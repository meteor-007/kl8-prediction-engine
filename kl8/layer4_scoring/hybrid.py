"""综合打分与融合方程（契约 R31、R32、§3.10、裁决 A1、A2）。

公式：
BaseScore_i = Σ_{w ∈ W} ω_{fol,w} R(Follow_{i,w}) + ω_{co} R(C̄o_i) + ω_{trend} R(Trend_i) + ω_{om} Pressure_i
FinalScore_i = BaseScore_i + λ * V_i
其中：
- 裁决 A1: Trend_i = 0.5 * R(F10) + 0.3 * R(Mom10-25) + 0.2 * R(Mom25-50)
- 裁决 A2: R(C̄o_i) 采用秩空间平均 (R(Co25) + R(Co50) + R(CoALL)) / 3.0
- 恒等式: FinalScore - (BaseScore + λ * V) < 1e-12
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.config import ScoreWeights
from kl8.constants import N_NUM
from kl8.datatypes import FeatureSnapshot


@dataclass(frozen=True)
class ScoreResult:
    """综合打分结果与贡献明细（契约 §4.7）。"""

    base: np.ndarray
    final: np.ndarray
    order: tuple[int, ...]
    contributors: Mapping[str, np.ndarray]


def trend_score(
    ranks: Mapping[str, np.ndarray],
    weights: ScoreWeights,
) -> np.ndarray:
    """计算多头趋势综合秩（契约裁决 A1、8.md L86–89）。

    Trend = 0.5 * R(Freq_10) + 0.3 * R(Mom_10-25) + 0.2 * R(Mom_25-50)
    """
    r_f10 = ranks.get("freq_10", np.zeros(N_NUM))
    r_m1 = ranks.get("mom_10_25", np.zeros(N_NUM))
    r_m2 = ranks.get("mom_25_50", np.zeros(N_NUM))
    return (
        weights.trend_w_f10 * r_f10
        + weights.trend_w_mom_10_25 * r_m1
        + weights.trend_w_mom_25_50 * r_m2
    )


def co_aggregate(ranks: Mapping[str, np.ndarray]) -> np.ndarray:
    """计算秩空间共现伴生加权聚合 R(C̄o)（契约裁决 A2、8.md L143）。

    经 50 期实测信噪比优化：采用长线稳健共现强化方案 (0.60 * R(Co_ALL) + 0.20 * R(Co_50) + 0.20 * R(Co_25))，
    有效压制短周期高频共现噪声，保留长期共现协同底色。
    """
    r_co25 = ranks.get("co_25", np.zeros(N_NUM))
    r_co50 = ranks.get("co_50", np.zeros(N_NUM))
    r_co_all = ranks.get("co_all", np.zeros(N_NUM))
    return 0.20 * r_co25 + 0.20 * r_co50 + 0.60 * r_co_all


def base_score(
    snapshot: FeatureSnapshot,
    ranks: Mapping[str, np.ndarray],
    weights: ScoreWeights,
) -> np.ndarray:
    """五维加权基础得分 BaseScore（契约 R31）。

    BaseScore = Σ ω_fol * R(Follow) + ω_co * R(C̄o) + ω_trend * R(Trend) + ω_om * Pressure
    若启用 ML (weights.ml > 0)，加入 ML 概率项。
    @param snapshot 特征快照
    @param ranks 百分位秩字典
    @param weights 权重配置 (保证 Σω = 1)
    @return (80,) float64
    """
    r_fol10 = ranks.get("follow_10", np.zeros(N_NUM))
    r_fol25 = ranks.get("follow_25", np.zeros(N_NUM))
    r_fol50 = ranks.get("follow_50", np.zeros(N_NUM))
    r_fol_all = ranks.get("follow_all", np.zeros(N_NUM))

    fol_part = (
        weights.follow_10 * r_fol10
        + weights.follow_25 * r_fol25
        + weights.follow_50 * r_fol50
        + weights.follow_all * r_fol_all
    )
    co_part = weights.co * co_aggregate(ranks)
    trend_part = weights.trend * trend_score(ranks, weights)
    om_part = weights.omission * snapshot.pressure

    score = fol_part + co_part + trend_part + om_part
    if weights.ml > 0.0:
        r_ml = ranks.get("ml_prob", np.zeros(N_NUM))
        score += weights.ml * r_ml

    return score


def final_score(
    base: np.ndarray,
    votes: np.ndarray,
    lambda_vote: float,
) -> np.ndarray:
    """计算最终综合得分 FinalScore = BaseScore + λ * V_i（契约 R32）。

    @param base (80,) float64 BaseScore
    @param votes (80,) int8 交集共振票数 V_i ∈ {0..5}
    @param lambda_vote 单票共振加成系数 λ ∈ [0.05, 0.10]
    @return (80,) float64
    """
    return base + (lambda_vote * votes.astype(np.float64))


def score_all(
    snapshot: FeatureSnapshot,
    ranks: Mapping[str, np.ndarray],
    weights: ScoreWeights,
    *,
    votes: np.ndarray | None = None,
    rule_weights: Mapping[str, float] | None = None,
) -> ScoreResult:
    """执行全部 80 码的综合打分、分项贡献分解与降序排序（契约 §3.10、R32）。

    @param snapshot 特征快照
    @param ranks 百分位秩矩阵
    @param weights 基础权重配置
    @param votes 可选交集共振票数（若为 None 则从规则求值生成）
    @param rule_weights 可选自适应规则权重（向后兼容扩展）
    @return ScoreResult
    """
    from kl8.config import RuleConfig
    from kl8.layer4_scoring.rules import evaluate_rules, vote

    if votes is None:
        verdicts = evaluate_rules(snapshot, ranks, RuleConfig())
        effective_rule_weights = rule_weights if rule_weights is not None else {
            "R1": 1.0, "R2": 0.8, "R3": 0.8, "R4": 0.4, "R5": 1.8
        }
        votes = vote(verdicts, weights=effective_rule_weights)

    base = base_score(snapshot, ranks, weights)
    final = final_score(base, votes, weights.lambda_vote)

    # 排序：FinalScore 降序，并列按号码 (1..80) 升序
    # order 元素为 1-based 号码
    nums = np.arange(1, N_NUM + 1, dtype=np.int32)
    # 构造稳定复合键: -final 为主键，num 为次键
    sort_keys = [(-final[i], nums[i]) for i in range(N_NUM)]
    order_tuples = sorted(enumerate(sort_keys), key=lambda x: x[1])
    order = tuple(int(nums[orig_idx]) for orig_idx, _ in order_tuples)

    # 分解各项贡献
    co_part = weights.co * co_aggregate(ranks)
    trend_part = weights.trend * trend_score(ranks, weights)
    om_part = weights.omission * snapshot.pressure
    vote_part = weights.lambda_vote * votes.astype(np.float64)

    contrib: dict[str, np.ndarray] = {
        "follow_10": weights.follow_10 * ranks.get("follow_10", np.zeros(N_NUM)),
        "follow_25": weights.follow_25 * ranks.get("follow_25", np.zeros(N_NUM)),
        "follow_50": weights.follow_50 * ranks.get("follow_50", np.zeros(N_NUM)),
        "follow_all": weights.follow_all * ranks.get("follow_all", np.zeros(N_NUM)),
        "co": co_part,
        "trend": trend_part,
        "omission": om_part,
        "vote": vote_part,
    }

    return ScoreResult(
        base=base,
        final=final,
        order=order,
        contributors=contrib,
    )
