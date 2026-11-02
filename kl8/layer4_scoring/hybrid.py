# -*- coding: utf-8 -*-
"""综合打分与融合方程（契约 R31、R32、§3.10、§5.1-B5、裁决 A1、A2）。

【公式（冻结）】

    BaseScore_i = clip_divide(
        Σ_{w∈{10G,25G,50G,ALL}} ω_fol,w · R(Follow_{i,w})
        + ω_co · R(C̄o_i)
        + ω_trend · R(Trend_i)
        + ω_om · Pressure_i
        + [ω_ml>0] ω_ml · R(ml_prob_i),
        core_sum)

    FinalScore_i = BaseScore_i + λ · Σ_r (w_r / w̄) · pass_{r,i},  w̄ = 1/5

其中：
- 裁决 A1：``Trend_i = 0.5·R(F10) + 0.3·R(Mom10-25) + 0.2·R(Mom25-50)``；
- 裁决 A2（v1.3.0 口径冻结）：``C̄o_i = mean(R(Co25), R(Co50), R(CoALL))``，
  **先秩后均、等权 ÷3、不含 10G**，任何加权版本均判偏离；
- 契约 B5：规则权重只作用于**票数加成**，``Σ_r w_r = 1``、``w̄ = 1/5``；
  ``w_r`` 全均匀（0.2）时 ``Σ(w_r/w̄)·pass_r ≡ V_i``，即退化为 8.md L144 原式
  ``FinalScore = Base + λ·V``；
- 契约 B4：``ml.weight>0`` 时 ML 项是 BaseScore 的**额外叠加项**，七维核心权重
  ``Σω=1`` 不变且 ml 不参与 core_sum 归一化（BaseScore 上界可 >1，报告须注明）。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

import numpy as np

from kl8.config import ScoreWeights
from kl8.constants import N_NUM, RULE_IDS
from kl8.datatypes import FeatureSnapshot, RuleVerdict

#: 裁决 A2 冻结的共现聚合窗口（等权、先秩后均、不含 10G）。
CO_AGG_WINDOWS: Final[tuple[str, ...]] = ("co_25", "co_50", "co_all")

#: 七维核心权重名（参与 Σω=1 校验与 BaseScore 归一化；ml 为额外叠加项不在其中）。
CORE_WEIGHT_NAMES: Final[tuple[str, ...]] = (
    "follow_10",
    "follow_25",
    "follow_50",
    "follow_all",
    "co",
    "trend",
    "omission",
)

#: 权重和容差（与 config.SUM_TOLERANCE 同口径）。
WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True)
class ScoreResult:
    """综合打分结果与贡献明细（契约 §4.7）。

    ``base`` / ``final`` 均已按契约 B5 归一化，满足 ``final - base == λ·weighted_votes``
    （浮点误差 < 1e-12），该恒等式由 ``tests/test_scoring.py`` 逐票断言。
    """

    base: np.ndarray
    """BaseScore（七维核心权重归一化后），(80,) float64。"""

    final: np.ndarray
    """FinalScore = base + λ·weighted_votes，(80,) float64。"""

    order: tuple[int, ...]
    """全部 80 码按 final 降序（并列按号码升序），1-based。"""

    contributors: Mapping[str, np.ndarray]
    """各分项贡献（follow_10/25/50/all、co、trend、omission、ml、vote）。"""

    weighted_votes: np.ndarray | None = None
    """契约 B5 的等效票数 Σ(w_r/w̄)·pass_r，(80,) float64；None 表示未计算。"""

    rule_weights: Mapping[str, float] = field(default_factory=dict)
    """实际参与加成的规则权重（归一化后，Σ=1），供报告单一数据源展示。"""


def trend_score(
    ranks: Mapping[str, np.ndarray],
    weights: ScoreWeights,
) -> np.ndarray:
    """计算多头趋势综合秩（契约裁决 A1、8.md L86–89）。

    Trend = 0.5·R(Freq_10) + 0.3·R(Mom_10-25) + 0.2·R(Mom_25-50)

    @param ranks 百分位秩字典
    @param weights 权重配置（提供 trend_w_* 三项）
    @return (80,) float64
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
    """计算秩空间共现伴生聚合 R(C̄o)（契约裁决 A2，v1.3.0 口径冻结）。

    ``C̄o_i = mean(R(Co_25), R(Co_50), R(Co_ALL))``：先秩后均、等权 ÷3、不含 10G。
    （历史事故：曾改为 0.20/0.20/0.60 并被测试标成「裁决 A2」，属偏离，已纠正。）

    @param ranks 百分位秩字典
    @return (80,) float64
    """
    parts = [np.asarray(ranks.get(name, np.zeros(N_NUM)), dtype=np.float64) for name in CO_AGG_WINDOWS]
    matrix = np.column_stack(parts)
    return np.asarray(matrix.mean(axis=1), dtype=np.float64)


def base_score(
    snapshot: FeatureSnapshot,
    ranks: Mapping[str, np.ndarray],
    weights: ScoreWeights,
) -> np.ndarray:
    """七维加权基础得分 BaseScore（契约 R31 / B4 / B5）。

    BaseScore = [Σ ω_fol·R(Follow) + ω_co·R(C̄o) + ω_trend·R(Trend) + ω_om·Pressure
                 (+ ω_ml·R(ml_prob) if ω_ml>0)] / Σω_core

    【为什么要除以 Σω_core】契约 B5 要求「均匀规则权重时 FinalScore 严格等于
    8.md 原式 Base + λ·V」。若用户把七维权重配成 Σω≠1（在校验允许的容差外不可能，
    但为防御性编程），不归一化会让 Base 量级漂移、共振加成相对被稀释。ml 项按 B4
    只做叠加、不参与该归一化。

    @param snapshot 特征快照
    @param ranks 百分位秩字典
    @param weights 权重配置（校验保证 Σω_core = 1）
    @return (80,) float64
    """
    fol_part = (
        weights.follow_10 * ranks.get("follow_10", np.zeros(N_NUM))
        + weights.follow_25 * ranks.get("follow_25", np.zeros(N_NUM))
        + weights.follow_50 * ranks.get("follow_50", np.zeros(N_NUM))
        + weights.follow_all * ranks.get("follow_all", np.zeros(N_NUM))
    )
    co_part = weights.co * co_aggregate(ranks)
    trend_part = weights.trend * trend_score(ranks, weights)
    om_part = weights.omission * np.asarray(snapshot.pressure, dtype=np.float64)

    score = fol_part + co_part + trend_part + om_part
    if weights.ml > 0.0:
        # B4：ML 为额外叠加项，不参与 core_sum 归一化
        score = score + weights.ml * ranks.get("ml_prob", np.zeros(N_NUM))

    core_sum = float(
        weights.follow_10
        + weights.follow_25
        + weights.follow_50
        + weights.follow_all
        + weights.co
        + weights.trend
        + weights.omission
    )
    if core_sum > WEIGHT_SUM_TOLERANCE:
        score = score / core_sum
    return np.asarray(score, dtype=np.float64)


def final_score(
    base: np.ndarray,
    votes: np.ndarray,
    lambda_vote: float,
) -> np.ndarray:
    """计算最终综合得分 FinalScore = BaseScore + λ·votes（契约 R32 / B5）。

    @param base (80,) float64 BaseScore（已归一化）
    @param votes (80,) float64 等效票数（= Σ(w_r/w̄)·pass_r；均匀权重时即整型 V）
    @param lambda_vote 单票共振加成系数 λ ∈ [0.05, 0.10]
    @return (80,) float64
    """
    return np.asarray(base, dtype=np.float64) + lambda_vote * np.asarray(votes, dtype=np.float64)


def normalize_rule_weights(
    rule_weights: Mapping[str, float] | None,
    rule_ids: Sequence[str],
) -> tuple[dict[str, float], tuple[str, ...]]:
    """归一化规则权重并做空安全兜底（契约 B5：Σ_r w_r = 1）。

    @param rule_weights 自适应状态给出的规则权重（可缺键/非归一/含负值）
    @param rule_ids 规则编号固定顺序
    @return (归一化权重字典, 告警元组)
    """
    warnings: list[str] = []
    count = len(rule_ids)
    if count == 0:
        return ({}, ())
    uniform = 1.0 / float(count)
    raw: dict[str, float] = {}
    for rule_id in rule_ids:
        value = 0.0 if rule_weights is None else float(rule_weights.get(rule_id, 0.0))
        raw[rule_id] = max(0.0, value)
    total = sum(raw.values())
    if total <= WEIGHT_SUM_TOLERANCE:
        warnings.append("规则权重缺失或全为 0，回退为均匀权重")
        return ({rule_id: uniform for rule_id in rule_ids}, tuple(warnings))
    normalized = {rule_id: value / total for rule_id, value in raw.items()}
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        warnings.append(f"规则权重之和 {total:.6f} ≠ 1，已按比例归一化")
    return (normalized, tuple(warnings))


def rule_weight_equivalence(
    verdicts: Sequence[RuleVerdict],
    rule_weights: Mapping[str, float] | None,
) -> tuple[np.ndarray, dict[str, float], tuple[str, ...]]:
    """计算契约 B5 的等效票数 ``Σ_r (w_r / w̄)·pass_r``（w̄ = 1/规则数）。

    【为什么除以 w̄】w_r 归一化到 Σ=1 后均值为 w̄=0.2，若不除，均匀权重下共振加成
    会被缩小 5 倍（历史 High 缺陷 H1）。除以 w̄ 后均匀权重严格退化为 ``V_i``。

    @param verdicts 五条规则判定（固定 R1..R5 顺序）
    @param rule_weights 规则权重（None → 均匀）
    @return (等效票数 (80,) float64, 归一化权重, 告警元组)
    """
    rule_ids = [verdict.rule_id for verdict in verdicts]
    weights_norm, warnings = normalize_rule_weights(rule_weights, rule_ids)
    if not rule_ids:
        return (np.zeros(N_NUM, dtype=np.float64), {}, tuple(warnings))
    w_bar = 1.0 / float(len(rule_ids))
    equivalence = np.zeros(N_NUM, dtype=np.float64)
    for verdict in verdicts:
        equivalence += (weights_norm[verdict.rule_id] / w_bar) * np.asarray(verdict.mask, dtype=np.float64)
    return (equivalence, weights_norm, tuple(warnings))


def _score_order(final: np.ndarray) -> tuple[int, ...]:
    """按 FinalScore 降序、并列按号码升序生成 1-based 顺序。

    @param final (80,) FinalScore
    @return 1-based 号码元组
    """
    numbers = np.arange(1, N_NUM + 1, dtype=np.int32)
    order_indices = np.lexsort((numbers, -np.asarray(final, dtype=np.float64)))
    return tuple(int(numbers[idx]) for idx in order_indices)


def score_all(
    snapshot: FeatureSnapshot,
    ranks: Mapping[str, np.ndarray],
    weights: ScoreWeights,
    *,
    votes: np.ndarray | None = None,
    verdicts: Sequence[RuleVerdict] | None = None,
    rule_weights: Mapping[str, float] | None = None,
) -> ScoreResult:
    """执行全部 80 码的综合打分、分项贡献分解与降序排序（契约 §3.10、R32、B5）。

    【等效票数口径（契约 B5）】
    ``FinalScore = BaseScore + λ·Σ_r (w_r/w̄)·pass_r``；当规则权重全相同时
    ``Σ_r (w_r/w̄)·pass_r ≡ 整型得票 V``，因此 ``FinalScore - BaseScore == λ·V`` 精确成立。

    @param snapshot 特征快照
    @param ranks 百分位秩矩阵
    @param weights 基础权重配置
    @param votes 整型共振票数 (80,)；与 ``verdicts`` 二选一，传入则直接作为等效票
    @param verdicts 五规则判定（``score_round`` 传入以避免重复计算；None 且 votes 为 None 时内部求值）
    @param rule_weights 自适应规则权重（None → 均匀先验，等效票数 == 整型 V）
    @return ScoreResult（含等效票数与归一化权重）
    """
    if votes is None and verdicts is None:
        from kl8.config import RuleConfig
        from kl8.layer4_scoring.rules import evaluate_rules

        verdicts = evaluate_rules(snapshot, ranks, RuleConfig())

    if votes is not None:
        equivalence = np.asarray(votes, dtype=np.float64)
        if rule_weights is None:
            weights_norm: Mapping[str, float] = {
                rule_id: 1.0 / len(RULE_IDS) for rule_id in RULE_IDS
            }
        else:
            _eq_check, weights_norm, _warns = rule_weight_equivalence(verdicts, rule_weights)
            if verdicts is None:
                weights_norm = normalize_rule_weights(rule_weights)
    else:
        equivalence, weights_norm, _warns = rule_weight_equivalence(verdicts, rule_weights)

    base = base_score(snapshot, ranks, weights)
    final = final_score(base, equivalence, weights.lambda_vote)

    co_part = weights.co * co_aggregate(ranks)
    trend_part = weights.trend * trend_score(ranks, weights)
    om_part = weights.omission * np.asarray(snapshot.pressure, dtype=np.float64)
    vote_part = weights.lambda_vote * equivalence

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
        order=_score_order(final),
        contributors=contrib,
        weighted_votes=equivalence,
        rule_weights=weights_norm,
    )