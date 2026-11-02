"""五维正交选号规则集与交集共振投票（契约 R25–R30、§3.10、8.md L131–L139）。

规则集定义：
- 规则 R1（多尺度强跟随共振）：4 个窗口的跟随得分百分位秩中，至少 3 个窗口 R >= 0.65 (R25)
- 规则 R2（温热多头趋势共振）：Freq_25 >= 0.24 且 Freq_50 >= 0.24 且 Freq_10 ∈ [0.20, 0.50] (R26)
- 规则 R3（动量金叉加速）：Freq_10 >= Freq_25 >= Freq_50 且 R(Follow_10) >= 0.60 (R27)
- 规则 R4（强共现伴生拉动）：0.5 * R(Co_25) + 0.5 * R(Co_50) >= 0.75 (R28)
- 规则 R5（全量回归临界回补）：Pressure >= 0.60 且 Gap <= 12 且 R(Follow_ALL) >= 0.70 (R29)
- 交集共振票数：V_i = Σ R_k(i) ∈ {0, 1, 2, 3, 4, 5} (R30)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from kl8.config import RuleConfig
from kl8.constants import N_NUM
from kl8.datatypes import FeatureSnapshot, RuleVerdict


def rule_r1(ranks: Mapping[str, np.ndarray], cfg: RuleConfig) -> np.ndarray:
    """规则 R1: 多尺度强跟随共振（契约 R25）。

    在 10G, 25G, 50G, ALL 四个窗口的 FollowScore 百分位秩中，至少有 r1_min_windows 个达到阈值。
    @return (80,) bool
    """
    keys = ("follow_10", "follow_25", "follow_50", "follow_all")
    hits = np.zeros(N_NUM, dtype=np.int32)
    for k in keys:
        if k in ranks:
            hits += (ranks[k] >= cfg.r1_rank_threshold).astype(np.int32)
    return hits >= cfg.r1_min_windows


def rule_r2(snapshot: FeatureSnapshot, cfg: RuleConfig) -> np.ndarray:
    """规则 R2: 温热多头趋势共振（契约 R26）。

    25G 与 50G 出现频率均不低于理论基线，且 10G 处于稳健温热区间 [0.20, 0.50]。
    @return (80,) bool
    """
    f10 = snapshot.freq_10
    f25 = snapshot.freq_25
    f50 = snapshot.freq_50
    return (
        (f25 >= cfg.r2_base_freq)
        & (f50 >= cfg.r2_base_freq)
        & (f10 >= cfg.r2_f10_low)
        & (f10 <= cfg.r2_f10_high)
    )


def rule_r3(
    snapshot: FeatureSnapshot,
    ranks: Mapping[str, np.ndarray],
    cfg: RuleConfig,
) -> np.ndarray:
    """规则 R3: 动量金叉加速（契约 R27）。

    短中长周期热力递增（Freq_10 >= Freq_25 >= Freq_50），且 10G 短期跟随排名进入前 40% (R >= 0.60)。
    @return (80,) bool
    """
    f10 = snapshot.freq_10
    f25 = snapshot.freq_25
    f50 = snapshot.freq_50
    heat_inc = (f10 >= f25) & (f25 >= f50)
    follow_ok = ranks.get("follow_10", np.zeros(N_NUM)) >= cfg.r3_follow_rank
    return heat_inc & follow_ok


def rule_r4(ranks: Mapping[str, np.ndarray], cfg: RuleConfig) -> np.ndarray:
    """规则 R4: 强共现伴生拉动（契约 R28）。

    25G 与 50G 共现伴生得分百分位秩加权均值进入前 25% (>= 0.75)。
    @return (80,) bool
    """
    r_co25 = ranks.get("co_25", np.zeros(N_NUM))
    r_co50 = ranks.get("co_50", np.zeros(N_NUM))
    combined = cfg.r4_w_co25 * r_co25 + cfg.r4_w_co50 * r_co50
    return combined >= cfg.r4_rank_score


def rule_r5(
    snapshot: FeatureSnapshot,
    ranks: Mapping[str, np.ndarray],
    cfg: RuleConfig,
) -> np.ndarray:
    """规则 R5: 全量回归临界回补（契约 R29）。

    处于黄金回补区 (Pressure >= 0.60 且 Gap <= 12) 且全量历史跟随排名进入前 30% (R >= 0.70)。
    @return (80,) bool
    """
    pressure_ok = snapshot.pressure >= cfg.r5_pressure
    gap_ok = snapshot.gap <= cfg.r5_max_gap
    follow_all_ok = ranks.get("follow_all", np.zeros(N_NUM)) >= cfg.r5_follow_all_rank
    return pressure_ok & gap_ok & follow_all_ok


def evaluate_rules(
    snapshot: FeatureSnapshot,
    ranks: Mapping[str, np.ndarray],
    cfg: RuleConfig,
) -> tuple[RuleVerdict, ...]:
    """对 80 码批量求值全部五条正交规则（契约 §3.10）。

    @param snapshot 特征快照
    @param ranks 归一化百分位秩矩阵
    @param cfg 规则配置
    @return (R1, R2, R3, R4, R5) 固定元组
    """
    m1 = rule_r1(ranks, cfg)
    m2 = rule_r2(snapshot, cfg)
    m3 = rule_r3(snapshot, ranks, cfg)
    m4 = rule_r4(ranks, cfg)
    m5 = rule_r5(snapshot, ranks, cfg)

    v1 = RuleVerdict(
        rule_id="R1",
        rule_name="多尺度强跟随共振",
        mask=m1,
        pass_count=int(m1.sum()),
        thresholds={"min_windows": float(cfg.r1_min_windows), "rank_threshold": cfg.r1_rank_threshold},
    )
    v2 = RuleVerdict(
        rule_id="R2",
        rule_name="温热多头趋势共振",
        mask=m2,
        pass_count=int(m2.sum()),
        thresholds={"base_freq": cfg.r2_base_freq, "f10_low": cfg.r2_f10_low, "f10_high": cfg.r2_f10_high},
    )
    v3 = RuleVerdict(
        rule_id="R3",
        rule_name="动量金叉加速",
        mask=m3,
        pass_count=int(m3.sum()),
        thresholds={"follow_rank": cfg.r3_follow_rank},
    )
    v4 = RuleVerdict(
        rule_id="R4",
        rule_name="强共现伴生拉动",
        mask=m4,
        pass_count=int(m4.sum()),
        thresholds={"rank_score": cfg.r4_rank_score, "w_co25": cfg.r4_w_co25, "w_co50": cfg.r4_w_co50},
    )
    v5 = RuleVerdict(
        rule_id="R5",
        rule_name="全量回归临界回补",
        mask=m5,
        pass_count=int(m5.sum()),
        thresholds={"pressure": cfg.r5_pressure, "max_gap": float(cfg.r5_max_gap), "follow_all_rank": cfg.r5_follow_all_rank},
    )

    return (v1, v2, v3, v4, v5)


def vote(
    verdicts: Sequence[RuleVerdict],
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> np.ndarray:
    """统计每个号码满足的规则条数或加权共振得分（契约 R30、R33）。

    【执行逻辑】
    - weights 为 None：统计整型共振票数 V_i ∈ {0..5}（返回 int8）；
    - weights 显式指定：按各规则经验胜率与信噪比进行加权折算（返回 float64）。

    @param verdicts 5 条规则的裁决结果
    @param weights 可选规则权重映射 (如 {"R5": 1.8, "R4": 0.4}) 或序列
    @return (80,) 共振票数或加权当量得分
    """
    if weights is None:
        votes = np.zeros(N_NUM, dtype=np.int8)
        for v in verdicts:
            votes += v.mask.astype(np.int8)
        return votes

    weighted_votes = np.zeros(N_NUM, dtype=np.float64)
    if isinstance(weights, Mapping):
        for v in verdicts:
            w = float(weights.get(v.rule_id, 1.0))
            weighted_votes += w * v.mask.astype(np.float64)
    else:
        for idx, v in enumerate(verdicts):
            w = float(weights[idx]) if idx < len(weights) else 1.0
            weighted_votes += w * v.mask.astype(np.float64)
    return weighted_votes
