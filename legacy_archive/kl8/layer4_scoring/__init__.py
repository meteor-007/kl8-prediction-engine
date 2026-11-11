"""快乐8 综合打分、五规则共振与候选池过滤层（Layer 4 Scoring）。

唯一打分入口 ``score_round``（P8-H1 修复后）：
- Borda 百分位秩归一化（``rank_matrix``）；
- 评估五条正交规则集并统计**整型**共振票数 V（报告展示口径）；
- 按契约 B5 计算**等效票数** ``Σ_r (w_r/w̄)·pass_r``，w_r 来自自适应状态或
  配置化均匀兜底，``FinalScore = BaseScore/Σω + λ·等效票数``；
- 硬约束贪心选池 + B6 重号配比 + B7 有序放宽留痕 + A5 金胆池。

所有下游（报告、回测、注单）只读 `ScoreRoundResult`，不得自行重算票数/权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from kl8.config import AppConfig
from kl8.datatypes import FeatureSnapshot, PoolResult, RuleVerdict
from kl8.layer4_scoring.adaptive import AdaptiveState, default_rule_weights
from kl8.layer4_scoring.borda import percentile_rank, rank_matrix
from kl8.layer4_scoring.filter import build_pool
from kl8.layer4_scoring.hybrid import ScoreResult, score_all
from kl8.layer4_scoring.rules import evaluate_rules, vote


@dataclass(frozen=True)
class ScoreRoundResult:
    """当期打分与候选池构建完整结果（契约 §4.7）。"""

    snapshot: FeatureSnapshot
    """特征快照。"""

    ranks: Mapping[str, np.ndarray]
    """百分位秩矩阵（键 → (80,)）。"""

    verdicts: tuple[RuleVerdict, ...]
    """五规则判定（固定 R1..R5 顺序）。"""

    votes: np.ndarray
    """整型共振票数 V，(80,) int8（报告/回测展示口径）。"""

    scores: ScoreResult
    """综合打分结果（含等效票数 weighted_votes 与实际 rule_weights）。"""

    pool: PoolResult
    """金胆池 + 复式池。"""

    weighted_votes: np.ndarray | None = None
    """契约 B5 等效票数 Σ(w_r/w̄)·pass_r，(80,) float64。"""

    rule_weights: Mapping[str, float] = field(default_factory=dict)
    """实际参与加成的规则权重（归一化后 Σ=1）。"""


def score_round(
    snapshot: FeatureSnapshot,
    cfg: AppConfig,
    *,
    adaptive_state: AdaptiveState | None = None,
) -> ScoreRoundResult:
    """唯一打分与选号池装配中心（契约 §3.10）。

    @param snapshot 特征工程快照
    @param cfg 全局配置
    @param adaptive_state 可选自适应权重状态（None → 配置化均匀兜底）
    @return ScoreRoundResult
    """
    # 1. 抽取用于规则与打分的特征进行 Borda 百分位秩归一化
    features_to_rank = {
        "freq_10": snapshot.freq_10,
        "freq_25": snapshot.freq_25,
        "freq_50": snapshot.freq_50,
        "freq_all": snapshot.freq_all,
        "mom_10_25": snapshot.mom_10_25,
        "mom_25_50": snapshot.mom_25_50,
        "follow_10": snapshot.follow_10,
        "follow_25": snapshot.follow_25,
        "follow_50": snapshot.follow_50,
        "follow_all": snapshot.follow_all,
        "co_10": snapshot.co_10,
        "co_25": snapshot.co_25,
        "co_50": snapshot.co_50,
        "co_all": snapshot.co_all,
        "ml_prob": snapshot.ml_prob,
    }
    ranks = rank_matrix(features_to_rank)

    # 2. 评估五大正交规则并统计整型共振票数（报告口径）
    verdicts = evaluate_rules(snapshot, ranks, cfg.rules)
    votes = vote(verdicts)

    # 3. 规则权重：自适应状态优先，否则走配置化均匀先验（禁止字面量兜底）
    if adaptive_state is not None:
        rule_weights: Mapping[str, float] = dict(adaptive_state.weights)
    else:
        rule_weights = default_rule_weights(cfg.adaptive)

    # 4. 综合打分（内部按 B5 计算等效票数 + 归一化）
    scores = score_all(
        snapshot,
        ranks,
        cfg.weights,
        votes=votes,
        verdicts=verdicts,
        rule_weights=rule_weights,
    )

    # 5. 上期开出的 20 个球号（遗漏为 0）用于重号配比控制
    last_draw = frozenset(int(i + 1) for i in range(len(snapshot.gap)) if snapshot.gap[i] == 0)

    # 6. 过滤构建号码池（复用已算好的得分/规则，零重复计算）
    pool = build_pool(
        snapshot,
        scores.final,
        votes,
        last_draw,
        pool_size=cfg.pool.main_default,
        cfg=cfg,
        ranks=ranks,
        verdicts=verdicts,
        base=scores.base,
    )

    equivalence = scores.weighted_votes
    return ScoreRoundResult(
        snapshot=snapshot,
        ranks=ranks,
        verdicts=verdicts,
        votes=votes,
        scores=scores,
        pool=pool,
        weighted_votes=equivalence,
        rule_weights=dict(scores.rule_weights),
    )


__all__ = [
    "ScoreRoundResult",
    "build_pool",
    "default_rule_weights",
    "evaluate_rules",
    "percentile_rank",
    "rank_matrix",
    "score_all",
    "score_round",
    "vote",
]