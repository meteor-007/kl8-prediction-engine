"""快乐8 综合打分、五规则共振与候选池过滤层（Layer 4 Scoring）。

唯一打分入口 score_round：
- 执行 Borda 百分位秩归一化
- 评估五条正交规则集并计算共振票数
- 执行加权打分与共振票数加成 (FinalScore = BaseScore + λ * V)
- 过滤组装核心金胆池与优选复式池
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.config import AppConfig
from kl8.datatypes import FeatureSnapshot, PoolResult, RuleVerdict
from kl8.layer4_scoring.adaptive import AdaptiveState
from kl8.layer4_scoring.borda import percentile_rank, rank_matrix
from kl8.layer4_scoring.filter import build_pool
from kl8.layer4_scoring.hybrid import ScoreResult, score_all
from kl8.layer4_scoring.rules import evaluate_rules, vote


@dataclass(frozen=True)
class ScoreRoundResult:
    """当期打分与候选池构建完整结果（契约 §4.7）。"""

    snapshot: FeatureSnapshot
    ranks: Mapping[str, np.ndarray]
    verdicts: tuple[RuleVerdict, ...]
    votes: np.ndarray
    scores: ScoreResult
    pool: PoolResult


def score_round(
    snapshot: FeatureSnapshot,
    cfg: AppConfig,
    *,
    adaptive_state: AdaptiveState | None = None,
) -> ScoreRoundResult:
    """唯一打分与选号池装配中心（契约 §3.10）。

    @param snapshot 特征工程快照
    @param cfg 全局配置
    @param adaptive_state 可选自适应权重状态
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

    # 2. 评估五大正交规则并统计共振票数
    verdicts = evaluate_rules(snapshot, ranks, cfg.rules)
    votes = vote(verdicts)

    # 3. 综合打分（应用实测信噪比加权共振当量）
    rule_weights = adaptive_state.weights if adaptive_state is not None else {
        "R1": 1.0, "R2": 0.8, "R3": 0.8, "R4": 0.4, "R5": 1.8
    }
    weighted_votes = vote(verdicts, weights=rule_weights)
    scores = score_all(snapshot, ranks, cfg.weights, votes=weighted_votes, rule_weights=rule_weights)

    # 4. 获取上期开出的 20 个球号用于重号配比控制
    # 遗漏为 0 的号码即上期开出
    last_draw = frozenset(int(i + 1) for i in range(len(snapshot.gap)) if snapshot.gap[i] == 0)

    # 5. 过滤构建号码池
    pool = build_pool(
        snapshot,
        scores.final,
        votes,
        last_draw,
        pool_size=cfg.pool.main_default,
        cfg=cfg,
    )

    return ScoreRoundResult(
        snapshot=snapshot,
        ranks=ranks,
        verdicts=verdicts,
        votes=votes,
        scores=scores,
        pool=pool,
    )


__all__ = [
    "ScoreRoundResult",
    "build_pool",
    "evaluate_rules",
    "percentile_rank",
    "rank_matrix",
    "score_all",
    "score_round",
    "vote",
]
