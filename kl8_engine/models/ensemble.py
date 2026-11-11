# -*- coding: utf-8 -*-
"""深度学习与统计图谱混合集成引擎（Hybrid Ensemble Engine）。

【业务背景与架构定位】
本模块作为预测系统的核心仲裁中心，实现双引擎协同集成：
1. 深度学习引擎（PyTorch Transformer）：
   从开奖历史序列中捕获多阶自注意力时序依赖，输出概率分布 $P_{\\text{DL}}$；
2. 图谱贝叶斯引擎（Dirichlet + Spectral Graph）：
   从全局与局部开奖分布中捕获号码协同共现引力与贝叶斯平滑势能，输出得分 $S_{\\text{Spectral}}$；
3. 正交融合仲裁：
   默认采用四分量等权 Borda 秩融合（DL / Dirichlet / 共现谱 / 动量），
   消除 sigmoid 概率与 min-max 谱分数之间的量纲差异；
   保留 weighted 加权凸组合模式用于回滚与 A/B 对比，输出 80 码 FinalScore 并精选候选梯队。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from kl8_engine.constants import GOLD_POOL_SIZE, MAIN_POOL_SIZE, N_NUM, SHRINKAGE_POOL_SIZE, get_zone4, is_prime
from kl8_engine.data import Dataset
from kl8_engine.models.spectral_bayesian import (
    combine_spectral_components,
    compute_cooccurrence_spectral_scores,
    compute_dirichlet_scores,
    compute_momentum_differential_scores,
)
from kl8_engine.models.transformer_net import train_and_predict_transformer

FUSION_MODES: Tuple[str, ...] = ("borda", "weighted")
"""融合方式: borda=四分量等权秩融合; weighted=原始加权概率和（保留回滚）。"""


def rank_percentile_scores(scores: np.ndarray) -> np.ndarray:
    """将 80 维分数转换为 Borda 秩得分（1.0=最好，最小=最差）。

    秩变换天然消除各分量的量纲差异，避免 sigmoid 概率（~0.25 量级）
    被 min-max 到 [0,1] 的谱分量在加权和里淹没。
    """
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(np.argsort(arr))  # 0-based 升序名次
    return (order.astype(np.float64) + 1.0) / float(N_NUM)


def borda_fusion_scores(
    module_scores: Dict[str, np.ndarray],
    weights: Dict[str, float] | None = None,
) -> np.ndarray:
    """对 DL / Dirichlet / 共现谱 / 动量四分量做加权 Borda 秩融合。

    @param module_scores 分模块得分字典
    @param weights 可选分量权重（默认等权）
    @return shape (80,) 融合得分，排序与平均秩一致
    """
    keys = ("dl", "dirichlet", "cooccurrence", "momentum")
    weights = weights or {key: 1.0 for key in keys}
    total_weight = sum(weights[key] for key in keys)
    total = np.zeros(N_NUM, dtype=np.float64)
    for key in keys:
        total += weights[key] * rank_percentile_scores(module_scores[key])
    return total / max(total_weight, 1e-12)


@dataclass(frozen=True)
class NumberScoreDetail:
    """单号码综合得分明细实体。"""

    number: int
    final_score: float
    dl_prob: float
    spectral_score: float
    rank: int
    omission: int
    is_repeat: bool
    is_prime: bool
    zone4: int


@dataclass(frozen=True)
class EnsembleResult:
    """混合集成预测完整结果集。"""

    target_period: int
    gold_pool: Tuple[int, ...]
    main_pool: Tuple[int, ...]
    shrinkage_pool_30: Tuple[int, ...]
    full_order: Tuple[int, ...]
    scores: np.ndarray
    details: Dict[int, NumberScoreDetail]
    training_loss: float
    module_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    """分模块得分明细: dl / dirichlet / cooccurrence / momentum，shape 均为 (80,)。"""


def _compute_component_scores(
    dataset: Dataset,
    seq_len: int,
    epochs: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """训练一次 Transformer 并计算四个分量得分（供单/双融合复用）。"""
    matrix = dataset.to_matrix()
    dl_probs, train_loss = train_and_predict_transformer(
        matrix,
        seq_len=seq_len,
        epochs=epochs,
    )
    dirichlet_scores = compute_dirichlet_scores(matrix, recent_window=35)
    cooccurrence_scores = compute_cooccurrence_spectral_scores(matrix, window_limit=150)
    momentum_scores, omissions = compute_momentum_differential_scores(
        matrix, short_window=10, long_window=40
    )
    spectral_scores = combine_spectral_components(
        dirichlet_scores, cooccurrence_scores, momentum_scores
    )
    module_scores: Dict[str, np.ndarray] = {
        "dl": np.asarray(dl_probs, dtype=np.float64),
        "dirichlet": np.asarray(dirichlet_scores, dtype=np.float64),
        "cooccurrence": np.asarray(cooccurrence_scores, dtype=np.float64),
        "momentum": np.asarray(momentum_scores, dtype=np.float64),
    }
    return module_scores, spectral_scores, omissions, train_loss


def _build_ensemble_result(
    dataset: Dataset,
    module_scores: Dict[str, np.ndarray],
    spectral_scores: np.ndarray,
    omissions: np.ndarray,
    train_loss: float,
    fusion_mode: str,
    dl_weight: float,
) -> EnsembleResult:
    """基于同一组分量得分，按指定融合方式装配预测结果。"""
    target_period = dataset.target_period()
    latest_numbers = set(dataset.latest.numbers)

    if fusion_mode == "borda":
        # 四分量等权秩融合：消除 DL 概率与谱分数的量纲差异
        final_scores = borda_fusion_scores(module_scores)
    else:
        final_scores = (
            dl_weight * module_scores["dl"] + (1.0 - dl_weight) * spectral_scores
        )

    # 产生全局 80 码降序排列 (1-based)
    order_0based = np.argsort(-final_scores)
    full_order = tuple(int(idx + 1) for idx in order_0based)

    # 逐号装配明细字典
    details: Dict[int, NumberScoreDetail] = {}
    for rank_idx, num in enumerate(full_order, start=1):
        idx0 = num - 1
        details[num] = NumberScoreDetail(
            number=num,
            final_score=float(final_scores[idx0]),
            dl_prob=float(module_scores["dl"][idx0]),
            spectral_score=float(spectral_scores[idx0]),
            rank=rank_idx,
            omission=int(omissions[idx0]),
            is_repeat=(num in latest_numbers),
            is_prime=is_prime(num),
            zone4=get_zone4(num),
        )

    # 选拔核心梯队
    gold_pool = tuple(sorted(full_order[:GOLD_POOL_SIZE]))

    candidate_main: List[int] = []
    repeat_count = 0
    max_repeats = 5
    for num in full_order:
        if len(candidate_main) >= MAIN_POOL_SIZE:
            break
        if num in latest_numbers:
            if repeat_count < max_repeats:
                candidate_main.append(num)
                repeat_count += 1
        else:
            candidate_main.append(num)
    if len(candidate_main) < MAIN_POOL_SIZE:
        for num in full_order:
            if num not in candidate_main:
                candidate_main.append(num)
                if len(candidate_main) >= MAIN_POOL_SIZE:
                    break
    main_pool = tuple(sorted(candidate_main))
    shrinkage_pool = tuple(sorted(full_order[:SHRINKAGE_POOL_SIZE]))

    return EnsembleResult(
        target_period=target_period,
        gold_pool=gold_pool,
        main_pool=main_pool,
        shrinkage_pool_30=shrinkage_pool,
        full_order=full_order,
        scores=final_scores,
        details=details,
        training_loss=train_loss,
        module_scores=module_scores,
    )


def run_hybrid_ensemble(
    dataset: Dataset,
    dl_weight: float = 0.55,
    seq_len: int = 10,
    epochs: int = 30,
    fusion_mode: str = "borda",
) -> EnsembleResult:
    """端到端执行混合多模型集成预测流水线。

    @param dataset 历史开奖数据集
    @param dl_weight weighted 融合模式下的深度学习权重 (0.0..1.0)
    @param seq_len 时序回看窗口长度
    @param epochs Transformer 微调轮数
    @param fusion_mode borda=等权秩融合 (默认); weighted=原始加权和
    @return 结构化完整集成结果 EnsembleResult
    """
    if fusion_mode not in FUSION_MODES:
        raise ValueError(f"未知融合方式: {fusion_mode}，可选 {FUSION_MODES}")
    module_scores, spectral, omissions, loss = _compute_component_scores(
        dataset, seq_len, epochs
    )
    return _build_ensemble_result(
        dataset, module_scores, spectral, omissions, loss, fusion_mode, dl_weight
    )


def run_dual_ensemble(
    dataset: Dataset,
    official_mode: str = "borda",
    comparison_mode: Optional[str] = None,
    dl_weight: float = 0.55,
    seq_len: int = 10,
    epochs: int = 30,
) -> Tuple[EnsembleResult, Optional[EnsembleResult]]:
    """训练一次，同时产出正式版与实验版两套融合结果（并排对比用）。

    @param official_mode 正式版融合方式
    @param comparison_mode 实验版融合方式；None 时只返回正式版
    @return (正式版结果, 实验版结果或 None)
    """
    for mode in (official_mode, comparison_mode):
        if mode is not None and mode not in FUSION_MODES:
            raise ValueError(f"未知融合方式: {mode}，可选 {FUSION_MODES}")

    module_scores, spectral, omissions, loss = _compute_component_scores(
        dataset, seq_len, epochs
    )
    official = _build_ensemble_result(
        dataset, module_scores, spectral, omissions, loss, official_mode, dl_weight
    )
    comparison = None
    if comparison_mode is not None:
        comparison = _build_ensemble_result(
            dataset, module_scores, spectral, omissions, loss, comparison_mode, dl_weight
        )
    return official, comparison
