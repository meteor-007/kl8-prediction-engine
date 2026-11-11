# -*- coding: utf-8 -*-
"""Dirichlet 贝叶斯分层后验与号码共现谱图特征提取器。

【业务背景与架构定位】
借鉴 GitHub 优秀开源项目 `KittenCN/kl8-lottery-analyzer` 的统计学与图谱挖掘思想：
1. 采用 Dirichlet-Multinomial 贝叶斯共轭先验，解决短期窗口内的高频噪声与小样本方差过大问题；
2. 构建衰减加权的 80×80 号码同现图邻接矩阵，通过谱分解（Spectral Centrality）提取主特征向量，
   捕捉 20 码组合内部的协同引力与隐式集聚特征；
3. 计算超短线（10G）与中长线（40G/160G）多尺度动量速度差，刻画号码活跃态趋势。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

from typing import Final, Tuple
import numpy as np

from kl8_engine.constants import N_NUM

# 统计分量经验融合权重：Dirichlet 45% / 共现谱 35% / 动量差分 20%
COMPONENT_WEIGHTS: Final[Tuple[float, float, float]] = (0.45, 0.35, 0.20)


def _normalise(arr: np.ndarray) -> np.ndarray:
    """将一维数组线性缩放到 [0.0, 1.0] 闭区间，严格防除零。"""
    v_min = float(arr.min())
    v_max = float(arr.max())
    diff = v_max - v_min
    if diff < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - v_min) / diff


def compute_dirichlet_scores(
    history_matrix: np.ndarray,
    recent_window: int = 30,
    prior_strength: float = 2.0,
    variance_penalty: float = 1.0,
) -> np.ndarray:
    """计算 Dirichlet-Multinomial 贝叶斯分层后验平滑得分。

    @param history_matrix shape (T, 80)
    @param recent_window 近期局部观察窗口
    @param prior_strength 全局先验强度
    @param variance_penalty 方差风险贴现系数（惩罚不稳定号码）
    @return shape (80,) 归一化得分
    """
    T = len(history_matrix)
    if T == 0:
        return np.zeros(N_NUM, dtype=np.float32)

    # 步骤 1: 统计全局出现次数与局部出现次数
    global_counts = history_matrix.sum(axis=0) + 1e-4
    global_total = global_counts.sum()
    global_prob = global_counts / max(global_total, 1.0)

    local_matrix = history_matrix[-min(T, recent_window) :]
    local_counts = local_matrix.sum(axis=0)

    # 步骤 2: 构建狄利克雷后验参数 alpha
    prior_alpha = global_prob * (prior_strength * N_NUM)
    posterior_alpha = prior_alpha + local_counts
    alpha_sum = float(posterior_alpha.sum())

    # 步骤 3: 提取后验均值与后验方差
    posterior_mean = posterior_alpha / max(alpha_sum, 1.0)
    posterior_var = (posterior_alpha * (alpha_sum - posterior_alpha)) / (
        (alpha_sum**2) * (alpha_sum + 1.0)
    )

    # 步骤 4: 风险折现（期望减去方差惩罚，奖励稳定上升码）
    adjusted = posterior_mean - variance_penalty * np.sqrt(np.maximum(posterior_var, 0.0))
    return _normalise(adjusted)


def compute_cooccurrence_spectral_scores(
    history_matrix: np.ndarray,
    window_limit: int = 150,
    decay_rate: float = 0.975,
) -> np.ndarray:
    """构建衰减加权共现图并提取主特征向量谱中心性（Spectral Centrality）。

    @param history_matrix shape (T, 80)
    @param window_limit 截取最近期数
    @param decay_rate 时间衰减系数（越近的开奖共现权重越高）
    @return shape (80,) 谱中心性得分
    """
    T = len(history_matrix)
    if T < 2:
        return np.full(N_NUM, 0.5, dtype=np.float32)

    effective_slice = history_matrix[-min(T, window_limit) :]
    n_samples = len(effective_slice)

    # 步骤 1: 向量化构建衰减共现矩阵
    weights = decay_rate ** np.arange(n_samples - 1, -1, -1)  # shape (n_samples,)
    # 加权内积: C = X^T * diag(weights) * X
    weighted_X = effective_slice * np.sqrt(weights[:, np.newaxis])
    cooc_matrix = np.dot(weighted_X.T, weighted_X)  # shape (80, 80)

    # 对角线置零（自身不与自身共现）
    np.fill_diagonal(cooc_matrix, 0.0)

    # 步骤 2: 计算实对称矩阵的谱分解 (eigh)
    try:
        eigvals, eigvecs = np.linalg.eigh(cooc_matrix)
        # 最大特征值对应的特征向量
        principal_vec = np.abs(eigvecs[:, -1])
        scores = _normalise(principal_vec)
    except Exception:
        # 降级为加权度数中心性
        degree = cooc_matrix.sum(axis=1)
        scores = _normalise(degree)

    return scores


def compute_momentum_differential_scores(
    history_matrix: np.ndarray,
    short_window: int = 10,
    long_window: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算多尺度动量差分与当前遗漏值。

    @param history_matrix shape (T, 80)
    @param short_window 短周期窗口
    @param long_window 长周期参考窗口
    @return (动量得分 shape (80,), 遗漏期数 shape (80,))
    """
    T = len(history_matrix)
    if T == 0:
        return np.full(N_NUM, 0.5), np.zeros(N_NUM, dtype=int)

    f_short = history_matrix[-min(T, short_window) :].mean(axis=0)
    f_long = history_matrix[-min(T, long_window) :].mean(axis=0)
    momentum_diff = _normalise(f_short - f_long)

    # 计算每个号码当前遗漏期数
    omissions = np.zeros(N_NUM, dtype=int)
    for num_idx in range(N_NUM):
        col = history_matrix[:, num_idx]
        hit_indices = np.where(col > 0.5)[0]
        if len(hit_indices) == 0:
            omissions[num_idx] = T
        else:
            omissions[num_idx] = (T - 1) - int(hit_indices[-1])

    return momentum_diff, omissions


def combine_spectral_components(
    dirichlet_scores: np.ndarray,
    cooccurrence_scores: np.ndarray,
    momentum_scores: np.ndarray,
) -> np.ndarray:
    """按经验权重融合 Dirichlet、共现谱与动量三分量并归一化。

    @param dirichlet_scores shape (80,) Dirichlet 后验得分
    @param cooccurrence_scores shape (80,) 共现谱中心性得分
    @param momentum_scores shape (80,) 动量差分得分
    @return shape (80,) 归一化综合统计得分
    """
    d_w, c_w, m_w = COMPONENT_WEIGHTS
    combined = (
        d_w * np.asarray(dirichlet_scores, dtype=np.float64)
        + c_w * np.asarray(cooccurrence_scores, dtype=np.float64)
        + m_w * np.asarray(momentum_scores, dtype=np.float64)
    )
    return _normalise(combined)


def compute_spectral_bayesian_scores(history_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """综合计算贝叶斯图谱混合得分。

    权重分配遵循高信噪比经验配比：
    - Dirichlet 贝叶斯后验: 45%
    - 共现谱图中心性: 35%
    - 时序动量差分: 20%

    @param history_matrix shape (T, 80)
    @return (综合统计得分 shape (80,), 遗漏期数 shape (80,))
    """
    dirichlet = compute_dirichlet_scores(history_matrix, recent_window=35)
    spectral = compute_cooccurrence_spectral_scores(history_matrix, window_limit=150)
    momentum, omissions = compute_momentum_differential_scores(history_matrix, short_window=10, long_window=40)

    combined = combine_spectral_components(dirichlet, spectral, momentum)
    return combined, omissions
