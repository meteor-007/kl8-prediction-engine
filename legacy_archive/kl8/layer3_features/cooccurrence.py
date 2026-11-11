"""同期共现矩阵与伴生拉动特征（契约 R19、R20、§3.9）。

实现特征：
- 窗口内两两共现频次矩阵与中心化共现矩阵 A_ij = C_ij / w - p_co (R19)
- 三元共现团计数 triad_counts (R19)
- 当期伴生拉动分 CoScore_j = Σ_{i ∈ S_t, i ≠ j} A_ij (R20)
- 四尺度伴生拉动分与原始均值聚合 co_centered_mean
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from kl8.constants import N_NUM, P_CO


def cooccurrence_counts(window_matrix: np.ndarray) -> np.ndarray:
    """计算切片窗口内的两两同期共现频次矩阵（对角线置 0）。

    @param window_matrix (T, 80) 二值切片矩阵
    @return (80, 80) int32 共现频次对称矩阵
    """
    mat = np.asarray(window_matrix, dtype=np.float64)
    if len(mat) == 0:
        return np.zeros((N_NUM, N_NUM), dtype=np.int32)
    counts = (mat.T @ mat).astype(np.int32)
    np.fill_diagonal(counts, 0)
    return counts


def centered_cooccurrence(
    counts: np.ndarray,
    w: int,
    *,
    p_co: float = P_CO,
) -> np.ndarray:
    """计算中心化共现矩阵 A_ij = C_ij / w - p_co（契约 R19）。

    正值表示实际共现高于理论基线（抱团亲和），负值表示排斥。
    @param counts (80, 80) 共现频次矩阵
    @param w 窗口期数
    @param p_co 理论两球共现概率，默认 ≈ 0.060127
    @return (80, 80) float64 对角为 0 的中心化矩阵
    """
    if w <= 0:
        return np.zeros((N_NUM, N_NUM), dtype=np.float64)
    c_float = counts.astype(np.float64)
    a_mat = (c_float / float(w)) - p_co
    np.fill_diagonal(a_mat, 0.0)
    return a_mat


def co_score(
    A: np.ndarray,
    current_set: frozenset[int],
) -> np.ndarray:
    """计算最新一期开奖号码对全部 80 码的伴生拉动分（契约 R20）。

    CoScore_j = Σ_{i ∈ S_t, i ≠ j} A_ij
    由于 A 对角线已置 0，直接将 S_t 对应行的向量求和即自动剔除自身。
    @param A (80, 80) 中心化共现矩阵
    @param current_set 最新期 20 码 (1-based)
    @return (80,) float64
    """
    if not current_set:
        return np.zeros(N_NUM, dtype=np.float64)
    indices = [num - 1 for num in current_set if 1 <= num <= N_NUM]
    if not indices:
        return np.zeros(N_NUM, dtype=np.float64)
    return np.sum(A[indices, :], axis=0)


def co_score_all_windows(
    A_map: Mapping[str, np.ndarray],
    current_set: frozenset[int],
) -> dict[str, np.ndarray]:
    """计算所有窗口的伴生拉动得分。"""
    res: dict[str, np.ndarray] = {}
    for name, a_mat in A_map.items():
        res[name] = co_score(a_mat, current_set)
    return res


def co_centered_mean(co_map: Mapping[str, np.ndarray]) -> np.ndarray:
    """计算 25G, 50G, ALL 窗口伴生拉动得分的原始均值（契约 §4.1、8.md L143）。

    供报告展示与打分参考。
    @param co_map 键为窗口名称的伴生拉动分字典
    @return (80,) float64
    """
    keys = [k for k in ("25G", "50G", "ALL") if k in co_map]
    if not keys:
        keys = list(co_map.keys())
    if not keys:
        return np.zeros(N_NUM, dtype=np.float64)
    stacked = np.stack([co_map[k] for k in keys], axis=0)
    return np.mean(stacked, axis=0)


def lift_matrix(
    counts: np.ndarray,
    w: int,
    *,
    p0: float = 0.25,
    eps: float = 1e-12,
) -> np.ndarray:
    """计算号码间提升度 Lift 矩阵：P(A, B) / (P(A) * P(B))。"""
    if w <= 0:
        return np.ones((N_NUM, N_NUM), dtype=np.float64)
    p_ab = counts.astype(np.float64) / float(w)
    # 先验独立期望
    p_indep = p0 * p0
    lift = p_ab / max(p_indep, eps)
    np.fill_diagonal(lift, 1.0)
    return lift


def triad_counts(
    counts: np.ndarray,
    *,
    top_n: int = 20,
) -> tuple[int, np.ndarray]:
    """统计强伴生网络中的三元共现团（3-clique 三角形）个数（契约 R19）。

    基于二值化强伴生图计算每个节点的三角形个数与全局三元团总数。
    @param counts (80, 80) 共现频次矩阵
    @param top_n 选取最高频的前 N 对作为强边（或按显著阈值）
    @return (全局三元团总数, 各号码参与的三元团个数(80,) int32)
    """
    # 构造强边邻接矩阵：取频次大于中位数且大于 0 的边
    c_arr = counts.copy()
    np.fill_diagonal(c_arr, 0)
    positive_vals = c_arr[c_arr > 0]
    if len(positive_vals) == 0:
        return (0, np.zeros(N_NUM, dtype=np.int32))

    threshold = np.quantile(positive_vals, 0.85) if len(positive_vals) > 20 else 0
    adj = (c_arr >= threshold) & (c_arr > 0)
    adj = adj.astype(np.float64)

    # 节点 i 参与的三角形数量为 (adj^3)[i, i] / 2
    adj2 = adj @ adj
    adj3 = adj2 @ adj
    diag_triads = (np.diag(adj3) / 2.0).astype(np.int32)
    total_triads = int(diag_triads.sum() // 3)

    return (total_triads, diag_triads)
