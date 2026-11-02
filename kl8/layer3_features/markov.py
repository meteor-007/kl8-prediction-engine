"""多尺度马尔可夫跟随转移矩阵与拉动得分（契约 R15–R18、§3.9）。

实现特征：
- 一阶拉普拉斯平滑条件转移概率 M^(1)(w) (R15)
- 四尺度中心化跟随聚合 FollowScore (R16)
- 重号对角惯性分与异号跟随分拆分 (R17)
- 二阶滞后转移矩阵 M^(2) 隔期回补 (R18)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.config import MarkovConfig
from kl8.constants import N_NUM, P0
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import WindowSpec


@dataclass(frozen=True)
class MarkovResult:
    """马尔可夫转移矩阵与跟随得分计算结果。"""

    follow: Mapping[str, np.ndarray]
    follow_diag: Mapping[str, np.ndarray]
    follow_offdiag: Mapping[str, np.ndarray]
    follow2_all: np.ndarray
    matrices: Mapping[str, np.ndarray]


def transition_matrix(
    window_matrix: np.ndarray,
    *,
    alpha: float,
    p0: float = P0,
) -> np.ndarray:
    """计算一阶拉普拉斯平滑条件转移矩阵 M^(1) ∈ R^(80×80)（契约 R15）。

    M_{i->j} = (Σ X_{τ-1, i} X_{τ, j} + α * p0) / (Σ X_{τ-1, i} + α)
    @param window_matrix (T, 80) 二值切片矩阵
    @param alpha 拉普拉斯平滑系数 α ∈ [0.5, 2.0]
    @param p0 先验概率，默认 0.25
    @return (80, 80) float64 转移概率矩阵
    """
    mat = np.asarray(window_matrix, dtype=np.float64)
    t_len = len(mat)
    if t_len < 2:
        return np.full((N_NUM, N_NUM), p0, dtype=np.float64)

    x_prev = mat[:-1]  # (T-1, 80)
    x_curr = mat[1:]   # (T-1, 80)

    # 跨期联合频次 N_ij
    n_ij = x_prev.T @ x_curr  # (80, 80)
    c_i = np.sum(x_prev, axis=0)  # (80,)

    denom = c_i[:, None] + alpha  # (80, 1)
    numer = n_ij + (alpha * p0)
    return numer / denom


def transition_matrix_order2(
    window_matrix: np.ndarray,
    *,
    alpha: float,
    p0: float = P0,
) -> np.ndarray:
    """计算二阶隔期滞后转移矩阵 M^(2) ∈ R^(80×80)（契约 R18）。

    M^(2)_{i->j} 量化 τ-2 期出现号码 i 对 τ 期号码 j 的拉动效应。
    @param window_matrix (T, 80) 二值切片矩阵
    @param alpha 拉普拉斯平滑系数
    @param p0 先验概率
    @return (80, 80) float64 转移概率矩阵
    """
    mat = np.asarray(window_matrix, dtype=np.float64)
    t_len = len(mat)
    if t_len < 3:
        return np.full((N_NUM, N_NUM), p0, dtype=np.float64)

    x_prev2 = mat[:-2]  # (T-2, 80)
    x_curr = mat[2:]    # (T-2, 80)

    n2_ij = x_prev2.T @ x_curr
    c2_i = np.sum(x_prev2, axis=0)

    denom = c2_i[:, None] + alpha
    numer = n2_ij + (alpha * p0)
    return numer / denom


def follow_score(
    matrix: np.ndarray,
    current_set: frozenset[int],
    *,
    p0: float = P0,
) -> np.ndarray:
    """计算最新期 20 码对全部 80 码的中心化跟随拉动得分（契约 R16）。

    FollowScore_j = Σ_{i ∈ S_t} (M_{i->j} - p0)
    @param matrix (80, 80) 转移矩阵
    @param current_set 最新一期开出的 20 个号码 (1-based)
    @param p0 先验概率基准
    @return (80,) float64
    """
    if not current_set:
        return np.zeros(N_NUM, dtype=np.float64)

    indices = [num - 1 for num in current_set if 1 <= num <= N_NUM]
    if not indices:
        return np.zeros(N_NUM, dtype=np.float64)

    # 提取第 i 行的条件概率相加并中心化扣除 p0
    sub_mat = matrix[indices, :]  # (len(indices), 80)
    centered = sub_mat - p0
    return np.sum(centered, axis=0)


def follow_score_all_windows(
    matrices: Mapping[str, np.ndarray],
    current_set: frozenset[int],
    *,
    p0: float = P0,
) -> dict[str, np.ndarray]:
    """计算所有配置窗口的中心化跟随得分。"""
    res: dict[str, np.ndarray] = {}
    for name, mat in matrices.items():
        res[name] = follow_score(mat, current_set, p0=p0)
    return res


def follow_split(
    matrices: Mapping[str, np.ndarray],
    current_set: frozenset[int],
    *,
    p0: float = P0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """将中心化跟随得分拆分为重号（对角）惯性分与异号跟随分（契约 R17）。

    @param matrices 各窗口转移矩阵
    @param current_set 最新一期号码集合 (1-based)
    @param p0 先验概率
    @return (follow_diag_map, follow_offdiag_map)
    """
    diag_map: dict[str, np.ndarray] = {}
    offdiag_map: dict[str, np.ndarray] = {}

    for name, mat in matrices.items():
        total_follow = follow_score(mat, current_set, p0=p0)
        diag = np.zeros(N_NUM, dtype=np.float64)
        for num in current_set:
            if 1 <= num <= N_NUM:
                idx = num - 1
                diag[idx] = mat[idx, idx] - p0
        offdiag = total_follow - diag
        diag_map[name] = diag
        offdiag_map[name] = offdiag

    return (diag_map, offdiag_map)


def build_markov(
    state: BinaryStateMatrix,
    end: int,
    windows: Mapping[str, WindowSpec],
    cfg: MarkovConfig,
) -> MarkovResult:
    """构建马尔可夫特征集合。

    @param state 状态矩阵
    @param end 历史截断端点
    @param windows 窗口规范
    @param cfg 马尔可夫配置
    @return MarkovResult
    """
    last_set = state.latest_set(end)

    matrices: dict[str, np.ndarray] = {}
    for name, spec in windows.items():
        w_mat = state.window_matrix(end, spec.size)
        matrices[name] = transition_matrix(w_mat, alpha=cfg.alpha, p0=P0)

    follow = follow_score_all_windows(matrices, last_set, p0=P0)
    follow_diag, follow_offdiag = follow_split(matrices, last_set, p0=P0)

    # 二阶马尔可夫
    if cfg.order >= 2:
        all_mat = state.window_matrix(end, None)
        m2_all = transition_matrix_order2(all_mat, alpha=cfg.alpha, p0=P0)
        follow2_all = follow_score(m2_all, last_set, p0=P0)
    else:
        follow2_all = np.zeros(N_NUM, dtype=np.float64)

    return MarkovResult(
        follow=follow,
        follow_diag=follow_diag,
        follow_offdiag=follow_offdiag,
        follow2_all=follow2_all,
        matrices=matrices,
    )
