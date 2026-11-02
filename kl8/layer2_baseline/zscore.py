"""窗口频次标准化偏离度 Z-Score 计算（契约 R8、§3.8）。

公式：
Z_i(w) = (c_i(w) - w * p0) / sqrt(w * p0 * (1 - p0))
当 |Z_i(w)| > 1.96 时表明在 95% 置信度下存在显著冷热偏态。
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from kl8.constants import N_NUM, P0, Z_SIG
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import WindowSpec


def window_z(
    state: BinaryStateMatrix,
    end: int,
    w: int | None,
    *,
    p: float = P0,
    eps: float = 1e-9,
) -> np.ndarray:
    """计算单个滑动窗口内的 80 码标准化偏离度 Z-Score。

    @param state 状态矩阵
    @param end 历史截断端点（含）
    @param w 窗口期数，None 表示 ALL 窗口
    @param p 理论期望概率，默认 0.25
    @param eps 防除零分母极小值
    @return (80,) float64 的 Z 统计量
    """
    if state.is_empty():
        return np.zeros(N_NUM, dtype=np.float64)

    counts = state.window_sum(end, w).astype(np.float64)
    # 实际有效窗口长度
    if w is None:
        w_eff = end + 1
    else:
        w_eff = min(w, end + 1)

    if w_eff <= 0:
        return np.zeros(N_NUM, dtype=np.float64)

    expected = float(w_eff) * p
    variance = float(w_eff) * p * (1.0 - p)
    std = math.sqrt(max(variance, eps))
    return (counts - expected) / std


def significant_mask(z: np.ndarray, threshold: float = Z_SIG) -> np.ndarray:
    """判断 Z 统计量是否达到双尾显著性水平。

    @param z (80,) 偏离度统计量
    @param threshold 显著性分位数，默认 1.96
    @return (80,) bool 显著性掩码
    """
    return np.abs(z) > threshold


def z_all_windows(
    state: BinaryStateMatrix,
    end: int,
    windows: Mapping[str, WindowSpec],
    *,
    p: float = P0,
) -> dict[str, np.ndarray]:
    """计算所有配置窗口的 Z-Score。

    @param state 状态矩阵
    @param end 历史截断端点
    @param windows 窗口字典
    @param p 理论期望概率
    @return 键为窗口名（如 '10G', '25G', '50G', 'ALL'），值为 (80,) float64 的字典
    """
    res: dict[str, np.ndarray] = {}
    for name, spec in windows.items():
        res[name] = window_z(state, end, spec.size, p=p)
    return res
