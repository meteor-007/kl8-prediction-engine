"""卡方拟合优度检验（契约 R10、§3.8）。

评估 80 个号码在给定窗口内的频次分布是否与均匀随机分布产生显著偏离。
自由度 dof = 80 - 1 = 79。
优先使用 scipy 计算生存函数 sf，无 scipy 时使用 Wilson-Hilferty 正态近似。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.constants import N_NUM, P0
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import WindowSpec

try:
    from scipy.stats import chi2 as scipy_chi2  # type: ignore[import-untyped]
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass(frozen=True)
class ChiSquareResult:
    """卡方拟合优度检验结果。"""

    statistic: float
    dof: int
    p_value: float
    expected: np.ndarray
    observed: np.ndarray
    method: str  # "scipy" | "wilson_hilferty" | "degenerate"


def _wilson_hilferty_p_value(stat: float, dof: int) -> float:
    """使用 Wilson-Hilferty 近似计算卡方分布的上尾 p 值。"""
    if stat <= 0.0 or dof <= 0:
        return 1.0
    factor = 2.0 / (9.0 * dof)
    z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - factor)) / math.sqrt(factor)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def chi_square_gof(observed: np.ndarray, w: int, *, p: float = P0) -> ChiSquareResult:
    """计算 80 码频次的卡方拟合优度检验。

    @param observed 长度为 80 的观测频次数组
    @param w 滑动窗口期数
    @param p 理论期望出现率，默认 0.25
    @return ChiSquareResult 包含统计量、自由度、p值、期望频次、方法
    """
    obs = np.asarray(observed, dtype=np.float64)
    if len(obs) != N_NUM:
        raise ValueError(f"观测数组长度必须为 {N_NUM}，当前为 {len(obs)}")

    if w <= 0:
        exp = np.zeros(N_NUM, dtype=np.float64)
        return ChiSquareResult(
            statistic=0.0,
            dof=N_NUM - 1,
            p_value=1.0,
            expected=exp,
            observed=obs,
            method="degenerate",
        )

    expected_val = float(w) * p
    expected = np.full(N_NUM, expected_val, dtype=np.float64)
    dof = N_NUM - 1

    stat = float(np.sum((obs - expected) ** 2 / expected_val))

    if _HAS_SCIPY:
        p_val = float(scipy_chi2.sf(stat, dof))
        method = "scipy"
    else:
        p_val = _wilson_hilferty_p_value(stat, dof)
        method = "wilson_hilferty"

    return ChiSquareResult(
        statistic=stat,
        dof=dof,
        p_value=max(0.0, min(1.0, p_val)),
        expected=expected,
        observed=obs,
        method=method,
    )


def chi_square_by_window(
    state: BinaryStateMatrix,
    end: int,
    windows: Mapping[str, WindowSpec],
) -> dict[str, ChiSquareResult]:
    """对所有窗口分别执行卡方拟合优度检验。

    @param state 状态矩阵
    @param end 历史截断端点
    @param windows 窗口规范字典
    @return 键为窗口名称，值为 ChiSquareResult
    """
    res: dict[str, ChiSquareResult] = {}
    for name, spec in windows.items():
        if spec.size is None:
            w_eff = end + 1
        else:
            w_eff = min(spec.size, end + 1)
        obs = state.window_sum(end, spec.size)
        res[name] = chi_square_gof(obs, w_eff)
    return res
