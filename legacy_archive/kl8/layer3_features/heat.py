"""冷热动力学、动量差与遗漏高斯回补特征（契约 R11–R14、§3.9）。

实现特征：
- 四尺度热力频率 Freq_i(w) (R11)
- 跨周期动量差 Mom_10-25 与 Mom_25-50 及金叉加速判定 (R12)
- 当前遗漏 O 与历史平均遗漏 μ_gap (R13)
- 高斯核黄金回补压力 Pressure (R14)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.constants import N_NUM
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import WindowSpec, effective_window


@dataclass(frozen=True)
class HeatResult:
    """冷热动力学特征计算结果。"""

    freq: Mapping[str, np.ndarray]
    mom_10_25: np.ndarray
    mom_25_50: np.ndarray
    golden_cross: np.ndarray
    gap: np.ndarray
    gap_mean: np.ndarray
    gap_ratio: np.ndarray
    pressure: np.ndarray


def multi_scale_freq(
    state: BinaryStateMatrix,
    end: int,
    windows: Mapping[str, WindowSpec],
) -> dict[str, np.ndarray]:
    """计算各时间窗口内的号码出现频率 Freq_i(w) = c_i(w) / w_eff（契约 R11）。

    @param state 状态矩阵
    @param end 历史截断端点
    @param windows 窗口规范
    @return 键为 "10G","25G","50G","ALL"，值为 (80,) float64 频率
    """
    res: dict[str, np.ndarray] = {}
    for name, spec in windows.items():
        w_eff = effective_window(spec.size, end + 1)
        c_i = state.window_sum(end, spec.size).astype(np.float64)
        res[name] = c_i / float(max(1, w_eff))
    return res


def momentum(freq: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """计算跨周期动量差（契约 R12）。

    Mom_10-25 = Freq_10G - Freq_25G
    Mom_25-50 = Freq_25G - Freq_50G
    @param freq 四尺度频率字典
    @return (mom_10_25, mom_25_50)
    """
    f10 = freq["10G"]
    f25 = freq["25G"]
    f50 = freq["50G"]
    mom_10_25 = f10 - f25
    mom_25_50 = f25 - f50
    return (mom_10_25, mom_25_50)


def golden_cross(mom_10_25: np.ndarray, mom_25_50: np.ndarray) -> np.ndarray:
    """判定动量金叉加速标签（契约 R12）。

    条件：双正且短中差 > 中长差（m1 > 0 且 m2 > 0 且 m1 > m2）。
    @param mom_10_25 短中动量差
    @param mom_25_50 中长动量差
    @return (80,) bool
    """
    return (mom_10_25 > 0.0) & (mom_25_50 > 0.0) & (mom_10_25 > mom_25_50)


def gap_series(state: BinaryStateMatrix, end: int) -> np.ndarray:
    """获取当前截止各号码的连续遗漏期数 O_{t, i}（契约 R13）。

    @param state 状态矩阵
    @param end 历史截断端点
    @return (80,) int32
    """
    return state.gap_vector(end)


def historical_mean_gap(
    state: BinaryStateMatrix,
    end: int,
    *,
    floor: float = 3.0,
) -> np.ndarray:
    """计算各号码的历史平均遗漏间隔 μ_gap,i（契约 R13）。

    根据全量历史统计，开出频次 c 时的平均间隔约为 (end+1) / c。
    若某号码从未开出或间隔异常小，保底取 floor。
    @param state 状态矩阵
    @param end 历史截断端点
    @param floor 保底平均遗漏，默认 3.0
    @return (80,) float64
    """
    if state.is_empty() or end < 0:
        return np.full(N_NUM, floor, dtype=np.float64)

    c_all = state.window_sum(end, None).astype(np.float64)
    n_hist = float(end + 1)
    mean_gap = np.where(c_all > 0.0, n_hist / np.maximum(c_all, 1e-12), floor)
    return np.maximum(floor, mean_gap)


def omission_pressure(
    gap: np.ndarray,
    gap_mean: np.ndarray,
    *,
    center: float = 1.2,
    sigma: float = 0.6,
    floor: float = 3.0,
) -> np.ndarray:
    """高斯核黄金回补压力 Pressure（契约 R14）。

    公式：
    ratio = O_{t,i} / max(μ_gap,i, floor)
    Pressure_i = exp(-0.5 * ((ratio - center) / sigma)^2)
    @param gap (80,) 当前遗漏
    @param gap_mean (80,) 历史平均遗漏
    @param center 高斯核中心（默认 1.2）
    @param sigma 高斯核带宽（默认 0.6）
    @param floor 保底分母（默认 3.0）
    @return (80,) float64，数值限制在 [0.0, 1.0]
    """
    g = np.asarray(gap, dtype=np.float64)
    m = np.asarray(gap_mean, dtype=np.float64)
    safe_m = np.maximum(floor, m)
    ratio = g / safe_m
    z = (ratio - center) / sigma
    pressure = np.exp(-0.5 * (z**2))
    return np.clip(pressure, 0.0, 1.0)


def build_heat(
    state: BinaryStateMatrix,
    end: int,
    windows: Mapping[str, WindowSpec],
) -> HeatResult:
    """一次性计算组装冷热动力学全部特征。

    @param state 状态矩阵
    @param end 历史截断端点
    @param windows 窗口规范
    @return HeatResult
    """
    freq = multi_scale_freq(state, end, windows)
    mom_10_25, mom_25_50 = momentum(freq)
    gc = golden_cross(mom_10_25, mom_25_50)
    gap = gap_series(state, end)
    mean_gap = historical_mean_gap(state, end)
    safe_m = np.maximum(3.0, mean_gap)
    gap_ratio = gap.astype(np.float64) / safe_m
    pressure = omission_pressure(gap, mean_gap)

    return HeatResult(
        freq=freq,
        mom_10_25=mom_10_25,
        mom_25_50=mom_25_50,
        golden_cross=gc,
        gap=gap,
        gap_mean=mean_gap,
        gap_ratio=gap_ratio,
        pressure=pressure,
    )
