"""序列游程检验 Wald-Wolfowitz Runs Test（契约 R9、§3.8）。

数学模型：
针对单个号码的 0-1 历史出现序列，统计连续出现或连续遗漏的游程（Run）数量。
若实际游程数显著少于理论期望 (z < -1.96)，判定为「聚簇连出」；
若实际游程数显著多于理论期望 (z > 1.96)，判定为「频繁跳跃」；
其余为「无偏」；全 0 或全 1 为「退化」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kl8.constants import N_NUM, Z_SIG


@dataclass(frozen=True)
class RunsStat:
    """单个号码序列的游程统计。"""

    n: int
    n1: int
    n0: int
    runs: int
    expected: float
    variance: float
    z: float
    label: str  # "聚簇连出" | "频繁跳跃" | "无偏" | "退化"


def runs_test(series: np.ndarray, *, eps: float = 1e-9) -> RunsStat:
    """计算单个 0-1 序列的 Wald-Wolfowitz 游程统计量。

    @param series 1D 数组（0 或 1）
    @param eps 防除零极小值
    @return RunsStat 包含 n, n1, n0, runs, expected, variance, z, label
    """
    arr = np.asarray(series, dtype=np.int32)
    n = len(arr)
    if n <= 1:
        return RunsStat(
            n=n,
            n1=int(arr.sum()) if n == 1 else 0,
            n0=int((1 - arr).sum()) if n == 1 else 0,
            runs=1 if n == 1 else 0,
            expected=1.0,
            variance=0.0,
            z=0.0,
            label="退化",
        )

    n1 = int((arr == 1).sum())
    n0 = n - n1

    # 退化情况：全 0 或全 1
    if n1 == 0 or n0 == 0:
        return RunsStat(
            n=n,
            n1=n1,
            n0=n0,
            runs=1,
            expected=1.0,
            variance=0.0,
            z=0.0,
            label="退化",
        )

    # 统计游程数：相邻异号切换次数 + 1
    diffs = np.diff(arr)
    runs = int(np.count_nonzero(diffs)) + 1

    # 理论期望与方差
    expected = (2.0 * n1 * n0) / float(n) + 1.0
    numerator = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n)
    denominator = float(n**2 * (n - 1))
    variance = max(0.0, numerator / denominator) if denominator > 0 else 0.0

    if variance <= eps:
        z = 0.0
    else:
        z = (float(runs) - expected) / math.sqrt(variance)

    if z < -Z_SIG:
        label = "聚簇连出"
    elif z > Z_SIG:
        label = "频繁跳跃"
    else:
        label = "无偏"

    return RunsStat(
        n=n,
        n1=n1,
        n0=n0,
        runs=runs,
        expected=expected,
        variance=variance,
        z=z,
        label=label,
    )


def runs_test_80(
    window_matrix: np.ndarray,
    *,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """对矩阵中的全部 80 个号码批量执行游程检验。

    @param window_matrix (T, 80) 二值切片矩阵
    @param eps 防除零极小值
    @return (z: (80,), runs: (80,) int32, labels: tuple[str, ...] len=80)
    """
    z_arr = np.zeros(N_NUM, dtype=np.float64)
    runs_arr = np.zeros(N_NUM, dtype=np.int32)
    labels: list[str] = []

    for col in range(N_NUM):
        stat = runs_test(window_matrix[:, col], eps=eps)
        z_arr[col] = stat.z
        runs_arr[col] = stat.runs
        labels.append(stat.label)

    return (z_arr, runs_arr, tuple(labels))
