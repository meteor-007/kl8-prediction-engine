"""NIST SP 800-22 轻量级随机性测试套件（契约 R10、§3.8）。

提供 4 项核心测试：
1. Monobit (频数单比特检验)
2. Frequency Within Block (块内频数检验)
3. Runs (游程检验)
4. Approximate Entropy (近似熵检验)
注意：这些测试仅用于序列物理随机性基线诊断，不直接参与预测打分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NistResult:
    """NIST 随机性测试单项结果。"""

    name: str
    statistic: float
    p_value: float
    passed: bool
    note: str


def _chi2_sf(stat: float, dof: int) -> float:
    """计算卡方生存函数 p 值（带兜底）。"""
    if stat <= 0.0 or dof <= 0:
        return 1.0
    factor = 2.0 / (9.0 * dof)
    z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - factor)) / math.sqrt(factor)
    return float(max(0.0, min(1.0, 0.5 * math.erfc(z / math.sqrt(2.0)))))


def flatten_history(hit: np.ndarray) -> np.ndarray:
    """按期号时序将 (T, 80) 二值矩阵展平为一维 0-1 比特流。

    @param hit (T, 80) 历史二值状态矩阵
    @return (T*80,) int8 比特流
    """
    return np.asarray(hit, dtype=np.int8).ravel()


def monobit(bits: np.ndarray) -> NistResult:
    """单比特频数检验 (Monobit Test)。"""
    n = len(bits)
    if n < 20:
        return NistResult("Monobit", 0.0, 1.0, True, "序列长度不足 20")
    s_n = float(np.sum(2 * bits - 1))
    s_obs = abs(s_n) / math.sqrt(float(n))
    p_val = math.erfc(s_obs / math.sqrt(2.0))
    passed = p_val >= 0.01
    note = "通过" if passed else "0/1 频数偏离显著"
    return NistResult("Monobit", float(s_obs), float(p_val), passed, note)


def frequency_within_block(bits: np.ndarray, block_size: int = 128) -> NistResult:
    """块内频数检验 (Frequency Test within a Block)。"""
    n = len(bits)
    if n < block_size:
        return NistResult("BlockFrequency", 0.0, 1.0, True, f"序列长度不足块大小 {block_size}")

    num_blocks = n // block_size
    trimmed_len = num_blocks * block_size
    blocks = bits[:trimmed_len].reshape(num_blocks, block_size)
    proportions = np.mean(blocks, axis=1)

    chi2 = float(4.0 * block_size * np.sum((proportions - 0.5) ** 2))
    p_val = _chi2_sf(chi2, num_blocks)
    passed = p_val >= 0.01
    note = "通过" if passed else "块内占比分布不均"
    return NistResult("BlockFrequency", chi2, p_val, passed, note)


def runs_nist(bits: np.ndarray) -> NistResult:
    """NIST 游程检验 (Runs Test)。"""
    n = len(bits)
    if n < 20:
        return NistResult("Runs", 0.0, 1.0, True, "序列长度不足 20")

    pi = float(np.mean(bits))
    # 前置判断：若 1 的占比严重偏离 0.5，则预检验不通过
    if abs(pi - 0.5) >= (2.0 / math.sqrt(float(n))):
        return NistResult("Runs", 0.0, 0.0, False, "1的占比严重失衡，前置检验失败")

    diffs = np.diff(bits)
    v_n = float(np.count_nonzero(diffs)) + 1.0
    expected = 2.0 * n * pi * (1.0 - pi)
    denominator = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    if denominator <= 1e-12:
        return NistResult("Runs", 0.0, 1.0, True, "方差接近 0")

    num = abs(v_n - expected) / denominator
    p_val = math.erfc(num)
    passed = p_val >= 0.01
    note = "通过" if passed else "游程总数偏离期望"
    return NistResult("Runs", v_n, float(p_val), passed, note)


def approximate_entropy(bits: np.ndarray, m: int = 2) -> NistResult:
    """近似熵检验 (Approximate Entropy Test)。"""
    n = len(bits)
    if n < 32:
        return NistResult("ApproximateEntropy", 0.0, 1.0, True, "序列过短")

    def _phi(block_len: int) -> float:
        padded = np.concatenate([bits, bits[: block_len - 1]])
        counts: dict[tuple[int, ...], int] = {}
        for i in range(n):
            pattern = tuple(padded[i : i + block_len])
            counts[pattern] = counts.get(pattern, 0) + 1
        phi = 0.0
        for count in counts.values():
            prob = float(count) / float(n)
            phi += prob * math.log(prob)
        return phi

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    apen = phi_m - phi_m1
    chi2 = float(2.0 * n * (math.log(2.0) - apen))
    dof = 1 << (m - 1)
    p_val = _chi2_sf(max(0.0, chi2), dof)
    passed = p_val >= 0.01
    note = "通过" if passed else "局部模式复杂度不足"
    return NistResult("ApproximateEntropy", apen, p_val, passed, note)


def nist_lite(
    bits: np.ndarray,
    *,
    block_size: int = 128,
    apen_m: int = 2,
    alpha: float = 0.01,
) -> tuple[NistResult, ...]:
    """执行 NIST-lite 固定 4 项测试套件。

    @param bits 1D 比特数组
    @param block_size 块大小
    @param apen_m 近似熵模式长度
    @param alpha 显著性阈值（默认 0.01）
    @return (Monobit, BlockFrequency, Runs, ApproximateEntropy) 固定元组
    """
    arr = np.asarray(bits, dtype=np.int8)
    res_mono = monobit(arr)
    res_block = frequency_within_block(arr, block_size=block_size)
    res_runs = runs_nist(arr)
    res_apen = approximate_entropy(arr, m=apen_m)
    return (res_mono, res_block, res_runs, res_apen)
