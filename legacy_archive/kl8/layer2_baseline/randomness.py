"""NIST SP 800-22 轻量级随机性测试套件（契约 R10、§3.8）。

提供 4 项核心测试：
1. Monobit（频数单比特检验）
2. Frequency Within Block（块内频数检验）
3. Runs（游程检验）
4. Approximate Entropy（近似熵检验）

【P8-M6 修复：偏置比特流口径】
本系统的输入不是等概率 0/1 比特流，而是「80 码是否开出」的偏置序列，单比特理论
出现率恒为 ``P0 = K_DRAW/N_NUM = 25%``。若直接套用 NIST 原文的 p=0.5 期望，Monobit/块内频数
会 100% 判定失败（属于"检验对象用错"，而非数据异常）。因此各项统计量统一按
``p0`` 参数化：零假设为「独立同分布 Bernoulli(p0)」，默认取业务理论值 ``P0``。

【适用性边界（必须在报告中同时披露）】
- 同一期内恰好开出 K_DRAW（20）个号码，位与位之间存在固定配额的负相关，四项检验只能作为
  "量级监控"，不能据此断言序列非随机；
- ApproximateEntropy 的自由度沿用 NIST 近似 ``2^(m-1)``，在偏置流下亦仅为近似口径。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kl8.constants import P0

#: 单项检验显著性阈值（契约 R10 固定 0.01）。
NIST_ALPHA: float = 0.01

#: 短序列保护门槛（低于该长度不做统计推断，直接判合格）。
NIST_MIN_LEN: int = 20

#: 近似熵检验的最小序列长度。
NIST_APEN_MIN_LEN: int = 32


@dataclass(frozen=True)
class NistResult:
    """NIST 随机性测试单项结果。"""

    name: str
    """检验项名称。"""

    statistic: float
    """检验统计量。"""

    p_value: float
    """p 值（越大越接近零假设）。"""

    passed: bool
    """是否通过（p ≥ alpha）。"""

    note: str
    """中文判定说明。"""


def _chi2_sf(stat: float, dof: int) -> float:
    """计算卡方生存函数 p 值（Wilson–Hilferty 正态近似，带兜底）。

    @param stat 卡方统计量
    @param dof 自由度
    @return p 值 ∈ [0, 1]
    """
    if stat <= 0.0 or dof <= 0:
        return 1.0
    factor = 2.0 / (9.0 * dof)
    z = ((stat / dof) ** (1.0 / 3.0) - (1.0 - factor)) / math.sqrt(factor)
    return float(max(0.0, min(1.0, 0.5 * math.erfc(z / math.sqrt(2.0)))))


def _check_p0(p0: float) -> float:
    """校验并归一化偏置概率参数。

    @param p0 单比特理论出现率
    @return 合法概率值
    @raises ValueError p0 不在 (0, 1) 区间
    """
    if not (0.0 < p0 < 1.0):
        raise ValueError(f"偏置概率 p0 必须位于 (0, 1)，实际收到 {p0}")
    return float(p0)


def flatten_history(hit: np.ndarray) -> np.ndarray:
    """按期号时序将 (T, 80) 二值矩阵展平为一维 0-1 比特流。

    @param hit (T, 80) 历史二值状态矩阵
    @return (T*80,) int8 比特流
    """
    return np.asarray(hit, dtype=np.int8).ravel()


def monobit(bits: np.ndarray, *, p0: float = P0) -> NistResult:
    """单比特频数检验（Monobit Test，按偏置率 p0 参数化）。

    统计量 ``S_obs = |ones - n·p0| / sqrt(n·p0·(1-p0))``，双侧 p 值 ``erfc(S_obs/√2)``。

    @param bits 1D 0-1 比特数组
    @param p0 单比特理论出现率（默认业务理论值 25%）
    @return NistResult
    """
    prob = _check_p0(p0)
    n = len(bits)
    if n < NIST_MIN_LEN:
        return NistResult("Monobit", 0.0, 1.0, True, f"序列长度不足 {NIST_MIN_LEN}")
    ones = float(np.count_nonzero(bits))
    expected = n * prob
    sd = math.sqrt(n * prob * (1.0 - prob))
    s_obs = abs(ones - expected) / sd
    p_val = math.erfc(s_obs / math.sqrt(2.0))
    passed = p_val >= NIST_ALPHA
    note = "通过（1 的占比符合理论偏置率）" if passed else "1 的占比偏离理论偏置率显著"
    return NistResult("Monobit", float(s_obs), float(p_val), passed, note)


def frequency_within_block(bits: np.ndarray, block_size: int = 128, *, p0: float = P0) -> NistResult:
    """块内频数检验（Frequency Test within a Block，按偏置率 p0 参数化）。

    统计量 ``chi2 = Σ (π_i - p0)² · M / (p0·(1-p0))``，自由度 = 块数。

    @param bits 1D 0-1 比特数组
    @param block_size 每块比特数
    @param p0 单比特理论出现率
    @return NistResult
    """
    prob = _check_p0(p0)
    n = len(bits)
    if n < block_size:
        return NistResult("BlockFrequency", 0.0, 1.0, True, f"序列长度不足块大小 {block_size}")

    num_blocks = n // block_size
    trimmed_len = num_blocks * block_size
    blocks = np.asarray(bits[:trimmed_len]).reshape(num_blocks, block_size)
    proportions = np.mean(blocks, axis=1)
    chi2 = float(np.sum((proportions - prob) ** 2) * block_size / (prob * (1.0 - prob)))
    p_val = _chi2_sf(chi2, num_blocks)
    passed = p_val >= NIST_ALPHA
    note = "通过（块内占比符合理论偏置率）" if passed else "块内占比分布偏离理论偏置率"
    return NistResult("BlockFrequency", chi2, p_val, passed, note)


def runs_nist(bits: np.ndarray, *, p0: float = P0) -> NistResult:
    """NIST 游程检验（Runs Test，按偏置率 p0 参数化）。

    游程数期望 ``E[V] = 1 + 2(n-1)·p0·(1-p0)``，
    方差 ``Var[V] = 2n·p0·q0·(2n·p0·q0 - 1) / (2n - 1)``，
    统计量 ``z = |V - E| / sqrt(Var)``，双侧 p 值 ``erfc(|z|/√2)``。

    @param bits 1D 0-1 比特数组
    @param p0 单比特理论出现率
    @return NistResult
    """
    prob = _check_p0(p0)
    n = len(bits)
    if n < NIST_MIN_LEN:
        return NistResult("Runs", 0.0, 1.0, True, f"序列长度不足 {NIST_MIN_LEN}")

    q = 1.0 - prob
    observed_pi = float(np.mean(bits))
    # 前置检验：实际占比若严重偏离理论偏置率，游程检验无意义
    if abs(observed_pi - prob) >= (2.0 / math.sqrt(float(n))):
        return NistResult("Runs", 0.0, 0.0, False, "1 的占比严重偏离理论偏置率，前置检验失败")

    diffs = np.diff(np.asarray(bits, dtype=np.int8))
    v_n = float(np.count_nonzero(diffs)) + 1.0
    expected = 1.0 + 2.0 * (n - 1) * prob * q
    base = 2.0 * n * prob * q
    variance = base * (base - 1.0) / (2.0 * n - 1.0)
    if variance <= 1e-12:
        return NistResult("Runs", v_n, 1.0, True, "理论方差接近 0")
    z_stat = abs(v_n - expected) / math.sqrt(variance)
    p_val = math.erfc(z_stat / math.sqrt(2.0))
    passed = p_val >= NIST_ALPHA
    note = "通过（游程总数符合理论期望）" if passed else "游程总数偏离理论期望"
    return NistResult("Runs", v_n, float(p_val), passed, note)


def approximate_entropy(bits: np.ndarray, m: int = 2, *, p0: float = P0) -> NistResult:
    """近似熵检验（Approximate Entropy Test，按偏置率 p0 计算零假设熵）。

    零假设熵 ``H0 = -p0·ln p0 - (1-p0)·ln(1-p0)``（原文 p0=0.5 时退化为 ln2），
    统计量 ``chi2 = 2n·(H0 + apen)``，其中 ``apen = φ(m) - φ(m+1)`` 为负值。

    @param bits 1D 0-1 比特数组
    @param m 模式长度
    @param p0 单比特理论出现率
    @return NistResult
    """
    prob = _check_p0(p0)
    n = len(bits)
    if n < NIST_APEN_MIN_LEN:
        return NistResult("ApproximateEntropy", 0.0, 1.0, True, "序列过短")

    def _phi(block_len: int) -> float:
        padded = np.concatenate([bits, bits[: block_len - 1]])
        counts: dict[tuple[int, ...], int] = {}
        for i in range(n):
            pattern = tuple(int(x) for x in padded[i : i + block_len])
            counts[pattern] = counts.get(pattern, 0) + 1
        phi = 0.0
        for count in counts.values():
            p_block = float(count) / float(n)
            phi += p_block * math.log(p_block)
        return phi

    apen = _phi(m) - _phi(m + 1)
    h0 = -(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob))
    chi2 = float(2.0 * n * (h0 + apen))
    dof = 1 << (m - 1)
    p_val = _chi2_sf(max(0.0, chi2), dof)
    passed = p_val >= NIST_ALPHA
    note = "通过（局部模式复杂度符合理论熵）" if passed else "局部模式复杂度不足"
    return NistResult("ApproximateEntropy", apen, p_val, passed, note)


def nist_lite(
    bits: np.ndarray,
    *,
    block_size: int = 128,
    apen_m: int = 2,
    alpha: float = NIST_ALPHA,
    p0: float = P0,
) -> tuple[NistResult, ...]:
    """执行 NIST-lite 固定 4 项测试套件（统一按偏置率 p0 口径）。

    @param bits 1D 比特数组（应为「号码是否开出」的偏置历史流）
    @param block_size 块大小
    @param apen_m 近似熵模式长度
    @param alpha 显著性阈值（默认 0.01；仅作结果判定展示用）
    @param p0 单比特理论出现率（默认业务理论值 25%）
    @return (Monobit, BlockFrequency, Runs, ApproximateEntropy) 固定元组
    """
    arr = np.asarray(bits, dtype=np.int8)
    res_mono = monobit(arr, p0=p0)
    res_block = frequency_within_block(arr, block_size=block_size, p0=p0)
    res_runs = runs_nist(arr, p0=p0)
    res_apen = approximate_entropy(arr, m=apen_m, p0=p0)
    if alpha != NIST_ALPHA:  # 显式保留参数语义：阈值变化需同步判定口径
        results = (res_mono, res_block, res_runs, res_apen)
        return tuple(
            NistResult(item.name, item.statistic, item.p_value, item.p_value >= alpha, item.note)
            for item in results
        )
    return (res_mono, res_block, res_runs, res_apen)