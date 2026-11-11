"""快乐8 随机性基准与局部偏态探测层（Layer 2 Baseline）。

导出基准先验、Z-Score、游程检验、卡方检验、NIST 轻量套件与综合随机性报告。
"""

from __future__ import annotations

from kl8.layer2_baseline.chisq import ChiSquareResult, chi_square_by_window, chi_square_gof
from kl8.layer2_baseline.entropy import (
    RandomnessReport,
    feature_redundancy,
    freq_entropy,
    mi_lag1,
    randomness_report,
    shannon_entropy,
)
from kl8.layer2_baseline.priors import (
    BaselineStats,
    expected_repeat,
    hypergeometric_baseline,
    p0,
    p_co,
    posterior_beta,
)
from kl8.layer2_baseline.randomness import (
    NistResult,
    approximate_entropy,
    flatten_history,
    frequency_within_block,
    monobit,
    nist_lite,
    runs_nist,
)
from kl8.layer2_baseline.runs import RunsStat, runs_test, runs_test_80
from kl8.layer2_baseline.zscore import significant_mask, window_z, z_all_windows

__all__ = [
    "BaselineStats",
    "ChiSquareResult",
    "NistResult",
    "RandomnessReport",
    "RunsStat",
    "approximate_entropy",
    "chi_square_by_window",
    "chi_square_gof",
    "expected_repeat",
    "feature_redundancy",
    "flatten_history",
    "freq_entropy",
    "frequency_within_block",
    "hypergeometric_baseline",
    "mi_lag1",
    "monobit",
    "nist_lite",
    "p0",
    "p_co",
    "posterior_beta",
    "randomness_report",
    "runs_nist",
    "runs_test",
    "runs_test_80",
    "shannon_entropy",
    "significant_mask",
    "window_z",
    "z_all_windows",
]
