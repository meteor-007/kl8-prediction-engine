"""随机性基准层：理论先验与超几何分布基准（契约 R5、R6、R7、§3.8）。

实现功能：
- 单球单期理论开出概率先验 p0 = K / N = 0.25 (R5)
- 任意两球同期共现理论概率 p_co = K(K-1) / (N(N-1)) ≈ 0.060127 (R6)
- 相邻两期理论重号（落球）个数期望 E(Repeat) = K * p0 = 5.0 (R7)
- 超几何基准分布与 Beta 后验估计
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.constants import K_DRAW, N_NUM, P0, P_CO, hypergeometric_mean, hypergeometric_pmf, hypergeometric_var


@dataclass(frozen=True)
class BaselineStats:
    """超几何理论基准统计。"""

    n_pick: int
    mean: float
    sd: float
    pmf: np.ndarray
    p_ge: Mapping[int, float]


def p0(n_draw: int = K_DRAW, n_total: int = N_NUM) -> float:
    """单球单期理论开出概率先验（契约 R5）。

    @param n_draw 每期开奖球数，默认 20
    @param n_total 总球数，默认 80
    @return 理论开出概率，默认 0.25
    """
    if n_total <= 0:
        raise ValueError("总球数 n_total 必须大于 0")
    return float(n_draw) / float(n_total)


def p_co(n_draw: int = K_DRAW, n_total: int = N_NUM) -> float:
    """任意两球同期共现理论概率（契约 R6）。

    @param n_draw 每期开奖球数，默认 20
    @param n_total 总球数，默认 80
    @return 理论同期共现概率，默认 20*19/(80*79) ≈ 0.06012658
    """
    if n_total < 2 or n_draw < 2:
        raise ValueError("计算共现概率要求 n_total >= 2 且 n_draw >= 2")
    return float(n_draw * (n_draw - 1)) / float(n_total * (n_total - 1))


def expected_repeat(n_draw: int = K_DRAW, p: float = P0) -> float:
    """相邻两期理论重号（落球）个数期望（契约 R7）。

    @param n_draw 每期开奖球数，默认 20
    @param p 先验开出概率，默认 0.25
    @return 期望重号个数，默认 5.0
    """
    return float(n_draw) * float(p)


def hypergeometric_baseline(n_pick: int, n_draw: int = K_DRAW, n_total: int = N_NUM) -> BaselineStats:
    """计算选 n_pick 码时的超几何理论基准分布与累积中奖概率。

    @param n_pick 选号个数 (1 <= n_pick <= n_total)
    @param n_draw 开奖个数，默认 20
    @param n_total 总号码数，默认 80
    @return BaselineStats 包含均值、标准差、PMF 与 >=k 概率
    """
    if n_pick < 0 or n_pick > n_total:
        raise ValueError(f"选号数 n_pick={n_pick} 必须在 0..{n_total} 之间")
    pmf = hypergeometric_pmf(n_pick, n_draw, n_total)
    mean = hypergeometric_mean(n_pick, n_draw, n_total)
    var = hypergeometric_var(n_pick, n_draw, n_total)
    sd = math.sqrt(max(0.0, var))

    # 计算 P(X >= k)
    p_ge: dict[int, float] = {}
    cum = 0.0
    for k in range(n_pick, -1, -1):
        cum += float(pmf[k])
        p_ge[k] = min(1.0, cum)

    return BaselineStats(
        n_pick=n_pick,
        mean=mean,
        sd=sd,
        pmf=pmf,
        p_ge=p_ge,
    )


def posterior_beta(success: int, trials: int, *, alpha: float = 1.0, beta: float = 1.0) -> tuple[float, float, float]:
    """Beta-二项后验估计。

    @param success 成功次数（开出次数）
    @param trials 试验次数（期数）
    @param alpha Beta 先验 alpha 参数
    @param beta Beta 先验 beta 参数
    @return (后验均值, 95% 置信区间下界近似, 95% 置信区间上界近似)
    """
    a_post = alpha + success
    b_post = beta + (trials - success)
    mean = a_post / (a_post + b_post)
    var = (a_post * b_post) / (((a_post + b_post) ** 2) * (a_post + b_post + 1.0))
    sd = math.sqrt(max(0.0, var))
    ci_low = max(0.0, mean - 1.96 * sd)
    ci_high = min(1.0, mean + 1.96 * sd)
    return (mean, ci_low, ci_high)
