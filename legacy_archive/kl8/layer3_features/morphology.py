"""宏观空间形态与组合统计特征（契约 R22、§3.9、裁决 A4）。

实现特征：
- 四区/八区分布、奇偶比、质合比、大中小分布
- 连号统计（二连号、三连号、最大连号长度）
- 同尾号统计（模 10 同余组数）
- 和值、跨度、和值偏离度 Z-Score
- 算术复杂度 AC 值（裁决 A4: |{两两正差集}| - (k - 1)）
- 空间均衡度校验 is_balanced
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from kl8.config import PoolConfig
from kl8.constants import (
    N_NUM,
    PRIMES_80,
    get_prime_label,
    size_class_of,
    tail_of,
    zone4_of,
    zone8_of,
)
from kl8.errors import DataFormatError


@dataclass(frozen=True)
class MorphologyStats:
    """宏观空间形态特征快照（契约 §4.7）。"""

    n: int
    zone4: Mapping[int, int]
    zone8: Mapping[int, int]
    odd: int
    even: int
    prime: int
    composite: int
    size: Mapping[str, int]
    consecutive: Mapping[str, int]
    same_tail: Mapping[int, int]
    sum_value: int
    sum_z: float
    span: int
    ac: int


def assert_valid_numbers(nums: Sequence[int]) -> None:
    """校验号码序列的合法性（1..80 且无重复），非法抛 DataFormatError。"""
    seen: set[int] = set()
    for x in nums:
        if not isinstance(x, (int,)):
            raise DataFormatError(f"号码必须为整数，收到: {type(x)}")
        if x < 1 or x > N_NUM:
            raise DataFormatError(f"号码越界: {x}，必须在 1..{N_NUM} 范围内")
        if x in seen:
            raise DataFormatError(f"号码序列中存在重复号码: {x}")
        seen.add(x)


def zone_distribution(nums: Sequence[int], *, scheme: str = "zone4") -> dict[int, int]:
    """统计四区或八区分布个数。"""
    if scheme == "zone4":
        dist = {i: 0 for i in range(4)}
        for x in nums:
            dist[zone4_of(x)] += 1
        return dist
    elif scheme == "zone8":
        dist = {i: 0 for i in range(8)}
        for x in nums:
            dist[zone8_of(x)] += 1
        return dist
    else:
        raise ValueError(f"未知分区方案: {scheme}，仅支持 'zone4' 或 'zone8'")


def odd_even_ratio(nums: Sequence[int]) -> tuple[int, int]:
    """统计奇偶比 (odd, even)。"""
    odd = sum(1 for x in nums if x % 2 == 1)
    return (odd, len(nums) - odd)


def prime_composite_ratio(nums: Sequence[int]) -> tuple[int, int, int]:
    """统计质合比与数字1个数 (prime, composite, count_of_1)。"""
    p = sum(1 for x in nums if x in PRIMES_80)
    c1 = sum(1 for x in nums if x == 1)
    comp = len(nums) - p - c1
    return (p, comp, c1)


def size_ratio(nums: Sequence[int]) -> dict[str, int]:
    """统计大中小比例（0=小 1..27，1=中 28..54，2=大 55..80）。"""
    res = {"小": 0, "中": 0, "大": 0}
    for x in nums:
        cls_idx = size_class_of(x)
        if cls_idx == 0:
            res["小"] += 1
        elif cls_idx == 1:
            res["中"] += 1
        else:
            res["大"] += 1
    return res


def consecutive_groups(nums: Sequence[int]) -> dict[str, int]:
    """统计连号特征（二连号组数、三连号组数、最大连续长度）。"""
    if len(nums) <= 1:
        return {"二连": 0, "三连": 0, "max_run": len(nums)}

    sorted_nums = sorted(nums)
    runs: list[int] = []
    current_len = 1

    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] == sorted_nums[i - 1] + 1:
            current_len += 1
        else:
            runs.append(current_len)
            current_len = 1
    runs.append(current_len)

    two_cons = sum(1 for r in runs if r == 2)
    three_cons = sum(1 for r in runs if r >= 3)
    max_run = max(runs) if runs else 1

    return {"二连": two_cons, "三连": three_cons, "max_run": max_run}


def same_tail_groups(nums: Sequence[int]) -> dict[int, int]:
    """统计同尾号分布（尾号 0..9 各自出现的球号个数）。"""
    tails: dict[int, int] = {t: 0 for t in range(10)}
    for x in nums:
        tails[tail_of(x)] += 1
    return tails


def sum_value(nums: Sequence[int]) -> int:
    """计算号码序列和值。"""
    return sum(nums)


def sum_deviation_z(nums: Sequence[int], *, n_pick: int | None = None) -> float:
    """计算和值相对于不放回理论期望的标准偏离 Z-Score。"""
    k = len(nums) if n_pick is None else n_pick
    if k <= 0:
        return 0.0
    actual_sum = float(sum(nums))
    expected = float(k) * 40.5
    # 有限总体方差修正
    n_pop = float(N_NUM)
    var = float(k) * ((n_pop**2 - 1.0) / 12.0) * ((n_pop - float(k)) / (n_pop - 1.0))
    std = math.sqrt(max(1e-9, var))
    return (actual_sum - expected) / std


def ac_value(nums: Sequence[int]) -> int:
    """计算算术复杂度 AC 值（契约裁决 A4、8.md L118、§6.2 手算对照）。

    公式：
    AC = |{两两不同正差集}| - (k - 1)
    @param nums 号码序列
    @return 整数 AC 值
    """
    k = len(nums)
    if k <= 1:
        return 0

    sorted_nums = sorted(nums)
    diffs: set[int] = set()
    for i in range(k):
        for j in range(i + 1, k):
            diffs.add(sorted_nums[j] - sorted_nums[i])

    return max(0, len(diffs) - (k - 1))


def morphology_stats(nums: Sequence[int]) -> MorphologyStats:
    """计算完整的宏观空间形态统计。"""
    assert_valid_numbers(nums)
    k = len(nums)
    z4 = zone_distribution(nums, scheme="zone4")
    z8 = zone_distribution(nums, scheme="zone8")
    odd, even = odd_even_ratio(nums)
    prime, composite, _ = prime_composite_ratio(nums)
    size_map = size_ratio(nums)
    cons = consecutive_groups(nums)
    same_tails = same_tail_groups(nums)
    sv = sum_value(nums)
    sz = sum_deviation_z(nums)
    span = max(nums) - min(nums) if k > 0 else 0
    ac = ac_value(nums)

    return MorphologyStats(
        n=k,
        zone4=z4,
        zone8=z8,
        odd=odd,
        even=even,
        prime=prime,
        composite=composite,
        size=size_map,
        consecutive=cons,
        same_tail=same_tails,
        sum_value=sv,
        sum_z=sz,
        span=span,
        ac=ac,
    )


def is_balanced(nums: Sequence[int], cfg: PoolConfig) -> tuple[bool, tuple[str, ...]]:
    """校验号码池是否满足形态硬约束（契约 R34）。

    硬约束：
    1. 单一四分区占比 <= cfg.max_zone4_ratio (0.45)
    2. 奇数或偶数单边占比 <= cfg.max_parity_ratio (0.70)
    3. 同尾号个数 <= cfg.max_same_tail (3)
    @param nums 号码池
    @param cfg 号码池约束配置
    @return (是否均衡通过, 未通过原因元组)
    """
    k = len(nums)
    if k == 0:
        return (True, ())

    reasons: list[str] = []
    # 1. 四区约束
    z4 = zone_distribution(nums, scheme="zone4")
    for z_idx, count in z4.items():
        ratio = count / float(k)
        if ratio > cfg.max_zone4_ratio + 1e-9:
            reasons.append(f"第 {z_idx + 1} 区号码占比 {ratio:.1%} > 上限 {cfg.max_zone4_ratio:.1%}")

    # 2. 奇偶单边约束
    odd, even = odd_even_ratio(nums)
    max_parity = max(odd, even)
    parity_ratio = max_parity / float(k)
    if parity_ratio > cfg.max_parity_ratio + 1e-9:
        label = "奇数" if odd > even else "偶数"
        reasons.append(f"{label}单边占比 {parity_ratio:.1%} > 上限 {cfg.max_parity_ratio:.1%}")

    # 3. 同尾约束
    tails = same_tail_groups(nums)
    for tail_digit, count in tails.items():
        if count > cfg.max_same_tail:
            reasons.append(f"尾数 {tail_digit} 包含 {count} 个号码 > 上限 {cfg.max_same_tail}")

    return (len(reasons) == 0, tuple(reasons))
