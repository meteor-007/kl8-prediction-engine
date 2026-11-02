# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 核心数据结构与时序抽象 (纯开奖数据驱动，零点位依赖)

【业务背景与架构定位】:
本模块专为快乐8「微积分八区分区预测」提供基础数据层。
完全解耦点位信息，100% 只读取官方真实开奖历史数据，
提供 Dataset 容器、Draw 实体以及时序滑动矩阵 TimeSeries。

@author czw
@since 2026-10-18
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import numpy as np

# 快乐8号码总数 (1..80)
N_NUM = 80
DEFAULT_HISTORY_FILE = "kl8_history.txt"
OUTPUT_DIR = "output"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"
WECHAT_DIR = f"{OUTPUT_DIR}/wechat"
EXCEL_DIR = f"{OUTPUT_DIR}/excel"

# -----------------------------------------------------------------------------
# 012路与质合数基础数学定义 (1..80 标准定义)
# -----------------------------------------------------------------------------
# 80以内质数集合 (严格数学定义，共22个，1既非质数亦非合数)
PRIMES_80: frozenset[int] = frozenset({
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79
})

# 012路号码划分 (模3同余集合)
ROAD_0_NUMS: frozenset[int] = frozenset(x for x in range(1, N_NUM + 1) if x % 3 == 0)  # 26个 (3, 6, 9...78)
ROAD_1_NUMS: frozenset[int] = frozenset(x for x in range(1, N_NUM + 1) if x % 3 == 1)  # 27个 (1, 4, 7...79)
ROAD_2_NUMS: frozenset[int] = frozenset(x for x in range(1, N_NUM + 1) if x % 3 == 2)  # 27个 (2, 5, 8...80)

# 单期20码的理论期望出号数 (基于均匀无放回超几何/二项分布逼近)
EXP_ROAD_0_PER_DRAW = 20.0 * (len(ROAD_0_NUMS) / float(N_NUM))  # 20 * 26 / 80 = 6.50
EXP_ROAD_1_PER_DRAW = 20.0 * (len(ROAD_1_NUMS) / float(N_NUM))  # 20 * 27 / 80 = 6.75
EXP_ROAD_2_PER_DRAW = 20.0 * (len(ROAD_2_NUMS) / float(N_NUM))  # 20 * 27 / 80 = 6.75

# 16码推荐池的理论期望数 (超几何分布均值)
EXP_ROAD_0_IN_16 = 16.0 * (len(ROAD_0_NUMS) / float(N_NUM))  # 16 * 26 / 80 = 5.20
EXP_ROAD_1_IN_16 = 16.0 * (len(ROAD_1_NUMS) / float(N_NUM))  # 16 * 27 / 80 = 5.40
EXP_ROAD_2_IN_16 = 16.0 * (len(ROAD_2_NUMS) / float(N_NUM))  # 16 * 27 / 80 = 5.40
EXP_PRIME_IN_16 = 16.0 * (len(PRIMES_80) / float(N_NUM))     # 16 * 22 / 80 = 4.40

# 非替换抽样 (超几何) 标准差 σ = √( n·p·(1-p)·(N-n)/(N-1) )，用于把结构偏差换算成统计显著度
def _hypergeometric_sd(pool_size: int, picked: int = 16, total: int = N_NUM) -> float:
    """「从 80 码中随机选 16 码」时，某属性 (pool_size 个码) 被选中个数的标准差。"""
    p = pool_size / float(total)
    return (picked * p * (1.0 - p) * (total - picked) / float(total - 1)) ** 0.5


ROAD_EXPECTED_COUNT_IN_16: dict[int, float] = {0: EXP_ROAD_0_IN_16, 1: EXP_ROAD_1_IN_16, 2: EXP_ROAD_2_IN_16}
ROAD_SD_IN_16: dict[int, float] = {
    0: _hypergeometric_sd(len(ROAD_0_NUMS)),
    1: _hypergeometric_sd(len(ROAD_1_NUMS)),
    2: _hypergeometric_sd(len(ROAD_2_NUMS)),
}
PRIME_EXPECTED_COUNT_IN_16 = EXP_PRIME_IN_16
PRIME_SD_IN_16 = _hypergeometric_sd(len(PRIMES_80))


def get_road_012(num: int) -> int:
    """获取号码的 012 路归属 (除 3 余数: 0, 1, 2)。

    @param num 号码 (1..80)
    @return 0, 1 或 2
    """
    return num % 3


def is_prime_num(num: int) -> bool:
    """判定号码是否为质数 (标准数学定义，1 既非质数亦非合数)。

    @param num 号码 (1..80)
    @return True 为质数，False 为非质数
    """
    return num in PRIMES_80


def get_prime_label(num: int) -> str:
    """获取号码的质合分类标签。

    @param num 号码 (1..80)
    @return '质' (素数), '合' (合数), 或 '特1' (数字1)
    """
    if num == 1:
        return "特1"
    return "质" if num in PRIMES_80 else "合"


def get_num_prop_tag(num: int) -> str:
    """生成号码的紧凑属性标签 (如 '0·合', '1·质', '2·合', '1·特1')。

    @param num 号码 (1..80)
    @return 属性字符串
    """
    return f"{get_road_012(num)}·{get_prime_label(num)}"


def analyze_number_group(numbers: Iterable[int]) -> dict:
    """对一组号码 (如20个开奖号码或16码推荐清单) 进行全维正交形态统计与健康度评估。

    @param numbers 号码集合或列表 (1..80)
    @return 包含012路比、质合比、奇偶比、大小比及统计指标的结构化字典
    """
    nums = sorted(set(numbers))
    total = len(nums)
    if total == 0:
        return {
            "total": 0,
            "road_counts": {0: 0, 1: 0, 2: 0},
            "road_ratio": "0:0:0",
            "prime_count": 0,
            "composite_count": 0,
            "special_1": False,
            "prime_ratio": "0:0",
            "odd_count": 0,
            "even_count": 0,
            "odd_ratio": "0:0",
            "small_count": 0,
            "big_count": 0,
            "size_ratio": "0:0",
            "sum_value": 0,
            "span": 0,
        }

    # 步骤 1: 统计 012 路分布
    r0 = sum(1 for x in nums if x % 3 == 0)
    r1 = sum(1 for x in nums if x % 3 == 1)
    r2 = sum(1 for x in nums if x % 3 == 2)

    # 步骤 2: 统计 质合分布 (1 标注为特1，计入非质数项)
    primes = sum(1 for x in nums if x in PRIMES_80)
    has_one = 1 in nums
    composites = sum(1 for x in nums if (x not in PRIMES_80 and x != 1))

    # 步骤 3: 奇偶、大小、和值与跨度
    odds = sum(1 for x in nums if x % 2 == 1)
    evens = total - odds
    smalls = sum(1 for x in nums if x <= 40)
    bigs = total - smalls
    sum_val = sum(nums)
    span = nums[-1] - nums[0] if total > 1 else 0

    return {
        "total": total,
        "road_counts": {0: r0, 1: r1, 2: r2},
        "road_ratio": f"{r0}:{r1}:{r2}",
        "prime_count": primes,
        "composite_count": composites,
        "special_1": has_one,
        "prime_ratio": f"{primes}:{composites + (1 if has_one else 0)}",
        "odd_count": odds,
        "even_count": evens,
        "odd_ratio": f"{odds}:{evens}",
        "small_count": smalls,
        "big_count": bigs,
        "size_ratio": f"{smalls}:{bigs}",
        "sum_value": sum_val,
        "span": span,
    }


def hypergeometric_baseline(n_pick: int = 16, n_draw: int = 20, n_total: int = 80) -> dict:
    """计算「完全随机选 n_pick 码」时的精确命中分布 (超几何分布)，作为命中率评估的唯一公正基准。

    快乐8 每期从 80 码中开出 20 码，任何选号方案若不含真实信息量，其命中数
    必然服从超几何分布 H(80, 20, n_pick)。本函数给出该基准的均值、标准差、
    各门槛概率与单区命中期望，用于对算法战绩做显著性检验。

    @param n_pick 推荐号码个数
    @param n_draw 每期开出号码个数
    @param n_total 号码池总数
    @return 结构化基准字典
    """
    from math import comb

    total_ways = comb(n_total, n_draw)
    pmf = []
    for k in range(0, n_pick + 1):
        if n_draw - k < 0 or n_draw - k > n_total - n_pick:
            pmf.append(0.0)
            continue
        pmf.append(comb(n_pick, k) * comb(n_total - n_pick, n_draw - k) / total_ways)

    mean = sum(k * p for k, p in enumerate(pmf))
    var = sum((k - mean) ** 2 * p for k, p in enumerate(pmf))

    def p_at_least(threshold: int) -> float:
        return sum(pmf[k] for k in range(threshold, len(pmf)))

    # 每区选 2 码时，单个分区至少命中 1 码的概率 = 1 - C(78,20)/C(80,20)
    p_zone = 1.0 - comb(n_total - 2, n_draw) / total_ways

    return {
        "n_pick": n_pick,
        "mean": float(mean),
        "sd": float(var ** 0.5),
        "pmf": pmf,
        "p_ge": {t: p_at_least(t) for t in (3, 4, 5, 6, 7, 8)},
        "zone_hit_prob": float(p_zone),
        "zone_coverage_expected": float(p_zone * (n_pick / 2)),
    }


CONFIG = {
    # 8区分区定义：80号划分为8个连续区间，每区10码，每区精选2码
    "calc_zone_n": 8,
    "calc_zone_size": 10,
    # 微积分时间窗口 (短窗10期，中窗20期，长窗40期)
    "window_calc": 40,
    "window_calc_short": 10,
    "window_calc_mid": 20,
    # 遗漏上限截断与弹性控制
    "gap_cap": 15.0,
    # 012路正交防撞车与势能补偿参数
    "road_orthogonal_enabled": True,
    # 当跨路候选势能与最高势能差异在容忍阈值内时，触发正交跨路优选
    "road_orthogonal_tolerance": 0.15,
    # --- 动力学打分权重 (原为散落在引擎内的硬编码常量，统一收敛到配置便于审计与复现) ---
    # 动能分 = w_integral*密度 + w_velocity*速度 + w_accel*加速度 - 冷号惩罚
    "weight_integral": 0.50,
    "weight_velocity": 0.30,
    "weight_accel": 0.20,
    # 连失惩罚：遗漏 > penalty_gap 期的号码在动能分上扣 penalty_score (动能只取爆发态)
    "penalty_gap": 4.0,
    "penalty_score": 0.40,
    # 势能分中「正速度拐点」的附加权重
    "potential_velocity_bonus": 0.25,
    # 012 路均值回归调制的斜率与限幅 (偏离系数 = clip(-slope*bias, ±clamp))
    "road_elasticity_slope": 0.10,
    "road_elasticity_clamp": 0.10,
    # 区内双码的综合分融合权重 (动能/势能各占一半)
    "combine_weight_kinetic": 0.50,
    # --- 区态标签阈值 (短窗10期·10码密度均值，随机基准 0.25、σ≈0.0406，
    #     取 ±1σ 为热/冷分界，使三区标签分布大致为 热:平:冷 = 1:1:1) ---
    "zone_label_hot_above": 0.29,
    "zone_label_cold_below": 0.21,
}


@dataclass(frozen=True)
class Draw:
    """单期开奖记录实体。"""
    date: str
    period: int
    numbers: frozenset[int]
    idx: int = -1


@dataclass
class Dataset:
    """开奖历史数据集容器。"""
    draws: list[Draw] = field(default_factory=list)
    draws_by_period: dict[int, Draw] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def T(self) -> int:
        return len(self.draws)

    def draw_matrix(self) -> np.ndarray:
        """生成形状为 (T, 80) 的历史开出 0-1 矩阵。"""
        matrix = np.zeros((self.T, N_NUM), dtype=float)
        for i, d in enumerate(self.draws):
            for n in d.numbers:
                matrix[i, n - 1] = 1.0
        return matrix


class TimeSeries:
    """高效时序缓存结构：提供 hit 0-1 矩阵与遗漏矩阵 gap 的预计算容器。"""

    def __init__(self, ds: Dataset):
        self.T = ds.T
        self.hit = ds.draw_matrix()
        self.gap = np.zeros((self.T, N_NUM), dtype=np.int32)
        if self.T > 0:
            cur = np.zeros(N_NUM, dtype=np.int32)
            for t in range(self.T):
                cur = np.where(self.hit[t] > 0.5, 0, cur + 1)
                self.gap[t] = cur


def parse_history_file(path: str | Path) -> tuple[list[tuple[str, int, frozenset[int]]], list[str]]:
    """解析开奖历史文件。格式: date:YYYY-MM-DD,period:XXXXXXX,numbers:01-02-..."""
    rows = []
    warnings = []
    pat = re.compile(r"date:(\d{4}-\d{2}-\d{2}),period:(\d+),numbers:(.+)")
    p = Path(path)
    if not p.exists():
        warnings.append(f"开奖历史文件不存在: {path}")
        return rows, warnings

    with open(p, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                warnings.append(f"第 {ln} 行格式无法解析，跳过")
                continue
            date, period_s, nums_s = m.groups()
            try:
                nums = [int(x) for x in nums_s.replace(",", "-").split("-") if x.strip()]
                assert len(nums) == 20 and len(set(nums)) == 20
                assert all(1 <= x <= N_NUM for x in nums)
            except Exception:
                warnings.append(f"第 {ln} 行号码校验失败（需20个1-80不重复码），跳过")
                continue
            rows.append((date, int(period_s), frozenset(nums)))
    return rows, warnings


def load_dataset(history_path: str = DEFAULT_HISTORY_FILE) -> Dataset:
    """加载开奖历史数据并按时间升序构建 Dataset。"""
    rows, warnings = parse_history_file(history_path)
    draws = [Draw(date=d, period=p, numbers=ns) for d, p, ns in rows]
    # 按日期与期号升序 (旧 -> 新)
    draws.sort(key=lambda x: (x.date, x.period))
    for i, d in enumerate(draws):
        object.__setattr__(d, "idx", i)
    draws_by_period = {d.period: d for d in draws}
    return Dataset(draws=draws, draws_by_period=draws_by_period, warnings=warnings)


def truncated(ds: Dataset, target_period: int) -> Dataset:
    """防前瞻泄漏截断：截取历史数据到 target_period 之前。"""
    if target_period in ds.draws_by_period:
        idx = ds.draws_by_period[target_period].idx
        sub_draws = ds.draws[:idx]
    else:
        # target_period 大于所有历史期（待开期）
        sub_draws = list(ds.draws)
    draws_by_p = {d.period: d for d in sub_draws}
    return Dataset(draws=sub_draws, draws_by_period=draws_by_p, warnings=list(ds.warnings))
