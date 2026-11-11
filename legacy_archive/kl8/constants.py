# -*- coding: utf-8 -*-
"""快乐8五层重构 —— 全局数学恒等式与号码静态属性（C1：零魔法数字的唯一归宿之一）。

【定位与边界】
本模块**只允许**放两类东西：

1. **数学恒等式**：由 ``N_NUM=80`` 与 ``K_DRAW=20`` 推导出来的概率/期望常量
   （``P0``、``P_CO``、``E_REPEAT``、``SUM_EXPECT`` 等），它们是快乐8玩法的
   定义性事实，不随业务调参变化；
2. **号码静态属性**：质数集合、012 路集合、四区/八区/大中小分界、尾号、
   超几何分布计算等纯函数。

**业务阈值/权重/窗口/上限一律进** ``kl8/config.py``（如 α、λ、R1–R5 阈值、
覆盖预算、回测 warmup），不要写在本模块，否则违反 C1 且审大关会判 High。

【AC 值定式说明（供 layer3 morphology 实现，R22）】
AC 值 = "号码两两差值的不同取值个数" - (k - 1)，其中 k 为号码个数。
例：号码 {01, 04, 09} 的两两差值集合为 {3, 5, 8}（|集合|=3），AC = 3 - (3-1) = 1。
本模块只在此处记录定式，具体计算在 ``kl8/layer3_features/morphology.py::ac_value``。

【奖金表口径（裁决 A10）】
仅「选五」奖级已被用户确认：中 5 = 1000 元、中 4 = 21 元、中 3 = 3 元，单注 2 元。
「选三」「选七」奖级**尚未确认**，一律记为 ``None``，只做诊断展示、不参与资金结算，
避免把未核实的奖金写进 ROI 而误导结论。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import math
import re
from typing import Final, Mapping

import numpy as np

# =============================================================================
# 一、玩法定义性常量（数学恒等式，不随业务调参）
# =============================================================================
N_NUM: Final[int] = 80
"""号码总数 N（1..80）。"""

K_DRAW: Final[int] = 20
"""每期开出号码个数 K。"""

P0: Final[float] = K_DRAW / N_NUM
"""单球单期开出理论概率 p0 = K/N = 0.25。"""

P_CO: Final[float] = K_DRAW * (K_DRAW - 1) / (N_NUM * (N_NUM - 1))
"""任意两球同期共现理论概率 p_co = K(K-1)/N(N-1) ≈ 0.0601265823。"""

E_REPEAT: Final[float] = K_DRAW * P0
"""相邻两期理论重号（落球）个数期望 E(Repeat) = K·p0 = 5.0。"""

SUM_EXPECT: Final[int] = K_DRAW * (N_NUM + 1) // 2
"""单期 20 码理论期望和值 810（= K·(N+1)/2）。"""

Z_SIG: Final[float] = 1.96
"""双尾 5% 显著性的标准正态临界值（|Z| > 1.96 判显著偏态）。"""

# =============================================================================
# 二、号码静态属性集合（1..80 上的固定划分）
# =============================================================================
PRIMES_80: Final[frozenset[int]] = frozenset(
    {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79,
    }
)
"""80 以内质数，共 22 个；1 既非质数亦非合数（单独标注为「特1」）。"""

ROAD_0_NUMS: Final[frozenset[int]] = frozenset(x for x in range(1, N_NUM + 1) if x % 3 == 0)
"""012 路之 0 路（模 3 余 0），共 26 个。"""

ROAD_1_NUMS: Final[frozenset[int]] = frozenset(x for x in range(1, N_NUM + 1) if x % 3 == 1)
"""012 路之 1 路（模 3 余 1），共 27 个。"""

ROAD_2_NUMS: Final[frozenset[int]] = frozenset(x for x in range(1, N_NUM + 1) if x % 3 == 2)
"""012 路之 2 路（模 3 余 2），共 27 个。"""

ZONE4_BOUNDS: Final[tuple[tuple[int, int], ...]] = ((1, 20), (21, 40), (41, 60), (61, 80))
"""四分区边界（闭区间）：01-20 / 21-40 / 41-60 / 61-80。"""

ZONE8_BOUNDS: Final[tuple[tuple[int, int], ...]] = tuple(
    (idx * 10 + 1, idx * 10 + 10) for idx in range(8)
)
"""八分区边界（闭区间）：01-10 / 11-20 / … / 71-80。"""

SIZE_BOUNDS: Final[tuple[tuple[int, int], ...]] = ((1, 27), (28, 54), (55, 80))
"""大中小分界（契约裁决 A3）：小 01-27 / 中 28-54 / 大 55-80。"""

ALL_NUMBERS: Final[np.ndarray] = np.arange(1, N_NUM + 1, dtype=np.int64)
"""号码全量数组 (80,) int64 = 1..80（只读约定，调用方不得原地修改）。"""

DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
"""标准日期形状 ``YYYY-MM-DD``（红线 6：时间一律标准化，日期字符串 + int 期号）。"""

HISTORY_LINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^date:(?P<date>\d{4}-\d{2}-\d{2}),"
    r"period:(?P<period>\d+),"
    r"numbers:(?P<numbers>[0-9]{1,2}(?:-[0-9]{1,2})*)$"
)
"""开奖历史行格式：``date:YYYY-MM-DD,period:NNNNNNN,numbers:NN-NN-...``。

注意：号码个数 / 去重 / 1..80 取值范围由 ``kl8/history.py::parse_line`` 做业务校验，
本正则只负责"形状"匹配，避免把业务规则塞进常量层。
"""

# =============================================================================
# 三、奖金与成本口径（静态事实；未确认玩法一律 None，只诊断不结算）
# =============================================================================
PRIZE_TABLE_PICK5: Final[Mapping[int, int]] = {5: 1000, 4: 21, 3: 3}
"""选五已确认奖级（整数元）：中 5 = 1000、中 4 = 21、中 3 = 3。"""

DEFAULT_PRIZE_TABLE: Final[Mapping[str, Mapping[int, int] | None]] = {
    "pick5": dict(PRIZE_TABLE_PICK5),
    "pick3": None,
    "pick7": None,
}
"""全玩法奖金表默认值：选三/选七奖级待用户确认，故为 None（禁止臆造金额）。"""

COST_PER_TICKET_YUAN: Final[int] = 2
"""单注投注成本（整数元）。"""

# =============================================================================
# 四、号码属性查询（纯函数）
# =============================================================================


def _require_number(num: int) -> int:
    """校验号码合法性。

    @param num 待校验号码
    @return 原号码（便于链式调用）
    @raises ValueError 号码非 int 或不在 1..80
    """
    if isinstance(num, bool) or not isinstance(num, (int, np.integer)):
        raise ValueError(f"号码 {num!r} 非法，必须是 1..{N_NUM} 的整数")
    value = int(num)
    if not 1 <= value <= N_NUM:
        raise ValueError(f"号码 {num} 非法，必须在 1..{N_NUM}")
    return value


def get_road_012(num: int) -> int:
    """获取号码的 012 路归属（模 3 同余）。

    @param num 号码（1..80）
    @return 0 / 1 / 2
    @raises ValueError 号码越界
    """
    return _require_number(num) % 3


def is_prime_num(num: int) -> bool:
    """判定号码是否为质数（1 非质非合，返回 False）。

    @param num 号码（1..80）
    @return True 为质数，否则 False
    @raises ValueError 号码越界
    """
    return _require_number(num) in PRIMES_80


def get_prime_label(num: int) -> str:
    """获取号码的质合分类标签。

    @param num 号码（1..80）
    @return ``"质"`` / ``"合"`` / ``"特1"``
    @raises ValueError 号码越界
    """
    value = _require_number(num)
    if value == 1:
        return "特1"
    return "质" if value in PRIMES_80 else "合"


def _bound_index(num: int, bounds: tuple[tuple[int, int], ...], what: str) -> int:
    """在给定闭区间划分中定位号码所属区间下标。

    @param num 号码（1..80）
    @param bounds 闭区间元组
    @param what 划分名称（用于异常消息）
    @return 区间下标（0-based）
    @raises ValueError 号码越界
    """
    value = _require_number(num)
    for idx, (lo, hi) in enumerate(bounds):
        if lo <= value <= hi:
            return idx
    raise ValueError(f"号码 {num} 不在{what}划分范围内，请检查 kl8/constants.py")


def zone4_of(num: int) -> int:
    """号码所属四分区下标（0..3，对应 01-20 / 21-40 / 41-60 / 61-80）。

    @param num 号码（1..80）
    @return 0..3
    @raises ValueError 号码越界
    """
    return _bound_index(num, ZONE4_BOUNDS, "四区")


def zone8_of(num: int) -> int:
    """号码所属八分区下标（0..7，每 10 个号一区）。

    @param num 号码（1..80）
    @return 0..7
    @raises ValueError 号码越界
    """
    return _bound_index(num, ZONE8_BOUNDS, "八区")


def size_class_of(num: int) -> int:
    """号码大中小分类。

    @param num 号码（1..80）
    @return 0=小（01-27）、1=中（28-54）、2=大（55-80）
    @raises ValueError 号码越界
    """
    return _bound_index(num, SIZE_BOUNDS, "大中小")


def tail_of(num: int) -> int:
    """号码尾号（模 10 同余，用于同尾组特征）。

    @param num 号码（1..80）
    @return 0..9（如 80 → 0，03 → 3）
    @raises ValueError 号码越界
    """
    return _require_number(num) % 10


def numbers_array() -> np.ndarray:
    """返回号码 1..80 的 float64 向量 (80,)。

    @return 新分配的 (80,) float64 数组，调用方可安全原地修改
    """
    return np.arange(1, N_NUM + 1, dtype=np.float64)


# =============================================================================
# 五、超几何分布基准（随机性评估的唯一公正基准）
# =============================================================================


def _require_pick(n_pick: int, n_draw: int, n_total: int) -> None:
    """校验超几何参数合法性。

    @param n_pick 选号个数
    @param n_draw 开出个数
    @param n_total 号码总数
    @raises ValueError 参数非法
    """
    for name, value in (("n_pick", n_pick), ("n_draw", n_draw), ("n_total", n_total)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"超几何参数 {name} 必须是整数，收到 {value!r}")
    if n_total <= 0:
        raise ValueError(f"超几何参数 n_total 必须为正整数，收到 {n_total}")
    if not 0 <= n_draw <= n_total:
        raise ValueError(f"超几何参数 n_draw 必须在 0..{n_total}，收到 {n_draw}")
    if not 0 <= n_pick <= n_total:
        raise ValueError(f"超几何参数 n_pick 必须在 0..{n_total}，收到 {n_pick}")


def hypergeometric_pmf(
    n_pick: int,
    n_draw: int = K_DRAW,
    n_total: int = N_NUM,
) -> np.ndarray:
    """计算「从 n_total 中随机选 n_pick 码」的精确命中数分布（超几何 pmf）。

    @param n_pick 选号个数（推荐池大小）
    @param n_draw 每期开出个数，默认 K=20
    @param n_total 号码总数，默认 N=80
    @return 长度 n_pick+1 的 float64 数组，下标 k 为"命中 k 码"的概率
    @raises ValueError 参数非法
    """
    _require_pick(n_pick, n_draw, n_total)
    total_ways = math.comb(n_total, n_draw)
    pmf = np.zeros(n_pick + 1, dtype=np.float64)
    for hit in range(n_pick + 1):
        miss = n_draw - hit
        if miss < 0 or miss > n_total - n_pick:
            continue
        pmf[hit] = math.comb(n_pick, hit) * math.comb(n_total - n_pick, miss) / total_ways
    return pmf


def hypergeometric_mean(
    n_pick: int,
    n_draw: int = K_DRAW,
    n_total: int = N_NUM,
) -> float:
    """超几何分布均值 E[Hits] = n_pick · n_draw / n_total。

    @param n_pick 选号个数
    @param n_draw 开出个数
    @param n_total 号码总数
    @return 理论平均命中个数
    @raises ValueError 参数非法
    """
    _require_pick(n_pick, n_draw, n_total)
    return n_pick * n_draw / n_total


def hypergeometric_var(
    n_pick: int,
    n_draw: int = K_DRAW,
    n_total: int = N_NUM,
) -> float:
    """超几何分布方差 Var = n·p·(1-p)·(N-n)/(N-1)。

    @param n_pick 选号个数
    @param n_draw 开出个数
    @param n_total 号码总数
    @return 理论方差（n_total<=1 时返回 0.0 的空安全分支）
    @raises ValueError 参数非法
    """
    _require_pick(n_pick, n_draw, n_total)
    if n_total <= 1:
        return 0.0
    p = n_draw / n_total
    return n_pick * p * (1.0 - p) * (n_total - n_pick) / (n_total - 1)


# =============================================================================
# 四、契约固定系数（非业务可调项，故不进 config；修改须升版契约）
# =============================================================================
GOLD_RATIO: Final[float] = 0.30
"""金胆池规模系数（契约 A5 自定：n_gold = clip(round(0.30×len(main)), gold_min, gold_max)）。

8.md L155 只给示例区间「如 3~5 码」，0.30 系数与「V≥3 优先」均为**契约自定**，
不随用户调参，因此收敛到本模块而非 config.py。
"""

GOLD_VOTE_PRIORITY: Final[int] = 3
"""金胆池优先票数门槛（契约 A5：先取 V ≥ 3 的号码按 FinalScore 降序，不足再补）。"""

VOTE_HIGH_THRESHOLD: Final[int] = 3
"""高共振票数组门槛（R45 归因：V ≥ 3 视为高票组）。"""

VOTE_LOW_MAX: Final[int] = 1
"""低共振票数组上限（R45 归因：V ≤ 1 视为低票组）。"""

PLAY_KT: Final[Mapping[str, tuple[int, int]]] = {"pick5": (5, 4), "pick7": (7, 5)}
"""玩法 → (单注码数 k, 保底命中数 t)（契约 A8：选五 20 注 / 选七 10 注）。

L5 的 `build_ticket_plan` 必须从本表取 k/t，禁止在函数体内写死 5/4/7/5（红线 C1）。
"""

ML_FEATURE_WARMUP: Final[int] = 10
"""ML 训练样本可用的最小前置期数：前 10 期窗口未满，不进入训练集（C1 收口）。"""

RULE_IDS: Final[tuple[str, ...]] = ("R1", "R2", "R3", "R4", "R5")
"""五条正交规则固定编号顺序（投票、权重、归因、报告共用的单一数据源）。"""

RULE_NAMES: Final[Mapping[str, str]] = {
    "R1": "多尺度强跟随共振",
    "R2": "温热多头趋势共振",
    "R3": "动量金叉加速",
    "R4": "强共现伴生拉动",
    "R5": "全量回归临界回补",
}
"""五条规则的中文名称（P8-M5：报告层禁止再写错名/硬编码，统一从此取）。"""

PLAY_LABELS: Final[Mapping[str, str]] = {"pick5": "选五", "pick7": "选七"}
"""玩法内部键 → 中文名（报告层唯一来源；未登记玩法原样回退）。"""

NO_BASELINE_TEXT: Final[str] = "首期无复盘基线（无可用的上期预测记录）"
"""首期无复盘基线固定文案（契约 §3.14.1：缺文件/损坏/无更早记录时必须逐字输出）。"""

ROI_NA_TEXT: Final[str] = "ROI: N/A（官方奖级未确认）"
"""奖级未确认时的 ROI 披露文案（契约裁决 A10 / §6.6，三类报告共用）。"""

ROI_NA_NO_TICKET_TEXT: Final[str] = "ROI: N/A（本次未生成注单）"
"""未生成注单时的 ROI 披露文案（P10b-L15：与"奖级未确认"区分，不得混用同一原因）。"""


def sum_band(n_pick: int, sigmas: float = 2.0) -> tuple[int, int]:
    """计算 n_pick 个号码和值的理论合理区间 ``E ± sigmas·σ``（无放回超几何抽样）。

    【为什么放这里】报告需要给出"合理和值区间"，但区间随 k 变化且必须有数学依据；
    旧实现写死固定和值区间（契约 §6.6 点名禁止）。本函数由 N_NUM 推导，零魔法数字。

    @param n_pick 选号个数
    @param sigmas 标准差倍数（默认 2σ，约 95% 覆盖）
    @return (下界, 上界) 整数和值区间
    """
    if n_pick <= 0:
        return (0, 0)
    expected = float(n_pick) * float(N_NUM + 1) / 2.0
    variance = (
        float(n_pick)
        * ((float(N_NUM) ** 2 - 1.0) / 12.0)
        * ((float(N_NUM) - float(n_pick)) / (float(N_NUM) - 1.0))
    )
    std = math.sqrt(max(0.0, variance))
    return (
        int(round(expected - sigmas * std)),
        int(round(expected + sigmas * std)),
    )