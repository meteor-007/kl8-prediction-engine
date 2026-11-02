# -*- coding: utf-8 -*-
"""快乐8五层重构 —— 应用配置中心（C1 零魔法数字的唯一业务归宿）。

【定位】
全系统所有**业务阈值/权重/窗口/预算/上限**都在本模块定义，并在这里做**严格校验**。
layer1–layer5 只允许通过 ``AppConfig`` 读取参数，严禁在业务代码里写裸数字。

【配置层级】
``AppConfig``
├─ ``windows``   :class:`WindowConfig`   四尺度窗口与冷启动下限
├─ ``markov``    :class:`MarkovConfig`   拉普拉斯平滑 α、阶数、重号/异号拆分
├─ ``weights``   :class:`ScoreWeights`   BaseScore 归一化权重 + λ 投票加成
├─ ``rules``     :class:`RuleConfig`     R1–R5 阈值
├─ ``adaptive``  :class:`AdaptiveConfig` 在线 Softmax 自适应与状态持久化
├─ ``pool``      :class:`PoolConfig`     金胆池/复式池与形态硬约束
├─ ``covering``  :class:`CoveringConfig` 覆盖设计 C(v,k,t,m) 与预算
├─ ``ml``        :class:`MLConfig`       LightGBM 训练/缓存/校准（默认仅诊断）
└─ ``backtest``  :class:`BacktestConfig` Walk-Forward 冷启动与指标口径

【深合并语义】
``load_config(overrides)`` 把 ``overrides`` 视作**稀疏补丁**递归合并到默认配置上，
未出现的字段保持默认值；合并后统一走 :func:`validate_config`，任何非法都会抛
:class:`ConfigError`（消息格式：``配置项 {路径} 非法：{现值}，要求 {约束}；建议 {修复}``）。
``load_config`` 承诺**不向调用方泄漏** ``KeyError`` / ``TypeError`` / ``ValueError``。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Final, Mapping, NoReturn

from kl8.constants import COST_PER_TICKET_YUAN, DEFAULT_PRIZE_TABLE, N_NUM
from kl8.errors import ConfigError

# =============================================================================
# 校验口径常量（属于"配置域"的工程参数，故放在 config.py 而非 constants.py）
# =============================================================================
SUM_TOLERANCE: Final[float] = 1e-9
"""权重求和容差：Σω 与 1.00 的允许绝对误差。"""

MARKOV_ALPHA_RANGE: Final[tuple[float, float]] = (0.5, 2.0)
"""拉普拉斯平滑系数 α 的合法闭区间（8.md §四模块三）。"""

LAMBDA_VOTE_RANGE: Final[tuple[float, float]] = (0.05, 0.10)
"""单票共振加成系数 λ 的合法闭区间（8.md §四模块四.3）。"""

PLAY_TIER_DEFAULTS: Final[Mapping[str, tuple[int, int]]] = {"pick5": (5, 4), "pick7": (7, 5)}
"""玩法 → (k, t) 默认映射：选五 k=5/t=4；选七 k=7/t=5。"""

SUPPORTED_PLAYS: Final[tuple[str, ...]] = tuple(PLAY_TIER_DEFAULTS)
"""支持的玩法键（与奖金表、预算表联动）。"""

PAYOUT_PLAYS: Final[tuple[str, ...]] = tuple(DEFAULT_PRIZE_TABLE)
"""奖金表允许出现的玩法键（含选三这类"奖级待确认"的占位玩法，其值必须为 None）。"""

RELAX_ORDER_ALLOWED: Final[frozenset[str]] = frozenset({"same_tail", "parity", "zone4", "repeat_ratio"})
"""形态硬约束放宽顺序的合法取值集合（R34）。"""


# =============================================================================
# 一、分层配置数据类（全部 frozen，禁止运行期改写）
# =============================================================================
@dataclass(frozen=True)
class WindowConfig:
    """四尺度时间窗口配置（R2 / R3）。"""

    short: int = 10
    """短线爆发窗 10G。"""

    mid: int = 25
    """中线趋势窗 25G。"""

    long: int = 50
    """中期稳态窗 50G。"""

    min_history: int = 40
    """低于该历史期数抛 ``InsufficientDataError``（窗口不可信）。"""

    cold_start_floor: int = 1
    """冷启动窗口下限 w_eff = max(floor, min(w, T))，防止 0 长度窗口。"""


@dataclass(frozen=True)
class MarkovConfig:
    """马尔可夫跟随转移配置（R15–R18）。"""

    alpha: float = 1.0
    """拉普拉斯平滑系数 α ∈ [0.5, 2.0]。"""

    order: int = 2
    """阶数：1=一阶；2=额外算二阶滞后矩阵（隔期回补）。"""

    diag_split: bool = True
    """是否拆分"重号惯性分（对角）/异号跟随分（非对角）"。"""


@dataclass(frozen=True)
class ScoreWeights:
    """BaseScore 归一化权重与投票加成（R31 / R32，Σω 必须 = 1）。

    经实测动力学与大数周期优化：提升高置信度 omission 回补与长线 follow_all，
    降低高频噪声 follow_10，权重严格归一化和为 1.0。
    """

    follow_10: float = 0.02
    """Follow_10G 权重 ω_fol,10（超短期噪声抑制）。"""

    follow_25: float = 0.08
    """Follow_25G 权重。"""

    follow_50: float = 0.20
    """Follow_50G 权重（中期动量跟随核心）。"""

    follow_all: float = 0.25
    """Follow_ALL 权重（长期大数法则稳健基准）。"""

    co: float = 0.05
    """共现伴生 C̄o 权重 ω_co（低权重降噪）。"""

    trend: float = 0.05
    """趋势 Trend 权重 ω_trend。"""

    omission: float = 0.35
    """遗漏回补压力 Pressure 权重 ω_om（高置信度核心反弹爆发特征）。"""

    lambda_vote: float = 0.05
    """单票共振加成 λ ∈ [0.05, 0.10]（取下界控制规则交集噪声）。"""

    trend_w_f10: float = 0.5
    """Trend 内部：R(F10) 权重（裁决 A1）。"""

    trend_w_mom_10_25: float = 0.3
    """Trend 内部：R(Mom_10-25) 权重。"""

    trend_w_mom_25_50: float = 0.2
    """Trend 内部：R(Mom_25-50) 权重。"""

    ml: float = 0.0
    """ML 概率权重，默认 0（裁决 A10：只诊断不进打分）。"""


@dataclass(frozen=True)
class RuleConfig:
    """五大正交规则阈值（R25–R30，逐项对应 8.md §四模块四.2）。"""

    r1_min_windows: int = 3
    """R1：四个窗口中至少几个进入前 35%。"""

    r1_rank_threshold: float = 0.65
    """R1：百分位秩门槛（前 35%）。"""

    r2_base_freq: float = 0.24
    """R2：25G / 50G 频率下限（≥ 该值）。"""

    r2_f10_low: float = 0.20
    """R2：10G 频率区间下界。"""

    r2_f10_high: float = 0.50
    """R2：10G 频率区间上界（防短期极端过热透支）。"""

    r3_follow_rank: float = 0.60
    """R3：R(Follow_10G) 门槛（前 40%）。"""

    r4_rank_score: float = 0.75
    """R4：0.5R(Co25)+0.5R(Co50) 门槛（前 25%）。"""

    r4_w_co25: float = 0.5
    """R4：Co25 秩权重。"""

    r4_w_co50: float = 0.5
    """R4：Co50 秩权重。"""

    r5_pressure: float = 0.60
    """R5：回补压力下限。"""

    r5_max_gap: int = 12
    """R5：当前遗漏上限（排除死冷号）。"""

    r5_follow_all_rank: float = 0.70
    """R5：R(Follow_ALL) 门槛（前 30%）。"""


@dataclass(frozen=True)
class AdaptiveConfig:
    """在线自适应权重配置（R33 / 裁决 A9）。"""

    lookback_m: int = 8
    """回看期数 M（裁决 A9：M=8）。"""

    min_periods: int = 3
    """低于该样本数退回均匀先验。"""

    laplace_alpha: float = 1.0
    """命中率 Laplace 平滑系数。"""

    temperature: float = 1.0
    """Softmax 温度（越大越平缓）。"""

    weight_floor: float = 0.05
    """单规则权重下限，防止某规则被彻底饿死。"""

    schema_version: int = 1
    """状态文件 schema 版本（版本不符则 .bak 重建）。"""

    state_path: str = "output/state/adaptive_weights.json"
    """自适应权重状态文件路径。"""

    in_memory: bool = False
    """纯内存模式（回测强制 True：不落盘、不污染线上状态）。"""

    persist: bool = True
    """日跑是否持久化状态。"""


@dataclass(frozen=True)
class PoolConfig:
    """候选池与空间形态硬约束（R34–R36 / 裁决 A6）。"""

    gold_min: int = 3
    """金胆池最小码数。"""

    gold_max: int = 5
    """金胆池最大码数。"""

    main_default: int = 12
    """复式池默认大小（裁决 A6：12）。"""

    main_min: int = 10
    """复式池最小码数。"""

    main_max: int = 15
    """复式池最大码数。"""

    max_zone4_ratio: float = 0.45
    """单四分区入选占比上限 45%。"""

    max_parity_ratio: float = 0.70
    """奇/偶单边占比上限 70%。"""

    max_same_tail: int = 3
    """同尾号最多 3 个。"""

    repeat_ratio: float = 0.30
    """上期重号目标占比 30%。"""

    repeat_ratio_min: float = 0.25
    """重号配比下限 25%。"""

    repeat_ratio_max: float = 0.35
    """重号配比上限 35%。"""

    relax_order: tuple[str, ...] = ("same_tail", "parity", "zone4", "repeat_ratio")
    """不可行时的放宽顺序：同尾 → 奇偶 → 分区 → 重号比。"""


@dataclass(frozen=True)
class CoveringConfig:
    """覆盖设计 C(v,k,t,m) 与预算配置（R37–R41 / 裁决 A7、A8）。"""

    play: str = "pick5"
    """当前玩法：pick5 | pick7。"""

    k: int = 5
    """单注号码个数。"""

    t: int = 4
    """保底命中数。"""

    m_mode: str = "auto"
    """m 取值模式：auto（t>=k 时 m=min(v,t)；t<k 时 m=min(v,t+m_margin)）| fixed（用 m_fixed）。"""

    m_fixed: int | None = None
    """m_mode=fixed 时的固定 m。"""

    m_margin: int = 1
    """auto 模式附加余量（契约裁决 A7 / v1.1.0：默认 1，合法区间 [1,3]）。

    m 是「池内预期命中码数」的**玩法假设参数**（8.md L164），不是算法预测值；
    报告必须注明该口径。与 ceil(0.25v)+2 的偏差 >2 时只写告警、不阻断。
    """

    budget: int = 20
    """当前玩法注数预算。"""

    budget_by_play: Mapping[str, int] = field(default_factory=lambda: {"pick5": 20, "pick7": 10})
    """各玩法默认预算（裁决 A8）。"""

    max_budget: int = 200
    """预算硬上限（防误配巨额注单）。"""

    max_pool_v: int = 20
    """覆盖池规模上限 v ≤ 20（防组合爆炸）。"""

    max_m_subsets: int = 200_000
    """m 子集穷举上限 C(v,m) ≤ 200000（防内存爆炸）。"""

    ilp_candidate_cap: int = 120
    """ILP 候选注数上限，超过则降级贪心。"""

    ilp_target_cap: int = 400
    """ILP m 子集目标数上限，超过则降级贪心。"""

    ilp_time_limit_s: float = 5.0
    """ILP 求解时间上限（秒）。"""

    sa_iterations: int = 2000
    """模拟退火迭代上限。"""

    sa_t0: float = 1.0
    """模拟退火初始温度。"""

    sa_cooling: float = 0.995
    """模拟退火冷却系数（0<cooling<1）。"""

    max_pair_overlap: int = 3
    """任意两注最大重合号码数（正交防撞）。"""

    max_consecutive_in_ticket: int = 2
    """单注内最大连号长度（剔除 3 连以上）。"""

    forbid_all_odd_even: bool = True
    """是否剔除全奇/全偶废注。"""

    sum_margin: float = 0.45
    """单注和值安全带宽（相对池均值 × k 的上下浮动比例）。"""

    table_path: str = "kl8/layer5_combinatorial/covering_tables.json"
    """经典参数覆盖表路径。"""


@dataclass(frozen=True)
class MLConfig:
    """LightGBM 训练与缓存配置（R23）。"""

    enabled: bool = False
    """是否启用 ML 特征（默认关闭：契约 B4 规定 ML 为「可配置开启的诊断通道」）。

    weight=0.0 时 ML 不参与 FinalScore，开启只会拖慢单期耗时（实测 0.34s → 2.03s），
    因此默认关闭；需要诊断或调参时用 CLI `--ml` 显式开启。
    """

    weight: float = 0.0
    """ML 概率在 FinalScore 中的权重（默认 0，仅诊断）。"""

    train_window: int = 160
    """训练回溯期数。"""

    refit_every: int = 25
    """重训间隔（期间命中缓存）。"""

    n_estimators: int = 40
    """弱学习器数量。"""

    learning_rate: float = 0.03
    """学习率。"""

    num_leaves: int = 12
    """最大叶子数（限制复杂度）。"""

    min_child_samples: int = 25
    """叶子最小样本数（防过拟合）。"""

    subsample: float = 0.85
    """行采样比例。"""

    colsample_bytree: float = 0.85
    """列采样比例。"""

    seed: int = 42
    """随机种子（C7 确定性）。"""

    calibrate: str = "pav"
    """校准方式：pav（自写保序回归）| none。"""

    fallback: str = "ema"
    """lightgbm 缺失时的降级方式：ema | uniform。"""

    cache_size: int = 8
    """进程内 ML 预测缓存条数。"""


@dataclass(frozen=True)
class BacktestConfig:
    """Walk-Forward 回测配置（R42–R46）。"""

    warmup: int = 300
    """冷启动期数 T0（前 T0 期只建矩阵，不参与评价）。"""

    default_n: int = 100
    """默认回测期数。"""

    ks: tuple[int, ...] = (3, 5, 7, 10, 12)
    """Lift@K 的 K 取值集合。"""

    t0_min: int = 60
    """冷启动下限（T ≤ T0 + t0_min 时抛 InsufficientDataError）。"""

    include_tickets: bool = True
    """回测是否生成注单（关闭可大幅提速）。"""

    alpha_sig: float = 0.05
    """显著性水平（单侧 t 检验）。"""

    lift_signal: float = 1.05
    """Lift@K 判定显著超额信号的阈值 ∈[1.0, 1.2]（契约 R43、v1.1.0）。"""


@dataclass(frozen=True)
class AppConfig:
    """快乐8五层系统总配置（唯一入口对象）。"""

    windows: WindowConfig = field(default_factory=WindowConfig)
    """窗口配置。"""

    markov: MarkovConfig = field(default_factory=MarkovConfig)
    """马尔可夫配置。"""

    weights: ScoreWeights = field(default_factory=ScoreWeights)
    """打分权重配置。"""

    rules: RuleConfig = field(default_factory=RuleConfig)
    """五规则阈值配置。"""

    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    """自适应权重配置。"""

    pool: PoolConfig = field(default_factory=PoolConfig)
    """候选池配置。"""

    covering: CoveringConfig = field(default_factory=CoveringConfig)
    """覆盖设计配置。"""

    ml: MLConfig = field(default_factory=MLConfig)
    """机器学习配置。"""

    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    """回测配置。"""

    history_path: str = "kl8_history.txt"
    """开奖历史文件路径。"""

    output_dir: str = "output"
    """产出根目录。"""

    seed: int = 20261102
    """全局随机种子（裁决：确定性 seed，C7）。"""

    eps: float = 1e-9
    """全局除零保护 epsilon（C8）。"""

    cost_per_ticket: int = 2
    """单注成本（整数元，C3 禁浮点金额）。"""

    prize_table: Mapping[str, Mapping[int, int] | None] = field(
        default_factory=lambda: {play: (dict(prize) if prize is not None else None)
                                 for play, prize in DEFAULT_PRIZE_TABLE.items()}
    )
    """奖金表：仅 pick5 已确认；pick3/pick7 为 None（待确认，不结算）。"""


# =============================================================================
# 二、校验工具
# =============================================================================
def _fail(path: str, value: Any, rule: str, suggestion: str) -> NoReturn:
    """抛出标准格式的配置异常。

    @param path 配置项路径（如 ``weights.co``）
    @param value 当前非法值
    @param rule 约束描述
    @param suggestion 修复建议
    @raises ConfigError 始终抛出
    """
    raise ConfigError(f"配置项 {path} 非法：{value!r}，要求 {rule}；建议 {suggestion}")


def _check_int(path: str, value: Any, lo: int, hi: int) -> int:
    """校验整数字段并返回归一化值。

    @param path 配置项路径
    @param value 待校验值（bool 视为非法，避免 True/1 混淆）
    @param lo 允许下界（含）
    @param hi 允许上界（含）
    @return int 值
    @raises ConfigError 类型或范围非法
    """
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, value, "整数", f"改为 {lo}..{hi} 之间的整数")
    if not lo <= value <= hi:
        _fail(path, value, f"{lo}..{hi} 的整数", f"改为 {lo}..{hi} 之间的整数")
    return int(value)


def _check_float(path: str, value: Any, lo: float, hi: float, *, inclusive: bool = True) -> float:
    """校验浮点字段并返回归一化值。

    @param path 配置项路径
    @param value 待校验值
    @param lo 下界
    @param hi 上界
    @param inclusive True=闭区间，False=开区间
    @return float 值
    @raises ConfigError 类型或范围非法
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, value, "数值", f"改为 {lo}..{hi} 之间的数值")
    number = float(value)
    ok = (lo <= number <= hi) if inclusive else (lo < number < hi)
    if not ok:
        bracket = "[]" if inclusive else "()"
        _fail(path, number, f"{bracket[0]}{lo}, {hi}{bracket[1]} 区间内数值", f"改为 {lo}..{hi} 之间的数值")
    return number


def _check_bool(path: str, value: Any) -> bool:
    """校验布尔字段。

    @param path 配置项路径
    @param value 待校验值
    @return bool 值
    @raises ConfigError 非布尔
    """
    if not isinstance(value, bool):
        _fail(path, value, "布尔值 True/False", "改为 True 或 False")
    return value


def _check_str(path: str, value: Any) -> str:
    """校验非空字符串字段。

    @param path 配置项路径
    @param value 待校验值
    @return 原字符串
    @raises ConfigError 非字符串或空串
    """
    if not isinstance(value, str) or not value.strip():
        _fail(path, value, "非空字符串", "填写有效路径或名称")
    return value


def _check_weights_sum(path: str, parts: Mapping[str, float], target: float = 1.0) -> None:
    """校验一组权重之和等于目标值（容差 SUM_TOLERANCE）。

    @param path 权重组路径（用于消息）
    @param parts 分量名 → 权重
    @param target 目标和，默认 1.0
    @raises ConfigError 求和不等于目标
    """
    total = sum(parts.values())
    if abs(total - target) > SUM_TOLERANCE:
        detail = " + ".join(f"{name}={value}" for name, value in parts.items())
        _fail(
            path,
            round(total, 12),
            f"权重之和 = {target}（容差 {SUM_TOLERANCE}）",
            f"当前 {detail} = {total:.12f}，请调整各分量使总和为 {target}",
        )


# =============================================================================
# 三、校验主函数（R48-a）
# =============================================================================
def validate_config(cfg: AppConfig) -> None:
    """对完整配置做严格校验，任何非法立即抛 :class:`ConfigError`。

    【覆盖的强校验】
    1. ``Σω = 1.00 ± 1e-9``（BaseScore 七维权重）；
    2. ``α ∈ [0.5, 2.0]``；
    3. ``λ ∈ [0.05, 0.10]``；
    4. ``gold_max < main_min``（金胆池必须严格小于复式池）；
    5. ``k ≤ 池大小``（覆盖设计单注码数不超过覆盖池上限与号码总数）；
    6. Trend 内部权重和 = 1、R4 内部权重和 = 1；
    7. 预算 ≤ 硬上限、重号配比区间有序、回测冷启动关系成立等。

    @param cfg 待校验配置
    @raises ConfigError 任一字段/约束非法
    """
    if not isinstance(cfg, AppConfig):
        _fail("AppConfig", type(cfg).__name__, "AppConfig 实例", "调用 load_config() 构造配置")

    # —— 窗口 ——
    win = cfg.windows
    if not isinstance(win, WindowConfig):
        _fail("windows", type(win).__name__, "WindowConfig", "使用 load_config() 深合并构造")
    _check_int("windows.short", win.short, 1, N_NUM)
    _check_int("windows.mid", win.mid, 1, N_NUM)
    _check_int("windows.long", win.long, 1, N_NUM)
    _check_int("windows.min_history", win.min_history, 1, 10_000)
    _check_int("windows.cold_start_floor", win.cold_start_floor, 1, N_NUM)
    if not win.short < win.mid < win.long:
        _fail(
            "windows.(short,mid,long)",
            (win.short, win.mid, win.long),
            "严格递增 short < mid < long",
            "短窗必须最小、长窗必须最大，否则动量差符号会与业务语义相反",
        )

    # —— 马尔可夫 ——
    _check_float("markov.alpha", cfg.markov.alpha, *MARKOV_ALPHA_RANGE)
    if cfg.markov.order not in (1, 2):
        _fail("markov.order", cfg.markov.order, "1 或 2", "只支持一阶(1)与二阶(2)滞后矩阵")
    _check_bool("markov.diag_split", cfg.markov.diag_split)

    # —— 打分权重 R31/R32 ——
    weights = cfg.weights
    if not isinstance(weights, ScoreWeights):
        _fail("weights", type(weights).__name__, "ScoreWeights", "使用 load_config() 深合并构造")
    weight_parts = {
        "follow_10": _check_float("weights.follow_10", weights.follow_10, 0.0, 1.0),
        "follow_25": _check_float("weights.follow_25", weights.follow_25, 0.0, 1.0),
        "follow_50": _check_float("weights.follow_50", weights.follow_50, 0.0, 1.0),
        "follow_all": _check_float("weights.follow_all", weights.follow_all, 0.0, 1.0),
        "co": _check_float("weights.co", weights.co, 0.0, 1.0),
        "trend": _check_float("weights.trend", weights.trend, 0.0, 1.0),
        "omission": _check_float("weights.omission", weights.omission, 0.0, 1.0),
    }
    _check_weights_sum("weights.(follow_10..omission)", weight_parts)
    _check_float("weights.lambda_vote", weights.lambda_vote, *LAMBDA_VOTE_RANGE)
    trend_parts = {
        "trend_w_f10": _check_float("weights.trend_w_f10", weights.trend_w_f10, 0.0, 1.0),
        "trend_w_mom_10_25": _check_float("weights.trend_w_mom_10_25", weights.trend_w_mom_10_25, 0.0, 1.0),
        "trend_w_mom_25_50": _check_float("weights.trend_w_mom_25_50", weights.trend_w_mom_25_50, 0.0, 1.0),
    }
    _check_weights_sum("weights.trend_w_*", trend_parts)
    _check_float("weights.ml", weights.ml, 0.0, 1.0)

    # —— 五规则阈值 ——
    rules = cfg.rules
    if not isinstance(rules, RuleConfig):
        _fail("rules", type(rules).__name__, "RuleConfig", "使用 load_config() 深合并构造")
    _check_int("rules.r1_min_windows", rules.r1_min_windows, 1, 4)
    _check_float("rules.r1_rank_threshold", rules.r1_rank_threshold, 0.0, 1.0)
    _check_float("rules.r2_base_freq", rules.r2_base_freq, 0.0, 1.0)
    _check_float("rules.r2_f10_low", rules.r2_f10_low, 0.0, 1.0)
    _check_float("rules.r2_f10_high", rules.r2_f10_high, 0.0, 1.0)
    if not rules.r2_f10_low <= rules.r2_f10_high:
        _fail(
            "rules.(r2_f10_low,r2_f10_high)",
            (rules.r2_f10_low, rules.r2_f10_high),
            "r2_f10_low ≤ r2_f10_high",
            "下界必须不大于上界，否则 R2 永远不可能命中",
        )
    _check_float("rules.r3_follow_rank", rules.r3_follow_rank, 0.0, 1.0)
    _check_float("rules.r4_rank_score", rules.r4_rank_score, 0.0, 1.0)
    r4_parts = {
        "r4_w_co25": _check_float("rules.r4_w_co25", rules.r4_w_co25, 0.0, 1.0),
        "r4_w_co50": _check_float("rules.r4_w_co50", rules.r4_w_co50, 0.0, 1.0),
    }
    _check_weights_sum("rules.(r4_w_co25, r4_w_co50)", r4_parts)
    _check_float("rules.r5_pressure", rules.r5_pressure, 0.0, 1.0)
    _check_int("rules.r5_max_gap", rules.r5_max_gap, 0, 10_000)
    _check_float("rules.r5_follow_all_rank", rules.r5_follow_all_rank, 0.0, 1.0)

    # —— 自适应 ——
    adaptive = cfg.adaptive
    if not isinstance(adaptive, AdaptiveConfig):
        _fail("adaptive", type(adaptive).__name__, "AdaptiveConfig", "使用 load_config() 深合并构造")
    _check_int("adaptive.lookback_m", adaptive.lookback_m, 1, 10_000)
    _check_int("adaptive.min_periods", adaptive.min_periods, 1, 10_000)
    if adaptive.min_periods > adaptive.lookback_m:
        _fail(
            "adaptive.(min_periods,lookback_m)",
            (adaptive.min_periods, adaptive.lookback_m),
            "min_periods ≤ lookback_m",
            "最小样本数不能超过回看窗口，否则永远走不到 Softmax 分支",
        )
    _check_float("adaptive.laplace_alpha", adaptive.laplace_alpha, 0.0, 1_000.0, inclusive=False)
    _check_float("adaptive.temperature", adaptive.temperature, 0.0, 1_000.0, inclusive=False)
    _check_float("adaptive.weight_floor", adaptive.weight_floor, 0.0, 1.0, inclusive=False)
    _check_int("adaptive.schema_version", adaptive.schema_version, 1, 1_000)
    _check_str("adaptive.state_path", adaptive.state_path)
    _check_bool("adaptive.in_memory", adaptive.in_memory)
    _check_bool("adaptive.persist", adaptive.persist)

    # —— 候选池 ——
    pool = cfg.pool
    if not isinstance(pool, PoolConfig):
        _fail("pool", type(pool).__name__, "PoolConfig", "使用 load_config() 深合并构造")
    _check_int("pool.gold_min", pool.gold_min, 1, N_NUM)
    _check_int("pool.gold_max", pool.gold_max, 1, N_NUM)
    _check_int("pool.main_min", pool.main_min, 1, N_NUM)
    _check_int("pool.main_max", pool.main_max, 1, N_NUM)
    _check_int("pool.main_default", pool.main_default, 1, N_NUM)
    if not pool.gold_min <= pool.gold_max:
        _fail(
            "pool.(gold_min,gold_max)",
            (pool.gold_min, pool.gold_max),
            "gold_min ≤ gold_max",
            "金胆池下界不能大于上界",
        )
    if not pool.gold_max < pool.main_min:
        _fail(
            "pool.(gold_max,main_min)",
            (pool.gold_max, pool.main_min),
            "gold_max < main_min（金胆池必须严格小于复式池）",
            "把 pool.main_min 调大或 pool.gold_max 调小，避免胆池与复式池倒挂",
        )
    if not pool.main_min <= pool.main_default <= pool.main_max:
        _fail(
            "pool.main_default",
            pool.main_default,
            f"{pool.main_min}..{pool.main_max}",
            "默认复式池大小必须落在 [main_min, main_max] 区间内",
        )
    _check_float("pool.max_zone4_ratio", pool.max_zone4_ratio, 0.0, 1.0, inclusive=False)
    _check_float("pool.max_parity_ratio", pool.max_parity_ratio, 0.5, 1.0)
    _check_int("pool.max_same_tail", pool.max_same_tail, 1, N_NUM)
    _check_float("pool.repeat_ratio_min", pool.repeat_ratio_min, 0.0, 1.0)
    _check_float("pool.repeat_ratio_max", pool.repeat_ratio_max, 0.0, 1.0)
    _check_float("pool.repeat_ratio", pool.repeat_ratio, pool.repeat_ratio_min, pool.repeat_ratio_max)
    if not isinstance(pool.relax_order, tuple) or not pool.relax_order:
        _fail(
            "pool.relax_order",
            pool.relax_order,
            "非空元组",
            f"取值来自 {sorted(RELAX_ORDER_ALLOWED)}，按放宽先后顺序排列",
        )
    unknown_relax = [name for name in pool.relax_order if name not in RELAX_ORDER_ALLOWED]
    if unknown_relax:
        _fail(
            "pool.relax_order",
            pool.relax_order,
            f"元素来自 {sorted(RELAX_ORDER_ALLOWED)}",
            f"移除未知约束名 {unknown_relax}",
        )

    # —— 覆盖设计 ——
    covering = cfg.covering
    if not isinstance(covering, CoveringConfig):
        _fail("covering", type(covering).__name__, "CoveringConfig", "使用 load_config() 深合并构造")
    if covering.play not in SUPPORTED_PLAYS:
        _fail("covering.play", covering.play, f"取值来自 {list(SUPPORTED_PLAYS)}", "选五用 pick5、选七用 pick7")
    _check_int("covering.k", covering.k, 1, N_NUM)
    _check_int("covering.t", covering.t, 1, covering.k)
    if covering.t < 1:
        _fail("covering.t", covering.t, "≥ 1", "保底命中数必须为正")
    if covering.m_mode not in ("auto", "fixed"):
        _fail("covering.m_mode", covering.m_mode, "'auto' 或 'fixed'", "自动推导用 auto，指定用 fixed")
    if covering.m_mode == "fixed":
        if covering.m_fixed is None:
            _fail("covering.m_fixed", covering.m_fixed, "m_mode=fixed 时必须给出整数 m", "填写 m_fixed")
        _check_int("covering.m_fixed", covering.m_fixed, covering.t, covering.max_pool_v)
    elif covering.m_fixed is not None:
        _check_int("covering.m_fixed", covering.m_fixed, covering.t, covering.max_pool_v)
    _check_int("covering.m_margin", covering.m_margin, 1, 3)
    _check_int("covering.budget", covering.budget, 1, covering.max_budget)
    _check_int("covering.max_budget", covering.max_budget, 1, 100_000)
    if not isinstance(covering.budget_by_play, Mapping):
        _fail("covering.budget_by_play", covering.budget_by_play, "映射", "形如 {'pick5': 20, 'pick7': 10}")
    for play, value in covering.budget_by_play.items():
        if play not in SUPPORTED_PLAYS:
            _fail("covering.budget_by_play", play, f"键来自 {list(SUPPORTED_PLAYS)}", "只允许 pick5/pick7")
        _check_int(f"covering.budget_by_play.{play}", value, 1, covering.max_budget)
    _check_int("covering.max_pool_v", covering.max_pool_v, 1, N_NUM)
    if covering.k > covering.max_pool_v:
        _fail(
            "covering.(k,max_pool_v)",
            (covering.k, covering.max_pool_v),
            "k ≤ 池大小 max_pool_v",
            "单注号码个数不能超过覆盖池规模，否则组合无意义",
        )
    _check_int("covering.max_m_subsets", covering.max_m_subsets, 1, 100_000_000)
    _check_int("covering.ilp_candidate_cap", covering.ilp_candidate_cap, 1, 1_000_000)
    _check_int("covering.ilp_target_cap", covering.ilp_target_cap, 1, 1_000_000)
    _check_float("covering.ilp_time_limit_s", covering.ilp_time_limit_s, 0.0, 600.0, inclusive=False)
    _check_int("covering.sa_iterations", covering.sa_iterations, 1, 10_000_000)
    _check_float("covering.sa_t0", covering.sa_t0, 0.0, 1_000.0, inclusive=False)
    _check_float("covering.sa_cooling", covering.sa_cooling, 0.0, 1.0, inclusive=False)
    _check_int("covering.max_pair_overlap", covering.max_pair_overlap, 0, covering.k)
    _check_int("covering.max_consecutive_in_ticket", covering.max_consecutive_in_ticket, 1, covering.k)
    _check_bool("covering.forbid_all_odd_even", covering.forbid_all_odd_even)
    _check_float("covering.sum_margin", covering.sum_margin, 0.0, 1.0, inclusive=False)
    _check_str("covering.table_path", covering.table_path)

    # —— 机器学习 ——
    ml = cfg.ml
    if not isinstance(ml, MLConfig):
        _fail("ml", type(ml).__name__, "MLConfig", "使用 load_config() 深合并构造")
    _check_bool("ml.enabled", ml.enabled)
    _check_float("ml.weight", ml.weight, 0.0, 1.0)
    _check_int("ml.train_window", ml.train_window, 1, 100_000)
    _check_int("ml.refit_every", ml.refit_every, 1, 100_000)
    _check_int("ml.n_estimators", ml.n_estimators, 1, 10_000)
    _check_float("ml.learning_rate", ml.learning_rate, 0.0, 1.0, inclusive=False)
    _check_int("ml.num_leaves", ml.num_leaves, 2, 100_000)
    _check_int("ml.min_child_samples", ml.min_child_samples, 1, 100_000)
    _check_float("ml.subsample", ml.subsample, 0.0, 1.0, inclusive=False)
    _check_float("ml.colsample_bytree", ml.colsample_bytree, 0.0, 1.0, inclusive=False)
    _check_int("ml.seed", ml.seed, 0, 2**32 - 1)
    if ml.calibrate not in ("pav", "none"):
        _fail("ml.calibrate", ml.calibrate, "'pav' 或 'none'", "默认 pave；关校准用 none")
    if ml.fallback not in ("ema", "uniform"):
        _fail("ml.fallback", ml.fallback, "'ema' 或 'uniform'", "lightgbm 缺失时的替代概率口径")
    _check_int("ml.cache_size", ml.cache_size, 0, 1_000)

    # —— 回测 ——
    backtest = cfg.backtest
    if not isinstance(backtest, BacktestConfig):
        _fail("backtest", type(backtest).__name__, "BacktestConfig", "使用 load_config() 深合并构造")
    _check_int("backtest.warmup", backtest.warmup, 1, 1_000_000)
    _check_int("backtest.default_n", backtest.default_n, 1, 1_000_000)
    if not isinstance(backtest.ks, tuple) or not backtest.ks:
        _fail("backtest.ks", backtest.ks, "非空整数元组", "如 (3, 5, 7, 10, 12)")
    previous = 0
    for k_value in backtest.ks:
        _check_int("backtest.ks[]", k_value, 1, N_NUM)
        if k_value <= previous:
            _fail("backtest.ks", backtest.ks, "严格递增的正整数元组", "去重并按升序排列")
        previous = k_value
    _check_int("backtest.t0_min", backtest.t0_min, 1, 1_000_000)
    if backtest.warmup <= backtest.t0_min:
        _fail(
            "backtest.(warmup,t0_min)",
            (backtest.warmup, backtest.t0_min),
            "warmup > t0_min",
            "冷启动期数必须大于最小可回测期数，否则永远凑不出回测样本",
        )
    _check_bool("backtest.include_tickets", backtest.include_tickets)
    _check_float("backtest.alpha_sig", backtest.alpha_sig, 0.0, 1.0, inclusive=False)
    _check_float("backtest.lift_signal", backtest.lift_signal, 1.0, 1.2, inclusive=True)

    # —— 全局 ——
    _check_str("history_path", cfg.history_path)
    _check_str("output_dir", cfg.output_dir)
    _check_int("seed", cfg.seed, 0, 2**32 - 1)
    _check_float("eps", cfg.eps, 0.0, 1.0, inclusive=False)
    _check_int("cost_per_ticket", cfg.cost_per_ticket, 1, 100_000)
    if not isinstance(cfg.prize_table, Mapping):
        _fail("prize_table", cfg.prize_table, "映射", "形如 {'pick5': {5: 1000, 4: 21, 3: 3}}")
    for play, prize in cfg.prize_table.items():
        if play not in PAYOUT_PLAYS:
            _fail(
                "prize_table",
                play,
                f"键来自 {list(PAYOUT_PLAYS)}",
                "只允许已登记玩法；未确认奖级请把值设为 None，禁止臆造金额",
            )
        if prize is None:
            continue
        if not isinstance(prize, Mapping):
            _fail(f"prize_table.{play}", prize, "映射或 None", "未确认奖级请填 None，禁止臆造金额")
        for tier, amount in prize.items():
            _check_int(f"prize_table.{play}.{tier}", tier, 1, N_NUM)
            _check_int(f"prize_table.{play}[{tier}]", amount, 0, 100_000_000)


# =============================================================================
# 四、构造 / 序列化 / 深合并
# =============================================================================
_SUBCONFIG_TYPES: Final[tuple[tuple[str, type], ...]] = (
    ("windows", WindowConfig),
    ("markov", MarkovConfig),
    ("weights", ScoreWeights),
    ("rules", RuleConfig),
    ("adaptive", AdaptiveConfig),
    ("pool", PoolConfig),
    ("covering", CoveringConfig),
    ("ml", MLConfig),
    ("backtest", BacktestConfig),
)
"""AppConfig 的嵌套子配置字段名 → 类型（顺序与 dataclass 字段一致）。"""

_APP_SCALAR_FIELDS: Final[tuple[str, ...]] = (
    "history_path",
    "output_dir",
    "seed",
    "eps",
    "cost_per_ticket",
    "prize_table",
)
"""AppConfig 的标量/映射字段名。"""


def default_config() -> AppConfig:
    """构造并校验一份默认配置。

    @return 通过 :func:`validate_config` 的默认 :class:`AppConfig`
    @raises ConfigError 默认值自身不合法（属于开发期错误）
    """
    cfg = AppConfig()
    validate_config(cfg)
    return cfg


def _field_default(dataclass_type: type, name: str) -> Any:
    """取出 dataclass 字段的默认值（兼容 default_factory）。

    @param dataclass_type 目标 dataclass 类型
    @param name 字段名
    @return 默认值；无默认返回 None
    """
    for item in dataclass_fields(dataclass_type):
        if item.name != name:
            continue
        if item.default is not dataclasses.MISSING:
            return item.default
        if item.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return item.default_factory()  # type: ignore[misc]
    return None


def _coerce_like(value: Any, default: Any) -> Any:
    """把 JSON 反序列化值按默认值类型归一（list → tuple）。

    @param value 原始值
    @param default 该字段默认值（用于类型推断）
    @return 归一后的值
    """
    if isinstance(default, tuple) and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _build_section(path: str, dataclass_type: type, payload: Any) -> Any:
    """由稀疏字典构造一个子配置对象（未给出的字段用默认值）。

    @param path 配置路径（用于错误消息）
    @param dataclass_type 目标 dataclass 类型
    @param payload 稀疏字典或 None
    @return dataclass 实例
    @raises ConfigError payload 不是映射或含未知字段
    """
    if payload is None:
        return dataclass_type()
    if not isinstance(payload, Mapping):
        _fail(path, payload, "映射（dict）", "形如 {\"字段名\": 值}")
    known = {item.name for item in dataclass_fields(dataclass_type)}
    unknown = sorted(set(payload) - known)
    if unknown:
        _fail(path, unknown, f"已知字段 {sorted(known)}", f"删除未知字段 {unknown}（检查拼写）")
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in payload:
            continue
        raw = payload[name]
        default = _field_default(dataclass_type, name)
        kwargs[name] = _coerce_like(raw, default)
    return dataclass_type(**kwargs)


def _normalize_prize_table(raw: Any) -> dict[str, dict[int, int] | None]:
    """把奖金表归一为 ``{玩法: {奖级:int 金额} | None}``（JSON 的字符串键转回 int）。

    @param raw 原始奖金表
    @return 归一后的奖金表
    @raises ConfigError 结构非法
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        _fail("prize_table", raw, "映射", "形如 {'pick5': {5: 1000}}")
    result: dict[str, dict[int, int] | None] = {}
    for play, prize in raw.items():
        if prize is None:
            result[str(play)] = None
            continue
        if not isinstance(prize, Mapping):
            _fail(f"prize_table.{play}", prize, "映射或 None", "未确认奖级请填 None")
        tiers: dict[int, int] = {}
        for tier, amount in prize.items():
            try:
                tiers[int(tier)] = int(amount)
            except (TypeError, ValueError) as exc:
                _fail(f"prize_table.{play}.{tier}", amount, "整数元", f"改为整数金额：{exc}")
        result[str(play)] = tiers
    return result


def from_dict(payload: Mapping[str, Any] | None) -> AppConfig:
    """由（可能稀疏的）字典构造配置，缺省字段自动补默认值并校验。

    @param payload 配置字典；None 等价于全默认
    @return 校验通过的 :class:`AppConfig`
    @raises ConfigError 字段未知/类型不符/约束冲突
    """
    if payload is None:
        return default_config()
    if not isinstance(payload, Mapping):
        _fail("AppConfig", payload, "映射（dict）", "传入配置字典或 None")
    known = {item.name for item in dataclass_fields(AppConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        _fail("AppConfig", unknown, f"已知顶层字段 {sorted(known)}", f"删除未知字段 {unknown}（检查拼写）")

    kwargs: dict[str, Any] = {}
    for name, dataclass_type in _SUBCONFIG_TYPES:
        if name in payload:
            kwargs[name] = _build_section(name, dataclass_type, payload[name])
    for name in _APP_SCALAR_FIELDS:
        if name not in payload:
            continue
        if name == "prize_table":
            kwargs[name] = _normalize_prize_table(payload[name])
            continue
        default = _field_default(AppConfig, name)
        kwargs[name] = _coerce_like(payload[name], default)

    cfg = AppConfig(**kwargs)
    validate_config(cfg)
    return cfg


def to_dict(cfg: AppConfig) -> dict[str, Any]:
    """把配置序列化为可 JSON 化的纯字典（深拷贝，改写不影响原配置）。

    @param cfg 配置对象
    @return 嵌套字典（tuple 保持 tuple，便于 ``from_dict`` 往返）
    @raises ConfigError 传入非 AppConfig
    """
    if not isinstance(cfg, AppConfig):
        _fail("AppConfig", type(cfg).__name__, "AppConfig 实例", "调用 default_config() 构造")
    return dataclasses.asdict(cfg)


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """递归深合并：``patch`` 覆盖 ``base``，两侧都是映射时逐键递归。

    @param base 基础字典（不会被修改）
    @param patch 覆盖字典
    @return 新字典
    """
    result: dict[str, Any] = dict(base)
    for key, value in patch.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _read_config_file(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置文件。

    @param path 配置文件路径
    @return 配置字典
    @raises ConfigError 文件不存在/不可读/不是合法 JSON/顶层不是对象
    """
    file_path = Path(path)
    if not file_path.exists():
        _fail("path", str(file_path), "存在的配置文件路径", "检查 --config 路径或先创建该文件")
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail("path", str(file_path), "可读的 UTF-8 JSON 文件", f"检查文件权限：{exc}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        _fail("path", str(file_path), "合法 JSON", f"修复第 {exc.lineno} 行 JSON 语法：{exc.msg}")
    if not isinstance(payload, dict):
        _fail("path", str(file_path), "顶层为 JSON 对象", "配置文件顶层必须是 {...}")
    return payload


def load_config(
    overrides: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> AppConfig:
    """加载配置：默认值 → 配置文件 → CLI 覆盖，深合并后严格校验。

    @param overrides 稀疏覆盖字典（通常来自 CLI），支持嵌套（如 ``{"weights": {"co": 0.2}}``）
    @param path 可选 JSON 配置文件路径；None 表示不读文件
    @return 校验通过的 :class:`AppConfig`
    @raises ConfigError 文件问题或任何字段非法
    """
    payload: dict[str, Any] = _read_config_file(path) if path is not None else {}
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            _fail("overrides", overrides, "映射（dict）或 None", "传入 {\"字段\": 值} 形式的补丁")
        payload = _deep_merge(payload, dict(overrides))
    return from_dict(payload)


def merge_cli_overrides(cfg: AppConfig, args: Any) -> AppConfig:
    """把 argparse 命名空间中的有效参数合并进配置（未给出的参数保持原值）。

    【支持的 CLI → 配置映射】
    ``--history`` → ``history_path``；``--seed`` → ``seed``；``--pool-size`` → ``pool.main_default``；
    ``--k`` → ``covering.k``（并联动 play/t）；``--budget`` → ``covering.budget``；
    ``--warmup`` → ``backtest.warmup``。

    @param cfg 当前配置
    @param args argparse.Namespace（允许缺失属性）
    @return 合并并校验后的新配置
    @raises ConfigError 合并后非法
    """
    patch: dict[str, Any] = {}
    history = getattr(args, "history", None)
    if history is not None:
        patch["history_path"] = str(history)
    seed = getattr(args, "seed", None)
    if seed is not None:
        patch["seed"] = int(seed)
    pool_size = getattr(args, "pool_size", None)
    if pool_size is not None:
        patch.setdefault("pool", {})["main_default"] = int(pool_size)
    warmup = getattr(args, "warmup", None)
    if warmup is not None:
        patch.setdefault("backtest", {})["warmup"] = int(warmup)

    covering_patch: dict[str, Any] = {}
    k_value = getattr(args, "k", None)
    if k_value is not None:
        covering_patch["k"] = int(k_value)
        matched_play = next(
            (play for play, (play_k, _play_t) in PLAY_TIER_DEFAULTS.items() if play_k == int(k_value)),
            None,
        )
        if matched_play is not None:
            covering_patch["play"] = matched_play
            covering_patch["t"] = PLAY_TIER_DEFAULTS[matched_play][1]
        else:
            raise ConfigError(
                f"CLI 参数 --k={k_value} 非法：未找到匹配的玩法配置（支持玩法: "
                f"{list(SUPPORTED_PLAYS)}，对应 k 值: {sorted({v[0] for v in PLAY_TIER_DEFAULTS.values()})}）"
            )
    budget = getattr(args, "budget", None)
    if budget is not None:
        covering_patch["budget"] = int(budget)
    if covering_patch:
        patch.setdefault("covering", {}).update(covering_patch)

    if not patch:
        return cfg
    return from_dict(_deep_merge(to_dict(cfg), patch))


def config_hash(cfg: AppConfig) -> str:
    """计算配置指纹（sha256 前 16 位十六进制），用于 ML 缓存 key 与回测血缘。

    @param cfg 配置对象
    @return 16 位十六进制字符串
    @raises ConfigError 传入非 AppConfig
    """
    payload = to_dict(cfg)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]