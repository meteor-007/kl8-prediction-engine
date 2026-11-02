# -*- coding: utf-8 -*-
"""快乐8五层重构 —— 全系统跨层数据结构（datatypes，contract.md §3.4 + §4 冻结）。

【定位】
本模块是**基础设施层**：只依赖 ``kl8.constants`` / ``kl8.errors``，被 layer1–layer5、
evaluation、reporting、pipeline 共同引用；**禁止** import 任何 ``layerN``，否则破坏
C5 五层单向依赖。

【不可变约定】
所有数据结构均为 ``@dataclass(frozen=True)``。强制不可变的原因：
1. 防止下游层把上游算好的特征数组原地改掉（跨层污染极难排查）；
2. 让"同一期只装配一次、多报告复用"成为可能（C6 零循环内重计算）。

【索引约定（C9）】
- ``np.ndarray`` 一律 0-based，下标 ``i`` ↔ 号码 ``i+1``（1..80）；
- 对外展示 / 存储的号码一律 1-based ``int``；
- ``as_of_index`` / ``end`` / ``t`` 是 0-based 且**含端点**，即竖直切片 ``X[0..t]``。

【跨层类型的落点说明】
``BaselineSnapshot`` 属于 §4.7 的共享结构，按 §2.1「datatypes.py 承载全部跨层数据结构」
落在本模块。其 ``chi2`` / ``nist`` 字段的值分别是 layer2 的 ``ChiSquareResult`` /
``NistResult`` 实例；若在此处硬 import 会形成 ``datatypes → layer2 → datatypes`` 循环，
故本模块用 ``Mapping[str, Any]`` / ``tuple[Any, ...]`` 做宽类型收口（字段名与语义不变）。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping

import numpy as np

from kl8.constants import DATE_PATTERN, K_DRAW, N_NUM
from kl8.errors import DataFormatError, InsufficientDataError

#: 80 维浮点向量别名（0-based 索引 ↔ 号码 i+1）。
NumArray = np.ndarray
#: 80 维布尔向量别名。
Mask = np.ndarray
#: 80 维投票向量别名（int8，取值 0..5）。
Votes = np.ndarray

SNAPSHOT_SCHEMA_VERSION: str = "fs-v1"
"""特征快照 schema 版本（契约 §4.1 固定值，ML 缓存 key 的组成部分）。"""


def assert_finite(arr: np.ndarray, name: str) -> None:
    """断言数组内不存在 NaN / Inf（C8：层的出口必须拦截非法数值）。

    @param arr 待检查数组
    @param name 字段名（用于中文错误消息定位）
    @raises DataFormatError 存在 NaN 或 Inf
    """
    values = np.asarray(arr)
    if values.size == 0:
        return
    if not np.all(np.isfinite(values)):
        bad = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        raise DataFormatError(
            f"字段 {name} 含 {bad} 个 NaN/Inf 非法数值；"
            "请检查该特征的除零防守（分母加 cfg.eps 或显式分支）后重跑"
        )


def _validate_number_iterable(numbers: Any, where: str) -> None:
    """校验号码集合：必须是 20 个 1..80 的不重复整数。

    @param numbers 待校验对象（任何可迭代）
    @param where 调用位置描述（用于错误消息）
    @raises DataFormatError 个数/取值/重复性不合规
    """
    try:
        values = [int(x) for x in numbers]
    except (TypeError, ValueError) as exc:
        raise DataFormatError(f"{where} 号码集合不可解析为整数：{exc}") from exc
    if len(values) != K_DRAW:
        raise DataFormatError(
            f"{where} 号码个数为 {len(values)}，要求恰好 {K_DRAW} 个；请检查原始开奖数据"
        )
    if len(set(values)) != len(values):
        dup = sorted({x for x in values if values.count(x) > 1})
        raise DataFormatError(f"{where} 号码存在重复：{dup}；每期 20 码必须互不重复")
    out_of_range = [x for x in values if not 1 <= x <= N_NUM]
    if out_of_range:
        raise DataFormatError(
            f"{where} 号码越界：{sorted(out_of_range)[:5]}，要求全部落在 1..{N_NUM}"
        )


# =============================================================================
# 一、开奖历史实体与时序容器（§3.4）
# =============================================================================
@dataclass(frozen=True)
class Draw:
    """单期开奖记录实体（不可变）。

    【字段】
    - ``date``：开奖日期字符串，严格 ``YYYY-MM-DD``；
    - ``period``：期号（7 位正整数，如 2026291）；
    - ``numbers``：20 个 1..80 不重复号码的冻结集合（1-based）；
    - ``idx``：该期在**升序** Dataset 中的 0-based 下标（由 ``history`` 赋值）。
    """

    date: str
    """开奖日期（``YYYY-MM-DD``）。"""

    period: int
    """期号（正整数，形如 2026291）。"""

    numbers: frozenset[int]
    """20 个 1..80 不重复号码（1-based 冻结集合）。"""

    idx: int = -1
    """在升序 Dataset 中的 0-based 索引；-1 表示尚未定位。"""

    def __post_init__(self) -> None:
        """构造期自校验（C8 空/非法安全）。

        @raises DataFormatError 日期/期号/号码不合规
        """
        if not isinstance(self.date, str) or not DATE_PATTERN.match(self.date):
            raise DataFormatError(
                f"开奖日期 {self.date!r} 非法，要求 YYYY-MM-DD（如 2026-11-01）；"
                "请检查历史文件行格式 date:..."
            )
        if isinstance(self.period, bool) or not isinstance(self.period, int) or self.period <= 0:
            raise DataFormatError(
                f"期号 {self.period!r} 非法，要求正整数（如 2026291）；请检查历史文件 period:..."
            )
        _validate_number_iterable(self.numbers, f"第 {self.period} 期")

    def as_asc(self) -> tuple[int, ...]:
        """返回升序号码元组（1-based，便于展示与工单比对）。

        @return 20 个号码的升序元组
        """
        return tuple(sorted(self.numbers))

    def row(self) -> np.ndarray:
        """返回该期的 80 维二值化开奖向量。

        @return shape (80,) float64，开出位置为 1.0，其余 0.0
        """
        row = np.zeros(N_NUM, dtype=np.float64)
        row[np.fromiter((n - 1 for n in self.numbers), dtype=np.int64, count=len(self.numbers))] = 1.0
        return row


@dataclass(frozen=True)
class Dataset:
    """开奖历史数据集（严格按 (date, period) 升序、不可变）。

    【字段】
    - ``draws``：严格升序的 Draw 元组；
    - ``draws_by_period``：期号 → Draw 的映射（O(1) 取期）；
    - ``source_path``：来源文件路径（可追溯）；
    - ``warnings``：解析期产生的告警（坏行跳过、期号断档等）。
    """

    draws: tuple[Draw, ...]
    """严格升序的开奖记录元组。"""

    draws_by_period: Mapping[int, Draw]
    """期号 → 开奖记录映射（与 draws 同一批对象）。"""

    source_path: str = ""
    """来源文件路径（可追溯血缘）。"""

    warnings: tuple[str, ...] = ()
    """解析/截断期告警（中文）。"""

    def __post_init__(self) -> None:
        """构造期自校验：升序、期号唯一、索引与映射自洽。

        @raises DataFormatError 顺序/唯一性/映射不一致
        """
        keys = [(draw.date, draw.period) for draw in self.draws]
        if keys != sorted(keys):
            raise DataFormatError(
                "Dataset.draws 必须按 (date, period) 严格升序；请先用 history.load_dataset 重排后再构造"
            )
        if len(self.draws_by_period) != len(self.draws):
            raise DataFormatError(
                f"Dataset 期号映射条数 {len(self.draws_by_period)} 与 draws 条数 {len(self.draws)} 不一致"
            )
        for draw in self.draws:
            if self.draws_by_period.get(draw.period) is not draw:
                raise DataFormatError(
                    f"Dataset.draws_by_period 与 draws 不自洽（期号 {draw.period}）；请重建映射"
                )

    @property
    def T(self) -> int:
        """历史总期数 T。

        @return 期数（可能为 0）
        """
        return len(self.draws)

    @property
    def periods(self) -> tuple[int, ...]:
        """按期序升序排列的期号元组。

        @return 期号元组（长度 T）
        """
        return tuple(draw.period for draw in self.draws)

    @property
    def latest(self) -> Draw:
        """最新一期开奖记录。

        @return 最后一期 Draw
        @raises InsufficientDataError 数据集为空
        """
        if not self.draws:
            raise InsufficientDataError("开奖数据为空，无法获取最新一期；请先同步或导入 kl8_history.txt")
        return self.draws[-1]

    @cached_property
    def _hit_matrix(self) -> np.ndarray:
        """构造并缓存 (T,80) 二值状态矩阵（一次性向量化，避免逐期重算）。

        @return shape (T,80) float64
        """
        total = len(self.draws)
        matrix = np.zeros((total, N_NUM), dtype=np.float64)
        if total == 0:
            return matrix
        index = np.fromiter(
            (n for draw in self.draws for n in draw.numbers), dtype=np.int64, count=total * K_DRAW
        ).reshape(total, K_DRAW)
        matrix[np.arange(total)[:, None], index - 1] = 1.0
        return matrix

    def hit_matrix(self) -> np.ndarray:
        """返回 (T,80) 二值状态矩阵 X（行和恒为 K=20）。

        @return shape (T,80) float64；T=0 时返回 (0,80) 空矩阵
        """
        return self._hit_matrix.copy()

    def index_of_period(self, period: int) -> int:
        """查询指定期号在升序序列中的 0-based 下标。

        @param period 期号
        @return 0-based 下标
        @raises KeyError 期号不存在
        """
        try:
            return self.draws_by_period[period].idx
        except KeyError as exc:
            first = self.periods[0] if self.periods else "空"
            last = self.periods[-1] if self.periods else "空"
            raise KeyError(
                f"期号 {period} 不在数据集内（共 {self.T} 期，范围 {first}..{last}）；"
                "请检查 --period 参数或先同步数据"
            ) from exc

    def target_period(self) -> int:
        """待预测期号 = 最新已开奖期号 + 1。

        @return 下一期期号
        @raises InsufficientDataError 数据集为空
        """
        return self.latest.period + 1


class TimeSeries:
    """时序预计算容器（不可变语义）：提供 0-1 矩阵与遗漏矩阵的 O(1) 取用。

    【缓存内容】
    - ``hit``：(T,80) float64 开奖矩阵；
    - ``gap``：(T,80) int32 遗漏矩阵，``gap[t, i]`` 表示第 t 期结束时号码 i+1 的连续未开期数
      （若第 t 期开出则为 0）。

    【为什么预计算 gap】
    遗漏特征是 layer3 的高频输入（R13/R14/R5），若每次现算会退化为 O(T) 重复扫描；
    这里用一次前缀最大值扫描向量化算完（C6：禁止 80 次 Python 循环做矩阵运算）。
    """

    def __init__(self, ds: Dataset) -> None:
        """基于数据集预计算时序矩阵。

        @param ds 升序开奖数据集
        """
        self.T: int = ds.T
        self.hit: np.ndarray = ds.hit_matrix()
        self.gap: np.ndarray = self._build_gap(self.hit)

    @staticmethod
    def _build_gap(hit: np.ndarray) -> np.ndarray:
        """向量化构造 (T,80) 遗漏矩阵。

        【算法】对每列求"最近一次开出位置"的前缀最大值：若当期开出则遗漏为 0，
        否则遗漏 = 当期下标 - 最近开出下标（从未开出时最近下标为 -1，故遗漏 = t+1）。

        @param hit (T,80) 0-1 矩阵
        @return (T,80) int32 遗漏矩阵；T=0 返回 (0,80)
        """
        total = hit.shape[0]
        if total == 0:
            return np.zeros((0, N_NUM), dtype=np.int32)
        index = np.arange(total, dtype=np.int64)[:, None]
        is_hit = hit > 0.5
        last_hit = np.maximum.accumulate(np.where(is_hit, index, -1), axis=0)
        gap = np.where(is_hit, 0, index - last_hit)
        return gap.astype(np.int32)

    def _check_end(self, end: int) -> int:
        """校验并归一化窗口右端点。

        @param end 0-based 含端点索引
        @return 原索引
        @raises IndexError end 越界或非整数
        """
        if isinstance(end, bool) or not isinstance(end, (int, np.integer)):
            raise IndexError(f"窗口右端点 end 必须是整数，收到 {end!r}")
        value = int(end)
        if value < 0 or value >= self.T:
            raise IndexError(f"窗口右端点 end={value} 越界（当前 T={self.T}，合法范围 0..{self.T - 1}）")
        return value

    def window_hits(self, end: int, w: int | None) -> np.ndarray:
        """取截止 end（含）的最近 w 期开奖矩阵。

        @param end 0-based 含端点索引
        @param w 窗口长度；``None`` 表示全量历史（ALL）
        @return shape (w_eff,80) float64，``w_eff = max(1, min(w, end+1))``
        @raises IndexError end 越界
        @raises ValueError w <= 0
        """
        value = self._check_end(end)
        if w is None:
            return self.hit[: value + 1]
        if isinstance(w, bool) or not isinstance(w, (int, np.integer)) or int(w) <= 0:
            raise ValueError(f"窗口长度 w 必须为正整数或 None(ALL)，收到 {w!r}")
        w_eff = max(1, min(int(w), value + 1))
        return self.hit[value + 1 - w_eff : value + 1]

    def freq(self, end: int, w: int | None) -> np.ndarray:
        """取截止 end 的最近 w 期频率向量 Freq_i(w)=c_i(w)/w_eff。

        @param end 0-based 含端点索引
        @param w 窗口长度；``None`` 表示 ALL
        @return shape (80,) float64，取值 [0,1]
        @raises IndexError end 越界
        @raises ValueError w <= 0
        """
        window = self.window_hits(end, w)
        return window.mean(axis=0)


# =============================================================================
# 二、随机性基准摘要（§4.7，layer2 产出、layer3 装入快照）
# =============================================================================
@dataclass(frozen=True)
class BaselineSnapshot:
    """随机性基准摘要（§4.7）：把 layer2 的偏态探测结果折叠成快照的一小块。

    【宽类型说明】``chi2`` 的值是 ``layer2_baseline.chisq.ChiSquareResult``，
    ``nist`` 的元素是 ``layer2_baseline.randomness.NistResult``；为避免
    ``datatypes → layer2 → datatypes`` 循环依赖，此处用 ``Any`` 收口（字段名不变）。
    """

    z_10G: np.ndarray
    """10G 窗口标准化偏离度 Z_i(10)，(80,) float64。"""

    z_25G: np.ndarray
    """25G 窗口 Z 值，(80,) float64。"""

    z_50G: np.ndarray
    """50G 窗口 Z 值，(80,) float64。"""

    z_ALL: np.ndarray
    """ALL 窗口 Z 值，(80,) float64。"""

    significant_any: np.ndarray
    """任一窗口 |Z|>1.96 的标记，(80,) bool。"""

    runs_z: np.ndarray
    """游程检验 z 统计量，(80,) float64。"""

    runs_labels: tuple[str, ...]
    """游程检验结论标签（长度 80，如 "聚簇连出"/"频繁跳跃"/"无偏"/"退化"）。"""

    chi2: Mapping[str, Any] = field(default_factory=dict)
    """卡方拟合优度结果：窗口键 → ChiSquareResult。"""

    nist: tuple[Any, ...] = ()
    """NIST 轻量随机性测试结果（固定 4 项、固定顺序）。"""

    entropy: Mapping[str, float] = field(default_factory=dict)
    """信息熵指标（如 {"freq_10G": …, "freq_ALL": …}）。"""

    mi_top_pairs: tuple[tuple[str, str, float], ...] = ()
    """跨期互信息最高的号码对（(号码A, 号码B, MI)）。"""

    expected_repeat: float = 0.0
    """相邻期理论重号期望 E(Repeat)=5.0。"""


# =============================================================================
# 三、特征快照（§4.1，layer3 唯一出口 / layer4 唯一入口）
# =============================================================================
_SNAPSHOT_ARRAY_FIELDS: tuple[str, ...] = (
    "freq_10",
    "freq_25",
    "freq_50",
    "freq_all",
    "mom_10_25",
    "mom_25_50",
    "gap_ratio",
    "pressure",
    "follow_10",
    "follow_25",
    "follow_50",
    "follow_all",
    "co_10",
    "co_25",
    "co_50",
    "co_all",
    "co_centered_mean",
    "pagerank",
    "eigen",
    "community_strength",
    "gap_mean",
    "follow2_all",
    "ml_prob",
    "gap",
    "golden_cross",
    "community_id",
    "community_size",
    "triad_count",
    "zone4",
    "zone8",
    "is_odd",
    "is_prime",
    "size_class",
    "tail",
    "road012",
)
"""FeatureSnapshot 全部 80 维数组字段（固定顺序，供形状校验与防穿越指纹使用）。"""


@dataclass(frozen=True)
class FeatureSnapshot:
    """单期特征快照：一次装配、全链路复用（R23 / C6 零循环内重计算）。

    【语义】``as_of_index = t``（0-based 含端点），只用 ``X[0..t]`` 构造，预测 ``t+1``；
    ``target_period`` 即 ``t+1`` 的期号。所有 ``np.ndarray`` 长度 80，
    0-based 下标 ``i`` ↔ 号码 ``i+1``。

    【装配入口】仅 ``kl8/layer3_features/features.py::build_feature_snapshot``；
    【消费入口】layer4 的 ``borda`` / ``rules`` / ``scoring`` 只读访问，禁止改写。
    """

    # —— 元信息 ——
    as_of_index: int
    """t（0-based 含端点）：用 X[0..t] 预测 t+1；``evaluation.max_accessed_t`` 必须等于它。"""

    target_period: int
    """t+1 的期号。"""

    n_history: int
    """可用期数 = t+1。"""

    effective_windows: Mapping[str, int]
    """各尺度实际生效窗口 {"10G":w_eff,"25G":…,"50G":…,"ALL":T}。"""

    cold_start: bool
    """任一窗口 w_eff < w 即为 True（冷启动回退已发生）。"""

    schema_version: str
    """快照 schema 版本，固定 "fs-v1"（ML 缓存 key 组成部分）。"""

    config_hash: str
    """配置指纹 config.config_hash(cfg)（ML 缓存 key 组成部分）。"""

    # —— 连续特征（float64, shape=(80,)）——
    freq_10: np.ndarray
    """10G 出现频率 Freq_i(10)=c_i(10)/w_eff。"""

    freq_25: np.ndarray
    """25G 出现频率。"""

    freq_50: np.ndarray
    """50G 出现频率。"""

    freq_all: np.ndarray
    """ALL 出现频率 c_i(ALL)/T。"""

    mom_10_25: np.ndarray
    """短中周期动量差 freq_10 - freq_25。"""

    mom_25_50: np.ndarray
    """中长周期动量差 freq_25 - freq_50。"""

    gap_ratio: np.ndarray
    """遗漏比 O_{t,i} / max(μ_gap,i, 3.0)。"""

    pressure: np.ndarray
    """高斯核遗漏回补压力 exp(-0.5((gap_ratio-1.2)/0.6)^2)，clip 到 [0,1]。"""

    follow_10: np.ndarray
    """10G 跟随分 Follow_j(10)=Σ_{i∈S_t}(M_{i→j}-p0)。"""

    follow_25: np.ndarray
    """25G 跟随分。"""

    follow_50: np.ndarray
    """50G 跟随分。"""

    follow_all: np.ndarray
    """ALL 窗口跟随分。"""

    co_10: np.ndarray
    """10G 当期共现伴生拉动分 CoScore_j(10)。"""

    co_25: np.ndarray
    """25G 共现伴生拉动分。"""

    co_50: np.ndarray
    """50G 共现伴生拉动分。"""

    co_all: np.ndarray
    """ALL 窗口共现伴生拉动分。"""

    co_centered_mean: np.ndarray
    """C̄o 原始值 mean(co_25, co_50, co_all)（仅报告展示；打分用秩空间，见裁决 A2）。"""

    pagerank: np.ndarray
    """正值共现图 PageRank 中心度（归一化到 [0,1]）。"""

    eigen: np.ndarray
    """特征向量中心度（归一化到 [0,1]）。"""

    community_strength: np.ndarray
    """所属社区内平均正共现强度（无社区时 0）。"""

    gap_mean: np.ndarray
    """历史平均遗漏 μ_gap,i（μ=0 时取 3.0）。"""

    follow2_all: np.ndarray
    """二阶滞后矩阵 M^(2) 的 ALL 窗跟随分（cfg.markov.order=1 时全 0）。"""

    ml_prob: np.ndarray
    """PAV 校准后开出概率 ∈[0,1]；ml.enabled=False 时全为 0.25。"""

    # —— 离散/整型/布尔特征 ——
    gap: np.ndarray
    """当前遗漏 O_{t,i}（0 = 上期开出），int32。"""

    golden_cross: np.ndarray
    """动量金叉加速标记，(80,) bool。"""

    community_id: np.ndarray
    """Louvain 社区编号（-1=孤立点/无图），int32。"""

    community_size: np.ndarray
    """所属社区节点数（孤立点 1），int32。"""

    triad_count: np.ndarray
    """该号码参与的三元共现团个数，int32。"""

    zone4: np.ndarray
    """四区归属 0..3，int8。"""

    zone8: np.ndarray
    """八区归属 0..7，int8。"""

    is_odd: np.ndarray
    """奇数标记，(80,) bool。"""

    is_prime: np.ndarray
    """质数标记（1 为 False），(80,) bool。"""

    size_class: np.ndarray
    """大中小分类 0=小 1=中 2=大（裁决 A3），int8。"""

    tail: np.ndarray
    """尾号 num % 10，int8。"""

    road012: np.ndarray
    """012 路 num % 3，int8。"""

    # —— 映射 / 附属字段（可缺省）——
    follow_diag: Mapping[str, np.ndarray] = field(default_factory=dict)
    """重号（对角）惯性分：窗口键 → (80,) 数组。"""

    follow_offdiag: Mapping[str, np.ndarray] = field(default_factory=dict)
    """异号跟随分（非对角）：窗口键 → (80,) 数组。"""

    companion: Mapping[int, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    """号码 → 前 3 伴生码及归一化亲和度（1.0 为最强）。"""

    co_matrices: Mapping[str, np.ndarray] = field(default_factory=dict)
    """窗口键 → (80,80) 中心化共现矩阵 A_ij。"""

    triads_total: int = 0
    """全局三元共现团总数。"""

    ml_available: bool = False
    """LightGBM 是否可用（降级时为 False）。"""

    ml_backend: str = "uniform"
    """ML 后端："lightgbm" / "ema" / "uniform"。"""

    baseline: Any = None
    """随机性基准摘要（layer2 的 BaselineSnapshot；允许 None 表示未计算）。"""

    warnings: tuple[str, ...] = ()
    """降级/退化/冷启动说明（中文可读）。"""

    def __post_init__(self) -> None:
        """构造期形状校验：所有 80 维数组必须是 (80,)。

        @raises DataFormatError 任一特征数组形状不是 (80,)
        """
        for name in _SNAPSHOT_ARRAY_FIELDS:
            value = getattr(self, name)
            shape = np.shape(value)
            if shape != (N_NUM,):
                raise DataFormatError(
                    f"FeatureSnapshot.{name} 形状为 {shape}，要求 ({N_NUM},)；"
                    "请检查 layer3 装配是否漏传或传错维度"
                )

    @property
    def max_accessed_t(self) -> int:
        """本快照实际访问过的历史最大索引（防穿越审计用，R47）。

        @return 0-based 最大索引；构造期已保证等于 ``as_of_index``
        """
        return self.as_of_index

    def arrays(self) -> dict[str, np.ndarray]:
        """导出全部 80 维数组（用于指纹哈希与自检遍历）。

        @return 字段名 → np.ndarray（固定顺序）
        """
        return {name: np.asarray(getattr(self, name)) for name in _SNAPSHOT_ARRAY_FIELDS}


# =============================================================================
# 四、规则与池（§4.2 / §4.3）
# =============================================================================
@dataclass(frozen=True)
class RuleVerdict:
    """单条正交规则（R1–R5）的判定结果（§4.2）。

    【为什么返回掩码而不是号码列表】掩码可与其他规则向量化对齐做交集/投票，
    避免 80 次 Python 循环（C6），也让归因统计（R45）直接复用同一份掩码。
    """

    rule_id: str
    """规则编号，固定 "R1".."R5"。"""

    rule_name: str
    """规则中文名（写入报告「五规则共振榜」）。"""

    mask: np.ndarray
    """命中掩码，(80,) bool。"""

    pass_count: int = -1
    """mask.sum() 的缓存；-1 表示由构造期自动计算。"""

    thresholds: Mapping[str, float] = field(default_factory=dict)
    """本次判定使用的阈值快照（来自 cfg.rules，供复现与审计）。"""

    def __post_init__(self) -> None:
        """形状与计数自校验（pass_count 为 -1 时自动填充）。

        @raises DataFormatError mask 形状不是 (80,)
        """
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != (N_NUM,):
            raise DataFormatError(
                f"RuleVerdict[{self.rule_id}].mask 形状为 {mask.shape}，要求 ({N_NUM},)"
            )
        if np.asarray(self.mask).dtype != np.bool_:
            object.__setattr__(self, "mask", mask)
        if self.pass_count < 0:
            object.__setattr__(self, "pass_count", int(mask.sum()))


@dataclass(frozen=True)
class NumberDetail:
    """单号码全维度明细（§4.3.1）：报告与 Excel 的**唯一数据源**。

    避免报告层再从 6 个数组各自 indexing（易错且无法审计）。
    """

    num: int
    """号码（1-based）。"""

    final: float
    """FinalScore。"""

    base: float
    """BaseScore（归一化加权和）。"""

    votes: int
    """共振票数 V_i ∈ {0..5}。"""

    rank: int
    """FinalScore 排名（1=最高）。"""

    zone4: int
    """四区下标 0..3。"""

    zone8: int
    """八区下标 0..7。"""

    is_odd: bool
    """是否奇数。"""

    is_prime: bool
    """是否质数。"""

    size_class: int
    """大中小 0/1/2。"""

    tail: int
    """尾号 0..9。"""

    is_repeat: bool
    """是否上期重号。"""

    r1: bool
    """R1 是否命中。"""

    r2: bool
    """R2 是否命中。"""

    r3: bool
    """R3 是否命中。"""

    r4: bool
    """R4 是否命中。"""

    r5: bool
    """R5 是否命中。"""

    freq_10: float
    """10G 频率。"""

    freq_25: float
    """25G 频率。"""

    freq_50: float
    """50G 频率。"""

    freq_all: float
    """ALL 频率。"""

    pressure: float
    """回补压力。"""

    gold_flag: bool
    """是否入选金胆池。"""

    in_main: bool
    """是否入选复式池。"""


@dataclass(frozen=True)
class PoolResult:
    """空间形态均衡过滤后的候选池结果（§4.3 / R34–R36）。"""

    gold: tuple[int, ...]
    """金胆池（1-based 升序，长度 ∈ [gold_min, gold_max]）。"""

    main: tuple[int, ...]
    """复式池（1-based 升序，长度 ∈ [main_min, main_max]，默认 12）。"""

    votes: np.ndarray
    """五规则共振票数 V_i ∈ {0..5}，(80,) int8。"""

    scores: np.ndarray
    """FinalScore，(80,) float64。"""

    order: tuple[int, ...]
    """全部 80 码按 FinalScore 降序（并列按号码升序），1-based。"""

    details: Mapping[int, NumberDetail]
    """号码(1-based) → 逐号明细，报告唯一数据源。"""

    warnings: tuple[str, ...] = ()
    """约束放宽/降级说明（中文可读）。"""

    relaxed: tuple[str, ...] = ()
    """已放宽的约束键（"same_tail"/"parity"/"zone4"/"repeat_ratio"）。"""

    feasible: bool = True
    """是否在全部硬约束下达成目标规模（False 表示已放宽）。"""

    repeat_count: int = 0
    """池内上期重号个数（main ∩ 上期开奖号）。"""

    new_count: int = -1
    """非重号个数 size-repeat_count；-1 表示构造期自动计算。"""

    size: int = -1
    """len(main)；-1 表示构造期自动计算。"""

    def __post_init__(self) -> None:
        """自动补齐派生字段（size/new_count），保证三者永远自洽。

        @raises DataFormatError votes/scores 形状不是 (80,)
        """
        for name in ("votes", "scores"):
            shape = np.shape(getattr(self, name))
            if shape != (N_NUM,):
                raise DataFormatError(f"PoolResult.{name} 形状为 {shape}，要求 ({N_NUM},)")
        size = len(self.main) if self.size < 0 else self.size
        object.__setattr__(self, "size", size)
        if self.new_count < 0:
            object.__setattr__(self, "new_count", size - self.repeat_count)


# =============================================================================
# 五、覆盖设计（§4.5）与注单方案（§4.4）
# =============================================================================
@dataclass(frozen=True)
class CoveringReport:
    """覆盖设计 C(v,k,t,m) 的求解明细（§4.5 / R37–R40）。"""

    pool: tuple[int, ...]
    """号码池（1-based 升序），v = len(pool)。"""

    v: int
    """池规模 v。"""

    k: int
    """单注号码个数 k。"""

    t: int
    """保底命中数 t。"""

    m: int
    """池内实际命中数假设 m（用户/玩法假设参数，非预测）。"""

    tickets_pos: tuple[tuple[int, ...], ...]
    """位置索引注单（0-based，0..v-1）—— 覆盖计算的内部真值。"""

    tickets: tuple[tuple[int, ...], ...]
    """映射到号码后的注单（1-based），与 tickets_pos 逐注对应。"""

    n_tickets: int
    """注数。"""

    coverage_ratio: float
    """已覆盖 m 子集比例 ∈[0,1]。"""

    achieved_t: int
    """实测最小 |B∩M|（穷举结果，必须 ≥ t）。"""

    verified: bool
    """verify_guarantee 是否通过（未通过禁止出单）。"""

    method: str
    """"table"/"ilp"/"greedy"/"anneal"/"greedy+anneal"。"""

    degraded: bool = False
    """是否发生降级（ILP 超规模/超时 → 贪心；查表未命中 → 求解器）。"""

    degrade_reason: str = ""
    """降级原因（中文；未降级为空串）。"""

    elapsed_ms: int = 0
    """求解耗时（毫秒）。"""

    warnings: tuple[str, ...] = ()
    """其它告警。"""


@dataclass(frozen=True)
class TicketPlan:
    """单个玩法的最终注单方案（§4.4 / R41）。"""

    play: str
    """玩法内部键："pick5" / "pick7"（中文名见 ``play_label``）。"""

    k: int
    """单注号码个数（选五 5，选七 7）。"""

    budget: int
    """预算注数（默认选五 20、选七 10；裁决 A8）。"""

    tickets: tuple[tuple[int, ...], ...]
    """注单（1-based 号码、每注升序、注间不重复），长度 ≤ budget。"""

    guarantee_level: Mapping[str, Any]
    """保底等级 {"t","m","text","coverage_ratio","verified"}。"""

    covering: CoveringReport
    """求解明细（含 method/降级原因）。"""

    cost_yuan: int
    """总成本（整数元）= len(tickets) × cfg.cost_per_ticket。"""

    method: str
    """"table"/"ilp"/"greedy"/"anneal"。"""

    warnings: tuple[str, ...] = ()
    """预算无法同时满足保底时的降级说明。"""

    @property
    def play_label(self) -> str:
        """玩法中文名。

        @return "选五" 或 "选七"（未知玩法回退为原文）
        """
        return {"pick5": "选五", "pick7": "选七"}.get(self.play, self.play)

    @property
    def n_tickets(self) -> int:
        """实际注数。

        @return len(tickets)
        """
        return len(self.tickets)


# =============================================================================
# 六、回测（§4.6）
# =============================================================================
@dataclass(frozen=True)
class BacktestRow:
    """Walk-Forward 单期回测明细（§4.6）：一期一行，记录"预测→揭晓→结算"闭环。"""

    period: int
    """被预测期（t+1）的真实期号。"""

    as_of_index: int
    """t（0-based，特征只用到 X[0..t]）。"""

    as_of_period: int
    """t 的期号。"""

    scores: np.ndarray
    """当期 FinalScore，(80,) float64。"""

    votes: np.ndarray
    """当期票数，(80,) int8。"""

    top_picks: Mapping[int, tuple[int, ...]]
    """{K: 排名前 K 的号码（1-based，降序)}，K ∈ cfg.backtest.ks。"""

    hit_at: Mapping[int, int]
    """{K: Top-K 命中个数}。"""

    gold: tuple[int, ...]
    """当期金胆池（1-based）。"""

    main: tuple[int, ...]
    """当期复式池（1-based）。"""

    gold_hit: int
    """金胆池命中个数。"""

    main_hit: int
    """复式池命中个数。"""

    tickets: tuple[tuple[int, ...], ...] = ()
    """当期注单（1-based；include_tickets=False 时为空）。"""

    best_ticket_hit: int = 0
    """单注最高命中数（无注单为 0）。"""

    cost_yuan: int = 0
    """当期投入（整数元）。"""

    prize_yuan: int = 0
    """当期奖金（整数元；未确认奖级玩法记 0 且 prize_known=False）。"""

    prize_known: bool = False
    """该玩法奖金表是否已确认（裁决 A10）。"""

    ok: bool = True
    """该期数据是否完整（历史够、无解析告警）。"""

    warnings: tuple[str, ...] = ()
    """该期降级/异常说明。"""


@dataclass(frozen=True)
class BacktestReport:
    """Walk-Forward 回测总报告（§4.6 / R42–R47）。"""

    n_sample: int
    """实际回测期数（= len(rows)）。"""

    t0: int
    """冷启动期数（默认 300）。"""

    start_period: int
    """回测覆盖起始期号。"""

    end_period: int
    """回测覆盖结束期号。"""

    rows: tuple[BacktestRow, ...]
    """逐期明细（按期号升序）。"""

    lift: Mapping[int, Any] = field(default_factory=dict)
    """{K: LiftStat}（LiftStat 定义于 evaluation.metrics）。"""

    high_tier: Mapping[str, Any] = field(default_factory=dict)
    """{玩法:奖级: TierStat}，如 "pick5:5" / "pick5:4+5" / "pick3:3"。"""

    attribution: Any = None
    """规则归因报告（AttributionReport）。"""

    bankroll: Any = None
    """资金曲线统计（BankrollStat）。"""

    config_hash: str = ""
    """配置指纹（回测血缘）。"""

    self_check: Mapping[str, bool] = field(default_factory=dict)
    """口径自洽位（n_sample==len(rows)、lift 可重算、bankroll 可重算、no_lookahead）。"""

    elapsed_s: float = 0.0
    """回测总耗时（秒）。"""

    warnings: tuple[str, ...] = ()
    """全局告警。"""

    def __post_init__(self) -> None:
        """自检位回填：``n_sample == len(rows)`` 必须恒成立。

        @raises DataFormatError n_sample 与行数不一致
        """
        if self.n_sample != len(self.rows):
            raise DataFormatError(
                f"BacktestReport.n_sample={self.n_sample} 与 rows 行数 {len(self.rows)} 不一致；"
                "请在构造前用 len(rows) 赋值，避免报告口径漂移"
            )