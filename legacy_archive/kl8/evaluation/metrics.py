"""回测核心评价指标：Lift@K、高奖级触发率、规则归因与资金曲线（契约 R43–R46、§3.12）。

统计量包括：
- Top-K 平均命中率与提升度 Lift@K 及单侧 t 检验；`is_signal` 为**三条件与**：
  ``n ≥ cfg.min_n_signal`` 且 ``lift > cfg.lift_signal`` 且 ``p_value < alpha_adj``，
  ``alpha_adj = alpha_sig / len(ks)``（Bonferroni，可配置关闭）—— 契约 v1.3.0 / R43；
- 高奖级实际触发率对比超几何理论基准（high_tier_rate），高奖级只在 n≥500 时才允许
  宣称"显著"（R44，措辞由报告层执行）；
- 五规则独立胜率与 `V≥3` vs `V≤1` 超额命中的两比例检验（rule_attribution，R45）；
- 整数元资金曲线、ROI（万分比整数 roi_bp 为权威口径）、最大回撤与最长连挂（bankroll）；
- 报告自洽性机检（self_check）：样本数、lift 独立重算、bankroll 独立重算、
  **防穿越真实计数**（P8-H4：禁止硬编码 True）。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from kl8.config import BacktestConfig
from kl8.constants import P0, ROI_NA_NO_TICKET_TEXT, ROI_NA_TEXT, RULE_IDS, hypergeometric_pmf
from kl8.datatypes import BacktestReport, BacktestRow

try:
    from scipy.stats import t as scipy_t  # type: ignore[import-untyped]
    _HAS_SCIPY_T = True
except ImportError:
    _HAS_SCIPY_T = False

#: 万分比基数（ROI 整数口径，禁止浮点金额累加）。
BASIS_POINTS: int = 10_000


@dataclass(frozen=True)
class LiftStat:
    """Top-K 统计提升度与显著性（契约 §3.12、§4.6 v1.3.0）。"""

    k: int
    """评价的 Top-K 值。"""

    mean_hit: float
    """Top-K 平均命中个数。"""

    expected: float
    """超几何理论期望 = K·P0。"""

    lift: float
    """Lift@K = mean_hit / expected。"""

    sd: float
    """命中个数样本标准差（ddof=1）。"""

    t_stat: float
    """单侧 t 检验统计量。"""

    p_value: float
    """单侧 p 值。"""

    ci95: tuple[float, float]
    """均值 95% 置信区间。"""

    n: int
    """参与检验的样本期数（v1.3.0：is_signal 的样本量门槛依据）。"""

    alpha_adj: float
    """校正后显著性水平（bonferroni=True 时 = alpha_sig/len(ks)，否则 = alpha_sig）。"""

    is_signal: bool
    """v1.3.0 三条件：n ≥ min_n_signal 且 lift > lift_signal 且 p_value < alpha_adj。"""


@dataclass(frozen=True)
class TierStat:
    """高奖级触发率对比。"""

    play: str
    tier: str
    n_rounds: int
    observed_rate: float
    theory_rate: float
    ratio: float
    n_trials: int = 0
    """实测分母口径：单注口径时=总注数；无逐注数据时=期数（下游应在展示中标明）。"""


@dataclass(frozen=True)
class RuleStat:
    """单条规则独立表现（R45：由真实命中数据算出，禁止常量）。"""

    rule_id: str
    pool_size_avg: float
    hits: int
    hit_rate: float
    lift: float
    z: float


@dataclass(frozen=True)
class AttributionReport:
    """五规则正交性与归因报告。"""

    per_rule: Mapping[str, RuleStat]
    high_votes: dict[str, float]
    low_votes: dict[str, float]


@dataclass(frozen=True)
class BankrollStat:
    """资金投入与收益曲线统计（契约 R46）。"""

    n_rounds: int
    total_cost_yuan: int
    total_prize_yuan: int
    net_yuan: int
    roi: float
    max_drawdown_yuan: int
    max_losing_streak: int
    best_round_yuan: int
    roi_bp: int = 0
    """ROI 万分比整数（权威口径；报告一律由本字段格式化，禁止浮点金额累加）。"""

    roi_available: bool = True
    """ROI 是否可披露（奖级未确认 / 无投入 → False，报告必须打印 N/A）。"""


def adjusted_alpha(cfg: BacktestConfig, ks: Sequence[int]) -> float:
    """计算多重比较校正后的显著性水平（契约 R43 v1.3.0）。

    @param cfg 回测配置（alpha_sig / bonferroni）
    @param ks 参与检验的 K 集合
    @return alpha_adj = alpha_sig / len(ks)（bonferroni=True）或 alpha_sig
    """
    n_tests = len(ks) if cfg.bonferroni else 1
    return float(cfg.alpha_sig) / float(max(1, n_tests))


def t_test_one_sided(
    mean: float,
    expected: float,
    sd: float,
    n: int,
) -> tuple[float, float]:
    """单侧单样本 t 检验：H0: μ <= expected vs H1: μ > expected。

    @param mean 样本均值
    @param expected 理论期望
    @param sd 样本标准差
    @param n 样本量
    @return (t 统计量, p 值)；n<2 或 sd≈0 → (0.0, 1.0)
    """
    if n < 2 or sd < 1e-9:
        return (0.0, 1.0)
    se = sd / math.sqrt(float(n))
    t_stat = (mean - expected) / se

    if _HAS_SCIPY_T:
        p_val = float(scipy_t.sf(t_stat, df=n - 1))
    else:
        # 正态近似（scipy 缺失时的降级路径）
        p_val = float(0.5 * math.erfc(t_stat / math.sqrt(2.0)))
    return (t_stat, max(0.0, min(1.0, p_val)))


def lift_at_k(
    rows: Sequence[BacktestRow],
    ks: Sequence[int],
    cfg: BacktestConfig,
) -> dict[int, LiftStat]:
    """计算各 Top-K 的平均命中数、提升倍数与显著性（契约 R43 v1.3.0）。

    @param rows 回测结果行序列
    @param ks 评价的 K 值列表（如 3, 5, 7, 10, 12）
    @param cfg 回测配置（min_n_signal / lift_signal / alpha_sig / bonferroni）
    @return 键为 K，值为 LiftStat
    """
    res: dict[int, LiftStat] = {}
    n = len(rows)
    alpha_adj = adjusted_alpha(cfg, ks)
    if n == 0:
        for k_value in ks:
            expected = P0 * float(k_value)
            res[k_value] = LiftStat(
                k=k_value,
                mean_hit=0.0,
                expected=expected,
                lift=0.0,
                sd=0.0,
                t_stat=0.0,
                p_value=1.0,
                ci95=(0.0, 0.0),
                n=0,
                alpha_adj=alpha_adj,
                is_signal=False,
            )
        return res

    for k_value in ks:
        hits = [float(row.hit_at.get(k_value, 0)) for row in rows]
        mean_hit = float(np.mean(hits))
        expected = P0 * float(k_value)
        lift = mean_hit / expected if expected > 0.0 else 1.0
        sd = float(np.std(hits, ddof=1)) if n > 1 else 0.0

        t_stat, p_val = t_test_one_sided(mean_hit, expected, sd, n)
        se = sd / math.sqrt(float(n)) if n > 0 else 0.0
        # 95% 区间用 1.96 个标准误（正态近似；scipy 缺失时同样适用）
        ci95 = (mean_hit - 1.96 * se, mean_hit + 1.96 * se)

        # v1.3.0 三条件显著性：样本量门槛 + 效应量门槛 + Bonferroni 校正后 p 值
        is_signal = (
            (n >= cfg.min_n_signal)
            and (lift > cfg.lift_signal)
            and (p_val < alpha_adj)
        )

        res[k_value] = LiftStat(
            k=k_value,
            mean_hit=mean_hit,
            expected=expected,
            lift=lift,
            sd=sd,
            t_stat=t_stat,
            p_value=p_val,
            ci95=ci95,
            n=n,
            alpha_adj=alpha_adj,
            is_signal=is_signal,
        )
    return res


def high_tier_rate(
    rows: Sequence[BacktestRow],
    play: str,
    prize_table: Mapping[str, Mapping[int, int] | None],
) -> dict[str, TierStat]:
    """统计高奖级触发率并对比精确超几何分布理论概率（契约 R44）。

    【口径铁律】理论概率 ``P(选五中五)=C(20,5)/C(80,5)`` 是**单注**概率，因此实测
    分母必须是"总注数"而非"总期数"。若用"每期最佳票"去比，17 注/期会把触发率
    放大成十几倍，得出 23% vs 1.27% 这类毫无意义的天文比值（P10 修复）。
    仅当行内没有逐注数据（手工构造行）时，才退化为逐期 ``best_ticket_hit`` 口径，
    并在 ``n_trials`` 上如实反映分母。

    @param rows 回测结果行（优先读取 ``ticket_hits``，回退 ``best_ticket_hit``）
    @param play 玩法（当前仅 pick5 有已确认奖级）
    @param prize_table 奖金表（未确认奖级为 None）
    @return 键为 "玩法:奖级" 的 TierStat 映射
    """
    n = len(rows)
    if n == 0:
        return {}

    stats: dict[str, TierStat] = {}
    if play == "pick5":
        pmf = hypergeometric_pmf(5)
        p_5 = float(pmf[5])
        p_4_5 = float(pmf[4] + pmf[5])

        per_ticket: list[int] = []
        for row in rows:
            per_ticket.extend(int(hit) for hit in getattr(row, "ticket_hits", ()) or ())
        if not per_ticket and not any(int(row.best_ticket_hit) > 0 for row in rows):
            # 既无逐注数据也无最佳票数据（如 daily 回测未生成注单）→ 高奖级不可评，
            # 披露 N/A 而不是把 0% 当结论（0% 会被误读成"从未触发"）。
            return {}
        if per_ticket:
            trials = len(per_ticket)
            cnt_5 = sum(1 for hit in per_ticket if hit >= 5)
            cnt_4 = sum(1 for hit in per_ticket if hit >= 4)
            n_trials = trials
        else:
            trials = n
            cnt_5 = sum(1 for row in rows if row.best_ticket_hit >= 5)
            cnt_4 = sum(1 for row in rows if row.best_ticket_hit >= 4)
            n_trials = n

        obs_5 = cnt_5 / float(trials)
        obs_4 = cnt_4 / float(trials)

        stats["pick5:5"] = TierStat(
            play="pick5",
            tier="选五中五",
            n_rounds=n,
            observed_rate=obs_5,
            theory_rate=p_5,
            ratio=obs_5 / max(1e-9, p_5),
            n_trials=n_trials,
        )
        stats["pick5:4+5"] = TierStat(
            play="pick5",
            tier="选五中四及以上",
            n_rounds=n,
            observed_rate=obs_4,
            theory_rate=p_4_5,
            ratio=obs_4 / max(1e-9, p_4_5),
            n_trials=n_trials,
        )

    return stats


def _proportion_z(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """两比例差异 z 统计量（合并方差），任一组为空返回 0。

    @param hits_a 组 A 命中数
    @param n_a 组 A 样本数
    @param hits_b 组 B 命中数
    @param n_b 组 B 样本数
    @return z 值
    """
    if n_a <= 0 or n_b <= 0:
        return 0.0
    p_a = hits_a / float(n_a)
    p_b = hits_b / float(n_b)
    pooled = (hits_a + hits_b) / float(n_a + n_b)
    var = pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b)
    if var <= 1e-12:
        return 0.0
    return (p_a - p_b) / math.sqrt(var)


def _group_summary(hits: int, total: int) -> dict[str, float]:
    """把 (命中, 总数) 汇总成高/低票组披露字段。

    @param hits 命中号码数
    @param total 组内号码数
    @return {"n","hit_rate","lift","z"}
    """
    if total <= 0:
        return {"n": 0.0, "hit_rate": 0.0, "lift": 0.0, "z": 0.0}
    hit_rate = hits / float(total)
    lift = hit_rate / P0 if P0 > 0.0 else 0.0
    z = (hit_rate - P0) / math.sqrt(P0 * (1.0 - P0) / float(total))
    return {"n": float(total), "hit_rate": float(hit_rate), "lift": float(lift), "z": float(z)}


def rule_attribution(rows: Sequence[BacktestRow]) -> AttributionReport:
    """分析规则正交性与高共振票数超额命中率归因（契约 R45，P8-H3 真算）。

    数据来源：``BacktestRow.rule_stats``（规则 → (命中, 池规模)）与
    ``BacktestRow.vote_group_stats``（high/low → (命中, 组内码数)），均由
    ``evaluation.walkforward`` 在开奖揭晓后回填。缺失数据时返回 0 值而非假常量。

    @param rows 回测结果行
    @return AttributionReport
    """
    hits_acc = {rule_id: 0 for rule_id in RULE_IDS}
    pool_acc = {rule_id: 0 for rule_id in RULE_IDS}
    rows_with_rule = 0
    high_hits = high_total = 0
    low_hits = low_total = 0

    for row in rows:
        stats = getattr(row, "rule_stats", None) or {}
        if stats:
            rows_with_rule += 1
        for rule_id in RULE_IDS:
            pair = stats.get(rule_id)
            if not pair:
                continue
            hit_count, pool_size = int(pair[0]), int(pair[1])
            if pool_size > 0:
                hits_acc[rule_id] += hit_count
                pool_acc[rule_id] += pool_size

        groups = getattr(row, "vote_group_stats", None) or {}
        pair_high = groups.get("high")
        if pair_high:
            high_hits += int(pair_high[0])
            high_total += int(pair_high[1])
        pair_low = groups.get("low")
        if pair_low:
            low_hits += int(pair_low[0])
            low_total += int(pair_low[1])

    per_rule: dict[str, RuleStat] = {}
    for rule_id in RULE_IDS:
        total = pool_acc[rule_id]
        hit_rate = hits_acc[rule_id] / float(total) if total > 0 else P0
        lift = hit_rate / P0 if P0 > 0.0 else 0.0
        z = (hit_rate - P0) / math.sqrt(P0 * (1.0 - P0) / float(total)) if total > 0 else 0.0
        per_rule[rule_id] = RuleStat(
            rule_id=rule_id,
            pool_size_avg=(total / float(rows_with_rule)) if rows_with_rule > 0 else 0.0,
            hits=hits_acc[rule_id],
            hit_rate=float(hit_rate),
            lift=float(lift),
            z=float(z),
        )

    high_votes = _group_summary(high_hits, high_total)
    low_votes = _group_summary(low_hits, low_total)
    # 两组差异检验（V≥3 vs V≤1）；组缺失时 z 记 0，报告按"样本不足"处理
    high_votes["z"] = _proportion_z(high_hits, high_total, low_hits, low_total)

    return AttributionReport(
        per_rule=per_rule,
        high_votes=high_votes,
        low_votes=low_votes,
    )


def _ratio_bp(numerator: int, denominator: int) -> int:
    """整数万分比（四舍五入到最近），避免浮点金额。

    @param numerator 分子（整数元）
    @param denominator 分母（整数元，须 > 0）
    @return 万分比整数
    """
    if denominator <= 0:
        return 0
    scaled = numerator * BASIS_POINTS
    quotient, remainder = divmod(abs(scaled), denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return quotient if scaled >= 0 else -quotient


def format_roi_bp(roi_bp: int) -> str:
    """把万分比整数格式化为百分比字符串（纯整数运算，零浮点）。

    @param roi_bp 万分比整数（如 15000 → "150.00%"）
    @return 形如 "-58.33%" 的字符串
    """
    sign = "-" if roi_bp < 0 else ""
    value = abs(int(roi_bp))
    return f"{sign}{value // 100}.{value % 100:02d}%"


def roi_text(stat: "BankrollStat") -> str:
    """ROI 展示口径的唯一出口：能算则给万分比，不能算则**分原因**披露（P10b-L15）。

    - ``roi_available=True`` → 万分比整数格式化（如 ``-70.12%``）；
    - 本期未生成注单（``total_cost_yuan == 0``）→ ``ROI: N/A（本次未生成注单）``；
    - 有注单但奖级未确认 → ``ROI: N/A（官方奖级未确认）``。

    @param stat 资金曲线统计
    @return 展示字符串
    """
    if stat.roi_available:
        return format_roi_bp(stat.roi_bp)
    if int(stat.total_cost_yuan) <= 0:
        return ROI_NA_NO_TICKET_TEXT
    return ROI_NA_TEXT


def bankroll(
    rows: Sequence[BacktestRow],
    prize_table: Mapping[str, Mapping[int, int] | None] | None = None,
    *,
    cost_per_ticket: int = 2,
) -> BankrollStat:
    """计算回测周期的资金投入、奖金收入、净利润与最大回撤（契约 R46）。

    金额严格使用 int 整数元；ROI 的权威口径是万分比整数 ``roi_bp``，
    ``roi`` 字段仅为契约兼容的展示冗余（= net/cost，未参与任何累加）。

    @param rows 回测结果行
    @param prize_table 奖金表（未确认奖级为 None）
    @param cost_per_ticket 单注成本（整数元）
    @return BankrollStat
    """
    n = len(rows)
    if n == 0:
        return BankrollStat(0, 0, 0, 0, 0.0, 0, 0, 0, roi_bp=0, roi_available=False)

    total_cost = sum(int(row.cost_yuan) for row in rows)
    total_prize = sum(int(row.prize_yuan) for row in rows)
    net_yuan = total_prize - total_cost

    roi = float(net_yuan) / float(total_cost) if total_cost > 0 else 0.0
    roi_bp = _ratio_bp(net_yuan, total_cost)
    # 奖级未确认（rows 全部 prize_known=False）时不披露 ROI，避免把假设当结论
    roi_available = total_cost > 0 and all(bool(getattr(row, "prize_known", False)) for row in rows)

    # 逐期累计资金净值曲线计算最大回撤
    cum_net = 0
    peak = 0
    max_dd = 0
    losing_streak = 0
    max_streak = 0
    best_round = 0

    for row in rows:
        round_net = int(row.prize_yuan) - int(row.cost_yuan)
        if int(row.prize_yuan) > best_round:
            best_round = int(row.prize_yuan)

        cum_net += round_net
        if cum_net > peak:
            peak = cum_net
        dd = peak - cum_net
        if dd > max_dd:
            max_dd = dd

        if round_net <= 0:
            losing_streak += 1
            if losing_streak > max_streak:
                max_streak = losing_streak
        else:
            losing_streak = 0

    return BankrollStat(
        n_rounds=n,
        total_cost_yuan=total_cost,
        total_prize_yuan=total_prize,
        net_yuan=net_yuan,
        roi=roi,
        max_drawdown_yuan=max_dd,
        max_losing_streak=max_streak,
        best_round_yuan=best_round,
        roi_bp=roi_bp,
        roi_available=roi_available,
    )


def self_check(
    report: BacktestReport,
    cfg: BacktestConfig | None = None,
) -> dict[str, bool]:
    """报告口径自洽性自检（契约 §4.6；P8-H4：no_lookahead 必须真判定）。

    机检项：
    1. ``sample_count_matches``：n_sample == len(rows)；
    2. ``lift_recalculated``：用 rows 的 hit_at **独立重算** mean_hit/expected/n 并与
       报告值比对（1e-9 容差）；给了 cfg 时还复核 is_signal 三条件；
    3. ``bankroll_recalculated``：用 rows 独立重算 cost/prize/net/n_rounds；
    4. ``no_lookahead``：``lookahead_checked == n_sample and not lookahead_violations``
       —— 不是常量 True，而是 walkforward 逐期校验后的真实计数。

    @param report 回测报告
    @param cfg 回测配置（可选；提供时复核显著性三条件）
    @return 四项布尔机检结果
    """
    rows = report.rows
    n_rows = len(rows)

    lift_ok = True
    if rows and report.lift:
        for k_value, stat in report.lift.items():
            mean_hit = float(np.mean([float(row.hit_at.get(k_value, 0)) for row in rows]))
            if abs(mean_hit - stat.mean_hit) > 1e-9:
                lift_ok = False
                break
            if abs(P0 * float(k_value) - stat.expected) > 1e-12:
                lift_ok = False
                break
            if stat.n != n_rows:
                lift_ok = False
                break
            if cfg is not None:
                alpha_adj = adjusted_alpha(cfg, tuple(report.lift.keys()))
                if abs(alpha_adj - stat.alpha_adj) > 1e-12:
                    lift_ok = False
                    break
                expected_signal = (
                    (stat.n >= cfg.min_n_signal)
                    and (stat.lift > cfg.lift_signal)
                    and (stat.p_value < alpha_adj)
                )
                if expected_signal != stat.is_signal:
                    lift_ok = False
                    break
    elif report.lift and not rows:
        lift_ok = False

    bank_ok = True
    broll = report.bankroll
    if broll is not None:
        cost = sum(int(row.cost_yuan) for row in rows)
        prize = sum(int(row.prize_yuan) for row in rows)
        bank_ok = (
            broll.n_rounds == n_rows
            and broll.total_cost_yuan == cost
            and broll.total_prize_yuan == prize
            and broll.net_yuan == prize - cost
            and broll.max_drawdown_yuan >= 0
            and broll.max_losing_streak >= 0
        )

    no_lookahead = (report.lookahead_checked == n_rows) and (not report.lookahead_violations)

    return {
        "sample_count_matches": report.n_sample == n_rows,
        "lift_recalculated": lift_ok,
        "bankroll_recalculated": bank_ok,
        "no_lookahead": no_lookahead,
    }