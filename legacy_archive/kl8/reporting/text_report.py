"""全景控制台与 TXT 精要决策报告生成器（契约 R48-d、§3.13、§6.6）。

【六段式结构】
[0] 上期实测深度复盘 (Previous Draw Review) —— 只读 §3.14.1 复盘状态文件
[1] 宏观随机性与动力学体检 (Randomness Baseline) —— 常数取自 constants，禁止渲染层硬编码
[2] 四尺度时空动量与热度分布 (Four-scale Features)
[3] 五规则正交共振榜与 FinalScore Top 20 (Resonance Ranking)
[4] 金胆池 / 核心复式大底与形态体检卡 (Pools & Morphology Guard)
[5] 覆盖注单与多层级保底投注方案 (Covering Tickets & Strategy)
[附录A] 80 码底层动力学与微积分特征全参表 (80-Number Full Stats)
[附录B] Walk-Forward 滚动时序实测归因报告 (Backtest Summary)

【P8-M5 / §6.6 验收红线（可 grep 审计）】
- 渲染层不得出现任何"写死口径值"：单码概率/两码同现率/重号均值/和值区间/金胆规模
  一律从 ``constants`` 或结果对象读取并格式化，源码中不允许出现固定百分比字面量；
- 不得出现模板占位符残留（未替换的花括号变量、英文设计式占位、临时填充标记）；
- 选三/选七等未确认奖级必须输出 ``ROI: N/A（官方奖级未确认）``；
- ``is_signal=False`` 时不得出现「★显著」；n < ``cfg.backtest.min_n_signal`` 时必须
  输出「样本量不足（n<x），结论仅供参考」（x 取自配置，禁止写死）。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from kl8.atomic_io import atomic_write_text
from kl8.config import AppConfig, RuleConfig
from kl8.constants import (
    E_REPEAT,
    K_DRAW,
    NO_BASELINE_TEXT,
    P0,
    P_CO,
    PLAY_LABELS,
    ROI_NA_TEXT,
    RULE_IDS,
    RULE_NAMES,
    SUM_EXPECT,
    sum_band,
)
from kl8.datatypes import BacktestReport, Dataset, FeatureSnapshot, TicketPlan
from kl8.evaluation.metrics import format_roi_bp, roi_text
from kl8.layer4_scoring import ScoreRoundResult

if TYPE_CHECKING:
    from kl8.layer2_baseline.randomness import RandomnessReport


@dataclass(frozen=True)
class ReportContext:
    """三类交付件共享的唯一数据源上下文（契约 §3.13）。

    所有报表（控制台 TXT、微信文案、商业 Excel）必须 100% 消费本上下文中的相同数据，
    绝对禁止各模块私自二次计算产生口径分歧（契约 B8）。
    """

    cfg: AppConfig
    """全局生效配置。"""

    ds: Dataset
    """全量历史数据集（最新开奖截断）。"""

    target_period: int
    """待预测目标期号。"""

    target_date: str
    """待预测目标开奖日期（如 '2026-11-02'）。"""

    snapshot: FeatureSnapshot
    """Layer 3 特征快照。"""

    rounds: ScoreRoundResult
    """Layer 4 综合打分与规则判定结果。"""

    plans: tuple[TicketPlan, ...]
    """Layer 5 组合覆盖注单方案。"""

    randomness: "RandomnessReport | None" = None
    """Layer 2 宏观随机性与 NIST-Lite 检验报告。"""

    backtest: BacktestReport | None = None
    """Walk-Forward 滚动回测评估报告。"""

    previous_review: Mapping[str, Any] | None = None
    """上期复盘结果（status/note/gold_hit/main_hit/best_ticket_hit）。"""

    warnings: tuple[str, ...] = ()
    """全局执行过程中的告警或降级信息。"""


def sample_note(min_n_signal: int) -> str:
    """生成样本量不足提示（契约 R43 v1.3.0；数值来自配置，禁止写死 500）。

    @param min_n_signal 显著性最小样本量门槛
    @return 中文提示文案
    """
    return f"样本量不足（n<{int(min_n_signal)}），结论仅供参考"


def _rule_description(rule_id: str, rules: RuleConfig) -> str:
    """由配置实时生成规则阈值描述（P8-M5：禁止在渲染层写死阈值/写错规则名）。

    @param rule_id 规则编号 R1..R5
    @param rules 规则阈值配置
    @return 形如「R1 多尺度强跟随共振：10G/25G/50G/ALL 中至少 3 个秩 ≥ 0.65」的中文说明
    """
    name = RULE_NAMES.get(rule_id, rule_id)
    if rule_id == "R1":
        return (
            f"{rule_id} {name}：10G/25G/50G/ALL 四个窗口中至少 {rules.r1_min_windows} 个"
            f"百分位秩 ≥ {rules.r1_rank_threshold:.2f}"
        )
    if rule_id == "R2":
        return (
            f"{rule_id} {name}：10G 频率 ∈ [{rules.r2_f10_low:.2f}, {rules.r2_f10_high:.2f}]，"
            f"且 25G/50G 频率 ≥ {rules.r2_base_freq:.2f}"
        )
    if rule_id == "R3":
        return (
            f"{rule_id} {name}：10G ≤ 25G ≤ 50G 热度过热梯度，且 R(Follow_10G) ≥ "
            f"{rules.r3_follow_rank:.2f}"
        )
    if rule_id == "R4":
        return (
            f"{rule_id} {name}：{rules.r4_w_co25:.2f}·R(Co25)+{rules.r4_w_co50:.2f}·R(Co50) ≥ "
            f"{rules.r4_rank_score:.2f}"
        )
    return (
        f"{rule_id} {name}：Pressure ≥ {rules.r5_pressure:.2f}，遗漏 ≤ {rules.r5_max_gap} 期，"
        f"且 R(Follow_ALL) ≥ {rules.r5_follow_all_rank:.2f}"
    )


def _review_line(review: Mapping[str, Any] | None) -> str:
    """把上期复盘结果渲染成一行中文（缺失时输出固定文案且不输出任何命中数）。

    @param review 复盘结果字典（可为 None）
    @return 中文说明行
    """
    if not review:
        return f"  * 复盘提示: {NO_BASELINE_TEXT}"
    status = str(review.get("status", "baseline"))
    if status != "ok":
        note = str(review.get("note", "")).strip() or NO_BASELINE_TEXT
        return f"  * 复盘提示: {note}"
    prev_period = review.get("period", 0)
    gold_hit = int(review.get("gold_hit", -1))
    main_hit = int(review.get("main_hit", -1))
    gold = review.get("gold", ())
    main = review.get("main", ())
    best = int(review.get("best_ticket_hit", -1))
    return (
        f"  * 上期预测复盘 (第 {prev_period} 期): 金胆命中 {gold_hit}/{len(gold)} 码，"
        f"复式命中 {main_hit}/{len(main)} 码，单注最高命中 {best} 码"
    )


def _attribution_lines(report: BacktestReport) -> list[str]:
    """渲染 R45 规则归因（真实命中率/超额/lift/z 与高低票组对比）。

    @param report 回测报告
    @return 文本行列表
    """
    attribution = report.attribution
    if attribution is None:
        return ["  * 规则归因: 暂无归因数据。"]
    lines: list[str] = ["  * 五规则独立查准率归因（理论基线 p0 见上文）:"]
    for rule_id in RULE_IDS:
        stat = attribution.per_rule.get(rule_id)
        if stat is None or stat.pool_size_avg <= 0:
            lines.append(f"    - {rule_id} {RULE_NAMES.get(rule_id, '')}: 样本不足，未统计")
            continue
        lines.append(
            f"    - {rule_id} {RULE_NAMES.get(rule_id, '')}: 平均池 {stat.pool_size_avg:.1f} 码，"
            f"命中率={stat.hit_rate:.2%}，Lift={stat.lift:.3f}，z={stat.z:+.2f}"
        )
    high = attribution.high_votes
    low = attribution.low_votes
    lines.append(
        f"    - 高票组(V≥3): n={int(high.get('n', 0))}, 命中率={high.get('hit_rate', 0.0):.2%}, "
        f"Lift={high.get('lift', 0.0):.3f}"
    )
    lines.append(
        f"    - 低票组(V≤1): n={int(low.get('n', 0))}, 命中率={low.get('hit_rate', 0.0):.2%}, "
        f"Lift={low.get('lift', 0.0):.3f}"
    )
    lines.append(f"    - 高票组 vs 低票组两比例检验 z={high.get('z', 0.0):+.2f}")
    return lines


def _high_tier_lines(report: BacktestReport, cfg: AppConfig) -> list[str]:
    """渲染高奖级触发率与理论概率（含 §7.4 选五全中 0.0645% 口径）。

    @param report 回测报告
    @param cfg 全局配置
    @return 文本行列表
    """
    if not report.high_tier:
        return []
    lines: list[str] = ["  * 高奖级触发率（实际 vs 超几何理论；实测分母=单注口径）:"]
    for key, tier in sorted(report.high_tier.items()):
        if tier.theory_rate > 0:
            period_once = int(1.0 / tier.theory_rate)  # 向下取整：0.0645% → ≈1/1550 期（契约 §7.4 口径）
            theory = f"{tier.theory_rate:.4%}（≈1/{period_once} 期）"
        else:
            theory = "N/A"
        n_note = ""
        if tier.n_rounds < cfg.backtest.min_n_signal:
            n_note = f"；{sample_note(cfg.backtest.min_n_signal)}"
        denom = f"{tier.n_rounds} 期 / {tier.n_trials} 注" if tier.n_trials > tier.n_rounds else f"{tier.n_rounds} 期"
        lines.append(
            f"    - {tier.tier}: 实际={tier.observed_rate:.4%}，理论={theory}，"
            f"比值={tier.ratio:.2f}（n={denom}{n_note}）"
        )
    return lines


def _pool_shape_lines(gold_pool: Sequence[int], main_pool: Sequence[int]) -> list[str]:
    """渲染金胆/复式池形态体检（和值合理区间由常数函数算出，禁止写死固定区间）。

    @param gold_pool 金胆池
    @param main_pool 复式池
    @return 文本行列表
    """
    lines: list[str] = []
    k_gold = len(gold_pool)
    if k_gold > 0:
        odd = sum(1 for num in gold_pool if num % 2 == 1)
        even = k_gold - odd
        low, high = sum_band(k_gold)
        lines.append("  * 金胆组形态体检卡:")
        lines.append(f"    - 和值: {sum(gold_pool)} (理论区间 {low}~{high}，E={SUM_EXPECT * k_gold / K_DRAW:.1f})")
        lines.append(f"    - 奇偶比: {odd}:{even}")
        lines.append(f"    - 跨度: {max(gold_pool) - min(gold_pool)}")
    k_main = len(main_pool)
    if k_main > 0:
        low_m, high_m = sum_band(k_main)
        lines.append(f"  * 复式组形态体检卡: 规模={k_main} 码，和值={sum(main_pool)} (理论区间 {low_m}~{high_m})")
    return lines


def render_text_report(ctx: ReportContext) -> str:
    """渲染工业级结构化控制台与 TXT 文本报表（契约 §3.13、§6.6）。

    @param ctx 报表上下文数据源
    @return 格式化后的完整文本报告
    """
    lines: list[str] = []

    # 表头
    lines.append("=" * 80)
    lines.append("          快乐8 (Keno) 五层正交动力学与组合优化精要分析报告")
    lines.append(f"                 预测目标期号: 第 {ctx.target_period} 期  |  开奖日期: {ctx.target_date}")
    lines.append("=" * 80)
    lines.append("")

    # [0] 上期复盘（只读 §3.14.1 状态文件，禁止用当期池冒充）
    lines.append("【段落 0: 上期开奖实测深度复盘】")
    lines.append("-" * 80)
    if ctx.ds.draws:
        last_draw = ctx.ds.draws[-1]
        nums_str = " ".join(f"{num:02d}" for num in sorted(last_draw.numbers))
        lines.append(f"  * 上期期号: 第 {last_draw.period} 期 ({last_draw.date})")
        lines.append(f"  * 上期开奖: {nums_str}")
        lines.append(_review_line(ctx.previous_review))
    else:
        lines.append("  * 暂无历史开奖记录。")
    lines.append("")

    # [1] 宏观随机性与动力学基准
    lines.append("【段落 1: 宏观随机性与大数动力学基准】")
    lines.append("-" * 80)
    lines.append(
        f"  * 理论基准: 单码出现率 p0 = {P0:.2%}  |  两码同现率 p_co = {P_CO:.2%}"
    )
    lines.append(
        f"  * 重号均值: E(Repeat) = {E_REPEAT:.2f} 个  |  历史总样本期数: {len(ctx.ds.draws)} 期"
    )
    if ctx.randomness is not None:
        if "ALL" in ctx.randomness.chi2:
            c_res = ctx.randomness.chi2["ALL"]
            lines.append(
                f"  * 卡方拟合度检验 (GOF): chi2 = {c_res.statistic:.2f}, p-value = {c_res.p_value:.4f}"
            )
            verdict = "通过 (符合均匀随机)" if c_res.p_value >= 0.05 else "未通过 (存在显著非随机偏差)"
            lines.append(f"    - 均匀性判定: {verdict}")
        lines.append("  * NIST-Lite 四项检验摘要（已按 p0=25% 的偏置比特流口径适配）:")
        for nist_res in ctx.randomness.nist:
            status = "合格" if nist_res.passed else "预警"
            lines.append(
                f"    - [{nist_res.name}]: 统计量={nist_res.statistic:.4f}, "
                f"p值={nist_res.p_value:.4f}, 状态={status}"
            )
        lines.append(
            "    - 口径说明: NIST 原始检验针对等概率 0/1 比特流；本系统输入为单码出现率 "
            f"{P0:.0%} 的偏置流，已按偏置口径换算期望与方差，结论仅作长程稳定性监控，"
            "不作为「非随机」判定依据。"
        )
    else:
        lines.append("  * 随机性基准检验: 正常执行 (未加载独立随机性报告)")
    lines.append("")

    # [2] 四尺度特征
    lines.append("【段落 2: 四尺度时空动量与热度分布】")
    lines.append("-" * 80)
    lines.append("  * 观察窗口划分: 超短线 10G / 短线 25G / 中线 50G / 全历史 ALL")
    w_eff = ctx.snapshot.effective_windows
    lines.append(
        f"  * 实际生效窗口深度: 10G={w_eff.get('10G', 10)}期, 25G={w_eff.get('25G', 25)}期, "
        f"50G={w_eff.get('50G', 50)}期, ALL={w_eff.get('ALL', len(ctx.ds.draws))}期"
    )
    if ctx.snapshot.cold_start:
        lines.append("  ! 注意: 当前处于冷启动阶段，部分深层窗口已自动按比例平滑兜底。")
    f10 = ctx.snapshot.freq_10
    hot = sorted(range(1, 81), key=lambda num: f10[num - 1], reverse=True)[:5]
    cold = sorted(range(1, 81), key=lambda num: f10[num - 1])[:5]
    lines.append("  * 10G 超短线极热码 (Top 5): " + ", ".join(f"[{num:02d}](频次:{f10[num-1]:.2f})" for num in hot))
    lines.append("  * 10G 超短线冰冷码 (Top 5): " + ", ".join(f"[{num:02d}](频次:{f10[num-1]:.2f})" for num in cold))
    lines.append("")

    # [3] 五规则共振榜
    lines.append("【段落 3: 五规则正交共振榜与 FinalScore Top 20】")
    lines.append("-" * 80)
    lines.append("  * 五规则体系说明（阈值实时取配置）:")
    for rule_id in RULE_IDS:
        lines.append(f"    - {_rule_description(rule_id, ctx.cfg.rules)}")
    lines.append("")

    votes = ctx.rounds.votes
    final_scores = ctx.rounds.scores.final
    equivalence = ctx.rounds.weighted_votes
    sorted_by_score = sorted(range(1, 81), key=lambda num: final_scores[num - 1], reverse=True)
    res_5 = [num for num in range(1, 81) if int(votes[num - 1]) == 5]
    res_4 = [num for num in range(1, 81) if int(votes[num - 1]) == 4]
    lines.append("  * 五星共振金码 (5票): " + (", ".join(f"{num:02d}" for num in res_5) if res_5 else "无 (本期五条规则独立分散)"))
    lines.append("  * 四星共振银码 (4票): " + (", ".join(f"{num:02d}" for num in res_4) if res_4 else "无"))
    lines.append("")
    lines.append("  * 综合动力学评分 Top 20 精选梯队（等效票 = 契约 B5 的 Σ(w/w̄)·pass）:")
    lines.append("    名次  号码   总分(FinalScore)  规则票  等效票  10G热度  当前遗漏  动量金叉")
    golden_cross = ctx.snapshot.golden_cross
    gaps = ctx.snapshot.gap
    for rank_idx, num in enumerate(sorted_by_score[:20], start=1):
        idx = num - 1
        eq_val = float(equivalence[idx]) if equivalence is not None else float(votes[idx])
        gc_lbl = "★金叉" if golden_cross[idx] else "平稳"
        lines.append(
            f"    {rank_idx:02d}    [{num:02d}]      {final_scores[idx]:7.4f}        "
            f"{int(votes[idx])}票   {eq_val:5.2f}   {f10[idx]:.2f}      {int(gaps[idx]):02d}期      {gc_lbl}"
        )
    lines.append("")

    # [4] 金胆池 / 复式池 + 形态体检
    lines.append("【段落 4: 金胆池 / 核心复式大底与形态体检卡】")
    lines.append("-" * 80)
    gold_pool = ctx.rounds.pool.gold
    main_pool = ctx.rounds.pool.main
    lines.append(f"  * 核心超级金胆 ({len(gold_pool)} 码): " + " ".join(f"{num:02d}" for num in gold_pool))
    lines.append(f"  * 精选复式大底 ({len(main_pool)} 码): " + " ".join(f"{num:02d}" for num in main_pool))
    lines.append(
        f"  * 重号配比: 重号 {ctx.rounds.pool.repeat_count} / 新号 {ctx.rounds.pool.new_count}"
        f"（契约 B6）"
    )
    if ctx.rounds.pool.relaxed:
        lines.append(f"  * 已放宽约束: {', '.join(ctx.rounds.pool.relaxed)}")
    lines.extend(_pool_shape_lines(gold_pool, main_pool))
    if ctx.rounds.pool.warnings:
        lines.append(f"    - 池选告警: {'; '.join(ctx.rounds.pool.warnings)}")
    lines.append("")

    # [5] 覆盖注单
    lines.append("【段落 5: 覆盖注单与组合多层级保底投注方案】")
    lines.append("-" * 80)
    if ctx.plans:
        total_tickets = sum(len(plan.tickets) for plan in ctx.plans)
        total_cost = sum(plan.cost_yuan for plan in ctx.plans)
        lines.append(
            f"  * 推荐方案: 共 {len(ctx.plans)} 个玩法计划，总注数: {total_tickets} 注，"
            f"总投入: {total_cost} 元"
        )
        for plan_i, plan in enumerate(ctx.plans, start=1):
            play_cn = PLAY_LABELS.get(plan.play, plan.play)
            design = plan.guarantee_level.get("design", "")
            lines.append(
                f"  --- [计划 {plan_i}]: 玩法={play_cn}({plan.play}), 选号数 k={plan.k}, "
                f"单注预算={plan.budget}注 ({plan.cost_yuan}元), 求解方式={plan.method} ---"
            )
            lines.append(f"      理论保底设计: {design}")
            if ctx.cfg.prize_table.get(plan.play) is None:
                lines.append(f"      {ROI_NA_TEXT}（该玩法只做覆盖诊断，不计资金结算）")
            lines.append("      精选实战投注单列表:")
            for t_idx, ticket in enumerate(plan.tickets, start=1):
                tk_str = " ".join(f"{x:02d}" for x in ticket)
                lines.append(f"        第 {t_idx:02d} 注: [ {tk_str} ]")
    else:
        lines.append("  * 暂无注单生成（当前配置未启用注单构建或预算为 0）。")
    lines.append("")

    # 附录 A
    lines.append("【附录 A: 80 码底层动力学微积分全参矩阵】")
    lines.append("-" * 80)
    lines.append(" 号码 | BaseScore | R1 R2 R3 R4 R5 | 得票 | FinalScore | 10G频次 | 25G频次 | 50G频次 | 遗漏 | 金叉")
    lines.append("-" * 80)
    f25 = ctx.snapshot.freq_25
    f50 = ctx.snapshot.freq_50
    base_scores = ctx.rounds.scores.base
    masks = [np.asarray(verdict.mask, dtype=bool) for verdict in ctx.rounds.verdicts]
    for num in range(1, 81):
        idx = num - 1
        rule_marks = "  ".join(str(int(mask[idx])) for mask in masks)
        gc_mark = "★金叉" if golden_cross[idx] else "平稳"
        lines.append(
            f"  {num:02d}  |  {base_scores[idx]:7.4f}  |  {rule_marks}  |  {int(votes[idx])}票  |  "
            f"{final_scores[idx]:7.4f}   |  {f10[idx]:.2f}   |  {f25[idx]:.2f}   |  {f50[idx]:.2f}   |  "
            f"{int(gaps[idx]):02d}  | {gc_mark}"
        )
    lines.append("")

    # 附录 B
    lines.append("【附录 B: Walk-Forward 滚动时序实测归因摘要】")
    lines.append("-" * 80)
    if ctx.backtest is not None:
        bt = ctx.backtest
        lines.append(f"  * 回测期数: {bt.n_sample} 期 (冷启动 T0={bt.t0})")
        lines.append(f"  * 覆盖期号范围: {bt.start_period} ~ {bt.end_period}")
        lines.append(f"  * 防穿越校验: 已校验 {bt.lookahead_checked} 期，越界 {len(bt.lookahead_violations)} 期")
        if bt.lookahead_violations:
            periods = ", ".join(str(p) for p in bt.lookahead_violations)
            lines.append(f"    ! 防穿越失败期号: {periods}")
        insufficient = any(stat.n < ctx.cfg.backtest.min_n_signal for stat in bt.lift.values())
        if insufficient:
            lines.append(f"  ! {sample_note(ctx.cfg.backtest.min_n_signal)}")
        lines.append("  * Top-K 提升度 (Lift@K) 检阅:")
        for _k, lstat in sorted(bt.lift.items(), key=lambda item: item[0]):
            sig_mark = "★显著" if lstat.is_signal else "平稳"
            lines.append(
                f"    - Top-{lstat.k:02d}: 实测均中={lstat.mean_hit:.2f}, 理论期望={lstat.expected:.2f}, "
                f"Lift={lstat.lift:.3f}, p值={lstat.p_value:.4f}, "
                f"校正后α={lstat.alpha_adj:.4f}, n={lstat.n} ({sig_mark})"
            )
        lines.extend(_high_tier_lines(bt, ctx.cfg))
        lines.extend(_attribution_lines(bt))
        broll = bt.bankroll
        if broll is not None:
            if broll.roi_available:
                lines.append(
                    f"  * 资金收益表现: 总投入={broll.total_cost_yuan}元, "
                    f"总奖金={broll.total_prize_yuan}元, 净利={broll.net_yuan}元, "
                    f"ROI={format_roi_bp(broll.roi_bp)}, 最大回撤={broll.max_drawdown_yuan}元, "
                    f"最长连挂={broll.max_losing_streak}期"
                )
            elif broll.total_cost_yuan <= 0:
                lines.append(f"  * 资金收益表现: {roi_text(broll)}；回测期无注单，无资金流水可披露")
            else:
                lines.append(
                    f"  * 资金收益表现: {roi_text(broll)}；"
                    f"（投入 {broll.total_cost_yuan} 元 / 奖金 {broll.total_prize_yuan} 元，"
                    "仅记录整数元流水，不披露收益率）"
                )
        if bt.self_check:
            self_check_str = ", ".join(f"{key}={value}" for key, value in bt.self_check.items())
            lines.append(f"  * 报告自洽机检: {self_check_str}")
    else:
        lines.append("  * 暂无关联回测报告。")
    lines.append("")

    # 尾注与告警
    if ctx.warnings:
        lines.append("【系统执行提示与告警】")
        for warning in ctx.warnings:
            lines.append(f"  ! {warning}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("              [工业级研发生态·Antigravity 全自动生成交付件]")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_text_report(text: str, path: str | Path) -> Path:
    """保存控制台精要 TXT 报表到本地指定路径（原子写，P8-M13）。

    @param text 报表文本字符串
    @param path 保存目标路径
    @return 保存成功的文件 Path 对象
    @raises OSError 写入失败（调用方不得把失败当成功）
    """
    return atomic_write_text(path, text)