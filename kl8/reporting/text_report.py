"""全景控制台与 TXT 精要决策报告生成器（契约 R48-d、§3.13）。

【六段式结构】
[0] 上期实测深度复盘 (Previous Draw Review)
[1] 宏观随机性与动力学体检 (Randomness Baseline)
[2] 四尺度时空动量与热度分布 (Four-scale Features)
[3] 五规则正交共振榜与 FinalScore Top 20 (Resonance Ranking)
[4] 金胆池 / 核心复式大底与形态体检卡 (Pools & Morphology Guard)
[5] 覆盖注单与多层级保底投注方案 (Covering Tickets & Strategy)
[附录A] 80 码底层动力学与微积分特征全参表 (80-Number Full Stats)
[附录B] T0=300 Walk-Forward 滚动时序实测归因报告 (Backtest Summary)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from kl8.config import AppConfig
from kl8.datatypes import BacktestReport, Dataset, FeatureSnapshot, TicketPlan
from kl8.layer4_scoring import ScoreRoundResult

if TYPE_CHECKING:
    from kl8.layer2_baseline.randomness import RandomnessReport


@dataclass(frozen=True)
class ReportContext:
    """三类交付件共享的唯一数据源上下文（契约 §3.13）。

    所有报表（控制台 TXT、微信文案、商业 Excel）必须 100% 消费本上下文中的相同数据，
    绝对禁止各模块私自二次计算产生口径分歧。
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
    """上期开奖命中与得失复盘数据字典。"""

    warnings: tuple[str, ...] = ()
    """全局执行过程中的告警或降级信息。"""


def render_text_report(ctx: ReportContext) -> str:
    """渲染工业级结构化控制台与 TXT 文本报表（契约 §3.13）。

    @param ctx 报表上下文数据源
    @return 格式化后的完整文本报告
    """
    lines: list[str] = []

    # 步骤 1: 报告表头与元信息
    lines.append("=" * 80)
    lines.append(f"          快乐8 (Keno) 五层正交动力学与组合优化精要分析报告")
    lines.append(f"                 预测目标期号: 第 {ctx.target_period} 期  |  开奖日期: {ctx.target_date}")
    lines.append("=" * 80)
    lines.append("")

    # 步骤 2: [0] 上期实测深度复盘
    lines.append("【段落 0: 上期开奖实测深度复盘】")
    lines.append("-" * 80)
    if ctx.ds.draws:
        last_draw = ctx.ds.draws[-1]
        sorted_last_nums = sorted(list(last_draw.numbers))
        nums_str = " ".join(f"{n:02d}" for n in sorted_last_nums)
        lines.append(f"  * 上期期号: 第 {last_draw.period} 期 ({last_draw.date})")
        lines.append(f"  * 上期开奖: {nums_str}")

        if ctx.previous_review:
            hits = ctx.previous_review.get("hit_count", 0)
            target_p = ctx.previous_review.get("target_period", last_draw.period)
            lines.append(f"  * 前期预测复盘 (第 {target_p} 期): 命中 {hits} 码")
            if "details" in ctx.previous_review:
                lines.append(f"    - 复盘摘要: {ctx.previous_review['details']}")
        else:
            lines.append("  * 复盘提示: 上期开奖基准已加载，无独立上一期持久化预测记录对比。")
    else:
        lines.append("  * 暂无历史开奖记录。")
    lines.append("")

    # 步骤 3: [1] 宏观随机性与动力学体检
    lines.append("【段落 1: 宏观随机性与大数动力学基准】")
    lines.append("-" * 80)
    lines.append(f"  * 理论基准: 单码出现率 p0 = 20/80 = 25.00%  |  两码同现率 p_co = 6.01%")
    lines.append(f"  * 重号均值: E(Repeat) = 5.00 个  |  历史总样本期数: {len(ctx.ds.draws)} 期")
    if ctx.randomness is not None:
        if "ALL" in ctx.randomness.chi2:
            c_res = ctx.randomness.chi2["ALL"]
            lines.append(f"  * 卡方拟合度检验 (GOF): chi2 = {c_res.statistic:.2f}, p-value = {c_res.p_value:.4f}")
            pass_str = "通过 (符合均匀随机)" if c_res.p_value >= 0.05 else "未通过 (存在显著非随机偏差)"
            lines.append(f"    - 均匀性判定: {pass_str}")
        lines.append(f"  * NIST-Lite 四项检验摘要:")
        for nist_res in ctx.randomness.nist:
            lines.append(f"    - [{nist_res.name}]: 统计量={nist_res.statistic:.4f}, p值={nist_res.p_value:.4f}, 状态={'合格' if nist_res.passed else '预警'}")
    else:
        lines.append("  * 随机性基准检验: 正常执行 (未加载独立随机性报告)")
    lines.append("")

    # 步骤 4: [2] 四尺度特征与动量
    lines.append("【段落 2: 四尺度时空动量与热度分布】")
    lines.append("-" * 80)
    lines.append("  * 观察窗口划分: 超短线 10G / 短线 25G / 中线 50G / 全历史 ALL")
    w_eff = ctx.snapshot.effective_windows
    lines.append(f"  * 实际生效窗口深度: 10G={w_eff.get('10G', 10)}期, 25G={w_eff.get('25G', 25)}期, 50G={w_eff.get('50G', 50)}期, ALL={w_eff.get('ALL', len(ctx.ds.draws))}期")
    if ctx.snapshot.cold_start:
        lines.append("  ! 注意: 当前处于冷启动阶段，部分深层窗口已自动按比例平滑兜底。")

    # 提取热度最高与最低的号码
    f10 = ctx.snapshot.freq_10
    top_hot_10 = sorted(range(1, 81), key=lambda x: f10[x - 1], reverse=True)[:5]
    top_cold_10 = sorted(range(1, 81), key=lambda x: f10[x - 1])[:5]
    lines.append(f"  * 10G 超短线极热码 (Top 5): " + ", ".join(f"[{n:02d}](频次:{f10[n-1]:.2f})" for n in top_hot_10))
    lines.append(f"  * 10G 超短线冰冷码 (Top 5): " + ", ".join(f"[{n:02d}](频次:{f10[n-1]:.2f})" for n in top_cold_10))
    lines.append("")

    # 步骤 5: [3] 五规则共振榜 + FinalScore Top 20
    lines.append("【段落 3: 五规则正交共振榜与 FinalScore Top 20】")
    lines.append("-" * 80)
    lines.append("  * 五规则体系说明:")
    lines.append("    - R1: 强连共振码 (Markov/近期连续同现)")
    lines.append("    - R2: 热度龙头码 (10G/25G 多尺度多头排列)")
    lines.append("    - R3: 跨期互补码 (跟随转移高概率引导)")
    lines.append("    - R4: 强伴随防撞码 (共现协方差网络中枢)")
    lines.append("    - R5: 全局冷热反弹码 (Runs游程聚集与回补)")
    lines.append("")

    votes = ctx.rounds.votes
    final_scores = ctx.rounds.scores.final
    sorted_by_score = sorted(range(1, 81), key=lambda x: final_scores[x - 1], reverse=True)

    # 寻找 5 票与 4 票高共振码
    res_5 = [n for n in range(1, 81) if votes[n - 1] == 5]
    res_4 = [n for n in range(1, 81) if votes[n - 1] == 4]
    lines.append(f"  * 五星共振金码 (5票): " + (", ".join(f"{n:02d}" for n in res_5) if res_5 else "无 (本期五条规则独立分散)"))
    lines.append(f"  * 四星共振银码 (4票): " + (", ".join(f"{n:02d}" for n in res_4) if res_4 else "无"))
    lines.append("")

    lines.append("  * 综合动力学评分 Top 20 精选梯队:")
    lines.append("    名次  号码   总分(FinalScore)  五规则得票  10G热度  当前遗漏  动量金叉")
    golden_cross = ctx.snapshot.golden_cross
    gaps = ctx.snapshot.gap
    for rank_idx, num in enumerate(sorted_by_score[:20], start=1):
        idx = num - 1
        score_val = final_scores[idx]
        vote_val = votes[idx]
        f_val = f10[idx]
        o_val = int(gaps[idx])
        gc_lbl = "★金叉" if golden_cross[idx] else "平稳"
        lines.append(f"    {rank_idx:02d}    [{num:02d}]      {score_val:7.4f}           {vote_val}票       {f_val:.2f}      {o_val:02d}期      {gc_lbl}")
    lines.append("")

    # 步骤 6: [4] 金胆池/复式池 + 形态体检
    lines.append("【段落 4: 金胆池 / 核心复式大底与形态体检卡】")
    lines.append("-" * 80)
    gold_pool = ctx.rounds.pool.gold
    main_pool = ctx.rounds.pool.main
    lines.append(f"  * 核心超级金胆 (5 码): " + " ".join(f"{n:02d}" for n in gold_pool))
    lines.append(f"  * 精选复式大底 ({len(main_pool)} 码): " + " ".join(f"{n:02d}" for n in main_pool))
    lines.append("")

    # 形态体检
    sum_gold = sum(gold_pool) if gold_pool else 0
    odds_gold = sum(1 for n in gold_pool if n % 2 == 1)
    evens_gold = len(gold_pool) - odds_gold
    span_gold = (max(gold_pool) - min(gold_pool)) if gold_pool else 0
    lines.append("  * 金胆组形态体检卡:")
    lines.append(f"    - 和值: {sum_gold} (合理区间 150~250)")
    lines.append(f"    - 奇偶比: {odds_gold}:{evens_gold}")
    lines.append(f"    - 跨度: {span_gold}")
    if ctx.rounds.pool.warnings:
        lines.append(f"    - 池选告警: {'; '.join(ctx.rounds.pool.warnings)}")
    lines.append("")

    # 步骤 7: [5] 覆盖注单与保底方案
    lines.append("【段落 5: 覆盖注单与组合多层级保底投注方案】")
    lines.append("-" * 80)
    if ctx.plans:
        total_tickets = sum(len(p.tickets) for p in ctx.plans)
        total_cost = sum(p.cost_yuan for p in ctx.plans)
        lines.append(f"  * 推荐方案: 共 {len(ctx.plans)} 个玩法计划，总注数: {total_tickets} 注，总投入: {total_cost} 元")
        for plan_i, plan in enumerate(ctx.plans, start=1):
            lines.append(f"  --- [计划 {plan_i}]: 玩法={plan.play.upper()}, 选号数 k={plan.k}, 单注预算={plan.budget}注 ({plan.cost_yuan}元), 求解方式={plan.method} ---")
            lines.append(f"      理论保底设计: {plan.guarantee_level.get('design', 'C(v,k,t)')}")
            lines.append("      精选实战投注单列表:")
            for t_idx, ticket in enumerate(plan.tickets, start=1):
                tk_str = " ".join(f"{x:02d}" for x in ticket)
                lines.append(f"        第 {t_idx:02d} 注: [ {tk_str} ]")
    else:
        lines.append("  * 暂无注单生成（当前配置未启用注单构建或预算为 0）。")
    lines.append("")

    # 步骤 8: [附录A] 80 码底层全参表
    lines.append("【附录 A: 80 码底层动力学微积分全参矩阵】")
    lines.append("-" * 80)
    lines.append(" 号码 | BaseScore | R1 R2 R3 R4 R5 | 得票 | FinalScore | 10G频次 | 25G频次 | 50G频次 | 遗漏 | 金叉")
    lines.append("-" * 80)
    f25 = ctx.snapshot.freq_25
    f50 = ctx.snapshot.freq_50
    base_scores = ctx.rounds.scores.base
    for num in range(1, 81):
        i = num - 1
        b_sc = base_scores[i]
        f_sc = final_scores[i]
        vt = votes[i]
        v_r1 = int(ctx.rounds.verdicts[0].mask[i])
        v_r2 = int(ctx.rounds.verdicts[1].mask[i])
        v_r3 = int(ctx.rounds.verdicts[2].mask[i])
        v_r4 = int(ctx.rounds.verdicts[3].mask[i])
        v_r5 = int(ctx.rounds.verdicts[4].mask[i])
        gc_mark = "★金叉" if golden_cross[i] else "平稳"
        lines.append(
            f"  {num:02d}  |  {b_sc:7.4f}  |  {v_r1}  {v_r2}  {v_r3}  {v_r4}  {v_r5}  |  {vt}票  |  {f_sc:7.4f}   |  {f10[i]:.2f}   |  {f25[i]:.2f}   |  {f50[i]:.2f}   |  {int(gaps[i]):02d}  | {gc_mark}"
        )
    lines.append("")

    # 步骤 9: [附录B] 回测与归因摘要
    lines.append("【附录 B: T0=300 Walk-Forward 滚动时序实测归因摘要】")
    lines.append("-" * 80)
    if ctx.backtest is not None:
        lines.append(f"  * 回测期数: {ctx.backtest.n_sample} 期 (冷启动 T0={ctx.backtest.t0})")
        lines.append(f"  * 覆盖期号范围: {ctx.backtest.start_period} ~ {ctx.backtest.end_period}")
        lines.append(f"  * Top-K 提升度 (Lift@K) 检阅:")
        for k_val, lstat in sorted(ctx.backtest.lift.items(), key=lambda x: x[0]):
            sig_mark = "★显著" if lstat.is_signal else "平稳"
            lines.append(f"    - Top-{k_val:02d}: 实测均中={lstat.mean_hit:.2f}, 理论期望={lstat.expected:.2f}, Lift={lstat.lift:.3f}, p值={lstat.p_value:.4f} ({sig_mark})")
        if ctx.backtest.bankroll is not None:
            broll = ctx.backtest.bankroll
            lines.append(f"  * 资金收益表现: 总投入={broll.total_cost_yuan}元, 总奖金={broll.total_prize_yuan}元, 净利={broll.net_yuan}元, ROI={broll.roi*100:.2f}%, 最大回撤={broll.max_drawdown_yuan}元")
    else:
        lines.append("  * 暂无关联回测报告。")
    lines.append("")

    # 尾注与告警
    if ctx.warnings:
        lines.append("【系统执行提示与告警】")
        for w in ctx.warnings:
            lines.append(f"  ! {w}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("              [工业级研发生态·Antigravity 全自动生成交付件]")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_text_report(text: str, path: str | Path) -> Path:
    """保存控制台精要 TXT 报表到本地指定路径。

    @param text 报表文本字符串
    @param path 保存目标路径
    @return 保存成功的文件 Path 对象
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p
