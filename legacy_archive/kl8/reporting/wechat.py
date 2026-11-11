"""微信与手机移动端发布文案生成器（契约 R48-d、§3.13、§6.6）。

【业务与交互红线】
1. 字符数严格控制在 1200 字符以内（≤1200），防止手机转发截断；
2. 零晦涩物理数学符号（严禁出现 k, p, v, a, g, due 等符号堆砌）；
3. 纯正通俗业务表达（追热龙头、防冷反弹、共振金胆、跨路互补）；
4. 视觉结构精美、带醒目 emoji 标识，手机端一键全选即可直接转发；
5. P8-M5/§6.6：禁止渲染层硬编码（码数/和值区间/常数全部动态生成），
   奖级未确认玩法必须标注 ``ROI: N/A（官方奖级未确认）``。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

from pathlib import Path

from kl8.atomic_io import atomic_write_text
from kl8.constants import NO_BASELINE_TEXT, PLAY_LABELS, ROI_NA_TEXT, sum_band
from kl8.reporting.text_report import ReportContext

#: 微信文案硬上限（契约 §3.13：超过会被手机端截断）。
WECHAT_MAX_CHARS: int = 1200

#: 展示的最大推荐票数（避免超长）。
WECHAT_MAX_TICKETS: int = 6


def _review_block(ctx: ReportContext) -> list[str]:
    """生成上期复盘短句（§3.14.1：缺失时输出固定文案，且不输出任何命中数）。

    @param ctx 报表上下文
    @return 文本行列表
    """
    review = ctx.previous_review
    if not review:
        return [f"📋【上期预测复盘】{NO_BASELINE_TEXT}"]
    status = str(review.get("status", "baseline"))
    if status != "ok":
        note = str(review.get("note", "")).strip() or NO_BASELINE_TEXT
        return [f"📋【上期预测复盘】{note}"]
    gold_hit = int(review.get("gold_hit", -1))
    main_hit = int(review.get("main_hit", -1))
    gold = review.get("gold", ())
    main = review.get("main", ())
    return [
        f"📋【上期预测复盘】第{review.get('period', 0)}期："
        f"金胆命中 {gold_hit}/{len(gold)}，复式命中 {main_hit}/{len(main)}"
    ]


def render_wechat(ctx: ReportContext) -> str:
    """渲染极简精巧、手机端一键直发微信文案（契约 §3.13、§6.6）。

    @param ctx 报表上下文数据源
    @return 纯净紧凑的微信发布文案（长度严格控制在 1200 字以内）
    """
    lines: list[str] = []

    # 标题区
    lines.append(f"🎯【快乐8当期五维精选·第{ctx.target_period}期】")
    lines.append(f"📅 开奖日期：{ctx.target_date}")
    lines.append("──────────────────────")

    # 上期开奖
    if ctx.ds.draws:
        last_draw = ctx.ds.draws[-1]
        sorted_last = sorted(list(last_draw.numbers))
        lines.append(f"🔄【上期开奖】第{last_draw.period}期")
        lines.append(f"号码: {' '.join(f'{num:02d}' for num in sorted_last[:10])}")
        lines.append(f"      {' '.join(f'{num:02d}' for num in sorted_last[10:])}")
        lines.append("──────────────────────")

    # 上期复盘（只读 §3.14.1 状态文件）
    lines.extend(_review_block(ctx))
    lines.append("──────────────────────")

    # 金胆
    gold_pool = ctx.rounds.pool.gold
    gold_str = " ".join(f"{num:02d}" for num in gold_pool)
    lines.append(f"🔥【核心超级金胆 ({len(gold_pool)}码)】")
    lines.append(f"👉 [ {gold_str} ]")
    lines.append("──────────────────────")

    # 复式大底
    main_pool = ctx.rounds.pool.main
    main_str = " ".join(f"{num:02d}" for num in main_pool)
    lines.append(f"⭐【精选复式大底 ({len(main_pool)}码)】")
    lines.append(f"👉 {main_str}")
    lines.append("──────────────────────")

    # 实战推荐票
    lines.append("🎟️【核心保底实战推荐票】")
    if ctx.plans:
        ticket_count = 0
        unknown_play = False
        for plan in ctx.plans:
            play_cn = PLAY_LABELS.get(plan.play, plan.play)
            design = plan.guarantee_level.get("design", "")
            lines.append(f"📌 {play_cn}计划（单注{plan.k}码，设计:{design}）")
            if ctx.cfg.prize_table.get(plan.play) is None:
                unknown_play = True
            for ticket in plan.tickets:
                ticket_count += 1
                if ticket_count <= WECHAT_MAX_TICKETS:
                    tk_str = " ".join(f"{x:02d}" for x in ticket)
                    lines.append(f"  • 注{ticket_count}: [ {tk_str} ]")
        if ticket_count > WECHAT_MAX_TICKETS:
            lines.append(f"  (其余 {ticket_count - WECHAT_MAX_TICKETS} 注完整清单详见控制台/Excel报告)")
        if unknown_play:
            lines.append(f"⚠️ {ROI_NA_TEXT}")
    else:
        lines.append("  (当期采用大底守号策略，未拆单)")
    lines.append("──────────────────────")

    # 形态体检
    odd = sum(1 for num in gold_pool if num % 2 == 1)
    even = len(gold_pool) - odd
    lines.append("📊【宏观形态体检卡】")
    if gold_pool:
        low, high = sum_band(len(gold_pool))
        lines.append(f"  • 金胆和值: {sum(gold_pool)}（理论区间 {low}~{high}）")
    lines.append(f"  • 奇偶比例: {odd}奇 : {even}偶")
    lines.append("──────────────────────")

    # 理性购彩提示
    lines.append("💡【理性购彩提示】")
    lines.append("本数据由五层正交动力学模型计算生成，仅供参考。大数定律防冲动，请理性投资！")

    text = "\n".join(lines)
    if len(text) > WECHAT_MAX_CHARS:
        text = text[: WECHAT_MAX_CHARS - 10] + "\n...(略)"
    return text


def save_wechat(text: str, path: str | Path) -> Path:
    """保存微信文案到指定文件（原子写，P8-M13）。

    @param text 微信文案字符串
    @param path 保存路径
    @return 保存路径 Path 对象
    @raises OSError 写入失败
    """
    return atomic_write_text(path, text)