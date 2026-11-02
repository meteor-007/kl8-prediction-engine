"""微信与手机移动端发布文案生成器（契约 R48-d、§3.13）。

【业务与交互红线】
1. 字符数严格控制在 1200 字符以内（≤1200），防止手机转发截断；
2. 零晦涩物理数学符号（严禁出现 k, p, v, a, g, due 等符号堆砌）；
3. 纯正通俗业务表达（追热龙头、防冷反弹、共振金胆、跨路互补）；
4. 视觉结构精美、带醒目 emoji 标识，手机端一键全选即可直接转发。
"""

from __future__ import annotations

from pathlib import Path

from kl8.reporting.text_report import ReportContext


def render_wechat(ctx: ReportContext) -> str:
    """渲染极简精巧、手机端一键直发微信文案（契约 §3.13）。

    @param ctx 报表上下文数据源
    @return 纯净紧凑的微信发布文案（长度严格控制在 1200 字以内）
    """
    lines: list[str] = []

    # 标题区
    lines.append(f"🎯【快乐8当期五维精选·第{ctx.target_period}期】")
    lines.append(f"📅 开奖日期：{ctx.target_date}")
    lines.append("──────────────────────")

    # 上期简要复盘
    if ctx.ds.draws:
        last_draw = ctx.ds.draws[-1]
        sorted_last = sorted(list(last_draw.numbers))
        lines.append(f"🔄【上期开奖】第{last_draw.period}期")
        lines.append(f"号码: {' '.join(f'{n:02d}' for n in sorted_last[:10])}")
        lines.append(f"      {' '.join(f'{n:02d}' for n in sorted_last[10:])}")
        lines.append("──────────────────────")

    # 超级金胆
    gold_pool = ctx.rounds.pool.gold
    gold_str = " ".join(f"{n:02d}" for n in gold_pool)
    lines.append("🔥【核心超级金胆 (5码)】")
    lines.append(f"👉 [ {gold_str} ]")
    lines.append("──────────────────────")

    # 精选复式大底
    main_pool = ctx.rounds.pool.main
    main_str = " ".join(f"{n:02d}" for n in main_pool)
    lines.append(f"⭐【精选复式大底 ({len(main_pool)}码)】")
    lines.append(f"👉 {main_str}")
    lines.append("──────────────────────")

    # 核心保底实战票
    lines.append("🎟️【核心保底实战推荐票】")
    if ctx.plans:
        # 展示前 5 注推荐票，避免超长
        ticket_count = 0
        for plan in ctx.plans:
            lines.append(f"📌 {plan.play.upper()} 计划 (单注{plan.k}码，设计:{plan.guarantee_level.get('design', '保底')})")
            for tk in plan.tickets:
                ticket_count += 1
                if ticket_count <= 6:
                    tk_str = " ".join(f"{x:02d}" for x in tk)
                    lines.append(f"  • 注{ticket_count}: [ {tk_str} ]")
        if ticket_count > 6:
            lines.append(f"  (其余 {ticket_count - 6} 注完整清单详见控制台/Excel报告)")
    else:
        lines.append("  (当期采用大底守号策略，未拆单)")
    lines.append("──────────────────────")

    # 形态体检与风控卡
    sum_gold = sum(gold_pool) if gold_pool else 0
    odds_gold = sum(1 for n in gold_pool if n % 2 == 1)
    evens_gold = len(gold_pool) - odds_gold
    lines.append("📊【宏观形态体检卡】")
    lines.append(f"  • 金胆和值: {sum_gold} (合理区间 150~250)")
    lines.append(f"  • 奇偶比例: {odds_gold}奇 : {evens_gold}偶")
    lines.append("──────────────────────")

    # 理性投资说明
    lines.append("💡【理性购彩提示】")
    lines.append("本数据由五层正交动力学模型计算生成，仅供参考。大数定律防冲动，请理性投资！")

    text = "\n".join(lines)

    # 强力字符数保障：若超出 1200 字执行安全截断
    if len(text) > 1200:
        text = text[:1190] + "\n...(略)"

    return text


def save_wechat(text: str, path: str | Path) -> Path:
    """保存微信文案到指定文件。

    @param text 微信文案字符串
    @param path 保存路径
    @return 保存路径 Path 对象
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p
