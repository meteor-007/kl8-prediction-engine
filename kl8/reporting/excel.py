"""商业级 Excel 多工作表决策报表生成器（契约 R48-d、§3.13）。

【六大标准工作表】
1. 「当期总览」: 核心胆组、复式池、形态体检指标、推荐方案摘要
2. 「80码五规则全参」: 80 码底层动力学指标、五规则投票矩阵、高亮选号
3. 「覆盖注单」: 组合覆盖算法生成的推荐实战注单与理论保底方案
4. 「回测归因」: Walk-Forward 滚动回测 Lift@K 提升度与资金曲线
5. 「随机性基准」: NIST-Lite 四项检验、卡方拟合度与大数理论期望
6. 「上期复盘」: 上期真实开奖与实测命中复盘
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from kl8.reporting.text_report import ReportContext


@dataclass(frozen=True)
class ExcelStyle:
    """Excel 商业报表渲染风格规范（契约 §3.13）。"""

    header_fill: str = "1F497D"
    """表头背景深蓝配色。"""

    font_name: str = "微软雅黑"
    """全局中文字体。"""

    hit_fill: str = "FFF2CC"
    """入选金胆/大底的高亮浅黄色背景。"""

    sheet_names: tuple[str, ...] = (
        "当期总览",
        "80码五规则全参",
        "覆盖注单",
        "回测归因",
        "随机性基准",
        "上期复盘",
    )


def _apply_header_style(cell: Any, style: ExcelStyle) -> None:
    """给表头单元格应用深蓝白字、居中加粗样式。"""
    cell.fill = PatternFill(
        start_color=style.header_fill,
        end_color=style.header_fill,
        fill_type="solid",
    )
    cell.font = Font(
        name=style.font_name,
        size=11,
        bold=True,
        color="FFFFFF",
    )
    cell.alignment = Alignment(horizontal="center", vertical="center")


def _apply_data_style(
    cell: Any,
    style: ExcelStyle,
    *,
    highlight: bool = False,
    align_center: bool = True,
) -> None:
    """应用常规数据单元格样式与可选浅黄高亮。"""
    cell.font = Font(name=style.font_name, size=10)
    cell.alignment = Alignment(
        horizontal="center" if align_center else "left",
        vertical="center",
    )
    if highlight:
        cell.fill = PatternFill(
            start_color=style.hit_fill,
            end_color=style.hit_fill,
            fill_type="solid",
        )


def _autofit_columns(ws: Any) -> None:
    """自动微调列宽以容纳内容。"""
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            # 简单估计中英文字符宽度
            val_len = sum(2 if ord(c) > 127 else 1 for c in val_str)
            if val_len > max_len:
                max_len = val_len
        ws.column_dimensions[col_letter].width = max(max_len + 3, 10)


def render_excel(
    ctx: ReportContext,
    path: str | Path,
    *,
    style: ExcelStyle = ExcelStyle(),
) -> Path:
    """渲染多 Sheet 商业级 Excel 分析报表（契约 §3.13）。

    包含文件占用异常防护：若文件被 Excel 独占，尝试重试 3 次；
    若依然失败，记录日志并不阻断其余交付件生成。

    @param ctx 报表上下文数据源
    @param path 保存路径
    @param style 报表视觉样式配置
    @return 最终落盘的 Path
    """
    wb = openpyxl.Workbook()
    # 移除默认 sheet
    if wb.active:
        wb.remove(wb.active)

    # ─────────────────────────────────────────────────────────────
    # Sheet 1: 当期总览
    # ─────────────────────────────────────────────────────────────
    ws_overview = wb.create_sheet(title=style.sheet_names[0])
    ws_overview.append(["快乐8 (Keno) 当期决策总览卡", ""])
    ws_overview.merge_cells("A1:B1")
    _apply_header_style(ws_overview["A1"], style)

    ws_overview.append(["指标项", "内容与数值"])
    _apply_header_style(ws_overview["A2"], style)
    _apply_header_style(ws_overview["B2"], style)

    gold_pool = ctx.rounds.pool.gold
    main_pool = ctx.rounds.pool.main
    overview_data = [
        ("预测目标期号", f"第 {ctx.target_period} 期"),
        ("预测开奖日期", ctx.target_date),
        ("核心超级金胆 (5码)", " ".join(f"{n:02d}" for n in gold_pool)),
        ("精选复式大底 (码)", " ".join(f"{n:02d}" for n in main_pool)),
        ("大底规模", f"{len(main_pool)} 码"),
        ("金胆和值", sum(gold_pool) if gold_pool else 0),
        ("金胆奇偶比", f"{sum(1 for n in gold_pool if n % 2 == 1)}奇 : {sum(1 for n in gold_pool if n % 2 == 0)}偶"),
        ("金胆跨度", (max(gold_pool) - min(gold_pool)) if gold_pool else 0),
        ("推荐玩法注数", sum(len(p.tickets) for p in ctx.plans) if ctx.plans else 0),
        ("总投资预算 (元)", sum(p.cost_yuan for p in ctx.plans) if ctx.plans else 0),
    ]

    for row_idx, (k, v) in enumerate(overview_data, start=3):
        ws_overview.append([k, v])
        _apply_data_style(ws_overview[f"A{row_idx}"], style, align_center=False)
        _apply_data_style(ws_overview[f"B{row_idx}"], style, highlight=(k.startswith("核心") or k.startswith("精选")))

    _autofit_columns(ws_overview)

    # ─────────────────────────────────────────────────────────────
    # Sheet 2: 80码五规则全参
    # ─────────────────────────────────────────────────────────────
    ws_params = wb.create_sheet(title=style.sheet_names[1])
    param_headers = [
        "号码",
        "BaseScore",
        "R1(强连)",
        "R2(热度)",
        "R3(跨期)",
        "R4(伴随)",
        "R5(冷热)",
        "总票数",
        "FinalScore",
        "10G频次",
        "25G频次",
        "50G频次",
        "当前遗漏",
        "游程状态",
        "金胆入选",
        "大底入选",
    ]
    ws_params.append(param_headers)
    for col_i in range(1, len(param_headers) + 1):
        _apply_header_style(ws_params.cell(row=1, column=col_i), style)

    f10 = ctx.snapshot.freq_10
    f25 = ctx.snapshot.freq_25
    f50 = ctx.snapshot.freq_50
    gaps = ctx.snapshot.gap
    golden_cross = ctx.snapshot.golden_cross
    base_scores = ctx.rounds.scores.base
    final_scores = ctx.rounds.scores.final
    votes = ctx.rounds.votes
    gold_set = set(gold_pool)
    main_set = set(main_pool)

    for num in range(1, 81):
        idx = num - 1
        is_gold = num in gold_set
        is_main = num in main_set
        row_vals = [
            f"{num:02d}",
            round(float(base_scores[idx]), 4),
            int(ctx.rounds.verdicts[0].mask[idx]),
            int(ctx.rounds.verdicts[1].mask[idx]),
            int(ctx.rounds.verdicts[2].mask[idx]),
            int(ctx.rounds.verdicts[3].mask[idx]),
            int(ctx.rounds.verdicts[4].mask[idx]),
            int(votes[idx]),
            round(float(final_scores[idx]), 4),
            round(float(f10[idx]), 2),
            round(float(f25[idx]), 2),
            round(float(f50[idx]), 2),
            int(gaps[idx]),
            "★金叉" if golden_cross[idx] else "平稳",
            "★金胆" if is_gold else "-",
            "✔入选" if is_main else "-",
        ]
        ws_params.append(row_vals)
        r_num = num + 1
        for col_j in range(1, len(row_vals) + 1):
            _apply_data_style(
                ws_params.cell(row=r_num, column=col_j),
                style,
                highlight=(is_gold or is_main),
            )

    _autofit_columns(ws_params)

    # ─────────────────────────────────────────────────────────────
    # Sheet 3: 覆盖注单
    # ─────────────────────────────────────────────────────────────
    ws_tickets = wb.create_sheet(title=style.sheet_names[2])
    ticket_headers = [
        "序号",
        "计划名称",
        "玩法",
        "选号数(k)",
        "注单号码",
        "理论设计",
        "单注成本(元)",
    ]
    ws_tickets.append(ticket_headers)
    for col_i in range(1, len(ticket_headers) + 1):
        _apply_header_style(ws_tickets.cell(row=1, column=col_i), style)

    ticket_row_idx = 2
    global_seq = 1
    for plan in ctx.plans:
        design_str = plan.guarantee_level.get("design", "保底设计")
        for tk in plan.tickets:
            tk_str = " ".join(f"{x:02d}" for x in tk)
            ws_tickets.append([
                global_seq,
                f"计划-{plan.play.upper()}",
                plan.play.upper(),
                plan.k,
                tk_str,
                design_str,
                2,
            ])
            for c_i in range(1, len(ticket_headers) + 1):
                _apply_data_style(
                    ws_tickets.cell(row=ticket_row_idx, column=c_i),
                    style,
                )
            ticket_row_idx += 1
            global_seq += 1

    _autofit_columns(ws_tickets)

    # ─────────────────────────────────────────────────────────────
    # Sheet 4: 回测归因
    # ─────────────────────────────────────────────────────────────
    ws_backtest = wb.create_sheet(title=style.sheet_names[3])
    ws_backtest.append(["Walk-Forward 滚动时序实测指标归因卡", ""])
    ws_backtest.merge_cells("A1:G1")
    _apply_header_style(ws_backtest["A1"], style)

    ws_backtest.append([
        "Top-K",
        "实测均中",
        "理论期望",
        "提升度(Lift)",
        "t 统计量",
        "p-value",
        "显著信号",
    ])
    for c_i in range(1, 8):
        _apply_header_style(ws_backtest.cell(row=2, column=c_i), style)

    b_row_num = 3
    if ctx.backtest is not None:
        for k_val, lstat in sorted(ctx.backtest.lift.items(), key=lambda x: x[0]):
            ws_backtest.append([
                f"Top-{k_val}",
                round(float(lstat.mean_hit), 2),
                round(float(lstat.expected), 2),
                round(float(lstat.lift), 3),
                round(float(lstat.t_stat), 3),
                round(float(lstat.p_value), 4),
                "★显著信号" if lstat.is_signal else "平稳",
            ])
            for c_i in range(1, 8):
                _apply_data_style(
                    ws_backtest.cell(row=b_row_num, column=c_i),
                    style,
                    highlight=lstat.is_signal,
                )
            b_row_num += 1

        b_row_num += 1
        ws_backtest.cell(row=b_row_num, column=1, value="资金收益概览")
        ws_backtest.merge_cells(f"A{b_row_num}:B{b_row_num}")
        _apply_header_style(ws_backtest[f"A{b_row_num}"], style)
        b_row_num += 1

        if ctx.backtest.bankroll is not None:
            broll = ctx.backtest.bankroll
            b_metrics = [
                ("回测期数", f"{ctx.backtest.n_sample} 期"),
                ("累计投入 (元)", broll.total_cost_yuan),
                ("累计奖金 (元)", broll.total_prize_yuan),
                ("资金净收益 (元)", broll.net_yuan),
                ("投资回报率 ROI", f"{broll.roi * 100:.2f}%"),
                ("最大回撤 (元)", broll.max_drawdown_yuan),
                ("最长连挂 (期)", broll.max_losing_streak),
                ("单期最佳收益 (元)", broll.best_round_yuan),
            ]
            for m_k, m_v in b_metrics:
                ws_backtest.append([m_k, m_v])
                _apply_data_style(ws_backtest[f"A{b_row_num}"], style, align_center=False)
                _apply_data_style(ws_backtest[f"B{b_row_num}"], style)
                b_row_num += 1
    else:
        ws_backtest.append(["暂无回测报告", "-", "-", "-", "-", "-", "-"])

    _autofit_columns(ws_backtest)

    # ─────────────────────────────────────────────────────────────
    # Sheet 5: 随机性基准
    # ─────────────────────────────────────────────────────────────
    ws_rand = wb.create_sheet(title=style.sheet_names[4])
    ws_rand.append(["宏观大数基准与检验项", "统计量/结果", "判定说明"])
    for c_i in range(1, 4):
        _apply_header_style(ws_rand.cell(row=1, column=c_i), style)

    rand_rows = [
        ("理论单码概率 p0", "25.00%", "20 / 80"),
        ("理论两码同现概率 p_co", "6.01%", "20*19 / (80*79)"),
        ("重号理论均值 E(Repeat)", "5.00", "20 * 0.25"),
    ]
    if ctx.randomness is not None:
        if "ALL" in ctx.randomness.chi2:
            c_res = ctx.randomness.chi2["ALL"]
            rand_rows.append((
                "卡方拟合度检验 (GOF)",
                f"chi2={c_res.statistic:.2f}, p={c_res.p_value:.4f}",
                "均匀随机" if c_res.p_value >= 0.05 else "非均匀随机",
            ))
        for v in ctx.randomness.nist:
            rand_rows.append((
                f"NIST-Lite [{v.name}]",
                f"stat={v.statistic:.4f}, p={v.p_value:.4f}",
                "合格" if v.passed else "预警",
            ))

    for r_i, (k, val, desc) in enumerate(rand_rows, start=2):
        ws_rand.append([k, val, desc])
        for c_j in range(1, 4):
            _apply_data_style(ws_rand.cell(row=r_i, column=c_j), style)

    _autofit_columns(ws_rand)

    # ─────────────────────────────────────────────────────────────
    # Sheet 6: 上期复盘
    # ─────────────────────────────────────────────────────────────
    ws_rev = wb.create_sheet(title=style.sheet_names[5])
    ws_rev.append(["上期开奖指标与实测复盘", "详细数值"])
    for c_i in range(1, 3):
        _apply_header_style(ws_rev.cell(row=1, column=c_i), style)

    rev_rows = []
    if ctx.ds.draws:
        last = ctx.ds.draws[-1]
        sorted_last = sorted(list(last.numbers))
        rev_rows.append(("上期开奖期号", f"第 {last.period} 期"))
        rev_rows.append(("上期开奖日期", last.date))
        rev_rows.append(("开奖号码清单", " ".join(f"{x:02d}" for x in sorted_last)))
        if ctx.previous_review:
            for pr_k, pr_v in ctx.previous_review.items():
                rev_rows.append((f"前期复盘-{pr_k}", str(pr_v)))
    else:
        rev_rows.append(("历史数据", "无"))

    for r_i, (k, v) in enumerate(rev_rows, start=2):
        ws_rev.append([k, v])
        _apply_data_style(ws_rev[f"A{r_i}"], style, align_center=False)
        _apply_data_style(ws_rev[f"B{r_i}"], style)

    _autofit_columns(ws_rev)

    # ─────────────────────────────────────────────────────────────
    # 保存与文件占用 3 次重试保护
    # ─────────────────────────────────────────────────────────────
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            wb.save(p)
            return p
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                # 达到重试上限，捕获并安全返回
                print(f"[Warning] Excel 报表保存失败 (文件可能正被打开占用): {p}")
                return p
        except Exception as e:
            print(f"[Warning] Excel 保存异常: {e}")
            return p

    return p
