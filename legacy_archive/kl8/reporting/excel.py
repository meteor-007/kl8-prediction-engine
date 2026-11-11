"""商业级 Excel 多工作表决策报表生成器（契约 R48-d、§3.13、§6.6）。

【六大标准工作表】
1. 「当期总览」: 核心胆组、复式池、形态体检指标、推荐方案摘要
2. 「80码五规则全参」: 80 码底层动力学指标、五规则投票矩阵、高亮选号
3. 「覆盖注单」: 组合覆盖算法生成的推荐实战注单与理论保底方案
4. 「回测归因」: Walk-Forward Lift@K、显著性、规则归因与资金曲线
5. 「随机性基准」: NIST-Lite 四项检验、卡方拟合度与大数理论期望
6. 「上期复盘」: 上期真实开奖与 §3.14.1 复盘状态文件的命中核对

【P8-M13 修复】保存改为「同目录临时文件 → os.replace 原子替换 → 旧件 .bak」，
失败直接抛 :class:`KL8Error` 由上层记告警，**禁止 print 后仍返回路径冒充成功**。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl
import numpy as np
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from kl8.atomic_io import atomic_copy
from kl8.constants import E_REPEAT, NO_BASELINE_TEXT, P0, P_CO, PLAY_LABELS, ROI_NA_TEXT, RULE_IDS, RULE_NAMES
from kl8.errors import KL8Error
from kl8.evaluation.metrics import roi_text
from kl8.reporting.text_report import ReportContext, sample_note

#: Excel 被占用时的重试次数与间隔（秒）。
SAVE_MAX_RETRIES: int = 3
SAVE_RETRY_INTERVAL_S: float = 0.5


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
    cell.font = Font(name=style.font_name, size=11, bold=True, color="FFFFFF")
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
            val_len = sum(2 if ord(char) > 127 else 1 for char in val_str)
            if val_len > max_len:
                max_len = val_len
        ws.column_dimensions[col_letter].width = max(max_len + 3, 10)


def _overview_sheet(wb: Any, ctx: ReportContext, style: ExcelStyle) -> None:
    """渲染「当期总览」工作表。"""
    ws = wb.create_sheet(title=style.sheet_names[0])
    ws.append(["快乐8 (Keno) 当期决策总览卡", ""])
    ws.merge_cells("A1:B1")
    _apply_header_style(ws["A1"], style)
    ws.append(["指标项", "内容与数值"])
    _apply_header_style(ws["A2"], style)
    _apply_header_style(ws["B2"], style)

    gold_pool = ctx.rounds.pool.gold
    main_pool = ctx.rounds.pool.main
    total_tickets = sum(len(plan.tickets) for plan in ctx.plans) if ctx.plans else 0
    total_cost = sum(plan.cost_yuan for plan in ctx.plans) if ctx.plans else 0
    overview_data = [
        ("预测目标期号", f"第 {ctx.target_period} 期"),
        ("预测开奖日期", ctx.target_date),
        (f"核心超级金胆 ({len(gold_pool)}码)", " ".join(f"{num:02d}" for num in gold_pool)),
        (f"精选复式大底 ({len(main_pool)}码)", " ".join(f"{num:02d}" for num in main_pool)),
        ("大底规模", f"{len(main_pool)} 码"),
        ("金胆和值", sum(gold_pool) if gold_pool else 0),
        (
            "金胆奇偶比",
            f"{sum(1 for num in gold_pool if num % 2 == 1)}奇 : "
            f"{sum(1 for num in gold_pool if num % 2 == 0)}偶",
        ),
        ("金胆跨度", (max(gold_pool) - min(gold_pool)) if gold_pool else 0),
        ("大底重号/新号", f"{ctx.rounds.pool.repeat_count} / {ctx.rounds.pool.new_count}"),
        ("已放宽约束", "、".join(ctx.rounds.pool.relaxed) if ctx.rounds.pool.relaxed else "无"),
        ("推荐玩法注数", total_tickets),
        ("总投资预算 (元)", total_cost),
    ]
    for row_idx, (metric, value) in enumerate(overview_data, start=3):
        ws.append([metric, value])
        _apply_data_style(ws[f"A{row_idx}"], style, align_center=False)
        _apply_data_style(
            ws[f"B{row_idx}"],
            style,
            highlight=(metric.startswith("核心") or metric.startswith("精选")),
        )
    _autofit_columns(ws)


def _params_sheet(wb: Any, ctx: ReportContext, style: ExcelStyle) -> None:
    """渲染「80码五规则全参」工作表。"""
    ws = wb.create_sheet(title=style.sheet_names[1])
    headers = [
        "号码",
        "BaseScore",
        "R1(多尺度强跟随)",
        "R2(温热多头)",
        "R3(动量金叉)",
        "R4(强共现拉动)",
        "R5(全量回补)",
        "整型票数",
        "等效票数",
        "FinalScore",
        "10G频次",
        "25G频次",
        "50G频次",
        "当前遗漏",
        "游程状态",
        "金胆入选",
        "大底入选",
    ]
    ws.append(headers)
    for col_i in range(1, len(headers) + 1):
        _apply_header_style(ws.cell(row=1, column=col_i), style)

    f10 = ctx.snapshot.freq_10
    f25 = ctx.snapshot.freq_25
    f50 = ctx.snapshot.freq_50
    gaps = ctx.snapshot.gap
    golden_cross = ctx.snapshot.golden_cross
    base_scores = ctx.rounds.scores.base
    final_scores = ctx.rounds.scores.final
    votes = ctx.rounds.votes
    equivalence = ctx.rounds.weighted_votes
    masks = [np.asarray(verdict.mask, dtype=bool) for verdict in ctx.rounds.verdicts]
    gold_set = set(ctx.rounds.pool.gold)
    main_set = set(ctx.rounds.pool.main)

    for num in range(1, 81):
        idx = num - 1
        is_gold = num in gold_set
        is_main = num in main_set
        eq_val = float(equivalence[idx]) if equivalence is not None else float(votes[idx])
        row_vals = [
            f"{num:02d}",
            round(float(base_scores[idx]), 4),
            *[int(mask[idx]) for mask in masks],
            int(votes[idx]),
            round(eq_val, 2),
            round(float(final_scores[idx]), 4),
            round(float(f10[idx]), 2),
            round(float(f25[idx]), 2),
            round(float(f50[idx]), 2),
            int(gaps[idx]),
            "★金叉" if golden_cross[idx] else "平稳",
            "★金胆" if is_gold else "-",
            "✔入选" if is_main else "-",
        ]
        ws.append(row_vals)
        row_num = num + 1
        for col_j in range(1, len(row_vals) + 1):
            _apply_data_style(
                ws.cell(row=row_num, column=col_j),
                style,
                highlight=(is_gold or is_main),
            )
    _autofit_columns(ws)


def _tickets_sheet(wb: Any, ctx: ReportContext, style: ExcelStyle) -> None:
    """渲染「覆盖注单」工作表。"""
    ws = wb.create_sheet(title=style.sheet_names[2])
    headers = ["序号", "计划名称", "玩法", "选号数", "注单号码", "理论设计", "单注成本(元)", "资金口径"]
    ws.append(headers)
    for col_i in range(1, len(headers) + 1):
        _apply_header_style(ws.cell(row=1, column=col_i), style)

    row_idx = 2
    global_seq = 1
    for plan in ctx.plans:
        design_str = str(plan.guarantee_level.get("design", ""))
        play_cn = PLAY_LABELS.get(plan.play, plan.play)
        known = ctx.cfg.prize_table.get(plan.play) is not None
        money_note = "参与资金结算" if known else ROI_NA_TEXT
        for ticket in plan.tickets:
            tk_str = " ".join(f"{num:02d}" for num in ticket)
            ws.append([
                global_seq,
                f"计划-{play_cn}",
                plan.play,
                plan.k,
                tk_str,
                design_str,
                int(ctx.cfg.cost_per_ticket),
                money_note,
            ])
            for col_i in range(1, len(headers) + 1):
                _apply_data_style(ws.cell(row=row_idx, column=col_i), style)
            row_idx += 1
            global_seq += 1
    _autofit_columns(ws)


def _backtest_sheet(wb: Any, ctx: ReportContext, style: ExcelStyle) -> None:
    """渲染「回测归因」工作表（显著性三条件、规则归因、资金曲线）。"""
    ws = wb.create_sheet(title=style.sheet_names[3])
    ws.append(["Walk-Forward 滚动时序实测指标归因卡", ""])
    ws.merge_cells("A1:I1")
    _apply_header_style(ws["A1"], style)

    headers = ["Top-K", "实测均中", "理论期望", "提升度", "t 统计量", "p 值", "样本数 n", "校正后 α", "显著性判定"]
    ws.append(headers)
    for col_i in range(1, len(headers) + 1):
        _apply_header_style(ws.cell(row=2, column=col_i), style)
    row_num = 3

    if ctx.backtest is not None:
        bt = ctx.backtest
        insufficient = any(stat.n < ctx.cfg.backtest.min_n_signal for stat in bt.lift.values())
        if insufficient:
            ws.append([sample_note(ctx.cfg.backtest.min_n_signal)])
            ws.merge_cells(f"A{row_num}:I{row_num}")
            _apply_data_style(ws[f"A{row_num}"], style, align_center=False)
            row_num += 1
        for _k, stat in sorted(bt.lift.items(), key=lambda item: item[0]):
            ws.append([
                f"Top-{stat.k}",
                round(float(stat.mean_hit), 2),
                round(float(stat.expected), 2),
                round(float(stat.lift), 3),
                round(float(stat.t_stat), 3),
                round(float(stat.p_value), 4),
                int(stat.n),
                round(float(stat.alpha_adj), 4),
                "★显著信号" if stat.is_signal else "平稳",
            ])
            for col_i in range(1, len(headers) + 1):
                _apply_data_style(
                    ws.cell(row=row_num, column=col_i), style, highlight=stat.is_signal
                )
            row_num += 1

        row_num += 1
        ws.cell(row=row_num, column=1, value="规则归因（真实命中，R45）")
        ws.merge_cells(f"A{row_num}:D{row_num}")
        _apply_header_style(ws[f"A{row_num}"], style)
        row_num += 1
        if bt.attribution is not None:
            for rule_id in RULE_IDS:
                stat = bt.attribution.per_rule.get(rule_id)
                if stat is None:
                    continue
                ws.append([
                    f"{rule_id} {RULE_NAMES.get(rule_id, '')}",
                    round(float(stat.pool_size_avg), 1),
                    f"{stat.hit_rate:.2%}",
                    round(float(stat.lift), 3),
                    round(float(stat.z), 2),
                ])
                for col_i in range(1, 6):
                    _apply_data_style(ws.cell(row=row_num, column=col_i), style)
                row_num += 1
            high = bt.attribution.high_votes
            low = bt.attribution.low_votes
            ws.append([
                "高票组(V≥3) vs 低票组(V≤1)",
                f"n={int(high.get('n', 0))} / {int(low.get('n', 0))}",
                f"{high.get('hit_rate', 0.0):.2%} / {low.get('hit_rate', 0.0):.2%}",
                round(float(high.get("z", 0.0)), 2),
            ])
            for col_i in range(1, 5):
                _apply_data_style(ws.cell(row=row_num, column=col_i), style)
            row_num += 1

        row_num += 1
        ws.cell(row=row_num, column=1, value="高奖级实测触发率 vs 超几何理论（R44；理论概率为常数，禁止写死）")
        ws.merge_cells(f"A{row_num}:F{row_num}")
        _apply_header_style(ws[f"A{row_num}"], style)
        row_num += 1
        if bt.high_tier:
            for tier_key in sorted(bt.high_tier.keys()):
                tier = bt.high_tier[tier_key]
                theory_period = int(1.0 / tier.theory_rate) if tier.theory_rate > 0 else 0
                tier_denom = (
                    f"{int(tier.n_rounds)} 期 / {int(tier.n_trials)} 注"
                    if tier.n_trials > tier.n_rounds
                    else f"{int(tier.n_rounds)} 期"
                )
                ws.append([
                    str(tier_key),
                    tier_denom,
                    f"{tier.observed_rate:.4%}",
                    f"{tier.theory_rate:.4%}（≈1/{theory_period} 期）",
                    round(float(tier.ratio), 2),
                ])
                for col_i in range(1, 6):
                    _apply_data_style(ws.cell(row=row_num, column=col_i), style)
                row_num += 1
        else:
            ws.append([ROI_NA_TEXT])
            _apply_data_style(ws[f"A{row_num}"], style, align_center=False)
            row_num += 1

        row_num += 1
        ws.cell(row=row_num, column=1, value="资金收益概览")
        ws.merge_cells(f"A{row_num}:B{row_num}")
        _apply_header_style(ws[f"A{row_num}"], style)
        row_num += 1
        broll = bt.bankroll
        if broll is not None:
            metrics = [
                ("回测期数", f"{bt.n_sample} 期"),
                ("累计投入 (元)", broll.total_cost_yuan),
                ("累计奖金 (元)", broll.total_prize_yuan),
                ("资金净收益 (元)", broll.net_yuan),
                (
                    "投资回报率 ROI",
                    roi_text(broll),
                ),
                ("最大回撤 (元)", broll.max_drawdown_yuan),
                ("最长连挂 (期)", broll.max_losing_streak),
                ("单期最佳收益 (元)", broll.best_round_yuan),
            ]
            for metric, value in metrics:
                ws.append([metric, value])
                _apply_data_style(ws[f"A{row_num}"], style, align_center=False)
                _apply_data_style(ws[f"B{row_num}"], style)
                row_num += 1

        ws.append(["防穿越校验", f"已校验 {bt.lookahead_checked} 期 / 越界 {len(bt.lookahead_violations)} 期"])
        _apply_data_style(ws[f"A{row_num}"], style, align_center=False)
        _apply_data_style(ws[f"B{row_num}"], style)
        row_num += 1
        for check_key, check_value in bt.self_check.items():
            ws.append([f"自洽机检-{check_key}", str(check_value)])
            _apply_data_style(ws[f"A{row_num}"], style, align_center=False)
            _apply_data_style(ws[f"B{row_num}"], style, highlight=not check_value)
            row_num += 1
    else:
        ws.append(["暂无回测报告", "-", "-", "-", "-", "-", "-", "-", "-"])
    _autofit_columns(ws)


def _randomness_sheet(wb: Any, ctx: ReportContext, style: ExcelStyle) -> None:
    """渲染「随机性基准」工作表（常数取自 constants，禁止写死百分比）。"""
    ws = wb.create_sheet(title=style.sheet_names[4])
    ws.append(["宏观大数基准与检验项", "统计量/结果", "判定说明"])
    for col_i in range(1, 4):
        _apply_header_style(ws.cell(row=1, column=col_i), style)

    rand_rows: list[tuple[str, str, str]] = [
        ("理论单码概率 p0", f"{P0:.2%}", "抽 20 码 / 共 80 码"),
        ("理论两码同现概率 p_co", f"{P_CO:.2%}", "组合数比值 C(20,2)/C(80,2)"),
        ("重号理论均值 E(Repeat)", f"{E_REPEAT:.2f}", "抽 20 码 × 单码出现率"),
    ]
    if ctx.randomness is not None:
        if "ALL" in ctx.randomness.chi2:
            c_res = ctx.randomness.chi2["ALL"]
            rand_rows.append((
                "卡方拟合度检验 (GOF)",
                f"chi2={c_res.statistic:.2f}, p={c_res.p_value:.4f}",
                "均匀随机" if c_res.p_value >= 0.05 else "非均匀随机",
            ))
        for nist_res in ctx.randomness.nist:
            rand_rows.append((
                f"NIST-Lite [{nist_res.name}]",
                f"stat={nist_res.statistic:.4f}, p={nist_res.p_value:.4f}",
                "合格" if nist_res.passed else "预警",
            ))
        rand_rows.append((
            "NIST 口径说明",
            "已按 p0 偏置比特流适配",
            "原始 NIST 针对等概率 0/1；本结论仅作长程稳定性监控",
        ))

    for row_i, (metric, value, desc) in enumerate(rand_rows, start=2):
        ws.append([metric, value, desc])
        for col_j in range(1, 4):
            _apply_data_style(ws.cell(row=row_i, column=col_j), style)
    _autofit_columns(ws)


def _review_sheet(wb: Any, ctx: ReportContext, style: ExcelStyle) -> None:
    """渲染「上期复盘」工作表（§3.14.1：无记录时只输出固定文案，不输出命中数）。"""
    ws = wb.create_sheet(title=style.sheet_names[5])
    ws.append(["上期开奖指标与实测复盘", "详细数值"])
    for col_i in range(1, 3):
        _apply_header_style(ws.cell(row=1, column=col_i), style)

    rev_rows: list[tuple[str, str]] = []
    if ctx.ds.draws:
        last = ctx.ds.draws[-1]
        rev_rows.append(("上期开奖期号", f"第 {last.period} 期"))
        rev_rows.append(("上期开奖日期", last.date))
        rev_rows.append(("开奖号码清单", " ".join(f"{num:02d}" for num in sorted(last.numbers))))
    else:
        rev_rows.append(("历史数据", "无"))

    review = ctx.previous_review
    if not review:
        rev_rows.append(("预测复盘", NO_BASELINE_TEXT))
    else:
        status = str(review.get("status", "baseline"))
        if status == "ok":
            gold = review.get("gold", ())
            main = review.get("main", ())
            rev_rows.append(("复盘状态", "已核对上期预测"))
            rev_rows.append(("复盘期号", f"第 {review.get('period', 0)} 期"))
            rev_rows.append(("金胆命中", f"{int(review.get('gold_hit', -1))}/{len(gold)}"))
            rev_rows.append(("复式命中", f"{int(review.get('main_hit', -1))}/{len(main)}"))
            rev_rows.append(("单注最高命中", str(int(review.get("best_ticket_hit", -1)))))
        else:
            rev_rows.append(("预测复盘", str(review.get("note", "")).strip() or NO_BASELINE_TEXT))

    for row_i, (metric, value) in enumerate(rev_rows, start=2):
        ws.append([metric, value])
        _apply_data_style(ws[f"A{row_i}"], style, align_center=False)
        _apply_data_style(ws[f"B{row_i}"], style)
    _autofit_columns(ws)


def render_excel(
    ctx: ReportContext,
    path: str | Path,
    *,
    style: ExcelStyle = ExcelStyle(),
) -> Path:
    """渲染多 Sheet 商业级 Excel 分析报表（契约 §3.13、P8-M13）。

    保存策略：先写目标同目录临时文件 → 3 次重试 ``os.replace`` 原子替换 → 旧件备份 .bak；
    达重试上限或其它写失败 → 抛 :class:`KL8Error`（调用方记告警，**不得静默返回路径**）。

    @param ctx 报表上下文数据源
    @param path 保存路径
    @param style 报表视觉样式配置
    @return 最终落盘的 Path
    @raises KL8Error 保存失败（文件被占用/磁盘异常等）
    """
    wb = openpyxl.Workbook()
    if wb.active:
        wb.remove(wb.active)

    _overview_sheet(wb, ctx, style)
    _params_sheet(wb, ctx, style)
    _tickets_sheet(wb, ctx, style)
    _backtest_sheet(wb, ctx, style)
    _randomness_sheet(wb, ctx, style)
    _review_sheet(wb, ctx, style)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".xlsx.tmp")
    tmp_path = Path(handle.name)
    handle.close()
    try:
        wb.save(tmp_path)
    except OSError as exc:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise KL8Error(f"Excel 临时文件写入失败：{exc}") from exc

    last_error: BaseException | None = None
    for attempt in range(SAVE_MAX_RETRIES):
        try:
            atomic_copy(tmp_path, target, backup=True)
            return target
        except OSError as exc:
            last_error = exc
            if attempt < SAVE_MAX_RETRIES - 1:
                time.sleep(SAVE_RETRY_INTERVAL_S + attempt * SAVE_RETRY_INTERVAL_S)
    try:
        tmp_path.unlink()
    except OSError:
        pass
    raise KL8Error(
        f"Excel 报表保存失败（文件可能正被打开占用，已重试 {SAVE_MAX_RETRIES} 次）：{last_error}"
    )