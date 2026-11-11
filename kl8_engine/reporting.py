# -*- coding: utf-8 -*-
"""全渠道商业级交付报告渲染器（TXT精要、微信发布文案、Excel决策报表）。

【业务背景与架构定位】
本模块作为全渠道三维交付中心，严格落实工程铁律：
1. 结构化控制台/TXT精要；
2. 移动端/微信群一键转发版精美文案（带 emoji、手机一键全选）；
3. 商业级 Excel 决策工作簿（openpyxl 渲染标准配色、多 Sheet、零浮点失真）；
4. 【集群三元组】格式四铁律。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from kl8_engine.constants import (
    COST_PER_TICKET_YUAN,
    GOLD_POOL_SIZE,
    MAIN_POOL_SIZE,
    N_NUM,
    PRIZE_TABLE_PICK5,
)
from kl8_engine.data import Dataset
from kl8_engine.models.ensemble import EnsembleResult
from kl8_engine.points import PointsAnalysis, format_points_console
from kl8_engine.review import (
    ReviewResult,
    WindowHitStat,
    format_hit_rate_console,
    format_review_console,
)
from kl8_engine.shrinkage import ShrinkageReport, extract_cluster_triplets


def format_wechat_main_copy(
    ensemble: EnsembleResult,
    dataset: Dataset,
    tickets: Sequence[Tuple[int, ...]],
) -> str:
    """渲染微信主发布文案（格式紧凑、带 emoji 标识、手机端一键全选转发）。"""
    prev_draw = dataset.latest
    gold_str = " ".join(f"{x:02d}" for x in ensemble.gold_pool)
    main_str = " ".join(f"{x:02d}" for x in ensemble.main_pool)
    prev_nums_str = " ".join(f"{x:02d}" for x in prev_draw.numbers)

    # 计算金胆理论和值与奇偶比
    gold_sum = sum(ensemble.gold_pool)
    odd_count = sum(1 for x in ensemble.gold_pool if x % 2 == 1)
    even_count = len(ensemble.gold_pool) - odd_count

    lines = [
        f"🎯【快乐8当期五维精选·第{ensemble.target_period}期】",
        f"📅 目标期号：第 {ensemble.target_period} 期",
        "──────────────────────",
        f"🔄【上期开奖】第 {prev_draw.period} 期",
        f"号码: {prev_nums_str[:29]}",
        f"      {prev_nums_str[30:]}",
        "──────────────────────",
        f"🔥【核心超级金胆 ({len(ensemble.gold_pool)}码)】",
        f"👉 [ {gold_str} ]",
        "──────────────────────",
        f"⭐【精选复式大底 ({len(ensemble.main_pool)}码)】",
        f"👉 {main_str}",
        "──────────────────────",
        "🎟️【核心保底实战推荐票】",
        f"📌 选五计划（单注5码，推荐 {len(tickets)} 注）",
    ]
    for idx, t in enumerate(tickets[:10], start=1):
        lines.append(f"  • 注{idx:02d}: [ {' '.join(f'{x:02d}' for x in t)} ]")
    if len(tickets) > 10:
        lines.append(f"  (其余 {len(tickets) - 10} 注详见控制台与 Excel 报表)")
    lines.extend([
        "──────────────────────",
        "📊【宏观形态体检卡】",
        f"  • 金胆和值: {gold_sum}（理论区间 71~253）",
        f"  • 奇偶比例: {odd_count}奇 : {even_count}偶",
        "──────────────────────",
        "💡【理性购彩提示】",
        "本数据由深度学习 Transformer 与贝叶斯共现谱模型融合计算，仅供参考。大数定律防冲动，请理性投资！🍀",
    ])
    return "\n".join(lines)


_MODULE_LABELS: Tuple[Tuple[str, str], ...] = (
    ("dl", "Transformer DL概率"),
    ("dirichlet", "Dirichlet 贝叶斯后验"),
    ("cooccurrence", "共现谱中心性"),
    ("momentum", "动量差分(10G-40G)"),
)


def _format_top_k(scores: Sequence[float], k: int = 5) -> str:
    """将 80 维得分渲染为 Top-K 号码(分数) 串。"""
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr)[:k]
    return " ".join(f"{int(idx) + 1:02d}({arr[idx]:.4f})" for idx in order)


def _format_numbers(numbers: Sequence[int]) -> str:
    """号码列表格式化为两位补零空格串。"""
    return " ".join(f"{n:02d}" for n in numbers)


def format_comparison_console(
    official: EnsembleResult,
    official_shrinkage: ShrinkageReport,
    comparison: EnsembleResult,
    comparison_shrinkage: Optional[ShrinkageReport],
    official_mode: str = "borda",
    comparison_mode: Optional[str] = None,
) -> str:
    """渲染双融合并排对比板块（正式版 vs 实验版）。"""
    labels = {"borda": "Borda 秩融合", "weighted": "加权概率和"}
    official_label = labels.get(official_mode, official_mode)
    comparison_label = labels.get(comparison_mode or "", comparison_mode or "未知")

    gold_overlap = len(set(official.gold_pool) & set(comparison.gold_pool))
    main_overlap = len(set(official.main_pool) & set(comparison.main_pool))
    top20_overlap = len(set(official.full_order[:20]) & set(comparison.full_order[:20]))
    pool30_overlap = len(set(official.shrinkage_pool_30) & set(comparison.shrinkage_pool_30))

    def _shrink_line(label: str, shrink: Optional[ShrinkageReport]) -> str:
        if shrink is None:
            return f"      💎 {label}缩水: 未计算"
        return (
            f"      💎 {label}缩水: 留存 {shrink.after_morphology_count:,} 注 -> "
            f"{shrink.final_tickets_count} 注 / {shrink.total_cost_yuan} 元 | "
            f"保4率 {shrink.hit4_or_better_prob:.2f}% | "
            f"期望奖金 {shrink.expected_prize_per_draw:.2f} 元"
        )

    lines = [
        "【段落 9: 双融合并排对比（正式 vs 实验）】",
        "-" * 80,
        f"  * 正式版: {official_label}（进入推荐日志与全渠道交付）",
        f"  * 实验版: {comparison_label}（仅对比，不进入日志/正式交付件）",
        f"  * 重合度: 金胆 {gold_overlap}/4 | 主选 {main_overlap}/12 | "
        f"Top20 {top20_overlap}/20 | 30码大底 {pool30_overlap}/30",
        "  * 正式版:",
        f"      🔥 金胆: {_format_numbers(official.gold_pool)}",
        f"      ⭐ 主选: {_format_numbers(official.main_pool)}",
        f"      💎 30码: {_format_numbers(official.shrinkage_pool_30)}",
        _shrink_line("", official_shrinkage),
        "  * 实验版:",
        f"      🔥 金胆: {_format_numbers(comparison.gold_pool)}",
        f"      ⭐ 主选: {_format_numbers(comparison.main_pool)}",
        f"      💎 30码: {_format_numbers(comparison.shrinkage_pool_30)}",
        _shrink_line("", comparison_shrinkage),
    ]
    return "\n".join(lines)


def format_console_text_report(
    ensemble: EnsembleResult,
    dataset: Dataset,
    shrinkage: ShrinkageReport,
    tickets: Sequence[Tuple[int, ...]],
    review: Optional[ReviewResult] = None,
    points_analysis: Optional[PointsAnalysis] = None,
    points_mode: str = "compare",
    fusion_mode: str = "borda",
    hit_stats: Sequence[WindowHitStat] = (),
    comparison_ensemble: Optional[EnsembleResult] = None,
    comparison_shrinkage: Optional[ShrinkageReport] = None,
    comparison_fusion_mode: Optional[str] = None,
) -> str:
    """渲染控制台与 TXT 全景决策精要报告。"""
    line_sep = "=" * 80
    sub_sep = "-" * 80
    prev_draw = dataset.latest
    fusion_label = "Borda 秩融合" if fusion_mode == "borda" else "加权概率和"
    gold_str = " ".join(f"{x:02d}" for x in ensemble.gold_pool)
    main_str = " ".join(f"{x:02d}" for x in ensemble.main_pool)
    shrink_str = " ".join(f"{x:02d}" for x in shrinkage.pool)

    lines = [
        line_sep,
        "          快乐8 (Keno) 深度学习与图谱贝叶斯混合集成预测精要报告",
        f"                 预测目标期号: 第 {ensemble.target_period} 期  |  模型状态: 训练收敛 (Loss: {ensemble.training_loss:.4f})  |  融合: {fusion_label}",
        line_sep,
        "【段落 0: 数据概况与时序基准】",
        sub_sep,
        f"  * 上期期号: 第 {prev_draw.period} 期 ({prev_draw.date})",
        f"  * 上期开奖: {' '.join(f'{x:02d}' for x in prev_draw.numbers)}",
        f"  * 历史样本库总深度: {dataset.total_draws} 期 (严格时序因果，杜绝前瞻泄露)",
        "",
    ]

    if review is not None:
        lines.append(format_review_console(review))
        lines.append("")

    if points_analysis is not None:
        lines.append(format_points_console(points_analysis))
        lines.append("")

    lines.extend([
        "【段落 3: 核心选拔梯队（金胆与精选大底）】",
        sub_sep,
        f"  * 🔥 核心超级金胆 ({len(ensemble.gold_pool)} 码): [ {gold_str} ]",
        f"  * ⭐ 精选复式大底 ({len(ensemble.main_pool)} 码): [ {main_str} ]",
        f"  * 金胆和值: {sum(ensemble.gold_pool)} | 奇偶比: {sum(1 for x in ensemble.gold_pool if x % 2 == 1)}:{len(ensemble.gold_pool) - sum(1 for x in ensemble.gold_pool if x % 2 == 1)}",
        "",
        "【段落 4: 80 码 FinalScore 综合排序 Top 20 梯队（Borda 融合得分）】",
        sub_sep,
        "  名次  号码   FinalScore   DL序列概率   图谱贝叶斯   当前遗漏   重号属性   数论质数",
    ])

    for rank in range(1, 21):
        num = ensemble.full_order[rank - 1]
        det = ensemble.details[num]
        rep_str = "★重号" if det.is_repeat else " 新号 "
        prm_str = "质数" if det.is_prime else "合数"
        lines.append(
            f"  {rank:02d}    [{num:02d}]     {det.final_score:.4f}       {det.dl_prob:.4f}       {det.spectral_score:.4f}       {det.omission:02d}期      {rep_str}     {prm_str}"
        )

    lines.extend([
        "",
        "【段落 5: 预测分模块明细（Top 5）】",
        sub_sep,
    ])
    module_scores: Mapping[str, Sequence[float]] = getattr(ensemble, "module_scores", {}) or {}
    for key, label in _MODULE_LABELS:
        if key in module_scores:
            lines.append(f"  * {label}: {_format_top_k(module_scores[key])}")
        else:
            lines.append(f"  * {label}: 未提供")

    lines.extend([
        "",
        "【段落 6: 实战推荐保底投注单清单】",
        sub_sep,
        f"  * 推荐方案: 选五(Pick 5)，预算 {len(tickets)} 注，总成本 {len(tickets) * COST_PER_TICKET_YUAN} 元",
    ])
    for idx, t in enumerate(tickets, start=1):
        lines.append(f"    第 {idx:02d} 注: [ {' '.join(f'{x:02d}' for x in t)} ]")

    lines.extend([
        "",
        "【段落 7: 30 码黄金大底级联精细缩水方案执行概况】",
        sub_sep,
        f"  * 30 码大底: {shrink_str}",
        f"  * 漏斗缩水: 初始 C(30,5)={shrinkage.raw_combinations_count:,} 注 -> 形态剪枝留存 {shrinkage.after_morphology_count:,} 注 -> 精选落地 {shrinkage.final_tickets_count} 注 ({shrinkage.total_cost_yuan}元)",
        f"  * 蒙特卡洛 5000 次审计保奖率 (出9码场景): 保5千元爆发率 {shrinkage.hit5_guarantee_prob:.2f}% | 保4率 {shrinkage.hit4_or_better_prob:.2f}% | 保3率 {shrinkage.hit3_or_better_prob:.2f}% | 期望奖金 {shrinkage.expected_prize_per_draw:.2f} 元",
    ])

    if hit_stats:
        lines.extend(["", format_hit_rate_console(hit_stats)])

    if comparison_ensemble is not None:
        lines.extend([
            "",
            format_comparison_console(
                ensemble,
                shrinkage,
                comparison_ensemble,
                comparison_shrinkage,
                official_mode=fusion_mode,
                comparison_mode=comparison_fusion_mode,
            ),
        ])

    lines.append(line_sep)
    return "\n".join(lines)


def generate_excel_workbook(
    ensemble: EnsembleResult,
    dataset: Dataset,
    shrinkage: ShrinkageReport,
    filepath: str,
    review: Optional[ReviewResult] = None,
    points_analysis: Optional[PointsAnalysis] = None,
    comparison_ensemble: Optional[EnsembleResult] = None,
    comparison_shrinkage: Optional[ShrinkageReport] = None,
    official_fusion_mode: str = "borda",
    comparison_fusion_mode: Optional[str] = None,
) -> None:
    """生成商业级 Excel 决策报表 (.xlsx)。

    包含 4 个基础 Sheet（决策推荐 / 80码全参矩阵 / 30码缩水实战票 / 开奖实测复盘），
    可选追加 2 个 Sheet：
    5. 昨日推荐复盘 (Review)
    6. 用户点位对比 (Points)
    """
    wb = openpyxl.Workbook()
    # 默认移除初始 sheet
    default_sheet = wb.active
    if default_sheet is not None:
        wb.remove(default_sheet)

    # 样式定义
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")  # 经典深蓝
    header_font = Font(name="Microsoft YaHei", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Microsoft YaHei", size=10)
    bold_font = Font(name="Microsoft YaHei", size=10, bold=True)
    highlight_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # 浅黄高亮
    gold_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")  # 浅橙高亮
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # =========================================================================
    # Sheet 1: 决策推荐精要
    # =========================================================================
    ws1 = wb.create_sheet(title="决策推荐")
    ws1.views.sheetView[0].showGridLines = True

    title_row = [f"快乐8第 {ensemble.target_period} 期 深度学习与图谱贝叶斯决策推荐"]
    ws1.append(title_row)
    ws1.merge_cells("A1:G1")
    ws1["A1"].font = Font(name="Microsoft YaHei", size=14, bold=True, color="1F497D")
    ws1["A1"].alignment = center_align

    ws1.append([])
    ws1.append(["推荐类别", "号码规模", "精选号码清单", "形态和值", "奇偶配比", "重号个数", "实战建议"])
    for col_idx in range(1, 8):
        cell = ws1.cell(row=3, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    gold_str = " ".join(f"{x:02d}" for x in ensemble.gold_pool)
    main_str = " ".join(f"{x:02d}" for x in ensemble.main_pool)
    prev_numbers = set(dataset.latest.numbers)
    gold_reps = len(set(ensemble.gold_pool) & prev_numbers)
    main_reps = len(set(ensemble.main_pool) & prev_numbers)

    row_gold = [
        "🔥 超级金胆",
        f"{len(ensemble.gold_pool)} 码",
        gold_str,
        sum(ensemble.gold_pool),
        f"{sum(1 for x in ensemble.gold_pool if x % 2 == 1)}:{len(ensemble.gold_pool) - sum(1 for x in ensemble.gold_pool if x % 2 == 1)}",
        gold_reps,
        "高信噪比核心胆码，适合选一/选二/作胆",
    ]
    row_main = [
        "⭐ 精选复式大底",
        f"{len(ensemble.main_pool)} 码",
        main_str,
        sum(ensemble.main_pool),
        f"{sum(1 for x in ensemble.main_pool if x % 2 == 1)}:{len(ensemble.main_pool) - sum(1 for x in ensemble.main_pool if x % 2 == 1)}",
        main_reps,
        "平衡重号与热号，适合常规小复式",
    ]
    row_shrink = [
        "💎 黄金大底",
        f"{len(shrinkage.pool)} 码",
        " ".join(f"{x:02d}" for x in shrinkage.pool),
        sum(shrinkage.pool),
        f"{sum(1 for x in shrinkage.pool if x % 2 == 1)}:{len(shrinkage.pool) - sum(1 for x in shrinkage.pool if x % 2 == 1)}",
        len(set(shrinkage.pool) & prev_numbers),
        "大纵深覆盖防守，供 20 注精细缩水使用",
    ]

    for r_data in [row_gold, row_main, row_shrink]:
        ws1.append(r_data)
        cur_row = ws1.max_row
        for c_idx in range(1, 8):
            cell = ws1.cell(row=cur_row, column=c_idx)
            cell.font = data_font
            cell.border = thin_border
            cell.alignment = left_align if c_idx in (3, 7) else center_align
            if cur_row == 4:
                cell.fill = gold_fill
            elif cur_row == 5:
                cell.fill = highlight_fill

    # =========================================================================
    # Sheet 2: 80 码全参矩阵
    # =========================================================================
    ws2 = wb.create_sheet(title="80码全参矩阵")
    ws2.views.sheetView[0].showGridLines = True

    headers_ws2 = [
        "综合排名", "号码", "综合得分(Final)", "Transformer深度概率",
        "图谱贝叶斯得分", "当前遗漏期数", "是否上期重号", "数论属性(质/合)", "四分区归属"
    ]
    ws2.append(headers_ws2)
    for col_idx in range(1, len(headers_ws2) + 1):
        cell = ws2.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    for rank in range(1, N_NUM + 1):
        num = ensemble.full_order[rank - 1]
        det = ensemble.details[num]
        r_vals = [
            rank,
            f"{num:02d}",
            round(det.final_score, 4),
            round(det.dl_prob, 4),
            round(det.spectral_score, 4),
            det.omission,
            "重号" if det.is_repeat else "新号",
            "质数" if det.is_prime else "合数",
            f"第{det.zone4 + 1}区",
        ]
        ws2.append(r_vals)
        c_row = ws2.max_row
        for c_idx in range(1, len(headers_ws2) + 1):
            c_cell = ws2.cell(row=c_row, column=c_idx)
            c_cell.font = data_font
            c_cell.border = thin_border
            c_cell.alignment = center_align
            if num in ensemble.gold_pool:
                c_cell.fill = gold_fill
            elif num in ensemble.main_pool:
                c_cell.fill = highlight_fill

    # =========================================================================
    # Sheet 3: 30码级联缩水实战票
    # =========================================================================
    ws3 = wb.create_sheet(title="30码缩水实战票")
    ws3.views.sheetView[0].showGridLines = True

    ws3.append(["注单编号", "玩法", "选号1", "选号2", "选号3", "选号4", "选号5", "注单和值", "奇偶比", "单注成本(元)"])
    for col_idx in range(1, 11):
        cell = ws3.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    for idx, t in enumerate(shrinkage.tickets, start=1):
        odds = sum(1 for x in t if x % 2 == 1)
        r_t = [
            f"第 {idx:02d} 注",
            "选五",
            f"{t[0]:02d}", f"{t[1]:02d}", f"{t[2]:02d}", f"{t[3]:02d}", f"{t[4]:02d}",
            sum(t),
            f"{odds}:{5 - odds}",
            COST_PER_TICKET_YUAN,
        ]
        ws3.append(r_t)
        c_row = ws3.max_row
        for c_idx in range(1, 11):
            c_cell = ws3.cell(row=c_row, column=c_idx)
            c_cell.font = data_font
            c_cell.border = thin_border
            c_cell.alignment = center_align

    # =========================================================================
    # Sheet 4: 开奖实测复盘表 (Recent Draws Review)
    # =========================================================================
    ws4 = wb.create_sheet(title="开奖实测复盘")
    ws4.views.sheetView[0].showGridLines = True

    from kl8_engine.constants import is_prime

    headers_ws4 = ["期号", "开奖日期", "开奖号码 (20码全量)", "当期和值", "奇偶配比", "质合配比", "较前期重号数"]
    ws4.append(headers_ws4)
    for col_idx in range(1, len(headers_ws4) + 1):
        cell = ws4.cell(row=1, column=col_idx)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    recent_slice = list(dataset.draws[-20:])
    for idx in range(len(recent_slice) - 1, -1, -1):
        cur_draw = recent_slice[idx]
        cur_nums = cur_draw.numbers
        prev_nums_set = set(recent_slice[idx - 1].numbers) if idx > 0 else set()
        rep_count = len(set(cur_nums) & prev_nums_set) if idx > 0 else "-"
        odd_cnt = sum(1 for x in cur_nums if x % 2 == 1)
        prime_cnt = sum(1 for x in cur_nums if is_prime(x))

        r_draw = [
            f"第{cur_draw.period}期",
            cur_draw.date,
            " ".join(f"{x:02d}" for x in cur_nums),
            sum(cur_nums),
            f"{odd_cnt}:{20 - odd_cnt}",
            f"{prime_cnt}:{20 - prime_cnt}",
            rep_count,
        ]
        ws4.append(r_draw)
        cur_r = ws4.max_row
        for c_idx in range(1, len(headers_ws4) + 1):
            cell = ws4.cell(row=cur_r, column=c_idx)
            cell.font = data_font
            cell.border = thin_border
            cell.alignment = left_align if c_idx == 3 else center_align
            if idx == len(recent_slice) - 1:
                # 最新一期开奖高亮
                cell.fill = highlight_fill

    sheets = [ws1, ws2, ws3, ws4]

    # =========================================================================
    # Sheet 5: 昨日推荐复盘（可选）
    # =========================================================================
    if review is not None:
        ws5 = wb.create_sheet(title="昨日推荐复盘")
        ws5.views.sheetView[0].showGridLines = True
        headers_ws5 = ["复盘指标", "命中数", "池规模", "理论期望", "Lift", "说明"]
        ws5.append(headers_ws5)
        for col_idx in range(1, len(headers_ws5) + 1):
            cell = ws5.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align

        review_rows = [
            ["超级金胆", review.gold_hits, review.gold_total, 1.0, round(review.gold_lift, 2), ""],
            ["精选复式大底", review.main_hits, review.main_total, 3.0, round(review.main_lift, 2), ""],
            ["Top20", review.top20_hits, review.top20_total, 5.0,
             round(review.top20_lift, 2) if review.top20_lift is not None else "-", ""],
            ["30码黄金大底", review.pool30_hits, review.pool30_total, 7.5,
             round(review.pool30_lift, 2) if review.pool30_lift is not None else "-", ""],
            ["推荐票", review.best_ticket_hits, 5, "-", "-",
             f"投入 {review.total_cost_yuan} 元 / 回报 {review.total_prize_yuan} 元 / 盈亏 {review.net_profit_yuan:+d} 元"],
        ]
        for r_data in review_rows:
            ws5.append(r_data)
            cur_r = ws5.max_row
            for c_idx in range(1, len(headers_ws5) + 1):
                cell = ws5.cell(row=cur_r, column=c_idx)
                cell.font = data_font
                cell.border = thin_border
                cell.alignment = left_align if c_idx == 6 else center_align

        ws5.append([])
        ws5.append([f"复盘对象: 第 {review.target_period} 期推荐 | 实际开奖: 第 {review.actual_period} 期 ({review.actual_date})"])
        ws5.append(["开奖号码: " + " ".join(f"{x:02d}" for x in review.actual_numbers)])
        sheets.append(ws5)

    # =========================================================================
    # Sheet 6: 用户点位对比（可选）
    # =========================================================================
    if points_analysis is not None:
        ws6 = wb.create_sheet(title="用户点位对比")
        ws6.views.sheetView[0].showGridLines = True
        headers_ws6 = ["点位号码", "模型排名", "综合得分", "金胆", "主选", "30码大底", "与上期重号", "共识码(Top20)", "逆势码(排名>40)"]
        ws6.append(headers_ws6)
        for col_idx in range(1, len(headers_ws6) + 1):
            cell = ws6.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align

        for num in points_analysis.points:
            rank = points_analysis.ranks.get(num)
            score = points_analysis.scores.get(num)
            row = [
                f"{num:02d}",
                rank if rank is not None else "-",
                round(score, 4) if score is not None else "-",
                "是" if num in points_analysis.overlap_gold else "否",
                "是" if num in points_analysis.overlap_main else "否",
                "是" if num in points_analysis.overlap_pool30 else "否",
                "是" if num in points_analysis.repeat_with_latest else "否",
                "是" if num in points_analysis.consensus_top20 else "否",
                "是" if num in points_analysis.contrarian else "否",
            ]
            ws6.append(row)
            cur_r = ws6.max_row
            for c_idx in range(1, len(headers_ws6) + 1):
                cell = ws6.cell(row=cur_r, column=c_idx)
                cell.font = data_font
                cell.border = thin_border
                cell.alignment = center_align
                if num in points_analysis.overlap_gold:
                    cell.fill = gold_fill
                elif num in points_analysis.consensus_top20:
                    cell.fill = highlight_fill
        sheets.append(ws6)

    # =========================================================================
    # Sheet 7: 双融合并排对比（可选）
    # =========================================================================
    if comparison_ensemble is not None:
        ws7 = wb.create_sheet(title="融合并排对比")
        ws7.views.sheetView[0].showGridLines = True
        headers_ws7 = ["项目", "正式版", "实验版", "重合/差异"]
        ws7.append(headers_ws7)
        for col_idx in range(1, len(headers_ws7) + 1):
            cell = ws7.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align

        labels = {"borda": "Borda 秩融合", "weighted": "加权概率和"}
        gold_overlap = len(set(ensemble.gold_pool) & set(comparison_ensemble.gold_pool))
        main_overlap = len(set(ensemble.main_pool) & set(comparison_ensemble.main_pool))
        top20_overlap = len(
            set(ensemble.full_order[:20]) & set(comparison_ensemble.full_order[:20])
        )
        pool30_overlap = len(
            set(ensemble.shrinkage_pool_30) & set(comparison_ensemble.shrinkage_pool_30)
        )

        rows_ws7 = [
            [
                "融合方式",
                labels.get(official_fusion_mode, official_fusion_mode),
                labels.get(comparison_fusion_mode or "", comparison_fusion_mode or "-"),
                "-",
            ],
            [
                "超级金胆 (4)",
                " ".join(f"{x:02d}" for x in ensemble.gold_pool),
                " ".join(f"{x:02d}" for x in comparison_ensemble.gold_pool),
                f"{gold_overlap}/4",
            ],
            [
                "精选大底 (12)",
                " ".join(f"{x:02d}" for x in ensemble.main_pool),
                " ".join(f"{x:02d}" for x in comparison_ensemble.main_pool),
                f"{main_overlap}/12",
            ],
            [
                "Top20",
                " ".join(f"{x:02d}" for x in ensemble.full_order[:20]),
                " ".join(f"{x:02d}" for x in comparison_ensemble.full_order[:20]),
                f"{top20_overlap}/20",
            ],
            [
                "30码大底",
                " ".join(f"{x:02d}" for x in ensemble.shrinkage_pool_30),
                " ".join(f"{x:02d}" for x in comparison_ensemble.shrinkage_pool_30),
                f"{pool30_overlap}/30",
            ],
        ]
        if comparison_shrinkage is not None:
            rows_ws7.extend([
                [
                    "缩水留存注数",
                    f"{shrinkage.after_morphology_count:,}",
                    f"{comparison_shrinkage.after_morphology_count:,}",
                    "-",
                ],
                [
                    "精选注数 / 成本",
                    f"{shrinkage.final_tickets_count} 注 / {shrinkage.total_cost_yuan} 元",
                    f"{comparison_shrinkage.final_tickets_count} 注 / {comparison_shrinkage.total_cost_yuan} 元",
                    "-",
                ],
                [
                    "保4率",
                    f"{shrinkage.hit4_or_better_prob:.2f}%",
                    f"{comparison_shrinkage.hit4_or_better_prob:.2f}%",
                    "-",
                ],
                [
                    "期望奖金",
                    f"{shrinkage.expected_prize_per_draw:.2f} 元",
                    f"{comparison_shrinkage.expected_prize_per_draw:.2f} 元",
                    "-",
                ],
            ])
        for r_data in rows_ws7:
            ws7.append(r_data)
            cur_r = ws7.max_row
            for c_idx in range(1, len(headers_ws7) + 1):
                cell = ws7.cell(row=cur_r, column=c_idx)
                cell.font = data_font
                cell.border = thin_border
                cell.alignment = left_align if c_idx in (2, 3) else center_align
        sheets.append(ws7)

    # 自动调整列宽
    for ws in sheets:
        for col in ws.columns:
            max_len = 0
            for cell in col:
                val_str = str(cell.value or "")
                max_len = max(max_len, len(val_str.encode("utf-8")))
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 11)

    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    wb.save(filepath)
