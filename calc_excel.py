# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— Excel 报表导出模块

【业务背景与架构定位】:
本模块负责生成符合商业交付标准的专业 Excel 预测报表 (.xlsx)：
1. Sheet 1: 【八区分区预测】—— 八区分桶明细、追热龙头与防冷反弹、012路与质合属性、16码大底及选号建议；
2. Sheet 2: 【80码动力学全参】—— 全量 80 码的密度积分、一阶速度、二阶加速度、遗漏期数、动能分、势能分；
3. Sheet 3: 【上期开奖复盘】—— 真实开奖20码明细、红底白字高亮命中、八区逐区命中核算。

@author czw
@since 2026-10-19
"""
from __future__ import annotations

from pathlib import Path
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from calc_core import (
    EXCEL_DIR,
    Dataset,
    TimeSeries,
    analyze_number_group,
    get_road_012,
    truncated,
)
from calc_engine import run_calculus_engine


def _create_styles():
    """定义一套工业级配色与字体样式。"""
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")  # 经典深蓝
    header_font = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")

    sub_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")  # 浅蓝表头
    sub_font = Font(name="微软雅黑", size=10, bold=True, color="1F497D")

    hit_fill = PatternFill(start_color="C00000", end_color="C00000", fill_type="solid")  # 中奖红
    hit_font = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")

    pick_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # 精选浅黄
    pick_font = Font(name="微软雅黑", size=10, bold=True, color="B25900")

    regular_font = Font(name="微软雅黑", size=10)
    title_font = Font(name="微软雅黑", size=14, bold=True, color="1F497D")

    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    right_align = Alignment(horizontal="right", vertical="center")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    header_border = Border(
        left=Side(style="thin", color="FFFFFF"),
        right=Side(style="thin", color="FFFFFF"),
        top=Side(style="medium", color="1F497D"),
        bottom=Side(style="medium", color="1F497D"),
    )

    return {
        "header_fill": header_fill,
        "header_font": header_font,
        "sub_fill": sub_fill,
        "sub_font": sub_font,
        "hit_fill": hit_fill,
        "hit_font": hit_font,
        "pick_fill": pick_fill,
        "pick_font": pick_font,
        "regular_font": regular_font,
        "title_font": title_font,
        "center_align": center_align,
        "left_align": left_align,
        "right_align": right_align,
        "thin_border": thin_border,
        "header_border": header_border,
    }


def generate_excel_report(
    ds: Dataset,
    engine_res: dict,
    target_period: int,
    target_date: str,
    output_dir: str | Path = EXCEL_DIR,
    backtest_result: dict | None = None,
    sweep: list[dict] | None = None,
) -> Path:
    """生成专业的快乐8微积分分区预测 Excel 工作簿。

    @param ds 当前已加载的数据集
    @param engine_res 当期动力学引擎核算结果
    @param target_period 目标期号
    @param target_date 目标日期
    @param output_dir 输出目录
    @param backtest_result 可选：走查回测结果 (生成「命中率显著性检验」页)
    @param sweep 可选：参数寻优扫描结果 (写入检验页底部)
    @return 生成的 Excel 文件绝对路径
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    clean_date = target_date.replace("-", "")
    file_path = out_path / f"快乐8微积分八区预测_{clean_date}_第{target_period}期.xlsx"

    wb = openpyxl.Workbook()
    styles = _create_styles()

    # -------------------------------------------------------------
    # Sheet 1: 【八区分区预测】
    # -------------------------------------------------------------
    ws1 = wb.active
    ws1.title = "当期八区分区预测"
    ws1.views.sheetView[0].showGridLines = True

    # 标题区
    ws1.merge_cells("A1:I1")
    ws1["A1"] = f"快乐8「微积分八区分区预测系统」· 每日实战推荐表 (第 {target_period} 期 · {target_date})"
    ws1["A1"].font = styles["title_font"]
    ws1["A1"].alignment = styles["center_align"]
    ws1.row_dimensions[1].height = 36

    # 基础信息栏
    last_d = ds.draws[-1]
    ws1.merge_cells("A2:I2")
    ws1["A2"] = (
        f"开奖历史基准: 第 {last_d.period} 期 ({last_d.date}) ｜ "
        f"样本深度: T={ds.T} 期 ｜ 算法: 纯微积分动力学 + 012路区内正交防撞车 + 三态温号梯队"
    )
    ws1["A2"].font = Font(name="微软雅黑", size=9, italic=True, color="595959")
    ws1["A2"].alignment = styles["center_align"]
    ws1.row_dimensions[2].height = 20

    # 表头
    headers_s1 = [
        "分区编号", "号码范围", "走势研判", "预估出号",
        "追热龙头码 (热)", "中继温号码 (温)", "防冷反弹码 (冷)",
        "区内发散度与跨路", "战术运作指引"
    ]
    for col_idx, h in enumerate(headers_s1, 1):
        cell = ws1.cell(row=3, column=col_idx, value=h)
        cell.fill = styles["header_fill"]
        cell.font = styles["header_font"]
        cell.alignment = styles["center_align"]
        cell.border = styles["header_border"]
    ws1.row_dimensions[3].height = 26

    # 数据行
    zones = engine_res["zones"]
    for r_idx, z in enumerate(zones, 4):
        t0 = z["top_k_item"]
        tw = z.get("top_w_item", t0)
        t1 = z["top_p_item"]
        r0 = t0.get("road", get_road_012(t0["num"]))
        r1 = t1.get("road", get_road_012(t1["num"]))
        road_status = f"{z.get('dispersion_label', '正常发散')} [{r0}+{r1}路]"

        row_data = [
            f"第 {z['zone_id']} 区",
            f"{z['lo']:02d} - {z['hi']:02d}",
            z.get("z_trend", "⚖️ 平稳常态"),
            z.get("z_exp_range", "2~3码"),
            f"{t0['num']:02d} ({t0['prop_tag']})",
            f"{tw['num']:02d} ({tw['prop_tag']})",
            f"{t1['num']:02d} ({t1['prop_tag']})",
            road_status,
            z.get("z_tactical_adv", "标准配置"),
        ]
        for col_idx, val in enumerate(row_data, 1):
            cell = ws1.cell(row=r_idx, column=col_idx, value=val)
            cell.font = styles["regular_font"]
            cell.alignment = styles["center_align"] if col_idx != 9 else styles["left_align"]
            cell.border = styles["thin_border"]
            if col_idx in (5, 7):
                cell.fill = styles["pick_fill"]
                cell.font = styles["pick_font"]
            elif col_idx == 6:
                cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")  # 温和浅绿
                cell.font = Font(name="微软雅黑", size=10, bold=True, color="375623")
        ws1.row_dimensions[r_idx].height = 24

    # 实战分级推荐区
    bankers3 = engine_res.get("bankers3") or []
    pick5 = engine_res.get("pick5") or []
    pick7 = engine_res.get("pick7") or []
    pick10 = engine_res.get("pick10") or []
    hot_zones = engine_res.get("hot_zones") or []
    dyn_picks = sorted(engine_res.get("dynamic_picks") or [])
    picks16 = sorted(engine_res["picks16"])
    picks24 = sorted(engine_res.get("picks24") or [])

    curr_row = len(zones) + 5

    # 1. 金银铜胆 + 选五 + 选七 + 选十精选
    ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
    ws1.cell(row=curr_row, column=1, value="💎【实战分级高胜率投注推荐】(多模态时序非线性与拓扑优选):").font = Font(name="微软雅黑", size=11, bold=True, color="1F497D")
    ws1.cell(row=curr_row, column=1).alignment = styles["left_align"]
    curr_row += 1

    tier_rows = [
        ("👑 金银铜胆 (3码·核心定胆)", "   ".join(f"{n:02d}" for n in bankers3), "C00000", 12),
        ("🎯 选五精选 (5码·高密度实投)", "   ".join(f"{n:02d}" for n in pick5), "B25900", 11),
        ("🎯 选七精选 (7码·进阶实投)", "   ".join(f"{n:02d}" for n in pick7), "1F497D", 11),
        ("🏆 选十主力 (10码·主力实投)", "   ".join(f"{n:02d}" for n in pick10), "C00000", 11.5),
    ]
    for title, nums_s, color_code, fsize in tier_rows:
        ws1.cell(row=curr_row, column=1, value=title).font = Font(name="微软雅黑", size=10, bold=True, color="333333")
        ws1.cell(row=curr_row, column=1).alignment = styles["left_align"]
        ws1.merge_cells(start_row=curr_row, start_column=2, end_row=curr_row, end_column=9)
        c = ws1.cell(row=curr_row, column=2, value=nums_s)
        c.font = Font(name="Consolas", size=fsize, bold=True, color=color_code)
        c.alignment = styles["left_align"]
        c.fill = PatternFill(start_color="F9FBFD", end_color="F9FBFD", fill_type="solid")
        ws1.row_dimensions[curr_row].height = 24
        curr_row += 1

    # 2. 主攻放量双区
    if hot_zones:
        ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
        ws1.cell(row=curr_row, column=1, value="🚀【主攻放量双区 · 重兵突击】(动能爆发区饱和攻击):").font = Font(name="微软雅黑", size=10.5, bold=True, color="375623")
        ws1.cell(row=curr_row, column=1).alignment = styles["left_align"]
        curr_row += 1
        for hz in hot_zones:
            hz_title = f"· {hz['label']} [{hz['trend']} · 上期{hz['c_last']}码]"
            hz_picks_s = "   ".join(f"{n:02d}" for n in hz["picks"])
            ws1.cell(row=curr_row, column=1, value=hz_title).font = Font(name="微软雅黑", size=9.5, bold=True, color="375623")
            ws1.merge_cells(start_row=curr_row, start_column=2, end_row=curr_row, end_column=9)
            c = ws1.cell(row=curr_row, column=2, value=hz_picks_s)
            c.font = Font(name="Consolas", size=11, bold=True, color="1F497D")
            c.alignment = styles["left_align"]
            ws1.row_dimensions[curr_row].height = 22
            curr_row += 1

    # 3. 动态自适应配额推荐大名单 (17码)
    ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
    ws1.cell(row=curr_row, column=1, value=f"★ 打法 A【动态自适应大名单】({len(dyn_picks)} 码 · 升序排列):").font = Font(name="微软雅黑", size=11, bold=True, color="1F497D")
    ws1.cell(row=curr_row, column=1).alignment = styles["left_align"]
    curr_row += 1

    ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
    cell_dyn = ws1.cell(row=curr_row, column=1, value="   ".join(f"{n:02d}" for n in dyn_picks))
    cell_dyn.font = Font(name="Consolas", size=13, bold=True, color="C00000")
    cell_dyn.alignment = styles["center_align"]
    cell_dyn.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    ws1.row_dimensions[curr_row].height = 28
    curr_row += 1

    # 4. 16 码主力基准大名单
    ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
    ws1.cell(row=curr_row, column=1, value="★ 本期八区核心【精选 16 码主力推荐大名单】(升序 · 全局多模态优选):").font = Font(name="微软雅黑", size=10.5, bold=True, color="1F497D")
    ws1.cell(row=curr_row, column=1).alignment = styles["left_align"]
    curr_row += 1

    ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
    cell_p = ws1.cell(row=curr_row, column=1, value="   ".join(f"{n:02d}" for n in picks16))
    cell_p.font = Font(name="Consolas", size=12, bold=True, color="1F497D")
    cell_p.alignment = styles["center_align"]
    cell_p.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    ws1.row_dimensions[curr_row].height = 26
    curr_row += 1

    # 5. 24 码豪华全景大底
    ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
    ws1.cell(row=curr_row, column=1, value="★ 本期八区全景【24 码豪华大底名单】(升序 · 防踏空全覆盖):").font = Font(name="微软雅黑", size=10.5, bold=True, color="375623")
    ws1.cell(row=curr_row, column=1).alignment = styles["left_align"]
    curr_row += 1

    ws1.merge_cells(start_row=curr_row, start_column=1, end_row=curr_row, end_column=9)
    cell_w = ws1.cell(row=curr_row, column=1, value="   ".join(f"{n:02d}" for n in picks24))
    cell_w.font = Font(name="Consolas", size=11, bold=True, color="375623")
    cell_w.alignment = styles["center_align"]
    cell_w.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    ws1.row_dimensions[curr_row].height = 26
    curr_row += 1

    # 形态体检
    macro = engine_res.get("macro_analysis_16") or analyze_number_group(picks16)
    row_m = curr_row + 1
    ws1.merge_cells(start_row=row_m, start_column=1, end_row=row_m, end_column=9)
    ws1.cell(row=row_m, column=1, value="【16码宏观形态体检】:").font = Font(name="微软雅黑", size=10, bold=True, color="1F497D")

    health_info = [
        f"· 012路分布比: 0路 {macro['road_counts'][0]}码 ｜ 1路 {macro['road_counts'][1]}码 ｜ 2路 {macro['road_counts'][2]}码 (比例 {macro['road_ratio']})",
        f"· 质数合数分布: 质数 {macro['prime_count']}码 ｜ 合数 {macro['composite_count']}码 (比例 {macro['prime_ratio']})",
        f"· 大小与奇偶比: 大小比 {macro['size_ratio']} ｜ 奇偶比 {macro['odd_ratio']} ｜ 和值 {macro['sum_value']} ｜ 跨度 {macro['span']}",
        "· 实战玩法指引: 稳健型选五/选七精选；大底型以自适应17码为主力、24码为后备池；突击型直击主攻放量双区。",
    ]
    for idx, h_line in enumerate(health_info, 1):
        r = row_m + idx
        ws1.merge_cells(start_row=r, start_column=1, end_row=r, end_column=9)
        c = ws1.cell(row=r, column=1, value=h_line)
        c.font = Font(name="微软雅黑", size=9.5, color="333333")
        c.alignment = styles["left_align"]
        ws1.row_dimensions[r].height = 20

    # -------------------------------------------------------------
    # Sheet 2: 【80码动力学与多模态全参】
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="80码动力学与多模态全参")
    ws2.views.sheetView[0].showGridLines = True

    headers_s2 = [
        "号码", "所属分区", "012路", "质合属性", "属性标签",
        "多模态综合分", "LightGBM概率", "图网络中心度",
        "动能冲顶分k", "势能反弹分p", "密度积分∫", "速度导数v", "加速度a", "已遗漏期数", "精选大名单"
    ]
    for col_idx, h in enumerate(headers_s2, 1):
        cell = ws2.cell(row=1, column=col_idx, value=h)
        cell.fill = styles["header_fill"]
        cell.font = styles["header_font"]
        cell.alignment = styles["center_align"]
        cell.border = styles["header_border"]
    ws2.row_dimensions[1].height = 26

    picks_set = set(picks16)
    dyn_set = set(engine_res.get("dynamic_picks") or [])
    picks24_set = set(picks24)
    fusion_scores = engine_res.get("fusion_scores")
    lgb_probs = engine_res.get("lgb_probs")
    graph_scores = engine_res.get("graph_scores")

    row_cnt = 2
    for z in zones:
        for item in z["ranked"]:
            n = item["num"]
            n_idx = n - 1
            is_picked = n in picks_set
            is_dyn = n in dyn_set
            is_24 = n in picks24_set

            status_label = "——"
            if is_dyn:
                status_label = "★ 打法A动态自适应"
            elif is_picked:
                status_label = "★ 16码主力"
            elif is_24:
                status_label = "24码豪华大底"

            f_score = round(float(fusion_scores[n_idx]), 3) if fusion_scores is not None else round(item["score"], 3)
            lgb_p = round(float(lgb_probs[n_idx]), 3) if lgb_probs is not None else 0.250
            g_score = round(float(graph_scores[n_idx]), 3) if graph_scores is not None else 0.000

            row_data = [
                f"{n:02d}",
                f"第 {z['zone_id']} 区 ({z['lo']:02d}-{z['hi']:02d})",
                f"{item['road']}路",
                item["prime_label"],
                item["prop_tag"],
                f_score,
                lgb_p,
                g_score,
                round(item["k_score"], 2),
                round(item["p_score"], 2),
                round(item["integral"], 2),
                round(item["velocity"], 2),
                round(item["accel"], 2),
                int(item["gap"]),
                status_label,
            ]
            for col_idx, val in enumerate(row_data, 1):
                cell = ws2.cell(row=row_cnt, column=col_idx, value=val)
                cell.font = styles["regular_font"]
                cell.alignment = styles["center_align"]
                cell.border = styles["thin_border"]
                if is_dyn or is_picked:
                    cell.fill = styles["pick_fill"]
                    if col_idx in (1, 6, 15):
                        cell.font = styles["pick_font"]
            ws2.row_dimensions[row_cnt].height = 20
            row_cnt += 1

    # -------------------------------------------------------------
    # Sheet 3: 【上期开奖与实测复盘】
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="上期开奖与实测复盘")
    ws3.views.sheetView[0].showGridLines = True

    if ds.T >= 2:
        prev = ds.draws[-1]
        prev_nums = sorted(prev.numbers)
        # 针对前一期回溯预测
        tds = truncated(ds, prev.period)
        ts_p = TimeSeries(tds)
        res_p = run_calculus_engine(tds, ts_p)
        prev_picks16 = set(res_p["picks16"])
        hit_nums = sorted(prev_picks16 & prev.numbers)

        ws3.merge_cells("A1:F1")
        ws3["A1"] = f"第 {prev.period} 期 ({prev.date}) 真实开奖全景与 16 码推荐命中复盘"
        ws3["A1"].font = styles["title_font"]
        ws3["A1"].alignment = styles["center_align"]
        ws3.row_dimensions[1].height = 32

        # 真实开奖 20 码矩阵
        ws3.merge_cells("A2:F2")
        ws3["A2"] = f"官方开奖号码 (20码): 成功命中 {len(hit_nums)} 码 (红底白字表示命中)"
        ws3["A2"].font = Font(name="微软雅黑", size=10, bold=True, color="1F497D")

        # 摆放20个开奖号
        for idx, num in enumerate(prev_nums):
            r = 3 + (idx // 10)
            c = (idx % 10) + 1
            cell = ws3.cell(row=r, column=c, value=f"{num:02d}")
            cell.alignment = styles["center_align"]
            cell.border = styles["thin_border"]
            if num in prev_picks16:
                cell.fill = styles["hit_fill"]
                cell.font = styles["hit_font"]
            else:
                cell.font = Font(name="Consolas", size=11, bold=True)
                cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
            ws3.row_dimensions[r].height = 26

        # 分区命中明细表
        row_z_head = 6
        headers_s3 = ["分区编号", "号码范围", "走势研判", "追热龙头", "中继温号", "防冷反弹", "动态配额", "命中号码清单", "命中状态"]
        for col_idx, h in enumerate(headers_s3, 1):
            cell = ws3.cell(row=row_z_head, column=col_idx, value=h)
            cell.fill = styles["header_fill"]
            cell.font = styles["header_font"]
            cell.alignment = styles["center_align"]
            cell.border = styles["header_border"]
        ws3.row_dimensions[row_z_head].height = 24

        prev_picks_dyn = res_p.get("dynamic_picks", prev_picks16)

        for r_offset, z in enumerate(res_p["zones"], 1):
            cur_r = row_z_head + r_offset
            q = z.get("z_quota", 2)
            z_dyn = z["top3"] if q >= 3 else (z["top2"] if q == 2 else [z["top_k_item"]["num"]])
            zh = sorted(set(z_dyn) & prev.numbers)
            zh_all = sorted(set(z["top3"]) & prev.numbers)
            t0 = z["top_k_item"]
            tw = z.get("top_w_item", t0)
            t1 = z["top_p_item"]
            trend_s = z.get("z_trend", "⚖️ 平稳常态").split()[0]
            quota_s = f"{q} 码"
            if zh:
                status_str = f"命中 {len(zh)} 码"
                hit_str = " ".join(f"{x:02d}" for x in zh)
            elif zh_all:
                status_str = "温码命中(大底)"
                hit_str = " ".join(f"{x:02d}" for x in zh_all)
            else:
                status_str = "未中"
                hit_str = "——"

            row_data = [
                f"第 {z['zone_id']} 区",
                f"{z['lo']:02d} - {z['hi']:02d}",
                trend_s,
                f"{t0['num']:02d}",
                f"{tw['num']:02d}",
                f"{t1['num']:02d}",
                quota_s,
                hit_str,
                status_str,
            ]
            for col_idx, val in enumerate(row_data, 1):
                cell = ws3.cell(row=cur_r, column=col_idx, value=val)
                cell.font = styles["regular_font"]
                cell.alignment = styles["center_align"]
                cell.border = styles["thin_border"]
                if (zh or zh_all) and col_idx in (8, 9):
                    cell.font = Font(name="微软雅黑", size=10, bold=True, color="C00000")
            ws3.row_dimensions[cur_r].height = 22

    # -------------------------------------------------------------
    # Sheet 4: 【命中率显著性检验】(仅在传入回测结果时生成)
    # -------------------------------------------------------------
    if backtest_result and backtest_result.get("n_sample", 0) > 0:
        bt = backtest_result
        base = bt["baseline"]
        ws4 = wb.create_sheet(title="命中率显著性检验")
        ws4.views.sheetView[0].showGridLines = True

        ws4.merge_cells("A1:D1")
        ws4["A1"] = f"最近 {bt['n_sample']} 期严格走查回测 —— 16 码命中 vs 纯随机基准"
        ws4["A1"].font = styles["title_font"]
        ws4["A1"].alignment = styles["center_align"]
        ws4.row_dimensions[1].height = 32

        ws4.merge_cells("A2:D2")
        ws4["A2"] = ("基准定义: 从 80 码中任取 16 码，每期开出 20 码，命中数服从超几何分布 "
                     "H(80,20,16)，其数学期望恒为 4.00 码 —— 这是评判任何算法的唯一公正标尺。")
        ws4["A2"].font = Font(name="微软雅黑", size=9, italic=True, color="595959")
        ws4["A2"].alignment = styles["left_align"]
        ws4.row_dimensions[2].height = 26

        kpi = [
            ("【打法A】动态配额实测平均命中", f"{bt.get('avg_hits_dyn', bt['avg_hits']):.3f} 码/期", "实测"),
            ("【打法A】动态配额 ≥5码胜率", f"{bt['ge_dyn'][5]['pct']:.1f}%" if "ge_dyn" in bt and 5 in bt["ge_dyn"] else "—", "实测"),
            ("【打法A】动态配额 ≥6码胜率", f"{bt['ge_dyn'][6]['pct']:.1f}%" if "ge_dyn" in bt and 6 in bt["ge_dyn"] else "—", "实测"),
            ("模型实测平均命中(基准16码)", f"{bt['avg_hits']:.3f} 码/期", "实测"),
            ("随机基准数学期望", f"{base['mean']:.3f} 码/期", "理论"),
            ("基准标准差 σ", f"{base['sd']:.3f} 码", "理论"),
            ("超额优势", f"{bt['avg_hits'] - base['mean']:+.3f} 码/期", "实测"),
            ("显著性 z 值", f"{bt['z_score']:+.2f}", "检验"),
            ("单侧 p 值", f"{bt['p_value']:.4f}", "检验"),
            ("评级", f"{bt['grade']} 级", "检验"),
            ("统计结论", bt["verdict"], "检验"),
            ("单期最高 / 最低", f"{bt['max_hits']} 码 / {bt['min_hits']} 码", "实测"),
            ("八区平均命中覆盖", f"{bt['avg_zone_coverage']:.2f} / 8 区", "实测"),
            ("八区覆盖随机基准", f"{base['zone_coverage_expected']:.2f} / 8 区", "理论"),
            ("热号龙头平均命中", f"{bt['avg_kinetic_hits']:.2f} / 8 码", "实测"),
            ("势能冷号平均命中", f"{bt['avg_potential_hits']:.2f} / 8 码", "实测"),
        ]
        r4 = 4
        for col_idx, h in enumerate(["指标", "数值", "性质"], 1):
            c = ws4.cell(row=r4, column=col_idx, value=h)
            c.fill = styles["header_fill"]
            c.font = styles["header_font"]
            c.alignment = styles["center_align"]
            c.border = styles["header_border"]
        for k, v, kind in kpi:
            r4 += 1
            for col_idx, val in enumerate([k, v, kind], 1):
                c = ws4.cell(row=r4, column=col_idx, value=val)
                c.font = styles["regular_font"]
                c.alignment = styles["center_align"] if col_idx != 2 else styles["left_align"]
                c.border = styles["thin_border"]
                if k in ("显著性 z 值", "评级", "统计结论"):
                    c.fill = styles["pick_fill"]
                    c.font = Font(name="微软雅黑", size=10, bold=True, color="C00000")

        r4 += 2
        ws4.cell(row=r4, column=1, value="命中门槛实测 vs 基准应有").font = Font(
            name="微软雅黑", size=11, bold=True, color="1F497D")
        r4 += 1
        for col_idx, h in enumerate(["门槛", "实测期数", "实测占比", "基准应有占比"], 1):
            c = ws4.cell(row=r4, column=col_idx, value=h)
            c.fill = styles["sub_fill"]
            c.font = styles["sub_font"]
            c.alignment = styles["center_align"]
            c.border = styles["thin_border"]
        for thr, v in sorted(bt["ge"].items()):
            r4 += 1
            for col_idx, val in enumerate([f"≥{thr} 码", v["count"], f"{v['pct']:.1f}%",
                                           f"{v['baseline_pct']:.1f}%"], 1):
                c = ws4.cell(row=r4, column=col_idx, value=val)
                c.font = styles["regular_font"]
                c.alignment = styles["center_align"]
                c.border = styles["thin_border"]

        if sweep:
            r4 += 2
            ws4.cell(row=r4, column=1, value="参数寻优扫描 (同窗口对比，用于判断调参是否真有效)").font = Font(
                name="微软雅黑", size=11, bold=True, color="1F497D")
            r4 += 1
            for col_idx, h in enumerate(["配置变体", "平均命中", "z 值", "≥5码占比"], 1):
                c = ws4.cell(row=r4, column=col_idx, value=h)
                c.fill = styles["sub_fill"]
                c.font = styles["sub_font"]
                c.alignment = styles["center_align"]
                c.border = styles["thin_border"]
            for v in sweep:
                r4 += 1
                for col_idx, val in enumerate([v["name"], f"{v['avg']:.3f}", f"{v['z']:+.2f}",
                                               f"{v['ge5_pct']:.1f}%"], 1):
                    c = ws4.cell(row=r4, column=col_idx, value=val)
                    c.font = styles["regular_font"]
                    c.alignment = styles["center_align"] if col_idx != 1 else styles["left_align"]
                    c.border = styles["thin_border"]

    # 自动调整列宽
    for ws in (ws1, ws2, ws3, *( [wb["命中率显著性检验"]] if "命中率显著性检验" in wb.sheetnames else [] )):
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                # 中文字符按双倍长度计算
                char_len = sum(2 if ord(c) > 127 else 1 for c in val)
                if char_len > max_len:
                    max_len = char_len
            ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 72)

    wb.save(file_path)
    return file_path
