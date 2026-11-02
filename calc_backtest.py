# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 走查式回测与命中率显著性检验模块

【业务背景与架构定位】:
本模块解决「命中率到底怎么样」这一唯一严肃问题。核心原则：
1. 严格防前瞻泄漏：预测第 t 期时，只允许使用第 t 期之前的开奖数据 (walk-forward)；
2. 公正基准：快乐8每期从 80 码开出 20 码，任何 16 码方案的命中数服从
   超几何分布 H(80,20,16)，均值恰为 4.00。因此「平均命中 4.17」本身
   **不构成任何优势证据**，必须做假设检验后才能下结论；
3. 显著性判定：给出 z 值与单侧 p 值，并在结论中明确标注「统计上与随机不可区分」
   还是「显著优于随机」，杜绝把正常波动包装成算法战绩。

@Author czw
@Since 2026-10-21
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from calc_core import (
    EXCEL_DIR,
    Dataset,
    TimeSeries,
    hypergeometric_baseline,
    truncated,
)
from calc_engine import run_calculus_engine


def walk_forward_backtest(ds: Dataset, n_test: int = 100) -> dict:
    """对最近 n_test 期做严格 walk-forward 回测，逐期重跑完整微积分引擎。

    @param ds 全量历史数据集
    @param n_test 回测期数
    @return 含逐期明细与显著性检验结论的结构化字典
    """
    total_t = ds.T
    n_sample = max(0, min(n_test, total_t - 40))
    if n_sample < 5:
        return {"n_sample": 0}

    baseline = hypergeometric_baseline(16)
    rows = []
    for i in range(total_t - n_sample, total_t):
        actual = ds.draws[i]
        sub_ds = truncated(ds, actual.period)
        res = run_calculus_engine(sub_ds, TimeSeries(sub_ds))
        picks = set(res["picks16"])
        dyn_picks = set(res.get("dynamic_picks") or res["picks16"])
        hit_nums = sorted(picks & actual.numbers)
        hit_dyn_nums = sorted(dyn_picks & actual.numbers)

        k_items = {z["top_k_item"]["num"] for z in res["zones"]}
        p_items = {z["top_p_item"]["num"] for z in res["zones"]}
        rows.append({
            "period": actual.period,
            "date": actual.date,
            "hits": len(hit_nums),
            "hit_nums": hit_nums,
            "picks16": sorted(picks),
            "hits_dyn": len(hit_dyn_nums),
            "hit_dyn_nums": hit_dyn_nums,
            "dynamic_picks": sorted(dyn_picks),
            "dyn_len": len(dyn_picks),
            "zone_coverage": sum(1 for z in res["zones"] if set(z["top2"]) & actual.numbers),
            "kinetic_hits": len(k_items & actual.numbers),
            "potential_hits": len(p_items & actual.numbers),
            "road_ratio": res["macro_analysis_16"]["road_ratio"],
            "prime_ratio": res["macro_analysis_16"]["prime_ratio"],
        })

    hits = np.array([r["hits"] for r in rows], dtype=float)
    hits_dyn = np.array([r["hits_dyn"] for r in rows], dtype=float)
    zcov = np.array([r["zone_coverage"] for r in rows], dtype=float)
    kin = np.array([r["kinetic_hits"] for r in rows], dtype=float)
    pot = np.array([r["potential_hits"] for r in rows], dtype=float)

    mean_hit = float(hits.mean())
    mean_hit_dyn = float(hits_dyn.mean())
    avg_dyn_len = float(np.mean([r["dyn_len"] for r in rows]))
    ge_dyn = {}
    for thr in (3, 4, 5, 6, 7):
        c_d = int((hits_dyn >= thr).sum())
        ge_dyn[thr] = {"count": c_d, "pct": c_d / float(n_sample) * 100.0}
    # 基准均值的抽样标准误 (命中数本身按超几何 σ 波动，n_sample 期取均值后收窄)
    se = baseline["sd"] / math.sqrt(n_sample)
    z_score = (mean_hit - baseline["mean"]) / se if se > 0 else 0.0
    # 单侧 p 值：H0「算法=随机」时观测到当前均值的概率 (正态近似)
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))

    dist = {k: int((hits == k).sum()) for k in sorted(set(int(h) for h in hits))}
    ge = {}
    for thr in (3, 4, 5, 6, 7):
        c = int((hits >= thr).sum())
        ge[thr] = {"count": c, "pct": c / float(n_sample) * 100.0,
                   "baseline_pct": baseline["p_ge"][thr] * 100.0}

    if z_score >= 2.58:
        verdict, grade = ("显著优于随机基准 (p<0.01)：存在可复核的统计学优势", "A")
    elif z_score >= 1.645:
        verdict, grade = ("边际优于随机基准 (p<0.05)：优势微弱，需更长样本继续验证", "B")
    elif z_score >= -1.645:
        verdict, grade = ("统计上与纯随机选号不可区分 (|z|<1.645)：当前均值差属正常抽样波动", "C")
    else:
        verdict, grade = ("显著劣于随机基准：模型存在系统性缺陷，需立即排查", "D")

    return {
        "n_sample": n_sample,
        "rows": rows,
        "baseline": baseline,
        "avg_hits": mean_hit,
        "avg_hits_dyn": mean_hit_dyn,
        "avg_dyn_len": avg_dyn_len,
        "sd_hits": float(hits.std(ddof=1)) if n_sample > 1 else 0.0,
        "max_hits": int(hits.max()),
        "min_hits": int(hits.min()),
        "distribution": dist,
        "ge": ge,
        "ge_dyn": ge_dyn,
        "avg_zone_coverage": float(zcov.mean()),
        "baseline_zone_coverage": baseline["zone_coverage_expected"],
        "avg_kinetic_hits": float(kin.mean()),
        "avg_potential_hits": float(pot.mean()),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "verdict": verdict,
        "grade": grade,
    }


def find_best_param_sweep(ds: Dataset, n_test: int = 100) -> list[dict]:
    """对可解释的结构参数做小范围走查，报告各变体的回测均值。

    仅用于「是否还有深度优化空间」的量化回答，不自动改写线上配置。

    @param ds 全量历史数据集
    @param n_test 每个变体的回测期数
    @return 按平均命中降序排列的变体结果列表
    """
    from calc_core import CONFIG

    variants = {
        "线上配置 (40/20/10窗)": {},
        "纯动能 (每区取2个最热)": {"combine_weight_kinetic": 1.0},
        "纯势能 (每区取2个最冷)": {"combine_weight_kinetic": 0.0},
        "关闭012路均值回归调制": {"road_elasticity_slope": 0.0},
        "关闭区内跨路防撞": {"road_orthogonal_enabled": False},
        "长窗 80/40/20": {"window_calc": 80, "window_calc_mid": 40, "window_calc_short": 20},
        "短窗 20/10/5": {"window_calc": 20, "window_calc_mid": 10, "window_calc_short": 5},
        "取消连失惩罚": {"penalty_score": 0.0},
    }

    out = []
    for name, override in variants.items():
        saved = dict(CONFIG)
        try:
            CONFIG.update(override)
            total_t = ds.T
            n_sample = max(0, min(n_test, total_t - 40))
            if n_sample < 5:
                continue
            hits = []
            for i in range(total_t - n_sample, total_t):
                actual = ds.draws[i]
                sub_ds = truncated(ds, actual.period)
                res = run_calculus_engine(sub_ds, TimeSeries(sub_ds))
                hits.append(len(set(res["picks16"]) & actual.numbers))
            h = np.array(hits, dtype=float)
            base = hypergeometric_baseline(16)
            se = base["sd"] / math.sqrt(n_sample)
            out.append({
                "name": name,
                "n": n_sample,
                "avg": float(h.mean()),
                "z": float((h.mean() - base["mean"]) / se),
                "ge5_pct": float((h >= 5).mean() * 100),
            })
        finally:
            CONFIG.clear()
            CONFIG.update(saved)

    out.sort(key=lambda r: -r["avg"])
    return out


def build_backtest_console_report(bt: dict) -> str:
    """把回测结果渲染成控制台美观文本。"""
    b = bt["baseline"]
    L = []
    sep = "═" * 78
    L.append(sep)
    L.append(f"  📈【最近 {bt['n_sample']} 期严格走查回测 (Walk-Forward · 零前瞻泄漏)】")
    L.append(sep)
    L.append(f"    基准定义: 从 80 码中随机选 16 码，命中数服从超几何分布 H(80,20,16)")
    L.append(f"             该基准数学期望 = {b['mean']:.2f} 码，标准差 σ = {b['sd']:.2f} 码")
    L.append("")
    L.append(f"    ▸ 模型实测平均命中   : {bt['avg_hits']:.3f} 码/期  (样本标准差 {bt['sd_hits']:.2f})")
    L.append(f"    ▸ 与随机基准之差     : {bt['avg_hits'] - b['mean']:+.3f} 码/期")
    L.append(f"    ▸ 显著性检验         : z = {bt['z_score']:+.2f}   单侧 p 值 = {bt['p_value']:.3f}")
    L.append(f"    ▸ 判定结论           : [{bt['grade']} 级] {bt['verdict']}")
    L.append("")
    L.append(f"    ▸ 单期最高 / 最低命中 : {bt['max_hits']} 码 / {bt['min_hits']} 码")
    L.append(f"    ▸ 八区平均命中覆盖   : {bt['avg_zone_coverage']:.2f} / 8 区  (随机基准 {bt['baseline_zone_coverage']:.2f})")
    L.append(f"    ▸ 热号龙头平均命中   : {bt['avg_kinetic_hits']:.2f} / 8 码   势能冷号平均命中: {bt['avg_potential_hits']:.2f} / 8 码")
    L.append("")
    if "avg_hits_dyn" in bt and "ge_dyn" in bt:
        dyn_avg = bt["avg_hits_dyn"]
        dyn_len = bt.get("avg_dyn_len", 17.0)
        ge5_dyn = bt["ge_dyn"].get(5, {}).get("pct", 0.0)
        ge6_dyn = bt["ge_dyn"].get(6, {}).get("pct", 0.0)
        ge5_base = bt["ge"].get(5, {}).get("pct", 0.0)
        ge6_base = bt["ge"].get(6, {}).get("pct", 0.0)
        L.append(f"    ⚡【打法 A：动态自适应配额模型 (最佳性价比)】(依时序走势调度 1~3 码 · 均 {dyn_len:.1f} 码):")
        L.append(f"      · 动态模型平均单期命中 : {dyn_avg:.3f} 码/期 (对比静态基准 16 码净提升 {dyn_avg - bt['avg_hits']:+.2f} 码)")
        L.append(f"      · 命中 ≥4 码胜率       : {bt['ge_dyn'].get(4, {}).get('pct', 0.0):.1f}%")
        L.append(f"      · 命中 ≥5 码高倍胜率   : {ge5_dyn:.1f}% (基准 16 码为 {ge5_base:.1f}%，净提升 {ge5_dyn - ge5_base:+.1f}%)")
        L.append(f"      · 命中 ≥6 码爆发胜率   : {ge6_dyn:.1f}% (基准 16 码为 {ge6_base:.1f}%，净提升 {ge6_dyn - ge6_base:+.1f}%)")
        L.append("")
    L.append("    命中门槛达成情况 (静态基准 16 码实测 vs 随机基准应有概率):")
    for thr, v in sorted(bt["ge"].items()):
        diff = v["pct"] - v["baseline_pct"]
        bar_len = int(round(v["pct"] / 5.0))
        bar = "█" * bar_len
        L.append(f"      命中 ≥{thr} 码: {v['pct']:5.1f}% ({v['count']:3d}/{bt['n_sample']}期)"
                 f"  基准应有 {v['baseline_pct']:5.1f}%  差值 {diff:+5.1f}pp  |{bar}")
    L.append("")
    L.append("    逐期命中分布直方图 (实测期数 vs 随机基准应有期数):")
    for k in range(0, 10):
        exp_cnt = b["pmf"][k] * bt["n_sample"]
        act_cnt = bt["distribution"].get(k, 0)
        if act_cnt == 0 and exp_cnt < 0.5:
            continue
        bar = "▇" * int(round(act_cnt / 2.0)) if act_cnt else ""
        L.append(f"      命中 {k:2d} 码: 实测 {act_cnt:3d} 期 ｜ 基准应有 {exp_cnt:5.1f} 期  {bar}")
    L.append(sep)
    return "\n".join(L)


def save_backtest_excel(bt: dict, output_dir: str | Path = EXCEL_DIR) -> Path:
    """导出逐期回测明细 Excel (命中高亮 + 汇总看板)。"""
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"快乐8微积分_命中率走查回测_{bt['n_sample']}期.xlsx"

    wb = openpyxl.Workbook()

    deep_header_fill = PatternFill("solid", start_color="1F497D")
    deep_header_font = Font(name="微软雅黑", size=11, bold=True, color="FFFFFF")
    red_font = Font(name="微软雅黑", size=10, bold=True, color="C00000")
    reg_font = Font(name="微软雅黑", size=10)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    thin = Border(*(Side(style="thin", color="D9D9D9"),) * 4)

    # ---- Sheet 1 汇总看板 ----
    ws = wb.active
    ws.title = "命中率看板"
    b = bt["baseline"]
    ws.merge_cells("A1:D1")
    ws["A1"] = f"快乐8·微积分八区 16 码推荐 —— 最近 {bt['n_sample']} 期走查回测看板"
    ws["A1"].font = Font(name="微软雅黑", size=14, bold=True, color="1F497D")
    ws["A1"].alignment = center
    ws.row_dimensions[1].height = 34

    board = [
        ("【打法A】动态配额模型平均命中", f"{bt.get('avg_hits_dyn', bt['avg_hits']):.3f} 码/期 (净提升 {bt.get('avg_hits_dyn', bt['avg_hits']) - bt['avg_hits']:+.3f} 码)" if "avg_hits_dyn" in bt else f"{bt['avg_hits']:.3f} 码/期"),
        ("【打法A】动态配额 ≥5码胜率", f"{bt['ge_dyn'][5]['pct']:.1f}% (基准 16 码 {bt['ge'][5]['pct']:.1f}%)" if "ge_dyn" in bt and 5 in bt["ge_dyn"] else "—"),
        ("【打法A】动态配额 ≥6码胜率", f"{bt['ge_dyn'][6]['pct']:.1f}% (基准 16 码 {bt['ge'][6]['pct']:.1f}%)" if "ge_dyn" in bt and 6 in bt["ge_dyn"] else "—"),
        ("静态 16 码基准平均命中", f"{bt['avg_hits']:.3f} 码/期"),
        ("纯随机选号基准期望", f"{b['mean']:.3f} 码/期 (超几何分布 H(80,20,16))"),
        ("基准标准差 σ", f"{b['sd']:.3f} 码"),
        ("超额优势", f"{bt['avg_hits'] - b['mean']:+.3f} 码/期"),
        ("显著性 z 值", f"{bt['z_score']:+.2f}"),
        ("单侧 p 值", f"{bt['p_value']:.4f}"),
        ("评级", f"{bt['grade']} 级"),
        ("统计结论", bt["verdict"]),
        ("单期最高 / 最低", f"{bt['max_hits']} 码 / {bt['min_hits']} 码"),
        ("八区平均命中覆盖", f"{bt['avg_zone_coverage']:.2f} / 8 区 (随机基准 {bt['baseline_zone_coverage']:.2f})"),
        ("热号龙头平均命中", f"{bt['avg_kinetic_hits']:.2f} / 8 码"),
        ("势能冷号平均命中", f"{bt['avg_potential_hits']:.2f} / 8 码"),
    ]
    r = 3
    for k, v in board:
        ws.cell(row=r, column=1, value=k).font = Font(name="微软雅黑", size=10, bold=True, color="1F497D")
        ws.cell(row=r, column=1).border = thin
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        c = ws.cell(row=r, column=2, value=v)
        c.font = red_font if k in ("显著性 z 值", "评级", "统计结论") else reg_font
        c.alignment = left
        c.border = thin
        ws.row_dimensions[r].height = 22
        r += 1

    r += 1
    ws.cell(row=r, column=1, value="命中门槛达成对比").font = Font(name="微软雅黑", size=11, bold=True, color="1F497D")
    r += 1
    for col_idx, h in enumerate(["门槛", "实测期数", "基准16码占比", "打法A动态占比", "随机应有占比"], 1):
        c = ws.cell(row=r, column=col_idx, value=h)
        c.fill = deep_header_fill
        c.font = deep_header_font
        c.alignment = center
        c.border = thin
    for thr, v in sorted(bt["ge"].items()):
        r += 1
        dyn_pct_str = f"{bt['ge_dyn'][thr]['pct']:.1f}%" if "ge_dyn" in bt and thr in bt["ge_dyn"] else "—"
        for col_idx, val in enumerate([f"≥{thr} 码", v["count"], f"{v['pct']:.1f}%", dyn_pct_str, f"{v['baseline_pct']:.1f}%"], 1):
            c = ws.cell(row=r, column=col_idx, value=val)
            c.font = reg_font
            c.alignment = center
            c.border = thin

    for col_idx in range(1, 6):
        ws.column_dimensions[get_column_letter(col_idx)].width = [22, 14, 18, 18, 22][col_idx - 1]

    # ---- Sheet 2 逐期明细 ----
    ws2 = wb.create_sheet("逐期回测明细")
    heads = ["期号", "开奖日期", "16码命中", "16命中号码", "【打法A】动态命中", "动态命中号码",
             "八区命中覆盖", "热号龙头命中", "势能冷号命中", "打法A动态名单", "16码名单", "012路比", "质合比"]
    for col_idx, h in enumerate(heads, 1):
        c = ws2.cell(row=1, column=col_idx, value=h)
        c.fill = deep_header_fill
        c.font = deep_header_font
        c.alignment = center
        c.border = thin
    ws2.row_dimensions[1].height = 26

    good_fill = PatternFill("solid", start_color="FFF2CC")
    for r_off, row in enumerate(bt["rows"], 2):
        vals = [
            row["period"], row["date"], row["hits"],
            " ".join(f"{n:02d}" for n in row["hit_nums"]) or "—",
            row.get("hits_dyn", row["hits"]),
            " ".join(f"{n:02d}" for n in row.get("hit_dyn_nums", [])) or "—",
            f"{row['zone_coverage']}/8",
            f"{row['kinetic_hits']}/8",
            f"{row['potential_hits']}/8",
            " ".join(f"{n:02d}" for n in row.get("dynamic_picks", [])),
            " ".join(f"{n:02d}" for n in row["picks16"]),
            row["road_ratio"], row["prime_ratio"],
        ]
        for col_idx, val in enumerate(vals, 1):
            c = ws2.cell(row=r_off, column=col_idx, value=val)
            c.font = reg_font
            c.alignment = center
            c.border = thin
            if (col_idx == 3 and row["hits"] >= 5) or (col_idx == 5 and row.get("hits_dyn", 0) >= 5):
                c.fill = good_fill
                c.font = red_font
    widths = [12, 13, 10, 20, 14, 20, 13, 13, 13, 56, 56, 10, 10]
    for col_idx, w in enumerate(widths, 1):
        ws2.column_dimensions[get_column_letter(col_idx)].width = w
    ws2.freeze_panes = "A2"

    wb.save(file_path)
    return file_path
