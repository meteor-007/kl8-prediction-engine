# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 核心动力学数学计算引擎

【业务背景与架构定位】:
本模块专为快乐8构建「纯开奖数据驱动」的离散微积分物理动力学引擎。
摒弃任何点位主观干扰，严格根据历史开奖的时间滑动矩阵计算：
1. 密度积分 (Integral): 短窗 10 期、中窗 20 期、长窗 40 期出号频率；
2. 一阶速度导数 (Velocity): v = s_s - s_l (动量变化率)；
3. 二阶加速度导数 (Acceleration): a = (s_s - s_m) - (s_m - s_l) (曲率加剧度)；
4. 弹性势能回补 (Potential Elasticity): 适度遗漏号的恢复引力；
5. 双态对冲决策 (Dual-State Decision):
   每个分区选出：1 码「动能顺势龙头」 + 1 码「势能反弹拐点」，共 16 码。

@author czw
@since 2026-10-18
"""
from __future__ import annotations

import numpy as np
from calc_core import (
    CONFIG,
    EXP_ROAD_0_PER_DRAW,
    EXP_ROAD_1_PER_DRAW,
    EXP_ROAD_2_PER_DRAW,
    N_NUM,
    Dataset,
    TimeSeries,
    analyze_number_group,
    get_num_prop_tag,
    get_prime_label,
    get_road_012,
    is_prime_num,
)


def calc_zone_bounds(zone_id: int, n_zones: int = 8, zone_size: int = 10) -> tuple[int, int]:
    """区号 0..7 映射到区间 1-based 端点 (lo, hi)。
    Z1: 01-10, Z2: 11-20, ..., Z8: 71-80。
    """
    lo = zone_id * zone_size + 1
    hi = min((zone_id + 1) * zone_size, N_NUM)
    return lo, hi


def run_calculus_engine(ds: Dataset, ts: TimeSeries) -> dict:
    """执行微积分八区动力学全量核算，融入 012 路动力学偏离与区内正交防撞车机制。

    @param ds 当前已截断的历史开奖数据集
    @param ts 对应的时序矩阵缓存
    @return 结构化核算字典，包含八区明细、16码推荐清单、宏观正交形态及动力学参数
    """
    n_zones = int(CONFIG["calc_zone_n"])
    zone_size = int(CONFIG["calc_zone_size"])
    W = min(int(CONFIG["window_calc"]), ds.T)
    Ws = min(int(CONFIG["window_calc_short"]), W)
    Wm = min(int(CONFIG["window_calc_mid"]), W)
    T = ds.T
    gap_cap = float(CONFIG["gap_cap"])
    ortho_enabled = bool(CONFIG.get("road_orthogonal_enabled", True))
    ortho_tol = float(CONFIG.get("road_orthogonal_tolerance", 0.15))
    w_int = float(CONFIG.get("weight_integral", 0.50))
    w_vel = float(CONFIG.get("weight_velocity", 0.30))
    w_acc = float(CONFIG.get("weight_accel", 0.20))
    pen_gap = float(CONFIG.get("penalty_gap", 4.0))
    pen_score = float(CONFIG.get("penalty_score", 0.40))
    p_vel_bonus = float(CONFIG.get("potential_velocity_bonus", 0.25))
    elas_slope = float(CONFIG.get("road_elasticity_slope", 0.10))
    elas_clamp = float(CONFIG.get("road_elasticity_clamp", 0.10))
    w_comb_k = float(CONFIG.get("combine_weight_kinetic", 0.50))
    label_hot = float(CONFIG.get("zone_label_hot_above", 0.29))
    label_cold = float(CONFIG.get("zone_label_cold_below", 0.21))

    # 步骤 1: 统计短窗内 012 路宏观偏离度 (Macro Road Bias & Elasticity)
    road_bias = {0: 0.0, 1: 0.0, 2: 0.0}
    road_hits_window = {0: 0.0, 1: 0.0, 2: 0.0}
    if T > 0 and Ws > 0:
        win_hits = ts.hit[T - Ws:T, :]  # shape: (Ws, 80)
        road_mask = np.array([[1.0 if n % 3 == r else 0.0 for n in range(1, N_NUM + 1)]
                              for r in range(3)])  # (3, 80)
        road_actual = road_mask @ win_hits.sum(axis=0)  # 三路各自短窗出号总数
        road_expect = np.array([
            Ws * EXP_ROAD_0_PER_DRAW,
            Ws * EXP_ROAD_1_PER_DRAW,
            Ws * EXP_ROAD_2_PER_DRAW,
        ])
        for r in range(3):
            road_hits_window[r] = float(road_actual[r])
            if road_expect[r] > 0.0:
                road_bias[r] = float((road_actual[r] - road_expect[r]) / road_expect[r])

    integ = np.zeros(N_NUM)
    vel = np.zeros(N_NUM)
    acc = np.zeros(N_NUM)
    gaps = np.zeros(N_NUM)
    k_scores = np.zeros(N_NUM)
    p_scores = np.zeros(N_NUM)

    # 步骤 2: 逐号计算微积分动力学物理量与属性增强
    for i in range(N_NUM):
        n = i + 1
        x = ts.hit[T - W:T, i].astype(float) if T else np.zeros(0)
        if len(x) == 0:
            continue
        # 短/中/长窗密度积分
        s_s = float(x[-Ws:].mean()) if Ws else 0.0
        s_m = float(x[-Wm:].mean()) if Wm else s_s
        s_l = float(x.mean())
        # 一阶导数（速度，动量）与二阶导数（加速度，曲率）
        v = s_s - s_l
        a = (s_s - s_m) - (s_m - s_l)
        g = float(ts.gap[T - 1, i]) if T > 0 else 0.0
        g_n = min(g, gap_cap) / gap_cap

        integ[i], vel[i], acc[i], gaps[i] = s_s, v, a, g

        # 动能分 (Kinetic): 顺势速度v + 加速度a + 密度s_s (连失>penalty_gap期实施惩罚，锁定短线冲顶爆发态)
        k_scores[i] = (w_int * s_s + w_vel * v + w_acc * a
                       - (pen_score if g > pen_gap else 0.0))

        # 012路弹性均值回归调制: 欠饱和路数获得适度势能补偿，过度过热路数轻微降温
        road = n % 3
        bias_val = road_bias.get(road, 0.0)
        # 偏离系数限幅在 ±road_elasticity_clamp 之间，严守微积分物理基础，防止噪声喧宾夺主
        road_elasticity = max(min(-elas_slope * bias_val, elas_clamp), -elas_clamp)

        # 势能分 (Potential): 针对适度遗漏号的弹性恢复力 + 拐点反弹势能 + 012路均值回归引力
        base_potential = g_n * (1.0 - s_s) + p_vel_bonus * max(v, 0.0)
        p_scores[i] = base_potential * (1.0 + road_elasticity)

    zones = []
    picks16 = []

    # 步骤 3: 八区分区闭环决策 (融入 012 路正交防撞车)
    for z in range(n_zones):
        lo, hi = calc_zone_bounds(z, n_zones, zone_size)
        members = list(range(lo, hi + 1))
        sl = slice(lo - 1, hi)
        zden = float(integ[sl].mean()) if hi >= lo else 0.25

        rows = []
        for n in members:
            i = n - 1
            sc = float(w_comb_k * k_scores[i] + (1.0 - w_comb_k) * p_scores[i])
            road = get_road_012(n)
            p_lbl = get_prime_label(n)
            p_tag = get_num_prop_tag(n)
            row = {
                "num": n,
                "integral": float(integ[i]),
                "velocity": float(vel[i]),
                "accel": float(acc[i]),
                "gap": float(gaps[i]),
                "k_score": float(k_scores[i]),
                "p_score": float(p_scores[i]),
                "score": sc,
                "road": road,
                "is_prime": is_prime_num(n),
                "prime_label": p_lbl,
                "prop_tag": p_tag,
            }
            rows.append(row)

        if not rows:
            continue

        # 双态选号：
        # 1. 动能顺势龙头 (Kinetic Top) 选最高动能分号码
        k_sorted = sorted(rows, key=lambda r: (-r["k_score"], -r["score"], r["num"]))
        top_k_item = k_sorted[0]
        k_road = top_k_item["road"]

        # 2. 势能反弹拐点 (Potential Top) 候选列表 (剔除动能码)
        p_candidates = [r for r in rows if r["num"] != top_k_item["num"]]
        p_sorted = sorted(p_candidates, key=lambda r: (-r["p_score"], -r["score"], r["num"]))

        top_p_item = p_sorted[0] if p_sorted else top_k_item

        # 步骤 4: 区内 012 路正交防撞车与物理发散优选 (Intra-Zone Road Orthogonality & Spatial Dispersion)
        # 若势能第一候选与动能码同路，且开启了正交覆盖，则在容忍度内优先选取跨路次优候选
        if ortho_enabled and len(p_sorted) > 1 and top_p_item["road"] == k_road:
            diff_road_candidates = [r for r in p_sorted if r["road"] != k_road]
            if diff_road_candidates:
                best_diff = diff_road_candidates[0]
                # 只有当跨路候选势能与最高势能差距 <= ortho_tol 时才采用，确保动力学强度不失真
                if best_diff["p_score"] >= (top_p_item["p_score"] - ortho_tol):
                    top_p_item = best_diff

        # 物理发散度惩罚与防扎堆优化：若热码与冷码跨度 <= 1 (连号死角)，在容忍度内优先选开阔候选
        if len(p_sorted) > 1 and abs(top_k_item["num"] - top_p_item["num"]) <= 1:
            dispersed_candidates = [r for r in p_sorted if abs(top_k_item["num"] - r["num"]) >= 3]
            if dispersed_candidates:
                best_disp = dispersed_candidates[0]
                if best_diff_disp := [r for r in dispersed_candidates if r["road"] != k_road]:
                    best_disp = best_diff_disp[0]
                if best_disp["p_score"] >= (top_p_item["p_score"] - ortho_tol * 1.5):
                    top_p_item = best_disp

        # 3. 选定中继温号 (Warm Number Top): 遗漏 3~7 期、动能与势能平衡的中坚码
        w_candidates = [
            r for r in rows
            if r["num"] not in (top_k_item["num"], top_p_item["num"]) and 3 <= r["gap"] <= 7
        ]
        if w_candidates:
            w_sorted = sorted(w_candidates, key=lambda r: (-r["score"], -r["k_score"], r["num"]))
            top_w_item = w_sorted[0]
        else:
            remaining = [r for r in rows if r["num"] not in (top_k_item["num"], top_p_item["num"])]
            rem_sorted = sorted(remaining, key=lambda r: (-r["score"], -r["k_score"], r["num"])) if remaining else [top_k_item]
            top_w_item = rem_sorted[0]

        # 4. 区内物理发散度指标计算 (跨度与奇偶互补)
        intra_span = abs(top_k_item["num"] - top_p_item["num"])
        is_parity_comp = (top_k_item["num"] % 2) != (top_p_item["num"] % 2)
        dispersion_label = f"跨度{intra_span}位·" + ("奇偶互补" if is_parity_comp else "同奇偶")

        # 5. 分区出号容量与形态走势动力学推演 (Zone Output Capacity & Trend Projection)
        if T > 0:
            zone_counts = ts.hit[:, lo - 1:hi].sum(axis=1).astype(float)
            c_last = int(zone_counts[-1])
            c_3 = float(zone_counts[-3:].mean()) if T >= 3 else float(zone_counts.mean())
            c_5 = float(zone_counts[-5:].mean()) if T >= 5 else c_3
            c_10 = float(zone_counts[-10:].mean()) if T >= 10 else c_5
            c_30 = float(zone_counts[-30:].mean()) if T >= 30 else float(zone_counts.mean())

            z_vel = c_5 - c_30
            z_acc = (c_5 - c_10) - (c_10 - c_30)

            if c_last >= 4 or (z_vel > 0.4 and z_acc >= 0):
                z_trend = "📈 爆发冲顶"
                z_exp_range = "3~4码"
                z_tactical_adv = "出号主攻放量区，建议重点关注或出胆"
                z_quota = 3
            elif c_last <= 1 and (z_acc > 0 or c_3 <= 1.5):
                z_trend = "🔄 触底反弹"
                z_exp_range = "2~3码"
                z_tactical_adv = "低位势能积聚，重点防冷号反弹"
                z_quota = 2
            elif z_vel > 0.2 and z_acc < -0.2:
                z_trend = "📉 见顶降温"
                z_exp_range = "1~2码"
                z_tactical_adv = "动能见顶回落，轻度防守防踏空"
                z_quota = 1
            else:
                z_trend = "⚖️ 平稳常态"
                z_exp_range = "2~3码"
                z_tactical_adv = "出号节奏平稳，标配追热+防冷"
                z_quota = 2
        else:
            c_last, c_3, z_vel, z_acc = 2, 2.5, 0.0, 0.0
            z_trend = "⚖️ 平稳常态"
            z_exp_range = "2~3码"
            z_tactical_adv = "基准均衡配置"
            z_quota = 2

        top = [top_k_item, top_p_item]
        top_nums = [top_k_item["num"], top_p_item["num"]]
        top3 = [top_k_item, top_w_item, top_p_item]
        top3_nums = [top_k_item["num"], top_w_item["num"], top_p_item["num"]]

        zone_hits = float(ts.hit[T - W:T, lo - 1:hi].sum()) if T else 0.0
        label = ("区热" if zden > label_hot else
                 ("区冷" if zden < label_cold else "区平"))

        zones.append({
            "zone_id": z + 1,
            "lo": lo,
            "hi": hi,
            "members": members,
            "top2": top_nums,
            "top3": top3_nums,
            "top_k_item": top_k_item,
            "top_w_item": top_w_item,
            "top_p_item": top_p_item,
            "top2_detail": top,
            "top3_detail": top3,
            "ranked": sorted(rows, key=lambda r: (-r["score"], r["num"])),
            "energy_hits": zone_hits,
            "zone_density": zden,
            "label": label,
            "c_last": c_last,
            "c_3": c_3,
            "z_vel": z_vel,
            "z_acc": z_acc,
            "z_trend": z_trend,
            "z_exp_range": z_exp_range,
            "z_tactical_adv": z_tactical_adv,
            "z_quota": z_quota,
            "intra_span": intra_span,
            "is_parity_comp": is_parity_comp,
            "dispersion_label": dispersion_label,
        })
        picks16.extend(top_nums)

    # 步骤 5: 融入新一代 GBDT/LightGBM + 共现拓扑图网络多模态协同模型
    from calc_ensemble import compute_multimodal_scores, solve_global_constrained_picks

    fusion_scores, lgb_probs, graph_scores, companion_dict = compute_multimodal_scores(
        ts, T, k_scores, p_scores
    )

    # 步骤 6: 全局带约束求解最优组合 (多模态自适应驱动，集成实战分级投注体系)
    dynamic_picks, picks16, picks24, extra_tiers = solve_global_constrained_picks(
        zones, fusion_scores, companion_dict, return_extra=True
    )

    # 步骤 7: 计算 16 码、24 码与动态自适应配额宏观正交形态分析
    macro_16 = analyze_number_group(picks16)
    macro_24 = analyze_number_group(picks24)
    macro_dynamic = analyze_number_group(dynamic_picks)

    return {
        "n_zones": n_zones,
        "zone_size": zone_size,
        "window": W,
        "picks16": picks16,
        "picks24": picks24,
        "dynamic_picks": dynamic_picks,
        "bankers3": extra_tiers.get("bankers3", []),
        "pick5": extra_tiers.get("pick5", []),
        "pick7": extra_tiers.get("pick7", []),
        "pick10": extra_tiers.get("pick10", []),
        "picks30": extra_tiers.get("picks30", []),
        "hot_zones": extra_tiers.get("hot_zones", []),
        "macro_analysis_16": macro_16,
        "macro_analysis_24": macro_24,
        "macro_analysis_dynamic": macro_dynamic,
        "road_bias": road_bias,
        "road_hits_window": road_hits_window,
        "zones": zones,
        "integ": integ,
        "vel": vel,
        "acc": acc,
        "gaps": gaps,
        "k_scores": k_scores,
        "p_scores": p_scores,
        "fusion_scores": fusion_scores,
        "lgb_probs": lgb_probs,
        "graph_scores": graph_scores,
        "companion_dict": companion_dict,
    }
