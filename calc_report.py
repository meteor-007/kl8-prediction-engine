# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 深度分析报告与昨日复盘展示模块

【业务背景与架构定位】:
本模块负责生成结构化分析文本，提供：
1. 昨日真实开奖复盘：开奖形态统计（大小、奇偶、和值、连号）与微积分 16 码逐区命中实测核算；
2. 当期八区微积分动力学全量指标呈现（动能顺势龙头 vs 势能反弹拐点）；
3. 控制台精简/详细双模式美观排版与 TXT 报告持久化。

@author czw
@since 2026-10-18
"""
from __future__ import annotations

from pathlib import Path
from calc_core import (
    EXCEL_DIR,
    Dataset,
    EXP_PRIME_IN_16,
    EXP_ROAD_0_IN_16,
    EXP_ROAD_1_IN_16,
    EXP_ROAD_2_IN_16,
    PRIME_EXPECTED_COUNT_IN_16,
    PRIME_SD_IN_16,
    REPORTS_DIR,
    ROAD_EXPECTED_COUNT_IN_16,
    ROAD_SD_IN_16,
    WECHAT_DIR,
    TimeSeries,
    analyze_number_group,
    get_num_prop_tag,
    get_road_012,
    hypergeometric_baseline,
    truncated,
)
from calc_engine import run_calculus_engine


def build_previous_review(ds: Dataset) -> list[str]:
    """构建前一期（昨日）真实开奖全景复盘与微积分 16 码实测结算，以通俗易懂的方式呈现。"""
    lines = []
    T = ds.T
    if T < 2:
        return ["  无足够历史数据进行复盘"]

    prev = ds.draws[T - 1]
    pp = ds.draws[T - 2]
    actual_sorted = sorted(prev.numbers)
    repeats = sorted(prev.numbers & pp.numbers)

    # 步骤 1: 真实开奖全维度形态分析 (大小、奇偶、012路、质合、和值)
    act_analysis = analyze_number_group(actual_sorted)

    lines.append(f"  【昨日官方真实开奖 (第 {prev.period} 期 · {prev.date})】:")
    lines.append("    · 20个中奖号码: " + " ".join(f"{n:02d}" for n in actual_sorted))
    lines.append(
        f"    · 基础分布形态: 大小比(1-40:41-80) 为 {act_analysis['size_ratio']} ｜ "
        f"奇偶比 为 {act_analysis['odd_ratio']} ｜ 和值 {act_analysis['sum_value']} ｜ 跨度 {act_analysis['span']}"
    )
    lines.append(
        f"    · 012路与质合: 0路 {act_analysis['road_counts'][0]}个 : 1路 {act_analysis['road_counts'][1]}个 : 2路 {act_analysis['road_counts'][2]}个 ｜ "
        f"质数 {act_analysis['prime_count']}个 : 合数 {act_analysis['composite_count']}{'+特1' if act_analysis['special_1'] else ''}个"
    )
    lines.append(
        f"    · 连开重号追踪: 与前一期重叠 {len(repeats)} 个 → "
        + (" ".join(f"{n:02d}" for n in repeats) if repeats else "无重号")
    )

    # 步骤 2: 回溯上期微积分推演与核算
    try:
        tds = truncated(ds, prev.period)
        ts_p = TimeSeries(tds)
        res_p = run_calculus_engine(tds, ts_p)
        picks16_p = res_p["picks16"]
        h16_p = sorted(set(picks16_p) & prev.numbers)
        dyn_p = res_p.get("dynamic_picks") or picks16_p
        h_dyn_p = sorted(set(dyn_p) & prev.numbers)
        picks24_p = res_p.get("picks24") or []
        h24_p = sorted(set(picks24_p) & prev.numbers)
        bankers3_p = sorted(res_p.get("bankers3") or [])
        h_b3 = sorted(set(bankers3_p) & prev.numbers)

        lines.append("")
        if bankers3_p:
            if h_b3:
                lines.append(f"  👑【昨日系统 · 金银铜胆 (3码) 复盘】: 成功命中 {len(h_b3)} 码 (命中率 {len(h_b3)/len(bankers3_p)*100:.1f}%) → 中奖号: {' '.join(f'{n:02d}' for n in h_b3)}")
            else:
                lines.append("  👑【昨日系统 · 金银铜胆 (3码) 复盘】: 当期未命中")
        lines.append("  ⚡【昨日系统 · 打法 A 动态自适应配额推荐实战复盘】(核心主推方案):")
        base_dyn = hypergeometric_baseline(len(dyn_p))
        if h_dyn_p:
            hit_dyn_str = " ".join(f"{n:02d}" for n in h_dyn_p)
            lines.append(f"    · 实战战绩: 推荐 {len(dyn_p)} 码，成功命中 {len(h_dyn_p)} 码 (命中率 {len(h_dyn_p)/len(dyn_p)*100:.1f}%) → 中奖号: {hit_dyn_str}")
        else:
            lines.append(f"    · 实战战绩: 动态自适应 {len(dyn_p)} 码当期未命中")
        lines.append(
            f"    · 数学定位: 纯随机选 {len(dyn_p)} 码的期望命中为 {base_dyn['mean']:.2f} 码 (σ={base_dyn['sd']:.2f})，"
            f"单期命中 {len(h_dyn_p)} 码处于正常波动带内 (单日结论无统计意义，须看百期累计)。"
        )

        lines.append("")
        lines.append("  🔥【昨日系统 · 基准 16 码与豪华 24 码大底实测对照】:")
        if h16_p:
            lines.append(f"    · 基准 16 码命中: 成功命中 {len(h16_p)} 码 → 中奖号: {' '.join(f'{n:02d}' for n in h16_p)}")
        else:
            lines.append("    · 基准 16 码命中: 当期未命中")
        if h24_p:
            lines.append(f"    · 豪华 24 码大底: 成功命中 {len(h24_p)} 码 (覆盖率 {len(h24_p)/len(picks24_p)*100:.1f}%) → 中奖号: {' '.join(f'{n:02d}' for n in h24_p)}")

        lines.append("")
        lines.append("    · 八个分区推荐逐区走势推演与命中明细:")
        for z in res_p["zones"]:
            q = z.get("z_quota", 2)
            if q >= 3:
                z_dyn = z["top3"]
            elif q == 2:
                z_dyn = z["top2"]
            else:
                z_dyn = [z["top_k_item"]["num"]]
            zh_dyn = sorted(set(z_dyn) & prev.numbers)
            zh_all = sorted(set(z["top3"]) & prev.numbers)

            k_n = z["top_k_item"]["num"]
            p_n = z["top_p_item"]["num"]
            w_n = z.get("top_w_item", z["top_k_item"])["num"]
            trend_s = z.get("z_trend", "⚖️ 平稳常态").split()[0]
            quota_s = f"{q}码"
            k_tag = z["top_k_item"].get("prop_tag", get_num_prop_tag(k_n))
            p_tag = z["top_p_item"].get("prop_tag", get_num_prop_tag(p_n))
            w_tag = z.get("top_w_item", {}).get("prop_tag", get_num_prop_tag(w_n))

            if zh_dyn:
                mark = "✓ [命中]"
                hit_res = f" → 命中 {', '.join(f'{x:02d}' for x in zh_dyn)}"
            elif zh_all:
                mark = "~ [大底中]"
                hit_res = f" → 温码命中 {', '.join(f'{x:02d}' for x in zh_all)}"
            else:
                mark = "· [未中]"
                hit_res = ""

            lines.append(
                f"      {mark:<7} 第{z['zone_id']}区({z['lo']:02d}-{z['hi']:02d}) [{trend_s}·配{quota_s}]: "
                f"热 {k_n:02d}({k_tag}) ｜ 温 {w_n:02d}({w_tag}) ｜ 冷 {p_n:02d}({p_tag}){hit_res}"
            )
    except Exception as e:
        lines.append(f"    [提示] 历史回溯复盘异常: {e}")

    return lines


def evaluate_macro_health(macro: dict) -> tuple[str, str]:
    """综合评估 16 码宏观正交形态的结构健康度。

    判定不再使用拍脑袋的绝对偏差值，而是把 012 路 / 质合 的实测个数换算成
    「完全随机选 16 码」时的超几何分布标准分数 z，用统一的统计尺度衡量偏态，
    并对三路 + 质合四个指标取最大偏离 |z| (多重比较下的最坏维度)。

    @param macro analyze_number_group 输出的 16 码形态字典
    @return (评级标题, 诊断说明)
    """
    rc = macro["road_counts"]
    max_z = 0.0
    worst = ""
    for road, cnt in rc.items():
        z = abs(cnt - ROAD_EXPECTED_COUNT_IN_16[road]) / ROAD_SD_IN_16[road]
        if z > max_z:
            max_z, worst = z, f"{road}路{cnt}码"
    z_prime = abs(macro["prime_count"] - PRIME_EXPECTED_COUNT_IN_16) / PRIME_SD_IN_16
    if z_prime > max_z:
        max_z, worst = z_prime, f"质数{macro['prime_count']}码"

    if max_z <= 1.0:
        return ("★★★★★ [结构高度均衡 · 黄金抗偏态区间]",
                f"012路与质合分布落在随机基准 ±1σ 内 (最大偏离 |z|={max_z:.2f})，"
                "与纯随机选号的统计形态中心高度吻合，无扎堆偏科风险")
    elif max_z <= 2.0:
        return ("★★★★☆ [结构良好 · 正常波动区间]",
                f"路数与质合形态处于正常波动区间 (最大偏离 |z|={max_z:.2f}，主要来自{worst})，"
                "号码分布均匀，覆盖面佳")
    else:
        return ("★★★☆☆ [结构显著偏态 · 需留意]",
                f"最大偏离 |z|={max_z:.2f} 已超出随机波动 2σ (主要来自{worst})，"
                "属小概率结构形态，已由区内跨路对冲做部分纠偏")


def build_report_text(
    ds: Dataset,
    engine_res: dict,
    target_period: int,
    target_date: str,
    backtest_text: str = "",
    audit_text: str = "",
) -> str:
    """生成完整微积分八区预测报告文本（清晰通俗版）。

    @param ds 当前开奖数据集
    @param engine_res 动力学引擎结果
    @param target_period 目标期号
    @param target_date 目标日期
    @param backtest_text 可选：命中率走查回测文本 (置于附录之后)
    @param audit_text 可选：特征预测力审计文本 (置于附录之后)
    @return 完整报告文本
    """
    L = []
    sep = "=" * 80
    line = "-" * 80
    last = ds.draws[-1]
    baseline = hypergeometric_baseline(16)

    L.append(sep)
    L.append("  快乐8「微积分八区分区预测系统」· 每日实战推荐报告")
    L.append(f"  预测目标期: 第 {target_period} 期   开奖日期: {target_date}   模式: 纯开奖数据驱动")
    L.append(f"  数据基准截至: 第 {last.period} 期 ({last.date}) ｜ 历史样本: T={ds.T} 期")
    L.append(sep)

    # 1. 昨日深度复盘
    L.append("[0] 上期真实开奖与推荐战绩复盘")
    L.append(line)
    L.extend(build_previous_review(ds))
    L.append("")

    # 2. 分区出号容量与走势动力学推演
    L.append("[1] 八大分区出号容量与形态走势推演 (Zone Output Capacity & Trend Projection)")
    L.append(line)
    L.append("  【推演原理】: 追踪每个物理分区在短/中/长周期的出号动量变化率与曲率加速度，推演其所处的生命周期走势。")
    L.append("")
    L.append("  分区      号码范围    上期出号  近3期均  出号速度v  加速度a    形态走势研判    预估出号    战术运作指引")
    L.append("  " + "-" * 90)

    zones = engine_res["zones"]
    for z in zones:
        c_last_s = f"{z.get('c_last', 2):2d} 码"
        c_3_s = f"{z.get('c_3', 2.5):.1f} 码"
        v_s = f"{z.get('z_vel', 0.0):+.2f}"
        a_s = f"{z.get('z_acc', 0.0):+.2f}"
        trend_s = f"{z.get('z_trend', '⚖️ 平稳常态')}"
        exp_s = f"{z.get('z_exp_range', '2~3码')}"
        adv_s = f"{z.get('z_tactical_adv', '标准配置')}"

        L.append(
            f"  第 {z['zone_id']} 区   {z['lo']:02d} - {z['hi']:02d}    "
            f"{c_last_s}    {c_3_s}   {v_s:<9}  {a_s:<9}  {trend_s:<12}  {exp_s:<8}  {adv_s}"
        )
    L.append("  " + "-" * 90)
    L.append("")

    # 3. 本期微积分动力学预测 (升级为三态战术梯队 + 发散度)
    L.append("[2] 本期微积分·八区分区战术推荐（每区精选3码梯队：追热龙头 + 中继温号 + 防冷反弹）")
    L.append(line)
    L.append("  【选号梯队】: 每区输出 [追热龙头] + [中继温号] + [防冷反弹]，兼顾短线冲顶、中线温和中继与大冷拐点反弹。")
    L.append("  【发散防撞】: 严防连号死角 (跨度>=3位)，双码尽量跨路互补，避免同频共振。")
    L.append("")
    L.append("  分区      号码范围    区态走势        追热龙头码(热)       中继温号码(温)       防反弹拐点码(冷)     区内发散度与跨路")
    L.append("  " + "-" * 105)

    for z in zones:
        t0 = z["top_k_item"]
        tw = z.get("top_w_item", t0)
        t1 = z["top_p_item"]
        p0_tag = t0.get("prop_tag", get_num_prop_tag(t0["num"]))
        pw_tag = tw.get("prop_tag", get_num_prop_tag(tw["num"]))
        p1_tag = t1.get("prop_tag", get_num_prop_tag(t1["num"]))
        r0 = t0.get("road", get_road_012(t0["num"]))
        r1 = t1.get("road", get_road_012(t1["num"]))
        road_match = f"[{r0}+{r1}路]"

        col_k = f"{t0['num']:02d} ({p0_tag})"
        col_w = f"{tw['num']:02d} ({pw_tag})"
        col_p = f"{t1['num']:02d} ({p1_tag})"
        trend_s = z.get("z_trend", "⚖️ 平稳常态")
        disp_s = f"{z.get('dispersion_label', '正常发散')} {road_match}"

        L.append(
            f"  第 {z['zone_id']} 区   {z['lo']:02d} - {z['hi']:02d}    "
            f"{trend_s:<14}  {col_k:<18}  {col_w:<18}  {col_p:<18}  {disp_s}"
        )

    picks16 = engine_res["picks16"]
    sorted_16 = sorted(picks16)
    picks16_s = "  ".join(f"{n:02d}" for n in sorted_16)

    dyn_picks = sorted(engine_res.get("dynamic_picks") or [])
    dyn_s = "  ".join(f"{n:02d}" for n in dyn_picks)

    picks24 = engine_res.get("picks24") or []
    sorted_24 = sorted(picks24)
    picks24_s = "  ".join(f"{n:02d}" for n in sorted_24)

    bankers3 = engine_res.get("bankers3") or []
    pick5 = engine_res.get("pick5") or []
    pick7 = engine_res.get("pick7") or []
    pick10 = engine_res.get("pick10") or []
    hot_zones = engine_res.get("hot_zones") or []

    L.append("  " + "-" * 105)
    L.append("")
    if bankers3:
        L.append("  👑【实战核心 · 金银铜胆 3 码锚点】 (全局多模态最高置信度极核 · 选二/定胆必备):")
        L.append(f"     {'  '.join(f'{n:02d}' for n in bankers3)}")
        L.append("")
    if pick5:
        L.append("  🎯【实战精选 · 高胜率选五组合】 (5 码高密度精选单 · 兼顾连号与强伴生):")
        L.append(f"     {'  '.join(f'{n:02d}' for n in pick5)}")
        L.append("")
    if pick7:
        L.append("  🎯【实战精选 · 进阶选七组合】 (7 码进阶实投单):")
        L.append(f"     {'  '.join(f'{n:02d}' for n in pick7)}")
        L.append("")
    if pick10:
        L.append("  🏆【实战精选 · 选十主力推荐组合】 (10 码主力实投单):")
        L.append(f"     {'  '.join(f'{n:02d}' for n in pick10)}")
        L.append("")
    if hot_zones:
        L.append("  🚀【主攻放量双区 · 重兵饱和突击】 (锁定当前动能冲顶与高密度爆发区，集束出击):")
        for hz in hot_zones:
            hz_picks_s = "  ".join(f"{n:02d}" for n in hz["picks"])
            L.append(f"     · {hz['label']} [{hz['trend']} · 上期出{hz['c_last']}码]: 精选突击码 -> {hz_picks_s}")
        L.append("")
    if dyn_picks:
        L.append(f"  ★ 本期时序容量【动态自适应配额推荐大名单】 ({len(dyn_picks)} 码 · 依各区走势自动调度 1~4 码重防):")
        L.append(f"     {dyn_s}")
        L.append("")
    L.append("  ★ 本期八区核心【精选 16 码主力基准推荐名单】 (升序排列 · 全局多模态前16优选):")
    L.append(f"     {picks16_s}")
    L.append("")
    if picks24:
        L.append("  ★ 本期八区全景【24 码豪华大底名单】 (升序排列 · 全覆盖防踏空大名单):")
        L.append(f"     {picks24_s}")
        L.append("")
    L.append("  【实战运用提示】:")
    L.append("    · 稳健型: 每个分区各选 1 码主力（组 8 码精选单）；")
    L.append("    · 进阶型: 重点关注 [📈 爆发冲顶] 与 [🔄 触底反弹] 分区，选用该区 2~3 码梯队；")
    L.append("    · 选十/大底型: 以 16 码为主力底池、24 码为后备池，任选 10 码组合；")
    L.append("    · 胆拖型: 挑选 2~3 个走势最明朗的区各定 1 个热/冷码作胆，其余分区选码作拖。")
    L.append(f"    · 战绩预期: 16 码每期平均命中约 {baseline['mean']:.2f} 码（这是纯随机选号的数学期望），")
    L.append(f"      命中≥4码概率本身就有 {baseline['p_ge'][4] * 100:.1f}%、≥5码 {baseline['p_ge'][5] * 100:.1f}%、≥7码仅 {baseline['p_ge'][7] * 100:.1f}%；")
    L.append("      单日战绩围绕该基准上下波动属正常统计现象，请以 100 期以上的累计均值评判方案优劣。")
    L.append("")

    # 4. 宏观属性与正交形态评估
    macro = engine_res.get("macro_analysis_16") or analyze_number_group(picks16)
    r0_cnt = macro["road_counts"][0]
    r1_cnt = macro["road_counts"][1]
    r2_cnt = macro["road_counts"][2]
    primes = macro["prime_count"]
    composites = macro["composite_count"]
    health_title, health_memo = evaluate_macro_health(macro)

    L.append("[3] 16码宏观形态体检卡（大数定律校验，拒绝偏科）")
    L.append(line)
    L.append(f"  · 结构健康评级: {health_title}")
    L.append(f"    体检诊断: {health_memo}")
    L.append(
        f"  · 012路分布比: 0路选了 {r0_cnt} 个 ｜ 1路选了 {r1_cnt} 个 ｜ 2路选了 {r2_cnt} 个 "
        f"(比例 {macro['road_ratio']}，随机基准期望 {EXP_ROAD_0_IN_16:.1f}:{EXP_ROAD_1_IN_16:.1f}:{EXP_ROAD_2_IN_16:.1f})"
    )
    L.append(
        f"  · 质数合数分布: 质数选了 {primes} 个 ｜ 合数选了 {composites} 个 "
        f"(比例 {macro['prime_ratio']}，随机基准期望质数 {EXP_PRIME_IN_16:.1f} 个)"
    )
    L.append(
        f"  · 大小与奇偶比: 大小比 {macro['size_ratio']} ｜ 奇偶比 {macro['odd_ratio']} ｜ 和值 {macro['sum_value']} ｜ 跨度 {macro['span']}"
    )
    L.append("")

    # 4. 详细附录：全80码微积分动力学与多模态参数表
    L.append("[附录一] 80个号码底层多模态与动力学全参表（高阶专业分析专用，可略过）")
    L.append(line)
    fusion_scores = engine_res.get("fusion_scores")
    lgb_probs = engine_res.get("lgb_probs")
    graph_scores = engine_res.get("graph_scores")
    dyn_set = set(engine_res.get("dynamic_picks") or [])
    p16_set = set(picks16)

    for z in zones:
        L.append(f"  —— 第 {z['zone_id']} 区 ({z['lo']:02d}-{z['hi']:02d}) [{z['label']}] ——")
        for r in z["ranked"]:
            n = r["num"]
            n_idx = n - 1
            if n in dyn_set:
                mark = "★[打法A]"
            elif n in p16_set:
                mark = "★[16主力]"
            else:
                mark = "        "
            ptag = r.get("prop_tag", get_num_prop_tag(n))
            f_s = f"{fusion_scores[n_idx]:+.3f}" if fusion_scores is not None else f"{r['score']:+.3f}"
            lgb_s = f"{lgb_probs[n_idx]:.3f}" if lgb_probs is not None else "0.250"
            g_s = f"{graph_scores[n_idx]:.2f}" if graph_scores is not None else "0.00"

            L.append(
                f"   {mark} {n:02d}号({ptag}) ｜ 多模态分={f_s} "
                f"(ML概率={lgb_s}, 图中心度={g_s}) "
                f"｜ 速度v={r['velocity']:+.2f} 加速度a={r['accel']:+.2f} 已遗漏g={int(r['gap']):2d}期"
            )
    L.append("")
    if backtest_text:
        L.append("[附录二] 命中率严格走查回测 (Walk-Forward · 零前瞻泄漏)")
        L.append(line)
        L.append(backtest_text)
        L.append("")
    if audit_text:
        L.append("[附录三] 特征预测力审计与深度优化空间裁决")
        L.append(line)
        L.append(audit_text)
        L.append("")
    L.append(sep)
    L.append("  温馨提示: 快乐8每期开出20码，任何科学算法均为统计学概率优化工具，请理性投注、量力而行。")
    L.append(sep)

    return "\n".join(L)


def generate_wechat_text(
    ds: Dataset,
    engine_res: dict,
    target_period: int,
    target_date: str,
    backtest_result: dict | None = None,
) -> str:
    """生成适用于微信群/朋友圈一键复制发布的精美分桶预测文本（微信分桶原文）。

    @param ds 当前开奖数据集
    @param engine_res 引擎计算结果
    @param target_period 目标预测期号
    @param target_date 目标预测日期
    @param backtest_result 可选：走查回测结果，用于附真实战绩统计
    @return 格式优美的微信分桶发布文案
    """
    W = []
    line_dash = "---------------------------------------------"
    W.append("🎯【快乐8·微积分八区分区推荐】(多模态协同预测)")
    W.append(f"📅 预测期号：第 {target_period} 期 ｜ 日期：{target_date}")
    W.append("🔬 算法模型：新一代多模态协同架构 (LightGBM + 图共现网络 + 二阶动力学 + 全局带约束求解)")
    W.append(line_dash)
    W.append("【八区分桶精选一览】(每区精选三态梯队: 1热号龙头 + 1中继温号 + 1冷号反弹)")
    W.append("  ※ 走势研判: 追踪分区出号速度与加速度，推演当前生命周期与容量走势。")

    zones = engine_res["zones"]
    for z in zones:
        t0 = z["top_k_item"]
        tw = z.get("top_w_item", t0)
        t1 = z["top_p_item"]
        p0_tag = t0.get("prop_tag", get_num_prop_tag(t0["num"]))
        pw_tag = tw.get("prop_tag", get_num_prop_tag(tw["num"]))
        p1_tag = t1.get("prop_tag", get_num_prop_tag(t1["num"]))
        r0 = t0.get("road", get_road_012(t0["num"]))
        r1 = t1.get("road", get_road_012(t1["num"]))
        trend_short = z.get("z_trend", "⚖️ 平稳常态").split()[0]
        exp_range = z.get("z_exp_range", "2~3码")

        W.append(
            f"  🏷️ 第{z['zone_id']}区({z['lo']:02d}-{z['hi']:02d}) [{trend_short}·估{exp_range}]："
            f"热 {t0['num']:02d}({p0_tag}) ｜ 温 {tw['num']:02d}({pw_tag}) ｜ 冷 {t1['num']:02d}({p1_tag})"
        )

    picks16 = sorted(engine_res["picks16"])
    row1_8 = "  ".join(f"{n:02d}" for n in picks16[:8])
    row2_8 = "  ".join(f"{n:02d}" for n in picks16[8:])

    dyn_picks = sorted(engine_res.get("dynamic_picks") or [])
    bankers3 = engine_res.get("bankers3") or []
    pick5 = engine_res.get("pick5") or []
    pick7 = engine_res.get("pick7") or []
    pick10 = engine_res.get("pick10") or []
    hot_zones = engine_res.get("hot_zones") or []

    if bankers3 or pick5 or pick10:
        W.append(line_dash)
        W.append("💎【实战分级投注核心推荐】(高胜率精选单)：")
        if bankers3:
            W.append(f"  👑 金银铜胆 (3码)：{'  '.join(f'{n:02d}' for n in bankers3)}")
        if pick5:
            W.append(f"  🎯 选五精选 (5码)：{'  '.join(f'{n:02d}' for n in pick5)}")
        if pick7:
            W.append(f"  🎯 选七精选 (7码)：{'  '.join(f'{n:02d}' for n in pick7)}")
        if pick10:
            W.append(f"  🏆 选十主力 (10码)：{'  '.join(f'{n:02d}' for n in pick10)}")

    if hot_zones:
        W.append(line_dash)
        W.append("🚀【主攻放量双区 · 重兵突击】(动能冲顶放量区)：")
        for hz in hot_zones:
            W.append(f"  · {hz['label']} [{hz['trend']}]: {'  '.join(f'{n:02d}' for n in hz['picks'])}")

    if dyn_picks:
        half_dyn = (len(dyn_picks) + 1) // 2
        row1_dyn = "  ".join(f"{n:02d}" for n in dyn_picks[:half_dyn])
        row2_dyn = "  ".join(f"{n:02d}" for n in dyn_picks[half_dyn:])
        W.append(line_dash)
        W.append(f"⚡【打法A · 动态自适应大名单】({len(dyn_picks)}码 · 依走势自适应调度)：")
        W.append(f"  {row1_dyn}")
        W.append(f"  {row2_dyn}")

    W.append(line_dash)
    W.append("🔥【主力 16 码基准推荐名单】(全局多模态优选 · 升序)：")
    W.append(f"  {row1_8}")
    W.append(f"  {row2_8}")

    picks24 = sorted(engine_res.get("picks24") or [])
    if picks24:
        row1_12 = "  ".join(f"{n:02d}" for n in picks24[:12])
        row2_12 = "  ".join(f"{n:02d}" for n in picks24[12:])
        W.append(line_dash)
        W.append("🌟【豪华 24 码全景大底】(全覆盖防踏空 · 升序)：")
        W.append(f"  {row1_12}")
        W.append(f"  {row2_12}")
    W.append(line_dash)

    macro = engine_res.get("macro_analysis_16") or analyze_number_group(picks16)
    health_title, _ = evaluate_macro_health(macro)

    W.append("💡【实战投注组合建议】：")
    W.append("  ① 稳健选八：每个分区各挑 1 码组选八单；")
    W.append("  ② 激进选十：以 16 码为大底池任选 10 码；")
    W.append("  ③ 黄金胆拖：挑选 2~3 个最看好的区各定 1 胆码。")
    W.append(line_dash)
    W.append(f"🩺【形态体检】：012路比 {macro['road_ratio']} ｜ 质合比 {macro['prime_ratio']} ｜ 大小比 {macro['size_ratio']}")
    W.append(f"⭐ 结构健康度：{health_title}")

    if backtest_result and backtest_result.get("n_sample", 0) > 0:
        bt = backtest_result
        best_zone = max((r["zone_coverage"] for r in bt["rows"]), default=0)
        W.append(line_dash)
        W.append(f"📊【近 {bt['n_sample']} 期实测战绩】(严格 walk-forward 逐期回溯)：")
        if "avg_hits_dyn" in bt:
            dyn_avg = bt["avg_hits_dyn"]
            ge5_d = bt.get("ge_dyn", {}).get(5, {}).get("pct", 0.0)
            ge6_d = bt.get("ge_dyn", {}).get(6, {}).get("pct", 0.0)
            W.append(f"  · ⚡【打法A·动态自适应17码】平均命中 {dyn_avg:.2f} 码/期 (净提升 {dyn_avg - bt['avg_hits']:+.2f} 码)")
            W.append(f"    胜率表现: ≥4码 {bt.get('ge_dyn', {}).get(4, {}).get('pct', 0.0):.0f}% ｜ ≥5码 {ge5_d:.0f}% ｜ ≥6码 {ge6_d:.0f}%")
        W.append(f"  · 16码基准平均命中 {bt['avg_hits']:.2f} 码/期 ｜ 单期最高 {bt['max_hits']} 码")
        W.append(f"  · 基准胜率: 命中≥4码 {bt['ge'][4]['pct']:.0f}% ｜ ≥5码 {bt['ge'][5]['pct']:.0f}% ｜ ≥6码 {bt['ge'][6]['pct']:.0f}%")
        W.append(f"  · 八区平均覆盖 {bt['avg_zone_coverage']:.2f}/8 区 ｜ 单期最高覆盖 {best_zone}/8 区")
        W.append(f"  · 统计判定：[{bt['grade']}级] {bt['verdict']}")
        W.append(f"  · 参照基准：纯随机选16码期望命中恰为 {bt['baseline']['mean']:.2f} 码")

    W.append(line_dash)
    W.append("⚠️ 理性参考，量力而行，祝君好运大奖！")
    return "\n".join(W)


def generate_and_save_report(
    ds: Dataset,
    engine_res: dict,
    target_period: int,
    target_date: str,
    backtest_text: str = "",
    audit_text: str = "",
) -> tuple[str, Path]:
    """生成并保存 TXT 报告文件。"""
    rep_text = build_report_text(ds, engine_res, target_period, target_date,
                                 backtest_text=backtest_text, audit_text=audit_text)
    out_dir = Path(REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_date = target_date.replace("-", "")
    file_path = out_dir / f"微积分八区预测报告_{clean_date}.txt"
    file_path.write_text(rep_text + "\n", encoding="utf-8")
    return rep_text, file_path


def generate_and_save_all_reports(
    ds: Dataset,
    engine_res: dict,
    target_period: int,
    target_date: str,
    backtest_text: str = "",
    audit_text: str = "",
    backtest_result: dict | None = None,
    sweep: list[dict] | None = None,
) -> dict[str, str | Path]:
    """一键生成并持久化所有格式的交付报告：TXT 报告、微信分桶原文、Excel 报表。

    @param ds 当前已加载开奖数据集
    @param engine_res 动力学核算结果
    @param target_period 目标期号
    @param target_date 目标日期
    @param backtest_text 命中率走查回测文本 (写入 TXT 附录二)
    @param audit_text 特征预测力审计文本 (写入 TXT 附录三)
    @param backtest_result 回测结构化结果 (用于微信战绩提示与 Excel 看板)
    @param sweep 参数寻优扫描结果 (写入 Excel 看板)
    @return 包含各项报告路径与内容的汇总字典
    """
    clean_date = target_date.replace("-", "")

    # 1. 保存详细 TXT 报告
    txt_rep, txt_path = generate_and_save_report(
        ds, engine_res, target_period, target_date,
        backtest_text=backtest_text, audit_text=audit_text,
    )

    # 2. 生成并保存微信分桶原文
    wechat_text = generate_wechat_text(ds, engine_res, target_period, target_date,
                                       backtest_result=backtest_result)
    wechat_dir = Path(WECHAT_DIR)
    wechat_dir.mkdir(parents=True, exist_ok=True)
    wechat_path = wechat_dir / f"微信分桶推荐_{clean_date}_第{target_period}期.txt"
    wechat_path.write_text(wechat_text + "\n", encoding="utf-8")

    # 3. 生成并保存 Excel 工作簿
    from calc_excel import generate_excel_report
    excel_path = generate_excel_report(ds, engine_res, target_period, target_date,
                                       output_dir=EXCEL_DIR,
                                       backtest_result=backtest_result, sweep=sweep or [])

    bundle: dict[str, str | Path] = {
        "txt_report": txt_rep,
        "txt_path": txt_path,
        "wechat_text": wechat_text,
        "wechat_path": wechat_path,
        "excel_path": excel_path,
    }

    # 4. 若具备回测结果，额外导出逐期回测明细 Excel
    if backtest_result and backtest_result.get("n_sample", 0) > 0:
        from calc_backtest import save_backtest_excel
        bundle["backtest_excel_path"] = save_backtest_excel(backtest_result, output_dir=EXCEL_DIR)

    return bundle
