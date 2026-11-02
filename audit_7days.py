# -*- coding: utf-8 -*-
"""最近 7 期滚动窗口严格盲测核算脚本"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from calc_core import load_dataset, TimeSeries, truncated
from calc_engine import run_calculus_engine

ds = load_dataset()
T = ds.T
print(f"本地数据集总期数: {T} (最新已开奖: {ds.draws[-1].period})")
recent_7 = ds.draws[-7:]

results = []
for d in recent_7:
    target_p = d.period
    actual_nums = sorted(d.numbers)
    # 严格截断到 target_p 之前 (纯盲测，零未来泄漏)
    tds = truncated(ds, target_p)
    ts = TimeSeries(tds)
    res = run_calculus_engine(tds, ts)

    dyn_picks = sorted(res.get("dynamic_picks") or [])
    p16 = sorted(res.get("picks16") or [])
    p24 = sorted(res.get("picks24") or [])

    h_dyn = sorted(set(dyn_picks) & set(actual_nums))
    h_16 = sorted(set(p16) & set(actual_nums))
    h_24 = sorted(set(p24) & set(actual_nums))

    results.append({
        "period": d.period,
        "date": d.date,
        "actual": actual_nums,
        "dyn_len": len(dyn_picks),
        "dyn_picks": dyn_picks,
        "h_dyn": h_dyn,
        "p16": p16,
        "h_16": h_16,
        "p24": p24,
        "h_24": h_24,
    })

print("=" * 80)
print("  最近 7 天 (7期) 严格滚动窗口盲测 (Walk-Forward) 逐期详细核算")
print("=" * 80)
for r in results:
    print(f"【第 {r['period']} 期 · {r['date']}】")
    print(f"  · 官方真实开出 (20码): {' '.join(f'{n:02d}' for n in r['actual'])}")
    print(f"  · ⚡【打法A】动态大底 ({r['dyn_len']}码): 命中 {len(r['h_dyn'])} 码 -> {' '.join(f'{n:02d}' for n in r['h_dyn']) if r['h_dyn'] else '未中'}")
    print(f"     推荐名单: {' '.join(f'{n:02d}' for n in r['dyn_picks'])}")
    print(f"  · 🔥【基准16码】: 命中 {len(r['h_16'])} 码 -> {' '.join(f'{n:02d}' for n in r['h_16']) if r['h_16'] else '未中'}")
    print(f"     推荐名单: {' '.join(f'{n:02d}' for n in r['p16'])}")
    print(f"  · 🌟【豪华24码大底】: 命中 {len(r['h_24'])} 码 -> {' '.join(f'{n:02d}' for n in r['h_24']) if r['h_24'] else '未中'}")
    print("-" * 80)

# 统计汇总
dyn_hits = [len(r['h_dyn']) for r in results]
base_hits = [len(r['h_16']) for r in results]
p24_hits = [len(r['h_24']) for r in results]

print("\n【7 天盲测指标统计汇总】:")
print(f"  · ⚡【打法A动态大底】平均命中: {sum(dyn_hits)/7:.2f} 码/期 ｜ 命中分布: {dyn_hits} ｜ 最高单期: {max(dyn_hits)} 码")
print(f"    · >=4 码场次: {sum(1 for h in dyn_hits if h >= 4)}/7 期 ({sum(1 for h in dyn_hits if h >= 4)/7*100:.1f}%)")
print(f"    · >=5 码场次: {sum(1 for h in dyn_hits if h >= 5)}/7 期 ({sum(1 for h in dyn_hits if h >= 5)/7*100:.1f}%)")
print(f"  · 🔥【基准16码】平均命中: {sum(base_hits)/7:.2f} 码/期 ｜ 命中分布: {base_hits} ｜ 最高单期: {max(base_hits)} 码")
print(f"  · 🌟【豪华24码】平均命中: {sum(p24_hits)/7:.2f} 码/期 ｜ 命中分布: {p24_hits} ｜ 最高单期: {max(p24_hits)} 码")
