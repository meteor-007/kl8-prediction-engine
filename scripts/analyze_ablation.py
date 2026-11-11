# -*- coding: utf-8 -*-
"""金胆池消融实验的二次诊断分析。

读取 ablation_gold.py 产出的 JSON，补充：
1. 金胆池粘滞性（连续 Jaccard、完全相同连续段、有效样本量）；
2. 选择器与随机基线/生产基线的配对检验；
3. 前后 30 期分半稳定性；
4. 金胆池与热号/冷号/重号选择器的重合结构。

@author czw
@since 2026-11-10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def lag1_autocorr(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 3 or arr.std() == 0:
        return 0.0
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """配对均值差的 z 检验（a-b）。"""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else float("inf")
    se = sd / math.sqrt(n) if n > 1 else float("inf")
    z = mean / se if se > 0 else 0.0
    return {"mean_diff": mean, "sd_diff": sd, "se": se, "z": z, "p": normal_p(z)}


def analyze(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: List[dict] = data["records"]
    n = len(records)
    selector_summary = data["selector_summary"]
    auc_summary = data["auc_summary"]

    hits = {
        name: np.array([rec["hits"][name] for rec in records], dtype=np.float64)
        for name in selector_summary
    }
    fused_hits = hits["fused_top4"]
    random_hits = hits["random_top4"]

    lines = [
        "# 金胆池消融实验：二次诊断分析",
        "",
        f"- 样本: {n} 期（严格 Walk-Forward）",
        f"- 数据来源: `{path.as_posix()}`",
        "",
        "## 一、金胆池粘滞性诊断",
        "",
    ]

    pools = [rec["selectors"]["fused_top4"] for rec in records]
    turnovers = [4 - len(set(pools[i]) & set(pools[i - 1])) for i in range(1, n)]
    jaccards = [jaccard(pools[i - 1], pools[i]) for i in range(1, n)]

    # 完全相同池的连续段
    runs: List[int] = []
    current = 1
    for i in range(1, n):
        if set(pools[i]) == set(pools[i - 1]):
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)

    # 每个号码进入金胆池后的最长连续停留
    max_streak: Dict[int, int] = {}
    current_streak: Dict[int, int] = {}
    for pool in pools:
        pool_set = set(pool)
        for num in list(current_streak.keys()):
            if num not in pool_set:
                current_streak.pop(num, None)
        for num in pool_set:
            current_streak[num] = current_streak.get(num, 0) + 1
            max_streak[num] = max(max_streak.get(num, 0), current_streak[num])

    distinct_pools = len({tuple(sorted(p)) for p in pools})
    distinct_numbers = len({num for pool in pools for num in pool})
    hit_rho = lag1_autocorr(fused_hits.tolist())
    n_eff = n * (1 - hit_rho) / (1 + hit_rho) if hit_rho > -1 else float("inf")

    lines.extend([
        f"- 不同金胆池数量: **{distinct_pools} / {n}** 期（重复率 {1 - distinct_pools / n:.1%}）",
        f"- 平均连续两期 Jaccard 相似度: **{np.mean(jaccards):.3f}**（连续两期平均换号 {np.mean(turnovers):.2f} / 4 码）",
        f"- 完全相同池的最长连续段: **{max(runs)} 期**；连续段长度分布: {sorted(runs, reverse=True)[:10]}",
        f"- 全窗口出现过的不同号码数: **{distinct_numbers} / 80**",
        f"- 单个号码连续停留在金胆池的最长纪录: **{max(max_streak.values())} 期**（号码 {max(max_streak, key=max_streak.get)}）",
        f"- 金胆命中数的 lag-1 自相关: **{hit_rho:+.3f}**（若显著为正，60 期的有效独立样本约 **{n_eff:.1f}** 期）",
        "",
        "## 二、配对检验（同一期、同一模型，消除期际波动）",
        "",
        "| 对比 | 平均差 | z | p 值 |",
        "| :--- | ---: | ---: | ---: |",
    ])

    for name in selector_summary:
        if name in ("fused_top4", "random_top4"):
            continue
        vs_random = paired_test(hits[name], random_hits)
        vs_fused = paired_test(hits[name], fused_hits)
        lines.append(
            f"| {name} vs 随机 | {vs_random['mean_diff']:+.3f} | {vs_random['z']:+.2f} | {vs_random['p']:.4f} |"
        )
        lines.append(
            f"| {name} vs 生产 fused | {vs_fused['mean_diff']:+.3f} | {vs_fused['z']:+.2f} | {vs_fused['p']:.4f} |"
        )

    lines.extend([
        "",
        "## 三、分半稳定性（前 30 期 vs 后 30 期）",
        "",
        "| 选择器 | 前半均值 | 后半均值 | 前半 Lift | 后半 Lift |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    half = n // 2
    for name in selector_summary:
        first = float(hits[name][:half].mean())
        second = float(hits[name][half:].mean())
        lines.append(f"| {name} | {first:.3f} | {second:.3f} | {first:.2f}x | {second:.2f}x |")

    lines.extend([
        "",
        "| 得分源 | 前半 AUC | 后半 AUC |",
        "| :--- | ---: | ---: |",
    ])
    for name in auc_summary:
        first = float(np.mean([rec["auc"][name] for rec in records[:half]]))
        second = float(np.mean([rec["auc"][name] for rec in records[half:]]))
        lines.append(f"| {name} | {first:.4f} | {second:.4f} |")

    lines.extend([
        "",
        "## 四、金胆池成分结构（与其它选择器的平均重合数，最多 4）",
        "",
        "| 对比选择器 | 平均重合码数 |",
        "| :--- | ---: |",
    ])
    for other in ["hot50_top4", "cold50_top4", "dirichlet_top4", "cooc_top4", "dl_top4", "repeatonly_fused_top4", "norepeat_fused_top4"]:
        overlaps = [
            len(set(rec["selectors"]["fused_top4"]) & set(rec["selectors"][other]))
            for rec in records
        ]
        lines.append(f"| {other} | {np.mean(overlaps):.2f} |")

    lines.extend([
        "",
        "## 五、解读提示",
        "",
        "- 若连续 Jaccard 很高且完全相同池连续多期，说明金胆池并非逐期重新优化，"
        "而是被慢变分量锁死；此时 60 期命中数的有效样本远小于 60。",
        "- 配对检验比“各自对 1.0 做单样本检验”更敏感，因为它消除了每期开奖整体难易程度的波动。",
        "- 分半稳定性用于判断某个选择器的优势是否只出现在半个窗口；只在一半窗口有效的差异不应采纳。",
    ])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/reports/金胆池消融实验_20261110.json")
    parser.add_argument("--output", default="output/reports/金胆池消融实验_20261110_诊断.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    report = analyze(input_path)
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写入: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())