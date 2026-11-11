# -*- coding: utf-8 -*-
"""金胆池消融实验 (Gold Pool Ablation Study)。

【实验目标】
现有 4 码金胆池在 30 期回测中仅 0.67x 命中（随机期望 1.0），需要回答：
1. 低命中是模型缺陷，还是 4 码小样本的统计噪声？
2. 金胆池的排序信号来自哪个分量（Transformer / Dirichlet / 共现谱 / 动量）？
3. 更换选择规则（Borda 秩融合、去重号、分区分散、奇偶平衡）能否改善？
4. 历史频率（热号/冷号）与"重号偏好"是否在拖后腿？

【方法】
严格 Walk-Forward：回测第 T 期时只用 1..T-1 期历史，训练一次 Transformer，
从同一组因果得分派生 17 种选择器，对比命中均值、Lift、P(>=2) 与排序 AUC。
AUC 定义为"随机抽一个开奖号与一个未开奖号，开奖号得分更高的概率"，随机=0.5。

@author czw
@since 2026-11-10
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from kl8_engine.constants import N_NUM
from kl8_engine.data import Dataset, load_dataset
from kl8_engine.models.spectral_bayesian import (
    combine_spectral_components,
    compute_cooccurrence_spectral_scores,
    compute_dirichlet_scores,
    compute_momentum_differential_scores,
)
from kl8_engine.models.transformer_net import train_and_predict_transformer

K = 4
DRAWN = 20
EXPECTED_GOLD = K * DRAWN / N_NUM  # 1.0


def top_k(scores: Sequence[float], k: int = K, exclude: set[int] | None = None) -> Tuple[int, ...]:
    """按得分降序取前 k 个号码（可选排除集合）。"""
    exclude = exclude or set()
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    picked = [int(i) + 1 for i in order if int(i) + 1 not in exclude][:k]
    return tuple(sorted(picked))


def rank_fusion(score_list: Sequence[Sequence[float]], k: int = K) -> Tuple[int, ...]:
    """Borda 秩融合：各分量按升序排名后取平均秩最小者。"""
    total = np.zeros(N_NUM, dtype=np.float64)
    for scores in score_list:
        total += np.argsort(np.argsort(np.asarray(scores, dtype=np.float64)))
    order = np.argsort(total / len(score_list))
    return tuple(sorted(int(i) + 1 for i in order[:k]))


def diverse_top_k(scores: Sequence[float], max_per_zone: int, k: int = K) -> Tuple[int, ...]:
    """分区分散约束下的贪心选号（四个 20 号分区，每区最多 max_per_zone 个）。"""
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    chosen: List[int] = []
    zone_counts = [0, 0, 0, 0]
    for idx in order:
        num = int(idx) + 1
        zone = (num - 1) // 20
        if zone_counts[zone] >= max_per_zone:
            continue
        chosen.append(num)
        zone_counts[zone] += 1
        if len(chosen) == k:
            break
    if len(chosen) < k:
        for idx in order:
            num = int(idx) + 1
            if num not in chosen:
                chosen.append(num)
            if len(chosen) == k:
                break
    return tuple(sorted(chosen))


def balanced_top_k(scores: Sequence[float], k: int = K) -> Tuple[int, ...]:
    """奇偶平衡约束下的贪心选号（奇数、偶数各最多 2 个）。"""
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    chosen: List[int] = []
    odd = even = 0
    for idx in order:
        num = int(idx) + 1
        if num % 2 == 1 and odd >= 2:
            continue
        if num % 2 == 0 and even >= 2:
            continue
        chosen.append(num)
        odd += num % 2
        even += 1 - num % 2
        if len(chosen) == k:
            break
    if len(chosen) < k:
        for idx in order:
            num = int(idx) + 1
            if num not in chosen:
                chosen.append(num)
            if len(chosen) == k:
                break
    return tuple(sorted(chosen))


def ranking_auc(scores: Sequence[float], drawn: Sequence[int]) -> float:
    """排序 AUC：随机取一个开奖号和一个未开奖号，开奖号得分更高的概率。"""
    ranks = np.argsort(np.argsort(np.asarray(scores, dtype=np.float64))) + 1  # 1..80 升序
    drawn_ranks = ranks[[n - 1 for n in drawn]]
    n1, n2 = len(drawn), N_NUM - len(drawn)
    return float((drawn_ranks.sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def normal_p_value(z: float) -> float:
    """双侧正态近似 p 值。"""
    return math.erfc(abs(z) / math.sqrt(2.0))


def hypergeometric_ge2_prob() -> float:
    """4 码池在 80 选 20 下至少命中 2 码的理论概率。"""
    total = math.comb(N_NUM, K)
    p0 = math.comb(N_NUM - DRAWN, K) / total
    p1 = math.comb(DRAWN, 1) * math.comb(N_NUM - DRAWN, K - 1) / total
    return 1.0 - p0 - p1


def binom_two_sided(x: int, n: int, p: float) -> float:
    """精确二项分布双侧 p 值。"""
    def tail(lo: int, hi: int) -> float:
        return sum(
            math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(lo, hi + 1)
        )

    if x < n * p:
        pval = 2.0 * tail(0, x)
    else:
        pval = 2.0 * tail(x, n)
    return min(1.0, pval)


def build_selectors(
    dl: np.ndarray,
    dirichlet: np.ndarray,
    cooc: np.ndarray,
    momentum: np.ndarray,
    spectral: np.ndarray,
    fused: np.ndarray,
    hot50: np.ndarray,
    latest_numbers: set[int],
    period: int,
) -> Dict[str, Tuple[int, ...]]:
    """从同一组因果得分派生全部候选选择器。"""
    rng = random.Random(2026000 + period)
    return {
        "fused_top4": top_k(fused),
        "dl_top4": top_k(dl),
        "dirichlet_top4": top_k(dirichlet),
        "cooc_top4": top_k(cooc),
        "momentum_top4": top_k(momentum),
        "spectral_top4": top_k(spectral),
        "borda4_top4": rank_fusion([dl, dirichlet, cooc, momentum]),
        "borda3_top4": rank_fusion([dirichlet, cooc, momentum]),
        "borda_dl_spectral_top4": rank_fusion([dl, spectral]),
        "hot50_top4": top_k(hot50),
        "cold50_top4": top_k(-hot50),
        "norepeat_fused_top4": top_k(fused, exclude=latest_numbers),
        "repeatonly_fused_top4": top_k(fused, exclude=set(range(1, N_NUM + 1)) - latest_numbers),
        "zone1_fused_top4": diverse_top_k(fused, max_per_zone=1),
        "zone2_fused_top4": diverse_top_k(fused, max_per_zone=2),
        "balanced_fused_top4": balanced_top_k(fused),
        "random_top4": tuple(sorted(rng.sample(range(1, N_NUM + 1), K))),
    }


def run_ablation(
    history_path: str,
    periods: int,
    epochs: int,
    seq_len: int,
    dl_weight: float,
    offset: int = 0,
) -> Dict[str, object]:
    """执行金胆池消融回测并返回结构化结果。

    @param offset 回测窗口距最新一期的偏移量（0=最近 N 期；60=再往前 60 期）
    """
    full_ds = load_dataset(history_path)
    total = len(full_ds.draws)
    start = total - periods - offset
    end = total - offset
    if start < 20:
        raise ValueError(
            f"历史数据不足：总 {total} 期，无法回测偏移 {offset} 的最近 {periods} 期"
        )

    records: List[Dict[str, object]] = []
    t_start = time.perf_counter()

    for idx in range(start, end):
        actual = full_ds.draws[idx]
        history = Dataset(draws=tuple(full_ds.draws[:idx]))
        matrix = history.to_matrix()
        step_t0 = time.perf_counter()

        dl, loss = train_and_predict_transformer(matrix, seq_len=seq_len, epochs=epochs)
        dirichlet = compute_dirichlet_scores(matrix, recent_window=35)
        cooc = compute_cooccurrence_spectral_scores(matrix, window_limit=150)
        momentum, omissions = compute_momentum_differential_scores(
            matrix, short_window=10, long_window=40
        )
        spectral = combine_spectral_components(dirichlet, cooc, momentum)
        fused = dl_weight * dl + (1.0 - dl_weight) * spectral
        hot50 = matrix[-min(len(matrix), 50):].mean(axis=0)
        latest_numbers = set(history.latest.numbers)

        selectors = build_selectors(
            dl, dirichlet, cooc, momentum, spectral, fused, hot50, latest_numbers, actual.period
        )
        actual_set = set(actual.numbers)
        hits = {name: len(set(pool) & actual_set) for name, pool in selectors.items()}

        score_sources = {
            "fused": fused,
            "dl": dl,
            "dirichlet": dirichlet,
            "cooccurrence": cooc,
            "momentum": momentum,
            "hot50": hot50,
        }
        auc = {name: ranking_auc(scores, actual.numbers) for name, scores in score_sources.items()}

        zone_counts = [0, 0, 0, 0]
        for num in selectors["fused_top4"]:
            zone_counts[(num - 1) // 20] += 1

        record = {
            "period": actual.period,
            "date": actual.date,
            "actual_numbers": list(actual.numbers),
            "selectors": {name: list(pool) for name, pool in selectors.items()},
            "hits": hits,
            "auc": auc,
            "scores": {
                name: [round(float(v), 6) for v in np.asarray(values, dtype=np.float64)]
                for name, values in score_sources.items()
            },
            "diagnostics": {
                "fused_repeat_count": len(set(selectors["fused_top4"]) & latest_numbers),
                "fused_max_zone_count": max(zone_counts),
                "fused_odd_count": sum(1 for n in selectors["fused_top4"] if n % 2 == 1),
                "fused_mean_omission": float(
                    np.mean([omissions[n - 1] for n in selectors["fused_top4"]])
                ),
                "dl_loss": float(loss),
            },
        }
        records.append(record)
        elapsed = time.perf_counter() - step_t0
        print(
            f"  [{actual.period}] 生产金胆 {hits['fused_top4']}/4 | "
            f"随机 {hits['random_top4']}/4 | AUC(fused) {auc['fused']:.3f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    total_time = time.perf_counter() - t_start
    return summarize(records, periods, epochs, seq_len, dl_weight, total_time, offset)


def summarize(
    records: List[Dict[str, object]],
    periods: int,
    epochs: int,
    seq_len: int,
    dl_weight: float,
    total_time: float,
    offset: int = 0,
) -> Dict[str, object]:
    """汇总选择器命中与排序质量统计。"""
    n = len(records)
    selector_names = list(records[0]["hits"].keys())  # type: ignore[union-attr]
    selector_summary: Dict[str, Dict[str, float]] = {}
    p_ge2 = hypergeometric_ge2_prob()

    for name in selector_names:
        hits = np.array([rec["hits"][name] for rec in records], dtype=np.float64)  # type: ignore[index]
        mean = float(hits.mean())
        sd = float(hits.std(ddof=1))
        se = sd / math.sqrt(n) if n > 1 else float("inf")
        z = (mean - EXPECTED_GOLD) / se if se > 0 else 0.0
        ge2_count = int((hits >= 2).sum())
        selector_summary[name] = {
            "mean_hits": mean,
            "sd": sd,
            "se": se,
            "lift": mean / EXPECTED_GOLD,
            "z_vs_random": z,
            "p_value": normal_p_value(z),
            "p_ge2_observed": ge2_count / n,
            "p_ge2_expected": p_ge2,
            "p_ge2_binom_p": binom_two_sided(ge2_count, n, p_ge2),
            "p_ge3_observed": int((hits >= 3).sum()) / n,
            "zero_rate": int((hits == 0).sum()) / n,
        }

    score_sources = list(records[0]["auc"].keys())  # type: ignore[union-attr]
    auc_summary: Dict[str, Dict[str, float]] = {}
    for name in score_sources:
        aucs = np.array([rec["auc"][name] for rec in records], dtype=np.float64)  # type: ignore[index]
        mean = float(aucs.mean())
        sd = float(aucs.std(ddof=1))
        se = sd / math.sqrt(n) if n > 1 else float("inf")
        z = (mean - 0.5) / se if se > 0 else 0.0
        auc_summary[name] = {
            "mean_auc": mean,
            "sd": sd,
            "se": se,
            "z_vs_random": z,
            "p_value": normal_p_value(z),
        }

    diag = {
        key: np.array([rec["diagnostics"][key] for rec in records], dtype=np.float64)  # type: ignore[index]
        for key in records[0]["diagnostics"].keys()  # type: ignore[union-attr]
    }
    fused_hits = np.array([rec["hits"]["fused_top4"] for rec in records], dtype=np.float64)  # type: ignore[index]
    diagnostics_summary: Dict[str, float] = {}
    for key, values in diag.items():
        if key == "dl_loss":
            diagnostics_summary[f"{key}_mean"] = float(values.mean())
            continue
        corr = float(np.corrcoef(values, fused_hits)[0, 1]) if values.std() > 0 else 0.0
        diagnostics_summary[f"{key}_mean"] = float(values.mean())
        diagnostics_summary[f"{key}_corr_with_hits"] = corr

    return {
        "config": {
            "periods": periods,
            "offset": offset,
            "epochs": epochs,
            "seq_len": seq_len,
            "dl_weight": dl_weight,
            "k": K,
            "expected_gold_hits": EXPECTED_GOLD,
            "p_ge2_expected": p_ge2,
            "elapsed_seconds": total_time,
            "bonferroni_alpha": 0.05 / max(len(selector_names), 1),
        },
        "records": records,
        "selector_summary": selector_summary,
        "auc_summary": auc_summary,
        "diagnostics_summary": diagnostics_summary,
    }


def write_markdown(result: Dict[str, object], path: Path) -> None:
    """写出消融实验 Markdown 报告。"""
    cfg = result["config"]  # type: ignore[index]
    selector_summary = result["selector_summary"]  # type: ignore[index]
    auc_summary = result["auc_summary"]  # type: ignore[index]
    diagnostics = result["diagnostics_summary"]  # type: ignore[index]
    records = result["records"]  # type: ignore[index]

    lines = [
        "# 金胆池消融实验报告",
        "",
        f"- 生成日期: {date.today().isoformat()}",
        f"- 回测区间: 最近 {cfg['periods']} 期（偏移 {cfg.get('offset', 0)} 期，严格 Walk-Forward 因果切片）",
        f"- Transformer: epochs={cfg['epochs']}, seq_len={cfg['seq_len']}, 融合权重 dl={cfg['dl_weight']}",
        f"- 随机期望: 4 码金胆 {cfg['expected_gold_hits']:.2f} 码/期; P(>=2 码)={cfg['p_ge2_expected']:.3f}",
        f"- Bonferroni 阈值: {cfg['bonferroni_alpha']:.4f}",
        f"- 总耗时: {cfg['elapsed_seconds']:.1f}s",
        "",
        "## 一、选择器命中汇总（按平均命中降序）",
        "",
        "| 选择器 | 平均命中 | Lift | z vs 随机 | p 值 | P(>=2) | 二项 p | 0 命中率 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in sorted(
        selector_summary.items(), key=lambda item: item[1]["mean_hits"], reverse=True
    ):
        lines.append(
            f"| {name} | {stats['mean_hits']:.3f} | {stats['lift']:.3f}x | "
            f"{stats['z_vs_random']:+.2f} | {stats['p_value']:.4f} | "
            f"{stats['p_ge2_observed']:.3f} | {stats['p_ge2_binom_p']:.4f} | "
            f"{stats['zero_rate']:.3f} |"
        )

    lines.extend([
        "",
        "## 二、排序质量 AUC（随机=0.5）",
        "",
        "| 得分源 | 平均 AUC | z vs 0.5 | p 值 |",
        "| :--- | ---: | ---: | ---: |",
    ])
    for name, stats in sorted(auc_summary.items(), key=lambda item: item[1]["mean_auc"], reverse=True):
        lines.append(
            f"| {name} | {stats['mean_auc']:.4f} | {stats['z_vs_random']:+.2f} | {stats['p_value']:.4f} |"
        )

    lines.extend([
        "",
        "## 三、金胆池结构诊断（与命中数的相关系数）",
        "",
        "| 诊断项 | 均值 | 与命中数相关系数 |",
        "| :--- | ---: | ---: |",
    ])
    for key in sorted(diagnostics):
        if key.endswith("_mean"):
            base = key[:-5]
            corr = diagnostics.get(f"{base}_corr_with_hits")
            corr_str = f"{corr:+.3f}" if corr is not None else "-"
            lines.append(f"| {base} | {diagnostics[key]:.3f} | {corr_str} |")

    lines.extend([
        "",
        "## 四、逐期明细（前 20 期）",
        "",
        "| 期号 | 生产金胆 | 命中 | 随机基线命中 | AUC(fused) |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ])
    for rec in records[:20]:
        lines.append(
            f"| {rec['period']} | {' '.join(f'{n:02d}' for n in rec['selectors']['fused_top4'])} | "
            f"{rec['hits']['fused_top4']} | {rec['hits']['random_top4']} | {rec['auc']['fused']:.3f} |"
        )

    lines.extend([
        "",
        "## 五、说明",
        "",
        "- 本报告由 `scripts/ablation_gold.py` 自动生成；原始逐期数据见同名 JSON。",
        "- AUC 衡量的是 80 码整体排序质量，比 4 码命中更少受小样本噪声影响：AUC 显著大于 0.5 才说明模型有排序信号。",
        "- 所有 p 值为双侧；多选择器比较需按 Bonferroni 阈值判断，单次 p<0.05 不足以宣称有效。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="金胆池消融实验")
    parser.add_argument("--history", default="kl8_history.txt")
    parser.add_argument("--periods", type=int, default=60)
    parser.add_argument("--offset", type=int, default=0, help="回测窗口距最新一期的偏移量")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--dl-weight", type=float, default=0.55)
    parser.add_argument("--output-dir", default="output/reports")
    args = parser.parse_args()

    result = run_ablation(
        args.history, args.periods, args.epochs, args.seq_len, args.dl_weight, args.offset
    )

    stamp = date.today().strftime("%Y%m%d")
    tag = f"_offset{args.offset}" if args.offset else ""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"金胆池消融实验_{stamp}{tag}.md"
    json_path = out_dir / f"金胆池消融实验_{stamp}{tag}.json"
    write_markdown(result, md_path)
    json_path.write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("  金胆池消融实验汇总（按平均命中降序）")
    print("=" * 78)
    print(f"  {'选择器':<28} {'均值':>6} {'Lift':>7} {'z':>7} {'p':>8} {'P(>=2)':>8}")
    for name, stats in sorted(
        result["selector_summary"].items(), key=lambda item: item[1]["mean_hits"], reverse=True
    ):
        print(
            f"  {name:<28} {stats['mean_hits']:>6.3f} {stats['lift']:>6.2f}x "
            f"{stats['z_vs_random']:>+7.2f} {stats['p_value']:>8.4f} {stats['p_ge2_observed']:>8.3f}"
        )
    print("\n  排序质量 AUC（随机=0.5）:")
    for name, stats in sorted(result["auc_summary"].items(), key=lambda item: item[1]["mean_auc"], reverse=True):
        print(
            f"  {name:<28} AUC={stats['mean_auc']:.4f} z={stats['z_vs_random']:+.2f} "
            f"p={stats['p_value']:.4f}"
        )
    print(f"\n  报告: {md_path}")
    print(f"  数据: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())