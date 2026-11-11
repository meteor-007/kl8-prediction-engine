# -*- coding: utf-8 -*-
"""跨窗口汇总金胆池消融结果，并验证 Borda 全排序对主选/大底的影响。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def paired(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(len(diff))
    z = mean / se if se > 0 else 0.0
    return mean, normal_p(z)


def borda_order(scores: Dict[str, Sequence[float]]) -> np.ndarray:
    total = np.zeros(80, dtype=np.float64)
    for key in ("dl", "dirichlet", "cooccurrence", "momentum"):
        total += np.argsort(np.argsort(np.asarray(scores[key], dtype=np.float64)))
    return np.argsort(total)


def fused_order(scores: Dict[str, Sequence[float]]) -> np.ndarray:
    return np.argsort(-np.asarray(scores["fused"], dtype=np.float64))


def top_hits(order: np.ndarray, k: int, actual: set[int]) -> int:
    picked = {int(i) + 1 for i in order[:k]}
    return len(picked & actual)


def main() -> None:
    base = Path("output/reports")
    files = [
        base / "金胆池消融实验_20261110.json",
        base / "金胆池消融实验_20261110_offset60.json",
        base / "金胆池消融实验_20261110_offset120.json",
    ]

    all_borda, all_fused, all_rand = [], [], []
    windows: List[dict] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        recs = data["records"]
        b = [r["hits"]["borda4_top4"] for r in recs]
        f = [r["hits"]["fused_top4"] for r in recs]
        rnd = [r["hits"]["random_top4"] for r in recs]
        all_borda += b
        all_fused += f
        all_rand += rnd
        windows.append(
            {
                "path": path.name,
                "period_range": f"{recs[0]['period']}-{recs[-1]['period']}",
                "borda_mean": float(np.mean(b)),
                "fused_mean": float(np.mean(f)),
                "random_mean": float(np.mean(rnd)),
                "borda_minus_fused": paired(b, f),
                "borda_minus_random": paired(b, rnd),
            }
        )

    pooled_bf = paired(all_borda, all_fused)
    pooled_br = paired(all_borda, all_rand)
    print("== 跨窗口汇总（3 × 60 = 180 期）==")
    for w in windows:
        print(
            f"{w['period_range']}: Borda {w['borda_mean']:.3f} | fused {w['fused_mean']:.3f} | "
            f"random {w['random_mean']:.3f} | Δ(B-F)={w['borda_minus_fused'][0]:+.3f} "
            f"p={w['borda_minus_fused'][1]:.4f}"
        )
    print(
        f"合并 180 期: Borda {np.mean(all_borda):.3f} | fused {np.mean(all_fused):.3f} | "
        f"random {np.mean(all_rand):.3f}"
    )
    print(
        f"合并 Δ(Borda-Fused)={pooled_bf[0]:+.3f}, p={pooled_bf[1]:.4f} | "
        f"Δ(Borda-Random)={pooled_br[0]:+.3f}, p={pooled_br[1]:.4f}"
    )

    print("\n== Borda 全排序 vs fused 全排序（仅含 scores 的窗口 2/3）==")
    for path in files[1:]:
        data = json.loads(path.read_text(encoding="utf-8"))
        recs = data["records"]
        if "scores" not in recs[0]:
            continue
        stats = {
            "borda_main": [], "fused_main": [],
            "borda_top20": [], "fused_top20": [],
            "borda_pool30": [], "fused_pool30": [],
        }
        auc_borda = []
        for rec in recs:
            actual = set(rec["actual_numbers"])
            bo = borda_order(rec["scores"])
            fo = fused_order(rec["scores"])
            stats["borda_main"].append(top_hits(bo, 12, actual))
            stats["fused_main"].append(top_hits(fo, 12, actual))
            stats["borda_top20"].append(top_hits(bo, 20, actual))
            stats["fused_top20"].append(top_hits(fo, 20, actual))
            stats["borda_pool30"].append(top_hits(bo, 30, actual))
            stats["fused_pool30"].append(top_hits(fo, 30, actual))
            ranks = np.argsort(np.argsort(bo)) + 1
            drawn_ranks = ranks[[n - 1 for n in actual]]
            auc_borda.append((drawn_ranks.sum() - 210) / (20 * 60))
        print(f"\n{path.name} ({recs[0]['period']}-{recs[-1]['period']})")
        for pool in ("main", "top20", "pool30"):
            b_mean = float(np.mean(stats[f"borda_{pool}"]))
            f_mean = float(np.mean(stats[f"fused_{pool}"]))
            diff, p = paired(stats[f"borda_{pool}"], stats[f"fused_{pool}"])
            print(
                f"  {pool:<7} Borda {b_mean:.3f} | fused {f_mean:.3f} | "
                f"Δ={diff:+.3f} p={p:.4f}"
            )
        print(f"  Borda 排序 AUC: {np.mean(auc_borda):.4f}")


if __name__ == "__main__":
    main()