# -*- coding: utf-8 -*-
"""快乐8 (Keno) 新引擎前瞻实测走查回测系统 (Walk-Forward Backtesting Engine)。

【业务背景与架构定位】
严格遵循因果律（Zero Lookahead Bias），以滑动时间窗口对历史真实期次进行逐期回测：
1. 确定性因果切片：回测第 T 期时，模型仅能获取截至第 T-1 期的历史数据；
2. 评测目标：
   - 4 码金胆池（理论期望 1.0 码）
   - 12 码主选池（理论期望 3.0 码）
   - 30 码黄金大底（理论期望 7.5 码）
   - Top 20 号码（理论期望 5.0 码）
3. 计算 Lift（增益倍数 = 实际均值 / 理论期望）与胜率分布。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# 将项目根目录加入模块搜索路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kl8_engine.data import Dataset, Draw, load_dataset
from kl8_engine.models.ensemble import run_hybrid_ensemble
from kl8_engine.models.spectral_bayesian import compute_spectral_bayesian_scores


def evaluate_slice(
    history_draws: List[Draw],
    actual_draw: Draw,
    use_transformer: bool = True,
    epochs: int = 12,
    dl_weight: float = 0.55,
) -> Dict[str, object]:
    """对单期执行因果预测与真实命中核算。"""
    sub_dataset = Dataset(draws=tuple(history_draws))
    matrix = sub_dataset.to_matrix()

    if use_transformer:
        res = run_hybrid_ensemble(
            sub_dataset,
            dl_weight=dl_weight,
            seq_len=10,
            epochs=epochs,
        )
        gold_pool = res.gold_pool
        main_pool = res.main_pool
        pool_30 = res.shrinkage_pool_30
        top_20 = res.full_order[:20]
    else:
        spectral_scores, _ = compute_spectral_bayesian_scores(matrix)
        order_0based = np.argsort(-spectral_scores)
        full_order = tuple(int(idx + 1) for idx in order_0based)
        gold_pool = full_order[:4]
        main_pool = full_order[:12]
        pool_30 = full_order[:30]
        top_20 = full_order[:20]

    actual_set = set(actual_draw.numbers)
    gold_hits = len(set(gold_pool) & actual_set)
    main_hits = len(set(main_pool) & actual_set)
    pool30_hits = len(set(pool_30) & actual_set)
    top20_hits = len(set(top_20) & actual_set)

    return {
        "period": actual_draw.period,
        "date": actual_draw.date,
        "gold_pool": gold_pool,
        "gold_hits": gold_hits,
        "main_hits": main_hits,
        "pool30_hits": pool30_hits,
        "top20_hits": top20_hits,
        "actual_numbers": actual_draw.numbers,
    }


def run_walk_forward_backtest(
    history_path: str = "kl8_history.txt",
    test_periods: int = 20,
    use_transformer: bool = True,
    epochs: int = 12,
) -> None:
    """执行滑动窗口回测并打印详尽统计报告。"""
    print(f"================================================================================")
    print(f"  快乐8 新引擎前瞻实测回测 (Walk-Forward Historical Backtest)")
    print(f"  模式: {'混合集成 (Transformer + 谱贝叶斯)' if use_transformer else '谱贝叶斯基线'}")
    print(f"  回测期数: 最近 {test_periods} 期 | 历史因果切片防未来函数: 100% 启用")
    print(f"================================================================================\n")

    full_ds = load_dataset(history_path)
    total_draws = len(full_ds.draws)
    if total_draws < test_periods + 20:
        raise ValueError("历史开奖数据不足，无法回测！")

    start_idx = total_draws - test_periods
    records: List[Dict[str, object]] = []

    t_start = time.perf_counter()
    for idx in range(start_idx, total_draws):
        actual_draw = full_ds.draws[idx]
        history_draws = list(full_ds.draws[:idx])
        step_t0 = time.perf_counter()
        rec = evaluate_slice(
            history_draws,
            actual_draw,
            use_transformer=use_transformer,
            epochs=epochs,
        )
        step_elapsed = time.perf_counter() - step_t0
        rec["elapsed"] = step_elapsed
        records.append(rec)
        print(
            f"  [期次 {rec['period']}] "
            f"金胆(4码): {rec['gold_hits']}/4 | "
            f"主选(12码): {rec['main_hits']}/12 | "
            f"大底(30码): {rec['pool30_hits']}/30 | "
            f"Top20: {rec['top20_hits']}/20 "
            f"({step_elapsed:.1f}s)"
        )

    total_time = time.perf_counter() - t_start

    # 汇总计算
    n = len(records)
    gold_hit_list = [r["gold_hits"] for r in records]
    main_hit_list = [r["main_hits"] for r in records]
    pool30_hit_list = [r["pool30_hits"] for r in records]
    top20_hit_list = [r["top20_hits"] for r in records]

    mean_gold = float(np.mean(gold_hit_list))
    mean_main = float(np.mean(main_hit_list))
    mean_pool30 = float(np.mean(pool30_hit_list))
    mean_top20 = float(np.mean(top20_hit_list))

    lift_gold = mean_gold / 1.00  # 4 * (20/80) = 1.0
    lift_main = mean_main / 3.00  # 12 * (20/80) = 3.0
    lift_pool30 = mean_pool30 / 7.50  # 30 * (20/80) = 7.5
    lift_top20 = mean_top20 / 5.00  # 20 * (20/80) = 5.0

    print("\n" + "=" * 80)
    print(f"                       【回测汇总成绩单 (最近 {n} 期)】")
    print("=" * 80)
    print(f" 评测指标            理论随机期望     实测均值命中     增益倍数 (Lift)     极值范围 [Min, Max]")
    print("-" * 80)
    print(f" 4码金胆池 (Gold)        1.00 个         {mean_gold:.2f} 个         {lift_gold:.2f}x ({(lift_gold-1)*100:+.1f}%)        [{min(gold_hit_list)}, {max(gold_hit_list)}]")
    print(f" 12码主选池 (Main)       3.00 个         {mean_main:.2f} 个         {lift_main:.2f}x ({(lift_main-1)*100:+.1f}%)        [{min(main_hit_list)}, {max(main_hit_list)}]")
    print(f" 20码核心层 (Top20)      5.00 个         {mean_top20:.2f} 个         {lift_top20:.2f}x ({(lift_top20-1)*100:+.1f}%)        [{min(top20_hit_list)}, {max(top20_hit_list)}]")
    print(f" 30码黄金大底 (Pool30)   7.50 个         {mean_pool30:.2f} 个         {lift_pool30:.2f}x ({(lift_pool30-1)*100:+.1f}%)        [{min(pool30_hit_list)}, {max(pool30_hit_list)}]")
    print("-" * 80)

    # 4码金胆命中分布
    gold_ge_1 = sum(1 for h in gold_hit_list if h >= 1) / n * 100
    gold_ge_2 = sum(1 for h in gold_hit_list if h >= 2) / n * 100
    gold_ge_3 = sum(1 for h in gold_hit_list if h >= 3) / n * 100
    print(f" 【4码金胆分布】至少中1码概率: {gold_ge_1:.1f}% (理论68.4%) | 至少中2码概率: {gold_ge_2:.1f}% (理论26.2%) | 命中3码+: {gold_ge_3:.1f}%")

    # 30码大底命中分布
    p30_ge_8 = sum(1 for h in pool30_hit_list if h >= 8) / n * 100
    p30_ge_9 = sum(1 for h in pool30_hit_list if h >= 9) / n * 100
    p30_ge_10 = sum(1 for h in pool30_hit_list if h >= 10) / n * 100
    print(f" 【30码大底分布】中8码+概率: {p30_ge_8:.1f}% | 中9码+概率: {p30_ge_9:.1f}% | 中10码+概率: {p30_ge_10:.1f}% (最高单期命中 {max(pool30_hit_list)} 码)")
    print(f" 总耗时: {total_time:.1f} 秒 (平均每期 {total_time/n:.1f}s)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="快乐8新引擎回测工具")
    parser.add_argument("--periods", type=int, default=15, help="回测期数 (默认 15)")
    parser.add_argument("--epochs", type=int, default=10, help="Transformer 轮数 (默认 10)")
    parser.add_argument("--spectral-only", action="store_true", help="仅运行谱贝叶斯")
    args = parser.parse_args()

    run_walk_forward_backtest(
        test_periods=args.periods,
        use_transformer=not args.spectral_only,
        epochs=args.epochs,
    )
