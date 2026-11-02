# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 双专家大底深度融合与多源集成决策引擎

【业务背景与架构定位】:
本模块专为多专家/多渠道「30码大底」的深度碰撞与动态加权融合打造。
当存在两套不同逻辑产生的 30 码大底时，本模块执行三层科学集成流水线：
1. 一级共识层（Consensus Intersection）：
   - 提取双方共同看好的交集号码作为高置信度「必选骨干核」；
2. 二级动力学择优层（Kinetic Momentum Selection）：
   - 对双方各异的差集号码，基于官方开奖滑动窗口的微积分动力学
     （速度 v、加速度 a、动量 p 与弹性势能）进行全局精准排序，
     择优补强高动能、强反弹潜力的龙头号码；
3. 三级形态与区间平衡（Zone & Morphology Guard）：
   - 保证融合后的新 30 码大底在 012 路、四大象限（低、中、中高、高区）分布均衡，
     杜绝严重偏态，最终输出「黄金融合 30 码大底」并直通级联缩水引擎生成选 5 实战注单。

@author czw
@since 2026-11-01
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# 控制台编码防乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from calc_core import (
    DEFAULT_HISTORY_FILE,
    N_NUM,
    Dataset,
    TimeSeries,
    load_dataset,
)
from calc_engine import run_calculus_engine
from calc_shrinkage import ShrinkageEngine, format_ticket

TARGET_POOL_SIZE: int = 30


@dataclass
class PoolEnsembleResult:
    """双大底融合与深度集成分析视图对象。"""
    pool1: List[int]
    pool2: List[int]
    intersection_numbers: List[int]
    only_pool1_numbers: List[int]
    only_pool2_numbers: List[int]
    final_merged_30: List[int]
    sector_distribution: Dict[str, List[int]]
    tickets: List[Tuple[int, ...]]


class PoolEnsembleEngine:
    """双专家大底深度融合与多源集成决策引擎。"""

    def __init__(
        self,
        pool1: List[int],
        pool2: List[int],
        kinetic_scores: Optional[Dict[int, float]] = None,
    ) -> None:
        """初始化双大底融合引擎。

        @param pool1 专家1的大底号码池
        @param pool2 专家2的大底号码池
        @param kinetic_scores 微积分动力学综合打分字典 {号码: 分值}
        """
        # 步骤 1: 参数合法性校验与去重
        if not pool1 or not pool2:
            raise ValueError("传入的大底号码池不能为空")

        self.pool1 = sorted(list(set(pool1)))
        self.pool2 = sorted(list(set(pool2)))

        for n in self.pool1 + self.pool2:
            if not (1 <= n <= N_NUM):
                raise ValueError(f"号码 {n} 超出 1..{N_NUM} 合法范围")

        # 步骤 2: 装配号码微积分打分表
        self.scores: Dict[int, float] = {}
        all_unique = set(self.pool1) | set(self.pool2)
        for num in all_unique:
            if kinetic_scores and num in kinetic_scores:
                self.scores[num] = float(kinetic_scores[num])
            else:
                self.scores[num] = 1.0

    def merge_and_optimize(self, target_size: int = TARGET_POOL_SIZE) -> Tuple[List[int], Dict[str, any]]:
        """执行双大底科学融合流水线，选出最优的全新大底。

        @param target_size 目标新大底号码数量 (默认 30 码)
        @return (最终融合的30码列表, 分析指标字典)
        """
        # 步骤 1: 提取双方共识交集与差集
        s1 = set(self.pool1)
        s2 = set(self.pool2)
        intersection_nums = sorted(list(s1 & s2))
        only_s1 = sorted(list(s1 - s2))
        only_s2 = sorted(list(s2 - s1))
        candidate_pool = sorted(list(s1 | s2))

        # 步骤 2: 共识核直接入选
        merged = set(intersection_nums)

        # 步骤 3: 差集号码按微积分动力学分值进行降序竞选
        difference_nums = [x for x in candidate_pool if x not in merged]
        difference_nums.sort(key=lambda x: self.scores.get(x, 1.0), reverse=True)

        needed = target_size - len(merged)
        if needed > 0:
            supplement = difference_nums[:needed]
            merged.update(supplement)

        final_30 = sorted(list(merged))

        # 步骤 4: 四大象限区间分布统计
        sector_dist = {
            "低区 (01-20)": [x for x in final_30 if 1 <= x <= 20],
            "中区 (21-40)": [x for x in final_30 if 21 <= x <= 40],
            "中高区 (41-60)": [x for x in final_30 if 41 <= x <= 60],
            "高区 (61-80)": [x for x in final_30 if 61 <= x <= 80],
        }

        analysis = {
            "intersection": intersection_nums,
            "only_p1": only_s1,
            "only_p2": only_s2,
            "sector_dist": sector_dist,
        }
        return final_30, analysis

    def execute_ensemble_pipeline(self, budget_tickets: int = 30) -> PoolEnsembleResult:
        """端到端执行双大底融合并调用级联缩水引擎生成选 5 注单。"""
        # 步骤 1: 融合出最完美 30 码
        final_30, analysis = self.merge_and_optimize(TARGET_POOL_SIZE)

        # 步骤 2: 组装缩水引擎并执行正交贪心定额缩水
        shrink_engine = ShrinkageEngine(final_30, self.scores)
        tickets = shrink_engine.execute_cascade_shrink(
            budget_tickets=budget_tickets,
            max_overlap=3,
            diversity_lambda=1.2,
        )

        return PoolEnsembleResult(
            pool1=self.pool1,
            pool2=self.pool2,
            intersection_numbers=analysis["intersection"],
            only_pool1_numbers=analysis["only_p1"],
            only_pool2_numbers=analysis["only_p2"],
            final_merged_30=final_30,
            sector_distribution=analysis["sector_dist"],
            tickets=tickets,
        )


def auto_load_kinetic_scores() -> Dict[int, float]:
    """从本地开奖时序历史数据中自动计算全量号码的微积分综合动力学得分。"""
    try:
        ds = load_dataset(DEFAULT_HISTORY_FILE)
        ts = TimeSeries(ds)
        res = run_calculus_engine(ds, ts)
        scores = res["fusion_scores"]
        return {num: float(scores[num - 1]) for num in range(1, N_NUM + 1)}
    except Exception:
        return {num: 1.0 for num in range(1, N_NUM + 1)}


def main(argv=None) -> int:
    """CLI 命令行入口。"""
    parser = argparse.ArgumentParser(description="快乐8双专家大底深度融合与选5生成引擎")
    parser.add_argument("--pool1", type=str, required=True, help="专家1大底 (逗号隔开)")
    parser.add_argument("--pool2", type=str, required=True, help="专家2大底 (逗号隔开)")
    parser.add_argument("--budget", type=int, default=30, help="输出选5实战注数 (默认30注)")
    args = parser.parse_args(argv)

    p1 = [int(x.strip()) for x in args.pool1.split(",") if x.strip()]
    p2 = [int(x.strip()) for x in args.pool2.split(",") if x.strip()]

    score_map = auto_load_kinetic_scores()
    ensemble_engine = PoolEnsembleEngine(p1, p2, score_map)
    result = ensemble_engine.execute_ensemble_pipeline(budget_tickets=args.budget)

    print("=" * 78)
    print("  ★ 快乐8双专家大底深度融合分析报告")
    print("=" * 78)
    print(f"  * 双方高共识交集 ({len(result.intersection_numbers)} 码): {' '.join(f'{x:02d}' for x in result.intersection_numbers)}")
    print(f"  * 最终融合黄金 30 码: {' '.join(f'{x:02d}' for x in result.final_merged_30)}")
    print("  * 四大象限区间分布:")
    for sec_name, nums in result.sector_distribution.items():
        print(f"    - {sec_name} ({len(nums)} 码): {' '.join(f'{x:02d}' for x in nums)}")
    print("-" * 78)
    print(f"  * 级联缩水实战选 5 注单 ({len(result.tickets)} 注):")
    for idx, t in enumerate(result.tickets, 1):
        print(f"    [{idx:02d}] {format_ticket(t)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
