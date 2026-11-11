# -*- coding: utf-8 -*-
"""快乐8 30码黄金大底级联精细缩水模块单元测试。

【测试覆盖范围】
1. 宏观形态体检器 (MorphologyFilterConfig) 与单注剪枝有效性；
2. 30码规模与边界校验（POOL_SIZE_30=30，合法号码 1..80）；
3. 空间组合展开与形态过滤留存率；
4. 位运算正交防撞车与动量赋权贪心选取（生成 20 注实战票）；
5. 蒙特卡洛（Monte Carlo）实测抽样模拟准确性与奖金计算；
6. 格式化控制台输出与微信文案文本渲染健壮性。

@author czw
@since 2026-11-03
"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.layer5_combinatorial.shrinkage import (
    MorphologyFilterConfig,
    ShrinkageEngine,
    ShrinkageReport,
    build_wechat_shrinkage_text,
    format_shrinkage_console_summary,
    run_30_shrinkage,
)


class TestShrinkageEngine(unittest.TestCase):
    """30码黄金大底级联精细缩水引擎单元测试用例集。"""

    def setUp(self) -> None:
        """初始化测试环境与固定测试样本。"""
        # 构造一组标准的 30 码池
        self.sample_pool = tuple(range(1, 31))
        # 构造均匀递减的得分模拟动力学打分
        self.sample_scores = {n: float(100 - n) for n in self.sample_pool}

    def test_pool_validation_success(self) -> None:
        """验证合法 30 码池正常初始化。"""
        engine = ShrinkageEngine(
            pool_numbers=self.sample_pool,
            kinetic_scores=self.sample_scores,
        )
        self.assertEqual(len(engine.pool_numbers), 30)
        self.assertEqual(engine.pool_numbers[0], 1)
        self.assertEqual(engine.pool_numbers[-1], 30)

    def test_pool_validation_invalid_length(self) -> None:
        """验证池大小不为 30 码时严格抛出 ValueError。"""
        with self.assertRaises(ValueError):
            ShrinkageEngine(pool_numbers=tuple(range(1, 29)))
        with self.assertRaises(ValueError):
            ShrinkageEngine(pool_numbers=tuple(range(1, 32)))

    def test_pool_validation_out_of_range(self) -> None:
        """验证超出 1..80 范围的号码严格抛出 ValueError。"""
        invalid_pool = list(range(1, 30)) + [81]
        with self.assertRaises(ValueError):
            ShrinkageEngine(pool_numbers=invalid_pool)

    def test_pool_validation_duplicates(self) -> None:
        """验证存在重复号码时严格抛出 ValueError。"""
        duplicate_pool = list(range(1, 30)) + [1]
        with self.assertRaises(ValueError):
            ShrinkageEngine(pool_numbers=duplicate_pool)

    def test_morphology_filter_basic(self) -> None:
        """验证形态体检规则能正确拦截极端形态并放行平衡形态。"""
        config = MorphologyFilterConfig()
        engine = ShrinkageEngine(pool_numbers=self.sample_pool, config=config)

        # 平衡注：[10, 15, 20, 25, 30] -> 跨度 20，奇偶 3:2，合规
        valid_ticket = (10, 15, 20, 25, 30)
        self.assertTrue(engine.is_valid_morphology(valid_ticket))

        # 极端和值过小：[1, 2, 3, 4, 5] -> 和值 15 < 40，应被拦截
        extreme_low_sum = (1, 2, 3, 4, 5)
        self.assertFalse(engine.is_valid_morphology(extreme_low_sum))

        # 极端跨度过小：[10, 11, 12, 13, 14] -> 跨度 4 < 14，应被拦截
        extreme_small_span = (10, 11, 12, 13, 14)
        self.assertFalse(engine.is_valid_morphology(extreme_small_span))

        # 极端五奇数：[1, 3, 5, 7, 9] -> 奇数 5 个，应被拦截
        all_odd = (1, 3, 5, 7, 9)
        self.assertFalse(engine.is_valid_morphology(all_odd))

    def test_generate_tickets_and_report(self) -> None:
        """验证三级级联缩水与实战注单生成完整性。"""
        engine = ShrinkageEngine(
            pool_numbers=self.sample_pool,
            kinetic_scores=self.sample_scores,
        )
        report = engine.generate_full_report(budget_tickets=20, n_simulations=500)

        self.assertEqual(report.raw_combinations_count, 142506)
        self.assertGreater(report.after_morphology_count, 0)
        self.assertLess(report.after_morphology_count, report.raw_combinations_count)
        self.assertEqual(report.final_tickets_count, 20)
        self.assertEqual(len(report.tickets), 20)
        self.assertEqual(report.total_cost_yuan, 40)

        # 检查各注号码的升序与 5 码属性
        for tk in report.tickets:
            self.assertEqual(len(tk), 5)
            self.assertEqual(tk, tuple(sorted(tk)))
            for n in tk:
                self.assertIn(n, self.sample_pool)

        # 验证蒙特卡洛评估指标合理性 (出9码场景下)
        self.assertGreater(report.hit3_or_better_prob, 50.0)
        self.assertGreater(report.expected_prize_per_draw, 0.0)

    def test_run_30_shrinkage_pipeline(self) -> None:
        """验证一键执行 run_30_shrinkage 接口与文本生成。"""
        # 构造 80 码顺序
        order = list(range(1, 81))
        final_scores = np.linspace(100, 1, 80)

        report, console_text, wechat_text = run_30_shrinkage(
            order=order,
            final_scores=final_scores,
            target_period=2026293,
            budget=20,
        )

        self.assertIsInstance(report, ShrinkageReport)
        self.assertEqual(report.final_tickets_count, 20)
        self.assertIn("2026293", console_text)
        self.assertIn("30码黄金大底", console_text)
        self.assertIn("第 20 注", console_text)
        self.assertIn("💎【快乐8第2026293期·30码大底精细缩水精选】💎", wechat_text)


if __name__ == "__main__":
    unittest.main()
