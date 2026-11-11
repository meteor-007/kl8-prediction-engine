# -*- coding: utf-8 -*-
"""快乐8 30码黄金大底三级级联缩水模块单元测试。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import unittest
from kl8_engine.shrinkage import (
    CascadeShrinkageEngine,
    MorphologyFilterConfig,
    ShrinkageReport,
    extract_cluster_triplets,
    format_shrinkage_console,
    format_wechat_shrinkage,
)


class TestShrinkage(unittest.TestCase):
    """测试 30 码级联缩水漏斗、蒙特卡洛审计与集群三元组。"""

    def setUp(self) -> None:
        self.sample_pool = tuple(range(1, 31))
        self.scores = {n: float(100 - n) for n in self.sample_pool}

    def test_pool_assertions(self) -> None:
        """测试 30 码大底规模与去重校验。"""
        # 数量不足
        with self.assertRaises(ValueError):
            CascadeShrinkageEngine(pool_30=tuple(range(1, 29)))
        # 重复号码
        with self.assertRaises(ValueError):
            CascadeShrinkageEngine(pool_30=list(range(1, 30)) + [1])
        # 超出范围
        with self.assertRaises(ValueError):
            CascadeShrinkageEngine(pool_30=list(range(1, 30)) + [81])

    def test_morphology_filter(self) -> None:
        """测试形态体检过滤器。"""
        engine = CascadeShrinkageEngine(pool_30=self.sample_pool, kinetic_scores=self.scores)
        # 平衡组合
        self.assertTrue(engine.is_valid_morphology((10, 15, 20, 25, 30)))
        # 极端过小和值
        self.assertFalse(engine.is_valid_morphology((1, 2, 3, 4, 5)))
        # 极端全奇数
        self.assertFalse(engine.is_valid_morphology((1, 3, 5, 7, 9)))

    def test_generate_report_and_audit(self) -> None:
        """测试缩水生成注单与蒙特卡洛审计。"""
        engine = CascadeShrinkageEngine(pool_30=self.sample_pool, kinetic_scores=self.scores)
        report = engine.generate_report(budget=20)

        self.assertIsInstance(report, ShrinkageReport)
        self.assertEqual(report.raw_combinations_count, 142506)
        self.assertGreater(report.after_morphology_count, 0)
        self.assertLess(report.after_morphology_count, report.raw_combinations_count)
        self.assertEqual(report.final_tickets_count, 20)
        self.assertEqual(len(report.tickets), 20)
        self.assertEqual(report.total_cost_yuan, 40)

        # 检查各注号码升序与 5 码属性
        for t in report.tickets:
            self.assertEqual(len(t), 5)
            self.assertEqual(t, tuple(sorted(t)))

        # 文本渲染校验
        console_txt = format_shrinkage_console(report, period=2026293)
        self.assertIn("2026293", console_txt)
        self.assertIn("第 20 注", console_txt)

        wechat_txt = format_wechat_shrinkage(report, period=2026293)
        self.assertIn("【快乐8第2026293期·30码大底精细缩水精选】", wechat_txt)

    def test_extract_cluster_triplets(self) -> None:
        """测试【集群三元组】Top 8 连号清单（格式四铁律）。"""
        order = tuple(range(1, 81))
        triplets = extract_cluster_triplets(order, top_n=8)
        self.assertEqual(len(triplets), 8)
        for trp in triplets:
            parts = trp.split("-")
            self.assertEqual(len(parts), 3)
            # 必须为 XX-XX-XX
            self.assertEqual(len(parts[0]), 2)
            self.assertEqual(len(parts[1]), 2)
            self.assertEqual(len(parts[2]), 2)


if __name__ == "__main__":
    unittest.main()
