# -*- coding: utf-8 -*-
"""快乐8 30码大底出9保5级联精细缩水引擎 —— 白盒单元测试套件

【测试范围】:
1. 初始化合法性与边界防守（非法号码、号码数量不足、去重校验）；
2. 宏观形态体检器校验（和值、跨度、奇偶、质合、连号截断、012路平衡）；
3. 三级级联缩水与正交贪心算法输出校验（预算注数匹配、防撞车重叠度控制）；
4. 胆码锚定功能校验（指定胆码必出、多胆码覆盖）；
5. 蒙特卡洛回测审计数值稳定性与无 NaN 校验。

@author czw
@since 2026-11-01
"""
from __future__ import annotations

import unittest
from typing import List, Tuple

from calc_shrinkage import (
    MorphologyFilterConfig,
    ShrinkageEngine,
    ShrinkageReport,
    auto_load_latest_top30,
    format_ticket,
)


class TestShrinkageEngine(unittest.TestCase):
    """缩水引擎白盒单元测试类。"""

    def setUp(self):
        """测试前置夹具：构建标准 30 码测试池。"""
        self.mock_pool: List[int] = [
            1, 3, 4, 5, 8, 13, 15, 16, 20, 28,
            33, 35, 41, 42, 44, 45, 46, 47, 48, 50,
            54, 57, 59, 60, 62, 65, 68, 72, 73, 78
        ]
        self.mock_scores = {num: 1.0 + (num % 5) * 0.1 for num in self.mock_pool}
        self.engine = ShrinkageEngine(self.mock_pool, self.mock_scores)

    def test_01_initialization_safety(self):
        """测试初始化参数校验与空安全边界防守。"""
        # 测试号码池为空或不足5个号码
        with self.assertRaises(ValueError):
            ShrinkageEngine([])

        with self.assertRaises(ValueError):
            ShrinkageEngine([1, 2, 3])

        # 测试包含非法号码 (如 0 或 81)
        with self.assertRaises(ValueError):
            ShrinkageEngine([0, 1, 2, 3, 4])

        with self.assertRaises(ValueError):
            ShrinkageEngine([1, 2, 3, 4, 85])

        # 测试自动去重
        engine_dup = ShrinkageEngine([1, 1, 2, 2, 3, 4, 5])
        self.assertEqual(len(engine_dup.pool), 5)

    def test_02_morphology_filter_rules(self):
        """测试宏观形态体检器的单项过滤规则准确性。"""
        # 正常组合：和值适中、跨度合理、奇偶平衡、无长连号
        valid_combo = (4, 15, 45, 48, 65)
        self.assertTrue(self.engine.is_valid_morphology(valid_combo))

        # 异常情况 1: 3连号 (违规：最大连号超过2)
        consec_combo = (10, 11, 12, 40, 50)
        self.assertFalse(self.engine.is_valid_morphology(consec_combo))

        # 异常情况 2: 全奇数 (违规：奇数5个，不在{1,2,3,4}白名单)
        all_odd_combo = (3, 5, 13, 15, 45)
        self.assertFalse(self.engine.is_valid_morphology(all_odd_combo))

        # 异常情况 3: 全偶数 (违规：奇数0个)
        all_even_combo = (4, 8, 16, 20, 48)
        self.assertFalse(self.engine.is_valid_morphology(all_even_combo))

        # 异常情况 4: 极端微小跨度 (违规：跨度 < 14)
        tight_span_combo = (20, 22, 24, 26, 28)
        self.assertFalse(self.engine.is_valid_morphology(tight_span_combo))

    def test_03_cascade_shrink_budget(self):
        """测试三级级联缩水预算注数与正交防撞车约束。"""
        for target_budget in [10, 20]:
            tickets = self.engine.execute_cascade_shrink(
                budget_tickets=target_budget,
                max_overlap=3,
                guarantee_mode="出9保5"
            )
            self.assertEqual(len(tickets), target_budget)

            # 验证每注均为 5 个不重复号码且严格排序
            for t in tickets:
                self.assertEqual(len(t), 5)
                self.assertEqual(list(t), sorted(list(t)))
                self.assertTrue(all(x in self.mock_pool for x in t))

            # 验证前若干注单之间的正交防撞车重合度不超过 3
            if len(tickets) >= 2:
                for i in range(min(5, len(tickets))):
                    for j in range(i + 1, min(5, len(tickets))):
                        overlap = len(set(tickets[i]) & set(tickets[j]))
                        self.assertLessEqual(overlap, 3, f"注单 {tickets[i]} 与 {tickets[j]} 重合度过高: {overlap}")

    def test_04_banker_anchoring(self):
        """测试胆码锚定功能的刚性约束。"""
        target_banker = 4
        tickets = self.engine.execute_cascade_shrink(
            budget_tickets=10,
            bankers=[target_banker],
            guarantee_mode="出9保5"
        )
        self.assertGreater(len(tickets), 0)
        # 验证所有生成的注单中均严格包含胆码 4
        for t in tickets:
            self.assertIn(target_banker, t, f"注单 {t} 未包含指定胆码 {target_banker}")

    def test_05_monte_carlo_audit(self):
        """测试蒙特卡洛回测审计输出指标的完整性与数值范围。"""
        tickets = self.engine.execute_cascade_shrink(budget_tickets=10)
        audit_res = self.engine.run_monte_carlo_audit(tickets, n_simulations=500, drawn_in_pool=9)

        self.assertIn("hit5_prob", audit_res)
        self.assertIn("hit4_or_better_prob", audit_res)
        self.assertIn("hit3_or_better_prob", audit_res)
        self.assertIn("avg_prize", audit_res)
        self.assertIn("roi_percentage", audit_res)

        # 概率必须在 [0, 100] 之间
        self.assertTrue(0.0 <= audit_res["hit5_prob"] <= 100.0)
        self.assertTrue(0.0 <= audit_res["hit4_or_better_prob"] <= 100.0)
        self.assertTrue(0.0 <= audit_res["hit3_or_better_prob"] <= 100.0)
        # 保3或以上概率应大于等于保4或以上概率
        self.assertGreaterEqual(audit_res["hit3_or_better_prob"], audit_res["hit4_or_better_prob"])

    def test_06_format_ticket_helper(self):
        """测试注单格式化函数对齐显示。"""
        formatted = format_ticket((1, 5, 20, 45, 78))
        self.assertEqual(formatted, "01 05 20 45 78")

    def test_07_full_report_integration(self):
        """测试端到端报告生成流程。"""
        report: ShrinkageReport = self.engine.generate_full_report(budget_tickets=10)
        self.assertEqual(report.raw_pool_count, 30)
        self.assertEqual(report.final_tickets_count, 10)
        self.assertEqual(report.total_cost_yuan, 20)
        self.assertGreater(report.after_morphology_count, 0)
        self.assertGreater(report.hit3_or_better_prob, 50.0)


if __name__ == "__main__":
    unittest.main()
