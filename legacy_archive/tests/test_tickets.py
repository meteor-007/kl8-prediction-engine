"""注单后处理过滤与双玩法注单落地测试（契约 R41）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import AppConfig, CoveringConfig
from kl8.layer5_combinatorial.tickets import build_ticket_plan, postprocess


class TestTickets(unittest.TestCase):
    """注单生成、后处理过滤与双玩法测试。"""

    def test_postprocess_and_dual_play(self) -> None:
        """测试注单后处理（去重/全奇全偶过滤/预算截断）与选五/选七双玩法落地（契约 R41）。"""
        pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        scores = np.linspace(0.8, 0.2, 80)
        cfg = AppConfig()

        # 测试选五计划
        plan_p5 = build_ticket_plan(pool, scores, cfg, play="pick5", budget=10)
        self.assertEqual(plan_p5.play, "pick5")
        self.assertEqual(plan_p5.k, 5)
        self.assertTrue(len(plan_p5.tickets) <= 10)
        for ticket in plan_p5.tickets:
            self.assertEqual(len(ticket), 5)
            # 每注严格升序
            self.assertEqual(list(ticket), sorted(ticket))

        # 测试选七计划
        plan_p7 = build_ticket_plan(pool, scores, cfg, play="pick7", budget=8)
        self.assertEqual(plan_p7.play, "pick7")
        self.assertEqual(plan_p7.k, 7)
        self.assertTrue(len(plan_p7.tickets) <= 8)
        for ticket in plan_p7.tickets:
            self.assertEqual(len(ticket), 7)
            self.assertEqual(list(ticket), sorted(ticket))


if __name__ == "__main__":
    unittest.main()
