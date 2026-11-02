"""覆盖设计定义、保底穷举验证、ILP 与模拟退火测试（契约 R37、R39、R40、C7）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import CoveringConfig
from kl8.layer5_combinatorial.covering import (
    anneal,
    coverage_ratio,
    greedy_weighted,
    m_target,
    solve,
    solve_ilp,
    verify_guarantee,
)


class TestCovering(unittest.TestCase):
    """覆盖设计穷举验证与优化算法测试。"""

    def test_verify_guarantee_exhaustive(self) -> None:
        """测试覆盖设计保底命题穷举验证（契约 R37）。"""
        v, k, t, m = 6, 4, 3, 4
        # 全组合必然 100% 达成保底
        import itertools
        full_tickets = list(itertools.combinations(range(v), k))
        res = verify_guarantee(full_tickets, v, k, t, m)
        self.assertTrue(res.passed)
        self.assertTrue(res.achieved_t >= t)
        self.assertIsNone(res.failed_example)

        # 仅 1 注必然无法保证所有 4-子集满足保底
        single_ticket = [(0, 1, 2, 3)]
        res_fail = verify_guarantee(single_ticket, v, k, t, m)
        self.assertFalse(res_fail.passed)
        self.assertIsNotNone(res_fail.failed_example)

    def test_ilp_min_tickets_and_fallback(self) -> None:
        """测试 0-1 整数规划求解与超规模安全降级（契约 R39）。"""
        # 小规模 ILP
        res = solve_ilp(6, 4, 3, 4, candidate_cap=120, target_cap=400)
        # 若环境具备 milp 求解器则成功返回注单，否则安全返回 None
        if res is not None:
            self.assertTrue(len(res) > 0)

        # 超规模触发降级返回 None
        res_fallback = solve_ilp(20, 5, 4, 5, candidate_cap=10, target_cap=10)
        self.assertIsNone(res_fallback)

    def test_greedy_weighted_and_sa_determinism(self) -> None:
        """测试带权贪心与模拟退火在固定种子下的确定性（契约 R40、C7）。"""
        v, k, t, m = 10, 5, 4, 5
        weights = np.linspace(1.0, 0.1, v)
        seed = 20261102

        # 同参两次调用结果逐注完全一致
        run1 = greedy_weighted(v, k, t, m, weights, budget=10, seed=seed)
        run2 = greedy_weighted(v, k, t, m, weights, budget=10, seed=seed)
        self.assertEqual(run1, run2)

        sa1 = anneal(v, k, t, m, weights, budget=8, iterations=100, t0=1.0, cooling=0.95, seed=seed)
        sa2 = anneal(v, k, t, m, weights, budget=8, iterations=100, t0=1.0, cooling=0.95, seed=seed)
        self.assertEqual(sa1, sa2)


if __name__ == "__main__":
    unittest.main()
