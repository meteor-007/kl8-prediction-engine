# -*- coding: utf-8 -*-
"""P7 QA —— 组合覆盖设计：v=6..10 穷举保底、预算严格、ILP 降级、SA 确定性。

【契约口径】
- R37：``verify_guarantee`` 对任意 C(v,m) 子集穷举验证 |B∩M| ≥ t；
- R38：经典覆盖表命中优先；R39：ILP；R40：贪心 + 模拟退火（固定 seed 可复现）；
- R41：预算硬截断必须留痕，且 ``n_tickets ≤ budget``；
- A7：m 判定为玩法假设参数，偏差 >2 仅告警。
"""
from __future__ import annotations

import itertools
import time
import unittest

import numpy as np

from kl8.config import CoveringConfig
from kl8.layer5_combinatorial.covering import (
    anneal,
    coverage_ratio,
    greedy_weighted,
    lookup_table,
    m_target,
    solve,
    verify_guarantee,
)
from kl8.layer5_combinatorial.tickets import positions_to_numbers


TABLE_PATH = "kl8/layer5_combinatorial/covering_tables.json"


class TestMTargetModes(unittest.TestCase):
    """A7 m 取值：auto / fixed / 边界保护。"""

    def test_fixed_mode_uses_given_value(self) -> None:
        """m_mode=fixed 且 m_fixed=6 → 6；m_fixed 越界时夹到 [t,v]。"""
        self.assertEqual(m_target(12, 5, 4, "fixed", 6), 6)
        self.assertEqual(m_target(12, 5, 4, "fixed", 1), 4)
        self.assertEqual(m_target(12, 5, 4, "fixed", 20), 12)
        self.assertEqual(m_target(12, 5, 4, "fixed", None), 5)

    def test_auto_margin_effect(self) -> None:
        """auto 模式 margin 越大 m 越大（受 v 截断）。"""
        self.assertEqual(m_target(12, 5, 4, margin=1), 5)
        self.assertEqual(m_target(12, 5, 4, margin=2), 6)
        self.assertEqual(m_target(12, 5, 4, margin=3), 7)
        self.assertEqual(m_target(10, 5, 4, margin=3), 7)


class TestExhaustiveGuaranteeV6ToV10(unittest.TestCase):
    """R37：v=6..10 覆盖结果必须经穷举复核通过（achieved_t ≥ t）。"""

    def test_v6_to_v10_solver_covers_all_subsets(self) -> None:
        """每个 v 求解后独立重跑 verify_guarantee，必须 passed 且 coverage=1.0。"""
        weights = np.ones(20, dtype=np.float64)
        for v in range(6, 11):
            with self.subTest(v=v):
                cfg = CoveringConfig(budget=200, sa_iterations=0, table_path=TABLE_PATH)
                report = solve(v, 5, 4, 5, weights[:v], cfg)
                self.assertLessEqual(report.n_tickets, cfg.budget)
                self.assertEqual(report.n_tickets, len(report.tickets_pos))
                self.assertEqual(report.n_tickets, len(report.tickets))
                self.assertGreaterEqual(report.achieved_t, 4)
                self.assertAlmostEqual(report.coverage_ratio, 1.0, delta=1e-12)
                self.assertTrue(report.verified, f"v={v} 求解结果未通过保底")
                recheck = verify_guarantee(report.tickets_pos, v, 5, 4, 5)
                self.assertTrue(recheck.passed)
                self.assertEqual(report.verified, recheck.passed)
                self.assertEqual(recheck.n_subsets, len(list(itertools.combinations(range(v), 5))))

    def test_v10_table_hit_is_minimal_21_tickets(self) -> None:
        """v=10 必须命中经典表（21 注）并 100% 覆盖。"""
        cfg = CoveringConfig(budget=200, sa_iterations=0, table_path=TABLE_PATH)
        report = solve(10, 5, 4, 5, np.ones(10), cfg)
        self.assertEqual(report.method, "table")
        self.assertEqual(report.n_tickets, 21)
        self.assertTrue(report.verified)
        self.assertFalse(report.degraded)


class TestBudgetStrictness(unittest.TestCase):
    """R41：预算硬截断与 n_tickets 严格性。"""

    def test_budget_truncates_table_and_warns(self) -> None:
        """42 注表预算 5 → 截断为 5 注、verified=False、告警留痕。"""
        cfg = CoveringConfig(budget=5, sa_iterations=0, table_path=TABLE_PATH)
        report = solve(12, 5, 4, 5, np.ones(12), cfg)
        self.assertEqual(report.n_tickets, 5)
        self.assertEqual(len(report.tickets_pos), 5)
        self.assertFalse(report.verified)
        self.assertLess(report.coverage_ratio, 1.0)
        self.assertTrue(any("预算" in w and "截断" in w for w in report.warnings))

    def test_budget_ge_table_size_verifies(self) -> None:
        """预算 ≥ 表注数时必须完整保留并通过保底验证。"""
        cfg = CoveringConfig(budget=100, sa_iterations=0, table_path=TABLE_PATH)
        report = solve(12, 5, 4, 5, np.ones(12), cfg)
        self.assertEqual(report.n_tickets, 42)
        self.assertTrue(report.verified)
        self.assertAlmostEqual(report.coverage_ratio, 1.0, delta=1e-12)

    def test_guarantee_fails_for_partial_cover(self) -> None:
        """任意少一注的 41 注表是否仍覆盖？若不覆盖必须如实报告（不得虚报）。"""
        cfg = CoveringConfig(budget=100, sa_iterations=0, table_path=TABLE_PATH)
        full = solve(12, 5, 4, 5, np.ones(12), cfg)
        partial = full.tickets_pos[:-1]
        result = verify_guarantee(partial, 12, 5, 4, 5)
        self.assertEqual(result.passed, coverage_ratio(partial, 12, 5, 4, 5) == 1.0)


class TestTableAndFallbackPaths(unittest.TestCase):
    """R38/R39/R40：查表命中 vs 未命中降级。"""

    def test_lookup_table_hit_and_miss(self) -> None:
        """直接查表：已知键返回注单，未知键返回 None，缺文件返回 None。"""
        hit = lookup_table(8, 5, 4, 5, path=TABLE_PATH)
        self.assertIsNotNone(hit)
        self.assertEqual(len(hit), 5)
        self.assertIsNone(lookup_table(10, 5, 4, 6, path=TABLE_PATH))
        self.assertIsNone(lookup_table(8, 5, 4, 5, path="not_exists.json"))

    def test_ilp_degradation_path(self) -> None:
        """表未命中 + 规模超 ILP 上限 → 降级贪心+退火并留 degrade_reason。"""
        cfg = CoveringConfig(budget=5, sa_iterations=0, table_path=TABLE_PATH)
        report = solve(10, 5, 4, 6, np.ones(10), cfg)
        self.assertEqual(report.method, "greedy+anneal")
        self.assertTrue(report.degraded)
        self.assertTrue(report.degrade_reason)
        self.assertLessEqual(report.n_tickets, cfg.budget)
        self.assertTrue(any("降级" in w or "ILP" in w for w in report.warnings))

    def test_greedy_weighted_deterministic(self) -> None:
        """贪心覆盖固定 seed 必须逐位可复现。"""
        weights = np.linspace(1.0, 0.1, 10)
        a = greedy_weighted(10, 5, 4, 6, weights, budget=8, seed=42)
        b = greedy_weighted(10, 5, 4, 6, weights, budget=8, seed=42)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 8)

    def test_anneal_deterministic(self) -> None:
        """SA 固定 seed 两次结果逐位相同，且注数不超过预算。"""
        weights = np.linspace(1.0, 0.1, 10)
        a = anneal(10, 5, 4, 6, weights, budget=8, iterations=120, t0=1.0, cooling=0.99, seed=7)
        b = anneal(10, 5, 4, 6, weights, budget=8, iterations=120, t0=1.0, cooling=0.99, seed=7)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 8)
        self.assertTrue(all(len(t) == 5 for t in a))

    def test_solve_deterministic_same_config(self) -> None:
        """同一配置连续两次 solve 结果注单逐位相同（C7 确定性）。"""
        cfg = CoveringConfig(budget=6, sa_iterations=120, table_path=TABLE_PATH)
        a = solve(10, 5, 4, 6, np.ones(10), cfg)
        b = solve(10, 5, 4, 6, np.ones(10), cfg)
        self.assertEqual(a.tickets_pos, b.tickets_pos)
        self.assertEqual(a.method, b.method)


class TestA7SanityWarning(unittest.TestCase):
    """A7：m 偏离经验期望 >2 时只告警、不阻断。"""

    def test_a7_warning_only(self) -> None:
        """m=12（经验期望 5）→ 正常返回 + 告警；不抛异常。"""
        cfg = CoveringConfig(budget=5, sa_iterations=0, table_path=TABLE_PATH)
        report = solve(12, 5, 4, 12, np.ones(12), cfg)
        self.assertTrue(any("假设命中数" in w for w in report.warnings))
        self.assertTrue(any("非算法预测值" in w for w in report.warnings))
        self.assertLessEqual(report.n_tickets, cfg.budget)

    def test_covering_report_field_consistency(self) -> None:
        """CoveringReport 字段自洽：pool=1..v、elapsed_ms≥0、achieved_t∈[0,k]。"""
        cfg = CoveringConfig(budget=100, sa_iterations=0, table_path=TABLE_PATH)
        report = solve(12, 5, 4, 5, np.ones(12), cfg)
        self.assertEqual(report.pool, tuple(range(1, 13)))
        self.assertEqual(report.v, 12)
        self.assertEqual(report.k, 5)
        self.assertEqual(report.t, 4)
        self.assertEqual(report.m, 5)
        self.assertGreaterEqual(report.elapsed_ms, 0)
        self.assertGreaterEqual(report.achieved_t, 0)
        self.assertLessEqual(report.achieved_t, report.k)
        self.assertGreaterEqual(report.coverage_ratio, 0.0)
        self.assertLessEqual(report.coverage_ratio, 1.0)
        # tickets 是 positions 的 1-based 占位映射
        mapped = positions_to_numbers(report.tickets_pos, report.pool)
        self.assertEqual(mapped, report.tickets)
        self.assertLess(report.elapsed_ms, 3000)


if __name__ == "__main__":
    unittest.main()