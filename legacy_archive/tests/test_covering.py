"""覆盖设计定义、保底穷举验证、ILP 与模拟退火测试（契约 R37、R39、R40、C7）。

【P10 修复】M14「假绿」：ILP 成功分支不再用 ``if res is not None`` 静默跳过，
改为显式 ``skipUnless`` + 成功路径强制穷举验证；M8 新增 seed 透传探针测试。
"""

from __future__ import annotations

import itertools
import unittest
from unittest import mock

import numpy as np

import kl8.layer5_combinatorial.covering as cov_module
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

try:
    from scipy.optimize import milp as _scipy_milp  # noqa: F401  # type: ignore[import-untyped]

    _HAS_MILP = True
except ImportError:
    _HAS_MILP = False


class TestCovering(unittest.TestCase):
    """覆盖设计穷举验证与优化算法测试。"""

    def test_verify_guarantee_exhaustive(self) -> None:
        """测试覆盖设计保底命题穷举验证（契约 R37）。"""
        v, k, t, m = 6, 4, 3, 4
        # 全组合必然 100% 达成保底
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

    @unittest.skipUnless(_HAS_MILP, "环境未安装 scipy>=1.9 的 milp 求解器，ILP 分支不可用")
    def test_ilp_min_tickets_and_fallback(self) -> None:
        """ILP 小规模必须真求解成功且穷举通过；超规模必须安全返回 None（契约 R39）。"""
        res = solve_ilp(6, 4, 3, 4, candidate_cap=120, target_cap=400)
        self.assertIsNotNone(res, "规模未超限时必须由 MILP 求真解，不得静默降级")
        self.assertGreater(len(res), 0)
        verify = verify_guarantee(list(res), 6, 4, 3, 4)
        self.assertTrue(verify.passed, "ILP 解必须通过穷举保底验证")
        self.assertGreaterEqual(verify.achieved_t, 3)

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

    def test_postprocess_rejects_three_consecutive(self) -> None:
        """M11 自证：连号阈值不得差一——允许 2 连、必须剔除 3 连，且剔除必须留痕（R41）。"""
        from kl8.layer5_combinatorial.tickets import postprocess

        cfg = CoveringConfig(budget=10)
        pool = tuple(range(1, 13))
        scores_by_number = {num: 1.0 for num in pool}
        # tickets 为池内位置索引（0-based）：(0,1,4,5,8) → 号码 (1,2,5,6,9) 仅 2 连
        tickets = [
            (0, 1, 4, 5, 8),      # 两个 2 连：允许
            (0, 1, 2, 6, 10),     # 号码 (1,2,3,7,11)：3 连 → 必须剔除
            (3, 4, 5, 6, 11),     # 号码 (4,5,6,7,12)：4 连 → 必须剔除
        ]
        kept, warns = postprocess(tickets, pool, cfg=cfg, scores_by_number=scores_by_number)
        self.assertIn((1, 2, 5, 6, 9), kept, f"合法 2 连注单被误杀：{kept}")
        self.assertNotIn((1, 2, 3, 7, 11), kept, "3 连注单未被剔除（off-by-one）")
        self.assertNotIn((4, 5, 6, 7, 12), kept, "4 连注单未被剔除")
        self.assertTrue(any("连号" in warn for warn in warns), f"连号剔除必须留痕：{warns}")

    def test_postprocess_enforces_pair_overlap_cap(self) -> None:
        """M11 自证：``max_pair_overlap`` 必须真实生效（配置项曾 0 引用）。"""
        from kl8.layer5_combinatorial.tickets import postprocess

        cfg = CoveringConfig(budget=10, max_pair_overlap=2, forbid_all_odd_even=False)
        pool = tuple(range(1, 13))
        scores_by_number = {num: 1.0 for num in pool}
        # 号码：(1,2,5,10,11) 与 (1,2,5,10,12) 重合 4 码 > 上限 2 → 必有一注被剔除
        tickets = [
            (0, 1, 4, 9, 10),
            (0, 1, 4, 9, 11),
            (2, 3, 6, 7, 8),
        ]
        kept, warns = postprocess(tickets, pool, cfg=cfg, scores_by_number=scores_by_number)
        self.assertIn((1, 2, 5, 10, 11), kept)
        self.assertNotIn((1, 2, 5, 10, 12), kept, "重合超限注单未被剔除")
        for left, right in itertools.combinations(kept, 2):
            self.assertLessEqual(len(set(left) & set(right)), 2, f"注间重合超限：{left} vs {right}")
        self.assertTrue(any("重叠" in warn for warn in warns), f"重叠剔除必须留痕：{warns}")

    def test_solve_plumbs_seed_from_caller(self) -> None:
        """M8 自证：``solve(seed=...)`` 必须把种子真实传给降级求解器（禁止 seed=42 硬编码）。"""
        captured: dict[str, int] = {}
        real_greedy = cov_module.greedy_weighted

        def spy(v: int, k: int, t: int, m: int, weights: np.ndarray, *, budget: int, seed: int, max_iters: int = 2000):
            captured["seed"] = int(seed)
            return real_greedy(v, k, t, m, weights, budget=budget, seed=seed, max_iters=max_iters)

        cfg = CoveringConfig(table_path="__no_such_table__.json", budget=4)
        with mock.patch.object(cov_module, "greedy_weighted", spy):
            report = solve(10, 5, 4, 5, np.linspace(1.0, 0.1, 10), cfg, seed=987654)
        self.assertEqual(captured.get("seed"), 987654, "seed 未从调用方透传到降级求解器")
        self.assertTrue(report.method.startswith("greedy"), f"降级路径标识异常：{report.method}")
        self.assertLessEqual(report.n_tickets, cfg.budget)


if __name__ == "__main__":
    unittest.main()
