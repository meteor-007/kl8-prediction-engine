"""评价指标与统计量单测（契约 R43–R46、§3.12）。"""

import unittest
import numpy as np

from kl8.config import BacktestConfig
from kl8.constants import DEFAULT_PRIZE_TABLE
from kl8.datatypes import BacktestReport, BacktestRow
from kl8.evaluation.metrics import (
    bankroll,
    high_tier_rate,
    lift_at_k,
    rule_attribution,
    self_check,
    t_test_one_sided,
)


class TestMetrics(unittest.TestCase):
    """回测评价指标测试集。"""

    def setUp(self):
        self.cfg = BacktestConfig(ks=(3, 5), lift_signal=1.05, alpha_sig=0.05)

    def test_t_test_one_sided_boundary(self):
        """单侧 t 检验边界：n < 2 返回 (0.0, 1.0)。"""
        t, p = t_test_one_sided(1.5, 1.25, 0.5, 1)
        self.assertEqual(t, 0.0)
        self.assertEqual(p, 1.0)

    def test_lift_at_k_handcalc(self):
        """Top-K 提升度手算比对测试（契约 R43）。"""
        # 构造 10 期样本，每期 Top-5 命中 2 个（理论预期 5 * 0.25 = 1.25）
        rows = []
        for i in range(10):
            rows.append(BacktestRow(
                period=2026001 + i,
                as_of_index=i,
                as_of_period=2026000 + i,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={3: (1, 2, 3), 5: (1, 2, 3, 4, 5)},
                hit_at={3: 1, 5: 2},  # Top-5 恒定命中 2
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
            ))

        lift_res = lift_at_k(rows, (3, 5), self.cfg)
        self.assertIn(5, lift_res)
        stat5 = lift_res[5]

        # 预期 mean_hit = 2.0, expected = 1.25, lift = 2.0 / 1.25 = 1.6
        self.assertAlmostEqual(stat5.mean_hit, 2.0, places=6)
        self.assertAlmostEqual(stat5.expected, 1.25, places=6)
        self.assertAlmostEqual(stat5.lift, 1.6, places=6)
        self.assertEqual(stat5.k, 5)

    def test_high_tier_vs_hypergeometric(self):
        """高奖级实际触发率对比超几何理论概率（契约 R44）。"""
        rows = [
            BacktestRow(
                period=2026001,
                as_of_index=0,
                as_of_period=2026000,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                best_ticket_hit=5,  # 选五中五
            ),
            BacktestRow(
                period=2026002,
                as_of_index=1,
                as_of_period=2026001,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                best_ticket_hit=3,
            ),
        ]

        tier_res = high_tier_rate(rows, "pick5", DEFAULT_PRIZE_TABLE)
        self.assertIn("pick5:5", tier_res)
        self.assertEqual(tier_res["pick5:5"].observed_rate, 0.5)
        self.assertGreater(tier_res["pick5:5"].theory_rate, 0.0)

    def test_bankroll_integer_yuan(self):
        """资金曲线与最大回撤纯整数元核算（契约 R46）。"""
        rows = [
            BacktestRow(
                period=2026001,
                as_of_index=0,
                as_of_period=2026000,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                cost_yuan=20,
                prize_yuan=0,
            ),
            BacktestRow(
                period=2026002,
                as_of_index=1,
                as_of_period=2026001,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                cost_yuan=20,
                prize_yuan=100,
            ),
        ]

        b_stat = bankroll(rows, DEFAULT_PRIZE_TABLE)
        self.assertEqual(b_stat.n_rounds, 2)
        self.assertEqual(b_stat.total_cost_yuan, 40)
        self.assertEqual(b_stat.total_prize_yuan, 100)
        self.assertEqual(b_stat.net_yuan, 60)
        self.assertAlmostEqual(b_stat.roi, 1.5, places=4)
        self.assertEqual(b_stat.max_drawdown_yuan, 20)
        self.assertEqual(b_stat.best_round_yuan, 100)

    def test_rule_attribution(self):
        """规则归因与投票共振统计。"""
        votes = np.array([3] * 10 + [1] * 70, dtype=np.int8)
        scores = np.ones(80)
        rows = [
            BacktestRow(
                period=2026001,
                as_of_index=0,
                as_of_period=2026000,
                scores=scores,
                votes=votes,
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
            )
        ]
        attrib = rule_attribution(rows)
        self.assertIn("n", attrib.high_votes)
        self.assertIn("n", attrib.low_votes)


if __name__ == "__main__":
    unittest.main()
