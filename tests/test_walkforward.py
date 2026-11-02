"""Walk-Forward 滚动时序回测与因果流测试（契约 R42）。"""

from datetime import date, timedelta
import unittest
import numpy as np

from kl8.config import AppConfig
from kl8.datatypes import Dataset, Draw
from kl8.errors import InsufficientDataError
from kl8.evaluation.walkforward import iter_windows, run_walkforward


def _make_dummy_dataset(n: int = 75) -> Dataset:
    """生成合规的测试用假数据集。"""
    draws = []
    rng = np.random.default_rng(42)
    start_date = date(2026, 1, 1)
    for i in range(n):
        cur_date = start_date + timedelta(days=i)
        nums = frozenset(int(x) for x in rng.choice(np.arange(1, 81), size=20, replace=False))
        draws.append(Draw(
            date=cur_date.strftime("%Y-%m-%d"),
            period=2026001 + i,
            numbers=nums,
            idx=i,
        ))
    return Dataset(
        draws=tuple(draws),
        draws_by_period={d.period: d for d in draws},
    )


class TestWalkForward(unittest.TestCase):
    """Walk-Forward 测试集。"""

    def setUp(self):
        self.ds = _make_dummy_dataset(75)
        self.cfg = AppConfig()

    def test_insufficient_data_raises(self):
        """测试当数据集期数小于等于冷启动期数时抛出 InsufficientDataError。"""
        with self.assertRaises(InsufficientDataError):
            list(iter_windows(self.ds, t0=100))

        with self.assertRaises(InsufficientDataError):
            run_walkforward(self.ds, self.cfg, t0=100)

    def test_iter_windows_causality(self):
        """验证 iter_windows 产生的子数据集严格小于当前被预测期号。"""
        windows = list(iter_windows(self.ds, t0=60, n_test=5))
        self.assertEqual(len(windows), 5)

        for t, sub_ds in windows:
            # 严格截断保证
            self.assertEqual(len(sub_ds.draws), t + 1)
            target_period = self.ds.draws[t + 1].period
            for d in sub_ds.draws:
                self.assertLess(d.period, target_period)

    def test_walkforward_flow_no_leak(self):
        """测试完整 Walk-Forward 回测流程无前瞻泄露并生成自检通过的报告。"""
        report = run_walkforward(
            self.ds,
            self.cfg,
            t0=60,
            n_test=5,
            include_tickets=False,
        )
        self.assertEqual(report.n_sample, 5)
        self.assertEqual(len(report.rows), 5)
        self.assertEqual(report.t0, 60)
        self.assertTrue(report.self_check.get("sample_count_matches", False))
        self.assertTrue(report.self_check.get("no_lookahead", False))

        # 验证逐期行对象完整
        for row in report.rows:
            self.assertGreater(row.period, row.as_of_period)
            self.assertEqual(len(row.scores), 80)
            self.assertEqual(len(row.votes), 80)
            self.assertIn(5, row.hit_at)


if __name__ == "__main__":
    unittest.main()
