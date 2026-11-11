"""防穿越与前瞻泄露审计单测（契约 R47、C4、§3.12）。"""

import unittest
import numpy as np

from kl8.datatypes import Dataset, Draw, FeatureSnapshot
from kl8.errors import LeakageError
from kl8.evaluation.leakage import (
    assert_no_lookahead,
    audit_grep_no_shuffle,
    fingerprint_snapshot,
    max_accessed_t,
    poison_future,
)


def _make_sample_dataset() -> Dataset:
    """构造基准测试数据集。"""
    draws = []
    for i in range(10):
        draws.append(Draw(
            date=f"2026-01-{i+1:02d}",
            period=2026001 + i,
            numbers=frozenset(range(1, 21)),
            idx=i,
        ))
    return Dataset(
        draws=tuple(draws),
        draws_by_period={d.period: d for d in draws},
    )


class TestLeakage(unittest.TestCase):
    """防穿越测试集。"""

    def test_assert_no_lookahead(self):
        """合法索引不抛异常，越界时必须抛出 LeakageError。"""
        # 合法访问
        assert_no_lookahead(max_accessed_t=5, as_of_t=5)
        assert_no_lookahead(max_accessed_t=3, as_of_t=5)

        # 越界非法前瞻
        with self.assertRaises(LeakageError):
            assert_no_lookahead(max_accessed_t=6, as_of_t=5)

    def test_poison_future_invariance(self):
        """测试未来投毒后历史数据严格纯净不变，未来数据发生改变。"""
        ds = _make_sample_dataset()
        as_of_period = 2026005

        poisoned_ds = poison_future(ds, as_of_period, seed=123)

        self.assertEqual(len(poisoned_ds.draws), len(ds.draws))

        # 历史期号保持完全一致
        for orig, poisoned in zip(ds.draws, poisoned_ds.draws):
            if orig.period <= as_of_period:
                self.assertEqual(orig.numbers, poisoned.numbers)
            else:
                # 未来期号码被随机污染
                self.assertNotEqual(orig.numbers, poisoned.numbers)

    def test_audit_grep_no_shuffle(self):
        """审计 kl8 源码目录不包含违规的 KFold 或 shuffle。"""
        violations = audit_grep_no_shuffle("kl8")
        self.assertEqual(violations, (), f"发现违规打乱时间序列的代码: {violations}")


if __name__ == "__main__":
    unittest.main()
