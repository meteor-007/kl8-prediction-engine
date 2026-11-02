"""宏观空间形态与 AC 值计算测试（契约 R22、裁决 A4、§6.2 手算对照）。"""

from __future__ import annotations

import unittest

from kl8.config import PoolConfig
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.morphology import (
    ac_value,
    is_balanced,
    morphology_stats,
    sum_deviation_z,
)


class TestMorphology(unittest.TestCase):
    """形态学统计与手算对照。"""

    def test_morphology_stats_full_matrix(self) -> None:
        """测试真实单期开奖号码的完整形态学快照（契约 R22）。"""
        ds = load_dataset("kl8_history.txt")
        state = BinaryStateMatrix.from_dataset(ds)
        nums = sorted(state.latest_set(100))
        self.assertEqual(len(nums), 20)

        stats = morphology_stats(nums)
        self.assertEqual(stats.n, 20)
        self.assertEqual(sum(stats.zone4.values()), 20)
        self.assertEqual(sum(stats.zone8.values()), 20)
        self.assertEqual(stats.odd + stats.even, 20)
        self.assertEqual(sum(stats.size.values()), 20)
        self.assertEqual(sum(stats.same_tail.values()), 20)
        self.assertEqual(stats.sum_value, sum(nums))
        self.assertTrue(stats.ac >= 0)

    def test_ac_value_handcalc(self) -> None:
        """手算对照: {1, 4, 6, 9, 11} 的正差集为 {2, 3, 5, 7, 8, 10}（6 个）-> AC = 6 - (5 - 1) = 2（契约 §6.2、裁决 A4）。"""
        nums = [1, 4, 6, 9, 11]
        self.assertEqual(ac_value(nums), 2)

    def test_is_balanced_constraint(self) -> None:
        """测试形态均衡度约束校验与放宽。"""
        cfg = PoolConfig()
        # 极端不均衡：全奇数、全扎堆在第一区
        unbalanced = list(range(1, 21, 2))  # [1, 3, 5, ..., 19]
        balanced, reasons = is_balanced(unbalanced, cfg)
        self.assertFalse(balanced)
        self.assertTrue(len(reasons) > 0)


if __name__ == "__main__":
    unittest.main()
