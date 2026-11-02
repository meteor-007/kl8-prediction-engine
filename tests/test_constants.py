"""数学恒等式与静态常数单元测试（契约 §3.2、C1、C9）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.constants import (
    ALL_NUMBERS,
    E_REPEAT,
    K_DRAW,
    N_NUM,
    P0,
    P_CO,
    PRIMES_80,
    SUM_EXPECT,
    Z_SIG,
    get_prime_label,
    get_road_012,
    hypergeometric_mean,
    hypergeometric_pmf,
    hypergeometric_var,
    is_prime_num,
    numbers_array,
    size_class_of,
    tail_of,
    zone4_of,
    zone8_of,
)


class TestConstants(unittest.TestCase):
    """数学常量与辅助函数验证。"""

    def test_constants_values(self) -> None:
        self.assertEqual(N_NUM, 80)
        self.assertEqual(K_DRAW, 20)
        self.assertAlmostEqual(P0, 0.25, places=9)
        self.assertAlmostEqual(P_CO, 20 * 19 / (80 * 79), places=9)
        self.assertAlmostEqual(E_REPEAT, 5.0, places=9)
        self.assertEqual(SUM_EXPECT, 810)
        self.assertAlmostEqual(Z_SIG, 1.96, places=9)
        self.assertEqual(len(PRIMES_80), 22)
        self.assertNotIn(1, PRIMES_80)
        self.assertIn(2, PRIMES_80)
        self.assertIn(79, PRIMES_80)

    def test_partition_helpers(self) -> None:
        # 012 路
        self.assertEqual(get_road_012(3), 0)
        self.assertEqual(get_road_012(1), 1)
        self.assertEqual(get_road_012(2), 2)
        with self.assertRaises(ValueError):
            get_road_012(0)
        with self.assertRaises(ValueError):
            get_road_012(81)

        # 质合
        self.assertFalse(is_prime_num(1))
        self.assertTrue(is_prime_num(2))
        self.assertEqual(get_prime_label(1), "特1")
        self.assertEqual(get_prime_label(2), "质")
        self.assertEqual(get_prime_label(4), "合")

        # 四区八区
        self.assertEqual(zone4_of(1), 0)
        self.assertEqual(zone4_of(20), 0)
        self.assertEqual(zone4_of(21), 1)
        self.assertEqual(zone4_of(80), 3)

        self.assertEqual(zone8_of(1), 0)
        self.assertEqual(zone8_of(10), 0)
        self.assertEqual(zone8_of(75), 7)

        # 大中小
        self.assertEqual(size_class_of(1), 0)
        self.assertEqual(size_class_of(27), 0)
        self.assertEqual(size_class_of(28), 1)
        self.assertEqual(size_class_of(54), 1)
        self.assertEqual(size_class_of(55), 2)
        self.assertEqual(size_class_of(80), 2)

        # 尾数
        self.assertEqual(tail_of(15), 5)
        self.assertEqual(tail_of(80), 0)

        # numbers_array
        arr = numbers_array()
        self.assertEqual(len(arr), 80)
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[-1], 80)

    def test_hypergeometric(self) -> None:
        pmf = hypergeometric_pmf(n_pick=5, n_draw=20, n_total=80)
        self.assertEqual(len(pmf), 6)
        self.assertAlmostEqual(float(pmf.sum()), 1.0, places=9)
        mean = hypergeometric_mean(n_pick=5, n_draw=20, n_total=80)
        self.assertAlmostEqual(mean, 5 * 0.25, places=9)


if __name__ == "__main__":
    unittest.main()
