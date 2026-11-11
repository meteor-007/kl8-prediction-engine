# -*- coding: utf-8 -*-
"""快乐8 常量层与号码静态属性单元测试。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import unittest
from kl8_engine.constants import (
    K_DRAW,
    N_NUM,
    P0,
    PRIMES_80,
    PRIZE_TABLE_PICK5,
    get_road_012,
    get_zone4,
    is_prime,
)


class TestConstants(unittest.TestCase):
    """测试玩法定义常量与纯函数属性。"""

    def test_basic_constants(self) -> None:
        """测试号码总数、每期开奖数与理论概率。"""
        self.assertEqual(N_NUM, 80)
        self.assertEqual(K_DRAW, 20)
        self.assertAlmostEqual(P0, 0.25, places=5)
        self.assertEqual(PRIZE_TABLE_PICK5[5], 1000)
        self.assertEqual(PRIZE_TABLE_PICK5[4], 21)
        self.assertEqual(PRIZE_TABLE_PICK5[3], 3)

    def test_prime_numbers(self) -> None:
        """测试 1..80 范围内的质数判断。"""
        self.assertFalse(is_prime(1))  # 1 非质非合
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(79))
        self.assertFalse(is_prime(80))
        self.assertEqual(len(PRIMES_80), 22)

    def test_road_012(self) -> None:
        """测试 012 路模 3 同余分类。"""
        self.assertEqual(get_road_012(3), 0)
        self.assertEqual(get_road_012(1), 1)
        self.assertEqual(get_road_012(2), 2)
        with self.assertRaises(ValueError):
            get_road_012(0)
        with self.assertRaises(ValueError):
            get_road_012(81)

    def test_zone4(self) -> None:
        """测试四分区划分。"""
        self.assertEqual(get_zone4(1), 0)
        self.assertEqual(get_zone4(20), 0)
        self.assertEqual(get_zone4(21), 1)
        self.assertEqual(get_zone4(40), 1)
        self.assertEqual(get_zone4(41), 2)
        self.assertEqual(get_zone4(60), 2)
        self.assertEqual(get_zone4(61), 3)
        self.assertEqual(get_zone4(80), 3)


if __name__ == "__main__":
    unittest.main()
