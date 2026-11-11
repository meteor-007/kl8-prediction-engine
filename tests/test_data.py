# -*- coding: utf-8 -*-
"""快乐8 数据解析、校验与状态矩阵抽象单元测试。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from kl8_engine.data import Dataset, Draw, load_dataset, parse_line, save_dataset


class TestData(unittest.TestCase):
    """测试 Draw 与 Dataset 解析、二值化与持久化。"""

    def setUp(self) -> None:
        self.sample_line = "date:2026-11-02,period:2026292,numbers:02-05-10-17-22-25-26-36-43-45-50-51-52-57-62-64-68-70-71-75"

    def test_parse_valid_line(self) -> None:
        """测试正常格式行解析。"""
        draw = parse_line(self.sample_line)
        self.assertIsNotNone(draw)
        assert draw is not None
        self.assertEqual(draw.period, 2026292)
        self.assertEqual(draw.date, "2026-11-02")
        self.assertEqual(len(draw.numbers), 20)
        self.assertEqual(draw.numbers[0], 2)
        self.assertEqual(draw.numbers[-1], 75)

    def test_draw_multihot(self) -> None:
        """测试二值化向量映射。"""
        draw = parse_line(self.sample_line)
        assert draw is not None
        vec = draw.to_multihot()
        self.assertEqual(vec.shape, (80,))
        self.assertEqual(vec.sum(), 20.0)
        self.assertEqual(vec[1], 1.0)  # number 2 -> index 1
        self.assertEqual(vec[0], 0.0)  # number 1 -> index 0

    def test_invalid_lines(self) -> None:
        """测试非法行错误防御。"""
        # 数量不足
        with self.assertRaises(ValueError):
            parse_line("date:2026-11-02,period:2026292,numbers:02-05-10")
        # 超出范围
        with self.assertRaises(ValueError):
            nums = "-".join(f"{x:02d}" for x in range(1, 20)) + "-85"
            parse_line(f"date:2026-11-02,period:2026292,numbers:{nums}")
        # 重复号码
        with self.assertRaises(ValueError):
            nums = "-".join(f"{x:02d}" for x in range(1, 20)) + "-01"
            parse_line(f"date:2026-11-02,period:2026292,numbers:{nums}")

    def test_dataset_save_and_load(self) -> None:
        """测试数据集保存与加载回环一致性。"""
        d1 = parse_line("date:2026-11-01,period:2026291,numbers:" + "-".join(f"{x:02d}" for x in range(1, 21)))
        d2 = parse_line(self.sample_line)
        assert d1 is not None and d2 is not None
        ds = Dataset(draws=(d1, d2))

        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, "test_history.txt")
            save_dataset(ds, fpath)
            loaded = load_dataset(fpath)

            self.assertEqual(loaded.total_draws, 2)
            self.assertEqual(loaded.latest.period, 2026292)
            self.assertEqual(loaded.target_period(), 2026293)
            mat = loaded.to_matrix()
            self.assertEqual(mat.shape, (2, 80))


if __name__ == "__main__":
    unittest.main()
