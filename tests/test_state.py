"""二值化状态矩阵单元测试（契约 R1、C6、C9）。

对应契约需求 R1：
- 二值化状态矩阵 X ∈ {0,1}^(T×80)，每行行和恒为 20。
- 形状、取值、行和强校验；异常抛 DataFormatError。
- window_matrix、window_sum、window_mean、gap_vector 等切片功能。
"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.constants import K_DRAW, N_NUM
from kl8.errors import DataFormatError
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix


class TestBinaryStateMatrix(unittest.TestCase):
    """状态矩阵核心测试集。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)

    def test_binary_matrix_shape_and_row_sum(self) -> None:
        """验证二值化矩阵形状为 (T, 80)，且每行行和严格为 20（契约 R1）。"""
        self.assertEqual(self.state.T, self.ds.T)
        self.assertEqual(self.state.hit.shape, (self.ds.T, N_NUM))
        # 校验取值仅为 0 或 1
        unique_vals = set(np.unique(self.state.hit))
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))
        # 校验每行行和恒为 K_DRAW (20)
        row_sums = self.state.hit.sum(axis=1)
        np.testing.assert_array_equal(row_sums, np.full(self.ds.T, K_DRAW, dtype=np.float64))
        # 校验 row_sum 方法
        self.assertEqual(self.state.row_sum(0), K_DRAW)
        self.assertEqual(self.state.row_sum(self.state.T - 1), K_DRAW)

    def test_invalid_shape_or_values_rejected(self) -> None:
        """非法形状或非法数值必须抛 DataFormatError。"""
        # 错误列数
        bad_cols = np.zeros((10, 79), dtype=np.float64)
        with self.assertRaises(DataFormatError):
            BinaryStateMatrix(bad_cols)
        # 非 0/1 取值
        bad_vals = np.zeros((10, 80), dtype=np.float64)
        bad_vals[:, :20] = 0.5
        with self.assertRaises(DataFormatError):
            BinaryStateMatrix(bad_vals)
        # 行和不等于 20
        bad_sum = np.zeros((10, 80), dtype=np.float64)
        bad_sum[:, :19] = 1.0
        with self.assertRaises(DataFormatError):
            BinaryStateMatrix(bad_sum)

    def test_window_operations(self) -> None:
        """验证窗口切片、求和、均值、遗漏向量及号码集合。"""
        end = 100
        # 10G 窗口切片
        w_mat = self.state.window_matrix(end, 10)
        self.assertEqual(w_mat.shape, (10, 80))
        # 窗口求和
        w_sum = self.state.window_sum(end, 10)
        self.assertEqual(w_sum.shape, (80,))
        self.assertEqual(w_sum.sum(), 10 * K_DRAW)
        # 窗口均值
        w_mean = self.state.window_mean(end, 10)
        np.testing.assert_allclose(w_mean, w_sum / 10.0)
        # ALL 窗口
        w_all_mat = self.state.window_matrix(end, None)
        self.assertEqual(w_all_mat.shape, (end + 1, 80))
        # latest_set
        latest_nums = self.state.latest_set(end)
        self.assertEqual(len(latest_nums), K_DRAW)
        self.assertTrue(all(1 <= n <= 80 for n in latest_nums))
        # gap_vector
        gap = self.state.gap_vector(end)
        self.assertEqual(gap.shape, (80,))
        self.assertTrue(np.all(gap >= 0))
        # 开出的号码遗漏必为 0
        for num in latest_nums:
            self.assertEqual(gap[num - 1], 0)


if __name__ == "__main__":
    unittest.main()
