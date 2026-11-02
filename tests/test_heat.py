"""冷热动力学与高斯回补单元测试（契约 R11–R14、§6.2 手算对照）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import WindowConfig
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import resolve_windows
from kl8.layer3_features.heat import (
    gap_series,
    golden_cross,
    historical_mean_gap,
    momentum,
    multi_scale_freq,
    omission_pressure,
)


class TestHeat(unittest.TestCase):
    """冷热特征与手算对照测试。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)
        self.windows = resolve_windows(WindowConfig(), self.ds.T)

    def test_multi_scale_freq(self) -> None:
        """测试四尺度出现频率 Freq_i(w)（契约 R11）。"""
        freq = multi_scale_freq(self.state, end=self.ds.T - 1, windows=self.windows)
        self.assertEqual(set(freq.keys()), {"10G", "25G", "50G", "ALL"})
        for name, arr in freq.items():
            self.assertEqual(arr.shape, (80,))
            self.assertTrue(np.all((arr >= 0.0) & (arr <= 1.0)))

    def test_momentum_and_golden_cross(self) -> None:
        """测试跨周期动量差与金叉加速标签（契约 R12）。"""
        freq = multi_scale_freq(self.state, end=self.ds.T - 1, windows=self.windows)
        m1, m2 = momentum(freq)
        self.assertEqual(m1.shape, (80,))
        self.assertEqual(m2.shape, (80,))
        gc = golden_cross(m1, m2)
        self.assertEqual(gc.shape, (80,))
        self.assertEqual(gc.dtype, bool)
        # 验证金叉条件
        for i in range(80):
            if gc[i]:
                self.assertTrue(m1[i] > 0.0)
                self.assertTrue(m2[i] > 0.0)
                self.assertTrue(m1[i] > m2[i])

    def test_gap_and_mean_gap_floor(self) -> None:
        """测试当前遗漏与历史平均遗漏保底（契约 R13）。"""
        gap = gap_series(self.state, end=self.ds.T - 1)
        self.assertEqual(gap.shape, (80,))
        self.assertTrue(np.all(gap >= 0))
        mean_gap = historical_mean_gap(self.state, end=self.ds.T - 1, floor=3.0)
        self.assertEqual(mean_gap.shape, (80,))
        self.assertTrue(np.all(mean_gap >= 3.0))

    def test_pressure_kernel_peak_at_1_2x(self) -> None:
        """手算对照: O/μ=1.2 -> 1.0; O/μ=0.6 -> exp(-0.5) ≈ 0.60653（契约 R14、§6.2）。"""
        # 手算峰值点：gap = 12, mean_gap = 10 -> ratio = 1.2
        peak_gap = np.array([12.0] * 80)
        peak_mean = np.array([10.0] * 80)
        pressure_peak = omission_pressure(peak_gap, peak_mean)
        self.assertAlmostEqual(float(pressure_peak[0]), 1.0, places=9)

        # 手算偏离点：gap = 6, mean_gap = 10 -> ratio = 0.6, (0.6 - 1.2)/0.6 = -1.0 -> exp(-0.5)
        dev_gap = np.array([6.0] * 80)
        dev_mean = np.array([10.0] * 80)
        pressure_dev = omission_pressure(dev_gap, dev_mean)
        expected_dev = np.exp(-0.5)
        self.assertAlmostEqual(float(pressure_dev[0]), expected_dev, places=7)


if __name__ == "__main__":
    unittest.main()
