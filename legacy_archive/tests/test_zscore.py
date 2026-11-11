"""窗口频次 Z-Score 偏离度单元测试（契约 R8、§6.2 手算对照）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import WindowConfig
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import resolve_windows
from kl8.layer2_baseline.zscore import significant_mask, window_z, z_all_windows


class TestZScore(unittest.TestCase):
    """Z-Score 计算与手算对照。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)

    def test_zscore_formula_and_threshold(self) -> None:
        """手算对照: w=10, c=5 -> (5 - 2.5) / sqrt(10*0.25*0.75) ≈ 1.825741858（契约 R8、§6.2）。"""
        expected_handcalc = (5.0 - 2.5) / np.sqrt(10.0 * 0.25 * 0.75)
        self.assertAlmostEqual(expected_handcalc, 1.8257418583505538, places=8)

        # 构造人工状态验证公式一致性
        synthetic_hit = np.zeros((10, 80), dtype=np.float64)
        # 让号码 1 (索引 0) 在 10 期中出现 5 次
        synthetic_hit[:5, 0] = 1.0
        synthetic_hit[5:, 1] = 1.0
        # 补全每行 20 个球
        synthetic_hit[:5, 2:21] = 1.0
        synthetic_hit[5:, 2:21] = 1.0
        mock_state = BinaryStateMatrix(synthetic_hit)

        z = window_z(mock_state, end=9, w=10)
        self.assertAlmostEqual(float(z[0]), expected_handcalc, places=7)

        # 显著性阈值掩码
        mask = significant_mask(z)
        self.assertEqual(mask.shape, (80,))
        # 1.8257 不大于 1.96，因此不显著
        self.assertFalse(mask[0])

    def test_z_all_windows(self) -> None:
        """测试对真实历史四窗口计算 Z-Score。"""
        cfg = WindowConfig()
        windows = resolve_windows(cfg, self.ds.T)
        res = z_all_windows(self.state, end=self.ds.T - 1, windows=windows)
        self.assertEqual(set(res.keys()), {"10G", "25G", "50G", "ALL"})
        for name, z_arr in res.items():
            self.assertEqual(z_arr.shape, (80,))
            self.assertTrue(np.all(np.isfinite(z_arr)))


if __name__ == "__main__":
    unittest.main()
