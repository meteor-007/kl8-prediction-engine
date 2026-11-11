"""NIST 轻量测试套件单元测试（契约 R10）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.layer2_baseline.randomness import (
    approximate_entropy,
    flatten_history,
    frequency_within_block,
    monobit,
    nist_lite,
    runs_nist,
)


class TestRandomness(unittest.TestCase):
    """NIST 随机性套件四项测试。"""

    def test_nist_lite_four_items(self) -> None:
        """测试 nist_lite 返回固定 4 项、固定顺序（契约 R10）。"""
        rng = np.random.default_rng(20261102)
        bits = rng.integers(0, 2, size=1024, dtype=np.int8)

        results = nist_lite(bits)
        self.assertEqual(len(results), 4)
        names = [r.name for r in results]
        self.assertEqual(names, ["Monobit", "BlockFrequency", "Runs", "ApproximateEntropy"])

        for r in results:
            self.assertTrue(0.0 <= r.p_value <= 1.0)
            self.assertTrue(isinstance(r.passed, bool))

    def test_flatten_history(self) -> None:
        """测试历史状态矩阵时序展平。"""
        mat = np.ones((10, 80), dtype=np.float64)
        flat = flatten_history(mat)
        self.assertEqual(flat.shape, (800,))
        self.assertEqual(int(flat.sum()), 800)


if __name__ == "__main__":
    unittest.main()
