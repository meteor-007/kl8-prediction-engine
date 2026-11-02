"""香农熵与特征冗余度测试（契约 R10）。"""

from __future__ import annotations

import math
import unittest
import numpy as np

from kl8.constants import N_NUM
from kl8.layer2_baseline.entropy import (
    feature_redundancy,
    freq_entropy,
    mi_lag1,
    shannon_entropy,
)


class TestEntropy(unittest.TestCase):
    """熵值、互信息与特征冗余度测试。"""

    def test_entropy_and_redundancy(self) -> None:
        """测试均匀分布香农熵达到 log2(N) 与特征冗余度排序（契约 R10）。"""
        # 均匀分布熵
        uniform_p = np.full(N_NUM, 1.0 / N_NUM)
        ent = shannon_entropy(uniform_p, base=2.0)
        self.assertAlmostEqual(ent, math.log2(N_NUM), places=7)

        # 单点确定性分布熵为 0
        point_p = np.zeros(N_NUM)
        point_p[0] = 1.0
        self.assertAlmostEqual(shannon_entropy(point_p), 0.0, places=7)

        # 特征冗余度
        feat_a = np.linspace(0, 1, 80)
        feat_b = feat_a * 2.0 + 1.0  # 完全线性相关
        feat_c = np.sin(feat_a * 10)
        features = {"fa": feat_a, "fb": feat_b, "fc": feat_c}

        pairs = feature_redundancy(features, top_n=2)
        self.assertTrue(len(pairs) >= 1)
        # 最相关对必然是 fa 和 fb
        top_pair = pairs[0]
        self.assertTrue(set(top_pair[:2]) == {"fa", "fb"})
        self.assertAlmostEqual(top_pair[2], 1.0, places=5)

    def test_mi_lag1_shape(self) -> None:
        """测试滞后 1 阶互信息矩阵形状为 (80, 80)。"""
        mat = np.zeros((50, 80), dtype=np.float64)
        mat[:, :20] = 1.0
        mi = mi_lag1(mat)
        self.assertEqual(mi.shape, (80, 80))
        self.assertTrue(np.all(mi >= 0.0))


if __name__ == "__main__":
    unittest.main()
