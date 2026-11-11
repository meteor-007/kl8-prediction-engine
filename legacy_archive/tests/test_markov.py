"""马尔可夫转移矩阵与跟随得分测试（契约 R15–R18、§6.2 手算对照）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import WindowConfig
from kl8.constants import P0
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import resolve_windows
from kl8.layer3_features.markov import (
    follow_score,
    follow_split,
    transition_matrix,
    transition_matrix_order2,
)


class TestMarkov(unittest.TestCase):
    """马尔可夫转移平滑与跟随得分手算对照。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)

    def test_alpha_smoothing_handcalc(self) -> None:
        """手算对照一阶拉普拉斯平滑: (N_ij + α*p0) / (c_i + α)（契约 R15、§6.2）。"""
        # 构造简单测试切片：2 期数据
        # 期 1 开出 [1..20]，期 2 开出 [1..10, 21..30]
        hit_toy = np.zeros((2, 80), dtype=np.float64)
        hit_toy[0, :20] = 1.0
        hit_toy[1, :10] = 1.0
        hit_toy[1, 20:30] = 1.0

        alpha = 1.0
        p_prior = 0.25
        mat = transition_matrix(hit_toy, alpha=alpha, p0=p_prior)

        # 号码 1 (索引 0): c_0 = 1.0. 期 2 号码 1 开出 -> N_0,0 = 1.0
        # 期望: (1.0 + 1.0 * 0.25) / (1.0 + 1.0) = 1.25 / 2.0 = 0.625
        self.assertAlmostEqual(mat[0, 0], 0.625, places=9)

        # 号码 1 (索引 0) 到号码 15 (索引 14): 期 2 未开出 -> N_0,14 = 0.0
        # 期望: (0.0 + 1.0 * 0.25) / (1.0 + 1.0) = 0.25 / 2.0 = 0.125
        self.assertAlmostEqual(mat[0, 14], 0.125, places=9)

        # 号码 25 (索引 24): 期 1 未开出 -> c_24 = 0.0, N_24,j = 0.0
        # 期望: (0.0 + 1.0 * 0.25) / (0.0 + 1.0) = 0.25
        self.assertAlmostEqual(mat[24, 0], 0.25, places=9)

    def test_follow_score_centered_dot(self) -> None:
        """测试跟随聚合得分: FollowScore_j = Σ_{i ∈ S_t} (M_{i->j} - p0)（契约 R16）。"""
        last_set = self.state.latest_set(100)
        w_mat = self.state.window_matrix(100, 25)
        mat = transition_matrix(w_mat, alpha=1.0, p0=P0)
        score = follow_score(mat, last_set, p0=P0)
        self.assertEqual(score.shape, (80,))
        self.assertTrue(np.all(np.isfinite(score)))

    def test_follow_diag_offdiag_split(self) -> None:
        """测试对角重号惯性分与非对角异号跟随分拆分（契约 R17）。"""
        last_set = self.state.latest_set(100)
        w_mat = self.state.window_matrix(100, 25)
        mat = transition_matrix(w_mat, alpha=1.0, p0=P0)
        diag_map, offdiag_map = follow_split({"25G": mat}, last_set, p0=P0)
        diag = diag_map["25G"]
        offdiag = offdiag_map["25G"]
        total = follow_score(mat, last_set, p0=P0)
        np.testing.assert_allclose(diag + offdiag, total, atol=1e-12)
        # 只有在 last_set 中的号码其对角分才允许非 0
        for i in range(80):
            if (i + 1) not in last_set:
                self.assertEqual(diag[i], 0.0)

    def test_second_order_transition(self) -> None:
        """测试二阶转移矩阵计算（契约 R18）。"""
        w_mat = self.state.window_matrix(100, None)
        m2 = transition_matrix_order2(w_mat, alpha=1.0, p0=P0)
        self.assertEqual(m2.shape, (80, 80))
        self.assertTrue(np.all((m2 >= 0.0) & (m2 <= 1.0)))


if __name__ == "__main__":
    unittest.main()
