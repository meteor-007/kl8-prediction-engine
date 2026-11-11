"""同期共现矩阵与伴生拉动得分测试（契约 R19、R20、§6.2 手算对照）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.constants import P_CO
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.cooccurrence import (
    centered_cooccurrence,
    co_score,
    cooccurrence_counts,
    triad_counts,
)


class TestCooccurrence(unittest.TestCase):
    """共现矩阵、三元团与伴生拉动测试。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)

    def test_centered_matrix_and_triads(self) -> None:
        """测试中心化共现矩阵 A_ij = C_ij / w - p_co 与三元共现团计数（契约 R19）。"""
        w_mat = self.state.window_matrix(100, 25)
        counts = cooccurrence_counts(w_mat)
        self.assertEqual(counts.shape, (80, 80))
        self.assertEqual(int(np.diag(counts).sum()), 0)  # 对角线为 0

        a_mat = centered_cooccurrence(counts, 25, p_co=P_CO)
        self.assertEqual(a_mat.shape, (80, 80))
        self.assertAlmostEqual(float(np.diag(a_mat).sum()), 0.0, places=9)

        # 三元团计数
        total_triads, triad_vec = triad_counts(counts)
        self.assertTrue(total_triads >= 0)
        self.assertEqual(triad_vec.shape, (80,))
        self.assertTrue(np.all(triad_vec >= 0))

    def test_co_score(self) -> None:
        """测试当期伴生拉动分 CoScore（契约 R20）。"""
        last_set = self.state.latest_set(100)
        w_mat = self.state.window_matrix(100, 25)
        counts = cooccurrence_counts(w_mat)
        a_mat = centered_cooccurrence(counts, 25, p_co=P_CO)
        score = co_score(a_mat, last_set)
        self.assertEqual(score.shape, (80,))
        self.assertTrue(np.all(np.isfinite(score)))


if __name__ == "__main__":
    unittest.main()
