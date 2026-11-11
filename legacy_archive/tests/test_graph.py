"""共现图网络拓扑特征测试（契约 R21）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.constants import P_CO
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.cooccurrence import centered_cooccurrence, cooccurrence_counts
from kl8.layer3_features.graph import build_graph_features


class TestGraph(unittest.TestCase):
    """图网络 PageRank、特征向量中心度与 Louvain 社区测试。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)

    def test_pagerank_eigen_and_communities(self) -> None:
        """测试 PageRank、特征向量中心度与 Louvain 社区发现（契约 R21）。"""
        w_mat = self.state.window_matrix(100, 25)
        counts = cooccurrence_counts(w_mat)
        a_mat = centered_cooccurrence(counts, 25, p_co=P_CO)

        res = build_graph_features(a_mat, seed=20261102, k=3)
        self.assertEqual(res.pagerank.shape, (80,))
        self.assertTrue(np.all((res.pagerank >= 0.0) & (res.pagerank <= 1.0)))

        self.assertEqual(res.eigen.shape, (80,))
        self.assertTrue(np.all((res.eigen >= 0.0) & (res.eigen <= 1.0)))

        self.assertEqual(res.community_id.shape, (80,))
        self.assertEqual(res.community_size.shape, (80,))
        self.assertTrue(np.all(res.community_size >= 1))

        # 伴生码提取测试
        self.assertEqual(len(res.companion), 80)
        self.assertEqual(len(res.companion[1]), 3)


if __name__ == "__main__":
    unittest.main()
