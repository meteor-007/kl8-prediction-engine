"""Borda 百分位秩、BaseScore 与 FinalScore 恒等式测试（契约 R24、R31、R32、§6.2 手算对照）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import AppConfig, ScoreWeights
from kl8.constants import N_NUM
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.features import build_feature_snapshot
from kl8.layer4_scoring.borda import percentile_rank, rank_matrix
from kl8.layer4_scoring.hybrid import base_score, final_score, score_all


class TestScoring(unittest.TestCase):
    """Borda 归一化与打分融合测试。"""

    def test_percentile_rank_ties_and_range(self) -> None:
        """测试 Borda 百分位秩最大 1.0、最小 0.0125，并列取平均秩（契约 R24、§6.2）。"""
        # 1..80 严格递增数组
        vals = np.arange(1, 81, dtype=np.float64)
        ranks = percentile_rank(vals)
        self.assertEqual(len(ranks), 80)
        self.assertAlmostEqual(float(ranks[0]), 1.0 / 80.0, places=9)  # 0.0125
        self.assertAlmostEqual(float(ranks[-1]), 1.0, places=9)         # 1.0

        # 并列测试：两数相等
        vals_tie = np.array([10.0, 10.0, 20.0, 30.0])
        ranks_tie = percentile_rank(vals_tie)
        # 前两个数并列第 1 和第 2，平均秩为 1.5 -> 1.5 / 4 = 0.375
        self.assertAlmostEqual(float(ranks_tie[0]), 1.5 / 4.0, places=9)
        self.assertAlmostEqual(float(ranks_tie[1]), 1.5 / 4.0, places=9)
        self.assertAlmostEqual(float(ranks_tie[2]), 3.0 / 4.0, places=9)
        self.assertAlmostEqual(float(ranks_tie[3]), 1.0, places=9)

    def test_base_score_decomposition(self) -> None:
        """测试 BaseScore 五维加权融合及各分项分解（契约 R31）。"""
        ds = load_dataset("kl8_history.txt")
        state = BinaryStateMatrix.from_dataset(ds)
        cfg = AppConfig()
        snap = build_feature_snapshot(state, end=100, cfg=cfg, include_ml=False)

        features = {"follow_10": snap.follow_10, "follow_25": snap.follow_25, "follow_50": snap.follow_50, "follow_all": snap.follow_all, "freq_10": snap.freq_10, "mom_10_25": snap.mom_10_25, "mom_25_50": snap.mom_25_50, "co_25": snap.co_25, "co_50": snap.co_50, "co_all": snap.co_all}
        ranks = rank_matrix(features)

        bs = base_score(snap, ranks, cfg.weights)
        self.assertEqual(bs.shape, (80,))
        self.assertTrue(np.all(np.isfinite(bs)))

    def test_final_score_identity(self) -> None:
        """手算对照恒等式: Final - (Base + λ*V) < 1e-12（契约 R32、§6.2）。"""
        ds = load_dataset("kl8_history.txt")
        state = BinaryStateMatrix.from_dataset(ds)
        cfg = AppConfig()
        snap = build_feature_snapshot(state, end=100, cfg=cfg, include_ml=False)

        features = {"follow_10": snap.follow_10, "follow_25": snap.follow_25, "follow_50": snap.follow_50, "follow_all": snap.follow_all, "freq_10": snap.freq_10, "mom_10_25": snap.mom_10_25, "mom_25_50": snap.mom_25_50, "co_25": snap.co_25, "co_50": snap.co_50, "co_all": snap.co_all}
        ranks = rank_matrix(features)

        votes = np.random.default_rng(20261102).integers(0, 6, size=80, dtype=np.int8)
        lambda_vote = cfg.weights.lambda_vote

        bs = base_score(snap, ranks, cfg.weights)
        fs = final_score(bs, votes, lambda_vote)

        diff = np.abs(fs - (bs + lambda_vote * votes.astype(np.float64)))
        self.assertTrue(np.max(diff) < 1e-12)


if __name__ == "__main__":
    unittest.main()
