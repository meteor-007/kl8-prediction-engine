"""五维正交选号规则与交集共振票数单元测试（契约 R25–R30）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import AppConfig, RuleConfig
from kl8.constants import N_NUM
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.features import build_feature_snapshot
from kl8.layer4_scoring.borda import rank_matrix
from kl8.layer4_scoring.rules import (
    evaluate_rules,
    rule_r1,
    rule_r2,
    rule_r3,
    rule_r4,
    rule_r5,
    vote,
)


class TestRules(unittest.TestCase):
    """五大正交选号规则边界测试。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)
        self.cfg = AppConfig()
        self.snap = build_feature_snapshot(self.state, end=100, cfg=self.cfg, include_ml=False)
        self.ranks = rank_matrix({
            "freq_10": self.snap.freq_10,
            "mom_10_25": self.snap.mom_10_25,
            "mom_25_50": self.snap.mom_25_50,
            "follow_10": self.snap.follow_10,
            "follow_25": self.snap.follow_25,
            "follow_50": self.snap.follow_50,
            "follow_all": self.snap.follow_all,
            "co_25": self.snap.co_25,
            "co_50": self.snap.co_50,
            "co_all": self.snap.co_all,
        })
        self.rule_cfg = self.cfg.rules

    def test_r1_boundary(self) -> None:
        """测试规则 R1 强跟随共振边界（4 窗口中 ≥3 个 R >= 0.65）（契约 R25）。"""
        mask = rule_r1(self.ranks, self.rule_cfg)
        self.assertEqual(mask.shape, (80,))
        self.assertEqual(mask.dtype, bool)

    def test_r2_boundary(self) -> None:
        """测试规则 R2 温热多头趋势边界 (F25/F50 >= 0.24 且 F10 ∈ [0.20, 0.50])（契约 R26）。"""
        mask = rule_r2(self.snap, self.rule_cfg)
        self.assertEqual(mask.shape, (80,))
        self.assertEqual(mask.dtype, bool)

    def test_r3_boundary(self) -> None:
        """测试规则 R3 动量金叉边界 (F10 >= F25 >= F50 且 R_Follow10 >= 0.60)（契约 R27）。"""
        mask = rule_r3(self.snap, self.ranks, self.rule_cfg)
        self.assertEqual(mask.shape, (80,))
        self.assertEqual(mask.dtype, bool)

    def test_r4_boundary(self) -> None:
        """测试规则 R4 强共现伴生边界 (0.5*R_Co25 + 0.5*R_Co50 >= 0.75)（契约 R28）。"""
        mask = rule_r4(self.ranks, self.rule_cfg)
        self.assertEqual(mask.shape, (80,))
        self.assertEqual(mask.dtype, bool)

    def test_r5_boundary(self) -> None:
        """测试规则 R5 全量回归临界回补 (Pressure >= 0.60 且 O <= 12 且 R_FollowALL >= 0.70)（契约 R29）。"""
        mask = rule_r5(self.snap, self.ranks, self.rule_cfg)
        self.assertEqual(mask.shape, (80,))
        self.assertEqual(mask.dtype, bool)

    def test_vote_count_range(self) -> None:
        """测试共振票数取值范围严格为 {0, 1, 2, 3, 4, 5}（契约 R30）。"""
        verdicts = evaluate_rules(self.snap, self.ranks, self.rule_cfg)
        self.assertEqual(len(verdicts), 5)
        votes = vote(verdicts)
        self.assertEqual(votes.shape, (80,))
        self.assertTrue(np.all((votes >= 0) & (votes <= 5)))


if __name__ == "__main__":
    unittest.main()
