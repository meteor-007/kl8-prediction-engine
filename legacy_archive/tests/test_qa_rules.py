# -*- coding: utf-8 -*-
"""P7 QA —— R1–R5 规则「刚好通过 / 差一点不通过」双点边界测试（契约 R25–R30、B2）。

【为什么必须双点】
阈值型规则只测“全满足/全不满足”无法证明边界被正确实现（> 与 >= 混用是
高频缺陷）。本文件对 R1–R5 每条规则与每个阈值端点各取「恰好等于」与
「略低于/略高于」两个点，确保实现与 8.md L131–L139 的 ≥/≤ 语义一致。
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from kl8.config import RuleConfig
from kl8.constants import N_NUM
from kl8.layer4_scoring.rules import (
    evaluate_rules,
    rule_r1,
    rule_r2,
    rule_r3,
    rule_r4,
    rule_r5,
    vote,
)


def arr_with(value: float, idx: int = 0, default: float = 0.0) -> np.ndarray:
    """构造 (80,) 数组：idx 位置为 value，其余为 default。"""
    arr = np.full(N_NUM, default, dtype=np.float64)
    arr[idx] = value
    return arr


def fake_snapshot(**overrides) -> SimpleNamespace:
    """构造仅含规则所需字段的轻量快照（不依赖特征构建，保证单测毫秒级）。"""
    base = {
        "freq_10": np.zeros(N_NUM),
        "freq_25": np.zeros(N_NUM),
        "freq_50": np.zeros(N_NUM),
        "freq_all": np.zeros(N_NUM),
        "pressure": np.zeros(N_NUM),
        "gap": np.zeros(N_NUM),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestR1Boundary(unittest.TestCase):
    """R1：4 窗口中至少 3 个 Follow 秩 ≥ 0.65。"""

    def setUp(self) -> None:
        self.cfg = RuleConfig()

    def test_r1_just_pass_three_windows_exactly_threshold(self) -> None:
        """恰好 3 个窗口 = 0.65（其余 0.649）→ 通过（>= 语义）。"""
        ranks = {
            "follow_10": arr_with(0.65),
            "follow_25": arr_with(0.65),
            "follow_50": arr_with(0.65),
            "follow_all": arr_with(0.649),
        }
        mask = rule_r1(ranks, self.cfg)
        self.assertTrue(bool(mask[0]))

    def test_r1_just_fail_only_two_windows(self) -> None:
        """只有 2 个窗口 ≥0.65 → 不通过。"""
        ranks = {
            "follow_10": arr_with(0.65),
            "follow_25": arr_with(0.65),
            "follow_50": arr_with(0.6499),
            "follow_all": arr_with(0.0),
        }
        mask = rule_r1(ranks, self.cfg)
        self.assertFalse(bool(mask[0]))

    def test_r1_missing_window_not_counted(self) -> None:
        """缺失窗口键不得计数（不得用默认 0 以外的隐式通过）。"""
        ranks = {"follow_10": arr_with(1.0), "follow_25": arr_with(1.0)}
        self.assertFalse(bool(rule_r1(ranks, self.cfg)[0]))

    def test_r1_threshold_read_from_config(self) -> None:
        """B2：阈值必须来自 cfg，改 cfg 即改变判定。"""
        ranks = {"follow_10": arr_with(0.50), "follow_25": arr_with(0.50), "follow_50": arr_with(0.50)}
        loose = RuleConfig(r1_rank_threshold=0.50)
        self.assertTrue(bool(rule_r1(ranks, loose)[0]))
        strict = RuleConfig(r1_rank_threshold=0.51)
        self.assertFalse(bool(rule_r1(ranks, strict)[0]))


class TestR2Boundary(unittest.TestCase):
    """R2：Freq25/Freq50 ≥ 0.24 且 Freq10 ∈ [0.20, 0.50]。"""

    def setUp(self) -> None:
        self.cfg = RuleConfig()

    def _snapshot(self, f10: float, f25: float, f50: float) -> SimpleNamespace:
        return fake_snapshot(
            freq_10=arr_with(f10),
            freq_25=arr_with(f25),
            freq_50=arr_with(f50),
        )

    def test_r2_just_pass_low_boundaries(self) -> None:
        """f10=0.20、f25=f50=0.24 恰好等于下界 → 通过。"""
        self.assertTrue(bool(rule_r2(self._snapshot(0.20, 0.24, 0.24), self.cfg)[0]))

    def test_r2_just_fail_f10_below_low(self) -> None:
        """f10=0.1999 < 0.20 → 不通过。"""
        self.assertFalse(bool(rule_r2(self._snapshot(0.1999, 0.30, 0.30), self.cfg)[0]))

    def test_r2_just_fail_f10_above_high(self) -> None:
        """f10=0.5001 > 0.50 → 不通过（防短期过热透支）。"""
        self.assertFalse(bool(rule_r2(self._snapshot(0.5001, 0.30, 0.30), self.cfg)[0]))

    def test_r2_high_boundary_pass(self) -> None:
        """f10=0.50 恰好等于上界 → 通过。"""
        self.assertTrue(bool(rule_r2(self._snapshot(0.50, 0.30, 0.30), self.cfg)[0]))

    def test_r2_just_fail_f25_or_f50_below_base(self) -> None:
        """f25=0.2399 或 f50=0.2399 → 不通过。"""
        self.assertFalse(bool(rule_r2(self._snapshot(0.30, 0.2399, 0.30), self.cfg)[0]))
        self.assertFalse(bool(rule_r2(self._snapshot(0.30, 0.30, 0.2399), self.cfg)[0]))


class TestR3Boundary(unittest.TestCase):
    """R3：Freq10 ≥ Freq25 ≥ Freq50 且 R(Follow_10) ≥ 0.60。"""

    def setUp(self) -> None:
        self.cfg = RuleConfig()

    def test_r3_just_pass_equal_heat_and_rank(self) -> None:
        """等频（0.30,0.30,0.30）且 follow rank=0.60 → 通过（>= 允许相等）。"""
        snap = fake_snapshot(
            freq_10=arr_with(0.30), freq_25=arr_with(0.30), freq_50=arr_with(0.30)
        )
        ranks = {"follow_10": arr_with(0.60)}
        self.assertTrue(bool(rule_r3(snap, ranks, self.cfg)[0]))

    def test_r3_just_fail_rank_below(self) -> None:
        """follow rank=0.5999 < 0.60 → 不通过。"""
        snap = fake_snapshot(
            freq_10=arr_with(0.30), freq_25=arr_with(0.30), freq_50=arr_with(0.30)
        )
        ranks = {"follow_10": arr_with(0.5999)}
        self.assertFalse(bool(rule_r3(snap, ranks, self.cfg)[0]))

    def test_r3_just_fail_heat_not_increasing(self) -> None:
        """Freq10=0.2999 < Freq25=0.30 → 不通过（即使 follow rank 满分）。"""
        snap = fake_snapshot(
            freq_10=arr_with(0.2999), freq_25=arr_with(0.30), freq_50=arr_with(0.30)
        )
        ranks = {"follow_10": arr_with(1.0)}
        self.assertFalse(bool(rule_r3(snap, ranks, self.cfg)[0]))

    def test_r3_just_fail_mid_below_long(self) -> None:
        """Freq25=0.2999 < Freq50=0.30 → 不通过。"""
        snap = fake_snapshot(
            freq_10=arr_with(0.40), freq_25=arr_with(0.2999), freq_50=arr_with(0.30)
        )
        ranks = {"follow_10": arr_with(1.0)}
        self.assertFalse(bool(rule_r3(snap, ranks, self.cfg)[0]))


class TestR4Boundary(unittest.TestCase):
    """R4：0.5·R(Co25) + 0.5·R(Co50) ≥ 0.75。"""

    def setUp(self) -> None:
        self.cfg = RuleConfig()

    def test_r4_just_pass_exactly_threshold(self) -> None:
        """均值恰好 0.75 → 通过；对称权重 0.5/0.5。"""
        ranks = {"co_25": arr_with(0.75), "co_50": arr_with(0.75)}
        self.assertTrue(bool(rule_r4(ranks, self.cfg)[0]))
        ranks2 = {"co_25": arr_with(1.0), "co_50": arr_with(0.5)}
        self.assertTrue(bool(rule_r4(ranks2, self.cfg)[0]))

    def test_r4_just_fail_just_below(self) -> None:
        """均值 0.7499 → 不通过。"""
        ranks = {"co_25": arr_with(0.7499), "co_50": arr_with(0.7499)}
        self.assertFalse(bool(rule_r4(ranks, self.cfg)[0]))

    def test_r4_missing_key_treated_as_zero(self) -> None:
        """仅 Co50 满分、Co25 缺失 → 均值 0.5 < 0.75 → 不通过。"""
        ranks = {"co_50": arr_with(1.0)}
        self.assertFalse(bool(rule_r4(ranks, self.cfg)[0]))


class TestR5Boundary(unittest.TestCase):
    """R5：Pressure ≥ 0.60 且 Gap ≤ 12 且 R(Follow_ALL) ≥ 0.70。"""

    def setUp(self) -> None:
        self.cfg = RuleConfig()

    def _snap(self, pressure: float, gap: float) -> SimpleNamespace:
        return fake_snapshot(pressure=arr_with(pressure), gap=arr_with(gap))

    def test_r5_just_pass_all_boundaries(self) -> None:
        """pressure=0.60、gap=12、rank=0.70 恰好通过。"""
        ranks = {"follow_all": arr_with(0.70)}
        self.assertTrue(bool(rule_r5(self._snap(0.60, 12.0), ranks, self.cfg)[0]))

    def test_r5_just_fail_pressure(self) -> None:
        """pressure=0.5999 → 不通过。"""
        ranks = {"follow_all": arr_with(1.0)}
        self.assertFalse(bool(rule_r5(self._snap(0.5999, 12.0), ranks, self.cfg)[0]))

    def test_r5_just_fail_gap(self) -> None:
        """gap=13 > 12 → 不通过（排除死冷号）。"""
        ranks = {"follow_all": arr_with(1.0)}
        self.assertFalse(bool(rule_r5(self._snap(1.0, 13.0), ranks, self.cfg)[0]))

    def test_r5_just_fail_follow_rank(self) -> None:
        """follow_all rank=0.6999 → 不通过。"""
        ranks = {"follow_all": arr_with(0.6999)}
        self.assertFalse(bool(rule_r5(self._snap(1.0, 0.0), ranks, self.cfg)[0]))


class TestEvaluateAndVote(unittest.TestCase):
    """规则批量求值形状/票数一致性 + 加权投票。"""

    def setUp(self) -> None:
        self.cfg = RuleConfig()
        ranks = {k: arr_with(1.0) for k in (
            "follow_10", "follow_25", "follow_50", "follow_all", "co_25", "co_50"
        )}
        snap = fake_snapshot(
            freq_10=arr_with(0.30),
            freq_25=arr_with(0.30),
            freq_50=arr_with(0.30),
            pressure=arr_with(1.0),
            gap=arr_with(0.0),
        )
        self.verdicts = evaluate_rules(snap, ranks, self.cfg)

    def test_evaluate_rules_shapes_and_order(self) -> None:
        """返回固定 5 条 (R1..R5)，mask 长度 80、dtype bool、pass_count 一致。"""
        self.assertEqual([v.rule_id for v in self.verdicts], ["R1", "R2", "R3", "R4", "R5"])
        for verdict in self.verdicts:
            self.assertEqual(verdict.mask.shape, (N_NUM,))
            self.assertEqual(verdict.mask.dtype, np.dtype(bool))
            self.assertEqual(verdict.pass_count, int(verdict.mask.sum()))

    def test_vote_counts_all_rules(self) -> None:
        """全部规则命中时共振票数 = 5，且 dtype 为 int8、值域 0..5。"""
        votes = vote(self.verdicts)
        self.assertEqual(votes.dtype, np.dtype(np.int8))
        self.assertEqual(int(votes[0]), 5)
        self.assertGreaterEqual(int(votes.min()), 0)
        self.assertLessEqual(int(votes.max()), 5)

    def test_vote_weighted_mapping_and_sequence(self) -> None:
        """加权投票：mapping 与 sequence 两种入口都必须按权重折算。"""
        weighted = vote(self.verdicts, {"R1": 2.0, "R2": 0.5})
        # R1=2.0 + R2=0.5 + R3/R4/R5 各 1.0 = 5.5
        self.assertAlmostEqual(float(weighted[0]), 5.5, delta=1e-12)
        seq = vote(self.verdicts, [2.0, 0.5, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(float(seq[0]), 5.5, delta=1e-12)

    def test_vote_partial_rules_zero_fill(self) -> None:
        """未命中的规则不贡献票数；空 verdicts → 全 0。"""
        self.assertEqual(int(vote(()) [0]), 0)


if __name__ == "__main__":
    unittest.main()