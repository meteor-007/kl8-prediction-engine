"""号码池过滤、硬约束与配比控制测试（契约 R34–R36）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import AppConfig, PoolConfig
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.features import build_feature_snapshot
from kl8.layer4_scoring.filter import apply_repeat_quota, build_pool, select_gold


class TestFilter(unittest.TestCase):
    """候选池硬约束、金胆与重号配比测试。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)
        self.cfg = AppConfig()
        self.snap = build_feature_snapshot(self.state, end=100, cfg=self.cfg, include_ml=False)

    def test_hard_constraints(self) -> None:
        """测试空间形态均衡过滤硬约束与冲突有序放宽（契约 R34）。"""
        scores = np.linspace(0.8, 0.1, 80)
        votes = np.full(80, 2, dtype=np.int8)
        last_draw = self.state.latest_set(100)
        pool = build_pool(self.snap, scores, votes, last_draw, pool_size=12, cfg=self.cfg)
        self.assertEqual(len(pool.main), 12)
        self.assertEqual(pool.size, 12)

    def test_gold_and_main_sizes(self) -> None:
        """测试核心金胆池 (3~5 码) 与复式池 (10~15 码，默认 12) 规模（契约 R35）。"""
        scores = np.linspace(0.8, 0.1, 80)
        votes = np.full(80, 2, dtype=np.int8)
        last_draw = self.state.latest_set(100)
        pool = build_pool(self.snap, scores, votes, last_draw, pool_size=12, cfg=self.cfg)
        self.assertTrue(3 <= len(pool.gold) <= 5)
        self.assertTrue(10 <= len(pool.main) <= 15)
        self.assertEqual(len(pool.main), 12)
        # 金胆必须包含在复式池中
        self.assertTrue(set(pool.gold).issubset(set(pool.main)))

    def test_repeat_quota(self) -> None:
        """测试重号配比控制在 25%~35%（默认 30%）（契约 R36）。"""
        order = list(range(1, 81))
        last_draw = frozenset(range(1, 21))
        p_cfg = PoolConfig(repeat_ratio=0.30, repeat_ratio_min=0.25, repeat_ratio_max=0.35)
        res, warns = apply_repeat_quota(order, last_draw, p_cfg, size=12)
        self.assertEqual(len(res), 12)
        rep_count = sum(1 for n in res if n in last_draw)
        # 12 * 0.30 = 3.6 -> 4 码 (或在 3..4 之间)
        self.assertTrue(3 <= rep_count <= 4)


if __name__ == "__main__":
    unittest.main()
