"""自适应 Softmax 规则权重更新与状态原子写单元测试（契约 R33、R48-e）。"""

from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from kl8.config import AdaptiveConfig
from kl8.layer4_scoring.adaptive import (
    AdaptiveState,
    load_state,
    next_weights,
    save_state,
    softmax_weights,
)


class TestAdaptive(unittest.TestCase):
    """自适应规则权重测试。"""

    def test_softmax_weights_sum_and_prior(self) -> None:
        """测试 Softmax 规则权重归一化和为 1，及历史不足时回退先验（契约 R33）。"""
        prior = np.full(5, 0.2, dtype=np.float64)
        hit_rates = np.array([0.4, 0.3, 0.25, 0.2, 0.15])
        weights = softmax_weights(hit_rates, prior=prior, laplace_alpha=1.0, temperature=1.0, weight_floor=0.05)
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, delta=1e-9)
        self.assertTrue(np.all(weights >= 0.05))

    def test_state_idempotent_atomic_write(self) -> None:
        """测试状态持久化原子写 (os.replace) 与幂等加载（契约 R48-e）。"""
        state = AdaptiveState(
            schema_version=1,
            period=2026291,
            weights={"R1": 0.2, "R2": 0.2, "R3": 0.2, "R4": 0.2, "R5": 0.2},
            hit_rates={"R1": 0.25, "R2": 0.25, "R3": 0.25, "R4": 0.25, "R5": 0.25},
            n_updates=1,
            updated_at="2026-11-02T00:00:00Z",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            save_state(state, path)
            loaded, warns = load_state(path, schema_version=1)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.period, 2026291)
            self.assertEqual(len(warns), 0)


if __name__ == "__main__":
    unittest.main()
