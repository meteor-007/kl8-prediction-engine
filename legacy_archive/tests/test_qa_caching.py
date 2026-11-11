# -*- coding: utf-8 -*-
"""P7 QA —— ML 缓存命中/未命中 与 PAV 保序校准（契约 R23、§3.9）。

【覆盖口径】
- ``MLCache`` LRU 语义：put/get/clear、超容淘汰最久未用；
- ``train_predict`` 同一窗口（end）第二次命中缓存（refit=False），不同 end 未命中（重训）；
- ML 默认关闭 → EMA/均匀先验降级且不写缓存；
- ``build_feature_snapshot(include_ml=True/False)`` 的 ML 概率通道与默认关闭一致性；
- ``pav_calibrate`` 单调非降 + 值域 [0,1] 手算样例。
"""
from __future__ import annotations

import unittest

import numpy as np

from kl8.config import AppConfig, MLConfig
from kl8.layer1_representation import BinaryStateMatrix
from kl8.layer3_features import build_feature_snapshot
from kl8.layer3_features.ml import (
    MLCache,
    ema_fallback,
    ml_feature_names,
    pav_calibrate,
    train_predict,
)


def synthetic_state(n: int = 120, seed: int = 20261102) -> BinaryStateMatrix:
    """n 期随机 20 码状态矩阵（行和恒 20，合法）。"""
    rng = np.random.default_rng(seed)
    mat = np.zeros((n, 80), dtype=np.float64)
    for i in range(n):
        nums = rng.choice(np.arange(80), size=20, replace=False)
        mat[i, nums] = 1.0
    return BinaryStateMatrix(mat)


class TestMLCacheLRU(unittest.TestCase):
    """LRU 缓存容器行为。"""

    def test_put_get_clear(self) -> None:
        """put 保存副本、get 命中、clear 清空。"""
        cache = MLCache(max_size=2)
        key = (10, "fs-v1", "20261102")
        prob = np.full(80, 0.25)
        cache.put(key, prob)
        prob[0] = 0.99  # 修改外部数组不得污染缓存副本
        got = cache.get(key)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(float(got[0]), 0.25)
        self.assertEqual(len(cache), 1)
        cache.clear()
        self.assertIsNone(cache.get(key))
        self.assertEqual(len(cache), 0)

    def test_lru_eviction(self) -> None:
        """超出 max_size 淘汰最久未用键。"""
        cache = MLCache(max_size=2)
        k1, k2, k3 = (1, "fs-v1", "a"), (2, "fs-v1", "a"), (3, "fs-v1", "a")
        cache.put(k1, np.full(80, 0.1))
        cache.put(k2, np.full(80, 0.2))
        cache.get(k1)  # k1 变最近使用
        cache.put(k3, np.full(80, 0.3))  # 淘汰 k2
        self.assertIsNotNone(cache.get(k1))
        self.assertIsNone(cache.get(k2))
        self.assertIsNotNone(cache.get(k3))

    def test_max_size_floor(self) -> None:
        """max_size<=0 必须夹到 1，避免无限增长/崩溃。"""
        cache = MLCache(max_size=0)
        self.assertEqual(cache.max_size, 1)


class TestTrainPredictCaching(unittest.TestCase):
    """train_predict 同窗口命中、不同窗口未命中。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = synthetic_state(120)
        cls.cfg = MLConfig(enabled=True, train_window=40, n_estimators=8, refit_every=25)
        cls.end = 119

    def test_same_window_cache_hit(self) -> None:
        """同一 end 第二次调用 refit=False 且概率逐位相同，缓存条目不增长。"""
        cache = MLCache(max_size=4)
        first = train_predict(self.state, self.end, self.cfg, cache=cache)
        self.assertTrue(first.available, "本机应安装 lightgbm；若未安装请查看告警")
        self.assertEqual(first.prob.shape, (80,))
        self.assertTrue(np.all(np.isfinite(first.prob)))
        self.assertTrue(np.all(first.prob >= 0.05 - 1e-12))
        self.assertTrue(np.all(first.prob <= 0.95 + 1e-12))
        self.assertTrue(first.refit)
        self.assertEqual(len(cache), 1)

        second = train_predict(self.state, self.end, self.cfg, cache=cache)
        self.assertTrue(second.available)
        self.assertFalse(second.refit)
        np.testing.assert_array_equal(first.prob, second.prob)
        self.assertEqual(len(cache), 1)

    def test_different_window_cache_miss(self) -> None:
        """不同 end 未命中缓存 → refit=True 并新增缓存条目。"""
        cache = MLCache(max_size=4)
        r1 = train_predict(self.state, 119, self.cfg, cache=cache)
        r2 = train_predict(self.state, 118, self.cfg, cache=cache)
        self.assertTrue(r1.refit)
        self.assertTrue(r2.refit)
        self.assertEqual(len(cache), 2)
        self.assertFalse(np.array_equal(r1.prob, r2.prob))

    def test_disabled_ml_uses_fallback_without_cache(self) -> None:
        """ML 默认关闭 → EMA 降级、available=False、不写缓存。"""
        cache = MLCache(max_size=4)
        disabled = MLConfig(enabled=False)
        result = train_predict(self.state, self.end, disabled, cache=cache)
        self.assertFalse(result.available)
        self.assertFalse(result.refit)
        self.assertEqual(result.backend, "ema")
        self.assertEqual(len(cache), 0)
        self.assertEqual(result.n_train, 0)

    def test_ema_fallback_edges(self) -> None:
        """空状态/end<0 → 均匀先验 0.25；正常状态值域 [0.05,0.95]。"""
        empty = BinaryStateMatrix(np.zeros((0, 80), dtype=np.float64))
        np.testing.assert_allclose(ema_fallback(empty, 0), np.full(80, 0.25))
        np.testing.assert_allclose(ema_fallback(self.state, -1), np.full(80, 0.25))
        prob = ema_fallback(self.state, self.end)
        self.assertTrue(np.all(prob >= 0.05 - 1e-12))
        self.assertTrue(np.all(prob <= 0.95 + 1e-12))

    def test_feature_snapshot_ml_channel(self) -> None:
        """include_ml=False 默认不训练；cfg.ml.enabled=True 时 ml_available 且 ml_prob ∈[0,1]。"""
        cfg = AppConfig()
        off = build_feature_snapshot(self.state, self.end, cfg, include_ml=False)
        self.assertFalse(off.ml_available)
        np.testing.assert_allclose(off.ml_prob, np.full(80, 0.25))
        # 仅 include_ml=True 但 cfg.ml.enabled=False：不得越权训练（B4 默认关闭）
        blocked = build_feature_snapshot(self.state, self.end, cfg, include_ml=True)
        self.assertFalse(blocked.ml_available)
        ml_cfg = AppConfig(ml=MLConfig(enabled=True, train_window=40, n_estimators=8))
        on = build_feature_snapshot(
            self.state, self.end, ml_cfg, include_ml=True, ml_cache=MLCache(4)
        )
        self.assertTrue(on.ml_available)
        self.assertTrue(np.all(np.isfinite(on.ml_prob)))
        self.assertTrue(np.all(on.ml_prob >= 0.0))
        self.assertTrue(np.all(on.ml_prob <= 1.0))


class TestPAVCalibration(unittest.TestCase):
    """PAV 保序回归：单调非降、值域 [0,1]、长度与索引还原。"""

    def test_pav_merges_violating_blocks(self) -> None:
        """scores=[1,2,3], labels=[1,0,1] → 校准 [0.5,0.5,1.0]。"""
        out = pav_calibrate(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 1]))
        np.testing.assert_allclose(out, np.array([0.5, 0.5, 1.0]), atol=1e-12)

    def test_pav_monotone_and_range(self) -> None:
        """乱序输入按原索引返回；按分数排序后必须单调非降且 ∈[0,1]。"""
        scores = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
        labels = np.array([1, 0, 0, 0, 1])
        out = pav_calibrate(scores, labels)
        self.assertEqual(out.shape, scores.shape)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))
        order = np.argsort(scores)
        self.assertTrue(np.all(np.diff(out[order]) >= -1e-12))

    def test_pav_perfect_separation(self) -> None:
        """完全可分时校准值应保持 0/1 阶梯。"""
        out = pav_calibrate(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
        np.testing.assert_allclose(out, np.array([0.0, 0.0, 1.0, 1.0]), atol=1e-12)

    def test_feature_names_fixed_18(self) -> None:
        """ML 特征列名固定 18 列且无重复（R23）。"""
        names = ml_feature_names()
        self.assertEqual(len(names), 18)
        self.assertEqual(len(set(names)), 18)
        self.assertIn("pressure", names)
        self.assertIn("co_all", names)


if __name__ == "__main__":
    unittest.main()