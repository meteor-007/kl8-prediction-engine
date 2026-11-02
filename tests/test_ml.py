"""机器学习概率校准与 FeatureSnapshot 装配测试（契约 R23）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.config import AppConfig, MLConfig
from kl8.constants import N_NUM
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.features import SCHEMA_VERSION, build_feature_snapshot, validate_snapshot
from kl8.layer3_features.ml import MLCache, pav_calibrate, train_predict


class TestML(unittest.TestCase):
    """机器学习概率预测与快照装配测试。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)
        self.cfg = AppConfig()

    def test_ml_prob_bounded_and_cache(self) -> None:
        """测试概率预测有界性、PAV 保序校准与 MLCache 缓存功能（契约 R23）。"""
        # 测试 PAV 保序校准
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        cal = pav_calibrate(scores, labels)
        self.assertEqual(cal.shape, (4,))
        self.assertTrue(np.all((cal >= 0.0) & (cal <= 1.0)))

        # 测试 train_predict
        cache = MLCache(max_size=4)
        ml_cfg = MLConfig(enabled=True, train_window=60, n_estimators=10)
        res1 = train_predict(self.state, end=100, cfg=ml_cfg, cache=cache)
        self.assertEqual(res1.prob.shape, (N_NUM,))
        self.assertTrue(np.all((res1.prob >= 0.0) & (res1.prob <= 1.0)))
        self.assertEqual(len(cache), 1)

        # 二次调用命中缓存
        res2 = train_predict(self.state, end=100, cfg=ml_cfg, cache=cache)
        self.assertFalse(res2.refit)
        np.testing.assert_array_equal(res1.prob, res2.prob)

    def test_build_snapshot_schema(self) -> None:
        """测试 FeatureSnapshot 完整装配、字段形状与数值健全性（契约 R23）。"""
        snap = build_feature_snapshot(self.state, end=100, cfg=self.cfg, include_ml=False)
        self.assertEqual(snap.schema_version, SCHEMA_VERSION)
        self.assertEqual(snap.as_of_index, 100)
        self.assertEqual(snap.n_history, 101)
        # 校验所有连续特征与离散特征形状均为 80
        for name, arr in snap.arrays().items():
            self.assertEqual(arr.shape, (N_NUM,), f"字段 {name} 形状不为 80")
            self.assertTrue(np.all(np.isfinite(arr)), f"字段 {name} 包含非有限值")
        # validate_snapshot 必须正常通过不抛异常
        validate_snapshot(snap)


if __name__ == "__main__":
    unittest.main()
