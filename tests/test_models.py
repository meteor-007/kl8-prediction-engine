# -*- coding: utf-8 -*-
"""快乐8 深度学习 Transformer 与图谱贝叶斯模型单元测试。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from kl8_engine.data import Dataset, Draw
from kl8_engine.models.ensemble import (
    borda_fusion_scores,
    rank_percentile_scores,
    run_dual_ensemble,
    run_hybrid_ensemble,
)
from kl8_engine.models.spectral_bayesian import (
    compute_cooccurrence_spectral_scores,
    compute_dirichlet_scores,
    compute_momentum_differential_scores,
    compute_spectral_bayesian_scores,
)
from kl8_engine.models.transformer_net import KL8TransformerNet, train_and_predict_transformer


class TestModels(unittest.TestCase):
    """测试 Transformer、谱图贝叶斯与集成引擎。"""

    def setUp(self) -> None:
        """构造模拟时序开奖数据集。"""
        rng = np.random.RandomState(42)
        draws = []
        for i in range(1, 40):
            # 每期随机采样 20 个不重复号码
            nums = sorted(rng.choice(range(1, 81), size=20, replace=False).tolist())
            draws.append(Draw(period=2026000 + i, date="2026-01-01", numbers=tuple(nums)))
        self.dataset = Dataset(draws=tuple(draws))
        self.matrix = self.dataset.to_matrix()

    def test_transformer_net_forward(self) -> None:
        """测试 Transformer 前向形状与 Sigmoid 输出有界性。"""
        model = KL8TransformerNet(d_in=80, d_model=64, nhead=2, num_layers=1, dim_feedforward=128)
        x = torch.randn(4, 5, 80)
        out = model(x)
        self.assertEqual(out.shape, (4, 80))
        probs = torch.sigmoid(out)
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_train_and_predict_transformer(self) -> None:
        """测试轻量训练并获得下一期预测概率。"""
        probs, loss = train_and_predict_transformer(
            self.matrix,
            seq_len=5,
            epochs=5,
            batch_size=8,
            lr=0.01,
        )
        self.assertEqual(probs.shape, (80,))
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())
        self.assertGreater(loss, 0.0)

    def test_spectral_bayesian(self) -> None:
        """测试 Dirichlet、共现谱与动量差分计算。"""
        dirichlet = compute_dirichlet_scores(self.matrix, recent_window=10)
        self.assertEqual(dirichlet.shape, (80,))
        self.assertTrue((dirichlet >= 0.0).all() and (dirichlet <= 1.0).all())

        spectral = compute_cooccurrence_spectral_scores(self.matrix, window_limit=30)
        self.assertEqual(spectral.shape, (80,))
        self.assertTrue((spectral >= 0.0).all() and (spectral <= 1.0).all())

        momentum, omissions = compute_momentum_differential_scores(self.matrix, short_window=5, long_window=20)
        self.assertEqual(momentum.shape, (80,))
        self.assertEqual(omissions.shape, (80,))
        self.assertTrue((omissions >= 0).all())

        scores, _ = compute_spectral_bayesian_scores(self.matrix)
        self.assertEqual(scores.shape, (80,))

    def test_run_hybrid_ensemble(self) -> None:
        """测试混合多模型集成引擎端到端执行。"""
        res = run_hybrid_ensemble(self.dataset, dl_weight=0.5, seq_len=5, epochs=5)
        self.assertEqual(len(res.gold_pool), 4)
        self.assertEqual(len(res.main_pool), 12)
        self.assertEqual(len(res.shrinkage_pool_30), 30)
        self.assertEqual(len(res.full_order), 80)
        self.assertEqual(len(res.details), 80)

        # 检查金胆池升序与号码合法性
        self.assertEqual(res.gold_pool, tuple(sorted(res.gold_pool)))
        for n in res.gold_pool:
            self.assertTrue(1 <= n <= 80)

    def test_borda_fusion_prefers_consensus(self) -> None:
        """测试 Borda 秩融合对四分量一致排序的保持。"""
        base = np.arange(80, dtype=np.float64)
        module = {
            "dl": base,
            "dirichlet": base,
            "cooccurrence": base,
            "momentum": base,
        }
        fused = borda_fusion_scores(module)
        self.assertEqual(fused.shape, (80,))
        self.assertEqual(int(np.argmax(fused)) + 1, 80)
        self.assertAlmostEqual(float(fused[79]), 1.0)
        ranks = rank_percentile_scores(base)
        self.assertEqual(int(np.argmin(ranks)) + 1, 1)

    def test_invalid_fusion_mode_rejected(self) -> None:
        """测试未知融合方式被拒绝。"""
        with self.assertRaises(ValueError):
            run_hybrid_ensemble(self.dataset, fusion_mode="unknown")

    def test_run_dual_ensemble(self) -> None:
        """测试一次训练同时产出正式版与实验版。"""
        official, comparison = run_dual_ensemble(
            self.dataset,
            official_mode="borda",
            comparison_mode="weighted",
            seq_len=5,
            epochs=5,
        )
        self.assertIsNotNone(comparison)
        assert comparison is not None
        for res in (official, comparison):
            self.assertEqual(len(res.gold_pool), 4)
            self.assertEqual(len(res.main_pool), 12)
            self.assertEqual(len(res.shrinkage_pool_30), 30)
            self.assertEqual(len(res.full_order), 80)
        # 同一模型只训练一次，分模块得分与训练损失必须完全一致
        np.testing.assert_allclose(official.module_scores["dl"], comparison.module_scores["dl"])
        self.assertEqual(official.training_loss, comparison.training_loss)

        official_only, none_comparison = run_dual_ensemble(
            self.dataset,
            official_mode="borda",
            comparison_mode=None,
            seq_len=5,
            epochs=5,
        )
        self.assertIsNone(none_comparison)
        self.assertEqual(len(official_only.gold_pool), 4)


if __name__ == "__main__":
    unittest.main()
