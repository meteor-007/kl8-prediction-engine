"""序列游程检验单元测试（契约 R9）。

对应契约需求：
- runs_test 与 runs_test_80
- 聚簇连出、频繁跳跃、无偏与退化四种标签
"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.layer2_baseline.runs import runs_test, runs_test_80


class TestRuns(unittest.TestCase):
    """游程统计与标签分类测试。"""

    def test_runs_cluster_vs_jump_labels(self) -> None:
        """测试聚簇连出 (z < -1.96) 与频繁跳跃 (z > 1.96) 标签（契约 R9）。"""
        # 聚簇序列：连续长串 1，然后连续长串 0，游程极少
        cluster_seq = np.array([1] * 20 + [0] * 20, dtype=np.int32)
        stat_cluster = runs_test(cluster_seq)
        self.assertEqual(stat_cluster.runs, 2)
        self.assertTrue(stat_cluster.z < -1.96)
        self.assertEqual(stat_cluster.label, "聚簇连出")

        # 频繁跳跃序列：1 0 1 0 1 0 ... 频繁切换，游程极多
        jump_seq = np.array([1, 0] * 20, dtype=np.int32)
        stat_jump = runs_test(jump_seq)
        self.assertEqual(stat_jump.runs, 40)
        self.assertTrue(stat_jump.z > 1.96)
        self.assertEqual(stat_jump.label, "频繁跳跃")

        # 退化序列：全 1 或全 0
        degen_seq = np.zeros(30, dtype=np.int32)
        stat_degen = runs_test(degen_seq)
        self.assertEqual(stat_degen.label, "退化")

    def test_runs_test_80(self) -> None:
        """测试 80 码批量游程检验输出形状。"""
        rng = np.random.default_rng(20261102)
        mat = (rng.random((100, 80)) < 0.25).astype(np.float64)
        z_arr, runs_arr, labels = runs_test_80(mat)
        self.assertEqual(z_arr.shape, (80,))
        self.assertEqual(runs_arr.shape, (80,))
        self.assertEqual(len(labels), 80)
        valid_labels = {"聚簇连出", "频繁跳跃", "无偏", "退化"}
        self.assertTrue(all(lbl in valid_labels for lbl in labels))


if __name__ == "__main__":
    unittest.main()
