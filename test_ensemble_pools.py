# -*- coding: utf-8 -*-
"""双专家大底融合与集成模块单元测试套件

@author czw
@since 2026-11-01
"""
import unittest
from calc_ensemble_pools import PoolEnsembleEngine, TARGET_POOL_SIZE

class TestPoolEnsemble(unittest.TestCase):
    def setUp(self):
        self.p1 = [5, 7, 8, 9, 10, 11, 14, 17, 26, 30, 34, 36, 37, 39, 44, 50, 53, 55, 56, 58, 59, 60, 66, 68, 70, 73, 74, 75, 78, 80]
        self.p2 = [1, 2, 4, 5, 8, 10, 11, 12, 13, 16, 18, 21, 28, 32, 35, 37, 43, 47, 48, 51, 53, 54, 58, 59, 61, 63, 65, 70, 74, 80]
        self.mock_scores = {n: 1.0 + (n % 10) * 0.1 for n in range(1, 81)}
        self.engine = PoolEnsembleEngine(self.p1, self.p2, self.mock_scores)

    def test_01_merge_size(self):
        merged, analysis = self.engine.merge_and_optimize(TARGET_POOL_SIZE)
        self.assertEqual(len(merged), TARGET_POOL_SIZE)
        self.assertEqual(len(set(merged)), TARGET_POOL_SIZE)
        # 验证共识交集 11 码必须 100% 包含
        self.assertTrue(set(analysis["intersection"]).issubset(set(merged)))

    def test_02_ensemble_pipeline(self):
        res = self.engine.execute_ensemble_pipeline(budget_tickets=10)
        self.assertEqual(len(res.final_merged_30), TARGET_POOL_SIZE)
        self.assertEqual(len(res.tickets), 10)
        for t in res.tickets:
            self.assertEqual(len(t), 5)
            self.assertTrue(all(x in res.final_merged_30 for x in t))

if __name__ == "__main__":
    unittest.main()
