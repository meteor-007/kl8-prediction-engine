"""卡方拟合优度检验单元测试（契约 R10）。"""

from __future__ import annotations

import unittest
import numpy as np

from kl8.constants import N_NUM
from kl8.layer2_baseline.chisq import chi_square_gof


class TestChiSquare(unittest.TestCase):
    """卡方拟合优度测试。"""

    def test_gof_uniform_and_biased(self) -> None:
        """测试均匀观测与严重偏态观测的卡方统计量与 p 值（契约 R10）。"""
        w = 100
        # 均匀观测：每个号码严格出现 25 次
        uniform_obs = np.full(N_NUM, 25.0, dtype=np.float64)
        res_uniform = chi_square_gof(uniform_obs, w=w)
        self.assertAlmostEqual(res_uniform.statistic, 0.0, delta=1e-9)
        self.assertAlmostEqual(res_uniform.p_value, 1.0, delta=1e-3)
        self.assertEqual(res_uniform.dof, 79)

        # 极端偏态：前面号码超频，后面号码全为 0
        biased_obs = np.zeros(N_NUM, dtype=np.float64)
        biased_obs[:20] = 100.0  # 前 20 个号全部拉满
        res_biased = chi_square_gof(biased_obs, w=w)
        self.assertTrue(res_biased.statistic > 200.0)
        self.assertTrue(res_biased.p_value < 1e-5)


if __name__ == "__main__":
    unittest.main()
