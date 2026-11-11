"""理论先验与超几何基准测试（契约 R5、R6、R7、§6.2 手算对照）。"""

from __future__ import annotations

import unittest

from kl8.layer2_baseline.priors import (
    expected_repeat,
    hypergeometric_baseline,
    p0,
    p_co,
    posterior_beta,
)


class TestPriors(unittest.TestCase):
    """先验值与超几何手算对照。"""

    def test_p0_value(self) -> None:
        """手算对照 p0 = 20 / 80 = 0.25（契约 R5、§6.2）。"""
        self.assertAlmostEqual(p0(), 0.25, delta=1e-9)

    def test_p_co_value(self) -> None:
        """手算对照 p_co = 20*19 / (80*79) ≈ 0.0601265823（契约 R6、§6.2）。"""
        expected = 20 * 19 / (80 * 79)
        self.assertAlmostEqual(p_co(), expected, delta=1e-9)

    def test_expected_repeat(self) -> None:
        """手算对照相邻两期重号期望 E(Repeat) = 20 * 0.25 = 5.0（契约 R7、§6.2）。"""
        self.assertAlmostEqual(expected_repeat(), 5.0, delta=1e-9)

    def test_hypergeometric_baseline(self) -> None:
        """测试选五玩法下的超几何理论基准与选五全中理论概率。"""
        base = hypergeometric_baseline(n_pick=5)
        self.assertEqual(base.n_pick, 5)
        self.assertAlmostEqual(base.mean, 1.25, delta=1e-9)
        # 精确超几何 C(5,5)*C(75,15)/C(80,20) ≈ 0.00064492
        exact_p5 = (20 * 19 * 18 * 17 * 16) / (80 * 79 * 78 * 77 * 76)
        self.assertAlmostEqual(base.p_ge[5], exact_p5, delta=1e-9)

    def test_posterior_beta(self) -> None:
        """测试 Beta-二项后验估计。"""
        mean, low, high = posterior_beta(success=25, trials=100)
        self.assertAlmostEqual(mean, 26.0 / 102.0, delta=1e-4)
        self.assertTrue(0.0 <= low <= mean <= high <= 1.0)


if __name__ == "__main__":
    unittest.main()
