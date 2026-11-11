# -*- coding: utf-8 -*-
"""P7 QA —— 数值正确性手算对照测试（契约 §6.2，容差 1e-9）。

【覆盖口径】
- 理论先验：p0=0.25、p_co=20·19/(80·79)、E(Repeat)=5、和值期望 810；
- 超几何：选五全中 exact PMF、均值/方差；
- Z 值：contract §6.2 手算样例 (w=10,c=5 → 1.8257418583505538)、空状态、从未开出码；
- 游程 z：Wald-Wolfowitz 手算、退化序列、聚簇/跳跃标签；
- 马尔可夫 α 平滑：逐项手算对照（一阶/二阶）；
- A=C/w−p_co、CoScore 自排除；
- Borda 百分位秩：手算、并列平均秩、(0,1] 值域；
- Trend / C̄o 聚合 / FinalScore=Base+λV 恒等 / BaseScore 分解回加；
- Pressure 高斯核峰值与半峰、μ_gap 手算与 floor；
- AC 值（契约 A4 样例）、连号/同尾/和值偏离。
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from kl8.config import AppConfig, ScoreWeights
from kl8.constants import (
    E_REPEAT,
    K_DRAW,
    N_NUM,
    P0,
    P_CO,
    SUM_EXPECT,
    hypergeometric_mean,
    hypergeometric_pmf,
    hypergeometric_var,
)
from kl8.history import load_dataset
from kl8.layer1_representation import BinaryStateMatrix
from kl8.layer1_representation.windows import effective_window, is_cold_start
from kl8.layer2_baseline.priors import expected_repeat, p0, p_co
from kl8.layer2_baseline.runs import runs_test, runs_test_80
from kl8.layer2_baseline.zscore import significant_mask, window_z
from kl8.layer3_features import build_feature_snapshot
from kl8.layer3_features.cooccurrence import centered_cooccurrence, co_score, cooccurrence_counts
from kl8.layer3_features.heat import (
    historical_mean_gap,
    omission_pressure,
)
from kl8.layer3_features.markov import (
    follow_score,
    follow_split,
    transition_matrix,
    transition_matrix_order2,
)
from kl8.layer3_features.morphology import (
    ac_value,
    consecutive_groups,
    same_tail_groups,
    sum_deviation_z,
    sum_value,
)
from kl8.layer4_scoring import rank_matrix, score_all
from kl8.layer4_scoring.borda import percentile_rank
from kl8.layer4_scoring.hybrid import co_aggregate, final_score, trend_score


def make_state(rows: list[list[int]]) -> BinaryStateMatrix:
    """由 20 码行列表构造合规 BinaryStateMatrix。"""
    mat = np.zeros((len(rows), N_NUM), dtype=np.float64)
    for idx, nums in enumerate(rows):
        arr = np.asarray(nums, dtype=np.int64) - 1
        mat[idx, arr] = 1.0
    return BinaryStateMatrix(mat)


class TestPriorsAndHypergeometric(unittest.TestCase):
    """R5–R7 + §6.2 理论基准手算。"""

    def test_p0_p_co_e_repeat_exact(self) -> None:
        """p0=0.25、p_co=20*19/(80*79)、E(Repeat)=5.0 必须逐项精确。"""
        self.assertAlmostEqual(p0(), 0.25, delta=1e-12)
        self.assertAlmostEqual(p0(), K_DRAW / N_NUM, delta=1e-12)
        self.assertAlmostEqual(p_co(), 20.0 * 19.0 / (80.0 * 79.0), delta=1e-12)
        self.assertAlmostEqual(p_co(), 0.060126582278481, delta=1e-12)
        self.assertAlmostEqual(expected_repeat(), 5.0, delta=1e-12)
        self.assertAlmostEqual(P0, 0.25, delta=1e-12)
        self.assertAlmostEqual(P_CO, 20.0 * 19.0 / (80.0 * 79.0), delta=1e-12)
        self.assertAlmostEqual(E_REPEAT, 5.0, delta=1e-12)

    def test_sum_expect_810(self) -> None:
        """和值期望 SUM_EXPECT = 20*(1+80)/2 = 810。"""
        self.assertEqual(SUM_EXPECT, 810)
        self.assertEqual(K_DRAW * (N_NUM + 1) // 2, 810)
        self.assertEqual(sum(range(1, 81)) // 4, 810)
        self.assertAlmostEqual(np.mean(np.arange(1, 81)), 40.5, delta=1e-12)
        self.assertAlmostEqual(20 * 40.5, 810.0, delta=1e-12)

    def test_hypergeometric_mean_var_pmf(self) -> None:
        """超几何均值 E=n·p、方差有限总体修正、PMF 归一。"""
        n_pick = 5
        mean = hypergeometric_mean(n_pick)
        var = hypergeometric_var(n_pick)
        self.assertAlmostEqual(mean, 5 * 0.25, delta=1e-12)
        p = 20.0 / 80.0
        expected_var = n_pick * p * (1.0 - p) * (80 - n_pick) / 79.0
        self.assertAlmostEqual(var, expected_var, delta=1e-12)
        pmf = hypergeometric_pmf(n_pick)
        self.assertAlmostEqual(float(np.sum(pmf)), 1.0, delta=1e-12)

    def test_pick5_all_hit_exact_probability(self) -> None:
        """选五全中概率必须等于 C(5,5)C(75,15)/C(80,20)（≈0.064492%，约 1550.6 期一遇）。"""
        expected = math.comb(5, 5) * math.comb(75, 15) / math.comb(80, 20)
        self.assertAlmostEqual(float(hypergeometric_pmf(5)[5]), expected, delta=1e-15)
        self.assertAlmostEqual(1.0 / expected, 1550.5686274509806, delta=1e-9)
        # 代码口径以精确超几何为准（不采用 8.md L190 的 0.155% 口径）
        self.assertNotAlmostEqual(expected, 0.00155, delta=1e-5)


class TestWindowZScore(unittest.TestCase):
    """R8 + §6.2 Z 值手算。"""

    def test_z_handcalc_contract_example(self) -> None:
        """w=10,c=5 时 Z=(5-2.5)/sqrt(10*0.25*0.75)=1.8257418583505538。"""
        rows = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ] * 5 + [
            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        ] * 5
        state = make_state(rows)
        z = window_z(state, end=9, w=10)
        expected = (5.0 - 2.5) / math.sqrt(10.0 * 0.25 * 0.75)
        self.assertAlmostEqual(float(z[0]), expected, delta=1e-12)
        self.assertAlmostEqual(float(z[0]), 1.8257418583505538, delta=1e-12)

    def test_z_empty_state_returns_zeros(self) -> None:
        """T=0 空状态必须返回 (80,) 全零，不做除零。"""
        state = BinaryStateMatrix(np.zeros((0, N_NUM), dtype=np.float64))
        z = window_z(state, end=0, w=10)
        np.testing.assert_array_equal(z, np.zeros(N_NUM))

    def test_z_never_seen_number_finite(self) -> None:
        """从未开出号码的 Z 必须有限且为负，不得出现 NaN/Inf。"""
        state = make_state([[i for i in range(1, 21)]] * 10)
        z = window_z(state, end=9, w=10)
        self.assertTrue(np.all(np.isfinite(z)))
        expected = (0.0 - 2.5) / math.sqrt(10.0 * 0.25 * 0.75)
        self.assertAlmostEqual(float(z[79]), expected, delta=1e-12)

    def test_z_significant_threshold_strict(self) -> None:
        """显著性判定必须是严格 |Z| > 1.96。"""
        z = np.array([1.95, 1.96, 1.97, -1.97])
        mask = significant_mask(z)
        np.testing.assert_array_equal(mask, np.array([False, False, True, True]))


class TestColdStartWindow(unittest.TestCase):
    """R3 冷启动公式 w_eff = max(floor, min(w, T))。"""

    def test_effective_window_formula(self) -> None:
        """T<w 回退 T，T=0 回退 floor=1，ALL 用 T。"""
        self.assertEqual(effective_window(50, 5), 5)
        self.assertEqual(effective_window(10, 10), 10)
        self.assertEqual(effective_window(10, 200), 10)
        self.assertEqual(effective_window(None, 5), 5)
        self.assertEqual(effective_window(50, 0), 1)
        self.assertEqual(effective_window(50, 0, cold_start_floor=3), 3)
        self.assertTrue(is_cold_start(50, 5))
        self.assertFalse(is_cold_start(50, 50))
        self.assertFalse(is_cold_start(50, 60))
        self.assertTrue(is_cold_start(None, 0))


class TestRunsTest(unittest.TestCase):
    """R9 游程检验手算与退化边界。"""

    def test_runs_handcalc(self) -> None:
        """序列 1100110001：runs=5, n1=n0=5, E=6, Var=2.2222, z=-0.67082, 无偏。"""
        series = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 1], dtype=np.int32)
        stat = runs_test(series)
        self.assertEqual(stat.n1, 5)
        self.assertEqual(stat.n0, 5)
        self.assertEqual(stat.runs, 5)
        self.assertAlmostEqual(stat.expected, 6.0, delta=1e-12)
        self.assertAlmostEqual(stat.variance, 50.0 * 40.0 / 900.0, delta=1e-12)
        self.assertAlmostEqual(stat.z, -1.0 / math.sqrt(2.2222222222222223), delta=1e-12)
        self.assertEqual(stat.label, "无偏")

    def test_runs_degenerate_all_zero_all_one(self) -> None:
        """全 0 / 全 1 / 空 / 单点均为「退化」，z=0，禁止 NaN。"""
        for series in (
            np.zeros(10, dtype=np.int32),
            np.ones(10, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.ones(1, dtype=np.int32),
        ):
            with self.subTest(n=len(series)):
                stat = runs_test(series)
                self.assertEqual(stat.label, "退化")
                self.assertEqual(stat.z, 0.0)
        self.assertEqual(runs_test(np.zeros(10, dtype=np.int32)).runs, 1)
        self.assertEqual(runs_test(np.zeros(0, dtype=np.int32)).runs, 0)

    def test_runs_cluster_vs_alternating_labels(self) -> None:
        """聚簇 [1]*10+[0]*10 → 聚簇连出；交替 1010… → 频繁跳跃。"""
        clustered = runs_test(np.array([1] * 10 + [0] * 10, dtype=np.int32))
        self.assertEqual(clustered.runs, 2)
        self.assertEqual(clustered.label, "聚簇连出")
        alternating = runs_test(np.array(([1, 0] * 20), dtype=np.int32))
        self.assertEqual(alternating.runs, 40)
        self.assertEqual(alternating.label, "频繁跳跃")

    def test_runs_test_80_shapes(self) -> None:
        """80 码批量游程检验输出形状/标签长度（含全 0 列）。"""
        state = make_state([[i for i in range(1, 21)]] * 20)
        z, runs, labels = runs_test_80(state.hit)
        self.assertEqual(z.shape, (N_NUM,))
        self.assertEqual(runs.shape, (N_NUM,))
        self.assertEqual(len(labels), N_NUM)
        self.assertEqual(labels[79], "退化")
        self.assertTrue(np.all(np.isfinite(z)))


class TestMarkovAlphaSmoothing(unittest.TestCase):
    """R15–R18 马尔可夫 α 平滑逐项手算。"""

    def test_transition_matrix_alpha_handcalc(self) -> None:
        """M[i→j]=(n_ij+α·p0)/(c_i+α) 逐项手算（α=1，3 期窗口）。"""
        rows = [
            list(range(1, 21)),
            list(range(1, 21)),
            list(range(21, 41)),
        ]
        mat = np.asarray(
            [[1.0 if n in row else 0.0 for n in range(1, 81)] for row in rows],
            dtype=np.float64,
        )
        m = transition_matrix(mat, alpha=1.0, p0=0.25)
        # 号码 1：上一期（row0,row1）出现 2 次，下一期（row1,row2）出现 1 次
        self.assertAlmostEqual(float(m[0, 0]), (1.0 + 0.25) / 3.0, delta=1e-12)
        # 号码 1→21：row1→row2 发生一次跟随
        self.assertAlmostEqual(float(m[0, 20]), (1.0 + 0.25) / 3.0, delta=1e-12)
        # 号码 21：上一期（row0,row1）从未出现 → c=0，分母 α=1，退化为 0.25
        self.assertAlmostEqual(float(m[20, 20]), 0.25, delta=1e-12)
        # 窗口不足 2 期 → 均匀先验 p0
        single = transition_matrix(mat[:1], alpha=1.0, p0=0.25)
        np.testing.assert_allclose(single, np.full((80, 80), 0.25))

    def test_transition_order2_handcalc(self) -> None:
        """二阶 M^(2)：τ-2 期 i 对 τ 期 j 的拉动，α=1 手算。"""
        rows = [
            list(range(1, 21)),
            list(range(21, 41)),
            list(range(21, 41)),
            list(range(1, 21)),
        ]
        mat = np.asarray(
            [[1.0 if n in row else 0.0 for n in range(1, 81)] for row in rows],
            dtype=np.float64,
        )
        m2 = transition_matrix_order2(mat, alpha=1.0, p0=0.25)
        # i=1 仅出现在 row0（c=1），τ-2 到 τ 无一次命中（n=0）→ (0+0.25)/2
        self.assertAlmostEqual(float(m2[0, 0]), (0.0 + 0.25) / 2.0, delta=1e-12)
        # 不足 3 期 → 均匀先验
        np.testing.assert_allclose(
            transition_matrix_order2(mat[:2], alpha=1.0, p0=0.25),
            np.full((80, 80), 0.25),
        )

    def test_follow_score_and_split(self) -> None:
        """FollowScore_j=Σ_{i∈S_t}(M_ij−p0)；对角+异号拆分必须精确回加。"""
        matrix = np.full((80, 80), 0.5, dtype=np.float64)
        current = frozenset(range(1, 21))
        follow = follow_score(matrix, current, p0=0.25)
        np.testing.assert_allclose(follow, np.full(80, 20 * 0.25), atol=1e-12)
        diag, offdiag = follow_split({"ALL": matrix}, current, p0=0.25)
        np.testing.assert_allclose(
            diag["ALL"] + offdiag["ALL"], follow, atol=1e-12
        )
        # 对角分量仅对当期号码非零
        self.assertEqual(float(diag["ALL"][0]), 0.25)
        self.assertEqual(float(diag["ALL"][79]), 0.0)


class TestCooccurrenceCentered(unittest.TestCase):
    """R19/R20 A=C/w−p_co 与 CoScore 手算。"""

    def test_centered_cooccurrence_handcalc(self) -> None:
        """两期均含 1..20：C_12=2, w=2 → A_12=1−p_co；对角恒为 0。"""
        rows = [list(range(1, 21)), list(range(1, 21))]
        mat = np.asarray(
            [[1.0 if n in row else 0.0 for n in range(1, 81)] for row in rows],
            dtype=np.float64,
        )
        counts = cooccurrence_counts(mat)
        self.assertEqual(int(counts[0, 1]), 2)
        self.assertEqual(int(counts[0, 0]), 0)
        a = centered_cooccurrence(counts, 2)
        self.assertAlmostEqual(float(a[0, 1]), 1.0 - P_CO, delta=1e-12)
        self.assertAlmostEqual(float(a[0, 79]), -P_CO, delta=1e-12)
        self.assertEqual(float(a[0, 0]), 0.0)

    def test_co_score_self_excluded(self) -> None:
        """CoScore_j=Σ_{i∈S_t,i≠j}A_ij；j∈S_t 时自身项因对角为 0 自动剔除。"""
        rows = [list(range(1, 21)), list(range(1, 21))]
        mat = np.asarray(
            [[1.0 if n in row else 0.0 for n in range(1, 81)] for row in rows],
            dtype=np.float64,
        )
        a = centered_cooccurrence(cooccurrence_counts(mat), 2)
        co = co_score(a, frozenset(range(1, 21)))
        self.assertAlmostEqual(float(co[0]), 19 * (1.0 - P_CO), delta=1e-12)
        self.assertAlmostEqual(float(co[20]), -20 * P_CO, delta=1e-12)


class TestBordaRanks(unittest.TestCase):
    """R24 Borda 百分位秩 R=Rank/80 ∈ (0,1]。"""

    def test_percentile_rank_handcalc(self) -> None:
        """升序 4 元 [10,20,30,40] → 0.25/0.5/0.75/1.0。"""
        ranks = percentile_rank(np.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_allclose(ranks, np.array([0.25, 0.5, 0.75, 1.0]), atol=1e-12)

    def test_percentile_rank_ties_average(self) -> None:
        """并列名次取平均秩：[5,5,7] → 0.5,0.5,1.0。"""
        ranks = percentile_rank(np.array([5.0, 5.0, 7.0]))
        np.testing.assert_allclose(ranks, np.array([0.5, 0.5, 1.0]), atol=1e-12)
        all_same = percentile_rank(np.ones(8))
        np.testing.assert_allclose(all_same, np.full(8, 0.5625), atol=1e-12)

    def test_percentile_rank_range_80(self) -> None:
        """80 码随机指标：值域 ∈ (0,1]，最大 1.0，最小 1/80=0.0125。"""
        vals = np.random.default_rng(7).normal(size=80)
        ranks = percentile_rank(vals)
        self.assertTrue(np.all(ranks > 0.0))
        self.assertTrue(np.all(ranks <= 1.0 + 1e-15))
        self.assertAlmostEqual(float(np.max(ranks)), 1.0, delta=1e-15)
        self.assertAlmostEqual(float(np.min(ranks)), 1.0 / 80.0, delta=1e-15)


class TestHybridIdentities(unittest.TestCase):
    """R31/R32 + A1/A2 恒等式与分解回加。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = AppConfig()
        ds = load_dataset("kl8_history.txt")
        state = BinaryStateMatrix.from_dataset(ds)
        cls.snapshot = build_feature_snapshot(state, end=100, cfg=cls.cfg, include_ml=False)
        cls.ranks = rank_matrix(
            {
                "freq_10": cls.snapshot.freq_10,
                "mom_10_25": cls.snapshot.mom_10_25,
                "mom_25_50": cls.snapshot.mom_25_50,
                "follow_10": cls.snapshot.follow_10,
                "follow_25": cls.snapshot.follow_25,
                "follow_50": cls.snapshot.follow_50,
                "follow_all": cls.snapshot.follow_all,
                "co_10": cls.snapshot.co_10,
                "co_25": cls.snapshot.co_25,
                "co_50": cls.snapshot.co_50,
                "co_all": cls.snapshot.co_all,
            }
        )

    def test_final_score_identity(self) -> None:
        """FinalScore = BaseScore + λ·V 必须对 λ∈[0.05,0.10] 精确成立（<1e-12）。"""
        rng = np.random.default_rng(20261102)
        base = rng.random(80)
        votes = rng.integers(0, 6, size=80).astype(np.int8)
        for lam in (0.05, 0.07, 0.10):
            final = final_score(base, votes, lam)
            residual = np.max(np.abs(final - (base + lam * votes.astype(np.float64))))
            self.assertLess(residual, 1e-12)

    def test_score_all_final_identity_and_decomposition(self) -> None:
        """score_all 的 final 恒等且 BaseScore 五维分解回加一致（B5 全均匀时 λ·V）。"""
        votes = np.arange(80, dtype=np.int8) % 6
        result = score_all(self.snapshot, self.ranks, self.cfg.weights, votes=votes)
        residual = np.max(
            np.abs(
                result.final
                - (result.base + self.cfg.weights.lambda_vote * votes.astype(np.float64))
            )
        )
        self.assertLess(residual, 1e-12)
        contrib = result.contributors
        rebuilt = (
            contrib["follow_10"]
            + contrib["follow_25"]
            + contrib["follow_50"]
            + contrib["follow_all"]
            + contrib["co"]
            + contrib["trend"]
            + contrib["omission"]
        )
        np.testing.assert_allclose(rebuilt, result.base, atol=1e-12)
        # 排序：FinalScore 降序，且 order 是 1..80 的排列
        self.assertEqual(len(result.order), 80)
        self.assertEqual(sorted(result.order), list(range(1, 81)))
        ordered_final = np.array([result.final[n - 1] for n in result.order])
        self.assertTrue(np.all(np.diff(ordered_final) <= 1e-12))

    def test_trend_score_formula(self) -> None:
        """Trend = 0.5R(F10)+0.3R(Mom10-25)+0.2R(Mom25-50)（裁决 A1）。"""
        ranks = {
            "freq_10": np.full(80, 0.4),
            "mom_10_25": np.full(80, 0.8),
            "mom_25_50": np.full(80, 0.6),
        }
        trend = trend_score(ranks, self.cfg.weights)
        np.testing.assert_allclose(trend, np.full(80, 0.5 * 0.4 + 0.3 * 0.8 + 0.2 * 0.6), atol=1e-12)

    def test_co_aggregate_formula(self) -> None:
        """契约 §5 A2：C̄o = mean(R(Co25), R(Co50), R(CoALL))，等权、不含 10G。"""
        ranks = {
            "co_25": np.full(80, 0.3),
            "co_50": np.full(80, 0.6),
            "co_all": np.full(80, 0.9),
            "co_10": np.full(80, 0.1),
        }
        agg = co_aggregate(ranks)
        expected = (0.3 + 0.6 + 0.9) / 3.0
        np.testing.assert_allclose(agg, np.full(80, expected), atol=1e-12)
        # 显式确认 10G 不参与聚合
        ranks_without_10 = {k: v for k, v in ranks.items() if k != "co_10"}
        np.testing.assert_allclose(
            co_aggregate(ranks_without_10), agg, atol=1e-12
        )

    def test_default_ml_weight_zero(self) -> None:
        """B4：默认 ScoreWeights.ml=0，ML 概率不得进入 BaseScore。"""
        self.assertEqual(ScoreWeights().ml, 0.0)


class TestPressureAndGapMean(unittest.TestCase):
    """R13/R14 遗漏回补压力手算与 μ_gap 边界。"""

    def test_pressure_gaussian_peak_and_half(self) -> None:
        """O/μ=1.2 → Pressure=1.0；O/μ=0.6 → exp(-0.5)=0.6065306597。"""
        gap = np.array([12.0, 6.0])
        gap_mean = np.array([10.0, 10.0])
        pressure = omission_pressure(gap, gap_mean)
        self.assertAlmostEqual(float(pressure[0]), 1.0, delta=1e-12)
        self.assertAlmostEqual(float(pressure[1]), math.exp(-0.5), delta=1e-12)

    def test_pressure_zero_gap_mean_floor(self) -> None:
        """μ_gap=0 时必须以 floor=3 作分母，输出有限且为 exp(-2)。"""
        pressure = omission_pressure(np.array([0.0]), np.array([0.0]), floor=3.0)
        self.assertAlmostEqual(float(pressure[0]), math.exp(-2.0), delta=1e-12)
        self.assertTrue(np.all(np.isfinite(pressure)))

    def test_historical_mean_gap_handcalc(self) -> None:
        """号码 1 只在 10 期中出现 1 次 → μ_gap=10；从未开出 → floor=3。"""
        rows = [[1] + list(range(2, 21))] + [list(range(21, 41))] * 9
        state = make_state(rows)
        mean_gap = historical_mean_gap(state, end=9)
        self.assertAlmostEqual(float(mean_gap[0]), 10.0, delta=1e-12)
        self.assertAlmostEqual(float(mean_gap[20]), 3.0, delta=1e-12)
        self.assertTrue(np.all(mean_gap >= 3.0))


class TestMorphologyHandcalc(unittest.TestCase):
    """R22 + 裁决 A4 手算。"""

    def test_ac_value_contract_example(self) -> None:
        """契约 §6.2：{1,4,6,9,11} 差集 {2,3,5,7,8,10}(6) → AC=6-(5-1)=2。"""
        self.assertEqual(ac_value([1, 4, 6, 9, 11]), 2)
        self.assertEqual(ac_value([1, 4, 9]), 1)
        self.assertEqual(ac_value([5]), 0)

    def test_consecutive_and_same_tail(self) -> None:
        """连号 {1,2,3,5,6} → 三连1/二连1/max_run3；同尾 1,11,21 → 尾1计3。"""
        cons = consecutive_groups([1, 2, 3, 5, 6])
        self.assertEqual(cons["三连"], 1)
        self.assertEqual(cons["二连"], 1)
        self.assertEqual(cons["max_run"], 3)
        tails = same_tail_groups([1, 11, 21, 32])
        self.assertEqual(tails[1], 3)
        self.assertEqual(tails[2], 1)

    def test_sum_value_and_deviation(self) -> None:
        """和值=810 时偏离 z=0；全小和值偏离为负。"""
        self.assertEqual(sum_value(range(1, 21)), 210)
        zero_dev = list(range(1, 11)) + list(range(71, 81))  # 和值 55+755=810
        self.assertEqual(sum(zero_dev), 810)
        self.assertAlmostEqual(sum_deviation_z(zero_dev), 0.0, delta=1e-12)
        self.assertLess(sum_deviation_z(list(range(1, 21))), 0.0)
        self.assertEqual(sum_deviation_z([]), 0.0)


if __name__ == "__main__":
    unittest.main()