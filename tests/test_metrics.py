"""评价指标与统计量单测（契约 R43–R46、§3.12）。

【P10 自证项】
- v1.3.0 三条件显著性：``n ≥ min_n_signal`` + ``lift > lift_signal`` + ``p < alpha_adj``，
  其中 ``alpha_adj = alpha_sig/len(ks)``（Bonferroni）；三个条件必须**逐条**可证伪；
- §6.6 静态红线：渲染层零硬编码/零占位符、全仓零裸 except、零硬编码 ``no_lookahead=True``。
"""
import unittest
from pathlib import Path

import numpy as np

from kl8.config import BacktestConfig
from kl8.constants import DEFAULT_PRIZE_TABLE
from kl8.datatypes import BacktestReport, BacktestRow
from kl8.evaluation.metrics import (
    bankroll,
    high_tier_rate,
    lift_at_k,
    rule_attribution,
    self_check,
    t_test_one_sided,
)


class TestMetrics(unittest.TestCase):
    """回测评价指标测试集。"""

    def setUp(self):
        self.cfg = BacktestConfig(ks=(3, 5), lift_signal=1.05, alpha_sig=0.05)

    def test_t_test_one_sided_boundary(self):
        """单侧 t 检验边界：n < 2 返回 (0.0, 1.0)。"""
        t, p = t_test_one_sided(1.5, 1.25, 0.5, 1)
        self.assertEqual(t, 0.0)
        self.assertEqual(p, 1.0)

    def test_lift_at_k_handcalc(self):
        """Top-K 提升度手算比对测试（契约 R43）。"""
        # 构造 10 期样本，每期 Top-5 命中 2 个（理论预期 5 * 0.25 = 1.25）
        rows = []
        for i in range(10):
            rows.append(BacktestRow(
                period=2026001 + i,
                as_of_index=i,
                as_of_period=2026000 + i,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={3: (1, 2, 3), 5: (1, 2, 3, 4, 5)},
                hit_at={3: 1, 5: 2},  # Top-5 恒定命中 2
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
            ))

        lift_res = lift_at_k(rows, (3, 5), self.cfg)
        self.assertIn(5, lift_res)
        stat5 = lift_res[5]

        # 预期 mean_hit = 2.0, expected = 1.25, lift = 2.0 / 1.25 = 1.6
        self.assertAlmostEqual(stat5.mean_hit, 2.0, places=6)
        self.assertAlmostEqual(stat5.expected, 1.25, places=6)
        self.assertAlmostEqual(stat5.lift, 1.6, places=6)
        self.assertEqual(stat5.k, 5)

    def test_high_tier_vs_hypergeometric(self):
        """高奖级实际触发率对比超几何理论概率（契约 R44）。"""
        rows = [
            BacktestRow(
                period=2026001,
                as_of_index=0,
                as_of_period=2026000,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                best_ticket_hit=5,  # 选五中五
            ),
            BacktestRow(
                period=2026002,
                as_of_index=1,
                as_of_period=2026001,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                best_ticket_hit=3,
            ),
        ]

        tier_res = high_tier_rate(rows, "pick5", DEFAULT_PRIZE_TABLE)
        self.assertIn("pick5:5", tier_res)
        self.assertEqual(tier_res["pick5:5"].observed_rate, 0.5)
        self.assertGreater(tier_res["pick5:5"].theory_rate, 0.0)

    def test_high_tier_uses_per_ticket_denominator(self):
        """R44 口径自证：有逐注命中时，分母必须是总注数而非期数（P10 修复）。"""
        rows = [
            BacktestRow(
                period=2026001 + i,
                as_of_index=i,
                as_of_period=2026000 + i,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                best_ticket_hit=5,
                # 每期 4 注：1 注中 5、1 注中 4、2 注中 1 → 单注口径 5 中奖率 1/4
                ticket_hits=(5, 4, 1, 1),
            )
            for i in range(10)
        ]
        tier = high_tier_rate(rows, "pick5", DEFAULT_PRIZE_TABLE)
        self.assertEqual(tier["pick5:5"].n_rounds, 10)
        self.assertEqual(tier["pick5:5"].n_trials, 40)
        self.assertAlmostEqual(tier["pick5:5"].observed_rate, 0.25, places=12)
        # 「中四及以上」= 每期 2 注（5 与 4）× 10 期 / 40 注 = 50%
        self.assertAlmostEqual(tier["pick5:4+5"].observed_rate, 0.5, places=12)
        # 单注口径下实测比率不应出现"每期最佳票"式的 10 倍级放大
        self.assertLess(tier["pick5:4+5"].ratio, 100.0)

    def test_bankroll_integer_yuan(self):
        """资金曲线与最大回撤纯整数元核算（契约 R46）。"""
        rows = [
            BacktestRow(
                period=2026001,
                as_of_index=0,
                as_of_period=2026000,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                cost_yuan=20,
                prize_yuan=0,
            ),
            BacktestRow(
                period=2026002,
                as_of_index=1,
                as_of_period=2026001,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                cost_yuan=20,
                prize_yuan=100,
            ),
        ]

        b_stat = bankroll(rows, DEFAULT_PRIZE_TABLE)
        self.assertEqual(b_stat.n_rounds, 2)
        self.assertEqual(b_stat.total_cost_yuan, 40)
        self.assertEqual(b_stat.total_prize_yuan, 100)
        self.assertEqual(b_stat.net_yuan, 60)
        self.assertAlmostEqual(b_stat.roi, 1.5, places=4)
        self.assertEqual(b_stat.max_drawdown_yuan, 20)
        self.assertEqual(b_stat.best_round_yuan, 100)

    def test_rule_attribution(self):
        """规则归因与投票共振统计（M14 反假绿：校验真实数值而非只查键存在）。"""
        votes = np.array([3] * 10 + [1] * 70, dtype=np.int8)
        scores = np.ones(80)
        rows = [
            BacktestRow(
                period=2026001 + i,
                as_of_index=i,
                as_of_period=2026000 + i,
                scores=scores,
                votes=votes,
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                # R1 每期命中 10 个 / 池 10 码；R2 命中 0 个 / 池 10 码
                rule_stats={"R1": (10, 10), "R2": (0, 10)},
                # 高票组 20 期 × 10 码全命中；低票组 20 期 × 70 码全不中
                vote_group_stats={"high": (10, 10), "low": (0, 70)},
            )
            for i in range(20)
        ]
        attrib = rule_attribution(rows)
        self.assertIn("R1", attrib.per_rule)
        stat_r1 = attrib.per_rule["R1"]
        self.assertAlmostEqual(stat_r1.pool_size_avg, 10.0, places=9)
        # 20 期 × 10 命中 / 200 池 = 100%
        self.assertAlmostEqual(stat_r1.hit_rate, 1.0, places=9)
        # 理论基线 p0 = 25% → Lift = 4.0
        self.assertAlmostEqual(stat_r1.lift, 4.0, places=6)
        self.assertGreater(stat_r1.z, 3.0)

        stat_r2 = attrib.per_rule["R2"]
        self.assertAlmostEqual(stat_r2.hit_rate, 0.0, places=9)
        self.assertAlmostEqual(stat_r2.lift, 0.0, places=9)

        self.assertEqual(int(attrib.high_votes["n"]), 20 * 10)
        self.assertEqual(int(attrib.low_votes["n"]), 20 * 70)
        # 高票组命中率 100% vs 低票组 0% → 两比例检验必须显著为正
        self.assertGreater(attrib.high_votes["z"], 3.0)


def _rows_with_hits(hits: list[int], k: int = 5) -> list[BacktestRow]:
    """构造只关心 Top-k 命中数的回测行序列。"""
    rows: list[BacktestRow] = []
    for i, hit in enumerate(hits):
        rows.append(
            BacktestRow(
                period=2026001 + i,
                as_of_index=i,
                as_of_period=2026000 + i,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={k: tuple(range(1, k + 1))},
                hit_at={k: hit},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
            )
        )
    return rows


class TestLiftSignalV130(unittest.TestCase):
    """v1.3.0：LiftStat 三条件显著性（n 门槛 + Lift 门槛 + Bonferroni p 门槛）。"""

    def setUp(self) -> None:
        self.cfg = BacktestConfig(
            ks=(3, 5),
            lift_signal=1.05,
            alpha_sig=0.05,
            min_n_signal=500,
            bonferroni=True,
        )

    def test_lift_signal_requires_n500_and_bonferroni(self) -> None:
        """n=501 → 显著；n=499 → 仅因样本量不足不显著；p 不达标 → 不显著。"""
        # 交替命中 3/2：均值 2.5（期望 1.25，Lift=2.0），有方差 → t 检验 p 极小
        hits_sig = [3, 2] * 250 + [3]  # 501 期
        res_ok = lift_at_k(_rows_with_hits(hits_sig), (3, 5), self.cfg)
        stat_ok = res_ok[5]
        self.assertEqual(stat_ok.n, 501)
        self.assertAlmostEqual(stat_ok.alpha_adj, self.cfg.alpha_sig / 2.0, places=12)
        self.assertGreater(stat_ok.lift, self.cfg.lift_signal)
        self.assertLess(stat_ok.p_value, stat_ok.alpha_adj)
        self.assertTrue(stat_ok.is_signal, "n≥500、Lift、p 三条件全满足必须判显著")

        # 同样统计形态，但少 2 期（n=499）→ 只差样本量门槛，必须不显著
        hits_short = hits_sig[:499]
        res_short = lift_at_k(_rows_with_hits(hits_short), (3, 5), self.cfg)
        stat_short = res_short[5]
        self.assertEqual(stat_short.n, 499)
        self.assertGreater(stat_short.lift, self.cfg.lift_signal)
        self.assertLess(stat_short.p_value, stat_short.alpha_adj)
        self.assertFalse(stat_short.is_signal, "n=499 < min_n_signal 时不得判显著")
        # 期望显著性与非显著性差异只来自 n，其余统计量应同量级
        self.assertAlmostEqual(stat_ok.mean_hit, stat_short.mean_hit, places=3)

        # p 边界：501 期但每期恒定命中 2 个 → 方差 0 → p=1，尽管 Lift=1.6 也不显著
        res_flat = lift_at_k(_rows_with_hits([2] * 501), (3, 5), self.cfg)
        stat_flat = res_flat[5]
        self.assertEqual(stat_flat.n, 501)
        self.assertGreater(stat_flat.lift, self.cfg.lift_signal)
        self.assertEqual(stat_flat.p_value, 1.0)
        self.assertFalse(stat_flat.is_signal, "p 值不达标时不得判显著")

    def test_bonferroni_switch_changes_alpha(self) -> None:
        """bonferroni=False → alpha_adj=alpha_sig；True → alpha_sig/len(ks)。"""
        hits = [3, 2] * 250 + [3]
        rows = _rows_with_hits(hits)
        cfg_nb = BacktestConfig(
            ks=(3, 5), lift_signal=1.05, alpha_sig=0.05, min_n_signal=500, bonferroni=False
        )
        stat_nb = lift_at_k(rows, (3, 5), cfg_nb)[5]
        self.assertAlmostEqual(stat_nb.alpha_adj, cfg_nb.alpha_sig, places=12)
        self.assertTrue(stat_nb.is_signal)
        stat_b = lift_at_k(rows, (3, 5), self.cfg)[5]
        self.assertAlmostEqual(stat_b.alpha_adj, self.cfg.alpha_sig / 2.0, places=12)
        self.assertLess(stat_b.alpha_adj, stat_nb.alpha_adj)


class TestP10StaticGuards(unittest.TestCase):
    """P10 静态红线：可 grep 审计的零命中项（M5/M7/H4/H6/M8）。"""

    @staticmethod
    def _py_files() -> list[Path]:
        files = sorted(Path("kl8").rglob("*.py"))
        files.extend([Path("main.py")])
        files.extend(sorted(Path("tools").rglob("*.py")))
        return files

    def test_no_bare_except_anywhere(self) -> None:
        """全仓禁止裸 ``except Exception`` / ``except:``（P8-M7）。

        唯一例外：``main.py`` CLI 最外层兜底（必须返回退出码 2），且只允许出现 1 处。
        """
        offenders: list[str] = []
        for path in self._py_files():
            if path == Path("main.py"):
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("except Exception") or stripped == "except:":
                    offenders.append(f"{path}:{lineno}")
        self.assertEqual(offenders, [], f"仍存在裸 except：{offenders}")

        main_text = Path("main.py").read_text(encoding="utf-8")
        catch_all = [
            line.strip()
            for line in main_text.splitlines()
            if line.strip().startswith("except Exception")
        ]
        self.assertEqual(len(catch_all), 1, f"main.py 只允许 1 处 CLI 兜底：{catch_all}")

    def test_no_hardcoded_no_lookahead_true(self) -> None:
        """禁止把 ``no_lookahead`` 硬编码为 True（P8-H4：假装验证过）。"""
        offenders: list[str] = []
        for path in self._py_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                compact = line.replace(" ", "")
                if "no_lookahead" in compact and ("=True" in compact or ":True" in compact):
                    offenders.append(f"{path}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], f"防穿越结论被硬编码：{offenders}")

    def test_reporting_no_placeholders_or_hardcodes(self) -> None:
        """渲染层零占位符、零硬编码口径值（§6.6，命令 B2 要求 0 命中）。"""
        banned = ["25.00%", "6.01%", "150~250", "（5码）", "(5码)", "20/80", "C(v,k,t)", "{t}", "{period}", "TODO", "待填"]
        offenders: list[str] = []
        for path in sorted(Path("kl8/reporting").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in banned:
                if token in text:
                    offenders.append(f"{path}: {token}")
        self.assertEqual(offenders, [], f"渲染层仍存在硬编码/占位符：{offenders}")

    def test_no_hardcoded_seed_in_covering(self) -> None:
        """覆盖求解器随机种子必须来自 cfg.seed（B10），禁止字面量 seed=42（P8-M8）。"""
        source = Path("kl8/layer5_combinatorial/covering.py").read_text(encoding="utf-8")
        self.assertNotIn("seed=42", source)
        self.assertNotIn("if False", source)


class TestRoiText(unittest.TestCase):
    """P10b-L15：ROI 不可算时必须**分原因**披露，不得两者共用同一文案。"""

    def test_roi_text_three_branches(self) -> None:
        """可算 → 万分比；无注单 → 「本次未生成注单」；奖级未确认 → 「官方奖级未确认」。"""
        from kl8.constants import ROI_NA_NO_TICKET_TEXT, ROI_NA_TEXT
        from kl8.evaluation.metrics import BankrollStat, roi_text

        computed = BankrollStat(
            n_rounds=100,
            total_cost_yuan=3464,
            total_prize_yuan=1035,
            net_yuan=-2429,
            roi=-0.7012,
            max_drawdown_yuan=2429,
            max_losing_streak=27,
            best_round_yuan=100,
            roi_bp=-7012,
            roi_available=True,
        )
        self.assertEqual(roi_text(computed), "-70.12%")

        no_ticket = BankrollStat(
            n_rounds=30,
            total_cost_yuan=0,
            total_prize_yuan=0,
            net_yuan=0,
            roi=0.0,
            max_drawdown_yuan=0,
            max_losing_streak=30,
            best_round_yuan=0,
            roi_bp=0,
            roi_available=False,
        )
        self.assertEqual(roi_text(no_ticket), ROI_NA_NO_TICKET_TEXT)
        self.assertIn("未生成注单", ROI_NA_NO_TICKET_TEXT)

        unknown_prize = BankrollStat(
            n_rounds=30,
            total_cost_yuan=340,
            total_prize_yuan=0,
            net_yuan=-340,
            roi=-1.0,
            max_drawdown_yuan=340,
            max_losing_streak=30,
            best_round_yuan=0,
            roi_bp=-10000,
            roi_available=False,
        )
        self.assertEqual(roi_text(unknown_prize), ROI_NA_TEXT)
        self.assertNotEqual(ROI_NA_NO_TICKET_TEXT, ROI_NA_TEXT)


if __name__ == "__main__":
    unittest.main()
