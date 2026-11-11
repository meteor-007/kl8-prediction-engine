# -*- coding: utf-8 -*-
"""P7 终局 QA —— P8 体检卡 H1–H6 独立验证（契约 R33/R34/R45/R47/§3.14.1/M13）。

> 与码小全的 `tests/test_*.py` 相互独立：本文件用**外部构造**的数据反证
> 「真算、真执行、真判定、真幂等、真去占位符」，不复制实现内部断言。

覆盖：
- H1 自适应规则权重真实现 + B5 等效票数（FinalScore ≡ Base + λ·Σ(w/w̄)·pass）；
- H2 空间形态硬约束真执行（同区 Top-12 必须换号）/ 全约束不可行必须抛错；
- H3 规则归因真统计（禁止 0.25/1.12 常量）；
- H4 防穿越真判定（正常路径 True + 注入穿越 False 反证）；
- H5 `prev_prediction.json` 按 period 幂等 upsert + 首期无基线文案；
- H6 交付件零占位符（`C(v,k,t)`/TODO/待填/★显著）+ `ROI: N/A` 披露。
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from openpyxl import load_workbook

from kl8.config import AdaptiveConfig, AppConfig
from kl8.datatypes import BacktestRow
from kl8.errors import InfeasibleConstraintError, LeakageError
from kl8.evaluation.leakage import assert_no_lookahead
from kl8.evaluation.metrics import rule_attribution
from kl8.evaluation.walkforward import run_walkforward
from kl8.history import load_dataset
from kl8.layer1_representation import BinaryStateMatrix
from kl8.layer3_features import build_feature_snapshot
from kl8.layer4_scoring import build_pool, score_all, score_round
from kl8.layer4_scoring.adaptive import next_weights
from kl8.pipeline import (
    read_prev_prediction,
    run_daily,
    write_prev_prediction,
)


@lru_cache(maxsize=1)
def _fixture():
    """真实历史上的公共夹具（只构建一次，避免每个用例重复 0.35s 特征）。"""
    ds = load_dataset("kl8_history.txt")
    state = BinaryStateMatrix.from_dataset(ds)
    cfg = AppConfig()
    snapshot = build_feature_snapshot(state, 1800, cfg, include_ml=False)
    rounds = score_round(snapshot, cfg)
    return ds, cfg, snapshot, rounds


def _mk_row(period: int, rule_stats: dict, vote_group_stats: dict) -> BacktestRow:
    """构造最小合法 BacktestRow（只需 rule_stats / vote_group_stats）。"""
    return BacktestRow(
        period=period,
        as_of_index=period - 1,
        as_of_period=period - 1,
        scores=np.zeros(80, dtype=np.float64),
        votes=np.zeros(80, dtype=np.int8),
        top_picks={},
        hit_at={},
        gold=(),
        main=(),
        gold_hit=0,
        main_hit=0,
        rule_stats=rule_stats,
        vote_group_stats=vote_group_stats,
    )


class TestH1AdaptiveWeights(unittest.TestCase):
    """H1/B5：自适应权重真算 + FinalScore 等效票数恒等式。"""

    def test_final_identity_with_uniform_default(self) -> None:
        """默认（无自适应态）必须满足 Final == Base + λ·V，且 Σw_r=1。"""
        _ds, cfg, _snap, rounds = _fixture()
        lam = cfg.weights.lambda_vote
        np.testing.assert_array_equal(rounds.scores.final, rounds.scores.base + lam * rounds.weighted_votes)
        # 均匀权重时等效票数严格退化为整型票数
        np.testing.assert_array_equal(rounds.weighted_votes, rounds.votes.astype(np.float64))
        self.assertAlmostEqual(float(sum(rounds.rule_weights.values())), 1.0, places=12)
        self.assertEqual(set(rounds.rule_weights), {"R1", "R2", "R3", "R4", "R5"})

    def test_final_identity_with_nonuniform_rule_weights(self) -> None:
        """非均匀规则权重下恒等式仍逐位成立，且等效票数确有变化（B5 除 w̄）。"""
        _ds, cfg, snap, rounds = _fixture()
        skew = {"R1": 0.4, "R2": 0.1, "R3": 0.2, "R4": 0.2, "R5": 0.1}
        skewed = score_all(
            snap, rounds.ranks, cfg.weights, verdicts=rounds.verdicts, rule_weights=skew
        )
        lam = cfg.weights.lambda_vote
        np.testing.assert_array_equal(skewed.final, skewed.base + lam * skewed.weighted_votes)
        self.assertAlmostEqual(float(sum(skewed.rule_weights.values())), 1.0, places=12)
        # 与整数票数不同：证明不是「读到了但没用」
        self.assertGreater(int(np.sum(~np.isclose(skewed.weighted_votes, rounds.votes.astype(float)))), 0)

    def test_next_weights_reacts_to_rule_stats(self) -> None:
        """两期差异极大的 rule_stats → 权重不同且 Σ=1、均 ≥ floor（查「读到未用」）。"""
        cfg = AdaptiveConfig()
        rows_a = [
            SimpleNamespace(
                rule_stats={"R1": (20, 20), "R2": (5, 20), "R3": (5, 20), "R4": (5, 20), "R5": (0, 20)}
            )
        ] * 3
        rows_b = [
            SimpleNamespace(
                rule_stats={"R1": (0, 20), "R2": (5, 20), "R3": (5, 20), "R4": (5, 20), "R5": (20, 20)}
            )
        ] * 3
        state_a, warns_a = next_weights(rows_a, None, cfg, period=2026292)
        state_b, warns_b = next_weights(rows_b, None, cfg, period=2026292)
        self.assertEqual(warns_a, ())
        self.assertEqual(warns_b, ())
        self.assertNotEqual(dict(state_a.weights), dict(state_b.weights))
        for state in (state_a, state_b):
            self.assertAlmostEqual(float(sum(state.weights.values())), 1.0, places=12)
            for value in state.weights.values():
                self.assertGreaterEqual(value, cfg.weight_floor - 1e-12)
        self.assertGreater(state_a.weights["R1"], state_b.weights["R1"])
        self.assertGreater(state_b.weights["R5"], state_a.weights["R5"])
        self.assertEqual(state_a.n_updates, 1)

    def test_next_weights_small_sample_falls_back_uniform(self) -> None:
        """可用样本 < min_periods → 回退均匀先验 0.2 并写中文告警。"""
        cfg = AdaptiveConfig()
        rows = [
            SimpleNamespace(rule_stats={"R1": (20, 20), "R2": (0, 20), "R3": (0, 20), "R4": (0, 20), "R5": (0, 20)})
        ] * 2  # 2 期 < 默认 min_periods=3
        state, warns = next_weights(rows, None, cfg, period=2026292)
        np.testing.assert_allclose(list(state.weights.values()), [0.2] * 5, atol=1e-12)
        self.assertTrue(any("回退均匀先验" in w for w in warns))


class TestH2HardConstraints(unittest.TestCase):
    """H2/R34：空间形态硬约束真执行与不可行硬失败。"""

    def test_zone4_top12_is_rejected_and_pool_rewritten(self) -> None:
        """人为把 12 个高分区全设为第四区 → 最终池单区 ≤5 码且无放宽（真换号）。"""
        _ds, cfg, snap, _rounds = _fixture()
        final = np.full(80, 0.1, dtype=np.float64)
        final[60:80] = 1.0  # 61..80 高分（同为第四区）
        final[0:4] = 0.9  # 1..4 次高分（作为上期重号候选）
        last_draw = frozenset(list(range(1, 5)) + list(range(41, 57)))
        pool = build_pool(
            snap, final, np.zeros(80, dtype=np.int8), last_draw, pool_size=12, cfg=cfg
        )
        zone4_count = sum(1 for num in pool.main if 61 <= num <= 80)
        self.assertLessEqual(zone4_count, 5, f"单四区 {zone4_count} 码 > 45% 上限")
        self.assertEqual(len(pool.main), 12)
        self.assertEqual(pool.relaxed, ())
        self.assertTrue(pool.feasible)
        # 原始 Top-12 全是 61..80，实际池必须与之不同
        self.assertNotEqual(set(pool.main), set(range(61, 81)))

    def test_empty_last_draw_relaxes_repeat_ratio_with_trace(self) -> None:
        """上期为空集 → 按 B7 有序放宽 repeat_ratio 即应满 12 码并留痕（不再硬失败）。

        契约 v1.3.0 B7 / M16 修复后语义：`greedy_pool` 不再"无约束补齐"，凑不满时如实返回
        不足码数，由 `build_pool` 按 same_tail → parity → zone4 → repeat_ratio 逐项放宽、每步
        重跑；只有全部 4 项放宽后仍不足才抛 `InfeasibleConstraintError`（该路径由
        `tests/test_filter.py::test_raises_only_after_all_relaxations` 以 mock 锁定）。

        【口径修订记录】本用例原断言"必须抛错"，其前提是 M16 修复前的旧缺陷行为
        （补齐吞掉放宽分支 → `relaxed` 恒空）。M16 修复后改为正向断言。
        """
        _ds, cfg, snap, _rounds = _fixture()
        final = np.full(80, 0.1, dtype=np.float64)
        final[60:80] = 1.0
        pool = build_pool(
            snap, final, np.zeros(80, dtype=np.int8), frozenset(), pool_size=12, cfg=cfg
        )
        self.assertEqual(len(pool.main), 12)
        self.assertIn("repeat_ratio", pool.relaxed)
        self.assertTrue(
            any("repeat_ratio" in w for w in pool.warnings),
            f"缺少 repeat_ratio 放宽留痕: {pool.warnings}",
        )
        self.assertEqual(pool.relaxed, tuple(
            x for x in cfg.pool.relax_order if x in pool.relaxed
        ), "relaxed 必须是 relax_order 的有序前缀")

    def test_infeasible_pool_size_raises(self) -> None:
        """全约束不可行（目标规模 0 / 81）必须抛 InfeasibleConstraintError，禁止带病出单。"""
        _ds, cfg, snap, _rounds = _fixture()
        final = np.linspace(1.0, 0.0, 80)
        for bad_size in (0, 81):
            with self.subTest(size=bad_size):
                with self.assertRaises(InfeasibleConstraintError):
                    build_pool(
                        snap,
                        final,
                        np.zeros(80, dtype=np.int8),
                        frozenset(range(1, 21)),
                        pool_size=bad_size,
                        cfg=cfg,
                    )


class TestH3Attribution(unittest.TestCase):
    """H3/R45：规则归因必须由真实 rule_stats/vote_group_stats 算出。"""

    def test_per_rule_hand_calc(self) -> None:
        """R1=(10,40)×2 → 命中率 0.25；R5=(20,20)×2 → 1.0/lift4；禁止 0.28/1.12 常量。"""
        stats = {"R1": (10, 40), "R2": (5, 20), "R3": (5, 20), "R4": (5, 20), "R5": (20, 20)}
        groups = {"high": (10, 20), "low": (2, 20)}
        report = rule_attribution([_mk_row(10, stats, groups), _mk_row(11, stats, groups)])
        r1 = report.per_rule["R1"]
        r5 = report.per_rule["R5"]
        self.assertAlmostEqual(r1.hit_rate, 0.25, places=12)
        self.assertAlmostEqual(r1.lift, 1.0, places=12)
        self.assertAlmostEqual(r1.pool_size_avg, 40.0, places=12)
        self.assertAlmostEqual(r5.hit_rate, 1.0, places=12)
        self.assertAlmostEqual(r5.lift, 4.0, places=12)
        self.assertGreater(r5.z, 0.0)
        # 旧假常量自证：若实现回退常量，这里必红
        self.assertNotAlmostEqual(r5.hit_rate, 0.28, places=6)
        self.assertNotAlmostEqual(r5.lift, 1.12, places=6)

    def test_vote_group_excess_and_z(self) -> None:
        """高票组 10/20 vs 低票组 2/20 → 高组 0.5、lift 2.0、两比例 z>0。"""
        groups = {"high": (10, 20), "low": (2, 20)}
        report = rule_attribution([_mk_row(10, {}, groups), _mk_row(11, {}, groups)])
        self.assertAlmostEqual(report.high_votes["hit_rate"], 0.5, places=12)
        self.assertAlmostEqual(report.high_votes["lift"], 2.0, places=12)
        self.assertAlmostEqual(report.low_votes["hit_rate"], 0.1, places=12)
        self.assertAlmostEqual(report.low_votes["lift"], 0.4, places=12)
        self.assertGreater(report.high_votes["z"], 0.0)


class TestH4Lookahead(unittest.TestCase):
    """H4/R47：防穿越是真判定（正常 True，注入穿越必须 False）。"""

    def test_normal_path_is_true_and_counted(self) -> None:
        """正常回测：每期都真实校验，checked == n_sample 且 self_check 全 True。"""
        ds, cfg, _snap, _rounds = _fixture()
        report = run_walkforward(ds, cfg, n_test=3, t0=300, include_tickets=False)
        self.assertEqual(report.lookahead_checked, report.n_sample)
        self.assertEqual(report.lookahead_checked, 3)
        self.assertEqual(report.lookahead_violations, ())
        self.assertEqual(
            report.self_check,
            {
                "sample_count_matches": True,
                "lift_recalculated": True,
                "bankroll_recalculated": True,
                "no_lookahead": True,
            },
        )

    def test_injected_lookahead_makes_self_check_false(self) -> None:
        """注入穿越（mock assert_no_lookahead 抛 LeakageError）→ no_lookahead 必须 False。"""
        ds, cfg, _snap, _rounds = _fixture()
        with mock.patch(
            "kl8.evaluation.walkforward.assert_no_lookahead",
            side_effect=LeakageError("注入穿越：模拟未来数据泄漏"),
        ):
            report = run_walkforward(ds, cfg, n_test=2, t0=300, include_tickets=False)
        self.assertEqual(report.lookahead_checked, 0)
        self.assertEqual(len(report.lookahead_violations), 2)
        self.assertFalse(report.self_check["no_lookahead"])
        self.assertTrue(all(not row.ok for row in report.rows))
        self.assertTrue(
            any("防穿越校验失败" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_guard_raises_only_when_access_exceeds_as_of(self) -> None:
        """索引守卫边界：等于 as_of_t 通过，超过 1 即 LeakageError。"""
        assert_no_lookahead(max_accessed_t=10, as_of_t=10)
        with self.assertRaises(LeakageError):
            assert_no_lookahead(max_accessed_t=11, as_of_t=10)


class TestH5PrevPredictionIdempotent(unittest.TestCase):
    """H5/§3.14.1：复盘状态按 period 幂等 upsert + 首期无基线文案。"""

    def test_same_period_repeat_is_byte_identical(self) -> None:
        """同 period 同内容重复写：逐字节相同，created_at 不被刷新。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig(output_dir=tmp, adaptive=AdaptiveConfig(state_path=str(Path(tmp) / "state.json")))
            kwargs = dict(
                period=2026292,
                target_date="2026-11-02",
                gold=(1, 2, 3),
                main=tuple(range(1, 13)),
                plans=(),
            )
            path, warns1 = write_prev_prediction(cfg, created_at="2026-11-02T10:00:00", **kwargs)
            first = path.read_bytes()
            path2, warns2 = write_prev_prediction(cfg, created_at="2027-01-01T00:00:00", **kwargs)
            self.assertEqual(path2, path)
            self.assertEqual(first, path2.read_bytes())
            self.assertEqual(warns1, ())
            self.assertEqual(warns2, ())
            payload = json.loads(first.decode("utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["predictions"]["2026292"]["created_at"], "2026-11-02T10:00:00")

    def test_multi_period_upsert_preserves_history(self) -> None:
        """不同 period 追加不覆盖旧期；读取严格早于目标期的最近一条。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig(output_dir=tmp, adaptive=AdaptiveConfig(state_path=str(Path(tmp) / "state.json")))
            for period in (2026290, 2026291):
                write_prev_prediction(
                    cfg,
                    period=period,
                    target_date="2026-10-31",
                    gold=(period % 80 + 1,),
                    main=tuple(range(1, 11)),
                    plans=(),
                    created_at="2026-10-31T10:00:00",
                )
            payload = json.loads(Path(cfg.output_dir, "state", "prev_prediction.json").read_text("utf-8"))
            self.assertEqual(set(payload["predictions"]), {"2026290", "2026291"})
            record, warns = read_prev_prediction(cfg, target_period=2026292)
            self.assertEqual(record["period"], 2026291)
            self.assertEqual(warns, ())
            record_earlier, _ = read_prev_prediction(cfg, target_period=2026290)
            self.assertIsNone(record_earlier)

    def test_daily_rerun_same_period_keeps_prev_bytes(self) -> None:
        """完整日跑两次（同目标期）→ prev_prediction.json 逐字节不变（幂等）。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig(
                history_path="kl8_history.txt",
                output_dir=str(Path(tmp) / "output"),
                adaptive=AdaptiveConfig(state_path=str(Path(tmp) / "state" / "adaptive.json")),
            )
            first = run_daily(
                cfg, no_sync=True, target_period=2026292, target_date="2026-11-02", backtest_n=0
            )
            prev_path = first.report_paths["prev_prediction"]
            bytes_first = prev_path.read_bytes()
            second = run_daily(
                cfg, no_sync=True, target_period=2026292, target_date="2026-11-02", backtest_n=0
            )
            self.assertEqual(second.report_paths["prev_prediction"], prev_path)
            self.assertEqual(bytes_first, prev_path.read_bytes())

    def test_missing_baseline_text_has_no_hit_numbers(self) -> None:
        """首期无复盘基线：固定文案逐字输出，且该行不含任何命中数字。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = AppConfig(
                history_path="kl8_history.txt",
                output_dir=str(Path(tmp) / "output"),
                adaptive=AdaptiveConfig(state_path=str(Path(tmp) / "state" / "adaptive.json")),
            )
            result = run_daily(
                cfg, no_sync=True, target_period=2026292, target_date="2026-11-02", backtest_n=0
            )
            text = result.report_paths["text"].read_text("utf-8")
            self.assertIn("首期无复盘基线（无可用的上期预测记录）", text)
            review_lines = [line for line in text.splitlines() if "复盘提示" in line]
            self.assertEqual(len(review_lines), 1)
            self.assertFalse(
                any(ch.isdigit() for ch in review_lines[0]),
                f"首期基线行不应出现命中数字: {review_lines[0]!r}",
            )


class TestH6DeliverablesNoPlaceholders(unittest.TestCase):
    """H6/M13：三类交付件零占位符 + 奖级未确认必须 ROI N/A。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cfg = AppConfig(
            history_path="kl8_history.txt",
            output_dir=str(tmp / "output"),
            adaptive=AdaptiveConfig(state_path=str(tmp / "state" / "adaptive.json")),
        )
        cls.result = run_daily(
            cfg, no_sync=True, target_period=2026292, target_date="2026-11-02", backtest_n=5
        )
        cls.text = cls.result.report_paths["text"].read_text("utf-8")
        cls.wechat = cls.result.report_paths["wechat"].read_text("utf-8")
        workbook = load_workbook(cls.result.report_paths["excel"], read_only=True)
        cls.excel_text = "\n".join(
            str(cell) for sheet in workbook.worksheets for row in sheet.iter_rows() for cell in row if cell
        )
        workbook.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_no_forbidden_placeholders(self) -> None:
        """`C(v,k,t)` / TODO / 待填 / ★显著 在 TXT+微信+Excel 中均为 0。"""
        for token in ("C(v,k,t)", "TODO", "待填"):
            with self.subTest(token=token):
                self.assertEqual(self.text.count(token), 0)
                self.assertEqual(self.wechat.count(token), 0)
                self.assertEqual(self.excel_text.count(token), 0)
        # n=5 < 500，任何玩法都不得标「★显著」
        self.assertEqual(self.text.count("★显著"), 0)
        self.assertEqual(self.wechat.count("★显著"), 0)

    def test_guarantee_design_is_real(self) -> None:
        """理论保底设计必须是真实 C(v,k,t,m) 而非占位符。"""
        designs = re.findall(r"理论保底设计:\s*C\((\d+),(\d+),(\d+),(\d+)\)\s*\d*注?", self.text)
        self.assertGreaterEqual(len(designs), 1)
        self.assertNotIn("C(v,k,t)", self.text)

    def test_roi_na_disclosed(self) -> None:
        """未确认奖级玩法必须输出 `ROI: N/A（官方奖级未确认）` ≥1 处。"""
        total = (
            self.text.count("ROI: N/A")
            + self.wechat.count("ROI: N/A")
            + self.excel_text.count("ROI: N/A")
        )
        self.assertGreaterEqual(total, 1)
        self.assertIn("ROI: N/A（官方奖级未确认）", self.text)

    def test_daily_report_self_check_all_true(self) -> None:
        """日跑交付件自检四项全 True（no_lookahead 非恒真，须真实计数）。"""
        report = self.result.backtest
        self.assertIsNotNone(report)
        self.assertEqual(report.lookahead_checked, report.n_sample)
        self.assertEqual(report.lookahead_violations, ())
        self.assertEqual(set(report.self_check.values()), {True})
        self.assertIn("报告自洽机检", self.text)


if __name__ == "__main__":
    unittest.main()