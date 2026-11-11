# -*- coding: utf-8 -*-
"""P7 QA —— 边界/退化、None 与类型防守、状态文件与空报告测试。

【覆盖口径】
- 边界退化：T=0 空数据集、T<窗口、全 0/全 1 序列、μ_gap=0、空池/单号码池、
  非法号码 0/81、重复号码、期号重复、乱序 Dataset、非法状态矩阵；
- 类型防守：config=None、Dataset 空、BacktestReport 无 rows、状态文件不存在/损坏/旧版本；
- 覆盖空输入：coverage_ratio / verify_guarantee / positions_to_numbers / postprocess；
- 多条件筛选：硬约束全满足 / 冲突放宽留痕 / 不可行输入兜底不崩；
- 自适应权重：softmax 数值稳定、min_periods 回退、save/load 原子往返。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kl8.config import (
    AppConfig,
    CoveringConfig,
    PoolConfig,
    config_hash,
    default_config,
    load_config,
    to_dict,
    validate_config,
)
from kl8.datatypes import BacktestReport, BacktestRow, Dataset, Draw
from kl8.errors import (
    ConfigError,
    DataFormatError,
    InsufficientDataError,
)
from kl8.evaluation.metrics import (
    bankroll,
    high_tier_rate,
    lift_at_k,
    self_check,
)
from kl8.evaluation.walkforward import iter_windows, run_walkforward
from kl8.history import load_dataset, parse_line, truncated
from kl8.layer1_representation import BinaryStateMatrix
from kl8.layer3_features import build_feature_snapshot
from kl8.layer3_features.morphology import assert_valid_numbers, is_balanced
from kl8.layer4_scoring.adaptive import (
    AdaptiveState,
    load_state,
    next_weights,
    save_state,
    softmax_weights,
)
from kl8.layer4_scoring.filter import apply_repeat_quota, enforce_constraints, greedy_pool
from kl8.layer5_combinatorial.covering import coverage_ratio, verify_guarantee
from kl8.layer5_combinatorial.tickets import positions_to_numbers, postprocess
from kl8.constants import N_NUM


def make_draw(period: int, numbers, date: str = "2026-11-01") -> Draw:
    """构造合法 Draw（默认 20 码）。"""
    return Draw(date=date, period=period, numbers=frozenset(numbers))


def make_dataset(n: int = 5, seed: int = 1) -> Dataset:
    """构造 n 期升序数据集（每期 20 个唯一号码）。"""
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 81), size=20, replace=False))
        draws.append(make_draw(2026001 + i, nums, date=f"2026-10-{1 + i:02d}"))
    return Dataset(draws=tuple(draws), draws_by_period={d.period: d for d in draws})


class TestEmptyDatasetAndState(unittest.TestCase):
    """T=0 / 空序列退化路径。"""

    def test_empty_dataset_properties(self) -> None:
        """空 Dataset：T=0、periods 空、latest/target_period 抛 InsufficientDataError。"""
        ds = Dataset(draws=(), draws_by_period={})
        self.assertEqual(ds.T, 0)
        self.assertEqual(ds.periods, ())
        self.assertEqual(ds.hit_matrix().shape, (0, N_NUM))
        with self.assertRaises(InsufficientDataError):
            _ = ds.latest
        with self.assertRaises(InsufficientDataError):
            ds.target_period()

    def test_empty_state_matrix_queries(self) -> None:
        """T=0 状态矩阵：窗口查询空安全，gap/latest_set 抛 IndexError。"""
        state = BinaryStateMatrix(np.zeros((0, N_NUM), dtype=np.float64))
        self.assertTrue(state.is_empty())
        self.assertEqual(state.prefix().shape, (1, N_NUM))
        self.assertEqual(state.window_matrix(0, 10).shape, (0, N_NUM))
        np.testing.assert_array_equal(state.window_sum(0, 10), np.zeros(N_NUM))
        np.testing.assert_array_equal(state.window_mean(0, 10), np.zeros(N_NUM))
        with self.assertRaises(IndexError):
            state.gap_vector(0)
        with self.assertRaises(IndexError):
            state.latest_set(0)
        with self.assertRaises(IndexError):
            state.row_sum(0)

    def test_state_matrix_illegal_shapes_and_values(self) -> None:
        """状态矩阵形状/取值/行和非法必须抛 DataFormatError；矩阵只读。"""
        with self.assertRaises(DataFormatError):
            BinaryStateMatrix(np.zeros((5, 79)))
        with self.assertRaises(DataFormatError):
            BinaryStateMatrix(np.zeros(80))
        with self.assertRaises(DataFormatError):
            BinaryStateMatrix(np.full((2, N_NUM), 0.5))
        bad_row = np.zeros((1, N_NUM))
        bad_row[0, :19] = 1.0
        with self.assertRaises(DataFormatError):
            BinaryStateMatrix(bad_row)
        good = np.zeros((1, N_NUM))
        good[0, :20] = 1.0
        state = BinaryStateMatrix(good)
        self.assertFalse(state.hit.flags.writeable)

    def test_dataset_unsorted_and_map_mismatch(self) -> None:
        """Dataset draws 非升序 / 映射不一致必须抛 DataFormatError。"""
        d1 = make_draw(2026002, sorted(range(1, 21)), "2026-11-02")
        d2 = make_draw(2026001, sorted(range(21, 41)), "2026-11-01")
        with self.assertRaises(DataFormatError):
            Dataset(draws=(d1, d2), draws_by_period={d1.period: d1, d2.period: d2})
        with self.assertRaises(DataFormatError):
            Dataset(draws=(d2, d1), draws_by_period={d2.period: d2})

    def test_walkforward_insufficient_data(self) -> None:
        """空数据集 / T≤T0 时 iter_windows 与 run_walkforward 必须抛 InsufficientDataError。"""
        ds = Dataset(draws=(), draws_by_period={})
        with self.assertRaises(InsufficientDataError):
            list(iter_windows(ds, t0=300, n_test=10))
        with self.assertRaises(InsufficientDataError):
            run_walkforward(ds, default_config(), n_test=10, include_tickets=False)

    def test_feature_snapshot_cold_start_and_insufficient(self) -> None:
        """T<min_history 抛 InsufficientDataError；T=40 允许但 cold_start=True。"""
        ds40 = make_dataset(40)
        state40 = BinaryStateMatrix.from_dataset(ds40)
        cfg = default_config()
        snap = build_feature_snapshot(state40, end=39, cfg=cfg, include_ml=False)
        self.assertTrue(snap.cold_start)
        self.assertEqual(snap.as_of_index, 39)
        self.assertLessEqual(snap.n_history, 40)
        ds39 = make_dataset(39)
        with self.assertRaises(InsufficientDataError):
            build_feature_snapshot(BinaryStateMatrix.from_dataset(ds39), end=38, cfg=cfg)
        with self.assertRaises(IndexError):
            build_feature_snapshot(state40, end=40, cfg=cfg)

    def test_all_zero_and_all_one_columns(self) -> None:
        """全 0 列（从未开出）与全 1 列（每期开出）特征必须有限、无 NaN。"""
        rows = [[i for i in range(1, 21)]] * 40  # 号码 1 全 1，号码 21..80 全 0
        mat = np.zeros((40, N_NUM), dtype=np.float64)
        for idx, nums in enumerate(rows):
            mat[idx, np.asarray(nums) - 1] = 1.0
        snap = build_feature_snapshot(
            BinaryStateMatrix(mat), end=39, cfg=default_config(), include_ml=False
        )
        for name in ("freq_10", "freq_25", "freq_50", "freq_all", "pressure", "gap_ratio"):
            arr = getattr(snap, name)
            self.assertTrue(np.all(np.isfinite(arr)), f"{name} 含 NaN/Inf")
        np.testing.assert_array_equal(snap.gap[:20], np.zeros(20))
        self.assertTrue(np.all(snap.gap[20:] == 40.0))


class TestCoveringEmptyInputs(unittest.TestCase):
    """空池 / 单号码池 / 空注单。"""

    def test_covering_empty_and_single_pool(self) -> None:
        """空注单：coverage_ratio=0、verify_guarantee 未通过；positions 映射单号码池。"""
        self.assertEqual(coverage_ratio((), 6, 5, 4, 5), 0.0)
        result = verify_guarantee((), 6, 5, 4, 5)
        self.assertFalse(result.passed)
        self.assertEqual(result.n_subsets, 0)
        self.assertEqual(result.failed_example, ())
        self.assertEqual(positions_to_numbers((), (42,)), ())
        self.assertEqual(positions_to_numbers(((0,),), (42,)), ((42,),))
        self.assertEqual(positions_to_numbers(((0, 1),), (7, 42)), ((7, 42),))

    def test_postprocess_empty_and_filter_fallback(self) -> None:
        """空注单输入返回空元组 + 告警；过滤全灭时回退原始去重注单。"""
        cfg = CoveringConfig(budget=5)
        tickets, warns = postprocess([], (1, 2, 3), cfg=cfg, scores_by_number={})
        self.assertEqual(tickets, ())
        self.assertTrue(warns)
        # 单注全奇（1,3,5,7,9）会被形态过滤剔除，但回退保留基础注单
        cfg2 = CoveringConfig(budget=5, forbid_all_odd_even=True)
        tickets2, warns2 = postprocess(
            [(0, 1, 2, 3, 4)], (1, 3, 5, 7, 9), cfg=cfg2, scores_by_number={}
        )
        self.assertEqual(tickets2, ((1, 3, 5, 7, 9),))
        self.assertTrue(any("回退" in w for w in warns2))


class TestIllegalNumbersAndPeriods(unittest.TestCase):
    """非法号码 0/81、重复号码、期号重复。"""

    def test_draw_number_validation(self) -> None:
        """0/81、重复、数量不足 20、非整数必须抛 DataFormatError。"""
        for bad in (
            [0] + list(range(2, 21)),
            [81] + list(range(1, 20)),
            [1] * 20,
            list(range(1, 20)),
            list(range(1, 22)),
        ):
            with self.subTest(numbers=bad[:3]):
                with self.assertRaises(DataFormatError):
                    make_draw(2026001, bad)

    def test_parse_line_and_validator_edges(self) -> None:
        """parse_line 对缺字段/坏号码抛 DataFormatError；morphology 校验直接暴露 0/81。"""
        for line in (
            "",
            "period:2026001,numbers:" + "-".join(f"{i:02d}" for i in range(1, 21)),
            "date:2026-11-01,numbers:" + "-".join(f"{i:02d}" for i in range(1, 21)),
            "date:2026-11-01,period:2026001",
            "date:2026-11-01,period:2026001,numbers:00-02-03-04-05-06-07-08-09-10-11-12-13-14-15-16-17-18-19-20",
        ):
            with self.subTest(line=line[:24]):
                with self.assertRaises(DataFormatError):
                    parse_line(line, 1)
        with self.assertRaises(DataFormatError):
            assert_valid_numbers([0] + list(range(2, 21)))
        with self.assertRaises(DataFormatError):
            assert_valid_numbers(list(range(1, 20)) + [81])

    def test_load_dataset_missing_and_invalid(self) -> None:
        """文件不存在/空/坏行/期号重复/乱序必须按契约报错或告警。"""
        cfg = default_config()
        with self.assertRaises(DataFormatError):
            load_dataset("definitely_not_exists_kl8.txt")
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            empty = tmp / "empty.txt"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(InsufficientDataError):
                load_dataset(empty)
            dup = tmp / "dup.txt"
            nums = "-".join(f"{i:02d}" for i in range(1, 21))
            dup.write_text(
                f"date:2026-11-02,period:2026002,numbers:{nums}\n"
                f"date:2026-11-01,period:2026002,numbers:{nums}\n",
                encoding="utf-8",
            )
            with self.assertRaises(DataFormatError):
                load_dataset(dup)
            mixed = tmp / "mixed.txt"
            mixed.write_text(
                "garbage line\n"
                f"date:2026-11-02,period:2026003,numbers:{nums}\n"
                f"date:2026-11-01,period:2026001,numbers:{nums}\n",
                encoding="utf-8",
            )
            ds = load_dataset(mixed)
            self.assertEqual(ds.T, 2)
            self.assertTrue(any("跳过" in w for w in ds.warnings))
            self.assertTrue(any("断档" in w for w in ds.warnings))
        self.assertIsNotNone(cfg)

    def test_truncated_boundaries(self) -> None:
        """目标期 ≤ 最早期 → 空+warning；> 最新期 → 全量；非法期号 → 全量+warning。"""
        ds = make_dataset(5)
        earliest = ds.periods[0]
        empty = truncated(ds, earliest)
        self.assertEqual(empty.T, 0)
        self.assertTrue(any("空历史" in w for w in empty.warnings))
        one = truncated(ds, earliest + 1)
        self.assertEqual(one.T, 1)
        full = truncated(ds, ds.periods[-1] + 1)
        self.assertEqual(full.T, 5)
        bad = truncated(ds, "not-a-period")
        self.assertEqual(bad.T, 5)
        self.assertTrue(any("无法解析" in w for w in bad.warnings))


class TestConfigNoneAndTypeDefense(unittest.TestCase):
    """config=None / 类型防守 / hash 稳定。"""

    def test_load_config_none_and_roundtrip(self) -> None:
        """load_config(None) 返回默认配置；to_dict→load_config 往返一致。"""
        cfg = load_config(None)
        self.assertEqual(cfg, default_config())
        self.assertEqual(load_config(to_dict(cfg)), cfg)
        self.assertIsNone(validate_config(cfg))

    def test_validate_config_type_errors(self) -> None:
        """validate_config(None/dict/字符串) 必须抛 ConfigError。"""
        for bad in (None, {"windows": {}}, "config", 42):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ConfigError):
                    validate_config(bad)

    def test_config_hash_stability(self) -> None:
        """config_hash 同配置稳定、字段变更敏感，长度 16。"""
        a = default_config()
        b = default_config()
        self.assertEqual(config_hash(a), config_hash(b))
        self.assertEqual(len(config_hash(a)), 16)
        c = load_config({"covering": {"m_margin": 2}})
        self.assertNotEqual(config_hash(a), config_hash(c))


class TestBacktestReportEmpty(unittest.TestCase):
    """BacktestReport 无 rows 与空指标。"""

    def test_empty_report_valid_and_self_check(self) -> None:
        """n_sample=0/rows=() 合法，自检全绿；n_sample 与 rows 不一致必须报错。"""
        report = BacktestReport(n_sample=0, t0=300, start_period=0, end_period=0, rows=())
        self.assertEqual(report.n_sample, 0)
        checks = self_check(report)
        self.assertTrue(all(checks.values()), checks)
        with self.assertRaises(DataFormatError):
            BacktestReport(n_sample=3, t0=300, start_period=1, end_period=3, rows=())

    def test_metrics_empty_rows(self) -> None:
        """空 rows 下 Lift 全 0、高奖级空、资金全 0，禁止除零。"""
        cfg = default_config().backtest
        lift = lift_at_k([], (3, 5), cfg)
        self.assertEqual(set(lift), {3, 5})
        for stat in lift.values():
            self.assertEqual(stat.mean_hit, 0.0)
            self.assertEqual(stat.lift, 0.0)
            self.assertFalse(stat.is_signal)
        self.assertEqual(high_tier_rate([], "pick5", {}), {})
        b = bankroll([], {})
        self.assertEqual((b.n_rounds, b.total_cost_yuan, b.roi), (0, 0, 0.0))
        self.assertEqual(b.max_drawdown_yuan, 0)

    def test_bankroll_handcalc_and_losing_streak(self) -> None:
        """资金曲线手算：成本/奖金/ROI/回撤/最长连挂（整数元）。"""
        def row(prize: int, cost: int = 20) -> BacktestRow:
            return BacktestRow(
                period=1,
                as_of_index=0,
                as_of_period=1,
                scores=np.zeros(80),
                votes=np.zeros(80, dtype=np.int8),
                top_picks={},
                hit_at={},
                gold=(),
                main=(),
                gold_hit=0,
                main_hit=0,
                cost_yuan=cost,
                prize_yuan=prize,
            )

        rows = [row(0), row(0), row(50), row(0), row(0), row(0)]
        stat = bankroll(rows, None)
        self.assertEqual(stat.total_cost_yuan, 120)
        self.assertEqual(stat.total_prize_yuan, 50)
        self.assertEqual(stat.net_yuan, -70)
        self.assertAlmostEqual(stat.roi, -70.0 / 120.0, delta=1e-12)
        self.assertEqual(stat.max_drawdown_yuan, 70)
        self.assertEqual(stat.max_losing_streak, 3)
        self.assertEqual(stat.best_round_yuan, 50)


class TestStateFileDefense(unittest.TestCase):
    """状态文件不存在/损坏/旧版本/原子往返。"""

    def test_missing_state_file(self) -> None:
        """状态文件不存在 → (None, ())，且不得创建文件。"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "missing.json"
            state, warns = load_state(path, 1)
            self.assertIsNone(state)
            self.assertEqual(warns, ())
            self.assertFalse(path.exists())

    def test_corrupt_state_backs_up(self) -> None:
        """损坏 JSON → 备份为 .json.bak 并告警。"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            path.write_text("{not-json", encoding="utf-8")
            state, warns = load_state(path, 1)
            self.assertIsNone(state)
            self.assertTrue(warns)
            self.assertIn("损坏", warns[0])
            self.assertFalse(path.exists())
            self.assertTrue(path.with_suffix(".json.bak").exists())

    def test_old_version_state_backs_up(self) -> None:
        """schema_version 不符 → 备份并重置。"""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            payload = {
                "schema_version": 0,
                "period": 2026001,
                "weights": {f"R{i}": 0.2 for i in range(1, 6)},
                "hit_rates": {f"R{i}": 0.25 for i in range(1, 6)},
                "n_updates": 3,
                "updated_at": "2026-11-01T00:00:00+00:00",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            state, warns = load_state(path, 1)
            self.assertIsNone(state)
            self.assertTrue(any("版本" in w for w in warns))
            self.assertTrue(path.with_suffix(".json.bak").exists())

    def test_save_load_roundtrip_atomic(self) -> None:
        """save_state→load_state 往返一致，父目录自动创建，无临时残留。"""
        state = AdaptiveState(
            schema_version=1,
            period=2026292,
            weights={f"R{i}": 0.2 for i in range(1, 6)},
            hit_rates={f"R{i}": 0.25 for i in range(1, 6)},
            n_updates=7,
            updated_at="2026-11-02T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "state.json"
            save_state(state, path)
            loaded, warns = load_state(path, 1)
            self.assertEqual(warns, ())
            self.assertEqual(loaded, state)
            self.assertEqual(
                sorted(p.name for p in path.parent.iterdir()), ["state.json"]
            )

    def test_softmax_stability_and_fallback(self) -> None:
        """softmax 极端输入不溢出、和恒 1；样本不足回退均匀先验并告警。"""
        prior = np.full(5, 0.2)
        # 真实命中率量级 [0,1] 下必须满足 floor 与归一
        w = softmax_weights(np.array([1.0, 0.0, 0.5, 0.25, 0.75]), prior=prior)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertAlmostEqual(float(np.sum(w)), 1.0, delta=1e-12)
        self.assertTrue(np.all(w >= 0.05 - 1e-12))
        # 病态量级仅要求数值稳定（不溢出/不出 NaN），不承诺 floor
        extreme = softmax_weights(np.array([1e6, -1e6, 0.0, 0.5, 0.2]), prior=prior)
        self.assertTrue(np.all(np.isfinite(extreme)))
        self.assertAlmostEqual(float(np.sum(extreme)), 1.0, delta=1e-12)
        self.assertEqual(softmax_weights(np.array([]), prior=np.array([])).size, 0)
        state, warns = next_weights([], None, AppConfig().adaptive, period=2026001)
        self.assertEqual(state.n_updates, 1)
        np.testing.assert_allclose(list(state.weights.values()), np.full(5, 0.2))
        self.assertTrue(warns)
        state2, _ = next_weights([], state, AppConfig().adaptive, period=2026002)
        self.assertEqual(state2.n_updates, 2)


class TestFilterBoundaries(unittest.TestCase):
    """多条件筛选：全满足 / 冲突放宽 / 不可行兜底。"""

    def test_repeat_quota_within_band_keeps_order(self) -> None:
        """天然重号数在 [25%,35%] 区间时零置换（完全保留高分顺序）。"""
        cfg = PoolConfig()
        order = list(range(1, 81))
        # 上期只开 4 个高频码，Top-12 天然重号 4 个，落在 [25%,35%] 区间
        last_draw = frozenset({1, 2, 3, 4}) | frozenset(range(61, 77))
        pool, warns = apply_repeat_quota(order, last_draw, cfg, size=12)
        self.assertEqual(pool, order[:12])
        self.assertEqual(warns, ())

    def test_repeat_quota_overflow_and_deficit(self) -> None:
        """重号超限/不足时最小差额置换，且池大小与重号配比落在契约区间。"""
        cfg = PoolConfig()
        order = list(range(1, 81))
        # 上期开 1..20，前 12 名全是重号 → 超过 35%（4.2 → 4 码）
        overflow, _ = apply_repeat_quota(order, frozenset(range(1, 21)), cfg, size=12)
        self.assertEqual(len(overflow), 12)
        rep = sum(1 for n in overflow if n <= 20)
        self.assertLessEqual(rep, int(0.35 * 12))
        # 上期开 61..80，前 12 名无重号 → 低于 25%（3 码）
        deficit, _ = apply_repeat_quota(order, frozenset(range(61, 81)), cfg, size=12)
        self.assertEqual(len(deficit), 12)
        rep2 = sum(1 for n in deficit if n >= 61)
        self.assertGreaterEqual(rep2, 3)

    def test_is_balanced_and_relax_chain(self) -> None:
        """硬约束全满足 → (True, ())；全同区冲突 → False+原因；放宽链不崩且留痕。"""
        cfg = PoolConfig()
        balanced_nums = [1, 2, 3, 24, 25, 26, 47, 48, 49, 60, 71, 72]
        ok, reasons = is_balanced(balanced_nums, cfg)
        self.assertTrue(ok, reasons)
        self.assertEqual(reasons, ())
        skewed = list(range(1, 13))  # 全部落在第一四分区
        ok2, reasons2 = is_balanced(skewed, cfg)
        self.assertFalse(ok2)
        self.assertTrue(reasons2)
        filtered, warns = enforce_constraints(skewed, None, cfg, size=12)
        self.assertEqual(len(filtered), 12)
        self.assertTrue(warns)
        for name in ("same_tail", "parity", "zone4", "repeat_ratio"):
            self.assertIn(name, cfg.relax_order)

    def test_greedy_pool_impossible_fallback(self) -> None:
        """不可行形态约束下 greedy_pool 不得崩溃，必须补齐 size 并留告警。"""
        cfg = PoolConfig()
        scores = np.linspace(1.0, 0.0, 80)
        pool, warns = greedy_pool(scores, None, cfg, size=12)
        self.assertEqual(len(pool), 12)
        self.assertTrue(warns)


if __name__ == "__main__":
    unittest.main()