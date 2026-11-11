# -*- coding: utf-8 -*-
"""P7 QA —— 发布护栏与契约增量回归测试（Lead 追加 a–e 五条）。

【覆盖口径】
a) A7 覆盖参数：``m_target`` 五组验算 + ``m_margin`` 越界（0/4）抛 ConfigError；
b) B4 ML 默认关闭：``load_config().ml.enabled is False``，且 ``main --ml`` 能显式开启；
c) 进度回调：``run_backtest(progress=cb)`` 回调次数 == 期数、``run_daily(progress=cb)`` ≥8 步；
d) 性能护栏：``build_tickets`` 走查表路径（method == "table"）且耗时 < 0.1 s；
e) ``运行微积分预测.bat``：存在、UTF-8 BOM（EF BB BF）、含 ``main.py`` 与 ``--no-sync``；
f) 契约 v1.1.0 新增配置项：``backtest.lift_signal`` 越界（0.99/1.25）抛 ConfigError。

【铁律】本文件只做断言，不改业务代码；所有期望值均来自 contract.md / 8.md 手算口径。
"""
from __future__ import annotations

import argparse
import dataclasses
import tempfile
import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from kl8.config import (
    AdaptiveConfig,
    AppConfig,
    CoveringConfig,
    default_config,
    load_config,
)
from kl8.datatypes import PoolResult
from kl8.errors import ConfigError
from kl8.layer5_combinatorial import build_tickets, m_target
from kl8.pipeline import run_backtest, run_daily
from main import build_arg_parser, main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_synthetic_history(path: Path, n: int, seed: int = 20261102) -> None:
    """生成 n 期合规历史文件（降序原格式，期号 2024001..）。"""
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    draws = []
    for i in range(n):
        nums = sorted(int(x) for x in rng.choice(np.arange(1, 81), size=20, replace=False))
        draws.append(
            (
                (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                2024001 + i,
                nums,
            )
        )
    lines = [
        f"date:{d},period:{p},numbers:" + "-".join(f"{x:02d}" for x in nums)
        for d, p, nums in reversed(draws)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_pool_result(main_pool=(2, 5, 8, 11, 14, 17, 22, 27, 33, 41, 52, 63)) -> PoolResult:
    """构造最小 PoolResult（build_tickets 只消费 main 字段）。"""
    votes = np.full(80, 2, dtype=np.int8)
    scores = np.linspace(1.0, 0.1, 80)
    order = tuple(range(1, 81))
    return PoolResult(
        gold=tuple(main_pool[:5]),
        main=tuple(main_pool),
        votes=votes,
        scores=scores,
        order=order,
        details={},
        warnings=(),
        relaxed=(),
        feasible=True,
        repeat_count=3,
        new_count=3,
        size=len(main_pool),
    )


class TestA7MTargetAndMargins(unittest.TestCase):
    """a) A7 覆盖参数 m 与 m_margin 校验。"""

    def test_m_target_contract_parameter_groups(self) -> None:
        """§5.0 五组参数验算必须逐条复现（t<k 用 t+m_margin，t≥k 用 t）。"""
        cases = [
            (12, 5, 4, 5),  # C(12,5,4,5)
            (12, 7, 5, 6),  # C(12,7,5,6)
            (15, 5, 4, 5),  # C(15,5,4,5)
            (15, 7, 5, 6),  # C(15,7,5,6)
            (10, 5, 5, 5),  # C(10,5,5,5)
        ]
        for v, k, t, expected in cases:
            with self.subTest(v=v, k=k, t=t):
                self.assertEqual(m_target(v, k, t), expected)

    def test_m_target_bounds_t_le_m_le_v(self) -> None:
        """任意合法 (v,k,t) 下必须满足 t ≤ m ≤ v（契约 A7 强制关系）。"""
        for v in (6, 8, 10, 12, 15, 20):
            for k in (5, 7):
                if k > v:
                    continue
                for t in range(1, min(k, v) + 1):
                    with self.subTest(v=v, k=k, t=t):
                        m = m_target(v, k, t)
                        self.assertGreaterEqual(m, t)
                        self.assertLessEqual(m, v)

    def test_m_margin_zero_rejected(self) -> None:
        """m_margin=0 越界（契约范围 [1,3]）必须抛 ConfigError。"""
        with self.assertRaises(ConfigError) as ctx:
            load_config({"covering": {"m_margin": 0}})
        self.assertIn("m_margin", str(ctx.exception))

    def test_m_margin_four_rejected(self) -> None:
        """m_margin=4 越界（契约范围 [1,3]）必须抛 ConfigError。"""
        with self.assertRaises(ConfigError):
            load_config({"covering": {"m_margin": 4}})

    def test_m_margin_boundaries_accepted(self) -> None:
        """m_margin 的闭区间端点 1 与 3 必须合法。"""
        self.assertEqual(load_config({"covering": {"m_margin": 1}}).covering.m_margin, 1)
        self.assertEqual(load_config({"covering": {"m_margin": 3}}).covering.m_margin, 3)

    def test_default_m_margin_is_one(self) -> None:
        """默认 m_margin 必须为 1（v1.1.0/A7；写成 2 会造成覆盖表全部未命中）。"""
        self.assertEqual(default_config().covering.m_margin, 1)
        self.assertEqual(CoveringConfig().m_margin, 1)


class TestLiftSignalConfig(unittest.TestCase):
    """f) backtest.lift_signal 契约范围 [1.0, 1.2]。"""

    def test_lift_signal_below_range_rejected(self) -> None:
        """lift_signal=0.99 越界必须抛 ConfigError。"""
        with self.assertRaises(ConfigError):
            load_config({"backtest": {"lift_signal": 0.99}})

    def test_lift_signal_above_range_rejected(self) -> None:
        """lift_signal=1.25 越界必须抛 ConfigError。"""
        with self.assertRaises(ConfigError):
            load_config({"backtest": {"lift_signal": 1.25}})

    def test_lift_signal_boundaries_accepted(self) -> None:
        """1.0 与 1.2 端点合法，默认值 1.05。"""
        self.assertEqual(load_config({"backtest": {"lift_signal": 1.0}}).backtest.lift_signal, 1.0)
        self.assertEqual(load_config({"backtest": {"lift_signal": 1.2}}).backtest.lift_signal, 1.2)
        self.assertEqual(default_config().backtest.lift_signal, 1.05)


class TestMLDefaultOff(unittest.TestCase):
    """b) B4：ML 默认关闭，仅 --ml 显式开启。"""

    def test_default_config_ml_disabled(self) -> None:
        """load_config()/default_config() 的 ml.enabled 必须为 False。"""
        self.assertIs(load_config().ml.enabled, False)
        self.assertIs(default_config().ml.enabled, False)

    def test_ml_can_be_enabled_by_patch(self) -> None:
        """显式配置 ml.enabled=True 时必须生效（能力具备、默认关闭）。"""
        cfg = load_config({"ml": {"enabled": True}})
        self.assertIs(cfg.ml.enabled, True)

    def test_main_ml_flag_enables_runtime_config(self) -> None:
        """main(["--ml"]) 传给 run_daily 的 cfg.ml.enabled 必须为 True，且默认调用为 False。"""
        captured: list[AppConfig] = []

        def fake_run_daily(cfg, **kwargs):  # noqa: ANN001
            captured.append(cfg)
            return SimpleNamespace(
                target_period=2026292,
                target_date="2026-11-02",
                pool=SimpleNamespace(gold=(1, 2, 3, 4, 5), main=tuple(range(1, 13))),
                plans=(),
                report_paths={
                    "text": REPO_ROOT / "output" / "reports" / "x.txt",
                    "wechat": REPO_ROOT / "output" / "wechat" / "x.txt",
                    "excel": REPO_ROOT / "output" / "excel" / "x.xlsx",
                    "state": REPO_ROOT / "output" / "state" / "x.json",
                },
                warnings=(),
                elapsed_s=0.0,
            )

        with mock.patch("main.run_daily", side_effect=fake_run_daily):
            code_off = main(["--no-sync", "--backtest", "0"])
            code_on = main(["--no-sync", "--backtest", "0", "--ml"])

        self.assertEqual(code_off, 0)
        self.assertEqual(code_on, 0)
        self.assertEqual(len(captured), 2)
        self.assertIs(captured[0].ml.enabled, False)
        self.assertIs(captured[1].ml.enabled, True)


class TestProgressCallbacks(unittest.TestCase):
    """c) 进度回调不丢不重。"""

    def test_backtest_progress_count_equals_periods(self) -> None:
        """run_backtest(..., progress=cb) 的回调次数必须恰好等于回测期数，且步序 1..n。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hist = tmp / "hist.txt"
            _write_synthetic_history(hist, 75)
            cfg = AppConfig(
                history_path=str(hist),
                output_dir=str(tmp / "out"),
                adaptive=AdaptiveConfig(state_path=str(tmp / "state.json")),
            )
            steps: list[tuple[int, int]] = []
            report = run_backtest(
                cfg,
                n=5,
                t0=60,
                include_tickets=False,
                progress=lambda done, total: steps.append((done, total)),
            )
        self.assertEqual(report.n_sample, 5)
        self.assertEqual(len(steps), 5)
        self.assertEqual([s[0] for s in steps], [1, 2, 3, 4, 5])
        self.assertTrue(all(s[1] == 5 for s in steps))

    def test_daily_progress_has_eight_steps(self) -> None:
        """run_daily(..., progress=cb) 至少收到 8 条步骤消息，且步序覆盖 [1/8]..[8/8]。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            hist = tmp / "hist.txt"
            _write_synthetic_history(hist, 400)
            cfg = AppConfig(
                history_path=str(hist),
                output_dir=str(tmp / "out"),
                adaptive=AdaptiveConfig(state_path=str(tmp / "state.json")),
            )
            messages: list[str] = []
            res = run_daily(
                cfg,
                no_sync=True,
                target_period=2024401,
                target_date="2025-02-04",
                backtest_n=3,
                progress=messages.append,
            )
            # 交付件必须在临时目录内（不污染仓库 output/）
            self.assertTrue(res.report_paths["text"].is_relative_to(tmp))
            self.assertTrue(res.report_paths["excel"].is_relative_to(tmp))
        self.assertGreaterEqual(len(messages), 8)
        for idx in range(1, 9):
            self.assertTrue(
                any(m.startswith(f"[{idx}/8]") for m in messages),
                f"缺少第 {idx}/8 步进度消息：{messages}",
            )


class TestPerformanceGuard(unittest.TestCase):
    """d) 稳定性能护栏：覆盖表命中路径必须快且方法可辨识。"""

    def test_build_tickets_uses_table_and_is_fast(self) -> None:
        """v=12 选五/选七必须命中覆盖表，且 build_tickets 总耗时 < 0.1 s（2000× 回归护栏）。"""
        cfg = load_config()
        pool_result = _make_pool_result()
        scores = np.linspace(1.0, 0.1, 80)

        t0 = time.perf_counter()
        plans = build_tickets(pool_result, scores, cfg)
        elapsed = time.perf_counter() - t0

        self.assertGreaterEqual(len(plans), 2)
        for plan in plans:
            with self.subTest(play=plan.play):
                self.assertEqual(plan.covering.method, "table", "覆盖表未命中将回落 ILP，触发性能缺陷")
                self.assertLessEqual(plan.covering.n_tickets, plan.budget)
                self.assertEqual(plan.cost_yuan, cfg.cost_per_ticket * len(plan.tickets))
        self.assertLess(elapsed, 0.1, f"build_tickets 耗时 {elapsed:.3f}s 超过护栏 0.1s")

    def test_covering_report_elapsed_ms_small(self) -> None:
        """表命中路径的 CoveringReport.elapsed_ms 必须远小于 3000ms 预算。"""
        cfg = load_config()
        plans = build_tickets(_make_pool_result(), np.linspace(1.0, 0.1, 80), cfg)
        for plan in plans:
            self.assertLess(plan.covering.elapsed_ms, 500)


class TestBatEntry(unittest.TestCase):
    """e) 一键入口 .bat 静态校验。"""

    def test_bat_exists_utf8_bom_and_commands(self) -> None:
        """文件存在、前 3 字节为 EF BB BF（UTF-8 BOM）、含 main.py 与 --no-sync。"""
        bat = REPO_ROOT / "运行微积分预测.bat"
        self.assertTrue(bat.exists(), "运行微积分预测.bat 必须存在")
        raw = bat.read_bytes()
        self.assertEqual(raw[:3], b"\xef\xbb\xbf", "必须带 UTF-8 BOM 以保证 cmd 中文不乱码")
        text = raw.decode("utf-8-sig")
        self.assertIn("main.py", text)
        self.assertIn("--no-sync", text)
        self.assertIn("--backtest", text)
        self.assertIn("--full-print", text)


class TestCliDefaults(unittest.TestCase):
    """CLI 兼容契约补充：--backtest 的模式化默认值与 --no-tickets。"""

    def test_backtest_default_is_none_and_mode_dependent(self) -> None:
        """CLI 解析层 --backtest 缺省必须为 None，交由 main 按模式分派（daily=30/backtest=100）。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--no-sync"])
        self.assertIsNone(args.backtest)
        self.assertFalse(args.ml)
        self.assertFalse(args.no_tickets)
        self.assertEqual(args.mode, "daily")

    def test_cli_accepts_ml_and_no_tickets(self) -> None:
        """--ml / --no-tickets 必须被 CLI 接受并置 True。"""
        parser = build_arg_parser()
        args = parser.parse_args(["--ml", "--no-tickets", "--mode", "backtest", "--backtest", "100"])
        self.assertTrue(args.ml)
        self.assertTrue(args.no_tickets)
        self.assertEqual(args.mode, "backtest")
        self.assertEqual(args.backtest, 100)


if __name__ == "__main__":
    unittest.main()