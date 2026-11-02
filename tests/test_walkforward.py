"""Walk-Forward 滚动时序回测与因果流测试（契约 R42）。

【P10 自证项（H4 / H5 / H6）】
- H4：**人为注入一次前瞻泄露**，断言报告 ``self_check["no_lookahead"]`` 变为 False、
  越界期号被记录——证明防穿越不是硬编码的 "True"；
- H5：§3.14.1 跨期预测状态文件：幂等 upsert（重复运行逐字节不变）、只读严格更早期号、
  缺失时输出固定文案「首期无复盘基线」且**不输出任何命中数**；
- H6：ML 代码中不存在 ``if False`` 死分支，PAV 校准为真实实现（静态 + 冒烟双证）。
"""
from datetime import date, timedelta
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from kl8.config import AdaptiveConfig, AppConfig
from kl8.constants import NO_BASELINE_TEXT, RULE_IDS
from kl8.datatypes import CoveringReport, Dataset, Draw, TicketPlan
from kl8.errors import InsufficientDataError, LeakageError
from kl8.evaluation import walkforward as wf_module
from kl8.evaluation.walkforward import iter_windows, run_walkforward
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.features import build_feature_snapshot
from kl8.layer4_scoring import score_round
from kl8.pipeline import (
    PREV_PREDICTION_SCHEMA_VERSION,
    _build_review,
    prev_prediction_path,
    read_prev_prediction,
    write_prev_prediction,
)
from kl8.reporting.text_report import ReportContext, render_text_report


def _make_dummy_dataset(n: int = 75) -> Dataset:
    """生成合规的测试用假数据集。"""
    draws = []
    rng = np.random.default_rng(42)
    start_date = date(2026, 1, 1)
    for i in range(n):
        cur_date = start_date + timedelta(days=i)
        nums = frozenset(int(x) for x in rng.choice(np.arange(1, 81), size=20, replace=False))
        draws.append(Draw(
            date=cur_date.strftime("%Y-%m-%d"),
            period=2026001 + i,
            numbers=nums,
            idx=i,
        ))
    return Dataset(
        draws=tuple(draws),
        draws_by_period={d.period: d for d in draws},
    )


class TestWalkForward(unittest.TestCase):
    """Walk-Forward 测试集。"""

    def setUp(self):
        self.ds = _make_dummy_dataset(75)
        self.cfg = AppConfig()

    def test_insufficient_data_raises(self):
        """测试当数据集期数小于等于冷启动期数时抛出 InsufficientDataError。"""
        with self.assertRaises(InsufficientDataError):
            list(iter_windows(self.ds, t0=100))

        with self.assertRaises(InsufficientDataError):
            run_walkforward(self.ds, self.cfg, t0=100)

    def test_iter_windows_causality(self):
        """验证 iter_windows 产生的子数据集严格小于当前被预测期号。"""
        windows = list(iter_windows(self.ds, t0=60, n_test=5))
        self.assertEqual(len(windows), 5)

        for t, sub_ds in windows:
            # 严格截断保证
            self.assertEqual(len(sub_ds.draws), t + 1)
            target_period = self.ds.draws[t + 1].period
            for d in sub_ds.draws:
                self.assertLess(d.period, target_period)

    def test_walkforward_flow_no_leak(self):
        """测试完整 Walk-Forward 回测流程无前瞻泄露并生成自检通过的报告。"""
        report = run_walkforward(
            self.ds,
            self.cfg,
            t0=60,
            n_test=5,
            include_tickets=False,
        )
        self.assertEqual(report.n_sample, 5)
        self.assertEqual(len(report.rows), 5)
        self.assertEqual(report.t0, 60)
        self.assertTrue(report.self_check.get("sample_count_matches", False))
        self.assertTrue(report.self_check.get("no_lookahead", False))
        self.assertEqual(report.lookahead_checked, 5)
        self.assertEqual(report.lookahead_violations, ())

        # 验证逐期行对象完整
        for row in report.rows:
            self.assertGreater(row.period, row.as_of_period)
            self.assertEqual(len(row.scores), 80)
            self.assertEqual(len(row.votes), 80)
            self.assertIn(5, row.hit_at)


class TestInjectedLookaheadDetected(unittest.TestCase):
    """H4 自证：注入一次真实前瞻泄露，机检必须报 False（旧实现恒为 True）。"""

    def test_injected_leak_flips_no_lookahead_false(self) -> None:
        """第 3 期注入 LeakageError → no_lookahead=False 且记录了越界期号。"""
        ds = _make_dummy_dataset(75)
        cfg = AppConfig()
        real_guard = wf_module.assert_no_lookahead
        calls = {"n": 0}

        def leaking_guard(*, max_accessed_t: int, as_of_t: int) -> None:
            calls["n"] += 1
            if calls["n"] == 3:
                raise LeakageError(
                    f"注入穿越：max_accessed_t={max_accessed_t + 5} > as_of_t={as_of_t}"
                )
            real_guard(max_accessed_t=max_accessed_t, as_of_t=as_of_t)

        with mock.patch.object(wf_module, "assert_no_lookahead", leaking_guard):
            report = run_walkforward(ds, cfg, t0=60, n_test=5, include_tickets=False)

        self.assertEqual(calls["n"], 5, "防穿越守卫必须逐期真实调用 5 次")
        self.assertFalse(report.self_check.get("no_lookahead", True), "注入泄露后仍判无穿越 = 假装验证")
        self.assertEqual(report.lookahead_checked, 4)
        self.assertEqual(len(report.lookahead_violations), 1)
        bad_rows = [row for row in report.rows if not row.ok]
        self.assertEqual(len(bad_rows), 1)
        self.assertEqual(report.lookahead_violations[0], bad_rows[0].period)
        self.assertTrue(bad_rows[0].warnings)
        self.assertEqual(report.n_sample, 5)


def _make_ticket_plan() -> TicketPlan:
    """构造一个最小合规注单方案，用于复盘状态文件往返测试。"""
    covering = CoveringReport(
        pool=tuple(range(1, 13)),
        v=12,
        k=5,
        t=4,
        m=5,
        tickets_pos=((0, 1, 2, 3, 4),),
        tickets=((1, 2, 3, 4, 5),),
        n_tickets=1,
        coverage_ratio=1.0,
        achieved_t=4,
        verified=True,
        method="table",
    )
    return TicketPlan(
        play="pick5",
        k=5,
        budget=1,
        tickets=((1, 2, 3, 4, 5),),
        guarantee_level={"t": 4, "m": 5, "design": "C(12,5,4,5) 1注", "verified": True},
        covering=covering,
        cost_yuan=2,
        method="table",
    )


class TestPrevPredictionState(unittest.TestCase):
    """H5 自证：§3.14.1 跨期预测状态文件（幂等 upsert + 只读更早期 + 首期固定文案）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.cfg = AppConfig(
            output_dir=str(tmp / "output"),
            history_path=str(tmp / "history.txt"),
            adaptive=AdaptiveConfig(state_path=str(tmp / "state" / "adaptive_weights.json")),
        )
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_roundtrip_idempotent_upsert(self) -> None:
        """写入 → 重复写入逐字节不变 → 只读严格更早期记录。"""
        plan = _make_ticket_plan()
        path, warns = write_prev_prediction(
            self.cfg,
            period=2026292,
            target_date="2026-11-02",
            gold=(1, 2, 3, 4),
            main=tuple(range(1, 13)),
            plans=(plan,),
        )
        self.assertTrue(warns == ())
        self.assertEqual(path, prev_prediction_path(self.cfg))
        first_bytes = path.read_bytes()

        path2, _ = write_prev_prediction(
            self.cfg,
            period=2026292,
            target_date="2026-11-02",
            gold=(1, 2, 3, 4),
            main=tuple(range(1, 13)),
            plans=(plan,),
        )
        self.assertEqual(path2, path)
        self.assertEqual(first_bytes, path2.read_bytes(), "同一期同内容重复写入必须逐字节一致")

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], PREV_PREDICTION_SCHEMA_VERSION)
        record = payload["predictions"]["2026292"]
        for field_name in ("period", "target_date", "gold", "main", "plans", "created_at"):
            self.assertIn(field_name, record)
        self.assertEqual(record["plans"][0]["play"], "pick5")
        self.assertEqual(record["plans"][0]["tickets"], [[1, 2, 3, 4, 5]])

        # 只读「严格早于目标期」的最近记录：目标期 2026293 → 命中 2026292
        found, _ = read_prev_prediction(self.cfg, target_period=2026293)
        self.assertIsNotNone(found)
        self.assertEqual(found["period"], 2026292)
        # 目标期等于记录期 → 不得自读（禁止用当期预测冒充上期预测）
        same, _ = read_prev_prediction(self.cfg, target_period=2026292)
        self.assertIsNone(same)

    def test_missing_baseline_has_no_hit_counts(self) -> None:
        """无状态文件 → status=baseline、固定文案，且报告段落 0 不出现任何命中数。"""
        review, _warns = _build_review(self.cfg, self.ds, target_period=2026292)
        self.assertEqual(review["status"], "baseline")
        self.assertEqual(review["note"], NO_BASELINE_TEXT)
        self.assertEqual(review["gold_hit"], -1)
        self.assertEqual(review["main_hit"], -1)

        cfg = AppConfig()
        snapshot = build_feature_snapshot(self.state, end=len(self.ds.draws) - 1, cfg=cfg, include_ml=False)
        ctx = ReportContext(
            cfg=cfg,
            ds=self.ds,
            target_period=2026292,
            target_date="2026-11-02",
            snapshot=snapshot,
            rounds=score_round(snapshot, cfg),
            plans=(),
            previous_review=review,
            warnings=(),
        )
        text = render_text_report(ctx)
        self.assertIn(NO_BASELINE_TEXT, text)
        section0 = text.split("【段落 1")[0]
        self.assertNotIn("命中", section0, f"首期基线不得输出命中数：{section0[-400:]}")
        self.assertIn("段落 1: 宏观随机性与大数动力学基准", text)


class TestMLNoDeadBranch(unittest.TestCase):
    """H6 自证：ML 不得留下 ``if False`` 假校准分支（静态 + 冒烟）。"""

    def test_no_if_false_branch_and_pav_smoke(self) -> None:
        """源码零 ``if False``；pav 校准调用可跑通并落在 [0.05,0.95]。"""
        source = Path("kl8/layer3_features/ml.py").read_text(encoding="utf-8")
        self.assertNotIn("if False", source, "ML 中仍存在 if False 死分支（P8-H6）")
        self.assertNotIn("pav_calibrate(raw_prob, raw_prob", source)

        from kl8.config import MLConfig
        from kl8.layer3_features.ml import MLCache, train_predict

        rng = np.random.default_rng(7)
        mat = np.zeros((120, 80), dtype=np.float64)
        for i in range(120):
            mat[i, rng.choice(np.arange(80), size=20, replace=False)] = 1.0
        state = BinaryStateMatrix(mat)
        cfg = MLConfig(enabled=True, train_window=40, n_estimators=8, refit_every=25, calibrate="pav")
        result = train_predict(state, 119, cfg, cache=MLCache(4))
        if result.available:
            self.assertEqual(result.prob.shape, (80,))
            self.assertTrue(np.all(result.prob >= 0.05 - 1e-12))
            self.assertTrue(np.all(result.prob <= 0.95 + 1e-12))
        else:
            self.assertTrue(result.warnings, "ML 不可用时必须给出降级告警")


if __name__ == "__main__":
    unittest.main()