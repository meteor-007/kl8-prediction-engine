"""三维报表渲染模块单测（契约 R48-d、§3.13）。"""

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import openpyxl

from kl8.config import AppConfig
from kl8.datatypes import Dataset, Draw, PoolResult, TicketPlan
from kl8.layer1_representation import BinaryStateMatrix
from kl8.layer3_features import build_feature_snapshot
from kl8.layer4_scoring import ScoreResult, ScoreRoundResult
from kl8.reporting.excel import ExcelStyle, render_excel
from kl8.reporting.text_report import ReportContext, render_text_report, save_text_report
from kl8.reporting.wechat import render_wechat, save_wechat


def _make_mock_context() -> ReportContext:
    """构建报表测试用 Mock ReportContext。"""
    cfg = AppConfig()
    draws = []
    start_date = date(2026, 1, 1)
    for i in range(100):
        d_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        draws.append(Draw(
            date=d_str,
            period=2026001 + i,
            numbers=frozenset(range(1, 21)),
            idx=i,
        ))
    ds = Dataset(
        draws=tuple(draws),
        draws_by_period={d.period: d for d in draws},
    )

    state = BinaryStateMatrix.from_dataset(ds)
    snapshot = build_feature_snapshot(state, end=99, cfg=cfg, include_ml=False)

    scores = ScoreResult(
        base=np.ones(80),
        final=np.ones(80),
        order=tuple(range(1, 81)),
        contributors={},
    )
    pool = PoolResult(
        gold=(1, 2, 3, 4, 5),
        main=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        votes=np.ones(80, dtype=np.int8) * 3,
        scores=np.ones(80),
        order=tuple(range(1, 81)),
        details={},
        warnings=(),
        relaxed=(),
        feasible=True,
        repeat_count=2,
        new_count=3,
        size=12,
    )

    class MockVerdict:
        mask = np.ones(80, dtype=bool)

    rounds = ScoreRoundResult(
        snapshot=snapshot,
        ranks={},
        verdicts=tuple(MockVerdict() for _ in range(5)),
        votes=np.ones(80, dtype=np.int8) * 3,
        scores=scores,
        pool=pool,
    )

    plan = TicketPlan(
        play="pick5",
        k=5,
        budget=4,
        tickets=(
            (1, 2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (3, 4, 5, 6, 7),
            (4, 5, 6, 7, 8),
        ),
        guarantee_level={"design": "C(12,5,3)"},
        covering=None,
        cost_yuan=8,
        method="table",
        warnings=(),
    )

    return ReportContext(
        cfg=cfg,
        ds=ds,
        target_period=2026101,
        target_date="2026-04-11",
        snapshot=snapshot,
        rounds=rounds,
        plans=(plan,),
        randomness=None,
        backtest=None,
        previous_review={"hit_count": 4},
        warnings=(),
    )


class TestReporting(unittest.TestCase):
    """报表生成测试集。"""

    def setUp(self):
        self.ctx = _make_mock_context()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_render_text_report(self):
        """测试控制台 TXT 报告渲染完整六段式内容。"""
        text = render_text_report(self.ctx)
        self.assertIn("段落 0: 上期开奖实测深度复盘", text)
        self.assertIn("段落 1: 宏观随机性与大数动力学基准", text)
        self.assertIn("段落 2: 四尺度时空动量与热度分布", text)
        self.assertIn("段落 3: 五规则正交共振榜与 FinalScore Top 20", text)
        self.assertIn("段落 4: 金胆池 / 核心复式大底与形态体检卡", text)
        self.assertIn("段落 5: 覆盖注单与组合多层级保底投注方案", text)
        self.assertIn("附录 A: 80 码底层动力学微积分全参矩阵", text)

        out_path = Path(self.temp_dir.name) / "report.txt"
        saved = save_text_report(text, out_path)
        self.assertTrue(saved.exists())
        self.assertGreater(saved.stat().st_size, 100)

    def test_render_wechat(self):
        """测试微信文案渲染合规性与字符长度约束（≤1200 字符）。"""
        text = render_wechat(self.ctx)
        self.assertLessEqual(len(text), 1200, "微信发布文案不能超过 1200 字符！")
        self.assertIn("核心超级金胆", text)
        self.assertIn("精选复式大底", text)
        self.assertIn("核心保底实战推荐票", text)

        out_path = Path(self.temp_dir.name) / "wechat.txt"
        saved = save_wechat(text, out_path)
        self.assertTrue(saved.exists())

    def test_render_excel(self):
        """测试 Excel 商业报表包含完整的 6 个工作表。"""
        out_path = Path(self.temp_dir.name) / "report.xlsx"
        saved = render_excel(self.ctx, out_path)
        self.assertTrue(saved.exists())

        wb = openpyxl.load_workbook(saved)
        self.assertEqual(len(wb.sheetnames), 6)
        expected_sheets = ["当期总览", "80码五规则全参", "覆盖注单", "回测归因", "随机性基准", "上期复盘"]
        for s_name in expected_sheets:
            self.assertIn(s_name, wb.sheetnames)


if __name__ == "__main__":
    unittest.main()
