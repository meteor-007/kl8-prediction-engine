# -*- coding: utf-8 -*-
"""快乐8 历史推荐复盘、命中率统计与推荐日志单元测试。

@author czw
@since 2026-11-06
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path
import numpy as np

from kl8_engine.data import Dataset, Draw
from kl8_engine.models.ensemble import EnsembleResult, NumberScoreDetail
from kl8_engine.reporting import format_console_text_report
from kl8_engine.review import (
    PublishedRecommendation,
    append_prediction_journal,
    build_review,
    collect_published_recommendations,
    compute_recent_hit_stats,
    evaluate_recommendation,
    find_published_for_period,
    load_journal,
    parse_report_text,
)
from kl8_engine.shrinkage import ShrinkageReport

LEGACY_REPORT = """================================================================================
          快乐8 (Keno) 深度学习与图谱贝叶斯混合集成预测精要报告
                 预测目标期号: 第 2026295 期  |  模型状态: 训练收敛 (Loss: 0.5480)
================================================================================
【段落 0: 上期开奖实测复盘与时序基准】
--------------------------------------------------------------------------------
  * 上期期号: 第 2026294 期 (2026-11-04)
  * 上期开奖: 03 07 10 16 21 25 29 33 35 43 45 52 58 60 64 68 69 72 76 80

【段落 1: 核心选拔梯队（金胆与精选大底）】
--------------------------------------------------------------------------------
  * 🔥 核心超级金胆 (4 码): [ 09 25 56 73 ]
  * ⭐ 精选复式大底 (12 码): [ 09 25 27 43 50 52 56 68 72 73 75 78 ]

【段落 2: 80 码 FinalScore 综合排序 Top 20 梯队】
--------------------------------------------------------------------------------
  名次  号码   FinalScore   DL序列概率   图谱贝叶斯   当前遗漏   重号属性   数论质数
  01    [09]     0.5790       0.2346       1.0000       07期       新号      合数
  02    [73]     0.5778       0.2390       0.9919       01期       新号      质数
  03    [25]     0.5457       0.2691       0.8839       00期      ★重号     合数

【段落 3: 实战推荐保底投注单清单】
--------------------------------------------------------------------------------
  * 推荐方案: 选五(Pick 5)，预算 2 注，总成本 4 元
    第 01 注: [ 09 25 56 72 73 ]
    第 02 注: [ 16 25 43 56 73 ]

【段落 4: 30 码黄金大底级联精细缩水方案执行概况】
--------------------------------------------------------------------------------
  * 30 码大底: 02 03 07 09 16 17 19 25 27 34 43 44 45 47 50 51 52 54 56 58 60 61 62 64 66 68 72 73 75 78
  * 漏斗缩水: 初始 C(30,5)=142,506 注 -> 形态剪枝留存 122,607 注 -> 精选落地 20 注 (40元)
================================================================================
"""


def _make_ensemble(target_period: int) -> EnsembleResult:
    details = {
        n: NumberScoreDetail(
            number=n,
            final_score=0.5,
            dl_prob=0.25,
            spectral_score=0.5,
            rank=n,
            omission=1,
            is_repeat=False,
            is_prime=False,
            zone4=0,
        )
        for n in range(1, 81)
    }
    return EnsembleResult(
        target_period=target_period,
        gold_pool=(9, 50, 56, 73),
        main_pool=(2, 9, 24, 27, 50, 52, 56, 61, 68, 72, 73, 78),
        shrinkage_pool_30=tuple(range(1, 31)),
        full_order=tuple(range(1, 81)),
        scores=np.full(80, 0.5),
        details=details,
        training_loss=0.5,
    )


def _make_shrinkage() -> ShrinkageReport:
    return ShrinkageReport(
        raw_pool_count=30,
        raw_combinations_count=142506,
        after_morphology_count=110000,
        morphology_filter_rate=22.8,
        final_tickets_count=20,
        total_cost_yuan=40,
        hit5_guarantee_prob=1.8,
        hit4_or_better_prob=31.0,
        hit3_or_better_prob=97.0,
        expected_prize_per_draw=33.5,
        roi_percentage=-16.0,
        tickets=tuple((i, i + 1, i + 2, i + 3, i + 4) for i in range(1, 21)),
        pool=tuple(range(1, 31)),
    )


class TestReportParsing(unittest.TestCase):
    """测试报告解析与复盘核算。"""

    def test_parse_legacy_report(self) -> None:
        rec = parse_report_text(LEGACY_REPORT, source_path="legacy.txt")
        assert rec is not None
        self.assertEqual(rec.target_period, 2026295)
        self.assertEqual(rec.gold, (9, 25, 56, 73))
        self.assertEqual(len(rec.main), 12)
        self.assertEqual(rec.top20, (9, 73, 25))
        self.assertEqual(len(rec.pool30), 30)
        self.assertEqual(rec.tickets, ((9, 25, 56, 72, 73), (16, 25, 43, 56, 73)))

    def test_evaluate_recommendation(self) -> None:
        rec = PublishedRecommendation(
            target_period=2026295,
            gold=(9, 25, 56, 73),
            main=(9, 25, 27, 43, 50, 52, 56, 68, 72, 73, 75, 78),
            top20=tuple(range(1, 21)),
            pool30=tuple(range(1, 31)),
            tickets=((9, 25, 56, 72, 73), (1, 2, 3, 4, 5), (11, 12, 13, 14, 15)),
            source_path="test.txt",
        )
        actual = Draw(period=2026295, date="2026-11-05", numbers=tuple(range(1, 21)))
        result = evaluate_recommendation(rec, actual)

        self.assertEqual(result.gold_hits, 1)
        self.assertEqual(result.main_hits, 1)
        self.assertEqual(result.top20_hits, 20)
        self.assertEqual(result.pool30_hits, 20)
        self.assertEqual(result.best_ticket_hits, 5)
        # 第 2、3 注各命中 5 码（1000 元），第 1 注命中 0
        self.assertEqual(result.total_prize_yuan, 2000)
        self.assertEqual(result.total_cost_yuan, 6)
        self.assertEqual(result.net_profit_yuan, 1994)
        self.assertAlmostEqual(result.pool30_lift, 20 / 7.5)


class TestHitStatsAndJournal(unittest.TestCase):
    """测试命中率窗口统计与推荐日志。"""

    def setUp(self) -> None:
        self.dataset = Dataset(
            draws=(
                Draw(period=2026292, date="2026-11-02", numbers=tuple(range(1, 21))),
                Draw(period=2026293, date="2026-11-03", numbers=tuple(range(21, 41))),
                Draw(period=2026294, date="2026-11-04", numbers=tuple(range(41, 61))),
            )
        )

    def _write_report(self, reports_dir: Path, date_str: str, period: int) -> None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        ensemble = _make_ensemble(period)
        shrinkage = _make_shrinkage()
        text = format_console_text_report(
            ensemble, self.dataset, shrinkage, shrinkage.tickets[:10]
        )
        (reports_dir / f"Keno预测精要_{date_str}_第{period}期.txt").write_text(text, encoding="utf-8")

    def test_review_and_recent_stats(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            reports_dir = output_dir / "reports"
            self._write_report(reports_dir, "20261102", 2026293)
            self._write_report(reports_dir, "20261103", 2026294)

            rec = find_published_for_period(reports_dir, 2026293)
            self.assertIsNotNone(rec)
            assert rec is not None
            self.assertEqual(rec.gold, (9, 50, 56, 73))

            collected = collect_published_recommendations(output_dir)
            self.assertEqual(set(collected), {2026293, 2026294})

            stats = compute_recent_hit_stats(output_dir, self.dataset, windows=(5,))
            self.assertEqual(len(stats), 1)
            stat = stats[0]
            self.assertEqual(stat.periods, 2)
            self.assertGreaterEqual(stat.mean_pool30, 0.0)

    def test_journal_dedupe_and_priority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            first = append_prediction_journal(
                output_dir,
                target_period=2026296,
                gold=(1, 2, 3, 4),
                main=tuple(range(1, 13)),
                top20=tuple(range(1, 21)),
                pool30=tuple(range(1, 31)),
                tickets=((1, 2, 3, 4, 5),),
            )
            second = append_prediction_journal(
                output_dir,
                target_period=2026296,
                gold=(5, 6, 7, 8),
                main=tuple(range(5, 17)),
                top20=tuple(range(5, 25)),
                pool30=tuple(range(5, 35)),
                tickets=((5, 6, 7, 8, 9),),
            )
            self.assertTrue(first)
            self.assertFalse(second)

            journal = load_journal(output_dir)
            self.assertEqual(len(journal), 1)
            self.assertEqual(journal[2026296]["gold"], [1, 2, 3, 4])

    def test_find_published_prefers_earliest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports_dir = Path(td) / "reports"
            self._write_report(reports_dir, "20261103", 2026294)
            self._write_report(reports_dir, "20261102", 2026294)
            rec = find_published_for_period(reports_dir, 2026294)
            assert rec is not None
            self.assertIn("20261102", rec.source_path)

    def test_ranking_auc_from_full_order(self) -> None:
        """测试由 80 码全序计算排序 AUC。"""
        actual = Draw(period=2026295, date="2026-11-05", numbers=tuple(range(1, 21)))
        rec = PublishedRecommendation(
            target_period=2026295,
            gold=(1, 2, 3, 4),
            main=tuple(range(1, 13)),
            top20=tuple(range(1, 21)),
            pool30=tuple(range(1, 31)),
            tickets=(),
            source_path="test.txt",
            full_order=tuple(range(1, 81)),
        )
        self.assertAlmostEqual(evaluate_recommendation(rec, actual).ranking_auc, 1.0)

        worst = dataclasses.replace(rec, full_order=tuple(range(80, 0, -1)))
        self.assertAlmostEqual(evaluate_recommendation(worst, actual).ranking_auc, 0.0)

    def test_build_review_prefers_journal(self) -> None:
        """复盘应优先使用推荐日志中的首份推荐，而非报告文件。"""
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            self._write_report(output_dir / "reports", "20261103", 2026294)
            append_prediction_journal(
                output_dir,
                target_period=2026294,
                gold=(41, 42, 43, 44),
                main=tuple(range(41, 53)),
                top20=tuple(range(41, 61)),
                pool30=tuple(range(31, 61)),
                tickets=((41, 42, 43, 44, 45),),
                full_order=tuple(range(41, 81)) + tuple(range(1, 41)),
            )
            result = build_review(output_dir, self.dataset)
            assert result is not None
            self.assertEqual(result.gold_hits, 4)
            self.assertIn("prediction_journal.jsonl", result.source_path)
            self.assertAlmostEqual(result.ranking_auc, 1.0)

    def test_legacy_report_tickets_backfilled_from_shrinkage(self) -> None:
        """旧报告仅含 10 注时，应从缩水专报回填完整 20 注。"""
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            self._write_report(output_dir / "reports", "20261102", 2026293)
            shrink_dir = output_dir / "shrinkage"
            shrink_dir.mkdir()
            ticket_lines = [
                f"第 {i:02d} 注: [ {' '.join(f'{n:02d}' for n in range(i, i + 5))} ]"
                for i in range(1, 21)
            ]
            (shrink_dir / "Keno30码精细缩水_20261102_第2026293期.txt").write_text(
                "\n".join(ticket_lines), encoding="utf-8"
            )
            recs = collect_published_recommendations(output_dir)
            self.assertEqual(len(recs[2026293].tickets), 20)

    def test_build_review_falls_back_when_latest_has_no_recommendation(self) -> None:
        """最新开奖期无推荐时，应回退到最近一份可复盘推荐并标注断档。"""
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            append_prediction_journal(
                output_dir,
                target_period=2026293,
                gold=(21, 22, 23, 24),
                main=tuple(range(21, 33)),
                top20=tuple(range(21, 41)),
                pool30=tuple(range(21, 51)),
                tickets=((21, 22, 23, 24, 25),),
            )
            result = build_review(output_dir, self.dataset)
            assert result is not None
            self.assertEqual(result.target_period, 2026293)
            self.assertEqual(result.latest_draw_period, 2026294)
            self.assertEqual(result.gold_hits, 4)


if __name__ == "__main__":
    unittest.main()