# -*- coding: utf-8 -*-
"""快乐8 用户点位录入与一致性分析单元测试。

@author czw
@since 2026-11-06
"""

from __future__ import annotations

import unittest
import numpy as np

from kl8_engine.data import Dataset, Draw
from kl8_engine.models.ensemble import EnsembleResult, NumberScoreDetail
from kl8_engine.points import (
    PointsError,
    analyze_points,
    attach_boost_coverage,
    format_points_console,
    normalize_points,
    parse_points,
)

SAMPLE_POINTS = "07 08 13 16 17 18 22 28 32 33 37 38 40 42 43 45 47 48 56 67"


class TestPointsParsing(unittest.TestCase):
    """测试点位输入解析与校验。"""

    def test_parse_space_separated(self) -> None:
        points = parse_points(SAMPLE_POINTS)
        self.assertEqual(len(points), 20)
        self.assertEqual(points[0], 7)
        self.assertEqual(points[-1], 67)
        self.assertEqual(points, tuple(sorted(points)))

    def test_parse_comma_and_chinese_separator(self) -> None:
        self.assertEqual(parse_points("07,08，13、16"), (7, 8, 13, 16))

    def test_parse_empty(self) -> None:
        self.assertEqual(parse_points(""), ())
        self.assertEqual(parse_points(None), ())

    def test_parse_duplicates_rejected(self) -> None:
        with self.assertRaises(PointsError):
            parse_points("07 07 08")

    def test_parse_out_of_range_rejected(self) -> None:
        with self.assertRaises(PointsError):
            parse_points("07 81")

    def test_parse_invalid_token_rejected(self) -> None:
        with self.assertRaises(PointsError):
            parse_points("07 abc")

    def test_normalize_points(self) -> None:
        self.assertEqual(normalize_points([13, 7, 8, 7]), (7, 8, 13))
        with self.assertRaises(PointsError):
            normalize_points([0, 7])


class TestPointsAnalysis(unittest.TestCase):
    """测试点位与模型输出的重合分析。"""

    def setUp(self) -> None:
        draw = Draw(period=2026292, date="2026-11-02", numbers=tuple(range(1, 21)))
        self.dataset = Dataset(draws=(draw,))
        details = {
            n: NumberScoreDetail(
                number=n,
                final_score=1.0 - n / 100.0,
                dl_prob=0.2,
                spectral_score=0.8,
                rank=n,
                omission=2,
                is_repeat=(n in range(1, 21)),
                is_prime=False,
                zone4=0,
            )
            for n in range(1, 81)
        }
        self.ensemble = EnsembleResult(
            target_period=2026293,
            gold_pool=(9, 50, 56, 73),
            main_pool=(2, 9, 24, 27, 50, 52, 56, 61, 68, 72, 73, 78),
            shrinkage_pool_30=tuple(range(1, 31)),
            full_order=tuple(range(1, 81)),
            scores=np.full(80, 0.5),
            details=details,
            training_loss=0.5,
        )

    def test_analyze_points(self) -> None:
        points = parse_points(SAMPLE_POINTS)
        analysis = analyze_points(points, self.ensemble, self.dataset, mode="compare")

        # Top20 = 1..20，点位与之交集为 7,8,13,16,17,18
        self.assertEqual(analysis.consensus_top20, (7, 8, 13, 16, 17, 18))
        # 金胆与主选仅重合 56
        self.assertEqual(analysis.overlap_gold, (56,))
        self.assertEqual(analysis.overlap_main, (56,))
        # 30 码大底 = 1..30
        self.assertEqual(analysis.overlap_pool30, (7, 8, 13, 16, 17, 18, 22, 28))
        # 模型排名 = 号码本身，>40 为逆势码
        self.assertEqual(analysis.contrarian, (42, 43, 45, 47, 48, 56, 67))
        # 上期开奖 = 1..20
        self.assertEqual(analysis.repeat_with_latest, (7, 8, 13, 16, 17, 18))

        text = format_points_console(analysis)
        self.assertIn("用户点位分析", text)
        self.assertIn("共识码 (6 个)", text)

    def test_attach_boost_coverage(self) -> None:
        points = parse_points(SAMPLE_POINTS)
        analysis = analyze_points(points, self.ensemble, self.dataset, mode="boost")
        tickets = (
            (7, 41, 42, 43, 44),
            (1, 2, 3, 4, 5),
            (16, 56, 57, 58, 59),
        )
        updated = attach_boost_coverage(analysis, self.ensemble.shrinkage_pool_30, tickets)
        self.assertEqual(updated.boost_tickets_covered, 2)
        self.assertEqual(updated.boost_points_covered, (7, 16, 42, 43, 56))
        # 56/67 等不在 30 码大底内
        self.assertIn(56, updated.boost_points_unavailable)
        self.assertIn(67, updated.boost_points_unavailable)
        self.assertNotIn(7, updated.boost_points_unavailable)


if __name__ == "__main__":
    unittest.main()