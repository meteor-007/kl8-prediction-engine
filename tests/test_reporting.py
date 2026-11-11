# -*- coding: utf-8 -*-
"""快乐8 全渠道报表生成单元测试。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
import numpy as np
import openpyxl

from kl8_engine.data import Dataset, Draw
from kl8_engine.models.ensemble import EnsembleResult, NumberScoreDetail
from kl8_engine.reporting import (
    format_console_text_report,
    format_wechat_main_copy,
    generate_excel_workbook,
)
from kl8_engine.shrinkage import ShrinkageReport


class TestReporting(unittest.TestCase):
    """测试全渠道三维交付物渲染与落盘。"""

    def setUp(self) -> None:
        draw = Draw(period=2026292, date="2026-11-02", numbers=tuple(range(1, 21)))
        self.dataset = Dataset(draws=(draw,))
        details = {
            n: NumberScoreDetail(
                number=n,
                final_score=0.8,
                dl_prob=0.8,
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
            scores=np.full(80, 0.8),
            details=details,
            training_loss=0.5,
        )
        self.shrinkage = ShrinkageReport(
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

    def test_wechat_and_console_text(self) -> None:
        """测试微信文案与控制台精要格式。"""
        wechat = format_wechat_main_copy(self.ensemble, self.dataset, self.shrinkage.tickets[:10])
        self.assertIn("🎯【快乐8当期五维精选·第2026293期】", wechat)
        self.assertIn("🔥【核心超级金胆 (4码)】", wechat)

        console = format_console_text_report(self.ensemble, self.dataset, self.shrinkage, self.shrinkage.tickets[:10])
        self.assertIn("预测目标期号: 第 2026293 期", console)
        self.assertIn("80 码 FinalScore 综合排序 Top 20 梯队", console)

    def test_generate_excel_workbook(self) -> None:
        """测试商业 Excel 文件生成与多 Sheet 验证。"""
        with tempfile.TemporaryDirectory() as td:
            xl_path = os.path.join(td, "test_decision.xlsx")
            generate_excel_workbook(self.ensemble, self.dataset, self.shrinkage, xl_path)
            self.assertTrue(os.path.exists(xl_path))
            self.assertGreater(os.path.getsize(xl_path), 5000)

    def test_comparison_console_and_excel(self) -> None:
        """测试双融合并排对比的 TXT 与 Excel 输出。"""
        comparison = dataclasses.replace(
            self.ensemble,
            gold_pool=(1, 2, 3, 4),
            main_pool=tuple(range(1, 13)),
            shrinkage_pool_30=tuple(range(31, 61)),
        )
        console = format_console_text_report(
            self.ensemble,
            self.dataset,
            self.shrinkage,
            self.shrinkage.tickets[:10],
            comparison_ensemble=comparison,
            comparison_shrinkage=self.shrinkage,
            fusion_mode="borda",
            comparison_fusion_mode="weighted",
        )
        self.assertIn("双融合并排对比", console)
        self.assertIn("重合度", console)

        with tempfile.TemporaryDirectory() as td:
            xl_path = os.path.join(td, "test_compare.xlsx")
            generate_excel_workbook(
                self.ensemble,
                self.dataset,
                self.shrinkage,
                xl_path,
                comparison_ensemble=comparison,
                comparison_shrinkage=self.shrinkage,
                official_fusion_mode="borda",
                comparison_fusion_mode="weighted",
            )
            wb = openpyxl.load_workbook(xl_path)
            self.assertIn("融合并排对比", wb.sheetnames)


if __name__ == "__main__":
    unittest.main()
