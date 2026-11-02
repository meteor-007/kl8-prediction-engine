"""开奖历史数据清洗、校验与防穿越截断单元测试（契约 R4、C4）。

对应契约需求：
- R4: parse_line, load_dataset, save_dataset, 格式解析、20码去重、1..80、升序重排、降序首行校验。
- C4: 防穿越唯一截断入口 truncated(ds, target_period)。
"""

from __future__ import annotations

import os
import tempfile
import unittest

from kl8.errors import DataFormatError, InsufficientDataError
from kl8.history import (
    append_draws,
    load_dataset,
    parse_line,
    save_dataset,
    truncated,
)


class TestHistory(unittest.TestCase):
    """历史开奖数据加载与清洗校验。"""

    def test_parse_validate_and_sort(self) -> None:
        """测试标准行解析、错误格式拒绝与升序重排校验（契约 R4）。"""
        line = "date:2026-11-01,period:2026291,numbers:01-02-03-04-05-06-07-08-09-10-11-12-13-14-15-16-17-18-19-20"
        draw = parse_line(line, line_no=1)
        self.assertEqual(draw.date, "2026-11-01")
        self.assertEqual(draw.period, 2026291)
        self.assertEqual(len(draw.numbers), 20)
        self.assertEqual(min(draw.numbers), 1)
        self.assertEqual(max(draw.numbers), 20)

        # 错误行测试
        # 重复号码
        bad_line1 = "date:2026-11-01,period:2026291,numbers:01-01-03-04-05-06-07-08-09-10-11-12-13-14-15-16-17-18-19-20"
        with self.assertRaises(DataFormatError):
            parse_line(bad_line1)
        # 号码超出 1..80
        bad_line2 = "date:2026-11-01,period:2026291,numbers:00-02-03-04-05-06-07-08-09-10-11-12-13-14-15-16-17-18-19-20"
        with self.assertRaises(DataFormatError):
            parse_line(bad_line2)
        bad_line3 = "date:2026-11-01,period:2026291,numbers:01-02-03-04-05-06-07-08-09-10-11-12-13-14-15-16-17-18-19-81"
        with self.assertRaises(DataFormatError):
            parse_line(bad_line3)
        # 球数不足 20
        bad_line4 = "date:2026-11-01,period:2026291,numbers:01-02-03-04-05-06-07-08-09-10"
        with self.assertRaises(DataFormatError):
            parse_line(bad_line4)

    def test_load_real_history(self) -> None:
        """测试加载真实历史文件 kl8_history.txt。"""
        ds = load_dataset("kl8_history.txt")
        self.assertEqual(ds.T, 2112)
        # 升序排列：最旧期在 index 0，最新期在 index -1
        self.assertEqual(ds.draws[0].period, 2020001)
        self.assertEqual(ds.draws[-1].period, 2026291)
        self.assertEqual(ds.latest.period, 2026291)
        self.assertEqual(ds.target_period(), 2026292)

    def test_truncated_no_leakage(self) -> None:
        """测试防穿越唯一截断入口 truncated（契约 C4）。"""
        ds = load_dataset("kl8_history.txt")
        # 截断到 2026291，必须只包含严格早于 2026291 的期数（即最新期为 2026290）
        sub = truncated(ds, 2026291)
        self.assertEqual(sub.T, ds.T - 1)
        self.assertEqual(sub.latest.period, 2026290)
        self.assertNotIn(2026291, sub.draws_by_period)

        # 截断早于最早期的期号
        sub_early = truncated(ds, 2020000)
        self.assertEqual(sub_early.T, 0)
        self.assertTrue(len(sub_early.warnings) > 0)

    def test_save_and_append_roundtrip(self) -> None:
        """测试持久化与追加幂等。"""
        ds = load_dataset("kl8_history.txt")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = os.path.join(tmpdir, "test_hist.txt")
            save_dataset(ds, tmppath)
            ds_loaded = load_dataset(tmppath)
            self.assertEqual(ds_loaded.T, ds.T)
            self.assertEqual(ds_loaded.latest.period, ds.latest.period)

            # 幂等追加同一期
            ds_appended = append_draws(ds_loaded, [ds.latest])
            self.assertEqual(ds_appended.T, ds.T)


if __name__ == "__main__":
    unittest.main()
