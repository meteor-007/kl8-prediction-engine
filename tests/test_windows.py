"""四尺度窗口解析与冷启动回退单元测试（契约 R2、R3）。

对应契约需求：
- R2: 四尺度窗口 W={10G, 25G, 50G, ALL} 定义与切片。
- R3: 冷启动窗口回退 w_eff = min(w, T)，T < w 时不报错。
"""

from __future__ import annotations

import unittest

from kl8.config import WindowConfig
from kl8.layer1_representation.windows import (
    effective_window,
    is_cold_start,
    resolve_windows,
    window_label,
)


class TestWindows(unittest.TestCase):
    """窗口规范与有效窗口计算测试。"""

    def setUp(self) -> None:
        self.cfg = WindowConfig(short=10, mid=25, long=50)

    def test_four_scale_windows(self) -> None:
        """验证四尺度窗口解析结果包含 10G, 25G, 50G, ALL 四个固定键（契约 R2）。"""
        windows = resolve_windows(self.cfg, T=100)
        self.assertEqual(list(windows.keys()), ["10G", "25G", "50G", "ALL"])
        self.assertEqual(windows["10G"].size, 10)
        self.assertEqual(windows["25G"].size, 25)
        self.assertEqual(windows["50G"].size, 50)
        self.assertIsNone(windows["ALL"].size)
        self.assertEqual(window_label(10), "10G")
        self.assertEqual(window_label(None), "ALL")

    def test_cold_start_fallback(self) -> None:
        """验证 T < w 时的冷启动回退机制（契约 R3）。"""
        # T = 5 < 10
        self.assertEqual(effective_window(10, T=5), 5)
        self.assertTrue(is_cold_start(10, T=5))
        # T = 20: 10G 不冷启动，25G 冷启动
        self.assertEqual(effective_window(10, T=20), 10)
        self.assertFalse(is_cold_start(10, T=20))
        self.assertEqual(effective_window(25, T=20), 20)
        self.assertTrue(is_cold_start(25, T=20))
        # ALL 窗口有效大小为 T
        self.assertEqual(effective_window(None, T=20), 20)
        self.assertFalse(is_cold_start(None, T=20))
        # 极小样本下保底为 cold_start_floor
        self.assertEqual(effective_window(10, T=0, cold_start_floor=1), 1)


if __name__ == "__main__":
    unittest.main()
