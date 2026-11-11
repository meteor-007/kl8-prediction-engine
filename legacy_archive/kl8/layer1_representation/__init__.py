# -*- coding: utf-8 -*-
"""快乐8五层重构 —— layer1 数据表征层（R1–R4）。

【职责】把原始开奖历史固化为可高效查询的二值状态矩阵与四尺度窗口规格。
【单向依赖】本层只允许依赖 ``config`` / ``constants`` / ``datatypes`` / ``errors`` / ``history``，
**禁止** import layer2 及以上（C5）。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import (
    WINDOW_KEYS,
    WindowSpec,
    effective_window,
    is_cold_start,
    resolve_windows,
    window_label,
)

__all__ = [
    "BinaryStateMatrix",
    "WINDOW_KEYS",
    "WindowSpec",
    "effective_window",
    "is_cold_start",
    "resolve_windows",
    "window_label",
]