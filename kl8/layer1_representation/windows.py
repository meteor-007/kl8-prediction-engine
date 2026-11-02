# -*- coding: utf-8 -*-
"""快乐8五层重构 —— layer1 数据表征层：多尺度时间窗口定义（R2 / R3）。

【定位】
把"四尺度窗口 {10G, 25G, 50G, ALL}"从业务概念固化成可引用的规格对象，
并提供冷启动回退口径 ``w_eff = max(floor, min(w, T))``：

- 历史不足 w 期时**不报错**，自动退化为"用当前全部可用期数"（8.md §六.2）；
- 真正不可用（T < ``windows.min_history``）的判定放在 layer3 的装配入口，
  以便把 ``InsufficientDataError`` 抛在"语义正确"的位置。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from kl8.config import WindowConfig

WINDOW_KEYS: Final[tuple[str, ...]] = ("10G", "25G", "50G", "ALL")
"""四尺度窗口的固定键与固定顺序（下游 dict 迭代顺序依赖此约定）。"""


@dataclass(frozen=True)
class WindowSpec:
    """单个尺度窗口的规格。

    @ivar name 窗口显示名（``"10G"`` / ``"25G"`` / ``"50G"`` / ``"ALL"``）
    @ivar size 窗口期数；``None`` 表示全量历史（ALL）
    """

    name: str
    """窗口显示名。"""

    size: int | None
    """窗口期数；None 表示 ALL（全量）。"""


def _check_history_length(T: int) -> int:
    """校验历史期数。

    @param T 历史总期数
    @return 归一化的 int
    @raises ValueError T 为负或非整数
    """
    if isinstance(T, bool) or not isinstance(T, (int, float)):
        raise ValueError(f"历史期数 T 必须是数值，收到 {T!r}")
    total = int(T)
    if total < 0:
        raise ValueError(f"历史期数 T 不能为负，收到 {T}")
    return total


def resolve_windows(cfg: WindowConfig, T: int) -> dict[str, WindowSpec]:
    """把窗口配置解析为四尺度规格字典（固定 4 键、固定顺序）。

    @param cfg 窗口配置（``WindowConfig``）
    @param T 当前历史总期数（仅用于合法性校验，不影响规格本身）
    @return ``{"10G": WindowSpec, "25G": WindowSpec, "50G": WindowSpec, "ALL": WindowSpec}``
    @raises ValueError T 非法或配置窗口非正
    """
    _check_history_length(T)
    sizes = {"10G": int(cfg.short), "25G": int(cfg.mid), "50G": int(cfg.long), "ALL": None}
    for name in WINDOW_KEYS[:3]:
        size = sizes[name]
        if size is None or size <= 0:
            raise ValueError(f"窗口 {name} 的期数必须为正整数，收到 {size}；请检查 kl8/config.py WindowConfig")
    return {name: WindowSpec(name=name, size=sizes[name]) for name in WINDOW_KEYS}


def effective_window(w: int | None, T: int, cold_start_floor: int = 1) -> int:
    """计算实际生效窗口 w_eff（冷启动回退的核心公式）。

    【口径】``w=None`` → ``max(floor, T)``（ALL）；否则 ``max(floor, min(w, T))``。
    因此 ``T < w`` 时不会报错，只会用满可用历史（8.md §六.2 冷启动处理）。

    @param w 目标窗口期数；None 表示 ALL
    @param T 当前历史总期数
    @param cold_start_floor 窗口下限（≥1，避免 0 长度窗口造成除零）
    @return 实际生效窗口期数（≥ cold_start_floor）
    @raises ValueError w <= 0、T 非法或 floor < 1
    """
    total = _check_history_length(T)
    floor = int(cold_start_floor)
    if floor < 1:
        raise ValueError(f"冷启动下限 cold_start_floor 必须 ≥ 1，收到 {cold_start_floor}")
    if w is None:
        return max(floor, total)
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        raise ValueError(f"窗口长度 w 必须是整数或 None(ALL)，收到 {w!r}")
    size = int(w)
    if size <= 0:
        raise ValueError(f"窗口长度 w 必须为正整数或 None(ALL)，收到 {w}")
    return max(floor, min(size, total))


def window_label(w: int | None) -> str:
    """生成窗口显示标签。

    @param w 窗口期数；None 表示 ALL
    @return ``"ALL"`` 或 ``"{w}G"``
    """
    return "ALL" if w is None else f"{int(w)}G"


def is_cold_start(w: int | None, T: int) -> bool:
    """判断当前是否处于冷启动（历史不足以填满目标窗口）。

    @param w 目标窗口期数；None 表示 ALL（只有 T==0 才算冷启动）
    @param T 当前历史总期数
    @return True 表示需要按 w_eff 回退
    @raises ValueError T 非法
    """
    total = _check_history_length(T)
    if total == 0:
        return True
    if w is None:
        return False
    return total < int(w)