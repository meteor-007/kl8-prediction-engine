# -*- coding: utf-8 -*-
"""快乐8五层重构 —— layer1 数据表征层：二值化状态矩阵（R1 / R2）。

【定位】
把开奖历史固化成 ``X ∈ {0,1}^{T×80}`` 的二值状态矩阵，并提供窗口聚合的
**O(1) 前缀和查询**与当前遗漏向量。layer2 / layer3 的所有特征都从这里取数，
因此本类是整个系统的"数据地基"，必须做到：

1. **形状与行和强校验**：shape 必须是 ``(T,80)``，每行和恒为 ``K=20``，取值只允许 0/1；
   任何不合规立刻 :class:`DataFormatError`（宁可中断，不得带着脏矩阵继续算）；
2. **不可变**：内部矩阵设为只读，防止下游把"原始数据"改掉导致跨层污染；
3. **空安全**：``T=0``（空矩阵）构造合法，窗口类查询在非法 end 时抛 ``IndexError``，
   ``T=0`` 下调用窗口查询会自然触发越界而非除零。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import numpy as np

from kl8.constants import K_DRAW, N_NUM
from kl8.datatypes import Dataset
from kl8.errors import DataFormatError
from kl8.layer1_representation.windows import effective_window


class BinaryStateMatrix:
    """二值化开奖状态矩阵 ``X ∈ {0,1}^{T×80}``（行和恒为 20）。

    【索引约定（C9）】矩阵为 0-based；列下标 ``i`` ↔ 号码 ``i+1``（1..80）。
    """

    def __init__(self, hit: np.ndarray) -> None:
        """由 (T,80) 0-1 矩阵构造状态矩阵，并做形状/取值/行和强校验。

        @param hit 开奖矩阵，(T,80)，取值仅允许 0/1；T=0 时允许形状 (0,80)
        @raises DataFormatError 维度/列数/取值/行和不符合快乐8规则
        """
        matrix = np.asarray(hit, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != N_NUM:
            raise DataFormatError(
                f"状态矩阵形状 {matrix.shape} 非法，要求 (T,{N_NUM})；"
                "请先用 kl8.io 的 Dataset.hit_matrix() 生成矩阵"
            )
        if matrix.size:
            invalid = not bool(np.all((matrix == 0.0) | (matrix == 1.0)))
            if invalid:
                raise DataFormatError(
                    "状态矩阵含 0/1 以外的取值；开奖矩阵必须是严格二值化结果"
                )
        row_sums = matrix.sum(axis=1) if matrix.size else np.zeros(0, dtype=np.float64)
        bad_rows = np.flatnonzero(row_sums != K_DRAW)
        if bad_rows.size:
            first = int(bad_rows[0])
            raise DataFormatError(
                f"状态矩阵第 {first} 行行和为 {int(row_sums[first])}，要求每行恒为 {K_DRAW}；"
                "请检查该期号码是否恰好 20 个、是否有重复码漏检"
            )

        matrix.setflags(write=False)  # 不可变：下游只能读，不能改原始数据
        self.hit: np.ndarray = matrix
        """(T,80) float64 只读开奖矩阵。"""

        self.T: int = int(matrix.shape[0])
        """历史总期数 T（可能为 0）。"""

        prefix = np.zeros((self.T + 1, N_NUM), dtype=np.int32)
        if self.T:
            prefix[1:] = np.cumsum(matrix, axis=0).astype(np.int32)
        prefix.setflags(write=False)
        self._prefix: np.ndarray = prefix

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "BinaryStateMatrix":
        """由开奖数据集构造状态矩阵。

        @param ds 升序开奖数据集
        @return :class:`BinaryStateMatrix` 实例
        @raises DataFormatError 矩阵形状/行和非法
        """
        return cls(ds.hit_matrix())

    def prefix(self) -> np.ndarray:
        """返回 (T+1,80) int32 前缀和矩阵（只读视图，O(1) 区间求和用）。

        @return 前缀和矩阵 ``prefix[t, i] = Σ_{τ<t} X_{τ,i}``
        """
        return self._prefix

    def _check_end(self, end: int) -> int:
        """校验窗口右端点（0-based 含端点）。

        @param end 右端点索引
        @return 归一化索引
        @raises IndexError end 越界或非整数
        """
        if isinstance(end, bool) or not isinstance(end, (int, np.integer)):
            raise IndexError(f"窗口右端点 end 必须是整数，收到 {end!r}")
        value = int(end)
        if value < 0 or value >= self.T:
            raise IndexError(
                f"窗口右端点 end={value} 越界（T={self.T}，合法范围 0..{self.T - 1}）；"
                "请确认已按 history.truncated() 截断到目标期之前"
            )
        return value

    def _window_bounds(self, end: int, w: int | None) -> tuple[int, int]:
        """计算窗口的 [start, end] 闭区间边界。

        @param end 右端点（含）
        @param w 窗口期数；None 表示 ALL
        @return (start, end) 闭区间
        @raises IndexError end 越界
        @raises ValueError w <= 0
        """
        value = self._check_end(end)
        w_eff = effective_window(w, value + 1)
        return value + 1 - w_eff, value

    def row_sum(self, t: int) -> int:
        """返回第 t 期（0-based）的行和（应恒为 K=20，用于自检）。

        @param t 期索引（0-based）
        @return 行和
        @raises IndexError t 越界
        """
        value = self._check_end(t)
        return int(self._prefix[value + 1].sum() - self._prefix[value].sum())

    def window_matrix(self, end: int, w: int | None) -> np.ndarray:
        """返回截止 end（含）最近 w 期的开奖子矩阵。

        【空安全（契约 §3.7）】``T==0`` 时返回形状 (0,80) 的空矩阵，**不抛异常**；
        其余情况 ``end`` 越界 → ``IndexError``，``w<=0`` → ``ValueError``。

        @param end 右端点（0-based 含端点）
        @param w 窗口期数；None 表示 ALL
        @return (w_eff,80) float64 只读视图，``w_eff = max(1, min(w, end+1))``
        @raises IndexError end 越界（仅 T>0）
        @raises ValueError w <= 0
        """
        if self.T == 0:
            return np.zeros((0, N_NUM), dtype=np.float64)
        start, stop = self._window_bounds(end, w)
        return self.hit[start : stop + 1]

    def window_sum(self, end: int, w: int | None) -> np.ndarray:
        """返回窗口内各号码出现次数向量（前缀和 O(1) 查询）。

        【空安全（契约 §3.7）】``T==0`` 时返回 (80,) 全零向量，**不抛异常**。

        @param end 右端点（0-based 含端点）
        @param w 窗口期数；None 表示 ALL
        @return (80,) int32，``c_i(w) = Σ_{τ} X_{τ,i}``
        @raises IndexError end 越界（仅 T>0）
        @raises ValueError w <= 0
        """
        if self.T == 0:
            return np.zeros(N_NUM, dtype=np.int32)
        start, stop = self._window_bounds(end, w)
        counts = self._prefix[stop + 1] - self._prefix[start]
        return np.ascontiguousarray(counts, dtype=np.int32)

    def window_mean(self, end: int, w: int | None) -> np.ndarray:
        """返回窗口内各号码出现频率 ``Freq_i(w) = c_i(w) / w_eff``。

        【空安全（契约 §3.7）】``T==0`` 时返回 (80,) 全零向量，**不抛异常**。

        @param end 右端点（0-based 含端点）
        @param w 窗口期数；None 表示 ALL
        @return (80,) float64，取值 [0,1]
        @raises IndexError end 越界（仅 T>0）
        @raises ValueError w <= 0
        """
        if self.T == 0:
            return np.zeros(N_NUM, dtype=np.float64)
        start, stop = self._window_bounds(end, w)
        w_eff = stop + 1 - start
        return self.window_sum(end, w).astype(np.float64) / float(w_eff)

    def latest_set(self, end: int) -> frozenset[int]:
        """返回第 end 期开出的号码集合（1-based）。

        @param end 期索引（0-based）
        @return 20 个号码的 frozenset
        @raises IndexError end 越界
        """
        value = self._check_end(end)
        return frozenset(int(index) + 1 for index in np.flatnonzero(self.hit[value]))

    def gap_vector(self, end: int) -> np.ndarray:
        """返回第 end 期结束时各号码的当前遗漏期数 ``O_{t,i}``。

        【算法】向量化取"最近一次开出下标"：当期开出则遗漏为 0，否则
        ``遗漏 = end - 最近开出下标``（从未开出时最近下标为 -1，故遗漏 = end+1）。

        @param end 期索引（0-based 含端点）
        @return (80,) int32 当前遗漏
        @raises IndexError end 越界
        """
        value = self._check_end(end)
        window = self.hit[: value + 1]
        is_hit = window > 0.5
        index = np.arange(value + 1, dtype=np.int64)[:, None]
        last_hit = np.maximum.accumulate(np.where(is_hit, index, -1), axis=0)[-1]
        gap = np.where(is_hit[-1], 0, value - last_hit)
        return gap.astype(np.int32)

    def is_empty(self) -> bool:
        """判断是否为空矩阵（T=0）。

        @return True 表示没有任何历史期
        """
        return self.T == 0