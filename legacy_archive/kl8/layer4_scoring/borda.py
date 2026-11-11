"""Borda 百分位秩归一化（契约 R24、§3.10、8.md L126–L129）。

公式：
R(v_i) = Rank(v_i) / 80 ∈ (0, 1]
并列取平均秩（Average Rank）。
数值越接近 1，代表该号码在该指标下处于当期全市场最顶尖水平。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from kl8.constants import N_NUM


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """计算一维数组的 Borda 百分位秩归一化得分（契约 R24）。

    @param values (80,) 一维连续指标
    @return (80,) float64 百分位秩，范围 ∈ (0, 1]
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)

    # 升序排序索引：值最小的 rank 为 1，值最大的 rank 为 n
    sorter = np.argsort(arr)
    inv = np.empty(n, dtype=np.int64)
    inv[sorter] = np.arange(n)

    # 处理并列取平均秩
    arr_sorted = arr[sorter]
    # 找到不相等处的分界线
    diff = np.diff(arr_sorted)
    splits = np.flatnonzero(diff != 0) + 1
    splits = np.concatenate([[0], splits, [n]])

    ranks_sorted = np.empty(n, dtype=np.float64)
    for start, end in zip(splits[:-1], splits[1:]):
        # 1-based 秩为 [start + 1, end]，平均秩为 (start + 1 + end) / 2
        avg_rank = (float(start + 1) + float(end)) / 2.0
        ranks_sorted[start:end] = avg_rank

    ranks = ranks_sorted[inv]
    return ranks / float(n)


def rank_matrix(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """对特征字典中所有一维数组批量执行百分位秩归一化。

    @param features 键为特征名，值为 (80,) 数组
    @return 键为特征名，值为 (80,) float64 百分位秩
    """
    return {k: percentile_rank(v) for k, v in features.items()}


def top_percentile_mask(ranks: np.ndarray, threshold: float) -> np.ndarray:
    """判断百分位秩是否达到指定阈值（例如 R >= 0.65）。"""
    return ranks >= threshold
