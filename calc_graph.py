# -*- coding: utf-8 -*-
"""快乐8新一代预测系统 —— 共现图网络与伴生挖掘引擎

【业务背景与架构定位】:
快乐8每期开出20个号码，号码之间存在显著的局部伴生聚合与共现拓扑网络关系。
原有系统假设每个号码完全孤立，忽视了多码同出的联合图结构。
本模块基于 NetworkX 构建近 W 期的加权共现无向图 (Co-occurrence Graph)，
计算各号码在拓扑网络中的加权度中心度 (Degree Centrality) 与 PageRank 枢纽度，
并挖掘高置信度伴生号码对 (Companion Pairs)，为全局组合优化提供网络协同特征。

@author czw
@since 2026-10-30
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np

from calc_core import N_NUM, TimeSeries

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_WINDOW: int = 50  # 图网络分析滑动窗口期数 (近期50期共现最敏感)


def build_cooccurrence_matrix(ts: TimeSeries, current_t: int, window: int = DEFAULT_GRAPH_WINDOW) -> np.ndarray:
    """计算近 window 期 80 个号码之间的联合共现频次矩阵 (形状: 80 × 80)。

    // 步骤 1: 截取滑动窗口历史开奖 0-1 矩阵
    // 步骤 2: 矩阵转置点积快速计算任意两码同出频次: C = H^T * H
    // 步骤 3: 对角线清零 (排除自身与自身共现) 并返回整数对称阵

    @param ts 时序数据集缓存对象
    @param current_t 当前目标截断期
    @param window 滑动计算窗口期数
    @return 形状为 (80, 80) 的非负整数对称矩阵
    """
    total_t = current_t if current_t is not None else ts.T
    if total_t <= 0:
        return np.zeros((N_NUM, N_NUM), dtype=float)

    start_t = max(0, total_t - window)
    recent_hit = ts.hit[start_t:total_t]  # (W, 80)
    co_matrix = recent_hit.T @ recent_hit  # (80, 80)
    np.fill_diagonal(co_matrix, 0.0)
    return co_matrix


def analyze_graph_centrality(
    co_matrix: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[tuple[int, float]]]]:
    """基于 NetworkX 分析共现拓扑网络，输出 80 码中心度得分与各码的顶尖伴生号码列表。

    // 步骤 1: 根据共现权重矩阵构建无向加权图
    // 步骤 2: 计算图节点的加权度中心度与 PageRank 枢纽度
    // 步骤 3: 归一化综合图网络协同得分 Score_graph ∈ [0, 1]
    // 步骤 4: 提取每个号码的前 3 大高频伴生号码对及亲和度

    @param co_matrix 形状为 (80, 80) 的共现矩阵
    @return (形状为 80 的图协同度数组, 号码编号 -> [(伴生号码, 归一化亲和度)] 的映射字典)
    """
    companion_dict: dict[int, list[tuple[int, float]]] = {}
    if np.all(co_matrix == 0):
        return np.zeros(N_NUM, dtype=float), companion_dict

    # 步骤 1: 提取各号码的前 top-3 伴生亲和对
    for i in range(N_NUM):
        num = i + 1
        row = co_matrix[i]
        top_indices = np.argsort(-row)[:3]
        max_val = float(row[top_indices[0]]) if row[top_indices[0]] > 0 else 1.0
        pairs = []
        for idx in top_indices:
            if row[idx] > 0:
                pairs.append((idx + 1, float(row[idx]) / max_val))
        companion_dict[num] = pairs

    # 步骤 2: 计算网络拓扑指标
    try:
        import networkx as nx

        graph = nx.from_numpy_array(co_matrix)
        degree_dict = dict(graph.degree(weight="weight"))
        pagerank_dict = nx.pagerank(graph, weight="weight", max_iter=100)

        deg_arr = np.array([degree_dict.get(i, 0.0) for i in range(N_NUM)], dtype=float)
        pr_arr = np.array([pagerank_dict.get(i, 0.0) for i in range(N_NUM)], dtype=float)

        # 步骤 3: 归一化融合
        norm_deg = deg_arr / (deg_arr.max() + 1e-6)
        norm_pr = pr_arr / (pr_arr.max() + 1e-6)
        graph_scores = 0.50 * norm_deg + 0.50 * norm_pr
        return np.asarray(graph_scores, dtype=float), companion_dict

    except Exception as exc:
        logger.warning("NetworkX 图算法计算异常，启动度数累加兜底: %s", exc)
        row_sums = co_matrix.sum(axis=1)
        max_sum = float(row_sums.max()) if row_sums.max() > 0 else 1.0
        return row_sums / max_sum, companion_dict
