"""多尺度特征工程统一装配入口（契约 R23、§3.9、§4.1）。

唯一特征快照装配中心：
- 严格遵循因果律，仅使用 X[0..end] 历史切片
- 装配冷热、马尔可夫、共现图、宏观形态与机器学习概率
- 出口执行严格的数值健全性校验与防穿越断言
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from kl8.config import AppConfig, config_hash
from kl8.constants import (
    E_REPEAT,
    N_NUM,
    P0,
    Z_SIG,
    get_road_012,
    is_prime_num,
    size_class_of,
    tail_of,
    zone4_of,
    zone8_of,
)
from kl8.datatypes import BaselineSnapshot, FeatureSnapshot, assert_finite
from kl8.errors import DataFormatError, InsufficientDataError
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import effective_window, is_cold_start, resolve_windows
from kl8.layer2_baseline.chisq import chi_square_by_window
from kl8.layer2_baseline.entropy import feature_redundancy, freq_entropy, mi_lag1
from kl8.layer2_baseline.randomness import flatten_history, nist_lite
from kl8.layer2_baseline.runs import runs_test_80
from kl8.layer2_baseline.zscore import z_all_windows
from kl8.layer3_features.cooccurrence import (
    centered_cooccurrence,
    co_centered_mean,
    co_score_all_windows,
    cooccurrence_counts,
    triad_counts,
)
from kl8.layer3_features.graph import build_graph_features
from kl8.layer3_features.heat import build_heat
from kl8.layer3_features.markov import build_markov
from kl8.layer3_features.ml import MLCache, MLResult, train_predict

SCHEMA_VERSION: str = "fs-v1"


def validate_snapshot(snapshot: FeatureSnapshot) -> None:
    """校验 FeatureSnapshot 全部特征数组的数值健全性（无 NaN/Inf、长度为 80）。

    @param snapshot 特征快照对象
    @raises DataFormatError 存在 NaN、Inf 或维度不符
    """
    for name, arr in snapshot.arrays().items():
        assert_finite(arr, f"FeatureSnapshot.{name}")


def feature_matrix(snapshot: FeatureSnapshot, fields: Sequence[str]) -> np.ndarray:
    """提取快照中指定字段构成的特征矩阵 (80, len(fields))。"""
    cols: list[np.ndarray] = []
    for f in fields:
        val = getattr(snapshot, f)
        cols.append(np.asarray(val, dtype=np.float64))
    return np.column_stack(cols)


def build_feature_snapshot(
    state: BinaryStateMatrix,
    end: int,
    cfg: AppConfig,
    *,
    include_ml: bool | None = None,
    ml_cache: MLCache | None = None,
) -> FeatureSnapshot:
    """唯一特征快照装配中心（契约 §3.9、R23）。

    语义：使用 X[0..end] 预测 end+1。
    @param state 状态矩阵
    @param end 历史最大截断索引（0-based，含端点）
    @param cfg 全局配置
    @param include_ml 是否启用 ML 预测，默认根据 cfg.ml.enabled
    @param ml_cache 可选 ML LRU 缓存
    @return 完整装配并校验通过的 FeatureSnapshot
    @raises IndexError end 越界
    @raises InsufficientDataError 历史样本不足以支撑基础窗口
    """
    if end < 0 or end >= state.T:
        raise IndexError(f"历史截断索引 end={end} 越界，可用范围 [0, {state.T - 1}]")

    n_hist = end + 1
    if n_hist < cfg.windows.min_history:
        raise InsufficientDataError(
            f"历史期数 {n_hist} 小于最小历史需求 {cfg.windows.min_history} 期，"
            "无法可靠构建多尺度统计特征"
        )

    windows = resolve_windows(cfg.windows, n_hist)
    eff_windows = {k: effective_window(v.size, n_hist) for k, v in windows.items()}
    cold_start = any(is_cold_start(v.size, n_hist) for v in windows.values())
    cfg_hash = config_hash(cfg)
    last_set = state.latest_set(end)
    warnings: list[str] = []

    # 1. 冷热动力学特征
    heat = build_heat(state, end, windows)

    # 2. 马尔可夫转移跟随特征
    markov = build_markov(state, end, windows, cfg.markov)

    # 3. 同期共现矩阵与伴生拉动特征
    co_mats: dict[str, np.ndarray] = {}
    for name, spec in windows.items():
        w_mat = state.window_matrix(end, spec.size)
        c_cnt = cooccurrence_counts(w_mat)
        co_mats[name] = centered_cooccurrence(c_cnt, eff_windows[name])
    co_scores = co_score_all_windows(co_mats, last_set)
    co_mean = co_centered_mean(co_scores)

    # 全量三元团统计
    all_mat = state.window_matrix(end, None)
    all_co_cnt = cooccurrence_counts(all_mat)
    triads_tot, triad_cnt = triad_counts(all_co_cnt)

    # 4. 共现图拓扑特征 (以主力趋势窗口 25G 为基准)
    graph_res = build_graph_features(co_mats["25G"], seed=cfg.seed)
    warnings.extend(graph_res.warnings)

    # 5. 宏观空间形态固定静态属性 (80 维)
    z4 = np.array([zone4_of(i + 1) for i in range(N_NUM)], dtype=np.int8)
    z8 = np.array([zone8_of(i + 1) for i in range(N_NUM)], dtype=np.int8)
    odd = np.array([(i + 1) % 2 == 1 for i in range(N_NUM)], dtype=bool)
    prime = np.array([is_prime_num(i + 1) for i in range(N_NUM)], dtype=bool)
    s_cls = np.array([size_class_of(i + 1) for i in range(N_NUM)], dtype=np.int8)
    tails = np.array([tail_of(i + 1) for i in range(N_NUM)], dtype=np.int8)
    roads = np.array([get_road_012(i + 1) for i in range(N_NUM)], dtype=np.int8)

    # 6. 机器学习概率预测
    should_ml = cfg.ml.enabled if include_ml is None else include_ml
    if should_ml:
        ml_res = train_predict(state, end, cfg.ml, cache=ml_cache)
        warnings.extend(ml_res.warnings)
    else:
        ml_res = MLResult(
            prob=np.full(N_NUM, P0, dtype=np.float64),
            available=False,
            backend="uniform",
            refit=False,
            n_train=0,
            warnings=(),
        )

    # 7. 随机性基准快照 (BaselineSnapshot)
    z_map = z_all_windows(state, end, windows, p=P0)
    sig_any = np.zeros(N_NUM, dtype=bool)
    for z_arr in z_map.values():
        sig_any |= np.abs(z_arr) > Z_SIG

    runs_z, _runs_cnt, runs_lbls = runs_test_80(all_mat)
    chi2_map = chi_square_by_window(state, end, windows)
    flat_bits = flatten_history(all_mat)
    nist_tuple = nist_lite(flat_bits)

    ent_map = {
        "10G": freq_entropy(state.window_sum(end, 10), eff_windows["10G"]),
        "ALL": freq_entropy(state.window_sum(end, None), n_hist),
    }
    recent_mat = state.window_matrix(end, min(200, n_hist))
    mi_mat = mi_lag1(recent_mat)
    mi_pairs: list[tuple[str, str, float]] = []
    for r in range(N_NUM):
        for c in range(r + 1, N_NUM):
            mi_pairs.append((f"{r + 1:02d}", f"{c + 1:02d}", float(mi_mat[r, c])))
    mi_pairs.sort(key=lambda x: x[2], reverse=True)
    mi_top = tuple(mi_pairs[:10])

    baseline_snap = BaselineSnapshot(
        z_10G=z_map["10G"],
        z_25G=z_map["25G"],
        z_50G=z_map["50G"],
        z_ALL=z_map["ALL"],
        significant_any=sig_any,
        runs_z=runs_z,
        runs_labels=runs_lbls,
        chi2=chi2_map,
        nist=nist_tuple,
        entropy=ent_map,
        mi_top_pairs=mi_top,
        expected_repeat=E_REPEAT,
    )

    # 8. 实例化 FeatureSnapshot
    target_period = state.target_period if hasattr(state, "target_period") else end + 1
    snapshot = FeatureSnapshot(
        as_of_index=end,
        target_period=target_period,
        n_history=n_hist,
        effective_windows=eff_windows,
        cold_start=cold_start,
        schema_version=SCHEMA_VERSION,
        config_hash=cfg_hash,
        freq_10=heat.freq["10G"],
        freq_25=heat.freq["25G"],
        freq_50=heat.freq["50G"],
        freq_all=heat.freq["ALL"],
        mom_10_25=heat.mom_10_25,
        mom_25_50=heat.mom_25_50,
        gap_ratio=heat.gap_ratio,
        pressure=heat.pressure,
        follow_10=markov.follow["10G"],
        follow_25=markov.follow["25G"],
        follow_50=markov.follow["50G"],
        follow_all=markov.follow["ALL"],
        co_10=co_scores["10G"],
        co_25=co_scores["25G"],
        co_50=co_scores["50G"],
        co_all=co_scores["ALL"],
        co_centered_mean=co_mean,
        pagerank=graph_res.pagerank,
        eigen=graph_res.eigen,
        community_strength=graph_res.community_strength,
        gap_mean=heat.gap_mean,
        follow2_all=markov.follow2_all,
        ml_prob=ml_res.prob,
        gap=heat.gap,
        golden_cross=heat.golden_cross,
        community_id=graph_res.community_id,
        community_size=graph_res.community_size,
        triad_count=triad_cnt,
        zone4=z4,
        zone8=z8,
        is_odd=odd,
        is_prime=prime,
        size_class=s_cls,
        tail=tails,
        road012=roads,
        follow_diag=markov.follow_diag,
        follow_offdiag=markov.follow_offdiag,
        companion=graph_res.companion,
        co_matrices=co_mats,
        triads_total=triads_tot,
        ml_available=ml_res.available,
        ml_backend=ml_res.backend,
        baseline=baseline_snap,
        warnings=tuple(warnings),
    )

    validate_snapshot(snapshot)
    return snapshot
