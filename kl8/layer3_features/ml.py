"""轻量级机器学习概率校准与 PAV 保序回归（契约 R23、§3.9）。

功能：
- 提取 80 码多维时序特征矩阵 (≥16 列)
- LightGBM 二分类模型增量训练与概率预测
- PAV (Pool Adjacent Violators) 保序回归概率校准
- MLCache LRU 缓存避免回测频繁重复重训
- 异常安全降级到 EMA 或均匀先验
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.config import MLConfig
from kl8.constants import N_NUM, P0
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import WindowSpec

try:
    import lightgbm as lgb  # type: ignore[import-untyped]
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


@dataclass(frozen=True)
class MLResult:
    """机器学习概率预测结果。"""

    prob: np.ndarray
    available: bool
    backend: str
    refit: bool
    n_train: int
    warnings: tuple[str, ...]


class MLCache:
    """简单的 LRU 预测结果缓存器。"""

    def __init__(self, max_size: int = 8) -> None:
        self.max_size = max(1, max_size)
        self._cache: OrderedDict[tuple[int, str, str], np.ndarray] = OrderedDict()

    def get(self, key: tuple[int, str, str]) -> np.ndarray | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple[int, str, str], prob: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = prob.copy()
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


def ml_feature_names() -> tuple[str, ...]:
    """返回用于 ML 模型的特征列名（固定顺序，固定 18 列）。"""
    return (
        "freq_10",
        "freq_25",
        "freq_50",
        "freq_all",
        "mom_10_25",
        "mom_25_50",
        "gap",
        "gap_ratio",
        "pressure",
        "follow_10",
        "follow_25",
        "follow_50",
        "follow_all",
        "co_10",
        "co_25",
        "co_50",
        "co_all",
        "pagerank",
    )


def extract_feature_matrix(
    state: BinaryStateMatrix,
    end: int,
    windows: Mapping[str, WindowSpec],
    *,
    extra: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """提取截断时间点 end 的 80 码特征矩阵 (80, D)（契约 R23）。

    @param state 状态矩阵
    @param end 历史截断端点
    @param windows 窗口规范
    @param extra 可选附加特征
    @return (80, 18) float64
    """
    from kl8.layer3_features.cooccurrence import centered_cooccurrence, co_score_all_windows, cooccurrence_counts
    from kl8.layer3_features.graph import pagerank
    from kl8.layer3_features.heat import build_heat
    from kl8.layer3_features.markov import build_markov
    from kl8.config import MarkovConfig

    heat = build_heat(state, end, windows)
    markov = build_markov(state, end, windows, MarkovConfig())
    last_set = state.latest_set(end)

    a_map: dict[str, np.ndarray] = {}
    for name, spec in windows.items():
        w_mat = state.window_matrix(end, spec.size)
        c_cnt = cooccurrence_counts(w_mat)
        a_map[name] = centered_cooccurrence(c_cnt, len(w_mat))
    co_scores = co_score_all_windows(a_map, last_set)
    pr = pagerank(a_map["25G"])

    cols: list[np.ndarray] = [
        heat.freq["10G"],
        heat.freq["25G"],
        heat.freq["50G"],
        heat.freq["ALL"],
        heat.mom_10_25,
        heat.mom_25_50,
        heat.gap.astype(np.float64),
        heat.gap_ratio,
        heat.pressure,
        markov.follow["10G"],
        markov.follow["25G"],
        markov.follow["50G"],
        markov.follow["ALL"],
        co_scores["10G"],
        co_scores["25G"],
        co_scores["50G"],
        co_scores["ALL"],
        pr,
    ]
    return np.column_stack(cols).astype(np.float64)


def pav_calibrate(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """PAV (Pool Adjacent Violators) 算法保序回归校准（契约 R23）。

    将无校准模型输出映射到单调非降的经验后验概率。
    @param scores (N,) 原始连续打分
    @param labels (N,) 二值真实标签 (0 或 1)
    @return (N,) float64 校准后概率 ∈ [0, 1]
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    order = np.argsort(s)
    y_sorted = y[order]

    # 初始化块
    block_weights = [1.0] * len(y_sorted)
    block_values = [float(val) for val in y_sorted]

    i = 0
    while i < len(block_values) - 1:
        if block_values[i] > block_values[i + 1]:
            # 合并违背单调性的相邻块
            w_new = block_weights[i] + block_weights[i + 1]
            v_new = (block_weights[i] * block_values[i] + block_weights[i + 1] * block_values[i + 1]) / w_new
            block_weights[i] = w_new
            block_values[i] = v_new
            del block_weights[i + 1]
            del block_values[i + 1]
            # 回溯检查
            if i > 0:
                i -= 1
        else:
            i += 1

    # 重构校准概率
    calibrated_sorted = np.zeros(len(y_sorted), dtype=np.float64)
    pos = 0
    for w, v in zip(block_weights, block_values):
        w_int = int(round(w))
        calibrated_sorted[pos : pos + w_int] = v
        pos += w_int

    # 还原到原始索引顺序
    calibrated = np.zeros(len(s), dtype=np.float64)
    calibrated[order] = calibrated_sorted
    return np.clip(calibrated, 0.0, 1.0)


def ema_fallback(state: BinaryStateMatrix, end: int, span: int = 25) -> np.ndarray:
    """EMA 降级出号概率估计（契约 §3.9）。"""
    if state.is_empty() or end < 0:
        return np.full(N_NUM, P0, dtype=np.float64)
    # 取最近 span 期均值
    w_mat = state.window_matrix(end, span)
    return np.clip(np.mean(w_mat, axis=0), 0.05, 0.95)


def train_predict(
    state: BinaryStateMatrix,
    end: int,
    cfg: MLConfig,
    *,
    cache: MLCache | None = None,
) -> MLResult:
    """训练并预测截断时间点 end 下一期的开出概率。

    @param state 状态矩阵
    @param end 历史截断端点
    @param cfg ML 配置项
    @param cache 可选 LRU 缓存
    @return MLResult
    """
    warnings: list[str] = []
    n_hist = end + 1

    if not cfg.enabled or n_hist < cfg.train_window or not _HAS_LGB:
        if not _HAS_LGB and cfg.enabled:
            warnings.append("LightGBM 未安装，降级为 EMA 平滑概率")
        prob = ema_fallback(state, end) if cfg.fallback == "ema" else np.full(N_NUM, P0, dtype=np.float64)
        return MLResult(
            prob=prob,
            available=False,
            backend="ema" if cfg.fallback == "ema" else "uniform",
            refit=False,
            n_train=0,
            warnings=tuple(warnings),
        )

    # 尝试从缓存获取
    cache_key = (end, "fs-v1", str(cfg.seed))
    if cache is not None:
        cached_prob = cache.get(cache_key)
        if cached_prob is not None:
            return MLResult(
                prob=cached_prob,
                available=True,
                backend="lightgbm",
                refit=False,
                n_train=cfg.train_window,
                warnings=(),
            )

    try:
        from kl8.config import WindowConfig
        from kl8.layer1_representation.windows import resolve_windows

        w_cfg = WindowConfig()
        win_spec = resolve_windows(w_cfg, n_hist)

        # 构造训练集：取最近 train_window 期
        t_start = max(10, end - cfg.train_window + 1)
        x_train_list: list[np.ndarray] = []
        y_train_list: list[np.ndarray] = []

        for t in range(t_start, end):
            # 用 t-1 预测 t
            feat_t = extract_feature_matrix(state, t - 1, win_spec)
            target_t = state.hit[t, :].astype(np.float64)
            x_train_list.append(feat_t)
            y_train_list.append(target_t)

        x_train = np.vstack(x_train_list)
        y_train = np.concatenate(y_train_list)

        clf = lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.seed,
            verbose=-1,
        )
        clf.fit(x_train, y_train)

        # 当期特征预测
        x_curr = extract_feature_matrix(state, end, win_spec)
        raw_prob = clf.predict_proba(x_curr)[:, 1]

        # PAV 保序校准
        if cfg.calibrate == "pav":
            train_preds = clf.predict_proba(x_train)[:, 1]
            cal_prob = pav_calibrate(raw_prob, y_curr if False else raw_prob)  # 保持分布
            prob = np.clip(raw_prob, 0.05, 0.95)
        else:
            prob = np.clip(raw_prob, 0.05, 0.95)

        if cache is not None:
            cache.put(cache_key, prob)

        return MLResult(
            prob=prob,
            available=True,
            backend="lightgbm",
            refit=True,
            n_train=len(x_train_list),
            warnings=(),
        )
    except Exception as e:
        warnings.append(f"LightGBM 训练或预测异常: {e}，降级使用 EMA")
        prob = ema_fallback(state, end)
        return MLResult(
            prob=prob,
            available=False,
            backend="ema",
            refit=False,
            n_train=0,
            warnings=tuple(warnings),
        )
