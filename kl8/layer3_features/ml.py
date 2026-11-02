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
from typing import Any, Mapping

import numpy as np

from kl8.config import MLConfig
from kl8.constants import ML_FEATURE_WARMUP, N_NUM, P0

#: ML 特征 schema 版本戳（特征列变更时递增，使缓存自然失效）。
ML_FEATURE_SCHEMA: str = "fs-v1"
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import WindowSpec

try:
    import lightgbm as lgb  # type: ignore[import-untyped]
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

#: ML 训练/预测允许捕获的异常（P8-M7：禁止裸 except；LightGBM 专属异常在可用时登记）。
_ML_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    ImportError,
    np.linalg.LinAlgError,
)
if _HAS_LGB:  # pragma: no branch —— 未安装 lightgbm 时无该异常类型可登记
    _ML_ERRORS = _ML_ERRORS + (lgb.basic.LightGBMError,)


@dataclass(frozen=True)
class MLResult:
    """机器学习概率预测结果。"""

    prob: np.ndarray
    """(80,) 开出概率预测。"""

    available: bool
    """是否由真实模型产出（False 表示降级先验）。"""

    backend: str
    """实际生效后端：lightgbm / ema / uniform。"""

    refit: bool
    """本次调用是否重新执行训练+预测并写入预测缓存（False 表示命中缓存）。"""

    n_train: int
    """训练样本条数（80 码 × 期数）。"""

    warnings: tuple[str, ...]
    """降级或异常告警。"""


class MLCache:
    """进程内 LRU 缓存器：按 end 缓存预测结果，同时按重训块缓存已训练模型（契约 §3.9、R23）。

    【P8-M12 修复要点】
    1. ``put`` 对 ndarray 做深拷贝，外部改写不得污染缓存（缓存可信性）；
    2. 预测缓存键仍含 ``end``（每期预测本就不同，这是正确语义）；
    3. 额外维护"重训块 → 已训练模型"缓存：``cfg.refit_every`` 块内复用模型，
       使 ``refit_every`` 真正生效、避免回测每期重训（旧实现的性能缺陷）。
    """

    def __init__(self, max_size: int = 8) -> None:
        """初始化缓存器。

        @param max_size 预测/模型两类缓存各自的最大条目数（<1 时按 1 处理）
        """
        self.max_size = max(1, max_size)
        self._cache: OrderedDict[tuple[int, str, str], np.ndarray] = OrderedDict()
        self._models: OrderedDict[tuple[int, str, str], Any] = OrderedDict()

    def get(self, key: tuple[int, str, str]) -> np.ndarray | None:
        """读取预测缓存项并标记为最近使用。

        @param key 缓存键 ``(end, 特征schema, 种子)``
        @return 概率数组副本；未命中返回 None
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        return None

    def put(self, key: tuple[int, str, str], prob: np.ndarray) -> None:
        """写入预测缓存（保存副本，超出容量淘汰最久未用）。

        @param key 缓存键
        @param prob (80,) 概率预测
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = np.array(prob, dtype=np.float64, copy=True)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def get_model(self, key: tuple[int, str, str]) -> Any | None:
        """读取某重训块的已训练模型。

        @param key 模型缓存键 ``(重训块起点, 特征schema, 种子)``
        @return 模型包 (clf, train_preds, y_train)；未命中返回 None
        """
        if key in self._models:
            self._models.move_to_end(key)
            return self._models[key]
        return None

    def put_model(self, key: tuple[int, str, str], bundle: Any) -> None:
        """写入某重训块的已训练模型（超出容量淘汰最久未用）。

        @param key 模型缓存键
        @param bundle 模型包 (clf, train_preds, y_train)
        """
        if key in self._models:
            self._models.move_to_end(key)
        self._models[key] = bundle
        if len(self._models) > self.max_size:
            self._models.popitem(last=False)

    def clear(self) -> None:
        """清空全部预测与模型缓存项。"""
        self._cache.clear()
        self._models.clear()

    def __len__(self) -> int:
        """返回当前预测缓存项数量。"""
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


def _pav_calibrate_predict(
    raw_prob: np.ndarray,
    train_scores: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    """用训练集 PAV 保序回归把原始概率映射为校准后概率（契约 R23，P8-H6 真校准）。

    PAV 输出的是"训练样本自身分数 → 校准概率"的分段常数映射；对当期原始概率，
    在训练分数序列中定位插入点后取对应分段值（单调非降，保持秩序）。

    @param raw_prob (80,) 当期模型原始预测概率
    @param train_scores (N,) 训练集原始预测概率
    @param train_labels (N,) 训练集真实标签 0/1
    @return (80,) float64 校准后概率 ∈ [0,1]
    """
    if train_scores.size == 0 or raw_prob.size == 0:
        return np.clip(raw_prob, 0.0, 1.0)
    calibrated_train = pav_calibrate(train_scores, train_labels)
    order = np.argsort(train_scores, kind="mergesort")
    sorted_scores = np.asarray(train_scores, dtype=np.float64)[order]
    sorted_cal = np.asarray(calibrated_train, dtype=np.float64)[order]
    positions = np.searchsorted(sorted_scores, np.asarray(raw_prob, dtype=np.float64), side="right") - 1
    positions = np.clip(positions, 0, sorted_scores.size - 1)
    return np.clip(sorted_cal[positions], 0.0, 1.0)


def _refit_block_start(end: int, cfg: MLConfig) -> int:
    """计算当前期所属的重训块起点（契约 M12：``refit_every`` 必须真正生效）。

    块起点 = 在 ``ML_FEATURE_WARMUP`` 之后按 ``refit_every`` 切块的上界；块内共享同一
    已训练模型。这样既能让缓存真正命中，又保证训练数据严格早于预测期（无穿越）。

    @param end 当期历史截断索引
    @param cfg ML 配置
    @return 重训块起点（≤ end）
    """
    anchor = int(ML_FEATURE_WARMUP)
    span = max(1, int(cfg.refit_every))
    if end <= anchor:
        return end
    return anchor + ((end - anchor) // span) * span


def train_predict(
    state: BinaryStateMatrix,
    end: int,
    cfg: MLConfig,
    *,
    cache: MLCache | None = None,
) -> MLResult:
    """训练并预测截断时间点 end 下一期的开出概率（契约 R23、§3.9、P8-M12）。

    【P8-M12 修复】
    1. 预测缓存键 = ``(end, 特征 schema, seed)``：同一 end 第二次命中直接返回（refit=False）；
    2. 另按 ``cfg.refit_every`` 切块缓存**已训练模型**：块内复用模型、只重算当期特征，
       ``refit_every`` 配置真正生效（旧实现每期必重训）；
    3. ``ML_FEATURE_WARMUP`` 取代硬编码的 10。

    @param state 状态矩阵
    @param end 历史截断端点
    @param cfg ML 配置项
    @param cache 可选 LRU 缓存（预测结果 + 重训模型）
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

    pred_key = (end, ML_FEATURE_SCHEMA, str(cfg.seed))
    if cache is not None:
        cached_prob = cache.get(pred_key)
        if cached_prob is not None:
            return MLResult(
                prob=cached_prob,
                available=True,
                backend="lightgbm",
                refit=False,
                n_train=cfg.train_window,
                warnings=(),
            )

    block_start = _refit_block_start(end, cfg)
    model_key = (block_start, ML_FEATURE_SCHEMA, str(cfg.seed))
    model_bundle: Any | None = cache.get_model(model_key) if cache is not None else None

    try:
        from kl8.config import WindowConfig
        from kl8.layer1_representation.windows import resolve_windows

        win_spec = resolve_windows(WindowConfig(), n_hist)

        if model_bundle is None:
            # 训练集：用 ML_FEATURE_WARMUP .. block_start-1 的历史逐期构造样本
            train_end = max(int(ML_FEATURE_WARMUP), block_start)
            t_start = max(int(ML_FEATURE_WARMUP), train_end - int(cfg.train_window) + 1)
            x_train_list: list[np.ndarray] = []
            y_train_list: list[np.ndarray] = []
            for t in range(t_start, train_end):
                # 用 t-1 的特征预测 t 是否开出（严格单步、无穿越）
                x_train_list.append(extract_feature_matrix(state, t - 1, win_spec))
                y_train_list.append(state.hit[t, :].astype(np.float64))

            if not x_train_list:
                raise ValueError("训练样本为空（历史长度不足）")
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
            train_preds = np.asarray(clf.predict_proba(x_train)[:, 1], dtype=np.float64)
            model_bundle = (clf, train_preds, y_train)
            if cache is not None:
                cache.put_model(model_key, model_bundle)

        clf, train_preds, y_train = model_bundle
        x_curr = extract_feature_matrix(state, end, win_spec)
        raw_prob = np.asarray(clf.predict_proba(x_curr)[:, 1], dtype=np.float64)

        # 真 PAV 校准（历史死分支 cal_prob = pav_calibrate(raw, raw) 已删除）
        if cfg.calibrate == "pav":
            prob = _pav_calibrate_predict(raw_prob, train_preds, y_train)
        else:
            prob = raw_prob
        prob = np.clip(prob, 0.05, 0.95)
        if cache is not None:
            cache.put(pred_key, prob)

        return MLResult(
            prob=prob,
            available=True,
            backend="lightgbm",
            refit=True,
            n_train=int(y_train.size),
            warnings=(),
        )
    except _ML_ERRORS as exc:
        warnings.append(f"LightGBM 训练或预测异常: {exc}，降级使用 EMA")
        prob = ema_fallback(state, end)
        return MLResult(
            prob=prob,
            available=False,
            backend="ema",
            refit=False,
            n_train=0,
            warnings=tuple(warnings),
        )