"""信息论、熵值、互信息与综合随机性基准报告（契约 R10、§3.8、§4.7）。

实现功能：
- 香农熵与频次分布熵计算
- 滞后 1 阶跨期互信息矩阵 mi_lag1
- 特征冗余度评估 feature_redundancy
- 指数移动平均平滑与威布尔生存诊断
- 综合随机性报告装配 randomness_report
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kl8.constants import N_NUM, P0, Z_SIG
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer1_representation.windows import WindowSpec
from kl8.layer2_baseline.chisq import ChiSquareResult, chi_square_by_window
from kl8.layer2_baseline.randomness import NistResult, flatten_history, nist_lite
from kl8.layer2_baseline.runs import runs_test_80
from kl8.layer2_baseline.zscore import z_all_windows


@dataclass(frozen=True)
class RandomnessReport:
    """综合随机性基准检验报告（契约 §4.7）。"""

    n_history: int
    p0: float
    z_by_window: Mapping[str, np.ndarray]
    significant_count: int
    chi2: Mapping[str, ChiSquareResult]
    nist: tuple[NistResult, ...]
    entropy_80: float
    entropy_max: float
    mi_top_pairs: tuple[tuple[str, str, float], ...]
    runs_summary: Mapping[str, int]
    notes: tuple[str, ...]


def shannon_entropy(p: np.ndarray, *, base: float = 2.0, eps: float = 1e-12) -> float:
    """计算离散概率分布的香农熵。

    @param p 概率分布数组（需归一化）
    @param base 对数底数，默认 2.0（单位 bit）
    @param eps 防 log(0) 极小值
    @return 香农熵标量
    """
    arr = np.asarray(p, dtype=np.float64)
    total = float(arr.sum())
    if total <= eps:
        return 0.0
    probs = arr / total
    probs = probs[probs > eps]
    log_p = np.log(probs) / math.log(base)
    return float(-np.sum(probs * log_p))


def freq_entropy(window_sum: np.ndarray, w: int) -> float:
    """计算 80 码出现频次的分布熵（理想均匀分布为 log2(80) ≈ 6.321928 bit）。

    @param window_sum (80,) 出现次数
    @param w 窗口期数
    @return 实际分布熵
    """
    if w <= 0:
        return math.log2(float(N_NUM))
    return shannon_entropy(window_sum.astype(np.float64), base=2.0)


def mi_lag1(window_matrix: np.ndarray) -> np.ndarray:
    """计算号码间的 1 阶滞后互信息矩阵 I(X_{t-1, i}; X_{t, j})。

    @param window_matrix (T, 80) 二值切片矩阵
    @return (80, 80) float64 互信息矩阵
    """
    mat = np.asarray(window_matrix, dtype=np.float64)
    t_len = len(mat)
    if t_len < 2:
        return np.zeros((N_NUM, N_NUM), dtype=np.float64)

    x_prev = mat[:-1]  # (T-1, 80)
    x_curr = mat[1:]   # (T-1, 80)
    n = float(t_len - 1)

    # 边缘概率
    p1_prev = np.mean(x_prev, axis=0)  # (80,)
    p0_prev = 1.0 - p1_prev
    p1_curr = np.mean(x_curr, axis=0)  # (80,)
    p0_curr = 1.0 - p1_curr

    # 联合频次与概率 (80, 80)
    # n11[i, j] = sum_t (x_prev[t, i] * x_curr[t, j])
    n11 = x_prev.T @ x_curr
    p11 = n11 / n
    p10 = (x_prev.T @ (1.0 - x_curr)) / n
    p01 = ((1.0 - x_prev).T @ x_curr) / n
    p00 = ((1.0 - x_prev).T @ (1.0 - x_curr)) / n

    mi = np.zeros((N_NUM, N_NUM), dtype=np.float64)
    eps = 1e-12

    for p_joint, p_x, p_y in [
        (p11, p1_prev[:, None], p1_curr[None, :]),
        (p10, p1_prev[:, None], p0_curr[None, :]),
        (p01, p0_prev[:, None], p1_curr[None, :]),
        (p00, p0_prev[:, None], p0_curr[None, :]),
    ]:
        valid = (p_joint > eps) & (p_x > eps) & (p_y > eps)
        ratio = np.where(valid, p_joint / (p_x * p_y + eps), 1.0)
        ratio = np.maximum(ratio, eps)
        contrib = np.where(valid, p_joint * np.log2(ratio), 0.0)
        mi += contrib

    return np.maximum(0.0, mi)


def feature_redundancy(
    features: Mapping[str, np.ndarray],
    top_n: int = 20,
) -> tuple[tuple[str, str, float], ...]:
    """评估输入特征字典各维度之间的相关性冗余对。

    @param features 键为特征名，值为 (80,) 数组
    @param top_n 返回最高相关的前 N 对
    @return ((feat_a, feat_b, abs_corr), ...) 降序排列
    """
    names = list(features.keys())
    if len(names) < 2:
        return ()

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(names)):
        v_i = np.asarray(features[names[i]], dtype=np.float64)
        std_i = float(np.std(v_i))
        if std_i < 1e-9:
            continue
        for j in range(i + 1, len(names)):
            v_j = np.asarray(features[names[j]], dtype=np.float64)
            std_j = float(np.std(v_j))
            if std_j < 1e-9:
                continue
            corr = float(np.corrcoef(v_i, v_j)[0, 1])
            if not math.isnan(corr):
                pairs.append((names[i], names[j], abs(corr)))

    pairs.sort(key=lambda item: item[2], reverse=True)
    return tuple(pairs[:top_n])


def ema_smooth(x: np.ndarray, span: int) -> np.ndarray:
    """一维序列指数移动平均平滑。"""
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) == 0 or span <= 1:
        return arr
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def weibull_survival_diagnostic(gaps: np.ndarray) -> dict[str, float]:
    """号码遗漏分布的威布尔生存分析诊断参数（仅诊断）。"""
    arr = np.asarray(gaps, dtype=np.float64)
    positive_gaps = arr[arr > 0]
    if len(positive_gaps) < 5:
        return {"shape_k": 1.0, "scale_lambda": float(np.mean(arr)) if len(arr) > 0 else 3.0}
    mean_val = float(np.mean(positive_gaps))
    var_val = float(np.var(positive_gaps))
    cv = math.sqrt(max(0.0, var_val)) / max(1e-9, mean_val)
    # k ≈ cv^(-1.086) 近似经验估计
    k_approx = max(0.2, min(5.0, cv ** (-1.086))) if cv > 0 else 1.0
    return {"shape_k": k_approx, "scale_lambda": mean_val}


def randomness_report(
    state: BinaryStateMatrix,
    end: int,
    windows: Mapping[str, WindowSpec],
) -> RandomnessReport:
    """生成综合随机性基准诊断报告。

    @param state 状态矩阵
    @param end 历史截断端点
    @param windows 窗口规范字典
    @return RandomnessReport
    """
    n_hist = end + 1
    z_map = z_all_windows(state, end, windows, p=P0)

    # 显著性号码数统计（任一窗口 |Z| > 1.96）
    sig_any = np.zeros(N_NUM, dtype=bool)
    for z_arr in z_map.values():
        sig_any |= np.abs(z_arr) > Z_SIG
    sig_count = int(np.count_nonzero(sig_any))

    # 卡方拟合优度
    chi2_res = chi_square_by_window(state, end, windows)

    # NIST-lite 4 项
    w_mat = state.window_matrix(end, None)
    flat_bits = flatten_history(w_mat)
    nist_res = nist_lite(flat_bits)

    # 香农分布熵
    obs_all = state.window_sum(end, None)
    ent_80 = freq_entropy(obs_all, n_hist)
    ent_max = math.log2(float(N_NUM))

    # 互信息 Top 对
    recent_mat = state.window_matrix(end, min(200, n_hist))
    mi_mat = mi_lag1(recent_mat)
    # 提取非对角 top 5 对
    mi_pairs: list[tuple[str, str, float]] = []
    for r in range(N_NUM):
        for c in range(r + 1, N_NUM):
            score = float(0.5 * (mi_mat[r, c] + mi_mat[c, r]))
            mi_pairs.append((f"{r + 1:02d}", f"{c + 1:02d}", score))
    mi_pairs.sort(key=lambda x: x[2], reverse=True)
    mi_top = tuple(mi_pairs[:10])

    # 游程检验摘要
    _z_runs, _r_runs, run_labels = runs_test_80(w_mat)
    run_counts: dict[str, int] = {}
    for lbl in run_labels:
        run_counts[lbl] = run_counts.get(lbl, 0) + 1

    notes = (
        f"样本期数: {n_hist} 期",
        f"显著冷热偏态号码数: {sig_count} / {N_NUM}",
        f"全量分布熵: {ent_80:.4f} bit (理论上限: {ent_max:.4f} bit)",
    )

    return RandomnessReport(
        n_history=n_hist,
        p0=P0,
        z_by_window=z_map,
        significant_count=sig_count,
        chi2=chi2_res,
        nist=nist_res,
        entropy_80=ent_80,
        entropy_max=ent_max,
        mi_top_pairs=mi_top,
        runs_summary=run_counts,
        notes=notes,
    )
