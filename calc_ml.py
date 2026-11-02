# -*- coding: utf-8 -*-
"""快乐8新一代预测系统 —— GBDT/LightGBM 高维时序特征非线性预测引擎

【业务背景与架构定位】:
本模块作为系统「机器学习非线性排序引擎」，彻底打破原有单一线性微积分加权的局限。
提取号码过去 3/5/10/20/40 期的滑动窗口出号率、一阶差分速度、二阶差分加速度、
遗漏偏离度、连开状态、同区分区密度等多维时序特征，
利用正则化轻量级 LightGBM 决策树拟合样本历史，输出每个号码下期开出的置信概率 P(Y=1|X)。

@author czw
@since 2026-10-30
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np

from calc_core import N_NUM, PRIMES_80, TimeSeries

logger = logging.getLogger(__name__)

# 默认特征提取参数常量
DEFAULT_TRAIN_WINDOW: int = 160  # 训练回溯期数 (兼顾样本量与模型时效性)
DEFAULT_N_ESTIMATORS: int = 40   # 决策树弱学习器数量 (轻量防过拟合)
DEFAULT_LEARNING_RATE: float = 0.03  # 学习率
DEFAULT_NUM_LEAVES: int = 12     # 最大叶子节点数 (严格限制树复杂度)
DEFAULT_MIN_CHILD_SAMPLES: int = 25  # 叶子节点最小样本数 (防针对极少样本过度拟合)


def extract_features_for_draw(ts: TimeSeries, t: int) -> np.ndarray:
    """提取第 t 期预测所需的 80 个号码特征矩阵 (形状: 80 × 16)。

    // 步骤 1: 提取各尺度滑动窗口历史开出率矩阵
    // 步骤 2: 提取遗漏期数与连开重叠指标
    // 步骤 3: 提取 012 路、质合、奇偶、大小属性与分区微环境密度
    // 步骤 4: 构建并返回标准化 2D 特征数组

    @param ts 时序数据集缓存对象
    @param t 目标截断时间步 (预测第 t 期，仅能观测到 t-1 期以前的数据)
    @return 形状为 (80, 16) 的特征浮点数组
    """
    feature_matrix = np.zeros((N_NUM, 16), dtype=float)
    if t <= 0:
        return feature_matrix

    past_hit = ts.hit[:t]
    gaps = ts.gap[t - 1]

    # 多尺度滑动窗口均值
    f_3 = past_hit[-3:].mean(axis=0) if t >= 3 else past_hit.mean(axis=0)
    f_5 = past_hit[-5:].mean(axis=0) if t >= 5 else f_3
    f_10 = past_hit[-10:].mean(axis=0) if t >= 10 else f_5
    f_20 = past_hit[-20:].mean(axis=0) if t >= 20 else f_10
    f_40 = past_hit[-40:].mean(axis=0) if t >= 40 else f_20

    # 循环提取 80 个号码特征
    for num_idx in range(N_NUM):
        actual_num = num_idx + 1

        # 1. 遗漏与时序动量特征
        feature_matrix[num_idx, 0] = float(gaps[num_idx])  # 当前遗漏期数
        feature_matrix[num_idx, 1] = float(f_3[num_idx])   # 近3期开出率
        feature_matrix[num_idx, 2] = float(f_5[num_idx])   # 近5期开出率
        feature_matrix[num_idx, 3] = float(f_10[num_idx])  # 近10期开出率
        feature_matrix[num_idx, 4] = float(f_20[num_idx])  # 近20期开出率
        feature_matrix[num_idx, 5] = float(f_40[num_idx])  # 近40期开出率

        # 2. 动量速度与二阶加速度差分
        vel = f_3[num_idx] - f_20[num_idx]
        acc = (f_3[num_idx] - f_10[num_idx]) - (f_10[num_idx] - f_40[num_idx])
        feature_matrix[num_idx, 6] = float(vel)
        feature_matrix[num_idx, 7] = float(acc)

        # 3. 连号与状态特征
        feature_matrix[num_idx, 8] = float(past_hit[-1, num_idx]) if t >= 1 else 0.0  # 上期是否命中

        # 4. 固有数学属性 (012路、质合、奇偶、大小、所属分区)
        feature_matrix[num_idx, 9] = float(actual_num % 3)  # 012路类别
        feature_matrix[num_idx, 10] = 1.0 if actual_num in PRIMES_80 else 0.0  # 质数标志
        feature_matrix[num_idx, 11] = 1.0 if actual_num % 2 == 1 else 0.0     # 奇数标志
        feature_matrix[num_idx, 12] = 1.0 if actual_num >= 41 else 0.0        # 大号标志
        feature_matrix[num_idx, 13] = float(num_idx // 10)                     # 物理分区编号 (0~7)

        # 5. 分区微环境密度 (捕捉同分区群体爆发态势)
        zone_start = (num_idx // 10) * 10
        feature_matrix[num_idx, 14] = float(f_3[zone_start : zone_start + 10].sum())   # 同区近3期出球总数
        feature_matrix[num_idx, 15] = float(f_10[zone_start : zone_start + 10].sum())  # 同区近10期出球总数

    return feature_matrix


def train_and_predict_lgb(
    ts: TimeSeries,
    current_t: int | None = None,
    train_window: int = DEFAULT_TRAIN_WINDOW,
) -> np.ndarray:
    """基于滚动时间窗口执行样本构建、LightGBM 弱分类器训练并输出 80 码预测概率。

    // 步骤 1: 边界防御与有效训练区间确定 (严格防前瞻泄漏)
    // 步骤 2: 滑动构建滚动训练集 X_train, y_train
    // 步骤 3: 提取待预测期特征 X_predict
    // 步骤 4: 训练轻量级 LightGBM 模型并预测开出概率
    // 步骤 5: 降级容错防护 (当环境无 LightGBM 或样本过少时退化为指数滑动平均)

    @param ts 时序数据集缓存对象
    @param current_t 当前目标截断期 (默认使用全部历史 T)
    @param train_window 训练回溯期数
    @return 形状为 (80,) 的号码开出概率数组，元素取值在 [0.0, 1.0] 之间
    """
    total_t = current_t if current_t is not None else ts.T
    if total_t < 40:
        # 样本过少时返回均匀概率兜底
        return np.full(N_NUM, 0.25, dtype=float)

    # 步骤 1: 确定训练起止范围
    start_t = max(40, total_t - train_window)
    train_x_list: list[np.ndarray] = []
    train_y_list: list[int] = []

    # 步骤 2: 提取滑动特征与标签
    for step_t in range(start_t, total_t):
        feat = extract_features_for_draw(ts, step_t)
        label = ts.hit[step_t]  # (80,)
        train_x_list.append(feat)
        train_y_list.extend(int(v) for v in label)

    # 步骤 3: 构建 NumPy 训练数组与待预测特征
    x_train = np.vstack(train_x_list)  # (N_samples * 80, 16)
    y_train = np.array(train_y_list, dtype=int)
    x_predict = extract_features_for_draw(ts, total_t)  # (80, 16)

    # 步骤 4: 模型拟合与预测
    try:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            learning_rate=DEFAULT_LEARNING_RATE,
            num_leaves=DEFAULT_NUM_LEAVES,
            min_child_samples=DEFAULT_MIN_CHILD_SAMPLES,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(x_train, y_train)
        pred_probs = clf.predict_proba(x_predict)[:, 1]
        return np.asarray(pred_probs, dtype=float)

    except Exception as exc:
        logger.warning("LightGBM 预测异常或环境不可用，启动指数动量平滑降级: %s", exc)
        # 步骤 5: 优雅降级算法：基于短中长窗口平滑加权得分
        past_hit = ts.hit[:total_t]
        short_rate = past_hit[-5:].mean(axis=0) if total_t >= 5 else past_hit.mean(axis=0)
        mid_rate = past_hit[-15:].mean(axis=0) if total_t >= 15 else short_rate
        heuristic_prob = 0.60 * short_rate + 0.40 * mid_rate
        # 归一化至均值 0.25 附近
        scale = 0.25 / (heuristic_prob.mean() + 1e-6)
        return np.clip(heuristic_prob * scale, 0.05, 0.60)
