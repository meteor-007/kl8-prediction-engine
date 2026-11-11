# -*- coding: utf-8 -*-
"""基于 PyTorch Transformer 的快乐8序列自注意力深度学习预测网络。

【业务背景与架构定位】
借鉴 GitHub 优秀开源项目 `KittenCN/predict_Lottery_ticket_pytorch` 的序列多标签建模思想，
将历史开奖序列编码为连续多期的 80 维 Multi-Hot 状态张量。
通过多头自注意力机制（Multi-Head Self-Attention）提取跨期开奖号之间的潜在时序转移与深层模式，
最终利用二进制交叉熵损失函数（BCEWithLogitsLoss）优化并输出下一期各号码的独立激活概率。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from kl8_engine.constants import N_NUM


class PositionalEncoding(nn.Module):
    """时序位置编码模块（为序列中的每一期注入相对时序距离）。"""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.05) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """叠加位置编码。

        @param x shape (batch_size, seq_len, d_model)
        @return shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class KL8TransformerNet(nn.Module):
    """快乐8多标签时序 Transformer 分类网络。

    【网络架构】
    输入层 -> 线性映射至隐藏层 (d_model) -> 时序位置编码 ->
    2 层 TransformerEncoder (4 头自注意力) -> 均值池化与末端特征提取 ->
    全连接分类头 -> 80 维对数几率 (Logits)。
    """

    def __init__(
        self,
        d_in: int = N_NUM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(d_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=100, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播计算。

        @param x shape (batch_size, seq_len, 80)
        @return shape (batch_size, 80) 未归一化对数几率 (Logits)
        """
        # 步骤 1: 投影至潜在特征空间
        feat = self.embedding(x)
        # 步骤 2: 叠加位置信息
        feat = self.pos_encoder(feat)
        # 步骤 3: 提取全序自注意力特征
        out = self.transformer_encoder(feat)
        # 步骤 4: 汇聚最后一期及整体时序上下文
        last_step = out[:, -1, :]
        mean_context = out.mean(dim=1)
        pooled = last_step + 0.5 * mean_context
        # 步骤 5: 输出预测对数几率
        logits = self.head(pooled)
        return logits


def train_and_predict_transformer(
    history_matrix: np.ndarray,
    seq_len: int = 10,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.003,
    seed: int = 20261104,
) -> Tuple[np.ndarray, float]:
    """训练 Transformer 模型并输出对下一期的 80 维预测概率分布。

    @param history_matrix shape (T, 80) 历史 Multi-Hot 状态矩阵
    @param seq_len 滑动窗口大小（如 10 期）
    @param epochs 训练轮数
    @param batch_size 批次大小
    @param lr 初始学习率
    @param seed 随机种子，确保完全可复现
    @return (80 维预测概率数组 shape (80,), 最终训练损失 float)
    """
    # 步骤 1: 固定随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    T = len(history_matrix)
    if T <= seq_len + 5:
        # 样本不足时安全降级为近期频率
        recent_freq = history_matrix[-seq_len:].mean(axis=0)
        return recent_freq, 0.0

    # 步骤 2: 构建 (X, y) 滑动窗口数据集
    # 截取最近 400 期加速训练并捕捉有效近景模式
    effective_matrix = history_matrix[-min(T, 450) :]
    eff_T = len(effective_matrix)

    X_list = []
    y_list = []
    for i in range(eff_T - seq_len):
        X_list.append(effective_matrix[i : i + seq_len])
        y_list.append(effective_matrix[i + seq_len])

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)

    X_tensor = torch.from_numpy(X_arr)
    y_tensor = torch.from_numpy(y_arr)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 步骤 3: 实例化网络与优化器
    model = KL8TransformerNet(d_in=N_NUM, d_model=96, nhead=4, num_layers=2, dim_feedforward=192, dropout=0.08)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # 步骤 4: 模型轻量高效微调
    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        scheduler.step()
        final_loss = epoch_loss / max(batches, 1)

    # 步骤 5: 执行下一期因果预测（仅使用截至最新期的输入窗口）
    model.eval()
    with torch.no_grad():
        latest_window = torch.from_numpy(history_matrix[-seq_len:]).unsqueeze(0).float()
        pred_logits = model(latest_window).squeeze(0)
        pred_probs = torch.sigmoid(pred_logits).numpy()

    return pred_probs, final_loss
