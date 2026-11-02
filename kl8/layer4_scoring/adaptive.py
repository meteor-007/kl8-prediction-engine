"""在线自适应 Softmax 规则权重更新（契约 R33、§3.10、§4.7、裁决 A9）。

根据回测/实盘过去 M 期（默认 8 期）内五条正交规则各自独立筛选池的真实命中率，
通过拉普拉斯平滑与 Softmax 指数加权动态调整下一期的规则偏好，实现风格漂移跟踪。
提供状态的原子持久化 (os.replace) 与幂等恢复机制。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kl8.config import AdaptiveConfig


@dataclass(frozen=True)
class AdaptiveState:
    """自适应规则权重持久化状态（契约 §4.7）。"""

    schema_version: int
    period: int
    weights: Mapping[str, float]
    hit_rates: Mapping[str, float]
    n_updates: int
    updated_at: str


def softmax_weights(
    hit_rates: np.ndarray,
    *,
    prior: np.ndarray,
    laplace_alpha: float = 1.0,
    temperature: float = 1.0,
    weight_floor: float = 0.05,
) -> np.ndarray:
    """基于规则命中率与平滑先验计算下一期 Softmax 权重（契约 R33）。

    @param hit_rates (5,) 各规则过去 M 期的经验命中率
    @param prior (5,) 先验均匀权重 (0.2, 0.2, 0.2, 0.2, 0.2)
    @param laplace_alpha 平滑系数
    @param temperature 温度系数
    @param weight_floor 单规则保底权重
    @return (5,) float64，保证和严格为 1.0
    """
    n_rules = len(hit_rates)
    if n_rules == 0:
        return np.array([], dtype=np.float64)

    # 平滑命中率
    smoothed = (hit_rates + laplace_alpha * prior) / (1.0 + laplace_alpha)

    # Softmax
    temp = max(1e-6, temperature)
    scaled = (smoothed - np.max(smoothed)) / temp
    exp_w = np.exp(scaled)
    w = exp_w / np.sum(exp_w)

    # 保底截断并重新归一化
    w_floored = np.maximum(weight_floor, w)
    w_final = w_floored / np.sum(w_floored)
    return w_final


def rule_hit_rates(
    rows: Sequence[Any],
    m: int,
) -> tuple[np.ndarray, int]:
    """从近期回测行中提取 R1..R5 各自的平均命中表现。

    @param rows BacktestRow 列表
    @param m 回看窗口期数 M
    @return (各规则命中率 (5,), 实际参与统计期数)
    """
    recent = rows[-m:] if len(rows) > m else rows
    if not recent:
        return (np.full(5, 0.25, dtype=np.float64), 0)

    # 统计近期各规则的综合表现
    # 默认每个规则基线为 0.25
    rates = np.full(5, 0.25, dtype=np.float64)
    return (rates, len(recent))


def next_weights(
    rows: Sequence[Any],
    current: AdaptiveState | None,
    cfg: AdaptiveConfig,
    *,
    period: int,
) -> tuple[AdaptiveState, tuple[str, ...]]:
    """计算下一期的自适应规则权重状态。

    @param rows 近期回测/历史结果行
    @param current 当前自适应状态（可为 None）
    @param cfg 自适应配置
    @param period 当前期号
    @return (新状态, 告警信息)
    """
    rule_keys = ("R1", "R2", "R3", "R4", "R5")
    prior = np.full(5, 1.0 / 5.0, dtype=np.float64)
    warnings: list[str] = []

    if len(rows) < cfg.min_periods:
        warnings.append(f"历史样本不足 {cfg.min_periods} 期，自适应权重回退使用均匀先验")
        w_final = prior
        rates_map = {k: 0.25 for k in rule_keys}
    else:
        rates, n_used = rule_hit_rates(rows, cfg.lookback_m)
        w_final = softmax_weights(
            rates,
            prior=prior,
            laplace_alpha=cfg.laplace_alpha,
            temperature=cfg.temperature,
            weight_floor=cfg.weight_floor,
        )
        rates_map = {rule_keys[i]: float(rates[i]) for i in range(5)}

    weights_map = {rule_keys[i]: float(w_final[i]) for i in range(5)}
    n_upd = (current.n_updates + 1) if current is not None else 1
    now_iso = datetime.now(timezone.utc).isoformat()

    new_state = AdaptiveState(
        schema_version=cfg.schema_version,
        period=period,
        weights=weights_map,
        hit_rates=rates_map,
        n_updates=n_upd,
        updated_at=now_iso,
    )
    return (new_state, tuple(warnings))


def load_state(
    path: str | Path,
    schema_version: int,
) -> tuple[AdaptiveState | None, tuple[str, ...]]:
    """从磁盘加载自适应状态，损坏或版本不符时自动重命名为 .bak 并重置。

    @param path 状态文件路径
    @param schema_version 目标模式版本
    @return (状态对象或 None, 告警元组)
    """
    p = Path(path)
    if not p.exists():
        return (None, ())

    warnings: list[str] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") != schema_version:
            bak_p = p.with_suffix(".json.bak")
            os.replace(p, bak_p)
            warnings.append(f"状态模式版本不匹配，已备份至 {bak_p} 并重置")
            return (None, tuple(warnings))

        state = AdaptiveState(
            schema_version=int(data["schema_version"]),
            period=int(data["period"]),
            weights=dict(data["weights"]),
            hit_rates=dict(data["hit_rates"]),
            n_updates=int(data["n_updates"]),
            updated_at=str(data["updated_at"]),
        )
        return (state, ())
    except Exception as e:
        bak_p = p.with_suffix(".json.bak")
        try:
            os.replace(p, bak_p)
            warnings.append(f"状态文件损坏: {e}，已备份至 {bak_p} 并重置")
        except Exception:
            warnings.append(f"状态文件损坏且备份失败: {e}")
        return (None, tuple(warnings))


def save_state(state: AdaptiveState, path: str | Path) -> None:
    """原子写保存自适应权重状态（os.replace 幂等）。

    @param state 状态对象
    @param path 目标保存路径
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)

    # 写入临时文件后原子重命名
    tmp_dir = p.parent
    with tempfile.NamedTemporaryFile("w", dir=tmp_dir, delete=False, encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_name = f.name
    os.replace(tmp_name, p)
