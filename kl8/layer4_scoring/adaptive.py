# -*- coding: utf-8 -*-
"""在线自适应 Softmax 规则权重更新（契约 R33、§3.10、§4.7、裁决 A9、B5）。

【真实算法（P8-H1 修复）】
1. ``rule_hit_rates``：从近 M 期 ``BacktestRow.rule_stats`` 里**真算**各规则的筛选池
   命中率 —— 命中率 = Σ(该规则命中码被开出的个数) / Σ(该规则命中码个数)，基线 ``P0``；
2. 样本数 < ``cfg.min_periods`` → 回退**均匀先验**（不编造命中率，并写中文告警）；
3. ``softmax_weights``：Laplace 平滑（α 来自配置）→ Softmax（温度来自配置）→
   单规则保底截断 → 重新归一化，保证 **Σw = 1**（契约 B5）；
4. 状态持久化：原子写 ``os.replace``，损坏/旧版本自动 ``.bak`` 重建且不外抛。

【为什么不能用"假常量"】旧实现无论入参如何都返回 0.25，导致自适应权重恒为 0.2，
8.md L147–148「回看 M=8 期各规则实际命中率 + Softmax 动态调权」完全落空 —— 这是
P8 体检卡 H1 的核心缺陷，本模块现在由 ``tests/test_adaptive.py`` 用"合成命中率"
反向验证权重确实随命中率变化。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from kl8.config import AdaptiveConfig
from kl8.constants import P0

#: 五条正交规则的固定编号顺序（报告、投票、权重共用）。
RULE_IDS: Final[tuple[str, ...]] = ("R1", "R2", "R3", "R4", "R5")

#: 状态文件损坏时的可捕获异常（禁止裸 except，见 P8-M7）。
_STATE_ERRORS: Final[tuple[type[BaseException], ...]] = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    json.JSONDecodeError,
)


@dataclass(frozen=True)
class AdaptiveState:
    """自适应规则权重持久化状态（契约 §4.7）。"""

    schema_version: int
    """状态 schema 版本（版本不符则 .bak 重建）。"""

    period: int
    """该权重对应的期号（用于血缘追踪）。"""

    weights: Mapping[str, float]
    """规则权重 R1..R5（Σ=1）。"""

    hit_rates: Mapping[str, float]
    """本次计算所依据的规则命中率（供审计复现）。"""

    n_updates: int
    """累计更新次数（幂等校验用）。"""

    updated_at: str
    """ISO8601 更新时间戳。"""


def softmax_weights(
    hit_rates: np.ndarray,
    *,
    prior: np.ndarray,
    laplace_alpha: float = 1.0,
    temperature: float = 1.0,
    weight_floor: float = 0.05,
) -> np.ndarray:
    """基于规则命中率与平滑先验计算下一期 Softmax 权重（契约 R33）。

    @param hit_rates (n,) 各规则过去 M 期的经验命中率
    @param prior (n,) 先验权重（样本不足时回均匀）
    @param laplace_alpha 拉普拉斯平滑系数
    @param temperature Softmax 温度（越大越平缓）
    @param weight_floor 单规则保底权重
    @return (n,) float64，和严格为 1.0；空输入返回空数组
    """
    rates = np.asarray(hit_rates, dtype=np.float64)
    n_rules = rates.size
    if n_rules == 0:
        return np.array([], dtype=np.float64)
    prior_arr = np.asarray(prior, dtype=np.float64)
    if prior_arr.size != n_rules:
        raise ValueError(f"先验维度 {prior_arr.size} 与命中率维度 {n_rules} 不一致")

    # 拉普拉斯平滑：把经验命中率向先验收缩，小样本不发散
    smoothed = (rates + laplace_alpha * prior_arr) / (1.0 + laplace_alpha)

    # Softmax（减去最大值防 exp 溢出）
    temp = max(1e-6, float(temperature))
    scaled = (smoothed - float(np.max(smoothed))) / temp
    exp_w = np.exp(scaled)
    total = float(np.sum(exp_w))
    weights = exp_w / total if total > 0.0 else np.full(n_rules, 1.0 / float(n_rules))

    # 保底截断并重新归一化，防止某规则被彻底饿死
    floored = np.maximum(float(weight_floor), weights)
    return floored / float(np.sum(floored))


def uniform_prior(n_rules: int) -> np.ndarray:
    """构造均匀先验权重（1/n）。

    @param n_rules 规则条数
    @return (n,) float64，和 = 1.0
    """
    count = max(0, int(n_rules))
    if count == 0:
        return np.array([], dtype=np.float64)
    return np.full(count, 1.0 / float(count), dtype=np.float64)


def default_rule_weights(cfg: AdaptiveConfig) -> dict[str, float]:
    """无可用自适应状态时的**兜底规则权重**（契约 B5：Σ=1，禁止硬编码字面量）。

    兜底走与正式路径完全相同的 Softmax 管线（均匀命中率 + 配置化平滑/温度/保底），
    因此结果恒为均匀 0.2；这正是「均匀权重时 FinalScore 退化为 Base+λ·V」的保证。

    @param cfg 自适应配置
    @return 规则号 → 权重（Σ=1）
    """
    prior = uniform_prior(len(RULE_IDS))
    weights = softmax_weights(
        prior,
        prior=prior,
        laplace_alpha=cfg.laplace_alpha,
        temperature=cfg.temperature,
        weight_floor=cfg.weight_floor,
    )
    return {rule_id: float(weights[i]) for i, rule_id in enumerate(RULE_IDS)}


def rule_hit_rates(
    rows: Sequence[Any],
    m: int,
) -> tuple[np.ndarray, int]:
    """从近期回测行中**真算** R1..R5 各自的筛选池命中率（契约 R33、P8-H1）。

    【口径】单规则命中率 = 该规则在近 M 期命中的号码中，最终被开出的总个数 ÷ 该
    规则命中号码总个数（即"规则的查准率"）；理论基线为 ``P0=0.25``。空池规则不参与
    统计（保持基线 P0），避免除零。

    @param rows BacktestRow 序列（按时间升序）
    @param m 回看窗口期数 M（<=0 视为全部）
    @return (各规则命中率 (5,), 实际参与统计的期数)
    """
    recent = list(rows[-m:]) if m is not None and m > 0 and len(rows) > m else list(rows)
    hits = np.zeros(len(RULE_IDS), dtype=np.float64)
    pools = np.zeros(len(RULE_IDS), dtype=np.float64)
    used = 0
    for row in recent:
        stats = getattr(row, "rule_stats", None)
        if not stats:
            continue
        used += 1
        for index, rule_id in enumerate(RULE_IDS):
            pair = stats.get(rule_id)
            if not pair:
                continue
            hit_count, pool_size = int(pair[0]), int(pair[1])
            if pool_size > 0:
                hits[index] += float(hit_count)
                pools[index] += float(pool_size)
    rates = np.where(pools > 0.0, hits / np.maximum(pools, 1.0), P0)
    return (np.asarray(rates, dtype=np.float64), used)


def next_weights(
    rows: Sequence[Any],
    current: AdaptiveState | None,
    cfg: AdaptiveConfig,
    *,
    period: int,
) -> tuple[AdaptiveState, tuple[str, ...]]:
    """计算下一期的自适应规则权重状态（契约 R33）。

    @param rows 近期回测/历史结果行（需带 ``rule_stats`` 才可能进入 Softmax 分支）
    @param current 当前自适应状态（可为 None）
    @param cfg 自适应配置
    @param period 当前期号
    @return (新状态, 中文告警元组)
    """
    prior = uniform_prior(len(RULE_IDS))
    warnings: list[str] = []

    rates, n_used = rule_hit_rates(rows, cfg.lookback_m)
    if n_used < cfg.min_periods:
        warnings.append(
            f"近期可用回测样本 {n_used} 期 < 最小样本 {cfg.min_periods} 期，自适应权重回退均匀先验"
        )
        weights = default_rule_weights(cfg)
        rates_map = {rule_id: P0 for rule_id in RULE_IDS}
    else:
        w_final = softmax_weights(
            rates,
            prior=prior,
            laplace_alpha=cfg.laplace_alpha,
            temperature=cfg.temperature,
            weight_floor=cfg.weight_floor,
        )
        weights = {rule_id: float(w_final[i]) for i, rule_id in enumerate(RULE_IDS)}
        rates_map = {rule_id: float(rates[i]) for i, rule_id in enumerate(RULE_IDS)}

    n_upd = (current.n_updates + 1) if current is not None else 1
    new_state = AdaptiveState(
        schema_version=cfg.schema_version,
        period=period,
        weights=weights,
        hit_rates=rates_map,
        n_updates=n_upd,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return (new_state, tuple(warnings))


def load_state(
    path: str | Path,
    schema_version: int,
) -> tuple[AdaptiveState | None, tuple[str, ...]]:
    """从磁盘加载自适应状态；损坏/旧版本自动重命名为 ``.bak`` 并重置（不外抛）。

    @param path 状态文件路径
    @param schema_version 目标 schema 版本
    @return (状态对象或 None, 中文告警元组)
    """
    file_path = Path(path)
    if not file_path.exists():
        return (None, ())

    warnings: list[str] = []
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if int(data.get("schema_version", -1)) != int(schema_version):
            os.replace(file_path, backup_path)
            warnings.append(f"自适应状态 schema 版本不匹配，已备份至 {backup_path.name} 并重置")
            return (None, tuple(warnings))
        state = AdaptiveState(
            schema_version=int(data["schema_version"]),
            period=int(data["period"]),
            weights={str(k): float(v) for k, v in dict(data["weights"]).items()},
            hit_rates={str(k): float(v) for k, v in dict(data["hit_rates"]).items()},
            n_updates=int(data["n_updates"]),
            updated_at=str(data["updated_at"]),
        )
        return (state, ())
    except _STATE_ERRORS as exc:
        try:
            os.replace(file_path, backup_path)
            warnings.append(f"自适应状态文件损坏（{exc}），已备份至 {backup_path.name} 并重置")
        except OSError as backup_exc:
            warnings.append(f"自适应状态文件损坏（{exc}）且备份失败（{backup_exc}），已按空状态继续")
        return (None, tuple(warnings))


def save_state(state: AdaptiveState, path: str | Path) -> None:
    """原子写保存自适应权重状态（同目录临时文件 → ``os.replace``，幂等）。

    @param state 状态对象
    @param path 目标保存路径
    @raises OSError 目录不可创建或原子替换失败（由调用方降级为告警）
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    # dir 必须与目标同盘，os.replace 才能保证原子性
    with tempfile.NamedTemporaryFile(
        "w", dir=file_path.parent, delete=False, encoding="utf-8", suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        tmp_name = handle.name
    os.replace(tmp_name, file_path)