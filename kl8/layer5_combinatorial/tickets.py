"""注单后处理、预算截断与双玩法落地（契约 R41、§3.11、§4.4）。

核心功能：
- positions_to_numbers: 将 0-based 覆盖位置索引映射为 1-based 真实号码并升序排序
- postprocess: 执行废注过滤（全奇/全偶、极端和值、过多连号、重叠上限、去重与预算截断）
- build_ticket_plan: 面向目标玩法（选五 / 选七）生成完整投注计划与保底级别描述
- plan_cost: 计算注单成本（整型元）
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from kl8.config import AppConfig, CoveringConfig
from kl8.datatypes import CoveringReport, TicketPlan
from kl8.layer3_features.morphology import ac_value, consecutive_groups, odd_even_ratio, sum_value
from kl8.layer5_combinatorial.covering import m_target, solve


def positions_to_numbers(
    tickets: Sequence[Sequence[int]],
    pool: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """将覆盖求解器输出的位置索引映射为 1-based 真实号码，每注内升序排列。

    @param tickets 位置索引注单（0..v-1）
    @param pool 1-based 候选号码池
    @return (1-based 注单, ...)
    """
    pool_arr = list(pool)
    res: list[tuple[int, ...]] = []
    for t_pos in tickets:
        nums = sorted(pool_arr[idx] for idx in t_pos if 0 <= idx < len(pool_arr))
        res.append(tuple(nums))
    return tuple(res)


def postprocess(
    tickets: Sequence[Sequence[int]],
    pool: Sequence[int],
    *,
    cfg: CoveringConfig,
    scores_by_number: Mapping[int, float],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """注单后处理过滤（契约 R41）。

    过滤规则：
    1. 去重
    2. 全奇/全偶剔除 (cfg.forbid_all_odd_even)
    3. 连号过多剔除 (连号长度 > cfg.max_consecutive_in_ticket)
    4. 极端和值剔除 (偏离池均值*k 超过 cfg.sum_margin)
    5. 重叠上限与预算截断
    @param tickets 位置或号码注单
    @param pool 候选号码池
    @param cfg 覆盖参数配置
    @param scores_by_number 号码打分映射
    @return (处理后的注单元组, 告警信息元组)
    """
    raw_nums_tickets = positions_to_numbers(tickets, pool)
    warnings: list[str] = []

    # 1. 唯一去重
    unique_tickets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for t in raw_nums_tickets:
        if t not in seen:
            seen.add(t)
            unique_tickets.append(t)

    # 计算池理论和值基准
    k = cfg.k
    pool_mean = float(sum(pool)) / float(max(1, len(pool)))
    expected_sum = pool_mean * float(k)
    min_sum = expected_sum * (1.0 - cfg.sum_margin)
    max_sum = expected_sum * (1.0 + cfg.sum_margin)

    filtered: list[tuple[int, ...]] = []
    for t in unique_tickets:
        # 全奇/全偶
        if cfg.forbid_all_odd_even:
            odd, even = odd_even_ratio(t)
            if odd == len(t) or even == len(t):
                continue

        # 连号长度
        cons = consecutive_groups(t)
        if cons.get("max_run", 1) > cfg.max_consecutive_in_ticket + 1:
            continue

        # 和值带宽
        sv = sum_value(t)
        if sv < min_sum or sv > max_sum:
            continue

        filtered.append(t)

    # 若过滤太严导致注单为空，回退使用原始去重注单并记录告警
    if not filtered:
        warnings.append("注单形态过滤全部命中，回退保留基础去重注单")
        filtered = unique_tickets

    # 预算截断
    if len(filtered) > cfg.budget:
        # 优先保留注单号码打分之和最高的注单
        filtered.sort(
            key=lambda b: sum(scores_by_number.get(num, 0.0) for num in b),
            reverse=True,
        )
        filtered = filtered[: cfg.budget]

    return (tuple(filtered), tuple(warnings))


def build_ticket_plan(
    pool: Sequence[int],
    scores: np.ndarray,
    cfg: AppConfig,
    *,
    play: str | None = None,
    budget: int | None = None,
) -> TicketPlan:
    """构建目标玩法（选五 / 选七）的完整注单计划（契约 R41、§3.11）。

    @param pool 优选复式池 (1-based)
    @param scores (80,) float64 FinalScore
    @param cfg 全局配置
    @param play 指定玩法内部键（"pick5" 或 "pick7"）
    @param budget 指定预算注数
    @return TicketPlan
    """
    target_play = play if play is not None else cfg.covering.play
    k = 5 if target_play == "pick5" else (7 if target_play == "pick7" else cfg.covering.k)
    t = 4 if target_play == "pick5" else (5 if target_play == "pick7" else cfg.covering.t)

    target_budget = budget
    if target_budget is None:
        target_budget = cfg.covering.budget_by_play.get(target_play, cfg.covering.budget)

    cov_cfg = CoveringConfig(
        play=target_play,
        k=k,
        t=t,
        m_mode=cfg.covering.m_mode,
        m_fixed=cfg.covering.m_fixed,
        m_margin=cfg.covering.m_margin,
        budget=target_budget,
        table_path=cfg.covering.table_path,
        forbid_all_odd_even=cfg.covering.forbid_all_odd_even,
        sum_margin=cfg.covering.sum_margin,
    )

    v = len(pool)
    m = m_target(v, k, t, m_mode=cov_cfg.m_mode, m_fixed=cov_cfg.m_fixed, margin=cov_cfg.m_margin)

    # 提取池内号码的局部权重
    pool_weights = np.array([float(scores[n - 1]) for n in pool], dtype=np.float64)

    # 求解覆盖设计
    cov_report = solve(v, k, t, m, pool_weights, cov_cfg)

    # 注单后处理
    scores_map = {n: float(scores[n - 1]) for n in pool}
    final_tickets, post_warns = postprocess(
        cov_report.tickets_pos,
        pool,
        cfg=cov_cfg,
        scores_by_number=scores_map,
    )

    all_warnings = list(cov_report.warnings) + list(post_warns)
    cost = len(final_tickets) * cfg.cost_per_ticket

    guarantee_desc = {
        "t": t,
        "m": m,
        "text": f"出{m}保{t}（池中任意 {m} 码至少 1 注中 {t}）",
        "coverage_ratio": cov_report.coverage_ratio,
        "verified": cov_report.verified,
    }

    return TicketPlan(
        play=target_play,
        k=k,
        budget=target_budget,
        tickets=final_tickets,
        guarantee_level=guarantee_desc,
        covering=cov_report,
        cost_yuan=cost,
        method=cov_report.method,
        warnings=tuple(all_warnings),
    )


def plan_cost(plan: TicketPlan) -> int:
    """计算注单计划投入资金（整数元）。"""
    return plan.cost_yuan
