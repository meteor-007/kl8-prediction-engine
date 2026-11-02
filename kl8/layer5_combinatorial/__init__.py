"""快乐8 运筹学组合覆盖设计与注单生成层（Layer 5 Combinatorial）。

唯一注单生成入口 build_tickets：
- 针对优选复式池执行旋转矩阵覆盖设计缩水
- 生成选五（出5保4）与选七（出6保5）高性价比实战注单
- 输出包含穷举保底验证与成本预算的 TicketPlan
"""

from __future__ import annotations

import numpy as np

from kl8.config import AppConfig
from kl8.datatypes import PoolResult, TicketPlan
from kl8.layer5_combinatorial.covering import (
    GuaranteeResult,
    coverage_ratio,
    lookup_table,
    m_target,
    solve,
    verify_guarantee,
)
from kl8.layer5_combinatorial.tickets import build_ticket_plan, postprocess


def build_tickets(
    pool_result: PoolResult,
    scores: np.ndarray,
    cfg: AppConfig,
) -> tuple[TicketPlan, ...]:
    """唯一注单计划构建中心（契约 §3.11）。

    默认生成选五主计划，若支持选七则配套生成选七计划。
    @param pool_result 选拔出的号码池（包含 main 复式池）
    @param scores (80,) FinalScore
    @param cfg 全局配置
    @return (TicketPlan, ...)
    """
    plans: list[TicketPlan] = []
    main_pool = pool_result.main

    # 1. 主玩法注单（默认选五 pick5）
    primary_play = cfg.covering.play
    plan_primary = build_ticket_plan(main_pool, scores, cfg, play=primary_play)
    plans.append(plan_primary)

    # 2. 若主玩法不是 pick7，且配置中存在 pick7 预算，则配套生成选七注单
    if primary_play != "pick7" and "pick7" in cfg.covering.budget_by_play:
        plan_pick7 = build_ticket_plan(main_pool, scores, cfg, play="pick7")
        plans.append(plan_pick7)

    return tuple(plans)


__all__ = [
    "GuaranteeResult",
    "build_ticket_plan",
    "build_tickets",
    "coverage_ratio",
    "lookup_table",
    "m_target",
    "postprocess",
    "solve",
    "verify_guarantee",
]
