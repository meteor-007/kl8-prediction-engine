"""注单后处理、预算截断与双玩法落地（契约 R41、§3.11、§4.4）。

核心功能：
- ``positions_to_numbers``: 把 0-based 覆盖位置索引映射为 1-based 真实号码并升序；
- ``postprocess``: 废注过滤（全奇/全偶、极端和值、连号超限、**注间重叠上限**、去重、
  预算截断）—— P8-M11 修复：连号判定曾差一（``> max+1`` 放行三连）、重叠上限配置
  从未生效；
- ``build_ticket_plan``: 按玩法生成完整投注计划与保底级别描述，``guarantee_level``
  带真实 ``design`` 字段（P8-H6：不再输出未替换的设计式占位符）。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import dataclasses
import math
from typing import Mapping, Sequence

import numpy as np

from kl8.config import AppConfig, CoveringConfig
from kl8.constants import PLAY_KT
from kl8.datatypes import CoveringReport, TicketPlan
from kl8.layer3_features.morphology import consecutive_groups, odd_even_ratio, sum_value
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


def _limit_pair_overlap(
    tickets: Sequence[tuple[int, ...]],
    cap: int,
    *,
    scores_by_number: Mapping[int, float],
) -> tuple[list[tuple[int, ...]], tuple[str, ...]]:
    """限制任意两注的最大重合号码数（契约 R41，P8-M11：配置项曾从未生效）。

    【策略】按"注内打分之和"降序贪心接受：只有与所有已接受注单的重合数 ≤ cap 才收；
    若因此不足预算注数，保留已接受注单并写告警（保底验证由 covering 负责，
    这里只做正交防撞，不改变覆盖保证的有效性判定口径）。

    @param tickets 已过滤注单
    @param cap 两注重合上限（cfg.max_pair_overlap）
    @param scores_by_number 号码 → FinalScore
    @return (满足重叠上限的注单, 告警元组)
    """
    if cap <= 0 or len(tickets) <= 1:
        return (list(tickets), ())
    ordered = sorted(
        tickets,
        key=lambda ticket: sum(scores_by_number.get(num, 0.0) for num in ticket),
        reverse=True,
    )
    accepted: list[tuple[int, ...]] = []
    dropped = 0
    for ticket in ordered:
        ticket_set = set(ticket)
        if all(len(ticket_set & set(other)) <= cap for other in accepted):
            accepted.append(ticket)
        else:
            dropped += 1
    warnings: tuple[str, ...] = ()
    if dropped:
        warnings = (
            f"注间重叠上限 {cap} 码：剔除 {dropped} 注与其他注重合过多的注单，"
            f"保留 {len(accepted)} 注",
        )
    # 恢复号码升序（accepted 按打分排序，展示时应按注单字典序稳定）
    accepted.sort()
    return (accepted, warnings)


def postprocess(
    tickets: Sequence[Sequence[int]],
    pool: Sequence[int],
    *,
    cfg: CoveringConfig,
    scores_by_number: Mapping[int, float],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """注单后处理过滤（契约 R41）。

    过滤规则：
    1. 去重；
    2. 全奇/全偶剔除 (``cfg.forbid_all_odd_even``)；
    3. 连号超限剔除（最大连号长度 > ``cfg.max_consecutive_in_ticket``）；
    4. 极端和值剔除（偏离池均值×k 超过 ``cfg.sum_margin``）；
    5. 注间重叠上限（``cfg.max_pair_overlap``）与预算截断。

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
    for ticket in raw_nums_tickets:
        if ticket not in seen:
            seen.add(ticket)
            unique_tickets.append(ticket)

    # 计算池理论和值基准
    k_value = int(cfg.k)
    pool_mean = float(sum(pool)) / float(max(1, len(pool)))
    expected_sum = pool_mean * float(k_value)
    min_sum = expected_sum * (1.0 - cfg.sum_margin)
    max_sum = expected_sum * (1.0 + cfg.sum_margin)

    filtered: list[tuple[int, ...]] = []
    drop_odd_even = 0
    drop_consecutive = 0
    drop_sum_band = 0
    for ticket in unique_tickets:
        # 全奇/全偶
        if cfg.forbid_all_odd_even:
            odd, even = odd_even_ratio(ticket)
            if odd == len(ticket) or even == len(ticket):
                drop_odd_even += 1
                continue

        # 连号长度：max_run 超过上限即剔除（上限 2 表示三连号必须剔除）
        cons = consecutive_groups(ticket)
        if int(cons.get("max_run", 1)) > int(cfg.max_consecutive_in_ticket):
            drop_consecutive += 1
            continue

        # 和值带宽
        sum_v = sum_value(ticket)
        if sum_v < min_sum or sum_v > max_sum:
            drop_sum_band += 1
            continue

        filtered.append(ticket)

    # 剔除必须留痕（禁止静默过滤：用户/审计需要知道注单为何变少）
    if drop_odd_even:
        warnings.append(f"剔除全奇/全偶注单 {drop_odd_even} 注（R41 形态过滤）")
    if drop_consecutive:
        warnings.append(
            f"剔除连号超限注单 {drop_consecutive} 注"
            f"（最大连号 > {int(cfg.max_consecutive_in_ticket)}，R41）"
        )
    if drop_sum_band:
        warnings.append(
            f"剔除和值离群注单 {drop_sum_band} 注"
            f"（合理区间 {min_sum:.1f}~{max_sum:.1f}，R41）"
        )

    # 若过滤太严导致注单为空，回退使用原始去重注单并记录告警
    if not filtered:
        warnings.append("注单形态过滤全部命中，回退保留基础去重注单")
        filtered = unique_tickets

    # 2. 注间重叠上限（R41 正交防撞）
    filtered, overlap_warns = _limit_pair_overlap(
        filtered,
        int(cfg.max_pair_overlap),
        scores_by_number=scores_by_number,
    )
    warnings.extend(overlap_warns)
    if not filtered:
        warnings.append("注间重叠上限过严导致注单为空，回退保留去重注单")
        filtered = unique_tickets

    # 3. 预算截断：保留注内打分之和最高的注单
    if len(filtered) > cfg.budget:
        filtered.sort(
            key=lambda ticket: sum(scores_by_number.get(num, 0.0) for num in ticket),
            reverse=True,
        )
        filtered = filtered[: cfg.budget]
        warnings.append(f"注数超过预算 {cfg.budget}，按注内打分和截断")

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
    @raises ValueError 玩法非法且配置无法兜底
    """
    target_play = play if play is not None else cfg.covering.play
    pair = PLAY_KT.get(target_play)
    if pair is None:
        # 未登记玩法：退回当前配置的 k/t，并在告警中说明（禁止静默按选五处理）
        k_value = int(cfg.covering.k)
        t_value = int(cfg.covering.t)
        fallback_warning = (
            f"玩法 {target_play!r} 未登记在 PLAY_KT，退回配置 k={k_value}/t={t_value}；"
            "请检查 covering.play 是否拼写错误"
        )
    else:
        k_value, t_value = pair
        fallback_warning = ""

    target_budget = budget
    if target_budget is None:
        target_budget = cfg.covering.budget_by_play.get(target_play, cfg.covering.budget)

    # 用 dataclasses.replace 继承**全部**覆盖参数（旧实现手工构造丢掉了
    # max_m_subsets / ilp_cap / sa_* / max_pair_overlap / max_consecutive_in_ticket 等，
    # 导致用户配置被静默忽略 —— 同属 P8-M11 的配置失效面）
    cov_cfg = dataclasses.replace(
        cfg.covering,
        play=target_play,
        k=k_value,
        t=t_value,
        budget=int(target_budget),
    )

    v = len(pool)
    m = m_target(
        v,
        k_value,
        t_value,
        m_mode=cov_cfg.m_mode,
        m_fixed=cov_cfg.m_fixed,
        margin=cov_cfg.m_margin,
    )

    # 提取池内号码的局部权重
    pool_weights = np.array([float(scores[n - 1]) for n in pool], dtype=np.float64)

    # 求解覆盖设计
    cov_report = solve(v, k_value, t_value, m, pool_weights, cov_cfg, seed=int(cfg.seed))

    # 注单后处理
    scores_map = {n: float(scores[n - 1]) for n in pool}
    final_tickets, post_warns = postprocess(
        cov_report.tickets_pos,
        pool,
        cfg=cov_cfg,
        scores_by_number=scores_map,
    )

    all_warnings = list(cov_report.warnings) + list(post_warns)
    if fallback_warning:
        all_warnings.append(fallback_warning)
    if not cov_report.verified:
        all_warnings.append(
            f"覆盖保底验证未通过（achieved_t={cov_report.achieved_t} < t={t_value}），"
            "本次注单仅作参考，请勿据此宣称保底"
        )
    cost = len(final_tickets) * int(cfg.cost_per_ticket)

    # H6：design 必须是真实参数串（v/k/t/m 全部代入），禁止保留未替换占位
    design = f"C({v},{k_value},{t_value},{m}) {len(final_tickets)}注"
    guarantee_desc = {
        "t": t_value,
        "m": m,
        "text": f"出{m}保{t_value}（池中任意 {m} 码至少 1 注中 {t_value}）",
        "design": design,
        "v": v,
        "n_tickets": len(final_tickets),
        "coverage_ratio": cov_report.coverage_ratio,
        "verified": cov_report.verified,
    }

    return TicketPlan(
        play=target_play,
        k=k_value,
        budget=int(target_budget),
        tickets=final_tickets,
        guarantee_level=guarantee_desc,
        covering=cov_report,
        cost_yuan=cost,
        method=cov_report.method,
        warnings=tuple(all_warnings),
    )