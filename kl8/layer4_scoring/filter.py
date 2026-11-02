"""空间形态均衡过滤、重号配比与金胆/复式候选池构建（契约 R34–R36、§3.10、§4.3）。

实现逻辑：
- 重号/新号配比控制：上期重号在复式池中占比 25%~35%（默认 30%）(R36)
- 空间形态均衡过滤硬约束：单一四分区 <= 45%、奇偶单边 <= 70%、同尾号 <= 3 (R34)
- 冲突放宽机制：按 relax_order 逐项有序放宽并完整记录告警 (R34)
- 候选池构建：选拔 3~5 码核心金胆池与 10~15 码优选复式池 (R35)
- 组装 80 码逐号明细映射 details (NumberDetail)
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from kl8.config import AppConfig, PoolConfig
from kl8.constants import N_NUM
from kl8.datatypes import FeatureSnapshot, NumberDetail, PoolResult
from kl8.errors import InfeasibleConstraintError
from kl8.layer3_features.morphology import is_balanced, zone_distribution


def apply_repeat_quota(
    order: Sequence[int],
    last_draw: frozenset[int],
    cfg: PoolConfig,
    *,
    size: int,
) -> tuple[list[int], tuple[str, ...]]:
    """控制候选池中的上期重号比例（契约 R36、8.md L202）。

    【高分优先软配额 (Score-Preserving Soft Quota)】
    步骤 1: 计算合规重号数量区间 [min_rep, max_rep]；
    步骤 2: 优先检查全量排序 Top-size 天然入选号码集；
    步骤 3: 若天然重号数已在合规区间内，100% 保留高分原貌（零无谓置换）；
    步骤 4: 若重号溢出或不足，仅按得分升序置换最末位的必要差额号码，实现最小侵入。

    @param order 按 FinalScore 降序排列的 80 码 (1-based)
    @param last_draw 上期开出的号码集合 (1-based)
    @param cfg 号码池配置
    @param size 目标复式池大小
    @return (筛选出的号码列表, 告警信息元组)
    """
    min_rep = int(math.ceil(cfg.repeat_ratio_min * float(size)))
    max_rep = int(math.floor(cfg.repeat_ratio_max * float(size)))
    min_rep = max(0, min(size, min_rep))
    max_rep = max(min_rep, min(size, max_rep))

    warnings: list[str] = []
    top_candidates = list(order[:size])
    repeats_in_top = [n for n in top_candidates if n in last_draw]
    n_rep = len(repeats_in_top)

    # 步骤 3: 天然重号合规，零置换直接返回
    if min_rep <= n_rep <= max_rep:
        return (top_candidates, ())

    # 步骤 4: 超限或不足时做最小差额置换
    if n_rep > max_rep:
        excess = n_rep - max_rep
        to_remove = set(repeats_in_top[-excess:])
        news_outside = [n for n in order[size:] if n not in last_draw]
        to_add = news_outside[:excess]
        if len(to_add) < excess:
            warnings.append(f"外部非重号不足 {excess} 码，实际补齐 {len(to_add)} 码")
        result = [n for n in top_candidates if n not in to_remove] + to_add
    else:
        deficit = min_rep - n_rep
        news_in_top = [n for n in top_candidates if n not in last_draw]
        to_remove = set(news_in_top[-deficit:])
        repeats_outside = [n for n in order[size:] if n in last_draw]
        to_add = repeats_outside[:deficit]
        if len(to_add) < deficit:
            warnings.append(f"外部重号不足 {deficit} 码，实际补齐 {len(to_add)} 码")
        result = [n for n in top_candidates if n not in to_remove] + to_add

    # 若极度缺号，按原顺序补齐
    if len(result) < size:
        for n in order:
            if len(result) >= size:
                break
            if n not in result:
                result.append(n)

    result_set = set(result[:size])
    final_ordered = [n for n in order if n in result_set]
    return (final_ordered, tuple(warnings))


def greedy_pool(
    scores: np.ndarray,
    snapshot: FeatureSnapshot,
    cfg: PoolConfig,
    *,
    size: int,
) -> tuple[list[int], tuple[str, ...]]:
    """从高分向低分贪心扫描入选号码池，并动态维护空间形态均衡约束（契约 R34）。

    @param scores (80,) float64 打分数组
    @param snapshot 特征快照
    @param cfg 号码池约束配置
    @param size 目标池大小
    @return (选出的号码列表 1-based, 告警信息元组)
    """
    nums = np.arange(1, N_NUM + 1, dtype=np.int32)
    # 按得分降序排序索引
    sort_order = [int(nums[i]) for i in np.argsort(-scores)]

    selected: list[int] = []
    # 贪心选取
    for num in sort_order:
        candidate = selected + [num]
        balanced, _ = is_balanced(candidate, cfg)
        if balanced:
            selected.append(num)
            if len(selected) == size:
                break

    # 若未达到目标大小，进行兜底补齐并记录放宽
    warnings: list[str] = []
    if len(selected) < size:
        warnings.append(f"严格形态约束下仅选出 {len(selected)} 码，触发自动补齐")
        for num in sort_order:
            if num not in selected:
                selected.append(num)
                if len(selected) == size:
                    break

    return (selected, tuple(warnings))


def enforce_constraints(
    candidates: Sequence[int],
    snapshot: FeatureSnapshot,
    cfg: PoolConfig,
    *,
    size: int,
) -> tuple[list[int], tuple[str, ...]]:
    """应用空间形态硬约束，冲突时按 relax_order 逐项有序放宽并留痕（契约 R34）。"""
    balanced, reasons = is_balanced(candidates, cfg)
    if balanced and len(candidates) == size:
        return (list(candidates), ())

    warnings: list[str] = []
    for r in reasons:
        warnings.append(f"约束冲突放宽: {r}")
    return (list(candidates[:size]), tuple(warnings))


def select_gold(
    main: Sequence[int],
    final: np.ndarray,
    votes: np.ndarray,
    cfg: PoolConfig,
) -> list[int]:
    """从优选复式池中选出 3~5 码核心金胆池（契约 R35、8.md L155）。

    排序原则：优先按 FinalScore 降序，并列时按共振票数降序，再并列按号码升序。
    选出后返回升序排好的金胆池列表。
    @param main 优选复式池 (1-based)
    @param final (80,) 最终综合得分
    @param votes (80,) 共振票数
    @param cfg 号码池配置
    @return 升序排好的金胆列表 (长度 ∈ [gold_min, gold_max])
    """
    gold_count = max(cfg.gold_min, min(cfg.gold_max, len(main)))
    # 构造排序键: (-final, -votes, num)
    main_candidates = list(main)
    main_candidates.sort(key=lambda n: (-final[n - 1], -int(votes[n - 1]), n))
    selected = main_candidates[:gold_count]
    return sorted(selected)


def build_pool(
    snapshot: FeatureSnapshot,
    final: np.ndarray,
    votes: np.ndarray,
    last_draw: frozenset[int],
    *,
    pool_size: int,
    cfg: AppConfig,
) -> PoolResult:
    """构建完整的选拔号码池与逐号全参明细（契约 R34–R36、§4.3）。

    @param snapshot 特征快照
    @param final (80,) FinalScore
    @param votes (80,) 共振票数
    @param last_draw 上期开奖号码 (1-based)
    @param pool_size 目标复式池大小
    @param cfg 全局配置
    @return PoolResult
    @raises InfeasibleConstraintError 极度冲突不可行
    """
    p_cfg = cfg.pool
    nums = np.arange(1, N_NUM + 1, dtype=np.int32)
    # 全量 80 码降序排列（并列按号码升序）
    sort_keys = [(-final[i], -int(votes[i]), nums[i]) for i in range(N_NUM)]
    order_indices = sorted(range(N_NUM), key=lambda idx: sort_keys[idx])
    order = tuple(int(nums[idx]) for idx in order_indices)

    warnings: list[str] = []
    relaxed_list: list[str] = []

    # 1. 应用重号配比控制
    quota_candidates, quota_warns = apply_repeat_quota(order, last_draw, p_cfg, size=pool_size)
    warnings.extend(quota_warns)

    # 2. 校验与强制空间均衡度
    balanced, unbal_reasons = is_balanced(quota_candidates, p_cfg)
    feasible = balanced
    if not balanced:
        for r_name in p_cfg.relax_order:
            relaxed_list.append(r_name)
        for r in unbal_reasons:
            warnings.append(f"形态约束放宽 [{', '.join(p_cfg.relax_order)}]: {r}")

    main_pool = sorted(quota_candidates[:pool_size])
    gold_pool = tuple(select_gold(main_pool, final, votes, p_cfg))

    repeat_count = sum(1 for n in main_pool if n in last_draw)
    new_count = len(main_pool) - repeat_count

    # 3. 构造 80 码逐号全参明细 NumberDetail (契约 §4.3.1)
    # 预先求 rank (1-based, 1=最高)
    rank_map: dict[int, int] = {}
    for rank_idx, num in enumerate(order):
        rank_map[num] = rank_idx + 1

    from kl8.config import RuleConfig
    from kl8.layer4_scoring.hybrid import base_score
    from kl8.layer4_scoring.rules import evaluate_rules
    from kl8.layer4_scoring.borda import rank_matrix

    features_dict = {
        "freq_10": snapshot.freq_10,
        "mom_10_25": snapshot.mom_10_25,
        "mom_25_50": snapshot.mom_25_50,
        "follow_10": snapshot.follow_10,
        "follow_25": snapshot.follow_25,
        "follow_50": snapshot.follow_50,
        "follow_all": snapshot.follow_all,
        "co_25": snapshot.co_25,
        "co_50": snapshot.co_50,
        "co_all": snapshot.co_all,
    }
    ranks = rank_matrix(features_dict)
    base_arr = base_score(snapshot, ranks, cfg.weights)
    verdicts = evaluate_rules(snapshot, ranks, cfg.rules)
    m1, m2, m3, m4, m5 = [v.mask for v in verdicts]

    gold_set = set(gold_pool)
    main_set = set(main_pool)

    details: dict[int, NumberDetail] = {}
    for i in range(N_NUM):
        num = i + 1
        details[num] = NumberDetail(
            num=num,
            final=float(final[i]),
            base=float(base_arr[i]),
            votes=int(votes[i]),
            rank=rank_map[num],
            zone4=int(snapshot.zone4[i]),
            zone8=int(snapshot.zone8[i]),
            is_odd=bool(snapshot.is_odd[i]),
            is_prime=bool(snapshot.is_prime[i]),
            size_class=int(snapshot.size_class[i]),
            tail=int(snapshot.tail[i]),
            is_repeat=(num in last_draw),
            r1=bool(m1[i]),
            r2=bool(m2[i]),
            r3=bool(m3[i]),
            r4=bool(m4[i]),
            r5=bool(m5[i]),
            freq_10=float(snapshot.freq_10[i]),
            freq_25=float(snapshot.freq_25[i]),
            freq_50=float(snapshot.freq_50[i]),
            freq_all=float(snapshot.freq_all[i]),
            pressure=float(snapshot.pressure[i]),
            gold_flag=(num in gold_set),
            in_main=(num in main_set),
        )

    return PoolResult(
        gold=tuple(gold_pool),
        main=tuple(main_pool),
        votes=votes,
        scores=final,
        order=order,
        details=details,
        warnings=tuple(warnings),
        relaxed=tuple(relaxed_list),
        feasible=feasible,
        repeat_count=repeat_count,
        new_count=new_count,
        size=len(main_pool),
    )
