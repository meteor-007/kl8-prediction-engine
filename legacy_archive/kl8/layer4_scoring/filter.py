# -*- coding: utf-8 -*-
"""空间形态均衡过滤、重号配比与金胆/复式候选池构建（契约 R34–R36、§3.10、§4.3）。

【P8-H2 修复要点：硬约束必须"真执行"】
旧实现 `build_pool` 只在事后调用 `is_balanced` 写一条告警，既不换码也不重选，
还把 4 项约束无差别塞进 `relaxed` —— 上线交付件里出现"第 3 区占比 50% > 45%"的池子。
现在的实现：

1. **贪心扫描中即时执行**：按 FinalScore 降序逐码尝试，只有加入后仍满足全部
   **启用中**约束才入选 —— 约束由构造过程保证，不靠事后告警；
2. **B6 重号配比**：先算 `repeat_target = clip(round(repeat_ratio×size),
   ceil(repeat_ratio_min×size), floor(repeat_ratio_max×size))`，贪心中硬约束
   "重号数 ≤ target 且 新号数 ≤ size-target"；
3. **B7 有序放宽**：选不满 size 时按 `same_tail → parity → zone4 → repeat_ratio`
   逐项放宽，`PoolResult.relaxed` 只记录**真实放宽**的项，`warnings` 写放宽后的真实
   统计值；全部放宽仍不满 → `InfeasibleConstraintError`（禁止带病出单）；
4. **A5 金胆池**：`n_gold = clip(round(0.30×len(main)), gold_min, gold_max)`
   （`main=12 → 4`），先取 `V≥3` 按 FinalScore 降序，不足再补。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import math
from typing import Any, Final, Mapping, Sequence

import numpy as np

from kl8.config import AppConfig, PoolConfig
from kl8.constants import GOLD_RATIO, GOLD_VOTE_PRIORITY, N_NUM, P0, tail_of, zone4_of
from kl8.datatypes import FeatureSnapshot, NumberDetail, PoolResult, RuleVerdict
from kl8.errors import InfeasibleConstraintError
from kl8.layer3_features.morphology import is_balanced

#: 约束键（与 `cfg.pool.relax_order` 元素一一对应，B7 固定顺序）。
CONSTRAINT_KEYS: Final[tuple[str, ...]] = ("same_tail", "parity", "zone4", "repeat_ratio")


def _score_order(scores: np.ndarray) -> list[int]:
    """按打分降序、并列按号码升序生成 1-based 号码序列。

    @param scores (80,) FinalScore
    @return 1-based 号码列表（长度 80）
    """
    numbers = np.arange(1, N_NUM + 1, dtype=np.int32)
    order_idx = np.lexsort((numbers, -np.asarray(scores, dtype=np.float64)))
    return [int(numbers[i]) for i in order_idx]


def _caps(cfg: PoolConfig, size: int) -> tuple[int, int, int]:
    """把比例约束换算成整数个数上限（避免每步浮点比较）。

    @param cfg 号码池配置
    @param size 目标池大小
    @return (单四区上限, 奇偶单边上限, 同尾上限)
    """
    zone_cap = int(math.floor(cfg.max_zone4_ratio * float(size) + 1e-9))
    parity_cap = int(math.floor(cfg.max_parity_ratio * float(size) + 1e-9))
    return (max(1, zone_cap), max(1, parity_cap), cfg.max_same_tail)


def repeat_quota(size: int, cfg: PoolConfig) -> tuple[int, int, int]:
    """计算 B6 重号配比区间与目标值。

    @param size 复式池目标规模
    @param cfg 号码池配置
    @return (min_rep, max_rep, target_rep)
    """
    min_rep = max(0, min(size, int(math.ceil(cfg.repeat_ratio_min * float(size)))))
    max_rep = max(min_rep, min(size, int(math.floor(cfg.repeat_ratio_max * float(size)))))
    target = int(math.floor(cfg.repeat_ratio * float(size) + 0.5))
    target = max(min_rep, min(max_rep, target))
    return (min_rep, max_rep, target)


def apply_repeat_quota(
    order: Sequence[int],
    last_draw: frozenset[int],
    cfg: PoolConfig,
    *,
    size: int,
) -> tuple[list[int], tuple[str, ...]]:
    """控制候选池中的上期重号比例（契约 R36 / B6）。

    【最小侵入置换】只调整"最末位必要差额"的号码，尽量保留高分原貌；天然重号数
    已在 `[min_rep, max_rep]` 区间时零置换直接返回。

    @param order 按 FinalScore 降序排列的 80 码（1-based）
    @param last_draw 上期开出的号码集合（1-based）
    @param cfg 号码池配置
    @param size 目标复式池大小
    @return (调整后的号码列表, 中文告警元组)
    """
    min_rep, max_rep, _target = repeat_quota(size, cfg)
    warnings: list[str] = []
    top = list(order[:size])
    repeats_in_top = [num for num in top if num in last_draw]
    n_rep = len(repeats_in_top)

    if min_rep <= n_rep <= max_rep:
        return (top, ())

    if n_rep > max_rep:
        excess = n_rep - max_rep
        remove = set(repeats_in_top[-excess:])
        add = [num for num in order[size:] if num not in last_draw][:excess]
        result = [num for num in top if num not in remove] + add
        if len(add) < excess:
            warnings.append(f"池外非重号不足 {excess} 码，实际补充 {len(add)} 码")
    else:
        deficit = min_rep - n_rep
        remove = set([num for num in top if num not in last_draw][-deficit:])
        add = [num for num in order[size:] if num in last_draw][:deficit]
        result = [num for num in top if num not in remove] + add
        if len(add) < deficit:
            warnings.append(f"池外重号不足 {deficit} 码，实际补充 {len(add)} 码")

    for num in order:
        if len(result) >= size:
            break
        if num not in result:
            result.append(num)

    selected = set(result[:size])
    return ([num for num in order if num in selected], tuple(warnings))


def greedy_pool(
    scores: np.ndarray,
    snapshot: FeatureSnapshot,
    cfg: PoolConfig,
    *,
    size: int,
    active: frozenset[str] | None = None,
    repeat_target: int | None = None,
    last_draw: frozenset[int] | None = None,
) -> tuple[list[int], tuple[str, ...]]:
    """按打分降序贪心选池，**在扫描过程中真实执行**启用的硬约束（契约 R34）。

    @param scores (80,) FinalScore
    @param snapshot 特征快照（保留参数以对齐契约签名，当前实现只需静态属性表）
    @param cfg 号码池约束配置
    @param size 目标池大小
    @param active 启用中的约束键集合；None 表示全部启用
    @param repeat_target 重号目标个数；None 表示不约束重号比
    @param last_draw 上期开奖号码集合（1-based）
    @return (选出的号码列表 1-based 升序, 中文告警元组)；**凑不满时如实返回不足码数**，
        由上层 ``build_pool`` 按 B7 逐项放宽后重跑，禁止在贪心内部无约束补齐（P10b-M16：
        补齐会塞入"因违约被跳过"的号码，使放宽分支不可达且最终复核直接抛错）。
    """
    active_keys = frozenset(cfg.relax_order) if active is None else frozenset(active)
    repeats = frozenset(last_draw or ())
    zone_cap, parity_cap, tail_cap = _caps(cfg, size)
    target_size = int(size)
    if target_size <= 0:
        return ([], ("目标池大小为 0，无需选池",))

    warnings: list[str] = []
    if "repeat_ratio" in active_keys and repeat_target is None:
        warnings.append("缺少上期开奖/重号目标，本次选池未执行重号配比约束（仅按形态与得分选池）")

    selected: list[int] = []
    zone_count = [0, 0, 0, 0]
    tail_count: dict[int, int] = {}
    odd_count = 0
    repeat_count = 0

    for num in _score_order(scores):
        if len(selected) >= target_size:
            break
        zone = zone4_of(num)
        tail = tail_of(num)
        is_repeat = num in repeats
        taken = len(selected)

        if "zone4" in active_keys and zone_count[zone] + 1 > zone_cap:
            continue
        if "parity" in active_keys:
            next_odd = odd_count + (1 if num % 2 == 1 else 0)
            next_even = (taken + 1) - next_odd
            if max(next_odd, next_even) > parity_cap:
                continue
        if "same_tail" in active_keys and tail_count.get(tail, 0) + 1 > tail_cap:
            continue
        if "repeat_ratio" in active_keys and repeat_target is not None:
            if is_repeat and repeat_count + 1 > repeat_target:
                continue
            if (not is_repeat) and (taken + 1 - repeat_count) > target_size - repeat_target:
                continue

        selected.append(num)
        zone_count[zone] += 1
        tail_count[tail] = tail_count.get(tail, 0) + 1
        odd_count += 1 if num % 2 == 1 else 0
        repeat_count += 1 if is_repeat else 0

    # 凑不满时**如实返回不足码数**（P10b-M16）：不在此处无约束补齐，交由 build_pool
    # 按 B7 的 relax_order 逐项放宽并留痕；补齐会引入违约号码、使放宽分支不可达。
    return (sorted(selected), tuple(warnings))


def enforce_constraints(
    candidates: Sequence[int],
    snapshot: FeatureSnapshot,
    cfg: PoolConfig,
    *,
    size: int,
    active: frozenset[str] | None = None,
) -> tuple[list[int], tuple[str, ...]]:
    """对已选号码池做硬约束**复核**，返回违反原因（契约 R34）。

    @param candidates 候选号码池（1-based）
    @param snapshot 特征快照（对齐契约签名）
    @param cfg 号码池约束配置
    @param size 目标池大小
    @param active 启用中的约束键集合；None 表示全部启用
    @return (截断到 size 的号码池, 违反原因元组)
    """
    active_keys = frozenset(cfg.relax_order) if active is None else frozenset(active)
    pool = sorted(int(num) for num in candidates)[: int(size)]
    if not pool:
        return (pool, ("候选池为空",))

    zone_cap, parity_cap, tail_cap = _caps(cfg, int(size))
    reasons: list[str] = []
    if "zone4" in active_keys:
        zone_count = [0, 0, 0, 0]
        for num in pool:
            zone_count[zone4_of(num)] += 1
        worst = max(zone_count)
        if worst > zone_cap:
            reasons.append(f"单四区最多 {worst} 码 > 上限 {zone_cap} 码")
    if "parity" in active_keys:
        odd = sum(1 for num in pool if num % 2 == 1)
        if max(odd, len(pool) - odd) > parity_cap:
            reasons.append(f"奇偶单边最多 {max(odd, len(pool) - odd)} 码 > 上限 {parity_cap} 码")
    if "same_tail" in active_keys:
        tail_count: dict[int, int] = {}
        for num in pool:
            tail_count[tail_of(num)] = tail_count.get(tail_of(num), 0) + 1
        worst_tail = max(tail_count.items(), key=lambda item: item[1])
        if worst_tail[1] > tail_cap:
            reasons.append(f"尾数 {worst_tail[0]} 出现 {worst_tail[1]} 次 > 上限 {tail_cap} 次")
    return (pool, tuple(reasons))


def select_gold(
    main: Sequence[int],
    final: np.ndarray,
    votes: np.ndarray,
    cfg: PoolConfig,
) -> list[int]:
    """按契约 A5 选取金胆池（P8-M2：不再恒取 5 码）。

    ``n_gold = clip(round(0.30 × len(main)), gold_min, gold_max)``；
    选取顺序：先取 ``V ≥ 3`` 的号码按 FinalScore 降序，不足再用 FinalScore 降序补齐；
    结果恒为 ``main`` 的子集（硬约束 ``gold ⊂ main``）。

    @param main 复式池（1-based 升序）
    @param final (80,) FinalScore
    @param votes (80,) 共振票数
    @param cfg 号码池配置
    @return 金胆池（1-based 升序）
    """
    pool = list(main)
    if not pool:
        return []
    raw_count = int(math.floor(GOLD_RATIO * float(len(pool)) + 0.5))
    n_gold = max(cfg.gold_min, min(cfg.gold_max, raw_count))
    n_gold = min(n_gold, len(pool))

    def sort_key(num: int) -> tuple[float, int]:
        return (-float(final[num - 1]), int(num))

    high_votes = [num for num in pool if int(votes[num - 1]) >= GOLD_VOTE_PRIORITY]
    high_votes.sort(key=sort_key)
    chosen = high_votes[:n_gold]
    if len(chosen) < n_gold:
        rest = [num for num in pool if num not in chosen]
        rest.sort(key=sort_key)
        chosen.extend(rest[: n_gold - len(chosen)])
    return sorted(chosen)


def _relax_detail(
    key: str,
    selected: Sequence[int],
    size: int,
    cfg: PoolConfig,
    last_draw: frozenset[int],
) -> str:
    """生成"放宽到多少"的中文留痕（B7 要求可读）。

    @param key 被放宽的约束键
    @param selected 最终号码池
    @param size 目标池大小
    @param cfg 号码池配置
    @param last_draw 上期开奖号码集合
    @return 中文说明
    """
    pool = list(selected)
    if key == "same_tail":
        tail_count: dict[int, int] = {}
        for num in pool:
            tail_count[tail_of(num)] = tail_count.get(tail_of(num), 0) + 1
        worst = max(tail_count.items(), key=lambda item: item[1])
        return f"放宽 [same_tail]：尾数 {worst[0]} 达 {worst[1]} 个（原上限 {cfg.max_same_tail}）"
    if key == "parity":
        odd = sum(1 for num in pool if num % 2 == 1)
        return (
            f"放宽 [parity]：奇数 {odd} / 偶数 {len(pool) - odd}，单边占比 "
            f"{max(odd, len(pool) - odd) / max(1, len(pool)):.1%}（原上限 {cfg.max_parity_ratio:.1%}）"
        )
    if key == "zone4":
        zone_count = [0, 0, 0, 0]
        for num in pool:
            zone_count[zone4_of(num)] += 1
        return (
            f"放宽 [zone4]：单四区最多 {max(zone_count)} / {len(pool)} 码，占比 "
            f"{max(zone_count) / max(1, len(pool)):.1%}（原上限 {cfg.max_zone4_ratio:.1%}）"
        )
    if key == "repeat_ratio":
        rep = sum(1 for num in pool if num in last_draw)
        return (
            f"放宽 [repeat_ratio]：重号 {rep} / 新号 {len(pool) - rep}，重号占比 "
            f"{rep / max(1, len(pool)):.1%}（目标区间 "
            f"[{cfg.repeat_ratio_min:.1%}, {cfg.repeat_ratio_max:.1%}]）"
        )
    return f"放宽 [{key}]（未知约束键）"


def _build_details(
    snapshot: FeatureSnapshot,
    final: np.ndarray,
    votes: np.ndarray,
    base: np.ndarray,
    verdicts: Sequence[RuleVerdict] | None,
    last_draw: frozenset[int],
    gold: Sequence[int],
    main: Sequence[int],
) -> dict[int, NumberDetail]:
    """构造 80 码逐号明细（报告唯一数据源，契约 §4.3.1）。

    @param snapshot 特征快照
    @param final (80,) FinalScore
    @param votes (80,) 共振票数
    @param base (80,) BaseScore
    @param verdicts 五规则判定（可为 None，则 r1..r5 全 False）
    @param last_draw 上期开奖号码
    @param gold 金胆池
    @param main 复式池
    @return 号码 → NumberDetail
    """
    masks = {verdict.rule_id: np.asarray(verdict.mask, dtype=bool) for verdict in (verdicts or ())}
    rank_of: dict[int, int] = {}
    order_idx = np.lexsort((np.arange(1, N_NUM + 1), -np.asarray(final, dtype=np.float64)))
    for rank, index in enumerate(order_idx, start=1):
        rank_of[int(index) + 1] = rank

    gold_set = set(gold)
    main_set = set(main)
    details: dict[int, NumberDetail] = {}
    for index in range(N_NUM):
        num = index + 1
        details[num] = NumberDetail(
            num=num,
            final=float(final[index]),
            base=float(base[index]),
            votes=int(votes[index]),
            rank=rank_of[num],
            zone4=int(snapshot.zone4[index]),
            zone8=int(snapshot.zone8[index]),
            is_odd=bool(snapshot.is_odd[index]),
            is_prime=bool(snapshot.is_prime[index]),
            size_class=int(snapshot.size_class[index]),
            tail=int(snapshot.tail[index]),
            is_repeat=num in last_draw,
            r1=bool(masks.get("R1", np.zeros(N_NUM, dtype=bool))[index]),
            r2=bool(masks.get("R2", np.zeros(N_NUM, dtype=bool))[index]),
            r3=bool(masks.get("R3", np.zeros(N_NUM, dtype=bool))[index]),
            r4=bool(masks.get("R4", np.zeros(N_NUM, dtype=bool))[index]),
            r5=bool(masks.get("R5", np.zeros(N_NUM, dtype=bool))[index]),
            freq_10=float(snapshot.freq_10[index]),
            freq_25=float(snapshot.freq_25[index]),
            freq_50=float(snapshot.freq_50[index]),
            freq_all=float(snapshot.freq_all[index]),
            pressure=float(snapshot.pressure[index]),
            gold_flag=num in gold_set,
            in_main=num in main_set,
        )
    return details


def build_pool(
    snapshot: FeatureSnapshot,
    final: np.ndarray,
    votes: np.ndarray,
    last_draw: frozenset[int],
    *,
    pool_size: int,
    cfg: AppConfig,
    ranks: Mapping[str, np.ndarray] | None = None,
    verdicts: Sequence[RuleVerdict] | None = None,
    base: np.ndarray | None = None,
) -> PoolResult:
    """构建候选池：真实执行硬约束 + 有序放宽留痕 + A5 金胆（契约 R34–R36）。

    @param snapshot 特征快照
    @param final (80,) FinalScore
    @param votes (80,) 共振票数（int）
    @param last_draw 上期开奖号码集合
    @param pool_size 复式池目标规模
    @param cfg 全局配置
    @param ranks 百分位秩（仅用于对齐契约签名，可空）
    @param verdicts 五规则判定（用于逐号明细，可空）
    @param base (80,) BaseScore（可空则记 0）
    @return PoolResult
    @raises InfeasibleConstraintError 全部约束放宽后仍无法凑满目标规模
    """
    pool_cfg = cfg.pool
    size = int(pool_size)
    if size <= 0:
        raise InfeasibleConstraintError("复式池目标规模必须为正整数；请检查 pool.main_default")
    if size > N_NUM:
        raise InfeasibleConstraintError(f"复式池目标规模 {size} 超过号码总数 {N_NUM}")

    _min_rep, _max_rep, repeat_target = repeat_quota(size, pool_cfg)
    score_order = _score_order(final)
    warnings: list[str] = []
    relaxed: list[str] = []
    selected: list[int] = []

    order_steps: list[str | None] = [None] + list(pool_cfg.relax_order)
    step_warnings: tuple[str, ...] = ()
    # 逐项放宽必须**累积**（P10b-M16）：每轮再放开一项后重跑 greedy；若每轮重置为
    # "全部约束 - 当前项"，则前面放开过的项会被重新启用，"全部放宽"分支永不可达。
    active = set(pool_cfg.relax_order)
    for relax_key in order_steps:
        if relax_key is not None:
            active.discard(relax_key)
            relaxed.append(relax_key)
        selected, step_warnings = greedy_pool(
            final,
            snapshot,
            pool_cfg,
            size=size,
            active=frozenset(active),
            repeat_target=repeat_target,
            last_draw=last_draw,
        )
        if len(selected) >= size:
            break

    if len(selected) < size:
        raised = "、".join(relaxed) if relaxed else "（无）"
        raise InfeasibleConstraintError(
            f"在放宽 {len(order_steps) - 1} 项形态约束（{raised}）后仍只能选出 "
            f"{len(selected)}/{size} 码；请调低 pool.main_default 或放宽 "
            "pool.max_zone4_ratio 等阈值后重跑"
        )

    # 被接受步骤自身的告警（如重号目标缺失）必须带出，禁止静默丢弃
    warnings.extend(step_warnings)

    # B7：只记录**真实用到**的放宽项（relax_order 的顺序前缀），并写明放宽到多少
    if relaxed:
        warnings.append(
            "本次选池按 B7 有序放宽 " + str(len(relaxed)) + " 项（顺序："
            + " → ".join(relaxed) + "），下列为逐项放开值"
        )
    for relax_key in relaxed:
        warnings.append(_relax_detail(relax_key, selected, size, pool_cfg, last_draw))

    # R36 兜底：若重号比仍越界（repeat_ratio 已被放宽），尝试最小侵入置换
    min_rep, max_rep, _target = repeat_quota(size, pool_cfg)
    repeat_now = sum(1 for num in selected if num in last_draw)
    if not min_rep <= repeat_now <= max_rep:
        adjusted, quota_warns = apply_repeat_quota(score_order, last_draw, pool_cfg, size=size)
        _checked, shape_reasons = enforce_constraints(
            adjusted, snapshot, pool_cfg, size=size, active=frozenset(pool_cfg.relax_order)
        )
        if adjusted and len(adjusted) >= size and not shape_reasons:
            selected = adjusted[:size]
            warnings.append(
                f"重号配比按 R36 最小侵入置换修正：重号 {repeat_now} → "
                f"{sum(1 for num in selected if num in last_draw)} / {size}"
            )
        else:
            warnings.append("重号配比越界且无法在不破坏形态约束的前提下置换修正，保留当前池")
        warnings.extend(quota_warns)

    # 最终复核（构造即满足；此处作为防御性自检，违反即抛错，禁止带病出单）
    checked, reasons = enforce_constraints(
        selected, snapshot, pool_cfg, size=size, active=frozenset(pool_cfg.relax_order)
    )
    remaining_relaxed = [key for key in relaxed if key in pool_cfg.relax_order]
    if reasons and not remaining_relaxed:
        raise InfeasibleConstraintError(
            "候选池违反硬约束且未记录任何放宽项：" + "；".join(reasons) + "；请检查选池实现"
        )

    main_pool = sorted(int(num) for num in checked[:size])
    gold_pool = select_gold(main_pool, final, votes, pool_cfg)
    balanced, shape_reasons = is_balanced(main_pool, pool_cfg)
    feasible = bool(balanced and not relaxed)
    if shape_reasons and feasible:
        # is_balanced 的浮点占比口径比整数上限更严格，二者不一致时以告警披露
        warnings.append("形态体检提示（非阻断）：" + "；".join(shape_reasons))

    base_arr = np.zeros(N_NUM, dtype=np.float64) if base is None else np.asarray(base, dtype=np.float64)
    details = _build_details(
        snapshot, final, votes, base_arr, verdicts, last_draw, gold_pool, main_pool
    )
    repeat_count = sum(1 for num in main_pool if num in last_draw)
    return PoolResult(
        gold=tuple(gold_pool),
        main=tuple(main_pool),
        votes=np.asarray(votes, dtype=np.int8),
        scores=np.asarray(final, dtype=np.float64),
        order=tuple(score_order),
        details=details,
        warnings=tuple(warnings),
        relaxed=tuple(relaxed),
        feasible=feasible,
        repeat_count=repeat_count,
        new_count=len(main_pool) - repeat_count,
        size=len(main_pool),
    )