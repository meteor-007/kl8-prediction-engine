"""运筹学组合覆盖设计与旋转矩阵求解（契约 R37–R40、§3.11、§4.5、裁决 A7）。

核心数学机制：
- 覆盖设计 C(v, k, t, m)：从大小为 v 的优选池中挑选最少注数 k 码注单，
  使得池中任意大小为 m 的子集 M 都至少有一注满足 |B ∩ M| >= t。
- 裁决 A7（v1.1.0 修正）：
  * t >= k: m = min(v, t)
  * t < k: m = min(v, t + margin)（margin 默认 1，保证 m >= t+1）
- 查表映射 lookup_table -> 0-1 整数规划 solve_ilp -> 带权贪心 greedy_weighted -> 模拟退火 anneal
- 穷举位掩码保底验证 verify_guarantee（未通过禁止出单）
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kl8.config import CoveringConfig
from kl8.datatypes import CoveringReport
from kl8.errors import CoveringError, InfeasibleConstraintError


@dataclass(frozen=True)
class GuaranteeResult:
    """覆盖设计保底命题穷举验证结果（契约 §3.11）。"""

    passed: bool
    achieved_t: int
    n_subsets: int
    failed_example: tuple[int, ...] | None


def m_target(
    v: int,
    k: int,
    t: int,
    m_mode: str = "auto",
    m_fixed: int | None = None,
    margin: int = 1,
) -> int:
    """计算目标命中假设参数 m（契约裁决 A7、v1.1.0 修正）。

    规则：
    1. m_mode == 'fixed' 且指定 m_fixed 时直接采用；
    2. t >= k: m = min(v, t)
    3. t < k: m = min(v, t + margin)
    @param v 候选池大小
    @param k 单注球数
    @param t 期望保底命中数 (t <= k)
    @param m_mode 'auto' | 'fixed'
    @param m_fixed 固定 m 取值
    @param margin auto 模式下的外推余量（默认 1）
    @return 整数 m，满足 t <= m <= v
    """
    if m_mode == "fixed" and m_fixed is not None:
        m_val = m_fixed
    elif t >= k:
        m_val = min(v, t)
    else:
        m_val = min(v, t + max(1, margin))

    # 边界保护
    m_val = max(t, min(v, m_val))
    return m_val


def verify_guarantee(
    tickets: Sequence[Sequence[int]],
    v: int,
    k: int,
    t: int,
    m: int,
) -> GuaranteeResult:
    """位掩码穷举验证覆盖保底命题（契约 R37、§6.2）。

    检验是否对任意 C(v, m) 个子集 M，均存在至少一注 B 满足 |B ∩ M| >= t。
    @param tickets 注单列表（元素为 0..v-1 的位置索引）
    @param v 候选池大小
    @param k 单注大小
    @param t 保底命中数
    @param m 假设命中数
    @return GuaranteeResult
    """
    if not tickets or v < k or m > v or t > k:
        return GuaranteeResult(passed=False, achieved_t=0, n_subsets=0, failed_example=())

    ticket_masks = [sum(1 << int(x) for x in b) for b in tickets]
    all_indices = range(v)
    subsets = list(itertools.combinations(all_indices, m))
    n_subsets = len(subsets)

    achieved_min = k
    for subset in subsets:
        m_mask = sum(1 << int(x) for x in subset)
        max_overlap = 0
        for t_mask in ticket_masks:
            overlap = bin(t_mask & m_mask).count("1")
            if overlap > max_overlap:
                max_overlap = overlap
                if max_overlap >= t:
                    break
        if max_overlap < achieved_min:
            achieved_min = max_overlap
        if max_overlap < t:
            return GuaranteeResult(
                passed=False,
                achieved_t=achieved_min,
                n_subsets=n_subsets,
                failed_example=tuple(subset),
            )

    return GuaranteeResult(
        passed=True,
        achieved_t=achieved_min,
        n_subsets=n_subsets,
        failed_example=None,
    )


def coverage_ratio(
    tickets: Sequence[Sequence[int]],
    v: int,
    k: int,
    t: int,
    m: int,
) -> float:
    """计算已达成 |B ∩ M| >= t 保底的 m-子集覆盖比例 ∈ [0, 1]。"""
    if not tickets or v < k or m > v:
        return 0.0

    ticket_masks = [sum(1 << int(x) for x in b) for b in tickets]
    subsets = list(itertools.combinations(range(v), m))
    if not subsets:
        return 1.0

    covered_count = 0
    for subset in subsets:
        m_mask = sum(1 << int(x) for x in subset)
        for t_mask in ticket_masks:
            if bin(t_mask & m_mask).count("1") >= t:
                covered_count += 1
                break

    return float(covered_count) / float(len(subsets))


def lookup_table(
    v: int,
    k: int,
    t: int,
    m: int,
    *,
    path: str | Path,
) -> tuple[tuple[int, ...], ...] | None:
    """从经典区组库中查表检索最优覆盖设计（契约 R38）。

    @param v 池大小
    @param k 单注大小
    @param t 保底数
    @param m 假设命中数
    @param path 覆盖表 JSON 文件路径
    @return 位置注单元组或 None
    """
    p = Path(path)
    if not p.exists():
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            tables = json.load(f)
        key = f"{v},{k},{t},{m}"
        if key in tables:
            raw_tickets = tables[key]
            # 校验并返回
            return tuple(tuple(int(x) for x in ticket) for ticket in raw_tickets)
        return None
    except (OSError, ValueError, TypeError, KeyError) as e:
        raise CoveringError(f"读取覆盖表文件 {p} 失败: {e}") from e


def solve_ilp(
    v: int,
    k: int,
    t: int,
    m: int,
    *,
    candidate_cap: int,
    target_cap: int,
    time_limit_s: float = 5.0,
) -> tuple[tuple[int, ...], ...] | None:
    """使用 0-1 整数线性规划求解理论最少注数（契约 R39）。

    当组合规模超过上限时返回 None，安全降级。
    """
    n_candidates = math.comb(v, k)
    n_targets = math.comb(v, m)
    if n_candidates > candidate_cap or n_targets > target_cap:
        return None

    try:
        from scipy.optimize import milp, LinearConstraint  # type: ignore[import-untyped]
    except ImportError:
        return None

    candidates = list(itertools.combinations(range(v), k))
    cand_masks = [sum(1 << x for x in c) for c in candidates]
    targets = list(itertools.combinations(range(v), m))
    target_masks = [sum(1 << x for x in s) for s in targets]

    # 构建约束矩阵 A @ x >= 1
    # A_ij = 1 如果 cand_i 覆盖 target_j
    matrix = np.zeros((len(targets), len(candidates)), dtype=np.float64)
    for j, t_mask in enumerate(target_masks):
        for i, c_mask in enumerate(cand_masks):
            if bin(c_mask & t_mask).count("1") >= t:
                matrix[j, i] = 1.0

    c = np.ones(len(candidates), dtype=np.float64)
    integrality = np.ones(len(candidates), dtype=np.int8)  # 纯整数规划 0-1
    constraints = LinearConstraint(matrix, lb=1.0, ub=np.inf)

    try:
        res = milp(c=c, integrality=integrality, constraints=constraints)
        if res.success and res.x is not None:
            chosen_indices = np.where(res.x > 0.5)[0]
            chosen = tuple(tuple(candidates[idx]) for idx in chosen_indices)
            return chosen
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        # MILP 求解器在退化/超时下可能抛数值异常；返回 None 由上层降级贪心+退火
        return None

    return None


def greedy_weighted(
    v: int,
    k: int,
    t: int,
    m: int,
    weights: np.ndarray,
    *,
    budget: int,
    seed: int,
    max_iters: int = 2000,
) -> tuple[tuple[int, ...], ...]:
    """带概率权重的贪心集合覆盖求解器（契约 R40）。

    优先挑选既能覆盖最多未覆盖子集、自身包含号码概率得分之和又最高的注单。
    @param v 池大小
    @param k 单注大小
    @param t 保底数
    @param m 假设命中数
    @param weights (v,) 池内各位置号码的综合权重得分
    @param budget 最大预算注数
    @param seed 确定性种子
    @return 位置注单元组
    """
    candidates = list(itertools.combinations(range(v), k))
    cand_masks = [sum(1 << x for x in c) for c in candidates]
    cand_weights = [sum(float(weights[x]) for x in c) for c in candidates]

    subsets = list(itertools.combinations(range(v), m))
    subset_masks = [sum(1 << x for x in s) for s in subsets]
    uncovered = set(range(len(subset_masks)))

    # 预计算注单覆盖关系
    covers_map: list[list[int]] = []
    for c_mask in cand_masks:
        covered = [
            s_idx for s_idx, s_mask in enumerate(subset_masks)
            if bin(c_mask & s_mask).count("1") >= t
        ]
        covers_map.append(covered)

    chosen: list[tuple[int, ...]] = []
    while uncovered and len(chosen) < budget:
        best_cand = -1
        best_new_cov = -1
        best_w = -1e9

        for c_idx in range(len(candidates)):
            new_cov = sum(1 for s_idx in covers_map[c_idx] if s_idx in uncovered)
            w_score = cand_weights[c_idx]
            if (new_cov > best_new_cov) or (new_cov == best_new_cov and w_score > best_w):
                best_new_cov = new_cov
                best_w = w_score
                best_cand = c_idx

        if best_cand == -1 or best_new_cov <= 0:
            break

        chosen.append(tuple(candidates[best_cand]))
        for s_idx in covers_map[best_cand]:
            uncovered.discard(s_idx)

    return tuple(chosen)


def anneal(
    v: int,
    k: int,
    t: int,
    m: int,
    weights: np.ndarray,
    *,
    budget: int,
    iterations: int,
    t0: float,
    cooling: float,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """模拟退火固定预算覆盖优化（契约 R40、C7 确定性）。"""
    rng = np.random.default_rng(seed)
    candidates = list(itertools.combinations(range(v), k))
    if len(candidates) <= budget:
        return tuple(tuple(c) for c in candidates)

    # 初始解：贪心选取
    initial = greedy_weighted(v, k, t, m, weights, budget=budget, seed=seed)
    if len(initial) == budget:
        current_set = set(initial)
    else:
        current_set = set(initial)
        for c in candidates:
            if len(current_set) >= budget:
                break
            current_set.add(tuple(c))

    def _eval(t_set: set[tuple[int, ...]]) -> float:
        cov = coverage_ratio(list(t_set), v, k, t, m)
        w_sum = sum(sum(float(weights[x]) for x in b) for b in t_set)
        return cov * 1000.0 + w_sum

    current_score = _eval(current_set)
    best_set = set(current_set)
    best_score = current_score

    temp = t0
    cand_list = [tuple(c) for c in candidates]
    for _ in range(iterations):
        # 随机替换一注
        current_list = list(current_set)
        out_ticket = current_list[int(rng.integers(0, len(current_list)))]
        in_ticket = cand_list[rng.integers(0, len(cand_list))]
        if in_ticket in current_set:
            continue

        neighbor = (current_set - {out_ticket}) | {in_ticket}
        neighbor_score = _eval(neighbor)
        delta = neighbor_score - current_score

        if delta > 0.0 or math.exp(delta / max(1e-9, temp)) > rng.random():
            current_set = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(neighbor)

        temp *= cooling

    return tuple(sorted(best_set))


def solve(
    v: int,
    k: int,
    t: int,
    m: int,
    weights: np.ndarray,
    cfg: CoveringConfig,
    *,
    seed: int = 0,
) -> CoveringReport:
    """组合覆盖全流程求解调度（契约 R37–R40、§3.11）。

    流程：查表 -> ILP -> 贪心 + 模拟退火 -> 穷举校验保底并生成报告。

    @param v 覆盖池规模
    @param k 单注号码个数
    @param t 保底命中数
    @param m 池内命中数假设
    @param weights (v,) 池内号码权重
    @param cfg 覆盖配置
    @param seed 随机种子（契约 B10：唯一随机源为 cfg.seed，调用方透传）
    @return CoveringReport
    @raises InfeasibleConstraintError 规模超限
    """
    t_start = time.perf_counter()
    if v > cfg.max_pool_v:
        raise InfeasibleConstraintError(f"号码池规模 v={v} 超过上限 {cfg.max_pool_v}")

    n_m_subsets = math.comb(v, m)
    if n_m_subsets > cfg.max_m_subsets:
        raise InfeasibleConstraintError(
            f"目标 m 子集数 C({v}, {m})={n_m_subsets} 超过上限 {cfg.max_m_subsets}"
        )

    warnings: list[str] = []
    degraded = False
    degrade_reason = ""
    method = "table"

    # 0. 契约 A7 合理性校验：m 是玩法假设参数，与「池内期望命中数」偏离过大时只告警
    heuristic_m = min(v, math.ceil(0.25 * v) + 2)
    if abs(m - heuristic_m) > 2:
        warnings.append(
            f"假设命中数 m={m} 与经验期望 ceil(0.25v)+2={heuristic_m} 偏离超过 2，"
            "请在报告中注明「m 为用户/玩法假设参数，非算法预测值」"
        )

    # 1. 尝试查表
    tickets_pos: tuple[tuple[int, ...], ...] | None = None
    if cfg.table_path:
        try:
            tickets_pos = lookup_table(v, k, t, m, path=cfg.table_path)
        except CoveringError as exc:
            warnings.append(f"覆盖表读取失败，回落到 ILP 求解：{exc}")

    # 2. 查表未命中，尝试 ILP
    if tickets_pos is None:
        method = "ilp"
        tickets_pos = solve_ilp(
            v, k, t, m,
            candidate_cap=cfg.ilp_candidate_cap,
            target_cap=cfg.ilp_target_cap,
            time_limit_s=cfg.ilp_time_limit_s,
        )
        if tickets_pos is None:
            degraded = True
            degrade_reason = "规模超出 ILP 上限或无求解器，降级使用带权贪心与模拟退火"
            warnings.append(degrade_reason)
            method = "greedy+anneal"
            tickets_pos = greedy_weighted(v, k, t, m, weights, budget=cfg.budget, seed=seed)
            if len(tickets_pos) < cfg.budget and cfg.sa_iterations > 0:
                tickets_pos = anneal(
                    v, k, t, m, weights,
                    budget=cfg.budget,
                    iterations=cfg.sa_iterations,
                    t0=cfg.sa_t0,
                    cooling=cfg.sa_cooling,
                    seed=seed,
                )

    # 3. 截断到最大预算
    if len(tickets_pos) > cfg.budget:
        warnings.append(f"注数 {len(tickets_pos)} 超出预算 {cfg.budget}，已截断")
        tickets_pos = tickets_pos[:cfg.budget]

    # 4. 穷举验证保底
    g_res = verify_guarantee(tickets_pos, v, k, t, m)
    cov_ratio = coverage_ratio(tickets_pos, v, k, t, m)
    elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    # 虚拟映射占位（上层 tickets.py 将位置转换为实际号码）
    dummy_nums = tuple(tuple(idx + 1 for idx in b) for b in tickets_pos)

    return CoveringReport(
        pool=tuple(range(1, v + 1)),
        v=v,
        k=k,
        t=t,
        m=m,
        tickets_pos=tickets_pos,
        tickets=dummy_nums,
        n_tickets=len(tickets_pos),
        coverage_ratio=cov_ratio,
        achieved_t=g_res.achieved_t,
        verified=g_res.passed,
        method=method,
        degraded=degraded,
        degrade_reason=degrade_reason,
        elapsed_ms=elapsed_ms,
        warnings=tuple(warnings),
    )
