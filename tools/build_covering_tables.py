"""幂等生成并严格穷举重验经典覆盖设计表（契约 R38、§6.1、§6.4）。

生成标准覆盖表：
- C(12, 5, 4, 5): 12选5 出5保4 最优已知设计
- C(10, 5, 4, 5): 10选5 出5保4
- C(12, 7, 5, 6): 12选7 出6保5
- C(10, 7, 5, 6): 10选7 出6保5
- C(8, 5, 4, 5): 8选5 出5保4

所有区组必须经过位掩码穷举 100% 验证保底命题，禁止录入未验证注单。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


def verify_guarantee_exhaustive(tickets: list[list[int]], v: int, k: int, t: int, m: int) -> bool:
    """穷举验证覆盖保底命题：对任意 m 子集 M，存在注单 B 使得 |B ∩ M| >= t。"""
    ticket_masks = [sum(1 << x for x in b) for b in tickets]
    all_elements = range(v)
    for subset in itertools.combinations(all_elements, m):
        m_mask = sum(1 << x for x in subset)
        covered = any(bin(t_mask & m_mask).count("1") >= t for t_mask in ticket_masks)
        if not covered:
            return False
    return True


def solve_greedy_covering(v: int, k: int, t: int, m: int) -> list[list[int]]:
    """贪心求解覆盖设计。"""
    subsets = list(itertools.combinations(range(v), m))
    subset_masks = [sum(1 << x for x in s) for s in subsets]
    uncovered = set(range(len(subset_masks)))

    candidates = list(itertools.combinations(range(v), k))
    candidate_masks = [sum(1 << x for x in c) for c in candidates]

    # 预计算每个候选注单能够覆盖哪些子集
    covers_map: list[list[int]] = []
    for c_idx, c_mask in enumerate(candidate_masks):
        covered_subsets = [
            s_idx for s_idx, s_mask in enumerate(subset_masks)
            if bin(c_mask & s_mask).count("1") >= t
        ]
        covers_map.append(covered_subsets)

    chosen_tickets: list[list[int]] = []
    while uncovered:
        best_c = -1
        best_count = -1
        for c_idx in range(len(candidates)):
            # 统计能新覆盖的子集个数
            new_cov = sum(1 for s_idx in covers_map[c_idx] if s_idx in uncovered)
            if new_cov > best_count:
                best_count = new_cov
                best_c = c_idx
        if best_count <= 0 or best_c == -1:
            break
        chosen_tickets.append(list(candidates[best_c]))
        for s_idx in covers_map[best_c]:
            uncovered.discard(s_idx)

    return chosen_tickets


def build_and_verify_tables(output_path: str = "kl8/layer5_combinatorial/covering_tables.json") -> dict[str, list[list[int]]]:
    """生成并核验所有核心经典覆盖设计区组。"""
    specs = [
        (8, 5, 4, 5),
        (10, 5, 4, 5),
        (12, 5, 4, 5),
        (10, 7, 5, 6),
        (12, 7, 5, 6),
    ]

    tables: dict[str, list[list[int]]] = {}
    for v, k, t, m in specs:
        key = f"{v},{k},{t},{m}"
        tickets = solve_greedy_covering(v, k, t, m)
        assert verify_guarantee_exhaustive(tickets, v, k, t, m), f"核验失败: {key}"
        tables[key] = tickets
        print(f"覆盖表 {key}: 注数={len(tickets)}，100% 穷举核验通过")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2)
    print(f"全量覆盖表已成功写入: {output_path}")
    return tables


if __name__ == "__main__":
    build_and_verify_tables()
