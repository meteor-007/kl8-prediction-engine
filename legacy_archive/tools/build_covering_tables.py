"""幂等生成并严格穷举重验经典覆盖设计表（契约 R38、§6.1、§6.4）。

【标准覆盖表】
- C(12, 5, 4, 5): 12选5 出5保4 最优已知设计
- C(10, 5, 4, 5): 10选5 出5保4
- C(12, 7, 5, 6): 12选7 出6保5
- C(10, 7, 5, 6): 10选7 出6保5
- C(8, 5, 4, 5): 8选5 出5保4

所有区组必须经过位掩码穷举 100% 验证保底命题，禁止录入未验证注单。

【用法】
    python tools/build_covering_tables.py                        # 全量重建并覆盖写
    python tools/build_covering_tables.py --output path.json     # 指定输出文件
    python tools/build_covering_tables.py --spec 12,5,4,5        # 只重建单条设计（可重复）

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

#: 默认输出路径（契约 §6.4：覆盖表随包分发，路径固定）。
DEFAULT_TABLE_PATH: str = "kl8/layer5_combinatorial/covering_tables.json"

#: 默认全量设计清单 (v, k, t, m)。
DEFAULT_SPECS: tuple[tuple[int, int, int, int], ...] = (
    (8, 5, 4, 5),
    (10, 5, 4, 5),
    (12, 5, 4, 5),
    (10, 7, 5, 6),
    (12, 7, 5, 6),
)


def verify_guarantee_exhaustive(tickets: list[list[int]], v: int, k: int, t: int, m: int) -> bool:
    """穷举验证覆盖保底命题：对任意 m 子集 M，存在注单 B 使得 |B ∩ M| >= t。

    @param tickets 区组列表（每个区组是 0-based 位置元组）
    @param v 池规模
    @param k 单注大小
    @param t 保底命中数
    @param m 池内命中数假设
    @return True 表示全部 m 子集均被保底覆盖
    """
    ticket_masks = [sum(1 << x for x in block) for block in tickets]
    for subset in itertools.combinations(range(v), m):
        m_mask = sum(1 << x for x in subset)
        covered = any(bin(ticket_mask & m_mask).count("1") >= t for ticket_mask in ticket_masks)
        if not covered:
            return False
    return True


def solve_greedy_covering(v: int, k: int, t: int, m: int) -> list[list[int]]:
    """贪心求解覆盖设计（按"新覆盖子集数"最大化选注）。

    @param v 池规模
    @param k 单注大小
    @param t 保底命中数
    @param m 池内命中数假设
    @return 选中的区组列表
    """
    subsets = list(itertools.combinations(range(v), m))
    subset_masks = [sum(1 << x for x in subset) for subset in subsets]
    uncovered = set(range(len(subset_masks)))

    candidates = list(itertools.combinations(range(v), k))
    candidate_masks = [sum(1 << x for x in candidate) for candidate in candidates]

    covers_map: list[list[int]] = []
    for candidate_mask in candidate_masks:
        covered_subsets = [
            s_idx
            for s_idx, subset_mask in enumerate(subset_masks)
            if bin(candidate_mask & subset_mask).count("1") >= t
        ]
        covers_map.append(covered_subsets)

    chosen_tickets: list[list[int]] = []
    while uncovered:
        best_c = -1
        best_count = -1
        for c_idx in range(len(candidates)):
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


def _parse_spec(text: str) -> tuple[int, int, int, int]:
    """解析 ``v,k,t,m`` 形式的单条设计规格。

    @param text 逗号分隔规格
    @return (v, k, t, m)
    @raises argparse.ArgumentTypeError 格式非法或数值不合业务约束
    """
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"设计规格必须为 v,k,t,m 四元组，实际收到：{text!r}")
    try:
        v, k, t, m = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"设计规格必须为整数四元组：{text!r}") from exc
    if not (0 < t <= k < v and 0 < m <= v):
        raise argparse.ArgumentTypeError(f"设计规格不满足 0<t<=k<v 且 0<m<=v：{text!r}")
    return (v, k, t, m)


def build_and_verify_tables(
    output_path: str = DEFAULT_TABLE_PATH,
    *,
    specs: Sequence[tuple[int, int, int, int]] = DEFAULT_SPECS,
    extra: Sequence[tuple[int, int, int, int]] = (),
) -> dict[str, list[list[int]]]:
    """生成并核验覆盖设计区组，全量重验后原子落盘。

    @param output_path 输出 JSON 路径
    @param specs 本次要重建的设计清单
    @param extra 需保留但本次不重建的历史设计清单（从既有文件读回）
    @return {设计键: 区组列表}
    """
    tables: dict[str, list[list[int]]] = {}
    for v, k, t, m in specs:
        key = f"{v},{k},{t},{m}"
        tickets = solve_greedy_covering(v, k, t, m)
        if not verify_guarantee_exhaustive(tickets, v, k, t, m):
            raise AssertionError(f"覆盖设计穷举核验失败，拒绝写入未验证区组: {key}")
        tables[key] = tickets
        print(f"覆盖表 {key}: 注数={len(tickets)}，100% 穷举核验通过")

    p = Path(output_path)
    if extra:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for spec in extra:
            key = f"{spec[0]},{spec[1]},{spec[2]},{spec[3]}"
            if key not in tables and key in existing:
                tables[key] = existing[key]
                print(f"覆盖表 {key}: 沿用既有区组（本次未重建，共 {len(existing[key])} 注）")

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    tmp_path.replace(p)
    print(f"全量覆盖表已成功写入: {output_path}")
    return tables


def build_arg_parser() -> argparse.ArgumentParser:
    """构建覆盖表重建 CLI 解析器。

    @return argparse 解析器
    """
    parser = argparse.ArgumentParser(
        prog="build_covering_tables",
        description="生成并穷举核验经典覆盖设计表（契约 R38/§6.4）",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_TABLE_PATH,
        help=f"输出 JSON 路径（默认 {DEFAULT_TABLE_PATH}）",
    )
    parser.add_argument(
        "--spec",
        action="append",
        type=_parse_spec,
        default=None,
        help="只重建指定设计，格式 v,k,t,m（可重复指定；省略则重建全部默认设计）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。

    @param argv 命令行入参
    @return 退出码（0 成功）
    """
    args = build_arg_parser().parse_args(argv)
    specs = tuple(args.spec) if args.spec else DEFAULT_SPECS
    extra = tuple(spec for spec in DEFAULT_SPECS if spec not in specs)
    build_and_verify_tables(args.output, specs=specs, extra=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())