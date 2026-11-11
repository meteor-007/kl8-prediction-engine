# -*- coding: utf-8 -*-
"""用户点位录入、校验与模型一致性分析模块。

【业务背景与架构定位】
点位是用户每日自选的候选号码组（可 1~80 码，常规为 20 码），需要与模型输出做
正交对比而非直接篡改模型分数。本模块提供：
1. 严格输入解析与校验（空格/逗号/顿号分隔，去重升序）；
2. 点位与金胆、主选、30 码大底、Top20 的重合分析；
3. 共识码 / 逆势码 / 上期重号点位标注；
4. boost 模式下缩水注单的点位覆盖审计。

@author czw
@since 2026-11-06
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

from kl8_engine.constants import N_NUM
from kl8_engine.data import Dataset
from kl8_engine.models.ensemble import EnsembleResult

_SPLIT_RE = re.compile(r"[,\s，、;；]+")
_TOKEN_RE = re.compile(r"\d{1,2}")

CONTRARIAN_RANK_THRESHOLD = 40
"""模型排名低于该阈值视为逆势码（即排名 > 40）。"""

POINTS_MODES = ("compare", "boost")
"""点位模式: compare=仅对比标注; boost=参与缩水覆盖加权。"""


class PointsError(ValueError):
    """点位输入非法。"""


def parse_points(raw: Optional[str]) -> Tuple[int, ...]:
    """解析点位字符串为去重升序号码元组。

    @param raw 空格/逗号/顿号分隔的点位字符串
    @return 升序去重后的号码元组；空输入返回空元组
    @raises PointsError 存在非法字符、越界或重复号码
    """
    if raw is None:
        return ()
    tokens = [t for t in _SPLIT_RE.split(raw.strip()) if t]
    if not tokens:
        return ()

    numbers = []
    for token in tokens:
        if not _TOKEN_RE.fullmatch(token):
            raise PointsError(f"点位 '{token}' 不是有效号码")
        num = int(token)
        if not (1 <= num <= N_NUM):
            raise PointsError(f"点位 {num} 超出 1..{N_NUM} 范围")
        numbers.append(num)

    duplicated = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicated:
        raise PointsError(f"点位存在重复号码: {' '.join(f'{n:02d}' for n in duplicated)}")

    return tuple(sorted(numbers))


def normalize_points(points: Optional[Iterable[int]]) -> Tuple[int, ...]:
    """校验并规范化号码序列（供程序内部调用）。

    @param points 号码可迭代对象
    @return 升序去重元组
    @raises PointsError 号码越界
    """
    if points is None:
        return ()
    normalized = sorted({int(n) for n in points})
    for num in normalized:
        if not (1 <= num <= N_NUM):
            raise PointsError(f"点位 {num} 超出 1..{N_NUM} 范围")
    return tuple(normalized)


@dataclass(frozen=True)
class PointsAnalysis:
    """点位与模型输出的一致性分析结果。"""

    points: Tuple[int, ...]
    mode: str
    ranks: Dict[int, int]
    scores: Dict[int, float]
    consensus_top20: Tuple[int, ...]
    overlap_gold: Tuple[int, ...]
    overlap_main: Tuple[int, ...]
    overlap_pool30: Tuple[int, ...]
    contrarian: Tuple[int, ...]
    repeat_with_latest: Tuple[int, ...]
    boost_points_covered: Tuple[int, ...] = ()
    boost_tickets_covered: int = 0
    boost_points_unavailable: Tuple[int, ...] = ()


def analyze_points(
    points: Sequence[int],
    ensemble: EnsembleResult,
    dataset: Dataset,
    mode: str = "compare",
) -> PointsAnalysis:
    """分析点位与模型金胆/主选/大底/Top20 的重合关系。

    @param points 已校验的点位号码
    @param ensemble 混合集成结果
    @param dataset 历史数据集（用于上期重号判定）
    @param mode compare 或 boost
    @return 结构化点位分析结果
    """
    if mode not in POINTS_MODES:
        raise ValueError(f"未知点位模式: {mode}，可选 {POINTS_MODES}")

    point_set = set(points)
    latest_numbers = set(dataset.latest.numbers)

    ranks: Dict[int, int] = {}
    scores: Dict[int, float] = {}
    for num in points:
        detail = ensemble.details.get(num)
        if detail is None:
            continue
        ranks[num] = detail.rank
        scores[num] = detail.final_score

    consensus = tuple(n for n in ensemble.full_order[:20] if n in point_set)
    overlap_gold = tuple(n for n in ensemble.gold_pool if n in point_set)
    overlap_main = tuple(n for n in ensemble.main_pool if n in point_set)
    overlap_pool30 = tuple(n for n in ensemble.shrinkage_pool_30 if n in point_set)
    contrarian = tuple(sorted(n for n in points if ranks.get(n, 999) > CONTRARIAN_RANK_THRESHOLD))
    repeats = tuple(sorted(point_set & latest_numbers))

    return PointsAnalysis(
        points=tuple(sorted(points)),
        mode=mode,
        ranks=ranks,
        scores=scores,
        consensus_top20=consensus,
        overlap_gold=overlap_gold,
        overlap_main=overlap_main,
        overlap_pool30=overlap_pool30,
        contrarian=contrarian,
        repeat_with_latest=repeats,
    )


def attach_boost_coverage(
    analysis: PointsAnalysis,
    shrinkage_pool: Sequence[int],
    tickets: Sequence[Sequence[int]],
) -> PointsAnalysis:
    """统计 boost 模式下点位在缩水注单中的覆盖情况。

    @param analysis 原点位分析
    @param shrinkage_pool 30 码大底
    @param tickets 缩水精选注单
    @return 补充覆盖字段后的新分析对象
    """
    point_set = set(analysis.points)
    pool_set = set(shrinkage_pool)
    covered: set[int] = set()
    covered_tickets = 0
    for ticket in tickets:
        hit = point_set & set(ticket)
        if hit:
            covered_tickets += 1
            covered |= hit

    return dataclasses.replace(
        analysis,
        boost_points_covered=tuple(sorted(covered)),
        boost_tickets_covered=covered_tickets,
        boost_points_unavailable=tuple(sorted(point_set - pool_set)),
    )


def _fmt_numbers(numbers: Iterable[int]) -> str:
    """号码列表格式化为两位补零空格串。"""
    return " ".join(f"{n:02d}" for n in numbers)


def format_points_console(analysis: PointsAnalysis) -> str:
    """渲染点位分析控制台/TXT 板块。"""
    mode_desc = (
        "compare（仅对比标注，不改变模型结果）"
        if analysis.mode == "compare"
        else "boost（点位参与缩水覆盖加权）"
    )
    lines = [
        "【段落 2: 用户点位分析】",
        "-" * 80,
        f"  * 点位规模: {len(analysis.points)} 码 | 点位模式: {mode_desc}",
        f"  * 点位号码: {_fmt_numbers(analysis.points)}",
        f"  * 与模型 Top20 共识码 ({len(analysis.consensus_top20)} 个): "
        f"{_fmt_numbers(analysis.consensus_top20) if analysis.consensus_top20 else '无'}",
        f"  * 与超级金胆(4码) 重合 ({len(analysis.overlap_gold)} 个): "
        f"{_fmt_numbers(analysis.overlap_gold) if analysis.overlap_gold else '无'}",
        f"  * 与精选大底(12码) 重合 ({len(analysis.overlap_main)} 个): "
        f"{_fmt_numbers(analysis.overlap_main) if analysis.overlap_main else '无'}",
        f"  * 与30码大底 重合 ({len(analysis.overlap_pool30)} 个): "
        f"{_fmt_numbers(analysis.overlap_pool30) if analysis.overlap_pool30 else '无'}",
        f"  * 逆势码 (模型排名 > {CONTRARIAN_RANK_THRESHOLD}) ({len(analysis.contrarian)} 个): "
        f"{_fmt_numbers(analysis.contrarian) if analysis.contrarian else '无'}",
        f"  * 与上期开奖重号点位 ({len(analysis.repeat_with_latest)} 个): "
        f"{_fmt_numbers(analysis.repeat_with_latest) if analysis.repeat_with_latest else '无'}",
        "  * 点位逐号对照:",
        "      号码   模型排名   综合得分   金胆   主选   30码大底",
    ]

    for num in analysis.points:
        rank = analysis.ranks.get(num)
        score = analysis.scores.get(num)
        rank_str = f"{rank:02d}" if rank is not None else "--"
        score_str = f"{score:.4f}" if score is not None else " --  "
        gold_flag = " ★ " if num in analysis.overlap_gold else " - "
        main_flag = " ★ " if num in analysis.overlap_main else " - "
        pool_flag = " ★ " if num in analysis.overlap_pool30 else " - "
        lines.append(
            f"      [{num:02d}]     {rank_str}      {score_str}    {gold_flag}   {main_flag}    {pool_flag}"
        )

    if analysis.mode == "boost":
        lines.append(
            f"  * 缩水覆盖（boost）: {analysis.boost_tickets_covered} 注含点位 | "
            f"覆盖 {len(analysis.boost_points_covered)}/{len(analysis.points)} 个点位 | "
            f"不在30码大底的点位: "
            f"{_fmt_numbers(analysis.boost_points_unavailable) if analysis.boost_points_unavailable else '无'}"
        )

    return "\n".join(lines)