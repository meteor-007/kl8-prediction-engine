# -*- coding: utf-8 -*-
"""历史发布推荐实测复盘、近期命中率统计与推荐日志模块。

【业务背景与架构定位】
新引擎此前只输出当期预测，不评估"昨天推荐中了几个"。本模块补齐闭环：
1. 从 output/reports 历史精要报告中解析已发布推荐（金胆/主选/Top20/30码/票单），
   兼容旧版报告格式；
2. 与最新开奖逐项核算命中数、理论期望与 Lift，并计算票单实际回报；
3. 统计最近 5/10/30 期命中率与 ROI；
4. 通过 output/state/prediction_journal.jsonl 持久化每期首份推荐，
   防止开奖后重跑污染复盘（同一目标期只记录第一份推荐）。

@author czw
@since 2026-11-06
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from kl8_engine.constants import COST_PER_TICKET_YUAN, N_NUM, PRIZE_TABLE_PICK5
from kl8_engine.data import Dataset, Draw

# 各池型理论期望命中数（超几何分布）
EXPECTED_HITS: Dict[str, float] = {
    "gold": 1.0,
    "main": 3.0,
    "top20": 5.0,
    "pool30": 7.5,
}

JOURNAL_FILENAME = "prediction_journal.jsonl"

_PERIOD_RE = re.compile(r"预测目标期号\s*[:：]\s*第\s*(\d+)\s*期")
_GOLD_RE = re.compile(r"核心超级金胆\s*\(\s*\d+\s*码\s*\)\s*[:：]\s*\[([0-9\s]+)\]")
_MAIN_RE = re.compile(r"精选复式大底\s*\(\s*\d+\s*码\s*\)\s*[:：]\s*\[([0-9\s]+)\]")
_POOL30_RE = re.compile(r"30\s*码大底\s*[:：]\s*([0-9][0-9\s]*)")
_TICKET_RE = re.compile(r"第\s*\d+\s*注\s*[:：]\s*\[([0-9\s]+)\]")
_TOP20_ROW_RE = re.compile(r"\[(\d{2})\]")
_SHRINKAGE_PERIOD_RE = re.compile(r"第(\d+)期")


def _parse_number_list(raw: str) -> Tuple[int, ...]:
    """将空格分隔的号码串解析为升序元组。"""
    return tuple(sorted(int(x) for x in raw.split() if x.strip()))


def _extract_section(text: str, keyword: str) -> str:
    """按段落关键词截取正文片段，兼容新旧报告的段落编号。

    @param text 报告全文
    @param keyword 段落标题中的稳定关键词
    @return 从标题下一行到下一个"【段落"标记之间的文本；未找到返回空串
    """
    idx = text.find(keyword)
    if idx < 0:
        return ""
    line_end = text.find("\n", idx)
    if line_end < 0:
        return ""
    next_section = text.find("【段落", line_end)
    return text[line_end: next_section if next_section >= 0 else len(text)]


@dataclass(frozen=True)
class PublishedRecommendation:
    """一份已发布的历史推荐。"""

    target_period: int
    gold: Tuple[int, ...]
    main: Tuple[int, ...]
    top20: Tuple[int, ...]
    pool30: Tuple[int, ...]
    tickets: Tuple[Tuple[int, ...], ...]
    source_path: str
    full_order: Tuple[int, ...] = ()
    """80 码完整排序（可选；用于计算排序 AUC）。"""


@dataclass(frozen=True)
class TicketReview:
    """单注实测命中与回报。"""

    ticket: Tuple[int, ...]
    hits: int
    prize_yuan: int


@dataclass(frozen=True)
class ReviewResult:
    """一期推荐的实测复盘结果。"""

    target_period: int
    actual_period: int
    actual_date: str
    actual_numbers: Tuple[int, ...]
    source_path: str
    gold_hits: int
    main_hits: int
    top20_hits: int
    pool30_hits: int
    gold_total: int
    main_total: int
    top20_total: int
    pool30_total: int
    ticket_reviews: Tuple[TicketReview, ...]
    total_cost_yuan: int
    total_prize_yuan: int
    latest_draw_period: int = 0
    """最新开奖期号；与复盘期号不同表示当日断档、已回退到最近一份推荐。"""
    ranking_auc: Optional[float] = None
    """80 码排序 AUC（0.5=随机；仅当推荐记录了完整排序时可用）。"""

    @property
    def gold_lift(self) -> float:
        return self.gold_hits / EXPECTED_HITS["gold"]

    @property
    def main_lift(self) -> float:
        return self.main_hits / EXPECTED_HITS["main"]

    @property
    def top20_lift(self) -> Optional[float]:
        if self.top20_total <= 0:
            return None
        return self.top20_hits / EXPECTED_HITS["top20"]

    @property
    def pool30_lift(self) -> Optional[float]:
        if self.pool30_total <= 0:
            return None
        return self.pool30_hits / EXPECTED_HITS["pool30"]

    @property
    def net_profit_yuan(self) -> int:
        return self.total_prize_yuan - self.total_cost_yuan

    @property
    def best_ticket_hits(self) -> int:
        return max((t.hits for t in self.ticket_reviews), default=0)


@dataclass(frozen=True)
class WindowHitStat:
    """单个统计窗口的命中率汇总。"""

    window: int
    periods: int
    mean_gold: float
    lift_gold: float
    mean_main: float
    lift_main: float
    mean_top20: float
    lift_top20: float
    mean_pool30: float
    lift_pool30: float
    cost_yuan: int
    prize_yuan: int
    roi_pct: float
    mean_auc: Optional[float] = None
    """窗口内可用排序 AUC 的均值（无全序记录的期次不参与）。"""


def parse_report_text(text: str, source_path: str = "") -> Optional[PublishedRecommendation]:
    """从精要报告文本中解析已发布推荐。

    @param text 报告全文
    @param source_path 来源文件路径（仅用于审计展示）
    @return 解析结果；关键字段缺失时返回 None
    """
    period_match = _PERIOD_RE.search(text)
    gold_match = _GOLD_RE.search(_extract_section(text, "核心选拔梯队"))
    main_match = _MAIN_RE.search(_extract_section(text, "核心选拔梯队"))
    if not (period_match and gold_match and main_match):
        return None

    pool_section = _extract_section(text, "30 码黄金大底级联精细缩水方案执行概况")
    pool_match = _POOL30_RE.search(pool_section) if pool_section else None
    pool30 = _parse_number_list(pool_match.group(1)) if pool_match else ()

    top_section = _extract_section(text, "Top 20 梯队")
    top20 = tuple(int(x) for x in _TOP20_ROW_RE.findall(top_section)) if top_section else ()

    ticket_section = _extract_section(text, "实战推荐保底投注单清单")
    tickets = []
    if ticket_section:
        for raw in _TICKET_RE.findall(ticket_section):
            nums = _parse_number_list(raw)
            if len(nums) == 5:
                tickets.append(nums)

    return PublishedRecommendation(
        target_period=int(period_match.group(1)),
        gold=_parse_number_list(gold_match.group(1)),
        main=_parse_number_list(main_match.group(1)),
        top20=top20,
        pool30=pool30,
        tickets=tuple(tickets),
        source_path=source_path,
    )


def _display_path(path: Path) -> str:
    """尽量以项目相对路径展示文件来源，保证报告可复现。"""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        return str(path)


def parse_report_file(filepath: Path) -> Optional[PublishedRecommendation]:
    """解析单个精要报告文件。"""
    try:
        text = filepath.read_text(encoding="utf-8")
    except OSError:
        return None
    return parse_report_text(text, source_path=_display_path(filepath))


def find_published_for_period(reports_dir: Path, period: int) -> Optional[PublishedRecommendation]:
    """查找目标期号对应的最早一份已发布推荐。

    同一天重跑会覆盖同名文件，不同日期同目标期号时取日期最早的一份，
    保证复盘对象是当时真实对外发布的推荐。
    """
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob("Keno预测精要_*.txt"))
    for path in candidates:
        rec = parse_report_file(path)
        if rec is not None and rec.target_period == period:
            return rec
    return None


def _ranking_auc(full_order: Sequence[int], actual_numbers: Sequence[int]) -> Optional[float]:
    """由 80 码完整排序计算排序 AUC（0.5=随机）。"""
    if len(full_order) != N_NUM:
        return None
    rank = {int(num): idx + 1 for idx, num in enumerate(full_order)}
    if any(int(n) not in rank for n in actual_numbers):
        return None
    drawn_ranks = [rank[int(n)] for n in actual_numbers]
    n1 = len(drawn_ranks)
    n2 = N_NUM - n1
    if n1 <= 0 or n2 <= 0:
        return None
    # 名次 1 为最好，转换为得分 N+1-rank 后套用 Mann-Whitney AUC 公式
    return float(
        (n1 * (N_NUM + 1) - sum(drawn_ranks) - n1 * (n1 + 1) / 2) / (n1 * n2)
    )


def evaluate_recommendation(rec: PublishedRecommendation, actual: Draw) -> ReviewResult:
    """将一份推荐与真实开奖逐项核算命中与票单回报。"""
    actual_set = set(actual.numbers)
    gold_hits = len(set(rec.gold) & actual_set)
    main_hits = len(set(rec.main) & actual_set)
    top20_hits = len(set(rec.top20) & actual_set)
    pool30_hits = len(set(rec.pool30) & actual_set)

    ticket_reviews = []
    total_prize = 0
    for ticket in rec.tickets:
        hits = len(set(ticket) & actual_set)
        prize = PRIZE_TABLE_PICK5.get(hits, 0)
        total_prize += prize
        ticket_reviews.append(TicketReview(ticket=ticket, hits=hits, prize_yuan=prize))

    return ReviewResult(
        target_period=rec.target_period,
        actual_period=actual.period,
        actual_date=actual.date,
        actual_numbers=actual.numbers,
        source_path=rec.source_path,
        gold_hits=gold_hits,
        main_hits=main_hits,
        top20_hits=top20_hits,
        pool30_hits=pool30_hits,
        gold_total=len(rec.gold),
        main_total=len(rec.main),
        top20_total=len(rec.top20),
        pool30_total=len(rec.pool30),
        ticket_reviews=tuple(ticket_reviews),
        total_cost_yuan=len(rec.tickets) * COST_PER_TICKET_YUAN,
        total_prize_yuan=total_prize,
        ranking_auc=_ranking_auc(rec.full_order, actual.numbers),
    )


def build_review(output_dir: Path, dataset: Dataset) -> Optional[ReviewResult]:
    """复盘最近一份已发布推荐（含断档回退）。

    优先选择"最新已开奖期号"对应的首份推荐；若该期无推荐（例如当日未运行），
    自动回退到最近一份已发布且有实际开奖可对照的推荐，并在结果中标注
    latest_draw_period，供报告提示断档。

    @param output_dir 交付件根目录（推荐报告位于其 reports 子目录）
    @param dataset 已同步至最新开奖的数据集
    @return 复盘结果；完全无可复盘推荐时返回 None
    """
    latest = dataset.latest
    draw_map = {d.period: d for d in dataset.draws}
    recs = collect_published_recommendations(Path(output_dir))
    reviewable = sorted((p for p in recs if p in draw_map), reverse=True)
    if not reviewable:
        return None

    target = reviewable[0]
    result = evaluate_recommendation(recs[target], draw_map[target])
    if target != latest.period:
        result = dataclasses.replace(result, latest_draw_period=latest.period)
    return result


# =============================================================================
# 推荐日志（防止重跑污染复盘）
# =============================================================================


def _journal_path(output_dir: Path) -> Path:
    """推荐日志文件路径。"""
    return Path(output_dir) / "state" / JOURNAL_FILENAME


def load_journal(output_dir: Path) -> Dict[int, dict]:
    """加载推荐日志，按期号索引；同一期号保留首条记录。"""
    path = _journal_path(output_dir)
    entries: Dict[int, dict] = {}
    if not path.exists():
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                period = int(record["target_period"])
            except (ValueError, KeyError, TypeError):
                continue
            entries.setdefault(period, record)
    return entries


def append_prediction_journal(
    output_dir: Path,
    target_period: int,
    gold: Sequence[int],
    main: Sequence[int],
    top20: Sequence[int],
    pool30: Sequence[int],
    tickets: Sequence[Sequence[int]],
    points: Sequence[int] = (),
    points_mode: str = "compare",
    full_order: Sequence[int] = (),
    fusion_mode: str = "borda",
) -> bool:
    """追加一期首份推荐到日志。

    @return 是否实际写入；该期已有记录时返回 False（不覆盖，防复盘污染）
    """
    existing = load_journal(output_dir)
    if target_period in existing:
        return False

    path = _journal_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "target_period": int(target_period),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gold": list(gold),
        "main": list(main),
        "top20": list(top20),
        "pool30": list(pool30),
        "tickets": [list(t) for t in tickets],
        "points": list(points),
        "points_mode": points_mode,
        "full_order": list(full_order),
        "fusion_mode": fusion_mode,
        "engine": "kl8_engine",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def _journal_recommendation(record: dict) -> Optional[PublishedRecommendation]:
    """将日志记录转换为统一推荐实体。"""
    try:
        return PublishedRecommendation(
            target_period=int(record["target_period"]),
            gold=tuple(sorted(int(x) for x in record.get("gold", []))),
            main=tuple(sorted(int(x) for x in record.get("main", []))),
            top20=tuple(int(x) for x in record.get("top20", [])),
            pool30=tuple(sorted(int(x) for x in record.get("pool30", []))),
            tickets=tuple(tuple(sorted(int(x) for x in t)) for t in record.get("tickets", [])),
            source_path="",
            full_order=tuple(int(x) for x in record.get("full_order", [])),
        )
    except (ValueError, KeyError, TypeError):
        return None


def _find_shrinkage_tickets(output_dir: Path, period: int) -> Tuple[Tuple[int, ...], ...]:
    """从缩水专报中提取该期完整的 20 注实战票。

    精要报告只列常规 10 注，早期推荐日志也未保存全量票单；
    命中率统计需要统一按 20 注实战口径计算 ROI，故回填缩水专报。
    """
    shrink_dir = Path(output_dir) / "shrinkage"
    if not shrink_dir.exists():
        return ()
    for path in sorted(shrink_dir.glob("Keno30码精细缩水_*.txt")):
        match = _SHRINKAGE_PERIOD_RE.search(path.name)
        if not match or int(match.group(1)) != period:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        tickets = []
        for raw in _TICKET_RE.findall(text):
            nums = _parse_number_list(raw)
            if len(nums) == 5:
                tickets.append(nums)
        if tickets:
            return tuple(tickets)
    return ()


def collect_published_recommendations(output_dir: Path) -> Dict[int, PublishedRecommendation]:
    """汇总日志与历史报告中的推荐，按期号索引。

    日志优先（机器可读且记录首份推荐）；历史报告用于兼容旧引擎时期。
    """
    output_dir = Path(output_dir)
    collected: Dict[int, PublishedRecommendation] = {}

    for period, record in load_journal(output_dir).items():
        rec = _journal_recommendation(record)
        if rec is not None:
            rec = PublishedRecommendation(
                target_period=rec.target_period,
                gold=rec.gold,
                main=rec.main,
                top20=rec.top20,
                pool30=rec.pool30,
                tickets=rec.tickets,
                source_path=_display_path(_journal_path(output_dir)),
                full_order=rec.full_order,
            )
            collected[period] = rec

    reports_dir = output_dir / "reports"
    if reports_dir.exists():
        for path in sorted(reports_dir.glob("Keno预测精要_*.txt")):
            rec = parse_report_file(path)
            if rec is None or rec.target_period in collected:
                continue
            extra_tickets = _find_shrinkage_tickets(output_dir, rec.target_period)
            if len(extra_tickets) > len(rec.tickets):
                rec = dataclasses.replace(rec, tickets=extra_tickets)
            collected[rec.target_period] = rec

    return collected


def compute_recent_hit_stats(
    output_dir: Path,
    dataset: Dataset,
    windows: Sequence[int] = (5, 10, 30),
) -> Tuple[WindowHitStat, ...]:
    """统计最近 N 期已发布推荐的命中均值、Lift 与票单 ROI。"""
    draw_map = {d.period: d for d in dataset.draws}
    recs = collect_published_recommendations(output_dir)
    periods = sorted((p for p in recs if p in draw_map), reverse=True)
    if not periods:
        return ()

    stats: List[WindowHitStat] = []
    for window in windows:
        selected = periods[:window]
        if not selected:
            continue
        reviews = [evaluate_recommendation(recs[p], draw_map[p]) for p in selected]
        n = len(reviews)
        mean_gold = sum(r.gold_hits for r in reviews) / n
        mean_main = sum(r.main_hits for r in reviews) / n
        mean_top20 = sum(r.top20_hits for r in reviews) / n
        mean_pool30 = sum(r.pool30_hits for r in reviews) / n
        cost = sum(r.total_cost_yuan for r in reviews)
        prize = sum(r.total_prize_yuan for r in reviews)
        roi = ((prize - cost) / cost * 100.0) if cost > 0 else 0.0
        aucs = [r.ranking_auc for r in reviews if r.ranking_auc is not None]
        mean_auc = sum(aucs) / len(aucs) if aucs else None
        stats.append(
            WindowHitStat(
                window=window,
                periods=n,
                mean_gold=mean_gold,
                lift_gold=mean_gold / EXPECTED_HITS["gold"],
                mean_main=mean_main,
                lift_main=mean_main / EXPECTED_HITS["main"],
                mean_top20=mean_top20,
                lift_top20=mean_top20 / EXPECTED_HITS["top20"],
                mean_pool30=mean_pool30,
                lift_pool30=mean_pool30 / EXPECTED_HITS["pool30"],
                cost_yuan=cost,
                prize_yuan=prize,
                roi_pct=roi,
                mean_auc=mean_auc,
            )
        )
    return tuple(stats)


# =============================================================================
# 渲染
# =============================================================================


def _format_numbers(numbers: Iterable[int]) -> str:
    """号码列表格式化为两位补零空格串。"""
    return " ".join(f"{n:02d}" for n in numbers)


def format_review_console(result: ReviewResult) -> str:
    """渲染单期复盘控制台/TXT 板块。"""
    lifts = [result.gold_lift, result.main_lift]
    if result.top20_lift is not None:
        lifts.append(result.top20_lift)
    if result.pool30_lift is not None:
        lifts.append(result.pool30_lift)
    avg_lift = sum(lifts) / len(lifts) if lifts else 0.0

    named = [
        ("金胆池(4码)", result.gold_lift),
        ("主选池(12码)", result.main_lift),
    ]
    if result.top20_lift is not None:
        named.append(("Top20", result.top20_lift))
    if result.pool30_lift is not None:
        named.append(("30码大底", result.pool30_lift))
    best_name, best_lift = max(named, key=lambda item: item[1])
    worst_name, worst_lift = min(named, key=lambda item: item[1])

    top20_line = (
        f"  * Top20  命中: {result.top20_hits:02d}/{result.top20_total}  "
        f"(期望 {EXPECTED_HITS['top20']:.1f}, Lift {result.top20_lift:.2f}x)"
        if result.top20_lift is not None
        else "  * Top20  命中: 报告未提供 Top20 明细"
    )
    pool30_line = (
        f"  * 30码大底 命中: {result.pool30_hits:02d}/{result.pool30_total}  "
        f"(期望 {EXPECTED_HITS['pool30']:.1f}, Lift {result.pool30_lift:.2f}x)"
        if result.pool30_lift is not None
        else "  * 30码大底 命中: 报告未提供 30 码清单"
    )

    lines = [
        "【段落 1: 昨日推荐实测复盘】",
        "-" * 80,
        f"  * 复盘对象: 第 {result.target_period} 期推荐 | 来源: {result.source_path}",
        f"  * 实际开奖: 第 {result.actual_period} 期 ({result.actual_date})",
        f"      {_format_numbers(result.actual_numbers)}",
    ]
    if result.latest_draw_period and result.latest_draw_period != result.actual_period:
        lines.append(
            f"  * 断档说明: 最新开奖第 {result.latest_draw_period} 期无已发布推荐（当日未运行），"
            f"本次自动回退复盘最近一份推荐（第 {result.actual_period} 期）。"
        )
    lines.extend([
        "  * 命中核算:",
        f"  * 金胆池  命中: {result.gold_hits:02d}/{result.gold_total}  "
        f"(期望 {EXPECTED_HITS['gold']:.1f}, Lift {result.gold_lift:.2f}x)",
        f"  * 主选池  命中: {result.main_hits:02d}/{result.main_total}  "
        f"(期望 {EXPECTED_HITS['main']:.1f}, Lift {result.main_lift:.2f}x)",
        top20_line,
        pool30_line,
    ])
    if result.ranking_auc is not None:
        lines.append(
            f"  * 排序质量 AUC: {result.ranking_auc:.4f}（0.5=随机；全序 80 码）"
        )
    if result.ticket_reviews:
        win_count = sum(1 for t in result.ticket_reviews if t.prize_yuan > 0)
        lines.append(
            f"  * 推荐票: {len(result.ticket_reviews)} 注 | 最高命中 {result.best_ticket_hits}/5 | "
            f"中奖 {win_count} 注 | 投入 {result.total_cost_yuan} 元 | "
            f"回报 {result.total_prize_yuan} 元 | 盈亏 {result.net_profit_yuan:+d} 元"
        )
    else:
        lines.append("  * 推荐票: 该报告未记录票单")
    lines.append(
        f"  * 复盘结论: 平均 Lift {avg_lift:.2f}x，"
        f"{'整体高于' if avg_lift >= 1.0 else '整体低于'}理论期望；"
        f"最优项 {best_name} ({best_lift:.2f}x)，最弱项 {worst_name} ({worst_lift:.2f}x)。"
    )
    return "\n".join(lines)


def format_hit_rate_console(stats: Sequence[WindowHitStat]) -> str:
    """渲染近期命中率统计板块。"""
    lines = [
        "【段落 8: 近期命中率统计（已发布推荐 × 实际开奖）】",
        "-" * 80,
        "  窗口      样本   金胆均值(Lift)    主选均值(Lift)    Top20均值(Lift)   大底均值(Lift)    票面ROI",
    ]
    for stat in stats:
        lines.append(
            f"  最近{stat.window}期  {stat.periods:>3}    "
            f"{stat.mean_gold:.2f}({stat.lift_gold:.2f}x)      "
            f"{stat.mean_main:.2f}({stat.lift_main:.2f}x)      "
            f"{stat.mean_top20:.2f}({stat.lift_top20:.2f}x)      "
            f"{stat.mean_pool30:.2f}({stat.lift_pool30:.2f}x)     "
            f"{stat.roi_pct:+.1f}%"
        )
    if stats:
        lines.append(
            "  * 理论期望: 金胆 1.00 | 主选 3.00 | Top20 5.00 | 30码大底 7.50；"
            "Lift=1.00 表示与随机抽取持平。"
        )
        auc_line = " | ".join(
            f"最近{s.window}期 {s.mean_auc:.4f}" for s in stats if s.mean_auc is not None
        )
        if auc_line:
            lines.append(f"  * 排序质量 AUC: {auc_line}（0.5=随机；仅统计有全序记录的期次）")
    return "\n".join(lines)