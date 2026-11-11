# -*- coding: utf-8 -*-
"""快乐8 (Keno) 深度学习与图谱贝叶斯端到端预测流水线执行中心。

【业务背景与架构定位】
本模块作为业务调度引擎，统筹调度：
1. 官方数据增量同步与因果加载；
2. PyTorch Transformer 自注意力微调与推理；
3. Dirichlet 贝叶斯后验与共现图谱分析；
4. 30 码黄金大底三级级联缩水漏斗与蒙特卡洛审计；
5. 全渠道三维交付件（TXT 精要、微信文案、商业 Excel、集群三元组）持久化落盘。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from kl8_engine.constants import POINT_BOOST_BONUS
from kl8_engine.data import Dataset, load_dataset, sync_history
from kl8_engine.models.ensemble import FUSION_MODES, EnsembleResult, run_dual_ensemble
from kl8_engine.points import (
    POINTS_MODES,
    PointsAnalysis,
    analyze_points,
    attach_boost_coverage,
    normalize_points,
)
from kl8_engine.reporting import (
    format_console_text_report,
    format_wechat_main_copy,
    generate_excel_workbook,
)
from kl8_engine.review import (
    ReviewResult,
    WindowHitStat,
    append_prediction_journal,
    build_review,
    compute_recent_hit_stats,
)
from kl8_engine.shrinkage import (
    CascadeShrinkageEngine,
    ShrinkageReport,
    extract_cluster_triplets,
    format_shrinkage_console,
    format_wechat_shrinkage,
)


@dataclass(frozen=True)
class PipelineResult:
    """流水线执行终态交付产出物。"""

    target_period: int
    ensemble: EnsembleResult
    shrinkage: ShrinkageReport
    cluster_triplets: Tuple[str, ...]
    regular_tickets: Tuple[Tuple[int, ...], ...]
    console_report: str
    wechat_main: str
    wechat_shrinkage: str
    shrinkage_console: str
    report_paths: Dict[str, Path]
    elapsed_seconds: float
    warnings: Tuple[str, ...]
    review: Optional[ReviewResult] = None
    points_analysis: Optional[PointsAnalysis] = None
    points_mode: str = "compare"
    fusion_mode: str = "borda"
    comparison: bool = True
    hit_stats: Tuple[WindowHitStat, ...] = ()
    comparison_ensemble: Optional[EnsembleResult] = None
    comparison_shrinkage: Optional[ShrinkageReport] = None
    comparison_fusion_mode: Optional[str] = None


def run_pipeline(
    history_path: str = "kl8_history.txt",
    output_dir: str = "output",
    no_sync: bool = False,
    dl_weight: float = 0.55,
    seq_len: int = 10,
    epochs: int = 25,
    budget_shrinkage: int = 20,
    points: Optional[Sequence[int]] = None,
    points_mode: str = "compare",
    fusion_mode: str = "borda",
    comparison: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """端到端执行快乐8完整预测、缩水与全渠道交付流水线。

    @param history_path 历史数据文件路径
    @param output_dir 输出根目录
    @param no_sync 是否跳过网络增量同步
    @param dl_weight 深度学习 Transformer 分数权重
    @param seq_len 时序窗口大小
    @param epochs 模型微调轮数
    @param budget_shrinkage 缩水实战注数预算
    @param points 用户点位号码（可空）
    @param points_mode compare=仅对比标注 / boost=点位参与缩水覆盖加权
    @param fusion_mode borda=四分量秩融合（默认）/ weighted=原始加权和
    @param comparison 是否同时生成另一融合方式的并排对比版
    @param progress 步骤进度回调
    @return 结构化完整流水线成果 PipelineResult
    """
    if points_mode not in POINTS_MODES:
        raise ValueError(f"未知点位模式: {points_mode}，可选 {POINTS_MODES}")
    if fusion_mode not in FUSION_MODES:
        raise ValueError(f"未知融合方式: {fusion_mode}，可选 {FUSION_MODES}")
    normalized_points = normalize_points(points)

    t0 = time.perf_counter()
    accumulated_warnings: List[str] = []

    def _step(msg: str) -> None:
        if progress is not None:
            progress(msg)

    # 步骤 1: 增量开奖数据同步
    if not no_sync:
        _step("[1/5] 增量抓取最新官方开奖数据...")
        added_count, sync_warns = sync_history(history_path)
        accumulated_warnings.extend(sync_warns)
    else:
        _step("[1/5] 跳过网络同步，直接加载本地数据...")

    # 步骤 2: 加载时序数据集并进行因果断言
    _step("[2/5] 加载开奖历史并构建因果状态矩阵...")
    dataset = load_dataset(history_path)
    target_period = dataset.target_period()

    # 步骤 2.5: 复盘上一期已发布推荐，并统计近期命中率
    _step("[2.5/5] 复盘上期已发布推荐并统计近期命中率...")
    review = build_review(Path(output_dir), dataset)
    hit_stats = compute_recent_hit_stats(Path(output_dir), dataset)

    # 步骤 3: 运行 PyTorch Transformer 与图谱贝叶斯混合集成模型
    _step("[3/5] 执行 Transformer 深度学习微调与图谱贝叶斯集成预测（双融合对比）...")
    comparison_mode = None
    if comparison:
        comparison_mode = "weighted" if fusion_mode == "borda" else "borda"
    ensemble, comparison_ensemble = run_dual_ensemble(
        dataset,
        official_mode=fusion_mode,
        comparison_mode=comparison_mode,
        dl_weight=dl_weight,
        seq_len=seq_len,
        epochs=epochs,
    )

    # 步骤 3.5: 用户点位与模型输出一致性分析
    points_analysis: Optional[PointsAnalysis] = None
    if normalized_points:
        points_analysis = analyze_points(normalized_points, ensemble, dataset, points_mode)

    # 步骤 4: 运行 30 码黄金大底三级级联缩水漏斗与蒙特卡洛审计
    _step("[4/5] 运行 30 码黄金大底三级级联精细缩水与蒙特卡洛审计...")
    kinetic_scores = {n: float(ensemble.scores[n - 1]) for n in ensemble.shrinkage_pool_30}
    if points_mode == "boost":
        for num in normalized_points:
            if num in kinetic_scores:
                kinetic_scores[num] += POINT_BOOST_BONUS
    shrink_engine = CascadeShrinkageEngine(
        pool_30=ensemble.shrinkage_pool_30,
        kinetic_scores=kinetic_scores,
    )
    shrinkage_report = shrink_engine.generate_report(budget=budget_shrinkage)

    comparison_shrinkage: Optional[ShrinkageReport] = None
    if comparison_ensemble is not None:
        comparison_engine = CascadeShrinkageEngine(
            pool_30=comparison_ensemble.shrinkage_pool_30,
            kinetic_scores={
                n: float(comparison_ensemble.scores[n - 1])
                for n in comparison_ensemble.shrinkage_pool_30
            },
        )
        comparison_shrinkage = comparison_engine.generate_report(budget=budget_shrinkage)

    if points_analysis is not None and points_mode == "boost":
        points_analysis = attach_boost_coverage(
            points_analysis,
            ensemble.shrinkage_pool_30,
            shrinkage_report.tickets,
        )

    # 记录每期首份推荐（防止开奖后重跑污染复盘）
    append_prediction_journal(
        Path(output_dir),
        target_period=target_period,
        gold=ensemble.gold_pool,
        main=ensemble.main_pool,
        top20=ensemble.full_order[:20],
        pool30=ensemble.shrinkage_pool_30,
        tickets=shrinkage_report.tickets,
        points=normalized_points,
        points_mode=points_mode,
        full_order=ensemble.full_order,
        fusion_mode=fusion_mode,
    )

    # 生成选五常规推荐票（取缩水精选的前 10 注或直接使用缩水票）
    regular_tickets = shrinkage_report.tickets[:10]

    # 提取格式四铁律集群三元组
    triplets = tuple(extract_cluster_triplets(ensemble.full_order, top_n=8))

    # 步骤 5: 渲染全渠道三维交付物并持久化落盘
    _step("[5/5] 渲染并落盘全渠道三维交付物（TXT/微信/Excel/缩水报表）...")
    out_path = Path(output_dir)
    rep_dir = out_path / "reports"
    wechat_dir = out_path / "wechat"
    excel_dir = out_path / "excel"
    shrink_dir = out_path / "shrinkage"

    for d in [rep_dir, wechat_dir, excel_dir, shrink_dir]:
        d.mkdir(parents=True, exist_ok=True)

    import datetime
    target_date_str = datetime.date.today().strftime("%Y%m%d")

    txt_file = rep_dir / f"Keno预测精要_{target_date_str}_第{target_period}期.txt"
    wechat_file = wechat_dir / f"Keno微信发布_{target_date_str}_第{target_period}期.txt"
    excel_file = excel_dir / f"Keno动力学决策_{target_date_str}_第{target_period}期.xlsx"
    shrink_txt_file = shrink_dir / f"Keno30码精细缩水_{target_date_str}_第{target_period}期.txt"
    shrink_wechat_file = wechat_dir / f"Keno微信30码缩水_{target_date_str}_第{target_period}期.txt"

    # 渲染文本
    console_report = format_console_text_report(
        ensemble,
        dataset,
        shrinkage_report,
        regular_tickets,
        review=review,
        points_analysis=points_analysis,
        points_mode=points_mode,
        fusion_mode=fusion_mode,
        hit_stats=hit_stats,
        comparison_ensemble=comparison_ensemble,
        comparison_shrinkage=comparison_shrinkage,
        comparison_fusion_mode=comparison_mode,
    )
    wechat_main = format_wechat_main_copy(ensemble, dataset, regular_tickets)
    shrinkage_console = format_shrinkage_console(shrinkage_report, target_period)
    wechat_shrinkage = format_wechat_shrinkage(shrinkage_report, target_period)

    # 写入磁盘
    txt_file.write_text(console_report, encoding="utf-8")
    wechat_file.write_text(wechat_main, encoding="utf-8")
    shrink_txt_file.write_text(shrinkage_console, encoding="utf-8")
    shrink_wechat_file.write_text(wechat_shrinkage, encoding="utf-8")

    # 生成商业 Excel
    generate_excel_workbook(
        ensemble,
        dataset,
        shrinkage_report,
        str(excel_file),
        review=review,
        points_analysis=points_analysis,
        comparison_ensemble=comparison_ensemble,
        comparison_shrinkage=comparison_shrinkage,
        official_fusion_mode=fusion_mode,
        comparison_fusion_mode=comparison_mode,
    )

    report_paths = {
        "text": txt_file,
        "wechat": wechat_file,
        "excel": excel_file,
        "shrinkage_txt": shrink_txt_file,
        "shrinkage_wechat": shrink_wechat_file,
    }

    elapsed = time.perf_counter() - t0

    return PipelineResult(
        target_period=target_period,
        ensemble=ensemble,
        shrinkage=shrinkage_report,
        cluster_triplets=triplets,
        regular_tickets=regular_tickets,
        console_report=console_report,
        wechat_main=wechat_main,
        wechat_shrinkage=wechat_shrinkage,
        shrinkage_console=shrinkage_console,
        report_paths=report_paths,
        elapsed_seconds=elapsed,
        warnings=tuple(accumulated_warnings),
        review=review,
        points_analysis=points_analysis,
        points_mode=points_mode,
        fusion_mode=fusion_mode,
        hit_stats=hit_stats,
        comparison_ensemble=comparison_ensemble,
        comparison_shrinkage=comparison_shrinkage,
        comparison_fusion_mode=comparison_mode,
    )
