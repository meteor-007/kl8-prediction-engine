"""全流程端到端业务执行流水线（契约 R48、§3.14）。

【核心入口】
1. run_daily: 日常实盘预测端到端全流程（多源同步 -> 状态构建 -> 五层特征打分 -> 组合覆盖 -> 三维报表落盘 -> 幂等自适应归档）；
2. run_backtest: 滚动时序前瞻 Walk-Forward 回测入口；
3. run_audit: 全仓防穿越与模型自洽性审计入口。
"""

from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from kl8.config import AppConfig, merge_cli_overrides
from kl8.datatypes import BacktestReport, Dataset, FeatureSnapshot, PoolResult, TicketPlan
from kl8.errors import DataFormatError, InsufficientDataError, KL8Error
from kl8.evaluation.leakage import audit_grep_no_shuffle
from kl8.evaluation.walkforward import run_walkforward
from kl8.history import load_dataset, truncated
from kl8.layer1_representation import BinaryStateMatrix
from kl8.layer1_representation.windows import resolve_windows
from kl8.layer2_baseline import randomness_report
from kl8.layer3_features import build_feature_snapshot
from kl8.layer4_scoring import ScoreRoundResult, adaptive, score_round
from kl8.layer5_combinatorial import build_tickets
from kl8.reporting import (
    ReportContext,
    render_excel,
    render_text_report,
    render_wechat,
    save_text_report,
    save_wechat,
)
from kl8.sync import sync_history


@dataclass(frozen=True)
class DailyResult:
    """日常预测全流程执行交付结果（契约 §3.14）。"""

    target_period: int
    """预测目标期号。"""

    target_date: str
    """预测目标开奖日期。"""

    pool: PoolResult
    """胆组与精选复式大底。"""

    plans: tuple[TicketPlan, ...]
    """覆盖注单推荐方案。"""

    report_paths: Mapping[str, Path]
    """三类交付件及状态文件的保存路径，包含 "text", "wechat", "excel", "state"。"""

    backtest: BacktestReport | None
    """关联的滚动回测报告。"""

    warnings: tuple[str, ...]
    """执行过程中的告警信息。"""

    elapsed_s: float
    """总耗时（秒）。"""


def run_daily(
    cfg: AppConfig,
    *,
    no_sync: bool = False,
    target_period: int | None = None,
    target_date: str | None = None,
    backtest_n: int = 30,
    quick: bool = True,
    progress: Callable[[str], None] | None = None,
) -> DailyResult:
    """执行日常实盘预测端到端全流程（契约 R48、§3.14）。

    【执行步骤】
    步骤 1: 开奖数据增量同步（支持无网络安全降级）；
    步骤 2: 加载全量历史数据并确定预测目标期号与开奖日期；
    步骤 3: 严格截断历史数据集构建 Layer 1 状态矩阵与 Layer 3 动量特征快照；
    步骤 4: 执行 Layer 2 宏观随机性检验与基准对齐；
    步骤 5: 加载磁盘幂等自适应权重，执行 Layer 4 综合打分与规则判定；
    步骤 6: 执行 Layer 5 组合覆盖优化生成保底注单；
    步骤 7: 执行可选滚动时序回测评估（Walk-Forward），并幂等落盘自适应状态；
    步骤 8: 渲染并落盘全渠道三维交付件（TXT 精要、微信发布文案、Excel 商业报表）。

    @param cfg 全局配置
    @param no_sync 是否跳过网络增量同步
    @param target_period 指定预测期号，None 时默认为历史最新期 + 1
    @param target_date 指定预测开奖日期，None 时默认为最新日期 + 1 天
    @param backtest_n 伴随滚动回测期数（0 表示关闭）
    @param quick 增量同步模式（True 极速抓取最新，False 全量校验）
    @return 结构化日常预测交付件结果 DailyResult
    """
    start_time = time.perf_counter()
    accumulated_warnings: list[str] = []

    def _step(idx: int, text: str) -> None:
        """打印单步进度（无回调时静默，供库调用方复用）。"""
        if progress is not None:
            progress(f"[{idx}/8] {text}")

    # 步骤 1: 增量开奖数据同步
    _step(1, "开奖数据增量同步" + ("（已跳过）" if no_sync else ""))
    if not no_sync:
        try:
            sync_res = sync_history(cfg.history_path, quick=quick)
            if sync_res.warnings:
                accumulated_warnings.extend(sync_res.warnings)
        except Exception as e:
            accumulated_warnings.append(f"开奖数据同步异常降级（使用本地历史）: {e}")

    # 步骤 2: 加载全量历史数据
    _step(2, "加载历史开奖数据")
    ds = load_dataset(cfg.history_path)
    if not ds.draws:
        raise DataFormatError(f"历史数据文件为空或不存在有效开奖记录: {cfg.history_path}")

    latest_draw = ds.draws[-1]
    if target_period is None:
        eff_target_period = latest_draw.period + 1
    else:
        eff_target_period = int(target_period)

    if target_date is None:
        try:
            dt = datetime.strptime(latest_draw.date, "%Y-%m-%d")
            eff_target_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            eff_target_date = "2026-11-02"
    else:
        eff_target_date = str(target_date)

    # 步骤 3: 截断历史并构建状态矩阵与特征快照
    _step(3, "构建 Layer1 状态矩阵与 Layer3 多尺度特征")
    cur_ds = truncated(ds, eff_target_period)
    state = BinaryStateMatrix.from_dataset(cur_ds)
    end_idx = len(cur_ds.draws) - 1
    snapshot = build_feature_snapshot(
        state,
        end=end_idx,
        cfg=cfg,
        include_ml=cfg.ml.enabled,
    )

    # 步骤 4: Layer 2 宏观随机性与基准分析
    _step(4, "Layer2 随机性检验与超几何基准对齐")
    win_specs = resolve_windows(cfg.windows, len(cur_ds.draws))
    rand_report = randomness_report(state, end_idx, win_specs)

    # 步骤 5: 加载磁盘幂等自适应权重，执行 Layer 4 综合打分
    _step(5, "Layer4 五规则共振打分与自适应权重")
    adaptive_state, a_warns = adaptive.load_state(
        cfg.adaptive.state_path,
        cfg.adaptive.schema_version,
    )
    if a_warns:
        accumulated_warnings.extend(a_warns)

    score_round_res = score_round(
        snapshot,
        cfg,
        adaptive_state=adaptive_state,
    )

    # 步骤 6: Layer 5 组合覆盖优化与注单生成
    _step(6, "Layer5 组合覆盖设计与保底注单生成")
    plans = build_tickets(
        score_round_res.pool,
        score_round_res.scores.final,
        cfg,
    )

    # 步骤 7: 伴随时序回测与自适应权重更新
    bt_report: BacktestReport | None = None
    if backtest_n > 0:
        _step(7, f"伴随 Walk-Forward 回测（{backtest_n} 期）与自适应权重更新")
        try:
            bt_report = run_walkforward(
                cur_ds,
                cfg,
                n_test=backtest_n,
                include_tickets=False,
            )
            # 状态更新并原子落盘
            new_adaptive_state, _ = adaptive.next_weights(
                bt_report.rows,
                adaptive_state,
                cfg.adaptive,
                period=eff_target_period,
            )
            adaptive.save_state(new_adaptive_state, cfg.adaptive.state_path)
        except Exception as e:
            accumulated_warnings.append(f"伴随回测与自适应权重更新跳过: {e}")

    # 步骤 8: 渲染并保存三类交付件
    _step(8, "渲染并落盘 TXT/微信/Excel 三类交付件")
    out_dir = Path(cfg.output_dir)
    date_compact = eff_target_date.replace("-", "")

    txt_filename = f"Keno预测精要_{date_compact}_第{eff_target_period}期.txt"
    wechat_filename = f"Keno微信发布_{date_compact}_第{eff_target_period}期.txt"
    excel_filename = f"Keno动力学决策_{date_compact}_第{eff_target_period}期.xlsx"

    txt_path = out_dir / "reports" / txt_filename
    wechat_path = out_dir / "wechat" / wechat_filename
    excel_path = out_dir / "excel" / excel_filename
    state_path = Path(cfg.adaptive.state_path)

    # 提取上期复盘信息
    previous_review_dict: dict[str, Any] = {}
    if len(ds.draws) >= 2:
        last_d = ds.draws[-1]
        previous_review_dict = {
            "period": last_d.period,
            "date": last_d.date,
            "hit_count": len(set(score_round_res.pool.gold) & last_d.numbers),
        }

    report_ctx = ReportContext(
        cfg=cfg,
        ds=cur_ds,
        target_period=eff_target_period,
        target_date=eff_target_date,
        snapshot=snapshot,
        rounds=score_round_res,
        plans=plans,
        randomness=rand_report,
        backtest=bt_report,
        previous_review=previous_review_dict,
        warnings=tuple(accumulated_warnings),
    )

    # 渲染与持久化
    txt_content = render_text_report(report_ctx)
    save_text_report(txt_content, txt_path)

    wechat_content = render_wechat(report_ctx)
    save_wechat(wechat_content, wechat_path)

    render_excel(report_ctx, excel_path)

    report_paths = {
        "text": txt_path,
        "wechat": wechat_path,
        "excel": excel_path,
        "state": state_path,
    }

    elapsed = time.perf_counter() - start_time

    return DailyResult(
        target_period=eff_target_period,
        target_date=eff_target_date,
        pool=score_round_res.pool,
        plans=plans,
        report_paths=report_paths,
        backtest=bt_report,
        warnings=tuple(accumulated_warnings),
        elapsed_s=elapsed,
    )


def run_backtest(
    cfg: AppConfig,
    n: int,
    *,
    t0: int | None = None,
    include_tickets: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestReport:
    """独立运行 Walk-Forward 滚动时序前瞻回测流水线（契约 R48、§3.14）。

    @param cfg 全局配置
    @param n 评估期数
    @param t0 冷启动期数
    @param include_tickets 是否生成注单并核算奖金，默认读取 cfg.backtest.include_tickets
    @param progress 进度回调函数，入参为 (已完成步数, 总步数)；为 None 时不打印
    @return 结构化完整回测报告
    """
    ds = load_dataset(cfg.history_path)
    with_tickets = (
        bool(include_tickets)
        if include_tickets is not None
        else bool(cfg.backtest.include_tickets)
    )
    return run_walkforward(
        ds,
        cfg,
        n_test=n,
        t0=t0,
        include_tickets=with_tickets,
        progress=progress,
    )


def run_audit(cfg: AppConfig, *, n: int = 100) -> dict[str, Any]:
    """执行系统级代码防穿越与统计模型自洽性审计（契约 R48、§3.14）。

    @param cfg 全局配置
    @param n 审计评估样本量
    @return 审计检验指标报告字典
    """
    violations = audit_grep_no_shuffle("kl8")

    ds = load_dataset(cfg.history_path)
    bt_report = run_walkforward(
        ds,
        cfg,
        n_test=n,
        include_tickets=False,
    )

    return {
        "code_audit_clean": len(violations) == 0,
        "code_violations": violations,
        "backtest_self_check": bt_report.self_check,
        "backtest_samples": bt_report.n_sample,
        "lift_at_k": {k: v.lift for k, v in bt_report.lift.items()},
        "config_hash": bt_report.config_hash,
    }
