"""全流程端到端业务执行流水线（契约 R48、§3.14、§3.14.1）。

【核心入口】
1. ``run_daily``: 日常实盘预测端到端全流程（多源同步 → 状态构建 → 五层特征打分 →
   组合覆盖 → 三维报表落盘 → 幂等自适应归档 → **上期预测复盘状态落盘**）；
2. ``run_backtest``: 滚动时序前瞻 Walk-Forward 回测入口；
3. ``run_audit``: 全仓防穿越与模型自洽性审计入口。

【§3.14.1 复盘状态契约（P8-H5 修复）】
``output/state/prev_prediction.json`` 记录**每期自己的预测**，下一期据此复盘真实命中；
**禁止**用"当期预测池 ∩ 最新开奖号"冒充上期命中（旧实现正是如此，属数据造假）。
缺失/损坏/未开奖时报告固定输出「首期无复盘基线（无可用的上期预测记录）」且不输出
任何命中数。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from kl8.config import AppConfig, merge_cli_overrides
from kl8.constants import NO_BASELINE_TEXT
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

#: 复盘状态文件 schema 版本（契约 §3.14.1 固定为 1）。
PREV_PREDICTION_SCHEMA_VERSION: int = 1

#: 首期无复盘基线固定文案（契约 §3.14.1：缺文件/损坏/无更早记录时必须逐字输出）。
NO_BASELINE_TEXT: str = "首期无复盘基线（无可用的上期预测记录）"

#: 状态文件相对 output_dir 的路径片段。
_PREV_PREDICTION_RELPATH: tuple[str, str] = ("state", "prev_prediction.json")

#: 允许的异常类型（禁止裸 except，P8-M7）。
_STATE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    json.JSONDecodeError,
)


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
    """交付件路径，包含 "text", "wechat", "excel", "state", "prev_prediction"。"""

    backtest: BacktestReport | None
    """关联的滚动回测报告。"""

    warnings: tuple[str, ...]
    """执行过程中的告警信息。"""

    elapsed_s: float
    """总耗时（秒）。"""

    previous_review: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    """上期复盘结果（``status``/``note``/命中数），报告层只读本字段。"""


def prev_prediction_path(cfg: AppConfig) -> Path:
    """返回复盘状态文件路径 ``<output_dir>/state/prev_prediction.json``。

    @param cfg 全局配置
    @return 文件路径
    """
    return Path(cfg.output_dir).joinpath(*_PREV_PREDICTION_RELPATH)


def _plan_to_payload(plan: TicketPlan) -> dict[str, Any]:
    """把注单方案序列化为 JSON 友好的字典（契约 §3.14.1 plans 字段）。

    @param plan 注单方案
    @return 可 JSON 序列化字典
    """
    level = dict(plan.guarantee_level)
    return {
        "play": plan.play,
        "k": int(plan.k),
        "budget": int(plan.budget),
        "t": int(level.get("t", 0)),
        "m": int(level.get("m", 0)),
        "n_tickets": int(plan.n_tickets),
        "tickets": [[int(num) for num in ticket] for ticket in plan.tickets],
        "cost_yuan": int(plan.cost_yuan),
    }


def write_prev_prediction(
    cfg: AppConfig,
    *,
    period: int,
    target_date: str,
    gold: Sequence[int],
    main: Sequence[int],
    plans: Sequence[TicketPlan],
    created_at: str | None = None,
) -> tuple[Path, tuple[str, ...]]:
    """幂等 upsert 本期预测到复盘状态文件（原子写，契约 §3.14.1）。

    【幂等】按 ``str(period)`` 覆盖；若除 ``created_at`` 外内容与已有记录完全一致，
    沿用旧 ``created_at``，保证同一期反复运行产生**逐字节相同**的文件。

    @param cfg 全局配置
    @param period 本期期号
    @param target_date 本期开奖日期
    @param gold 金胆池（1-based）
    @param main 复式池（1-based）
    @param plans 注单方案
    @param created_at 覆盖创建时间（测试用）
    @return (状态文件路径, 告警元组)
    @raises OSError 落盘失败（调用方须降级为告警，不中断主流程）
    """
    path = prev_prediction_path(cfg)
    warnings: list[str] = []
    payload = _read_prev_payload(path, warnings)
    predictions: dict[str, dict[str, Any]] = payload["predictions"]

    record: dict[str, Any] = {
        "period": int(period),
        "target_date": str(target_date),
        "gold": [int(num) for num in gold],
        "main": [int(num) for num in main],
        "plans": [_plan_to_payload(plan) for plan in plans],
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
    }
    key = str(int(period))
    previous = predictions.get(key)
    changed = True
    if previous is not None:
        previous_core = {k: v for k, v in previous.items() if k != "created_at"}
        record_core = {k: v for k, v in record.items() if k != "created_at"}
        if previous_core == record_core:
            record["created_at"] = str(previous.get("created_at", record["created_at"]))
            changed = False
    predictions[key] = record

    if not changed:
        # 幂等 no-op：同一期同内容重复运行不触碰文件（保持逐字节一致与 mtime 稳定）
        return (path, tuple(warnings))

    payload["schema_version"] = PREV_PREDICTION_SCHEMA_VERSION
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", suffix=".tmp"
    )
    tmp_name = handle.name
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return (path, tuple(warnings))


def _empty_payload() -> dict[str, Any]:
    """构造空的复盘状态文档。

    @return {"schema_version":1,"updated_at":"","predictions":{}}
    """
    return {
        "schema_version": PREV_PREDICTION_SCHEMA_VERSION,
        "updated_at": "",
        "predictions": {},
    }


def _read_prev_payload(path: Path, warnings: list[str]) -> dict[str, Any]:
    """读取复盘状态文件；缺失/损坏/版本不符时备份并返回空文档（不外抛）。

    @param path 状态文件路径
    @param warnings 告警收集列表（原地追加）
    @return 合法文档字典
    """
    if not path.exists():
        return _empty_payload()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("顶层不是 JSON 对象")
        if int(payload.get("schema_version", -1)) != PREV_PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"schema_version={payload.get('schema_version')} 与契约 1 不符")
        predictions = payload.get("predictions")
        if not isinstance(predictions, dict):
            raise ValueError("predictions 不是对象")
        return payload
    except _STATE_ERRORS as exc:
        backup = path.with_name(path.name + ".bak")
        try:
            os.replace(path, backup)
            warnings.append(f"复盘状态文件损坏（{exc}），已备份至 {backup.name} 并按空状态继续")
        except OSError as backup_exc:
            warnings.append(f"复盘状态文件损坏（{exc}）且备份失败（{backup_exc}），按空状态继续")
        return _empty_payload()


def read_prev_prediction(
    cfg: AppConfig,
    *,
    target_period: int,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """读取**严格早于**目标期的最近一条预测记录（契约 §3.14.1）。

    @param cfg 全局配置
    @param target_period 当前目标期号
    @return (记录字典或 None, 告警元组)
    """
    path = prev_prediction_path(cfg)
    warnings: list[str] = []
    payload = _read_prev_payload(path, warnings)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for key, value in payload.get("predictions", {}).items():
        try:
            period = int(key)
        except (TypeError, ValueError):
            warnings.append(f"复盘状态中存在非法期号键 {key!r}，已忽略")
            continue
        if period < int(target_period) and isinstance(value, dict):
            candidates.append((period, value))
    if not candidates:
        return (None, tuple(warnings))
    candidates.sort(key=lambda item: item[0])
    return (candidates[-1][1], tuple(warnings))


def _build_review(
    cfg: AppConfig,
    ds: Dataset,
    *,
    target_period: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """构造上期复盘结果（只读上一期自己的预测记录，绝不与当期池混用）。

    @param cfg 全局配置
    @param ds 当前截断数据集（只含 ≤ target_period-1 的历史）
    @param target_period 当前目标期号
    @return (复盘结果字典, 告警元组)
    """
    record, warnings = read_prev_prediction(cfg, target_period=target_period)
    if record is None:
        return (
            {
                "status": "baseline",
                "period": 0,
                "target_date": "",
                "gold": (),
                "main": (),
                "gold_hit": -1,
                "main_hit": -1,
                "best_ticket_hit": -1,
                "note": NO_BASELINE_TEXT,
            },
            warnings,
        )

    period = int(record.get("period", 0))
    draw = next((item for item in ds.draws if item.period == period), None)
    if draw is None:
        return (
            {
                "status": "pending",
                "period": period,
                "target_date": str(record.get("target_date", "")),
                "gold": tuple(int(x) for x in record.get("gold", ())),
                "main": tuple(int(x) for x in record.get("main", ())),
                "gold_hit": -1,
                "main_hit": -1,
                "best_ticket_hit": -1,
                "note": "上期预测尚未开奖（历史中无可核对开奖记录）",
            },
            warnings,
        )

    drawn = set(int(x) for x in draw.numbers)
    gold = tuple(int(x) for x in record.get("gold", ()))
    main = tuple(int(x) for x in record.get("main", ()))
    best_ticket_hit = 0
    for plan in record.get("plans", ()):
        if not isinstance(plan, dict):
            continue
        for ticket in plan.get("tickets", ()):
            best_ticket_hit = max(best_ticket_hit, len(set(int(x) for x in ticket) & drawn))
    return (
        {
            "status": "ok",
            "period": period,
            "target_date": str(record.get("target_date", "")),
            "gold": gold,
            "main": main,
            "gold_hit": len(set(gold) & drawn),
            "main_hit": len(set(main) & drawn),
            "best_ticket_hit": best_ticket_hit,
            "note": "",
        },
        warnings,
    )


def _resolve_target_date(
    ds: Dataset,
    latest_date: str,
    warnings: list[str],
) -> str:
    """由最新开奖日期推导下一期日期；无法解析时回溯最近合法日期。

    @param ds 历史数据集
    @param latest_date 最新一期日期字符串
    @param warnings 告警收集列表（原地追加）
    @return "YYYY-MM-DD" 格式日期
    @raises DataFormatError 全部历史日期均不可解析
    """
    for draw in reversed(ds.draws):
        try:
            dt = datetime.strptime(str(draw.date).strip(), "%Y-%m-%d")
        except (TypeError, ValueError):
            continue
        if str(draw.date) != latest_date:
            warnings.append(
                f"最新一期日期 {latest_date!r} 无法解析，已回溯到第 {draw.period} 期的合法日期 {draw.date}"
            )
        return (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    raise DataFormatError(f"历史数据中没有任何合法 YYYY-MM-DD 日期（最新值 {latest_date!r}）")


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
    """执行日常实盘预测端到端全流程（契约 R48、§3.14、§3.14.1）。

    【执行步骤】
    步骤 1: 开奖数据增量同步（支持无网络安全降级）；
    步骤 2: 加载全量历史数据并确定预测目标期号与开奖日期；
    步骤 3: 严格截断历史数据集构建 Layer 1 状态矩阵与 Layer 3 动量特征快照；
    步骤 4: 执行 Layer 2 宏观随机性检验与基准对齐；
    步骤 5: 加载磁盘幂等自适应权重，执行 Layer 4 综合打分与规则判定；
    步骤 6: 执行 Layer 5 组合覆盖优化生成保底注单；
    步骤 7: 执行可选滚动时序回测评估（Walk-Forward），并幂等落盘自适应状态；
    步骤 8: 渲染并落盘全渠道三维交付件（TXT 精要、微信发布文案、Excel 商业报表），
            并幂等落盘本期预测到复盘状态文件（供下一期真实复盘）。

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
        except (OSError, KL8Error) as exc:
            accumulated_warnings.append(f"开奖数据同步异常降级（使用本地历史）: {exc}")

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
        eff_target_date = _resolve_target_date(ds, latest_draw.date, accumulated_warnings)
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
            new_adaptive_state, aw = adaptive.next_weights(
                bt_report.rows,
                adaptive_state,
                cfg.adaptive,
                period=eff_target_period,
            )
            accumulated_warnings.extend(aw)
            try:
                adaptive.save_state(new_adaptive_state, cfg.adaptive.state_path)
            except OSError as exc:
                accumulated_warnings.append(f"自适应权重状态落盘失败（不影响本期出单）: {exc}")
        except (KL8Error, OSError) as exc:
            accumulated_warnings.append(f"伴随回测与自适应权重更新跳过: {exc}")

    # 步骤 8: 渲染并保存三类交付件 + 复盘状态
    _step(8, "渲染并落盘 TXT/微信/Excel 三类交付件与复盘状态")
    out_dir = Path(cfg.output_dir)
    date_compact = eff_target_date.replace("-", "")

    txt_path = out_dir / "reports" / f"Keno预测精要_{date_compact}_第{eff_target_period}期.txt"
    wechat_path = out_dir / "wechat" / f"Keno微信发布_{date_compact}_第{eff_target_period}期.txt"
    excel_path = out_dir / "excel" / f"Keno动力学决策_{date_compact}_第{eff_target_period}期.xlsx"
    state_path = Path(cfg.adaptive.state_path)

    # 上期复盘：只读上一期自己的预测记录（§3.14.1，禁止与当期池混用）
    previous_review, review_warns = _build_review(cfg, cur_ds, target_period=eff_target_period)
    accumulated_warnings.extend(review_warns)

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
        previous_review=previous_review,
        warnings=tuple(accumulated_warnings),
    )

    # 渲染与持久化（落盘失败必须抛错，禁止"print 后照常返回路径"）
    txt_content = render_text_report(report_ctx)
    save_text_report(txt_content, txt_path)

    wechat_content = render_wechat(report_ctx)
    save_wechat(wechat_content, wechat_path)

    report_paths: dict[str, Path] = {
        "text": txt_path,
        "wechat": wechat_path,
        "state": state_path,
    }
    try:
        render_excel(report_ctx, excel_path)
        report_paths["excel"] = excel_path
    except (OSError, KL8Error) as exc:
        accumulated_warnings.append(f"Excel 报表生成失败（TXT/微信已成功落盘）: {exc}")

    # 步骤 8b: 幂等落盘本期预测（供下一期真实复盘，§3.14.1）
    try:
        prev_path, p_warns = write_prev_prediction(
            cfg,
            period=eff_target_period,
            target_date=eff_target_date,
            gold=score_round_res.pool.gold,
            main=score_round_res.pool.main,
            plans=plans,
        )
        accumulated_warnings.extend(p_warns)
        report_paths["prev_prediction"] = prev_path
    except OSError as exc:
        accumulated_warnings.append(f"复盘状态文件落盘失败（不影响本期出单）: {exc}")

    elapsed = time.perf_counter() - start_time
    clean_warnings = tuple(dict.fromkeys(accumulated_warnings))

    return DailyResult(
        target_period=eff_target_period,
        target_date=eff_target_date,
        pool=score_round_res.pool,
        plans=plans,
        report_paths=report_paths,
        backtest=bt_report,
        warnings=clean_warnings,
        elapsed_s=elapsed,
        previous_review=previous_review,
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
    @return 审计检验指标报告字典（含真实穿越计数与自检判定）
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
        "backtest_self_check": dict(bt_report.self_check),
        "backtest_samples": bt_report.n_sample,
        "lookahead_checked": bt_report.lookahead_checked,
        "lookahead_violations": list(bt_report.lookahead_violations),
        "lift_at_k": {k: v.lift for k, v in bt_report.lift.items()},
        "lift_signals": {k: v.is_signal for k, v in bt_report.lift.items()},
        "config_hash": bt_report.config_hash,
    }