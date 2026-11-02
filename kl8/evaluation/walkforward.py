"""Walk-Forward 滚动前瞻时序回测执行引擎（契约 R42、§3.12）。

【核心机制】
1. 严禁使用随机打乱交叉验证（如 K-Fold / StratifiedKFold / train_test_split），必须严格遵循时间单向因果流；
2. T0 冷启动期：使用历史 [0..t] 训练并生成决策，预测并与 t+1 期真实开奖比对；
3. 防穿越守卫：通过 history.truncated(ds, target_period) 物理隔绝未来信息；
4. 自适应状态：纯内存递推更新，保证回测过程不污染外部磁盘状态文件。
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Iterator, Sequence

import numpy as np

from kl8.config import AppConfig, config_hash
from kl8.constants import DEFAULT_PRIZE_TABLE
from kl8.datatypes import BacktestReport, BacktestRow, Dataset
from kl8.errors import InsufficientDataError
from kl8.evaluation.leakage import assert_no_lookahead
from kl8.evaluation.metrics import bankroll, high_tier_rate, lift_at_k, rule_attribution, self_check
from kl8.history import truncated
from kl8.layer1_representation import BinaryStateMatrix
from kl8.layer3_features import build_feature_snapshot
from kl8.layer4_scoring import AdaptiveState, adaptive, score_round
from kl8.layer5_combinatorial import build_tickets


def iter_windows(
    ds: Dataset,
    *,
    t0: int,
    n_test: int | None = None,
) -> Iterator[tuple[int, Dataset]]:
    """生成 Walk-Forward 滚动时序评估窗口切片（契约 R42）。

    每次迭代返回元组 (t, sub_ds)，其中：
    - t 为当前截断时间步（0-based，特征仅可见 0..t）；
    - sub_ds 为截断后的数据集，恰好包含前 t+1 期历史开奖数据。

    @param ds 全量历史数据集
    @param t0 冷启动所需最少样本期数
    @param n_test 回测评估期数，None 表示评估冷启动后的全部可用期数
    @return 生成器依次产生 (t, sub_ds)
    @raises InsufficientDataError 历史样本总量不足以支撑冷启动窗口
    """
    total_draws = len(ds.draws)
    if total_draws <= t0:
        raise InsufficientDataError(
            f"数据集历史样本总量 {total_draws} 期不足以支撑冷启动期数 T0={t0}！"
        )

    # 步骤 1: 计算可用测试步长
    total_available_test = total_draws - t0
    if n_test is not None:
        eval_steps = min(max(0, n_test), total_available_test)
    else:
        eval_steps = total_available_test

    start_t = (total_draws - 1) - eval_steps

    # 步骤 2: 逐期滚动推进
    for t in range(start_t, total_draws - 1):
        target_draw = ds.draws[t + 1]
        sub_ds = truncated(ds, target_draw.period)
        yield (t, sub_ds)


def run_walkforward(
    ds: Dataset,
    cfg: AppConfig,
    *,
    n_test: int | None = None,
    t0: int | None = None,
    include_tickets: bool | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestReport:
    """执行严格单向无前瞻的 Walk-Forward 滚动回测流水线（契约 R42、§3.12）。

    【执行步骤】
    步骤 1: 确定回测参数与边界前置校验（T0 与可用数据量）；
    步骤 2: 强制启用纯内存自适应权重模式（不污染磁盘持久化）；
    步骤 3: 逐期滚动生成预测、防穿越校验、注单装配与开奖比对结算；
    步骤 4: 内存递推更新自适应状态；
    步骤 5: 聚合统计 Lift@K、高奖级分布、规则归因与资金曲线并执行自检。

    @param ds 历史数据集
    @param cfg 全局应用配置
    @param n_test 回测期数，None 时读取 cfg.backtest.default_n
    @param t0 冷启动期数，None 时读取 cfg.backtest.warmup
    @param include_tickets 是否生成覆盖注单并结算奖金，None 时读取 cfg.backtest.include_tickets
    @param progress 进度回调函数，入参为 (当前步数, 总步数)
    @return 结构化完整回测报告 BacktestReport
    @raises InsufficientDataError 样本量不足
    """
    start_time = time.perf_counter()

    # 步骤 1: 确定回测参数与边界前置校验
    effective_t0 = int(t0) if t0 is not None else int(cfg.backtest.warmup)
    total_draws = len(ds.draws)
    if total_draws <= effective_t0:
        raise InsufficientDataError(
            f"数据集样本量 {total_draws} 期不足以支持冷启动窗口 T0={effective_t0}！"
        )

    effective_n_test = (
        int(n_test) if n_test is not None else int(cfg.backtest.default_n)
    )
    with_tickets = (
        bool(include_tickets)
        if include_tickets is not None
        else bool(cfg.backtest.include_tickets)
    )

    # 步骤 2: 强制克隆配置为纯内存自适应权重更新模式
    run_cfg = dataclasses.replace(
        cfg,
        adaptive=dataclasses.replace(cfg.adaptive, in_memory=True),
    )

    windows = list(iter_windows(ds, t0=effective_t0, n_test=effective_n_test))
    total_steps = len(windows)
    rows: list[BacktestRow] = []
    adaptive_state: AdaptiveState | None = None

    # 步骤 3: 逐期滚动评估
    for step_idx, (t, sub_ds) in enumerate(windows):
        target_draw = ds.draws[t + 1]
        target_period = target_draw.period
        target_numbers = target_draw.numbers

        # 3.1 状态矩阵构建与特征快照装配
        state = BinaryStateMatrix.from_dataset(sub_ds)
        snapshot = build_feature_snapshot(
            state,
            end=len(sub_ds.draws) - 1,
            cfg=run_cfg,
            include_ml=run_cfg.ml.enabled,
        )

        # 3.2 严格防穿越守卫
        assert_no_lookahead(max_accessed_t=snapshot.as_of_index, as_of_t=t)

        # 3.3 Layer 4 综合打分与规则判定
        score_round_res = score_round(
            snapshot,
            run_cfg,
            adaptive_state=adaptive_state,
        )

        # 3.4 Top-K 选号命中统计
        ranked_numbers = score_round_res.scores.order
        if not ranked_numbers:
            sorted_indices = np.argsort(-score_round_res.scores.final)
            ranked_numbers = tuple(int(x + 1) for x in sorted_indices)

        top_picks: dict[int, tuple[int, ...]] = {}
        hit_at: dict[int, int] = {}
        for k_val in run_cfg.backtest.ks:
            picks = ranked_numbers[:k_val]
            top_picks[k_val] = picks
            hit_at[k_val] = len(set(picks) & target_numbers)

        gold_hit = len(set(score_round_res.pool.gold) & target_numbers)
        main_hit = len(set(score_round_res.pool.main) & target_numbers)

        # 3.5 Layer 5 注单生成与奖金核算
        all_tickets: list[tuple[int, ...]] = []
        round_cost = 0
        round_prize = 0
        prize_known = False

        if with_tickets:
            plans = build_tickets(
                score_round_res.pool,
                score_round_res.scores.final,
                run_cfg,
            )
            for plan in plans:
                round_cost += int(plan.cost_yuan)
                ptable = run_cfg.prize_table.get(plan.play)
                if ptable is not None:
                    prize_known = True
                    for tk in plan.tickets:
                        all_tickets.append(tk)
                        hits = len(set(tk) & target_numbers)
                        round_prize += int(ptable.get(hits, 0))
                else:
                    for tk in plan.tickets:
                        all_tickets.append(tk)

        best_ticket_hit = (
            max((len(set(tk) & target_numbers) for tk in all_tickets), default=0)
            if all_tickets
            else 0
        )

        row = BacktestRow(
            period=target_period,
            as_of_index=t,
            as_of_period=ds.draws[t].period,
            scores=score_round_res.scores.final,
            votes=score_round_res.votes,
            top_picks=top_picks,
            hit_at=hit_at,
            gold=score_round_res.pool.gold,
            main=score_round_res.pool.main,
            gold_hit=gold_hit,
            main_hit=main_hit,
            tickets=tuple(all_tickets),
            best_ticket_hit=best_ticket_hit,
            cost_yuan=round_cost,
            prize_yuan=round_prize,
            prize_known=prize_known,
            ok=True,
            warnings=(),
        )
        rows.append(row)

        # 步骤 4: 内存自适应权重更新
        adaptive_state, _ = adaptive.next_weights(
            rows,
            adaptive_state,
            run_cfg.adaptive,
            period=target_period,
        )

        # 进度反馈
        if progress is not None:
            progress(step_idx + 1, total_steps)

    # 步骤 5: 聚合统计核心指标
    lift_stats = lift_at_k(rows, run_cfg.backtest.ks, run_cfg.backtest)
    tier_stats = high_tier_rate(rows, run_cfg.covering.play, run_cfg.prize_table)
    attribution_report = rule_attribution(rows)
    bankroll_stats = bankroll(rows, run_cfg.prize_table)

    report = BacktestReport(
        n_sample=len(rows),
        t0=effective_t0,
        start_period=rows[0].period if rows else 0,
        end_period=rows[-1].period if rows else 0,
        rows=tuple(rows),
        lift=lift_stats,
        high_tier=tier_stats,
        attribution=attribution_report,
        bankroll=bankroll_stats,
        config_hash=config_hash(run_cfg),
        self_check={},
        elapsed_s=time.perf_counter() - start_time,
        warnings=(),
    )

    # 报告自检
    chk = self_check(report)
    return dataclasses.replace(report, self_check=chk)
