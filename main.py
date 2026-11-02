"""快乐8 (Keno) 五层正交动力学与组合优化预测系统主入口（契约 R48、§3.14）。

【支持模式】
1. daily (默认): 端到端实盘预测，生成金胆、复式池、保底注单及全渠道三维交付件；
2. backtest: 严格时序前瞻 Walk-Forward 滚动回测评估；
3. audit: 全仓防穿越审计与统计模型自洽性机检。

【CLI 兼容契约】
全面兼容历史外围批处理脚本参数 (--history/--no-sync/--quick/--full-sync/--period/--date/--backtest/--full-print)
并支持现代微调参数 (--mode/--k/--budget/--warmup/--pool-size/--seed/--config)。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

# 强制统一控制台输出编码为 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        # 已重定向到不支持 reconfigure 的流（如管道/测试替身），保持原编码继续运行
        pass

from kl8.config import AppConfig, load_config, merge_cli_overrides
from kl8.constants import PLAY_LABELS, ROI_NA_TEXT
from kl8.errors import KL8Error
from kl8.evaluation.metrics import format_roi_bp
from kl8.pipeline import DailyResult, run_audit, run_backtest, run_daily
from kl8.reporting.text_report import sample_note

DEFAULT_HISTORY_PATH = "kl8_history.txt"
# 日常模式伴随回测期数（够用且不会让双击入口长时间静默）
DEFAULT_DAILY_BACKTEST_N = 30
# 纯回测模式默认期数（契约 R48-b）
DEFAULT_BACKTEST_N = 100


def _daily_progress(message: str) -> None:
    """日常流水线单步进度打印（契约 R48 的 8 步执行序）。"""
    print(f"   ⏳ {message}", flush=True)


def _backtest_progress(step: int, total: int) -> None:
    """回测进度打印：约每 5% 或结束时刷新一行，避免控制台长时间静默。"""
    if total <= 0:
        return
    stride = max(1, total // 20)
    if step % stride != 0 and step != total:
        return
    pct = int(step * 100 / total)
    sys.stdout.write(f"\r   ⏳ 回测进度 {step}/{total} ({pct}%)")
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """构建标准统一的 CLI 参数解析器（契约 R48-b）。"""
    parser = argparse.ArgumentParser(
        prog="kl8",
        description="快乐8 (Keno) 五层正交动力学与组合优化预测系统",
    )

    # 运行模式选择
    parser.add_argument(
        "--mode",
        choices=["daily", "backtest", "audit"],
        default="daily",
        help="运行模式: daily (日常实盘预测), backtest (滚动时序回测), audit (模型自洽性审计)",
    )

    # 历史数据与增量同步参数
    parser.add_argument(
        "--history",
        default=None,
        help=f"历史开奖数据文件路径 (默认: {DEFAULT_HISTORY_PATH})",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="跳过开奖数据网络同步，直接使用本地历史数据",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="极速同步模式 (仅拉取最新期次，默认开启)",
    )
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help="全量同步模式 (扫描校验全部历史开奖记录)",
    )

    # 预测目标指定
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="指定预测目标期号 (默认: 历史最新期 + 1)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="指定预测目标开奖日期 YYYY-MM-DD (默认: 历史最新日期 + 1天)",
    )

    # 回测与性能控制
    parser.add_argument(
        "--backtest",
        type=int,
        default=None,
        help=(
            f"回测期数 (daily 模式默认 {DEFAULT_DAILY_BACKTEST_N}；backtest/audit 模式默认 "
            f"{DEFAULT_BACKTEST_N}；daily 传 0 表示关闭伴随回测)"
        ),
    )
    parser.add_argument(
        "--no-tickets",
        action="store_true",
        help="回测时跳过注单退火生成与奖金核算，仅秒级评估 Top-K 命中率 Lift（大幅提速）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="回测冷启动窗口 T0 (默认: 300 期)",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="启用 LightGBM 诊断通道（默认关闭；契约 B4：weight=0 时仅诊断，开启会显著变慢）",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="兼容参数 (无操作)",
    )

    # 业务玩法与组合覆盖参数
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="玩法选号数 (5 = 选五，7 = 选七；其它取值会被配置校验拒绝)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="注单预算注数 (如 4 代表推荐 4 注)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="精选复式大底规模 (如 12 或 15 码)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="全局随机种子 (保证算法完全可复现)",
    )

    # 配置文件与控制台打印
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="自定义 JSON 配置文件路径",
    )
    parser.add_argument(
        "--full-print",
        action="store_true",
        help="是否在控制台打印完整的 TXT 全景决策报告",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """应用程序唯一 CLI 入口函数（契约 §3.14）。

    退出码标准：
    - 0: 执行成功并完成交付；
    - 1: 参数非法 (ConfigError) 或数据异常 (DataFormatError/InsufficientDataError)；
    - 2: 系统内部未知未捕获异常；
    - 130: 用户手动取消中断 (KeyboardInterrupt)。

    @param argv 命令行入参序列
    @return 状态码整数
    """
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as se:
        return se.code if isinstance(se.code, int) else 1

    try:
        # 步骤 1: 加载并合并 CLI 配置覆盖（--config 是文件路径，必须走 path 参数）
        cfg = load_config(path=args.config)
        cfg = merge_cli_overrides(cfg, args)

        # 步骤 1.1: 可选的 LightGBM 诊断通道开启（契约 B4：默认关闭，--ml 显式打开）
        if getattr(args, "ml", False):
            cfg = dataclasses.replace(
                cfg,
                ml=dataclasses.replace(cfg.ml, enabled=True),
            )

        # 步骤 1.2: 解析回测期数（daily 与 backtest 默认值不同）
        if args.backtest is None:
            bt_n = DEFAULT_DAILY_BACKTEST_N if args.mode == "daily" else DEFAULT_BACKTEST_N
        else:
            bt_n = int(args.backtest)

        # 步骤 2: 分流执行对应业务模式
        if args.mode == "daily":
            print("=" * 70)
            print("🚀 [KL8 快乐8] 五层正交动力学与组合优化实盘预测流水线启动")
            print("=" * 70)

            is_quick = not args.full_sync
            res: DailyResult = run_daily(
                cfg,
                no_sync=args.no_sync,
                target_period=args.period,
                target_date=args.date,
                backtest_n=bt_n,
                quick=is_quick,
                progress=_daily_progress,
            )

            # 打印精炼控制台摘要
            print(f"✔ 预测目标: 第 {res.target_period} 期 | 开奖日期: {res.target_date}")
            print(f"🔥 超级金胆 ({len(res.pool.gold)}码):   {' '.join(f'{n:02d}' for n in res.pool.gold)}")
            print(f"⭐ 精选大底 ({len(res.pool.main)}码): {' '.join(f'{n:02d}' for n in res.pool.main)}")

            if res.plans:
                print("🎟️ 推荐保底实战票:")
                for plan in res.plans:
                    play_cn = PLAY_LABELS.get(plan.play, plan.play)
                    design = plan.guarantee_level.get("design", "")
                    print(
                        f"   [{play_cn} 计划, 选号k={plan.k}, 预算={plan.budget}注, 设计={design}]:"
                    )
                    for t_idx, tk in enumerate(plan.tickets, start=1):
                        print(f"     第 {t_idx:02d} 注: [ {' '.join(f'{x:02d}' for x in tk)} ]")
                    if cfg.prize_table.get(plan.play) is None:
                        print(f"     {ROI_NA_TEXT}")

            print("\n📁 全渠道交付物已就绪:")
            print(f"   📄 TXT 精要报告:   {res.report_paths['text']}")
            print(f"   📱 微信发布文案:   {res.report_paths['wechat']}")
            print(f"   📊 商业 Excel:     {res.report_paths['excel']}")
            print(f"   💾 自适应状态文件: {res.report_paths['state']}")
            if "prev_prediction" in res.report_paths:
                print(f"   🧾 复盘状态文件:   {res.report_paths['prev_prediction']}")

            if res.warnings:
                print("\n⚠️ 执行提示与警告:")
                for w in res.warnings:
                    print(f"   ! {w}")

            print(f"\n⏱️ 全流程端到端执行耗时: {res.elapsed_s:.3f} 秒")

            # 若请求全量输出，打印 TXT 内容
            if args.full_print and res.report_paths["text"].exists():
                print("\n" + "=" * 70)
                print("📋 控制台全景决策精要报告正文:")
                print("=" * 70)
                with open(res.report_paths["text"], "r", encoding="utf-8") as f:
                    print(f.read())

            return 0

        elif args.mode == "backtest":
            print("=" * 70)
            print(f"🔬 [KL8 回测] Walk-Forward 滚动时序前瞻回测 ({bt_n} 期)")
            print("=" * 70)

            include_tk = not args.no_tickets
            bt = run_backtest(
                cfg,
                n=bt_n,
                t0=args.warmup,
                include_tickets=include_tk,
                progress=_backtest_progress,
            )

            print(f"✔ 实际评估样本量: {bt.n_sample} 期 (冷启动 T0={bt.t0})")
            print(f"✔ 覆盖期号范围:   第 {bt.start_period} 期 ~ 第 {bt.end_period} 期")
            print(
                f"✔ 防穿越校验:     已校验 {bt.lookahead_checked} 期，越界 {len(bt.lookahead_violations)} 期"
            )
            if bt.lookahead_violations:
                print(f"   ! 越界期号: {', '.join(str(p) for p in bt.lookahead_violations)}")

            if any(stat.n < cfg.backtest.min_n_signal for stat in bt.lift.values()):
                print(f"   ! {sample_note(cfg.backtest.min_n_signal)}")

            print("\n📈 Top-K 提升度 (Lift@K) 显著性检验（三条件：n 门槛 + Lift + Bonferroni p）:")
            print("   Top-K  实测均中  理论期望  提升度   t统计量   p值     校正α    n      判定")
            for _k_val, lstat in sorted(bt.lift.items(), key=lambda x: x[0]):
                sig_str = "★显著" if lstat.is_signal else "平稳"
                print(
                    f"   Top-{lstat.k:02d}   {lstat.mean_hit:5.2f}     {lstat.expected:5.2f}     "
                    f"{lstat.lift:6.3f}   {lstat.t_stat:6.3f}  {lstat.p_value:.4f}  "
                    f"{lstat.alpha_adj:.4f}  {lstat.n:4d}   {sig_str}"
                )

            print("\n📊 R45 规则独立查准率归因:")
            if bt.attribution is None:
                print("   暂无归因数据。")
            else:
                for rule_id, stat in sorted(bt.attribution.per_rule.items()):
                    print(
                        f"   {rule_id}: 平均池 {stat.pool_size_avg:4.1f} 码 | 命中率 {stat.hit_rate:6.2%} | "
                        f"Lift {stat.lift:5.3f} | z={stat.z:+.2f}"
                    )
                high = bt.attribution.high_votes
                low = bt.attribution.low_votes
                print(
                    f"   高票组(V≥3): n={int(high.get('n', 0))}, 命中率={high.get('hit_rate', 0.0):.2%}, "
                    f"Lift={high.get('lift', 0.0):.3f}"
                )
                print(
                    f"   低票组(V≤1): n={int(low.get('n', 0))}, 命中率={low.get('hit_rate', 0.0):.2%}, "
                    f"Lift={low.get('lift', 0.0):.3f} | 两比例检验 z={high.get('z', 0.0):+.2f}"
                )

            if bt.high_tier:
                print("\n🎯 高奖级触发率（实际 vs 超几何理论，单注口径；§7.4 勘误）:")
                for _tier_key, tier in sorted(bt.high_tier.items()):
                    theory = (
                        f"{tier.theory_rate:.4%}（≈1/{int(1.0 / tier.theory_rate)} 期）"
                        if tier.theory_rate > 0
                        else "N/A"
                    )
                    denom = (
                        f"{tier.n_rounds} 期 / {tier.n_trials} 注"
                        if tier.n_trials > tier.n_rounds
                        else f"{tier.n_rounds} 期"
                    )
                    print(
                        f"   {tier.tier}: 实际 {tier.observed_rate:.4%} | 理论 {theory} | "
                        f"比值 {tier.ratio:.2f} | n={denom}"
                    )

            if bt.bankroll is not None:
                b = bt.bankroll
                print("\n💰 资金收益与风险指标:")
                print(f"   总投入: {b.total_cost_yuan} 元 | 总奖金: {b.total_prize_yuan} 元 | 净利润: {b.net_yuan} 元")
                roi_str = format_roi_bp(b.roi_bp) if b.roi_available else ROI_NA_TEXT
                print(
                    f"   投资回报率 (ROI): {roi_str} | 最大回撤: {b.max_drawdown_yuan} 元 | "
                    f"最长连挂: {b.max_losing_streak} 期"
                )

            print(f"\n✔ 报告自洽机检: {bt.self_check}")
            print(f"\n⏱️ 回测总耗时: {bt.elapsed_s:.3f} 秒")
            return 0 if bt.self_check.get("no_lookahead", False) else 1

        elif args.mode == "audit":
            print("=" * 70)
            print("🛡️ [KL8 审计] 系统级防穿越与模型自洽性审计")
            print("=" * 70)

            audit_res = run_audit(cfg, n=bt_n)
            code_ok = audit_res["code_audit_clean"]
            print(f"✔ 代码级防穿越审计 (禁止非法打乱): {'通过 (100% 洁净)' if code_ok else '未通过'}")
            if not code_ok:
                for v in audit_res["code_violations"]:
                    print(f"   ! 发现违规代码: {v}")

            print(f"✔ 回测自洽性机检项: {audit_res['backtest_self_check']}")
            print(f"✔ 评估期数: {audit_res['backtest_samples']} 期")
            print(
                f"✔ 防穿越覆盖: 已校验 {audit_res.get('lookahead_checked', 0)} 期，"
                f"越界 {len(audit_res.get('lookahead_violations', ()))} 期"
            )
            for v_period in audit_res.get("lookahead_violations", ()):
                print(f"   ! 防穿越越界期号: {v_period}")
            signals = audit_res.get("lift_signals", {})
            sig_keys = [int(k) for k, is_sig in signals.items() if is_sig]
            if sig_keys:
                print(f"✔ 达到显著性门槛的 Top-K: {', '.join(f'Top-{k}' for k in sorted(sig_keys))}")
            else:
                print("✔ 达到显著性门槛的 Top-K: 无（未通过样本量/Lift/Bonferroni 三条件）")
            print(f"✔ 指纹血缘: {audit_res['config_hash']}")
            no_leak = audit_res["backtest_self_check"].get("no_lookahead", False)
            return 0 if (code_ok and no_leak) else 1

        return 0

    except KL8Error as e:
        sys.stderr.write(f"\n[错误 - 业务约束阻断] {e}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n\n[中断] 用户已手动取消操作。\n")
        return 130
    except Exception as e:
        sys.stderr.write(f"\n[致命内部异常] {e}\n")
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
