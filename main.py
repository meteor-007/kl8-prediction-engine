# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 日常运行主入口

【业务背景与架构定位】:
日常一键执行流水线 (7 步)：
1. 同步福彩最新开奖数据 (官网 API + 备用源双通道增量)；
2. 加载并校验本地开奖数据集；
3. 深度复盘昨日开奖全景与昨日 16 码实测战绩；
4. 严格 walk-forward 回测与特征预测力审计 (先算，供报告引用)；
5. 运行微积分动力学引擎，导出 TXT / 微信分桶 / Excel 三类交付件 (含回测与审计结论)；
6. 控制台输出当期八区预测全量明细与微信分桶原文；
7. 控制台输出命中率回测、参数寻优与优化空间裁决。

@Author czw
@Since 2026-10-18
"""
from __future__ import annotations

import argparse
import datetime
import sys

# 控制台编码防乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from calc_backtest import (
    build_backtest_console_report,
    find_best_param_sweep,
    save_backtest_excel,
    walk_forward_backtest,
)
from calc_core import DEFAULT_HISTORY_FILE, TimeSeries, load_dataset
from calc_engine import run_calculus_engine
from calc_report import (
    build_previous_review,
    generate_and_save_all_reports,
)
from calc_signal_audit import build_audit_console_report
from calc_sync import sync_history

BAR = "═" * 78


def print_banner(title: str) -> None:
    print()
    print(BAR)
    print(f"  {title}")
    print(BAR)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="快乐8纯微积分八区分区预测系统")
    ap.add_argument("--history", default=DEFAULT_HISTORY_FILE, help="开奖历史数据文件路径")
    ap.add_argument("--no-sync", action="store_true", help="跳过数据同步，直接使用本地开奖数据")
    ap.add_argument("--quick", action="store_true", help="快速增量同步模式 (默认)")
    ap.add_argument("--full-sync", action="store_true", help="全量同步历史数据")
    ap.add_argument("--period", type=int, default=None, help="指定预测目标期号 (默认最新开奖期+1)")
    ap.add_argument("--date", type=str, default=None, help="指定预测日期 YYYY-MM-DD (默认当天)")
    ap.add_argument("--backtest", type=int, default=100,
                    help="命中率走查回测期数 (默认100期；0 表示跳过回测与审计)")
    ap.add_argument("--no-sweep", action="store_true", help="跳过参数寻优扫描")
    ap.add_argument("--full-print", action="store_true", help="控制台打印全量参数表 (默认打印核心精要)")
    args = ap.parse_args(argv)

    print_banner("快乐8「纯微积分动力学」八区分区预测流水线")

    # =================================================================
    # 步骤 1: 数据同步
    # =================================================================
    print("\n▶ 步骤 1/7  检查并同步官方最新开奖数据...")
    if not args.no_sync:
        sync_history(args.history, quick=not args.full_sync)
    else:
        print("    已按 --no-sync 跳过远程同步，直接读取本地开奖历史库")

    # =================================================================
    # 步骤 2: 数据加载与校验
    # =================================================================
    print("\n▶ 步骤 2/7  加载开奖数据集并校验...")
    ds = load_dataset(args.history)
    if ds.T < 40:
        print(f"    [错误] 历史开奖期数不足 (当前 {ds.T} 期)，无法进行微积分滑动窗口计算。")
        return 1
    for w in ds.warnings:
        print(f"    [数据告警] {w}")

    last_draw = ds.draws[-1]
    target_period = args.period or (last_draw.period + 1)
    target_date = args.date or datetime.date.today().strftime("%Y-%m-%d")
    print(f"    · 本地样本总量: {ds.T} 期 ({ds.draws[0].date} → {last_draw.date})，号码合法性校验全部通过")
    print(f"    · 官方最新已开奖: 第 {last_draw.period} 期 ({last_draw.date})")
    print(f"    · 本次预测目标期: 第 {target_period} 期 ({target_date})")

    ts = TimeSeries(ds)

    # =================================================================
    # 步骤 3: 昨日深度复盘
    # =================================================================
    print_banner("步骤 3/7  昨日推荐实战深度复盘")
    review_lines = build_previous_review(ds)
    print("\n".join(review_lines))

    # =================================================================
    # 步骤 4: 命中率回测与特征审计 (先于交付件生成，供报告引用)
    # =================================================================
    print("\n▶ 步骤 4/7  执行严格走查回测与特征预测力审计...")
    bt = None
    bt_text = audit_text = ""
    sweep = []
    if args.backtest > 0:
        bt = walk_forward_backtest(ds, args.backtest)
        if bt.get("n_sample", 0) > 0:
            bt_text = build_backtest_console_report(bt)
            audit_text = build_audit_console_report(ts.hit, bt["rows"])
            if not args.no_sweep:
                sweep = find_best_param_sweep(ds, min(args.backtest, 60))
            print(f"    · 完成 {bt['n_sample']} 期 walk-forward 回测 (逐期重跑引擎，零前瞻泄漏)")
            print(f"    · 完成 5 项特征预测力统计检验 + {len(sweep)} 组参数寻优扫描")
        else:
            print("    历史样本不足以执行回测，已跳过。")
    else:
        print("    已按 --backtest 0 跳过回测与审计。")

    # =================================================================
    # 步骤 5: 动力学计算与多格式交付件导出
    # =================================================================
    print("\n▶ 步骤 5/7  运行微积分动力学计算并输出多格式交付件 (TXT / 微信文案 / Excel)...")
    engine_res = run_calculus_engine(ds, ts)
    reports_bundle = generate_and_save_all_reports(
        ds, engine_res, target_period, target_date,
        backtest_text=bt_text, audit_text=audit_text, backtest_result=bt, sweep=sweep,
    )
    print(f"    · [TXT 详细报告]   {reports_bundle['txt_path']}")
    print(f"    · [微信分桶文案]   {reports_bundle['wechat_path']}")
    print(f"    · [Excel 专业报表] {reports_bundle['excel_path']}")
    if reports_bundle.get("backtest_excel_path"):
        print(f"    · [Excel 回测明细] {reports_bundle['backtest_excel_path']}")

    # =================================================================
    # 步骤 6: 预测精要 + 全量动力学附录 + 微信原文
    # =================================================================
    print_banner("步骤 6/7  当期微积分八区预测全量明细")
    rep_text = reports_bundle["txt_report"]
    if args.full_print:
        print(rep_text)
    else:
        # 步骤 3 已完整打印「上期复盘」板块，此处渲染从 [1] 本期推荐开始，避免控制台重复
        summary_lines = []
        in_summary = False
        for ln in rep_text.splitlines():
            if ln.startswith("[1]"):
                in_summary = True
            if ln.startswith("[附录"):
                break
            if in_summary:
                summary_lines.append(ln)
        print("\n".join(summary_lines))
        print("\n    [提示] 完整 TXT 报告 (含上期复盘/80码动力学全表/回测与审计附录): "
              f"{reports_bundle['txt_path']}")

    print("\n" + BAR)
    print("  📱【微信分桶推荐文案原文】(一键全选复制，可直接发送微信群/朋友圈):")
    print(BAR)
    print(reports_bundle["wechat_text"])

    # =================================================================
    # 步骤 7: 回测结果 + 参数寻优 + 优化空间裁决
    # =================================================================
    print()
    print_banner("步骤 7/7  命中率回测与深度优化空间裁决")
    if bt and bt.get("n_sample", 0) > 0:
        print(bt_text)
        if sweep:
            print("\n" + BAR)
            print("  🔍【参数寻优扫描】各结构变体在同一走查窗口下的实测对比:")
            print(BAR)
            best = sweep[0]["avg"]
            for v in sweep:
                flag = "◀ 实测最高" if abs(v["avg"] - best) < 1e-9 else ""
                print(f"      {v['name']:<26s} 平均命中 {v['avg']:.3f} 码  "
                      f"z={v['z']:+.2f}  ≥5码占比 {v['ge5_pct']:.1f}%  {flag}")
            base_v = next((v for v in sweep if "线上配置" in v["name"]), None)
            gain = best - (base_v["avg"] if base_v else best)
            noise = bt["baseline"]["sd"] / (bt["n_sample"] ** 0.5)
            if base_v and gain > 3 * noise:
                print(f"      → 最优变体高出线上配置 {gain:+.3f} 码，超过 3 倍噪声幅度 (σ/√n={noise:.3f})，"
                      "具备进入更长样本复核的资格。")
            else:
                print(f"      → 最优变体仅高出线上配置 {gain:+.3f} 码，不足噪声幅度 σ/√n = {noise:.3f} 码的 3 倍，"
                      "判定为抽样波动而非真实增益 → 不调整线上参数。")
        print("\n" + audit_text)
    else:
        print("    本次运行未执行回测与审计 (--backtest 0)。")

    print("\n" + BAR)
    print("  🎉 全部计算与报告导出执行完毕！可查阅 output 目录下的 TXT / 微信文案 / Excel 报表。")
    print(BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
