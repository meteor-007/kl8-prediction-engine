# -*- coding: utf-8 -*-
"""快乐8 (Keno) 深度学习与图谱贝叶斯混合集成预测系统主入口。

【技术特性】
1. 深度学习序列建模：基于 PyTorch Multi-Head Self-Attention 自注意力机制；
2. 图谱贝叶斯平滑：基于 Dirichlet-Multinomial 后验与 80×80 共现谱分解；
3. 30 码黄金大底级联缩水：位运算正交防撞车与蒙特卡洛 5000 次实测审计；
4. 全渠道三维交付：控制台精要、微信一键复制文案、商业 Excel、集群三元组。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

# 统一控制台输出编码为 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from kl8_engine.pipeline import run_pipeline
from kl8_engine.points import PointsError, format_points_console, parse_points
from kl8_engine.reporting import format_comparison_console
from kl8_engine.review import format_hit_rate_console, format_review_console


def build_arg_parser() -> argparse.ArgumentParser:
    """构建标准命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="kl8",
        description="快乐8 (Keno) 深度学习与图谱贝叶斯混合集成预测系统",
    )
    parser.add_argument(
        "--history",
        default="kl8_history.txt",
        help="历史开奖文件路径 (默认: kl8_history.txt)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="输出交付件目录 (默认: output)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="跳过网络开奖数据同步，直接使用本地数据",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Transformer 模型微调轮数 (默认: 25)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="时序自注意力滑动窗口长度 (默认: 10)",
    )
    parser.add_argument(
        "--dl-weight",
        type=float,
        default=0.55,
        help="深度学习权重 (默认: 0.55, 统计图谱权重为 0.45)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="30码黄金大底缩水实战注数预算 (默认: 20注)",
    )
    parser.add_argument(
        "--points",
        default="",
        help='用户点位号码，空格/逗号分隔，例如 "07 08 13 16"（默认空）',
    )
    parser.add_argument(
        "--points-mode",
        choices=("compare", "boost"),
        default="compare",
        help="点位模式: compare=仅对比标注 (默认); boost=点位参与缩水覆盖加权",
    )
    parser.add_argument(
        "--fusion",
        choices=("borda", "weighted"),
        default="borda",
        help="正式版融合方式: borda=四分量秩融合 (默认); weighted=原始加权和",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="关闭双融合并排对比输出（默认开启，实验版不进入日志/交付）",
    )
    parser.add_argument(
        "--full-print",
        action="store_true",
        help="在控制台打印完整 TXT 决策报告",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """系统主执行入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        points = parse_points(args.points)
    except PointsError as exc:
        parser.error(str(exc))

    print("=" * 80)
    print("🚀 [KL8 快乐8] 深度学习 Transformer 与图谱贝叶斯混合集成预测系统启动")
    print("=" * 80)

    def _progress_cb(msg: str) -> None:
        print(f"   ⏳ {msg}", flush=True)

    try:
        res = run_pipeline(
            history_path=args.history,
            output_dir=args.output_dir,
            no_sync=args.no_sync,
            dl_weight=args.dl_weight,
            seq_len=args.seq_len,
            epochs=args.epochs,
            budget_shrinkage=args.budget,
            points=points or None,
            points_mode=args.points_mode,
            fusion_mode=args.fusion,
            comparison=not args.no_compare,
            progress=_progress_cb,
        )

        print("\n" + "=" * 80)
        print(f"✔ 预测目标期号: 第 {res.target_period} 期")
        fusion_label = "Borda 秩融合" if res.fusion_mode == "borda" else "加权概率和"
        print(f"🔀 正式版融合: {fusion_label}")
        if res.comparison_fusion_mode is not None:
            comp_label = "Borda 秩融合" if res.comparison_fusion_mode == "borda" else "加权概率和"
            print(f"🧪 实验版融合: {comp_label}（仅对比，不进入日志/交付）")
        if res.points_analysis is not None:
            print(f"🧭 点位模式: {res.points_mode} | 已录入点位 {len(res.points_analysis.points)} 码")
        print(f"🔥 超级金胆 ({len(res.ensemble.gold_pool)}码):   {' '.join(f'{x:02d}' for x in res.ensemble.gold_pool)}")
        print(f"⭐ 精选大底 ({len(res.ensemble.main_pool)}码): {' '.join(f'{x:02d}' for x in res.ensemble.main_pool)}")
        print("=" * 80)

        if not args.full_print:
            if res.review is not None:
                print("\n" + format_review_console(res.review))
            if res.points_analysis is not None:
                print("\n" + format_points_console(res.points_analysis))
            if res.comparison_ensemble is not None and res.comparison_shrinkage is not None:
                print(
                    "\n"
                    + format_comparison_console(
                        res.ensemble,
                        res.shrinkage,
                        res.comparison_ensemble,
                        res.comparison_shrinkage,
                        official_mode=res.fusion_mode,
                        comparison_mode=res.comparison_fusion_mode,
                    )
                )
            if res.hit_stats:
                print("\n" + format_hit_rate_console(res.hit_stats))

        # 打印【集群三元组】纯净专版（格式四铁律）
        print("\n📌【集群三元组】纯净可复制专版（Top 8 全量连号集群清单）:")
        print("```text")
        for trp in res.cluster_triplets:
            print(trp)
        print("```")

        # 打印 30 码黄金大底缩水方案
        print("\n" + res.shrinkage_console)

        # 交付物汇总
        print("\n📁 全渠道交付物已全部落盘就绪:")
        print(f"   📄 TXT 精要报告:   {res.report_paths['text']}")
        print(f"   📱 微信主发文案:   {res.report_paths['wechat']}")
        print(f"   📊 商业 Excel:     {res.report_paths['excel']}")
        print(f"   💎 30码缩水专报:   {res.report_paths['shrinkage_txt']}")
        print(f"   📱 30码微信专发:   {res.report_paths['shrinkage_wechat']}")

        if res.warnings:
            print("\n⚠️ 运行提示与告警:")
            for w in res.warnings:
                print(f"   ! {w}")

        print(f"\n⏱️ 全流程端到端执行耗时: {res.elapsed_seconds:.3f} 秒")

        if args.full_print:
            print("\n" + "=" * 80)
            print("📋 控制台全景决策报告正文:")
            print("=" * 80)
            print(res.console_report)

        return 0

    except Exception as e:
        import traceback

        print(f"\n❌ [致命执行异常]: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
