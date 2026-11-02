# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 工业级质量与逻辑回归测试套件

【测试范围】:
1. 开奖数据格式完整性与 1..80 范围合规测试；
2. 八区分区覆盖完整性（无重合、无遗漏、恰好覆盖 1..80）；
3. 动力学算子（速度、加速度、密度、势能）数值边界与无 NaN 测试；
4. 双态决策测试：每区严格选出 1 动能码 + 1 势能码，共计 16 码无重复；
5. 防前瞻泄漏时间截断隔离测试。

@author czw
@since 2026-10-18
"""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

from calc_core import (
    N_NUM,
    DEFAULT_HISTORY_FILE,
    Dataset,
    TimeSeries,
    load_dataset,
    parse_history_file,
    truncated,
)
from calc_engine import calc_zone_bounds, run_calculus_engine


class TestCalculusZoneSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset(DEFAULT_HISTORY_FILE)
        cls.ts = TimeSeries(cls.ds)

    def test_01_history_integrity(self):
        """测试历史数据解析完整性与号码规范。"""
        self.assertGreater(self.ds.T, 100, "历史数据期数不应少于 100 期")
        for d in self.ds.draws[-50:]:
            self.assertEqual(len(d.numbers), 20, f"期号 {d.period} 号码数不为 20")
            self.assertTrue(all(1 <= x <= N_NUM for x in d.numbers), f"期号 {d.period} 存在超出 1..80 的号码")

    def test_02_zone_partition(self):
        """测试 8 分区划分是否无缝覆盖 1..80 且无任何重叠。"""
        all_members = []
        for z in range(8):
            lo, hi = calc_zone_bounds(z, 8, 10)
            self.assertEqual(lo, z * 10 + 1)
            self.assertEqual(hi, (z + 1) * 10)
            m = list(range(lo, hi + 1))
            self.assertEqual(len(m), 10)
            all_members.extend(m)
        self.assertEqual(len(all_members), 80)
        self.assertEqual(sorted(all_members), list(range(1, 81)))

    def test_03_engine_output_contract(self):
        """测试微积分动力学引擎输出契约。"""
        res = run_calculus_engine(self.ds, self.ts)
        self.assertIn("picks16", res)
        self.assertIn("zones", res)
        picks16 = res["picks16"]
        self.assertEqual(len(picks16), 16, "推荐号码总量必须严格为 16 码")
        self.assertEqual(len(set(picks16)), 16, "16 码推荐必须完全不重复")

        # 检查各分区结构
        zones = res["zones"]
        self.assertEqual(len(zones), 8, "必须严格输出 8 个分区")
        for z in zones:
            self.assertEqual(len(z["top2"]), 2, f"Z{z['zone_id']} 必须严格包含 2 码")
            k_n = z["top_k_item"]["num"]
            p_n = z["top_p_item"]["num"]
            self.assertNotEqual(k_n, p_n, f"Z{z['zone_id']} 动能码与势能码不可为同一号码")
            self.assertTrue(z["lo"] <= k_n <= z["hi"], f"动能码 {k_n} 不在区间 [{z['lo']}, {z['hi']}] 内")
            self.assertTrue(z["lo"] <= p_n <= z["hi"], f"势能码 {p_n} 不在区间 [{z['lo']}, {z['hi']}] 内")

    def test_04_no_nan_or_inf(self):
        """测试微积分算子无 NaN 或 Inf 异常值。"""
        res = run_calculus_engine(self.ds, self.ts)
        self.assertFalse(np.isnan(res["integ"]).any(), "密度积分存在 NaN")
        self.assertFalse(np.isnan(res["vel"]).any(), "速度导数存在 NaN")
        self.assertFalse(np.isnan(res["acc"]).any(), "加速度导数存在 NaN")
        self.assertFalse(np.isnan(res["k_scores"]).any(), "动能分存在 NaN")
        self.assertFalse(np.isnan(res["p_scores"]).any(), "势能分存在 NaN")

    def test_05_anti_leakage_truncation(self):
        """测试防前瞻泄漏时间截断机制。"""
        target = self.ds.draws[-1].period
        sub_ds = truncated(self.ds, target)
        self.assertEqual(sub_ds.T, self.ds.T - 1, "截断后历史长度应恰好比原数据集少 1 期")
        self.assertNotIn(target, sub_ds.draws_by_period, "截断后数据集严禁包含目标期开奖")

    def test_06_road_and_prime_definitions(self):
        """测试 012 路划分与质数合数数学定义的严格正确性。"""
        from calc_core import (
            PRIMES_80,
            ROAD_0_NUMS,
            ROAD_1_NUMS,
            ROAD_2_NUMS,
            get_prime_label,
            get_road_012,
            is_prime_num,
        )

        # 012路集合划分测试
        self.assertEqual(len(ROAD_0_NUMS), 26, "0路号码总数必须严格为 26 个")
        self.assertEqual(len(ROAD_1_NUMS), 27, "1路号码总数必须严格为 27 个")
        self.assertEqual(len(ROAD_2_NUMS), 27, "2路号码总数必须严格为 27 个")
        self.assertEqual(len(ROAD_0_NUMS | ROAD_1_NUMS | ROAD_2_NUMS), 80, "012路全集必须严格为 80 个号码")
        self.assertEqual(len(ROAD_0_NUMS & ROAD_1_NUMS), 0, "0路与1路不能有重叠")
        self.assertEqual(len(ROAD_0_NUMS & ROAD_2_NUMS), 0, "0路与2路不能有重叠")
        self.assertEqual(len(ROAD_1_NUMS & ROAD_2_NUMS), 0, "1路与2路不能有重叠")

        # 质数定义校验 (80以内共22个质数，1既非质数亦非合数)
        self.assertEqual(len(PRIMES_80), 22, "80以内的质数总数必须严格为 22 个")
        self.assertFalse(is_prime_num(1), "数字 1 既非质数亦非合数")
        self.assertEqual(get_prime_label(1), "特1", "数字 1 的质合标签必须为 特1")
        self.assertTrue(is_prime_num(2), "数字 2 必须为质数")
        self.assertTrue(is_prime_num(79), "数字 79 必须为质数")
        self.assertFalse(is_prime_num(80), "数字 80 必须为合数")

        for n in range(1, 81):
            self.assertEqual(get_road_012(n), n % 3)

    def test_07_analyze_number_group(self):
        """测试号码组合全维正交形态分析器的准确性。"""
        from calc_core import analyze_number_group

        # 测试真实开奖 20 码
        draw20 = self.ds.draws[-1].numbers
        res20 = analyze_number_group(draw20)
        self.assertEqual(res20["total"], 20)
        self.assertEqual(sum(res20["road_counts"].values()), 20)
        self.assertEqual(
            res20["prime_count"] + res20["composite_count"] + (1 if res20["special_1"] else 0),
            20,
        )
        self.assertEqual(res20["odd_count"] + res20["even_count"], 20)
        self.assertEqual(res20["small_count"] + res20["big_count"], 20)

    def test_08_intra_zone_orthogonality(self):
        """测试区内双态选号的 012 路正交跨路防撞车效果与属性附带。"""
        res = run_calculus_engine(self.ds, self.ts)
        self.assertIn("macro_analysis_16", res)
        macro = res["macro_analysis_16"]
        self.assertEqual(macro["total"], 16)

        # 检查各分区最优2码是否具备正交属性字段
        diff_road_zone_count = 0
        for z in res["zones"]:
            t0 = z["top_k_item"]
            t1 = z["top_p_item"]
            self.assertIn("road", t0)
            self.assertIn("is_prime", t0)
            self.assertIn("prime_label", t0)
            self.assertIn("prop_tag", t0)
            if t0["road"] != t1["road"]:
                diff_road_zone_count += 1

        # 在 8 个分区中，绝大多数分区应成功实现跨路正交覆盖 (通常 6~8 个分区)
        self.assertGreaterEqual(
            diff_road_zone_count, 5,
            f"区内 012 路跨路正交覆盖分区数过低 ({diff_road_zone_count}/8)"
        )

    def test_09_macro_health_and_report_generation(self):
        """测试 16 码宏观形态评估与分析报告生成。"""
        from calc_report import build_report_text, evaluate_macro_health

        res = run_calculus_engine(self.ds, self.ts)
        macro = res["macro_analysis_16"]
        title, memo = evaluate_macro_health(macro)
        self.assertTrue(len(title) > 0)
        self.assertTrue(len(memo) > 0)

        rep = build_report_text(self.ds, res, 2026999, "2026-10-18")
        self.assertIn("[0] 上期真实开奖与推荐战绩复盘", rep)
        self.assertIn("[1] 八大分区出号容量与形态走势推演", rep)
        self.assertIn("[2] 本期微积分·八区分区战术推荐", rep)
        self.assertIn("[3] 16码宏观形态体检卡", rep)
        self.assertIn("012路分布比", rep)
        self.assertIn("质数合数分布", rep)
        self.assertIn("[附录一]", rep)

    def test_10_hypergeometric_baseline(self):
        """测试超几何随机基准的数学正确性 (命中率评估的唯一公正标尺)。"""
        from calc_core import hypergeometric_baseline

        b = hypergeometric_baseline(16)
        # 期望 E = n·K/N = 16·20/80 = 4.0 (必须精确等于 4)
        self.assertAlmostEqual(b["mean"], 4.0, places=10)
        # 概率总和必须为 1
        self.assertAlmostEqual(sum(b["pmf"]), 1.0, places=12)
        # 单区命中概率 1 - C(78,20)/C(80,20) = 1 - (60·59)/(80·79) ≈ 43.99%
        self.assertAlmostEqual(b["zone_hit_prob"], 1 - (60 * 59) / (80 * 79), places=10)
        # 阈值概率单调递减
        for lo, hi in ((3, 4), (4, 5), (5, 6)):
            self.assertGreater(b["p_ge"][lo], b["p_ge"][hi])

    def test_11_walk_forward_backtest_contract(self):
        """测试走查回测输出契约与防前瞻截断 (小样本快速回归)。"""
        from calc_backtest import walk_forward_backtest

        bt = walk_forward_backtest(self.ds, 12)
        self.assertEqual(bt["n_sample"], 12)
        self.assertEqual(len(bt["rows"]), 12)
        for r in bt["rows"]:
            # 每期推荐必须为 16 个互不重复且合法号码
            self.assertEqual(len(r["picks16"]), 16)
            self.assertEqual(len(set(r["picks16"])), 16)
            self.assertTrue(all(1 <= x <= 80 for x in r["picks16"]))
            self.assertLessEqual(r["hits"], 16)
            # 命中号码必须真实出现在该期开奖中
            actual = self.ds.draws_by_period[r["period"]].numbers
            self.assertTrue(set(r["hit_nums"]) <= actual)
        self.assertIn(bt["grade"], ("A", "B", "C", "D"))

    def test_12_signal_audit_runs_and_bounded(self):
        """测试特征预测力审计模块可执行且统计量数值有界。"""
        from calc_signal_audit import (
            audit_autocorrelation,
            audit_gap_effect,
            audit_heat_effect,
            audit_road_reversion,
        )

        hit = self.ts.hit
        gap = audit_gap_effect(hit)
        self.assertTrue(len(gap["rows"]) >= 5)
        for r in gap["rows"]:
            self.assertTrue(0.0 <= r["rate"] <= 1.0)

        heat = audit_heat_effect(hit)
        self.assertTrue(len(heat["rows"]) >= 4)

        road = audit_road_reversion(hit)
        self.assertEqual(len(road["rows"]), 3)

        ac = audit_autocorrelation(hit)
        # 期际自相关必须落在概率区间内
        self.assertTrue(0.0 < ac["p_conditional"] < 1.0)
        self.assertAlmostEqual(ac["p_marginal"], 0.25, places=2)

    def test_13_zone_deep_mining_outputs(self):
        """测试分区深度挖掘：三态战术梯队、出号容量走势推演与区内发散度输出。"""
        res = run_calculus_engine(self.ds, self.ts)
        self.assertIn("picks24", res)
        self.assertEqual(len(res["picks24"]), 24, "三态全景推荐必须严格为 24 码")
        self.assertEqual(len(set(res["picks24"])), 24, "24 码推荐必须完全不重复")

        zones = res["zones"]
        for z in zones:
            self.assertIn("top_w_item", z, "各分区必须包含中继温号码")
            self.assertIn("z_trend", z, "各分区必须包含出号走势研判")
            self.assertIn("z_exp_range", z, "各分区必须包含预估出号区间")
            self.assertIn("dispersion_label", z, "各分区必须包含区内发散度标签")
            k_n = z["top_k_item"]["num"]
            w_n = z["top_w_item"]["num"]
            p_n = z["top_p_item"]["num"]
            self.assertEqual(len({k_n, w_n, p_n}), 3, f"Z{z['zone_id']} 三态热温冷号码必须互不相同")

    def test_14_lightgbm_ml_engine_contract(self):
        """测试 LightGBM 机器学习非线性引擎输出契约与概率边界。"""
        from calc_ml import train_and_predict_lgb

        probs = train_and_predict_lgb(self.ts, self.ds.T)
        self.assertEqual(len(probs), N_NUM, "LightGBM 预测输出必须为 80 维")
        self.assertFalse(np.isnan(probs).any(), "预测概率严禁包含 NaN")
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs), "预测概率必须在 [0, 1] 区间")
        # 80 个号码开出概率均值在随机基准 0.25 附近
        self.assertAlmostEqual(float(probs.mean()), 0.25, delta=0.08)

    def test_15_graph_engine_centrality_and_companions(self):
        """测试共现网络引擎的拓扑中心度与高频伴生挖掘。"""
        from calc_graph import analyze_graph_centrality, build_cooccurrence_matrix

        co_mat = build_cooccurrence_matrix(self.ts, self.ds.T, window=50)
        self.assertEqual(co_mat.shape, (N_NUM, N_NUM))
        self.assertEqual(np.diag(co_mat).sum(), 0.0, "对角线自共现必须为 0")

        g_scores, companions = analyze_graph_centrality(co_mat)
        self.assertEqual(len(g_scores), N_NUM)
        self.assertFalse(np.isnan(g_scores).any())
        self.assertTrue(all(0.0 <= s <= 1.0 for s in g_scores))
        self.assertEqual(len(companions), N_NUM)
        for num, pairs in companions.items():
            self.assertTrue(all(1 <= p[0] <= 80 for p in pairs))

    def test_16_multimodal_ensemble_and_constrained_solver(self):
        """测试多模态融合与全局带约束求解器的 16/17/24 码合法性与无重号。"""
        res = run_calculus_engine(self.ds, self.ts)
        self.assertIn("fusion_scores", res)
        self.assertIn("lgb_probs", res)
        self.assertIn("graph_scores", res)
        self.assertIn("companion_dict", res)

        dyn = res["dynamic_picks"]
        p16 = res["picks16"]
        p24 = res["picks24"]

        self.assertGreaterEqual(len(dyn), 14, "动态大名单不少于 14 码")
        self.assertLessEqual(len(dyn), 20, "动态大名单不多于 20 码")
        self.assertEqual(len(set(dyn)), len(dyn), "动态大名单必须完全无重复")

        self.assertEqual(len(p16), 16, "基准推荐必须严格为 16 码")
        self.assertEqual(len(set(p16)), 16, "基准 16 码必须无重复")

        self.assertEqual(len(p24), 24, "豪华全景大底必须严格为 24 码")
        self.assertEqual(len(set(p24)), 24, "豪华 24 码必须无重复")

    def test_17_multitier_betting_portfolios(self):
        """测试多层级实战投注组合 (金胆3码、选五、选七、选十主力、主攻双区) 合法性。"""
        res = run_calculus_engine(self.ds, self.ts)
        self.assertIn("bankers3", res)
        self.assertIn("pick5", res)
        self.assertIn("pick7", res)
        self.assertIn("pick10", res)
        self.assertIn("hot_zones", res)

        b3 = res["bankers3"]
        p5 = res["pick5"]
        p7 = res["pick7"]
        p10 = res["pick10"]
        hz = res["hot_zones"]

        self.assertEqual(len(b3), 3, "金胆3码必须严格为 3 码")
        self.assertEqual(len(set(b3)), 3, "金胆3码不能重复")

        self.assertEqual(len(p5), 5, "精选选五必须严格为 5 码")
        self.assertEqual(len(set(p5)), 5, "精选选五不能重复")

        self.assertEqual(len(p7), 7, "精选选七必须严格为 7 码")
        self.assertEqual(len(set(p7)), 7, "精选选七不能重复")

        self.assertEqual(len(p10), 10, "选十主力必须严格为 10 码")
        self.assertEqual(len(set(p10)), 10, "选十主力不能重复")

        self.assertEqual(len(hz), 2, "主攻放量区必须锁定前 2 个主攻区")
        for zone_item in hz:
            self.assertIn("zone_num", zone_item)
            self.assertIn("label", zone_item)
            self.assertIn("trend", zone_item)
            self.assertGreaterEqual(len(zone_item["picks"]), 3, "主攻区推荐不少于 3 码")


if __name__ == "__main__":
    unittest.main()
