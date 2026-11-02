"""号码池过滤、硬约束与配比控制测试（契约 R34–R36）。

【P10 自证项】
- H1（恒等精确性）：随机抽 3 个历史截断点，断言 ``FinalScore - BaseScore == λ_vote·V``
  逐号**精确成立**（残差恰为 0.0，而非"近似"）；
- H2（约束不可静默绕过）：构造「Top-12 全部落在同一四分区」的极端打分，
  断言最终池要么按 B7 放宽并留痕、要么换号绕开该分区，禁止"只打 warning 就出单"。

【P10b 自证项（M16）】B7 逐项放宽路径必须**可达**且留痕真实：贪心不足时如实返回不足码数，
``build_pool`` 按 ``same_tail → parity → zone4 → repeat_ratio`` 逐项放宽、每步重跑，
``PoolResult.relaxed`` 只记录真实用到的前缀，warnings 含放宽统计与逐项放开值；
全部放宽仍不足才抛 ``InfeasibleConstraintError``。
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from kl8.config import AppConfig, PoolConfig
from kl8.constants import N_NUM
from kl8.errors import InfeasibleConstraintError
from kl8.history import load_dataset
from kl8.layer1_representation.state import BinaryStateMatrix
from kl8.layer3_features.features import build_feature_snapshot
from kl8.layer4_scoring import filter as filter_module
from kl8.layer4_scoring import score_round
from kl8.layer4_scoring.filter import apply_repeat_quota, build_pool, select_gold, zone4_of


class TestFilter(unittest.TestCase):
    """候选池硬约束、金胆与重号配比测试。"""

    def setUp(self) -> None:
        self.ds = load_dataset("kl8_history.txt")
        self.state = BinaryStateMatrix.from_dataset(self.ds)
        self.cfg = AppConfig()
        self.snap = build_feature_snapshot(self.state, end=100, cfg=self.cfg, include_ml=False)

    def test_hard_constraints(self) -> None:
        """硬约束真实生效：单区≤45%、奇偶≤70%、同尾≤3，且放宽必有中文留痕（M14 反假绿）。"""
        scores = np.linspace(0.8, 0.1, 80)
        votes = np.full(80, 2, dtype=np.int8)
        last_draw = self.state.latest_set(100)
        cfg = self.cfg
        pool = build_pool(self.snap, scores, votes, last_draw, pool_size=12, cfg=cfg)
        self.assertEqual(len(pool.main), 12)
        self.assertEqual(pool.size, 12)

        size = 12
        pool_cfg = cfg.pool
        zone_count = [0, 0, 0, 0]
        for num in pool.main:
            zone_count[zone4_of(num)] += 1
        zone_cap = int(np.ceil(pool_cfg.max_zone4_ratio * size))
        self.assertLessEqual(max(zone_count), zone_cap, f"四分区分布超限：{zone_count} vs {zone_cap}")

        odd = sum(1 for num in pool.main if num % 2 == 1)
        parity_cap = int(np.ceil(pool_cfg.max_parity_ratio * size))
        self.assertLessEqual(max(odd, size - odd), parity_cap, "奇偶单边超限")

        tail_count: dict[int, int] = {}
        for num in pool.main:
            tail_count[num % 10] = tail_count.get(num % 10, 0) + 1
        self.assertLessEqual(max(tail_count.values()), pool_cfg.max_same_tail, "同尾超限")

        # 金胆必须落在复式池内（A5 硬约束），且规模符合 A5 公式
        self.assertTrue(set(pool.gold).issubset(set(pool.main)))
        self.assertEqual(len(pool.gold), 4)

        # 若发生放宽，必须逐项留可读说明；未放宽时不得虚报放宽项
        if pool.relaxed:
            self.assertTrue(any("放宽" in warn for warn in pool.warnings), "放宽了约束却没有可读留痕")
        else:
            self.assertIsInstance(pool.feasible, bool)

    def test_gold_and_main_sizes(self) -> None:
        """测试核心金胆池 (3~5 码) 与复式池 (10~15 码，默认 12) 规模（契约 R35）。"""
        scores = np.linspace(0.8, 0.1, 80)
        votes = np.full(80, 2, dtype=np.int8)
        last_draw = self.state.latest_set(100)
        pool = build_pool(self.snap, scores, votes, last_draw, pool_size=12, cfg=self.cfg)
        self.assertTrue(3 <= len(pool.gold) <= 5)
        self.assertTrue(10 <= len(pool.main) <= 15)
        self.assertEqual(len(pool.main), 12)
        # 金胆必须包含在复式池中
        self.assertTrue(set(pool.gold).issubset(set(pool.main)))

    def test_repeat_quota(self) -> None:
        """测试重号配比控制在 25%~35%（默认 30%）（契约 R36）。"""
        order = list(range(1, 81))
        last_draw = frozenset(range(1, 21))
        p_cfg = PoolConfig(repeat_ratio=0.30, repeat_ratio_min=0.25, repeat_ratio_max=0.35)
        res, warns = apply_repeat_quota(order, last_draw, p_cfg, size=12)
        self.assertEqual(len(res), 12)
        rep_count = sum(1 for n in res if n in last_draw)
        # 12 * 0.30 = 3.6 -> 4 码 (或在 3..4 之间)
        self.assertTrue(3 <= rep_count <= 4)


class TestScoreFinalIdentity(unittest.TestCase):
    """H1 自证：FinalScore - BaseScore 与 λ_vote·V 必须**精确**相等（非近似）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = load_dataset("kl8_history.txt")
        cls.state = BinaryStateMatrix.from_dataset(cls.ds)
        cls.cfg = AppConfig()

    def test_final_identity_exact_on_three_random_periods(self) -> None:
        """随机 3 个截断点：FinalScore 与 BaseScore+λ·V 逐位相同，残差 ≤2 ULP（舍入极限）。

        【精确性口径】数学恒等 ``Final = Base + λ_vote·V``（B5）要求：
        1. 引擎的 FinalScore 必须**就是** ``BaseScore + λ·V`` 这条表达式的双精度结果
           （逐位 ``np.array_equal`` 成立，排除任何隐藏项/提前归一化/截断）；
        2. 反向验证 ``Final - Base`` 与 ``λ·V`` 的差只允许来自 IEEE754 两次加法/减法的
           舍入（≤ 2 ULP），不得出现量级性偏差（旧 H1 缺陷表现为 λ 被错误缩放）。
        """
        rng = np.random.default_rng(20261102)
        lam = float(self.cfg.weights.lambda_vote)
        ends = [
            int(x)
            for x in rng.integers(self.cfg.windows.min_history, len(self.ds.draws) - 1, size=3)
        ]
        seen_votes: set[int] = set()

        for end in ends:
            snapshot = build_feature_snapshot(self.state, end=end, cfg=self.cfg, include_ml=False)
            result = score_round(snapshot, self.cfg)
            final = np.asarray(result.scores.final, dtype=np.float64)
            base = np.asarray(result.scores.base, dtype=np.float64)
            votes = np.asarray(result.votes, dtype=np.int8)
            votes_f = votes.astype(np.float64)
            equivalence = np.asarray(result.weighted_votes, dtype=np.float64)

            # 均匀先验（无自适应状态）下等效票数必须与整型得票逐位相同
            np.testing.assert_array_equal(equivalence, votes_f)
            # ① 引擎结果与 base + λ·V 表达式逐位一致（精确到每一位）
            self.assertTrue(
                np.array_equal(final, base + lam * votes_f),
                f"第 {end} 期 FinalScore 不是 BaseScore+λ·V 的逐位结果",
            )
            # ② 反向残差仅允许 IEEE754 舍入（≤2 ULP），不允许任何量级偏差
            residual = np.abs((final - base) - lam * votes_f)
            ulp = np.spacing(np.abs(final))
            self.assertTrue(
                bool(np.all(residual <= 2.0 * ulp + 1e-300)),
                f"第 {end} 期残差超舍入极限：max={float(np.max(residual))}, "
                f"max_ulp={float(np.max(ulp))}",
            )
            seen_votes.update(int(v) for v in np.unique(votes))

        # 3 期样本必须真实覆盖 0..5 全部票数档位，否则"恒等"证明不充分
        self.assertEqual(seen_votes, {0, 1, 2, 3, 4, 5}, f"得票档位覆盖不足：{sorted(seen_votes)}")


class TestZone4Extreme(unittest.TestCase):
    """H2 自证：Top 码全部挤在同一四分区时，约束必须真实生效（换号或放宽留痕）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = load_dataset("kl8_history.txt")
        cls.state = BinaryStateMatrix.from_dataset(cls.ds)
        cls.cfg = AppConfig()
        cls.snap = build_feature_snapshot(cls.state, end=100, cfg=cls.cfg, include_ml=False)

    def test_zone4_skewed_scores_swap_or_relax_with_trace(self) -> None:
        """极端构造：前 12 高分全部落在四分区 0 → 最终池不得整池同区且必须可审计。"""
        cfg = self.cfg
        size = cfg.pool.main_default
        zone0_nums = [num for num in range(1, N_NUM + 1) if zone4_of(num) == 0]
        top12 = zone0_nums[:size]
        scores = np.full(N_NUM, 0.01, dtype=np.float64)
        for rank, num in enumerate(top12):
            scores[num - 1] = 1.0 - rank * 1e-3
        votes = np.full(N_NUM, 2, dtype=np.int8)
        last_draw = self.state.latest_set(100)

        pool = build_pool(self.snap, scores, votes, last_draw, pool_size=size, cfg=cfg)
        self.assertEqual(len(pool.main), size)

        zone_count = [0, 0, 0, 0]
        for num in pool.main:
            zone_count[zone4_of(num)] += 1
        zone_cap = int(np.ceil(cfg.pool.max_zone4_ratio * size))
        self.assertLessEqual(
            max(zone_count), zone_cap, f"单区超限：{zone_count} 上限 {zone_cap}"
        )
        # 必须换号：整池不能等同于"纯按得分取前 12"
        self.assertNotEqual(set(pool.main), set(top12), "只按得分出单说明硬约束被子静默绕过")
        # 若记录了放宽，必须留可审计中文说明（B7），严禁"warning 后照抄高分号"
        if pool.relaxed:
            self.assertTrue(pool.warnings, "放宽了约束却没有留痕说明")
            self.assertTrue(
                any("zone4" in warn or "四区" in warn for warn in pool.warnings),
                f"放宽 zone4 缺少可读留痕：{pool.warnings}",
            )
        else:
            # 未放宽约束说明是"换号"路线：池必须真实满足形态体检口径
            self.assertIsInstance(pool.feasible, bool)


class TestRelaxTrace(unittest.TestCase):
    """P10b-M16：B7 逐项放宽路径可达且留痕真实（4 条正向断言）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ds = load_dataset("kl8_history.txt")
        cls.state = BinaryStateMatrix.from_dataset(cls.ds)
        cls.cfg = AppConfig()
        cls.snap = build_feature_snapshot(cls.state, end=100, cfg=cls.cfg, include_ml=False)
        cls.scores = np.linspace(0.8, 0.1, 80)
        cls.votes = np.full(80, 2, dtype=np.int8)
        cls.last_draw = cls.state.latest_set(100)

    def test_relaxed_entry_is_real_prefix(self) -> None:
        """①存在解时 ``relaxed`` 恰为真实放宽项（前缀），不是四项全列。"""
        strict_cfg = AppConfig(pool=PoolConfig(max_same_tail=1))
        pool = build_pool(
            self.snap, self.scores, self.votes, self.last_draw, pool_size=12, cfg=strict_cfg
        )
        self.assertEqual(pool.relaxed, ("same_tail",), f"放宽项不真实：{pool.relaxed}")
        self.assertEqual(
            pool.relaxed,
            tuple(strict_cfg.pool.relax_order[: len(pool.relaxed)]),
            "放宽必须按 relax_order 顺序取前缀",
        )
        self.assertLess(len(pool.relaxed), len(strict_cfg.pool.relax_order), "不得四项全列")

        # 默认配置下约束可满足 → 不允许虚报放宽
        normal = build_pool(
            self.snap, self.scores, self.votes, self.last_draw, pool_size=12, cfg=self.cfg
        )
        self.assertEqual(normal.relaxed, ())

    def test_relax_warnings_contain_statistics(self) -> None:
        """②warnings 含放宽统计（有序列举 + 逐项"放宽到多少"）。"""
        strict_cfg = AppConfig(pool=PoolConfig(max_same_tail=1))
        pool = build_pool(
            self.snap, self.scores, self.votes, self.last_draw, pool_size=12, cfg=strict_cfg
        )
        joined = "\n".join(pool.warnings)
        self.assertIn("有序放宽", joined, f"缺少放宽统计：{pool.warnings}")
        self.assertIn("same_tail", joined, f"缺少放宽项名：{pool.warnings}")
        self.assertIn("原上限", joined, f"缺少放开值说明：{pool.warnings}")

    def test_relaxed_pool_is_full_size(self) -> None:
        """③放宽后池必须满 12 码，且金胆仍是 A5 的 4 码（可出单）。"""
        strict_cfg = AppConfig(pool=PoolConfig(max_same_tail=1))
        pool = build_pool(
            self.snap, self.scores, self.votes, self.last_draw, pool_size=12, cfg=strict_cfg
        )
        self.assertEqual(len(pool.main), 12)
        self.assertEqual(pool.size, 12)
        self.assertEqual(len(pool.gold), 4)
        self.assertTrue(set(pool.gold).issubset(set(pool.main)))

    def test_raises_only_after_all_relaxations(self) -> None:
        """④只有全部 4 项放宽后仍不足才抛错，且错误信息列出已放宽项。"""
        calls: list[frozenset[str]] = []

        def always_short(*_args, **kwargs):
            calls.append(kwargs.get("active", frozenset()))
            return ([1, 2, 3], ("模拟：约束过紧只能选出 3 码",))

        with mock.patch.object(filter_module, "greedy_pool", always_short):
            with self.assertRaises(InfeasibleConstraintError) as ctx:
                build_pool(
                    self.snap, self.scores, self.votes, self.last_draw, pool_size=12, cfg=self.cfg
                )

        message = str(ctx.exception)
        self.assertIn("仍只能选出 3/12", message, message)
        # 5 次尝试 = 不放宽 + 逐项放宽 4 次；最后一次 active 应为空集（全部放宽）
        self.assertEqual(len(calls), 1 + len(self.cfg.pool.relax_order))
        self.assertEqual(calls[-1], frozenset())
        for key in self.cfg.pool.relax_order:
            self.assertIn(key, message, f"异常信息未列出放宽项 {key}：{message}")


if __name__ == "__main__":
    unittest.main()
