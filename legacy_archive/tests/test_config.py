# -*- coding: utf-8 -*-
"""kl8.config 配置中心单测（R48-a）。

覆盖口径：
1. 默认配置自身合法，Σω = 1.00；
2. 深合并语义（稀疏补丁不冲掉同级其它字段）；
3. **正反用例**：Σω、α、λ、gold<main、k≤池大小 五类强校验，合法通过 / 非法抛 ConfigError；
4. JSON 文件加载、未知字段拦截、致命 JSON 语法错误转中文 ConfigError；
5. to_dict/from_dict 往返一致、config_hash 稳定性与敏感性；
6. CLI 覆盖映射（--k/--budget/--warmup/--pool-size/--seed/--history）。
"""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from kl8.config import (
    AppConfig,
    CoveringConfig,
    ScoreWeights,
    config_hash,
    default_config,
    from_dict,
    load_config,
    merge_cli_overrides,
    to_dict,
    validate_config,
)
from kl8.errors import ConfigError, KL8Error


class TestConfigDefaults(unittest.TestCase):
    """默认配置与基础结构。"""

    def test_default_config_is_valid_and_weights_sum_to_one(self) -> None:
        """默认配置通过校验，且 BaseScore 七维权重之和精确为 1。"""
        cfg = default_config()
        self.assertIsInstance(cfg, AppConfig)
        weights = cfg.weights
        total = (
            weights.follow_10
            + weights.follow_25
            + weights.follow_50
            + weights.follow_all
            + weights.co
            + weights.trend
            + weights.omission
        )
        self.assertAlmostEqual(total, 1.0, places=12)
        self.assertEqual(cfg.covering.k, 5)
        self.assertEqual(cfg.covering.t, 4)
        self.assertLess(cfg.pool.gold_max, cfg.pool.main_min)

    def test_error_hierarchy_is_kl8_error(self) -> None:
        """ConfigError 必须继承 KL8Error（统一兜底与分类的前提）。"""
        self.assertTrue(issubclass(ConfigError, KL8Error))

    def test_validate_config_rejects_non_config_object(self) -> None:
        """validate_config 收到非 AppConfig 时抛 ConfigError，而不是 AttributeError。"""
        with self.assertRaises(ConfigError):
            validate_config({"seed": 1})  # type: ignore[arg-type]


class TestDeepMerge(unittest.TestCase):
    """深合并与嵌套补丁。"""

    def test_deep_merge_keeps_siblings(self) -> None:
        """只覆盖 weights.co 与 trend 时，同级的 follow/omission 等必须保持默认值。"""
        cfg = load_config({"weights": {"co": 0.02, "trend": 0.08}})
        self.assertAlmostEqual(cfg.weights.co, 0.02, places=12)
        self.assertAlmostEqual(cfg.weights.trend, 0.08, places=12)
        self.assertAlmostEqual(cfg.weights.follow_10, 0.02, places=12)
        # 合计 = 0.02+0.08+0.20+0.25+0.02+0.08+0.35 = 1.00
        self.assertAlmostEqual(
            cfg.weights.follow_10
            + cfg.weights.follow_25
            + cfg.weights.follow_50
            + cfg.weights.follow_all
            + cfg.weights.co
            + cfg.weights.trend
            + cfg.weights.omission,
            1.0,
            places=12,
        )

    def test_none_overrides_returns_defaults(self) -> None:
        """load_config(None) 等价于默认配置。"""
        self.assertEqual(load_config(None), default_config())

    def test_unknown_nested_field_rejected(self) -> None:
        """嵌套未知字段（拼写错误）必须被拦截，避免静默失效。"""
        with self.assertRaises(ConfigError):
            load_config({"weights": {"follow10": 0.1}})

    def test_unknown_top_level_field_rejected(self) -> None:
        """顶层未知字段必须被拦截。"""
        with self.assertRaises(ConfigError):
            load_config({"weight": {}})


class TestStrictValidation(unittest.TestCase):
    """五类强校验的正反用例（契约 R48-a）。"""

    def test_weights_sum_violation_rejected(self) -> None:
        """Σω ≠ 1 必须抛 ConfigError（此处 1.75）。"""
        with self.assertRaises(ConfigError) as ctx:
            load_config({"weights": {"co": 0.9}})
        self.assertIn("权重之和", str(ctx.exception))

    def test_weights_sum_positive_boundary_accepted(self) -> None:
        """Σω 在容差内（差 5e-10）应当通过，验证容差不是过严。"""
        cfg = load_config({"weights": {"co": 0.05 + 5e-10}})
        self.assertAlmostEqual(cfg.weights.co, 0.05, delta=1e-8)

    def test_alpha_out_of_range_rejected_low(self) -> None:
        """α = 0.49 低于 0.5 必须被拒。"""
        with self.assertRaises(ConfigError):
            load_config({"markov": {"alpha": 0.49}})

    def test_alpha_out_of_range_rejected_high(self) -> None:
        """α = 2.01 高于 2.0 必须被拒。"""
        with self.assertRaises(ConfigError):
            load_config({"markov": {"alpha": 2.01}})

    def test_alpha_boundary_values_accepted(self) -> None:
        """α 取闭区间端点 0.5 / 2.0 均合法。"""
        for alpha in (0.5, 2.0):
            cfg = load_config({"markov": {"alpha": alpha}})
            self.assertAlmostEqual(cfg.markov.alpha, alpha, places=12)

    def test_lambda_out_of_range_rejected(self) -> None:
        """λ 超出 [0.05, 0.10] 必须被拒（例：0.2 与 0.01）。"""
        for lam in (0.2, 0.01):
            with self.subTest(lam=lam):
                with self.assertRaises(ConfigError):
                    load_config({"weights": {"lambda_vote": lam}})

    def test_lambda_boundary_values_accepted(self) -> None:
        """λ 取闭区间端点 0.05 / 0.10 均合法。"""
        for lam in (0.05, 0.10):
            cfg = load_config({"weights": {"lambda_vote": lam}})
            self.assertAlmostEqual(cfg.weights.lambda_vote, lam, places=12)

    def test_gold_not_less_than_main_rejected(self) -> None:
        """gold_max >= main_min 违反"金胆池严格小于复式池"，必须被拒。"""
        with self.assertRaises(ConfigError):
            load_config({"pool": {"gold_max": 12, "main_min": 10}})

    def test_gold_equal_main_min_rejected(self) -> None:
        """gold_max == main_min 属于边界违规（要求严格小于）。"""
        with self.assertRaises(ConfigError):
            load_config({"pool": {"gold_min": 3, "gold_max": 10, "main_min": 10}})

    def test_k_larger_than_pool_rejected(self) -> None:
        """k > 覆盖池大小必须被拒（组合无意义）。"""
        with self.assertRaises(ConfigError):
            load_config({"covering": {"k": 25, "max_pool_v": 20}})

    def test_k_equal_pool_accepted(self) -> None:
        """k == 池大小属于合法边界（整池一注）。"""
        cfg = load_config({"covering": {"k": 20, "t": 20, "max_pool_v": 20, "budget": 5}})
        self.assertEqual(cfg.covering.k, 20)

    def test_t_larger_than_k_rejected(self) -> None:
        """t > k 的保底命题不成立，必须被拒。"""
        with self.assertRaises(ConfigError):
            load_config({"covering": {"k": 5, "t": 6}})

    def test_windows_not_increasing_rejected(self) -> None:
        """窗口必须严格递增 short<mid<long，否则动量语义反转。"""
        with self.assertRaises(ConfigError):
            load_config({"windows": {"short": 50, "mid": 25, "long": 10}})

    def test_r2_interval_inverted_rejected(self) -> None:
        """R2 的 [f10_low, f10_high] 区间下界大于上界必须被拒。"""
        with self.assertRaises(ConfigError):
            load_config({"rules": {"r2_f10_low": 0.6, "r2_f10_high": 0.2}})

    def test_budget_over_max_rejected(self) -> None:
        """预算超过 max_budget 必须被拒。"""
        with self.assertRaises(ConfigError):
            load_config({"covering": {"budget": 500, "max_budget": 200}})

    def test_backtest_warmup_not_greater_than_t0_min_rejected(self) -> None:
        """warmup <= t0_min 时永远凑不出回测样本，必须被拒。"""
        with self.assertRaises(ConfigError):
            load_config({"backtest": {"warmup": 60, "t0_min": 60}})

    def test_bool_field_type_checked(self) -> None:
        """布尔字段传字符串必须被拒（避免 truthy 陷阱）。"""
        with self.assertRaises(ConfigError):
            load_config({"markov": {"diag_split": "yes"}})

    def test_int_field_rejects_bool(self) -> None:
        """整数字段传 True 必须被拒（bool 是 int 子类，需显式排除）。"""
        with self.assertRaises(ConfigError):
            load_config({"backtest": {"default_n": True}})


class TestJsonFileLoading(unittest.TestCase):
    """JSON 配置文件加载。"""

    def test_load_json_file_and_override(self) -> None:
        """文件中的补丁 + overrides 补丁应深合并，overrides 优先。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(
                json.dumps({"seed": 123, "covering": {"budget": 15}}),
                encoding="utf-8",
            )
            cfg = load_config({"seed": 456}, path=path)
            self.assertEqual(cfg.seed, 456)
            self.assertEqual(cfg.covering.budget, 15)

    def test_missing_file_raises_config_error(self) -> None:
        """配置文件不存在 → ConfigError（而不是 FileNotFoundError）。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_config(None, path=Path(tmp) / "absent.json")

    def test_broken_json_raises_config_error(self) -> None:
        """JSON 语法错误 → ConfigError，消息应包含行号线索。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text('{"seed": 1,}', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(None, path=path)

    def test_prize_table_string_keys_normalized_to_int(self) -> None:
        """JSON 的字符串键 "5" 必须归一为 int 5（奖金表结算依赖 int 键）。"""
        cfg = from_dict({"prize_table": {"pick5": {"5": 1000, "4": 21}}})
        prize = cfg.prize_table["pick5"]
        self.assertIsNotNone(prize)
        self.assertEqual(prize[5], 1000)  # type: ignore[index]
        self.assertEqual(prize[4], 21)  # type: ignore[index]


class TestSerializationAndHash(unittest.TestCase):
    """序列化往返与配置指纹。"""

    def test_dict_roundtrip_equality(self) -> None:
        """to_dict → from_dict 往返后配置完全相等。"""
        cfg = load_config({"seed": 7, "covering": {"budget": 12}})
        self.assertEqual(from_dict(to_dict(cfg)), cfg)

    def test_to_dict_is_deep_copy(self) -> None:
        """to_dict 返回深拷贝：改字典不应污染原配置。"""
        cfg = default_config()
        payload = to_dict(cfg)
        payload["covering"]["budget"] = 999  # type: ignore[index]
        self.assertEqual(cfg.covering.budget, 20)

    def test_config_hash_stable_for_same_config(self) -> None:
        """同配置两次哈希必须一致（回测血缘可复现）。"""
        first = config_hash(default_config())
        second = config_hash(default_config())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_config_hash_sensitive_to_change(self) -> None:
        """任一参数变化都必须改变哈希。"""
        base = config_hash(default_config())
        changed = config_hash(load_config({"seed": 42}))
        self.assertNotEqual(base, changed)

    def test_to_dict_rejects_non_config(self) -> None:
        """to_dict 收到非配置对象时抛 ConfigError。"""
        with self.assertRaises(ConfigError):
            to_dict(42)  # type: ignore[arg-type]


class TestCliOverrides(unittest.TestCase):
    """CLI → 配置映射。"""

    def test_merge_cli_overrides_maps_fields(self) -> None:
        """--k/--budget/--warmup/--pool-size/--seed/--history 全部落到正确字段。"""
        args = argparse.Namespace(
            history="other.txt",
            seed=7,
            pool_size=14,
            k=7,
            budget=10,
            warmup=200,
        )
        cfg = merge_cli_overrides(default_config(), args)
        self.assertEqual(cfg.history_path, "other.txt")
        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.pool.main_default, 14)
        self.assertEqual(cfg.covering.play, "pick7")
        self.assertEqual(cfg.covering.k, 7)
        self.assertEqual(cfg.covering.t, 5)
        self.assertEqual(cfg.covering.budget, 10)
        self.assertEqual(cfg.backtest.warmup, 200)

    def test_merge_cli_overrides_noop_when_all_none(self) -> None:
        """全部参数为 None 时原配置不变（返回同一对象）。"""
        args = argparse.Namespace(history=None, seed=None, pool_size=None, k=None, budget=None, warmup=None)
        base = default_config()
        self.assertIs(merge_cli_overrides(base, args), base)

    def test_merge_cli_overrides_tolerates_missing_attrs(self) -> None:
        """argparse.Namespace 缺少属性时不得抛 AttributeError。"""
        cfg = merge_cli_overrides(default_config(), argparse.Namespace())
        self.assertEqual(cfg, default_config())

    def test_merge_cli_overrides_invalid_still_rejected(self) -> None:
        """CLI 组合非法（如 k=9 无对应玩法但 t 未联动时）必须仍被校验拦截或合法落地。"""
        args = argparse.Namespace(k=9)
        with self.assertRaises(ConfigError):
            merge_cli_overrides(default_config(), args)


class TestConfigSections(unittest.TestCase):
    """分层配置对象类型与不可变性。"""

    def test_sections_are_frozen_dataclasses(self) -> None:
        """子配置必须 frozen：尝试改写应抛 FrozenInstanceError（防止运行期漂移）。"""
        weights = ScoreWeights()
        with self.assertRaises(Exception):
            weights.co = 0.5  # type: ignore[misc]

    def test_covering_default_matches_play(self) -> None:
        """默认 covering 的 k/t/budget 与 pick5 玩法自洽。"""
        covering = CoveringConfig()
        self.assertEqual(covering.play, "pick5")
        self.assertEqual(covering.budget_by_play["pick5"], covering.budget)


if __name__ == "__main__":
    unittest.main()