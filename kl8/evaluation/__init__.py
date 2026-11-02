"""回测与防穿越评估模块（契约 R42–R47、§3.12）。

提供：
- Walk-Forward 滚动前瞻时序回测与窗口生成器（run_walkforward, iter_windows）
- 核心指标统计：Lift@K、单侧 t 检验、高奖级对比、规则归因与资金曲线（metrics）
- 严格防穿越守卫、未来投毒不变性测试与代码审计（leakage）
"""

from kl8.evaluation.leakage import (
    assert_no_lookahead,
    audit_grep_no_shuffle,
    fingerprint_snapshot,
    max_accessed_t,
    poison_future,
)
from kl8.evaluation.metrics import (
    AttributionReport,
    BankrollStat,
    LiftStat,
    RuleStat,
    TierStat,
    bankroll,
    high_tier_rate,
    lift_at_k,
    rule_attribution,
    self_check,
    t_test_one_sided,
)
from kl8.evaluation.walkforward import iter_windows, run_walkforward

__all__ = [
    # walkforward
    "iter_windows",
    "run_walkforward",
    # metrics
    "LiftStat",
    "TierStat",
    "RuleStat",
    "AttributionReport",
    "BankrollStat",
    "lift_at_k",
    "high_tier_rate",
    "rule_attribution",
    "bankroll",
    "t_test_one_sided",
    "self_check",
    # leakage
    "max_accessed_t",
    "assert_no_lookahead",
    "poison_future",
    "fingerprint_snapshot",
    "audit_grep_no_shuffle",
]
