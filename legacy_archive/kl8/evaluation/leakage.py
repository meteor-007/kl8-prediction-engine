"""严格防穿越审计与前瞻泄露守卫（契约 R47、C4、§3.12）。

防穿越三条硬等价断言（§6.3）：
1. 唯一截断入口: truncated(ds, target_period)
2. 索引守卫: assert_no_lookahead(max_accessed_t <= as_of_t)，违反抛 LeakageError
3. 未来投毒逐位不变性: poison_future 污染未来数据后，当期特征指纹与预测排序完全相同
4. 禁止随机打乱: audit_grep_no_shuffle 扫描代码排除非法交叉验证
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

from kl8.datatypes import Dataset, Draw, FeatureSnapshot
from kl8.errors import LeakageError


def max_accessed_t(snapshot: FeatureSnapshot) -> int:
    """获取特征快照实际访问过的历史最大时间步索引（0-based）。"""
    return snapshot.as_of_index


def assert_no_lookahead(*, max_accessed_t: int, as_of_t: int) -> None:
    """断言无前瞻泄露，若访问了 > as_of_t 的时间步立即中断抛出 LeakageError。"""
    if max_accessed_t > as_of_t:
        raise LeakageError(
            f"防穿越守卫失败: 访问历史索引 {max_accessed_t} 超过当前期截断点 {as_of_t}！"
            "严禁在时间步 t 使用未来时间步的信息。"
        )


def poison_future(ds: Dataset, as_of_period: int, *, seed: int) -> Dataset:
    """对晚于 as_of_period 的所有期次开奖号码进行随机投毒污染（用于测试防穿越不变性）。

    @param ds 原始历史数据集
    @param as_of_period 截断期号（该期及更早的期次保持绝对纯净）
    @param seed 随机种子
    @return 包含未来投毒数据的全新 Dataset
    """
    rng = np.random.default_rng(seed)
    new_draws: list[Draw] = []

    for draw in ds.draws:
        if draw.period <= as_of_period:
            new_draws.append(draw)
        else:
            # 未来数据：替换为随机 20 码
            fake_nums = frozenset(int(x) for x in rng.choice(np.arange(1, 81), size=20, replace=False))
            poisoned_draw = Draw(
                date=draw.date,
                period=draw.period,
                numbers=fake_nums,
                idx=draw.idx,
            )
            new_draws.append(poisoned_draw)

    by_period = {d.period: d for d in new_draws}
    return Dataset(
        draws=tuple(new_draws),
        draws_by_period=by_period,
        source_path=ds.source_path,
        warnings=ds.warnings,
    )


def fingerprint_snapshot(snapshot: FeatureSnapshot) -> str:
    """计算 FeatureSnapshot 中全部特征数组的 SHA256 联合指纹哈希。"""
    hasher = hashlib.sha256()
    for name, arr in sorted(snapshot.arrays().items(), key=lambda x: x[0]):
        hasher.update(name.encode("utf-8"))
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def audit_grep_no_shuffle(root: str | Path = "kl8") -> tuple[str, ...]:
    """静态审计全仓源码，严禁任何 shuffle 或 KFold 型时间打乱操作（契约 R42、R47）。

    @param root 扫描根目录
    @return 命中非法特征的文件与行号元组（为空表示合规）
    """
    p_root = Path(root)
    forbidden_patterns = [
        re.compile(r"\bKFold\b"),
        re.compile(r"\bStratifiedKFold\b"),
        re.compile(r"\btrain_test_split\b"),
        re.compile(r"\bshuffle\s*=\s*True\b"),
        re.compile(r"random\.shuffle"),
    ]

    violations: list[str] = []
    for file_path in p_root.rglob("*.py"):
        if file_path.name == "leakage.py":
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                in_docstring = False
                doc_quote = ""
                for line_no, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    # 维护文档字符串状态
                    if not in_docstring:
                        if line.startswith('"""') or line.startswith("'''"):
                            doc_quote = line[:3]
                            if line.count(doc_quote) < 2:
                                in_docstring = True
                            continue
                        if line.startswith("#"):
                            continue
                    else:
                        if doc_quote in line:
                            in_docstring = False
                        continue

                    for pat in forbidden_patterns:
                        if pat.search(line):
                            violations.append(f"{file_path}:{line_no}: {line}")
        except (OSError, UnicodeDecodeError) as exc:
            # 单个文件不可读不应中断全仓审计，但必须留痕（禁止静默 pass）
            violations.append(f"{file_path}: 读取失败（{exc}），本次审计未覆盖该文件")

    return tuple(violations)
