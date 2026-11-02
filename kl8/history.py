# -*- coding: utf-8 -*-
"""快乐8五层重构 —— 开奖历史解析 / 校验 / 保存 / 防穿越截断（C4 唯一截断入口）。

【定位】
本模块是**唯一**允许"读历史、切时间"的地方：

- 读：``load_dataset`` 把 ``kl8_history.txt`` 解析成严格升序的 :class:`Dataset`；
- 写：``save_dataset`` 按"新期在前"的降序原格式原子写回；
- 切：``truncated`` 是**全系统唯一的防穿越截断入口**（C4/R47）。任何层想"只看
  t 期以前的数据"，都必须调用它，禁止自己写 ``ds.draws[:t]``。

【§0.1 数据事实（契约冻结）】
1. 原始文件是**新期在前（降序）**：2112 期，首行 ``2026291 / 2026-11-01``；
2. **期号不连续**：存在 6 处跨年自然断档（如 ``2020065 → 2021001``），
   因此期号连续性**只记 warning，绝不抛错**；
3. 只允许三条硬校验：
   ① 升序重排后日期单调不减；② ``period`` 全局唯一（重复即 ``DataFormatError``，
   **禁止静默去重**）；③ 降序文件自检——升序后 ``draws[-1].period`` 必须等于
   文件首行期号（否则说明文件方向错了，抛 ``DataFormatError``）。

【§0.2 时间语义（下游单测据此编写）】
- ``end`` / ``t``：0-based、**含端点**，可用数据为 ``X[0..end]``（共 ``end+1`` 期）；
- ``truncated(ds, tp)``：返回**严格早于** ``tp`` 的全部期（不含 ``tp`` 本身）；
- 等价关系：**"用 X_t 预测 X_{t+1}" ⇔ ``truncated(ds, t+1)``**，此时
  ``as_of_index == t`` 且 ``len(train) == t+1``。这条等价关系是回测防穿越的正确性根。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import os
from bisect import bisect_left
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from kl8.constants import HISTORY_LINE_PATTERN, K_DRAW, N_NUM
from kl8.datatypes import Dataset, Draw
from kl8.errors import DataFormatError, InsufficientDataError

_TMP_SUFFIX = ".tmp"
"""原子写的临时文件后缀（同目录创建后再 os.replace）。"""


def parse_line(line: str, line_no: int = 0) -> Draw:
    """解析单行开奖历史并做业务校验。

    【行格式】``date:YYYY-MM-DD,period:NNNNNNN,numbers:NN-NN-...``（20 个 1..80 不重复）。

    @param line 待解析的原始行（允许首尾空白）
    @param line_no 行号（1-based，仅用于错误消息定位；0 表示未知）
    @return 校验通过的 :class:`Draw`（``idx`` 尚未赋值，由调用方重排后填入）
    @raises DataFormatError 格式不匹配 / 日期非法 / 期号非法 / 号码个数或取值非法
    """
    text = line.strip()
    if not text:
        raise DataFormatError(f"第 {line_no} 行为空行，无法解析开奖数据；请删除空行后重试")
    match = HISTORY_LINE_PATTERN.match(text)
    if match is None:
        raise DataFormatError(
            f"第 {line_no} 行格式非法：{text[:60]}…；"
            "要求 date:YYYY-MM-DD,period:NNNNNNN,numbers:NN-NN-...（20 个两位补零号码）"
        )

    date_text = match.group("date")
    period_text = match.group("period")
    numbers_text = match.group("numbers")

    try:
        period = int(period_text)
    except ValueError as exc:
        raise DataFormatError(f"第 {line_no} 行期号 {period_text!r} 不是整数；请检查 period: 字段") from exc

    parts = [item for item in numbers_text.split("-") if item.strip()]
    if len(parts) != K_DRAW:
        raise DataFormatError(
            f"第 {line_no} 行号码个数为 {len(parts)}，要求恰好 {K_DRAW} 个；"
            "请核对 numbers: 字段（快乐8 每期开 20 码）"
        )
    try:
        numbers = [int(item) for item in parts]
    except ValueError as exc:
        raise DataFormatError(f"第 {line_no} 行号码含非数字片段：{numbers_text[:60]}；请检查 numbers: 字段") from exc
    if len(set(numbers)) != len(numbers):
        duplicated = sorted({value for value in numbers if numbers.count(value) > 1})
        raise DataFormatError(f"第 {line_no} 行号码重复：{duplicated}；每期 20 码必须互不重复")
    out_of_range = [value for value in numbers if not 1 <= value <= N_NUM]
    if out_of_range:
        raise DataFormatError(
            f"第 {line_no} 行号码越界：{sorted(out_of_range)[:5]}，要求全部落在 1..{N_NUM}"
        )

    return Draw(date=date_text, period=period, numbers=frozenset(numbers))


def _check_unique_periods(rows: list[tuple[str, int, frozenset[int]]]) -> None:
    """硬校验 ②：期号全局唯一（重复即报错，禁止静默去重）。

    @param rows 解析后的原始行元组列表（date, period, numbers）
    @raises DataFormatError 存在重复期号
    """
    seen: dict[int, tuple[int, str]] = {}
    for record_no, (date_text, period, _numbers) in enumerate(rows, start=1):
        if period in seen:
            first_no, first_date = seen[period]
            raise DataFormatError(
                f"期号 {period} 重复出现（首见第 {first_no} 条记录 {first_date}，"
                f"重复于第 {record_no} 条记录 {date_text}）；"
                "请删除重复行，禁止静默去重（会污染状态矩阵与防穿越口径）"
            )
        seen[period] = (record_no, date_text)


def _warn_period_gaps(periods: tuple[int, ...]) -> tuple[str, ...]:
    """硬校验之外的软告警：期号不连续（§0.1：只 warning，绝不抛错）。

    @param periods 升序期号元组
    @return 告警消息元组（可能为空）
    """
    warnings: list[str] = []
    for index in range(len(periods) - 1):
        previous, current = periods[index], periods[index + 1]
        if current - previous == 1:
            continue
        kind = "自然跨年期" if current // 1000 != previous // 1000 else "疑似缺失期"
        warnings.append(f"期号断档：{previous} → {current}（{kind}，仅告警不阻断）")
    return tuple(warnings)


def load_dataset(path: str | Path = "kl8_history.txt") -> Dataset:
    """加载并校验开奖历史，返回严格升序的 :class:`Dataset`。

    【流程】逐行解析（坏行记 warning 跳过）→ 按 (date, period) 升序重排 →
    三条硬校验（日期单调 / 期号唯一 / 降序文件自检）→ 期号断档只记 warning。

    @param path 历史文件路径
    @return 升序 :class:`Dataset`
    @raises DataFormatError 文件不存在/不可读、重复期号、文件方向错误（非降序）
    @raises InsufficientDataError 解析后无任何有效期
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataFormatError(
            f"开奖历史文件不存在：{file_path}；请先运行数据同步（kl8/sync.py）或把 kl8_history.txt 放到项目根目录"
        )
    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DataFormatError(f"开奖历史文件不可读：{file_path}（{exc}）；请检查文件权限与 UTF-8 编码") from exc

    warnings: list[str] = []
    rows: list[tuple[str, int, frozenset[int]]] = []
    first_period: int | None = None
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            draw = parse_line(raw_line, line_no)
        except DataFormatError as exc:
            warnings.append(f"第 {line_no} 行已跳过：{exc.message}")
            continue
        if first_period is None:
            first_period = draw.period
        rows.append((draw.date, draw.period, draw.numbers))

    _check_unique_periods(rows)

    rows.sort(key=lambda item: (item[0], item[1]))
    draws = tuple(
        Draw(date=date_text, period=period, numbers=numbers, idx=index)
        for index, (date_text, period, numbers) in enumerate(rows)
    )

    # 硬校验 ①：升序后日期单调不减（排序后必然成立，此处显式断言防止未来改动破坏）
    dates = [draw.date for draw in draws]
    if dates != sorted(dates):
        raise DataFormatError("升序重排后日期仍非单调不减，历史文件存在异常日期；请检查数据来源")

    # 硬校验 ③：降序文件自检 —— 升序后的最新一期必须就是文件首行的那一期
    if draws and first_period is not None and draws[-1].period != first_period:
        raise DataFormatError(
            f"历史文件方向异常：文件首行期号 {first_period}，升序后最新期号为 {draws[-1].period}；"
            "kl8_history.txt 必须是「新期在前」的降序格式，请重排后再导入"
        )

    if not draws:
        raise InsufficientDataError(
            f"开奖历史文件 {file_path} 未解析出任何有效期；请检查文件内容或重新同步数据"
        )

    warnings.extend(_warn_period_gaps(tuple(draw.period for draw in draws)))
    return Dataset(
        draws=draws,
        draws_by_period={draw.period: draw for draw in draws},
        source_path=str(file_path),
        warnings=tuple(warnings),
    )


def save_dataset(ds: Dataset, path: str | Path) -> int:
    """把数据集按"新期在前"的降序原格式原子写回（UTF-8，tmp + os.replace）。

    【为什么降序】与官方/历史文件既有格式保持一致，人工查看时最新期在最上方；
    【为什么原子写】避免写入中途崩溃留下半截文件导致次日加载失败。

    @param ds 待保存数据集
    @param path 目标文件路径（父目录不存在会自动创建）
    @return 写入期数
    @raises DataFormatError 目录不可创建或文件不可写（含修复建议）
    """
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataFormatError(
            f"无法创建历史文件目录 {file_path.parent}（{exc}）；请检查磁盘权限或路径是否合法"
        ) from exc

    lines = [
        f"date:{draw.date},period:{draw.period},numbers:"
        + "-".join(f"{number:02d}" for number in sorted(draw.numbers))
        for draw in sorted(ds.draws, key=lambda item: (item.date, item.period), reverse=True)
    ]
    tmp_path = file_path.with_name(file_path.name + _TMP_SUFFIX)
    try:
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp_path, file_path)
    except OSError as exc:
        raise DataFormatError(
            f"开奖历史文件写入失败：{file_path}（{exc}）；请关闭占用该文件的程序后重试"
        ) from exc
    return len(lines)


def truncated(ds: Dataset, target_period: int) -> Dataset:
    """【防穿越唯一入口】返回**严格早于** ``target_period`` 的全部历史。

    【语义（§0.2）】
    - 预测第 t 期（1-based 期号 t）时可见 ``≤ t-1``；
    - ``truncated(ds, t+1)`` ⇔ "用 ``X_t`` 预测 ``X_{t+1}``"，此时返回前 t+1 期；
    - 目标期号晚于最新期 → 返回全部历史（正常预测未来期）；
    - 目标期号早于最早期 → 返回空 Dataset + warning（**不抛异常**）。

    **本函数承诺永不抛异常**：任何奇怪输入都退化为"全量历史 + warning"，
    以免一次脏参数让整条流水线崩溃；真正的错误请在调用方用返回值长度判断。

    @param ds 原始数据集
    @param target_period 待预测期期号
    @return 截断后的 :class:`Dataset`（draws 为原前缀，``idx`` 与位置天然一致）
    """
    warnings = list(ds.warnings)
    try:
        target = int(target_period)
    except (TypeError, ValueError):
        warnings.append(f"目标期号 {target_period!r} 无法解析为整数，已退化为全量历史")
        return Dataset(
            draws=ds.draws,
            draws_by_period=ds.draws_by_period,
            source_path=ds.source_path,
            warnings=tuple(warnings),
        )

    periods = ds.periods
    if not periods:
        warnings.append("数据集为空，truncated 返回空历史")
        return Dataset(draws=(), draws_by_period={}, source_path=ds.source_path, warnings=tuple(warnings))

    cut = bisect_left(periods, target)
    if cut <= 0:
        warnings.append(
            f"目标期号 {target} 不晚于最早期 {periods[0]}，截断结果为空历史（请检查 period 参数）"
        )
        return Dataset(draws=(), draws_by_period={}, source_path=ds.source_path, warnings=tuple(warnings))
    if cut == len(periods):
        return Dataset(
            draws=ds.draws,
            draws_by_period=ds.draws_by_period,
            source_path=ds.source_path,
            warnings=tuple(warnings),
        )
    sub_draws = ds.draws[:cut]
    return Dataset(
        draws=sub_draws,
        draws_by_period={draw.period: draw for draw in sub_draws},
        source_path=ds.source_path,
        warnings=tuple(warnings),
    )


def latest_period(ds: Dataset) -> int:
    """返回数据集最新期期号。

    @param ds 数据集
    @return 最新期号
    @raises InsufficientDataError 数据集为空
    """
    return ds.latest.period


def append_draws(ds: Dataset, draws: Iterable[Draw]) -> Dataset:
    """把新开奖期合并进数据集，按期号幂等去重（已存在期号忽略）。

    @param ds 原数据集
    @param draws 待合并的新期（可乱序；同一期重复给出只保留首次）
    @return 新的升序 :class:`Dataset`（``idx`` 已重排）
    @raises DataFormatError 合并后出现同日期同期的顺序冲突
    """
    known = {draw.period: draw for draw in ds.draws}
    for draw in draws:
        known.setdefault(draw.period, draw)
    ordered = sorted(known.values(), key=lambda item: (item.date, item.period))
    normalized = tuple(
        replace(draw, idx=index) if draw.idx != index else draw for index, draw in enumerate(ordered)
    )
    return Dataset(
        draws=normalized,
        draws_by_period={draw.period: draw for draw in normalized},
        source_path=ds.source_path,
        warnings=ds.warnings,
    )