# -*- coding: utf-8 -*-
"""快乐8五层重构 —— 开奖数据增量同步（官网主源 + 一定牛备用源）。

【定位】
本模块把"远程拉号"与"本地落盘"封装成一个**绝不因网络问题崩溃**的动作：

- 网络/解析异常一律内部捕获，转 ``logging`` 告警 + :class:`SyncResult` 的
  ``warnings``（C2：零裸 except，但要分类记录并给中文提示）；
- 两个源都失败 → ``ok=False`` 且**不修改本地文件**（宁可少一期，不可写坏历史）；
- 只有本地文件不可写才抛 :class:`DataFormatError`（契约 §3.6）。

【数据源口径】
1. 主源：中国福彩官网 ``findDrawNotice`` 接口（分页 JSON）；
2. 备用源：一定牛年度历史页（HTML 表格），用于官网夜间延迟时补齐；
3. 合并策略：以**期号**为键，官网优先于备用源，已存在期号一律不覆盖（幂等）。

【落盘格式】
统一交给 :func:`kl8.history.save_dataset` —— 新期在前降序 + UTF-8 + 原子写。

@author czw
@since 2026-11-02
"""
from __future__ import annotations

import datetime
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

from kl8.constants import K_DRAW, N_NUM
from kl8.datatypes import Dataset, Draw
from kl8.errors import DataFormatError
from kl8.history import append_draws, load_dataset, save_dataset

logger = logging.getLogger(__name__)

CWL_URL_TEMPLATE: Final[str] = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    "?name=kl8&issueCount=&issueStart=&issueEnd=&dayStart=&dayEnd="
    "&pageNo={page}&pageSize={size}&week=&systemType=PC"
)
"""福彩官网快乐8开奖查询接口模板（page / size 占位）。"""

CWL_HEADERS: Final[Mapping[str, str]] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.cwl.gov.cn/",
}
"""官网接口请求头（缺失 Referer 会被风控拒绝）。"""

BACKUP_URL_TEMPLATE: Final[str] = "https://www.ydniu.com/open/kl8History-{year}.html"
"""一定牛年度历史页模板（year 占位）。"""

BACKUP_HEADERS: Final[Mapping[str, str]] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}
"""备用源请求头。"""

BACKUP_ROW_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<tr>\s*<td>(\d{7})</td>\s*<td>(\d{4}-\d{2}-\d{2})[^<]*</td>\s*<td[^>]*>(.*?)</td>",
    re.S | re.I,
)
"""一定牛表格行抽取正则：期号 / 日期 / 号码单元格。"""

PAGE_SIZE: Final[int] = 100
"""官网单页返回条数。"""

OFFICIAL_QUICK_PAGES: Final[int] = 2
"""快速增量模式只翻前 2 页（覆盖最近约 200 期）。"""

OFFICIAL_FULL_PAGES: Final[int] = 50
"""全量校准模式最多翻 50 页。"""


@dataclass(frozen=True)
class SyncResult:
    """一次同步动作的结果汇总（供 main/pipeline 展示，不抛异常）。"""

    ok: bool
    """是否成功拿到远程数据并完成落盘（False 表示已降级为"用本地历史"）。"""

    added: int
    """本次新增期数。"""

    total: int
    """同步后本地总期数。"""

    latest_period: int | None
    """同步后本地最新期号；本地为空时为 None。"""

    warnings: tuple[str, ...]
    """同步过程中的中文告警（网络失败、无新增、本地文件损坏等）。"""


def _http_get(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
    """发起一次带超时的 HTTP GET（仅本模块内部使用）。

    @param url 目标地址
    @param headers 请求头
    @param timeout 超时秒数
    @return 响应体字节
    @raises urllib.error.URLError 网络层错误（由调用方分类捕获）
    @raises TimeoutError 超时
    """
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 固定官方域名
        return response.read()


def _normalize_numbers(raw: object) -> tuple[int, ...] | None:
    """把远程号码片段归一为 20 个不重复、1..80 的升序元组。

    @param raw 远程值（逗号分隔字符串或数字列表）
    @return 合法则返回升序元组，否则 None（调用方跳过该条并告警）
    """
    if isinstance(raw, str):
        parts = [item for item in raw.replace(",", "-").split("-") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        parts = [str(item) for item in raw]
    else:
        return None
    try:
        numbers = sorted(int(item) for item in parts)
    except (TypeError, ValueError):
        return None
    if len(numbers) != K_DRAW or len(set(numbers)) != K_DRAW:
        return None
    if not all(1 <= value <= N_NUM for value in numbers):
        return None
    return tuple(numbers)


def fetch_official(max_pages: int = OFFICIAL_FULL_PAGES, timeout: float = 30.0) -> dict[int, tuple[str, tuple[int, ...]]]:
    """从福彩官网分页拉取快乐8开奖历史。

    @param max_pages 最多翻页数（快速增量传 2，全量传 50）
    @param timeout 单次请求超时秒数
    @return ``期号 → (日期, 升序号码元组)`` 的映射；全部失败时返回空字典
    """
    official: dict[int, tuple[str, tuple[int, ...]]] = {}
    for page in range(1, max(1, int(max_pages)) + 1):
        url = CWL_URL_TEMPLATE.format(page=page, size=PAGE_SIZE)
        try:
            payload = json.loads(_http_get(url, CWL_HEADERS, timeout).decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("福彩官网第 %d 页网络异常，停止翻页：%s", page, exc)
            break
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("福彩官网第 %d 页返回非 JSON，停止翻页：%s", page, exc)
            break
        if not isinstance(payload, dict):
            logger.warning("福彩官网第 %d 页返回结构异常（非对象），停止翻页", page)
            break
        rows = payload.get("result")
        if not isinstance(rows, list) or not rows:
            break
        for item in rows:
            if not isinstance(item, dict):
                continue
            code = item.get("code")
            date_text = item.get("date")
            numbers = _normalize_numbers(item.get("red"))
            if code is None or date_text is None or numbers is None:
                continue
            try:
                period = int(str(code))
            except (TypeError, ValueError):
                logger.warning("福彩官网返回期号无法解析：%r，已跳过", code)
                continue
            official[period] = (str(date_text).split("(")[0].strip(), numbers)
        if len(rows) < PAGE_SIZE:
            break
    return official


def fetch_backup(year: int | None = None, timeout: float = 20.0) -> dict[int, tuple[str, tuple[int, ...]]]:
    """从一定牛备用源拉取某一年度快乐8开奖历史。

    @param year 目标年份，None 表示当前年份
    @param timeout 请求超时秒数
    @return ``期号 → (日期, 升序号码元组)`` 的映射；失败时返回空字典
    """
    target_year = year if year is not None else datetime.date.today().year
    url = BACKUP_URL_TEMPLATE.format(year=target_year)
    try:
        html = _http_get(url, BACKUP_HEADERS, timeout).decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("一定牛备用源网络异常（年份 %d）：%s", target_year, exc)
        return {}
    except UnicodeDecodeError as exc:  # pragma: no cover - replace 策略下几乎不可能触发
        logger.warning("一定牛备用源编码异常（年份 %d）：%s", target_year, exc)
        return {}

    backup: dict[int, tuple[str, tuple[int, ...]]] = {}
    for match in BACKUP_ROW_PATTERN.finditer(html):
        period_text, date_text, cell = match.group(1), match.group(2), match.group(3)
        numbers = _normalize_numbers(re.findall(r"\d{1,2}", cell))
        if numbers is None:
            continue
        try:
            period = int(period_text)
        except ValueError:
            continue
        backup[period] = (date_text, numbers)
    return backup


def sync_history(
    path: str | Path = "kl8_history.txt",
    *,
    quick: bool = True,
    timeout: float = 30.0,
) -> SyncResult:
    """执行一次增量同步：拉远程 → 与本地合并 → 降序原子写回。

    【安全优先】
    - 本地文件损坏：**中止同步**并告警，绝不用远程数据覆盖可能只是格式问题的本地全量历史；
    - 远程全失败：返回 ``ok=False`` 且不改动本地文件；
    - 写入失败：抛 :class:`DataFormatError`（唯一允许外抛的情形）。

    @param path 本地历史文件路径
    @param quick True=快速增量（官网只翻 2 页）；False=全量校准（最多 50 页）
    @param timeout 单次网络请求超时秒数
    @return :class:`SyncResult`
    @raises DataFormatError 本地历史文件不可写/目录不可创建
    """
    file_path = Path(path)
    warnings: list[str] = []

    if file_path.exists():
        try:
            local_ds = load_dataset(file_path)
        except DataFormatError as exc:
            return SyncResult(
                ok=False,
                added=0,
                total=0,
                latest_period=None,
                warnings=(
                    f"本地历史文件损坏，已中止同步以免覆盖数据：{exc.message}；"
                    "请先人工修复或备份后删除该文件再重试",
                ),
            )
    else:
        local_ds = Dataset(draws=(), draws_by_period={}, source_path=str(file_path))
        warnings.append(f"本地历史文件不存在，将新建：{file_path}")

    official = fetch_official(max_pages=OFFICIAL_QUICK_PAGES if quick else OFFICIAL_FULL_PAGES, timeout=timeout)
    backup = fetch_backup(timeout=timeout)
    if not official and not backup:
        warnings.append("远程数据源均未返回数据（官网与备用源均失败），已跳过同步并保留本地历史")
        return SyncResult(
            ok=False,
            added=0,
            total=local_ds.T,
            latest_period=local_ds.latest.period if local_ds.T else None,
            warnings=tuple(warnings),
        )

    merged: dict[int, tuple[str, tuple[int, ...]]] = dict(backup)
    merged.update(official)  # 官网数据优先覆盖备用源同名期

    existing = set(local_ds.periods)
    new_draws: list[Draw] = []
    for period in sorted(merged):
        if period in existing:
            continue
        date_text, numbers = merged[period]
        try:
            new_draws.append(Draw(date=date_text, period=period, numbers=frozenset(numbers)))
        except DataFormatError as exc:
            warnings.append(f"期号 {period} 远程数据非法，已跳过：{exc.message}")
    if not new_draws:
        warnings.append("远程数据无新增期号，本地历史保持不变")
        return SyncResult(
            ok=True,
            added=0,
            total=local_ds.T,
            latest_period=local_ds.latest.period if local_ds.T else None,
            warnings=tuple(warnings),
        )

    combined = append_draws(local_ds, new_draws)
    written = save_dataset(combined, file_path)
    if written != combined.T:
        warnings.append(f"写回期数 {written} 与内存期数 {combined.T} 不一致，请检查文件完整性")
    return SyncResult(
        ok=True,
        added=len(new_draws),
        total=combined.T,
        latest_period=combined.latest.period,
        warnings=tuple(warnings),
    )