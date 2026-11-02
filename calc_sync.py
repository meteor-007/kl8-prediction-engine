# -*- coding: utf-8 -*-
"""微积分八区分区预测系统 —— 官方开奖历史自动同步模块

【业务背景与架构定位】:
支持从中国福彩官网官方开放接口实时拉取最新快乐8开奖数据，
并在官网延迟时代用「一定牛」备用源进行补齐，保证本地开奖数据 100% 准时最新。

@author czw
@since 2026-10-18
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import urllib.request
from pathlib import Path

# 控制台编码防乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from calc_core import DEFAULT_HISTORY_FILE

CWL_API = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    "?name=kl8&issueCount=&issueStart=&issueEnd=&dayStart=&dayEnd="
    "&pageNo={page}&pageSize={size}&week=&systemType=PC"
)


def _fetch_cwl_page(page: int, size: int = 100) -> dict:
    url = CWL_API.format(page=page, size=size)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://www.cwl.gov.cn/",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_official(max_pages: int = 50) -> dict[int, tuple[str, tuple[int, ...]]]:
    """返回 period -> (date, sorted_numbers)。"""
    official: dict[int, tuple[str, tuple[int, ...]]] = {}
    for page in range(1, max_pages + 1):
        try:
            data = _fetch_cwl_page(page, 100)
        except Exception as e:
            print(f"  [提示] 福彩官网拉取第 {page} 页网络异常: {e}")
            break
        rows = data.get("result") or []
        if not rows:
            break
        for item in rows:
            if not isinstance(item, dict):
                continue
            code_str = item.get("code")
            date_str = item.get("date")
            red_str = item.get("red")
            if not code_str or not date_str or not red_str:
                continue
            try:
                period = int(code_str)
                date = str(date_str).split("(")[0].strip()
                nums = tuple(sorted(int(x) for x in str(red_str).split(",") if x.strip()))
            except (ValueError, TypeError):
                continue
            if len(nums) != 20:
                continue
            official[period] = (date, nums)
        print(f"  拉取福彩官网第 {page} 页: +{len(rows)} 条，累计 {len(official)} 条")
        if len(rows) < 100:
            break
    return official


def fetch_ydniu_recent(year: int | None = None) -> dict[int, tuple[str, tuple[int, ...]]]:
    """备用源：一定牛开奖历史（防止官网接口夜间开奖延迟）。"""
    y = year or datetime.date.today().year
    url = f"https://www.ydniu.com/open/kl8History-{y}.html"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  [提示] 一定牛备用源拉取异常: {e}")
        return {}

    out: dict[int, tuple[str, tuple[int, ...]]] = {}
    pat = re.compile(
        r"<tr>\s*<td>(\d{7})</td>\s*<td>(\d{4}-\d{2}-\d{2})[^<]*</td>\s*<td[^>]*>(.*?)</td>",
        re.S | re.I,
    )
    for m in pat.finditer(html):
        period = int(m.group(1))
        date = m.group(2)
        nums = tuple(sorted(int(x) for x in re.findall(r"\d{1,2}", m.group(3))))
        if len(nums) != 20 or len(set(nums)) != 20:
            continue
        if not all(1 <= n <= 80 for n in nums):
            continue
        out[period] = (date, nums)
    return out


def load_local_history(path: Path) -> dict[int, tuple[str, tuple[int, ...], str]]:
    """读取本地历史文件 period -> (date, sorted_nums, raw_line)。"""
    local = {}
    pat = re.compile(r"date:([^,]+),period:(\d+),numbers:(.+)")
    if not path.exists():
        return local
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pat.match(line)
        if not m:
            continue
        date, period_s, nums_s = m.groups()
        period = int(period_s)
        nums = tuple(sorted(int(x) for x in nums_s.replace(",", "-").split("-") if x.strip()))
        local[period] = (date, nums, line)
    return local


def sync_history(history_path: str = DEFAULT_HISTORY_FILE, quick: bool = True) -> int:
    """主同步执行逻辑。"""
    path = Path(history_path)
    print("=" * 64)
    print("  快乐8 · 开奖数据同步 " + ("(快速增量)" if quick else "(全量校准)"))
    print("=" * 64)

    # 步骤 1: 拉取官网数据
    print("[1] 检查福彩官方 API...")
    official = fetch_official(max_pages=2 if quick else 50)

    # 步骤 2: 检查备用源
    print("[2] 检查一定牛备用源...")
    yd = fetch_ydniu_recent()
    latest_cwl = max(official) if official else 0
    latest_yd = max(yd) if yd else 0
    if yd:
        print(f"    备用源最新期: {latest_yd} ({yd[latest_yd][0]})")
        for p, (d, n) in yd.items():
            if p not in official:
                official[p] = (d, n)

    if not official:
        print("[警告] 远程数据源均未返回数据，跳过同步，使用本地历史。")
        return 0

    # 步骤 3: 与本地合并
    local = load_local_history(path)
    latest_loc = max(local) if local else 0
    latest_remote = max(official) if official else 0

    print(f"[3] 本地最新期: {latest_loc} ｜ 远程最新期: {latest_remote}")
    new_periods = [p for p in sorted(official.keys(), reverse=True) if p not in local]

    # 合并全量字典并重写文件（降序：新期在上，保证查看直观）
    merged = dict(local)
    for p, (d, n) in official.items():
        if p not in merged:
            line = f"date:{d},period:{p},numbers:" + "-".join(f"{x:02d}" for x in n)
            merged[p] = (d, n, line)

    all_sorted_periods = sorted(merged.keys(), reverse=True)
    lines = [merged[p][2] for p in all_sorted_periods]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[4] 同步完成: 本地总计 {len(lines)} 期，本次新增 {len(new_periods)} 期")
    if new_periods:
        top_p = all_sorted_periods[0]
        d, n, _ = merged[top_p]
        print(f"    最新期号: {top_p} ({d})")
        print("    开奖号码: " + " ".join(f"{x:02d}" for x in n))
    print("=" * 64)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="同步快乐8开奖数据")
    ap.add_argument("--history", default=DEFAULT_HISTORY_FILE, help="开奖历史文件路径")
    ap.add_argument("--full", action="store_true", help="全量同步")
    ap.add_argument("--quick", action="store_true", help="快速增量同步 (默认)")
    args = ap.parse_args()
    sys.exit(sync_history(args.history, quick=not args.full))
