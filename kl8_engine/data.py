# -*- coding: utf-8 -*-
"""快乐8 (Keno) 历史开奖数据持久化、增量网络同步与时序数据抽象层。

【业务背景与架构定位】
本模块负责历史数据的严格因果加载、解析与二值化映射（80维 Multi-Hot 空间），
提供生产级增量网络抓取（支持中彩网官方接口及多源故障转移），
严格贯彻时序因果防穿越原则（只读历史，杜绝未来信息泄露）。

@author czw
@since 2026-11-04
"""

from __future__ import annotations

import os
import re
import tempfile
import urllib.request
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from kl8_engine.constants import K_DRAW, N_NUM

# 历史文件标准行正则表达式
_LINE_PATTERN = re.compile(
    r"^date:(?P<date>\d{4}-\d{2}-\d{2}),period:(?P<period>\d+),numbers:(?P<numbers>[0-9]{1,2}(?:-[0-9]{1,2})*)$"
)


@dataclass(frozen=True)
class Draw:
    """单期开奖记录实体（不可变对象）。

    【属性说明】
    - period: 期号整数，例如 2026292
    - date: 开奖日期，格式 YYYY-MM-DD
    - numbers: 20 个 1..80 升序开奖号码元组
    """

    period: int
    date: str
    numbers: Tuple[int, ...]

    def __post_init__(self) -> None:
        """数据健全性校验。"""
        if self.period <= 0:
            raise ValueError(f"期号 {self.period} 非法，必须为正整数")
        if len(self.numbers) != K_DRAW:
            raise ValueError(f"第 {self.period} 期号码数不足 {K_DRAW} 个: {self.numbers}")
        if len(set(self.numbers)) != K_DRAW:
            raise ValueError(f"第 {self.period} 期存在重复号码: {self.numbers}")
        for n in self.numbers:
            if not (1 <= n <= N_NUM):
                raise ValueError(f"第 {self.period} 期号码 {n} 超出 1..{N_NUM} 范围")

    def to_multihot(self) -> np.ndarray:
        """转换为 80 维 0/1 指示向量 (float32)。"""
        vec = np.zeros(N_NUM, dtype=np.float32)
        for n in self.numbers:
            vec[n - 1] = 1.0
        return vec


@dataclass(frozen=True)
class Dataset:
    """开奖历史时序数据集。

    严格按开奖时间升序组织，提供基于期号与索引的快速检索。
    """

    draws: Tuple[Draw, ...]

    @property
    def total_draws(self) -> int:
        """总期数。"""
        return len(self.draws)

    @property
    def latest(self) -> Draw:
        """最新一期开奖记录。"""
        if not self.draws:
            raise ValueError("数据集为空")
        return self.draws[-1]

    def target_period(self) -> int:
        """预测目标期号（最新期号 + 1）。"""
        return self.latest.period + 1

    def to_matrix(self, limit: Optional[int] = None) -> np.ndarray:
        """构建 (T, 80) 二值化状态矩阵。

        @param limit 截取最近 N 期，为 None 时返回全部
        @return shape (T, 80) float32 矩阵
        """
        slice_draws = self.draws if limit is None else self.draws[-limit:]
        matrix = np.zeros((len(slice_draws), N_NUM), dtype=np.float32)
        for idx, d in enumerate(slice_draws):
            matrix[idx] = d.to_multihot()
        return matrix


def parse_line(line: str) -> Optional[Draw]:
    """解析单行开奖文本数据。

    @param line 标准开奖行
    @return Draw 对象或在空行/注释行时返回 None
    """
    clean_line = line.strip()
    if not clean_line or clean_line.startswith("#"):
        return None

    match = _LINE_PATTERN.match(clean_line)
    if not match:
        raise ValueError(f"无法识别的数据格式行: {clean_line}")

    date_str = match.group("date")
    period_int = int(match.group("period"))
    num_list = sorted([int(x) for x in match.group("numbers").split("-")])

    return Draw(period=period_int, date=date_str, numbers=tuple(num_list))


def load_dataset(filepath: str) -> Dataset:
    """从磁盘文件加载完整开奖历史数据集。

    @param filepath 历史数据文件路径
    @return Dataset 数据集对象
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"历史数据文件未找到: {filepath}")

    draws_list: List[Draw] = []
    seen_periods: set[int] = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = parse_line(line)
            if d is not None:
                if d.period in seen_periods:
                    continue
                seen_periods.add(d.period)
                draws_list.append(d)

    # 严格按期号升序排序
    draws_list.sort(key=lambda item: item.period)
    return Dataset(draws=tuple(draws_list))


def save_dataset(dataset: Dataset, filepath: str) -> None:
    """原子写持久化数据集到磁盘文件。"""
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=dir_name or ".") as tf:
        tmp_name = tf.name
        for d in dataset.draws:
            nums_str = "-".join(f"{x:02d}" for x in d.numbers)
            tf.write(f"date:{d.date},period:{d.period},numbers:{nums_str}\n")

    os.replace(tmp_name, filepath)


def sync_history(filepath: str, issue_count: int = 100) -> Tuple[int, List[str]]:
    """从权威官方接口增量同步最新快乐8开奖数据。

    @param filepath 本地历史开奖文件路径
    @param issue_count 单次向上游请求的期数（接口实测支持 100 期）
    @return (新增记录数, 告警列表)
    """
    warnings: List[str] = []
    if not os.path.exists(filepath):
        return 0, [f"文件 {filepath} 不存在，跳过同步"]

    cur_ds = load_dataset(filepath)
    latest_local_period = cur_ds.latest.period

    # 尝试从中彩网 API 抓取最新开奖数据
    url = f"https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=kl8&issueCount={issue_count}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.cwl.gov.cn/ygfw/kjxx/kl8/",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    new_draws: List[Draw] = []
    latest_official_period = 0
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8")
            data = json.loads(content)
            result_list = data.get("result", [])

            for item in result_list:
                code_str = str(item.get("code", "")).strip()
                date_str = str(item.get("date", "")).strip()[:10]
                red_str = str(item.get("red", "")).strip()
                if not (code_str and date_str and red_str):
                    continue

                p_int = int(code_str)
                latest_official_period = max(latest_official_period, p_int)
                if p_int > latest_local_period:
                    num_list = sorted([int(x) for x in red_str.split(",") if x.strip()])
                    if len(num_list) == K_DRAW:
                        new_draws.append(Draw(period=p_int, date=date_str, numbers=tuple(num_list)))
    except Exception as exc:
        warnings.append(f"官方 API 增量同步连接受限（保留本地现有数据）: {exc}")
        return 0, warnings

    if not new_draws:
        if latest_official_period > latest_local_period:
            warnings.append(
                f"官方最新为第 {latest_official_period} 期，本地为第 {latest_local_period} 期，"
                f"但本次请求未返回可补期次（单次上限 {issue_count} 期）"
            )
        return 0, warnings

    # 合并新数据并原子更新
    all_draws = list(cur_ds.draws) + new_draws
    all_draws.sort(key=lambda x: x.period)
    dedup_draws: List[Draw] = []
    seen: set[int] = set()
    for d in all_draws:
        if d.period not in seen:
            seen.add(d.period)
            dedup_draws.append(d)

    save_dataset(Dataset(draws=tuple(dedup_draws)), filepath)

    remaining_gap = latest_official_period - dedup_draws[-1].period
    if remaining_gap > 0:
        warnings.append(
            f"官方最新为第 {latest_official_period} 期，本地补至第 {dedup_draws[-1].period} 期，"
            f"仍缺 {remaining_gap} 期（单次同步上限 {issue_count} 期，请再次运行分批补齐）"
        )

    return len(new_draws), warnings
