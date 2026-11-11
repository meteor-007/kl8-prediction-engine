# -*- coding: utf-8 -*-
"""快乐8五层重构 —— 统一异常体系（C2：零裸 except / 零 silent pass）。

【定位】
本模块是全系统唯一的异常定义处。任何模块都不允许抛内置的 ``RuntimeError`` /
``ValueError`` 作为"业务错误"，也不允许出现 ``except: pass`` 之类的静默吞异常；
业务异常一律继承 :class:`KL8Error`，消息必须是**中文 + 可操作修复建议**。

【异常归属契约（contract.md §3.1）】
====================================  ==========================================
异常                                  只允许由这些模块抛出
====================================  ==========================================
:class:`ConfigError`                   ``kl8/config.py``
:class:`DataFormatError`               ``kl8/history.py`` / ``kl8/datatypes.py``
:class:`InsufficientDataError`         ``kl8/layer1_representation`` / ``evaluation`` / ``layer5_combinatorial``
:class:`InfeasibleConstraintError`     ``kl8/layer4_scoring`` / ``kl8/layer5_combinatorial``
:class:`LeakageError`                  ``kl8/evaluation/leakage.py``
:class:`CoveringError`                 ``kl8/layer5_combinatorial``
====================================  ==========================================

@author czw
@since 2026-11-02
"""
from __future__ import annotations


class KL8Error(Exception):
    """快乐8五层系统异常基类。

    【用法】只用于 ``except KL8Error`` 统一兜底与分类，**不要直接抛出本类**；
    具体错误请抛对应的子类，便于上层给出精准的中文处置建议。
    """

    def __init__(self, message: str) -> None:
        """构造异常。

        @param message 中文错误消息，必须包含「发生了什么 + 怎么修」两部分信息。
        """
        super().__init__(message)
        self.message: str = message

    def __str__(self) -> str:
        """返回中文错误消息本体。

        @return 错误消息字符串
        """
        return self.message


class ConfigError(KL8Error):
    """配置非法：字段缺失、类型不符、取值越界或约束冲突（Σω、α、λ、gold<main、k≤池大小 等）。

    【谁能抛】仅 ``kl8/config.py``（含它调用的 ``validate_config``）。
    【怎么修】按消息中的「配置项 {路径} 非法：…；建议 …」逐项改正后重跑。

    @author czw
    @since 2026-11-02
    """


class DataFormatError(KL8Error):
    """历史数据 / 数据容器格式非法：行格式、日期、期号、号码集合、矩阵形状或取值不合规。

    【谁能抛】``kl8/history.py``（文件与行级解析）与 ``kl8/datatypes.py``（容器自校验）。
    【怎么修】按消息中的行号/字段名定位脏数据；开奖历史行格式应为
    ``date:YYYY-MM-DD,period:NNNNNNN,numbers:NN-NN-...``（20 个 1..80 不重复号码）。

    @author czw
    @since 2026-11-02
    """


class InsufficientDataError(KL8Error):
    """历史期数不足以支撑窗口计算 / 回测冷启动 / 覆盖设计求解。

    【谁能抛】``kl8/layer1_representation``、``kl8/evaluation``、``kl8/layer5_combinatorial``。
    【怎么修】补足开奖历史（或降低窗口 ``windows.short/mid/long``、回测 ``backtest.warmup``）。
    注意冷启动 ``w_eff=min(w,T)`` 本身**不抛**本异常，只有低于 ``windows.min_history`` 才抛。

    @author czw
    @since 2026-11-02
    """


class InfeasibleConstraintError(KL8Error):
    """约束不可行：候选池过小、硬约束全部放宽后仍凑不齐池、组合规模超出上限。

    【谁能抛】``kl8/layer4_scoring``（形态均衡过滤）、``kl8/layer5_combinatorial``（覆盖规模封顶）。
    【怎么修】按消息提示放大 ``pool.main_default`` 或预算 ``covering.budget``，或放宽对应硬约束。

    @author czw
    @since 2026-11-02
    """


class LeakageError(KL8Error):
    """防穿越（未来信息泄露）断言失败 —— 宁可中断，绝不静默继续。

    【谁能抛】仅 ``kl8/evaluation/leakage.py::assert_no_lookahead``。
    【怎么修】检查是否绕过了唯一截断入口 ``kl8/history.py::truncated`` 直接切片历史矩阵。

    @author czw
    @since 2026-11-02
    """


class CoveringError(KL8Error):
    """覆盖设计表损坏 / 覆盖参数非法（v、k、t、m 关系不成立或表内注单不自洽）。

    【谁能抛】``kl8/layer5_combinatorial``。
    【怎么修】重跑 ``python tools/build_covering_tables.py`` 幂等重建覆盖表，并核对
    ``C(v,k,t,m)`` 参数关系（``t ≤ k`` 且 ``t ≤ m ≤ v``）。

    @author czw
    @since 2026-11-02
    """