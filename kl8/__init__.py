# -*- coding: utf-8 -*-
"""快乐8（Keno）五层重构包 kl8。

【分层结构（contract.md §2）】
======================  ============================================  ==============
包/模块                  职责                                          允许依赖
======================  ============================================  ==============
``constants``           数学恒等式与号码静态属性                      —
``errors``              统一异常体系                                  —
``datatypes``           跨层不可变数据结构                            constants/errors
``config``              业务参数与严格校验                            constants/errors
``history``             历史解析/校验/保存/``truncated``（唯一截断）   datatypes/constants
``sync``                远程增量同步                                  history/datatypes
``layer1_representation``  二值状态矩阵 + 四尺度窗口                  ≤ config
``layer2_baseline``     随机性基准与偏态探测                          ≤ layer1
``layer3_features``     四大特征族 + 快照装配                         ≤ layer2
``layer4_scoring``      Borda 秩 / 五规则 / 混合打分 / 形态过滤       ≤ layer3
``layer5_combinatorial``  覆盖设计 C(v,k,t,m) + 注单后处理            ≤ layer4
``evaluation``          Walk-Forward 回测 / 指标 / 防穿越审计         任意层（只读）
``reporting``           TXT / 微信 / Excel 三类交付件                 任意层（只读）
``pipeline``            唯一编排点 run_daily / run_backtest / run_audit 任意层
======================  ============================================  ==============

【包级约定】
- 本文件**只导出 ``__version__``**：禁止在这里 import 子模块或做任何重计算，
  否则 `import kl8` 会把整条流水线拖起来（也会制造循环依赖）。
- ``layerN`` 只能 import ``layerM (M<N)``，由 ``tests/test_architecture.py`` 用 AST 扫描断言。

@author czw
@since 2026-11-02
"""

__version__ = "1.0.0"
"""kl8 五层重构包版本号（与 contract.md v1.0.0 对齐）。"""