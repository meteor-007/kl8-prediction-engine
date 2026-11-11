# KL8 快乐8 预测与分析系统

基于 **Transformer 序列建模 + 贝叶斯图谱统计** 的快乐8（Keno 型，80 选 20）数据分析与组合优化系统。

> **免责声明**：快乐8 开奖是独立随机事件。本项目用于数据分析、统计检验与工程实践，**不构成任何投注建议**，也不承诺预测收益。请理性购彩。

---

## 特性

- **双引擎预测**
  - Transformer 多头自注意力网络（80 维 Multi-Hot 序列）
  - 统计图谱侧：Dirichlet 贝叶斯后验、80×80 共现谱中心性、10G/40G 动量差分
- **双融合并排输出**
  - 默认正式版：四分量等权 **Borda 秩融合**，避免 sigmoid 概率与 min-max 谱分数的量纲失衡
  - 实验版：原始加权概率和（`--fusion weighted` 可反转正式/实验版）
  - 两版金胆/主选/30码大底/缩水指标并排展示
- **每日复盘闭环**
  - 推荐日志只记录每个目标期号的**首份推荐**，防止开奖后重跑污染复盘
  - 自动核算金胆/主选/Top20/30码大底命中、票面回报、Lift
  - **排序质量 AUC** 监控（0.5 = 随机）
- **用户点位分析**
  - 支持 `compare`（仅对比标注）与 `boost`（参与缩水覆盖加权）
  - 输出点位逐号排名、共识码、逆势码、与上期重号
- **30 码级联缩水**
  - C(30,5)=142,506 注全展开 → 形态剪枝 → 位运算正交防撞 → 20 注精选
  - 蒙特卡洛 5000 次审计：保 5/保 4/保 3 概率与期望奖金
- **全渠道交付**：控制台/TXT 精要报告、微信文案、商业 Excel、缩水专报、集群三元组

## 快速开始

环境：Python 3.11

```bash
pip install numpy torch openpyxl pytest
```

每日预测（默认正式版 Borda，实验版 weighted，并排输出）：

```bash
python main.py --full-print --points "02 06 10 12 16 24 26 32 36 40 42 44 52 56 64 70 72 74 76 80"
```

常用选项：

```bash
# 跳过联网同步，使用本地历史
python main.py --no-sync --full-print

# 把 weighted 设为正式版，Borda 作为实验版
python main.py --full-print --fusion weighted

# 关闭双融合并排对比
python main.py --no-compare --full-print

# 调整缩水注数 / Transformer 轮数 / 窗口
python main.py --full-print --budget 20 --epochs 25 --seq-len 10
```

Windows 下也可直接双击 `运行微积分预测.bat`。

## 数据源

每日只需维护项目根目录的 `kl8_history.txt`（也可由程序自动增量同步）：

```text
date:2026-11-10,period:2026300,numbers:08-12-17-23-28-30-31-33-35-39-43-45-47-51-52-58-63-66-69-75
```

- 一行一期，UTF-8
- 20 个号码两位补零、升序、`-` 分隔
- 程序默认从中彩网官方接口增量抓取（单次最多 100 期），失败时保留本地数据并告警

## 目录结构

```text
kl8_engine/
  data.py                 # 历史解析、增量同步、状态矩阵
  models/
    transformer_net.py    # Transformer 自注意力网络
    spectral_bayesian.py  # Dirichlet / 共现谱 / 动量差分
    ensemble.py           # 单/双融合、Borda 秩融合、核心梯队
  points.py               # 用户点位解析与一致性分析
  review.py               # 复盘、推荐日志、命中率/AUC 统计
  shrinkage.py            # 30 码级联缩水与蒙特卡洛审计
  reporting.py            # TXT/微信/Excel 渲染
  pipeline.py             # 端到端流水线
main.py                   # CLI 入口
scripts/
  backtest_evaluation.py  # Walk-Forward 回测
  ablation_gold.py        # 金胆池多选择器消融
  analyze_ablation.py     # 粘滞性/配对检验/分半稳定性
  analyze_cross_window.py # 跨窗口合并与全排序验证
tests/                    # 40 个单元测试
legacy_archive/           # 旧五层架构归档（不参与运行）
kl8_history.txt           # 历史开奖数据
每日预测指令.txt           # 每日执行指令书
```

## 输出

| 路径 | 内容 |
| :--- | :--- |
| `output/reports/` | TXT 精要报告（复盘、点位、Top20、分模块、命中率、双融合对比、消融报告） |
| `output/wechat/` | 微信发布文案 + 30 码缩水专发 |
| `output/excel/` | 商业 Excel（决策推荐、80码矩阵、缩水票、复盘、点位、并排对比） |
| `output/shrinkage/` | 30 码缩水专报（20 注实战票） |
| `output/state/prediction_journal.jsonl` | 推荐日志（每期首份推荐 + 80 码全序） |

## 回测与消融

```bash
# 最近 30 期 Walk-Forward 回测
python scripts/backtest_evaluation.py --periods 30 --epochs 10

# 金胆池消融（60 期；可加 --offset 60/120 换窗口）
python scripts/ablation_gold.py --periods 60 --epochs 25

# 诊断与跨窗口汇总
python scripts/analyze_ablation.py
python scripts/analyze_cross_window.py
```

**已有结论（2026-11-10）**：3 个 60 期窗口、17 种选择器。Borda 金胆均值 1.206，生产加权融合 0.922，随机基线 1.017；Borda 修复了谱分量尺度绑架与金胆粘滞问题，但对随机基线仍未达统计显著（p≈0.058）。所有分量的排序 AUC 约 0.49，未发现稳定预测信号。详见 [金胆池消融与优化整改报告](output/reports/金胆池消融与优化整改_20261110.md)。

## 测试

```bash
python -m pytest tests -q
# 40 passed
```

## 设计说明

- **为什么默认 Borda**：`0.55×DL + 0.45×谱分数` 中两者量纲不可比，谱分量会主导 Top4；Borda 先转秩再融合，实际 DL 权重不再名不副实，且三个窗口金胆表现均优于原融合。
- **为什么复盘优先读日志**：推荐日志只保存每个目标期号的首份推荐，避免开奖后重跑或改参污染历史。
- **为什么看 AUC**：4 码金胆的命中方差极大（SD≈0.85），几十期命中率不足以判断模型优劣；80 码排序 AUC 更稳定，0.5 表示与随机无异。
- **彩票本质**：公平开奖下任意 4 码的期望命中恒为 1.0。任何“提升命中率”的调参都必须先通过与同期随机基线的配对检验与多重比较校正。

## License

[MIT](LICENSE)