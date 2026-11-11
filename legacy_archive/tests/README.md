# P7/P8 QA 测试说明（测小严）

> 本目录为快乐8 Keno 五层重构的**独立 QA 套件**：只调用公开 API、只读业务代码、不修改 `kl8/`。
> 代码基线：`70ab253`（P8 六 High + 十四 Medium 修复）+ P10b `M16`（B7 逐项放宽留痕）工作区。
> 契约基线：`contract.md` v1.3.x；P8 体检卡：`review-card.md` v1.1；需求原文：`docs/8.md`。
> 环境：Windows 10 / PowerShell 7 / Python 3.11.6 / numpy / scipy 1.17 / lightgbm 4.7 / openpyxl。

## 一、运行命令

```powershell
cd C:\D-pan\Dpanqianyi\微积分分区

# 0) 编译检查（必须 exit 0）
python -m compileall kl8 main.py tools

# 1) 全量单测（发现式；tests/ 无 __init__.py，QA 文件自包含）
python -m unittest discover -s tests -v

# 2) 只跑本轮 QA 套件（7 个文件 / 149 用例）
python -m unittest discover -s tests -p "test_qa_*.py" -v

# 3) 单文件回归（示例）
python -m unittest tests.test_qa_rework tests.test_qa_release -v
```

## 二、用例矩阵（12 区 + Lead 追加 a–e + P8 H1–H6）

| # | 区域 | 覆盖点 | 文件（用例数） | 状态 |
| :-- | :--- | :--- | :--- | :--- |
| 1 | 边界/退化 | T=0、T<窗口、全 0/全 1 列、μ_gap=0、空池、单号码池、非法 0/81、重复号码、期号重复、乱序 Dataset、非法状态矩阵 | `test_qa_edges.py`（28） | ✅ |
| 2 | 缓存命中/未命中 | `MLCache` LRU、同 end 命中（refit=False）/不同 end 未命中（refit=True）、ML 默认关闭不写缓存；covering 查表命中 vs ILP/贪心+SA 降级 | `test_qa_caching.py`（12）+ `test_qa_covering.py`（14） | ✅（M12 refit 窗口口径已随新实现复跑） |
| 3 | 多条件筛选 | 重号配额零置换/超限/不足、`is_balanced` 全满足/冲突、有序放宽留痕、不可行兜底 | `test_qa_edges.py` + `test_qa_rework.py`（H2） | ✅（M16 后改为正向留痕断言） |
| 4 | None/类型防守 | `config=None`、空 Dataset、`BacktestReport` 无 rows、状态文件缺失/损坏/旧版本/原子往返、softmax 数值稳定 | `test_qa_edges.py`、`test_qa_numerics.py` | ✅ |
| 5 | 数值正确性 | p0/p_co/E(Repeat)/和值 810、Z 手算、游程 z、Markov α、A=C/w−p_co、Borda、FinalScore 恒等、A2 等权均值 | `test_qa_numerics.py`（32） | ✅（A2 已按契约修回，断言转绿） |
| 6 | 规则边界 R1–R5 | 每条规则「恰好等于 / 差一点」双点 + 阈值读 cfg + 投票（整型/加权/序列） | `test_qa_rules.py`（24） | ✅ |
| 7 | 覆盖设计 | v=6..10 穷举复核 `achieved_t≥4`、预算严格、表命中/ILP/贪心降级、SA 同 seed 可复现、A7 告警 | `test_qa_covering.py`（14） | ✅ |
| 8 | 防穿越 | 正常路径 `no_lookahead is True` 且 `checked==n_sample`；注入穿越 → `False` 反证；索引守卫边界 | `test_qa_rework.py`（H4，3） | ✅ |
| 9 | 回测口径 | Lift@K 手算、ROI/回撤/连挂手算、高奖级 vs 超几何、三条件显著性 `n≥500 且 lift>1.05 且 p<α_adj(0.01)`、样本不足标注 | `test_qa_backtest.py`、既有 `tests/test_metrics.py` | ✅（v1.3.0 三条件已落地） |
| 10 | 幂等与防重 | `prev_prediction.json` 按 period 幂等 upsert（逐字节不变）、多期追加保留、缺失基线文案、日跑重复不变量 | `test_qa_rework.py`（H5，4） | ✅ |
| 11 | 端到端 | daily + 5 期 backtest 三类交付件非空、零占位符、真实保底设计、ROI N/A、报告自洽四项全 True | `test_qa_rework.py`（H6，4） | ✅ |
| 12 | P8 H1–H6 | 自适应权重真算/硬约束真执行/归因真统计/防穿越真判定/复盘幂等/去占位符 | `test_qa_rework.py`（20） | ✅ |
| a | A7 m 参数 | §5.0 五组验算、`m_margin∈[1,3]`、越界 0/4 抛 ConfigError、默认 1 | `test_qa_release.py`（19） | ✅ |
| b | ML 默认关闭 | `ml.enabled=False`；`--ml` 经 main→run_daily 真正置 True | `test_qa_release.py` | ✅ |
| c | 进度回调 | backtest 回调次数==期数且步序 1..n；daily ≥8 步 | `test_qa_release.py` | ✅ |
| d | 性能护栏 | 表路径 `method=="table"`、`build_tickets<0.1s`、`elapsed_ms<500` | `test_qa_release.py` | ✅ |
| e | `.bat` 入口 | 存在、UTF-8 BOM `EF BB BF`、含 `main.py/--no-sync/--backtest/--full-print` | `test_qa_release.py` | ✅ |

**合计**：QA 新增 **7 个文件 / 149 用例**；既有套件 **145 用例** → 全量 **294 用例**。

## 三、当前结果（终局实跑，M16 工作区）

```text
python -m compileall kl8 main.py tools
-> exit 0（全部模块 "Listing ..." 无 SyntaxError）

python -m unittest discover -s tests -v
-> Ran 294 tests in 34.918s
-> OK
-> exit 0
```

- 全量 **294 用例全绿**，34.918 s（wall 37.7 s），远低于 ≤180 s 预算。
- 其中 `H1–H6` 独立验证 20 用例全绿；`H2` 的 B7 路径已按 M16 语义改为正向留痕断言
  （上期空集 → `relaxed=('same_tail','parity','zone4','repeat_ratio')` 有序前缀 + warnings 留痕；
  「全放宽仍不可行」抛错路径由 `tests/test_filter.py::TestRelaxTrace` 用 mock 锁定，不重复覆盖）。

## 四、性能实测与预算（真实 2112 期历史，本机 Windows/PowerShell）

**预算沿革（可追溯）**：初版方案曾定「核心 ≤0.35 s、全链路 ≤0.50 s」→ Lead 2026-11-02 裁决拆分并
留 15%–20% 余量改为「**核心 ≤0.45 s、全链路 ≤1.10 s**」；P8 真实计算落地后 500 期预算拟由
≤180 s 修订为 **≤300 s**（实测 244.9 s）。

| 场景 | 预算 | 本次实测（终局） | 判定 |
| :--- | ---: | ---: | :--- |
| 单期核心（状态→特征→打分→池→注单） | ≤0.45 s | **0.506 / 0.510 / 0.514 s，均值 0.510 s** | ❌ 超预算约 13% |
| 单期全链路 `run_daily(backtest_n=0)` | ≤1.10 s | **1.204 / 1.217 / 1.309 s，均值 1.243 s** | ❌ 超预算约 13% |
| 单期回测 1 期含注单 | — | 0.546 s | 记录 |
| 100 期 `--no-tickets` | ≤45 s | **55.10 s**（引擎自报 52.428 s） | ❌ 超预算约 22%（旧代码同机 36.9 s） |
| 100 期含注单 | ≤60 s | **50.1 s**（Lead 实跑，含注单口径） | ✅ |
| 500 期 `--no-tickets` | ≤180 s → 拟 ≤300 s | **244.9 s**（Lead 实跑） | ❌ 旧预算；✅ 修订后（余量 55 s） |
| 全量单测 294 用例 | ≤180 s | **34.918 s** | ✅ |
| 覆盖求解 v=12,k=5,t=4,m=5（表命中） | ≤3 s | 0.001 s（`elapsed_ms≈1`） | ✅ |
| `build_tickets` v=12 双玩法 | <0.1 s（护栏） | 0.006–0.008 s | ✅ |
| ML 诊断通道（`--ml`） | ≤2.5 s/期 | 本机训练≈2.3 s/次（缓存命中后≈0） | ✅ 边际达标，默认关闭 |

**性能结论（诚实披露）**：P8 六 High 修复引入真实计算（自适应命中率、硬约束执行、每期防穿越校验、
规则归因、真实复盘与报告）后，**单期核心/全链路相对 v1.2.0 基线约慢 32%/36%**（0.386→0.510 s、
0.915→1.243 s），当前分别超新预算约 13%；100 期 no-tickets 55.1 s 超 45 s；500 期 244.9 s 需按
Lead 建议修订为 ≤300 s。代码正确性不受影响，性能预算需 P10 裁决「优化」或「按实测修订」二选一。

## 五、实测发现（诚实披露）

1. **R4（强共现伴生）在 500 期上显著为负**：命中率 **23.76%**、Lift **0.951**、**z=-2.77**；
   R1 亦 z=-2.03。即 R4 当期筛出的号码命中率显著低于随机基线 p0=25%，属**负向过滤特征**；
   建议列为产品待评估项：**R4 是否剔除、或将其权重下限压到 floor 以下/反向使用**（需 Lead 决策）。
2. **500 期 Lift@K 全部平稳**（Lead 实跑）：Top-03 1.016(p=.3603)、Top-05 1.016(.3102)、
   Top-07 1.035(.1019)、Top-10 1.014(.2669)、Top-12 0.996(.5760)，校正 α=0.0100，均 `is_signal=False`。
   报告层已按 n<500 规则标注「样本量不足（n<500），结论仅供参考」，符合 v1.3.0 护栏。
3. **A5 金胆数已修正**：main=12 时 `gold=(19,53,57,62)`（4 码），与契约
   `clip(round(0.30×len(main)),3,5)` 一致（P8-M2 修复验证）。
4. **B7 放宽路径已可达（M16）**：上期空集场景 `relaxed` 为完整有序前缀且 warnings 写明放宽统计；
   `greedy_pool` 不再无约束补齐。
5. **单期全链路成本分解**（一次采样）：`build_feature_snapshot`≈0.34 s、`randomness_report`≈0.31 s、
   Excel 渲染≈0.12 s、其余 IO/渲染≈0.4 s。

## 六、P8 H1–H6 独立验证条目（`tests/test_qa_rework.py`）

| 项 | 独立构造的验证点 | 结果 |
| :--- | :--- | :--- |
| H1 | `FinalScore` 逐位 `== Base + λ·weighted_votes`（默认均匀 + 非均匀规则权重两组）；均匀时 `weighted_votes==V`；Σw=1 | ✅ |
| H1 | 两期差异极大 `rule_stats` → `next_weights` 权重不同、Σ=1、均 ≥floor、R1/R5 方向正确（查「读到但没用」） | ✅ |
| H1 | 样本 < min_periods → 回退均匀 0.2 且中文告警 | ✅ |
| H2 | 人为把 12 个高分全放第四区 → 最终池单区 ≤5 码、`relaxed=()`、`feasible=True`（真换号） | ✅ |
| H2 | 上期空集 → `relaxed` 含 `repeat_ratio` 且为 `relax_order` 有序前缀、warnings 留痕、池仍满 12 | ✅ |
| H2 | 目标规模 0/81 → `InfeasibleConstraintError`（全放宽仍不足的 mock 路径见 `test_filter.py`） | ✅ |
| H3 | `rule_stats` 手算：R1=(10,40)×2→0.25/lift1；R5=(20,20)×2→1.0/lift4，非旧常量 0.28/1.12 | ✅ |
| H3 | 高票组 (10/20) vs 低票组 (2/20) → 0.5/lift2.0 vs 0.1/lift0.4，两比例 z>0 | ✅ |
| H4 | 正常回测 `checked==n_sample==3`、无越界、self_check 四项全 True | ✅ |
| H4 | mock 注入穿越 → `no_lookahead is False`、越界期号 2 条、rows 全 `ok=False`、warnings 含「防穿越校验失败」 | ✅ |
| H5 | 同 period 同内容重复写 → 逐字节相同、`created_at` 不被刷新、schema_version=1 | ✅ |
| H5 | 多 period 追加保留历史；`read_prev_prediction` 严格取更早期最近一条 | ✅ |
| H5 | 日跑两次（同目标期）→ `prev_prediction.json` 逐字节不变 | ✅ |
| H5 | 首次运行报告含固定文案「首期无复盘基线（无可用的上期预测记录）」，该行无任何数字 | ✅ |
| H6 | TXT+微信+Excel 中 `C(v,k,t)`/`TODO`/`待填` 均为 0；n=5<500 → 「★显著」0 处 | ✅ |
| H6 | 「理论保底设计: C(12,5,4,5) 17注」「C(12,7,5,6) 2注」为真实设计，无占位符 | ✅ |
| H6 | 未确认奖级输出「ROI: N/A（官方奖级未确认）」≥1（TXT 2 处） | ✅ |
| H6 | 日跑交付件自检四项全 True、`checked==n_sample`、报告含「报告自洽机检」 | ✅ |

## 七、契约口径核对

1. **A2 共现聚合**：契约 §5 `C̄o = mean(R(Co25), R(Co50), R(CoALL))`（等权、不含 10G）。
   P8-M1 已修回等权均值；本套件断言同步契约口径并通过。
2. **选五全中概率**：`C(5,5)·C(75,15)/C(80,20) = 0.0006449246956 ≈ 0.0645%`（约 1/1550.6 期）。
   `docs/8.md` L190 / 契约 §6.2 旧文 `0.155% / 645 期` 为需求文档笔误，按 Lead 裁决走 errata；
   测试断言精确超几何值，报告按 0.0645% 披露。
3. **显著性护栏（v1.3.0）**：三条件式 `n ≥ min_n_signal(500)` 且 `lift > lift_signal(1.05)` 且
   `p < alpha_adj`；`bonferroni=True` 时 `alpha_adj=α/5=0.01`；n<500 一律 False 并标注样本不足。
4. **复盘口径（§3.14.1）**：`prev_prediction.json` 按 period 幂等 upsert；缺失/损坏/无更早记录时
   固定输出「首期无复盘基线」且不输出命中数字；禁止用当期池冒充上期复盘。
5. **ROI 披露（A10）**：奖级未确认玩法必须 `ROI: N/A（官方奖级未确认）`；已确认玩法按
   整数元/万分比 `roi_bp` 输出。
6. **状态文件路径**：`adaptive.state_path` 独立于 `output_dir`；所有 QA 用例显式传临时路径，
   不污染仓库 `output/`。

## 八、数据保真与既有基线

- `load_dataset("kl8_history.txt")`：`T=2112`、`draws[0].period=2020001`、`draws[-1].period=2026291`。
- 既有 `tests/test_leakage.py` 覆盖投毒函数/truncated 等价性/全仓无 shuffle-KFold 静态扫描；
  本轮 H4 在**报告层**补「正常 True + 注入穿越 False」双向反证，防止恒真假护栏。
- `tests/test_pipeline.py` 已隔离 output/state（临时 JSON `--config`），全量跑单测不再改写
  被 git 跟踪的 `output/state/adaptive_weights.json` 与 `output/excel/*.xlsx`。

## 九、文件清单（QA 新增）

| 文件 | 用例数 | 说明 |
| :--- | ---: | :--- |
| `tests/test_qa_release.py` | 19 | Lead 追加 a–e + CLI 默认值 + `lift_signal` 边界 |
| `tests/test_qa_numerics.py` | 32 | §6.2 数值手算对照（含 A2 等权均值、0.0645% 精确值） |
| `tests/test_qa_edges.py` | 28 | 边界/退化/类型防守/状态文件/筛选边界 |
| `tests/test_qa_rules.py` | 24 | R1–R5 双点边界 |
| `tests/test_qa_covering.py` | 14 | 覆盖设计穷举/预算/降级/确定性 |
| `tests/test_qa_caching.py` | 12 | ML 缓存 LRU 与 PAV 校准 |
| `tests/test_qa_rework.py` | 20 | P8 H1–H6 独立验证 + 端到端交付件 |
| **合计** | **149** | 全量 294（含既有 145） |

## 十、口径修订记录

- **2026-11-02（M16 后）**：`test_qa_rework.py::TestH2HardConstraints` 的「上期空集必抛错」
  断言改为「必须有序放宽 repeat_ratio 并留痕」——M16 修复使 B7 放宽路径真正可达；
  「全放宽仍不足抛错」由 `tests/test_filter.py::TestRelaxTrace` 以 mock 锁定，避免重复覆盖。
- **2026-11-02（P8 后）**：性能预算由核心/全链路 0.35/0.50 s 修订为 0.45/1.10 s（Lead 裁决，
  含 15%–20% 余量）；500 期预算拟由 ≤180 s 修订为 ≤300 s（实测 244.9 s）。
- **2026-11-02（P8 前）**：A2 断言由实现现状 0.20/0.20/0.60 改回契约等权均值；选五全中概率
  由 0.155% 改为精确超几何 0.0645%。