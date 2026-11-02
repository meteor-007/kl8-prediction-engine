# P7 QA 测试说明（测小严）

> 本目录为快乐8 Keno 五层重构的**独立 QA 套件**：只调用公开 API、只读业务代码、不修改 `kl8/`。
> 代码基线：`251e795`（五层实现）+ `f170147`（清理与性能修正）+ `3281992`（工作区口径）。
> 契约基线：`contract.md` v1.2.0；P8 体检卡：`review-card.md` v1.1（阻断，6 High / 15 Medium）。
> 环境：Windows 10 / PowerShell 7 / Python 3.11.6 / numpy / scipy 1.17 / lightgbm 4.7。

## 一、运行命令

```powershell
cd C:\D-pan\Dpanqianyi\微积分分区

# 0) 编译检查（必须 exit 0）
python -m compileall kl8 main.py tools

# 1) 全量单测（发现式；tests/ 无 __init__.py，新增文件保持自包含）
python -m unittest discover -s tests -v

# 2) 只跑本轮新增 QA 套件（6 个文件 / 129 用例）
python -m unittest discover -s tests -p "test_qa_*.py" -v

# 3) 单文件回归（示例）
python -m unittest tests.test_qa_release tests.test_qa_edges -v
```

## 二、用例矩阵（12 区 + Lead 追加 a–e）

| # | 区域 | 覆盖点 | 文件（用例数） | 状态 |
| :-- | :--- | :--- | :--- | :--- |
| 1 | 边界/退化 | T=0、T<窗口、全 0/全 1 列、μ_gap=0、空池、单号码池、非法 0/81、重复号码、期号重复、乱序 Dataset、非法状态矩阵 | `test_qa_edges.py`（28） | ✅ 本批完成 |
| 2 | 缓存命中/未命中 | `MLCache` LRU、同 end 命中（refit=False）/不同 end 未命中（refit=True）、ML 默认关闭不写缓存；covering 查表命中 vs ILP/贪心+SA 降级 | `test_qa_caching.py`（12）+ `test_qa_covering.py`（14） | ✅ 本批完成（M12 改 refit 窗口口径后按 §五-8 重写） |
| 3 | 多条件筛选 | 重号配额零置换/超限/不足、`is_balanced` 全满足/冲突、放宽链留痕、不可行输入兜底不崩 | `test_qa_edges.py::TestFilterBoundaries`（4） | ✅ 本批完成；H2 真执行版待新代码 |
| 4 | None/类型防守 | `config=None`、空 Dataset、`BacktestReport` 无 rows、状态文件缺失/损坏/旧版本/原子往返、softmax 数值稳定 | `test_qa_edges.py` + `test_qa_numerics.py` | ✅ 本批完成 |
| 5 | 数值正确性 | p0/p_co/E(Repeat)/和值 810、Z 手算、游程 z、Markov α、A=C/w−p_co、Borda、FinalScore 恒等、A2 等权均值 | `test_qa_numerics.py`（32） | ✅ 本批完成（A2 断言已按 Lead 裁决改回契约口径，待 M1 实现修复后转绿） |
| 6 | 规则边界 R1–R5 | 每条规则「恰好等于 / 差一点」双点 + 阈值读 cfg + 投票（整型/加权/序列） | `test_qa_rules.py`（24） | ✅ 本批完成 |
| 7 | 覆盖设计 | v=6..10 穷举复核 `achieved_t≥4`、预算严格、表命中/ILP/贪心降级、SA 同 seed 可复现、A7 告警 | `test_qa_covering.py`（14） | ✅ 本批完成 |
| 8 | 防穿越 | 未来投毒后 snapshot/final/pool/注单逐位不变、全仓无 shuffle/KFold、`max_accessed_t` 守卫、self_check 反证 | `test_qa_leakage.py` | ⏸ 等 H4 修复后写（当前 `no_lookahead` 恒真，写必红） |
| 9 | 回测口径 | Lift@K 手算、ROI/回撤/连挂手算、高奖级 vs 超几何、显著性三条件 `n≥500 且 lift>1.05 且 p<α_adj(0.01)` + 样本不足标注 | `test_qa_backtest.py` | ⏸ 等契约 v1.3 实现落地 |
| 10 | 幂等与防重 | 状态文件按 period upsert、重复日跑不重复累计、交付件覆盖写（原子） | `test_qa_idempotency.py` | ⏸ 当前重复日跑 `n_updates` 累加（已报 Lead）；H5 新增 `prev_prediction.json` 一并覆盖 |
| 11 | 端到端 | 400 期临时目录 daily + backtest、三类交付件非空、关键段落标记、报告数字与引擎自洽、复盘口径 | `test_qa_e2e.py` | ⏸ 等 H5/H6/M13 报告修复 |
| a | A7 m 参数 | §5.0 五组验算、`m_margin∈[1,3]`、越界 0/4 抛 ConfigError、默认 1 | `test_qa_release.py`（19） | ✅ 本批完成 |
| b | ML 默认关闭 | `ml.enabled=False`、`--ml` 经 main→run_daily 真正置 True | `test_qa_release.py` | ✅ 本批完成 |
| c | 进度回调 | backtest 回调次数==期数且步序 1..n；daily ≥8 步且覆盖 `[1/8]..[8/8]` | `test_qa_release.py` | ✅ 本批完成 |
| d | 性能护栏 | 覆盖表路径 `method=="table"`、`build_tickets<0.1s`、`CoveringReport.elapsed_ms<500` | `test_qa_release.py` | ✅ 本批完成 |
| e | `.bat` 入口 | 存在、UTF-8 BOM `EF BB BF`、含 `main.py/--no-sync/--backtest/--full-print` | `test_qa_release.py` | ✅ 本批完成 |

**合计**：新增 6 个文件 / **129 用例**；既有套件 124 用例 → 全量 **253 用例**。

## 三、当前结果（基线 `3281992` 工作区，2026-11-02 18:2x 实跑）

```text
python -m compileall kl8 main.py tools          -> exit 0
python -m unittest discover -s tests -v         -> Ran 253 tests in 16.223s / FAILED (failures=2)
```

- 全量 253 用例，16.223s，满足 ≤180s 预算。
- 2 个失败均已有明确归属，**不是本轮测试缺陷**：
  1. `test_qa_numerics.py::TestHybridIdentities::test_co_aggregate_formula`：M1/A2，
     断言已按 Lead 裁决改为契约等权均值 `mean(R(Co25),R(Co50),R(CoALL))`；当前实现仍为
     0.20/0.20/0.60，等码小全按契约改回后自动转绿。
  2. `test_qa_caching.py::TestTrainPredictCaching::test_feature_snapshot_ml_channel`：
     本文件自身口径错误（误以为 `include_ml=True` 可越过 `cfg.ml.enabled=False`），
     已修正为「include_ml 不越权 + 显式开启后才训练」，单文件复跑 **12 tests OK**。
- **重要状态说明**：18:2x 起码小全正在改 `kl8/`（H1–H6 + Lead 指定 Medium），期间工作区出现
  中间态（例：`kl8/layer4_scoring/__init__.py` 调用 `score_all(..., votes=...)` 而
  `hybrid.score_all` 尚未同步 → `TypeError`）。本轮 253/16.2s 结论取自该改动开始前的
  自洽基线；新代码下发后按 Lead 要求**全量复跑并补 H1–H6 用例**。

## 四、性能实测（真实 2112 期历史，Windows/PowerShell，基线 `3281992`）

**预算口径（Lead 2026-11-02 正式裁决，替代初值）**：
- 初版方案曾定「单期核心 ≤0.35 s、单期全链路 ≤0.50 s」；实测后按「核心 / 全链路」拆分并各留
  15%–20% 余量，正式预算改为：
  - **核心决策链路**（状态矩阵 → 特征快照 → 打分 → 池 → 注单）：**≤0.45 s**；
  - **全链路**（核心 + 随机性诊断 + 三类交付件渲染 + IO）：**≤1.10 s**。
- 为可追溯，下表同时保留旧预算、新预算与实测值（旧 0.35/0.50 → 实测 0.386/0.94 → 新 0.45/1.10）。

| 场景 | 旧预算 | 新预算 | 本次实测 | 结论 |
| :--- | ---: | ---: | ---: | :--- |
| 单期核心（特征→打分→池） | ≤0.35 s | **≤0.45 s** | 0.343 / 0.371 / 0.408 s（均值 **0.386 s**） | ✅ 新预算达标（旧预算超约 10%） |
| 单期核心 + 注单 | — | 含在上行 | +0.006–0.010 s | ✅ |
| 单期回测 1 期（`run_backtest(...,1,include_tickets=True)`） | — | — | 0.405 s | 记录 |
| 单期全链路 `run_daily(backtest_n=0)` | ≤0.50 s | **≤1.10 s** | 0.866 / 0.935 / 0.943 s（均值 **0.915 s**） | ✅ 新预算达标（旧预算超 1.7–1.9×） |
| 100 期 `--no-tickets` | ≤45 s | ≤45 s | **36.90 s**（引擎自报 34.386 s） | ✅ |
| 100 期含注单 | ≤60 s | ≤60 s | **37.28 s**（引擎自报 34.699 s） | ✅ |
| 500 期 `--no-tickets` | ≤180 s | ≤180 s | **158.70 s**（引擎自报 156.160 s） | ✅（余量 21.3 s） |
| 500 期含注单 | ≤180 s | ≤180 s | 未测成（码小全改动中间态）；Lead 将随新 HEAD 提供 | ⏸ 新代码后复核 |
| 全量单测 253 用例 | ≤180 s | ≤180 s | **16.22 s** | ✅ |
| 覆盖求解 v=12,k=5,t=4,m=5（表命中） | ≤3 s | ≤3 s | 0.001 s（`elapsed_ms≈1`） | ✅ |
| `build_tickets` v=12 双玩法 | <0.1 s（护栏） | <0.1 s | 0.006–0.010 s | ✅ |
| ML 诊断通道（`--ml`） | ≤2.5 s/期 | ≤2.5 s/期 | 本机训练≈2.3 s/次（12 用例文件总 2.36 s，含 2 次训练） | ✅ 边际达标，默认关闭 |

单期全链路分项（一次采样，单位秒）：

| 分项 | 耗时 | 说明 |
| :--- | ---: | :--- |
| `load_dataset` | 0.043 | 2112 期 |
| `BinaryStateMatrix` 构造 | 0.005 | 前缀和 |
| `build_feature_snapshot` | 0.341 | 主要成本（图/马尔可夫/共现） |
| `score_round`（规则+池） | 0.002 | |
| `build_tickets` | 0.006 | 覆盖表命中 |
| `randomness_report` | 0.312 | 卡方/NIST/熵/图 |
| TXT / 微信渲染 | 0.001 / <0.001 | |
| Excel 渲染 | 0.117 | openpyxl |
| **`run_daily(backtest_n=0)`** | **0.866** | 0.341+0.312+0.117+IO |

> **性能结论（按新口径）**：单期核心均值 0.386 s ≤0.45 s ✅；单期全链路均值 0.915 s ≤1.10 s ✅；
> 100 期 36.90 s ≤45 s ✅；500 期 158.70 s ≤180 s ✅；全量单测 16.22 s ≤180 s ✅。
> 全链路成本主要来自 `build_feature_snapshot`(0.34 s)、`randomness_report`(0.31 s)、
> Excel 渲染(0.12 s)，后续若需再压缩优先看这三项。

## 五、契约口径核对（QA 独立结论）

1. **A2 共现聚合**：契约 §5 原文 `C̄o = mean(R(Co25), R(Co50), R(CoALL))`（等权、不含 10G）。
   实现 0.20/0.20/0.60 为偏离，Lead 已裁决按契约修回；本套件断言已同步为等权均值。
2. **选五全中概率**：`C(5,5)·C(75,15)/C(80,20) = 0.0006449246956 ≈ 0.0645%`（1/1550.6 期）。
   `docs/8.md` L190 与契约 §6.2 的 `0.155% / 645 期` 为需求文档笔误，按 Lead 裁决走契约
   §7 errata；测试断言精确超几何值。
3. **显著性护栏（契约 v1.3 / Lead 裁决）**：三条件式判定
   `n >= cfg.backtest.min_n_signal(500) 且 lift > cfg.backtest.lift_signal(1.05) 且 p < α_adj`；
   `bonferroni=True` 时 `α_adj = α/5 = 0.01`，`n<500` 一律 `False`，报告须标注样本不足。
   回测类用例待新实现落地后按此三条件补写（见 #9）。
4. **A5 金胆数**：契约公式 `clip(round(0.30×len(main)),3,5)`，main=12 → 4；当前实现恒 5，
   属 M2 已报 Lead（未在本轮写成失败断言，避免与新代码冲突）。
5. **`self_check.no_lookahead` 恒真（H4）**：本轮不写「穿越必失败」反证用例，等 H4 修复；
   否则会锁死假实现（review-card M14 已明确禁止循环论证）。
6. **状态文件路径**：`adaptive.state_path` 独立于 `output_dir`（默认相对 CWD 的
   `output/state/adaptive_weights.json`）。E2E/幂等测试必须显式传临时 `state_path`，
   否则会污染仓库 `output/`。
7. **既有 `tests/test_pipeline.py` 已隔离（Lead 授权，2026-11-02）**：`run_daily`/`run_backtest`
   显式传临时 `AdaptiveConfig(state_path=tmp)`；`main` CLI 用例改为写临时 JSON `--config`，
   把 `output_dir` 与 `adaptive.state_path` 一并重定向到 tmp，不再改写被 git 跟踪的
   `output/state/adaptive_weights.json` 与 `output/excel/*.xlsx`。
8. **ML 缓存口径（M12 修复后）**：键改为 `(窗口起点, refit 边界, schema_hash)`，允许同一
   `refit_every` 窗口内不同 `end` 复用；断言改为「同 refit 窗口内不同 end 命中（refit=False）、
   跨 refit 边界重训（refit=True）」。当前 `test_qa_caching.py` 仍是「同 end 命中」口径，
   新代码下发后按此重写。

## 六、数据保真与既有基线

- `load_dataset("kl8_history.txt")`：`T=2112`、`draws[0].period=2020001`、`draws[-1].period=2026291`。
- 既有 `tests/test_leakage.py` 覆盖：投毒函数本身、`truncated` 等价性、全仓无 KFold/shuffle
  静态扫描；**缺少**「投毒后当期推荐逐位不变」与 `self_check` 反证，本轮按 Lead 要求等新代码。
- P8 复核记录（review-card §8）：Lead 4 项修复（`m_margin=1/[1,3]`、ML 默认关闭 + `--ml`、
  8 步/回测进度回调、`.bat` UTF-8 BOM）均已被本套件 `test_qa_release.py` 实跑验证通过。

## 七、新增文件清单

| 文件 | 用例数 | 说明 |
| :--- | ---: | :--- |
| `tests/test_qa_release.py` | 19 | Lead 追加 a–e + CLI 默认值 + `lift_signal` 边界 |
| `tests/test_qa_numerics.py` | 32 | §6.2 数值手算对照 |
| `tests/test_qa_edges.py` | 28 | 边界/退化/类型防守/状态文件/筛选边界 |
| `tests/test_qa_rules.py` | 24 | R1–R5 双点边界 |
| `tests/test_qa_covering.py` | 14 | 覆盖设计穷举/预算/降级/确定性 |
| `tests/test_qa_caching.py` | 12 | ML 缓存 LRU 与 PAV 校准 |
| **合计** | **129** | 全量 253（含既有 124） |

> 另按 Lead 授权修改既有 `tests/test_pipeline.py`：为 `run_daily`/`run_backtest` 显式指定 tmp
> `adaptive.state_path`，并将 `main` CLI 用例改为临时 JSON `--config` 重定向输出，消除对仓库
> `output/` 的污染（不新增用例数）。