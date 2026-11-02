# 快乐8（Keno）五层动力学系统 · 上线运维手册（runbook）

> **文档状态**：**v1.0（P9 终态版，2026-11-02 19:2x）**
> **适用对象**：本机操作者（双击 `.bat` 的日常使用者）+ 维护者（回测/审计/排障）
> **仓库**：`C:\D-pan\Dpanqianyi\微积分分区`（Windows 11 + PowerShell，Python 3.11.6）
> **代码基线**：`d9bc139`（重构前快照）→ **当前 `HEAD=70ab253`（P8 返工已提交，代码工作区 clean）**
> **重要前提**：P8 体检卡 **6 High + 14 Medium 已修复提交**（M15 为文档勘误，走契约 §7.4 errata 关闭），**审大关复审中**；本手册性能数字为 **P9 终态实测**（未标来源者为 Lead 实跑，release-ops 复跑值单独标注）。

---

## 1. 环境要求

| 项 | 要求 | 本机实测（2026-11-02） | 说明 |
| :--- | :--- | :--- | :--- |
| 操作系统 | Windows 10/11（含 PowerShell 5.1+ 或 pwsh 7+） | Windows 11 | `.bat` 依赖 `cmd` + `chcp 65001` |
| Python | **3.11.x**（推荐 3.11.6） | **3.11.6** | 需在 PATH 中，`.bat` 用 `where python` 探测 |
| numpy | 2.x | **2.4.6** | 向量化核心，五层全依赖 |
| scipy | 1.x | **1.17.1** | 卡方/超几何/`milp`（覆盖求解降级路径） |
| openpyxl | 3.x | **3.1.5** | Excel 交付件渲染 |
| networkx | 3.x | **3.6.1** | 共现图 PageRank/特征向量/Louvain；缺失可降级 |
| lightgbm | 4.x | **4.7.0** | **可选**，仅 `--ml` 诊断通道；缺失自动降级 `ema/uniform` |
| 磁盘 | `output/` 可写，建议保留 ≥200 MB | — | 每次日跑覆盖写 3 类交付件 |
| 网络 | **非必需** | — | 同步失败可 `--no-sync` 全离线运行 |

**依赖结论**：本次重构**未新增第三方依赖**（numpy/scipy/openpyxl/networkx 原系统已在用，lightgbm 仅诊断用），仓库根目录**实测不存在** `requirements.txt` / `pyproject.toml`，依赖按上表版本手工安装。

### 1.1 依赖自检（一条命令）

```powershell
python -c "import sys,numpy,scipy,openpyxl,networkx;print(sys.version.split()[0]);print('numpy',numpy.__version__);print('scipy',scipy.__version__);print('openpyxl',openpyxl.__version__);print('networkx',networkx.__version__)"
```

期望：Python `3.11.6` + 四个版本号与上表一致，**exit 0**。`python -c "import lightgbm;print(lightgbm.__version__)"` 失败不阻断主流程。

---

## 2. 首次安装（5 步）

```powershell
# 0) 进入仓库（路径含中文，务必加引号）
cd "C:\D-pan\Dpanqianyi\微积分分区"

# 1) 取代码（二选一）
git clone <repo-url> "C:\D-pan\Dpanqianyi\微积分分区"   # 首次拉取
git pull                                                   # 已有仓库更新

# 2) 安装依赖（版本按 §1 锁定；无 requirements.txt）
python -m pip install numpy==2.4.6 scipy==1.17.1 openpyxl==3.1.5 networkx==3.6.1
python -m pip install lightgbm==4.7.0        # 仅需 ML 诊断通道时安装
# 国内网络可用镜像：-i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 依赖自检（见 §1.1），fail 则先修环境

# 4) 编译自检（必须 exit 0）
python -m compileall kl8 main.py tools

# 5) 离线冒烟（首次跑：跳过伴随回测，最快出交付件）
python main.py --no-sync --backtest 0
```

**首次跑通判据**：控制台打印 `✔ 预测目标: 第 N 期`、`🔥 超级金胆`、`⭐ 精选大底`、`📁 全渠道交付物已就绪`，且 `output/reports|wechat|excel` 三处各新增 1 个当日文件；退出码 0。

> 需要完整验证时再执行：`python -m unittest discover -s tests -v`（**用例数唯一口径 = 输出行的 `Ran N tests`**；P9 终态实测 **`Ran 269 tests in 21.916s` / `OK`**，演进 124→253→269）。

---

## 3. 日常运维（三种标准用法）

### 3.1 双击一键入口（推荐日常）

双击仓库根目录 **`运行微积分预测.bat`**，等价于：

```powershell
python main.py --no-sync --backtest 30 --full-print
```

| 特性 | 说明 |
| :--- | :--- |
| 编码 | 文件为 **UTF-8 BOM（`EF BB BF`，实测）** + `chcp 65001`，中文不乱码 |
| 工作目录 | `cd /d "%~dp0"`，从任意位置双击均可 |
| 行为 | 跳过联网同步（`--no-sync`）→ 日常预测 → **伴随 30 期回测** → 控制台打印完整 TXT 正文（`--full-print`） |
| 失败处理 | Python 缺失 / 流程非 0 → 中文提示 + `pause`，**不静默退出** |
| 产出 | §4 的四类交付件（当日覆盖写） |

### 3.2 纯回测模式（评估用，不产出当日交付件）

```powershell
# 标准 100 期 Walk-Forward（契约 R48-b：backtest 模式默认 100）
python main.py --no-sync --mode backtest --backtest 100

# 只看 Lift、跳过注单与奖金（秒级，约快 3~4 倍）
python main.py --no-sync --mode backtest --backtest 100 --no-tickets

# 小样本快速核对（<500 期一律不判"显著"，见 §5 口径说明）
python main.py --no-sync --mode backtest --backtest 30 --no-tickets
```

输出：样本量 / 覆盖期号区间 / Top-K `Lift@K` 表（含 t 统计量、p 值、`★显著`或`平稳`）/ 资金曲线（选五口径；未确认奖级出 `N/A`）。退出码 0。

### 3.3 审计模式（防穿越与自洽性机检）

```powershell
python main.py --no-sync --mode audit --backtest 30
```

输出：代码级防穿越审计是否洁净、回测自洽性机检项、评估期数、配置指纹（`config_hash`）。**任何审计项不通过 → 退出码 1**（0 成功 / 1 参数或数据错误 / 2 内部异常 / 130 手动中断）。

**P9 实测**：30 期越界 **0 期**，报告自洽四项全 `True`（`sample_count_matches / lift_recalculated / bankroll_recalculated / no_lookahead`），指纹 `9095ab31c5c7bb5e`，耗时 **17.2 s**（Lead）/ **21.5 s**（复跑）。

### 3.4 高级参数（按需，勿日常滥用）

```powershell
python main.py --no-sync --ml --backtest 0          # 开 LightGBM 诊断通道（显著变慢，见 §6-6）
python main.py --no-sync --k 7 --budget 10          # 切选七玩法、10 注
python main.py --no-sync --pool-size 15             # 复式池 15 码（契约区间 10~15）
python main.py --no-sync --seed 12345               # 复现同一随机结果
python main.py --no-sync --period 2026293 --date 2026-11-03   # 指定目标期/日期
python main.py --no-sync --config my_config.json    # 自定义配置（见 §5）
```

**参考耗时（P9 终态实测，HEAD=`70ab253`）**：单期核心 **0.481 s**（连续 3 期 0.488/0.484/0.473，均值；特征+打分 0.482/0.478/0.467，**注单生成 0.006 s（`table` 查表命中）**；真实 2112 期、ML 关、自适应内存态）；daily `--backtest 30` 端到端 **17.9 s**（≈0.60 s/期，含 30 期伴随回测与全渠道渲染）；100 期含注单 **50.1 s**（修复前 40 s，**+10.1 s 为新增真实计算成本**：硬约束真执行 + 归因真统计 + 双口径高奖级 + 防穿越计数）；`audit --backtest 30` **17.2 s**（复跑 21.5 s）；全量单测（269 例）**21.916 s**；**500 期 `--no-tickets` 244.9 s**；`--ml` 开启 2.03 s/期（v1.2.0 值，P9 未重测）。

**性能预算（P9，含修订建议）**：单期核心 原 ≤0.45 s → **建议 ≤0.55 s**（实测 0.481 s，超原 6.9%；修复轮把"只告警的硬约束"改为**真执行**并逐期累计归因，属正确性要求；修复前同口径 0.386 s，差额 0.095 s 全部来自新增真计算）/ 单期全链路 ≤1.10 s / 100 期 原 ≤45 s → **建议 ≤60 s**（实测 50.1 s）/ 500 期 原 ≤180 s → **建议 ≤300 s**（实测 244.9 s）/ 单测 ≤180 s。**修订待 Lead 批准；超预算须给优化证据，不作为 SLA。**

---

## 4. 产出文件清单

| 类别 | 路径模板 | 内容 | 读取方 |
| :--- | :--- | :--- | :--- |
| TXT 精要报告 | `output/reports/Keno预测精要_YYYYMMDD_第N期.txt` | 六段式：上期复盘 / 随机性基准 / 四尺度动量 / 五规则共振+FinalScore Top20 / 金胆池+复式池+形态体检 / 覆盖注单；附录 A 80 码全参、附录 B 回测归因 | 人工阅读、归档 |
| 微信发布文案 | `output/wechat/Keno微信发布_YYYYMMDD_第N期.txt` | 手机端 emoji 分块，≤1200 字，一键全选转发 | 微信群 |
| 商业 Excel | `output/excel/Keno动力学决策_YYYYMMDD_第N期.xlsx` | 多 Sheet（当期总览/80 码五规则全参/覆盖注单/回测归因/随机性基准/上期复盘） | 复盘、汇报 |
| 自适应权重状态 | `output/state/adaptive_weights.json` | 规则权重 R1–R5、命中率、`n_updates`、`updated_at`；按 period 幂等 upsert、原子写 | 引擎自读 |
| 上期预测复盘状态 | `output/state/prev_prediction.json` | **P8-H5 修复轮新增**：跨期预测持久化（`schema_version` + 按 period 幂等 upsert），供"上期预测命中"如实复盘 | 引擎自读 |

**当前实测产物（2026-11-02 19:2x 快照，P8 修复轮后，HEAD=`70ab253`）**：

```
output/reports/Keno预测精要_20261102_第2026292期.txt        (+ .txt.bak)
output/wechat/Keno微信发布_20261102_第2026292期.txt         (+ .txt.bak)
output/excel/Keno动力学决策_20261102_第2026292期.xlsx       (+ .xlsx.bak)
output/state/adaptive_weights.json
output/state/prev_prediction.json      # 已落地：schema_version=1，predictions keys=[2026292]
```

**注意**：`prev_prediction.json` **已随 P8-H5 修复落地**（原子写 + 按 period 幂等 upsert；首次运行或上期无记录时复盘固定输出「首期无复盘基线」，属正常）；`.bak` 为原子写前的上一版备份（可留 1 份回看，勿手工改名）。**旧命名交付件**（共 48 个）已随 `f170147` 删除，**不再生成**。

---

## 5. 配置覆盖方式

### 5.1 优先级与合并规则

`默认值 → --config JSON → CLI 参数`，**深合并**后执行 `validate_config` 严格校验；任何字段非法 → `ConfigError` + 中文修复建议 + **退出码 1**（不会带病运行）。

### 5.2 CLI 参数速查

| 参数 | 取值 | 作用 |
| :--- | :--- | :--- |
| `--mode {daily,backtest,audit}` | 默认 `daily` | 运行模式分派 |
| `--k {5,7}` | 5=选五，7=选七 | 玩法（联动 `covering.k/play/t`；其它值被校验拒绝） |
| `--budget N` | 默认选五 20、选七 10；上限 200 | 注单预算注数 |
| `--pool-size N` | 契约区间 10~15，默认 12 | 精选复式大底规模 |
| `--seed N` | 默认 `20261102` | 全局随机种子（Louvain/SA/ML/抽样唯一随机源） |
| `--ml` | flag，默认关 | 开启 LightGBM 诊断通道（**慢，仅诊断**） |
| `--warmup N` | 默认 300（下限 60） | 回测冷启动期数 T0 |
| `--no-tickets` | flag | 回测跳过注单生成与奖金核算（快速 Lift 评估） |
| `--backtest N` | 未给时：daily→30、backtest/audit→100；`0`=daily 关闭伴随回测 | 回测期数 |
| `--no-sync` / `--full-sync` | flag | 跳过同步 / 全量同步（默认增量） |
| `--history PATH` | 默认 `kl8_history.txt` | 历史数据文件 |
| `--period N` / `--date S` | 默认最新+1 | 指定预测目标期/日期 |
| `--config PATH` | JSON | 自定义配置（深合并） |
| `--full-print` | flag | 控制台打印完整 TXT 正文 |

### 5.3 `--config` JSON 示例（嵌套结构，只写需覆盖的字段）

```json
{
  "weights": { "co": 0.05, "omission": 0.35 },
  "rules": { "r1_rank_threshold": 0.65 },
  "covering": { "k": 5, "budget": 20, "m_margin": 1 },
  "backtest": { "warmup": 300, "min_n_signal": 500 },
  "seed": 20261102
}
```

### 5.4 校验红线（违反即拒绝启动）

- 七维打分权重 `Σω = 1.00 ± 1e-9`（`ml` 为额外叠加项，不计入 Σω）
- 拉普拉斯平滑 `α ∈ [0.5, 2.0]`；票数加成 `λ ∈ [0.05, 0.10]`
- `gold_max < main_min`（金胆池严格小于复式池）；`k ≤ 池大小`
- 预算 `≤ max_budget=200`；覆盖假设 `m_margin ∈ [1,3]`；重号配比区间有序
- Trend/R4 内部权重和 = 1；回测冷启动关系成立

---

## 6. 常见故障与处置（6 类）

### 6.1 开奖数据同步失败

- **现象**：控制台出现 `开奖数据同步异常降级（使用本地历史）: ...` 或 `SyncResult ok=False` 的 warning；**流程不中断**。
- **机制**：官网主源 + 备用源增量拉取，网络/解析异常内部捕获转 warning，**不修改本地文件**（两源全失也只有 `ok=False`，宁可少一期不写坏历史）。
- **处置**：
  1. 直接改用离线：`python main.py --no-sync ...`（日常一切照旧）；
  2. 需要补数据：确认代理/VPN，重跑 `--full-sync`；
  3. 核对 `kl8_history.txt` 首行期号（文件为**新期在前降序**）是否已推进；
  4. 仍失败：手工补行后跑 `python -m compileall` 无关，直接 `python main.py --no-sync --backtest 0` 验证能读。

### 6.2 Excel 被 Excel/WPS 进程占用

- **现象**：`PermissionError`；交付件 Excel 未更新（TXT/微信通常已写成功）。
- **机制**：渲染层对文件占用**重试 3 次**；达上限后记 warning 并返回路径。交付件改为**原子写**（先写临时文件再 `os.replace`，旧版留 `.bak`），P8-M13 已修复并实测生效。
- **处置**：
  1. 关闭所有 Excel/WPS 窗口后重跑（最有效）；
  2. 任务管理器确认无残留 `EXCEL.EXE`；
  3. 临时改日期/期号重跑生成新文件名，事后合并；
  4. **不要**手工删除正在写的 `*.xlsx` 及其 `~$` 锁文件以外的目录。

### 6.3 历史期数不足（T0=300 冷启动）

- **现象**：`[错误 - 业务约束阻断] 数据集样本量 X 期不足以支持冷启动窗口 T0=300！`（退出码 1）。
- **阈值**：日常预测最低 `min_history=40` 期；回测需 `T > T0 + t0_min(60)`。
- **处置**：
  1. 降冷启动：`--warmup 100`（不得低于 60，且必须 < 本地期数）；
  2. 缩短回测：`--backtest 10` 或 `0`；
  3. 补历史数据（`--full-sync` 或手工追加，注意降序 + 20 码去重 + 期号唯一，重复期号会抛 `DataFormatError`）。

### 6.4 状态文件损坏

- **现象**：`自适应权重` 相关 warning：`状态文件损坏: ...，已备份至 output/state/adaptive_weights.json.bak 并重置`；或复盘段落输出「**首期无复盘基线**（无可用的上期预测记录）」。
- **机制**：`adaptive_weights.json` 损坏/`schema_version` 不符 → **自动改名 `.bak` + 重置均匀权重**，不中断；`prev_prediction.json` 缺失/损坏 → 复盘固定输出"首期无复盘基线"，**绝不冒充上期命中**。
- **处置**：
  1. 无需人工干预，跑一轮后即重建；如需保留证据，先复制 `output/state/*.bak`；
  2. 想彻底重置：停止运行时删除 `output/state/adaptive_weights.json`（下次自动生成均匀先验）；
  3. 复盘基线缺失属正常（首次运行/新期号），不是故障。

### 6.5 `m` 假设偏离告警

- **现象**：覆盖段出现 warning：`m 假设与池内期望命中数偏离`（`|m - (ceil(0.25v)+2)| > 2`）。
- **定性（必读）**：`m` 是**用户/玩法假设参数**（"预期池中能命中几个开奖号"），**不是算法推导值、不是预测值**；报告与对外文案必须写明"m 为假设参数"。
- **处置**：
  1. 接受并在报告中注明；`m` 由玩法与池大小决定（选五 `C(v,5,4,m)`，`v=12→m=5`；选七 `C(v,7,5,m)`，`v=12→m=6`）；
  2. 调 `--pool-size`（10~15）或 `covering.m_margin`（1~3，默认 1）改变假设；
  3. 告警不等于错误，**不要**为了消警把 `m_margin` 调成非法值。

### 6.6 `--ml` 开启后明显变慢

- **现象**：单期 0.39 s → 约 2.03 s（v1.2.0 实测），回测整体慢数倍。
- **机制**：`--ml` 显式开启 LightGBM 训练/推理通道；**默认关闭**（`ml.enabled=False`、`ml.weight=0`，不进入 FinalScore）。
- **处置**：
  1. 日常/发版**不要**加 `--ml`；仅做诊断时短跑（如 `--backtest 0` 或 10 期）；
  2. `lightgbm` 缺失或训练异常 → 自动降级 `ema/uniform` 并写 warning，**不中断**；
  3. 需要长期开启时先做耗时预算评估（性能预算 ≤2.5 s/期）。

### 6.7 其它速查

| 现象 | 处置 |
| :--- | :--- |
| 退出码 2 + `[致命内部异常]` | 抓完整 traceback 与操作命令报维护者；多为数据/环境异常 |
| 退出码 130 | 用户 Ctrl+C 中断，重跑即可 |
| 控制台中文乱码 | 用 `.bat`（自带 `chcp 65001`）；PowerShell 手工运行时先 `chcp 65001` |
| 路径含中文报错 | 命令中给路径加引号；确认 `cd` 到仓库根（用 `Get-Location`） |
| 交付件日期/期号错位 | 检查 `--period/--date` 是否显式指定；P8-M9（日期兜底硬编码）修复后应报错而非静默兜底 |

---

## 7. 发布前 / 变更后自检清单

```powershell
cd "C:\D-pan\Dpanqianyi\微积分分区"
git rev-parse --short HEAD                                   # 确认目标提交
python -m compileall kl8 main.py tools; echo "COMPILE_EXIT=$LASTEXITCODE"      # 期望 0
python -m unittest discover -s tests -v; echo "UNITTEST_EXIT=$LASTEXITCODE"    # 期望 0/全绿（P9 基线：Ran 269）
python main.py --no-sync --mode audit --backtest 30; echo "AUDIT_EXIT=$LASTEXITCODE"  # 期望 0
python main.py --no-sync --backtest 30 --full-print          # 日常冒烟，期望 0
Get-ChildItem output/reports,output/wechat,output/excel,output/state -File | Select-Object Name,Length
```

残留引用检查（应为 0；豁免见 acceptance.md 附录 B）：

```powershell
git grep -n -E "calc_core|calc_engine|calc_backtest|calc_ensemble|calc_shrinkage|calc_report|calc_ml|calc_graph|calc_signal_audit|calc_sync|calc_excel" -- "*.py" "*.bat"
```

---

## 8. 回滚方案

| 场景 | 命令 | 说明 |
| :--- | :--- | :--- |
| 只回滚"删除旧系统" | `git revert f170147` | 恢复 12 个 `calc_*.py`、2 旧脚本、3 旧测试、48 旧命名交付件；**但 `main.py`/`.bat` 已重写，必须一并回滚对应提交才能回到旧入口** |
| 整体回到重构前基线 | `git checkout d9bc139` | detached HEAD，整套旧八区微积分系统；`git switch -c hotfix d9bc139` 更稳妥 |
| 回到上一稳定版 | `git revert <坏提交>` 或 `git checkout <上一 tag/commit>` | 优先 `revert`（保留历史），避免 `reset --hard` 丢工作区 |
| 仅回滚配置/状态 | 删除 `output/state/adaptive_weights.json`（自动重建）；`--config` 换回默认 | 不动代码 |

**回滚步骤（建议顺序）**：① 备份 `output/` 与 `kl8_history.txt`（`Copy-Item -Recurse`）；② 确认目标 commit；③ `git checkout`/`git revert`；④ `python -m compileall kl8 main.py tools`；⑤ `python main.py --no-sync --backtest 0` 冒烟；⑥ 用旧 `.bat`/旧脚本验证入口；⑦ 记录回滚 commit 与原因。

**回滚风险**：旧系统入口是 `audit_7days.py` / `generate_tonight_30.py`（已删），外部定时任务若仍指向它们需同步改回；旧交付件命名与新命名不兼容，消费方（微信群转发模板/Excel 汇总脚本）需同步切换。

---

## 9. 变更记录

| 版本 | 日期 | 变更 |
| :--- | :--- | :--- |
| v0.9-draft | 2026-11-02 | P9 提前启动稿：环境/安装/三种用法/产出清单/配置/6 类故障/回滚/自检；性能为 v1.2.0 参考值 |
| **v1.0** | 2026-11-02 | **P9 终态版**（HEAD=`70ab253`）：用例 269；compileall/audit/.bat 实测；性能换 P9 终态数字（含 100 期 50.1 s、500 期 244.9 s 超预算披露）；`prev_prediction.json` 已落地；交付件原子写 + `.bak`；回滚与自检口径复核 |