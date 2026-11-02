# 变更集（changeset）· 快乐8 五层重构

> **快照时间**：2026-11-02 18:11（Asia/Shanghai）
> **基线**：`d9bc139` chore: 重构前基线快照（旧 `calc_*` 八区微积分系统 + 2112 期数据 + 历史交付件）
> **当前 HEAD**：`3281992`（工作区仍有 P8 返工未提交，见 §7）
> **数据来源**：`git log --oneline d9bc139..HEAD`、`git diff --shortstat/--numstat/--name-status d9bc139..HEAD`、`git status --short`（均为本稿生成时实测）

---

## 1. 提交序列（`git log --oneline d9bc139..HEAD`，共 6 个）

| # | commit | 标题 | 内容 |
| :--- | :--- | :--- | :--- |
| 1 | `85b0883` | docs(P1): 架构契约 contract.md(48需求点/签名冻结) + config-schema.md + requirement-mapping.csv | 契约 v1.0：R1–R48 映射、五层签名冻结、A1–A10 裁决 |
| 2 | `ad69760` | docs(P1): 契约升版 v1.1.0 | A7 `m_target` 公式修正、A5/A8/A9 自定标注、A10 ROI-N/A、`lift_signal`、`margin=1`、B4 ML 说明 |
| 3 | `251e795` | feat(kl8): 快乐8五层动力学重构（契约 R1-R48 全落地，124 用例全绿） | 新增 `kl8/` 全量五层代码 + `tests/` 30 个测试文件 + `tools/` |
| 4 | `f170147` | chore: 删除旧 calc_* 八区微积分系统与旧命名交付件 | 删除 12 旧模块 + 2 旧脚本 + 3 旧测试 + 48 旧交付件 + 16 个根缓存 pyc |
| 5 | `e15d50c` | docs(P1): 契约 v1.2.0 | ML 默认关闭、`--ml`/`--no-tickets`、`--backtest` 按模式分派、权重实测值、性能实测表、`m_margin` 工程结论 |
| 6 | `3281992` | fix(cli): `--k` 帮助文字与实际支持的 5/7 玩法对齐 | main.py 帮助文案 1 行 |

---

## 2. 相对 `d9bc139` 的 diff 统计（实测）

```
$ git diff --shortstat d9bc139..HEAD
 166 files changed, 15580 insertions(+), 9318 deletions(-)

$ git diff --numstat d9bc139..HEAD  → 166 条记录
$ git diff --diff-filter=A --name-only d9bc139..HEAD | Measure-Object → 83
$ git diff --diff-filter=M --name-only d9bc139..HEAD                → 2
$ git diff --diff-filter=D --name-only d9bc139..HEAD                → 81
```

| 状态 | 文件数 | 说明 |
| :--- | ---: | :--- |
| 新增（A） | **83** | `kl8/` 43 个 `.py` + `covering_tables.json`、`tests/` 30 个、`tools/` 1 个、`docs/dsh-matrix/` 3 个、`.gitignore`、新命名交付件 4 个 |
| 修改（M） | **2** | `main.py`（重写）、`运行微积分预测.bat`（重写） |
| 删除（D） | **81** | 旧 `calc_*` 12 + 旧脚本 2 + 旧测试 3 + 根 `__pycache__` 16 + 旧命名交付件 48 |
| 合计 | **166** | +15,580 / −9,318 行 |

> 说明：`git diff --stat` 对二进制文件（`.pyc`/`.xlsx`）显示 `Bin`，故行数统计不含二进制；文件总数 166 = 83 + 2 + 81，自洽。

---

## 3. 新增清单（A=83）

### 3.1 代码：`kl8/` 43 个 `.py` + 1 个数据表

| 子包 | 新增数 | 文件 |
| :--- | ---: | :--- |
| `kl8/`（根） | 8 | `__init__.py`、`config.py`、`constants.py`、`datatypes.py`、`errors.py`、`history.py`、`sync.py`、`pipeline.py` |
| `layer1_representation` | 3 | `__init__.py`、`state.py`、`windows.py` |
| `layer2_baseline` | 7 | `__init__.py`、`priors.py`、`zscore.py`、`runs.py`、`chisq.py`、`randomness.py`、`entropy.py` |
| `layer3_features` | 8 | `__init__.py`、`heat.py`、`markov.py`、`cooccurrence.py`、`graph.py`、`morphology.py`、`ml.py`、`features.py` |
| `layer4_scoring` | 6 | `__init__.py`、`borda.py`、`rules.py`、`hybrid.py`、`adaptive.py`、`filter.py` |
| `layer5_combinatorial` | 4 | `__init__.py`、`covering.py`、`tickets.py`、**`covering_tables.json`**（已核验的经典旋转矩阵表） |
| `evaluation` | 4 | `__init__.py`、`walkforward.py`、`metrics.py`、`leakage.py` |
| `reporting` | 4 | `__init__.py`、`text_report.py`、`wechat.py`、`excel.py` |
| **小计** | **44**（43 `.py` + 1 `.json`） | 其中业务模块 35 个 + 包 `__init__` 8 个 |

> 口径差异提示：Lead 任务描述为「`kl8/` 39 模块」，实测为 **43 个 `.py`**（35 业务模块 + 8 包初始化）；按"不含 `__init__.py`"口径为 35，按"含 `__init__.py`"为 43，均非 39，验收时以实测数为准。

### 3.2 测试：`tests/` 30 个已提交（另 6 个 QA 待提交，见 §7）

`test_adaptive / test_architecture / test_chisq / test_config / test_constants / test_cooccurrence / test_covering / test_covering_tables / test_entropy / test_filter / test_graph / test_heat / test_history / test_leakage / test_markov / test_metrics / test_ml / test_morphology / test_pipeline / test_priors / test_randomness / test_reporting / test_rules / test_runs / test_scoring / test_state / test_tickets / test_walkforward / test_windows / test_zscore`（均为 `.py`）。

### 3.3 工具与文档

| 路径 | 新增 | 说明 |
| :--- | :--- | :--- |
| `tools/build_covering_tables.py` | 1 | 幂等生成并**重验**覆盖表（禁止照抄未核验数值） |
| `docs/dsh-matrix/20261102-kl8-five-layer/contract.md` | 1 | 架构契约（48 需求点、签名/字段冻结、A1–A10 裁决、DoD） |
| `docs/dsh-matrix/20261102-kl8-five-layer/config-schema.md` | 1 | `AppConfig` 全字段默认值/范围/校验/CLI 覆盖规则 |
| `docs/dsh-matrix/20261102-kl8-five-layer/requirement-mapping.csv` | 1 | 机检用 48 行映射（R1–R48） |
| `.gitignore` | 1 | 忽略 `__pycache__/`、`output/` 临时件等 |

### 3.4 新命名交付件（4）

| 路径 | 说明 |
| :--- | :--- |
| `output/reports/Keno预测精要_20261102_第2026292期.txt` | 六段式 TXT + 附录 A/B |
| `output/wechat/Keno微信发布_20261102_第2026292期.txt` | 手机端发布文案 |
| `output/excel/Keno动力学决策_20261102_第2026292期.xlsx` | 多 Sheet 商业报表 |
| `output/state/adaptive_weights.json` | 自适应权重状态（幂等 upsert） |

---

## 4. 修改清单（M=2）

| 路径 | 变更性质 | 内容摘要 |
| :--- | :--- | :--- |
| `main.py` | **重写**（340 行） | 新 CLI：`--mode {daily,backtest,audit}`、`--ml`、`--no-tickets`、`--k/--budget/--pool-size/--seed/--warmup/--config`；保留旧参数兼容；退出码 0/1/2/130；UTF-8 控制台重配 |
| `运行微积分预测.bat` | **重写**（43 行，UTF-8 BOM `EF BB BF`） | `chcp 65001` + `where python` 探测 + `python main.py --no-sync --backtest 30 --full-print`；失败分支中文提示 + `pause`，不静默退出；末尾列出常用命令 |

---

## 5. 删除清单（D=81，全部经 `git rm -r` 保留历史可恢复）

| 类别 | 数量 | 明细 |
| :--- | ---: | :--- |
| 旧业务模块 | **12** | `calc_core.py`、`calc_engine.py`、`calc_backtest.py`、`calc_ensemble.py`、`calc_ensemble_pools.py`、`calc_excel.py`、`calc_graph.py`、`calc_ml.py`、`calc_report.py`、`calc_shrinkage.py`、`calc_signal_audit.py`、`calc_sync.py` |
| 旧脚本 | **2** | `audit_7days.py`、`generate_tonight_30.py` |
| 旧测试 | **3** | `test_calculus.py`、`test_shrinkage.py`、`test_ensemble_pools.py` |
| 根缓存 | **16** | `__pycache__/`：12 个 `calc_*.pyc` + `main.pyc` + 3 个旧测试 `.pyc` |
| 旧命名交付件 | **48** | `output/excel/快乐8微积分*.xlsx` **17** 个；`output/reports/微积分八区预测报告_*.txt` **16** 个；`output/wechat/微信分桶推荐_*.txt` **15** 个（日期覆盖 20261018–20261102） |
| **合计** | **81** | 与 `--diff-filter=D` 实测一致 |

> 口径差异提示：Lead 任务描述为「44 个旧命名交付件」，实测为 **48 个**（17 Excel + 16 报告 + 15 微信），删除总数 81 与 diff 自洽。

**保留项**（未改动）：`kl8/`、`main.py`、`tests/`、`tools/`、`docs/`、`kl8_history.txt`（2112 期，未删未改）、`运行微积分预测.bat`、`output/` 下新命名交付件。

---

## 6. 执行顺序（部署步骤）

> **结论：本变更集为纯 Python CLI 重构，无数据库迁移、无配置中心、无服务端部署。** 按以下顺序执行即可（每步都有判据，任一步失败即停）：

| 步骤 | 操作 | 判据 |
| ---: | :--- | :--- |
| 1 | **备份**：`Copy-Item output -Destination output_bak -Recurse -Force`；复制 `kl8_history.txt` | 备份目录存在、期数 2112 |
| 2 | **切换代码**：`git checkout <目标提交>`（发布即 `HEAD`）或解压发布包 | `git rev-parse --short HEAD` 符合预期 |
| 3 | **依赖核对**：按 runbook §1.1 跑依赖自检；**无新增依赖**，无需重装 | exit 0，版本与 §1 一致 |
| 4 | **编译**：`python -m compileall kl8 main.py tools` | **exit 0** |
| 5 | **单测**：`python -m unittest discover -s tests -v` | **exit 0 且全绿**（用例数以终态实跑为准） |
| 6 | **离线冒烟**：`python main.py --no-sync --backtest 0` | exit 0，四类交付件生成（`prev_prediction.json` 待 P8-H5 落地后出现） |
| 7 | **日常全链路**：双击 `运行微积分预测.bat` 或 `python main.py --no-sync --backtest 30 --full-print` | exit 0，控制台打印金胆/大底/TXT 正文 |
| 8 | **审计**：`python main.py --no-sync --mode audit --backtest 30` | exit 0（审计项全通过） |
| 9 | **切换消费方**：群转发模板、Excel 汇总脚本、定时任务改指新命令/新文件名 | 旧脚本名（`audit_7days.py`/`generate_tonight_30.py`）与旧文件名不再被引用 |

**回滚**：见 `runbook.md` §8（`git revert f170147` 恢复旧系统文件；`git checkout d9bc139` 整体回到基线；回滚后重跑步骤 4–6）。

---

## 7. 工作区未提交项（截至快照，最终 changeset 会更大）

| 项 | 状态 | 说明 |
| :--- | :--- | :--- |
| `docs/dsh-matrix/20261102-kl8-five-layer/contract.md` | **M（未提交）** | 已升 **v1.3.0**：§7.4 概率勘误、A2 口径冻结、§3.14.1 `prev_prediction.json` 契约、R43/R44 显著性硬规则（n≥500 + Bonferroni α=0.01）、§6.6 报告禁止项 |
| `tests/test_qa_*.py` | **6 个未跟踪** | `test_qa_caching / test_qa_covering / test_qa_edges / test_qa_numerics / test_qa_release / test_qa_rules`（QA 补测，实测 `tests/` 现有 36 个 `test_*.py`） |
| `docs/dsh-matrix/20261102-kl8-five-layer/review-card.md` | **未跟踪** | P8 体检卡 v1.1（阻断结论：6 High / 15 Medium） |
| `output/excel/...xlsx`、`output/state/adaptive_weights.json` | **M（未提交）** | 运行产物，随每日跑变化，一般不入库或入库以最新为准 |

> **终态说明**：P8 返工（码小全修复 6 High + 指定 Medium）完成后，`kl8/` 与 `tests/` 会再产生改动，届时 **A/M/D 计数与行数统计需由 Lead 在最终提交后重跑本文件 §2 命令刷新**；本稿数字为 2026-11-02 18:11 快照。

---

## 8. 能力变更摘要（相对 d9bc139）

**新增能力**：五层解耦架构（数据表征→随机性基准→多尺度特征→五规则共振打分→覆盖缩水+Walk-Forward 回测）、防穿越唯一入口 `history.truncated`、零魔法数字配置化、Borda 秩归一化、金胆池(3~5)+复式池(10~15)+覆盖注单、三类标准化交付件、自适应权重状态、需求映射机检（48 项）。

**下线能力**：旧八区分桶推荐（`微积分八区预测`）、旧微信分桶文案（`微信分桶推荐`）、旧 Excel 命名（`快乐8微积分*.xlsx`）、12 个 `calc_*` 模块 API、2 个旧入口脚本。

**兼容性**：CLI 保留旧参数（`--history/--no-sync/--quick/--full-sync/--period/--date/--backtest/--full-print`）；`--no-sweep` 为空操作兼容；旧**交付件文件名与 TXT 版式不兼容**，消费方需同步调整。

---

## 9. 验证证据（占位，终态填 acceptance.md）

| 项 | 期望 | 现状 |
| :--- | :--- | :--- |
| `compileall kl8 main.py tools` | exit 0 | **待补**（P8 返工中） |
| `unittest discover -s tests -v` | 全绿 | **待补**（返工中；**用例数唯一口径 = 输出行 `Ran N tests`**，静态 `def test_` 253 仅参考） |
| `audit` 模式 | exit 0 | **待补** |
| 日跑 `.bat` 冒烟 | exit 0 + 4 类交付件 | **待补** |
| 残留引用 grep | `*.py`/`*.bat` 0 处 | **已初步实测 0 处**（`*.md` 15 处为归档豁免 + 契约自指，见 acceptance.md 附录 B） |

> 本文件不代 Lead 拍板；G1–G5 逐项结论见 `acceptance.md`（**v0.2：G1/G2/G5 已填，G3/G4 待补**）。