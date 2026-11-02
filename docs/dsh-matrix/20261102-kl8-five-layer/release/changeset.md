# 变更集（changeset）· 快乐8 五层重构

> **快照时间**：2026-11-02 19:2x（Asia/Shanghai，P9 定稿）
> **基线**：`d9bc139` chore: 重构前基线快照（旧 `calc_*` 八区微积分系统 + 2112 期数据 + 历史交付件）
> **当前 HEAD**：**`fe448cd`**（P9 定稿后 docs-only 提交：体检卡 v3 + `tests/README`）；**P9 代码终态 = `467b4e4`**（`5b2e462` 为 M16 修复提交）
> **数据来源**：`git log --oneline d9bc139..fe448cd`、`git diff --stat/--shortstat/--name-status d9bc139..467b4e4`（代码终态口径）与 `..fe448cd`（含文档）、`git status --short`（本稿生成时实测）

---

## 1. 提交序列（`git log --oneline d9bc139..HEAD`，共 12 个）

| # | commit | 标题 | 内容 |
| :--- | :--- | :--- | :--- |
| 1 | `85b0883` | docs(P1): 架构契约 contract.md(48需求点/签名冻结) + config-schema.md + requirement-mapping.csv | 契约 v1.0：R1–R48 映射、五层签名冻结、A1–A10 裁决 |
| 2 | `ad69760` | docs(P1): 契约升版 v1.1.0 | A7 `m_target` 公式修正、A5/A8/A9 自定标注、A10 ROI-N/A、`lift_signal`、`margin=1`、B4 ML 说明 |
| 3 | `251e795` | feat(kl8): 快乐8五层动力学重构（契约 R1-R48 全落地，124 用例全绿） | 新增 `kl8/` 全量五层代码 + `tests/` + `tools/` |
| 4 | `f170147` | chore: 删除旧 calc_* 八区微积分系统与旧命名交付件 | 删除 12 旧模块 + 2 旧脚本 + 3 旧测试 + 48 旧交付件 + 16 个根缓存 pyc |
| 5 | `e15d50c` | docs(P1): 契约 v1.2.0 | ML 默认关闭、`--ml`/`--no-tickets`、`--backtest` 按模式分派、权重实测值、性能实测表、`m_margin` 工程结论 |
| 6 | `3281992` | fix(cli): `--k` 帮助文字与实际支持的 5/7 玩法对齐 | main.py 帮助文案 1 行 |
| 7 | `95d1549` | docs(P1/P8): 契约 v1.3.0 + P8 体检卡 v1.1 | §7.4 概率勘误（0.0645%≈1/1550）、A2 口径冻结、§3.14.1 `prev_prediction.json` 契约、R43/R44 显著性硬规则（n≥500 + Bonferroni α=0.01）、§6.6 报告禁止项；体检卡阻断结论（6 High / 15 Medium） |
| 8 | `06611f5` | docs(P1): 映射校正 v1.3.1 | 修 R2/R46/R48 及 4 处用例名，48/48 全量校验通过 |
| 9 | `70ab253` | fix(p8): 归档 P8 体检卡 6 High + 14 Medium | 自适应权重真实现、硬约束真执行、归因真统计、`no_lookahead` 真判定、复盘持久化、去占位符、显著性样本门槛 + Bonferroni、报告零硬编码 |
| 10 | `5b2e462` | fix(p8): 归档 M16（B7 有序放宽路径可达）+ L15（ROI N/A 三态文案）+ 294 用例全绿 | 放宽路径改为可达 + 留痕；ROI N/A 三态文案；`test_qa_rework.py` 入库（+25 用例 → 294）；复审卡 v2 落盘 |
| 11 | `467b4e4` | chore(output): 刷新交付件（294 用例终态复跑） | 终态复跑刷新 `Keno动力学决策_*.xlsx`（6 Sheet） |
| 12 | `fe448cd` | docs(P7/P8): 体检卡 v3（G4 放行：High=0/Medium=0）+ tests/README（294 例终值与性能沿革） | 复审卡 **v3 §12**（G4 通过 PASS）；`tests/README` 更新 294 例与性能沿革 |

---

## 2. 相对 `d9bc139` 的 diff 统计（实测）

```
# 代码终态口径（P9 放行基线）
$ git diff --stat d9bc139..467b4e4 | tail -1
 184 files changed, 23052 insertions(+), 9318 deletions(-)

# 定稿时仓库 HEAD（含最后两个 docs-only 提交）
$ git diff --stat d9bc139..fe448cd | tail -1
 184 files changed, 23163 insertions(+), 9318 deletions(-)

$ git diff --name-status d9bc139..467b4e4 | 分类统计
 A=101    M=2    D=81
```

| 状态 | 文件数 | 说明 |
| :--- | ---: | :--- |
| 新增（A） | **101** | `kl8/` **44 个 `.py`** + `covering_tables.json`；`tests/` **37 个 `test_*.py`** + `README.md`；`tools/` 1；`docs/dsh-matrix/` 8；`.gitignore`；`output/` 8（4 交付件 + `prev_prediction.json` + 3 个 `.bak`） |
| 修改（M） | **2** | `main.py`（重写）、`运行微积分预测.bat`（重写） |
| 删除（D） | **81** | 旧 `calc_*` 12 + 旧脚本 2 + 旧测试 3 + 根 `__pycache__` 16 + 旧命名交付件 48 |
| 合计 | **184** | 代码终态口径 +23,052 / −9,318 行；含体检卡 v3 与 `tests/README` 的最终 HEAD（`fe448cd`）为 +23,163 |

> 自洽校验：101 + 2 + 81 = 184 ✓；`kl8/` 45（44 py + 1 json）+ `tests/` 38（37 py + README）+ `docs/` 8 + `output/` 8 + `tools/` 1 + `.gitignore` 1 = 101 ✓。

---

## 3. 新增清单（A=101）

### 3.1 代码：`kl8/` 44 个 `.py` + 1 个数据表

| 子包 | `.py` 数 | 文件 |
| :--- | ---: | :--- |
| `kl8/`（根） | 9 | `__init__.py`、`atomic_io.py`、`config.py`、`constants.py`、`datatypes.py`、`errors.py`、`history.py`、`pipeline.py`、`sync.py` |
| `layer1_representation` | 3 | `__init__.py`、`state.py`、`windows.py` |
| `layer2_baseline` | 7 | `__init__.py`、`priors.py`、`zscore.py`、`runs.py`、`chisq.py`、`randomness.py`、`entropy.py` |
| `layer3_features` | 8 | `__init__.py`、`heat.py`、`markov.py`、`cooccurrence.py`、`graph.py`、`morphology.py`、`ml.py`、`features.py` |
| `layer4_scoring` | 6 | `__init__.py`、`borda.py`、`rules.py`、`hybrid.py`、`adaptive.py`、`filter.py` |
| `layer5_combinatorial` | 3（+1 json） | `__init__.py`、`covering.py`、`tickets.py` + `covering_tables.json` |
| `evaluation` | 4 | `__init__.py`、`walkforward.py`、`metrics.py`、`leakage.py` |
| `reporting` | 4 | `__init__.py`、`text_report.py`、`wechat.py`、`excel.py` |
| **小计** | **44**（+1 json） | 业务模块 **36** + 包 `__init__` 8 |

### 3.2 测试：`tests/` 38 个文件（37 个 `test_*.py` + `README.md`）

`test_adaptive / test_architecture / test_chisq / test_config / test_constants / test_cooccurrence / test_covering / test_covering_tables / test_entropy / test_filter / test_graph / test_heat / test_history / test_leakage / test_markov / test_metrics / test_ml / test_morphology / test_pipeline / test_priors / test_qa_caching / test_qa_covering / test_qa_edges / test_qa_numerics / test_qa_release / test_qa_rework / test_qa_rules / test_randomness / test_reporting / test_rules / test_runs / test_scoring / test_state / test_tickets / test_walkforward / test_windows / test_zscore` + `tests/README.md`。

> 用例总数演进：**124** → **253** → **269**（P8 修复轮）→ **294**（终态，`Ran 294 tests in 34.861s` / `OK`；静态 `def test_` 参考值不等于该口径）。

### 3.3 工具与文档（10）

| 路径 | 说明 |
| :--- | :--- |
| `tools/build_covering_tables.py` | 幂等生成并**重验**覆盖表 |
| `docs/dsh-matrix/20261102-kl8-five-layer/contract.md` | 架构契约 **v1.3.1**（48 需求点、A1–A10、§7.4 errata、§6.6 报告禁止项） |
| `docs/.../config-schema.md` | `AppConfig` 全字段默认/范围/校验/CLI 覆盖规则 |
| `docs/.../requirement-mapping.csv` | 机检用 48 行映射（0 偏差） |
| `docs/.../review-card.md` | P8 体检卡 v2（复审：阻断仅 M16）+ §11 复审详情 |
| `docs/.../release/acceptance.md` | P9 验收报告 **v1.0 定稿**（G1–G5 全填，放行） |
| `docs/.../release/changeset.md` | 本文件 |
| `docs/.../release/runbook.md` | 上线运维手册 v1.0 |
| `docs/.../release/wechat-announcement.txt` | 发布文案（294 例、G4 放行、R4 负向披露；违规词 0） |
| `.gitignore` | 忽略 `__pycache__/`、`output/` 临时件等 |

### 3.4 新命名交付件与状态（8）

| 路径 | 说明 |
| :--- | :--- |
| `output/reports/Keno预测精要_20261102_第2026292期.txt`（+ `.bak`） | 六段式 TXT + 附录 A/B |
| `output/wechat/Keno微信发布_20261102_第2026292期.txt`（+ `.bak`） | 手机端发布文案 |
| `output/excel/Keno动力学决策_20261102_第2026292期.xlsx`（+ `.bak`） | 6 Sheet：当期总览 / 80 码五规则全参 / 覆盖注单 / 回测归因 / 随机性基准 / 上期复盘 |
| `output/state/adaptive_weights.json` | 自适应权重状态（`schema_version` + 幂等 upsert + 原子写） |
| `output/state/prev_prediction.json` | P8-H5：跨期预测持久化（`schema_version=1`） |

---

## 4. 修改清单（M=2）

| 路径 | 变更性质 | 内容摘要 |
| :--- | :--- | :--- |
| `main.py` | **重写**（340 行） | 新 CLI：`--mode {daily,backtest,audit}`、`--ml`、`--no-tickets`、`--k/--budget/--pool-size/--seed/--warmup/--config`；保留旧参数兼容；退出码 0/1/2/130；UTF-8 控制台重配 |
| `运行微积分预测.bat` | **重写**（43 行，UTF-8 BOM `EF BB BF`） | `chcp 65001` + `where python` 探测 + `python main.py --no-sync --backtest 30 --full-print`；失败分支中文提示 + `pause` |

---

## 5. 删除清单（D=81，`git rm -r` 保留历史可恢复）

| 类别 | 数量 | 明细 |
| :--- | ---: | :--- |
| 旧业务模块 | **12** | `calc_core.py`、`calc_engine.py`、`calc_backtest.py`、`calc_ensemble.py`、`calc_ensemble_pools.py`、`calc_excel.py`、`calc_graph.py`、`calc_ml.py`、`calc_report.py`、`calc_shrinkage.py`、`calc_signal_audit.py`、`calc_sync.py` |
| 旧脚本 | **2** | `audit_7days.py`、`generate_tonight_30.py` |
| 旧测试 | **3** | `test_calculus.py`、`test_shrinkage.py`、`test_ensemble_pools.py` |
| 根缓存 | **16** | `__pycache__/`：12 个 `calc_*.pyc` + `main.pyc` + 3 个旧测试 `.pyc` |
| 旧命名交付件 | **48** | `output/excel/快乐8微积分*.xlsx` **17**；`output/reports/微积分八区预测报告_*.txt` **16**；`output/wechat/微信分桶推荐_*.txt` **15** |
| **合计** | **81** | 与 `--diff-filter=D` 实测一致 |

**保留项**（未改动）：`kl8/`、`main.py`、`tests/`、`tools/`、`docs/`、`kl8_history.txt`（2112 期，未删未改）、`运行微积分预测.bat`、`output/` 新命名交付件。

---

## 6. 执行顺序（部署步骤）

> **纯 Python CLI 重构，无数据库迁移、无配置中心、无服务端部署。** 每步有判据，任一步失败即停：

| 步骤 | 操作 | 判据 |
| ---: | :--- | :--- |
| 1 | **备份**：`Copy-Item output -Destination output_bak -Recurse -Force`；复制 `kl8_history.txt` | 备份目录存在、期数 2112 |
| 2 | **切换代码**：`git checkout 467b4e4` | `git rev-parse --short HEAD` = `467b4e4` |
| 3 | **依赖核对**：按 runbook §1.1 跑依赖自检；**无新增依赖** | exit 0 |
| 4 | **编译**：`python -m compileall -q kl8 main.py tools` | **exit 0** |
| 5 | **单测**：`python -m unittest discover -s tests` | **`Ran 294 tests` / `OK`**（≈35–36 s） |
| 6 | **离线冒烟**：`python main.py --no-sync --backtest 0` | exit 0，交付件生成；`prev_prediction.json` 已落地 |
| 7 | **日常全链路**：双击 `运行微积分预测.bat` | exit 0（20.3 s），打印金胆/大底/TXT 正文 |
| 8 | **审计**：`python main.py --no-sync --mode audit --backtest 30` | exit 0（17.2/21.5 s，越界 0） |
| 9 | **切换消费方**：群转发模板、Excel 汇总脚本、定时任务改指新命令/新文件名 | 旧脚本名与旧文件名不再被引用 |

**回滚**：`git revert f170147` 恢复旧系统文件；`git checkout d9bc139` 整体回基线；回滚后重跑步骤 4–6。详见 `runbook.md` §8。

---

## 7. 工作区状态（截至本稿）

| 项 | 状态 | 说明 |
| :--- | :--- | :--- |
| 代码与既有文档 | **clean** | P9 代码终态 `467b4e4`；仓库 HEAD=`fe448cd`（docs-only）；`git status` 对代码无未提交改动 |
| 本 changeset / acceptance v1.0 / runbook / wechat | **工作区未提交** | P9 定稿在 `467b4e4` 之后继续完善终态证据，待 Lead 统一提交 |
| `output/` 交付件 | **随读变化** | 终态复跑已入库（`467b4e4`）；本地如再跑 `.bat` 会刷新，内容幂等 |

> 终态说明：本文件计数基于 `d9bc139..467b4e4`（184 files / A101·M2·D81，代码终态）；`fe448cd` 仅改文档故 A/M/D 不变、行数 +111。P9 文档（本文件、acceptance v1.0、runbook、wechat）提交后以 Lead 最终 `git diff --stat d9bc139..<新HEAD>` 为准。

---

## 8. 能力变更摘要（相对 d9bc139）

**新增**：五层解耦架构、防穿越唯一入口 `history.truncated`、零魔法数字配置化、Borda 秩归一化、金胆池(3~5)+复式池(10~15)+覆盖注单、三类标准化交付件、自适应权重状态、**跨期复盘持久化**、**原子写 + `.bak`**、**显著性硬门槛（n≥500 + Bonferroni α=0.01）**、需求映射机检（48 项）、**B7 有序放宽路径可达 + 留痕**。

**下线**：旧八区分桶推荐、旧微信分桶文案、旧 Excel 命名、12 个 `calc_*` API、2 个旧入口脚本。

**兼容性**：CLI 保留旧参数；`--no-sweep` 为空操作兼容；旧**交付件文件名与 TXT 版式不兼容**，消费方需同步调整。

---

## 9. 验证证据（P9 终态实测）

| 项 | 期望 | 实测 |
| :--- | :--- | :--- |
| `compileall -q kl8 main.py tools` | exit 0 | **达成**：exit 0，0 输出 |
| `unittest discover -s tests` | 全绿 | **达成**：`Ran 294 tests in 34.861s` / `OK`（测小严 36.285 s / 码小全 35.992 s） |
| `audit --backtest 30` | exit 0 | **达成**：exit 0，越界 0，`no_lookahead: True`，自检四项 True，指纹 `9095ab31c5c7bb5e` |
| `.bat` 冒烟 | exit 0 + 交付件 | **达成**：`BAT_EXIT=0`、20.3 s、中文无乱码、交付件全量重生成 |
| 残留引用 grep | `*.py`/`*.bat` 0 | **达成**：0 命中；`*.md` 18 处自指/归档（见 acceptance 附录 B） |
| 占位符 grep | 0 | **达成**：`C(v,k,t)`/`TODO`/`待填`/`★显著`=0 |
| 性能（两来源并列） | 核心 ≤0.55s / 全链路 ≤1.35s / 100 期 ≤60s / 500 期 ≤300s / 单测 ≤180s | 核心 **0.48–0.51 s**（Lead 0.481 / QA 0.510）；全链路 **1.243 s**；100 期无注单 **55.10 s**（QA，旧代码 36.9 s）/含注单 **50.1 s**（Lead）；500 期 **244.9 s**（Lead）；单测 **≈35–36 s**；注单生成 **0.006–0.008 s** |
| 性能因果 | 说明慢的代价 | **较 v1.2.0 基线（0.386s/0.915s）慢约 32%/36%**：硬约束真执行、归因逐期真算、每期防穿越校验、双口径高奖级；属正确性提升非退化；优化列下一迭代（热点：特征快照 0.34s、随机性报告 0.31s、Excel 渲染 0.12s） |
| 100 期量化 | Lift/p/ROI 诚实 | Lift@K 0.967~1.013 全平稳；ROI **-70.12%**、回撤 2429 元、连挂 27 期 |
| 500 期量化 | 显著性 | Lift@K 0.996~1.035 全平稳；**R4 Lift 0.951 / z=-2.77 显著为负**、R1 z=-2.03 → 建议下版剔除 R4 |
| G4 质检 | High/Medium 清零 | **达成**：High=0 / Medium=0（含 M16 关闭）；红线 11/12 严格通过 + 红线 1 余 5 处 Low 字面量（不阻断） |

> 本文件不代任何角色拍板；G1–G5 逐项结论见 `acceptance.md`（**v1.0 定稿：G1 达成 / G2 N-A / G3 达成 / G4 放行 / G5 达成**）。
