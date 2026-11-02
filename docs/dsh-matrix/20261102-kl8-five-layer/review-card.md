# P8 代码定稿体检卡（review-card）

> 审查人：审大关（review-audit）｜ 审查时间：2026-11-02 17:56–18:2x ｜ 模式：**只读审查**（未改任何业务代码）
> 卡版本：**v1.1**（2026-11-02：并入 Lead 三项裁决回执 + 返工/复审安排，见 §10；v1 为首轮审查结论，§1–§9 问题状态不变）
> 审查基线：`git diff d9bc139..HEAD`，HEAD=`3281992`（工作区），预重构快照 `d9bc139`
> 变更规模：**166 files changed, 15580 insertions(+), 9318 deletions(-)**
> 契约基线：`contract.md` **v1.2.0**（提交 `e15d50c`，与 HEAD 对齐）+ `config-schema.md` v1.2.0；需求原文 `docs/8.md`（202 行，**最高效力**）
> 审查方法：全量静态走查（关键路径逐行 + 全仓 grep/AST 扫描）+ 我方独立动态复跑（单测/编译/audit/回测）+ 交付件实物核对

---

## 0. 结论：**阻断（BLOCK）**

| 级别 | 数量 | 是否阻断 |
| :--- | ---: | :--- |
| High | **6** | 是 |
| Medium | **15**（其中 M15 已按 Lead 裁决降为"文档勘误"，非代码缺陷，由智小研 errata 关闭） | 是 |
| Low | 14 | 否（随本轮修） |
| 12 条红线（task-8 口径） | **7 条违规 / 4 条部分违规 / 1 条通过** | 是 |

一句话结论：**架构、分层、向量化、防穿越主链、覆盖表查表与穷举核验这几块是真做出来的、质量不错；但 layer4 的三处"展示级真、计算级假"（R33 自适应、R34 硬约束、R45 归因）和一处自检恒真（`no_lookahead`）导致"五规则共振 + 自适应权重"这条**核心卖点链路**在引擎里并没有真正闭环；交付件里已有可自证的矛盾（"均匀性通过"与 NIST 四项"预警"同屏、池子明确违反 45% 硬约束、理论保底显示占位符 `C(v,k,t)`、复盘口径算错）。按 G4 口径，有 High/Medium 即阻断。**

### 审查期间基线变动（需 Lead 知悉，非缺陷但影响举证）

1. 我方开工时 HEAD=`f170147`，审查过程中 HEAD 前移两次：`e15d50c`（契约 v1.2.0 提交，契约正文 +107/-… 对齐实现）→ `3281992`（仅改 `main.py` `--k` 帮助文字 1 行）。**本卡所有结论基于 `3281992` 工作区复跑。**
2. `tests/test_qa_numerics.py`、`tests/test_qa_release.py` 为**未提交**新文件（qa-test 并行产出，+51 用例）。我方最终单测证据为**含这两个文件的 175 用例**（见 §7）。
3. 契约 v1.2.0 属"**实现漂移后回写契约**"（ScoreWeights 默认值、`ml.enabled=False`、`run_daily backtest_n=30`、Σω 语义均按实现改写），且回写**不完整**：A2、B5、R33、R34、R41、R45 仍与实现不符（见下）。请 Lead 明确：回写契约不等于需求达成，**判定仍以 8.md 为准**。
4. **Lead 裁决回执（2026-11-02，详见 §10）**：① A2 **契约不改**，实现+测试双错；② 选五全中概率以精确超几何 `0.0645%（约 1/1550 期）` 为准，8.md L190 为**需求原文笔误**，不改 `docs/8.md`，走契约 §7 errata；③ 授权新增 `output/state/prev_prediction.json`（修 H5 前提）。

---

## 1. High 问题（6 项，必须修）

| # | 级别 | 位置（文件:行） | 问题与证据 | 修复建议 | Owner |
| :-- | :-- | :--- | :--- | :--- | :--- |
| H1 | High | `kl8/layer4_scoring/adaptive.py:71-88,113`；`kl8/layer4_scoring/__init__.py:73-80`；`kl8/layer4_scoring/hybrid.py:101-141` | **R33 自适应规则权重是假实现 + B5 归一化缺失，导致核心打分公式失真。** ① `rule_hit_rates()` 无论入参如何都 `return np.full(5, 0.25)`（第 83、87 行），`next_weights()` 的 `rates_map` 亦写死 `{k: 0.25}`（第 113 行）→ 自适应永远输出**均匀 0.2**，8.md L147–148「回看 M=8 期各规则实际命中率 + Softmax 动态调权」完全落空。② `__init__.py:76-78` 无自适应态时用的兜底权重 `{"R1":1.0,"R2":0.8,"R3":0.8,"R4":0.4,"R5":1.8}` 是**硬编码且 Σ≠1**（=4.8），违反 C1 与契约 B5（`Σw_r=1`）。③ B5 要求 `FinalScore = Base + λ·Σ(w_r/w̄)·pass_r`（`w̄=1/5`，`contract.md:1154`），实现 `hybrid.final_score` 直接 `base + λ·weighted_votes`，**漏掉除以 w̄**；均匀权重下本应恒等于 8.md 原式 `Base+λ·V`，实际把票数加成**缩小 5 倍**。④ **实测自证**：交付件 `output/reports/Keno预测精要_20261102_第2026292期.txt:45` 09 号显示"5票"，同文件 `:118`（附录 A）`BaseScore=0.9180 → FinalScore=0.9680`，差 **0.0500**；按 8.md L144 应为 `λ·V=0.05×5=0.2500`。⑤ 交付件"得票"展示用未加权 `votes`、引擎用 `weighted_votes` → 同时违反契约 **B8（报告单一数据源）**。 | 真算各规则历史窗口命中率（按钮单/池真实命中），`Σw=1` 走校验；`final_score` 按 B5 除 `w̄`；`score_round` 与报告统一用同一 `weighted_votes`；补"均匀权重恒等于 λ·V"单测（`tests/test_scoring.py` 现有用例未覆盖自适应路径）。 | 码小全 |
| H2 | High | `kl8/layer4_scoring/filter.py:176-219`；死函数 `:134-149`、`:88-131` | **R34 空间形态硬约束根本没执行。** `build_pool()` 只在 `is_balanced` 失败时**写告警**，并把 `relax_order` 的**全部 4 项**无差别塞进 `relaxed`（`:214-215`），既不重选也不调用 `enforce_constraints()`（`:134`）或 `greedy_pool()`（`:88`）——两者全仓 **0 调用（死代码）**，而契约 R34 明确把这两个函数列为实现点（`contract.md:1136`）。**实测自证**：交付件 `output/reports/...第2026292期.txt:75` 原文「池选告警: 形态约束放宽 [same_tail, parity, zone4, repeat_ratio]: **第 3 区号码占比 50.0% > 上限 45.0%**」——上线交付的 12 码池**确实违反 8.md L152 硬约束**，且"放宽留痕"是假的（4 项全列，未按 B7 逐项有序放宽并重算）。 | `build_pool` 改为：按 FinalScore 贪心 + `is_balanced` 硬约束选满；冲突时按 `same_tail→parity→zone4→repeat_ratio` **逐项**放宽并只记真实放宽项；全放宽仍不达标抛 `InfeasibleConstraintError`；删除或启用死函数；补"违反 45% 时必须换码"的真断言单测。 | 码小全 |
| H3 | High | `kl8/evaluation/metrics.py:203-227` | **R45 规则归因是纯造假常量。** `rule_attribution()` 对 R1–R5 一律返回 `pool_size_avg=15.0, hits=int(n*3.75), hit_rate=0.25, lift=1.0, z=0.0`；`high_votes={hit_rate:0.28, lift:1.12, z:2.1}`、`low_votes={0.22,0.88,-2.1}` 全部与 `rows` 无关（仅把 `n` 塞进去）。8.md L191–192 要求"分别统计 R1~R5 单一规则候选池独立胜率，以及 V≥3 高票组相对低票组的超额命中率，作为**阈值调优唯一依据**"——该能力现在为零。当前 Excel/TXT 尚未展示该函数（`excel.py` 仅 Lift 表），但契约 R45 列为必需交付，**且展示即等于向用户输出虚构 z=2.1**。 | 按 `BacktestRow` 里的 `votes` 与 `hit_at` 真算：各规则的 `pass` 掩码只能从逐期 `verdicts` 取（需在 `BacktestRow` 增存 rule mask 或由 `score_round` 回传），再算真实命中率/lift/两比例 z；`tests/test_metrics.py:141-162` 必须断言数值而非 `"n" in dict`。 | 码小全 |
| H4 | High | `kl8/evaluation/metrics.py:289-295`；`kl8/evaluation/walkforward.py:144`；`main.py:315-320`；`tests/test_walkforward.py:72-73` | **"防穿越自检"恒真，audit 模式是伪机检。** `self_check()` 里 `"no_lookahead": True` **字面量硬编码**（`:295`），`lift_recalculated` 只看 `len(lift)>0`、`bankroll_recalculated` 只比期数（`:293-294`），没有任何重算可比。`main.py --mode audit` 把它打印成「回测自洽性机检项」并通过（我方实测 exit 0、四项全 True）；`tests/test_walkforward.py:73` 又用 `assertTrue(self_check["no_lookahead"])` 反向锁死。**也就是说：一个"审计通过"的绿灯，其实什么都没审。** | `no_lookahead` 改为由 `walkforward` 真实累计（每期 `assert_no_lookahead` 通过计数、越界计数为 0 才 True）；`lift_recalculated` 用独立重算分支与报告值比对（1e-9）；补一条"故意注入越界访问 → self_check 必为 False"的反证测试。 | 码小全 |
| H5 | High | `kl8/pipeline.py:218-226`；`kl8/reporting/text_report.py:98-101`；交付件 `output/reports/...第2026292期.txt:10` | **交付件「前期预测复盘」统计口径错误，向用户输出错误事实。** `previous_review["hit_count"] = len(set(score_round_res.pool.gold) & last_d.numbers)` —— 这是"**当期目标期的金胆池** ∩ **最新一期开奖号**"（即重号数），被 `text_report.py:101` 渲染成「**前期预测复盘 (第 2026291 期): 命中 X 码**」，语义应为"上一期给出的预测号码命中几个"。交付件实文：`* 前期预测复盘 (第 2026291 期): 命中 0 码`。用户会据此认为"上期预测全错"，而真实上期预测命中数从未被计算，属**报告口径不诚实**（产品红线「命中率如实披露」）。 | 复盘必须取"上期输出（上期 `gold/main/top-picks`）∩ 上期开奖"。**【Lead 裁决回执 2026-11-02】已授权新增 `output/state/prev_prediction.json`**（`schema_version` + 原子写 `os.replace` + 按 period 幂等 upsert），由智小研补契约 §3.14 / config-schema；无上期记录时明确写"无上期预测记录"。 | 码小全 + 智小研（契约） |
| H6 | High | `kl8/layer5_combinatorial/tickets.py:176-182`；`kl8/reporting/text_report.py:212`；`kl8/reporting/excel.py:245`；`kl8/layer3_features/ml.py:292`；配置死项 `kl8/config.py` `ml.refit_every` / `covering.max_pair_overlap` / `backtest.t0_min` | **半成品伪代码 + 死配置（红线 11 / C1）。** ① `TicketPlan.guarantee_level` 只有 `t/m/text/coverage_ratio/verified`，**没有 `design` 键**，于是交付件 TXT 第 81、101 行与 Excel 第 245 行直接输出占位符「**理论保底设计: C(v,k,t)**」/「保底设计」——上线报告里出现未完成占位符。② `ml.py:292` `cal_prob = pav_calibrate(raw_prob, y_curr if False else raw_prob)  # 保持分布`：`if False` 死分支、`cal_prob` 计算后**从未使用**（下一行直接用 `raw_prob`）→ PAV 校准是假的。③`refit_every`（config 零引用，重训间隔未实现）、`max_pair_overlap`（注单正交上限未实现，R41）、`t0_min`（未使用）为死配置项。④ 死函数：`filter.greedy_pool`/`enforce_constraints`、`cooccurrence.lift_matrix`、`borda.top_percentile_mask`、`tickets.plan_cost` 全仓 0 调用。 | 补真实 `design` 描述（`C(v,k,t,m)`+注数，来自 `CoveringReport`）；删掉 `if False` 死分支，要么真做 PAV 并用 `cal_prob`，要么删除该函数与契约条目；三个死配置项要么实现要么删除（禁止"有 schema 无实现"）；死函数删除或接入。 | 码小全 |

---

## 2. Medium 问题（15 项）

| # | 级别 | 位置（文件:行） | 问题与证据 | 修复建议 | Owner |
| :-- | :-- | :--- | :--- | :--- | :--- |
| M1 | Medium | `kl8/layer4_scoring/hybrid.py:8,52-61` vs `contract.md:1124`（A2） | **A2 契约偏离被就地"改判"并写进测试。** 契约 A2：`C̄o=mean(R(Co25),R(Co50),R(CoALL))`（不含 10G）；实现 `co_aggregate` 为 `0.20·Co25+0.20·Co50+0.60·CoALL`，docstring 第 8 行还写着 `/3.0`（自相矛盾）。更严重：新增 `tests/test_qa_numerics.py:411-420` 把 **0.20/0.20/0.60 标成「裁决 A2」** 断言——等于用测试把偏离固化为"契约口径"。 | **【Lead 裁决回执 2026-11-02】契约不改**：`C̄o = mean(R(Co25), R(Co50), R(CoALL))`（A2 原文）；**归属：实现+测试双错，契约无误**。实现改回等权（删除 0.60 私改），`tests/test_qa_numerics.py:411-420` 标成「裁决 A2」的注释与断言同步改回契约口径。 | 码小全（实现） + 测小严（测试） |
| M2 | Medium | `kl8/layer4_scoring/filter.py:152-173`（`:168`）；`tests/test_filter.py:33-43` | **A5 金胆选取偏离契约。** 契约 A5：`n_gold = clip(round(0.30×len(main)),3,5)`，`main=12` → **4**；并按"先 `V≥3` 按 FinalScore 降序、不足再补"选。实现 `gold_count = max(gold_min, min(gold_max, len(main)))` 恒取 5，且**完全没实现 V≥3 优先**。交付件顶级展示"超级金胆 (5码)"、`main.py:242`/`wechat.py:42` 也都写死 5。测试只断言 `3<=len<=5`，抓不到。 | 按 A5 公式与票数优先实现；报告标题用 `len(gold)` 动态渲染。 | 码小全 |
| M3 | Medium | `kl8/evaluation/metrics.py:230-248`；`kl8/pipeline.py:186-202`；`main.py:299-303`；`kl8/reporting/excel.py:313-320` | **A10「未确认奖级不出金额」被违反，实测输出误导性 ROI。** `prize_table` 中 pick7/pick3 为 `None`，`walkforward` 里 `prize_known=False` 但仍累计成本，`bankroll()` 照算，`main.py` 照打。**我方实测**（`--mode backtest --backtest 20 --k 7`，含注单）：`总投入 378 元 | 总奖金 0 元 | 投资回报率 (ROI): -100.00% | 最长连挂 20 期`——选七奖级未确认却给出 -100% ROI。契约 A10 硬性措辞要求「ROI: N/A（官方奖级未确认）」且"ROI/回撤/连挂仅对选五出具"。 | `bankroll` 与三类报告按 `prize_known` 分支：任一计划奖级未知即 ROI/回撤/连挂输出 `N/A（官方奖级未确认）`，金额只统计已确认奖级。 | 码小全 |
| M4 | Medium | `kl8/evaluation/metrics.py:112-141`；`kl8/config.py` `backtest.lift_signal/alpha_sig`；交付件若跑默认 daily（`pipeline.py:79` `backtest_n=30`） | **小样本"★显著"是假阳性制造机，与 8.md L188 冲突。** 8.md：显著性结论要求"**长期大样本（如 500 期以上）**"；实现只要 `lift>1.05 且 p<0.05` 就标 `★显著`，无最小样本量门槛、无多重比较校正（同时检 K=3/5/7/10/12 五组）。**我方实测 30 期 no-tickets**：Lift=1.289/1.280/1.276/1.187/1.178，p=0.0658/0.0330/0.0082/0.0417/0.0498 → **4 组被判"★显著"**；而 Lead 提供的同一数据 100 期跑法为 1.027~1.072、p 全>0.05。同一特征体系 30 期"显著"、100 期"不显著"，是典型小样本噪声。默认 daily 恰恰就是 30 期。 | `is_signal` 增加 `n_sample ≥ 500` 门槛（或输出"样本不足，仅描述"）；多重比较用 Bonferroni/`p×len(ks)`；TXT 里注明样本量与判据。 | 码小全 + 智小研（契约口径） |
| M5 | Medium | `kl8/reporting/text_report.py:113-114,186,196`；`kl8/reporting/wechat.py:42,76`；`kl8/reporting/text_report.py:33-38`（对比 `kl8/layer4_scoring/rules.py`） | **报告硬编码口径与规则描述错误。** ① `p0 = 20/80 = 25.00%`、`p_co = 6.01%`、`E(Repeat)=5.00`、"合理区间 150~250"、"核心超级金胆 (5 码)" 全部写死在渲染层（应取 `constants.py`/`len(gold)`，C1+B8）。② 段落 3 的规则说明与 `rules.py` 实现和 8.md L133–137 **完全对不上**：交付件写 `R1: 强连共振码 (Markov/近期连续同现)`、`R2: 热度龙头码`、`R3: 跨期互补码`、`R4: 强伴随防撞码`、`R5: 全局冷热反弹码`；真实 R1=多尺度跟随共振、R2=温热多头、R3=动量金叉、R4=强共现伴生、R5=全量均值回归。用户看到的规则语义是错的。 | 数字全部改从 config/constants/实际对象取；规则中文名与说明改为从 `rules.py` 单一数据源导出（B8）。 | 码小全 |
| M6 | Medium | `kl8/layer2_baseline/randomness.py:39-58,133-149`；交付件 `...txt:17-22` | **NIST-Lite 方法误用，且交付件自相矛盾。** `flatten_history()` 把 (T,80) 的**二值开奖矩阵直接摊平**当比特流：每期恰好 20/80 个 1 → 全局 ones=25%，而 NIST Monobit/BlockFrequency/Runs/APE 的前提是**等概率 0/1 独立比特流**。结果 4 项全部 p=0.0000「预警」（Monobit 统计量 205.52 恰等于 `|2·42240-168960|/√168960`），而同屏第 17 行卡方又写「均匀性判定: 通过 (符合均匀随机)」。**这是"必然预警"而非发现非随机**，对用户是误导（会以为快乐8被人为操控）。 | 明确口径：要么对"每期 20 码之和/间隔/号码序列"做适配检验（如按 p0=0.25 的偏差检验），要么在报告中固定注明"本项对 20/80 结构天然不适用，仅作长程稳定性监控"，并与卡方结论合并解释；不得裸列 4 个"预警"。 | 码小全 + 智小研（口径） |
| M7 | Medium | `kl8/layer3_features/graph.py:96-97,133-134,181-182`；`kl8/layer5_combinatorial/covering.py:235-236`；`kl8/evaluation/leakage.py:122-123`；`main.py:28-29`；`kl8/pipeline.py:137-138` | **裸 `except Exception: pass` 共 6 处（红线 2）+ 静默降级不留痕。** graph 三处降级（pagerank/eigenvector/louvain）直接 `pass` 转 NumPy，且 `build_graph_features` 只在"networkx 未安装"时写 warning（`graph.py:233-235`），**装了就抛异常时不写任何 warning、`backend` 仍报 "networkx"** → 违反契约 B3（`contract.md:1152`："不可用/异常 → 降级 + `backend` 记录 + 写 warning"）。covering 的 milp 异常被吞后返回 None（会把 ILP 失败伪装成"超规模"）。leakage grep 审计异常被吞（审计自己失败时静默）。main 的 stdout reconfigure 失败被吞。 | 每处裸 except 改为捕获具体异常 + `warnings.append(中文原因)` + 记录实际 backend；milp 的 `except` 区分"无求解器/超时/数值失败"；审计失败必须显式降级并计入 `code_audit_clean=False`。 | 码小全 |
| M8 | Medium | `kl8/layer5_combinatorial/covering.py:427,435` | **`seed=42` 硬编码，`--seed`/`cfg.seed` 对 L5 无效**（红线 1 + 契约 B10 `cfg.seed` 是唯一随机源）。`greedy_weighted(..., seed=42)`、`anneal(..., seed=42)` 两处写死；`cfg.seed` 传入后在此断链，两次不同 seed 的复现实验在降级路径上结果相同。 | 改为 `seed=cfg.seed` 并补"不同 seed 覆盖结果可复现且可按 seed 区分"的单测。 | 码小全 |
| M9 | Medium | `kl8/pipeline.py:133-140` | **时间标准化兜底硬编码（红线 6）**：`try: ... except Exception: eff_target_date = "2026-11-02"`，日期解析失败时静默使用写死的项目基准日，交付件日期/文件名全部错位且无告警。 | 解析失败应抛 `DataFormatError`（或取 `latest_draw.date` 并写 warning），禁止写死日期。 | 码小全 |
| M10 | Medium | `kl8/evaluation/metrics.py:248`；`main.py:303`；`kl8/reporting/text_report.py:257`；`kl8/reporting/excel.py:320` | **金额/ROI 浮点（红线 3 / 契约 C3）**：`roi = float(net)/float(cost)` 为 float，报告 `roi*100:.2f` 展示；契约 C3 要求"万分比整数/Decimal，杜绝浮点精度丢失"。成本/奖金本身是 int ✓，但 ROI 出口破坏口径。 | ROI 以 `Decimal` 或"万分比整数"（`net*10000//cost`）产出与存储，展示时再格式化。 | 码小全 |
| M11 | Medium | `kl8/layer5_combinatorial/tickets.py:90,135-136`；`kl8/config.py` `covering.max_pair_overlap` | **R41 注单后处理缺项 + 连号阈值 off-by-one。** ① `max_pair_overlap`（任意两注最大重合，契约 R41 明列）**全仓 0 引用**，正交防撞未实现。② 第 90 行 `cons.get("max_run",1) > cfg.max_consecutive_in_ticket + 1`：配置语义是"剔除 3 连以上"（`config.py:325` 注释），但 `max_run=3 > 2+1=3` 为 False → **3 连号被放行**，4 连才剔除。③ 135-136 行 `k = 5 if pick5 else 7`、`t = 4/5` 硬编码双玩法参数，与 `cfg.covering.k/t` 重复且不受配置控制（C1）。 | 实现 `max_pair_overlap`（后处理两两重合上限 + 超限取舍）；连号判据改 `>=` 或改配置语义；k/t 从 config 表（`budget_by_play` 旁增加 `kt_by_play`）读取。 | 码小全 |
| M12 | Medium | `kl8/layer3_features/ml.py:238-246,259,292` | **R23 ML 通道半成品。** ① `cache_key = (end, "fs-v1", str(cfg.seed))` 含 `end`，而回测每期 `end` 都变 → **缓存永不命中**，契约设想的"refit_every=25 期间复用"退化为每期重训。② `refit_every` 零引用（重训间隔未实现）。③ `t_start = max(10, end - train_window + 1)` 中 `10` 魔法下限。④ PAV 校准为死代码（见 H6）。当前默认 `ml.enabled=False` 掩盖了问题，`--ml` 开启即暴露。 | cache key 改为不含 `end`（用"最后重训期 + refit_every"定位缓存）；实现 refit_every；魔法 10 进 config；PAV 二选一（真用或删）。 | 码小全 |
| M13 | Medium | `kl8/reporting/excel.py:405-418`；`kl8/reporting/text_report.py:283-286`；`kl8/reporting/wechat.py:100-103` | **交付件非原子写 + 假成功（红线 8 / 10）。** Excel `wb.save(p)` 直写目标路径，`except PermissionError` 只 `print` 警告，随后**仍返回 `p` 当作已保存**（调用方与用户看到"Excel 已就绪"）；TXT/微信 `open(p,"w")` 直接截断，写一半异常即毁旧件。契约 R48-e 要求"报告覆盖写"但结合幂等应"先临时文件再 `os.replace`"。 | 三类交付件统一"写 `.tmp` → `os.replace`"；Excel 失败必须抛业务异常/返回失败状态，禁止 print 后当成功。 | 码小全 |
| M14 | Medium | `tests/test_filter.py:24-31`；`tests/test_metrics.py:141-162`；`tests/test_covering.py:40-50`；`tests/test_walkforward.py:60-80` | **测试"假绿"4 处（G3 证据质量）。** ① `test_hard_constraints` 名为"硬约束与冲突有序放宽"，实际只断言 `len(main)==12`——**完全不校验 45%/70%/同尾≤3**，H2 就是被它放过的。② `test_rule_attribution` 只断言 `"n" in attrib.high_votes`，H3 的假常量照样绿。③ `test_ilp_min_tickets_and_fallback` 用 `if res is not None:` 条件断言，`None` 时静默跳过。④ `test_walkforward` 断言 `self_check["no_lookahead"]` 为真，而该值恒真（H4）——**循环论证**。 | 四处改为真断言：硬约束逐项校验、归因数值对拍、ILP 失败分支显式断言 `method=="greedy+anneal"`、自检项由真实引擎产出并用"注入越界应失败"反证。 | 测小严 + 码小全 |
| M15 | Medium·文档勘误 | `contract.md:1184`（§6.1-4）；`contract.md:1208`（§6.2）vs `docs/8.md:190`；`tests/test_qa_numerics.py:108-114` | **契约/原文自身存在错误口径，需 Lead 裁决（不能长期两套数）。** ① `contract.md:1184` 把数据保真写成 `draws[-1].period == 2020001`、`draws[0].period == 2026291`（**顺序写反**），而 `tests/test_history.py:59-60` 断言 `draws[0]=2020001`、`draws[-1]=2026291`（正确）。② 选五全中概率：8.md L190 与 `contract.md:1208` 写「0.155%，约 645 期一遇」；但 `C(20,5)/C(80,5)=0.000645`（**0.0645%，约 1550 期一遇**），8.md 的 0.155% 与"645 期"本身互相矛盾。实现取精确值 0.0645%（数学正确），QA 测试却在注释里写「不采用 8.md L190 口径」——等于测试跟最高效力原文打架。 | **【Lead 裁决回执 2026-11-02】定级：文档勘误（非实现缺陷）**。①契约 §6.1-4 期号顺序 → 智小研订正；②选五全中概率以精确超几何 `C(20,5)/C(80,5)=15504/24040016=0.000645=0.0645%（约 1/1550 期）` 为准；8.md L190「0.155%/645 期」为**需求原文笔误**，**不修改 docs/8.md**（保留原样以便追溯），由智小研在契约 §7 errata 留痕，报告/验收报告按精确值披露。 | 智小研（errata） |

---

## 3. Low 问题（14 项，随本轮一并清理）

| # | 位置 | 问题 | 修复建议 |
| :-- | :--- | :--- | :--- |
| L1 | `kl8/config.py:38`；`evaluation/leakage.py:13`；`evaluation/metrics.py:15`；`evaluation/walkforward.py:14,19`；`layer3_features/features.py:11,29,33`；`layer4_scoring/borda.py:15`；`layer4_scoring/filter.py:14,21,22,231`；`layer5_combinatorial/tickets.py:12,18,19`；`pipeline.py:11,12,17,19,21,22,23,31`；`reporting/excel.py:17,20`；`reporting/text_report.py:16,19,21` | **未使用导入 60+ 处**（AST 扫描：`COST_PER_TICKET_YUAN`、`os`、`Any`、`Sequence`、`DEFAULT_PRIZE_TABLE`、`Mapping`、`DataFormatError`、`feature_redundancy`、`get_prime_label`、`N_NUM`、`InfeasibleConstraintError`、`zone_distribution`、`RuleConfig`（函数内）、`math`、`CoveringReport`、`ac_value`、`dataclasses`、`np`、`merge_cli_overrides`、`Dataset`、`FeatureSnapshot`、`InsufficientDataError`、`KL8Error`、`ScoreRoundResult`、`Border`、`Side`、`RandomnessReport` 等）。 | 用 `ruff --select F401` 或等价清理；CI 加静态检查。 |
| L2 | `kl8/layer3_features/ml.py:50,56,63` | `MLCache.get/put/clear` 三个公开方法**缺中文 docstring**（C10 / 红线 12；全仓 AST 扫描仅此 3 处公开符号缺失）。 | 补中文 docstring。 |
| L3 | `evaluation/metrics.py:128,135,141,214`（0.25/1.96）；`layer3_features/morphology.py:151`（40.5）；`features.py:171`（`min(200, n_hist)`）；`layer5_combinatorial/covering.py:398`（`ceil(0.25v)+2`/`>2`）；`layer3_features/graph.py:74`（`alpha=0.85`）；`text_report.py:113-114` | **残留魔法数字**（0.25/1.96 应取 `constants.P0/Z_SIG`；40.5 应 `(N_NUM+1)/2`；200/0.85/0.25/2 应进 config）。 | 收敛到 `constants.py`/`config.py`。 |
| L4 | `kl8/reporting/excel.py:414,417` | 库函数内直接 `print()` 警告（分层：库向 stdout 输出；`pipeline` 的进度已改回调，此处遗漏）。 | 改为返回 warnings 或注入 logger。 |
| L5 | `kl8/reporting/excel.py:255` | 单注成本写死 `2`（应 `cfg.cost_per_ticket`）。 | 取配置。 |
| L6 | `layer4_scoring/filter.py:251`（`m1..m5`）；`main.py:219`（`bt_n`）；`layer3_features/graph.py:237`（`pr`）；`evaluation/walkforward.py:118`（`run_cfg`）；`covering.py`（`g_res`/`cov_ratio`） | 模糊命名/单字母缩写（红线 12，阿里规约）。 | 改名（`masks`/`backtest_n`/`pagerank_scores`…）。 |
| L7 | `layer3_features/cooccurrence.py` `triad_counts(top_n=20)`；`layer5_combinatorial/covering.py:438-441` | 参数未使用（`top_n`）、预算截断时只告警不重验保底（截断后 `verify_guarantee` 仍会跑，OK，但截断可能破坏保底而不阻断——建议截断后若 `verified=False` 抛错）。 | 删除无用参数；截断后保底不成立必须阻断。 |
| L8 | `layer5_combinatorial/covering.py:74-110` | `verify_guarantee` 无内部规模上限（仅靠 `solve()` 前置 `max_m_subsets` 守卫）——防御性不足。 | 函数内加 `C(v,m)>cfg.max_m_subsets` 断言或参数。 |
| L9 | `covering.py:346-367`（anneal 每轮全量重算覆盖）；`features.py` 每期重建矩阵/协方差 | 降级路径性能反模式（6000 组合 × 2000 迭代）；单期 0.39s 达标但余量小。 | 增量覆盖集维护；矩阵复用。 |
| L10 | `tests/test_qa_release.py:278` | 断言 `plan.cost_yuan == 2 * len(plan.tickets)` 把成本 2 写死进测试，配置改价即假绿。 | 断言 `== cfg.cost_per_ticket * len(...)`。 |
| L11 | `tools/build_covering_tables.py` 全文 | 工具无 `--output`/`--specs` 参数、无单测、`print` 直出（可接受但未纳入 G3 覆盖）。 | 加 argparse + 一个冒烟测试。 |
| L12 | `kl8/config.py` `BacktestConfig.t0_min`；`ml.refit_every`；`covering.max_pair_overlap` | 死配置项（"有 schema 无实现"，与 M12/H6 同源，此处记为清理项）。 | 实现或删除，二选一。 |
| L13 | `kl8/layer3_features/graph.py:219-263` | `GraphResult.backend` 仅在"未安装 networkx"时正确，异常降级不修正（与 M7 同源，此处记 API 一致性）。 | 降级分支同步改 `backend`。 |
| L14 | `kl8/reporting/text_report.py:283-286`；`wechat.py:100-103` | 覆盖写无 `.bak`（历史交付件被直接覆盖，无回溯能力）。 | 覆盖前留一份 `.bak` 或按日期归档。 |

---

## 4. 12 大红线逐条判定（task-8 派单口径，Python 映射）

| # | 红线 | 判定 | 关键证据 |
| :-- | :--- | :--- | :--- |
| 1 | 零魔法数字 | **❌ 违规** | H1（`layer4/__init__.py:76-78` 权重 Σ=4.8）、H6、M3、M8（`seed=42`）、M11（k/t 5/4/7/5、连号 `+1`）、M12（`max(10,…)`）、L3、L5（excel 成本 2） |
| 2 | 零裸 except / 零 silent pass | **❌ 违规** | M7：6 处 `except Exception: pass`（`graph.py:96,133,181`、`covering.py:235`、`leakage.py:122`、`main.py:28`）+ 4 处静默降级不留痕（`graph.backend` 异常时不改、`covering` milp、`pipeline` 日期） |
| 3 | 零浮点金额 | **❌ 违规** | M10 `metrics.py:248` `roi=float/float`，报告 `roi*100:.2f`；契约 C3 要求万分比整数/Decimal。成本/奖金 `int` ✓ |
| 4 | 循环与组合规模封顶 | **✅ 通过** | `covering.py:386-390` `C(v,m)≤max_m_subsets=20万`；`v≤max_pool_v=20`；`budget≤max_budget=200`；ILP 有 `candidate_cap/target_cap`；`combinations` 均在守卫之后（L8 建议加函数内自防御） |
| 5 | 除零与空安全 | ⚠️ 部分违规 | 多数分母已守卫（`state.window_sum`、`shannon_entropy`、`mi_lag1`、`lift_at_k`、`bankroll.roi`、`high_tier_rate n==0` ✓）；但 M9 静默兜底、M7 吞异常后继续、`tickets.py` 池为空/预算 0 的边界无用例 |
| 6 | 时间标准化 | **❌ 违规** | M9 `pipeline.py:137-138` `except Exception: eff_target_date="2026-11-02"`（写死日期且无告警）；其余 `YYYY-MM-DD` + 期号 `int` ✓ |
| 7 | 零循环内重计算 | **❌ 违规** | M12 ML 缓存 key 含 `end` → 回测每期重训（`refit_every` 未实现）；`filter.py:236-251` 在 `build_pool` 内重算 `rank_matrix`+`base_score`+`evaluate_rules`（`score_round` 已算过一遍，B8 口径分裂风险）；L9 `anneal._eval` 每轮全量重算覆盖；`features.py` 每期重建矩阵 |
| 8 | 幂等防重 | ⚠️ 部分违规 | `adaptive.save_state` 原子 `os.replace` 幂等 ✓、`history.append_draws` 重复期不追加 ✓、报告覆盖写 ✓；但覆盖写非原子 + Excel 失败仍返回成功路径（M13） |
| 9 | 确定性 | **❌ 违规** | M8 `covering.py:427,435` 写死 `seed=42`，`cfg.seed` 在 L5 断链；其余随机源（Louvain/ML/抽样）均取 `cfg.seed` ✓ |
| 10 | 资源释放 | ⚠️ 部分违规 | `sync` `with urlopen`+超时 ✓、`history` 原子写 ✓；但 Excel `wb.save` 直写、TXT/微信 `open("w")` 截断、无 `.bak`（M13/L14） |
| 11 | 零死代码 / 零半成品 | **❌ 违规（最重）** | H6：`guarantee_level` 缺 `design` → 交付件输出占位符 `C(v,k,t)`；`ml.py:292` `if False` 死分支 + `cal_prob` 未用；死函数 `greedy_pool`/`enforce_constraints`/`lift_matrix`/`top_percentile_mask`/`plan_cost`；死配置 `refit_every`/`max_pair_overlap`/`t0_min`；H1/H3/H4 三处假实现 |
| 12 | 中文文档与命名 | ⚠️ 部分违规 | 全仓中文注释/docstring 覆盖率很高（AST 扫描仅 `ml.py:50/56/63` 三个公开方法缺失，L2）；命名缩写见 L6（`m1..m5`/`bt_n`/`pr`/`g_res`） |

---

## 5. Lead 派单 5 项专项复核

| 专项 | 结论 | 证据 |
| :--- | :--- | :--- |
| 五层单向依赖 | **✅ 通过** | `tests/test_architecture.py` 用 AST 校验层间 import 方向；我复核 `layer1←layer2←layer3←layer4←layer5` 无反向引用；`layer5/tickets.py` 仅下行引用 `layer3_features.morphology`（允许）；无跨层循环依赖。 |
| 泄漏仅经 `history.truncated` | **⚠️ 结构通过 / 测试不足** | 全仓 `ds.draws[:cut]` 只在 `kl8/history.py::truncated`；`walkforward.py:144` 有索引守卫；`state.py` 的切片由 `end` 约束（layer1 内部，允许）。但契约 §6.3 要求的①指纹等价性②未来投毒后 `snapshot/score_round/pool.main/注单` 逐位不变，**均无测试**（`tests/test_leakage.py` 只测 `poison_future` 本身）→ M14 范畴。 |
| 真实测试断言 | **⚠️ 部分** | 175 用例、401+ 断言、无 `assertTrue(True)`；但存在 4 处弱断言/条件断言/循环论证（M14），且 R33/R34/R45/`self_check` 四个"假实现"**零覆盖**。 |
| 报告口径诚实性 | **❌ 不通过** | H5 复盘口径错误（交付件"命中 0 码"）；M6 NIST 四项"预警" vs "均匀性通过"；M4 小样本"★显著"；H6 占位符；M5 规则描述错误。 |
| 43/48 需求点覆盖 | **48 项为准（"43"为早期笔误，`contract.md:3`）**；`requirement-mapping.csv` 实测 48 行；抽样 10 点 9 通过 1 偏离；全表 3 项落空（R33/R34/R45）+ 3 项部分（R23/R41/R44） | 见 §6 |

## 6. 需求抽样对照（10 点，按 Lead 指定）

| 需求 | 原文（8.md） | 实现 | 判定 |
| :-- | :--- | :--- | :--- |
| R13 遗漏暂停/压力 | L90–92 | `heat.py`：`gap`、`μ_gap=n/c`、高斯核 `exp(-0.5((O/μ-1.2)/0.6)²)`，峰值/半峰手算对拍通过 | ✅ |
| R14 拉普拉斯平滑 | L97–98 | `markov.py:transition_matrix` `(n_ij+αp0)/(c_i+α)`，α 取自 config，手算对拍通过 | ✅ |
| R25 R1 强跟随共振 | L133 | `rules.py::rule_r1`：4 窗口 ≥3 个 `R≥0.65`，阈值全取自 `cfg.rules` | ✅ |
| R26 R2 温热多头 | L134 | `r2_base_freq=0.24`、`f10∈[0.20,0.50]` 与原文一致 | ✅ |
| R27 R3 动量金叉 | L135 | `f10≥f25≥f50` 且 `R(follow_10)≥0.60` | ✅ |
| R28 R4 共现伴生 | L136 | `0.5R(Co25)+0.5R(Co50)≥0.75` | ✅ |
| R29 R5 临界回补 | L137 | `Pressure≥0.60`、`gap≤12`、`R(follow_all)≥0.70` | ✅ |
| R37 覆盖设计 C(v,k,t,m) | L161–166 | `covering.solve`：查表→ILP→贪心+SA，`verify_guarantee` 对每条路径穷举校验（`covering.py:444`），`m_target` 与 A7 五组验算一致 | ✅ |
| R43 Lift@K 显著性 | L185–188 | 公式/CI 实现正确；但"**长期大样本（500 期以上）**"门槛缺失，30 期即判"★显著" | ⚠️ 偏离（M4） |
| R47 防穿越 | L198–201 | 转移对 τ≤t、`X_t` 预测 `X_{t+1}`、`truncated` 唯一入口、索引守卫齐全；缺不变性测试 | ⚠️ 结构✓/测试✗ |

> 全表层面：**R33（自适应）落空、R34（硬约束）落空、R45（归因）造假**为三处硬缺口；R23（ML）半成品、R41（后处理）缺 `max_pair_overlap`、R44（高奖级）仅 pick5 可算为部分落空。**因此"R1–R48 全落地"（提交 `251e795` 标题）与实际不符。**

---

## 7. 我方独立动态证据（命令 + exit code，均为本次审查实跑）

| # | 命令 | 结果 |
| :-- | :--- | :--- |
| 1 | `python -m compileall -q kl8 main.py tools` | **COMPILE_EXIT=0** |
| 2 | `python -m unittest discover -s tests -b`（含 2 个未提交 QA 文件） | **Ran 175 tests in 14.882s / OK / UNITTEST_EXIT=0**（Lead 上报 124 为加 QA 文件前口径；当前工作区 32 个测试文件） |
| 3 | `python main.py --no-sync --mode audit --backtest 30` | **AUDIT_EXIT=0**；机检四项 `{sample_count_matches:True, lift_recalculated:True, bankroll_recalculated:True, no_lookahead:True}`；指纹血缘 `1ae96b7084427293` —— **但该项为假自检（H4）** |
| 4 | `python main.py --no-sync --mode backtest --backtest 30 --no-tickets` | 14.3s；Lift 1.289/1.280/1.276/1.187/1.178，p=0.0658/0.0330/0.0082/0.0417/0.0498 → **4/5 标"★显著"**（小样本假阳性，M4） |
| 5 | `python main.py --no-sync --mode backtest --backtest 20 --k 7`（含注单） | 10.6s；**总投入 378 元 / 奖金 0 / ROI -100.00% / 最大回撤 378 / 最长连挂 20 期**（选七奖级未确认却出 ROI，M3） |
| 6 | `.bat` 字节检查 | `EF BB BF` + `chcp 65001` + `python main.py --no-sync --backtest 30 --full-print` ✅ |
| 7 | 交付件实物核对 | `output/reports/Keno预测精要_20261102_第2026292期.txt`：L10 复盘"命中 0 码"、L17 均匀性通过 vs L19–22 四项预警、L45 5 票 vs L118 加成 0.05、L75 违反 45% 约束、L81/L101 `C(v,k,t)` 占位 |

> 未复跑项（标注"未验证"，不代 Lead 补绿）：Lead 上报的 100 期回测数字（1.027~1.072，p 全>0.05）；500 期性能预算；`--ml` 开启后的端到端耗时（预估值 2.03s 未实测）。

---

## 8. Lead 4 项「已修复」声明复核

| # | 声明 | 判定 | 证据 |
| :-- | :--- | :--- | :--- |
| 1 | `m_margin` 默认改 1、合法区间 `[1,3]` | **✅ 已修复** | `config.py:281`（默认 1）、`config.py:750`（`_check_int(...,1,3)`）；`tests/test_qa_release.py:115-134` 实测 0/4 抛 `ConfigError`、1/3 通过 |
| 2 | ML 默认 `False` + `--ml` 显式开启 | **✅ 已修复** | `config.py` `MLConfig.enabled=False`；`main.py:137` 帮助文案；`tests/test_qa_release.py:162-199` 实测默认 False、`--ml` 后 True |
| 3 | 进度输出（8 步 + 回测进度回调） | **✅ 已修复** | `pipeline.run_daily/run_backtest` 用 `progress` 回调、库内不 print；`tests/test_qa_release.py:205-257` 实测回调次数=期数、daily 覆盖 `[1/8]..[8/8]`；唯一例外 `excel.py:414/417`（L4） |
| 4 | `.bat` 重写 UTF-8 BOM | **✅ 已修复** | 首 3 字节 `EF BB BF`、`chcp 65001`、`--no-sync --backtest 30 --full-print`；`tests/test_qa_release.py:292-302` 覆盖 |

---

## 9. 待修复清单（按 Owner 汇总，供 Lead 分派）

**码小全（代码，14 项）**：H1 H2 H3 H4 H5 H6；M1（A2 改回等权）、M2 M3 M4(实现侧) M5 M6(实现侧) M7 M8 M9 M10 M11 M12 M13；M14（配合测小严）；L1–L14。
**智小研（契约/口径，4 项）**：M4（显著性样本量口径）、M6（NIST 口径）、M15（契约 §6.1/§6.2 笔误 + §7 errata 记录选五全中概率，不改 8.md）、H5（`prev_prediction.json` 补 §3.14 / config-schema）。
**测小严（测试，2 项）**：M14（4 处弱断言）、补 R33/R34/R45/`self_check` 反证用例；L10。
**Lead（裁决，已闭环，3 项）**：A2 契约不改（§10.1-①）、选五全中概率文档勘误（§10.1-②）、H5 新文件授权（§10.1-③）。

### 复审口径（修复后请直接按此复跑）
1. 重新提交新 diff（`git diff d9bc139..HEAD` + 工作区），我方基于**新 diff** 复审，不复用本卡结论。
2. 必须新增/替换的验证证据：`weighted_votes` 除 `w̄` 后的期望手算（均匀权重 = λ·V）；12 码池强制满足 45%/70%/同尾≤3；`rule_attribution` 真实数值对拍；`self_check` 注入越界反证；选七 ROI 输出 `N/A`；`compileall` + 175+ 用例 + audit + 单期/30 期实测 exit code。
3. 本卡 High/Medium 未清零前，**G4 不放行**，release-ops 不得出验收报告。

---

## 10. Lead 裁决回执与复审执行清单（2026-11-02）

### 10.1 三项裁决（已并入正文对应行）

| # | 裁决（Lead 原文要点） | 卡内落点 |
| :-- | :--- | :--- |
| ① | **A2 不改契约**：`C̄o = mean(R(Co25), R(Co50), R(CoALL))`（先秩后均、不含 10G）。实现 0.20/0.20/0.60 属实现偏离，按缺陷修回；`tests/test_qa_numerics.py` 中标成「裁决 A2」的断言是错的，测小严改回契约口径。**归属：实现+测试双错，契约无误。** | M1（已加回执行） |
| ② | **选五全中概率以精确超几何为准**：`P=C(20,5)/C(80,5)=15504/24040016=0.000645=0.0645%（约 1/1550 期）`。8.md L190「0.155%/645 期」为**需求原文笔误**；契约 §7 errata 留痕，**不修改 `docs/8.md`**（保留原样追溯）。**定级：文档勘误（非实现缺陷）。** | M15（已降级为文档勘误，Owner=智小研） |
| ③ | **授权新增 `output/state/prev_prediction.json`**（上期预测持久化：`schema_version` + 原子写 + 按 period 幂等 upsert），作为修 H5 的必要前提；算契约扩展，智小研补契约 §3.14 / config-schema。 | H5（已加授权说明，Owner=码小全+智小研） |

> 判决不变：**G4 维持阻断**。裁决只解决"归谁修/以谁为准"，不降低任何 High/Medium 的严重度。

### 10.2 返工与复审流程（已与 Lead 对齐）

1. 码小全按本卡修 **High 6 项 + Lead 指定 Medium**，修完自报实测输出给 Lead；
2. Lead 转发复审指令给我（附**新 HEAD hash** + 自报实测输出）；
3. 我按下列顺序复跑并在卡内逐条改状态（**已修复 / 部分修复 / 仍存在 / 新增**），旧结论保留可追溯，最后**基于新 diff 重新给结论**：
   - `git rev-parse --short HEAD`、`git diff --stat d9bc139..<新HEAD>`、按缺陷文件 `git diff d9bc139..<新HEAD> -- <file>` 逐行核对
   - `python -m compileall -q kl8 main.py tools`（须 exit 0）
   - `python -m unittest discover -s tests -b`（须全绿，用例数 ≥ 当前 175）
   - `python main.py --no-sync --mode audit --backtest 30`（须 exit 0，且"自检"这次要能失败，见 10.3-③）
   - `python main.py --no-sync --mode backtest --backtest 30 --no-tickets`、`--backtest 20 --k 7`（A10 须输出 N/A）
   - 三类交付件实物核对（TXT/微信/Excel：不再有 `C(v,k,t)` 占位、规则中文名正确、复盘口径正确）
4. **只认新 diff 与实跑证据**，不认"已修"口头声明；自报数字与我实跑不一致时，以我实跑为准并记 Medium。

### 10.3 三项"防复发"验收断言（Lead 指定，复审逐条执行）

1. **H1：得票展示 == 引擎增量。** 随机抽 3 期（建议 `--backtest 3` 的 3 行 + 当日 1 期）：①交付件附录 A 的 80 行逐行断言 `|FinalScore − BaseScore − λ·V| < 1e-4`，且样本覆盖 `V=0..5` 各至少 1 点；②引擎侧 `score_round(...).scores.final / .scores.base / .votes` 同式断言 `< 1e-12`。任何一行不成立 → H1 判"仍未修复"（说明展示与引擎仍两套数，B8 未闭环）。
2. **H2：约束真被执行。** 构造"高分码全落在同一四区"的极端 `scores`（如 1–20 号给 0.9、其余 0.1）调用 `build_pool`：①最终 `pool.main` 必须满足 `zone4 ≤ 45%（12 码 ≤ 5 码）`；②若放宽，`pool.relaxed` 只能含**实际冲突**的约束项，不得把 `same_tail/parity/zone4/repeat_ratio` 四项无差别全列；③所有约束同时不可行时须抛 `InfeasibleConstraintError`，**不得只写 warning 就出单**。
3. **H4：自检能失败（反向证明）。** 故意制造穿越（如 `end` 越过 `t`、或 `poison_future` 后不截断直接喂入）→ 断言 `self_check["no_lookahead"] is False`；正常路径断言 `True`。若注入穿越后仍恒 True → H4 判"未修复"（护栏仍是假的）。

### 10.4 复审台账（新 HEAD 下发后逐格填写）

| 项 | 首轮状态（v1.1） | 复审状态（待填） | 证据/命令 |
| :--- | :--- | :--- | :--- |
| H1–H6 | 未修复（阻断） | — | 见 10.2 / 10.3 |
| M1–M15 | 未修复（M15=文档勘误） | — | 逐文件 diff 核对 |
| 红线 1/2/3/6/7/9/11 | 违规 | — | grep/AST + 实跑 |
| 红线 5/8/10/12 | 部分违规 | — | 交付件原子写实测 |
| 红线 4（规模封顶） | 通过 | — | 保持 |
| 12 红线总判定 / 最终结论 | 7 违规·4 部分·1 通过 / **阻断** | — | 全部清零才可放行 |

### 10.5 未验证项（保持标记，不代补绿）

- `--ml` 开启后**端到端耗时**：未实测（契约 v1.2 自报 ≈2.03s，仅为声明）。
- **500 期回测性能预算**：未实测，Lead 将自行执行并提供数据，我到时据其输出更新 §7。
- Lead 上报的**100 期 Lift/p 数字**（1.027~1.072，p 全>0.05）：我方未复跑，仅作为"30 期假阳性"的对照引用。

---

*本卡仅写入 `docs/dsh-matrix/20261102-kl8-five-layer/review-card.md`；未改任何业务代码、未提交、未删除文件。本卡 v1.1 仅并入 Lead 裁决回执与复审清单，首轮问题状态不变，结论维持**阻断**。*