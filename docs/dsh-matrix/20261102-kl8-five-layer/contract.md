# 快乐8（Keno）五层重构 · 架构契约（contract.md）

> **本契约需求点编号 R1–R48，共 48 项；项目文档与后续任务中的「43 点」为早期笔误，以本表 48 项为准。**
>
> - 版本：**v1.3.0**（2026-11-02，吸收 P8 体检卡暴露的**契约级缺口**：§7.4「8.md L190 概率勘误」、A2 口径冻结、新增 §3.14.1 复盘状态契约 `prev_prediction.json`、R43/R44 显著性硬规则（n≥500 + Bonferroni）、新增 §6.6「报告禁止项（可 grep）」；详见 §7.3）｜ 冻结日：2026-11-02 ｜ 作者：req-architect（智小研）
> - 适用阶段：P2（基础层）→ P3（layer2/3）→ P4（layer4）→ P5（layer5）→ P6（evaluation/reporting/pipeline/main）
> - 权威优先级：**docs/8.md（业务需求，唯一权威）> 本契约（工程落地口径与签名冻结）> 旧实现 `calc_*.py`（仅语义参考，P9 删除）**
> - 冻结效力：**§3 模块签名与 §4 数据结构字段名/顺序为冻结项，下游开发禁止改签名、禁止改字段名、禁止改返回顺序**；确需变更必须由 Lead 发起契约版本升级（v1.x）并同步 P2–P9。
> - 配套文档：[config-schema.md](./config-schema.md)（AppConfig 全字段默认值/范围/校验/CLI 覆盖规则）、[requirement-mapping.csv](./requirement-mapping.csv)（机检用三列映射）

---

## §0 全局硬约束（违反即 High 级缺陷）

| 编号 | 约束 | 落地方式 |
| :--- | :--- | :--- |
| C1 | **零魔法数字**：所有阈值/权重/窗口/上限来自 `kl8/config.py` 或 `kl8/constants.py` | 审大关 grep 硬编码常量；`constants.py` 只放数学恒等式（π 类常量），业务参数一律进 config |
| C2 | **零裸 except / 零 silent pass**：异常必须分类（`kl8/errors.py`）并给中文可操作提示 | 网络类异常内部捕获转 `warnings`；逻辑类异常抛出；禁止 `except: pass` |
| C3 | **零浮点金额**：成本/奖金/ROI/回撤一律整数元（`int`），ROI 以万分比整数或 Decimal 呈现 | `metrics.bankroll` 只用 `int` 累加 |
| C4 | **防穿越唯一入口**：任何"取历史"只能经 `history.truncated(ds, target_period)` | 全仓仅此函数可做时间截断；`evaluation/leakage.assert_no_lookahead` 守卫 |
| C5 | **五层单向依赖**：`layerN` 不得 import `layerN+1`；`layer1` 不得 import `layer3/4/5` | `tests/test_architecture.py` 扫 import 图断言 |
| C6 | **向量化**：80 码计算必须是 numpy 矩阵/向量运算，禁止 `for i in range(80)` 内做矩阵乘法 | 性能预算 §6.5；审大关抽查 |
| C7 | **确定性**：Louvain / 模拟退火 / ML / 抽样必须走 `cfg.seed`；同输入同输出 | 单测同参两次结果逐位相等 |
| C8 | **零除以零**：所有分母带 `cfg.eps` 或显式分支；NaN/Inf 必须在层出口被断言拦截 | `features.validate_snapshot`、`assert_finite` |
| C9 | **索引约定**：`np.ndarray` 用 0-based，索引 `i` ↔ 号码 `i+1`（1..80）；对外展示的号码一律 1-based `int`；`t` 一律 0-based **含**端点 | 所有签名注明 `end`/`t` 语义，见 §0.2 |
| C10 | **中文文档**：模块/类/公开函数/字段必须有中文 docstring；无拼音缩写歧义 | 审大关红线 12 |

### §0.1 原始数据事实（P2 实测，写死为契约）

- `kl8_history.txt`：**新期在前（降序）**，2112 期，首行 `2026291 / 2026-11-01`，末行 `2020001 / 2020-10-28`；行格式 `date:YYYY-MM-DD,period:NNNNNNN,numbers:NN-NN-...`（20 码，两位补零，`-` 分隔）。
- **期号不连续**：实测 6 处跨年自然断档（`2020065→2021001`、`2021351→2022001`、`2022351→2023001`、`2023351→2024001`、`2024352→2025001`、`2025351→2026001`）。
- 因此 `history.load_dataset` **只允许三条硬校验**：① 日期单调不减（升序重排后）；② `period` 全局唯一（**重复即 `DataFormatError`，禁止静默去重**）；③ 降序文件自检 `升序后 ds.draws[-1].period == 文件首行 period`（不成立抛 `DataFormatError`）。期号连续性**只记 warning、不得抛错**。

### §0.2 时间语义（下游据此写单测）

| 记号 | 含义 |
| :--- | :--- |
| `t` / `end` | 0-based、**含端点**的历史最大索引；可用数据为 `X[0..t]`（共 `t+1` 期） |
| `target_period` | 待预测期期号；数据中对应 `draws_by_period[target_period]` |
| 预测语义 | "用 `X_t` 预测 `X_{t+1}`" ⇔ `truncated(ds, t+1)` ⇔ `as_of_index == t` 且 `len(train)==t+1` |
| `truncated(ds, tp)` | **严格早于** `tp` 的全部期（不含 `tp` 本身） |

---

## §1 需求拆解与映射表（R1–R48，共 48 项）

### §1.1 分组与计数自检

| 分组 | 编号区间 | 条数 | 覆盖 8.md 章节 |
| :--- | :--- | ---: | :--- |
| A 数据表征层（layer1） | R1–R4 | 4 | §三.1、§四模块一（L45–58）、§六.2（L201） |
| B 随机性基准与偏态探测（layer2） | R5–R10 | 6 | §二（L17、L22）、§三.2（L32–33）、§四模块二（L61–77） |
| C 多尺度特征工程（layer3） | R11–R23 | 13 | §二（L18–20）、§三.3（L34–35）、§四模块三（L80–118） |
| D 打分与过滤（layer4） | R24–R36 | 13 | §四模块四（L122–155）、§六.3（L202） |
| E 组合覆盖缩水（layer5） | R37–R41 | 5 | §二（L21）、§四模块五（L159–170） |
| F 回测与工程落地 | R42–R48 | 7 | §三（L28–39）、§五（L174–194）、§六（L198–202） |
| **合计** | **R1–R48** | **48** | 8.md 全文 202 行 |

> 机检：`docs/dsh-matrix/20261102-kl8-five-layer/requirement-mapping.csv` 与本表一一对应，行数必须 = 48（P7/P9 用 `Import-Csv` 断言）。

### §1.2 主表（需求 ID ｜ 可验收需求摘要 ｜ 8.md 行号 ｜ 实现模块.函数 ｜ 测试文件::用例名）

| 需求 ID | 可验收需求摘要 | 8.md 行号 | 实现模块.函数 | 测试文件::用例名 |
| :--- | :--- | :--- | :--- | :--- |
| R1 | 二值化状态矩阵 `X∈{0,1}^{T×80}`，每行行和恒为 20，形状/取值/行和强校验 | L30、L47–50 | `kl8/layer1_representation/state.py::BinaryStateMatrix.from_dataset`、`BinaryStateMatrix.row_sum` | `tests/test_state.py::test_binary_matrix_shape_and_row_sum` |
| R2 | 四尺度窗口 `W={10G,25G,50G,ALL}` 的定义、切片与 O(1) 窗口查询 | L34–35、L52–57 | `kl8/layer1_representation/windows.py::resolve_windows`、`state.py::BinaryStateMatrix.window_matrix`、`window_sum` | `tests/test_windows.py::test_four_scale_windows` |
| R3 | 冷启动窗口回退 `w_eff=min(w,T)`，`T<w` 时不报错、方差由平滑抑制 | L57、L201 | `kl8/layer1_representation/windows.py::effective_window`、`is_cold_start` | `tests/test_windows.py::test_cold_start_fallback` |
| R4 | 原始开奖数据清洗/校验/标准化：格式解析、20 码去重、1..80、升序重排、去重期号、降序文件自检、不连续仅 warning | L30–31（+§0.1 实测） | `kl8/history.py::parse_line`、`load_dataset`、`save_dataset` | `tests/test_history.py::test_parse_validate_and_sort` |
| R5 | 单球单期理论先验 `p0=K/N=0.25` | L66、L97 | `kl8/layer2_baseline/priors.py::p0` | `tests/test_priors.py::test_p0_value` |
| R6 | 任意两球同期共现理论概率 `p_co=20·19/(80·79)≈0.060127` | L67、L108 | `priors.py::p_co` | `tests/test_priors.py::test_p_co_value` |
| R7 | 相邻两期理论重号期望 `E(Repeat)=20×0.25=5` | L68 | `priors.py::expected_repeat` | `tests/test_priors.py::test_expected_repeat` |
| R8 | 窗口频次标准化偏离度 `Z_i(w)=(c_i-w·p0)/√(w·p0(1-p0))`，`abs(Z)>1.96` 判显著 | L70–73 | `kl8/layer2_baseline/zscore.py::window_z`、`significant_mask` | `tests/test_zscore.py::test_zscore_formula_and_threshold` |
| R9 | 单号码 0-1 序列 Wald-Wolfowitz 游程检验：R/E[R]/Var/z + 聚簇/无偏/跳跃标签 + `n1=0/n2=0` 退化 | L75–76 | `kl8/layer2_baseline/runs.py::runs_test`、`runs_test_80` | `tests/test_runs.py::test_runs_cluster_vs_jump_labels` |
| R10 | 随机性基准套件：卡方拟合优度 + NIST-lite 4 项 + 80 码香农熵/互信息冗余对（**仅诊断，不进打分**） | L17、L22、L32 | `kl8/layer2_baseline/chisq.py::chi_square_gof`、`randomness.py::nist_lite`、`entropy.py::shannon_entropy`、`feature_redundancy` | `tests/test_chisq.py::test_gof_uniform_and_biased`、`tests/test_randomness.py::test_nist_lite_four_items`、`tests/test_entropy.py::test_entropy_and_redundancy` |
| R11 | 四尺度热力频率 `Freq_i(w)=c_i(w)/w` | L85 | `kl8/layer3_features/heat.py::multi_scale_freq` | `tests/test_heat.py::test_multi_scale_freq` |
| R12 | 跨周期动量差 `Mom_10-25`、`Mom_25-50` 与"动量金叉加速"标签 | L86–89 | `heat.py::momentum`、`golden_cross` | `tests/test_heat.py::test_momentum_and_golden_cross` |
| R13 | 当前遗漏 `O_t,i` 与历史平均遗漏 `μ_gap,i`（μ=0 兜底、截止窗口） | L90–91 | `heat.py::gap_series`、`historical_mean_gap` | `tests/test_heat.py::test_gap_and_mean_gap_floor` |
| R14 | 高斯核遗漏回补压力 `Pressure=exp(-0.5((O/μ-1.2)/0.6)^2)`，端点 clip | L91–92 | `heat.py::omission_pressure` | `tests/test_heat.py::test_pressure_kernel_peak_at_1_2x` |
| R15 | 一阶拉普拉斯平滑条件转移概率 `M^(1)(w)`（α∈[0.5,2.0]、先验 p0） | L96–98 | `kl8/layer3_features/markov.py::transition_matrix` | `tests/test_markov.py::test_alpha_smoothing_handcalc` |
| R16 | 四尺度中心化跟随聚合 `Follow_j(w)=Σ_{i∈S_t}(M_{i→j}-p0)` | L99–101 | `markov.py::follow_score`、`follow_score_all_windows` | `tests/test_markov.py::test_follow_score_centered_dot` |
| R17 | 重号（对角）惯性分与异号跟随分拆分 | L102 | `markov.py::follow_split` | `tests/test_markov.py::test_follow_diag_offdiag_split` |
| R18 | 二阶滞后矩阵 `M^(2)`（隔期回补/斜跳特征） | L102 | `markov.py::transition_matrix_order2` | `tests/test_markov.py::test_second_order_transition` |
| R19 | 中心化共现矩阵 `A_ij(w)=C_ij(w)/w-p_co`（去对角）+ 三元共现团计数 | L19、L107–108 | `kl8/layer3_features/cooccurrence.py::centered_cooccurrence`、`triad_counts` | `tests/test_cooccurrence.py::test_centered_matrix_and_triads` |
| R20 | 当期伴生拉动分 `CoScore_j(w)=Σ_{i∈S_t,i≠j}A_ij(w)` | L109–111 | `cooccurrence.py::co_score`、`co_score_all_windows` | `tests/test_cooccurrence.py::test_co_score` |
| R21 | 共现图网络：正值加权无向图 → PageRank + 特征向量中心度 + Louvain 社区（归属/规模/簇内强度）+ 伴生码 top-k + 降级后端 | L19、L112 | `kl8/layer3_features/graph.py::build_graph_features`、`pagerank`、`eigenvector_centrality`、`louvain_communities`、`companion_topk` | `tests/test_graph.py::test_pagerank_eigen_and_communities` |
| R22 | 宏观空间形态约束特征：四区/八区分布、奇偶比、质合比、大中小比、连号组（2/3 连）、同尾组、和值（期望 810）、AC 值 | L114–118 | `kl8/layer3_features/morphology.py::morphology_stats`、`ac_value`、`sum_deviation_z`、`is_balanced` | `tests/test_morphology.py::test_morphology_stats_full_matrix` |
| R23 | GBDT 非线性概率校准 + PAV 保序回归 + FeatureSnapshot 一次性装配（默认 `ml.weight=0` 仅诊断） | L20、L37 | `kl8/layer3_features/ml.py::train_predict`、`pav_calibrate`、`extract_feature_matrix`、`features.py::build_feature_snapshot` | `tests/test_ml.py::test_ml_prob_bounded_and_cache`、`tests/test_ml.py::test_build_snapshot_schema` |
| R24 | Borda 百分位秩归一化 `R(v_i)=Rank(v_i)/80∈(0,1]`（并列取平均秩） | L126–129 | `kl8/layer4_scoring/borda.py::percentile_rank`、`rank_matrix` | `tests/test_scoring.py::test_percentile_rank_ties_and_range` |
| R25 | 规则 R1 多尺度强跟随共振：4 窗口中 ≥3 个 `R(Follow)≥0.65` | L133 | `kl8/layer4_scoring/rules.py::rule_r1` | `tests/test_rules.py::test_r1_boundary` |
| R26 | 规则 R2 温热多头：`F25≥0.24` 且 `F50≥0.24` 且 `F10∈[0.20,0.50]` | L134 | `rules.py::rule_r2` | `tests/test_rules.py::test_r2_boundary` |
| R27 | 规则 R3 动量金叉：`F10≥F25≥F50` 且 `R(Follow_10G)≥0.60` | L135 | `rules.py::rule_r3` | `tests/test_rules.py::test_r3_boundary` |
| R28 | 规则 R4 强共现伴生：`0.5R(Co25)+0.5R(Co50)≥0.75` | L136 | `rules.py::rule_r4` | `tests/test_rules.py::test_r4_boundary` |
| R29 | 规则 R5 全量回归临界回补：`Pressure≥0.60` 且 `O≤12` 且 `R(Follow_ALL)≥0.70` | L137 | `rules.py::rule_r5` | `tests/test_rules.py::test_r5_boundary` |
| R30 | 交集共振票数 `V_i∈{0,1,2,3,4,5}` | L139 | `rules.py::evaluate_rules`、`vote` | `tests/test_rules.py::test_vote_count_range` |
| R31 | `BaseScore` 五维加权融合（4 窗口 Follow + C̄o + Trend + Pressure，`Σω=1`） | L143 | `kl8/layer4_scoring/hybrid.py::base_score`、`trend_score`、`co_aggregate` | `tests/test_scoring.py::test_base_score_decomposition` |
| R32 | `FinalScore=BaseScore+λ·V_i`（λ∈[0.05,0.10]），输出降序排序与分解明细 | L144–145 | `hybrid.py::final_score`、`score_all` | `tests/test_scoring.py::test_final_score_identity` |
| R33 | 在线自适应 Softmax 规则权重更新（回看 M=8 期各规则命中率，Laplace 平滑、样本不足回均匀先验） | L147–148 | `kl8/layer4_scoring/adaptive.py::next_weights`、`softmax_weights`、`load_state`、`save_state` | `tests/test_adaptive.py::test_softmax_weights_sum_and_prior` |
| R34 | 空间形态均衡过滤硬约束：单四区 ≤45%、奇/偶单边 ≤70%、同尾 ≤3，冲突按序放宽并留痕 | L150–154 | `kl8/layer4_scoring/filter.py::enforce_constraints`、`greedy_pool` | `tests/test_filter.py::test_hard_constraints` |
| R35 | 输出核心金胆池（3~5 码）与优选复式池（10~15 码，默认 12） | L155 | `filter.py::select_gold`、`build_pool` | `tests/test_filter.py::test_gold_and_main_sizes` |
| R36 | 重号/新号配比控制：上期重号占比 25%~35%（默认 30%） | L202 | `filter.py::apply_repeat_quota` | `tests/test_filter.py::test_repeat_quota` |
| R37 | 覆盖设计 `C(v,k,t,m)` 定义与保底命题穷举验证（任意 m 子集 M 都存在一注 B_j 使 `len(B_j ∩ M) ≥ t`） | L163–166 | `kl8/layer5_combinatorial/covering.py::m_target`、`verify_guarantee` | `tests/test_covering.py::test_verify_guarantee_exhaustive` |
| R38 | 标准旋转矩阵查表映射（经典参数最优区组表，生成后必须重验，禁止照抄未核验数值） | L169 | `covering.py::lookup_table`、`tools/build_covering_tables.py` | `tests/test_covering_tables.py::test_tables_verified` |
| R39 | 0-1 整数规划最小注数求解 + 规模超限自动降级（记录降级原因） | L21、L168–170 | `covering.py::solve_ilp`、`solve` | `tests/test_covering.py::test_ilp_min_tickets_and_fallback` |
| R40 | 带概率权重贪心集合覆盖 + 模拟退火固定预算优化（seed 固定） | L170 | `covering.py::greedy_weighted`、`anneal` | `tests/test_covering.py::test_greedy_weighted_and_sa_determinism` |
| R41 | 注单后处理与双玩法落地（极端和值/全奇全偶/3 连剔除、去重、预算截断、正交重叠上限、选五/选七） | L161–170 | `kl8/layer5_combinatorial/tickets.py::postprocess`、`build_ticket_plan` | `tests/test_tickets.py::test_postprocess_and_dual_play` |
| R42 | Walk-Forward 滚动回测：T0=300 冷启动、逐期推进 `[1..t]→t+1`、**严禁随机 K-Fold** | L176–182 | `kl8/evaluation/walkforward.py::run_walkforward`、`iter_windows` | `tests/test_walkforward.py::test_walkforward_flow_no_leak` |
| R43 | Top-K 平均命中率与 `Lift@K=mean_hit/(0.25K)` + 单侧 t 检验显著性 + 95% CI。**显著性硬规则（v1.3.0）**：① **样本量门槛**：`n < cfg.backtest.min_n_signal(500)` → `is_signal` 一律 **False**，报告必须标注「样本量不足（n<500），结论仅供参考」；② **多重比较校正**：对 `K∈cfg.backtest.ks`（5 次检验）做 Bonferroni，`alpha_adj = cfg.backtest.alpha_sig / len(ks) = 0.01`；③ `is_signal = (n ≥ min_n_signal) and (lift > lift_signal) and (p_value < alpha_adj)` | L185–188 | `kl8/evaluation/metrics.py::lift_at_k`、`t_test_one_sided` | `tests/test_metrics.py::test_lift_at_k_handcalc`、`test_lift_signal_requires_n500_and_bonferroni` |
| R44 | 高奖级触发率：实测频率 vs 精确超几何理论概率（选三中三 / 选五中四·五 / 选七中六·七）。**v1.3.0**：若同时输出显著性，同样受 R43 两条硬规则约束（`n≥500` + Bonferroni 校正 `α=0.01`）；否则只报点估计与理论值，**不得标「显著」** | L189–190 | `metrics.py::high_tier_rate`、`hypergeometric_baseline` | `tests/test_metrics.py::test_high_tier_vs_hypergeometric` |
| R45 | 规则正交性与归因：R1–R5 独立胜率 + `V≥3` vs 低票组超额命中与 z | L191–192 | `metrics.py::rule_attribution` | `tests/test_metrics.py::test_rule_attribution` |
| R46 | 资金曲线：ROI、最大回撤、最长连挂（整数元，注单成本×2 元） | L193–194 | `metrics.py::bankroll` | `tests/test_metrics.py::test_bankroll_roi_drawdown` |
| R47 | 严格防穿越审计：唯一截断入口 + `max_accessed_t≤t` 断言 + 未来投毒逐位不变性 + 无 shuffle 型交叉验证 | L28、L200 | `kl8/evaluation/leakage.py::assert_no_lookahead`、`poison_future`、`fingerprint_snapshot`；`history.truncated` | `tests/test_leakage.py::test_poison_future_invariance` |
| R48 | 工程落地：配置校验 + CLI + 数据同步 + 三类交付件（TXT/微信/Excel）+ 幂等状态 + 性能预算 | L30–39、L198–202 | `kl8/pipeline.py::run_daily`、`run_backtest`、`run_audit`；`kl8/reporting/*`；`kl8/config.py::load_config`；`kl8/sync.py::sync_history`；`main.py::main` | `tests/test_pipeline.py::test_end_to_end_daily_and_backtest`、`tests/test_config.py::test_config_validation` |

### §1.3 R48 子项拆解（避免单行过载，逐条可验收）

| 子项 | 内容 | 验收证据 |
| :--- | :--- | :--- |
| R48-a | `AppConfig` 深合并 + 严格校验（七维 Σω=1、α∈[0.5,2.0]、λ∈[0.05,0.10]、gold<main、k≤池大小），非法抛 `ConfigError` 中文提示 | `tests/test_config.py::test_config_validation`；config-schema.md §5 |
| R48-b | CLI：保留旧参数（`--history/--no-sync/--quick/--full-sync/--period/--date/--backtest/--full-print`）+ 现代参数（`--mode/--k/--budget/--warmup/--pool-size/--seed/--config`）+ **v1.2.0 新增 `--ml`、`--no-tickets`**；`--backtest` 未显式指定时**按模式分派**（daily→30、backtest/audit→100，`0`=daily 关闭伴随回测）；异常统一中文提示并返回非 0 | `python main.py --help`；`python main.py --no-sync --backtest 30`；`python main.py --mode backtest --no-sync --no-tickets` |
| R48-c | 数据同步：官网主源 + 备用源增量；网络失败仅告警不崩；文件降序写回并保留全量 | `kl8/sync.py::sync_history` 返回值 `SyncResult`；`tests/test_pipeline.py`（离线 mock） |
| R48-d | 三类交付件：`output/reports/Keno分析报告_YYYYMMDD.txt`（六段式 + 附录 A/B）、`output/wechat/Keno发布_YYYYMMDD_第N期.txt`、`output/excel/Keno分析_YYYYMMDD_第N期.xlsx`（6 Sheet） | `tests/test_pipeline.py::test_end_to_end_daily_and_backtest`；P7 端到端实测 |
| R48-e | 幂等：自适应状态按 period 幂等 upsert（原子写 `os.replace`）、重复日跑不重复累计、报告覆盖写 | `tests/test_adaptive.py::test_state_idempotent_atomic_write` |
| R48-f | 性能预算（v1.2.0 按实测修订）：单期（特征→打分→池）≤0.35s、单期含注单+报告 ≤0.50s、100 期回测 ≤45s（`--no-tickets` 30 期冒烟 ≤15s）、500 期 ≤180s、全量单测 ≤180s、ML 关闭为默认 | P7 实测表；超出需给优化证据 |

---

## §2 工程结构与依赖方向

### §2.1 目录树（P2–P6 落地目标）

```
kl8/
├─ __init__.py                 # 仅导出 __version__，禁止在此做重计算
├─ config.py                   # AppConfig + load/validate/merge（§3.3）
├─ constants.py                # 数学恒等式与静态属性（§3.2）
├─ datatypes.py                # 全部跨层数据结构（§3.4、§4）
├─ errors.py                   # 异常体系（§3.1）
├─ history.py                  # 解析/校验/保存/truncated（§3.5）
├─ sync.py                     # 增量同步（§3.6）
├─ layer1_representation/
│  ├─ __init__.py
│  ├─ state.py                 # BinaryStateMatrix（§3.7）
│  └─ windows.py               # 四尺度窗口（§3.7）
├─ layer2_baseline/
│  ├─ __init__.py              # 汇总导出 randomness_report()
│  ├─ priors.py  zscore.py  runs.py  chisq.py  randomness.py  entropy.py
├─ layer3_features/
│  ├─ __init__.py
│  ├─ heat.py  markov.py  cooccurrence.py  graph.py  morphology.py  ml.py
│  └─ features.py              # build_feature_snapshot（唯一装配入口）
├─ layer4_scoring/
│  ├─ __init__.py              # score_round（唯一打分入口）
│  ├─ borda.py  rules.py  hybrid.py  adaptive.py  filter.py
├─ layer5_combinatorial/
│  ├─ __init__.py              # build_tickets（唯一注单入口）
│  ├─ covering.py  tickets.py  covering_tables.json
├─ evaluation/
│  ├─ __init__.py
│  ├─ walkforward.py  metrics.py  leakage.py
├─ reporting/
│  ├─ __init__.py
│  ├─ text_report.py  wechat.py  excel.py
└─ pipeline.py                 # run_daily / run_backtest / run_audit（唯一编排点）
tests/                         # 见 §6.4 命令；文件清单见 §1.2 映射表
tools/
├─ build_covering_tables.py    # 幂等生成并重验覆盖表
├─ ab_legacy.py                # 可选旧系统对照，不进日常流水线
└─ rebuild_state.py            # 从历史重建自适应权重状态
main.py                        # CLI 入口（参数清单见 §3.14）
运行微积分预测.bat             # 双击一键入口（UTF-8 BOM；--no-sync --backtest 30 --full-print）
```

目标产物目录：`output/reports/`、`output/wechat/`、`output/excel/`、`output/state/`。

### §2.2 依赖规则（`tests/test_architecture.py` 逐条断言）

| 允许 | 说明 |
| :--- | :--- |
| `layerN → layerM (M<N)` | 可跨层调用下层 |
| `layerN → config/constants/datatypes/errors` | 任意层可依赖基础设施 |
| `layer1 → history` | 仅 `layer1` 可读 `Dataset` |
| `evaluation/reporting/pipeline → 任意层` | 只读消费，不得被下层反向 import |
| `main.py → pipeline` | CLI 只做参数解析与展示 |

| 禁止 | 断言方式 |
| :--- | :--- |
| `layerN` import `layerN+1`（含 `layer5 → evaluation`） | AST 扫 import 图，出现即失败 |
| `layer1/2/3` import `layer4/5` | 同上 |
| 任意层 import `main` / `pipeline` | 同上 |
| 除 `history.truncated` 外的任何时间截断实现 | `grep` 断言（`ds.draws[:`、`hit[:t]` 只允许出现在 `history.py`/`state.py` 的窗口函数内） |

---

## §3 模块签名契约（冻结）

### §3.0 通用约定

- 所有公开函数带中文 docstring，首行"做什么"，`@param`/`@return` 逐项说明，`@raises` 列出异常。
- 类型别名（`datatypes.py` 或 `typing`）：`NumArray = np.ndarray`（80 维 `float64`，0-based 索引 ↔ 号码 i+1）、`Mask = np.ndarray`（80 维 `bool`）、`Votes = np.ndarray`（80 维 `int8`，取值 0..5）。
- 所有返回 numpy 数组的函数**必须**保证 `shape==(80,)`、`dtype` 明确、无 NaN/Inf（`assert_finite` 在层出口调用）。
- 空/退化输入（`T=0`、`w_eff=0`、`n1=0`）不得抛 `ZeroDivisionError`，返回零向量/中性值并在 `warnings` 中说明。
- 异常一律继承 `kl8.errors.KL8Error`，消息为中文（含修复建议）。

### §3.1 `kl8/errors.py`

```python
class KL8Error(Exception): ...                    # 全系统基类（不直接抛）
class ConfigError(KL8Error): ...                  # 配置非法：字段/取值/约束冲突
class DataFormatError(KL8Error): ...              # 历史数据/报告上下文格式非法
class InsufficientDataError(KL8Error): ...        # 期数不足以支撑窗口/回测/覆盖
class InfeasibleConstraintError(KL8Error): ...    # 约束不可行（池太小、预算与覆盖冲突、组合规模超限）
class LeakageError(KL8Error): ...                 # 防穿越断言失败（宁可中断，不得静默）
class CoveringError(KL8Error): ...                # 覆盖表损坏/参数非法
```
异常契约：`ConfigError` 只能由 `config.py` 抛出；`DataFormatError` 只能由 `history.py`/`datatypes.py` 校验抛出；`InsufficientDataError` 由 `layer1`/`evaluation`/`layer5` 抛出；`LeakageError` 只由 `evaluation/leakage.py` 抛出。

### §3.2 `kl8/constants.py`（只放数学恒等式与静态属性，无业务阈值）

```python
N_NUM: int = 80                                   # 号码总数
K_DRAW: int = 20                                  # 每期开出个数
P0: float = 0.25                                  # = K_DRAW / N_NUM
P_CO: float = 20 * 19 / (80 * 79)                 # ≈ 0.0601265823
E_REPEAT: float = 5.0                             # = K_DRAW * P0
SUM_EXPECT: int = 810                             # 单期 20 码理论期望和值 (20*81/2)
Z_SIG: float = 1.96                               # 双尾 5% 显著性阈值
PRIMES_80: frozenset[int]                         # 22 个质数（1 非质非合）
ROAD_0_NUMS / ROAD_1_NUMS / ROAD_2_NUMS: frozenset[int]      # 模 3 同余 012 路
ZONE4_BOUNDS: tuple[tuple[int, int], ...]         # ((1,20),(21,40),(41,60),(61,80))
ZONE8_BOUNDS: tuple[tuple[int, int], ...]         # ((1,10),...,(71,80))
SIZE_BOUNDS: tuple[tuple[int, int], ...]          # ((1,27),(28,54),(55,80)) 见裁决 A3
ALL_NUMBERS: np.ndarray                           # arange(1, 81)
HISTORY_LINE_PATTERN: re.Pattern                  # date:(...),period:(...),numbers:(...)

def get_road_012(num: int) -> int: ...            # 号码 → 0/1/2；越界 ValueError
def is_prime_num(num: int) -> bool: ...
def get_prime_label(num: int) -> str: ...          # "质" | "合" | "特1"
def zone4_of(num: int) -> int: ...                # 0..3
def zone8_of(num: int) -> int: ...                # 0..7
def size_class_of(num: int) -> int: ...           # 0=小 1=中 2=大
def tail_of(num: int) -> int: ...                 # num % 10
def numbers_array() -> np.ndarray: ...            # (80,) float64 = 1..80
def hypergeometric_pmf(n_pick: int, n_draw: int = K_DRAW, n_total: int = N_NUM) -> np.ndarray: ...
def hypergeometric_mean(n_pick: int, n_draw: int = K_DRAW, n_total: int = N_NUM) -> float: ...
def hypergeometric_var(n_pick: int, ...) -> float: ...
```
异常：号码非 `int` 或不在 `1..N_NUM` → `ValueError("号码 {num} 非法，必须在 1..80")`。

### §3.3 `kl8/config.py`

```python
@dataclass(frozen=True)
class WindowConfig:
    short: int = 10; mid: int = 25; long: int = 50
    min_history: int = 40          # 低于此期数抛 InsufficientDataError
    cold_start_floor: int = 1

@dataclass(frozen=True)
class MarkovConfig:
    alpha: float = 1.0             # 拉普拉斯平滑 α ∈ [0.5, 2.0]
    order: int = 2                 # 1|2（二阶为可选特征）
    diag_split: bool = True        # 重号/异号拆分

@dataclass(frozen=True)
class ScoreWeights:                # 七维核心权重 Σω = 1.00 ± 1e-9 强校验（ml 为额外叠加项，不计入 Σω）
    follow_10: float = 0.02; follow_25: float = 0.08
    follow_50: float = 0.20; follow_all: float = 0.25      # Σ = 0.55
    co: float = 0.05; trend: float = 0.05; omission: float = 0.35   # 七维合计 1.00（v1.2.0 与实现对齐）
    lambda_vote: float = 0.05      # λ ∈ [0.05, 0.10]（v1.2.0 实现取下沿 0.05）
    trend_w_f10: float = 0.5; trend_w_mom_10_25: float = 0.3; trend_w_mom_25_50: float = 0.2  # 裁决 A1
    ml: float = 0.0                # **额外叠加项**（默认 0）；>0 时叠加到 BaseScore，不参与 Σω 归一化

@dataclass(frozen=True)
class RuleConfig:                  # R1–R5 阈值（全量见 config-schema.md §3）
    r1_min_windows: int = 3; r1_rank_threshold: float = 0.65
    r2_base_freq: float = 0.24; r2_f10_low: float = 0.20; r2_f10_high: float = 0.50
    r3_follow_rank: float = 0.60
    r4_rank_score: float = 0.75; r4_w_co25: float = 0.5; r4_w_co50: float = 0.5
    r5_pressure: float = 0.60; r5_max_gap: int = 12; r5_follow_all_rank: float = 0.70

@dataclass(frozen=True)
class AdaptiveConfig:
    lookback_m: int = 8            # 裁决 A9
    min_periods: int = 3
    laplace_alpha: float = 1.0
    temperature: float = 1.0
    weight_floor: float = 0.05
    schema_version: int = 1
    state_path: str = "output/state/adaptive_weights.json"
    in_memory: bool = False        # 回测强制 True
    persist: bool = True

@dataclass(frozen=True)
class PoolConfig:
    gold_min: int = 3; gold_max: int = 5
    main_default: int = 12; main_min: int = 10; main_max: int = 15     # 裁决 A6
    max_zone4_ratio: float = 0.45
    max_parity_ratio: float = 0.70
    max_same_tail: int = 3
    repeat_ratio: float = 0.30; repeat_ratio_min: float = 0.25; repeat_ratio_max: float = 0.35
    relax_order: tuple[str, ...] = ("same_tail", "parity", "zone4", "repeat_ratio")

@dataclass(frozen=True)
class CoveringConfig:
    play: str = "pick5"            # pick5 | pick7
    k: int = 5; t: int = 4
    m_mode: str = "auto"; m_fixed: int | None = None; m_margin: int = 1   # 裁决 A7（v1.1.0：默认 1，范围 [1,3]）
    budget: int = 20; budget_by_play: Mapping[str, int] = {"pick5": 20, "pick7": 10}  # 裁决 A8
    max_budget: int = 200
    max_pool_v: int = 20; max_m_subsets: int = 200_000
    ilp_candidate_cap: int = 120; ilp_target_cap: int = 400; ilp_time_limit_s: float = 5.0
    sa_iterations: int = 2000; sa_t0: float = 1.0; sa_cooling: float = 0.995
    max_pair_overlap: int = 3
    max_consecutive_in_ticket: int = 2
    forbid_all_odd_even: bool = True
    sum_margin: float = 0.45       # 单注和值安全带宽（相对池均值×k）
    table_path: str = "kl8/layer5_combinatorial/covering_tables.json"

@dataclass(frozen=True)
class MLConfig:
    enabled: bool = False          # v1.2.0：默认关闭（B4 诊断通道）；CLI `--ml` 显式开启
    weight: float = 0.0            # 额外叠加项，默认 0（仅诊断，不进 FinalScore）
    train_window: int = 160; refit_every: int = 25
    n_estimators: int = 40; learning_rate: float = 0.03
    num_leaves: int = 12; min_child_samples: int = 25
    subsample: float = 0.85; colsample_bytree: float = 0.85
    seed: int = 42; calibrate: str = "pav"; fallback: str = "ema"
    cache_size: int = 8

@dataclass(frozen=True)
class BacktestConfig:
    warmup: int = 300              # T0 冷启动
    default_n: int = 100           # 配置层默认；CLI 未指定时按模式分派：daily→30、backtest/audit→100（§3.14）
    ks: tuple[int, ...] = (3, 5, 7, 10, 12)
    t0_min: int = 60
    include_tickets: bool = True
    alpha_sig: float = 0.05
    lift_signal: float = 1.05      # 8.md L188「Lift@K > 1.05」判据阈值 ∈[1.0, 1.2]（v1.1.0 新增，禁止字面量）
    min_n_signal: int = 500        # v1.3.0：显著性最小样本量门槛；n < 本值 → is_signal 恒 False（8.md L188「500 期以上」）
    bonferroni: bool = True        # v1.3.0：K 组多重比较 Bonferroni 校正；True 时 alpha_adj = alpha_sig / len(ks)

@dataclass(frozen=True)
class AppConfig:
    windows: WindowConfig = WindowConfig()
    markov: MarkovConfig = MarkovConfig()
    weights: ScoreWeights = ScoreWeights()
    rules: RuleConfig = RuleConfig()
    adaptive: AdaptiveConfig = AdaptiveConfig()
    pool: PoolConfig = PoolConfig()
    covering: CoveringConfig = CoveringConfig()
    ml: MLConfig = MLConfig()
    backtest: BacktestConfig = BacktestConfig()
    history_path: str = "kl8_history.txt"
    output_dir: str = "output"
    seed: int = 20261102
    eps: float = 1e-9
    cost_per_ticket: int = 2
    prize_table: Mapping[str, Mapping[int, int] | None] = {"pick5": {5: 1000, 4: 21, 3: 3}, "pick3": None, "pick7": None}  # 裁决 A10

# —— 公开函数 ——
def default_config() -> AppConfig: ...
def load_config(overrides: Mapping[str, Any] | None = None,
                path: str | Path | None = None) -> AppConfig: ...   # 深合并 + validate
def validate_config(cfg: AppConfig) -> None: ...                    # 失败抛 ConfigError（中文，含修复建议）
def merge_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig: ...
def config_hash(cfg: AppConfig) -> str: ...        # sha256 前 16 hex，用于 ML 缓存 key 与回测血缘
def to_dict(cfg: AppConfig) -> dict[str, Any]: ...
def from_dict(payload: Mapping[str, Any]) -> AppConfig: ...
```
异常契约：任何字段缺失/类型不符/越界/约束冲突 → `ConfigError`（消息格式：`配置项 {路径} 非法：{现值}，要求 {约束}；建议 {修复}`）。`load_config` 不得抛 `KeyError`/`TypeError`/`ValueError`。

### §3.4 `kl8/datatypes.py`（类签名；字段细则见 §4）

```python
@dataclass(frozen=True)
class Draw:
    date: str            # "YYYY-MM-DD"（严格正则校验）
    period: int          # 期号（7 位正整数）
    numbers: frozenset[int]   # 20 个 1..80 不重复
    idx: int = -1        # 在升序 Dataset 中的 0-based 索引
    def as_asc(self) -> tuple[int, ...]: ...
    def row(self) -> np.ndarray: ...          # (80,) float64

@dataclass(frozen=True)
class Dataset:
    draws: tuple[Draw, ...]                   # 严格升序（date, period）
    draws_by_period: Mapping[int, Draw]
    source_path: str = ""
    warnings: tuple[str, ...] = ()
    @property
    def T(self) -> int: ...
    @property
    def periods(self) -> tuple[int, ...]: ...
    @property
    def latest(self) -> Draw: ...             # T=0 → InsufficientDataError
    def hit_matrix(self) -> np.ndarray: ...   # (T,80) float64
    def index_of_period(self, period: int) -> int: ...   # 不存在 → KeyError
    def target_period(self) -> int: ...       # latest.period + 1

class TimeSeries:                             # 预计算容器（不可变）
    def __init__(self, ds: Dataset) -> None: ...
    T: int; hit: np.ndarray                   # (T,80) float64
    gap: np.ndarray                           # (T,80) int32 遗漏矩阵
    def freq(self, end: int, w: int | None) -> np.ndarray: ...        # (80,)
    def window_hits(self, end: int, w: int | None) -> np.ndarray: ... # (w_eff,80)

@dataclass(frozen=True)
class FeatureSnapshot: ...    # 字段见 §4.1（冻结）
@dataclass(frozen=True)
class RuleVerdict: ...        # §4.2
@dataclass(frozen=True)
class PoolResult: ...         # §4.3
@dataclass(frozen=True)
class TicketPlan: ...         # §4.4
@dataclass(frozen=True)
class CoveringReport: ...     # §4.5
@dataclass(frozen=True)
class BacktestRow: ...        # §4.6
@dataclass(frozen=True)
class BacktestReport: ...     # §4.6

def assert_finite(arr: np.ndarray, name: str) -> None: ...   # NaN/Inf → DataFormatError
```

### §3.5 `kl8/history.py`

```python
def parse_line(line: str, line_no: int = 0) -> Draw: ...
    # 格式 date:YYYY-MM-DD,period:NNNNNNN,numbers:NN-NN-...；不合法 → DataFormatError（含行号）

def load_dataset(path: str | Path = "kl8_history.txt") -> Dataset: ...
    # 1) 逐行 parse_line，坏行记 warning 并跳过（坏行数 > 0 时 warnings 必非空）
    # 2) 按 (date, period) 升序重排
    # 3) 硬校验三条（§0.1）：日期单调不减 / period 唯一（重复→DataFormatError）/ 降序文件首行自检
    # 4) 期号不连续只记 warning（形如 "期号断档：2020065 → 2021001（自然跨年期）"）
    # 5) 文件不存在 → DataFormatError；解析后 T==0 → InsufficientDataError

def save_dataset(ds: Dataset, path: str | Path) -> int: ...
    # 降序（新期在前）写回，UTF-8，原子写（tmp + os.replace），返回写入期数

def truncated(ds: Dataset, target_period: int) -> Dataset: ...
    # 【防穿越唯一入口】返回严格早于 target_period 的全部期（不含 target_period）
    # 等价关系：predict X_{t+1} ⇔ truncated(ds, t+1)
    # target_period 晚于最新期 → 返回全部；早于最早期 → 返回空 Dataset + warning

def latest_period(ds: Dataset) -> int: ...
def append_draws(ds: Dataset, draws: Iterable[Draw]) -> Dataset: ...   # 按期号幂等去重
```
异常契约：文件级问题（不存在/不可读）→ `DataFormatError`；空数据集 → `InsufficientDataError`；`truncated` 永不抛异常。

### §3.6 `kl8/sync.py`

```python
@dataclass(frozen=True)
class SyncResult:
    ok: bool; added: int; total: int; latest_period: int | None
    warnings: tuple[str, ...]

def fetch_official(max_pages: int = 50, timeout: float = 30.0) -> dict[int, tuple[str, tuple[int, ...]]]: ...
def fetch_backup(year: int | None = None, timeout: float = 20.0) -> dict[int, tuple[str, tuple[int, ...]]]: ...
def sync_history(path: str | Path = "kl8_history.txt", *, quick: bool = True,
                 timeout: float = 30.0) -> SyncResult: ...
```
异常契约：**网络/解析异常一律内部捕获转 `warnings`，绝不外抛**；只有本地文件不可写才抛 `DataFormatError`。两源都失败 → `ok=False` 且不修改本地文件。

### §3.7 `kl8/layer1_representation/`

```python
# state.py
class BinaryStateMatrix:
    def __init__(self, hit: np.ndarray) -> None: ...
        # 校验 shape (T,80)、取值∈{0,1}、每行和 == K_DRAW（T=0 空安全）；否则 DataFormatError
    @classmethod
    def from_dataset(cls, ds: Dataset) -> "BinaryStateMatrix": ...
    T: int
    hit: np.ndarray                                  # (T,80) float64
    def prefix(self) -> np.ndarray: ...              # (T+1,80) int32 前缀和
    def row_sum(self, t: int) -> int: ...
    def window_matrix(self, end: int, w: int | None) -> np.ndarray: ...   # (w_eff,80)，end 含端点
    def window_sum(self, end: int, w: int | None) -> np.ndarray: ...     # (80,) int32
    def window_mean(self, end: int, w: int | None) -> np.ndarray: ...    # (80,) float64
    def latest_set(self, end: int) -> frozenset[int]: ...                # 1-based 号码集合
    def gap_vector(self, end: int) -> np.ndarray: ...                    # (80,) int32 当前遗漏
    def is_empty(self) -> bool: ...

# windows.py
@dataclass(frozen=True)
class WindowSpec:
    name: str        # "10G" | "25G" | "50G" | "ALL"
    size: int | None # None 表示 ALL

def resolve_windows(cfg: WindowConfig, T: int) -> dict[str, WindowSpec]: ...   # 固定 4 键、固定顺序
def effective_window(w: int | None, T: int, cold_start_floor: int = 1) -> int: ...  # max(1,min(w,T))；w=None→max(1,T)
def window_label(w: int | None) -> str: ...
def is_cold_start(w: int | None, T: int) -> bool: ...
```
异常：`end` 越界 → `IndexError`；`w<=0` → `ValueError`；`T==0` 时 `window_*` 返回零向量（不抛）。

### §3.8 `kl8/layer2_baseline/`

```python
# priors.py
def p0(n_draw: int = K_DRAW, n_total: int = N_NUM) -> float: ...
def p_co(n_draw: int = K_DRAW, n_total: int = N_NUM) -> float: ...
def expected_repeat(n_draw: int = K_DRAW, p: float = P0) -> float: ...
def hypergeometric_baseline(n_pick: int, n_draw: int = K_DRAW, n_total: int = N_NUM) -> BaselineStats: ...
    # BaselineStats(n_pick, mean, sd, pmf: np.ndarray(len n_pick+1), p_ge: Mapping[int,float])
def posterior_beta(success: int, trials: int, *, alpha: float = 1.0, beta: float = 1.0) -> tuple[float, float, float]: ...

# zscore.py
def window_z(state: BinaryStateMatrix, end: int, w: int | None, *, p: float = P0, eps: float = 1e-9) -> np.ndarray: ...
def significant_mask(z: np.ndarray, threshold: float = Z_SIG) -> np.ndarray: ...   # (80,) bool
def z_all_windows(state: BinaryStateMatrix, end: int,
                  windows: Mapping[str, WindowSpec], *, p: float = P0) -> dict[str, np.ndarray]: ...

# runs.py
@dataclass(frozen=True)
class RunsStat:
    n: int; n1: int; n0: int; runs: int
    expected: float; variance: float; z: float
    label: str        # "聚簇连出" | "频繁跳跃" | "无偏" | "退化"
def runs_test(series: np.ndarray, *, eps: float = 1e-9) -> RunsStat: ...
def runs_test_80(window_matrix: np.ndarray, *, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]: ...
    # 返回 (z:(80,), runs:(80,) int32, labels: tuple[str,...] len=80)

# chisq.py
@dataclass(frozen=True)
class ChiSquareResult:
    statistic: float; dof: int; p_value: float
    expected: np.ndarray; observed: np.ndarray
    method: str       # "scipy" | "wilson_hilferty" | "degenerate"
def chi_square_gof(observed: np.ndarray, w: int, *, p: float = P0) -> ChiSquareResult: ...
def chi_square_by_window(state: BinaryStateMatrix, end: int,
                         windows: Mapping[str, WindowSpec]) -> dict[str, ChiSquareResult]: ...

# randomness.py
@dataclass(frozen=True)
class NistResult:
    name: str; statistic: float; p_value: float; passed: bool; note: str
def monobit(bits: np.ndarray) -> NistResult: ...
def frequency_within_block(bits: np.ndarray, block_size: int = 128) -> NistResult: ...
def runs_nist(bits: np.ndarray) -> NistResult: ...
def approximate_entropy(bits: np.ndarray, m: int = 2) -> NistResult: ...
def nist_lite(bits: np.ndarray, *, block_size: int = 128, apen_m: int = 2,
              alpha: float = 0.01) -> tuple[NistResult, ...]: ...   # 固定 4 项、固定顺序
def flatten_history(hit: np.ndarray) -> np.ndarray: ...                # 按期序展平为 0-1 流

# entropy.py
def shannon_entropy(p: np.ndarray, *, base: float = 2.0, eps: float = 1e-12) -> float: ...
def freq_entropy(window_sum: np.ndarray, w: int) -> float: ...         # 与 log2(80)≈6.3219 对比
def mi_lag1(window_matrix: np.ndarray) -> np.ndarray: ...              # (80,80) 跨期互信息
def feature_redundancy(features: Mapping[str, np.ndarray], top_n: int = 20) -> tuple[tuple[str, str, float], ...]: ...
def ema_smooth(x: np.ndarray, span: int) -> np.ndarray: ...
def weibull_survival_diagnostic(gaps: np.ndarray) -> dict[str, float]: ...   # 仅诊断
def randomness_report(state: BinaryStateMatrix, end: int,
                      windows: Mapping[str, WindowSpec]) -> "RandomnessReport": ...   # §4.7
```
异常：全部按 `eps` 防守，退化输入返回中性值 + `note`；`n_pick<0` / `w<0` → `ValueError`。

### §3.9 `kl8/layer3_features/`

```python
# heat.py
@dataclass(frozen=True)
class HeatResult:
    freq: Mapping[str, np.ndarray]      # keys "10G","25G","50G","ALL"
    mom_10_25: np.ndarray; mom_25_50: np.ndarray
    golden_cross: np.ndarray            # (80,) bool
    gap: np.ndarray                     # (80,) int32
    gap_mean: np.ndarray; gap_ratio: np.ndarray; pressure: np.ndarray
def multi_scale_freq(state, end, windows) -> dict[str, np.ndarray]: ...
def momentum(freq: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]: ...
def golden_cross(mom_10_25: np.ndarray, mom_25_50: np.ndarray) -> np.ndarray:
    # 双正且短中差 > 中长差（m1>0 且 m2>0 且 m1>m2）
def gap_series(state, end) -> np.ndarray: ...
def historical_mean_gap(state, end, *, floor: float = 3.0) -> np.ndarray: ...   # μ=0 → floor
def omission_pressure(gap, gap_mean, *, center: float = 1.2, sigma: float = 0.6,
                      floor: float = 3.0) -> np.ndarray: ...   # 结果 clip 到 [0,1]
def build_heat(state, end, windows) -> HeatResult: ...

# markov.py
@dataclass(frozen=True)
class MarkovResult:
    follow: Mapping[str, np.ndarray]        # 四窗口 FollowScore
    follow_diag: Mapping[str, np.ndarray]   # 重号惯性分（对角）
    follow_offdiag: Mapping[str, np.ndarray]# 异号跟随分（非对角）
    follow2_all: np.ndarray                 # 二阶 M^(2) 的 ALL 窗跟随分
    matrices: Mapping[str, np.ndarray]      # keys 同 windows；每个 (80,80)
def transition_matrix(window_matrix: np.ndarray, *, alpha: float, p0: float) -> np.ndarray: ...
def transition_matrix_order2(window_matrix: np.ndarray, *, alpha: float, p0: float) -> np.ndarray: ...
def follow_score(matrix: np.ndarray, current_set: frozenset[int], *, p0: float) -> np.ndarray: ...
def follow_score_all_windows(matrices: Mapping[str, np.ndarray], current_set: frozenset[int],
                             *, p0: float) -> dict[str, np.ndarray]: ...
def follow_split(matrices: Mapping[str, np.ndarray], current_set: frozenset[int],
                 *, p0: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: ...
def build_markov(state, end, windows, cfg: MarkovConfig) -> MarkovResult: ...

# cooccurrence.py
def cooccurrence_counts(window_matrix: np.ndarray) -> np.ndarray: ...      # (80,80) int32，对角置 0
def centered_cooccurrence(counts: np.ndarray, w: int, *, p_co: float) -> np.ndarray: ...   # C/w - p_co
def co_score(A: np.ndarray, current_set: frozenset[int]) -> np.ndarray: ...# Σ_{i∈S_t,i≠j} A_ij
def co_score_all_windows(A_map: Mapping[str, np.ndarray],
                         current_set: frozenset[int]) -> dict[str, np.ndarray]: ...
def co_centered_mean(co_map: Mapping[str, np.ndarray]) -> np.ndarray: ...  # mean(co_25, co_50, co_all) 原始值
def lift_matrix(counts: np.ndarray, w: int, *, p0: float, eps: float = 1e-12) -> np.ndarray: ...
def triad_counts(counts: np.ndarray, *, top_n: int = 20) -> tuple[int, np.ndarray]: ...

# graph.py
@dataclass(frozen=True)
class GraphResult:
    pagerank: np.ndarray; eigen: np.ndarray
    community_id: np.ndarray; community_size: np.ndarray; community_strength: np.ndarray
    companion: Mapping[int, tuple[tuple[int, float], ...]]
    backend: str        # "networkx" | "numpy_power"
    warnings: tuple[str, ...]
def build_graph(A: np.ndarray, *, min_weight: float = 0.0) -> object | None: ...  # 仅正值边；空图返回 None
def pagerank(A: np.ndarray, *, alpha: float = 0.85, max_iter: int = 100,
             tol: float = 1e-6, backend: str = "auto") -> np.ndarray: ...
def eigenvector_centrality(A: np.ndarray, *, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray: ...
def louvain_communities(A: np.ndarray, *, seed: int, resolution: float = 1.0
                        ) -> tuple[np.ndarray, np.ndarray]: ...          # (community_id, community_size)
def companion_topk(A: np.ndarray, *, k: int = 3) -> dict[int, tuple[tuple[int, float], ...]]: ...
def build_graph_features(A: np.ndarray, *, seed: int, k: int = 3) -> GraphResult: ...

# morphology.py
@dataclass(frozen=True)
class MorphologyStats:
    n: int
    zone4: Mapping[int, int]; zone8: Mapping[int, int]
    odd: int; even: int; prime: int; composite: int
    size: Mapping[str, int]                       # {"小":x,"中":y,"大":z}
    consecutive: Mapping[str, int]                # {"二连":a,"三连":b,"max_run":m}
    same_tail: Mapping[int, int]                  # {尾号: 个数}
    sum_value: int; sum_z: float; span: int; ac: int
def zone_distribution(nums: Sequence[int], *, scheme: str = "zone4") -> dict[int, int]: ...
def odd_even_ratio(nums) -> tuple[int, int]: ...
def prime_composite_ratio(nums) -> tuple[int, int, int]: ...     # (质, 合, 1 的个数)
def size_ratio(nums) -> dict[str, int]: ...
def consecutive_groups(nums) -> dict[str, int]: ...
def same_tail_groups(nums) -> dict[int, int]: ...
def sum_value(nums) -> int: ...
def sum_deviation_z(nums: Sequence[int], *, n_pick: int | None = None) -> float: ...
def ac_value(nums: Sequence[int]) -> int: ...                    # 裁决 A4
def morphology_stats(nums: Sequence[int]) -> MorphologyStats: ...
def is_balanced(nums: Sequence[int], cfg: PoolConfig) -> tuple[bool, tuple[str, ...]]: ...
def assert_valid_numbers(nums: Sequence[int]) -> None: ...        # 越界/重复 → DataFormatError

# ml.py
@dataclass(frozen=True)
class MLResult:
    prob: np.ndarray; available: bool; backend: str
    refit: bool; n_train: int; warnings: tuple[str, ...]
class MLCache:
    def __init__(self, max_size: int = 8) -> None: ...
    def get(self, key: tuple[int, str, str]) -> np.ndarray | None: ...
    def put(self, key: tuple[int, str, str], prob: np.ndarray) -> None: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...
def ml_feature_names() -> tuple[str, ...]: ...          # 固定顺序，schema hash 依据（≥16 列）
def extract_feature_matrix(state, end, windows, *,
                           extra: Mapping[str, np.ndarray] | None = None) -> np.ndarray: ...  # (80, D)
def pav_calibrate(scores: np.ndarray, labels: np.ndarray) -> np.ndarray: ...                  # 自写保序回归
def ema_fallback(state, end) -> np.ndarray: ...                                              # (80,) 概率
def train_predict(state, end, cfg: MLConfig, *, cache: MLCache | None = None) -> MLResult: ...

# features.py
SCHEMA_VERSION: str = "fs-v1"
def build_feature_snapshot(state: BinaryStateMatrix, end: int, cfg: AppConfig, *,
                           include_ml: bool | None = None,
                           ml_cache: MLCache | None = None) -> FeatureSnapshot: ...
    # end 语义：使用 X[0..end] 预测 end+1；所有数组 len=80；出口调用 validate_snapshot
def validate_snapshot(snapshot: FeatureSnapshot) -> None: ...        # NaN/Inf/长度/枚举 → DataFormatError
def feature_matrix(snapshot: FeatureSnapshot, fields: Sequence[str]) -> np.ndarray: ...
```
异常契约：`end<0` → `IndexError`；`end+1 < cfg.windows.min_history` → `InsufficientDataError`；`networkx`/`lightgbm` 缺失或异常 → 降级分支 + `warnings`，**不得抛出**。

### §3.10 `kl8/layer4_scoring/`

```python
# borda.py
def percentile_rank(values: np.ndarray) -> np.ndarray: ...
    # R = Rank/80 ∈ (0,1]；并列取平均秩；空数组→空；全相等→(n+1)/(2n)
def rank_matrix(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]: ...
def top_percentile_mask(ranks: np.ndarray, threshold: float) -> np.ndarray: ...   # ranks >= threshold

# rules.py（五条规则均为纯函数，输入快照+秩，输出 (80,) bool）
def rule_r1(ranks: Mapping[str, np.ndarray], cfg: RuleConfig) -> np.ndarray: ...
def rule_r2(snapshot: FeatureSnapshot, cfg: RuleConfig) -> np.ndarray: ...
def rule_r3(snapshot: FeatureSnapshot, ranks: Mapping[str, np.ndarray], cfg: RuleConfig) -> np.ndarray: ...
def rule_r4(ranks: Mapping[str, np.ndarray], cfg: RuleConfig) -> np.ndarray: ...
def rule_r5(snapshot: FeatureSnapshot, ranks: Mapping[str, np.ndarray], cfg: RuleConfig) -> np.ndarray: ...
def evaluate_rules(snapshot: FeatureSnapshot, ranks: Mapping[str, np.ndarray],
                   cfg: RuleConfig) -> tuple[RuleVerdict, ...]: ...      # 固定 5 个，顺序 R1..R5
def vote(verdicts: Sequence[RuleVerdict]) -> np.ndarray: ...           # (80,) int8，0..5

# hybrid.py
def trend_score(ranks: Mapping[str, np.ndarray], weights: ScoreWeights) -> np.ndarray: ...   # 裁决 A1
def co_aggregate(ranks: Mapping[str, np.ndarray]) -> np.ndarray: ...    # 裁决 A2
def base_score(snapshot: FeatureSnapshot, ranks: Mapping[str, np.ndarray],
               weights: ScoreWeights) -> np.ndarray: ...
def final_score(base: np.ndarray, votes: np.ndarray, lambda_vote: float) -> np.ndarray: ...
@dataclass(frozen=True)
class ScoreResult:
    base: np.ndarray; final: np.ndarray; order: tuple[int, ...]        # order 为号码 1-based 降序
    contributors: Mapping[str, np.ndarray]     # 各分项（follow_10/…/co/trend/omission/vote）
def score_all(snapshot: FeatureSnapshot, ranks: Mapping[str, np.ndarray],
              weights: ScoreWeights) -> ScoreResult: ...

# adaptive.py
@dataclass(frozen=True)
class AdaptiveState:
    schema_version: int; period: int
    weights: Mapping[str, float]              # keys "R1".."R5"
    hit_rates: Mapping[str, float]
    n_updates: int; updated_at: str           # ISO8601
def rule_hit_rates(rows: Sequence[BacktestRow], m: int) -> tuple[np.ndarray, int]: ...
def softmax_weights(hit_rates: np.ndarray, *, prior: np.ndarray, laplace_alpha: float,
                    temperature: float, weight_floor: float) -> np.ndarray: ...   # 和 = 1
def next_weights(rows: Sequence[BacktestRow], current: AdaptiveState | None,
                 cfg: AdaptiveConfig, *, period: int) -> tuple[AdaptiveState, tuple[str, ...]]: ...
def load_state(path: str | Path, schema_version: int) -> tuple[AdaptiveState | None, tuple[str, ...]]: ...
    # 不存在→(None, ())；损坏/旧版本→自动 .bak 重建 + warning；不抛
def save_state(state: AdaptiveState, path: str | Path) -> None: ...   # 原子写 os.replace

# filter.py
def greedy_pool(scores: np.ndarray, snapshot: FeatureSnapshot, cfg: PoolConfig,
                *, size: int) -> tuple[list[int], tuple[str, ...]]: ...
def enforce_constraints(candidates: Sequence[int], snapshot: FeatureSnapshot,
                        cfg: PoolConfig, *, size: int) -> tuple[list[int], tuple[str, ...]]: ...
def apply_repeat_quota(order: Sequence[int], last_draw: frozenset[int], cfg: PoolConfig,
                       *, size: int) -> tuple[list[int], tuple[str, ...]]: ...
def select_gold(main: Sequence[int], final: np.ndarray, votes: np.ndarray,
                cfg: PoolConfig) -> list[int]: ...
def build_pool(snapshot: FeatureSnapshot, final: np.ndarray, votes: np.ndarray,
               last_draw: frozenset[int], *, pool_size: int, cfg: AppConfig) -> PoolResult: ...
    # 约束不可行：按 cfg.pool.relax_order 逐项放宽并写 warnings；仍不可行 → InfeasibleConstraintError

# __init__.py（唯一打分入口）
@dataclass(frozen=True)
class ScoreRoundResult:
    snapshot: FeatureSnapshot; ranks: Mapping[str, np.ndarray]
    verdicts: tuple[RuleVerdict, ...]; votes: np.ndarray
    scores: ScoreResult; pool: PoolResult
def score_round(snapshot: FeatureSnapshot, cfg: AppConfig, *,
                adaptive_state: AdaptiveState | None = None) -> ScoreRoundResult: ...
```

### §3.11 `kl8/layer5_combinatorial/`

```python
# covering.py
@dataclass(frozen=True)
class GuaranteeResult:
    passed: bool; achieved_t: int; n_subsets: int
    failed_example: tuple[int, ...] | None      # 未达保底的 m 子集（0-based 位置）
def m_target(v: int, k: int, t: int, m_mode: str = "auto", m_fixed: int | None = None,
             margin: int = 1) -> int: ...
    # 裁决 A7（v1.1.0 修正）：t >= k → m = min(v, t)；t < k → m = min(v, t + margin)
    # margin 默认 ≥1（cfg.covering.m_margin），保证 m ≥ t+1；|m-(ceil(0.25v)+2)|>2 时由 solve() 写 warning
def verify_guarantee(tickets: Sequence[Sequence[int]], v: int, k: int, t: int, m: int
                     ) -> GuaranteeResult: ...  # 位掩码穷举；tickets 为 0-based 位置（0..v-1）
def lookup_table(v: int, k: int, t: int, m: int, *, path: str | Path) -> tuple[tuple[int, ...], ...] | None: ...
def greedy_weighted(v: int, k: int, t: int, m: int, weights: np.ndarray, *,
                    budget: int, seed: int, max_iters: int = 2000) -> tuple[tuple[int, ...], ...]: ...
def anneal(v: int, k: int, t: int, m: int, weights: np.ndarray, *, budget: int,
           iterations: int, t0: float, cooling: float, seed: int) -> tuple[tuple[int, ...], ...]: ...
def solve_ilp(v: int, k: int, t: int, m: int, *, candidate_cap: int, target_cap: int,
              time_limit_s: float = 5.0) -> tuple[tuple[int, ...], ...] | None: ...
def coverage_ratio(tickets: Sequence[Sequence[int]], v: int, k: int, t: int, m: int) -> float: ...
def solve(v: int, k: int, t: int, m: int, weights: np.ndarray,
          cfg: CoveringConfig) -> CoveringReport: ...
    # 流程：表命中 → ILP（规模内）→ 贪心 + SA；结尾必 verify_guarantee，结果写进 CoveringReport

# tickets.py
def positions_to_numbers(tickets: Sequence[Sequence[int]], pool: Sequence[int]) -> tuple[tuple[int, ...], ...]: ...
def postprocess(tickets: Sequence[Sequence[int]], pool: Sequence[int], *,
                cfg: CoveringConfig, scores_by_number: Mapping[int, float]
                ) -> tuple[tuple[int, ...], tuple[str, ...]]: ...    # 剔除/去重/预算截断/重叠上限
def build_ticket_plan(pool: Sequence[int], scores: np.ndarray, cfg: AppConfig, *,
                      play: str | None = None, budget: int | None = None) -> TicketPlan: ...
def plan_cost(plan: TicketPlan) -> int: ...                            # 注数 × cost_per_ticket（整数元）

# __init__.py
def build_tickets(pool_result: PoolResult, scores: np.ndarray, cfg: AppConfig) -> tuple[TicketPlan, ...]: ...
```
异常契约：`v > cfg.max_pool_v` 或 `comb(v, m) > cfg.max_m_subsets` → `InfeasibleConstraintError`（中文说明规模上限）；`ilp` 不可用/超时 → 返回 `None`，`solve` 降级到贪心并在 `CoveringReport.warnings` 记录原因；覆盖表损坏 → `CoveringError`。

### §3.12 `kl8/evaluation/`

```python
# walkforward.py
def iter_windows(ds: Dataset, *, t0: int, n_test: int | None = None
                 ) -> Iterator[tuple[int, Dataset]]: ...      # (t, truncated(ds, t+1))
def run_walkforward(ds: Dataset, cfg: AppConfig, *, n_test: int | None = None,
                    t0: int | None = None, include_tickets: bool | None = None,
                    progress: Callable[[int, int], None] | None = None) -> BacktestReport: ...
    # 1) T0 = cfg.backtest.warmup；T <= T0+t0_min → InsufficientDataError
    # 2) 逐期 t：truncated(ds, t+1) → state → snapshot → score_round → tickets → 与 t+1 真实比对
    # 3) 自适应权重：纯内存（cfg.adaptive.in_memory=True 强制），按期幂等
    # 4) 返回 BacktestReport（§4.6），末尾 self_check

# metrics.py
@dataclass(frozen=True)
class LiftStat:
    k: int; mean_hit: float; expected: float; lift: float
    sd: float; t_stat: float; p_value: float; ci95: tuple[float, float]
    n: int            # v1.3.0：参与检验的样本期数（is_signal 的样本量门槛依据）
    alpha_adj: float  # v1.3.0：校正后显著性水平（bonferroni=True 时 = alpha_sig/len(ks)，否则 = alpha_sig）
    is_signal: bool   # v1.3.0：n >= min_n_signal 且 lift > lift_signal 且 p_value < alpha_adj（三条同时成立）
@dataclass(frozen=True)
class TierStat:
    play: str; tier: str; n_rounds: int
    observed_rate: float; theory_rate: float; ratio: float
@dataclass(frozen=True)
class RuleStat:
    rule_id: str; pool_size_avg: float; hits: int; hit_rate: float; lift: float; z: float
@dataclass(frozen=True)
class AttributionReport:
    per_rule: Mapping[str, RuleStat]
    high_votes: dict[str, float]     # {"n","hit_rate","lift","z"}，V>=3 组
    low_votes: dict[str, float]      # V<=1 组
@dataclass(frozen=True)
class BankrollStat:
    n_rounds: int; total_cost_yuan: int; total_prize_yuan: int
    net_yuan: int; roi: float
    max_drawdown_yuan: int; max_losing_streak: int; best_round_yuan: int
def lift_at_k(rows: Sequence[BacktestRow], ks: Sequence[int],
              cfg: BacktestConfig) -> dict[int, LiftStat]: ...
    # v1.1.0 新增 cfg 入参；v1.3.0：读 lift_signal/alpha_sig/min_n_signal/bonferroni 生成 is_signal
    # 禁止在实现内写死 1.05 / 0.05 / 500 / 5（K 组数取 len(ks)）
def high_tier_rate(rows: Sequence[BacktestRow], play: str,
                   prize_table: Mapping[str, Mapping[int, int] | None]) -> dict[str, TierStat]: ...
def rule_attribution(rows: Sequence[BacktestRow]) -> AttributionReport: ...
def bankroll(rows: Sequence[BacktestRow], prize_table, *, cost_per_ticket: int = 2) -> BankrollStat: ...
def t_test_one_sided(mean: float, sd: float, n: int) -> tuple[float, float]: ...   # (t, p)；n<2→(0,1)
def self_check(report: BacktestReport) -> dict[str, bool]: ...

# leakage.py
def max_accessed_t(snapshot: FeatureSnapshot) -> int: ...
def assert_no_lookahead(*, max_accessed_t: int, as_of_t: int) -> None: ...   # 违反 → LeakageError
def poison_future(ds: Dataset, as_of_period: int, *, seed: int) -> Dataset: ...
    # 把 > as_of_period 的所有期开奖号码替换为随机 20 码（测试专用）
def fingerprint_snapshot(snapshot: FeatureSnapshot) -> str: ...              # sha256(所有 80 维数组)
def audit_grep_no_shuffle(root: str | Path) -> tuple[str, ...]: ...           # 命中 shuffle/KFold 的文件:行
```
异常契约：`metrics` 全部函数对 `rows=()` 返回零值结构，不抛异常；`bankroll` 金额只用 `int`；`leakage.assert_no_lookahead` 抛 `LeakageError`。

### §3.13 `kl8/reporting/`

```python
# text_report.py
@dataclass(frozen=True)
class ReportContext:                 # 三类交付件共享入参（唯一数据源，禁止各算一套）
    cfg: AppConfig; ds: Dataset
    target_period: int; target_date: str
    snapshot: FeatureSnapshot
    rounds: ScoreRoundResult
    plans: tuple[TicketPlan, ...]
    randomness: "RandomnessReport | None"
    backtest: BacktestReport | None
    previous_review: Mapping[str, Any] | None
    warnings: tuple[str, ...]
def render_text_report(ctx: ReportContext) -> str: ...
    # 六段式：[0] 上期复盘 [1] 随机性基准 [2] 四尺度特征与动量
    #        [3] 五规则共振榜 + FinalScore Top20 [4] 金胆池/复式池 + 形态体检
    #        [5] 覆盖注单与保底 [附录A] 80 码全参 [附录B] 回测归因
def save_text_report(text: str, path: str | Path) -> Path: ...

# wechat.py
def render_wechat(ctx: ReportContext) -> str: ...      # 手机端 emoji 分块，≤1200 字
def save_wechat(text: str, path: str | Path) -> Path: ...

# excel.py
@dataclass(frozen=True)
class ExcelStyle:
    header_fill: str = "1F497D"; font_name: str = "微软雅黑"; hit_fill: str = "FFF2CC"
    sheet_names: tuple[str, ...] = ("当期总览", "80码五规则全参", "覆盖注单", "回测归因", "随机性基准", "上期复盘")
def render_excel(ctx: ReportContext, path: str | Path, *, style: ExcelStyle = ExcelStyle()) -> Path: ...
    # 失败（文件被占用）重试 3 次，仍失败记 warning 返回 Path，不中断其它交付件
```

### §3.14 `kl8/pipeline.py` 与 `main.py`

```python
@dataclass(frozen=True)
class DailyResult:
    target_period: int; target_date: str
    pool: PoolResult; plans: tuple[TicketPlan, ...]
    report_paths: Mapping[str, Path]       # keys "text","wechat","excel","state"
    backtest: BacktestReport | None
    warnings: tuple[str, ...]; elapsed_s: float

def run_daily(cfg: AppConfig, *, no_sync: bool = False, target_period: int | None = None,
              target_date: str | None = None, backtest_n: int = 30,
              quick: bool = True, progress: Callable[[str], None] | None = None) -> DailyResult: ...
def run_backtest(cfg: AppConfig, n: int, *, t0: int | None = None,
                 include_tickets: bool | None = None,
                 progress: Callable[[int, int], None] | None = None) -> BacktestReport: ...
def run_audit(cfg: AppConfig, *, n: int = 100) -> dict[str, Any]: ...

# main.py
def build_arg_parser() -> argparse.ArgumentParser: ...
def main(argv: Sequence[str] | None = None) -> int: ...   # 0 成功 / 1 参数或数据错误 / 2 内部错误 / 130 中断
```

**`main.py` CLI 参数清单（v1.2.0 与实现对齐，`python main.py --help` 为准）**

| 参数 | 类型/默认 | 语义 |
| :--- | :--- | :--- |
| `--mode {daily,backtest,audit}` | str，`daily` | 运行模式分派 `run_daily / run_backtest / run_audit` |
| `--history PATH` | str，`kl8_history.txt` | 历史数据文件路径（覆盖 `cfg.history_path`） |
| `--no-sync` | flag | 跳过网络同步 |
| `--quick` / `--full-sync` | flag | 增量/全量同步（`quick = not full_sync`） |
| `--period N` / `--date S` | int / str | 目标期号 / 目标日期 |
| `--backtest N` | int，**`None` 时按模式分派**：daily→**30**、backtest/audit→**100**；显式 `0` = daily 关闭伴随回测 | 回测期数 |
| `--no-tickets` | flag | 回测跳过注单生成与奖金核算（`include_tickets=False`，快速评估 Lift） |
| `--warmup N` | int，300 | 回测冷启动 T0（覆盖 `cfg.backtest.warmup`） |
| **`--ml`** | flag（**v1.2.0 新增**） | 开启 LightGBM 诊断通道（`ml.enabled=True`）；默认关闭，单期耗时 0.39s→2.03s |
| `--no-sweep` | flag | 兼容旧脚本的空操作（no-op） |
| `--k {5,7}` / `--budget N` / `--pool-size N` | int | 玩法码数（联动 `play`/`t`）/ 预算注数 / 复式池规模 |
| `--seed N` | int | 全局随机种子（Louvain/SA/抽样） |
| `--config PATH` | str | JSON 配置文件（深合并，CLI 优先级更高） |
| `--full-print` | flag | 控制台打印完整 TXT 报告 |

> **`.bat` 一键入口**（v1.2.0 重写）：`运行微积分预测.bat`（UTF-8 BOM + `chcp 65001`）执行 `python main.py --no-sync --backtest 30 --full-print`，双击即出全套交付件；失败分支给中文提示并 `pause`，不静默退出。
异常契约：`pipeline` 是对外唯一编排点，捕获 `KL8Error` 转中文可操作提示并返回非 0；Ctrl+C 返回 130。

#### §3.14.1 复盘状态文件契约 `output/state/prev_prediction.json`（v1.3.0 新增，修 P8-H5）

> 背景：现行交付件「前期预测复盘」用**当期金胆池 ∩ 最新一期开奖号** 冒充"上期预测命中数"（`pipeline.py`/`text_report.py`），向用户输出错误事实。修复必须引入**跨期预测持久化**，禁止任何"当期冒充上期"的替代实现。

**文件路径**：`{cfg.output_dir}/state/prev_prediction.json`（默认 `output/state/prev_prediction.json`；不新增配置项，由 `output_dir` 派生）。

**JSON 结构（`schema_version=1`，字段冻结）**

```json
{
  "schema_version": 1,
  "updated_at": "2026-11-02T18:00:00+08:00",
  "predictions": {
    "2026292": {
      "period": 2026292,
      "target_date": "2026-11-02",
      "gold": [3, 17, 28, 44, 61],
      "main": [3, 9, 17, 22, 28, 31, 40, 44, 52, 61, 68, 75],
      "plans": [{"play": "pick5", "k": 5, "budget": 20, "tickets": [[3, 9, 17, 22, 28]]}],
      "created_at": "2026-11-02T18:00:00+08:00"
    }
  }
}
```

| 要求 | 口径（可验收） |
| :--- | :--- |
| 必填字段 | 每条记录至少 `{period:int, target_date:"YYYY-MM-DD", gold:list[int], main:list[int], plans:list[dict], created_at:ISO8601}`；顶层 `{schema_version:int, updated_at:ISO8601, predictions:Mapping[str,record]}` |
| 写入时机 | `run_daily` 在**报告渲染前**把当期预测 upsert 入库；同一 `period` 重复日跑（幂等）覆盖为最新值，`n_records` 不增长 |
| 幂等 upsert | 键 = `str(period)`；已存在则整体替换该条，绝不追加重复期号 |
| 原子写 | 先写同目录临时文件（`prev_prediction.json.tmp`）→ `os.replace` 覆盖；**异常时保留旧文件**，禁止 `open(..., "w")` 直写 |
| 读取规则 | 复盘只允许读取 **`period < 当期 target_period` 的最大 period 记录**；比较对象为该 period 的**真实开奖号**（来自 `Dataset`），命中 = `len(set(record.gold) & set(numbers))` / `len(set(record.main) & set(numbers))` |
| 缺失/损坏 | 文件不存在 / JSON 非法 / `schema_version` 不匹配 / 无 `period < target_period` 记录 → 复盘段落**必须**输出固定文案「**首期无复盘基线**（无可用的上期预测记录）」，**不得**输出任何命中数、命中率或"上期预测"字样 |
| 禁止项 | **禁止用当期预测冒充上期预测**；禁止用"当期金胆 ∩ 最新开奖"充当复盘命中；禁止把 `gold ⊂ main` 重叠数当命中数 |
| 降级与告警 | 写失败 → 写入 `DailyResult.warnings`（「复盘状态写入失败：…」，不中断交付）；读失败 → 同上 + 走「首期无复盘基线」分支 |
| 报告消费 | 三类交付件（TXT/微信/Excel）的复盘段落只读 `prev_prediction` 结果，**不得各自重算**（B8 单一数据源） |

> 落地归属：`kl8/pipeline.py`（写）+ `kl8/reporting/text_report.py`/`wechat.py`/`excel.py`（读）；P8 修复轮需同步补单测 `tests/test_pipeline.py::test_prev_prediction_roundtrip_is_idempotent` 与 `test_prev_prediction_missing_shows_baseline_message`。

---

## §4 数据结构契约（字段冻结）

> 全部为 `@dataclass(frozen=True)`，字段名/顺序/单位冻结；新增字段只能追加在末尾并以 `DEFAULT` 给出（保证 `from_dict` 兼容）。所有 `np.ndarray` 长度 80、索引 `i` ↔ 号码 `i+1`。

### §4.1 `FeatureSnapshot`（layer3 唯一出口，layer4 唯一入口）

**元信息**

| 字段 | 类型 | 语义/来源 |
| :--- | :--- | :--- |
| `as_of_index` | `int` | `t`（0-based 含端点），用 `X[0..t]` 预测 `t+1`；`evaluation.max_accessed_t` 必须等于它 |
| `target_period` | `int` | `t+1` 的期号 |
| `n_history` | `int` | `t+1`（可用期数） |
| `effective_windows` | `Mapping[str, int]` | `{"10G":w_eff,"25G":…,"50G":…,"ALL":T}`（裁决 R3/`effective_window`） |
| `cold_start` | `bool` | 任一窗口 `w_eff < w` 即 True |
| `schema_version` | `str` | 固定 `"fs-v1"`（ML 缓存 key 组成部分） |
| `config_hash` | `str` | `config.config_hash(cfg)`（ML 缓存 key 组成部分） |

**连续特征（`np.ndarray`，`dtype=float64`，`shape=(80,)`）**

| 字段 | 公式/来源 | 8.md |
| :--- | :--- | :--- |
| `freq_10` | `c_i(10G)/10`（`heat.multi_scale_freq`） | L85 |
| `freq_25` | `c_i(25G)/25` | L85 |
| `freq_50` | `c_i(50G)/50` | L85 |
| `freq_all` | `c_i(ALL)/T` | L85 |
| `mom_10_25` | `freq_10 - freq_25` | L87 |
| `mom_25_50` | `freq_25 - freq_50` | L88 |
| `gap_ratio` | `O_{t,i} / max(μ_gap,i, 3.0)` | L90–92 |
| `pressure` | `exp(-0.5((gap_ratio-1.2)/0.6)^2)`，clip 到 `[0,1]` | L92 |
| `follow_10` | `Follow_j(10G)=Σ_{i∈S_t}(M_{i→j}-p0)` | L99–101 |
| `follow_25` | 同上（25G） | L99–101 |
| `follow_50` | 同上（50G） | L99–101 |
| `follow_all` | 同上（ALL） | L99–101 |
| `co_10` | `CoScore_j(10G)` | L109–111 |
| `co_25` | `CoScore_j(25G)` | L109–111 |
| `co_50` | `CoScore_j(50G)` | L109–111 |
| `co_all` | `CoScore_j(ALL)` | L109–111 |
| `co_centered_mean` | **原始值** `mean(co_25, co_50, co_all)`（仅供报告展示；打分用秩空间 `C̄o`，见裁决 A2） | L143 |
| `pagerank` | 正值共现图 PageRank（`graph.pagerank`，归一化到 `[0,1]`） | L112 |
| `eigen` | 特征向量中心度（归一化到 `[0,1]`） | L112 |
| `community_strength` | 所属社区内平均正共现强度（无社区时 0） | L19/L112 |
| `gap_mean` | `μ_gap,i`（`heat.historical_mean_gap`，μ=0 时取 3.0） | L90 |
| `follow2_all` | 二阶 `M^(2)` 的 ALL 窗跟随分（`cfg.markov.order=1` 时全 0） | L102 |
| `ml_prob` | PAV 校准后开出概率 ∈[0,1]；`ml.enabled=False` 时全为 0.25 | L20/L37 |

**离散/整型/布尔特征（`shape=(80,)`）**

| 字段 | 类型 | 语义 |
| :--- | :--- | :--- |
| `gap` | `int32` | 当前遗漏 `O_{t,i}`（0 = 上期开出） |
| `golden_cross` | `bool` | 动量金叉加速（`mom_10_25>0 且 mom_25_50>0 且 mom_10_25>mom_25_50`） |
| `community_id` | `int32` | Louvain 社区编号；`-1` = 孤立点/无图 |
| `community_size` | `int32` | 所属社区节点数（孤立点 1） |
| `triad_count` | `int32` | 该号码参与的三元共现团个数 |
| `zone4` | `int8` | 0..3（四区） |
| `zone8` | `int8` | 0..7（八区） |
| `is_odd` | `bool` | 奇数 |
| `is_prime` | `bool` | 质数（1 为 False） |
| `size_class` | `int8` | 0=小 1=中 2=大（裁决 A3） |
| `tail` | `int8` | `num % 10`（同尾判定） |
| `road012` | `int8` | `num % 3` |

**映射/附属字段**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `follow_diag` | `Mapping[str, np.ndarray]` | 重号（对角）惯性分，键 `"10G","25G","50G","ALL"`，各 `(80,)` |
| `follow_offdiag` | `Mapping[str, np.ndarray]` | 异号跟随分，键同上 |
| `companion` | `Mapping[int, tuple[tuple[int, float], ...]]` | 号码 → 前 3 伴生码及归一化亲和度（`1.0` 为最强） |
| `co_matrices` | `Mapping[str, np.ndarray]` | 键同上，各 `(80,80)` 中心化共现矩阵 `A_ij` |
| `triads_total` | `int` | 全局三元共现团总数 |
| `ml_available` | `bool` | LightGBM 是否可用（降级时 False） |
| `ml_backend` | `str` | `"lightgbm"` / `"ema"` / `"uniform"` |
| `baseline` | `BaselineSnapshot` | 随机性基准摘要（§4.7） |
| `warnings` | `tuple[str, ...]` | 降级/退化/冷启动说明（中文） |

### §4.2 `RuleVerdict`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `rule_id` | `str` | `"R1"`..`"R5"`（固定顺序） |
| `rule_name` | `str` | 中文名：多尺度强跟随共振 / 温热多头趋势共振 / 动量金叉加速 / 强共现伴生拉动 / 全量回归临界回补 |
| `mask` | `np.ndarray` (`bool`, 80) | 该规则命中掩码 |
| `pass_count` | `int` | `mask.sum()`，0..80 |
| `thresholds` | `Mapping[str, float]` | 生效阈值快照（来自 `cfg.rules`，供报告与审计复现） |

### §4.3 `PoolResult`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `gold` | `tuple[int, ...]` | 金胆池，1-based 号码，升序；长度 ∈ [`gold_min`,`gold_max`]=[3,5] |
| `main` | `tuple[int, ...]` | 复式池，1-based 号码，升序；长度 ∈ [10,15]，默认 12（裁决 A6） |
| `votes` | `np.ndarray` (`int8`, 80) | 交集共振票数 `V_i ∈ {0..5}` |
| `scores` | `np.ndarray` (`float64`, 80) | FinalScore |
| `order` | `tuple[int, ...]` | 全部 80 码按 FinalScore 降序（并列按号码升序） |
| `details` | `Mapping[int, NumberDetail]` | 80 码逐号明细（§4.3.1），报告唯一数据源 |
| `warnings` | `tuple[str, ...]` | 约束放宽/降级说明（中文，必须可读） |
| `relaxed` | `tuple[str, ...]` | 已放宽的约束键（`"same_tail"/"parity"/"zone4"/"repeat_ratio"`） |
| `feasible` | `bool` | 是否在全部硬约束下达成目标规模（False 表示已放宽） |
| `repeat_count` | `int` | 池内上期重号个数（`main` 与上期开奖号交集大小） |
| `new_count` | `int` | `len(main) - repeat_count` |
| `size` | `int` | `len(main)` |

**§4.3.1 `NumberDetail`**：`num:int`、`final:float`、`base:float`、`votes:int8`、`rank:int`（1=最高）、`zone4:int`、`zone8:int`、`is_odd:bool`、`is_prime:bool`、`size_class:int`、`tail:int`、`is_repeat:bool`、`r1..r5:bool`、`freq_10/25/50/all:float`、`pressure:float`、`gold_flag:bool`、`in_main:bool`。

### §4.4 `TicketPlan`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `play` | `str` | `"pick5"` / `"pick7"`（内部键）；`play_label` 为中文「选五」「选七」 |
| `k` | `int` | 单注号码个数（选五 5，选七 7） |
| `budget` | `int` | 预算注数（默认选五 20、选七 10；裁决 A8） |
| `tickets` | `tuple[tuple[int, ...], ...]` | 注单，**1-based 号码**、每注升序、注间不重复；长度 `≤ budget` |
| `guarantee_level` | `Mapping[str, Any]` | `{"t":4,"m":5,"text":"出5保4（池中任意 5 码至少 1 注中 4）","coverage_ratio":0.83,"verified":true}` |
| `covering` | `CoveringReport` | 求解明细（§4.5），含 method/降级原因 |
| `cost_yuan` | `int` | `len(tickets) * cfg.cost_per_ticket`（整数元） |
| `method` | `str` | `"table"` / `"ilp"` / `"greedy"` / `"anneal"` |
| `warnings` | `tuple[str, ...]` | 预算无法同时满足保底时的降级说明 |

### §4.5 `CoveringReport`

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `pool` | `tuple[int, ...]` | 号码池（1-based，升序），`v = len(pool)` |
| `v` / `k` / `t` / `m` | `int` | 覆盖参数；保证 `t ≤ k`、`t ≤ m ≤ v`（违反 → `InfeasibleConstraintError`） |
| `tickets_pos` | `tuple[tuple[int, ...], ...]` | 位置索引注单（0-based，`0..v-1`）——覆盖计算的内部真值 |
| `tickets` | `tuple[tuple[int, ...], ...]` | 映射到号码后的注单（1-based），与 `tickets_pos` 逐注对应 |
| `n_tickets` | `int` | 注数 |
| `coverage_ratio` | `float` | 已覆盖 m 子集比例 ∈[0,1] |
| `achieved_t` | `int` | 实测最小 `\|B∩M\|`（穷举结果，必须 `≥ t`） |
| `verified` | `bool` | `verify_guarantee` 通过标志（未通过禁止出单） |
| `method` | `str` | `"table"`/`"ilp"`/`"greedy"`/`"anneal"`/`"greedy+anneal"` |
| `degraded` | `bool` | 是否发生降级（ILP 超规模/超时 → 贪心；查表未命中 → 求解器） |
| `degrade_reason` | `str` | 降级原因（中文；未降级为空串） |
| `elapsed_ms` | `int` | 求解耗时（毫秒） |
| `warnings` | `tuple[str, ...]` | 其它告警 |

### §4.6 `BacktestRow` / `BacktestReport`

**`BacktestRow`（一期一行）**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `period` | `int` | 被预测期（`t+1` 的真实期号） |
| `as_of_index` | `int` | `t`（0-based） |
| `as_of_period` | `int` | `t` 的期号 |
| `scores` | `np.ndarray` (`float64`, 80) | 当期 FinalScore |
| `votes` | `np.ndarray` (`int8`, 80) | 当期票数 |
| `top_picks` | `Mapping[int, tuple[int, ...]]` | 键为 `cfg.backtest.ks`（3/5/7/10/12），值为排名前 K 的号码（1-based，降序） |
| `hit_at` | `Mapping[int, int]` | Top-K 命中个数 |
| `gold` / `main` | `tuple[int, ...]` | 当期金胆池/复式池（1-based） |
| `gold_hit` / `main_hit` | `int` | 池命中个数 |
| `tickets` | `tuple[tuple[int, ...], ...]` | 当期注单（1-based；`include_tickets=False` 时为空） |
| `best_ticket_hit` | `int` | 单注最高命中数（无注单为 0） |
| `cost_yuan` / `prize_yuan` | `int` | 当期投入 / 奖金（整数元；选七等未确认奖级 → `prize_yuan=0` 且 `prize_known=False`） |
| `prize_known` | `bool` | 该玩法奖金表是否已确认（裁决 A10） |
| `ok` | `bool` | 该期数据完整（历史够、无解析告警） |
| `warnings` | `tuple[str, ...]` | 该期降级/异常说明 |

**`BacktestReport`**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `n_sample` | `int` | 实际回测期数（= `len(rows)`） |
| `t0` | `int` | 冷启动期数（默认 300） |
| `start_period` / `end_period` | `int` | 回测覆盖期号区间 |
| `rows` | `tuple[BacktestRow, ...]` | 逐期明细 |
| `lift` | `Mapping[int, LiftStat]` | 键为 K |
| `high_tier` | `Mapping[str, TierStat]` | 键形如 `"pick5:5"`、`"pick5:4+5"`、`"pick3:3"` |
| `attribution` | `AttributionReport` | 规则归因（§3.12） |
| `bankroll` | `BankrollStat` | 资金曲线 |
| `config_hash` | `str` | 血缘标识 |
| `self_check` | `Mapping[str, bool]` | 口径自洽位（`n_sample==len(rows)`、`lift` 可重算、`bankroll` 可重算、`no_lookahead`） |
| `elapsed_s` | `float` | 回测总耗时（秒） |
| `warnings` | `tuple[str, ...]` | 全局告警 |

### §4.7 其它共享结构（字段一览）

| 结构 | 字段 |
| :--- | :--- |
| `BaselineSnapshot` | `z_10G/z_25G/z_50G/z_ALL: ndarray(80)`、`significant_any: bool(80)`、`runs_z: ndarray(80)`、`runs_labels: tuple[str,...](80)`、`chi2: Mapping[str, ChiSquareResult]`、`nist: tuple[NistResult,...]`、`entropy: Mapping[str,float]`、`mi_top_pairs: tuple[tuple[str,str,float],...]`、`expected_repeat: float` |
| `RandomnessReport` | `n_history: int`、`p0: float`、`z_by_window: Mapping[str,ndarray(80)]`、`significant_count: int`、`chi2: Mapping[str,ChiSquareResult]`、`nist: tuple[NistResult,...]`、`entropy_80: float`、`entropy_max: float`、`mi_top_pairs: tuple[...]`、`runs_summary: Mapping[str,int]`、`notes: tuple[str,...]` |
| `MorphologyStats` | 见 §3.9 `morphology.MorphologyStats`（`n/zone4/zone8/odd/even/prime/composite/size/consecutive/same_tail/sum_value/sum_z/span/ac`） |
| `AdaptiveState` | `schema_version/period/weights(R1..R5)/hit_rates/n_updates/updated_at` |
| `ScoreResult` | `base/final/order/contributors` |
| `ScoreRoundResult` | `snapshot/ranks/verdicts/votes/scores/pool` |
| `SyncResult` | `ok/added/total/latest_period/warnings` |
| `DailyResult` | `target_period/target_date/pool/plans/report_paths/backtest/warnings/elapsed_s` |
| `PrevPredictionState`（v1.3.0） | 非内存结构（JSON 状态文件，见 §3.14.1）：`schema_version: int`、`updated_at: str`、`predictions: Mapping[str, PrevPredictionRecord]`；`PrevPredictionRecord` = `period/target_date/gold/main/plans/created_at` |

---

## §5 8.md 未定义处的显式裁决表（A1–A10，冻结）

> 说明：8.md 只给符号不给定义、或同一对象两处规模不一致的，一律以本表为准；下游开发**不得自行另定**。每条给出「原文位置 → 裁决 → 理由/兼容性 → 落点」。

| 编号 | 8.md 原文缺口 | 裁决（可执行口径） | 理由 / 兼容性 | 落点 |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | L143 `R(Trend_i)` 未定义 Trend | `Trend_i = 0.5·R(Freq_10G) + 0.3·R(Mom_10-25) + 0.2·R(Mom_25-50)`，**秩空间**加权（先各自取百分位秩再加权） | 与 8.md「跨尺度量纲统一」+「动量金叉」一致；权重可配（`weights.trend_w_*`），三者同为 (0,1] 量纲，避免频率/差分混量纲 | `hybrid.trend_score`；`tests/test_scoring.py::test_trend_score_weights` |
| **A2** | L143 `R(C̄o_i)` 的 C̄o 聚合方式未定义 | `C̄o_i = mean(R(Co_25G), R(Co_50G), R(Co_ALL))`，**先秩后均**，**不含 10G** | 10G 共现样本仅 10 期，`CoScore` 方差最大，纳入会主导排序；25/50/ALL 覆盖中短中长期，与 R4 的窗口选择一致。**※ 口径冻结（v1.3.0）**：`C̄o` 必须是**等权先秩后均**（÷3），**实现与测试均不得改为加权式**（历史事故：曾改为 `0.20/0.20/0.60` 并在 `tests/test_qa_numerics.py` 把它标成"裁决 A2"断言）；**本口径自 v1.0.0 起未变更，任何加权版本均为偏离** | `hybrid.co_aggregate`；`FeatureSnapshot.co_centered_mean` 只作报告展示原始值，**禁止**替代 `R(C̄o)` |
| **A3** | L117「大中小比」未给分界 | 小 `01-27`、中 `28-54`、大 `55-80`（27/27/26）；「大小比」保留老口径 `01-40 / 41-80` 作为报告附加字段，**不参与均衡硬约束** | 8.md 未定义，等分最稳；老系统 1–40/41–80 用于兼容报告阅读习惯，但不作为过滤条件 | `constants.SIZE_BOUNDS`、`size_class_of`；`tests/test_morphology.py` |
| **A4** | L118「AC 值」未给公式 | `AC = \|{ \|a-b\| : a<b, a,b ∈ 注单 }\| - (k-1)`；单注 k 码最小 0；对 20 码开奖号同式（k=20） | 标准 AC 定义（算术复杂度）；8.md「衡量号码间距的离散度」即差集势减线性项 | `morphology.ac_value`；`tests/test_morphology.py::test_ac_value` |
| **A5** | L155「金胆池（如 3~5 码）」未给选取规则 | `n_gold = clip(round(0.30 × len(main)), 3, 5)`（`main=12` → **4**）；候选顺序：先取 `V_i ≥ 3` 的号码按 FinalScore 降序，不足再用 FinalScore 降序补；**硬约束 `gold ⊂ main`** | 大小随复式池自适应且落在 3~5 区间；优先共振票高的号码符合"核心金胆"语义。**※ 契约自定声明**：`0.3` 系数、`V_i ≥ 3` 优先序与 `gold ⊂ main` 均为**契约自定**，8.md L155 仅给示例区间「如 3~5 码」，**非原文要求** | `filter.select_gold`；`tests/test_filter.py::test_gold_and_main_sizes` |
| **A6** | L155「10~15 码」 vs L202「10~12 码」 冲突 | **同属一个复用对象**：`main` 池默认 **12**，合法区间 10~15；L202 的 10~12 是"日常下发玩法变体"（选五取 12、选七取 10），不是第二个池 | 消除两处规模表述冲突；默认 12 同时落在两个区间内，兼容两处原文 | `PoolConfig.main_default/main_min/main_max`；`filter.build_pool` |
| **A7** | L161–166 覆盖参数 `m`：原文为"**预期**候选池 v 中能够实际命中的开奖号码个数"，未给取值 | **m 定性（必读）**：`m` 是**用户/玩法假设参数**（8.md L164 的"预期"），**非算法推导值、非预测值**；报告与文案必须写「m 为假设参数」。**取值规则（v1.1.0 修正）**：① `t ≥ k` → `m = min(v, t)`（全保底场景）；② `t < k` → `m = min(v, t + cfg.covering.m_margin)`，`m_margin` 默认 **1**、范围 **[1,3]**；③ 合理性校验：`abs(m - (ceil(0.25·v)+2)) > 2` → 写 warning（提示 m 假设与池内期望命中数偏离，报告须注明）。玩法默认：选五 `C(v,5,4,m)`（`v=12 → m=5 = C(12,5,4,5)`）、选七 `C(v,7,5,m)`（`v=12 → m=6 = C(12,7,5,6)`）；强制 `t ≤ k`、`t ≤ m ≤ v` | 精确复现 8.md L169 全部命名参数组（验算见 §5.0）；`0.25v+2` 只作合理性校验而非定义 —— 原式与 k/t 无关，会把 pick7 算成 m=5，与原文 `C(12,7,5,6)` 冲突 | `covering.m_target`；`tests/test_covering.py::test_m_target` |
| **A8** | L170「固定只买 20 注」示例，无默认预算 | 默认预算：**选五 20 注（40 元）、选七 10 注（20 元）**；`--budget` 覆盖；硬上限 `max_budget=200` 注（400 元），超出抛 `ConfigError` | 沿用旧系统 20 注精选版习惯；选七单注成本高故 10 注；上限防误配。**※ 契约自定声明**：选七 **10 注**为契约自定（8.md L170 仅给"固定只买 20 注"示例），**非原文要求** | `CoveringConfig.budget_by_play/max_budget`；`main.py --budget` |
| **A9** | L148「M=5 或 10」未定 | 自适应回看期 **M=8**；`min_periods=3`（不足回均匀先验 `0.2/规则`）；Laplace `α=1.0`；temperature `1.0`；权重下限 `0.05`；**回测强制纯内存**（`in_memory=True`），日跑按 period 幂等 upsert | M=8 取两档中值，兼顾风格切换灵敏度与样本量（8 期×每规则约 3 码 ≈ 24 观测）；下限防单规则归零。**※ 契约自定声明**：8.md L148 仅示例「如 M=5 或 10」，**8 不在原文示例集合内**，为经 Lead 批准的契约裁决值，**非原文要求** | `AdaptiveConfig`；`tests/test_adaptive.py` |
| **A10** | L193–194「结合官方固定设奖规则」未给奖金表 | **只启用已确认奖级**：选五 `{中5:1000 元, 中4:21 元, 中3:3 元}`，单注 2 元；选三、选七奖金表为 `None` → 其高奖级只报**触发频次与超几何理论概率**，不出金额；ROI/回撤/连挂**仅对选五出具** | 未核实的奖级进金额计算会产出误导性 ROI；8.md 自身只给"选五全中理论概率 0.155%（约 645 期一遇）"这一确定值（**该数值为 8.md L190 笔误，精确值 0.0645% ≈ 1/1550 期，以 §7.4 errata 为准**）。**报告硬性措辞**：选三/选七段落必须输出 `ROI: N/A（官方奖级未确认）`，**禁止任何金额结论或暗示性收益表述** | `AppConfig.prize_table`；`metrics.bankroll/high_tier_rate`；`tests/test_metrics.py` |

### §5.0 A7 参数组验算（v1.1.0 修正后，必须逐条复现）

| 8.md L169 参数组 | v | k | t | 修正后规则 | m | 结论 |
| :--- | ---: | ---: | ---: | :--- | ---: | :--- |
| `C(12,5,4,5)` | 12 | 5 | 4 | `t<k` → `min(12, 4+1)` | **5** | ✓ 与原文一致 |
| `C(12,7,5,6)` | 12 | 7 | 5 | `t<k` → `min(12, 5+1)` | **6** | ✓ 与原文一致（旧式会算成 5，已修正） |
| `C(15,5,4,5)` | 15 | 5 | 4 | `t<k` → `min(15, 4+1)` | **5** | ✓ |
| `C(15,7,5,6)` | 15 | 7 | 5 | `t<k` → `min(15, 5+1)` | **6** | ✓ |
| `C(10,5,5,5)` | 10 | 5 | 5 | `t≥k` → `min(10, 5)` | **5** | ✓ |

> 验算用例写入 `tests/test_covering.py::test_m_target`：上表 5 组逐一断言 `m_target(v,k,t)==m`，外加上界 `m ≤ v`、`m ≥ t`、`m ≤ k+?` 无强制（只要求 `t ≤ m ≤ v`）。

### §5.1 补充口径（B 类：8.md 未提及但落地必须统一，非裁决表 10 条）

| 编号 | 口径 |
| :--- | :--- |
| B1 | 窗口键名全仓唯一写法：`"10G"`、`"25G"`、`"50G"`、`"ALL"`（大小写敏感，`Mapping` 键即此四字符串；`ALL` 无 `G` 后缀） |
| B2 | 五条规则阈值**一律从 `cfg.rules` 读取**；单测通过构造自定义 `RuleConfig` 造边界，禁止改源码常量（裁决阈值：`0.65 / 0.649`、`0.24` 边界等） |
| B3 | 社区检测：`networkx` 可用走 `louvain_communities(seed=cfg.seed)`；不可用/异常 → 降级 `numpy` 幂迭代 PageRank + 连通分量社区，`GraphResult.backend` 记录 `"networkx"/"numpy_power"` 并写 warning；无图时 `community_id=-1` |
| B4 | ML 为**默认关闭的诊断通道（v1.2.0 对齐实现）**：`ml.enabled=False`（默认）、`ml.weight=0.0`；`enabled` 只决定是否训练（关：`ml_prob` 恒 0.25、`ml_available=False`、单期 0.34s；开：≈2.03s），诊断时用 CLI `--ml`。`ml.weight` 是 **BaseScore 的额外叠加项**：七维核心权重 `Σω=1.00` 恒成立且**不含 ml**；`ml.weight>0` 时叠加 `ml·R(ml_prob)`，**不再归一化**（BaseScore 上界 >1，报告需注明）。`ml_prob` 必须 ∈[0,1]，不强制均值归一（PAV 校准后允许偏离 0.25）。**8.md L37 对齐说明**：L37 的「机器学习概率校准模型」以**可配置项交付**（能力具备、默认关闭），启用后进入 BaseScore |
| B5 | 自适应"规则权重"作用于票数加成：`FinalScore = Base + λ·Σ_r (w_r / w̄)·pass_r`，其中 `Σ_r w_r = 1`、`w̄ = 1/5`；**`w_r` 全均匀时恒等于 8.md 原式 `Base + λ·V_i`**。为此 §3.10 `hybrid.score_all` 签名追加 keyword-only 参数 `rule_weights: Mapping[str, float] \| None = None`（v1.0.1 补充，向后兼容；`final_score()` 保持原式不变） |
| B6 | 重号配比：`repeat_target = clip(round(cfg.pool.repeat_ratio × size), ceil(0.25×size), floor(0.35×size))`；`size=12` → 目标 4（区间 [3,4]）；先按 FinalScore 取足上期重号，再补新号 |
| B7 | 约束放宽顺序固定 `same_tail → parity → zone4 → repeat_ratio`，每次放宽必须在 `PoolResult.relaxed` 留痕并写中文 `warnings`；全部放宽仍不达标 → `InfeasibleConstraintError` |
| B8 | **报告单一数据源**：TXT/微信/Excel 三类交付件只能消费 `ReportContext`（同一 `ScoreRoundResult` + `TicketPlan` + `BacktestReport`），禁止任何交付件自行重算指标（保证"数值与引擎结果严格一致"） |
| B9 | 期号/日期：日期一律 `"YYYY-MM-DD"` 字符串，期号一律 `int`；报告标题中的期号为"目标期 = 最新已开期 + 1" |
| B10 | `cfg.seed` 是唯一随机源（Louvain / SA / ML / 抽样）；禁止使用全局 `random`/`np.random` 无 seed 调用 |

---

## §6 DoD 与验证

### §6.1 分层 DoD（每层可独立 import 通过）

| 层级 | 交付物 | 独立自检命令（必须 exit 0） |
| :--- | :--- | :--- |
| L0 基础设施 | `config/constants/datatypes/errors/history/sync` | `python -c "import kl8.config, kl8.constants, kl8.datatypes, kl8.errors, kl8.history, kl8.sync"` |
| L1 表征 | `layer1_representation/{state,windows}` | `python -c "import kl8.layer1_representation.state, kl8.layer1_representation.windows"` |
| L2 基准 | `layer2_baseline/*` | `python -c "import kl8.layer2_baseline.priors, kl8.layer2_baseline.zscore, kl8.layer2_baseline.runs, kl8.layer2_baseline.chisq, kl8.layer2_baseline.randomness, kl8.layer2_baseline.entropy"` |
| L3 特征 | `layer3_features/*` | `python -c "import kl8.layer3_features.features as f; print(f.SCHEMA_VERSION)"` |
| L4 打分 | `layer4_scoring/*` | `python -c "import kl8.layer4_scoring as s; print(s.score_round.__name__)"` |
| L5 覆盖 | `layer5_combinatorial/*` | `python -c "import kl8.layer5_combinatorial as t; print(t.build_tickets.__name__)"` |
| E 回测 | `evaluation/*` | `python -c "import kl8.evaluation.walkforward, kl8.evaluation.metrics, kl8.evaluation.leakage"` |
| R 报告 | `reporting/*` | `python -c "import kl8.reporting.text_report, kl8.reporting.wechat, kl8.reporting.excel"` |
| P 编排 | `pipeline.py` + `main.py` | `python -c "import kl8.pipeline"`；`python main.py --help` |

**通用 DoD（全阶段）**

1. **编译**：`python -m compileall kl8 main.py tools` exit 0，无 warning 级语法问题。
2. **单测**：`python -m unittest discover -s tests -v` 全绿，用例数 ≥90（P7 口径）。
3. **需求映射**：§1 主表 48 行、每行都有实现点与测试名；P7 必须逐行找到对应用例（缺失即不通过）。
4. **数据保真**：`load_dataset("kl8_history.txt").T == 2112`；**升序排序后** `draws[0].period == 2020001`（最早）、`draws[-1].period == 2026291`（最新）；且**降序文件的首行期号 == `draws[-1].period`**；warnings 中**不得出现**"期号断档"之外的解析错误（period 断档仅 warning，见 §0.1）。**（v1.3.0 勘误：v1.2.0 之前此处把 `draws[0]`/`draws[-1]` 写反，与 `tests/test_history.py` 断言冲突，已按实现改正）**
5. **防穿越**：`tests/test_leakage.py` 未来投毒后当期推荐逐位不变（§6.3）。
6. **无残留**：`grep -rn "calc_core\|calc_engine\|..." --include=*.py`（P9 执行）无引用。
7. **中文文档**：所有公开符号有中文 docstring（审大关红线 12）。

### §6.2 数值手算对照口径（单测必须按此断言，容差 `1e-9`）

| 量 | 手算口径 | 期望值 |
| :--- | :--- | :--- |
| `p0` | `20/80` | `0.25` |
| `p_co` | `20*19/(80*79)` | `0.060126582278481012...` |
| `E(Repeat)` | `20 * 0.25` | `5.0` |
| 和值期望 | `20 * (1+80)/2` | `810`（`SUM_EXPECT`） |
| `Z_i(w)` | `w=10, c=5 → (5-2.5)/sqrt(10*0.25*0.75)` | `1.8257418583505538` |
| 游程 `E[R]` | `2*n1*n0/n + 1`（`n=n1+n0`） | 按样本手算 |
| `Pressure` 峰值 | `O/μ = 1.2` | `1.0`；`O/μ=0.6` → `exp(-0.5)=0.6065...` |
| 马尔可夫 α 平滑 | `(Σ X_{τ-1,i}X_{τ,j} + α*0.25)/(Σ X_{τ-1,i} + α)` | 逐项手算对照 |
| `A_ij` | `C_ij/w - p_co` | 手算 |
| Borda | `Rank/80`，最大 `1.0`，最小 `0.0125`，并列取平均秩 | 手算 |
| `FinalScore` 恒等 | `Final - (Base + λ·V)` | `< 1e-12` |
| `Trend` | `0.5*R(F10)+0.3*R(Mom10-25)+0.2*R(Mom25-50)` | 手算（裁决 A1） |
| `AC` | `\|{差}\| - (k-1)`；例 `{1,4,6,9,11}` 的差集 `{2,3,5,7,8,10}`（6 个）→ `AC = 6-(5-1) = 2` | 手算（裁决 A4） |
| `Lift@K` | `mean_hit_K / (0.25*K)` | 手算 |
| `is_signal`（v1.3.0） | `n >= cfg.backtest.min_n_signal(500)` 且 `lift > cfg.backtest.lift_signal(1.05)` 且 `p_value < alpha_adj`；`alpha_adj = alpha_sig/len(ks) = 0.01`（`bonferroni=True`） | 手算（8.md L188「500 期以上」；实现禁止写死 1.05/0.05/500/5） |
| 选五全中概率（**v1.3.0 勘误**） | `C(20,5)/C(80,5) = C(75,15)/C(80,20)` | `15504/24040016 = 0.000644902...`（`≈0.000645`）→ **0.0645%，约 1/1550 期**；8.md L190 的「0.155%、约 645 期」为笔误，见 **§7.4 errata E1** |
| ROI | `(奖金 - 成本) / 成本`，金额整数元 | 手算 |
| 覆盖保底 | `v=6..10` 时穷举全部 `C(v,m)` 子集 | `achieved_t ≥ t` 且 `n_tickets ≤ budget` |

### §6.3 防穿越口径（三条硬等价，单测逐条实现）

1. **等价性**：`fingerprint_snapshot(build(snapshot, t, ds_full))` == `fingerprint_snapshot(build(snapshot, t, truncated(ds, t+1)))` —— 证明 `truncated` 是唯一截断口径。
2. **未来投毒不变性**：`poison_future(ds, t_period, seed)` 污染 `>t` 的全部期后，位置 `t` 的 snapshot 指纹、`score_round` 的 `final` 数组、`pool.main`、注单**逐位完全相同**。
3. **索引守卫**：`assert_no_lookahead(max_accessed_t=snapshot.as_of_index, as_of_t=t)`；遍历全仓无 `shuffle`/`KFold` 型交叉验证（`audit_grep_no_shuffle`）。

### §6.4 验证命令清单（每阶段收尾必跑，命令与 exit code 一并上报 Lead）

```powershell
# 0) 工作目录
cd C:\D-pan\Dpanqianyi\微积分分区

# 1) 编译检查（P2 起每阶段）
python -m compileall kl8 main.py tools

# 2) 全量单测（P2 起每阶段；P7 要求 ≥90 用例全绿）
python -m unittest discover -s tests -v

# 3) 生成/重验覆盖表（P5 起）
python tools/build_covering_tables.py

# 4) 日常日跑（P6 起；离线，不联网；daily 模式伴随 30 期回测）
python main.py --no-sync --backtest 30 --full-print

# 5) 纯回测 / 审计模式（默认 100 期；--no-tickets 跳过注单，秒级评估 Lift）
python main.py --mode backtest --no-sync --no-tickets
python main.py --mode audit --no-sync

# 6) 需求映射机检（48 行）
python -c "import csv;rows=list(csv.DictReader(open(r'docs/dsh-matrix/20261102-kl8-five-layer/requirement-mapping.csv',encoding='utf-8')));print(len(rows));assert len(rows)==48"

# 7) 依赖方向 / 防穿越 grep 断言（P7/P8）
python -m unittest tests.test_architecture -v
```

### §6.5 性能预算（v1.2.0 按实测修订；超预算必须给优化证据，否则 P7 阻断）

| 场景 | 预算 | 实测（v1.2.0，2026-11-02） | 测量方式 |
| :--- | :--- | :--- | :--- |
| 单期核心（特征→打分→池，不含注单） | ≤0.35 s | 0.39 s（含注单+报告口径） | `time.perf_counter` 包 `score_round`，P7 记录均值 |
| 单期全链路（含注单 + 报告渲染） | ≤0.50 s | **0.39 s** | `run_daily` 单期耗时 |
| 100 期 Walk-Forward（`--no-tickets`） | ≤45 s | **37.4 s**（优化前 176.5 s） | `python main.py --mode backtest --no-sync --no-tickets --backtest 100` |
| 100 期 Walk-Forward（含注单） | ≤60 s | **40 s**（优化前 >600 s） | `python main.py --mode backtest --no-sync --backtest 100` |
| 500 期 Walk-Forward | ≤180 s | 待 P7 补测 | `run_backtest(cfg, 500)` |
| 全量单测（124 用例） | ≤180 s | **14.6 s**（优化前 105.6 s） | `python -m unittest discover -s tests` 总耗时 |
| 覆盖求解 `v=12,k=5,t=4,m=5` | ≤3 s（含 ILP 降级） | 0.006 s（表命中路径） | `CoveringReport.elapsed_ms` |
| ML 诊断通道开启（`--ml`） | ≤2.5 s/期 | 2.03 s（默认关闭，不进入日常预算） | `time.perf_counter` 对比开/关 |
| 单期内存峰值 | ≤300 MB | 待 P7 补测 | P7 用 `tracemalloc` 抽查 |

### §6.6 报告交付禁止项（v1.3.0 新增，修 P8-H6/M5；**可 grep 验收**）

> 适用于三类交付件（TXT / 微信 / Excel）。P8 体检卡按下列 3 条逐项 grep，**命中即判 High**。验收命令按「**排除零命中**」执行（`-NoMatch` 或 `!` 断言）。

| # | 禁止项 | 违规示例（P8 实测命中） | 判定方式 |
| :--- | :--- | :--- | :--- |
| 1 | **占位符**：未替换的模板符号 | `C(v,k,t)`、`{t}`、`<period>`、`TODO`、`N/A 待填` | 命令 B1，命中数必须 = 0 |
| 2 | **硬编码常量**：渲染层写死的口径值（必须取 `constants.py` / 实际数据） | `25.00%`、`6.01%`、`E(Repeat)` 写死 `5.00`、和值区间写死 `150~250`、金胆写死 `(5码)`、`20/80` | 命令 B2，命中数必须 = 0 |
| 3 | **未标注 ROI 的金额结论**：选三/选七出现金额、收益、"回本"等 | 选七段落出现 `ROI: 12.5%`、`净收益 30 元`、"盈利" | 命令 B3（必须存在 N/A）+ 命令 B4（禁止收益字样） |

**验收命令（在仓库根目录执行；每条命中数必须 = 0，否则 P8 判 High）**

```powershell
# B1 占位符
Select-String -Path output/reports/*.txt,output/wechat/*.txt -Pattern 'C\(v,\s*k,\s*t\)|\{t\}|\{period\}|TODO|待填'

# B2 渲染层硬编码常量
Select-String -Path output/reports/*.txt,output/wechat/*.txt -Pattern '25\.00%|6\.01%|150~250|（5码）|\(5码\)|20/80'

# B3 选三/选七必须标注 ROI N/A（命中数必须 ≥1）
Select-String -Path output/reports/*.txt,output/wechat/*.txt -Pattern 'ROI: N/A（官方奖级未确认）'

# B4 禁止收益/回本类暗示（命中数必须 = 0）
Select-String -Path output/reports/*.txt,output/wechat/*.txt -Pattern '净收益|盈利|回本|投资回报率'

# B5 Excel 单元格内同样受限（打开工作簿后遍历单元格，断言同上 4 条全 0）
python -c "import openpyxl,re,glob;[print(p) for p in glob.glob('output/excel/*.xlsx')]"
```

**补充硬性要求**
1. **数值必须来自引擎**：报告中的 `p0`、`p_co`、`E(Repeat)`、金胆规模、和值区间等一律读 `constants.py` / 结果对象（B8 单一数据源），禁止渲染层自带副本；
2. **显著性披露**：`is_signal=False` 时禁止出现「★显著 / 显著超额」字样；`n<500` 必须输出「样本量不足（n<500），结论仅供参考」（R43）；
3. **复盘披露**：无上期记录时必须输出「首期无复盘基线」，禁止任何"上期预测命中 X 码"（§3.14.1）；
4. **m 假设披露**：覆盖段落必须写「m 为用户/玩法假设参数，非算法预测」（A7）；
5. **验收示例（P8 修复轮）**：`python main.py --no-sync --backtest 30 --full-print` 后，对 `output/reports/*.txt` 执行上表 3 条 grep，全部 0 命中；且 `grep -c 'ROI: N/A（官方奖级未确认）'` 在含 pick3/pick7 的报告中 ≥1。

---

## §7 附录

### §7.1 `requirement-mapping.csv` 列定义（机检用）

```
需求ID,分组,需求摘要,8md行号,实现模块,实现符号,测试文件,测试用例名
R1,A 数据表征层,layer1 BinaryStateMatrix,L30;L47-L50,kl8/layer1_representation/state.py,BinaryStateMatrix.from_dataset,tests/test_state.py,test_binary_matrix_shape_and_row_sum
...
```

- 行数必须 = 48；`需求ID` 必须恰好覆盖 `R1..R48` 无重复无缺号（P7/P9 用 `Import-Csv` 或 Python `csv` 断言）。
- 与 §1.2 主表内容一致；两者不一致时以 `contract.md` §1.2 为准并修正 CSV。

### §7.2 契约与下游任务的对应关系

| 阶段 | 消费本契约的章节 | 交付 |
| :--- | :--- | :--- |
| P2 | §0、§1（R1–R4/R48-a）、§3.1–§3.7、§4（类签名与 Dataset/TimeSeries） | `kl8` 基础层 + layer1 + 5 个测试文件 |
| P3 | §1（R5–R23）、§3.8–§3.9、§4.1（FeatureSnapshot 全字段）、§5（A1–A4） | layer2/layer3 + 12 个测试文件 |
| P4 | §1（R24–R36）、§3.10、§4.2–§4.3、§5（A5/A6/B5/B6/B7） | layer4 + 4 个测试文件 |
| P5 | §1（R37–R41）、§3.11、§4.4–§4.5、§5（A7/A8/A10） | layer5 + 3 个测试文件 + 覆盖表 |
| P6 | §1（R42–R48）、§3.12–§3.14、§4.6–§4.7、§5（A9/A10/B8） | evaluation + reporting + pipeline + main |
| P7/P8/P9 | §1 全表、§6 全章 | QA 用例矩阵、体检卡、验收报告 |

### §7.3 变更记录

| 版本 | 日期 | 变更 |
| :--- | :--- | :--- |
| v1.0.0 | 2026-11-02 | 首版：48 项需求映射（含 8.md 行号列）、五层签名冻结、数据结构字段冻结、A1–A10 裁决、DoD 与验证命令 |
| v1.0.1 | 2026-11-02 | B5：`hybrid.score_all` 追加 keyword-only `rule_weights`（自适应规则权重落点，向后兼容） |
| **v1.1.0** | 2026-11-02 | **5 类修改（Lead 批准）**：① **A7 修正**（`t≥k → m=min(v,t)`；`t<k → m=min(v,t+m_margin)`，`m_margin` 默认 1/范围 [1,3]；`0.25v+2` 降级为合理性 warning；`m` 定性为"用户/玩法假设参数，非预测"；新增 §5.0 五组参数验算表；`m_target` 签名追加 `t` 与 `margin=1`）；② **A5/A8/A9 自定标注**（0.3 系数与 `V≥3` 优先序、选七 10 注、M=8 均注明"契约自定，非 8.md 原文要求"）；③ **A10 N/A 措辞**（选三/选七报告必须输出 `ROI: N/A（官方奖级未确认）`，禁止金额结论）；④ **新增 `BacktestConfig.lift_signal: float = 1.05`**（范围 [1.0,1.2]）+ `LiftStat.is_signal` + `metrics.lift_at_k(rows, ks, cfg)`（消除字面量 1.05，报告措辞由该位驱动）；⑤ **B4 对齐说明**（8.md L37 的 ML 能力以可配置项交付，默认 `weight=0` 诊断通道，启用后进 BaseScore 并触发权重再归一化——**此句已被 v1.2.0 更正为"额外叠加、不归一化"**）。**附带修正**：v1.0.0 的"104 个叶子字段"计数有误（重复计入 2 个映射键），v1.1.0 实数 **102** 个叶子字段（Window 5 + Markov 3 + ScoreWeights 12 + Rule 12 + Adaptive 9 + Pool 12 + Covering 22 + ML 14 + Backtest 7 + AppConfig 6），已同步 config-schema.md §16 |
| **v1.2.0** | 2026-11-02 | **与落地代码 HEAD=`f170147` 对齐（6 类）**：① **`MLConfig.enabled` 默认 `True→False`**（B4 诊断通道、`weight=0` 时不进 FinalScore，却使单期 0.34s→2.03s；诊断时用 CLI `--ml` 开启，开启后 `ml_available=True`；`weight>0` 而 `enabled=False` 时**不阻断**（静默忽略 ML 通道、排序无实质影响），P8 记为低优先建议：后续版本补一条 `DailyResult.warnings` 提示）。② **CLI**：§3.14 新增完整参数表，含 **`--ml`**（→`ml.enabled=True`）与 **`--no-tickets`**（→`include_tickets=False`）、`--no-sweep`（兼容 no-op）；R48-b 同步。③ **`--backtest` 默认值按模式分派**：`daily`→**30**、`backtest/audit`→**100**，显式 `0` = daily 关闭伴随回测；`cfg.backtest.default_n` 仍为 100（配置层默认不变）。④ **`ScoreWeights` 七维默认值按实现对齐**：`0.02/0.08/0.20/0.25/0.05/0.05/0.35`（Σ=1.00），`lambda_vote 0.08→0.05`；并**更正 Σω 语义**：只有七维核心权重参与 `Σω=1` 强校验，`ml` 是**额外叠加项**（>0 时叠加、不归一化）。⑤ **性能预算按实测修订**（§6.5）：单期含注单 0.39s、100 期 `--no-tickets` 37.4s、含注单 40s、全量单测 14.6s、覆盖求解 0.006s、ML 开启 2.03s/期限额 2.5s。⑥ `.bat` 一键入口（UTF-8 BOM + `--no-sync --backtest 30 --full-print`）写入 §2.1 与 §3.14。 |
| **v1.2.0 工程结论** | 2026-11-02 | **"契约漂移导致的 2000× 性能缺陷"（P8 必查条目）**：实现侧把 `CoveringConfig.m_margin` 写成 **2**（契约 v1.1.0 明确要求 **1**）→ `m_target(12,5,4)=6` → 覆盖表键 `12,5,4,5` 未命中 → 每期回落 `scipy.milp` ILP，单期 8~13s。改回 1 后 `build_tickets` **13.66s → 0.006s（约 2000×）**。实测改善：100 期回测（`--no-tickets`）**176.5s → 37.4s**；含注单 **>600s → 40s**；全量单测 **105.6s → 14.6s**；单期 **0.39s**。**结论：契约默认值即性能护栏** —— 下游实现偏离契约默认值应视为缺陷而非"可调参数"，P8 质检需核对「实现默认值 == 契约默认值」逐项对齐（本版 §3.3 与 config-schema.md 已全部核对）。 |
| **v1.2.0 机检** | 2026-11-02 | `Import-Csv` 断言：`rows=48; ids_in_order=True; empty_fields_rows=0; PASS=True`（ID 严格 R1..R48 顺序）；§1.2 主表 48 行 × 5 列齐整；`default_config()` 实测默认值与 §3.3 / config-schema.md 逐字段一致（102 叶子字段，含 `ml.enabled=False`、`covering.m_margin=1`、`backtest.lift_signal=1.05`、七维权重 Σ=1.00） |
| **v1.3.0** | 2026-11-02 | **吸收 P8 体检卡（6 High / 15 Medium）暴露的 5 条契约级缺口**：① **§7.4 新增 Errata**：8.md L190「选五全中概率 0.155%、约 645 期」为笔误，精确值 `C(20,5)/C(80,5)=15504/24040016=**0.0645%（≈1/1550 期）**」→ 契约与报告一律按精确值披露，8.md 原文保留不改；全文 3 处引用（§6.2 手算行、§5 A10、旧版 §6.2 行）已同步。② **§5 A2 口径冻结**：`C̄o=mean(R(Co25),R(Co50),R(CoALL))` 等权先秩后均、不含 10G，**实现与测试均不得改为加权式**（历史事故 0.20/0.20/0.60 被写进测试当"裁决 A2"），本口径自 v1.0.0 未变更。③ **新增 §3.14.1 复盘状态契约** `output/state/prev_prediction.json`（修 P8-H5）：`schema_version`、按 period 幂等 upsert、原子写 `os.replace`、字段 `{period,target_date,gold,main,plans,created_at}`、缺失输出「首期无复盘基线」、**禁止当期预测冒充上期**。④ **R43/R44 显著性硬规则**（修 P8-M4）：`n<500` → `is_signal` 恒 False 且报告标注「样本量不足（n<500），结论仅供参考」；`K∈ks` 5 次检验 Bonferroni → `alpha_adj=0.01`；新增配置 `backtest.min_n_signal=500`、`backtest.bonferroni=True`，`LiftStat` 增 `n`/`alpha_adj`。⑤ **新增 §6.6 报告禁止项（可 grep 验收）**（修 P8-H6/M5）：禁占位符、禁渲染层硬编码常量（25.00%/6.01%/150~250/5 码/20/80）、禁未标注 ROI 的金额结论。**附带勘误**：§6.1-4 数据保真条目 `draws[0]/draws[-1]` 期号写反，已按实现改正（`draws[0]=2020001`、`draws[-1]=2026291`）。 |
| **v1.3.0 机检** | 2026-11-02 | `Import-Csv`：`rows=48; ids_in_order=True; empty_fields_rows=0; PASS=True`；两文件全部 Markdown 表格块 `bad_rows=0`；全文无残留 `0.155%`（除 A10/§7.4 的"笔误留痕"引用外）；config-schema.md 同步 `min_n_signal/bonferroni` 与 §17 状态契约 |

### §7.4 Errata（8.md 原文勘误留痕，**原文不改**）

> 原则：8.md 是原始需求留痕，**不修改原文**；凡原文数值/顺序有误，在此登记「原文 → 精确值 → 影响范围 → 以谁为准」，并由本契约与报告按精确值执行。

| # | 8.md 位置 | 原文表述 | 精确值 / 正确口径 | 影响范围 | 以谁为准 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| E1 | **L190** | 「选五全中理论概率 **0.155%**，即约 **645 期一遇**」 | **P = C(20,5)/C(80,5) = C(75,15)/C(80,20) = 15504/24040016 = 0.000644902…** → **0.0645%，约 1/1550 期**（原文放大约 2.4 倍；"645 期"应为"约 1550 期"） | `R44` 高奖级理论概率、`metrics.hypergeometric_baseline`、报告高奖级对照段、P7 数值用例（`tests/test_qa_numerics.py`） | **精确超几何公式**；报告与验收一律披露 `0.0645% / ≈1/1550 期`，不得沿用 0.155% |
| E2 | **L155 与 L202** | L155「复式池 10~15 码」；L202「生成最终 10~12 码复式池」 | 同一对象；默认 `main=12`，区间 10~15，L202 的 10~12 为玩法变体（选五 12 / 选七 10） | `A6`、`PoolConfig`、报告池规模描述 | **裁决 A6**（默认 12 同时满足两处） |
| E3 | **L164–166** | `m` 表述为"预期…实际命中个数" | `m` 是**用户/玩法假设参数**，非算法推导；取值规则见 A7（`m = min(v, t + m_margin)`，`t<k`） | `A7`、`CoveringConfig.m_margin`、报告覆盖段必须写「m 为假设参数」 | **裁决 A7** |
| E4 | **L148** | 「M=5 或 10」示例 | 契约取 **M=8**（自定值，Lead 批准，见 A9） | `AdaptiveConfig.lookback_m` | **裁决 A9** |
| E5 | （契约自身）§6.1-4 v1.2.0 及更早 | `draws[-1].period == 2020001`、`draws[0].period == 2026291` | **升序排序后** `draws[0].period == 2020001`（最早）、`draws[-1].period == 2026291`（最新） | 契约 §6.1-4，与 `tests/test_history.py` 断言对齐 | **本契约 v1.3.0**（已改） |

> 使用规则：任何文档/报告/用例发现与 8.md 数值冲突时，**先查本表**；本表未登记的冲突必须 send_message 给 Lead 裁决，禁止"就地改判"（如把 A2 改成加权式并写进测试）。

---

**契约冻结声明**：§3 与 §4 为冻结项。下游开发如发现签名无法实现 8.md 需求，必须 send_message 给 Lead 说明「签名/字段 + 阻碍 + 建议」并由 Lead 发起 v1.x 升级，**不得私自改签名或字段名**。