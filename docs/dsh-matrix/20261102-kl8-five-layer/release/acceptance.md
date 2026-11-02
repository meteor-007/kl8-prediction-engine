# P9 终态验收报告（acceptance）· 快乐8 五层重构

> ## ⚠️ 文档状态：**骨架 v0.2（2026-11-02：G1/G2/G5 已按 Lead 口径填写；G3/G4 留待终态证据）**
> **G3（编译与单测绿灯）与 G4（工业质检清零）仍为 `待补` —— 本文件当前不得作为放行依据。**
> G1/G2/G5 的依据为契约与文件的**静态实测**（全部命令与输出见附录 A）；性能/命中率/ROI 一律留待 Lead 终态实跑；本文件作者（release-ops）**不代 Lead 拍板**。
>
> | 项 | 值 |
> | :--- | :--- |
> | 需求原文 | `docs/8.md`（202 行，**唯一权威**） |
> | 架构契约 | `docs/dsh-matrix/20261102-kl8-five-layer/contract.md` **v1.3.0**（工作区未提交） |
> | 基线 | `d9bc139`（重构前快照） |
> | 当前 HEAD | `3281992`（工作区含 P8 返工未提交） |
> | 验收维度 | G1 需求保真 / G2 ACL+DB 脚本伴随 / G3 编译与单测绿灯 / G4 工业质检达标 / G5 发布就绪 |
> | 现状 | **P8 体检卡判定阻断（6 High / 15 Medium）；G4 未清零前不得放行** |

---

## 0. 填表规则（先读再填）

1. **结论**只能三选一：`达成` / `N-A + 理由` / `待补`；禁止「基本达成」「口头确认」「应该没问题」等模糊表述。
2. **证据**必须可复现，三选一或组合：
   - 代码：`文件路径:行号`（相对仓库根）；
   - 命令：**完整命令 + exit code + 关键输出行**（如 `UNITTEST_EXIT=0`、`Ran 175 tests ... OK`）；
   - 单测：`tests/test_xxx.py::test_name`（须为**实跑通过**，非静态存在）。
3. **无证据项一律写 `待补`**，不得因为「负责人说修好了」而填 `达成`（P8 复审口径：只认新 diff 与实跑证据）。
4. **数字口径**：性能/Lift/ROI/命中率必须取自 **P9 终态实跑**；**禁止**引用 P8 修复前快照（`output/` 中旧快照的"5 票 vs 加成 0.05""违反 45%""C(v,k,t) 占位""命中 0 码"均为**缺陷证据**而非验收证据）。显著性口径：`n≥500` 且 `p < α/len(ks)=0.01`，否则「样本量不足，仅供参考」。
5. **需求点数量更正**：早期任务书写「43 个需求点」为**笔误**，以契约 §1 与 `requirement-mapping.csv` 的 **48 项（R1–R48）** 为准（本文件按 48 项逐条验收）。

---

## 1. 总览（G1–G5）

| 维度 | 定义 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- |
| **G1 需求保真** | 48/48 条目均有 8.md 行号 + 实现模块/符号 + 测试映射，契约 §1.2 与 CSV 机检一致 | **达成**（契约映射保真；实现质量归 G3/G4） | §2；`contract.md:65-112`（§1.2）；`requirement-mapping.csv:2-49`；附录 A-9 |
| **G2 ACL+DB 脚本伴随** | 纯本地 Python CLI，**无 Controller/DB/ACL 变更 → N/A**；以配置契约/状态 schema/历史兼容用例作替代证据 | **N-A**（4 项 N-A 有理由 + 4 项替代证据达成） | §3 |
| **G3 编译与单测绿灯** | `compileall` exit 0 + `unittest` 全绿 + 架构/防穿越/残留/映射机检 | **待补** | §4 |
| **G4 工业质检达标** | P8 体检卡 6 High / 15 Medium 清零 + 12 红线通过 | **待补（当前阻断）** | §5 |
| **G5 发布就绪** | 四份发布文档齐备 + 执行顺序 + 回滚方案 + 发布文案口径诚实（违规词 0 命中） | **达成（文档级 5/10）**；交付件实物/演练/.bat 冒烟待补 | §6 |

---

## 2. G1 需求保真（R1–R48，48/48）

> 列说明：`实现` 与 `测试` 来自 `requirement-mapping.csv`（机读生成，48 行）；`结论/证据` 由 Lead 终态证据填写。
> 48 行完整性由 §4 的 `Import-Csv` 机检断言（`len(rows)==48`、ID 严格 R1..R48）。

| 需求ID | 需求摘要（可验收） | 8.md 行号 | 实现模块.符号 | 测试文件::用例 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | 二值化状态矩阵 X∈{0,1}^(T×80) 行和恒为 20 | L30;L47-L50 | `kl8/layer1_representation/state.py`<br>`BinaryStateMatrix.from_dataset` | `tests/test_state.py`<br>`test_binary_matrix_shape_and_row_sum` | 达成（契约映射保真） | `contract.md:65`（§1.2）<br>`requirement-mapping.csv:2` |
| R2 | 四尺度窗口 W={10G,25G,50G,ALL} 定义与切片 | L34-L35;L52-L57 | `kl8/layer1_representation/windows.py`<br>`resolve_windows + state.window_matrix` | `tests/test_windows.py`<br>`test_four_scale_windows` | 达成（条目存在·映射文本待订正） | `contract.md:66`（§1.2）<br>`requirement-mapping.csv:3` |
| R3 | 冷启动窗口回退 w_eff=min(w,T) | L57;L201 | `kl8/layer1_representation/windows.py`<br>`effective_window` | `tests/test_windows.py`<br>`test_cold_start_fallback` | 达成（契约映射保真） | `contract.md:67`（§1.2）<br>`requirement-mapping.csv:4` |
| R4 | 原始开奖数据清洗校验标准化(20码/1..80/升序/去重期号/降序文件自检) | L30-L31 | `kl8/history.py`<br>`parse_line + load_dataset + save_dataset` | `tests/test_history.py`<br>`test_parse_validate_and_sort` | 达成（契约映射保真） | `contract.md:68`（§1.2）<br>`requirement-mapping.csv:5` |
| R5 | 单球单期理论先验 p0=K/N=0.25 | L66;L97 | `kl8/layer2_baseline/priors.py`<br>`p0` | `tests/test_priors.py`<br>`test_p0_value` | 达成（契约映射保真） | `contract.md:69`（§1.2）<br>`requirement-mapping.csv:6` |
| R6 | 两球同期共现理论概率 p_co=20*19/(80*79) | L67;L108 | `kl8/layer2_baseline/priors.py`<br>`p_co` | `tests/test_priors.py`<br>`test_p_co_value` | 达成（契约映射保真） | `contract.md:70`（§1.2）<br>`requirement-mapping.csv:7` |
| R7 | 相邻两期理论重号期望 E(Repeat)=5 | L68 | `kl8/layer2_baseline/priors.py`<br>`expected_repeat` | `tests/test_priors.py`<br>`test_expected_repeat` | 达成（契约映射保真） | `contract.md:71`（§1.2）<br>`requirement-mapping.csv:8` |
| R8 | 窗口频次标准化偏离度 Z 与 /Z/>1.96 显著性 | L70-L73 | `kl8/layer2_baseline/zscore.py`<br>`window_z + significant_mask` | `tests/test_zscore.py`<br>`test_zscore_formula_and_threshold` | 达成（契约映射保真） | `contract.md:72`（§1.2）<br>`requirement-mapping.csv:9` |
| R9 | Wald-Wolfowitz 游程检验与聚簇/跳跃标签 | L75-L76 | `kl8/layer2_baseline/runs.py`<br>`runs_test + runs_test_80` | `tests/test_runs.py`<br>`test_runs_cluster_vs_jump_labels` | 达成（契约映射保真） | `contract.md:73`（§1.2）<br>`requirement-mapping.csv:10` |
| R10 | 卡方拟合优度+NIST-lite 4项+香农熵/互信息冗余(仅诊断) | L17;L22;L32 | `kl8/layer2_baseline/chisq.py<br>kl8/layer2_baseline/randomness.py<br>kl8/layer2_baseline/entropy.py`<br>`chi_square_gof + nist_lite + shannon_entropy + feature_redundancy` | `tests/test_chisq.py<br>tests/test_randomness.py<br>tests/test_entropy.py`<br>`test_gof_uniform_and_biased<br>test_nist_lite_four_items<br>test_entropy_and_redundancy` | 达成（契约映射保真） | `contract.md:74`（§1.2）<br>`requirement-mapping.csv:11` |
| R11 | 四尺度热力频率 Freq_i(w) | L85 | `kl8/layer3_features/heat.py`<br>`multi_scale_freq` | `tests/test_heat.py`<br>`test_multi_scale_freq` | 达成（契约映射保真） | `contract.md:75`（§1.2）<br>`requirement-mapping.csv:12` |
| R12 | 跨周期动量差 Mom10-25/Mom25-50 与金叉加速标签 | L86-L89 | `kl8/layer3_features/heat.py`<br>`momentum + golden_cross` | `tests/test_heat.py`<br>`test_momentum_and_golden_cross` | 达成（契约映射保真） | `contract.md:76`（§1.2）<br>`requirement-mapping.csv:13` |
| R13 | 当前遗漏 O 与历史平均遗漏 μ_gap | L90-L91 | `kl8/layer3_features/heat.py`<br>`gap_series + historical_mean_gap` | `tests/test_heat.py`<br>`test_gap_and_mean_gap_floor` | 达成（契约映射保真） | `contract.md:77`（§1.2）<br>`requirement-mapping.csv:14` |
| R14 | 高斯核遗漏回补压力 Pressure | L91-L92 | `kl8/layer3_features/heat.py`<br>`omission_pressure` | `tests/test_heat.py`<br>`test_pressure_kernel_peak_at_1_2x` | 达成（契约映射保真） | `contract.md:78`（§1.2）<br>`requirement-mapping.csv:15` |
| R15 | 一阶拉普拉斯平滑条件转移概率 M1(w) | L96-L98 | `kl8/layer3_features/markov.py`<br>`transition_matrix` | `tests/test_markov.py`<br>`test_alpha_smoothing_handcalc` | 达成（契约映射保真） | `contract.md:79`（§1.2）<br>`requirement-mapping.csv:16` |
| R16 | 四尺度中心化跟随聚合 FollowScore | L99-L101 | `kl8/layer3_features/markov.py`<br>`follow_score + follow_score_all_windows` | `tests/test_markov.py`<br>`test_follow_score_centered_dot` | 达成（契约映射保真） | `contract.md:80`（§1.2）<br>`requirement-mapping.csv:17` |
| R17 | 重号对角惯性分与异号跟随分拆分 | L102 | `kl8/layer3_features/markov.py`<br>`follow_split` | `tests/test_markov.py`<br>`test_follow_diag_offdiag_split` | 达成（契约映射保真） | `contract.md:81`（§1.2）<br>`requirement-mapping.csv:18` |
| R18 | 二阶滞后矩阵 M2 隔期回补(斜跳) | L102 | `kl8/layer3_features/markov.py`<br>`transition_matrix_order2` | `tests/test_markov.py`<br>`test_second_order_transition` | 达成（契约映射保真） | `contract.md:82`（§1.2）<br>`requirement-mapping.csv:19` |
| R19 | 中心化共现矩阵 A=C/w-p_co 与三元共现团 | L19;L107-L108 | `kl8/layer3_features/cooccurrence.py`<br>`centered_cooccurrence + triad_counts` | `tests/test_cooccurrence.py`<br>`test_centered_matrix_and_triads` | 达成（契约映射保真） | `contract.md:83`（§1.2）<br>`requirement-mapping.csv:20` |
| R20 | 当期伴生拉动分 CoScore | L109-L111 | `kl8/layer3_features/cooccurrence.py`<br>`co_score + co_score_all_windows` | `tests/test_cooccurrence.py`<br>`test_co_score` | 达成（契约映射保真） | `contract.md:84`（§1.2）<br>`requirement-mapping.csv:21` |
| R21 | 共现图 PageRank/特征向量中心度/Louvain社区/伴生码+降级 | L19;L112 | `kl8/layer3_features/graph.py`<br>`build_graph_features` | `tests/test_graph.py`<br>`test_pagerank_eigen_and_communities` | 达成（契约映射保真） | `contract.md:85`（§1.2）<br>`requirement-mapping.csv:22` |
| R22 | 宏观空间形态特征(四八区/奇偶/质合/大中小/连号/同尾/和值/AC) | L114-L118 | `kl8/layer3_features/morphology.py`<br>`morphology_stats + ac_value + sum_deviation_z + is_balanced` | `tests/test_morphology.py`<br>`test_morphology_stats_full_matrix` | 达成（契约映射保真） | `contract.md:86`（§1.2）<br>`requirement-mapping.csv:23` |
| R23 | GBDT概率校准+PAV保序回归+FeatureSnapshot装配 | L20;L37 | `kl8/layer3_features/ml.py<br>kl8/layer3_features/features.py`<br>`train_predict + pav_calibrate + build_feature_snapshot` | `tests/test_ml.py`<br>`test_ml_prob_bounded_and_cache<br>test_build_snapshot_schema` | 达成（契约映射保真） | `contract.md:87`（§1.2）<br>`requirement-mapping.csv:24` |
| R24 | Borda 百分位秩归一化 R=Rank/80∈(0,1] | L126-L129 | `kl8/layer4_scoring/borda.py`<br>`percentile_rank + rank_matrix` | `tests/test_scoring.py`<br>`test_percentile_rank_ties_and_range` | 达成（契约映射保真） | `contract.md:88`（§1.2）<br>`requirement-mapping.csv:25` |
| R25 | 规则R1 多尺度强跟随共振(>=3窗口 R>=0.65) | L133 | `kl8/layer4_scoring/rules.py`<br>`rule_r1` | `tests/test_rules.py`<br>`test_r1_boundary` | 达成（契约映射保真） | `contract.md:89`（§1.2）<br>`requirement-mapping.csv:26` |
| R26 | 规则R2 温热多头(F25/F50>=0.24 且 F10∈[0.20,0.50]) | L134 | `kl8/layer4_scoring/rules.py`<br>`rule_r2` | `tests/test_rules.py`<br>`test_r2_boundary` | 达成（契约映射保真） | `contract.md:90`（§1.2）<br>`requirement-mapping.csv:27` |
| R27 | 规则R3 动量金叉(F10>=F25>=F50 且 R Follow10>=0.60) | L135 | `kl8/layer4_scoring/rules.py`<br>`rule_r3` | `tests/test_rules.py`<br>`test_r3_boundary` | 达成（契约映射保真） | `contract.md:91`（§1.2）<br>`requirement-mapping.csv:28` |
| R28 | 规则R4 强共现伴生(0.5R Co25+0.5R Co50>=0.75) | L136 | `kl8/layer4_scoring/rules.py`<br>`rule_r4` | `tests/test_rules.py`<br>`test_r4_boundary` | 达成（契约映射保真） | `contract.md:92`（§1.2）<br>`requirement-mapping.csv:29` |
| R29 | 规则R5 全量回归回补(Pressure>=0.60 且 O<=12 且 R FollowALL>=0.70) | L137 | `kl8/layer4_scoring/rules.py`<br>`rule_r5` | `tests/test_rules.py`<br>`test_r5_boundary` | 达成（契约映射保真） | `contract.md:93`（§1.2）<br>`requirement-mapping.csv:30` |
| R30 | 交集共振票数 V_i∈{0..5} | L139 | `kl8/layer4_scoring/rules.py`<br>`evaluate_rules + vote` | `tests/test_rules.py`<br>`test_vote_count_range` | 达成（契约映射保真） | `contract.md:94`（§1.2）<br>`requirement-mapping.csv:31` |
| R31 | BaseScore 五维加权融合(Σω=1) | L143 | `kl8/layer4_scoring/hybrid.py`<br>`base_score + trend_score + co_aggregate` | `tests/test_scoring.py`<br>`test_base_score_decomposition` | 达成（契约映射保真） | `contract.md:95`（§1.2）<br>`requirement-mapping.csv:32` |
| R32 | FinalScore=BaseScore+λ*V_i(λ∈[0.05,0.10]) | L144-L145 | `kl8/layer4_scoring/hybrid.py`<br>`final_score + score_all` | `tests/test_scoring.py`<br>`test_final_score_identity` | 达成（契约映射保真） | `contract.md:96`（§1.2）<br>`requirement-mapping.csv:33` |
| R33 | 在线自适应 Softmax 规则权重更新(M=8) | L147-L148 | `kl8/layer4_scoring/adaptive.py`<br>`next_weights + softmax_weights + load_state + save_state` | `tests/test_adaptive.py`<br>`test_softmax_weights_sum_and_prior` | 达成（契约映射保真） | `contract.md:97`（§1.2）<br>`requirement-mapping.csv:34` |
| R34 | 空间形态均衡硬约束(四区45%/奇偶70%/同尾3)与按序放宽 | L150-L154 | `kl8/layer4_scoring/filter.py`<br>`enforce_constraints + greedy_pool` | `tests/test_filter.py`<br>`test_hard_constraints` | 达成（契约映射保真） | `contract.md:98`（§1.2）<br>`requirement-mapping.csv:35` |
| R35 | 金胆池3~5 与复式池10~15(默认12) | L155 | `kl8/layer4_scoring/filter.py`<br>`select_gold + build_pool` | `tests/test_filter.py`<br>`test_gold_and_main_sizes` | 达成（契约映射保真） | `contract.md:99`（§1.2）<br>`requirement-mapping.csv:36` |
| R36 | 重号新号配比控制 25%~35% | L202 | `kl8/layer4_scoring/filter.py`<br>`apply_repeat_quota` | `tests/test_filter.py`<br>`test_repeat_quota` | 达成（契约映射保真） | `contract.md:100`（§1.2）<br>`requirement-mapping.csv:37` |
| R37 | 覆盖设计 C(v,k,t,m) 定义与保底命题穷举验证 | L163-L166 | `kl8/layer5_combinatorial/covering.py`<br>`m_target + verify_guarantee` | `tests/test_covering.py`<br>`test_verify_guarantee_exhaustive` | 达成（契约映射保真） | `contract.md:101`（§1.2）<br>`requirement-mapping.csv:38` |
| R38 | 标准旋转矩阵查表映射(核验入库) | L169 | `kl8/layer5_combinatorial/covering.py`<br>`lookup_table + tools/build_covering_tables.py` | `tests/test_covering_tables.py`<br>`test_tables_verified` | 达成（契约映射保真） | `contract.md:102`（§1.2）<br>`requirement-mapping.csv:39` |
| R39 | 0-1整数规划最小注数求解与规模降级 | L21;L168-L170 | `kl8/layer5_combinatorial/covering.py`<br>`solve_ilp + solve` | `tests/test_covering.py`<br>`test_ilp_min_tickets_and_fallback` | 达成（契约映射保真） | `contract.md:103`（§1.2）<br>`requirement-mapping.csv:40` |
| R40 | 带概率权重贪心集合覆盖+模拟退火预算优化 | L170 | `kl8/layer5_combinatorial/covering.py`<br>`greedy_weighted + anneal` | `tests/test_covering.py`<br>`test_greedy_weighted_and_sa_determinism` | 达成（契约映射保真） | `contract.md:104`（§1.2）<br>`requirement-mapping.csv:41` |
| R41 | 注单后处理与双玩法(选五/选七)预算落地 | L161-L170 | `kl8/layer5_combinatorial/tickets.py`<br>`postprocess + build_ticket_plan` | `tests/test_tickets.py`<br>`test_postprocess_and_dual_play` | 达成（契约映射保真） | `contract.md:105`（§1.2）<br>`requirement-mapping.csv:42` |
| R42 | Walk-Forward 滚动回测(T0=300 逐期推进 禁随机K-Fold) | L176-L182 | `kl8/evaluation/walkforward.py`<br>`run_walkforward + iter_windows` | `tests/test_walkforward.py`<br>`test_walkforward_flow_no_leak` | 达成（契约映射保真） | `contract.md:106`（§1.2）<br>`requirement-mapping.csv:43` |
| R43 | Top-K 平均命中率与 Lift@K + 单侧t检验 | L185-L188 | `kl8/evaluation/metrics.py`<br>`lift_at_k + t_test_one_sided` | `tests/test_metrics.py`<br>`test_lift_at_k_handcalc` | 达成（契约映射保真） | `contract.md:107`（§1.2）<br>`requirement-mapping.csv:44` |
| R44 | 高奖级触发率 vs 超几何理论概率 | L189-L190 | `kl8/evaluation/metrics.py`<br>`high_tier_rate` | `tests/test_metrics.py`<br>`test_high_tier_vs_hypergeometric` | 达成（契约映射保真） | `contract.md:108`（§1.2）<br>`requirement-mapping.csv:45` |
| R45 | 规则正交性与归因分析(R1-R5独立胜率 V>=3 vs 低票) | L191-L192 | `kl8/evaluation/metrics.py`<br>`rule_attribution` | `tests/test_metrics.py`<br>`test_rule_attribution` | 达成（契约映射保真） | `contract.md:109`（§1.2）<br>`requirement-mapping.csv:46` |
| R46 | 资金曲线 ROI/最大回撤/最长连挂(整数元) | L193-L194 | `kl8/evaluation/metrics.py`<br>`bankroll` | `tests/test_metrics.py`<br>`test_bankroll_roi_drawdown` | 达成（条目存在·映射文本待订正） | `contract.md:110`（§1.2）<br>`requirement-mapping.csv:47` |
| R47 | 严格防穿越审计(唯一截断入口+max_accessed_t+未来投毒不变性) | L28;L200 | `kl8/evaluation/leakage.py`<br>`assert_no_lookahead + poison_future + fingerprint_snapshot` | `tests/test_leakage.py`<br>`test_poison_future_invariance` | 达成（契约映射保真） | `contract.md:111`（§1.2）<br>`requirement-mapping.csv:48` |
| R48 | 工程落地(配置校验/CLI/同步/三类交付件/幂等/性能) | L30-L39;L198-L202 | `kl8/pipeline.py<br>kl8/reporting<br>kl8/config.py<br>kl8/sync.py<br>main.py`<br>`run_daily + run_backtest + run_audit` | `tests/test_pipeline.py<br>tests/test_config.py`<br>`test_end_to_end_daily_and_backtest<br>test_config_validation` | 达成（条目存在·映射文本待订正） | `contract.md:112`（§1.2）<br>`requirement-mapping.csv:49` |

**G1 小计**：**48/48 条目达成（契约映射保真）**；N-A 0 项；待补 0 项。
机检依据（附录 A-9，2026-11-02 实测）：`rows=48`、`ids_in_order=true`、`missing_impl_files=[]`（48 行实现文件全部存在）、`missing_test_files=[]`（48 行测试文件全部存在）；契约 §1.2 行号 R1→L65 … R48→L112，与 CSV 行号 R1→L2 … R48→L49 一一对应。

> **偏差登记（3 处映射文本问题，不否定条目存在性，但须在终态订正 CSV）**：
> ① **R2** 符号 `state.window_matrix` 实际位于 `kl8/layer1_representation/state.py:131`，CSV「实现模块」仅列 `windows.py`（跨文件未列全）；
> ② **R46** CSV 用例名 `test_bankroll_roi_drawdown` 在 `tests/test_metrics.py` 中**不存在**（实测为 `test_bankroll_integer_yuan`，`test_metrics.py:98`）；
> ③ **R48** CSV 用例名 `test_config_validation` **不存在**（`tests/test_config.py` 实测为 `test_weights_sum_violation_rejected` 等细分用例，无该聚合名）。
> 处置：终态由智小研/测小严修订 `requirement-mapping.csv` 这 3 行（用例名对齐真实 `def test_*`），否则 G3.6 映射机检会判失败。

> **注意（G1 不得替代 G4）**：G1 只证明「需求→契约→实现点/测试名」的映射保真；P8 体检卡在映射**存在**的前提下判定其中 6 项实现有病（R23/R33/R34/R41/R44/R45，对应 H1/H2/H3/M3/M11/M12），其实现是否真正达标 **以 G3/G4 实跑证据为准**。


---

## 3. G2 ACL + DB 脚本伴随

> 本项目为**纯本地单机 Python CLI**（无 HTTP 服务/Controller、无用户体系、无数据库、无多租户）。**Lead 裁决：无 Controller/DB/ACL 变更 → G2 判 N/A**。为避免「N/A 等于不验收」，下列替代证据必须成立。

| 检查项 | 是否需要 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- |
| G2.1 权限模型 / ACL（角色、资源授权） | 不需要 | **N-A**：无服务端、无用户体系、无 HTTP API；仅本机单用户文件系统权限，不存在跨主体授权对象 | `main.py`（argparse → `pipeline`，无鉴权层）；`kl8/pipeline.py`（唯一编排点，无权限分支） |
| G2.2 数据库 schema / 建表 / 迁移脚本 | 不需要 | **N-A**：无数据库；持久化仅两个 JSON 状态文件（`adaptive_weights.json`、`prev_prediction.json`），以 `schema_version` + 原子写 `os.replace` + 按 period 幂等 upsert 替代迁移 | `contract.md:249`（§3.3 `AdaptiveConfig.schema_version`）；`contract.md:928`（§3.14.1 `prev_prediction.json` 契约）；`kl8/layer4_scoring/adaptive.py:226,267`（`load_state`/`save_state`）；`output/state/adaptive_weights.json`（实测 `schema_version=1`） |
| G2.3 状态损坏自愈（替代 DB 修复/回滚脚本） | 需要 | **达成**：状态文件损坏/版本不符 → 自动 `.bak` 备份并重置，写中文 warning，不中断运行 | `kl8/layer4_scoring/adaptive.py:226`（`load_state`）；`release/runbook.md` §6.4 |
| G2.4 外部网络调用降级与鉴权 | 需要（同步可选） | **达成**：网络失败仅转 `warnings`、`ok=False` 且**不改本地文件**；公开数据源无鉴权需求 | `kl8/sync.py:212`（`sync_history`）；`release/runbook.md` §6.1 |
| G2.5 凭据 / 密钥 / 敏感信息 | 不需要 | **N-A**：无账号/密钥/Token；历史数据与交付件不含个人隐私字段 | `kl8/sync.py`（公开 HTTP 源，无凭据读取）；`release/runbook.md` §1 |
| G2.6 配置契约（替代接口契约/ACL 校验） | 需要 | **达成**：`config-schema.md`（357 行）定义全字段默认值/范围/校验；`validate_config` 强校验（Σω=1±1e-9、α∈[0.5,2]、λ∈[0.05,0.10]、gold<main、k≤池、预算≤200），非法抛 `ConfigError`（中文可操作提示） | `docs/dsh-matrix/20261102-kl8-five-layer/config-schema.md`；`kl8/config.py:574`（`validate_config`）、`:1068`（`load_config`）、`:1087`（`merge_cli_overrides`） |
| G2.7 历史数据格式兼容（替代数据迁移脚本） | 需要 | **达成**：解析/校验/排序/期号唯一/降序文件自检用例在册（20 码 + 1..80 + 升序） | `tests/test_history.py:27`（`test_parse_validate_and_sort`）；`contract.md` §0.1 |
| G2.8 `tools/rebuild_state.py`（契约 §2.1 曾列） | 不需要 | **N-A（不存在）**：实测 `tools/` 仅 `build_covering_tables.py`；状态重建改为「删 state 文件自动重置」，无需专用脚本 | `Get-ChildItem tools -Filter *.py` → 仅 `build_covering_tables.py`；`release/runbook.md` §6.4 |

**G2 小计**：**N-A 4 项**（G2.1/G2.2/G2.5/G2.8，均有理由与替代证据）；**达成 4 项**（G2.3/G2.4/G2.6/G2.7）；待补 0 项。

---

## 4. G3 编译与单测绿灯

| # | 检查项 | 命令 | 期望 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| G3.1 | 编译 | `python -m compileall kl8 main.py tools` | **exit 0** | 待补 | 待补 |
| G3.2 | 全量单测 | `python -m unittest discover -s tests -v` | **exit 0、全绿**；用例数**唯一权威口径 = 输出行的 `Ran N tests`**（Lead 2026-11-02 定调；静态 `def test_` 253 仅作参考，见附录 C-6） | 待补 | 待补 |
| G3.3 | 分层 import 冒烟 | `python -c "import kl8.config, kl8.history, kl8.pipeline; import kl8.layer1_representation.state; import kl8.layer2_baseline.priors; import kl8.layer3_features.features; import kl8.layer4_scoring; import kl8.layer5_combinatorial; import kl8.evaluation.walkforward, kl8.reporting.text_report"` | exit 0 | 待补 | 待补 |
| G3.4 | 架构依赖（五层单向） | `python -m unittest tests.test_architecture -v` | exit 0 | 待补 | 待补 |
| G3.5 | 防穿越审计 | `python main.py --no-sync --mode audit --backtest 30` | **exit 0**，且自检项**能对注入穿越返回 False**（P8-H4 反证） | 待补 | 待补 |
| G3.6 | 需求映射机检 | `python -c "import csv;rows=list(csv.DictReader(open(r'docs/dsh-matrix/20261102-kl8-five-layer/requirement-mapping.csv',encoding='utf-8')));print(len(rows));assert len(rows)==48;assert [r['需求ID'] for r in rows]==[f'R{i}' for i in range(1,49)]"` | 打印 `48`，exit 0 | 待补（**行数=48/ID 顺序已实测通过**，附录 A-9；R2/R46/R48 映射文本偏差待订正，见 §2） | §2 偏差登记；附录 A-9 |
| G3.7 | 残留引用（代码） | `git grep -n -E "calc_core\|calc_engine\|calc_backtest\|calc_ensemble\|calc_shrinkage\|calc_report\|calc_ml\|calc_graph\|calc_signal_audit\|calc_sync\|calc_excel" -- "*.py" "*.bat"` | **0 处** | **已实测 0 处**（2026-11-02，`*.py`/`*.bat` 无命中；`*.md` 15 处豁免见附录 B） | 附录 B |
| G3.8 | 数据保真 | `python -c "from kl8.history import load_dataset;ds=load_dataset('kl8_history.txt');print(ds.T, ds.draws[0].period, ds.draws[-1].period)"` | `2112 2020001 2026291`，无意外 warning | 待补 | 待补 |
| G3.9 | 交付件实物 | `Get-ChildItem output/reports,output/wechat,output/excel,output/state -File` | 四类各 1 新件（+ `prev_prediction.json` 修复后出现） | 待补 | 待补 |

**G3 小计**：待补 `__/9` 通过（其中 **G3.6/G3.7 两项静态机检已实测通过**，待终态复跑确认）。

---

## 5. G4 工业质检达标（P8 体检卡清零）

> 来源：`docs/dsh-matrix/20261102-kl8-five-layer/review-card.md` **v1.1**（首轮判定：**阻断**，6 High / 15 Medium / 14 Low；12 红线 7 违规·4 部分·1 通过）。
> 放行条件：**High 6 + Medium 15 全部关闭（或经 Lead 裁决降级并留痕）+ 红线全部通过**；复审只认新 diff 与实跑证据。

### 5.1 High（6 项，必须清零）

| # | 问题摘要（首轮） | 复审状态 | 证据位置 |
| :--- | :--- | :--- | :--- |
| H1 | R33 自适应权重假实现 + B5 归一化缺失（权重硬编码 Σ=4.8、`FinalScore` 漏除 `w̄`、报告与引擎两套得票） | 待补 | 待补 |
| H2 | R34 空间形态硬约束未执行（只告警不重选；`enforce_constraints/greedy_pool` 死代码；实测池违反 45%） | 待补 | 待补 |
| H3 | R45 规则归因造假常量（R1–R5 恒 0.25/1.0/0；高/低票组固定值） | 待补 | 待补 |
| H4 | `self_check["no_lookahead"]` 恒 True，audit 模式为伪机检 | 待补 | 待补 |
| H5 | 复盘口径错误（当期金胆 ∩ 最新开奖冒充"上期预测命中"）→ 需 `prev_prediction.json` 跨期持久化 | 待补 | 待补 |
| H6 | 半成品/死代码/死配置（`C(v,k,t)` 占位符、PAV `if False`、`refit_every/max_pair_overlap/t0_min` 死配置、多处死函数） | 待补 | 待补 |

### 5.2 Medium（15 项）

| # | 问题摘要（首轮） | 复审状态 | 证据位置 |
| :--- | :--- | :--- | :--- |
| M1 | A2 契约偏离（`0.20/0.20/0.60` vs 等权 `mean(R(Co25),R(Co50),R(CoALL))`）且在 QA 测试中被固化；**Lead 裁决：契约不改，实现+测试双错** | 待补 | 待补 |
| M2 | A5 金胆选取偏离（恒取 5 码、无 `V≥3` 优先；契约 `clip(round(0.30·main),3,5)`，main=12→4） | 待补 | 待补 |
| M3 | A10 未确认奖级仍出金额（选七实测 ROI -100%，应为 `N/A（官方奖级未确认）`） | 待补 | 待补 |
| M4 | 小样本假显著（30 期 4/5 标"★显著"；缺 `n≥500` + Bonferroni） | 待补 | 待补 |
| M5 | 报告硬编码口径（p0/p_co/和值区间/金胆 5 码）+ 规则中文名与实现不符 | 待补 | 待补 |
| M6 | NIST-Lite 方法误用（(T,80) 二值矩阵直接摊平），且与卡方"均匀性通过"同屏矛盾 | 待补 | 待补 |
| M7 | 6 处裸 `except Exception: pass` + 降级不留痕（`backend` 仍报 networkx） | 待补 | 待补 |
| M8 | `seed=42` 硬编码，`cfg.seed` 在 L5 断链（不同 seed 不可复现/不可区分） | 待补 | 待补 |
| M9 | 日期兜底硬编码 `"2026-11-02"`（解析失败静默使用项目基准日） | 待补 | 待补 |
| M10 | ROI 浮点（违反 C3 零浮点金额；应为万分比整数/Decimal） | 待补 | 待补 |
| M11 | R41 缺 `max_pair_overlap`；连号判据 off-by-one（3 连被放行）；`k/t` 硬编码 | 待补 | 待补 |
| M12 | R23 ML 缓存 key 含 `end` 永不命中；`refit_every` 未实现；PAV 死代码 | 待补 | 待补 |
| M13 | 交付件非原子写 + 假成功（Excel `PermissionError` 仍返回成功路径） | 待补 | 待补 |
| M14 | 4 处测试假绿（硬约束/归因/ILP 条件断言/自检循环论证） | 待补 | 待补 |
| M15 | 文档勘误（契约 §6.1-4 期号顺序写反；8.md L190 选五全中概率应为 `15504/24040016=0.0645%≈1/1550`）——**Lead 裁决：不改 8.md，走契约 §7 errata** | 待补 | 待补 |

### 5.3 12 条工程红线（P8 口径）

| # | 红线 | 首轮判定 | 复审状态 | 证据位置 |
| ---: | :--- | :--- | :--- | :--- |
| 1 | 零魔法数字 | ❌ 违规 | 待补 | 待补 |
| 2 | 零裸 except / 零 silent pass | ❌ 违规 | 待补 | 待补 |
| 3 | 零浮点金额 | ❌ 违规 | 待补 | 待补 |
| 4 | 循环与组合规模封顶 | ✅ 通过 | 待补 | 待补 |
| 5 | 除零与空安全 | ⚠️ 部分违规 | 待补 | 待补 |
| 6 | 时间标准化 | ❌ 违规 | 待补 | 待补 |
| 7 | 零循环内重计算 | ❌ 违规 | 待补 | 待补 |
| 8 | 幂等防重 | ⚠️ 部分违规 | 待补 | 待补 |
| 9 | 确定性 | ❌ 违规 | 待补 | 待补 |
| 10 | 资源释放 | ⚠️ 部分违规 | 待补 | 待补 |
| 11 | 零死代码 / 零半成品 | ❌ 违规（最重） | 待补 | 待补 |
| 12 | 中文文档与命名 | ⚠️ 部分违规 | 待补 | 待补 |

### 5.4 复审三项"防复发"断言（Lead 指定，逐条留证）

| # | 断言 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- |
| G4.1 | **H1**：交付件附录 A 逐行 `|Final − Base − λ·V| < 1e-4`（覆盖 V=0..5），引擎侧同式 `< 1e-12` | 待补 | 待补 |
| G4.2 | **H2**：极端 `scores`（全落同一四区）下 `main` 仍满足四区 ≤45%；`relaxed` 只含真实冲突项；全不可行抛 `InfeasibleConstraintError` | 待补 | 待补 |
| G4.3 | **H4**：注入穿越（`end` 越界 / 投毒后不截断）→ `self_check["no_lookahead"] is False`；正常路径 True | 待补 | 待补 |

**G4 小计**：High 待补 `__/6` 关闭；Medium 待补 `__/15` 关闭；红线待补 `__/12` 通过。

---

## 6. G5 发布就绪

| # | 检查项 | 期望证据 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- |
| G5.1 | 上线手册 | 环境/安装/三用法/.bat/产出/配置/回滚/6 类故障齐备并经 Lead 复核 | **达成**（Lead 2026-11-02 回执「四件交付有效」） | `release/runbook.md`（17.7 KB：§1 环境、§2 安装、§3 用法、§4 产出、§5 配置、§6 六类故障、§8 回滚） |
| G5.2 | 发布文案 | 口径诚实（含 Lift≈1.0~1.07 不显著 / ROI 为负 / 不构成投注建议），**无**违规词 | **达成**（违规词实测 0 命中，附录 A-10；并已加「修复轮前样例不可作为最终交付样本」声明） | `release/wechat-announcement.txt`（2.5 KB） |
| G5.3 | 变更集 | 新增/修改/删除 + 执行顺序 + `d9bc139..HEAD` diff 统计 | **达成**（结构性齐备；§2 计数待终态刷新） | `release/changeset.md`（A83/M2/D81；166 files +15580/−9318；9 步执行顺序） |
| G5.4 | 验收报告 | 本文件 G1–G5 全部有证据、无 `待补` | **待补**（G3/G4 未填） | 本文件 §4、§5 |
| G5.5 | 回滚方案齐备 | 4 类回滚场景 + 步骤 + 风险 | **达成** | `release/runbook.md` §8；`release/changeset.md` §6 |
| G5.6 | 回滚演练 | `git revert f170147` / `git checkout d9bc139` 实跑 + 回滚后冒烟 exit 0 | 待补 | 待补 |
| G5.7 | 交付件实物 | TXT/微信/Excel/状态/`prev_prediction.json` 五类齐备；无 `C(v,k,t)` 占位、规则中文名正确、复盘口径正确 | 待补 | 待补 |
| G5.8 | `.bat` 双击冒烟 | 双击 exit 0，中文无乱码，四类交付件生成 | 待补 | 待补 |
| G5.9 | 文档口径一致 | 契约 §6.6 禁止项 + §7.4 Errata 已落笔；8.md 原文保留不改 | **达成**：`contract.md:1306`（§6.6）、`contract.md:1381`（§7.4）、`contract.md:1387`（E1 精确值 0.0645%≈1/1550）、`docs/8.md:190`（原文 0.155% 保留） | 见左 |
| G5.10 | 提交与标签 | Lead 统一提交；tag 由 Lead 决定（release-ops 不代提交） | 待补 | 待补 |

**G5 小计**：**达成 5/10**（G5.1/G5.2/G5.3/G5.5/G5.9）；**待补 5/10**（G5.4/G5.6/G5.7/G5.8/G5.10）。

---

## 附录 A：证据登记表（终态填写）

| # | 命令（完整可复现） | 执行人 | 执行时间 | exit code | 关键输出（原文摘录） | 对应维度 |
| ---: | :--- | :--- | :--- | ---: | :--- | :--- |
| 1 | `python -m compileall kl8 main.py tools` | 待补 | 待补 | 待补 | 待补 | G3.1 |
| 2 | `python -m unittest discover -s tests -v` | 待补 | 待补 | 待补 | 待补 | G3.2 |
| 3 | `python main.py --no-sync --mode audit --backtest 30` | 待补 | 待补 | 待补 | 待补 | G3.5 |
| 4 | `python main.py --no-sync --mode backtest --backtest 100 --no-tickets` | 待补 | 待补 | 待补 | 待补 | G1-R43 / G5 |
| 5 | `python main.py --no-sync --mode backtest --backtest 20 --k 7` | 待补 | 待补 | 待补 | 待补 | G4-M3（须输出 ROI N/A） |
| 6 | `python main.py --no-sync --backtest 30 --full-print` | 待补 | 待补 | 待补 | 待补 | G5.8 |
| 7 | 双击 `运行微积分预测.bat` | 待补 | 待补 | 待补 | 待补 | G5.8 |
| 8 | 交付件实物核对（TXT/微信/Excel/state） | 待补 | 待补 | — | 待补 | G5.7 |
| 9 | 48 行映射机检：`python` 脚本读 `requirement-mapping.csv` + `contract.md`，断言 48 行/ID 顺序/实现文件/测试文件/符号/用例名 | 发小顺 | 2026-11-02 18:2x | **0** | `rows=48, ids_in_order=true, contract_lines_found=true, csv_lines_found=true, missing_impl_files=[], missing_test_files=[]`；偏差 3 处：R2 符号跨文件、R46 用例名 `test_bankroll_roi_drawdown` 不存在（实为 `test_bankroll_integer_yuan`）、R48 `test_config_validation` 不存在 | **G1 / G3.6** |
| 10 | `Select-String -Path release/wechat-announcement.txt -Pattern "提升命中率\|跑赢随机\|稳赚\|必中\|包中"` | 发小顺 | 2026-11-02 18:2x | **0** | `hits=0` | **G5.2** |

---

## 附录 B：残留引用 grep 豁免清单（2026-11-02 18:11 实测）

**实测命令与结果**：

```
grep -rn "calc_core|calc_engine|calc_backtest|calc_ensemble|calc_shrinkage|calc_report|calc_ml|calc_graph|calc_signal_audit|calc_sync|calc_excel" --include=*.py   → 0 命中
                                                                                                                              --include=*.bat  → 0 命中
                                                                                                                              --include=*.md   → 15 命中（2 个文件）
```

| 文件 | 命中数 | 豁免理由 |
| :--- | ---: | :--- |
| `docs/knowledge_base/2026-10-20_Evaluating Lottery Analysis Methods_1b5a1b8e.md` | 14 | **历史知识归档**：记录旧 `calc_*` 八区系统的设计演进与改造建议（含 `file:///` 链接），仅作可追溯资料，**不被任何代码 import、不参与运行**。删除会破坏历史可追溯性，故保留并豁免 |
| `docs/dsh-matrix/20261102-kl8-five-layer/contract.md` | 1（第 1231 行） | **检查规范自指**：DoD 条款 `grep -rn "calc_core\|calc_engine\|..."` 本身就是在列举待检查关键词，属"规则文本"而非"代码引用"，删除会导致验收条款无法执行 |
| **合计** | **15** | `*.py` / `*.bat` **零残留**；`*.md` 全部为归档或规范自指，**无代码引用风险** |

> 结论：清理（`f170147`）在**代码层面无残留**；后续若新增文档，`*.md` 命中需按同口径逐条列明理由，不得默认豁免。

---

## 附录 C：未验证项 / 待核对项（**不得代补绿**）

| # | 项 | 现状 | 由谁补 |
| ---: | :--- | :--- | :--- |
| 1 | **P8 返工后的终态**（新 HEAD、新 diff、6H/15M 关闭证据） | 码小全修复中，**未放行** | Lead + 审大关复审 |
| 2 | 100 期 Lift/p/ROI 数字 | Lead 口径 `Lift≈1.03~1.07、p 全 >0.05、ROI 为负`，**需终态实跑复现** | Lead |
| 3 | 500 期回测性能预算（≤180 s） | 契约 v1.2.0 标注「待 P7 补测」 | Lead |
| 4 | `--ml` 开启后端到端耗时（契约自报 2.03 s/期） | 未实测 | Lead |
| 5 | 单期内存峰值（≤300 MB） | 未实测 | Lead |
| 6 | **用例数口径**（Lead 定调 2026-11-02）：**唯一权威 = `unittest discover` 输出行的 `Ran N tests`**；静态 `def test_` 计数 253（36 个 `test_*.py`）**仅作参考**（含循环内子断言/跳过/参数化，会失真）；正文数字一律用 `Ran N` 终值 | 口径已定；终值待 Lead 实跑提供（复审时 175 为当时口径） | Lead / 测小严 |
| 7 | 工作区未提交项（`contract.md` v1.3.0、6 个 `test_qa_*.py`、`review-card.md`） | 未入库，changeset 计数会变 | Lead 统一提交 |
| 8 | `prev_prediction.json` 实际生成与复盘口径 | 当前 `Test-Path` = False（P8-H5 修复中，**修复后出现**为正确预期） | 码小全 + Lead |
| 9 | **`requirement-mapping.csv` 3 处文本偏差**：R2 符号 `state.window_matrix` 属 `state.py` 却列在 `windows.py`；R46 用例名 `test_bankroll_roi_drawdown` 实为 `test_bankroll_integer_yuan`；R48 用例名 `test_config_validation` 不存在（无该聚合用例） | **已实测确认**（附录 A-9）；不影响 48 条目存在性，但会导致 G3.6 机检失败 | 智小研 / 测小严终态修订 |

---

## 附录 D：放行签署

**放行条件（全部满足才可签）**：

1. G1：48/48 逐条有结论与证据，N-A 项有 Lead 确认理由；
2. G2：N-A 项经 Lead 确认（或补齐陪伴脚本）；
3. G3：`compileall` exit 0 + `unittest` 全绿 + audit exit 0 + 残留 0 + 映射 48 行机检通过；
4. G4：High 6 / Medium 15 全部关闭（或 Lead 裁决降级留痕）+ 12 红线通过 + 三项防复发断言留证；
5. G5：四份发布文档落盘且口径诚实 + 回滚演练 + 交付件实物核对 + Lead 统一提交。

| 角色 | 姓名 | 结论 | 签字/时间 |
| :--- | :--- | :--- | :--- |
| 发布验收（release-ops） | 发小顺 | **G1/G2/G5 已填（v0.2）；G3/G4 待补，暂不放行** | 2026-11-02（v0.2） |
| 工业质检（review-audit） | 审大关 | 待补 | 待补 |
| 测试（qa-test） | 测小严 | 待补 | 待补 |
| 技术负责人（Lead） | — | 待补 | 待补 |

---

*本文件由 release-ops（发小顺）起草；**只写 `docs/dsh-matrix/20261102-kl8-five-layer/release/` 作用域**，未改业务代码、未提交、未删任何文件。所有 `待补` 必须由 Lead 终态证据填充后方可放行。*
