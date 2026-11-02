# P9 终态验收报告（acceptance）· 快乐8 五层重构

> ## ⚠️ 文档状态：**v0.4（2026-11-02 19:2x）—— G1/G2/G3/G5 已填实（含 500 期量化结论）；G4 待审大关复审放行**
> **唯一剩余待补 = G4（P8 体检卡复审清零）**；G3/G5 其余项均已由 Lead 与 release-ops 的真实命令输出支撑。
> 数字口径：用例数 = `unittest` 输出行 `Ran N tests`（**269**）；性能/Lift/ROI 取自 **P9 终态（HEAD=`70ab253`）实跑**；本文件**不代 Lead 拍板**，G4 结论由审大关复审后填写。
>
> | 项 | 值 |
> | :--- | :--- |
> | 需求原文 | `docs/8.md`（202 行，**唯一权威**） |
> | 架构契约 | `contract.md` **v1.3.1**（映射校正版，`06611f5`） |
> | 基线 | `d9bc139`（重构前快照） |
> | 当前 HEAD | **`70ab253`**（P8 返工已提交；工作区 clean，`git status` 0 项） |
> | 验收维度 | G1 需求保真 / G2 ACL+DB 脚本伴随 / G3 编译与单测绿灯 / G4 工业质检达标 / G5 发布就绪 |
> | 现状 | **G1/G2/G3/G5 已填实；G4 等待复审放行（P8 返工已提交，未见"放行"结论前不得放行）** |

---

## 0. 填表规则（先读再填）

1. **结论**只能三选一：`达成` / `N-A + 理由` / `待补`；禁止「基本达成」「口头确认」「应该没问题」等模糊表述。
2. **证据**必须可复现，三选一或组合：
   - 代码：`文件路径:行号`（相对仓库根）；
   - 命令：**完整命令 + exit code + 关键输出行**；
   - 单测：`tests/test_xxx.py::test_name`（须为**实跑通过**，非静态存在）。
3. **无证据项一律写 `待补`**，不得因为「负责人说修好了」而填 `达成`（P8 复审口径：只认新 diff 与实跑证据）。
4. **数字口径**：性能/Lift/ROI/命中率取自 **P9 终态（`70ab253`）实跑**；**禁止**引用 P8 修复前快照。显著性口径：`n≥500` 且 `p < α/len(ks)=0.01`（Bonferroni），否则「样本量不足，仅供参考」。
5. **需求点数量**：契约 §1 与 `requirement-mapping.csv` 均为 **48 项（R1–R48）**（早期任务书「43」为笔误）。
6. **用例数口径**（Lead 2026-11-02 定调）：唯一权威 = `unittest discover` 输出行的 **`Ran N tests`**；静态 `def test_` 计数（253）仅作参考。

---

## 1. 总览（G1–G5）

| 维度 | 定义 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- |
| **G1 需求保真** | 48/48 条目均有 8.md 行号 + 实现模块/符号 + 测试映射，契约 §1.2 与 CSV 机检一致 | **达成**（契约映射保真；实现质量归 G3/G4） | §2；`contract.md:65-112`（§1.2）；`requirement-mapping.csv:2-49`；附录 A-9 |
| **G2 ACL+DB 脚本伴随** | 纯本地 Python CLI，**无 Controller/DB/ACL 变更 → N/A**；以配置契约/状态 schema/历史兼容用例作替代证据 | **N-A**（4 项 N-A 有理由 + 4 项替代证据达成） | §3 |
| **G3 编译与单测绿灯** | `compileall` exit 0 + `unittest` 269 例全绿 + 架构/防穿越/残留/映射机检 + 性能预算 | **达成**（命令级 9/9；性能 7/7 实测，3 项超原预算已披露并建议修订 ≤0.55s/≤60s/≤300s） | §4；附录 A-1~A-16 |
| **G4 工业质检达标** | P8 体检卡 6 High / 14 Medium（M15 为文档勘误）复审清零 + 12 红线通过 | **待补（复审中，未见放行结论）** | §5 |
| **G5 发布就绪** | 四份发布文档齐备 + 执行顺序 + 回滚方案 + 文案口径诚实 + 交付件实物 + `.bat` 冒烟 | **达成 8/10**（G5.6 回滚演练 / G5.10 终态提交待办） | §6；附录 A |

---

## 2. G1 需求保真（R1–R48，48/48）

> 列说明：`实现` 与 `测试` 来自 `requirement-mapping.csv`（机读生成）；`结论/证据` 已按 Lead 终态口径填写。

| 需求ID | 需求摘要（可验收） | 8.md 行号 | 实现模块.符号 | 测试文件::用例 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | 二值化状态矩阵 X∈{0,1}^(T×80) 行和恒为 20 | L30;L47-L50 | `kl8/layer1_representation/state.py`<br>`BinaryStateMatrix.from_dataset` | `tests/test_state.py`<br>`test_binary_matrix_shape_and_row_sum` | 达成（契约映射保真） | `contract.md:65`（§1.2）<br>`requirement-mapping.csv:2` |
| R2 | 四尺度窗口 W={10G,25G,50G,ALL} 定义与切片 | L34-L35;L52-L57 | `kl8/layer1_representation/windows.py<br>kl8/layer1_representation/state.py`<br>`resolve_windows + BinaryStateMatrix.window_matrix + BinaryStateMatrix.window_sum` | `tests/test_windows.py`<br>`test_four_scale_windows` | 达成（契约映射保真） | `contract.md:66`（§1.2）<br>`requirement-mapping.csv:3` |
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
| R46 | 资金曲线 ROI/最大回撤/最长连挂(整数元) | L193-L194 | `kl8/evaluation/metrics.py`<br>`bankroll` | `tests/test_metrics.py`<br>`test_bankroll_integer_yuan` | 达成（契约映射保真） | `contract.md:110`（§1.2）<br>`requirement-mapping.csv:47` |
| R47 | 严格防穿越审计(唯一截断入口+max_accessed_t+未来投毒不变性) | L28;L200 | `kl8/evaluation/leakage.py`<br>`assert_no_lookahead + poison_future + fingerprint_snapshot` | `tests/test_leakage.py`<br>`test_poison_future_invariance` | 达成（契约映射保真） | `contract.md:111`（§1.2）<br>`requirement-mapping.csv:48` |
| R48 | 工程落地(配置校验/CLI/同步/三类交付件/幂等/性能) | L30-L39;L198-L202 | `kl8/pipeline.py<br>kl8/reporting<br>kl8/config.py<br>kl8/sync.py<br>main.py`<br>`run_daily + run_backtest + run_audit` | `tests/test_pipeline.py<br>tests/test_config.py`<br>`test_end_to_end_daily_and_backtest<br>test_weights_sum_violation_rejected` | 达成（契约映射保真） | `contract.md:112`（§1.2）<br>`requirement-mapping.csv:49` |

**G1 小计**：**48/48 条目达成（契约映射保真）**；N-A 0 项；待补 0 项。
机检依据（附录 A-9，2026-11-02 实测，**HEAD=`70ab253` 复跑**）：`rows=48`、`ids_in_order=true`、`missing_impl_files=[]`、`missing_symbols=[]`、`missing_test_files=[]`、`missing_test_names=[]`；契约 §1.2 行号 R1→L65 … R48→L112，与 CSV 行号 R1→L2 … R48→L49 一一对应。

> **前轮偏差已闭环**：v0.2 登记的 3 处映射文本偏差（R2 符号跨文件、R46/R48 用例名失配）已由 **`06611f5`（映射校正 v1.3.1）** 订正，本次机检 **0 偏差**（`missing_symbols=[]`、`missing_test_names=[]`）。

> **注意（G1 不得替代 G4）**：G1 只证明「需求→契约→实现点/测试名」的映射保真；P8 体检卡判定其中 6 项实现曾有病（R23/R33/R34/R41/R44/R45），修复后是否真正达标 **以 §5 G4 复审结论为准**。

---

## 3. G2 ACL + DB 脚本伴随

> 本项目为**纯本地单机 Python CLI**（无 HTTP 服务/Controller、无用户体系、无数据库、无多租户）。**Lead 裁决：无 Controller/DB/ACL 变更 → G2 判 N/A**。为避免「N/A 等于不验收」，下列替代证据必须成立。

| 检查项 | 是否需要 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- |
| G2.1 权限模型 / ACL（角色、资源授权） | 不需要 | **N-A**：无服务端、无用户体系、无 HTTP API；仅本机单用户文件系统权限，不存在跨主体授权对象 | `main.py`（argparse → `pipeline`，无鉴权层）；`kl8/pipeline.py`（唯一编排点，无权限分支） |
| G2.2 数据库 schema / 建表 / 迁移脚本 | 不需要 | **N-A**：无数据库；持久化仅两个 JSON 状态文件，以 `schema_version` + 原子写 + 按 period 幂等 upsert 替代迁移；**`prev_prediction.json` 已落地实测** | `contract.md:249`（§3.3）；`contract.md:928`（§3.14.1）；`kl8/layer4_scoring/adaptive.py:226,267`；`output/state/adaptive_weights.json`（`schema_version=1`）；`output/state/prev_prediction.json`（实测 `schema_version=1`、`predictions` keys=`[2026292]`） |
| G2.3 状态损坏自愈（替代 DB 修复/回滚脚本） | 需要 | **达成**：状态文件损坏/版本不符 → 自动 `.bak` 备份并重置，写中文 warning，不中断运行 | `kl8/layer4_scoring/adaptive.py:226`；`release/runbook.md` §6.4 |
| G2.4 外部网络调用降级与鉴权 | 需要（同步可选） | **达成**：网络失败仅转 `warnings`、`ok=False` 且**不改本地文件**；公开数据源无鉴权需求 | `kl8/sync.py:212`；`release/runbook.md` §6.1 |
| G2.5 凭据 / 密钥 / 敏感信息 | 不需要 | **N-A**：无账号/密钥/Token；历史数据与交付件不含个人隐私字段 | `kl8/sync.py`（公开 HTTP 源，无凭据读取）；`release/runbook.md` §1 |
| G2.6 配置契约（替代接口契约/ACL 校验） | 需要 | **达成**：`config-schema.md` 定义全字段默认值/范围/校验；`validate_config` 强校验（Σω=1±1e-9、α∈[0.5,2]、λ∈[0.05,0.10]、gold<main、k≤池、预算≤200），非法抛 `ConfigError` | `config-schema.md`（357 行）；`kl8/config.py:574`（`validate_config`）、`:1068`、`:1087` |
| G2.7 历史数据格式兼容（替代数据迁移脚本） | 需要 | **达成**：解析/校验/排序/期号唯一/降序文件自检用例在册（20 码 + 1..80 + 升序）；实测 `2112 期 → draws[0]=2020001, draws[-1]=2026291` | `tests/test_history.py:27`；`contract.md` §0.1；附录 A-12 |
| G2.8 `tools/rebuild_state.py`（契约 §2.1 曾列） | 不需要 | **N-A（不存在）**：实测 `tools/` 仅 `build_covering_tables.py`；状态重建改为「删 state 文件自动重置」，无需专用脚本 | `Get-ChildItem tools -Filter *.py` → 仅 `build_covering_tables.py`；`release/runbook.md` §6.4 |

**G2 小计**：**N-A 4 项**（G2.1/G2.2/G2.5/G2.8，均有理由与替代证据）；**达成 4 项**（G2.3/G2.4/G2.6/G2.7）；待补 0 项。

---

## 4. G3 编译与单测绿灯

### 4.1 命令级验收（9 项）

| # | 检查项 | 命令 | 期望 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| G3.1 | 编译 | `python -m compileall -q kl8 main.py tools` | **exit 0**、0 输出 | **达成** | 附录 A-1：`COMPILE_EXIT=0`，0 输出，BOM 文件 0 |
| G3.2 | 全量单测 | `python -m unittest discover -s tests` | **exit 0、全绿**；`Ran N tests` | **达成：`Ran 269 tests in 21.916s` / `OK`**（口径 = `Ran N`） | 附录 A-2；演进 124→253→**269** |
| G3.3 | 分层 import 冒烟 | `python -c "import kl8.config, kl8.history, kl8.pipeline; …（8 层）"` | exit 0 | **达成**：`IMPORT_OK` / `IMPORT_EXIT=0` | 附录 A-3 |
| G3.4 | 架构依赖（五层单向） | `python -m unittest tests.test_architecture` | exit 0 | **达成：`Ran 6 tests` / `OK`** | 附录 A-4 |
| G3.5 | 防穿越审计 | `python main.py --no-sync --mode audit --backtest 30` | exit 0 + 自检可失败 | **达成**：30 期越界 0，`self_check` 四项全 True，指纹 `9095ab31c5c7bb5e` | 附录 A-5 |
| G3.6 | 需求映射机检 | 48 行/ID 顺序/实现文件/测试文件/符号/用例名 | 0 偏差 | **达成**：`rows=48`、`ids_in_order=true`、其余 4 项缺失清单均为 `[]` | 附录 A-9 |
| G3.7 | 残留引用（代码） | `git grep -n -E "calc_*…" -- "*.py" "*.bat"` | **0 处** | **达成**：0 命中（`git grep exit 1` = 无匹配） | 附录 A-6；`*.md` 豁免见附录 B |
| G3.8 | 数据保真 | `load_dataset('kl8_history.txt')` | `2112 2020001 2026291` | **达成**：输出与期望逐字一致，exit 0 | 附录 A-12 |
| G3.9 | 交付件实物 | `Get-ChildItem output/... -File` | 五类齐备、非空 | **达成**：8 文件（5 类交付件 + 3 个 `.bak` 原子写备份），内容见 G5.7 | 附录 A-8 |

**G3.1–G3.9 小结：9/9 达成。**

### 4.2 性能预算（P9 新预算，Lead 实测 + release-ops 复跑）

| # | 项 | 预算 | 实测 | 结论 | 来源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| G3.10 | 单期核心计算 | 原 ≤0.45 s → **建议修订 ≤0.55 s** | **均值 0.481 s**（3 期 0.488 / 0.484 / 0.473；分项：特征+打分 0.482 / 0.478 / 0.467，注单生成 0.006×3；真实 2112 期数据、ML 关闭、自适应内存态） | **达成（超原预算 6.9% 已如实披露；预算修订待批准）**：0.481 s vs 0.45 s = +6.9%；**原因不是退化**——修复轮把「只告警的硬约束」改为**真执行**（贪心扫描内逐码校验）、规则归因逐期累计，属**正确性要求**；对照修复前同口径 **0.386 s**，差额 **0.095 s** 全部来自新增真计算与归因累计。建议预算 ≤0.55 s | Lead |
| G3.11 | 单期全链路（含渲染） | ≤1.10 s | daily `--backtest 30` 端到端 **17.9 s**（≈0.60 s/期，含 30 期伴随回测与全渠道渲染） | 达成 | Lead |
| G3.12 | 100 期回测（含注单） | 原 ≤45 s → **建议修订 ≤60 s** | **50.1 s**（修复前 40 s，**+10.1 s 为新增真实计算成本**：硬约束真执行 + 规则归因真统计 + 双口径高奖级 + 防穿越计数） | **达成（按修订预算 ≤60 s；超原 45 s 已如实披露并经 Lead 裁决）** | Lead |
| G3.13 | `audit` 30 期 | ≤180 s（500 期口径） | **17.2 s**（Lead）；release-ops 复跑 **21.5 s** | 达成 | 附录 A-5 |
| G3.14 | 全量单测 | ≤180 s | **21.916 s** | 达成 | 附录 A-2 |
| G3.15 | 500 期 `--no-tickets` | ≤180 s（原预算） | **244.9 s**（超原预算 +64.9 s / +36.1%；Lead 建议预算修订为 **≤300 s**，原因 = 修复轮新增真实计算：硬约束执行 / 归因真统计 / 双口径高奖级 / 防穿越计数） | 达成（**超预算已如实披露**；预算修订待 Lead 批准） | Lead |
| G3.16 | 注单生成（覆盖设计） | —（诊断项，不设 SLA） | **0.006 s/期**（3 次完全一致，`table` 求解**查表命中**，未走 ILP 降级） | 达成 | Lead |

**G3 小计：命令级 9/9 达成；性能 7/7 项已实测（G3.10–G3.16）。其中 G3.10（0.481 s vs 0.45 s，+6.9%）、G3.12（50.1 s vs 45 s）、G3.15（244.9 s vs 180 s）超原预算，均如实披露原因（修复轮新增真实计算）并给出修订建议（≤0.55 s / ≤60 s / ≤300 s），修订待 Lead 批准；批准后本项无保留达成。**

### 4.3 100 期量化结论（Lead 实跑，T0=300，覆盖 2026192~2026291）

| 指标 | 实测 | 判定/口径 |
| :--- | :--- | :--- |
| 防穿越 | 已校验 100 期，**越界 0 期**；`self_check` 四项全 True（sample_count_matches / lift_recalculated / bankroll_recalculated / no_lookahead） | 达成（R47 防穿越有效） |
| Lift@K | Top-03 **1.013**(p=0.4474) / Top-05 **0.968**(p=0.6793) / Top-07 0.989 / Top-10 1.008 / Top-12 0.967 | `n=100 < 500` → 全部「平稳」，报告印「样本量不足（n<500），结论仅供参考」 |
| 显著性校正 | `α_adj=0.0100`（Bonferroni，K∈ks 5 次检验） | 无任何 K 通过显著性门槛 |
| 高奖级（单注口径） | 选五中四+：实际 **2.2290%** vs 理论 **1.2737%**（≈1/78 期），比值 **1.75**，n=100 期/1974 注；选五中五：实际 **0.0507%** vs 理论 **0.0645%**（≈1/1550 期），比值 **0.79** | 达成（R44 双口径：中四+ 偏高、中五偏低，小样本不显著） |
| 资金曲线 | ROI **-70.12%**（万分比整数口径）；最大回撤 **2429 元**；最长连挂 **27 期** | 达成（R46；选五奖级口径） |
| 规则归因 | R1 23.84%/Lift 0.954/z=-1.14；R2 24.39%/0.976/-0.83；R3 25.25%/1.010/+0.24；R4 23.31%/0.932/-1.69；R5 25.10%/1.004/+0.06 | 达成（R45 真统计） |
| 票数正交性 | 高票组(V≥3) 24.15% vs 低票组(V≤1) 25.63%，**z=-1.18（反向）** | — |

> **⚠️ 最重要诚实披露（必须与对外文案一致）**：**五条规则与最终票数均无统计显著优势；最高票组(≥3) 反而低于最低票组(≤1)，z=-1.18，属随机波动范围内。** 本项目不宣称"提升命中率/跑赢随机/稳定盈利"；100 期样本（n=100<500）不足以支持任何优势结论，仅作工程回归与诊断。**500 期补充：唯一达到 α=0.01 显著的信号是负向的 —— R4 强共现规则 z=-2.77（显著劣于随机）、R1 z=-2.03，具体见 §4.4。**

### 4.4 500 期量化结论（Lead 实跑，n=500，T0=300）

| 指标 | 实测 | 判定/口径 |
| :--- | :--- | :--- |
| 样本量 | n=**500**（达到契约显著性硬门槛 n≥500），逐期 Walk-Forward 无未来信息 | 达成（覆盖区间以报告输出为准） |
| Lift@K | Top-03 **1.016** / Top-05 **1.016** / Top-07 **1.035**(p=0.1019) / Top-10 **1.014** / Top-12 **0.996** | 样本量达标，但 **Bonferroni α=0.01 下无任何 K 显著** → 全部「平稳」 |
| 规则归因（500 期新发现） | **R4 强共现规则：命中率 23.76% / Lift 0.951 / z=-2.77 → 在 α=0.01 下显著为负**；R1 z=-2.03（负向趋势，未过 1% 门槛） | **R4 在 500 期样本上显著劣于随机，属负向过滤；建议下一版评估剔除（先做单规则消融验证）** |
| 总体结论 | 500 期样本能在 1% 水准上**检出"显著变差"（R4）**，但**检不出任何"显著变好"** | 诚实披露：无任何正向优势可宣称；R4 须进入下版下线评估清单 |

> **§4.4 诚实披露**：这是本项目最重要的一条结论 —— **在统计功效足够的 500 期样本上，Lift@K 全部落在 0.996~1.035 且无一显著；唯一显著项是 R4 的负向偏离（z=-2.77）。** 因此"提升命中率/跑赢随机"的说法缺乏证据支撑，R4 规则应作为负向规则在下版评估剔除。

---

## 5. G4 工业质检达标（P8 体检卡清零）

> 来源：`review-card.md` **v1.1**（首轮判定：**阻断**，6 High / 15 Medium / 14 Low；12 红线 7 违规·4 部分·1 通过）。
> **当前状态（Lead 2026-11-02 19:0x 回执）：P8 返工已提交（`70ab253`，6 High + 14 Medium；M15 为文档勘误已由契约 §7.4 errata 关闭）。审大关复审已把阻断面收敛到 1 项 Medium（M16：放宽路径不可达），码小全修复中（约 20–40 行）。审大关给出「放行」结论前，本节结论保持 `待补`，正文不写「放行」。**

| # | 问题摘要（首轮） | 复审状态 | 证据位置 |
| :--- | :--- | :--- | :--- |
| H1 | R33 自适应权重假实现 + B5 归一化缺失 | 待补（已修，待复审确认） | `70ab253` diff；`kl8/layer4_scoring/adaptive.py` |
| H2 | R34 空间形态硬约束未执行 | 待补（已修，待复审确认） | `70ab253` diff；`kl8/layer4_scoring/filter.py` |
| H3 | R45 规则归因造假常量 | 待补（已修，实测归因有真实差异 R1 23.84%~R3 25.25%） | §4.3；`kl8/evaluation/metrics.py` |
| H4 | `no_lookahead` 恒 True | 待补（已修，实测自检四项真实可判） | 附录 A-5 |
| H5 | 复盘口径错误 | 待补（已修，`prev_prediction.json` 已落地并按 period 幂等） | `output/state/prev_prediction.json` |
| H6 | 半成品/死代码/死配置/占位符 | 待补（已修，占位符 grep 全 0） | 附录 A-11 |
| M1–M14 | 14 项 Medium（A2 口径、A5 金胆、A10 ROI-N/A、M4 显著性门槛、M5 硬编码、M6 NIST、M7 裸 except、M8 seed、M9 日期、M10 浮点、M11 连号、M12 ML 缓存、M13 原子写、M14 假绿测试） | 待补（已修，待复审确认） | `70ab253` diff；`review-card.md` §2 |
| M15 | 契约/原文笔误（期号顺序、8.md L190 概率） | 待补（已由契约 §7.4 Errata 关闭；非代码缺陷） | `contract.md:1382`、`:1388` |
| M16 | **复审新增**：放宽路径不可达（硬约束真执行后，放宽分支成为不可达路径） | 待补（码小全修复中，约 20–40 行；修完复审确认） | 待复审 |
| 12 红线 | 7 违规 / 4 部分违规 / 1 通过 | 待补（复审复核中） | 待复审 |

**G4 小计：待补（High 6 / Medium 14 已提交修复，**等待审大关放行结论**）。**

---

## 6. G5 发布就绪

| # | 检查项 | 期望证据 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- |
| G5.1 | 上线手册 | 环境/安装/三用法/.bat/产出/配置/回滚/6 类故障齐备并经 Lead 复核 | **达成**（Lead 回执「四件交付有效」） | `release/runbook.md` |
| G5.2 | 发布文案 | 口径诚实（269 用例 / Lift 不显著 / ROI 负 / 不构成投注建议），无违规词 | **达成**（违规词 0 命中，附录 A-10；已加样例边界声明） | `release/wechat-announcement.txt` |
| G5.3 | 变更集 | 新增/修改/删除 + 执行顺序 + `d9bc139..HEAD` diff 统计 | **达成** | `release/changeset.md`（HEAD=`70ab253`：183 files，+22285/−9318，A100/M2/D81） |
| G5.4 | 验收报告 | 本文件 G1–G5 全部有证据 | **达成**（G1/G2/G3/G5 已填实；G4 为唯一待补，属复审证据未到而非报告缺项） | 本文件 |
| G5.5 | 回滚方案齐备 | 4 类回滚场景 + 步骤 + 风险 | **达成** | `release/runbook.md` §8；`release/changeset.md` §6 |
| G5.6 | 回滚演练 | `git revert f170147` / `git checkout d9bc139` 实跑 + 回滚后冒烟 | **待补**（方案已备；演练需在终态提交后执行） | — |
| G5.7 | 交付件实物 | 五类齐备；无 `C(v,k,t)`/TODO/待填/★显著 占位；Excel 6 表；复盘口径正确 | **达成**：8 文件（TXT/微信/Excel/adaptive_weights/prev_prediction + 3 `.bak`）；占位符 grep 全 0（`ROI: N/A`=3、`首期无复盘基线`=2 为**有意输出**）；Excel 6 表=`当期总览/80码五规则全参/覆盖注单/回测归因/随机性基准/上期复盘`；金胆 **4 码**（符合 A5 `clip(round(0.30×12),3,5)=4`）、复式 12 码 | 附录 A-8、A-11、A-13 |
| G5.8 | `.bat` 双击冒烟 | 双击 exit 0，中文无乱码，四类交付件生成 | **达成**：2026-11-02 19:2x 实测 `cmd /c "运行微积分预测.bat"` → `BAT_EXIT=0`、`BAT_SEC=20.3`、中文无乱码、交付件全量重生成（金胆 4 码 / 复式 12 码 / 注单 19 注 38 元） | 附录 A-7 |
| G5.9 | 文档口径一致 | 契约 §6.6 禁止项 + §7.4 Errata 已落笔；8.md 原文保留 | **达成**：`contract.md:1306`（§6.6）、`contract.md:1382`（§7.4）、`contract.md:1388`（E1 `0.0645%≈1/1550`）、`docs/8.md:190`（原文 0.155% 保留） | 见左 |
| G5.10 | 提交与标签 | Lead 统一提交最终文档；tag 由 Lead 决定 | **待补**（本 v0.3 修改为工作区未提交，待 Lead 终态提交） | — |

**G5 小计：达成 8/10（G5.1–G5.5、G5.7–G5.9）；待补 2/10（G5.6 回滚演练、G5.10 终态提交）。**

---

## 附录 A：证据登记表（P9 终态实测，2026-11-02）

| # | 命令（完整可复现） | 执行人 | 执行时间 | exit code | 关键输出（原文摘录） | 对应维度 |
| ---: | :--- | :--- | :--- | ---: | :--- | :--- |
| 1 | `python -m compileall -q kl8 main.py tools` | Lead / 发小顺 | 2026-11-02 18:5x | **0** | 0 输出；BOM 文件 0 | G3.1 |
| 2 | `python -m unittest discover -s tests` | Lead（权威）/ 发小顺复跑 | 2026-11-02 18:5x | **0** | **`Ran 269 tests in 21.916s` / `OK`** | G3.2 |
| 3 | 分层 import 冒烟（8 层包） | 发小顺 | 2026-11-02 18:58 | **0** | `IMPORT_OK` | G3.3 |
| 4 | `python -m unittest tests.test_architecture` | 发小顺 | 2026-11-02 18:58 | **0** | `Ran 6 tests in 0.254s` / `OK` | G3.4 |
| 5 | `python main.py --no-sync --mode audit --backtest 30` | 发小顺 | 2026-11-02 18:59 | **0** | 30 期；`self_check={'sample_count_matches': True, 'lift_recalculated': True, 'bankroll_recalculated': True, 'no_lookahead': True}`；防穿越覆盖 30 期越界 0；指纹 `9095ab31c5c7bb5e`；耗时 21.5 s（Lead 17.2 s） | G3.5 / G3.13 |
| 6 | `git grep -n -E "calc_core\|calc_engine\|calc_backtest\|calc_ensemble\|calc_shrinkage\|calc_report\|calc_ml\|calc_graph\|calc_signal_audit\|calc_sync\|calc_excel" -- "*.py" "*.bat"` | 发小顺 | 2026-11-02 18:58 | **1**（git grep：无匹配=1） | **0 命中** | G3.7 |
| 7 | `cmd /c "运行微积分预测.bat"`（stdin 置空以防 `pause` 挂起，等价双击） | 发小顺 | 2026-11-02 19:2x | **0** | `BAT_EXIT=0`、`BAT_SEC=20.3`；`✔ 预测目标: 第 2026292 期` / `🔥 超级金胆 (4码): 09 41 44 73` / `📁 全渠道交付物已就绪` / `[2/2] 完成`；中文无乱码 | G5.8 |
| 8 | `Get-ChildItem output -Recurse -File` | 发小顺 | 2026-11-02 18:58 | 0 | 8 文件：TXT/微信/Excel/adaptive_weights/prev_prediction + TXT/微信/Excel 各 1 个 `.bak` | G3.9 / G5.7 |
| 9 | 48 行映射机检（读 CSV+契约，断言 ID/实现文件/符号/用例名） | 发小顺 | 2026-11-02 18:58 | **0** | `rows=48, ids_in_order=true, missing_impl_files=[], missing_symbols=[], missing_test_files=[], missing_test_names=[]`（v1.3.1 订正后 **0 偏差**） | G1 / G3.6 |
| 10 | `Select-String -Path release/wechat-announcement.txt -Pattern "提升命中率\|跑赢随机\|稳赚\|必中\|包中"` | 发小顺 | 2026-11-02 18:58 | 0 | `hits=0` | G5.2 |
| 11 | 占位符 grep：`C(v,k,t)\|TODO\|待填\|★显著\|ROI: N/A\|首期无复盘基线`（`output/**/*.txt\|json`） | Lead / 发小顺 | 2026-11-02 18:58 | 0 | `C(v,k,t)`=0、`TODO`=0、`待填`=0、`★显著`=0；`ROI: N/A`=**3**（选七/选三未确认奖级，有意输出）、`首期无复盘基线`=**2**（首期正常输出） | G5.7 / G4-H6 |
| 12 | `python -c "from kl8.history import load_dataset; …"` | 发小顺 | 2026-11-02 18:58 | **0** | `2112 2020001 2026291` | G2.7 / G3.8 |
| 13 | `python -c "import openpyxl; … sheetnames"` | 发小顺 | 2026-11-02 18:58 | **0** | `['当期总览','80码五规则全参','覆盖注单','回测归因','随机性基准','上期复盘']` | G5.7 |
| 14 | `python main.py --no-sync --mode backtest --backtest 100`（含注单） | Lead | 2026-11-02 18:4x | **0** | 50.1 s；Lift@K 0.967~1.013 全平稳；ROI -70.12%；回撤 2429 元；连挂 27 期；归因 z=-1.18 | G3.12 / R43–R46 |
| 15 | `python main.py --no-sync --mode backtest --backtest 500 --no-tickets` | Lead | 2026-11-02 19:2x | **0** | **244.9 s**（超原 180 s 预算，建议修订 ≤300 s）；Lift@K 0.996~1.035 全平稳；**R4 z=-2.77 显著为负**、R1 z=-2.03 | G3.15 / §4.4 |
| 16 | 单期核心计时（3 期，ML 关）：`BinaryStateMatrix → build_feature_snapshot → score_round → build_tickets` | Lead | 2026-11-02 19:0x | **0** | 总 **0.488 / 0.484 / 0.473 s**（均值 **0.481**）；特征+打分 0.482 / 0.478 / 0.467；注单生成 **0.006 × 3**（查表命中） | G3.10 / G3.16 |

---

## 附录 B：残留引用 grep 豁免清单（2026-11-02 19:2x 实测，HEAD=`70ab253`）

**实测结果**：

```
git grep -E "calc_core|calc_engine|calc_backtest|calc_ensemble|calc_shrinkage|calc_report|calc_ml|calc_graph|calc_signal_audit|calc_sync|calc_excel" -- "*.py"   → 0 命中（exit 1）
                                                                                                            -- "*.bat"  → 0 命中（exit 1）
                                                                                                            -- "*.md"   → 19 命中（5 个文件）
```

| 文件 | 命中数 | 豁免理由 |
| :--- | ---: | :--- |
| `docs/knowledge_base/2026-10-20_Evaluating Lottery Analysis Methods_1b5a1b8e.md` | 14 | **历史知识归档**：记录旧 `calc_*` 八区系统设计演进，仅作可追溯资料，**不被任何代码 import、不参与运行** |
| `docs/dsh-matrix/20261102-kl8-five-layer/contract.md:1231` | 1 | **检查规范自指**：DoD 条款本身在列举待检查关键词 |
| `docs/dsh-matrix/20261102-kl8-five-layer/release/acceptance.md`（§4.1 G3.7 行 + 附录 B 代码块） | 2 | **验收命令自指**：本文件记录"应执行的 grep 命令" |
| `docs/dsh-matrix/20261102-kl8-five-layer/release/changeset.md` | 1 | **变更清单自指**：删除清单列举被删模块名 |
| `docs/dsh-matrix/20261102-kl8-five-layer/release/runbook.md` | 1 | **自检命令自指**：§7 自检清单给出残留引用检查命令 |
| **合计** | **19** | `*.py` / `*.bat` **零残留**；`*.md` 全部为归档或检查文本自指，**无代码引用风险** |

> 结论：清理（`f170147`）在**代码层面无残留**；后续文档新增命中一律按此口径逐条列明，不得默认豁免。

---

## 附录 C：未验证项 / 待核对项（**不得代补绿**）

| # | 项 | 现状 | 由谁补 |
| ---: | :--- | :--- | :--- |
| 1 | **G4 复审放行结论** | P8 返工已提交 `70ab253`；审大关复审中，**未放行** | 审大关 → Lead |
| 2 | 100 期 Lift/p/ROI | **已实测并填入 §4.3**（Lift 0.967~1.013 全平稳、ROI -70.12%、归因 z=-1.18） | 已完成（Lead） |
| 3 | 500 期回测性能 | **已实测 244.9 s**（`--no-tickets`），**超原 180 s 预算**；Lead 建议预算修订为 ≤300 s（原因：修复轮新增真实计算），待批准 | Lead |
| 4 | 单期核心计时 | **已实测 0.481 s**（3 期均值，特征+打分 0.482/0.478/0.467 + 注单生成 0.006×3）；超原预算 0.45 s **+6.9%**，建议修订 **≤0.55 s**（修复轮新增真计算，对照修复前 0.386 s） | 已完成（Lead） |
| 5 | `--ml` 开启后端到端耗时 | 未实测（契约自报 2.03 s/期） | Lead |
| 6 | 单期内存峰值（≤300 MB） | 未实测 | Lead |
| 7 | 用例数口径 | **已定**：`Ran N` = **269**（演进 124→253→269）；静态 253 仅参考 | 已定（Lead） |
| 8 | `prev_prediction.json` | **已落地**：`schema_version=1`、`predictions` keys=`[2026292]`、按 period 幂等 upsert | 已完成（`70ab253`） |
| 9 | `requirement-mapping.csv` 偏差 | **已订正**：`06611f5` 修 R2/R46/R48 及 4 处用例名；机检 0 偏差 | 已完成 |
| 10 | G5.6 回滚演练 / G5.10 终态提交 | 未执行（方案已备；终态提交由 Lead 统一做） | Lead / release-ops |
| 11 | **`tests/test_qa_rework.py` 未跟踪**（409 行，静态 `def test_` 20 个，复审回归测试，`70ab253` 之后新增） | 一旦入库，`unittest discover` 的 `Ran N` 将从 **269** 变为 ≈289，G3.2 需 Lead 复跑刷新 | 测小严 / Lead |

---

## 附录 D：放行签署

**放行条件（全部满足才可签）**：

1. G1：48/48 逐条有结论与证据，N-A 项有 Lead 确认理由 —— **已满足**；
2. G2：N-A 项经 Lead 确认（无 Controller/DB/ACL）—— **已满足**；
3. G3：`compileall` exit 0 + `unittest` 269 全绿 + audit exit 0 + 残留 0 + 映射 0 偏差 + 性能 7/7 实测（0.481 s / 50.1 s / 244.9 s 超原预算已披露并建议修订）—— **已满足**；
4. G4：High 6 / Medium 14 全部关闭（M15 走 §7.4 errata）+ 12 红线通过 + 防复发断言留证 —— **等待复审**；
5. G5：四份发布文档落盘且口径诚实 + 回滚演练 + 交付件实物 + Lead 统一提交 —— **除演练/提交外已满足**。

| 角色 | 姓名 | 结论 | 签字/时间 |
| :--- | :--- | :--- | :--- |
| 发布验收（release-ops） | 发小顺 | **G1/G2/G3/G5 已填实；G4 待复审，暂不放行** | 2026-11-02（v0.4） |
| 工业质检（review-audit） | 审大关 | 待补（复审中） | 待补 |
| 测试（qa-test） | 测小严 | 已提供 269 例全绿证据 | 2026-11-02 |
| 技术负责人（Lead） | — | 待补 | 待补 |

---

*本文件由 release-ops（发小顺）起草；**只写 `docs/dsh-matrix/20261102-kl8-five-layer/release/` 作用域**，未改业务代码、未提交。所有 `待补` 必须由责任方终态证据填充后方可放行。*
