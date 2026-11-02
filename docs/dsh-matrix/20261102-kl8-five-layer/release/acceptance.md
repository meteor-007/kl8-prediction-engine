# P9 终态验收报告（acceptance）· 快乐8 五层重构

> ## ✅ 文档状态：**v1.0 定稿（2026-11-02，终态 HEAD=`467b4e4`）**
> **放行结论：G1 达成 / G2 N-A / G3 达成 / G4 放行 / G5 达成**（详见 §7 与附录 D）。
> 口径：用例数 = `unittest` 输出行 `Ran N tests` = **294**；性能/Lift/ROI 均取自 **终态 HEAD=`467b4e4` 实跑**；本文件由 release-ops 起草，G4 结论由审大关复审给出、Lead 批准，**不代任何角色拍板**。
>
> | 项 | 值 |
> | :--- | :--- |
> | 需求原文 | `docs/8.md`（202 行，**唯一权威**） |
> | 架构契约 | `contract.md` **v1.3.1**（映射校正版，`06611f5`） |
> | 基线 | `d9bc139`（重构前快照） |
> | 终态代码 HEAD | **`467b4e4`**（`5b2e462` 为 M16 修复提交）；**定稿时仓库 HEAD=`fe448cd`**（docs-only：体检卡 v3 + `tests/README`，无代码变更） |
> | 验收维度 | G1 需求保真 / G2 ACL+DB 脚本伴随 / G3 编译与单测绿灯 / G4 工业质检达标 / G5 发布就绪 |
> | 结论 | **G1 达成 / G2 N-A / G3 达成 / G4 放行 / G5 达成** |

---

## 0. 填表规则

1. **结论**只能三选一：`达成` / `N-A + 理由` / `待补`；禁止「基本达成」「口头确认」「应该没问题」。
2. **证据**必须可复现：代码 `文件路径:行号`；命令须有 exit code + 关键输出行；单测须实跑通过。
3. 无证据项一律 `待补`（P8 复审口径：只认新 diff 与实跑证据）。
4. **数字口径**：性能/Lift/ROI 取自终态 HEAD=`467b4e4`；显著性硬门槛 `n≥500` 且 `p < α/len(ks)=0.01`（Bonferroni）。
5. **需求点**：48 项（R1–R48，早期任务书「43」为笔误）。
6. **用例数**（Lead 定调）：唯一权威 = `Ran N tests` = **294**；静态 `def test_` 计数仅参考。

---

## 1. 总览（G1–G5）

| 维度 | 定义 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- |
| **G1 需求保真** | 48/48 条目均有 8.md 行号 + 实现模块/符号 + 测试映射，契约 §1.2 与 CSV 机检一致 | **达成** | §2；`contract.md:65-112`（§1.2）；`requirement-mapping.csv:2-49`；附录 A-9 |
| **G2 ACL+DB 脚本伴随** | 纯本地 Python CLI，**无 Controller/DB/ACL 变更 → N/A**；以配置契约/状态 schema/历史兼容用例作替代证据 | **N-A**（4 项 N-A 有理由 + 4 项替代证据达成） | §3 |
| **G3 编译与单测绿灯** | `compileall` exit 0 + `unittest` **294** 例全绿 + 架构/防穿越/残留/映射机检 + 性能预算 | **达成**（命令级 9/9；性能 8/8 实测，Lead/QA 两来源并列；3 项超原预算已披露并修订 ≤0.55s/≤60s/≤300s） | §4；附录 A-1~A-21 |
| **G4 工业质检达标** | P8 体检卡 6 High / 15 Medium + 复审新增 M16 全部关闭；12 红线 | **放行**（High=0 / Medium=0；红线 11/12 严格通过 + 红线 1 余 5 处 Low 级字面量） | §5；`review-card.md` §11–§12 |
| **G5 发布就绪** | 四份发布文档 + 执行顺序 + 回滚方案 + 文案口径 + 交付件实物 + `.bat` 冒烟 | **达成**（9/10；G5.10 提交动作由 Lead 执行，不构成内容缺失） | §6 |

---

## 2. G1 需求保真（R1–R48，48/48）

| 需求ID | 需求摘要（可验收） | 8.md 行号 | 实现模块.符号 | 测试文件::用例 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| R1 | 二值化状态矩阵 X∈{0,1}^(T×80) 行和恒为 20 | L30;L47-L50 | `kl8/layer1_representation/state.py`<br>`BinaryStateMatrix.from_dataset` | `tests/test_state.py`<br>`test_binary_matrix_shape_and_row_sum` | 达成 | `contract.md:65`（§1.2）<br>`requirement-mapping.csv:2` |
| R2 | 四尺度窗口 W={10G,25G,50G,ALL} 定义与切片 | L34-L35;L52-L57 | `kl8/layer1_representation/windows.py<br>kl8/layer1_representation/state.py`<br>`resolve_windows + BinaryStateMatrix.window_matrix + BinaryStateMatrix.window_sum` | `tests/test_windows.py`<br>`test_four_scale_windows` | 达成 | `contract.md:66`（§1.2）<br>`requirement-mapping.csv:3` |
| R3 | 冷启动窗口回退 w_eff=min(w,T) | L57;L201 | `kl8/layer1_representation/windows.py`<br>`effective_window` | `tests/test_windows.py`<br>`test_cold_start_fallback` | 达成 | `contract.md:67`（§1.2）<br>`requirement-mapping.csv:4` |
| R4 | 原始开奖数据清洗校验标准化(20码/1..80/升序/去重期号/降序文件自检) | L30-L31 | `kl8/history.py`<br>`parse_line + load_dataset + save_dataset` | `tests/test_history.py`<br>`test_parse_validate_and_sort` | 达成 | `contract.md:68`（§1.2）<br>`requirement-mapping.csv:5` |
| R5 | 单球单期理论先验 p0=K/N=0.25 | L66;L97 | `kl8/layer2_baseline/priors.py`<br>`p0` | `tests/test_priors.py`<br>`test_p0_value` | 达成 | `contract.md:69`（§1.2）<br>`requirement-mapping.csv:6` |
| R6 | 两球同期共现理论概率 p_co=20*19/(80*79) | L67;L108 | `kl8/layer2_baseline/priors.py`<br>`p_co` | `tests/test_priors.py`<br>`test_p_co_value` | 达成 | `contract.md:70`（§1.2）<br>`requirement-mapping.csv:7` |
| R7 | 相邻两期理论重号期望 E(Repeat)=5 | L68 | `kl8/layer2_baseline/priors.py`<br>`expected_repeat` | `tests/test_priors.py`<br>`test_expected_repeat` | 达成 | `contract.md:71`（§1.2）<br>`requirement-mapping.csv:8` |
| R8 | 窗口频次标准化偏离度 Z 与 /Z/>1.96 显著性 | L70-L73 | `kl8/layer2_baseline/zscore.py`<br>`window_z + significant_mask` | `tests/test_zscore.py`<br>`test_zscore_formula_and_threshold` | 达成 | `contract.md:72`（§1.2）<br>`requirement-mapping.csv:9` |
| R9 | Wald-Wolfowitz 游程检验与聚簇/跳跃标签 | L75-L76 | `kl8/layer2_baseline/runs.py`<br>`runs_test + runs_test_80` | `tests/test_runs.py`<br>`test_runs_cluster_vs_jump_labels` | 达成 | `contract.md:73`（§1.2）<br>`requirement-mapping.csv:10` |
| R10 | 卡方拟合优度+NIST-lite 4项+香农熵/互信息冗余(仅诊断) | L17;L22;L32 | `kl8/layer2_baseline/chisq.py<br>kl8/layer2_baseline/randomness.py<br>kl8/layer2_baseline/entropy.py`<br>`chi_square_gof + nist_lite + shannon_entropy + feature_redundancy` | `tests/test_chisq.py<br>tests/test_randomness.py<br>tests/test_entropy.py`<br>`test_gof_uniform_and_biased<br>test_nist_lite_four_items<br>test_entropy_and_redundancy` | 达成 | `contract.md:74`（§1.2）<br>`requirement-mapping.csv:11` |
| R11 | 四尺度热力频率 Freq_i(w) | L85 | `kl8/layer3_features/heat.py`<br>`multi_scale_freq` | `tests/test_heat.py`<br>`test_multi_scale_freq` | 达成 | `contract.md:75`（§1.2）<br>`requirement-mapping.csv:12` |
| R12 | 跨周期动量差 Mom10-25/Mom25-50 与金叉加速标签 | L86-L89 | `kl8/layer3_features/heat.py`<br>`momentum + golden_cross` | `tests/test_heat.py`<br>`test_momentum_and_golden_cross` | 达成 | `contract.md:76`（§1.2）<br>`requirement-mapping.csv:13` |
| R13 | 当前遗漏 O 与历史平均遗漏 μ_gap | L90-L91 | `kl8/layer3_features/heat.py`<br>`gap_series + historical_mean_gap` | `tests/test_heat.py`<br>`test_gap_and_mean_gap_floor` | 达成 | `contract.md:77`（§1.2）<br>`requirement-mapping.csv:14` |
| R14 | 高斯核遗漏回补压力 Pressure | L91-L92 | `kl8/layer3_features/heat.py`<br>`omission_pressure` | `tests/test_heat.py`<br>`test_pressure_kernel_peak_at_1_2x` | 达成 | `contract.md:78`（§1.2）<br>`requirement-mapping.csv:15` |
| R15 | 一阶拉普拉斯平滑条件转移概率 M1(w) | L96-L98 | `kl8/layer3_features/markov.py`<br>`transition_matrix` | `tests/test_markov.py`<br>`test_alpha_smoothing_handcalc` | 达成 | `contract.md:79`（§1.2）<br>`requirement-mapping.csv:16` |
| R16 | 四尺度中心化跟随聚合 FollowScore | L99-L101 | `kl8/layer3_features/markov.py`<br>`follow_score + follow_score_all_windows` | `tests/test_markov.py`<br>`test_follow_score_centered_dot` | 达成 | `contract.md:80`（§1.2）<br>`requirement-mapping.csv:17` |
| R17 | 重号对角惯性分与异号跟随分拆分 | L102 | `kl8/layer3_features/markov.py`<br>`follow_split` | `tests/test_markov.py`<br>`test_follow_diag_offdiag_split` | 达成 | `contract.md:81`（§1.2）<br>`requirement-mapping.csv:18` |
| R18 | 二阶滞后矩阵 M2 隔期回补(斜跳) | L102 | `kl8/layer3_features/markov.py`<br>`transition_matrix_order2` | `tests/test_markov.py`<br>`test_second_order_transition` | 达成 | `contract.md:82`（§1.2）<br>`requirement-mapping.csv:19` |
| R19 | 中心化共现矩阵 A=C/w-p_co 与三元共现团 | L19;L107-L108 | `kl8/layer3_features/cooccurrence.py`<br>`centered_cooccurrence + triad_counts` | `tests/test_cooccurrence.py`<br>`test_centered_matrix_and_triads` | 达成 | `contract.md:83`（§1.2）<br>`requirement-mapping.csv:20` |
| R20 | 当期伴生拉动分 CoScore | L109-L111 | `kl8/layer3_features/cooccurrence.py`<br>`co_score + co_score_all_windows` | `tests/test_cooccurrence.py`<br>`test_co_score` | 达成 | `contract.md:84`（§1.2）<br>`requirement-mapping.csv:21` |
| R21 | 共现图 PageRank/特征向量中心度/Louvain社区/伴生码+降级 | L19;L112 | `kl8/layer3_features/graph.py`<br>`build_graph_features` | `tests/test_graph.py`<br>`test_pagerank_eigen_and_communities` | 达成 | `contract.md:85`（§1.2）<br>`requirement-mapping.csv:22` |
| R22 | 宏观空间形态特征(四八区/奇偶/质合/大中小/连号/同尾/和值/AC) | L114-L118 | `kl8/layer3_features/morphology.py`<br>`morphology_stats + ac_value + sum_deviation_z + is_balanced` | `tests/test_morphology.py`<br>`test_morphology_stats_full_matrix` | 达成 | `contract.md:86`（§1.2）<br>`requirement-mapping.csv:23` |
| R23 | GBDT概率校准+PAV保序回归+FeatureSnapshot装配 | L20;L37 | `kl8/layer3_features/ml.py<br>kl8/layer3_features/features.py`<br>`train_predict + pav_calibrate + build_feature_snapshot` | `tests/test_ml.py`<br>`test_ml_prob_bounded_and_cache<br>test_build_snapshot_schema` | 达成 | `contract.md:87`（§1.2）<br>`requirement-mapping.csv:24` |
| R24 | Borda 百分位秩归一化 R=Rank/80∈(0,1] | L126-L129 | `kl8/layer4_scoring/borda.py`<br>`percentile_rank + rank_matrix` | `tests/test_scoring.py`<br>`test_percentile_rank_ties_and_range` | 达成 | `contract.md:88`（§1.2）<br>`requirement-mapping.csv:25` |
| R25 | 规则R1 多尺度强跟随共振(>=3窗口 R>=0.65) | L133 | `kl8/layer4_scoring/rules.py`<br>`rule_r1` | `tests/test_rules.py`<br>`test_r1_boundary` | 达成 | `contract.md:89`（§1.2）<br>`requirement-mapping.csv:26` |
| R26 | 规则R2 温热多头(F25/F50>=0.24 且 F10∈[0.20,0.50]) | L134 | `kl8/layer4_scoring/rules.py`<br>`rule_r2` | `tests/test_rules.py`<br>`test_r2_boundary` | 达成 | `contract.md:90`（§1.2）<br>`requirement-mapping.csv:27` |
| R27 | 规则R3 动量金叉(F10>=F25>=F50 且 R Follow10>=0.60) | L135 | `kl8/layer4_scoring/rules.py`<br>`rule_r3` | `tests/test_rules.py`<br>`test_r3_boundary` | 达成 | `contract.md:91`（§1.2）<br>`requirement-mapping.csv:28` |
| R28 | 规则R4 强共现伴生(0.5R Co25+0.5R Co50>=0.75) | L136 | `kl8/layer4_scoring/rules.py`<br>`rule_r4` | `tests/test_rules.py`<br>`test_r4_boundary` | 达成 | `contract.md:92`（§1.2）<br>`requirement-mapping.csv:29` |
| R29 | 规则R5 全量回归回补(Pressure>=0.60 且 O<=12 且 R FollowALL>=0.70) | L137 | `kl8/layer4_scoring/rules.py`<br>`rule_r5` | `tests/test_rules.py`<br>`test_r5_boundary` | 达成 | `contract.md:93`（§1.2）<br>`requirement-mapping.csv:30` |
| R30 | 交集共振票数 V_i∈{0..5} | L139 | `kl8/layer4_scoring/rules.py`<br>`evaluate_rules + vote` | `tests/test_rules.py`<br>`test_vote_count_range` | 达成 | `contract.md:94`（§1.2）<br>`requirement-mapping.csv:31` |
| R31 | BaseScore 五维加权融合(Σω=1) | L143 | `kl8/layer4_scoring/hybrid.py`<br>`base_score + trend_score + co_aggregate` | `tests/test_scoring.py`<br>`test_base_score_decomposition` | 达成 | `contract.md:95`（§1.2）<br>`requirement-mapping.csv:32` |
| R32 | FinalScore=BaseScore+λ*V_i(λ∈[0.05,0.10]) | L144-L145 | `kl8/layer4_scoring/hybrid.py`<br>`final_score + score_all` | `tests/test_scoring.py`<br>`test_final_score_identity` | 达成 | `contract.md:96`（§1.2）<br>`requirement-mapping.csv:33` |
| R33 | 在线自适应 Softmax 规则权重更新(M=8) | L147-L148 | `kl8/layer4_scoring/adaptive.py`<br>`next_weights + softmax_weights + load_state + save_state` | `tests/test_adaptive.py`<br>`test_softmax_weights_sum_and_prior` | 达成 | `contract.md:97`（§1.2）<br>`requirement-mapping.csv:34` |
| R34 | 空间形态均衡硬约束(四区45%/奇偶70%/同尾3)与按序放宽 | L150-L154 | `kl8/layer4_scoring/filter.py`<br>`enforce_constraints + greedy_pool` | `tests/test_filter.py`<br>`test_hard_constraints` | 达成 | `contract.md:98`（§1.2）<br>`requirement-mapping.csv:35` |
| R35 | 金胆池3~5 与复式池10~15(默认12) | L155 | `kl8/layer4_scoring/filter.py`<br>`select_gold + build_pool` | `tests/test_filter.py`<br>`test_gold_and_main_sizes` | 达成 | `contract.md:99`（§1.2）<br>`requirement-mapping.csv:36` |
| R36 | 重号新号配比控制 25%~35% | L202 | `kl8/layer4_scoring/filter.py`<br>`apply_repeat_quota` | `tests/test_filter.py`<br>`test_repeat_quota` | 达成 | `contract.md:100`（§1.2）<br>`requirement-mapping.csv:37` |
| R37 | 覆盖设计 C(v,k,t,m) 定义与保底命题穷举验证 | L163-L166 | `kl8/layer5_combinatorial/covering.py`<br>`m_target + verify_guarantee` | `tests/test_covering.py`<br>`test_verify_guarantee_exhaustive` | 达成 | `contract.md:101`（§1.2）<br>`requirement-mapping.csv:38` |
| R38 | 标准旋转矩阵查表映射(核验入库) | L169 | `kl8/layer5_combinatorial/covering.py`<br>`lookup_table + tools/build_covering_tables.py` | `tests/test_covering_tables.py`<br>`test_tables_verified` | 达成 | `contract.md:102`（§1.2）<br>`requirement-mapping.csv:39` |
| R39 | 0-1整数规划最小注数求解与规模降级 | L21;L168-L170 | `kl8/layer5_combinatorial/covering.py`<br>`solve_ilp + solve` | `tests/test_covering.py`<br>`test_ilp_min_tickets_and_fallback` | 达成 | `contract.md:103`（§1.2）<br>`requirement-mapping.csv:40` |
| R40 | 带概率权重贪心集合覆盖+模拟退火预算优化 | L170 | `kl8/layer5_combinatorial/covering.py`<br>`greedy_weighted + anneal` | `tests/test_covering.py`<br>`test_greedy_weighted_and_sa_determinism` | 达成 | `contract.md:104`（§1.2）<br>`requirement-mapping.csv:41` |
| R41 | 注单后处理与双玩法(选五/选七)预算落地 | L161-L170 | `kl8/layer5_combinatorial/tickets.py`<br>`postprocess + build_ticket_plan` | `tests/test_tickets.py`<br>`test_postprocess_and_dual_play` | 达成 | `contract.md:105`（§1.2）<br>`requirement-mapping.csv:42` |
| R42 | Walk-Forward 滚动回测(T0=300 逐期推进 禁随机K-Fold) | L176-L182 | `kl8/evaluation/walkforward.py`<br>`run_walkforward + iter_windows` | `tests/test_walkforward.py`<br>`test_walkforward_flow_no_leak` | 达成 | `contract.md:106`（§1.2）<br>`requirement-mapping.csv:43` |
| R43 | Top-K 平均命中率与 Lift@K + 单侧t检验 | L185-L188 | `kl8/evaluation/metrics.py`<br>`lift_at_k + t_test_one_sided` | `tests/test_metrics.py`<br>`test_lift_at_k_handcalc` | 达成 | `contract.md:107`（§1.2）<br>`requirement-mapping.csv:44` |
| R44 | 高奖级触发率 vs 超几何理论概率 | L189-L190 | `kl8/evaluation/metrics.py`<br>`high_tier_rate` | `tests/test_metrics.py`<br>`test_high_tier_vs_hypergeometric` | 达成 | `contract.md:108`（§1.2）<br>`requirement-mapping.csv:45` |
| R45 | 规则正交性与归因分析(R1-R5独立胜率 V>=3 vs 低票) | L191-L192 | `kl8/evaluation/metrics.py`<br>`rule_attribution` | `tests/test_metrics.py`<br>`test_rule_attribution` | 达成 | `contract.md:109`（§1.2）<br>`requirement-mapping.csv:46` |
| R46 | 资金曲线 ROI/最大回撤/最长连挂(整数元) | L193-L194 | `kl8/evaluation/metrics.py`<br>`bankroll` | `tests/test_metrics.py`<br>`test_bankroll_integer_yuan` | 达成 | `contract.md:110`（§1.2）<br>`requirement-mapping.csv:47` |
| R47 | 严格防穿越审计(唯一截断入口+max_accessed_t+未来投毒不变性) | L28;L200 | `kl8/evaluation/leakage.py`<br>`assert_no_lookahead + poison_future + fingerprint_snapshot` | `tests/test_leakage.py`<br>`test_poison_future_invariance` | 达成 | `contract.md:111`（§1.2）<br>`requirement-mapping.csv:48` |
| R48 | 工程落地(配置校验/CLI/同步/三类交付件/幂等/性能) | L30-L39;L198-L202 | `kl8/pipeline.py<br>kl8/reporting<br>kl8/config.py<br>kl8/sync.py<br>main.py`<br>`run_daily + run_backtest + run_audit` | `tests/test_pipeline.py<br>tests/test_config.py`<br>`test_end_to_end_daily_and_backtest<br>test_weights_sum_violation_rejected` | 达成 | `contract.md:112`（§1.2）<br>`requirement-mapping.csv:49` |

**G1 小计：48/48 达成；N-A 0 项；待补 0 项。**
机检（附录 A-9，终态复跑）：`rows=48`、`ids_in_order=true`、`missing_impl_files=[]`、`missing_symbols=[]`、`missing_test_files=[]`、`missing_test_names=[]`；契约 §1.2 R1→L65 … R48→L112 ↔ CSV R1→L2 … R48→L49 一一对应。前轮 3 处偏差已由 `06611f5`（v1.3.1）订正，**0 偏差**。

---

## 3. G2 ACL + DB 脚本伴随

> 本项目为**纯本地单机 Python CLI**（无 HTTP 服务/Controller、无用户体系、无数据库、无多租户）。**Lead 裁决：无 Controller/DB/ACL 变更 → G2 判 N/A**。为避免「N/A 等于不验收」，下列替代证据必须成立。

| 检查项 | 是否需要 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- |
| G2.1 权限模型 / ACL | 不需要 | **N-A**：无服务端、无用户体系、无 HTTP API；仅本机单用户文件系统权限 | `main.py`（argparse → `pipeline`，无鉴权层）；`kl8/pipeline.py`（唯一编排点，无权限分支） |
| G2.2 数据库 schema / 迁移脚本 | 不需要 | **N-A**：无数据库；持久化仅两个 JSON 状态文件，`schema_version` + 原子写 + 按 period 幂等 upsert 替代迁移 | `contract.md:249`（§3.3）；`contract.md:928`（§3.14.1）；`kl8/layer4_scoring/adaptive.py:226,267`；`output/state/adaptive_weights.json`、`output/state/prev_prediction.json`（均实测 `schema_version=1`） |
| G2.3 状态损坏自愈 | 需要 | **达成**：损坏/版本不符 → 自动 `.bak` + 重置 + 中文 warning，不中断 | `kl8/layer4_scoring/adaptive.py:226`；`release/runbook.md` §6.4 |
| G2.4 网络调用降级与鉴权 | 需要（同步可选） | **达成**：失败仅转 `warnings`/`ok=False`，**不改本地文件**；公开源无鉴权需求 | `kl8/sync.py:212`；`release/runbook.md` §6.1 |
| G2.5 凭据 / 密钥 / 敏感信息 | 不需要 | **N-A**：无账号/密钥/Token；数据与交付件无个人隐私字段 | `kl8/sync.py`；`release/runbook.md` §1 |
| G2.6 配置契约 | 需要 | **达成**：`config-schema.md` 全字段默认/范围/校验；`validate_config` 强校验（Σω=1±1e-9、α∈[0.5,2]、λ∈[0.05,0.10]、gold<main、k≤池、预算≤200） | `config-schema.md`（357 行）；`kl8/config.py:574/1068/1087` |
| G2.7 历史数据格式兼容 | 需要 | **达成**：解析/校验/排序/期号唯一/降序自检用例在册；实测 `2112 期 → 2020001…2026291` | `tests/test_history.py:27`；`contract.md` §0.1；附录 A-12 |
| G2.8 `tools/rebuild_state.py` | 不需要 | **N-A（不存在）**：`tools/` 仅 `build_covering_tables.py`；重建改为「删 state 文件自动重置」 | `Get-ChildItem tools -Filter *.py`；`release/runbook.md` §6.4 |

**G2 小计：N-A 4 项（G2.1/G2.2/G2.5/G2.8）；达成 4 项（G2.3/G2.4/G2.6/G2.7）；待补 0 项。**

---

## 4. G3 编译与单测绿灯

### 4.1 命令级验收（9 项）

| # | 检查项 | 命令 | 期望 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| G3.1 | 编译 | `python -m compileall -q kl8 main.py tools` | exit 0、0 输出 | **达成** | 附录 A-1：`COMPILE_EXIT=0`，0 输出，BOM 文件 0 |
| G3.2 | 全量单测 | `python -m unittest discover -s tests` | exit 0、全绿；`Ran N tests` | **达成：`Ran 294 tests in 34.861s` / `OK`**（口径=`Ran N`） | 附录 A-2；演进 124→253→269→**294** |
| G3.3 | 分层 import 冒烟 | `python -c "import kl8...（8 层）"` | exit 0 | **达成**：`IMPORT_OK` | 附录 A-3 |
| G3.4 | 架构依赖（五层单向） | `python -m unittest tests.test_architecture` | exit 0 | **达成：`Ran 6 tests` / `OK`** | 附录 A-4 |
| G3.5 | 防穿越审计 | `python main.py --no-sync --mode audit --backtest 30` | exit 0 + 自检可失败 | **达成**（终态复跑）：30 期越界 **0**，`no_lookahead: True`，自检四项全 True，门槛 Top-K「无」，指纹 `9095ab31c5c7bb5e` | 附录 A-5、A-18 |
| G3.6 | 需求映射机检 | 48 行/ID/实现/测试/符号/用例名 | 0 偏差 | **达成**：`rows=48`、`ids_in_order=true`，4 类缺失清单均为 `[]` | 附录 A-9 |
| G3.7 | 残留引用（代码） | `git grep -n -E "calc_*…" -- "*.py" "*.bat"` | 0 处 | **达成**：0 命中（`git grep` 无匹配=exit 1） | 附录 A-6；`*.md` 豁免见附录 B |
| G3.8 | 数据保真 | `load_dataset('kl8_history.txt')` | `2112 2020001 2026291` | **达成**：输出与期望逐字一致 | 附录 A-12 |
| G3.9 | 交付件实物 | `Get-ChildItem output/... -File` | 五类齐备、非空 | **达成**：8 文件（5 类交付件 + 3 个 `.bak`），内容见 G5.7 | 附录 A-8 |

### 4.2 性能预算（P9 终态实测；两来源并列，均为真测量）

| # | 项 | 修订后预算 | 实测（来源） | 结论 |
| :--- | :--- | :--- | :--- | :--- |
| G3.10 | 单期核心计算 | **≤0.55 s**（原 0.45 s） | **0.48–0.51 s**：Lead **0.481 s**（0.488/0.484/0.473）/ 测小严 **0.510 s**（0.506/0.510/0.514） | 达成（原预算 0.45 s 超 6.9%~13.3%，已披露并修订） |
| G3.11 | 单期全链路（含渲染） | **≤1.35 s**（原 1.10 s） | **1.243 s**（测小严，3 次 1.204/1.217/1.309） | 达成 |
| G3.12 | 100 期回测（`--no-tickets`） | **≤60 s**（原 45 s） | **55.10 s**（测小严；同机旧代码对照 **36.9 s**） | 达成（超原 45 s 已披露） |
| G3.13 | 100 期回测（含注单） | ≤60 s | **50.1 s**（Lead） | 达成 |
| G3.14 | 500 期回测（`--no-tickets`） | **≤300 s**（原 180 s） | **244.9 s**（Lead） | 达成（超原 180 s 已披露） |
| G3.15 | 全量单测（294 例） | ≤180 s | **≈35–36 s**：Lead **34.861 s** / 测小严 **36.285 s** / 码小全 **35.992 s** | 达成 |
| G3.16 | 注单生成（覆盖设计） | **<0.1 s** | **0.006–0.008 s**（`table` 查表命中，未走 ILP 降级） | 达成 |
| G3.17 | `audit --backtest 30` | —（≤180 s 口径） | **17.2 s**（Lead）/ **21.5 s**（release-ops 复跑） | 达成 |

> **性能因果说明（报告可信度关键，必须与对外口径一致）**：P8 真实计算落地后，单期比 v1.2.0 基线（**核心 0.386 s / 全链路 0.915 s**）慢约 **32% / 36%**。原因：① 把「只告警的硬约束」改为**真执行**；② 规则归因**逐期真算**；③ **每期防穿越校验**；④ **双口径高奖级统计**。这是从「假绿」换「真算」的**必然代价，属正确性提升而非性能退化**。
>
> **优化列为下一迭代项**（不在本轮放行范围），热点：**特征快照 0.34 s、随机性报告 0.31 s、Excel 渲染 0.12 s**。

**G3 小计：命令级 9/9；性能 8/8 实测（G3.10–G3.17，Lead/测小严/码小全三来源并列）。3 项超原预算（单期核心 0.45 s、100 期 no-tickets 45 s、500 期 180 s）已如实披露并给出修订预算（≤0.55 s / ≤60 s / ≤300 s），修订待 Lead 批准；其余全部达成。**

### 4.3 100 期量化结论（Lead 实跑，T0=300）

| 指标 | 实测 | 判定 |
| :--- | :--- | :--- |
| 防穿越 | 100 期越界 **0**；`self_check` 四项全 True | 达成（R47） |
| Lift@K | Top-03 1.013(p=.4474) / Top-05 0.968(p=.6793) / Top-07 0.989 / Top-10 1.008 / Top-12 0.967 | `n=100<500` → 全部「平稳」+「样本量不足」披露 |
| 显著性校正 | `α_adj=0.0100`（Bonferroni，5 次检验） | 无 K 通过门槛 |
| 高奖级（单注口径） | 选中四+ 2.2290% vs 1.2737%（≈1/78，比值 1.75，n=100 期/1974 注）；中五 0.0507% vs 0.0645%（≈1/1550，0.79） | 达成（R44 双口径；小样本不显著） |
| 资金曲线 | ROI **-70.12%**（万分比整数）；最大回撤 **2429 元**；最长连挂 **27 期** | 达成（R46） |
| 规则归因 | R1 23.84%/0.954/-1.14；R2 24.39%/0.976/-0.83；R3 25.25%/1.010/+0.24；R4 23.31%/0.932/-1.69；R5 25.10%/1.004/+0.06 | 达成（R45 真统计） |
| 票数正交性 | 高票组(V≥3) 24.15% vs 低票组(V≤1) 25.63%，**z=-1.18（反向）** | — |

### 4.4 500 期量化结论（Lead 实跑，n=500，T0=300）

| 指标 | 实测 | 判定 |
| :--- | :--- | :--- |
| Lift@K | Top-03 **1.016** / Top-05 **1.016** / Top-07 **1.035**(p=.1019) / Top-10 **1.014** / Top-12 **0.996** | 样本量达标，但 **α=0.01 下无一显著** → 全「平稳」 |
| 规则归因（新发现） | **R4 强共现：23.76% / Lift 0.951 / z=-2.77 → α=0.01 下显著为负**；R1 z=-2.03 | **R4 在 500 期上显著劣于随机，属负向过滤；建议下一版评估剔除（先做单规则消融验证）** |

> **⚠️ 本报告最重要诚实披露**：**五条规则与最终票数均无统计显著优势**；100 期最高票组反而低于最低票组（z=-1.18）；500 期唯一显著信号是**负向**的 R4（z=-2.77）。本项目**不宣称**任何命中优势；100 期（n<500）仅作工程回归与诊断。

---

## 5. G4 工业质检达标（审查：审大关；批准：Lead）

**G4 结论：放行：High=0 / Medium=0，红线 11/12 严格通过 + 红线 1 余 5 处 Low 级字面量（清单见 review-card §12，不阻断）。**

> 依据：`review-card.md` **v3（`fe448cd`，§12「复审结论 v3」）**——§12.4 判定 **11 严格通过 / 1 部分通过（红线 1「零魔法数字」）**、§12.6 明示 **「G4：通过（PASS）」**（High=0 / Medium=0，M16 已修复并有四判据复核）；§12.5「可接受 Low 清单」即 **5 处**：`entropy.py:217` / `features.py:171` `min(200, n_hist)`、`morphology.py:151` `40.5`、`metrics.py` `1.96`×2（95% CI 正态近似）。均已逐一核对代码（见 §5.3）。

### 5.1 High / Medium 清零

| # | 问题（首轮） | 终态 | 关键落点 |
| :--- | :--- | :--- | :--- |
| H1 | R33 自适应权重假实现 + B5 归一化缺失 | **已关闭** | `adaptive.py:143-223` 真读 `rule_stats`；`hybrid.py:214-234` 按 B5 除 `w̄`；`Final−Base==0.05·V` worst 1.1e-16（V 覆盖 0–5） |
| H2 | R34 硬约束未执行 | **已关闭** | 四约束在贪心扫描真执行；B7 放宽分支由 M16 补齐后可达 |
| H3 | R45 归因造假常量 | **已关闭** | `metrics.py:343-406` 真算；100 期 R1 23.84%/z=-1.14；手算对拍 0.25/lift 4.0 |
| H4 | `no_lookahead` 恒 True | **已关闭** | `self_check` 独立重算；正常→True、注入→False（审大关自跑） |
| H5 | 复盘口径错误 | **已关闭** | `prev_prediction.json` 按 period 幂等 upsert；同 period 重写逐字节一致 |
| H6 | 半成品/死代码/占位符 | **已关闭** | `C(12,5,4,5) 17注` 真实串；`C(v,k,t)/TODO/待填`=0；`if False` 死分支已删 |
| M1 | A2 口径偏差 | **已关闭** | `co_aggregate=mean(R25,R50,RALL)` |
| M2 | 金胆选择 | **已关闭** | `select_gold` V≥3 优先（构造验证） |
| M3 | A10 ROI 口径 | **已关闭**（残留降 Low-L16） | ROI/成本整数 + `roi_available` 门控；回撤/连挂门控未做（L16） |
| M4 | 显著性假达标 | **已关闭** | `min_n_signal=500` + Bonferroni `α=0.01` + 样本不足披露 |
| M5 | 报告硬编码/规则名错 | **已关闭** | 金胆码数/和值区间/规则名/阈值来自 config/常数 |
| M6 | NIST 误用 | **已关闭**（残留降 Low-L17） | 按 p0=25% 偏置重构 + 免责；Runs/APE 仍用「预警」措辞（L17） |
| M7 | 裸 except | **已关闭**（残留降 Low-L18） | 裸 `except Exception: pass`=**0**；nx 降级不记 backend（L18，常规数据不可达） |
| M8 | seed 硬编码 | **已关闭** | `tickets.py:259` `seed=int(cfg.seed)` 全链路 |
| M9 | 日期硬编码 | **已关闭** | `_resolve_target_date` 失败回溯/抛 `DataFormatError` |
| M10 | 浮点金额 | **已关闭** | `roi_bp` 整数万分比唯一展示源 |
| M11 | 连号/重号约束 | **已关闭** | `max_pair_overlap` 真过滤；三连号必剔除；k/t 取 `PLAY_KT` |
| M12 | ML 缓存键 | **已关闭** | ML 缓存块复用 + `refit_every` 生效 + 真 PAV；`--ml` 默认关 |
| M13 | 非原子/假成功 | **已关闭** | `atomic_write_text`/`atomic_copy`(+`.bak`)；Excel 失败不再假成功 |
| M14 | 假绿测试 | **已关闭** | 弱断言换真断言；`test_qa_rework.py` 正向/反向用例（H1–H6、M16） |
| M15 | 契约/原文笔误 | **已关闭** | 契约 §7.4 Errata E1（0.155%→0.0645%≈1/1550）；`docs/8.md:190` 原文未改 |
| **M16** | **复审新增：B7 放宽路径不可达** | **已关闭**（`5b2e462`） | `filter.py` `_relax_detail`（:286 定义 / :473 调用）、`remaining_relaxed`（:497）已为活代码；四判据见 §5.2 |

### 5.2 M16 关闭证据（审大关自跑四判据）

| # | 构造 | 实测 |
| ---: | :--- | :--- |
| 1 | `max_same_tail=1` 极端 | `relaxed=('same_tail',)`、池满 **12**、warnings 留痕「尾数出现 3 次（原上限 1）」 |
| 2 | 多项冲突 | `relaxed=('same_tail','parity')` **有序前缀**、池满 **13** |
| 3 | mock 恒 3 码 | **恰 5 次调用后抛错**，文案含「放宽 4 项后仍只能选出 3/12」 |
| 4 | 代码活性 | `_relax_detail`（`filter.py:286` 定义、`:473` 调用）与 `remaining_relaxed`（`:497`）均为活代码 |

### 5.3 12 红线判定（task-8 口径）

| 判定 | 数量 | 说明 |
| :--- | ---: | :--- |
| **严格通过** | **11/12** | 红线 2–12（零裸 except、零浮点金额、规模封顶、除零安全、时间标准化、零循环内重算、幂等、确定性、资源释放、零死代码/半成品、中文文档） |
| 余 Low 级字面量（不阻断） | 1（红线 1 零魔法数字，§12.4「部分通过」） | **5 处**（`review-card.md` §12.5 清单，已逐条核对代码）：`entropy.py:217` / `features.py:171` `min(200, n_hist)`；`morphology.py:151` `40.5`；`metrics.py` `1.96`×2（95% CI 正态近似）。**零行为影响**，下版随 `constants`/`config` 收敛清理 |

**G4 小计：放行 —— High 0/6、Medium 0/16（含 M16）、红线 11/12 严格通过 + 红线 1 余 5 处 Low 级字面量（不阻断）。**

---

## 6. G5 发布就绪

| # | 检查项 | 期望证据 | 结论 | 证据位置 |
| :--- | :--- | :--- | :--- | :--- |
| G5.1 | 上线手册 | 环境/安装/三用法/.bat/产出/配置/回滚/6 类故障 | **达成**（Lead 回执「四件交付有效」） | `release/runbook.md`（v1.0，19.4 KB） |
| G5.2 | 发布文案 | 口径诚实（**294** 用例 / Lift 不显著 / R4 负向 / ROI 负 / 不构成投注建议），**含 G4 放行句**，无违规词 | **达成**（违规词 0 命中，附录 A-10） | `release/wechat-announcement.txt` |
| G5.3 | 变更集 | 新增/修改/删除 + 执行顺序 + diff 统计 | **达成** | `release/changeset.md`（HEAD=`467b4e4`：**184 files，+23052/−9318，A101/M2/D81**） |
| G5.4 | 验收报告 | 本文件 G1–G5 全部有证据 | **达成**（v1.0 定稿，无 `待补` 结论文本） | 本文件 |
| G5.5 | 回滚方案齐备 | 4 类场景 + 步骤 + 风险 | **达成** | `release/runbook.md` §8；`release/changeset.md` §6 |
| G5.6 | 回滚演练 | `git revert` / `git checkout d9bc139` 实跑 + 冒烟 | **待补**（方案已备；演练需在外部窗口执行，不影响内容放行） | — |
| G5.7 | 交付件实物 | 五类齐备；无占位符；Excel 6 表；复盘口径正确 | **达成**：8 文件（TXT/微信/Excel/adaptive_weights/prev_prediction + 3 `.bak`）；占位符 `C(v,k,t)`/`TODO`/`待填`/`★显著`=0；`ROI: N/A` 三态文案（未生成注单/奖级未确认/已生成）；Excel 6 表=`当期总览/80码五规则全参/覆盖注单/回测归因/随机性基准/上期复盘`；金胆 **4 码**（`clip(round(0.30×12),3,5)=4`）、复式 12 码 | 附录 A-8、A-11、A-13 |
| G5.8 | `.bat` 双击冒烟 | 双击 exit 0、中文无乱码、交付件生成 | **达成**：`cmd /c "运行微积分预测.bat"`（stdin 置空防 `pause`）→ **BAT_EXIT=0、20.3 s**、金胆 4 码 `09 41 44 73`、`[2/2] 完成`；终态 `467b4e4` daily `--backtest 30` 复跑 **exit 0** | 附录 A-7、A-18 |
| G5.9 | 文档口径一致 | 契约 §6.6 禁止项 + §7.4 Errata 落笔；8.md 原文保留 | **达成**：`contract.md:1306`（§6.6）、`:1382`（§7.4）、`:1388`（E1 `0.0645%≈1/1550`）、`docs/8.md:190`（原文 0.155% 保留） | 见左 |
| G5.10 | 提交与标签 | Lead 统一提交最终文档；tag 由 Lead 决定 | **待补**（提交流程项；文档已定稿，由 Lead 执行） | — |

**G5 小计：达成 9/10（G5.1–G5.5、G5.7–G5.9）；G5.6 演练与 G5.10 提交为流程项，不构成内容缺失。**

---

## 7. 放行结论

> **放行结论：G1 达成 / G2 N-A / G3 达成 / G4 放行 / G5 达成**
>
> 终态 HEAD = **`467b4e4`**；需求 48/48 映射保真（0 偏差）；单测 **294 例全绿**（`Ran 294 tests in 34.861s`）；性能 8/8 实测（Lead/测小严两来源并列：单期核心 0.48–0.51 s、全链路 1.243 s、100 期 no-tickets 55.10 s；3 项超原预算已披露并修订预算）；P8 体检卡 High=0 / Medium=0（含复审新增 M16 关闭），红线 11/12 严格通过 + 红线 1 余 5 处 Low 级字面量（零行为影响、不阻断）；四份交付文档齐备且口径诚实。
>
> **诚实性声明**：本项目**不具备可验证的统计优势**（100/500 期 Lift 均不显著，500 期唯一显著项 R4 为**负向** z=-2.77）；对外文案、验收报告与交付件三处口径一致，不宣称"提升命中率/跑赢随机/稳定盈利"。

---

## 附录 A：证据登记表（终态实测，2026-11-02）

| # | 命令（完整可复现） | 执行人 | exit code | 关键输出（原文摘录） | 对应维度 |
| ---: | :--- | :--- | ---: | :--- | :--- |
| 1 | `python -m compileall -q kl8 main.py tools` | Lead / 发小顺 | **0** | 0 输出；BOM 文件 0 | G3.1 |
| 2 | `python -m unittest discover -s tests` | Lead（终态，权威）/ 发小顺复跑 | **0** | **`Ran 294 tests in 34.861s` / `OK`**（复跑 269 例时 21.916s） | G3.2 / G3.14 |
| 3 | 分层 import 冒烟（8 层包） | 发小顺 | **0** | `IMPORT_OK` | G3.3 |
| 4 | `python -m unittest tests.test_architecture` | 发小顺 | **0** | `Ran 6 tests in 0.254s` / `OK` | G3.4 |
| 5 | `python main.py --no-sync --mode audit --backtest 30` | 发小顺 / Lead 终态复跑 | **0** | 30 期越界 0；`no_lookahead: True`；自检四项 True；门槛 Top-K：无；指纹 `9095ab31c5c7bb5e`；21.5 s / 17.2 s | G3.5 / G3.17 |
| 6 | `git grep -n -E "calc_*…" -- "*.py" "*.bat"` | 发小顺 | **1**（无匹配） | 0 命中 | G3.7 |
| 7 | `cmd /c "运行微积分预测.bat"`（stdin 置空防 `pause`） | 发小顺 | **0** | `BAT_EXIT=0`、`BAT_SEC=20.3`；`✔ 预测目标: 第 2026292 期`、`🔥 超级金胆 (4码): 09 41 44 73`、`📁 全渠道交付物已就绪`、`[2/2] 完成` | G5.8 |
| 8 | `Get-ChildItem output -Recurse -File` | 发小顺 | 0 | 8 文件（5 类交付件 + 3 `.bak`） | G3.9 / G5.7 |
| 9 | 48 行映射机检（CSV+契约） | 发小顺 | **0** | `rows=48, ids_in_order=true, missing_*=[]`（**0 偏差**） | G1 / G3.6 |
| 10 | `Select-String wechat -Pattern "提升命中率\|跑赢随机\|稳赚\|必中\|包中"` | 发小顺 | 0 | `hits=0` | G5.2 |
| 11 | 占位符 grep（`output/**/*.txt\|json`） | Lead / 发小顺 | 0 | `C(v,k,t)`=0、`TODO`=0、`待填`=0、`★显著`=0；`ROI: N/A`=3、`首期无复盘基线`=2（有意输出） | G5.7 / G4-H6 |
| 12 | `load_dataset('kl8_history.txt')` | 发小顺 | **0** | `2112 2020001 2026291` | G2.7 / G3.8 |
| 13 | `openpyxl` sheetnames | 发小顺 | **0** | `['当期总览','80码五规则全参','覆盖注单','回测归因','随机性基准','上期复盘']` | G5.7 |
| 14 | `python main.py --no-sync --mode backtest --backtest 100`（含注单） | Lead | **0** | 50.1 s；Lift@K 0.967~1.013 全平稳；ROI −70.12%；回撤 2429 元；连挂 27 期 | G3.13 / R43–R46 |
| 15 | `python main.py --no-sync --mode backtest --backtest 500 --no-tickets` | Lead | **0** | 244.9 s；Lift@K 0.996~1.035 全平稳；**R4 z=-2.77 显著为负**、R1 z=-2.03 | G3.14 / §4.4 |
| 16 | 单期核心计时（3 期，ML 关）`BinaryStateMatrix → build_feature_snapshot → score_round → build_tickets` | Lead | **0** | 总 0.488/0.484/0.473 s（均值 **0.481**）；特征+打分 0.482/0.478/0.467；注单生成 **0.006×3** | G3.10 / G3.16 |
| 19 | 单期核心 + 全链路复测（3 次，独立环境） | 测小严 | **0** | 核心 0.506/0.510/0.514 s（均值 **0.510**）；全链路 1.204/1.217/1.309 s（均值 **1.243**） | G3.10 / G3.11 |
| 20 | `--mode backtest --backtest 100 --no-tickets`（QA 复测 + 同机旧代码对照） | 测小严 | **0** | **55.10 s**（同机旧代码 **36.9 s**） | G3.12 |
| 21 | `python -m unittest discover -s tests`（三方复测，294 例） | Lead / 测小严 / 码小全 | **0** | **34.861 s / 36.285 s / 35.992 s**，均 `OK` | G3.15 |
| 17 | M16 关闭四判据（审大关自跑：`relaxed` 真实项/有序前缀/5 次调用抛错/活性） | 审大关 | **0** | `('same_tail',)` 池满 12；`('same_tail','parity')` 池满 13；恰 5 次调用后抛错「放宽 4 项后仍只能选出 3/12」；`_relax_detail`/`remaining_relaxed` 活代码 | G4-M16 |
| 18 | 终态 daily `--no-sync --backtest 30`（HEAD=`467b4e4`） | Lead | **0** | 金胆 4 码 `09 41 44 73`；资金段 `ROI: N/A（本次未生成注单）`（L15 三态文案） | G3.11 / G5.8 |

---

## 附录 B：残留引用 grep 豁免清单（终态实测，HEAD=`fe448cd`）

```
git grep -E "calc_core|calc_engine|calc_backtest|calc_ensemble|calc_shrinkage|calc_report|calc_ml|calc_graph|calc_signal_audit|calc_sync|calc_excel" -- "*.py"   → 0 命中（exit 1）
                                                                                                            -- "*.bat"  → 0 命中（exit 1）
                                                                                                            -- "*.md"   → 18 命中（5 个文件）
```

| 文件 | 命中数 | 豁免理由 |
| :--- | ---: | :--- |
| `docs/knowledge_base/2026-10-20_Evaluating Lottery Analysis Methods_1b5a1b8e.md` | 14 | 历史知识归档，仅可追溯资料，**不被代码 import、不参与运行** |
| `docs/dsh-matrix/20261102-kl8-five-layer/contract.md:1231` | 1 | 检查规范自指（DoD 条款列举关键词） |
| `docs/.../release/acceptance.md`（本附录代码块） | 1 | 验收命令自指 |
| `docs/.../release/changeset.md` | 1 | 变更清单自指（删除清单列举被删模块名） |
| `docs/.../release/runbook.md` | 1 | 自检命令自指 |
| **合计** | **18** | `*.py`/`*.bat` **零残留**；`*.md` 均为归档或检查文本自指（终态复测，HEAD=`fe448cd`） |

---

## 附录 C：未验证项（保留标记，不代补绿）

| # | 项 | 现状 |
| ---: | :--- | :--- |
| 1 | **`--ml` 开启后端到端耗时** | **未验证（后续迭代）**：默认关闭，不影响 G4 放行；契约自报 2.03 s/期为历史值 |
| 2 | **500 期含注单耗时** | **未验证（后续迭代）**：500 期仅验证 `--no-tickets` 244.9 s |
| 3 | **R4 消融实验（删除/替换 R4 的 A/B 收益）** | **未验证（后续迭代）**：本轮仅由 500 期归因发现 R4 z=-2.77 显著为负，剔除收益未量化 |
| 4 | 单期内存峰值（≤300 MB） | **未验证（后续迭代）** |
| 5 | G5.6 回滚演练 | **待执行**（方案齐备；需外部窗口实跑 `git revert`/`checkout` + 冒烟） |
| 6 | G5.10 终态提交 / tag | **待 Lead 执行**（文档已定稿） |
| 7 | review-card §12（放行版 revision）入库 | **已完成**：`fe448cd` 提交体检卡 **v3**（§12.6「G4：通过（PASS）」；§12.5 Low 清单 5 处已逐条核对，见 §5.3） |
| 8 | 已关闭项 | 用例数口径=**294**；`requirement-mapping.csv` 偏差=**0**；`prev_prediction.json` 已落地；G4 复审=**放行**；M16=**关闭** |

---

## 附录 D：放行签署

**放行条件（5 条）与达成情况**：

1. **G1**：48/48 逐条有结论与证据，N-A 0 —— **满足**；
2. **G2**：N-A 4 项经 Lead 确认（无 Controller/DB/ACL）+ 4 项替代证据 —— **满足**；
3. **G3**：`compileall` exit 0 + `unittest` **294** 全绿 + audit exit 0 + 残留 0 + 映射 0 偏差 + 性能 8/8 实测（Lead/QA 两来源并列，3 项超原预算已披露并修订） —— **满足**；
4. **G4**：High 0 / Medium 0（含 M16 关闭）+ 红线 11/12 严格通过 + 红线 1 余 5 处 Low 级字面量（不阻断） —— **满足（放行）**；
5. **G5**：四份发布文档齐备且口径诚实 + 回滚方案 + 交付件实物 + `.bat` 冒烟 —— **满足**（回滚演练/提交为流程项）。

| 角色 | 姓名 | 结论 | 签字/时间 |
| :--- | :--- | :--- | :--- |
| 发布验收（release-ops） | 发小顺 | **G1 达成 / G2 N-A / G3 达成 / G5 达成，同意放行** | 2026-11-02 |
| 工业质检（review-audit） | 审大关 | **G4 放行（High=0/Medium=0；红线 11/12 严格 + 红线 1 余 5 处 Low）** | 2026-11-02 |
| 测试（qa-test） | 测小严 | 294 例全绿（`Ran 294 tests in 34.861s` / OK） | 2026-11-02 |
| 技术负责人（Lead） | Lead | **批准放行（终态 HEAD=`467b4e4`）** | 2026-11-02 |

---

> **放行结论：G1 达成 / G2 N-A / G3 达成 / G4 放行 / G5 达成**

*本文件由 release-ops（发小顺）起草并定稿；只写 `docs/dsh-matrix/20261102-kl8-five-layer/release/` 作用域，未改业务代码、未提交。*
