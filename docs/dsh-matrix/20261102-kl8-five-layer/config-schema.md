# 快乐8 五层重构 · 配置契约（config-schema.md）

> 配套文档：[contract.md](./contract.md)（§3.3 为 `AppConfig` 签名冻结源）
> 版本：**v1.1.0**（同步 contract.md v1.1.0：`m_margin` 语义修正、新增 `lift_signal`、叶子字段计数修正）｜ 2026-11-02 ｜ 作者：req-architect（智小研）
> 定位：本项目**不含数据库/ACL**，G2 的「ACL + DB 脚本伴随物」以**本配置契约**替代 —— 即"所有可调参数有 schema、有默认值、有合法范围、有校验规则、有 CLI 覆盖路径、有可审计的报错文案"。审大关可用本文件逐条核对"零魔法数字"红线。

---

## §1 配置对象总览

`kl8/config.py` 采用**分层 frozen dataclass**，全部字段在 `AppConfig` 下聚合，运行期不可变（回测/报告各自持有同一份 `AppConfig`，避免中途被改）。

```
AppConfig
├─ windows   : WindowConfig      # 四尺度窗口与冷启动
├─ markov    : MarkovConfig      # 拉普拉斯平滑与阶数
├─ weights   : ScoreWeights      # 融合权重、λ、Trend 子权重、ML 权重
├─ rules     : RuleConfig        # R1–R5 五条规则阈值
├─ adaptive  : AdaptiveConfig    # 在线 Softmax 自适应
├─ pool      : PoolConfig        # 金胆/复式池规模与形态硬约束
├─ covering  : CoveringConfig    # 覆盖设计 C(v,k,t,m)、预算、SA/ILP
├─ ml        : MLConfig          # LightGBM / PAV 校准
├─ backtest  : BacktestConfig    # Walk-Forward 与指标 K 集合
├─ history_path / output_dir / seed / eps / cost_per_ticket / prize_table
```

**加载优先级（高 → 低）**：`CLI 参数` > `config.json`（`load_config(path=...)`） > `dataclass 默认值`。
**不启用环境变量覆盖**（避免隐式行为与不可复现）。加载链路：`load_config(overrides) → merge_cli_overrides() → validate_config()`，**任何出单/回测前必须校验通过（fail-fast）**。

---

## §2 `WindowConfig`（`cfg.windows`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `short` | `int` | `10` | `[5, 20]` | 必须 `< mid` | R2、R11、R25–R29 |
| `mid` | `int` | `25` | `[20, 40]` | `short < mid < long` | R2、R11 |
| `long` | `int` | `50` | `[40, 100]` | `> mid`；建议 `long ≥ 2×mid` | R2、R11 |
| `min_history` | `int` | `40` | `[30, 500]` | `≥ long`；`T < min_history` → `InsufficientDataError` | R1–R4、R42 |
| `cold_start_floor` | `int` | `1` | `[1, 10]` | `≤ min(short, …)`；`w_eff = max(floor, min(w, T))` | R3 |

> 窗口键名派生规则（全仓唯一）：`{"10G","25G","50G","ALL"}`；`ALL` 键的 `size=None`，实际 `w_eff = max(1, T)`。

---

## §3 `MarkovConfig`（`cfg.markov`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `alpha` | `float` | `1.0` | `[0.5, 2.0]` | 8.md L97 明示区间；越界 → `ConfigError` | R15 |
| `order` | `int` | `2` | `{1, 2}` | 1 = 只算一阶（`follow2_all` 置 0）；2 = 追加二阶 | R15、R18 |
| `diag_split` | `bool` | `True` | `{True, False}` | False 时 `follow_diag/follow_offdiag` 仍返回但与 `follow_*` 一致口径 | R17 |

---

## §4 `ScoreWeights`（`cfg.weights`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `follow_10` | `float` | `0.10` | `[0, 1]` | 参与 Σω 校验 | R31 |
| `follow_25` | `float` | `0.15` | `[0, 1]` | 同上 | R31 |
| `follow_50` | `float` | `0.15` | `[0, 1]` | 同上 | R31 |
| `follow_all` | `float` | `0.15` | `[0, 1]` | 同上 | R31 |
| `co` | `float` | `0.15` | `[0, 1]` | 同上 | R28、R31 |
| `trend` | `float` | `0.20` | `[0, 1]` | 同上 | R27、R31 |
| `omission` | `float` | `0.10` | `[0, 1]` | 同上 | R14、R31 |
| `ml` | `float` | `0.0` | `[0, 0.30]` | 默认 0（只诊断）；`>0` 时计入 Σω | R23、R31 |
| `lambda_vote` | `float` | `0.08` | `[0.05, 0.10]` | 8.md L145 建议区间，越界 `ConfigError` | R32 |
| `trend_w_f10` | `float` | `0.5` | `[0, 1]` | 三项和 = 1（裁决 A1） | R31 |
| `trend_w_mom_10_25` | `float` | `0.3` | `[0, 1]` | 同上 | R31 |
| `trend_w_mom_25_50` | `float` | `0.2` | `[0, 1]` | 同上 | R31 |

**跨字段硬约束**

| 编号 | 约束 | 失败文案（示例） |
| :--- | :--- | :--- |
| V1 | `follow_10+follow_25+follow_50+follow_all+co+trend+omission+ml == 1.0 ± 1e-9` | `配置项 weights：融合权重之和 = 1.080000，要求 1.0（±1e-9）；建议按比例缩减 follow_50` |
| V2 | 每个 ω `≥ 0` | `配置项 weights.co = -0.15 非法：权重不得为负` |
| V3 | `trend_w_*` 三项和 `== 1.0 ± 1e-9` | 同上格式 |
| V4 | `lambda_vote ∈ [0.05, 0.10]` | `配置项 weights.lambda_vote = 0.20 非法：要求 ∈[0.05, 0.10]；建议 0.08` |

---

## §5 `RuleConfig`（`cfg.rules`）

| 字段 | 类型 | 默认值 | 合法范围 | 语义（8.md 原文） | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `r1_min_windows` | `int` | `3` | `[1, 4]` | R1：4 窗口中至少 N 个进前 35% | R25 |
| `r1_rank_threshold` | `float` | `0.65` | `[0.5, 1.0]` | R1 的 `R ≥ 0.65`（前 35%） | R25 |
| `r2_base_freq` | `float` | `0.24` | `[0.20, 0.30]` | R2：`F25、F50 ≥ 0.24` | R26 |
| `r2_f10_low` | `float` | `0.20` | `[0.10, 0.25]` | R2：`F10 ≥ 0.20` | R26 |
| `r2_f10_high` | `float` | `0.50` | `[0.35, 0.60]` | R2：`F10 ≤ 0.50`（防过热透支） | R26 |
| `r3_follow_rank` | `float` | `0.60` | `[0.5, 1.0]` | R3：`R(Follow_10G) ≥ 0.60`（前 40%） | R27 |
| `r4_rank_score` | `float` | `0.75` | `[0.5, 1.0]` | R4：`0.5R(Co25)+0.5R(Co50) ≥ 0.75` | R28 |
| `r4_w_co25` | `float` | `0.5` | `[0, 1]` | R4 的 25G 权重 | R28 |
| `r4_w_co50` | `float` | `0.5` | `[0, 1]` | R4 的 50G 权重；`r4_w_co25 + r4_w_co50 == 1` | R28 |
| `r5_pressure` | `float` | `0.60` | `[0, 1]` | R5：`Pressure ≥ 0.60`（黄金回补区） | R29 |
| `r5_max_gap` | `int` | `12` | `[1, 30]` | R5：当前遗漏 `≤ 12` 期（排除死冷号） | R29 |
| `r5_follow_all_rank` | `float` | `0.70` | `[0.5, 1.0]` | R5：`R(Follow_ALL) ≥ 0.70`（前 30%） | R29 |

**跨字段约束**：`r2_f10_low < r2_f10_high`；`r4_w_co25 + r4_w_co50 == 1.0 ± 1e-9`；阈值全部 ∈ 开区间 `(0,1)` 时严格，`1.0` 仅允许 `r1_rank_threshold`/`r3/r5` 的极端调试场景。

---

## §6 `AdaptiveConfig`（`cfg.adaptive`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lookback_m` | `int` | `8` | `[3, 30]` | 裁决 A9；`≥ min_periods` | R33 |
| `min_periods` | `int` | `3` | `[1, 10]` | 样本不足 → 均匀先验 `0.2/规则` | R33 |
| `laplace_alpha` | `float` | `1.0` | `(0, 10]` | 命中率平滑 `(hit+α)/(n+2α)` | R33 |
| `temperature` | `float` | `1.0` | `[0.1, 10]` | Softmax 温度；越小越尖锐 | R33 |
| `weight_floor` | `float` | `0.05` | `[0, 0.2]` | 单规则权重下限；`5×floor ≤ 1` | R33 |
| `schema_version` | `int` | `1` | `{1}` | 状态文件版本；不匹配走迁移/重建 | R33、R48-e |
| `state_path` | `str` | `"output/state/adaptive_weights.json"` | 非空 | 父目录自动创建；原子写 | R33 |
| `in_memory` | `bool` | `False` | `{True, False}` | 回测强制 `True`（不落盘） | R33、R42 |
| `persist` | `bool` | `True` | `{True, False}` | `False` 时即使 `in_memory=False` 也不写盘 | R33 |

---

## §7 `PoolConfig`（`cfg.pool`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gold_min` | `int` | `3` | `[1, 5]` | 8.md L155 | R35 |
| `gold_max` | `int` | `5` | `[3, 8]` | `gold_min ≤ gold_max` 且 `gold_max < main_min` | R35 |
| `main_default` | `int` | `12` | `[10, 15]` | 裁决 A6；`main_min ≤ 默认 ≤ main_max` | R35 |
| `main_min` | `int` | `10` | `[5, 15]` | 8.md L155 | R35 |
| `main_max` | `int` | `15` | `[10, 20]` | `> main_min` | R35 |
| `max_zone4_ratio` | `float` | `0.45` | `(0, 1]` | 单四区上限（8.md L152） | R34 |
| `max_parity_ratio` | `float` | `0.70` | `[0.5, 1]` | 奇/偶单边上限（8.md L153） | R34 |
| `max_same_tail` | `int` | `3` | `[1, 6]` | 同尾上限（8.md L154） | R34 |
| `repeat_ratio` | `float` | `0.30` | `[0.25, 0.35]` | 8.md L202；`round(ratio×size)` 后 clip | R36 |
| `repeat_ratio_min` | `float` | `0.25` | `[0, 0.5]` | `≤ repeat_ratio` | R36 |
| `repeat_ratio_max` | `float` | `0.35` | `[0.1, 0.6]` | `≥ repeat_ratio` | R36 |
| `relax_order` | `tuple[str, ...]` | `("same_tail","parity","zone4","repeat_ratio")` | 四键的排列 | 顺序即放宽优先级（B7）；缺键/重复键 → `ConfigError` | R34 |

**跨字段约束**：`gold_max < main_min`（金胆 ⊂ 复式池，避免重叠）；`ceil(max_zone4_ratio × main_default) ≥ 1`；`repeat_ratio_min ≤ repeat_ratio ≤ repeat_ratio_max`。

---

## §8 `CoveringConfig`（`cfg.covering`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `play` | `str` | `"pick5"` | `{"pick5","pick7"}` | 与 `k` 必须一致（5↔pick5、7↔pick7） | R41 |
| `k` | `int` | `5` | `{5, 7}` | `k ≤ 池大小 v` | R37、R41 |
| `t` | `int` | `4` | `[2, k]` | 保底命中数；`t ≤ m` | R37 |
| `m_mode` | `str` | `"auto"` | `{"auto","fixed"}` | `fixed` 时 `m_fixed` 必填 | R37 |
| `m_fixed` | `int \| None` | `None` | `[t, v]` | 仅 `m_mode="fixed"` 使用 | R37 |
| `m_margin` | `int` | `1` | `[1, 3]` | 裁决 A7（v1.1.0 修正）：`t<k` 时 `m = min(v, t + m_margin)`；`t≥k` 时 `m = min(v, t)`（margin 不生效）；与 `ceil(0.25v)+2` 偏差 >2 时写 warning | R37 |
| `budget` | `int` | `20` | `[1, max_budget]` | 裁决 A8 | R41 |
| `budget_by_play` | `Mapping[str,int]` | `{"pick5":20,"pick7":10}` | 值 `[1, max_budget]` | CLI `--budget` 只覆盖当前 `play` | R41、R48-b |
| `max_budget` | `int` | `200` | `[20, 1000]` | 硬上限（400 元级） | R41 |
| `max_pool_v` | `int` | `20` | `[8, 30]` | 超限 → `InfeasibleConstraintError` | R37、R40 |
| `max_m_subsets` | `int` | `200000` | `[1000, 2000000]` | `comb(v,m)` 超限 → 拒绝穷举、降级抽样 | R37、R39 |
| `ilp_candidate_cap` | `int` | `120` | `[20, 500]` | 候选注数超限 → 降级贪心 | R39 |
| `ilp_target_cap` | `int` | `400` | `[50, 2000]` | 目标注数超限 → 降级贪心 | R39 |
| `ilp_time_limit_s` | `float` | `5.0` | `(0, 60]` | ILP 超时 → 降级并记 `degrade_reason` | R39 |
| `sa_iterations` | `int` | `2000` | `[100, 100000]` | 模拟退火迭代数 | R40 |
| `sa_t0` | `float` | `1.0` | `(0, 10]` | 初温 | R40 |
| `sa_cooling` | `float` | `0.995` | `[0.90, 0.9999]` | 冷却率 | R40 |
| `max_pair_overlap` | `int` | `3` | `[1, k-1]` | 注间最大重合号码数（正交防撞车） | R41 |
| `max_consecutive_in_ticket` | `int` | `2` | `[1, 4]` | 单注最大连号长度，超过剔除 | R41 |
| `forbid_all_odd_even` | `bool` | `True` | `{True, False}` | 剔除全奇/全偶注 | R41 |
| `sum_margin` | `float` | `0.45` | `[0.2, 0.8]` | 单注和值安全带宽（相对 `池均值×k`） | R41 |
| `table_path` | `str` | `"kl8/layer5_combinatorial/covering_tables.json"` | 非空 | 不存在 → `CoveringError`（可 `--no-table` 跳过，见 §12） | R38 |

**跨字段约束（V5–V9）**

| 编号 | 约束 | 失败示例 |
| :--- | :--- | :--- |
| V5 | `t ≤ k` 且 `t ≤ m` 且 `m ≤ v`（`v` 运行期传入） | `配置项 covering：t=6 > k=5 非法；保底命中数不得超过单注号码数` |
| V6 | `budget ≤ max_budget` | `配置项 covering.budget = 500 超过 max_budget = 200` |
| V7 | `play` 与 `k` 匹配 | `配置项 covering：play=pick7 要求 k=7，当前 k=5` |
| V8 | `max_pair_overlap < k` | `配置项 covering.max_pair_overlap = 5 非法：必须 < k = 5` |
| V9 | `m_fixed` 为空时 `m_mode != "fixed"` | `配置项 covering.m_mode=fixed 但 m_fixed 为空` |

---

## §9 `MLConfig`（`cfg.ml`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `True` | `{True, False}` | False → `ml_prob` 全 0.25、`ml_available=False` | R23 |
| `weight` | `float` | `0.0` | `[0, 0.30]` | 默认 0（只诊断）；`>0` 计入 Σω | R23、R31 |
| `train_window` | `int` | `160` | `[60, 600]` | 训练回溯期数 | R23 |
| `refit_every` | `int` | `25` | `[1, 200]` | 重训间隔（缓存 key 含窗口起点） | R23、R48 |
| `n_estimators` | `int` | `40` | `[10, 500]` | LightGBM 弱学习器数 | R23 |
| `learning_rate` | `float` | `0.03` | `(0, 0.5]` | 学习率 | R23 |
| `num_leaves` | `int` | `12` | `[2, 128]` | 叶子数（限复杂度） | R23 |
| `min_child_samples` | `int` | `25` | `[5, 200]` | 叶最小样本 | R23 |
| `subsample` | `float` | `0.85` | `(0, 1]` | 行采样 | R23 |
| `colsample_bytree` | `float` | `0.85` | `(0, 1]` | 列采样 | R23 |
| `seed` | `int` | `42` | `[0, 2^31-1]` | 模型随机种子（确定性） | R23、C7 |
| `calibrate` | `str` | `"pav"` | `{"pav","none"}` | PAV 保序回归校准 | R23 |
| `fallback` | `str` | `"ema"` | `{"ema","uniform"}` | LightGBM 缺失时降级策略 | R23 |
| `cache_size` | `int` | `8` | `[1, 64]` | `MLCache` 容量（LRU） | R23 |

**跨字段约束**：`num_leaves ≤ 2^min(10, n_estimators)`（防过参数化，仅 warning 级）；`weight > 0` 时 `enabled` 必须 True。

---

## §10 `BacktestConfig`（`cfg.backtest`）

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `warmup` | `int` | `300` | `[60, 1000]` | 8.md L179 例：前 300 期冷启动 | R42 |
| `default_n` | `int` | `100` | `[1, 2000]` | 默认回测期数；CLI `--backtest` 覆盖 | R42、R48-b |
| `ks` | `tuple[int, ...]` | `(3, 5, 7, 10, 12)` | 子集 `[1,20]`、升序、无重复 | 8.md L186 的 K 集合 | R43 |
| `t0_min` | `int` | `60` | `[20, 300]` | `T - warmup < t0_min` → `InsufficientDataError` | R42 |
| `include_tickets` | `bool` | `True` | `{True, False}` | 回测是否生成注单（关掉可提速） | R42、R46 |
| `alpha_sig` | `float` | `0.05` | `(0, 0.2]` | 单侧 t 检验显著性水平 | R43 |
| `lift_signal` | `float` | `1.05` | `[1.0, 1.2]` | **v1.1.0 新增**：8.md L188「Lift@K > 1.05」判据阈值；`LiftStat.is_signal = lift > 本值 且 p_value < alpha_sig`；实现内**禁止写死 1.05** | R43 |

---

## §11 `AppConfig` 顶层字段

| 字段 | 类型 | 默认值 | 合法范围 | 校验规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `history_path` | `str` | `"kl8_history.txt"` | 非空 | 不存在 → `DataFormatError`（同步模式下先同步） | R4、R48 |
| `output_dir` | `str` | `"output"` | 非空 | 自动创建 `reports/wechat/excel/state` 子目录 | R48-d |
| `seed` | `int` | `20261102` | `[0, 2^31-1]` | 全局随机源唯一入口（Louvain/SA/抽样） | C7、R21、R40 |
| `eps` | `float` | `1e-9` | `(0, 1e-3]` | 除零保护常量 | C8 |
| `cost_per_ticket` | `int` | `2` | `{2}` | 单注 2 元（官方固定）；改动需 Lead 批准 | R46、A10 |
| `prize_table` | `Mapping[str, Mapping[int,int] \| None]` | `{"pick5":{5:1000,4:21,3:3},"pick3":None,"pick7":None}` | 只允许已确认奖级 | 裁决 A10；`None` → 只统计频次不出金额 | R44、R46 |

**顶层校验**：`seed` 必须同时注入全局 `random`/`np.random`（不得各模块自造）；`prize_table` 的键必须 ⊆ `{"pick3","pick5","pick7"}`。

---

## §12 CLI 覆盖规则（`main.py` → `merge_cli_overrides` → `validate_config`）

| CLI 参数 | 类型/取值 | 映射配置路径 | 覆盖规则 | 关联需求 |
| :--- | :--- | :--- | :--- | :--- |
| `--mode` | `{daily,backtest,audit}` | 不落配置 | 分派 `pipeline.run_daily/run_backtest/run_audit`；默认 `daily` | R48-b |
| `--history PATH` | `str` | `history_path` | 路径不存在 → 报错退出码 1（不静默回退） | R4、R48 |
| `--no-sync` | flag | 不落配置 | `run_daily(no_sync=True)`，跳过网络 | R48-c |
| `--quick` / `--full-sync` | flag（互斥） | 不落配置 | `sync_history(quick=…)`；同时给出 → `--quick` 优先并 warning | R48-c |
| `--period N` | `int` | 不落配置 | `target_period=N`，必须 `> 最新期`（否则 `DataFormatError`） | R48-b |
| `--date S` | `str` | 不落配置 | `target_date`，必须 `YYYY-MM-DD` 正则合法 | R48-b |
| `--backtest N` | `int` | `backtest.default_n` | `0` = 跳过回测与审计（报告 [附录B] 输出"未执行"）；`>0` 覆盖期数 | R42、R48-b |
| `--warmup N` | `int` | `backtest.warmup` | 覆盖冷启动期数；`N < t0_min` → `ConfigError` | R42 |
| `--pool-size N` | `int` | `pool.main_default` | 必须 ∈ `[pool.main_min, pool.main_max]` = `[10,15]` | R35 |
| `--k` | `{5,7}` | `covering.k` + `covering.play` | `5→pick5`、`7→pick7`；同时写 `play`（V7 校验） | R41 |
| `--budget N` | `int` | `covering.budget` + `budget_by_play[play]` | 只覆盖当前 `play` 的预算；`> max_budget` → `ConfigError` | R41、A8 |
| `--seed N` | `int` | `seed` | 覆盖全局随机源（影响 Louvain/SA/抽样；`ml.seed` 保持独立） | C7、R40 |
| `--full-print` | flag | 不落配置 | 控制台打印完整报告（否则只打核心精要） | R48-b |
| `--no-table` | flag | `covering.table_path=""`（视同未配置） | 强制跳过查表，走 ILP/贪心（调试与核验用） | R38 |
| `--config PATH` | `str` | `load_config(path=…)` | 与默认值深合并；CLI 仍可二次覆盖 | R48-a |

**覆盖优先级示例**：`--k 7 --budget 15` ⇒ `covering.k=7, covering.play="pick7", covering.budget=15, budget_by_play["pick7"]=15`，`budget_by_play["pick5"]` 保持 20 不变。
**禁止**：任何参数不得绕过 `validate_config`（先合并、再校验、后执行，单一路径）。

---

## §13 配置文件 `config.json`（可选）示例

```json
{
  "windows": { "short": 10, "mid": 25, "long": 50, "min_history": 40 },
  "markov": { "alpha": 1.0, "order": 2, "diag_split": true },
  "weights": {
    "follow_10": 0.10, "follow_25": 0.15, "follow_50": 0.15, "follow_all": 0.15,
    "co": 0.15, "trend": 0.20, "omission": 0.10, "ml": 0.0,
    "lambda_vote": 0.08,
    "trend_w_f10": 0.5, "trend_w_mom_10_25": 0.3, "trend_w_mom_25_50": 0.2
  },
  "rules": { "r1_min_windows": 3, "r1_rank_threshold": 0.65, "r2_base_freq": 0.24 },
  "adaptive": { "lookback_m": 8, "min_periods": 3 },
  "pool": { "main_default": 12, "repeat_ratio": 0.30 },
  "covering": { "play": "pick5", "k": 5, "t": 4, "budget": 20 },
  "ml": { "enabled": true, "weight": 0.0 },
  "backtest": { "warmup": 300, "default_n": 100 }
}
```

**深合并语义**：只覆盖出现的键；未出现的键沿用默认值；未知键 → `ConfigError`（`配置项 weights.foo 不存在；可用键：…`），禁止静默忽略（防拼写错误导致"以为改了其实没改"）。

---

## §14 校验错误消息规范（统一模板，便于审大关抽查）

```
配置项 {路径} 非法：{现值或现状}，要求 {约束}；建议 {修复动作}
```

| 场景 | 消息 |
| :--- | :--- |
| 权重和不为 1 | `配置项 weights：融合权重之和 = 1.080000，要求 1.0（±1e-9）；建议按比例缩减 follow_50` |
| α 越界 | `配置项 markov.alpha = 2.5 非法：要求 ∈[0.5, 2.0]；建议 1.0` |
| λ 越界 | `配置项 weights.lambda_vote = 0.20 非法：要求 ∈[0.05, 0.10]；建议 0.08` |
| 金胆不小于复式池 | `配置项 pool：gold_max = 15 必须 < main_min = 10；建议 gold_max = 5` |
| k 超池 | `配置项 covering：k = 7 > 池大小 v = 5 非法；请增大复式池或改用 --k 5` |
| 预算超上限 | `配置项 covering.budget = 500 超过 max_budget = 200；建议 ≤ 200` |
| 未知键 | `配置项 weights.foo 不存在；可用键：follow_10, follow_25, follow_50, follow_all, co, trend, omission, ml, lambda_vote, trend_w_f10, trend_w_mom_10_25, trend_w_mom_25_50` |
| 未知枚举 | `配置项 covering.play = "pick9" 非法：要求 {"pick5","pick7"}` |

---

## §15 配置项 → 交付件影响矩阵（改配置前必读）

| 配置 | 影响交付件 | 影响测试 | 变更风险 |
| :--- | :--- | :--- | :--- |
| `weights.*` | FinalScore、Top20 排名、金胆/复式池、全部报告与回测 | `test_scoring.py`、`test_pipeline.py` | 中（排名变化，回测结论变化） |
| `rules.*` | 五规则榜、票数 V、FinalScore 加成 | `test_rules.py`（边界双点） | 中（阈值边界必须成对改测试） |
| `pool.*` | 金胆/复式池规模、形态体检 | `test_filter.py` | 高（池规模影响覆盖注数与预算） |
| `covering.budget/k/t/m_*` | 注单数、成本、保底等级、ROI | `test_covering.py`、`test_tickets.py` | 高（保底与预算直接互斥） |
| `ml.weight` > 0 | FinalScore（新增通道）、Σω 重算 | `test_ml.py`、`test_scoring.py` | 高（需 Lead 批准） |
| `backtest.warmup/default_n` | 回测样本、Lift 显著性、资金曲线 | `test_walkforward.py` | 低 |
| `seed` | Louvain 社区、SA 注单、ML | `test_adaptive.py`、`test_covering.py` | 低（仅重跑） |

---

## §16 机检清单（P7/P8 可直接照此核验）

1. **零魔法数字**：`grep -rnE "(0\.(65|24|75|45|70|60|08|25|35)|\[\[?10|25|50)" --include=*.py kl8/` 命中处必须来自 `config.py`/`constants.py` 的字段读取，而非字面量直用。
2. **schema 完整性**：`python -c "from kl8.config import default_config, to_dict; print(len(to_dict(default_config())))"` 得到的键集合必须覆盖 §2–§11 全部字段（v1.1.0 实数 **102** 个可配置叶子字段：WindowConfig 5 + MarkovConfig 3 + ScoreWeights 12 + RuleConfig 12 + AdaptiveConfig 9 + PoolConfig 12 + CoveringConfig 22 + MLConfig 14 + BacktestConfig 7（含 `lift_signal`）+ AppConfig 6；`budget_by_play`/`prize_table` 两个映射键已含在上述 22/6 内，不重复计数。v1.0.0 曾误写 104，已修正）。
3. **校验可触发**：对每个 `V*` 约束构造非法配置，必须抛 `ConfigError` 且消息符合 §14 模板（`tests/test_config.py` 逐条覆盖）。
4. **CLI 覆盖可审计**：`python main.py --no-sync --k 7 --budget 5 --pool-size 12 --seed 7 --backtest 5 --full-print` 输出的配置摘要（不止阈值）必须与本文件 §12 一致。
5. **默认值可复现**：`config_hash(default_config())` 在所有机器上一致（不含时间戳/路径）。