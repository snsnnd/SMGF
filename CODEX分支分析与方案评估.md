# Codex 分支分析与方案评估

## 1. 分支与验证状态

已切换到远程分支：`codex/optimize-smgf-experiment-and-generate-report`。

本地验证结果：

1. 已成功 `fetch` 并切换到该分支。
2. 已运行测试：`uv run python -m unittest discover -s tests`
3. 测试结果：`3 tests, OK`

说明该分支新增的指标统计与实验脚本至少通过了当前的基础烟雾测试，但这不等价于实验结论已成立。

## 2. 该分支新增内容概览

相对 `main`，该分支主要新增了四类内容：

1. 指标分解增强
2. `S4` 窄通道调参脚本与结果
3. `S6` 预测时间扫描脚本与结果
4. `S5` easy / medium / hard 三档场景，以及少量测试代码

关键文件：

1. `src/smgf/metrics.py`
2. `src/smgf/experiments.py`
3. `src/smgf/scenarios.py`
4. `src/smgf/tune_s4.py`
5. `src/smgf/tune_s6_prediction.py`
6. `tests/test_metrics.py`
7. `tests/test_experiments.py`

## 3. 对新代码和数据的分析

### 3.1 指标系统：方向是对的，但还不够细

`src/smgf/metrics.py:82-111` 新增了以下字段：

1. `inside_any`
2. `inside_final`
3. `gmax_success`
4. `radius_success`
5. `sigma_success`
6. `no_collision`
7. `dwell_success`
8. `radius_error_final`
9. `success_geom_final`
10. `success_no_collision`
11. `gmax_reach_time`

这是正确方向，因为它把“失败”从单一 `success/collision` 拆成了几何达标、碰撞与时间过程三个层次。

但这套拆分仍然不够支持你贴出来的诊断目标，主要缺口有：

1. 还没有 `obs_collision` 和 `agent_collision`，因此不能判断是障碍碰撞还是智能体间碰撞。
2. 还没有 `time_to_inside`、`time_to_radius`、`time_to_full_geom`，因此不能判断几何约束到底卡在哪一步。
3. 还没有 `success_hold_time`、`post_success_violation_count`、`last_10s_inside_rate` 之类的保持性指标，因此不能判断成功是否稳定。
4. `success_no_collision` 目前只是 `no_collision` 的别名，信息增量不够，见 `src/smgf/metrics.py:143-145`。

额外问题：

`gmax_reach_time` 用的是固定阈值 `120 deg`，而不是 `params.gmax_threshold_deg`，见 `src/smgf/metrics.py:145-151`。这会导致评价阈值与场景配置不一致。

结论：

这一步属于“方向正确，但实现未完成”。如果要支持 S4/S2/S3 的细粒度失败归因，还需要继续补字段。

### 3.2 S4 调参：已经开始系统扫参，但结论实际上是否定性的

`src/smgf/tune_s4.py:77-83` 对以下参数做了网格扫描：

1. `eta_min`
2. `sigma_omega`
3. `k_t`
4. `r0`
5. `lambda_curl`

这是合理的起点，但仍少于你贴出方案中建议优先扫描的集合，因为当前没有扫：

1. `d_agent_safe`
2. `k_s`

而这两个量恰好与通道内部碰撞高度相关。

更关键的是，现有结果并没有证明 `Omega` 机制在 S4 成立。

证据一：
`outputs/tune_s4/tune_s4_summary.csv` 中当前展示的组合全部是：

1. `success_rate = 0.0`
2. `collision_rate = 1.0`

证据二：
`outputs/tune_s4/best_s4_params.json` 中所谓“best”参数，前几项仍然全部是：

1. `success_rate = 0.0`
2. `collision_rate = 1.0`
3. `radius_success_rate = 0.0`
4. `no_collision_rate = 0.0`

证据三：
`outputs/targeted_optimization/s4_best_ablation_summary.csv` 里，`M4/M5/M6/M7` 全部：

1. `success_rate = 0.0`
2. `collision_rate = 1.0`

这说明当前 S4 的实验事实是：

> 无论是否保留 `Omega` 调制，所有方法都失败，而且全部碰撞。

因此，当前分支不能支持“`Omega` 有效”这个主张。

进一步看：

1. `inside_any_rate = 1.0`
2. `inside_final_rate = 1.0`
3. `gmax_success_rate = 1.0`
4. `sigma_success_rate = 1.0`
5. `radius_success_rate = 0.0`
6. `no_collision_rate = 0.0`

说明现在的问题并不是“完全没形成几何结构”，而是：

1. 半径约束没达标
2. 安全性彻底失败

这正好支持你贴出方案中“失败要拆得更细”的判断。

结论：

你贴出的 S4 观点是对的，尤其是这句最重要：

> 如果 M7 仍不如 M4，那么 Omega 机制还不能写成正贡献。

基于当前数据，这句话成立，而且必须保守表述。

### 3.3 S2：场景被强化了，但 Psi 贡献仍未被证明

`src/smgf/scenarios.py:41-58` 把 `S2` 改成了更强的同侧紧密起始构型，这是合理的，因为它更能放大 `Psi` 调制的作用。

但从 `outputs/debug_after_metric_fix/summary_metrics.csv:4-9` 看：

1. `M4/M5/M6/M7` 全部 `success_rate = 0.0`
2. 全部 `collision_rate = 1.0`
3. `gmax_mean` 仍然非常大，约 `274-275 deg`

这意味着：

1. 当前 S2 不是“差异尚不明显”
2. 而是“所有多智能体拓扑方法一起失败了”

因此当前分支还不能支持“`Psi` 调制改善同侧展开”的结论。

你贴出的建议“不要只看最终 success，要增加 `gmax_reach_time`、`convoy_ratio_time`、`time_to_inside`、`final_gmax`、`mean_gmax_over_last_10s`”是对的。当前分支只实现了 `gmax_reach_time`，远远不够。

### 3.4 S3：仍然是当前最像正结果的场景，但保持性指标缺失

从 `outputs/debug_after_metric_fix/summary_metrics.csv:10-15` 看：

1. `M6/M7 success_rate = 1.0`
2. `M4/M5 success_rate = 0.666...`
3. `M6/M7 no_collision_rate = 1.0`

这说明当前 `S3` 依然是最接近论文主张的正结果场景。

但你贴出的提醒也是对的：

> 不能只看 completion_time，还要看完成后是否稳定保持。

当前代码里没有：

1. `success_hold_time`
2. `post_success_violation_count`
3. `last_10s_gmax_mean`
4. `last_10s_radius_error_mean`
5. `last_10s_inside_rate`

所以当前 S3 只能说明“达成过”，还不能充分说明“稳定保持”。

### 3.5 S5：已经分出 easy / medium / hard，但还不够分层

`src/smgf/scenarios.py:91-144` 新增了：

1. `s5_dense_tracking_easy`
2. `s5_dense_tracking_medium`
3. `s5_dense_tracking_hard`

这比原来直接只看 dense hard 更好，说明分支已经吸收了一部分“分层加难度”的思路。

但从 `outputs/debug_after_metric_fix/summary_metrics.csv:22-39` 看：

1. `easy/medium/hard` 三档里，所有方法 `success_rate = 0.0`
2. 三档里，所有方法 `collision_rate = 1.0`

因此当前分层还不够细。

你贴出的更细 ladder：

1. `S5-0 移动目标无障碍`
2. `S5-1 1 个障碍`
3. `S5-2 2-3 个障碍`
4. `S5-3 medium density`
5. `S5-4 hard density`

这个方案更正确，因为现在连 easy 都全失败，必须继续回退难度，直到找到最小可成功层级。

### 3.6 S6：做了预测扫描，但当前结果并不支持“存在明显最佳 T_p 区间”

`src/smgf/tune_s6_prediction.py:17-20` 已经扫描：

1. `t_pred = 0.0, 0.3, 0.7, 1.0, 1.3, 1.5` 中的主要子集结果
2. `beta_lead`
3. `r_c`

这与贴出方案的方向是一致的。

但从 `outputs/targeted_optimization/quick_tune_s6_summary.csv` 可见：

1. `M4/M5/M6/M7` 全部 `success_rate = 0.0`
2. 全部 `collision_rate = 1.0`
3. `M5` 与 `M4` 几乎完全一致
4. `M7` 与 `M6` 也几乎完全一致

这说明两件事：

1. 当前场景下，`Psi` 与 `Omega` 调制对 S6 没有形成可分辨贡献。
2. 预测时间虽然改变了误差数值，但还没有把系统带入成功工作区间。

所以你贴出的判断也成立：

> 关键不是证明预测时间越长越好，而是证明存在最优区间。

而当前分支距离这一步还差很远，因为现在连基本成功都没有形成。

## 4. 对粘贴方案的总体判断

### 4.1 我认为方案总体上是正确的

你的粘贴内容在研究路径上是正确的，尤其正确体现在以下几点：

1. 先固定评估标准，再补诊断指标，而不是频繁改成功定义。
2. 把 S4 作为最高优先级，因为 `Omega` 是否成立是论文关键。
3. 如果 S4 调不通，不要强行宣称主方法成立，而要考虑增强版或保守表述。
4. S2 不应只看最终成功率，而应看展开速度和角度覆盖过程。
5. S3 需要稳定保持指标，防止“短暂成功”误导。
6. S5 必须从更低难度台阶开始，而不是直接盯 dense hard。
7. S6 要做 `T_p` 区间扫描，而不是只测一个预测时间。
8. 最终必须扩大样本，至少 `30/50/100 seeds` 分层推进。

这是一个成熟、审稿友好的实验推进思路。

### 4.2 但有两点我会修正或补充

第一，`M9` 各向异性软化是合理增强方向，但不应该过早引入。

原因：

1. 当前 S4 连碰撞归因都没拆清楚。
2. 还没把 easy / medium / hard 通道体系搭出来。
3. 还没扫 `d_agent_safe` 与 `k_s` 这两个更直接的碰撞因素。

所以更合理顺序是：

1. 先补 `obs_collision` / `agent_collision`
2. 先把 S4 easy 跑通
3. 再判断是否需要 `M9`

也就是说，`M9` 是合理备选，但不是当前第一优先级。

第二，S4 排序标准应先按“是否无碰撞成功”过滤，而不是只在全失败结果里做相对排序。

当前 `best_s4_params.json` 的问题在于：

1. 全部候选都失败
2. 但文件名却叫 `best`

这会误导后续分析。更稳妥的做法是：

1. 若存在 `success_rate > 0` 的组合，再做多指标排序
2. 若全部 `success_rate = 0`，则明确标注为 `least_bad_candidates` 或 `failed_grid_summary`

## 5. 建议的下一步

按优先级，我建议这样推进：

1. 在 `TrialMetrics` 中补充 `obs_collision`、`agent_collision`、`time_to_inside`、`time_to_radius`、`time_to_full_geom`。
2. 将 `gmax_reach_time` 阈值改为使用 `params.gmax_threshold_deg`。
3. 把 `S4` 拆成 `easy / medium / hard`，先只要求 easy 跑通。
4. 在 `S4` 扫参中加入 `d_agent_safe` 与 `k_s`。
5. 若 easy 仍全部失败，再考虑 `M9` 各向异性软化。
6. 把 `S5` 继续向下拆成 `S5-0 / S5-1 / S5-2`。
7. 为 `S3` 增加保持性指标。
8. 在形成正结果后，再执行 `30 -> 50 -> 100 seeds` 的正式统计。

## 6. 最终结论

一句话总结：

> 这个 `codex` 分支在实验诊断和调参工具上前进了一步，但新增结果并没有证明 SMGF 的关键主张已经成立；你贴出的推进方案整体正确，而且比当前分支的实现更完整、更适合作为下一轮实验工作的主路线。
