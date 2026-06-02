# C组问题诊断与优化记录

## 1. 文档目的

本记录用于单独保存 `C_pressure_passage` 当前阶段的问题分析、实验趋势和优化结论，避免后续重复回溯。

本记录聚焦：

1. `C` 组当前失败的主因是什么；
2. 已尝试哪些主干内优化方向；
3. 这些方向各自呈现出什么趋势；
4. 当前最合理的工程决策是什么；
5. 这一决策对论文主线和创新点意味着什么。

## 2. 当前分析所基于的数据

核心正式结果：

1. `outputs/phase1_validation_v2/C_pressure_passage/summary_metrics.csv`
2. `outputs/phase1_validation_v2/C_pressure_passage/trial_metrics.csv`

本轮新增对照：

1. `outputs/c_group_rho_floor_v1/`
2. `outputs/topo_rho_floor_scan_medium_v1/`
3. `outputs/topo_rho_floor_scan_medium_fine_v1/`
4. `outputs/topo_soft_floor_medium_check_v1/`
5. `outputs/topo_soft_floor_nonlinear_medium_check_v1/`
6. `outputs/topo_hysteresis_floor_medium_check_v1/`
7. `outputs/eta_sigma_scan_medium_v1/`
8. `outputs/c_group_final_param_sweep_medium_v1/`
9. `outputs/c_group_candidate_full_compare_v1/`

## 3. 当前问题定位

### 3.1 不是障碍碰撞主导失败

`C` 组当前最重要的事实是：

1. `easy/medium` 中大多数失败不是 `obs_collision_rate` 主导；
2. 核心失败模式是 `agent_collision_rate = 1.0`；
3. 说明当前主要瓶颈不是“识别不到通道”，而是“进入通道后群体内部结构先崩”。

### 3.2 问题主链条已经比较清楚

当前最合理的机制解释为：

1. `Omega` 确实在起作用；
2. 但 `eta(Omega)` 会把 `rho` 压得过低；
3. `rho` 当前主要只调制 `F_topo`；
4. 当 `F_topo` 被整体压弱后，群体结构支撑不足；
5. `F_safe` 无法单独接管通道内的秩序维持；
6. 因而障碍还没撞上，智能体之间先碰撞。

换句话说：

> 当前 `C` 组失败的主因不是 `Omega` 无效，而是高压力区下 `F_topo` 被削弱过头，导致内部安全与结构重构失稳。

### 3.3 这不是主干整体失效，而是主干在通道场景中的结构性弱点

当前不能据此得出“整个 SMGF 架构有问题”的结论，原因是：

1. `A` 组已经证明基础导航、绕障和安全必要性成立；
2. `B` 组已经证明围捕与角度覆盖边界可解释；
3. `F` 组已经证明安全项必要但有代价；
4. 说明主干框架不是混乱或不可解释，而是 `C` 组暴露出调制层在高压力通道场景中的表达能力不足。

更准确的说法是：

> 当前第一阶段主干的弱点在于：标量压力调制会把拓扑协同整体压弱，但通道通过真正需要的是“保留必要结构支撑，同时软化不必要挤压”。

## 4. 已尝试的优化方向与趋势

### 4.1 路线一：固定拓扑下界 `fixed floor`

做法：

1. 在 `rho * F_topo` 上加入固定下界；
2. 先只在 `S4` 场景启用；
3. 扫描 `topo_rho_floor`。

关键结果：

1. `topo_rho_floor = 0.65` 时，`s4_narrow_passage_medium` 的 `M7 success_rate` 从 `0` 拉到 `0.0333`；
2. `M6 success_rate` 在同一扫描中可到 `0.10`；
3. `min_agent_distance_mean` 明显抬升；
4. `completion_time_mean` 明显下降；
5. 但 `topo_rho_floor` 过大时，`min_obs_distance_mean` 会下降，说明过强刚性会侵蚀障碍余量。

结论：

> `fixed floor` 是当前最有效的主干内工程修正，说明系统确实需要一个持续的结构支撑下界。

### 4.2 路线二：线性 soft floor

做法：

1. 将常数下界改为 `omega` 相关的连续 soft floor；
2. 低压区弱激活，高压区接近上限。

关键结果：

1. `medium` 中 `M6/M7 success_rate` 重新掉回 `0`；
2. `obs` 余量略改善；
3. 但 `agent` 间距没有接近 `fixed floor` 水平。

结论：

> 线性 soft floor 太弱，本质上是“补得不持续”。

### 4.3 路线三：更强非线性 soft floor

做法：

1. 用阈值后非线性激活；
2. 扫了一小圈 `onset/gamma`。

关键结果：

1. 所有测试组合仍然 `success_rate = 0`；
2. 行为比线性 soft floor 更激进，但仍不够替代 `fixed floor`。

结论：

> 不是单纯 soft floor 的曲线没调好，而是这种“仅在高压时刻局部托住”的思路本身支撑持续性不足。

### 4.4 路线四：滞回 + 慢释放的持续型 floor

做法：

1. 用 `on/off threshold` 控制进入与退出；
2. 用 `release_tau` 控制慢释放；
3. 只在 `S4` 场景启用。

关键结果：

1. 比线性 soft floor 更接近 `fixed floor`；
2. 单次动态检查中 `floor` 的激活并非太短；
3. 但 `medium` 里 `M6/M7 success_rate` 仍为 `0`；
4. `agent` 间距仍明显低于 `fixed floor`。

结论：

> “进入-保持-退出”方向是对的，但当前 `C` 组所需的不是局部记忆，而是更高、更稳定的持续结构支撑。

### 4.5 路线五：只重塑 `eta(Omega)`

做法：

1. 关闭 `topo floor`；
2. 扫描 `eta_min` 与 `sigma_omega`；
3. 只看 `s4_narrow_passage_medium` 下的 `M7`。

关键结果：

1. 全部组合 `success_rate = 0`；
2. `agent_collision_rate` 始终为 `1.0`；
3. 虽然完成时间和障碍余量会变化，但无法逼近 `fixed floor` 的结构支撑效果。

结论：

> 只靠更缓的压力衰减，不能替代 `fixed floor`。

### 4.6 路线六：在 `fixed floor` 基线上继续微调 `k_s / k_t / r0`

做法：

1. 固定 `topo_rho_floor = 0.65`；
2. 只看 `s4_narrow_passage_medium`；
3. 扫描 `k_s = {2.8, 3.4, 4.0}`；
4. 扫描 `k_t = {1.0, 1.2, 1.4}`；
5. 扫描 `r0 = {1.4, 1.6, 1.8}`。

关键结果：

1. 原始 `fixed floor` 基线：
   `M7 success_rate = 0.0333`；
2. 最优参数区集中在：
   `k_t = 1.4, r0 = 1.8`；
3. `k_s` 在 `2.8/3.4/4.0` 之间影响较小；
4. 候选最优点例如：
   `k_s = 3.4, k_t = 1.4, r0 = 1.8`；
5. 该点可把 `M7 success_rate` 提升到 `0.20`；
6. `min_agent_distance_mean` 从约 `0.306` 提升到约 `0.334`；
7. `obs_collision_rate` 仍为 `0.0`，但 `min_obs_distance_mean` 从约 `0.904` 降到约 `0.806`。

结论：

> 当前 `C2 medium` 的最有效提升来自更强、更大尺度的结构支撑，其中 `k_t` 与 `r0` 比 `k_s` 更关键。

### 4.7 路线七：把候选参数带回 `easy / medium / hard` 全组对照

做法：

1. 固定候选参数：
   `fixed floor = 0.65, k_s = 3.4, k_t = 1.4, r0 = 1.8`；
2. 对 `s4_narrow_passage_easy / medium / hard` 全部运行；
3. 比较 `M4/M5/M6/M7` 四种方法。

关键结果：

1. `easy`：
   - `M4/M5 success_rate = 1.0`
   - `M6 success_rate ≈ 0.767`
   - `M7 success_rate ≈ 0.733`
2. `medium`：
   - `M4 success_rate = 0.70`
   - `M5 success_rate = 0.30`
   - `M6 success_rate = 0.60`
   - `M7 success_rate = 0.20`
3. `hard`：
   - `M4 success_rate ≈ 0.067`
   - `M5/M6/M7 success_rate = 0`
4. 三个场景下 `obs_collision_rate` 仍全部为 `0.0`；
5. 说明本轮增强主要改善的是内部结构维持，而不是障碍通过几何本身。

结论：

> 当前候选参数已经能显著改善 `C1 easy` 与 `C2 medium`，但 `C3 hard` 仍未被打通；因此它适合作为第一阶段 `C` 组的工程性候选版本，但不能被包装成对高压力通道问题的完全解决。

## 5. 总体趋势总结

把本轮所有尝试合在一起，趋势已经比较清楚：

1. 问题确实在于高压力下 `F_topo` 被压得过低；
2. 仅仅平滑或缓和 `rho` 的变化不够；
3. 仅仅在高压瞬时局部补偿也不够；
4. 当前最有效的修正是：给 `F_topo` 一个明确且持续的结构保底。
5. 在此基础上，进一步提高 `k_t` 并增大 `r0` 能带来比继续磨 `soft floor` 或 `eta` 更直接的收益。

因此当前最可信的工程性结论是：

> `C` 组通道问题需要一个明确的持续结构支撑下界，而不只是更柔和的瞬时压力调制函数。

## 6. 当前最合理的工程决策

在第一阶段主干论文闭环的语境下，当前最合理的决策是：

1. 暂时保留 `fixed floor = 0.65` 作为 `C` 组工作基线；
2. 当前 `C` 组最强候选参数可定为：
   `fixed floor = 0.65, k_s = 3.4, k_t = 1.4, r0 = 1.8`；
3. 不再继续把主要精力放在 `soft floor` 花样或纯 `eta` 曲线重塑上；
4. 当前没有必要继续无限下探“更复杂 floor 机制”，因为已经足够确认大趋势；
5. 如需继续优化，应优先围绕 `fixed floor + k_t/r0` 小范围修正，而不是重开新机制分支。

## 7. 这样做会不会伤到创新点

### 7.1 不会直接伤到创新点，但会改变表述方式

如果直接在主干里加入固定结构保底，论文表述不能再写成：

> `Omega` 调制本身已经完整解决了窄通道问题。

更稳妥的写法应该是：

> 当前结果表明，纯标量压力调制在高压通道中会过度削弱拓扑协同，因此需要为拓扑结构保留一个最低支撑强度，才能兼顾通过性与群体内部稳定性。

也就是说：

1. 创新点不是没了；
2. 而是从“纯粹的连续标量调制已经足够”调整为“状态调制主干成立，但在高压通道中需要结构保底修正”。

### 7.2 当前不宜过度声称 `C` 组已经被理论性解决

如果后续直接使用 `fixed floor`，建议把它定位成：

1. 第一阶段主干中的工程性稳定化修正；
2. 用于让 `C` 组获得可写的正结果；
3. 而不是把它包装成已经完备的最终理论形态。

这样更真实，也更不容易被质疑“过度包装”。

## 8. 当前阶段建议

当前阶段更合理的推进方式是：

1. 先接受 `fixed floor` 作为 `C` 组当前最有效基线；
2. 当前可以先采用候选版本：
   `fixed floor = 0.65, k_s = 3.4, k_t = 1.4, r0 = 1.8`；
3. 先把第一阶段论文主线稳住；
4. 更深层的“如何彻底优雅解决通道问题”可以作为第二阶段增强方向，而不是现在继续无限拉长第一阶段闭环。

## 9. 一句话结论

> 当前 `C` 组的问题定位已经足够清楚：纯标量压力调制会在高压通道中过度削弱拓扑支撑；在所有已试主干内修正中，`fixed floor + (k_t = 1.4, r0 = 1.8)` 是当前最有效的工程候选，其中 `easy/medium` 已得到明显改善，而 `hard` 仍是未打通边界。因此，第一阶段更合理的策略是接受该候选版本并诚实陈述其适用范围，而不是继续无限深入追求更复杂机制。
