# 结构化能量池最小模型

## 1. 目的

本文给出一个直接从 `M43` 往下抽象的最小模型，用来回答 4 个具体问题：

1. 哪些变量属于场景层；
2. 哪些变量属于拓扑/结构层；
3. 全局可释放能量池 `T` 如何更新；
4. `drive` 如何分配给各个 agent。

这个模型对应当前代码中的最小原型方法：

`M47 = Xi-Polarity-Axis-Queue-Occupancy-TankEnergy-lite`

它不是最终统一理论，只是：

> 在不推翻 `M43` 结构骨架的前提下，把“固定高能量”替换成“场景层门控 + 拓扑层储能/释能 + 局部预算分配”的第一版实现。

## 2. 分层思路

核心分层是：

1. 场景层定义外部约束和当前阶段；
2. 拓扑层定义内部结构是否已经具备储能与释放条件；
3. 能量池 `T` 记录当前系统可用于 through 的输运能量；
4. 各个 agent 根据自身局部角色与队列位置，从 `T` 中获得不同的 `drive` 预算。

一句话概括：

> 场景层决定“该不该放”，拓扑层决定“能不能放、放给谁”。

## 3. 场景层变量

当前最小模型使用下面这些场景层变量。

### 3.1 `corridor_confidence`

含义：当前是否真的处于需要 corridor 组织的阶段。

### 3.2 `corridor_commitment`

含义：群体是否已经进入并承诺执行 corridor through 结构。

### 3.3 `release_need`

含义：当前是否已经进入该开始释放和推进的阶段。

### 3.4 `forward_free_length`

含义：沿主轴前方可通行自由长度。

### 3.5 `obstacle_pressure`

含义：当前前方障碍造成的约束压力。

### 3.6 `teammate_pressure`

含义：当前前方队友造成的拥堵压力。

## 4. 拓扑/结构层变量

当前最小模型使用下面这些拓扑层变量。

### 4.1 `slot_order_readiness`

含义：队列顺序和纵向槽位是否基本成形。

### 4.2 `lateral_order_readiness`

含义：横向收拢和主轴对齐是否足够稳定。

### 4.3 `forward_room_mean`

含义：局部前向空隙是否允许继续传播 through。

### 4.4 `release_ready`

含义：当前结构是否已具备释放条件。

### 4.5 `crowding_mean / crowding_peak`

含义：局部压缩程度。

这里不把 crowding 简单理解成坏事，而是把它理解成：

1. 太低：队列还没收起来；
2. 适中：最适合储能与 through；
3. 太高：接近碰撞，应该转成耗散与限流。

### 4.6 `queue_shape`

定义：

```text
queue_shape = 1 - lateral_width / (longitudinal_span + 0.45 r0)
```

含义：群体当前有多像“纵向 through 队列”。

它直接用全局结构宽度和纵向跨度定义，是当前最小模型里一个很关键的拓扑量。

## 5. 全局能量池 `T`

### 5.1 定义

`T` 不是总物理能量，而是：

> 当前系统允许用于 corridor through 的可释放输运能量。

归一化后：

```text
T in [0, 1]
```

### 5.2 场景信号

场景层先合成一个外部门控信号：

```text
scene_signal
= 0.32 corridor_confidence
+ 0.24 corridor_commitment
+ 0.22 release_need
+ 0.12 forward_free_norm
+ 0.10 (1 - obstacle_pressure)
```

它回答：

`当前场景是否允许高 through 能量进入工作区。`

### 5.3 拓扑信号

拓扑层再合成一个内部结构信号：

```text
topology_signal
= 0.24 slot_order_readiness
+ 0.20 lateral_order_readiness
+ 0.16 forward_room_mean
+ 0.15 release_ready
+ 0.12 compact_band
+ 0.13 queue_shape
```

其中：

```text
compact_band = 1 - 2 |crowding_mean - 0.55|
```

它回答：

`当前结构是否真的具备储能和释放能力。`

### 5.4 `T` 的更新

最小版更新写成：

```text
dT/dt = P_store - P_release - P_loss
```

当前实现中：

```text
P_store
= k_charge * scene_signal * topology_signal * compact_band * (1 - T)

P_release
= k_release * release_need * release_ready * (0.35 + 0.65 forward_free_norm) * T

P_loss
= k_loss * (0.45 crowding_peak + 0.30 obstacle_block + 0.15 obstacle_pressure + 0.10 teammate_pressure) * T
```

直觉上：

1. 队列既有序又压缩到合适区间时，`T` 容易上升；
2. 结构进入 release 阶段时，`T` 通过推进输出被消耗；
3. 拥挤、局部风险和堵塞会把 `T` 转成耗散，而不是推进。

## 6. `drive` 如何分给各个 agent

当前不直接平均分，而是根据局部角色给每个 agent 一个预算。

### 6.1 局部释放得分

先定义每个 agent 的局部 through 资格：

```text
local_release_i
= 0.26 leader_priority_i
+ 0.24 forward_room_i
+ 0.20 slot_alignment_i
+ 0.15 obstacle_clearance_i
+ 0.15 lateral_alignment_i
```

它回答：

`如果当前要放能量，谁更应该优先获得推进预算。`

### 6.2 占位限流

然后乘上 occupancy 给出的局部限流系数：

```text
queue_drive_scale_i in [0, 1]
```

如果后车进入前车占位区，它会掉推进，而不是继续硬冲。

### 6.3 局部预算

最小预算写成：

```text
budget_i = T * queue_drive_scale_i * (0.35 + 0.65 local_release_i)
```

然后再用一个非线性映射，把中等预算也放大到可工作的推进区：

```text
budget_effect_i = sqrt(budget_i)
```

### 6.4 推进和旋度增益

最后：

```text
drive_gain_i = g_min + (g_max - g_min) budget_effect_i
curl_gain_i  = c_min + (c_max - c_min) budget_effect_i
```

因此：

1. `T` 决定系统现在整体能放多少；
2. `queue_drive_scale_i` 决定谁该被限流；
3. `local_release_i` 决定谁更该优先获得 through 预算。

## 7. 当前实现对应关系

当前代码位置：

1. 方法开关与编号：`src/smgf/core.py`
2. 能量池与预算逻辑：`src/smgf/core.py`
3. 最小验证组：`AJ_structured_energy_tank_validation`

当前新增方法：

1. `M47 = Xi-Polarity-Axis-Queue-Occupancy-TankEnergy-lite`
2. `M48 = Xi-Polarity-Axis-Queue-Occupancy-DualTank-lite`

它保留了：

1. `Xi polarity axis`
2. `queue occupancy`
3. `pure soft corridor control`

但把 `M43` 的固定高能量版本，替换成了当前这个最小 `T`-based 版本。

## 8. 最小实验验证

当前已跑最小验证组：

`outputs/aj_structured_energy_tank_validation_v1/`

对比方法：

1. `M43`
2. `M46`
3. `M47`

场景：

1. `s4_narrow_passage_easy`
2. `s4_narrow_passage_medium`
3. `s4_narrow_passage_hard`

### 8.1 当前结果摘要

1. `M43` 仍然是当前最强 corridor through 版本：
   - `medium success_rate = 1.0`
   - `easy success_rate = 0.333`
   - `hard` 上会过冲碰撞
2. `M46` 仍偏保守：
   - `easy/medium/hard` 全部 `0 success`
3. `M47` 当前也仍偏保守：
   - `easy/medium/hard` 仍是 `0 success`
   - 但它已经把“场景层 + 拓扑层 + 全局 T + 局部预算分配”这套分层机制真正接进了现有低层执行链

### 8.2 这个最小验证说明了什么

它说明两件事：

1. 当前分层模型是**可实现的**，不是纯空谈；
2. 但当前 `T` 的储能/释能律还不够强，暂时更像 `M46` 的保守侧，而不是已经达到 `M43` 的 through 工作区。

因此当前最合理的结论不是“统一能量系统已经超过 `M43`”，而是：

> 最小分层模型已经成功嵌入现有 corridor executor，并形成了可运行的实验原型；它证明了“场景层门控 + 拓扑层储能 + 局部分配”这条路线可以真正落到代码里，但当前仍需要继续调强储能区间和释放分配，才能从保守侧进入 `M43` 当前所在的 through 工作区。

## 9. 下一步最自然的优化方向

当前最该继续补的不是再加新概念，而是沿着这 4 个点做小步优化：

1. 让 `T` 的储能更多依赖当前队列成形，而不只是滞后反馈；
2. 把 `drive` 分配从局部启发式进一步升级为显式归一化预算分配；
3. 让出口前后的 `release` 和 `recover` 分段更清楚；
4. 专门针对 `medium -> hard` 的过冲边界，增加“高压区耗散增强”而不是单纯压低全局能量。

## 10. 一句话总结

当前这个最小模型的核心表述可以写成：

> 场景层决定是否进入 corridor through 工作区，拓扑层决定结构是否已经具备储能与释放条件，全局能量池 `T` 记录当前可用于 through 的输运能量，各 agent 再根据队列位置、局部前向空间和占位约束获得不同的推进预算；`M47` 是这套分层机制的第一版可运行原型，但目前仍处于“保守可运行”而非“性能超过 `M43`”的阶段。

## 11. 再往前一步：真正可改代码的双阶段版本 `M48`

在 `M47` 的基础上，当前已经进一步实现了一个更接近“原始拓扑有预储能”的最小代码版本：

`M48 = Xi-Polarity-Axis-Queue-Occupancy-DualTank-lite`

它做了两件事。

### 11.1 把初始拓扑拆成 `E_ready` 和 `E_debt`

初始不再只给一个固定 `T(0)`，而是从原始结构里估计：

1. `E_ready`：可转成进场能量的预组织势能；
2. `E_debt`：当前结构无序、受挤压、需要先重构的负债。

当前最小定义使用：

1. `slot_alignment`
2. `compress_potential`
3. `continuity`
4. `obstacle_clear`
5. `queue_shape`
6. `teammate_pressure`
7. `obstacle_pressure`

然后构造：

```text
T_approach(0) = clip(E_ready - E_debt, 0, 1)
```

### 11.2 把单一 `T` 拆成双阶段

当前最小双阶段结构是：

1. `T_approach`
   - 用于“进场、收拢、形成可 through 工作队形”
2. `T_transport`
   - 用于“进入通道后的 through 输运与释放”

两者之间通过一个转换项连接：

```text
convert = k_convert * corridor_commitment * release_ready * T_approach
```

于是最小版更新可以写成：

```text
dT_approach/dt = P_store_approach - convert - P_loss_approach
dT_transport/dt = convert + P_store_transport - P_release_transport - P_loss_transport
```

局部预算则从：

1. `approach_budget`
2. `transport_budget`

叠加成最终 `budget_i`。

### 11.3 `M48` 当前实验现象

对应输出目录：

`outputs/ak_dual_stage_energy_validation_v1/`

当前现象是：

1. `M48` 已经真正拥有了非零 `T_approach(0)`；
2. `T_approach` 会逐步向 `T_transport` 转移；
3. `budget` 也明显高于 `M47` 的完全保守版本；
4. 但它仍未打通 `entry_progress`，说明“有预储能”还不等于“已经形成足够强的进场推进链”。

这说明：

> `M48` 已经把“原始拓扑结构有能量”这个想法真正写进代码里了；当前未解的问题不再是“这个想法能不能实现”，而是“如何让 `T_approach` 真正转化成足够强的进场 through 推进，而不是只形成中等但不够破局的局部预算”。

## 12. 从 `M59` 到 `M62`：为什么要把系统压成弹簧链

### 12.0 更新说明（2026-06-16）

本节原先把 `outputs/aw_*` 与 `outputs/ax_*` 作为最新依据，但仓库后续已经新增：

1. `outputs/ba_m62_transport_boost_v1/`
2. `outputs/bb_m62_extended_corridor_probe_v1/`
3. `outputs/bc_m43_m62_cross_type_validation_v1/`
4. `outputs/bd_m43_m62_dense_tracking_hard_v1/`

因此这里需要补一条更准确的当前判断：

1. `M62` 仍然稳定优于 `M59/M61`，尤其体现在 `release_proxy`、`effective_drive`、`entry_flux` 与正 `r_peak` 上；
2. 但在验证种子段里，`M62` 还没有稳定达到 `M43` 的 corridor success 水平；
3. 因而当前最稳妥的结论应是：`M62` 证明了 spring-chain reduced model 的方向有效，但它还不是 `M43` 的替代完成版。

### 12.1 `M59` 显式 release 映射后的新判断

最新结果位于：

1. `outputs/au_m59_transport_release_mapping_v1/`
2. `outputs/av_m59_frontloaded_release_v1/`

当前可以更准确地区分两个阶段：

1. 原始 `M59`
   - 已有 `Xi` 调度、global quota 和非零 `transport_tank`；
   - 但 `transport_tank -> release_drive` 太弱，导致几乎没有正的 `entry_flux`。
2. front-loaded `M59`
   - 显式把 `transport_tank` 映射到前半队列 release 上后；
   - `effective_drive`、`entry_flux` 和 `r_peak` 都明显改善；
   - 但仍未稳定 through。

这说明：

> 统一能量线原先确实卡在 `transport_tank -> release_drive`；
> 但把这条链接出来之后，又暴露出一个新事实：
> 单靠头部聚能本身，还不足以形成持续 through。

### 12.2 `M61` 与 `M62`

最新结果位于：

1. `outputs/aw_m61_leader_frontloaded_probe_v1/`
2. `outputs/ax_m62_spring_chain_probe_v1/`

新增两条方法原型：

1. `M61 = LeaderFrontLoaded-TransportRelease-lite`
2. `M62 = SpringChain-Reduced-TransportRelease-lite`

它们的分工是：

1. `M61`
   - 测试“参考 leader 方案，把 release 更集中到头部”是否足够；
   - 结果表明：单独强化队首并不优于 `M59`。
2. `M62`
   - 把系统改写成最小两段式 spring-chain：
     - `entry`：链式压缩储能 + 队首 release
     - `transport`：后段压缩支撑 through
   - 结果表明：
     - `medium` 上 `r_peak` 已经变正；
     - `hard` 上 `r_peak` 已非常接近 0。

当前新增的最小链式观测包括：

1. `spring_chain_storage`
2. `spring_head_release`
3. `spring_rear_support`
4. `spring_link_compression`

最重要的新判断是：

> 对当前 corridor through 来说，更合适的 reduced physics 不是“单点 leader 释放”，
> 而是“单主轴上的主动阻尼弹簧链”：
> 后段压缩储能通过链式支撑队首释放，队首释放再形成 through 通量。

## 13. 一句话总结

当前这个最小模型的最新表述可以更新为：

> 场景层决定是否进入 corridor through 工作区，拓扑层决定结构是否已经具备储能与释放条件，全局能量池 `T` 记录当前可用于 through 的输运能量；而在继续往前推进后，最新 `M59 -> M61 -> M62` 跟进已经说明：真正有前景的 reduced model 很可能不是继续细分更多局部 budget，而是把系统压成“entry -> transport”两段式的主动弹簧链 through 动力学。
