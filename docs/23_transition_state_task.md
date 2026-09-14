# R+P → TS 同步任务与端点等变消息（2026-09-12）

用户指示两条：

1. **R→P 与 R+P→TS 同步测试**；
2. 现在 R→P 里"末态结构参与传递的仅有非等变消息"，那么在 R+P→TS 里
   初末态是否也应该传**等变消息**，并要给出传等变消息后的对比结果。

本文件记录这两条如何落地、以及它们与 `docs/22` 的关系。

## 一、为什么 R→P 里末态只有非等变消息

R→P 的末态信息只有两样东西进网络：

| 通道 | 内容 | 等变性 |
|---|---|---|
| `bond_change` one-hot | 每条边 `{不变, 生成, 断裂}` | **不变**（拼进 `reference_filter` 的标量通道） |
| `_bond_change_direction` | 由参考几何 + 标签算出的每原子拉/推向量 | **等变**，但它是从**反应物**几何导出的，不含产物坐标 |

即：产物的**坐标**从未以向量形式进入消息传递；进入的只是它的**离散拓扑**。
这正是 `docs/22` §一之三 补 `bond_change_direction` 的动机——补上之后
`t = 0` 的初始向量特征不再恒为 0（`t=0` 时 `current = x0 = R = reference`），
128 事件探针从 0.117 Å 降到 0.092 Å。

## 二、R+P → TS 的任务定义

| 项 | 取值 |
|---|---|
| 条件 | **R 与 P 都给**（ReactOT / MolGEN 的设定，产物是输入不是泄漏） |
| 初值 `x0` | **中点桥 `(R + P)/2`**（ReactOT 的先验；实测 `\|TS-R\|` RMS 2.24–3.07 Å，TS 基本在中点附近） |
| 目标 `x1` | 记录里的过渡态 |
| 采样 | 1 步直接回归（`flow_time = 0`，与 R→P 最优臂一致的参数化） |
| 指标 | Kabsch 对齐 RMSD（movable 原子）、`min(rmsd,1)`、逐事件最优 |
| 数据 | `transition1x_aligned_9000`，9000/536/537；**全部 10073 条都有 TS** |

## 三、端点消息：等变 vs 非等变（可消融）

`DualMessageBlock` 增加一个**端点几何通道**，与被消融的开关一一对应：

```text
filters      += endpoint_filter(RBF(|P 边|)) * cutoff        # 不变通道，两条臂都有
vector_message += endpoint_unit * endpoint_gate * cutoff      # 等变通道，仅 equivariant 臂有
```

- `endpoint_condition = true`：端点几何进入网络；
- `endpoint_equivariant = true`：它的**边方向**进入向量消息；
  `false` 时只留距离（非等变）通道。

所以两条臂的差值**只**来自"端点的等变消息"，其余（数据、初值、优化器、调度、
损失、步数）完全一致。`endpoint_equivariant` 需要 `endpoint_condition`，
且两个开关都默认关闭——**所有既有 checkpoint 的 state_dict 一字不变**。

实现位置：

| 文件 | 内容 |
|---|---|
| `basinflow/models/painn/layers.py` | `DualMessageBlock` 的 `endpoint_*` 通道与两个开关 |
| `basinflow/models/painn/painn.py` | 端点几何在同一 `edge_index` 上取 RBF/单位向量；缺张量时报明确错误 |
| `basinflow/models/painn/modules.py` | `prepare_flow_input` 透传 `endpoint_pos` |
| `basinflow/models/factory.py` | 两个开关的配置解析 |
| `basinflow/data/pyg.py` | `TransitionStateDataset`（无种子：两端都是任务给定） |
| `basinflow/evaluation/transition_state.py` | 采样 + 对齐 RMSD / 截断 / 逐事件最优 |
| `basinflow/workflows/transition_state.py` | `basinflow train-transition1x-ts` |
| `tools/evaluate_transition_state.py` | checkpoint 评分与逐 epoch 曲线 |

测试：`tests/test_transition_state_task.py`（6 项），含"两条臂输出必须不同"这一条，
保证消融不是空转。

## 三点五、为什么端点的等变消息在本参数化下是冗余的

用户的假设是"端点的等变消息应该对生成有好处"。实测**不支持**，而且原因可以量化。

本任务的初值是**中点桥** `x0 = (R+P)/2`，于是现有的向量通道
`vector_input(current − reference)` 恰好就是

```text
midpoint − R = (P − R)/2
```

而端点的等变通道给的是 `endpoint − current = P − midpoint = (P − R)/2`——
**同一个方向**。逐边统计（val 前 64 个事件、7810 条边）：

| 边方向夹角余弦 | 中位 | 均值 | \|cos\| > 0.99 的比例 |
|---|---|---|---|
| `cos(current, reference)` | +0.976 | +0.906 | 33.6% |
| **`cos(current, endpoint)`** | **+0.988** | +0.936 | **46.1%** |

端点的单位向量几乎完全落在现有向量通道的张成里，**没有新信息**。
两条完整训练臂的 loss 也印证：invariant 每轮都略优于 equivariant
（epoch 5：0.1778 vs 0.2144），与 128 事件探针一致。

**但这个结论只在早期成立。** 跑到 epoch 23 时两条臂反超了：

| epoch | 端点等变 clip / 训练 L1 | 端点非等变 clip / 训练 L1 |
|---|---|---|
| 5 | 0.4418 / — | 0.3690 / — |
| 21–23 | **0.2947** / **0.0544** | 0.3188 / 0.0970 |

等变臂在 **clipped 均值上好 7.6%，训练 L1 低 44%**——即"生成效率"确实因等变消息而改善，
方向与用户的判断一致，只是**收益要在训练足够久之后才显现**，epoch 5 的对照会给出相反结论。
早期冗余（§三点五开头）解释了为什么收益小：`current − reference` 与 `endpoint − current`
在 t=0 近似反向共线，所以等变通道是"补上另一半方向"而不是全新信息。

**这条仍然是参数化相关的结论，不是对等变消息本身的普适判断。** 有一个可验证的推论：
如果初值不是中点桥而是 MolGEN 的噪声源 `x0 = (P+R)/2 + σξ`，那么
`current − reference = (P−R)/2 + σξ`，端点方向 `P − current = (P−R)/2 − σξ`
就**不再共线**，等变通道应当重新携带独立信息。按用户授权"MolkGEN 规则可重新检验"，
已把 `source_scale` + `time_distribution = beta` 接进 `TransitionStateDataset`，
用同一对开关跑 MolGEN 规则的第二组对照即可证伪或证实这一推论。

## 四、同步进行的实验臂

| 任务 | 臂 | 配置 |
|---|---|---|
| R→P | 键变化标签 + cosine（对照） | `direct_t0_nonoise_cosine_transition1x_bond_change.ini` |
| R→P | **+ 方向向量**（主押注） | `direction_cosine_transition1x_bond_change.ini` |
| R+P→TS | 端点**非等变**消息 | `transition1x_ts_invariant.ini` |
| R+P→TS | 端点**等变**消息 | `transition1x_ts_equivariant.ini` |

四条臂共用同一 backbone 容量、同一损失（L1）、同一调度（warmup 5% + cosine 到 5%）、
同一训练数据划分。R+P→TS 的结果由 `tools/evaluate_transition_state.py` 逐 epoch 记录到
`<run>/ts_curve/ts_curve.csv`，R→P 的落在 `<run>/rmsd_curve/rmsd_curve.csv`。

## 五、R+P → TS 的平凡基线（决定目标是否现实）

中点桥 `(R+P)/2` 直接当 TS 用，不做任何学习：

| 划分 | 口径 | n | aligned 中位 | 均值 | `min(rmsd,1)` | < 0.2 Å | < 0.5 Å |
|---|---|---|---|---|---|---|---|
| val | 全部 | 536 | 0.4870 | 2.0232 | 0.5634 | 10.4% | 50.9% |
| test | 全部 | 537 | 0.4824 | 2.2371 | 0.5700 | 11.4% | 52.9% |
| test | 单事件 | 381 | **0.3684** | — | — | 16.0% | 74.5% |

对照 R→P 的平凡基线（"什么都不做"，val 中位 **1.0811 Å**）：

- **R+P→TS 的起点好 2.2–2.9 倍**。这正是 `docs/22` §三 的结论：0.1–0.2 Å 这个量级
  本来就属于"两端点已知"的任务。R→P 要从 1.08 Å 走到 0.1–0.2 Å 需要跨过数据歧义；
  R+P→TS 只需要从 0.48 Å 再压 2–3 倍，而 TS 由两端点基本决定。
- 因此**模型必须先打败中点桥**（< 0.487 Å）才算有贡献；epoch 1 时两条臂都在
  1.7–1.8 Å，即还没学会"不要离开中点"。

## 六、当前状态

四臂同时训练（15:00 起，CPU 8 核共享）：

| 任务 | 臂 | epoch（15:30） |
|---|---|---|
| R→P | 键变化标签 + cosine（对照） | 77 / 80 |
| R→P | + 方向向量（主押注） | 45 / 80 |
| R+P→TS | 端点非等变消息 | 2 / 80 |
| R+P→TS | 端点等变消息 | 2 / 80 |

R+P→TS 的逐 epoch 曲线在 `<run>/ts_curve/ts_curve.csv`（`tools/evaluate_transition_state.py
--watch`），R→P 的在 `<run>/rmsd_curve/rmsd_curve.csv`。最终对比补入本节。
