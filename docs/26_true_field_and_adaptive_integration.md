# 真正的场 + 自适应积分：实现与验证（2026-09-13）

> **2026-09-15 更新**：本文描述的 `basinflow/sampling/adaptive_ode.py` 手写积分器已删除——`torchdiffeq` 实际可以安装，现改为直接调用 MolGEN 用的同一个解算器（`torchdiffeq.odeint(method="dopri5", atol=1e-6, rtol=1e-3)`，见 `basinflow/sampling/ode.py`），且 R→P 与 R+P→TS 两条路径共用它。`accepted_steps`/`rejected_steps` 不再报告（torchdiffeq 不暴露步数），改为 `saved_points`；`num_steps` 现在是保存点网格而非步数上限。本文其余部分为当时记录。

用户终态明确：**要学真正的场，采样用自适应积分。** 本文件记录为此做的三处改动、
自适应积分器的实现与验证，以及两条探针的配置。前因见 `docs/25`。

## 一、为什么需要这两件事（`docs/25` 的结论）

`docs/25` 用三个实验证明：常数速度路径 `v = x1 - x0` 使训练目标在整个 t 上恒等，
网络只需输出一个常向量，一次前向就等于全部位移。实测签名：`|v_pred| ≈ 3.2–3.7 Å`
在 `t = 0…1` 上近似常数，轨迹 `0.487 → 1.35 → 0.94` 过冲回摆，方向误差中位 66–71°。

两个必要条件因此都缺：

| 条件 | 缺了会怎样 |
|---|---|
| **场有 t 依赖、有不动点** | 一步映射伪装成流：`v` 不随 t 衰减，`x1` 不是不动点 |
| **积分有误差控制** | 固定步长每步 ±1 Å 地过冲，第 2 步就离开训练流形 |

## 二、改动

### 2.1 终点监督（endpoint / x1-prediction）

`basinflow/models/loss.py`：`FlowLossWeights` 新增 `target ∈ {velocity, endpoint}`。

```text
velocity:  监督 v = x1 - x0          （常向量，t 无关）
endpoint:  监督 x1 = x_t + (1-t)·v_hat
           采样时 v = (x1_hat - x_t) / (1 - t)
```

为什么这给出"真正的场"：

- **自限**：`t → 1` 时 `v → 0`，`x1` 是 ODE 的**不动点**；
- **处处有定义**：`x1_hat` 是对终点的预测，对任意 `x_t` 都有意义；
- **与评测同量纲**：监督的是 `x1`，正是要评的 TS 位置。

`(1 - t)` 的分母在采样侧用 `clamp_min(1e-6)` 保护，配合 `time_epsilon`（见 2.3）。

### 2.2 自适应积分器（无新依赖）

`basinflow/sampling/adaptive_ode.py`：`torchdiffeq` 在本机装不上（沙箱禁止写
`~/.local`，`Errno 1`），因此**自己实现**：

- **方法**：五阶 Dormand-Prince tableau + **步长加倍**（step doubling）。
  每个接受的步比较"一整步 h"与"两个 h/2 步"，用 `|y(half) - y(full)| / (2^5 - 1)`
  估计局部误差；接受值再做 Richardson 外推 `y* = y_half + (y_half - y_full)/31`。
  用步长加倍而不是嵌入式系数对，是因为只需要一套 tableau，没有配错的余地。
- **误差尺度**：`scale = max(atol + rtol·max(|y_n|, |y_{n+1}|), 8·eps·|y|)`。
  最后一项是必需的：`atol` 若比浮点舍入还小，误差估计纯粹是舍入噪声，ratio 永远
  > 1，步长会向零塌缩而不是前进（实测 `atol=1e-12` 时正是如此）。
- **报告**：返回 `accepted_steps` / `rejected_steps` / `field_evaluations`，
  `record=True` 时附带每步状态（用于写轨迹）。

`basinflow/evaluation/transition_state.py` 的采样循环改为调用它：
`num_steps` 从"固定步数"变成**步数上限**，`t` 从 0 积到 `1 - time_epsilon`，
场每次查询都用**当前** t；`target="endpoint"` 时在采样侧换算
`v = (x1_hat - x_t)/(1-t)`；固定原子的速度被置零（保持约束与不动点一致）。

### 2.3 路径：线性 → decaying（本节是本轮最关键的发现）

只把目标换成 x1 并不能得到自限的场。实测（60 步小样本探针）：

| 配置 | 场形态 |
|---|---|
| 线性路径 + 终点监督，`ε=0.01` | `\|v\|` 从 2.8 涨到 **2.95e6**（`t=1` 处奇异） |
| 线性路径 + 终点监督，`ε=0.1` | `\|v\|` 仍随 t **上升**：3.5 → 22.7 |
| 线性路径 + 终点监督 + loss 加权 `1/(1-t)` | 仍上升：2.9 → 36.5 |
| **decaying 路径 + 速度监督** | **`\|v\|` 4.27 → 5.24 → 4.09 → 2.96 → 1.50 → 1.00（真值 7.42 → 0）** |

原因：源的散布只有 0.2 Å 时，`x1` 几乎可由 `x_t` 完全决定，**t 对网络没有信息量**，
它干脆忽略 t；此时终点参数化的 `1/(1-t)` 只是在放大一个与 t 无关的常数。加 loss
加权也没用，因为问题不在损失权重，而在**路径本身没有把 t 变成有用的条件**。

`decaying` 路径（`path = "decaying"`，`basinflow/data/pyg.py:_interpolate`）：

```text
x_t   = x0 + (x1 - x0)(2t - t^2)         轨迹与线性路径完全相同
dx/dt = 2(1 - t)(x1 - x0)                 t=0 最大，t=1 归零
∫₀¹ v dt = x1 - x0                        精确，采样终点就是目标
```

它同时满足三件事：**轨迹不变**（仍是直线，条件路径正确）、**场自限**（t=1 是不动点，
过冲会被拉回）、**t 有信息量**（目标随 t 线性衰减，网络必须建模时间）。

配套地，监督回到 **velocity**（`target = velocity`）：x1 参数化在 t→1 处的 `1/(1-t)`
在数值上没有好处，而 decaying 路径已经把自限性放进场里了。

### 2.4 时间区间裁剪

`TransitionStateDataset` 保留 `time_epsilon`（采样时间落在 `[eps, 1-eps]`，默认 0）。
decaying 路径下不再需要它：`t = 1` 处速度自然归零，`t = 1` 也不是奇点。

## 三、验证（不依赖训练）

`tests/test_adaptive_ode.py`（**8 项**）：

| 测试 | 断言 |
|---|---|
| `test_exponential_decay_matches_the_analytic_solution` | `x' = -x` 到 `x(1) = 1/e`（atol/rtol 1e-8） |
| `test_adaptive_takes_far_fewer_steps_than_a_uniform_grid_on_a_flat_field` | 常值场 < 10 步且无拒绝 |
| `test_adaptive_shrinks_the_step_when_the_field_is_stiff` | `x' = 50x` 仍收敛（固定步长会炸） |
| `test_convergence_is_bounded_by_the_tolerance` | 容差收紧 → 误差下降 |
| `test_a_dict_returning_field_is_supported_and_steps_are_reported` | 支持 `{"velocity": ...}`，步数被报告 |
| `test_runaway_field_is_reported_rather_than_looping_forever` | 发散场抛 `max_steps` 而不是死循环 |
| **`test_method_order_is_fifth`** | 阶数必须真的是五阶（见下） |
| `test_invalid_arguments_are_rejected` | 参数校验 |

`test_method_order_is_fifth` 是本轮最重要的测试：实现过程中**两次**把方法降成一阶
（多乘一次 `h`；把 state 塞进 stage 列表使权重整体错位一格），而其余测试**全部照常
通过**（小步长下仍然准）。该测试直接测阶数，所以不允许再退化。

`tests/test_transition_state_task.py`（16 项）新增/改写：

- `test_sampler_integrates_the_field_with_adaptive_steps`：场被查询 ≥6 次、
  时间跨越 [0.2, 0.5] 区间、剪切场积分出完整位移（"一步制"给不出）；
- `test_endpoint_target_recovers_velocity_from_the_prediction`：`target="endpoint"` 通路；
- `test_sampler_rejects_an_unknown_target`；
- 轨迹写入测试改为按**接受的步数**校验帧数与位移。

## 四、两条探针（运行中）

| 臂 | `source_scale` | `target` | `path` | 积分 | 步数上限 |
|---|---:|---|---:|---|---:|
| A | 1.0 | velocity | decaying | Dormand-Prince 5(4) | 128 |
| B | 0.2 | velocity | decaying | 同上 | 128 |

其余与 `docs/25` 的探针一致（9000/536，256×6 PaiNN，batch 64，10 epoch，L1）。
输出 `/tmp/bf_walkthrough/ts_decay_sig{10,02}`。

**判定门**（对齐 `docs/25` §六之五）：

1. **场是否自限**：`t` 响应曲线应从 t=0 的最大值衰减到 `t→1` 时的 0。
   **实测未通过**（见 §5.1：输出反而随 t 上升）；
2. **积分是否单调**：轨迹的 aligned RMSD 应单调下降，不再出现 `0.487 → 1.35 → 0.94`；
3. **精度**：aligned 中位必须打败中点桥 **0.487 Å**，理想目标是回到直接回归的
   0.230 Å 水平，同时保留随机源、可积分的场与 best-of-N 语义。

## 五、10 epoch 探针结果（decaying 路径）

两臂都跑完（9000 训练 / 536 val，256×6 PaiNN，batch 64，Dormand-Prince ≤128 步）：

| 臂 | final L1 | aligned 中位 | aligned 均值 | 源基线中位 | 源基线均值 | 接受/拒绝步 |
|---|---:|---:|---:|---:|---:|---:|
| σ=1.0 | 0.7392 | 0.9458 | 0.9693 | 1.7098 | 2.9802 | 18.4 / 6.4 |
| σ=0.2 | 0.3497 | 1.0207 | 1.0501 | 0.5823 | 2.1154 | 17.2 / 4.7 |

**都没打败中点桥基线 0.487 Å。** σ=1.0 臂确实把源推近了（1.71 → 0.95 中位），
但 σ=0.2 臂反而**比它自己的源更差**（源 0.58 就是中点桥，模型的积分把它推偏了）。

### 5.1 场没有学到时间依赖

在**固定的真实几何**上只改 `t`，测量训练后模型的输出：

| 模型 | \|v\| at t=0, .2, .4, .6, .8, 1.0 | max/min |
|---|---|---:|
| σ=1.0 | 2.37 3.64 4.37 5.33 6.88 9.26 | 3.91 |
| σ=0.2 | 4.17 6.37 7.72 8.85 9.56 9.65 | 2.31 |

**输出随 t 单调上升，而 decaying 路径的真值是 `2(1-t)Δ` 单调下降到 0**——方向完全相反。
所以两臂学到的都不是"运输场"，而是某种与 t 弱相关、幅度近似常数的映射。

### 5.2 已做的架构修正（保留，供下一轮使用）

| 开关 | 作用 | 默认 |
|---|---|---|
| `time_scale` | 加宽时间 Fourier 基的频率（原频率过窄） | 1.0（等价原行为） |
| `time_per_layer` | 把时间嵌入注入**每一层**消息块（原先只在输入层拼一次，经 6 层后被稀释） | false（参数集不变，旧 checkpoint 兼容） |

两者都验证过"打开后模型参数确实改变、关闭时与旧 checkpoint 完全兼容"。

### 5.3 一处被作废的实验（记录以免重犯）

用"单事件过拟合"判断时间条件是否可用是**无效实验**：该设置（1 事件、batch=1、
lr=3e-3）本身不收敛，实测 4 步内 loss 0.79 → 20.9、grad_norm 122，
`train_product_flow` 记录的首末 loss 相同是**发散后的假象**，不是"记不住"。
判断架构能力必须用稳定训练的配置，或与训练分开的探针。

## 六、口径提醒

本轮所有数字仍是**严格口径 + 单次采样**：禁镜像（proper Kabsch）、不截断 1 Å。
与 OA-ReactDiff/MolGEN 的 0.058/0.076 Å 并列前，必须先做
`docs/17_reference_protocol_audit.md` 要求的双口径（允许镜像 + 截断）报告，
并说明 confidence 模型缺失（本机无 checkpoint）。
