# 29 · 生成原理图（方法总览图）：提示词

> **与 `docs/28` 的分工**（这是最容易搞混的一点）：
>
> | | `docs/28` 结构图 | 本文件 原理图 |
> |---|---|---|
> | 画什么 | 代码怎么组织：包、模块、类、调用链 | 方法怎么工作：几何、概率路径、场、积分、候选 |
> | 节点标签 | `basinflow/data/samples.py`、`EventFlowDataset` | 数学符号：`R`、`x_t`、`v_θ`、`Δ` |
> | 读者 | 接手代码的人 | 评审 / 组会 / 论文 Figure 1 |
> | 错的典型 | 画了不存在的模块 | 把 `candidate_sampler.py` 画进原理图；把 `P` 画进推理路径 |
>
> 本文件的提示词按 2026-09-15 工作树的**实际数学**核对过（`data/samples.py`、
> `models/painn/painn.py`、`sampling/ode.py`、`sampling/candidate_sampler.py`），
> 包含"当前配置处于哪种退化/非退化状态"的如实标注 —— 这类标注是原理图最容易漏、
> 评审最容易抓的地方。

---

## 0. 怎么用

| 你要的产物 | 用哪节 |
|---|---|
| 论文/组会的方法总览图（矢量、可改、公式排版正确） | **§1 主提示词**（输出后端选 TikZ 或 draw.io XML） |
| 先花 5 分钟把逻辑理顺，再谈排版 | **§2 Mermaid 变体**（画得丑但对，用来定内容） |
| 封面/海报的"示意图风格"视觉稿 | **§3 图像模型变体**（英文、公式一定糊，只看构图） |
| 核对生成结果 | **§4 公式与事实底稿**、**§5 易错清单**、**§6 自检** |

**一句话原则**：**原理图里只允许出现数学符号**。模块名、文件名、类名一律不进原理图 ——
它们属于 `docs/28` 那张结构图。两张图混在一起，读者既看不懂方法也找不到代码。

---

## 1. 主提示词（原理图 / 方法总览图）

> 整段复制。`[方括号]` 是需要你替换的项。

```text
# 角色
你是科研论文配图专家，擅长画机器学习方法总览图（paper Figure 1）：
信息层级清晰、符号统一、公式排版正确、黑白打印仍可读。

# 任务
为 BasinFlow（学习辅助的 KMC 事件提案框架）画一张**方法原理图**，
说明"给定一个反应物盆地，如何生成多个候选逃逸事件"。

# 交付格式
[三选一，默认第一个]
(a) TikZ（pdflatex 可直接编译，standalone document，无外部图片依赖）
(b) draw.io / diagrams.net 的 XML
(c) Mermaid（只用于快速核对逻辑，不要求美观）
只输出图本身；如用 TikZ/draw.io，同时给一段 ≤ 90 字的中文图注。

# 版面契约
1. 四面板布局 2×2，左上起顺时针标 (a)(b)(c)(d)，每个面板左上角一个加粗小标签：
   (a) 输入与提案目标   (b) 训练目标：条件流匹配
   (c) 模型：双几何等变场  (d) 推理：自适应 ODE 采样与候选评估
2. 面板之间用**细虚线分隔**，不用粗框；整图底部一条通栏的 "Notation & Invariants" 小字区。
3. 配色：主色 1 种（结构/几何，建议 #2F5D8C 蓝）+ 强调色 2 种
   （训练=#C1663A 赭，推理=#3F7A5E 绿）。**只靠颜色区分是不够的**：
   训练路径用实线、推理路径用实线加箭头标签"inference"、未实现阶段用灰色虚线 + "not implemented"。
4. 字体：正文 8 pt，公式 9 pt（数学模式），图注 7 pt；中文用思源宋体/Noto Serif CJK，
   符号用 Computer Modern。所有张量维度用 `\mathbb{R}^{N\times 3}` 形式标注在箭头旁。
5. 箭头必须带标签说明**流过去的是什么**（几何 / 条件 / 速度场 / 标量），不画无标签箭头。
6. 固定原子（`movable_mask = False`）在 (a) 和 (d) 里画成**灰色小球**，可动原子实心；
   并在 (a) 里用一个灰色小图例说明"灰色 = 固定"。
7. 全图禁止出现文件名、目录名、类名、Python 标识符（`basinflow/...`、`CandidateSampler` 等）。
   若确需给实现锚点，只允许在图注里写一句"实现见 `basinflow/sampling/ode.py`"。

# 各面板必须画的内容

## (a) 输入与提案目标
- 一个**反应物盆地** R（画 2 个例子：周期性板状晶体 + 一个小分子，体现"分子与周期体系共用一套接口"）。
- 标注"一个盆地 ↔ 多个有效事件"（one-to-many），这是本方法的前提，不要画成 R → 单一 P。
- 种子 seed 的三种进入形式，画成三个并列的小标签（这是本方法的核心设计，必须显式）：
  ① 标量条件：`m`（movable_mask）、`a`（active_prior）、seed 类型
  ② 等变向量条件：`d`（seed direction）
  ③ 流初始几何：`x_0 = R + δ`（`δ` = seed displacement）
- 明确的输出集合：`R_basin → { (P_k, TS_k, A_k, d_k) }_k`，`A_k` = 活跃原子，`d_k` = 事件方向。
- 一条醒目的注释："seed 不是答案，是多样性的来源（不同 seed → 不同探索方向）"。

## (b) 训练目标：条件流匹配
画两条并列的任务分支，共用同一套"几何插值 + 目标速度"的图示语言：

任务 A（R → P，分子）：
  x_0 = R + σξ，ξ ~ N(0, I)，仅可动原子加噪
  x_t = (1 − t)·x_0 + t·x_1，x_1 = MIC 提升后的产物 P
  v = x_1 − x_0        （与 t 无关）
  t ~ U(0,1)，每个 epoch 重新抽噪声与时间

任务 B（R + P → TS）：
  x_0 = (R + P)/2 + σξ        （σ = 0 时退化为确定性中点桥）
  x_t 由所选路径给出；**三条路径都要画成同一根直线上的不同"时间参数化"**，并写出切向量：
     linear:        x_t = (1−t)x_0 + t·x_1          v = x_1 − x_0
     decaying:      x_t = x_0 + (x_1−x_0)(2t − t²)  v = 2(1−t)(x_1 − x_0)
     trigonometric: x_t = x_0 + (x_1−x_0)sin(πt/2)  v = (π/2)cos(πt/2)(x_1 − x_0)
  用一个小 inset 曲线说明 decaying 的意义：轨迹完全相同，但 |v| 在 t→1 时归零（自限场），
  且 ∫₀¹ v dt = x_1 − x_0 精确成立。
  P 在这里是**条件输入**（与 A 中 P 是监督不同），必须用不同颜色/线型标出这一区别。

损失（两个任务共用，画一个方框）：
  L = ‖ v_θ(x_t, t) − v ‖²（或 L1），误差只在 `m = 1` 的原子（movable）上取平均
  旁注："唯一监督项是速度；活跃原子与方向是它的派生量，不另设监督头"

## (c) 模型：双几何等变场 v_θ
- 画同一结构的两份几何并排进入网络：
  reference = R（未扰动反应物）与 current = x_t（当前流态）；
  B 任务额外有第三个 geometry：endpoint = P（虚线接入，标注"仅 R+P→TS 任务"）。
- 图构造：每次场求值按截断半径 r_max 重建**两个半径图的并集**，边上带晶格平移向量
  （PBC：cell + cell_offsets 必须成对出现，画一个小示意说明"跨边界成键"）。
- 条件通道：
  节点标量：m（movable_mask）、a（active_prior）、时间嵌入 t
  边条件（可选）：bond_change ∈ {0 不变, 1 生成, 2 断裂} 的 one-hot，旁注"离散拓扑，不含产物坐标"
- 输出：v_θ ∈ R^{N×3}，标注"O(3) 等变"；并注明"固定原子的输出在损失与采样场中都被屏蔽"。
- 用一个小箭头标出网络内部的标量通道 + 向量通道（PaiNN 式消息传递），不必画层数细节。

## (d) 推理：自适应 ODE 采样与候选评估（无 oracle）
- 起点：x_0 = R + δ，δ 来自**零初始化或高斯初始化**（画 3～5 个不同 δ 的分支 → 多样性）。
- 只允许出现反应物侧信息：在此面板内**不得出现 P 或 TS**（它们只在评估时作为真值出现，
  且必须画在面板右侧、用一条明确的"仅评估用"灰虚线隔开）。
- ODE：dx/dt = v_θ(x, t)，t: 0 → 1，自适应求解器（dopri5，atol/rtol），
  旁注"num_steps 是保存点个数，不是步数"；固定原子速度在**场内部**置零（每一步都成立）。
- 产出 K 个候选：聚类去重 → 与已知产物做最近匹配（MIC-RMSD 与 Kabsch 对齐两种口径）。
- 指标框：recall@{0.1, 0.2, 0.5} Å、macro 最近产物 RMSD、候选数、场求值次数（成本）。
- 下游用**灰色虚线**分三级，并明确标 "not implemented"：
  EAM 结构弛豫 → 鞍点/TS 物理验证（NEB/dimer/ARTn） → 事件库与率表导出（KMC）。
  在灰虚线上写一句口径："几何候选召回 ≠ 鞍点验证过的 KMC 事件召回"。

# 底部 Notation & Invariants 通栏（小字，必须有）
符号：R 反应物盆地 / P 产物 / TS 过渡态 / x_0 流初始几何 / x_t 流中间态 / x_1 目标态 /
t ∈ [0,1] 流时间 / v_θ 学习到的速度场 / σ 初始化噪声尺度 / ξ ~ N(0,I) /
m movable_mask / a active_prior / δ seed displacement / d seed direction / Δ = x_1 − x_0
不变量（逐条列出，可用编号）：
1. movable_mask 是唯一的移动语义；固定原子 = ¬m
2. active_prior 是条件输入，不是监督
3. 唯一监督项是速度；无活跃/方向辅助头
4. 推理路径不含任何目标信息（无 P、无 TS、无 target_*）
5. 几何候选召回 ≠ 鞍点验证过的 KMC 事件召回
6. 全链路 PBC：cell 与 cell_offsets 成对使用

# 允许（且鼓励）的状态标注
在 (b) 面板右下角加一个浅灰小注（如"当前出厂配置"），如实说明：
- 任务 A：t 在 [0,1] 上均匀采样（非退化）；
- 任务 B：仓库现有 TS 配置把 t 钉在 0、σ = 0（确定性中点桥 + 单点监督 = 直接回归，
  不是完整的场）；完整配方需要 σ > 0 与 t ~ Beta(0.8,0.8)，decaying 路径为候选修复臂。
这条注解是加分项，不要省略 —— 方法图不写清这一点会被评审直接问住。

# 禁止（出现任何一条即不合格，请重画）
- 不得在图上出现文件名/类名/函数名/配置键（`samples.py`、`CandidateSampler`、`flow_time` 等）
- 不得在 (d) 面板内出现 P 或 TS，或用任何方式让产物信息"从左边溜进"推理路径
- 不得把鞍点验证、弛豫、事件库、率表导出画成已实现（必须灰虚线 + not implemented）
- 不得把方法画成 R → 单一 P 的一对一预测；一个盆地到多个事件是本方法的立论
- 不得给"活跃原子/方向"画独立的监督头或独立损失项
- 不得把 seed 画成输入特征拼接（必须体现"标量条件 + 等变向量条件 + 流初始几何"三种进入方式）

# 输出前自检（不满足就自己改，不要输出解释）
1. (d) 面板里能搜到 "P" 或 "TS" 吗？必须为"否"（只在右侧"仅评估用"区域出现）。
2. 每条路径公式与它的切向量是否配对正确？（linear↔x_1−x_0；decaying↔2(1−t)Δ；trig↔(π/2)cos(πt/2)Δ）
3. seed 的三种进入形式是否都在 (a) 上？
4. 未实现的三级下游是否都是灰虚线？
5. 六条不变量与符号表是否都在底部通栏？
6. 图上是否有任何 Python 标识符？必须为"否"。
```

---

## 2. 变体：Mermaid（5 分钟定内容）

拿到第一版 TikZ 前，建议先跑这一版把逻辑钉死：

```text
用 Mermaid `flowchart LR` 画出 BasinFlow 的生成原理（不是代码结构）。
要求：节点只用数学符号与中文短语，禁止出现文件名/类名。
链路必须严格是：
  [反应物盆地 R] --> [seed: 标量条件 m,a / 等变向量 d / 初始几何 x0 = R + δ]
  --> [无目标推理视图（零 target_*）]
  --> [自适应 ODE 积分 dx/dt = vθ(x,t), t: 0→1]
  --> [K 个候选结构 P̂_k]
  --> [聚类 + 最近已知产物匹配（MIC-RMSD / Kabsch）]
  --> [指标：recall@0.1/0.2/0.5 Å、macro 最近产物 RMSD、场求值次数]
另外用虚线画训练侧：
  [任务 A: x0 = R + σξ, x_t = (1-t)x0 + t·P, v = P - x0]
  [任务 B: x0 = (R+P)/2 + σξ, 路径 ∈ {linear, decaying, trigonometric}, v = dx_t/dt, 条件 P]
  --> [vθ（双几何：R, x_t, 可选 P；边条件 bond_change；PBC 图每步重建）]
  --> [L = ‖vθ − v‖² 仅 movable 原子]
硬约束：训练侧到推理侧只允许一条边——"训练好的 vθ"；推理侧不得出现 P/TS 节点。
再补一个灰色虚线尾链：EAM 弛豫 -.-> 鞍点验证 -.-> 事件库/率表（标 not implemented）。
```

---

## 3. 变体：图像生成模型（视觉稿）

```text
A four-panel scientific method overview figure for a machine-learning paper,
flat vector style, white background, 16:9, no photorealism, no 3D renders.

Panel a (top-left) "INPUT & TARGET": a small crystal slab and a small molecule drawn as
ball-and-stick, most atoms solid dark blue, a few atoms light grey (fixed atoms);
three small labelled tags branching from the structure; an arrow to a fan of
three different candidate structures (multiple outcomes, not one).

Panel b (top-right) "TRAINING": three curves on one straight line between two
endpoint dots, showing the same straight trajectory with different time pacing;
one curve decays to zero slope at the right end; clean minimal axes, no numbers.

Panel c (bottom-left) "MODEL": two copies of the same structure (slightly different)
entering a stack of rounded blocks; a third structure entering with a dashed arrow;
a periodic lattice cell wireframe with two atoms connected across the boundary;
one output arrow labelled with a vector symbol.

Panel d (bottom-right) "INFERENCE": five small structures fanning out from one reactant,
each passing through a curved arrow into a rounded "ODE" block, converging to a
cluster of candidate structures compared against a reference structure on the right
separated by a light grey divider labelled "evaluation only"; three dashed grey boxes
at the bottom labelled "relaxation", "saddle validation", "KMC rate table".

Style: muted academic palette (slate blue, muted orange, muted green), 2 px strokes,
generous whitespace, orthogonal arrows with small labels, consistent line weight,
publication-quality flat design, no gradients, no clip art.
Text: keep every label to one or two English words; do not render Chinese characters;
do not attempt mathematical formulas — leave the formula areas as clean empty boxes.
```

> 说明：图像模型写不对公式和中文。这张只用来定构图和配色；公式与符号请回到 §1。

---

## 4. 公式与事实底稿（供校对，已内嵌在 §1）

| 项 | 当前实现（代码位置） |
|---|---|
| A 源 | `x_0 = R + δ`，`δ` 来自 `GaussianInit(scale=σ)`，只作用于可动原子（`seeds/generators.py:57`） |
| A 路径 | 线性、硬编码：`pos = (1−t)·source + t·P`（`data/samples.py:283`） |
| A 目标速度 | `v = P − source`，固定原子上置零（`data/samples.py:284-285`） |
| A 时间 | `time_distribution="uniform"` 默认，`(seed, epoch, index)` 派生、每 epoch 重抽（`data/samples.py:255-262`） |
| B 源 | `x_0 = (R+P)/2 + σξ`，`σ = source_scale`，默认 0 → 确定性中点桥（`data/samples.py:507-514`） |
| B 路径 | `path ∈ {linear, trigonometric, decaying}`；`decaying: x_0 + Δ(2t−t²)`，`v = 2(1−t)Δ`（`data/samples.py:476-501`） |
| B 条件 | `endpoint_pos = P`（MIC 提升），`target_pos = TS`（`data/samples.py:525,543`） |
| B 时间 | 默认 `fixed` + `flow_time = 0.0` → 单点监督（`workflows/transition_state.py:99-105`） |
| 损失 | 仅 `velocity`，掩码 `movable_mask`，`mse` 或 `l1`（`models/loss.py:44-70`） |
| 模型输入 | `positions_1 = R`、`positions_2 = x_t`、`z`、`edge_index`、`shifts`、`time`、`node_conditions = [movable, active_prior]`、可选 `bond_change`、可选 `endpoint_pos`（`models/painn/painn.py:59-107`） |
| 模型输出 | `{"velocity": [N,3], "hidden": ...}`；无 active/direction 头 |
| 向量初值 | `v_init = v_in(current − reference)` → `t = 0` 时恒为 0，故可选 `bond_change_direction` 提供等变初始方向（`painn.py:91-103`） |
| 图构造 | 每次前向重建双几何半径图并集，带 `cell_offsets`（`painn/modules.py`、`geometry/radius_graph.py`） |
| 采样 | `sample_ode`：`torchdiffeq.odeint(method="dopri5", atol=1e-6, rtol=1e-3)`，`t: 0→1`，`num_steps` = 保存点数（`sampling/ode.py:44-100`） |
| 固定原子 | 场内部置零（不是步后回位）（`sampling/candidate_sampler.py:162-165`） |
| 候选 | 每个 initializer 一个候选；每次试验 = 一个 `GaussianInit(scale, random_seed)`（`candidate_sampler.py:116-214`、`evaluation/semantic_flow.py:487`） |
| 指标口径 | `evaluation_protocol = "basin_level_gaussian_trials_no_oracle"`；MIC-RMSD 与 Kabsch 对齐两种（`evaluation/basin_recall.py`） |
| 未实现 | 鞍点验证 / 事件库 / 率表导出 / EON 集成（Stage 5–7） |

---

## 5. 易错清单（原理图最常被画错的 10 条）

| # | 常见画法 | 为什么错 |
|---|---|---|
| 1 | 把 `P` 放在推理面板里当输入 | 推理视图 `BasinDataset` 不构造任何 `target_*`，画进去等于宣称用了 oracle |
| 2 | 画成 `R → P` 单箭头（一对一回归） | 项目立论是**盆地级一对多**事件提案 |
| 3 | 给活跃原子/方向画独立输出头或独立损失 | 已删除；唯一监督是 velocity |
| 4 | 把 seed 画成"拼接进特征向量" | 必须是标量条件 + 等变向量条件 + **流初始几何**三种进入方式 |
| 5 | 把 `active_prior` 标成监督/标签 | 它是模型输入条件 |
| 6 | 给三条路径配同一个切向量 | `linear/decaying/trigonometric` 的 `dx_t/dt` 各不相同，只有 decaying 在 `t→1` 自限 |
| 7 | 把弛豫/鞍点验证/KMC 导出画成已实现 | 均为 Stage 5–7 目标，仓库内零命中 |
| 8 | 把"无弛豫 RMSD 召回"写成"事件召回率" | 几何候选召回 ≠ 鞍点验证过的 KMC 事件召回 |
| 9 | 画成静态近邻表 | 图在**每次场求值**时按当前几何重建（自适应步长下"每 k 步重建"没有意义） |
| 10 | 把 `num_steps` 标成"积分步数" | 它是保存点个数（MolGEN 的 `--inference_steps` 语义），真实步数由误差控制决定 |

---

## 6. 自检（生成后逐条打勾）

- [ ] (d) 面板内除"仅评估用"隔离区外无 `P`/`TS`/`target_*`；
- [ ] 三条路径的 `x_t` 与 `dx_t/dt` 成对且正确；
- [ ] seed 的三种进入形式都在图上；
- [ ] 训练侧与推理侧之间**只有一条边**：训练好的 `v_θ`；
- [ ] 未实现的下游全部灰虚线 + `not implemented`；
- [ ] 六条不变量 + 符号表齐全；
- [ ] 图上零 Python 标识符；
- [ ] 图注写明"几何候选召回，未做鞍点验证"与版本锚点（2026-09-15）。

## 7. 迭代话术（拿到第一版后）

```text
按以下顺序修改，每轮只改一处，改完重新输出整段代码：
1. "(d) 面板里出现了 P，请移到右侧 evaluation-only 隔离区，并补上分隔线标签。"
2. "(b) 的 decaying 曲线没有体现 t→1 时斜率归零，请重画并标出 ∫₀¹v dt = x_1 − x_0。"
3. "把 bond_change 边条件的说明改为『离散拓扑标签，不含产物坐标』。"
4. "底部不变量区压缩到 6 条，字号 7 pt，去掉与图注重複的内容。"
5. "整体改为黑白可读：只保留线型与灰度差异，颜色作为辅助。"
```

**最终图注模板**（版本锚点必须写，否则下次数据层再动这张图会静默过期）：

> 图 1 · BasinFlow 生成原理：以反应物盆地与事件种子为条件，学习速度场 `v_θ`，
> 经自适应 ODE 积分得到多个候选产物，并以几何召回（MIC-RMSD / Kabsch）评估。
> 虚线部分为未实现的物理验证与 KMC 导出。对应工作树 2026-09-15；
> 报告数字为**无弛豫几何候选召回**，非鞍点验证过的 KMC 事件召回。
