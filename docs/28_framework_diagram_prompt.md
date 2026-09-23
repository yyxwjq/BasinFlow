# 28 · 框架结构流程图：生成用提示词

> 目的：给出**可直接复制**的提示词，让任意工具（LLM / 图像生成模型 / 绘图软件）产出一张
> **与当前代码一致**的 BasinFlow 框架结构流程图。
>
> 事实底稿（§5）按 2026-09-15 的工作树逐文件核对：`basinflow/` 45 个 `.py`、
> 数据层重写（`docs/27`）与积分器统一（`torchdiffeq dopri5`）之后的实际状态。
> 提示词里内嵌了这份底稿，因此**不需要生成方访问仓库**也不会编造模块。

---

## 0. 怎么用（30 秒）

| 场景 | 用哪一节 | 说明 |
|---|---|---|
| 让 ChatGPT / Claude / DeepSeek 输出可渲染的流程图代码 | **§2 主提示词** | 输出 Mermaid，粘进 Markdown / mermaid.live 即可渲染，可反复微调 |
| 让 Midjourney / GPT-Image / Nano Banana 出"示意图风格"的图 | **§3 变体 B** | 纯英文、少文字；图像模型画不准中文小字，别指望它承载模块名 |
| 要论文/组会级矢量图（draw.io / TikZ / PPT） | **§4 变体 C** | 输出 draw.io XML 或 TikZ；文字精确、可改线宽字体 |
| 校对生成结果 | **§5 事实底稿 + §6 禁止事项** | 逐条核对；§6 列的是这类图最常被编造的东西 |

**推荐组合**：先用 §2 拿到 Mermaid 把**内容**定死（哪个模块连哪个），再用 §4 把同一份内容
交给绘图工具排版。不要一上来就用图像生成模型 —— 它会把模块名写成错别字，你还得逐个检查。

> 要的不是**代码结构图**而是**方法原理图**（几何 → 概率路径 → 速度场 → ODE 积分 → 候选，
> 给评审/组会看）？用 `docs/29_generation_principle_diagram_prompt.md`。两张图不要混：
> 结构图的节点是模块名，原理图的节点是数学符号。

---

## 1. 给"能读仓库的 agent"的附加指令（可选）

如果目标工具能访问本仓库（Claude Code / Cursor / Codex 等），在 §2 前面加这一段：

```text
在开始画图前，先执行以下核对（只读，不要改任何文件）：
1. `find basinflow -name '*.py' | sort` —— 确认你画的每个模块都真实存在；
2. `sed -n '1,40p' basinflow/cli.py` —— 确认入口命令；
3. `sed -n '1,60p' basinflow/models/loss.py` —— 确认监督项只剩 velocity；
4. `ls basinflow/sampling/ basinflow/evaluation/` —— 确认积分器与评测模块。
若某节点在代码中不存在，删掉它，不要凭文档或记忆补。
注意 docs/ 里部分文件带"本文为历史记录、未回改"的表头，它们描述的接口可能已被删除。
```

---

## 2. 主提示词（Mermaid，推荐）

> 以下整段可复制。中括号里的是**你需要按需替换**的项，其余原样发送。

```text
# 角色
你是软件架构可视化专家，擅长把研究型 Python 代码库画成信息密度高、无歧义的架构流程图。

# 任务
为项目 BasinFlow（学习辅助的 KMC 事件提案框架）画一张"当前实现结构"的流程图，
交付格式为**一个**可直接渲染的 Mermaid `flowchart` 代码块。

# 交付契约（逐条必须满足）
1. 只输出一个 ```mermaid 代码块；不要解释、不要备选方案、不要分多张图。
2. 图类型固定 `flowchart TB`（自上而下：入口在上，未实现的下游阶段在最下）。
3. 用 `subgraph` 表示代码分层，子图标题格式：`L<n> <中文层名> · basinflow/<子包>`。
4. 节点 ID 用简短英文（如 `data_events`）；节点标签格式 `中文短语<br/><code>模块.符号</code>`。
   凡是代码标识符（模块名、类名、函数名、配置键、张量字段名）**必须保留英文原样**，
   不得翻译、改写、按记忆更正。
5. 节点总数 ≤ 45。只能使用《内容清单》里出现的符号；不得新增、改名、合并或"顺便补全"。
6. 箭头语义：
   - `-->` 实线：已实现的调用或数据流；
   - `-.->` 虚线：仅用于 (a) 未实现的下游阶段，(b) 仅诊断/oracle 的支路。
   图末尾必须有一个 `subgraph 图例` 说明实线、虚线、灰底虚线框三者的含义。
7. 用 `classDef` 按层分色（浅色填充 + 深色描边，白底，适合投影）：
   入口/配置、数据层、几何、种子、模型、训练、采样、评测、松弛、未实现。
   未实现节点：灰色填充 + 虚线描边。
8. 图上必须能分辨三条路径，并在边标签上标出路径归属（`A:` / `B:` / `C:`）：
   - A 分子 R→P 监督训练
   - B 分子 R+P→TS 监督训练
   - C 盆地级无目标候选提案
9. 必须把《不变量》6 条以注释节点或边标签的形式画进图里（例如边上写 `movable_mask 掩码`），
   不要只在图外解释。
10. 数据来源（EON `.con` 目录 / Transition1x pickle → `events_dir`）作为数据层左侧的上游输入。
11. 版面：正交折线优先；同一子图内节点尽量同列；避免长距离斜穿箭头；
    整体适合 A4 横向或 16:9 幻灯片。

# 内容清单（唯一事实来源：只能画这些）
## L0 入口与编排
- basinflow/cli.py::main —— 三个子命令 train-transition1x / train-transition1x-ts / benchmark
- examples/*.py 薄壳入口（train_transition1x_product_flow.py、benchmark_semantic_flow.py）
- configs/*.ini → workflows/config.py::read_config（未知键直接报错）
- workflows/dataset.py::load_configured_dataset（唯一数据源模式：events_dir + 可选 manifest）
- 编排：workflows/transition1x.py::run_transition1x、
        workflows/transition_state.py::run_transition1x_ts、
        workflows/semantic.py::run_semantic_benchmark
## L1 数据层 basinflow/data/
- structures.py::Structure（species/z、positions、cell、pbc、movable_mask）+ from_ase / to_ase
- events.py：load_events（event_*.traj / *.extxyz + basin_table.csv）、
             Event（reactant/product/transition_state 三帧内联）、
             EventCatalog（盆地索引 dict[basin_id, [event_id]]）、
             event_target / EventTarget（MIC 提升后的产物坐标）、
             basin_reactant、select_basins、single_event_scope
- splits.py：BasinSplit、split_basins、save_split / load_split、find_split_manifest、open_dataset
- samples.py：EventData（PyG Data）、EventFlowDataset（模式 A）、
              TransitionStateDataset（模式 B）、BasinDataset（无目标推理视图）
- 上游转换工具：tools/eon_to_events.py、tools/transition1x_to_events.py、
                tools/merge_event_partitions.py、tools/prepare_transition1x_explicit_split.py
## L2 几何 basinflow/geometry/
- mic.py：minimum_image_displacement、pairwise_displacements、normalize_cell / normalize_pbc
- radius_graph.py：radius_graph、edge_geometry、dual_radius_graph（纯 torch，每次前向重建图）
- graph.py：build_neighbor_graph（ASE 基准）、torch_neighbor_graph_from_batch（EGNN / legacy）
- bonds.py：bond_adjacency、bond_change_labels、preserves_fragments、fragment_count
## L3 种子 basinflow/seeds/
- records.py：EventSeed（标量条件 active_prior/movable_mask、向量条件 seed_direction、
              流初始几何 seed_displacement）、SeedContext、seed_type_id / SEED_TYPE_IDS
- generators.py：ZeroInit、GaussianInit（盆地推理主路径）、DirectionalInit（legacy 采样入口）、
                 ProductInit（仅诊断 / oracle）
- from_config.py：init_generators
## L4 模型 basinflow/models/
- factory.py：build_model / model_spec / load_model_checkpoint（backend ∈ {painn, egnn}）
- loss.py：flow_loss + FlowLossWeights(velocity, norm) —— 只有 velocity 一项监督，误差只取 movable_mask
- painn/layers.py：径向基与包络、DualMessageBlock、UpdateBlock、GatedEquivariantBlock
- painn/modules.py：prepare_flow_input、PaiNN（EventData → 张量接口适配）
- painn/painn.py：DualPaiNN（reference = R、current = x_t、endpoint = P 仅模式 B 使用）
- egnn/：兼容与对比后端（单几何图）
## L5 训练
- training/product_flow.py::train_product_flow（逐轮验证、checkpoint、resume、调度器）
## L6 采样 basinflow/sampling/
- candidate_sampler.py：CandidateSampler.sample → Candidate
  （candidate_id、generated_structure_id、seed_id、seed_type、sampling_config）+ generated_structures
- ode.py：sample_ode（默认 torchdiffeq dopri5，可选 euler；固定原子的速度场内置零）
## L7 评测 basinflow/evaluation/
- basin_recall.py：movable_mic_rmsd、kabsch_aligned_rmsd、nearest_product_match、
                   cluster_candidates、evaluate_basin_recall
- semantic_flow.py::run_gaussian_trials（Au/Pt 周期体系 trial 编排 + 出图）
- single_event_molecule.py::run_single_event_molecule_trials（分子体系 trial 编排）
- transition_state.py::run_transition_state_trials（R+P→TS，midpoint bridge 协议）
- product_flow_quality.py（历史 EGNN pairwise 评测）
## L8 松弛（可选）basinflow/relaxation/
- relax.py：relax(structure, calculator, RelaxationConfig)、eam_calculator
## L9 未实现（必须灰底虚线框）
- 鞍点/TS 物理验证（NEB / dimer / ARTn 对接）、EventLibrary、率表导出 KMCExporter、
  EON 侧 ml_proposal 运行时集成（对应开发计划 Stage 5–7）

# 三条路径（箭头必须按此连接，不得再加别的通路）
A 分子 R→P 监督训练：
   load_events + open_dataset → EventFlowDataset（pos = (1-t)·source + t·P，
   target_velocity = P − source，MIC 提升；target_pos = P 仅作监督）
   → PaiNN 骨干（reference = R，current = x_t，可选边条件 bond_change）
   → flow_loss（velocity，mask = movable_mask） → train_product_flow → 采样 → 评测
B 分子 R+P→TS 监督训练：
   TransitionStateDataset（source = 中点桥 / R-P 插值，endpoint_pos = P 作为条件输入，
   target_pos = TS） → 同一 PaiNN 骨干（可选 endpoint_equivariant）
   → 同一 flow_loss → run_transition_state_trials
C 盆地级无目标提案：
   BasinDataset（零 target_* 字段）+ EventSeed（ZeroInit / GaussianInit）
   → CandidateSampler + sample_ode → Candidate + 生成结构
   → basin_recall（聚类 / 最近已知产物匹配）或 semantic_flow trials →（可选）relax

# 不变量（必须画进图里，六条都要有）
1. movable_mask 是唯一移动语义来源；固定原子 = ~movable_mask（损失、采样、模型输出三处都用它）
2. active_prior 是模型输入条件，不是监督
3. 唯一监督项是 velocity；active/direction 输出头与 target_active_mask / target_direction 已删除
4. 推理视图 BasinDataset 不含任何 target_*；bond_change_source="oracle" 仅诊断用
5. 无弛豫 RMSD / 召回是几何候选召回，不等于鞍点验证的 KMC 事件召回
6. 全链路 PBC：cell 与 cell_offsets 必须成对使用

# 禁止（出现任何一条即视为不合格，请重画）
- 不得出现已删除的模块或接口：records.py、catalog.py、pyg.py、raw_events.py、partitions.py、
  adaptive_ode.py、graph_update_interval、每 k 步重建图、target_active_mask、target_direction、
  任何 active/direction 损失项
- 不得把鞍点验证、事件库、KMC 导出、EON 集成画成已实现（只能灰底虚线 + 标注"未实现"）
- 不得把 ProductInit（oracle）画进盆地推理路径（只能作为虚线诊断支路）
- 不得让任何 target_* 字段出现在 BasinDataset 一侧
- 不得声称几何召回 = 鞍点验证召回；不得声称 EAM 弛豫 = 事件验证

# 输出前自检（不满足就自己改，不要输出解释）
1. 每个节点都在《内容清单》里吗？有没有凭印象补的模块？
2. 三条路径是否都能从入口一路追到产物？
3. 推理路径（C）上是否存在任何 target_*？必须为"否"。
4. 未实现阶段是否都是灰底 + 虚线？
5. 六条不变量是否都在图上？
6. 代码标识符是否全部保留英文原样？
```

**可选替换项**：把 `flowchart TB` 改成 `flowchart LR`（更适合宽屏）；把 `≤ 45 节点` 收紧到 `≤ 35`
（如果只想要一页纸的主干图）；标签语言改为全英文（若要投稿英文期刊）。

---

## 3. 变体 B：图像生成模型（Midjourney / GPT-Image / Nano Banana）

> 图像模型**写不对中文、也写不对模块名**。这张图的用途是"组会/海报上的视觉示意"，
> 精确结构请用 §2 或 §4。提示词故意限制文字数量。

```text
A clean, flat vector architecture diagram for a scientific machine-learning project,
white background, 16:9, no 3D, no photorealism, no logos, no icons of people or robots.

Layout: five horizontal swim-lane bands stacked top to bottom, each a light pastel
rounded rectangle with a thin darker outline and one short English title on the left:
1) ENTRY & CONFIG   2) DATA LAYER   3) GEOMETRY + SEEDS   4) MODELS + TRAINING
5) SAMPLING, EVALUATION, RELAXATION

Inside each band, 3-5 small white rounded boxes with very short labels (one to three
words, English only, large readable sans-serif): "CLI", "Config INI", "Event Catalog",
"PyG Views", "Seeds", "PBC Graph", "PaiNN Backbone", "Flow Loss", "Trainer",
"Candidate Sampler", "ODE (dopri5)", "Recall & Clustering", "EAM Relaxation".

Arrows: solid thin dark-grey arrows flow downward between bands. Three accent colours
mark three parallel paths: blue for "R -> P", orange for "R + P -> TS", green for
"basin -> candidates". At the very bottom, one dashed grey box labelled "not implemented"
containing: "saddle validation", "event library", "KMC export".

Style: muted academic palette (slate, teal, sand, muted blue), consistent 2 px stroke,
generous whitespace, aligned grid, orthogonal arrows only, no crossing lines,
no decorative gradients, publication-quality flat design.

Text constraint: do NOT add any text other than the titles listed above;
never render Chinese characters; keep every label on a single line.
```

---

## 4. 变体 C：论文级矢量图（draw.io / TikZ）

把下面这段接到 §2 的《内容清单》《三条路径》《不变量》《禁止》之后：

```text
# 追加交付契约（替换 §2 的 Mermaid 部分）
- 不要输出 Mermaid。改为输出 [二选一，我指定：draw.io XML / TikZ（pdflatex 可直接编译）]。
- 布局：五个横向分层带（ENTRY & CONFIG / DATA / GEOMETRY+SEEDS / MODELS+TRAINING /
  SAMPLING+EVAL+RELAXATION），带宽自适应内容；层内节点统一高度、水平对齐到网格。
- 用带标签的圆角矩形表示模块；三色细边框区分 A/B/C 三条路径；同一模块被多条路径共用时
  只画一次，用边框多色或角标 (A,B) 标注。
- 所有连线走正交折线，尽量减少交叉；必要时用带编号的跳线（jump）而不是新增节点。
- 字号层级：层标题 10 pt 加粗、模块名 8 pt、字段名 7 pt 斜体等宽；黑白打印也能区分
  （用线型 + 灰度，不依赖颜色）。
- 未实现阶段：灰色虚线框 + "not implemented (Stage 5-7)"。
- 图例放右下角：实线=已实现数据流、虚线=诊断或未实现、色带=A/B/C 路径。
- 同时给出图注（caption）文字，≤ 80 字中文，说明这张图画的是"当前实现"而非完整路线图。
```

---

## 5. 事实底稿（供人工校对，已内嵌在 §2）

### 5.1 分层与模块（工作树实测）

| 层 | 包/文件 | 关键符号 |
|---|---|---|
| L0 入口 | `basinflow/cli.py` | `main`；子命令 `train-transition1x`、`train-transition1x-ts`、`benchmark` |
| L0 编排 | `basinflow/workflows/` | `config.read_config`、`dataset.load_configured_dataset`、`transition1x.run_transition1x`、`transition_state.run_transition1x_ts`、`semantic.run_semantic_benchmark` |
| L1 数据 | `basinflow/data/` | `structures.Structure`、`events.{load_events,Event,EventCatalog,event_target,basin_reactant,select_basins,single_event_scope}`、`splits.{BasinSplit,split_basins,save_split,load_split,find_split_manifest,open_dataset}`、`samples.{EventData,EventFlowDataset,TransitionStateDataset,BasinDataset}` |
| L2 几何 | `basinflow/geometry/` | `mic.*`、`radius_graph.{radius_graph,dual_radius_graph}`、`graph.build_neighbor_graph`、`bonds.bond_change_labels` |
| L3 种子 | `basinflow/seeds/` | `records.{EventSeed,SeedContext,seed_type_id}`、`generators.{ZeroInit,GaussianInit,DirectionalInit,ProductInit}` |
| L4 模型 | `basinflow/models/` | `factory.build_model`、`loss.flow_loss`、`painn.{layers,modules,painn}`、`egnn/*`（兼容） |
| L5 训练 | `basinflow/training/product_flow.py` | `train_product_flow` |
| L6 采样 | `basinflow/sampling/` | `candidate_sampler.{CandidateSampler,Candidate}`、`ode.sample_ode` |
| L7 评测 | `basinflow/evaluation/` | `basin_recall.*`、`semantic_flow.run_gaussian_trials`、`single_event_molecule.*`、`transition_state.run_transition_state_trials`、`product_flow_quality.*`（历史） |
| L8 松弛 | `basinflow/relaxation/relax.py` | `relax`、`eam_calculator`、`RelaxationConfig` |
| L9 未实现 | — | 鞍点验证、`EventLibrary`、率表导出、EON `ml_proposal` 集成 |

### 5.2 三条路径的字段级事实

| | A：R→P | B：R+P→TS | C：盆地推理 |
|---|---|---|---|
| 数据集 | `EventFlowDataset` | `TransitionStateDataset` | `BasinDataset` |
| 源分布 | seed 初始化 `x_0` | 中点桥 / R-P 插值 | 同 A（`ZeroInit` / `GaussianInit`） |
| 条件输入 | R、`movable_mask`、`active_prior`、可选 `bond_change` | 额外 `endpoint_pos = P` | 仅反应物侧，零 `target_*` |
| 监督 | `target_velocity`（`target_pos` 仅作参考量） | `target_velocity`（`target_pos` = TS） | 无 |
| 损失 | `flow_loss`（velocity，mask = `movable_mask`） | 同一 `flow_loss` | 无（只采样） |
| 积分 | `sample_ode`（dopri5） | 同 | 同 |
| 评测 | `single_event_molecule` / `basin_recall` | `transition_state`（midpoint bridge） | `basin_recall` / `semantic_flow` |

### 5.3 六条不变量

1. `movable_mask` 是唯一移动语义；固定原子 = `~movable_mask`。
2. `active_prior` 是输入条件，不是监督。
3. 唯一监督项是 velocity；active/direction 头已删除，`target_active_mask` / `target_direction` 不再产出。
4. `BasinDataset` 不构造任何 `target_*`；`bond_change_source="oracle"` 仅诊断，盆地级结论不得使用。
5. 无弛豫召回是**几何候选召回**，不是鞍点验证过的 KMC 事件召回。
6. 全链路 PBC：`cell` 与 `cell_offsets` 必须成对使用。

---

## 6. 禁止事项：这类图最常被编造的东西

这些名字在 `docs/02_architecture.md`、`docs/05_development_plan.md`、`docs/24_source_walkthrough.md`
里**仍然存在**（它们是设计目标或历史记录），但**当前代码里没有**或已删除。
生成方如果读过这些文档，几乎一定会画错：

| 会被误画的 | 真实状态 | 证据 |
|---|---|---|
| `SaddleValidator`、`EventLibrary`、`KMCExporter` | 未实现，Stage 5–7 目标 | `grep -rl` 在 `basinflow/` 零命中 |
| `ml_proposal`（EON 侧集成） | 未实现 | 同上 |
| `records.py` / `catalog.py` / `pyg.py` / `raw_events.py` / `partitions.py` | 已删除，拆成 `structures/events/splits/samples.py` | `docs/27` §12.2 |
| `CandidateRecord`、`BasinRecord` | 已删除；现为 `sampling.Candidate`（5 字段） | `docs/27` §12.2 |
| `adaptive_ode.py`、手写 Dormand-Prince | 已删除；统一到 `sampling/ode.py` 的 torchdiffeq `dopri5` | `docs/27` §13 |
| `graph_update_interval`、"每 k 步重建图" | 已删除；每次场求值都重建图 | 源码零命中（仅旧 `.pyc`） |
| `target_active_mask` / `target_direction` / active / direction 损失项 | 已删除；只剩 velocity | `docs/27` §12.2、`models/loss.py` |
| `active_threshold`、`active_weight`、`direction_weight` 配置键 | 已删除；未知键会被 `read_config` 拒绝 | `docs/27` §5 |
| 把 `ProductInit` 画成推理路径 | 它是 oracle 诊断 | `docs/27` §12.3 |
| 把 EGNN 画成主路径 | 全部配置为 `backend = painn`；EGNN 是兼容/对比 | `docs/21` §四 |
| 把"无弛豫 RMSD 召回"写成"事件召回/验证通过" | 口径不同，不可等同 | `README.md`、`docs/27` §12.5 |

---

## 7. 迭代建议

1. **先定内容再看排版**：把 §2 的输出贴进 mermaid.live，检查箭头是否符合 §5.2 的三条路径表；
   内容有错就地改提示词，不要手改代码块 —— 否则下次重生成又会错。
2. **拆两张图**：模块结构图（本文件）与"一次真实运行的调用时序"是两件事。
   时序图建议直接用 `docs/24_source_walkthrough.md` §2 的调用链，配 `sequenceDiagram`。
3. **版本锚点**：图注里写上"对应工作树 2026-09-15（数据层重写 + dopri5 之后）"，
   否则下次数据层再动，这张图会静默过期。
