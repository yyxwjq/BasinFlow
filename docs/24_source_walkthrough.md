# BasinFlow 源码走读清单（逐文件夹 / 逐函数）

本文件是一份**可勾选的走读路线图**，不是设计文档。目标：用最短路径把 `basinflow/`
的每个模块、每个关键函数讲到，并让每一步都可验证、可回退。

- 建立时间：2026-09-13
- 走读对象：`basinflow/`（**44 个 `.py` / 6272 行**，其中 9 个 `__init__.py` 共 156 行）、
  `examples/`（5 个入口）、`tests/`（33 个文件，**277 个测试**，collect 约 6.5 s）
- 实测命令：`conda run -n ifdiff python -m pytest --collect-only -q` → `277 tests collected`

**为什么按数据流分层读、而不按目录树顺序读**：目录树把 `geometry/` 排在最前，但真正
决定"这段代码在干什么"的是运行时调用链。`evaluation/semantic_flow.py` 有 622 行，其中
只有 184 行（`run_gaussian_trials`，L439–622）是逻辑，其余是 matplotlib 出图；先读它会
浪费一整晚而理解增量接近零。下面的层级顺序即真实依赖顺序，**无环**。

---

## 0. 走读前置（约 20 分钟）

- [ ] 读 `docs/21_core_api.md`：两个真实工作流 + 每模块「核心 / 历史 / 兼容」评级
- [ ] 读 `docs/02_architecture.md`：分层与接口约定
- [ ] 读 `docs/03_data_schema.md`：`EventRecord` / `BasinRecord` / seed 字段契约
- [ ] 读 `docs/19_source_layout_and_interfaces.md`：接口面
- [ ] 跑通基线，建立"我能验证"的信心：

```bash
conda run -n ifdiff python -m pytest -q                # 期望 277 passed
conda run -n ifdiff python -m pytest --collect-only -q | tail -3
```

> ⚠️ 仓库当前处于**未提交的大规模迁移中间态**（见 §7）。在新环境复现前先读 §7。

---

## 1. 依赖图（自动提取，可复查）

重新生成依赖图的脚本（复制即可跑）：

```python
import re, pathlib, collections
root = pathlib.Path('basinflow'); mods = {}
for p in root.rglob('*.py'):
    m = str(p.with_suffix('')).replace('/', '.')
    if m.endswith('.__init__'): m = m[:-9]
    mods[m] = p
edges = collections.defaultdict(set)
for m, p in mods.items():
    for r in re.findall(r'^\s*(?:from|import)\s+(basinflow[\w\.]*)', p.read_text(encoding='utf-8'), re.M):
        t = r
        while t not in mods and '.' in t: t = t.rsplit('.', 1)[0]
        if t in mods and t != m: edges[m].add(t)
for m in sorted(mods):
    print(f"{m:50s} -> {', '.join(sorted(d.replace('basinflow.','') for d in edges.get(m, ()))) or '(leaf)'}")
```

层级结论：

```text
L1 geometry  mic / bonds / radius_graph  ——叶子，无内部依赖
             graph ——只依赖 data.records（理由见 §8.1）
L2 data      records ← mic ;  catalog ← records + raw_events + bonds + mic
             pyg ← catalog + seeds ;  partitions ← catalog
L2' seeds    records ← data.records ;  generators ← seeds.records
L3 models    painn/modules ← geometry.radius_graph ;  painn/layers ← geometry.bonds
             loss(叶子) ;  factory(叶子)
L4 training  product_flow ← data.pyg + models
L4' sampling candidate_sampler ← data.pyg + geometry.{graph,mic,bonds}
L5 evaluation  basin_recall ← mic ；其余 4 个 evaluation ← basin_recall
L6 workflows  transition1x / transition_state / semantic ← 以上全部 + config
L7 cli        3 个子命令，纯转发
```

被依赖最多的"基础设施"（改它们之前先想清楚）：

```text
data.records   ← 12 个模块（几乎是全局契约）
data.catalog   ← 11 个模块
seeds.records  ← data.pyg / seeds.generators
geometry.mic   ← data.catalog / data.records / evaluation.basin_recall / evaluation.semantic_flow / sampling.candidate_sampler
geometry.bonds ← data.catalog / data.pyg / models.painn.{layers,painn} / sampling.candidate_sampler
```

---

## 2. 第一遍 · 跟一次真实调用走（约 2 小时）

只跟**工作流 B（Au/Pt basin 级提案）**，因为它是项目主张的那条链，且不依赖外部大数据。

```bash
python examples/benchmark_semantic_flow.py --config configs/0910/smoke_au_stable.ini --system au
```

> ⚠️ **实测注意**：`smoke_au_stable.ini` 的 `output_dir` 指向
> `/Users/wx/Desktop/benchmark/0910/smoke/au_stable`，该目录已有旧运行产物，直接跑会在
> `workflows/semantic.py:73` 抛 `FileExistsError: output contains a prior run`——**框架不允许
> 覆盖既有 run 目录**（`docs/21` 的约定，训练侧同样如此）。走读前把配置复制一份、只改
> `output_dir` 到一个新路径即可；其余字段保持不动（`max_steps = 2`、`trials_per_test_basin = 1`、
> `[relaxation] enabled = false`，因此这一步只需几十秒）。
> 输入路径已核对存在：`/Users/wx/Desktop/benchmark/au/events` 与该 split manifest。

调用顺序（括号为正文行数）：

- [ ] `examples/benchmark_semantic_flow.py`（45；**已确认是薄壳**，`main` L20–30 只解析
      `--config/--system`，立即转发 `run_semantic_benchmark`）—— 问题：CLI 参数如何变成 config
- [ ] `basinflow/workflows/semantic.py:run_semantic_benchmark`（L62）—— 问题：一次 benchmark 的
      全流程有哪几个产物文件、写在哪。**已实测的产物清单**（2026-09-13 用改过 `output_dir` 的
      smoke 配置真跑一次，`wall_time_seconds ≈ 0.51`，13 basins / 24 events / 9-2-2 划分）：

      ```text
      <output_dir>/checkpoint.pt              config.resolved.ini      data_sources.json
                    split_manifest.json       training.log             history.json   (semantic.py:130)
                    eval_metrics.json         (semantic.py:200)        loss_by_epoch.csv
                    loss_curves.png
      <output_dir>/sampling/trial_metrics.csv  progress.log  summary.json
      <output_dir>/sampling/plots/{<basin>_rmsd_100_trials.png, eam_rmsd_before_after.png, rmsd_by_basin.png}
      <output_dir>/sampling/representatives/<basin>.{json,traj}
      <output_dir>/sampling/generated_before_eam.traj   generated_after_eam.traj
      ```

      ⚠️ 两点实测更正（**不要照抄 `docs/21`/`README` 的产物清单**）：出图目录是
      `sampling/plots/`（`semantic_flow.py:463` 定义、L264 写入），**不在 output_dir 根**；
      `history.json` / `eval_metrics.json` 由 workflow 层写（`semantic.py:130/200`），
      不属于 evaluation 层。另：`[relaxation] enabled = false` 时 `summary.json` 里
      `eam_relaxed_rmsd_angstrom.count = 0`、`improvement_fraction = null`，但 EAM 相关的图
      仍会生成（空内容）——读到这些空字段别误判为 bug。
- [ ] `basinflow/data/partitions.py:load_data_partitions`（L31）+ `_namespaced`（L11）——
      问题：`train/val/test_events_dir` 三种声明方式如何归一成同一结构
- [ ] `basinflow/data/catalog.py:EventCatalog`（L149）—— 问题：catalog 索引了哪些 id，
      `subset()` 为什么是采样入口的一部分
- [ ] `basinflow/data/pyg.py:BasinDataset`（L351）—— 问题：为什么推理视图里**不可能**出现 target
      （对照 `EventFlowDataset` L95，看哪些字段被删掉了）
- [ ] `basinflow/seeds/generators.py`（L28/46/85/131）—— 问题：一个 `EventSeed` 到底带哪些
      几何、标量、掩码字段
- [ ] `models/painn/painn.py:DualPaiNN.forward`（L51–98）——
      问题：`reference / current / endpoint` 三条几何分别来自哪个 batch 字段
- [ ] `basinflow/sampling/candidate_sampler.py:CandidateSampler.sample`（L93–213）——
      问题：Euler 循环每一步更新了哪些 batch 字段，为什么每步要重建图
- [ ] `basinflow/evaluation/semantic_flow.py:run_gaussian_trials`（L439–622，184 行）——
      问题：候选怎么变成 recall 数字，relaxation 在链路里的位置

**形状追踪练习（第一遍结束前必须做一次）**：

```text
batch(pos, reactant_pos, endpoint_pos, edge_index, cell, cell_offsets, batch)
  → painn/modules.py:prepare_flow_input   (L14)  → positions_1 / positions_2 / node_conditions / endpoint_pos
  → painn/painn.py:DualPaiNN.forward             → scalar[N,F] / vector[N,F,3]
  → painn/layers.py:DualMessageBlock.forward (L142) → filters[E,4F] → vector_message[E,F,3]
  → painn/layers.py:GatedEquivariantBlock    (L209) → velocity[N,3]
```

（其中 `positions_1 = reference = reactant`，`positions_2 = current = x_t`，见 §3 B 层卡片。）

---

## 3. 第二遍 · 函数级精读（约 4–6 小时）

卡片格式：**契约（输入→输出）／关键实现（函数 + 行号）／自查证据／坑**。
行号由脚本机械提取（`grep -nE '^(class|def) '`），可随时复查。

### A 层 · 几何与约定（必须逐行；后面全建在它上面）

#### A1 `geometry/mic.py`（110 行，"核心"）
- 契约：`StructureRecord` 级纯 numpy 位移工具，不碰 torch
- 关键实现：`normalize_cell` L8 / `normalize_pbc` L24 / `minimum_image_displacement` L36 /
  `pairwise_displacements` L58 / `displacement_norms` L76 / `derive_active_atoms` L84 /
  `derive_event_direction` L92
- 自查：`tests/test_mic.py`（4 个测试）
- [ ] `derive_active_atoms` 的阈值语义（谁能进 active 集）
- [ ] `derive_event_direction` 的方向定义与 `painn.py:_bond_change_direction`（L101）的关系

#### A2 `geometry/bonds.py`（112 行，"核心"）
- 契约：`(positions, atomic_numbers, cell, pbc)` → 邻接矩阵 / 变化标签
- 关键实现：`bond_adjacency` L29 / `fragment_count` L71 / `preserves_fragments` L92 /
  `bond_change_labels` L103
- 自查：`tests/test_bond_change_condition.py`（18 个测试，本仓库最厚的行为测试之一）
- [ ] **`bond_change_labels` 是 R→P 任务里唯一的产物信息来源**（离散拓扑，不含产物坐标）
- [ ] `preserves_fragments` 在哪个环节把候选判为无效

#### A3 `geometry/radius_graph.py`（90 行，"核心"）
- 契约：纯 torch 张量级 radius graph，供 PaiNN 每步重建
- 关键实现：`radius_graph` L20 / `edge_geometry` L78 / `dual_radius_graph` L85
- 自查：`tests/test_tensor_neighbors.py`（5 个测试）
- [ ] `dual_radius_graph` 的**两图取并集**策略：为什么必须并集，漏掉会怎样
- [ ] `edge_geometry` 产出的 `cell_offsets` 与 `prepare_flow_input` 里 einsum 的对应关系

#### A4 `geometry/graph.py`（222 行，"核心"但一半是兼容路径）
- 契约：ASE 版建图（PBC 正确性基准）+ torch 适配器
- 关键实现：`build_neighbor_graph` L28 / `torch_neighbor_graph_from_structure` L101 /
  `torch_neighbor_graph_from_batch` L151
- 自查：`tests/test_torch_graph.py`（3）+ `tests/test_neighborlist.py`（4）+
  `tests/test_geometry_invariance.py`（4）
- [ ] `torch_neighbor_graph_from_batch` **只服务 EGNN 与 legacy 采样入口**（PaiNN 用自己的
      `painn/modules.py` 路径）——读到它时先确认调用者是谁
- [ ] 这个文件依赖 `data.records`，所以严格说它不是 L1 叶子（见 §8.1）

### B 层 · 数据契约（37 个测试压在这里，改它最容易坏）

#### B1 `data/records.py`（288 行，"核心"，**被 12 个模块依赖**）
- 关键实现：`StructureRecord` L16 / `EventRecord` L204 / `BasinRecord` L238 / `CandidateRecord` L254
- 自查：`tests/test_records.py`（5 个测试）+ 被 19 个测试文件直接 import（最高）
- [ ] 逐个字段抄一遍 `StructureRecord`（positions / atomic_numbers / cell / pbc /
      movable_mask / …），确认哪些字段是**不可缺省**的
- [ ] **`movable_mask` 是规范**：固定原子一律表示为 `~movable_mask`
- [ ] 校验逻辑（形状、cell、frame count）写在哪个方法里

#### B2 `data/catalog.py`（307 行，"核心"）
- 关键实现：`EventTarget` L20 / `BasinSplit` L34 / `EventCatalog` L149
- 自查：`tests/test_event_catalog.py`（19 个测试）
- [ ] `EventTarget` 的 `active_mask` 是怎么算出来的、`active_threshold` 在哪里生效
- [ ] `catalog.event_target(...)` 与 `catalog.structures[...]` / `catalog.basins[...]` 三种索引的区别
- [ ] `BasinSplit.select(catalog, "train"/"val"/"test")` 返回什么类型的对象

#### B3 `data/pyg.py`（411 行，"核心"，风险最高）
- 关键实现：`EventData` L41 / `_BaseEventDataset` L56 / `EventFlowDataset` L95 /
  `TransitionStateDataset` L225 / `_mask_unmovable_pairs` L336 / `BasinDataset` L351
- 自查：`tests/test_pyg_datasets.py`（10）+ `tests/test_transition_state_task.py`（7）+
  `tests/test_bond_change_condition.py`（18）
- [ ] **三个数据集视图的字段差异表**（自己画一张：监督字段 / 推理禁用字段 / 必需字段）
- [ ] `EventFlowDataset` 里 `reference = reactant_pos`、`source_pos` 由 seed 决定——这是 §8.2 坑的源头
- [ ] `BasinDataset` 为什么禁止 target 字段进入（`docs/21` §一「要点」）

#### B4 `data/partitions.py`（101 行，"核心"）
- 关键实现：`_namespaced` L11 / `load_data_partitions` L31
- 自查：`tests/test_data_partitions.py`（11）+ `tests/test_explicit_transition1x.py`（14）
- [ ] 三种声明方式的优先级：显式 `train/val/test_events_dir` > `events_dir + [split] manifest` >
      `events_dir + 比例`
- [ ] 分区不相交校验在哪一行（`docs/17` 提到的 basin/reaction 身份校验）

#### B5 `data/raw_events.py`（200 行，"核心"，Au/Pt 输入）
- 关键实现：`LoadedEvent` L16 / `read_event_file` L33 / `_read_basin_table` L102 / `load_eon_catalog` L123
- 自查：`tests/test_eon2data.py`（3）
- [ ] EON `.con` 目录 → catalog 的字段映射，`FixAtoms` 如何变成 `movable_mask`

#### B6 `seeds/records.py`（148 行，"核心"，**覆盖最薄的契约**）
- 关键实现：`EventSeed` L27 / `SeedContext` L98 / `_directions_from_displacement` L110 /
  `event_seed_from_context` L118
- 自查：`tests/test_event_seed.py`（4）
- [ ] `EventSeed` 的标量条件 / 向量条件 / 初始几何三部分分别是哪些字段（对应
      `AGENTS.md` 的 "scalar conditions + equivariant conditions + flow initial geometry"）
- [ ] `ProductInit` 产出的 seed 被文档标记为**诊断专用**，代码里靠什么保证它不进推理路径

#### B7 `seeds/generators.py`（150 行）+ `seeds/from_config.py`（32 行）
- 关键实现：`ZeroInit` L28 / `GaussianInit` L46 / `DirectionalInit` L85 / `ProductInit` L131 /
  `init_generators`（from_config L11）
- 自查：`tests/test_init_generators.py`（7）
- [ ] 四种 init 的哲学差异：谁是主路径、谁是诊断、谁是 legacy（对照 `docs/21` §二）
- [ ] `_rng_for_item`（L19）如何保证同一 `event_id + seed_id` 可复现

### C 层 · 模型（形状最多、坑最密）

#### C1 `models/painn/layers.py`（234 行，"核心"）
- 关键实现：`GaussianFourierBasis` L11 / `BesselBasis` L23 / `GaussianBasis` L36 /
  `PolynomialEnvelope` L48 / `CosineEnvelope` L66 / `RadialBasis` L76 / `_bound_vector` L90 /
  `DualMessageBlock` L102（forward L142）/ `UpdateBlock` L178 / `GatedEquivariantBlock` L209
- 自查：`tests/test_painn_product_flow.py`（24）+ `tests/test_bond_change_condition.py`（18）
- [ ] `RadialBasis.__call__` 返回 **(radial, cutoff)** 两件套，`cutoff` 是包络
- [ ] `DualMessageBlock.forward` 的 4 路 `chunk`：`scalar_message / vector_gate /
      reference_gate / current_gate` 各管什么
- [ ] 三条几何如何合成 `filters`（reference + current + endpoint），以及 `endpoint_equivariant`
      开关（L162–166）加在哪一步
- [ ] `stability_mode` 的 `scaled` vs `bounded` 差异（`vector_cap` / `_bound_vector`）

#### C2 `models/painn/modules.py`（89 行，"核心"）
- 关键实现：`prepare_flow_input` L14 / `PaiNN` L68
- 自查：间接（`tests/test_painn_product_flow.py` 走 `PaiNN`）
- [ ] `dual_radius_graph` 调用点、`shifts` 的 einsum 构造
- [ ] `endpoint_pos` 是可选字段（L60–64），缺失时报错点在哪
- [ ] `node_conditions = [movable, active_prior]`（L52）—— **`active_prior` 是输入，
      `target_active_mask` 是监督，两者不许混**

#### C3 `models/painn/painn.py`（133 行，"核心"）
- 关键实现：`DualPaiNN` L11 / `_bond_change_direction` L101 / `_bond_change_one_hot` L115
- 自查：`tests/test_bond_change_condition.py`、`tests/test_transition_state_task.py`
- [ ] 三条几何（reference/current/endpoint）各自的 `radial/unit` 计算（forward L51–98）
- [ ] `_bond_change_direction`：方向场**只由反应物几何 + 离散标签导出**，产物坐标不进网络
- [ ] `vector = self.vector_input((current - reference)[:, :, None])`：t=0 时为何恒为 0，
      以及 `condition_vector_input` 如何补上方向

#### C4 `models/loss.py`（102 行，"核心"，无直接测试）
- 关键实现：`FlowLossWeights` L31 / `flow_loss` L39
- 自查：`tests/test_painn_product_flow.py` 间接覆盖
- [ ] 四项权重 `velocity / active / direction / active_pos_weight` 各自监督什么
- [ ] `norm ∈ {mse, l1}` 的切换点

#### C5 `models/factory.py`（67 行，"核心"）
- 关键实现：`model_spec` L16 / `build_model` L48 / `model_cutoff` L54 / `load_model_checkpoint` L58
- 自查：`tests/test_model_factory.py`（5）
- [ ] 配置 flag → 构造参数的映射（新增开关时必改这里）
- [ ] `backend ∈ {painn, egnn}` 的分支

### D 层 · 三种使用方式（训练 / 采样 / 评测）—— 框架的分水岭

#### D1 `training/product_flow.py`（479 行，"核心"）
- 关键实现：`active_pos_weight` L26 / `train_product_flow_epoch` L87 /
  `evaluate_product_flow_velocity` L166 / `train_product_flow` L221 /
  `total_scheduler_steps` L438 / `build_scheduler` L447
- 自查：`tests/test_training_product_flow.py`（31）
- [ ] `_dataset_signature`（L67）为什么存在：它在 L293 被消费，决定了"换了数据集会不会被静默复用"
- [ ] epoch 检查点 / `resume_from` 的语义（完整状态 vs 仅权重）
- [ ] `batch_size == "full"` 的含义

#### D2 `sampling/candidate_sampler.py`（213 行，"核心"）
- 关键实现：`CandidateSamplingResult` L26 / `CandidateSampler` L32 /
  `_oracle_bond_change` L63 / **`sample` L93–213**
- 自查：`tests/test_candidate_sampler.py`（10）
- [ ] Euler 循环：每步更新 `batch.pos`、按 `graph_update_interval` 重建图、`t = step / num_steps`
- [ ] 输出如何同时产出 `CandidateRecord` 与 `generated_structures`
- [ ] `bond_change_source="oracle"` 的 docstring 明确写了"diagnostic only"——记住这条纪律

#### D3 `evaluation/basin_recall.py`（205 行，"核心"）
- 关键实现：`CandidateCluster` L14 / `movable_mic_rmsd` L22 / `kabsch_aligned_rmsd` L42 /
  `nearest_product_match` L70 / `cluster_candidates` L100 / `evaluate_basin_recall` L128
- 自查：`tests/test_basin_recall.py`（3）
- [ ] 三个指标定义：MIC-RMSD（不消除旋转）/ Kabsch（消除旋转）/ nearest-match
- [ ] 聚类阈值与"一个 basin 多个正解"的表达方式（`AGENTS.md` 的核心科学主张）

#### D4 `evaluation/transition_state.py`（171 行）
- 关键实现：`run_transition_state_trials` L57
- 自查：**无**（唯一的 TS 评测实现没有专属测试）
- [ ] "midpoint bridge" 协议的实现位置（对应 `report.json` 里的
      `evaluation_protocol` 字段）
- [ ] 指标口径：aligned / clipped(`min(rmsd,1)`) / best-of

#### D5 `evaluation/single_event_molecule.py`（310 行，"核心"）
- 关键实现：`_active_metrics` L37 / `_best_per_basin` L55 / `run_single_event_molecule_trials` L146
- 自查：`tests/test_single_event_molecule.py`（4）
- [ ] 与 D2/D3 的分工：这里是分子侧 trial 编排，跑在 `CandidateSampler` 之上

#### D6 `evaluation/semantic_flow.py`（622 行，**只有 L439–622 是逻辑**）
- 关键实现：`run_gaussian_trials` L439–622（184 行）；`write_loss_artifacts` L185 /
  `write_trial_plots` L233 / `_write_representatives` L314 为出图
- 自查：`tests/test_semantic_flow_benchmark.py`（2）
- [ ] `calculator_factory + RelaxationConfig` 如何解耦（`docs/21` §五）
- [ ] trial 产物文件清单（`trial_metrics.csv` 等）

#### D7 `relaxation/relax.py`（119 行）
- 关键实现：`RelaxationConfig` L25 / `relax` L41 / `eam_calculator` L81 /
  `calculator_factory_of` L93 / `prepare_eam_potential` L100
- 自查：`tests/test_relaxation.py`（4）
- [ ] `relax(structure, calculator, config)` 的 calculator 无关设计：EAM 只是便捷工厂
- [ ] 弛豫结果如何回写候选（以及"弛豫后召回 ≠ 无弛豫几何召回"这条口径）

### E 层 · 编排与入口（薄，约 30 分钟）

- [ ] `workflows/config.py`：`read_config` L9 / `parse_bool` L30 —— `REQUIRED` dict 就是每个
      config 的必填 schema，配合 `getfloat(..., fallback=...)` 读
- [ ] `workflows/transition1x.py`：`_device` L43 / `run_transition1x` L57 —— 分子 R→P 全流程与产物清单
- [ ] `workflows/transition_state.py`：`run_transition1x_ts` L39 —— 与上一个的差异只在数据集与评测。
      **产物名不同，注意**：此路径写 `report.json`（L165），**没有** `history.json` /
      `eval_metrics.json`；且它**不调用** `run_transition1x`，而是直接复用
      `training.train_product_flow`（L114 附近）。实测目录
      `/Users/wx/Desktop/benchmark/0910/official/transition1x_ts_equivariant_aligned/` 印证：
      多出 `final_val/`、`final_test/`、`ts_curve/`、`ts_epoch1/`，缺少那两个 json
- [ ] `workflows/semantic.py`：`run_semantic_benchmark` L62 —— 周期体系全流程
- [ ] `cli.py`：`build_parser` L11 / `main` L36 —— 54 行纯转发，3 个子命令

---

## 4. 明确跳过（第一轮不读）

- [ ] `models/egnn/*`（222 行）：**兼容路径**，`configs/` 里 17 个配置全部 `backend = painn`
- [ ] `evaluation/product_flow_quality.py`（251 行）：历史 EGNN pairwise 评测
- [ ] `evaluation/semantic_flow.py` 的 **L185–437**（253 行）：`write_loss_artifacts`
      (L185) / `write_trial_plots` (L233) / `_write_representatives` (L314) 等纯出图函数
- [ ] `tools/` 中除 `eon_to_events.py` / `transition1x_to_events.py` / `summarize_benchmark.py`
      之外的 12 个 `diagnose_*` / `watch_*`：按需查阅，不进主线
- [ ] `examples/train_product_flow.py`（700 行）与 `examples/sample_product_flow.py`（289 行）：
      历史 EGNN 入口（其余 3 个 example 都是薄壳）

---

## 5. 第三遍 · 用测试反查（约 1 小时）

**测试数量的真相**（实测 `pytest --collect-only` 按文件统计）：

| 测试文件 | 用例数 | 覆盖对象 |
|---|---:|---|
| `test_transition1x.py` | 44 | 数据转换 + `workflows.transition1x` |
| `test_training_product_flow.py` | 31 | `training/product_flow.py` |
| `test_painn_product_flow.py` | 24 | PaiNN 主干 |
| `test_event_catalog.py` | 19 | `data/catalog.py` |
| `test_bond_change_condition.py` | 18 | `geometry/bonds.py` + 条件通道 |
| `test_explicit_transition1x.py` | 14 | `data/partitions.py` |
| 其余 27 个文件 | 127 | 见 `pytest --collect-only -q` |

**直接 import 密度**（`grep -E '^(from|import) basinflow' tests/*.py`）：

```text
basinflow.data.records   19 个测试文件     ← 契约中心
basinflow.data.catalog   14
basinflow.seeds           8
basinflow.data.pyg        4
basinflow.geometry.*      mic 1 / graph 1 / bonds 1（其余靠行为测试）
basinflow.models.painn    ⚠ 无直接 import（全靠 test_painn_product_flow 等行为测试，共 29 例）
basinflow.relaxation      ⚠ 无直接 import（靠 test_relaxation.py，4 例）
basinflow.evaluation.transition_state  ⚠ 零覆盖
```

零/薄覆盖清单（**先读测试再读实现**，读完顺手补一个契约测试）：

- [ ] `evaluation/transition_state.py`（171 行，0 例）
- [ ] `models/painn/*` 无直接 import（用 `tests/test_painn_product_flow.py` 反查）
- [ ] `models/loss.py` 无直接 import
- [ ] `relaxation/relax.py` 无直接 import（用 `tests/test_relaxation.py` 反查）
- [ ] `seeds/records.py` 仅 4 例 / 148 行
- [ ] `geometry/mic.py` 仅 4 例 / 110 行

---

## 6. 不许改坏的契约（走读时对照检查）

1. `movable_mask` 是规范，固定原子 = `~movable_mask`
2. `active_prior` 是模型输入；`target_active_mask` 是监督
3. `bond_change_source ∈ {seed, oracle}`；`oracle` 仅上限诊断，**任何 basin 级结论不得使用**
4. `BasinDataset`（推理）与 `EventFlowDataset`（监督）的字段隔离是刻意的
5. 无弛豫召回是**几何候选召回**，不是 saddle-validated event recall
6. `Graph` 全链路 PBC：`cell` + `cell_offsets` 必须一起用，缺一就是对孤立团簇建模

## 7. 仓库当前状态（走读前必读）

`git status` 实测：`24 D / 19 M / 64 ??`——**整个 `basinflow/`、`tests/`、`docs/`、
`tools/`、`examples/` 都是 untracked**，只有旧包 `src/fscgp/**` 的 30 个 `.py` 仍在索引里
且已被删除。`git log` 只有 5 个提交，止于 `6585ed1 feat: add Stage 3 minimal training loop`，
即 HEAD 与工作树不是同一代代码。

后果与注意事项：

- **`git ls-files tests` 列出的 5 个文件在磁盘上不存在**：
  `test_dataset_collate.py`、`test_flow_targets.py`、`test_product_event_flow_interface.py`、
  `test_raw_events.py`、`test_seed_generators.py`。它们属于 `fscgp` 时代的索引快照，
  当前 277 个测试只来自磁盘上实际存在的 33 个文件。
- `tests/`、`tools/`、`examples/` 下还有一批**孤儿 `.pyc`**（无对应 `.py`）：
  `test_no_oracle_flow_contract`、`test_egnn_product_flow`、`test_eam_relaxation`、
  `tools/{debug_painn_training,eon2data,plot_total_loss,prepare_transition1x_benchmark,summarize_0910,transition1x2data}`、
  `examples/{e2e_stage3,smoke_product_flow,rebuild_semantic_flow_output,regenerate_semantic_flow}`。
- **不要用 `grep tests/` 判断覆盖**：`__pycache__` 里的旧字节码会让结论偏乐观。
  用 `pytest --collect-only -q` 才准。
- 走读结论若要在别处复现，先做一次初始提交或打 tag，否则"我读的那版"无法寻址。

## 8. 三个高危点（读到必须停下来确认）

1. **`reference` ≠ 源分布**。`EventFlowDataset` 里 `reference = reactant_pos`，而 `source_pos`
   由 seed 决定（噪声臂 `reactant + σξ`；TS 臂 `(R+P)/2`）。`vector_input(current − reference)`
   在两种参数化下语义完全不同——`docs/22`（875 行）整篇就是讲这个坑。
2. **三个掩码不许混**：`movable_mask` / `active_prior` / `target_active_mask`（见 §6）。
3. **`graph.py` 不是纯 L1**：`geometry/graph.py` import 了 `data.records`，所以几何层里它是
   唯一有内部依赖的文件；PaiNN 主线其实走 `painn/modules.py` + `geometry/radius_graph.py`。

## 9. 进度总表

层级行数由 `wc -l` 对本文件 §3 列出的文件求和得出，**口径核对**：
`5595（A–E 五层）+ 473（§4 跳过）+ 204（3 个未制卡模块 + 全部 `__init__.py`）= 6272`，
与 `find basinflow -name '*.py' | xargs wc -l` 的 6272 完全对齐，可复查：

| 层 | 模块数 | 行数 | 状态 |
|---|---:|---:|---|
| A 几何 | 4 | 534 | ☐ |
| B 数据与种子 | 7 | 1605 | ☐ |
| C 模型 | 5 | 625 | ☐ |
| D 训练/采样/评测 | 7 | 2119 | ☐ |
| E 编排与入口 | 5 | 712 | ☐ |
| 跳过（`egnn/*` + `product_flow_quality`） | 3 | 473 | — |
| 跳过（`semantic_flow` 出图段 L185–437） | — | 253 | — |
| 未制卡（`seeds/from_config.py`、`models/*/__init__.py`） | 3 | 48 | — |
| `__init__.py` 汇总 | 9 | 156 | — |

## 10. 走读记录（边读边填）

每个模块一行，**不许写"看懂了"**，只写三样：契约（输入→输出）／关键函数 + 行号／
我还不懂的问题。

| 模块 | 契约 | 关键实现 | 疑问 |
|---|---|---|---|
| `geometry/mic.py` | | | |
| `geometry/bonds.py` | | | |
| `geometry/radius_graph.py` | | | |
| `geometry/graph.py` | | | |
| `data/records.py` | | | |
| `data/catalog.py` | | | |
| `data/pyg.py` | | | |
| `data/partitions.py` | | | |
| `data/raw_events.py` | | | |
| `seeds/records.py` | | | |
| `seeds/generators.py` | | | |
| `models/painn/layers.py` | | | |
| `models/painn/modules.py` | | | |
| `models/painn/painn.py` | | | |
| `models/loss.py` | | | |
| `models/factory.py` | | | |
| `training/product_flow.py` | | | |
| `sampling/candidate_sampler.py` | | | |
| `evaluation/basin_recall.py` | | | |
| `evaluation/transition_state.py` | | | |
| `evaluation/single_event_molecule.py` | | | |
| `evaluation/semantic_flow.py` | | | |
| `relaxation/relax.py` | | | |
| `workflows/*.py` | | | |
| `cli.py` | | | |
