# 27 · `basinflow/data/` 最小化重写：删除清单 / 保留清单 / 证据

> 状态：**已实施完成**（任务书 §8.1 的分析 + §9 的九步执行均已完成）。
> §1–§8 是开工前的只读分析（保留原文，便于对照实际改动）；
> **§12 是交付说明**：实际删了什么、留下了什么、验收证据、与本文估计的偏差。
> 基线：工作树 `4ed04b3` + 未提交改动（`data/records.py` 一行注释、`data/pyg.py` 的 TS `time_epsilon`/`path`，
> 属本次之外的在研工作，重构不得回退）。
>
> ⚠️ **2026-09-16 更新：本文的 basin 层已被移除。** 本文设计的 `BasinSplit` / `split_manifest.json`
> / `basin_table.csv` / `basins` 索引全部删除，改用事件级 `split.csv`（见
> [`docs/21_core_api.md`](21_core_api.md) 第八节）。保留下来的结论是 §1 的一条证据：
> **7516 行 `basin_table.csv` = 7516 个互异 basin = 7516 个互异事件，即"一个 basin 一个事件"**——
> 正因如此，basin 分组在 transition1x 上从来没有起过作用，去掉它不改变任何数据划分。
> 本文其余关于 basin 数据结构的设计描述仅作历史记录，**不要照它写新代码**。

---

## 0. 结论摘要

当前数据层 **6 文件 / 1416 行**：

| 文件 | 行数 | 处置 |
|---|---|---|
| `__init__.py` | 21 | 重写（最小公共面） |
| `catalog.py` | 307 | 拆解为 `events.py` + `splits.py` |
| `partitions.py` | 101 | **整文件删除** |
| `pyg.py` | 500 | 拆解为 `samples.py` |
| `raw_events.py` | 200 | 并入 `events.py` |
| `records.py` | 287 | 拆解为 `structures.py` + `events.py` |

目标 **5 文件 / 约 535 行**（任务书 §5 目标 450–550 行）。

四条目标路径全部有生产调用者，删除项全部做到"先检索消费者、再删"：
删除 = **18 项**，保留 = **15 项**，调用点改造 = **约 40 处 py + 51 个 ini + 18 个测试文件**。

---

## 1. 必须先纠正的四个事实（与任务书表述不符）

### 1.1 `CandidateRecord` 不是"零写入零读取"

| 事实 | 证据 |
|---|---|
| 有写入 | `sampling/candidate_sampler.py:183-211` 构造，15 个字段全部赋值 |
| 有生产读取 | `evaluation/basin_recall.py:131,135,141,142,165,178` 读 `generated_structure_id` + `candidate_id`；`evaluation/single_event_molecule.py:231,243` 读 `predicted_active_atoms` |
| 有 example 读取 | `examples/sample_product_flow.py:92-103` 读 11 个字段 |
| 真正零读零写 | `relaxed_structure_id`(`records.py:263`)、`cluster_id`(`:264`)、`relaxation_config`(`:265`)、`priority_score`(`:269`)、`uncertainty`(`:270`) |

**裁定**：仍然从数据层删除整类，但删除动作包含两项迁移，不是纯删除：
1. `sampling/` 内部自留一个 3 字段结果结构（`candidate_id` / `generated_structure_id` / `sampling_config`），
   `basin_recall.py` 改读它；
2. `evaluation/single_event_molecule.py` 的 `target_active_atoms` / `predicted_active_atoms` /
   `active_precision` / `active_recall` / `active_f1` 五列按验收 §7.9 删除。

### 1.2 `split_manifest.json` 不在 `events/` 里，而在上一层，且 `test` 为空

```
/Users/wx/Desktop/benchmark/0910/data/transition1x_single_fragment/
  events/                 7516 × event_*.traj + basin_table.csv   （仅此 7517 个文件）
  split_manifest.json     {"seed": 0, "train": 6733, "val": 783, "test": []}
  source_partitions.json
```

- `events/` 内、`data/` 内、`0910/` 根均**没有** `split_manifest.json`（已实测 `find`）。
- manifest 的 7516 个 id 与 `basin_table.csv` 的 7516 个 basin **完全一致、零重叠**；
  `source_partition` 列（train 6733 / val 783）与 manifest 冗余互证。
- **`test` 为空是 0910 的真实情形**，不是假想边界。验收 §7.7 的空分区必须显式处理。
- 因此"单目录 + 清单"的落地形式是：`events_dir=<...>/events`，`manifest=<...>/split_manifest.json`
  （清单默认路径 = `events_dir/../split_manifest.json` 与 `events_dir/split_manifest.json` 两处探测）。

### 1.3 `basin_table.csv` 是"一个盆地一个事件"

7516 行 = 7516 个互异 basin = 7516 个互异 `file`，**max = min = mean = 1.0 事件/盆地**。
→ 0910 数据**无法**在真实数据上覆盖多事件盆地路径；`single_event` 筛选、盆地内多事件的一致性校验
只能由合成 fixture 覆盖（`tests/test_bond_change_condition.py:318`）。

### 1.4 `bond_change_direction` 根本不是数据层字段

任务书 §4.2 设想它是"输入向量条件样本字段"。实际：

- `configs/0910/*.ini` 里 6 个配置写 `bond_change_direction = true`，它是 **`[model]` 段的模型超参**；
- `models/factory.py:30` → `painn.py:19,33-41` 构造 `self.condition_vector_input`，
  由**离散 `bond_change` 标签**在模型内现算等变向量（`painn.py:100-103,110-121`）；
- 全仓**不存在**名为 `bond_change_direction` 的样本字段。

**裁定**：数据层只有离散的 `bond_change` 样本字段；`bond_change_direction` 保持为模型超参名不动
（`models/` 只允许删两个头，不得改名）。因此 §4.2 的"改名以免与已删监督方向混淆"**无对象**——
已删的监督方向是 `EventTarget.direction` / `target_direction` / `output["direction"]`，与它无关。

**另**：`seed_direction`（输入）与 `EventTarget.direction`（已删监督）本来就是两个名字，
满足 §1.4 的消歧要求，无需再改。

---

## 2. 删除清单（含全仓检索结论）

"写入不算消费"。每条给出**唯一的证据行**。

| # | 删除项 | 定义处 | 全仓消费结论 | 调用点处置 |
|---|---|---|---|---|
| 1 | `CandidateRecord` 整类 | `records.py:252-287` | 见 §1.1 | 迁移 3 处（见上） |
| 2 | `EventRecord.atom_mapping` | `records.py:216,227-228` | **零消费**。`tools/prepare_transition1x_split.py:85` 只是 provenance JSON 里的字符串 `"source_order_preserved"`，与字段无关 | 无 |
| 3 | `EventRecord.event_direction` | `records.py:218,231-232` | **零消费**。（`candidate.event_direction` 是另一个类；`derive_event_direction` 是几何函数） | 无 |
| 4 | `EventRecord.validation_status` | `records.py:219` | **零消费**，全仓仅 `raw_events.py:37,91` 传参 | `raw_events.py` 一并删除 |
| 5 | `EventRecord.active_atoms` | `records.py:217,229-230` | 唯一消费者 `catalog.py:239-243`（`event_target` 的 oracle 分支） | `event_target` 只走 MIC 位移；`examples/demo_pipeline.py:101-103` 重写 |
| 6 | `StructureRecord.charge` | `records.py:33` | **零读零写**（`grep -rn "\.charge\b"` 全仓 0 命中） | 无 |
| 7 | `StructureRecord.energy` / `forces` | `records.py:37-38,88-94,131-134,157-166` | 唯一消费者是 `to_ase`/`from_ase` 自身 + `tests/test_records.py:47,48` | 删测试函数；`from_ase` 不再调 `get_potential_energy()`/`get_forces()`——**7516 事件载入少两次全帧遍历，是净加速** |
| 8 | `StructureRecord.tags` | `records.py:36,57-63,118-119,168` | 仅两处复制传递：`candidate_sampler.py:158`、`product_flow_quality.py:48`，无语义读取 | 两处删 `tags=` 实参 |
| 9 | `StructureRecord.constraints` + 遗留入参分支 | `records.py:35,71-87` | 入参只在测试；唯一读取 `examples/demo_pipeline.py:46` | `to_ase` 的 `FixAtoms` 改由 `~movable_mask` 现算（不依赖该字段）；demo 重写 |
| 10 | `LoadedEvent` | `raw_events.py:15-30` | 唯一引用是 `raw_events.py:94-99` 自身；**含测试在内全仓 0 外部引用** | 随 `raw_events.py` 删除 |
| 11 | `read_event_file` | `raw_events.py:33-99` | 唯一调用者 `raw_events.py:169` | 并入 `load_events` |
| 12 | `EventTarget.active_mask` / `direction` | `catalog.py:28,30,239-244` | 读于 `tests/*`；生产侧 `training/product_flow.py:34-35`（随 #14 删除）、`single_event_molecule.py:231,242`（随 §1.1 删除） | 删除 |
| 13 | `event_target(active_threshold=...)` 参数 | `catalog.py:229,240` | `active_threshold` 全仓 42 个 ini + 15 个 py 读取点，全部服务已删的 active 监督 | 见 §5 配置迁移 |
| 14 | 样本字段 `target_active_mask` / `target_direction` | `pyg.py:204-205,220-221` | 仅 `models/loss.py:122,135-136` 与 `examples/demo_pipeline.py:85,89` | loss 随 §6 改造；demo 重写 |
| 15 | `partitions.py` 整文件（101 行） | `partitions.py:1-101`，含 `_namespaced:11-28`、反应级泄漏检查`:62-71`、嵌套 provenance`:72-81` | 9 个调用点；但被下游读取的 provenance 键**只有** `mode`（`semantic.py:79`、`transition1x.py:83`）和 `manifest`（同行） | provenance 塌缩为扁平 dict；9 个调用点改用 `load_events` + `load_split` |
| 16 | `EventCatalog.subset` | `catalog.py:284-307` | 8 个调用点：`candidate_sampler.py:104`、`transition_state.py:157`、`evaluate_checkpoint.py:56,116`、`evaluate_event_chain.py:112`、`diagnose_flow_fit.py:182`、`diagnose_flow_sampling.py:61`、`demo_pipeline.py:125` | → 自由函数 `select_basins(catalog, ids)` |
| 17 | `EventCatalog.single_event_subset` | `catalog.py:261-282` | 2 个生产调用点：`transition1x.py:82`、`transition_state.py:57` | → 自由函数 `single_event_scope(catalog)` |
| 18 | `EventCatalog.from_eon_directory` | `catalog.py:173-177` | 8 个调用点（tools 5、examples 3、tests 2） | → `load_events(events_dir, ...)` |
| 19 | `BasinSplit.create` / `.save` / `.load` / `.validate` / `.select` | `catalog.py:59-145` | `.select` 唯一调用 `tests/test_event_catalog.py:74`；其余 24 处调用点 | → `split_basins()` / `save_split()` / `load_split()`；`.validate` 并入 `select_basins` 的未知 id 报错 |
| 20 | `pyg.py` 视图层 `bond_change` 两处重复填充 | `pyg.py:215-219` 与 `487-499` | 同一逻辑两份实现 | 收敛为 `_bond_change_tensor(...)` |
| 21 | `__init__.__all__` 手工清单 | `__init__.py:9-21` | 导出 `CandidateRecord`/`BasinRecord` 等已不存在的公共面 | 改为与实现同步的最小导出 |

### 附：随调用点消失的死代码（已确认删除）

| 项 | 定义处 | 删除后 |
|---|---|---|
| `FlowLossWeights.active` / `direction` / `active_pos_weight` / `target` | `loss.py:33,34,35,37` | 只剩 `velocity` + `norm`，二者均被读（`loss.py:148`、`:65-66,115`） |
| `flow_loss` 的 active/direction 分支与返回键 | `loss.py:120-145,152-156` | 返回 `{"velocity_loss": ...}`；调用点 `training/product_flow.py:123,148-150` 必须同步 |
| `active_pos_weight()` 自动权重函数 | `training/product_flow.py:26-38` | 无配置使用 `auto` 后即死（见 §5） |
| `derive_active_atoms` / `derive_event_direction` | `geometry/mic.py:84-110` | **已确认删除**；唯一剩余调用者 `tests/test_mic.py:50,51` 一并改写 |
| `StructureRecord.tags`→`to_ase` 的 `SinglePointCalculator` 分支 | `records.py:130-136` | 与 #7 同批 |
| `EventFlowDataset.diagnostic_oracle` 之外的 `active_threshold` 透传 | `pyg.py:103,115,181` | 见 §5 |

---

## 3. 保留清单（谁在读）

写入不算消费；下表每一步都给出**至少一个读取点**。

| 保留项 | 谁在读（file:line） |
|---|---|
| `displacement`、`target_positions` | `pyg.py:195,200`；`candidate_sampler.py:87`；`single_event_molecule.py:207`；`product_flow_quality.py:98,194`；`summarize_benchmark.py:93` |
| `target_velocity` | `pyg.py:214,420`；`models/loss.py:91` |
| `target_pos` | `models/loss.py:89`；`evaluation/transition_state.py:236,237` |
| `source_pos` | `pyg.py:198`；`evaluation/product_flow_quality.py:105`；`evaluation/transition_state.py:191,221,237,238` |
| `endpoint_pos` | `models/painn/modules.py:60-64` → `painn.py:80-83,105` |
| `movable_mask` | `models/loss.py:63,93`；`egnn.py:96,111,123,137`；`modules.py:41,89`；`candidate_sampler.py:146,150,157,167,170,177` |
| `active_prior` | `egnn.py:97,112`；`modules.py:42,52` → `painn.py:88` |
| `seed_displacement` | `egnn.py:94,114,121,134`；`candidate_sampler.py:207` |
| `seed_direction` | `egnn.py:95,115,134` |
| `seed_type_id` | `egnn.py:58,102`（`seed_type_embedding`） |
| `cell`、`pbc` | `geometry/graph.py:171-172`；`modules.py:27-28`；`egnn.py:77-87` |
| `flow_time` | `models/loss.py:73,102`；`egnn.py:105`；`modules.py:33` |
| `num_nodes` | PyG 批处理与 `tools/diagnose_painn_stability.py:37`、`demo_pipeline.py:93` |
| `bond_change`（PaiNN 专用条件） | `modules.py:57-59` → `painn.py:92-94,132`；`candidate_sampler.py:121-122` |
| `z` | `egnn.py:100`；`modules.py:48`；`geometry/graph.py:169` |
| 盆地索引 `dict[basin_id, list[event_id]]` | `basin_recall.py:100,163`；`semantic_flow.py:484`；`transition1x.py:91,103-105`；`candidate_sampler.py:74,110` |
| `single_event` 范围筛选（自由函数） | `transition1x.py:82`；`transition_state.py:57` |
| 种子类型 → id 映射 | `egnn.py:58,102` |

**保留判据的三条不变量落点**：
1. MIC 提升坐标 = `event_target().target_positions`（`catalog.py:256`），速度目标 `x1 - x0` 在 `pyg.py:201`；
2. `movable_mask` 是唯一移动语义来源：`egnn.py:123,137`、`modules.py:89`、`loss.py:93` 都以它为准；
3. 推理视图零 `target_*`：已逐行核对 `candidate_sampler.py` 对 `BasinDataset` 批的 18 处读取，**无一处读 `target_*`**
   （`candidate_sampler.py:87` 读的是 `EventTarget.target_positions`，只在 `bond_change_source == "oracle"`
   且模型带 `bond_change_condition` 时触发，`:70-72` 双重门禁）。

---

## 4. 待确认项裁定（4 项已确认 + 5 项默认建议）

| # | 项 | 裁定 | 依据 |
|---|---|---|---|
| §4.1 | `bond_change` 条件 | **保留**（PaiNN 专用） | 16 个配置启用；`painn.py:92-94` 消费；样本 schema 注释标注来源 |
| §4.2 | `bond_change_direction` | **无需改名**，它不是数据字段 | 见 §1.4 |
| §4.3 | `FlowLossWeights.target` | **删除** | 51 个配置中设该键者 **0 个**；仅 `loss.py:67-68,71,95` 与 `transition_state.py:156,179` 读取。删除后 `report.json` 去掉 `loss_target` 键 |
| §4.4 | 批处理方式 | **保留 PyG `EventData` 子类** | `_GRAPH_FIELDS={cell,pbc,flow_time,seed_type_id}` 必须 `__cat_dim__ → None`；两个后端都依赖 `batch.batch` 索引（`egnn.py:59,72`；`modules.py:24`；`graph.py:167`）与 `DataLoader`/`.to(device)`/`.num_graphs`（`product_flow.py:129,145`）。改显式拼接等于重写批处理 + `batch` 索引。按要求补注释与断言 |
| §4.5 | 采样报告"几何活跃原子" | **不再报告** | 唯一生产读取者 `single_event_molecule.py:231,243` 随 §1.1 删除；连带删除 `derive_active_atoms`/`derive_event_direction` |
| **新 1** | §6 调用点范围 | **扩展到** `geometry/`、`relaxation/`、`seeds/`、`docs/`、`README.md`、`AGENTS.md`、`.hpc/payload/` | 见 §7.1（四处**必须**改，否则 import 断裂） |
| **新 2** | 三目录数据集 | **改 `prepare_transition1x_explicit_split.py` 产出规范布局，重新生成数据** | 见 §7.2 |
| **新 3** | 死几何函数 | **删除** `derive_active_atoms` / `derive_event_direction` | 符合"有没有代码读它"判据 |
| **新 4** | 种子类型 id 单一来源 | **放 `seeds/records.py`**，紧挨 `EventSeed.seed_type`；未知类型 `ValueError` | 概念归属正确；`seeds/` 已纳入改动范围 |

### 新增 1 的证据：§6 清单遗漏的**强制**改动点

| 文件 | 为什么不改就断 |
|---|---|
| `basinflow/geometry/graph.py:25` | `from basinflow.data.records import StructureRecord` —— 类被改名/移位即 `ImportError` |
| `basinflow/geometry/graph.py:187-193` | 直接构造 `StructureRecord(...)` 建临时结构 |
| `basinflow/geometry/graph.py:42,66,67,71,76,77,81` | 读 `to_ase()`/`cell`/`positions`/`structure_id`/`atomic_numbers`/`species`/`n_atoms` |
| `basinflow/relaxation/relax.py:19,42,47` | 同上 import + 类型注解 |
| `basinflow/relaxation/relax.py:59,65-69` | 读 `to_ase()`、`from_ase()`、`structure.structure_id`、`**structure.metadata` |
| `basinflow/seeds/records.py:9,103` | `TYPE_CHECKING` import + `SeedContext.reactant` 注解 |
| `.hpc/payload/config.ini:23-25,77-80` | 三目录 + `active_weight=0.1` + `direction_weight=0.1` + `active_pos_weight=auto`；硬校验后会**拒绝启动** |
| `.hpc/payload/config_ts.ini:22-24,81-83` | 三目录 + 已废键 |
| `docs/` 13 篇 + `README.md:38,136,145,159` + `AGENTS.md:164` | `AGENTS.md:164` 明写 "`target_active_mask` is supervision"，与新契约**直接冲突**，必须同步 |

`data_sources["mode"]` / `["manifest"]`（`semantic.py:79`、`transition1x.py:83`）是仅有的 provenance 读取键，
`data_sources.json` 只是归档产物、无人回读 → 嵌套 provenance 可安全塌缩。

---

## 5. 配置迁移清单（51 个 ini）

统计（脚本实测，非估算）：

| 键 | 出现文件数 | 处置 |
|---|---|---|
| `active_weight` | 49 | 删除 |
| `direction_weight` | 49 | 删除 |
| `active_pos_weight` | 49（其中 3 个值为 `auto`） | 删除；`auto` 语义随 `active_pos_weight()` 函数一并消失 |
| `active_threshold` | 42 | 删除 |
| `active_source` | 14 | 删除 |
| `train/val/test_events_dir` | 29 | 改为单 `events_dir` + `manifest` |
| `velocity_weight` | 49 | **保留**（映射到 `FlowLossWeights.velocity`） |
| `bond_change_condition` | 16 | 保留 |
| `bond_change_direction` | 6 | 保留（模型超参） |
| `pbc_override` | 14 | 保留 |
| `training_scope` | 1 | 保留（映射到 `single_event_scope`） |

迁移后必须新增行为：`workflows/config.py:21-27` 现在只校验"必需键存在"，**不拒绝未知键**。
按 §6 要求补未知键拒绝（在 `read_config` 内加 `allowed` 反向校验），否则旧配置会静默跑成"缺项实验"。

**4 个辅助损失非零、与 velocity-only 不可比的配置**（已在 §1.5 声明）：
`configs/semantic_flow_au.ini:33-34`、`configs/semantic_flow_pt.ini:33-34`、
`configs/product_flow_transition1x_use_ind_40epochs.ini`、`..._40epochs_batch500.ini`。

**已失效配置（数据目录不存在，`ls` 实测）**：
`configs/product_flow_transition1x_use_ind.ini`、`..._40epochs.ini`、`..._40epochs_batch500.ini`
指向 `/Users/wx/Desktop/benchmark/transition1x/train/events_use_ind_centered`，该目录不存在。

---

## 6. §7.2 三目录数据集（新确认项 2）的证据与方案

| 数据集 | 磁盘布局 | 是否能被"单目录 + 清单"表达 |
|---|---|---|
| `transition1x_single_fragment/` | `events/` + 上层 `split_manifest.json` + `source_partitions.json` | ✅ **就是规范形态**，由 `tools/prepare_transition1x_split.py:102-119` 生成（把各分区事件 rename 进**同一个** `events/`，写**一张**合并 `basin_table.csv` 带 `source_partition` 列） |
| `transition1x_aligned_9000/`、`_ts/`、`aligned_single_fragment/`、`unaligned_single_fragment/`、`explicit_9000/` | `<root>/{train,val,test}/`，每个分区**自带** `basin_table.csv`；`<root>/split_manifest.json` 与分区内容逐一相等（实测：train 9000/9000、val 536/536、test 537/537） | ❌ 三个目录的盆地互不相交，一个 `events_dir` 装不下 |

→ 方案：把 `tools/prepare_transition1x_explicit_split.py:128-141` 改成与 `prepare_transition1x_split.py`
相同的输出形态（单一 `events/` + 上层 `split_manifest.json` + `source_partition` 列），重新生成
`aligned_9000` / `_ts`，29 个配置指向新目录。数据层保持**一种**模式。
`basin_table.csv` 里已有 `source_partition` 列（0910 实测表头），说明该列就是为此设计的。

---

## 7. 测试处置（35 文件 / 224 个 `def test_*`）

**没有任何文件是纯 DELETE**：每处失效接口旁都有仍需保留的覆盖。因此按**函数**粒度删。

| 判定 | 文件数 | 文件 |
|---|---|---|
| REWRITE | 18 | `test_basin_recall` `test_bond_change_condition` `test_candidate_sampler` `test_data_partitions` `test_demo_pipeline` `test_event_catalog` `test_event_seed` `test_geometry_invariance` `test_pyg_datasets` `test_records` `test_semantic_flow_benchmark` `test_single_event_molecule` `test_stage3_au_scripts` `test_summarize_0910` `test_training_product_flow` `test_transition_state_task` `test_transition1x` `test_ts_flow_workflow` |
| KEEP | 17 | `test_adaptive_ode` `test_benchmark_provenance` `test_e2e_stage3` `test_entry_point_bootstrap` `test_eon2data` `test_explicit_transition1x` `test_flow_sampling_diagnostic`⚠︎ `test_init_generators` `test_mic` `test_model_factory` `test_neighborlist` `test_painn_product_flow` `test_plot_total_loss` `test_relaxation` `test_short_names`⚠ `test_tensor_neighbors` `test_torch_graph` |

⚠︎ `test_flow_sampling_diagnostic` 的被测对象 `tools/diagnose_flow_sampling.py:61,137,156` 用了 `.subset`/`BasinSplit.load`/`from_eon_directory`。
⚠ `test_short_names.py:38-52` 守卫 `basinflow.data` 命名空间，导出面变化时必须同步。

**整函数删除清单（约 20 个函数 / 约 26 个 collected item）**：
`test_data_partitions.py:33,48(×2),59,72(×2)`；
`test_event_catalog.py:65,90(×8),95,109(×3),116,123,145`；
`test_bond_change_condition.py:318`；`test_training_product_flow.py:204,210,216`；
`test_records.py:38,51`；`test_candidate_sampler.py:210`；
`test_single_event_molecule.py:127(×2)`。

**测试命令**：`python -m pytest`（`pyproject.toml:32-34` 只设 `pythonpath=["."]` + `testpaths=["tests"]`，
无 `addopts`/无 marker/无 CI）。`README.md:126` 指向的 `tests/test_raw_events.py` **已不存在**（顺带修）。

---

## 8. 目标结构与行数预算

```
basinflow/data/
  __init__.py     仅导出最小公共面                              ~20 行
  structures.py   Structure（species/positions/cell/pbc/movable_mask）
                  + from_ase（归一化 + 不变量校验）              ~85 行
  events.py       Event(event_id, basin_id, reactant/product/ts 内联三帧)
                  + load_events(traj + basin_table) + R/P/TS 校验
                  + EventTarget / event_target
                  + single_event_scope + basin_reactant           ~140 行
  splits.py       BasinSplit + split_basins / save_split / load_split
                  + select_basins                                ~90 行
  samples.py      共享样本构造 + 模式 A 视图 + 模式 B 视图
                  + 无目标推理视图 + EventData + 种子类型 id      ~200 行
                                                          合计 ≈ 535 行
```

关键结构决定：
- **内联三帧**：`Event` 直接持有 `reactant`/`product`/`transition_state` 三个 `Structure`，
  删除 `structures: dict[str, StructureRecord]` 注册表、`structure_id` 簿记、`BasinRecord` 类。
  盆地 reactant 由 `basin_reactant(catalog, basin_id)` 现取（loader 已保证同盆地 reactant 一致）。
- **metadata 上移到 `Event`**：实测三个 reader 全在事件级
  （`basin_recall.py:115` 读 `event.metadata["source_file"]`；
  `summarize_benchmark.py:294`、`single_event_molecule.py:203` 读 `event.metadata["dataset_kind"]`）。

---

## 9. 分步执行计划（每步后三条路径至少一条可运行）

| 步 | 动作 | 可运行性检查 |
|---|---|---|
| 1 | 建 `structures.py` + `structures` 侧的 `__init__` 导出；`records.StructureRecord` 暂时别名转发 | `python -c "import basinflow.data"`；`tests/test_records.py` 通过 |
| 2 | 建 `events.py`（`load_events`/`Event`/`event_target`/`single_event_scope`/`select_basins`）；旧名转发 | **路径 1** 可用：0910 全量载入断言（§10.1） |
| 3 | 建 `splits.py`（`split_basins`/`save_split`/`load_split`/`select_basins`） | 读入 0910 自带 manifest，`test=[]` 显式报错/空分区 |
| 4 | 建 `samples.py`（三视图）；删 `pyg.py`、`catalog.py`、`raw_events.py`、`partitions.py`、`records.py` | **路径 1+2+3** 全部经新 API |
| 5 | 改 `models/`（删两头的 6 个张量 + `forward` 返回）+ `loss.py`（只剩 `velocity`/`norm`） | EGNN `output.keys() == {"velocity"}` |
| 6 | 改 `training/`、`sampling/`、`evaluation/`、`geometry/`、`relaxation/`、`seeds/` 调用点 | 单测通过；采样两后端跑通 |
| 7 | 迁移 51 个 ini + `workflows/config.py` 未知键拒绝 | 旧键报错、新键跑通 |
| 8 | 改 `tools/`、`examples/`、`docs/`、`README.md`、`AGENTS.md`、`.hpc/payload/` | 端到端两种模式各一次极短训练 + 采样 |
| 9 | 改写 18 个测试文件、删约 20 个死测试函数 | `python -m pytest` |

---

## 10. 验收对照

| 验收项 | 对应实现 | 可验证断言 |
|---|---|---|
| 1 真实数据 | `load_events` | 事件数 7516 == `ls event_*.traj | wc -l`；每事件恰 3 帧、角色序 `reactant/product/transition_state`；盆地索引 == `basin_table.csv`（7516 basin，1:1）；cell 全零、pbc 全 False、`movable_mask` 全 True（实测 7516/7516 文件满足） |
| 2 模式 A 样本 | `samples.py` | 字段集合/dtype/形状对 §1.2；`target_pos` == P（MIC 提升坐标）；`(seed, epoch, index)` 逐元素复现（`sha256` 派生，禁 `hash`/全局随机） |
| 3 模式 B 样本 | `samples.py` | `endpoint_pos == P`；`target_pos == TS`；`active_prior` 全 False（`pyg.py:403` 已如此） |
| 4 无泄漏 | 推理视图 | 推理样本零 `target_*`；3 帧事件可构造推理样本 |
| 5 损失 | `loss.py` | 无未被读字段；EGNN 只回 `velocity`；两后端都能训 |
| 6 采样 | `candidate_sampler.py:127-137` | 该分支保留（EGNN 无 `builds_own_graph` ⇒ `False`；PaiNN `modules.py:71` ⇒ `True`） |
| 7 划分 | `splits.py` | `save_split`/`load_split` 往返一致；读入 0910 自带 manifest；`test=[]` 显式处理 |
| 8 端到端 | workflows | 两模式各产出 `checkpoint.pt`/`training.log`/`sampling/summary.json` |
| 9 报告口径 | evaluation | `active_*`/`direction_*` 列清零；带 `evaluation_protocol` 标记（`single_event_molecule.py:233,272`、`transition_state.py:252,274` 已有，`basin_recall`/`product_flow_quality`/`semantic_flow` 需补） |
| 10 交付证据 | 本文 | 删除项 → §2 检索结论；保留项 → §3 读者；配置清单 → §5；行数 → §8；无新依赖；无 torch 可导入（`_require_pyg` 惰性分支保留） |

---

## 11. 已知后果（需写进交付说明）

1. **EGNN 旧 checkpoint 无法严格加载**：删除的是 `state_dict` 里的 6 个张量
   （`active_head.weight/bias`、`direction_gate.0.weight/bias`、`direction_gate.2.weight/bias`），
   而 `models/factory.py:71` 是 `strict=True`。对比基线需重训。
2. **4 个配置的历史结果与 velocity-only 不可比**（见 §5 末）。
3. 采样器 `sampling_config` 里的 `"active_atoms_source"`/`"event_direction_source"` 与
   `metadata` 中同名键将消失，改为只报 `final_mic_displacement` 派生量或不再报告（§4.5）。
4. 两个后端的差异**不得顺手统一**：EGNN 单几何图（当前 `pos`，按 `graph_update_interval` 由采样器重建）
   vs PaiNN 双几何并集（模型内每步重建）。对比运行须同 catalog、同 split、同 seed、同 batch size、同时间采样策略。

---

## 12. 交付说明（实施完成后补写）

### 12.1 实际结构

```
basinflow/data/
  __init__.py     56 行   最小公共面（21 个导出，与实现同步）
  structures.py  108 行   Structure（五项 + 校验）、from_ase / to_ase
  events.py      295 行   Event、EventCatalog、load_events、event_target、
                          basin_reactant、select_basins、single_event_scope
  splits.py      180 行   BasinSplit、split_basins/save_split/load_split、
                          find_split_manifest、open_dataset
  samples.py     546 行   模式 A / 模式 B / 无目标推理三视图 + EventData
                 ─────
                 1185 行（代码 837 + 文档字符串 224）
```

**与 §8 估计（≈535 行）的偏差，如实说明**：文件行数 1185，是估计的 2.2 倍。
分解后 837 行是代码、224 行是文档字符串、其余为注释与空行。估计偏低的原因是
§1.3/§1.4 要求把五条不变量与语义澄清写进代码，同时 §7 要求校验帧数/角色/
原子数/species/cell/pbc/盆内一致性，这些校验与契约注释本身就是主体；删掉它们
能把文件压到 550 行，但会丢掉本次任务明确要求保留的东西。实际删除量：
旧的 1416 行 → 1185 行，且其中 828 行的 `pyg.py`+`catalog.py`+`records.py`
三文件职责被重新划分。若仍需 450–550 行，可另开一轮只做注释精简。

### 12.2 删除清单执行结果（逐项均有全仓检索结论）

| 删除项 | 结果 |
|---|---|
| `data/catalog.py` `data/records.py` `data/raw_events.py` `data/partitions.py` `data/pyg.py` | 5 文件删除 |
| `CandidateRecord` 整类 | 删除；`sampling.Candidate`（5 字段）替代，评估只读 `candidate_id`/`generated_structure_id` |
| `EventRecord.atom_mapping` / `event_direction` / `validation_status` / `active_atoms` | 删除（零消费已核实） |
| `StructureRecord.charge` / `energy` / `forces` / `tags` / `constraints` | 删除；`from_ase` 不再读 ASE 计算器（7516 事件载入因此少两轮全帧遍历） |
| `LoadedEvent` / `read_event_file` | 删除，并入 `load_events` |
| `EventTarget.active_mask` / `direction` | 删除；`event_target()` 不再接受 `active_threshold` |
| 样本字段 `target_active_mask` / `target_direction` | 不再产出 |
| `partitions.py` 三目录模式 / `_namespaced` / 反应级泄漏检查 / 嵌套 provenance | 删除；provenance 塌缩为 `{events_dir, manifest, manifest_sha256}` |
| `EventCatalog.subset` / `single_event_subset` / `from_eon_directory` | → `select_basins` / `single_event_scope` / `load_events` |
| `BasinSplit.create/save/load/validate/select` | → `split_basins` / `save_split` / `load_split`；未知 id 与空分区在选择时报错 |
| `FlowLossWeights.active/direction/active_pos_weight/target` 与 `active_loss`/`direction_loss` | 删除；只剩 `velocity` + `norm` |
| `EGNNFlow.active_head` / `direction_gate` 与两个输出键 | 删除；`forward` 只回 `{"velocity"}` |
| `training.active_pos_weight()` | 删除 |
| `geometry.mic.derive_active_atoms` / `derive_event_direction` | 删除（失去全部生产调用者） |
| `"active_logits"` / `"direction"` 模型输出键 | 生产者与消费者全部消失 |

删除**没有**触及：`models/painn/**` 的输入语义与建图方式、`EGNNFlow` 的标量特征
构成（`hidden_dim*2+6`）、向量基 `[vector_state, seed, seed_direction]`、
`seed_type_embedding`、`builds_own_graph` 的两种取值。

### 12.3 保留清单：谁在读

| 保留项 | 消费者 |
|---|---|
| `displacement`、`target_positions` | `event_target` → 两个模式的样本视图、采样 oracle 诊断、评估 RMSD |
| `target_pos`、`target_velocity` | `models/loss.py`、模式 A/B 样本 |
| `endpoint_pos` | `models/painn/{modules,painn}.py`（模式 B 条件） |
| `source_pos` | 样本视图与 TS 采样器积分起点 |
| `movable_mask` | 损失掩码、EGNN 更新掩码、PaiNN 输出掩码、采样器固定原子回位 |
| `active_prior` | EGNN 标量特征 + PaiNN `node_conditions` |
| `seed_displacement` / `seed_direction` / `seed_type_id` | EGNN 标量特征、向量基、`seed_type_embedding` |
| `cell` / `pbc` | `geometry/graph.py`、`geometry/radius_graph.py` 建图 |
| `flow_time` | 损失、EGNN 时间特征、PaiNN 时间嵌入 |
| `bond_change` | PaiNN 边条件（16 个配置启用） |
| 盆地索引 `dict[basin_id, [event_id]]` | 盆地级评估、推理视图、`single_event_scope` |
| `single_event_scope`（自由函数） | `workflows/transition1x.py`、`workflows/transition_state.py` |
| 种子类型→id 映射 | `seeds/records.py` 单一来源；未知类型 `ValueError` |

### 12.4 三条路径的落点

| 路径 | 数据入口 | 视图 | 监督 |
|---|---|---|---|
| A：R → P 训练 | `load_events` + `open_dataset` | `EventFlowDataset` | `target_pos` / `target_velocity`（P 只作监督） |
| B：R + P → TS 训练 | 同上 | `TransitionStateDataset` | `target_pos` / `target_velocity`（P 是 `endpoint_pos` 条件输入） |
| 盆地级推理采样 | 同上 | `BasinDataset` | 无：视图不构造任何 `target_*` |

### 12.5 验收证据（§7）

| 验收 | 结果 |
|---|---|
| 1 真实数据 | 7516 事件 == 7516 文件；每事件 3 帧、角色序正确；盆地索引 == `basin_table.csv`；cell 全零 / pbc 全 False / `movable_mask` 全 True（7516/7516） |
| 2 模式 A | 字段集合 == §1.2 硬契约；`target_pos` == MIC 提升 P；同 `(seed, epoch, index)` 逐元素一致 |
| 3 模式 B | `endpoint_pos` == P；`target_pos` == TS；`active_prior` 全 False |
| 4 无泄漏 | 推理视图零 `target_*`；3 帧事件可构造推理样本 |
| 5 损失 | `FlowLossWeights == {velocity, norm}`；返回 `{velocity_loss}`；EGNN 只回 `velocity`；两后端均可反传 |
| 6 采样 | 两后端跑通；EGNN 走"模型不自建图"分支（`builds_own_graph` 缺失 → 采样器重建），PaiNN 内部重建 |
| 7 划分 | `save_split`/`load_split` 往返一致；直接读入数据目录自带 manifest（6733/783/0）；空 `test` 在选择时显式报错 |
| 8 端到端 | 两种模式各跑一次极短训练 + 采样，产出 `checkpoint.pt`/`training.log`/`history.json`/`config.resolved.ini`/`split_manifest.json`/`data_sources.json`/`sampling/summary.json` |
| 9 报告口径 | `trial_metrics.csv` 与 `summary.json` 无 active/direction 列；带 `evaluation_protocol` 标记（`basin_recall` / `product_flow_quality` / `semantic_flow` 本次补上） |
| 10 交付证据 | 本文 §12.2/§12.3；`wc -l` 1185；`pyproject.toml` 未改（无新依赖）；屏蔽 torch 后 `structures`/`events`/`splits` 可导入并运行 |

测试：`python -m pytest` → **282 passed, 1 skipped**（原 224 个测试函数；
删掉约 20 个固化了已删接口的函数，新增 2 个文件覆盖单数据源与 `single_event` 接线）。

### 12.6 实施中发现并修掉的两个真实缺陷

1. **三个 workflow 提前选择未使用的分区**：即使 `[evaluation] split = val`，代码也先
   `split.select(catalog, "test")`；0910 的 manifest `test: []` 会让全部 5 个
   single_fragment 配置直接报错。改为只选择实际使用的分区。
2. **`workflows/transition1x.py` 仍在调用已删除的 `part.single_event_subset()`**：
   单元测试没有覆盖 `training_scope = single_event` 这条路径，全仓检索才暴露。
   已改为 `single_event_scope`，并补了一个端到端测试。

### 12.7 数据侧一次性操作（仓库外）

`tools/merge_event_partitions.py`（新增）把旧的三目录数据集转成规范布局，
事件文件用硬链接，不占额外空间。已为 5 个数据集执行：

| 新目录 | 事件数 | split (train/val/test) |
|---|---|---|
| `transition1x_aligned_9000_merged` | 10073 | 9000 / 536 / 537 |
| `transition1x_aligned_9000_ts_merged` | 10073 | 9000 / 536 / 537 |
| `transition1x_aligned_single_fragment_merged` | 7516 | 6733 / 392 / 391 |
| `transition1x_unaligned_single_fragment_merged` | 7516 | 6733 / 392 / 391 |
| `transition1x_explicit_9000_merged` | 10073 | 9000 / 536 / 537 |

`tools/prepare_transition1x_explicit_split.py` 已改为直接产出这一布局，
`.hpc/deploy_data.sh` 改为上传 `*_merged`。

### 12.8 已知后果（承接 §11，逐条确认）

1. EGNN 旧 checkpoint 无法严格加载（`state_dict` 少 6 个张量，`factory.py` 是
   `strict=True`），对比基线需重训。
2. 4 个辅助损失非零的配置（`semantic_flow_au.ini`、`semantic_flow_pt.ini`、
   `product_flow_transition1x_use_ind_40epochs{,_batch500}.ini`）的历史结果与
   velocity-only 不可比；其 ini 已迁移为新键集，只保留 `velocity_weight`/`norm`。
3. 3 个 `product_flow_transition1x_use_ind*.ini` 指向的
   `/Users/wx/Desktop/benchmark/transition1x/train/events_use_ind_centered`
   **本来就不存在**，配置已迁移但数据仍需自备。
4. 两个后端的差异未被统一：EGNN 单几何图（采样器按 `graph_update_interval` 重建）
   vs PaiNN 双几何并集（模型内每步重建）。
5. 历史 run 目录里的 `config.resolved.ini` 仍带已删键，新的 `read_config` 会拒绝它
   —— 这是"不许静默跑成缺项实验"的预期行为；重跑需用迁移后的配置。

---

## 13. 追加：积分策略统一到 MolGEN（2026-09-15，数据层交付之后）

### 13.1 问题

数据层交付时，两条路径的积分策略不一致：

| | MolGEN | 我们（改前） |
|---|---|---|
| 默认积分器 | `dopri5` 自适应（`mdgen/parsing.py:78`、`mdgen/transport/integrators.py:77`、`mdgen/equivariant_transport.py` 调用处 `sampler.sample_ode(...)`）| 模式 B 自适应（但为**手写** Dormand-Prince 5(4)+步长加倍）；模式 A **定步长 Euler** |
| `num_steps` 语义 | 保存点网格，真实步数由误差控制决定 | 模式 B 当步数上限；模式 A 当真实步数 |

`basinflow/sampling/adaptive_ode.py` 里写的"torchdiffeq 在本机装不上"是**错的**：
`pip download torchdiffeq` 正常，`pip install` 后 `import` 正常。手写解算器因此没有存在理由。

### 13.2 改动

- **新增 `basinflow/sampling/ode.py`**：`sample_ode(field, initial, ...)` 直接调用
  `torchdiffeq.odeint(method="dopri5", atol=1e-6, rtol=1e-3, options={"max_num_steps":..., "dtype": state.dtype})`，
  与 MolGEN 的 `Sampler.sample_ode` 同一库、同一方法、同一默认容差。`method="euler"` 保留，
  对应 MolGEN 的 `--sampling_method euler`。
- **删除** `sampling/adaptive_ode.py` 与 `tests/test_adaptive_ode.py`（新增 `tests/test_ode_sampler.py`）。
- **模式 A**（`sampling/candidate_sampler.py`）：Euler 循环 → `sample_ode`；固定原子由
  "步后回位" 改为 "场内置零"（与模式 B 一致，约束在每个 RK 级都成立）。
- **模式 B**（`evaluation/transition_state.py`）：`integrate_adaptive` → `sample_ode`。
- **其余三处手写 rollout** 一并统一：`evaluation/product_flow_quality.py`、
  `tools/diagnose_flow_fit.py`、`examples/train_product_flow.py::_rollout_rmsd`。
- **删除 `graph_update_interval` / `eval_graph_update_interval`**（41 个配置 + 全部调用点）：
  自适应步长不均匀，"每 k 步重建图"没有意义；MolGEN 的骨干每次前向都自建图，模式 B 本来也是如此。
  现在 EGNN 在**每次场求值**重建图，PaiNN 内部重建。
- **`num_steps` / `eval_num_steps` 语义**改为 MolGEN 的 `--inference_steps`：`[t0,t1]` 上的保存点个数，
  **必须 ≥ 2**（`num_steps=1` 会让 torchdiffeq 直接返回初值、完全不积分）。11 个原本写
  `eval_num_steps = 1` 的"一步"配置已改为 50。
- **配置**：新增 `[evaluation] integrator`（默认 `dopri5`）、`solver_atol`、`solver_rtol`；
  三个 workflow 把**生效值**写进 `config.resolved.ini`，避免"默认值不进记录"的溯源缺口。
- **依赖**：`pyproject.toml` 的 `models` extra 增加 `torchdiffeq`（MolGEN 本身就用它）。
  §7.10 的"无新增依赖"针对数据层；本次是你明确要求的求解器统一所需。

### 13.3 过程中发现的两个真实缺陷（都已修）

1. **torchdiffeq 在 MPS 上直接报错**：它把 `rtol`/`atol` 转成 float64 张量，而 MPS 不支持 float64
   （`TypeError: Cannot convert a MPS Tensor to float64`）。修法：把解算器的 *时间 dtype* 设为状态的
   dtype（`options={"dtype": state.dtype}`），float32 状态就在 MPS 上跑通。已加回归测试。
2. **未训练/粗糙的场会让解算器把步长压到 4e-17 并以库内部 `AssertionError: underflow in dt` 崩溃**——
   对使用者读起来像调用方的 bug，实际是"这个场不可积分"。现在统一转成 `FloatingPointError`，
   带上区间、容差、场求值次数，并提示可用 `integrator="euler"` 兜底；采样器再补上 basin/candidate 上下文。
   注意：**放宽容差救不了粗糙的场**（实测 atol=1e-3/rtol=1e-1 只会变成非有限值，被我们自己的守卫抓到），
   所以这不是一个可以用容差绕开的问题。

### 13.4 验证

- `python -m pytest` → **285 passed, 1 skipped**（新增 `tests/test_ode_sampler.py` 11 项）。
- 真实数据两种模式各跑一次：
  - 模式 A：`runs/_smoke/r_to_p` — `integrator=dopri5, atol=1e-6, rtol=1e-3`，
    `sampling/summary.json` 现在带 `integrator`/`solver`/`saved_points`。
  - 模式 B：`runs/_smoke/rp_to_ts` — 同样，`field_evaluations` 均值 20（2 个 trial，4 个保存点）。
- `config.resolved.ini` 与 `candidates.json` 均记录生效解算器。

### 13.5 这对已有数值意味着什么

**R→P 的历史指标全部不可直接比较**：它们来自定步长 Euler，现在换成自适应 dopri5。此外
11 个 `eval_num_steps = 1` 的配置在 Euler 语义下是"一步直接回归"，在 dopri5 语义下是"1 个保存点＝不积分"，
已改为 50 —— 这类配置需要重跑。R+P→TS 的数字也变了：手写步长加倍换成 torchdiffeq 的控制器，
容差相同但误差估计与步长序列不同。

### 13.6 仍未统一、需要你决定的一点

MolGEN 的时间律是 `t ~ Beta(0.8,0.8)` 覆盖整条路径（`mdgen/transport/transport.py:265`），
所以它的自适应解算器始终在训练分布内。我们有一批 R→P 配置**故意**钉住 `flow_time = 0.0`
（单点监督，docs/22）——这是有意为之的分歧，但它让"在 [0,1] 上自适应积分"变成对未训练区域的积分。
本次只统一了积分器，没有动这些配置的训练时间律。

### 13.7 「崩溃为什么 MolGEN 没有」——已实测定位

**先排除我们的封装**：把 BasinFlow 全部绕开，直接用 MolGEN 的调用形状
（`odeint(fn, y0, torch.linspace(0,1,50), method="dopri5", atol=1e-6, rtol=1e-3)`，
字段闭包照抄 `mdgen/transport/integrators.py:98-115`）喂我们那个失败模型的场：

| 训练预算 | |dv/dx|max | MolGEN 原始调用 |
|---|---|---|
| 5 epochs, lr 0.05 | 19.0 | `AssertionError: underflow in dt` |
| 5 epochs, lr 0.01 | 0.6 | OK |
| 40 epochs, lr 0.005 | 1.5 | OK |

所以**不是封装问题，是场的局部 Lipschitz 常数太大**：显式 RK 的步长受 `|∂v/∂x|` 限制，
误差控制不断拒绝步长、按 ×0.2 收缩，直到 float64 下 `t + dt == t`，torchdiffeq 用断言收场。

**为什么 MolGEN 遇不到**（结构性，不是运气）：

1. **它从不积分欠训练的场。** `Transition1x-inference.py` 只加载长训 checkpoint——代码里
   列出的候选是 `epoch=224-step=0035775`、`epoch=709-step=0112890`、`epoch=534-step=1555780`、
   `epoch=1159-step=0184440`，并且按 val RMSD 挑。我们的框架**故意**采样 `epoch_checkpoints/`
   里的中途检查点（`tools/evaluate_transition_state.py --watch`），还有 smoke/诊断跑在小数据上。
2. **它的场是全数据集的条件平均。** 训练集是完整 Transition1x train split，条件场回归到上万个
   (R,P) 对的均值；平均会抹平曲率。实测在同一份真实 0910 数据上，用同样小的 EGNN（16×1）
   和同样激进的 lr=0.05、5 epochs，**即使只给 1 个事件**也只有 `|dv/dx| = 0.4` 并且积分正常：

   | 训练事件数 | 1 | 8 | 32 | 128 | 512 |
   |---|---|---|---|---|---|
   | \|dv/dx\|max | 0.4 | 1.2 | 1.3 | 1.6 | 1.6 |
   | dopri5 | OK | OK | OK | OK | OK |

   之前的崩溃只出现在**合成玩具**（H-O-H、1 个可动原子、目标只是该原子 0.25 Å 的位移）
   配 5 epochs × lr 0.05 这个病态优化状态上。也就是说：这是玩具/欠训练产物，不是采样器的潜在缺陷，
   在真实数据 + 我们实际使用的预算下复现不出来。
   （诚实边界：MolGEN 需要 CUDA 与官方 checkpoint，本机跑不起来，所以"MolGEN 不会崩"是**结构性论证**，
   不是我在它上面跑出来的观测。）
3. 我原本猜"钉住 `flow_time = 0.0` 会让它更严重"，**实测反了**：钉住 t=0 的臂学出一个几乎与 x 无关的
   常向量场（`|dv/dx| ≈ 0`，积分当然成功），因为 `gaussian_scale = 0` 时目标 `P − R` 对每个样本都相同。
   这正是 docs/22 说的"直接回归"退化——它**不**引发崩溃，但也不是一个场。这个区分要留着。

### 13.8 可比性范围修正（EGNN 与 PaiNN 都要重算）

之前只说了"EGNN 旧 checkpoint 无法严格加载"，把范围说窄了。实测结论：

| 项 | EGNN | PaiNN |
|---|---|---|
| 网络 / `state_dict` | 删两个头 → 少 6 个张量 → `strict=True` 加载失败 → **必须重训** | 本次未改 `models/painn/**` 一行（该目录的未提交改动是本次之前的工作）→ 旧 checkpoint **能加载** |
| 模式 A 训练数据 | — | **逐元素一致**：搭出 HEAD 的旧数据层对比 16 样本 × 14 字段，`flow_time=0.5/None/0.0` 全部相同；`bond_change_condition` 两种 source 下模型契约字段也相同 |
| 模式 B 训练数据 | `seed_type_id` 由硬编码 0 改为映射值 4（`midpoint_bridge`）→ `seed_type_embedding` 换行 → 旧 EGNN TS 数字不可复现 | PaiNN 不消费任何 `seed_*` 字段 → **不受影响** |
| 损失 | 去头后变了 | aux 权重为 0 时数值**相同**；45 个 painn 配置现在只有 `velocity_weight` + `norm` |
| **采样（两条模式）** | **变了** | **变了** |
| `graph_update_interval` | 41 个配置的档位 → 改为每步重建 | 对该键**本来就是空转**（PaiNN 每次前向自建图），删除只少一个配置键 |

**结论**：所有 **PaiNN 报告数字**都要重算（解算器换了）；但 **R→P 的 PaiNN 各臂不需要重训**——
数据逐元素一致、损失在 aux=0 时数值一致、checkpoint 仍能加载，直接**重采样**即可。
真正必须重训的是 EGNN（`state_dict` + 采样器两重变化）。EGNN vs PaiNN 的框架对比要整体重跑，
因为 EGNN 那一侧无论如何都是新权重。

§12.8 第 2 条列的 4 个"辅助损失非零"配置，实际性质不同，已核实：

- `semantic_flow_au.ini`、`semantic_flow_pt.ini`：无 `backend` → semantic workflow 默认 **EGNN** →
  确实以 `active_weight = 0.1` / `direction_weight = 0.1` 跑过，属于"目标函数不同、不可比"。
- `product_flow_transition1x_use_ind_40epochs{,_batch500}.ini`：无 `backend` → `examples/train_product_flow.py`
  默认 **painn**，而旧 `flow_loss` 在 `active != 0` 时读 `output["active_logits"]`，PaiNN 从不产出该键
  → 这两个配置**按原样会直接 `KeyError` 崩**，加上它们指向的
  `transitionx1/train/events_use_ind_centered` 目录不存在。所以它们不是"不可比的基线"，而是**没有结果**。

---

## 14. Au 体系 benchmark（数据层 + 积分器改造后第一次跑通）

### 14.1 数据与设置

| 项 | 值 |
|---|---|
| 数据 | `/Users/wx/Desktop/benchmark/au/events`：24 事件 / 13 basin，全部 Au₁₀₁ |
| 可动原子 | **每个事件恰好 1 个**（索引 47），其余 100 个由 FixAtoms 固定 |
| 位移 | 可动原子跳 **2.6–3.5 Å**（12/13 basin 是 3.451 Å）；pbc 全 False、无 MIC 环绕 |
| split | 数据目录自带 v2 manifest：9 train / 2 test basin（**17 个训练事件**） |
| 配置 | `configs/0910/official_au_stable.ini` 的完整设置（PaiNN 128×4、r_max 5、scaled、lr 1e-3、1000 步、batch 4、16 saved points、16 trials/basin），只改 `output_dir` |
| 命令 | `python -c "from basinflow.workflows.semantic import run_semantic_benchmark; run_semantic_benchmark('runs/au_bench/stable.ini')"` |
| 产物 | `runs/au_bench/stable/{checkpoint.pt, training.log, config.resolved.ini, eval_metrics.json, sampling/{summary.json, trial_metrics.csv}}` |

`runs/au_bench/smoke.ini`（20 步 / 2 trials）先证明流程通过，再跑正式预算。

### 14.2 结果（dopri5/16，1000 步）

| arm | macro 最近产物 RMSD | 召回 @0.1 / 0.2 / 0.5 Å |
|---|---|---|
| **model** | **0.389711 Å** | **1.000 / 1.000 / 1.000** |
| zero_motion（R 本身） | 3.451083 Å | 0 / 0 / 0 |
| gaussian_only | 3.412819 Å | 0 / 0 / 0 |

32 个候选中 31 个严格优于 R 基线；固定原子漂移 0.0；模型可动原子位移 3.5169 Å（真值 3.451，幅度差 ~2%），
残余 0.39 Å 是方向/细节误差。**流程跑通，任务学到了**（2 个测试 basin 的已知跳跃通道都被覆盖）。
未做 EAM 松弛（配置 `enabled = false`）、未做 NEB/鞍点确认。

### 14.3 与 docs/13 的 0.170112 Å 对比：用保留下来的旧 checkpoint 精确分解

`/Users/wx/Desktop/benchmark/0910/official/au_stable/` 仍然在，可用它把差异拆开：

| 权重 | 积分器 | macro 最近产物 RMSD |
|---|---|---|
| 历史 checkpoint | `euler`/17（＝旧采样器 16 步 × 1/16） | **0.170112** ← 与 docs/13 完全一致 |
| 历史 checkpoint | `dopri5`/16（新） | 0.289354 |
| 本次 run1 | `euler`/17 | 0.324296 |
| 本次 run1 | `dopri5`/16 | 0.389711 |
| 本次 run2（**同配置同 seed**） | `dopri5`/16 | 0.339965 |

`euler`/17 精确复现 docs/13 的数字，说明积分器迁移在**旧采样器语义**上是逐位等价的。

**效应一：积分器。** 在同一份权重上 0.170 → 0.289。但这**不能读作"新解算器更差"**——
逐 basin 看符号是反的（真值 3.451 Å）：

| basin | dopri5 终点位移 / 误差 | euler 终点位移 / 误差 |
|---|---|---|
| 1 | 3.809 / 0.359 | 3.551 / **0.101** |
| 5 | 3.007 / **0.497** | 2.712 / 0.740 |

basin 1 上模型的场过冲，而 Euler 的左端点求积**系统性欠积分**（16 步只走到 t=15/16，用 16 个左端点的平均），
恰好抵消了过冲；basin 5 上模型欠冲，Euler 让误差更大。**2 个测试 basin 的宏均值由"哪个 basin 的模型偏置朝哪边"决定**，
不构成"哪个积分器更好"的证据。自适应解是正确的积分：dopri5 在 16 与 64 个保存点下结果**逐位相同**（0.389711）。

**效应二：训练本身不可复现（本次最重要的发现）。** 同一份配置、同一个 seed、连续两次运行：

| | 权重 | 最终 loss | dopri5/16 RMSD |
|---|---|---|---|
| run1 | — | 0.08418584 | 0.389711 |
| run2 | 最大差 **0.1826** | 0.07676078 | 0.339965 |

逐层定位到原因（200 步、同 seed、同进程内对比）：

| 设置 | 两次运行是否逐位相同 |
|---|---|
| `cpu_threads = 4` | **否**（最大权重差 1.8e-2） |
| `cpu_threads = 4` + `torch.use_deterministic_algorithms(True, warn_only=True)` | **是** |
| `cpu_threads = 1` | **是** |

即**多线程浮点归约顺序**（PaiNN 消息传递里的 index/scatter 归约）导致不可复现，不是 seed/RNG 缺陷。
历史 checkpoint 与本次 epoch-1 loss 只差 5e-5（就是这点浮点差异），1000 步 Adam 之后放大成 0.18 的权重差
和 0.22 Å 的指标差。

**结论：docs/13 那张表里每个数字都是单次抽样**，而 run-to-run 的离散度（本次 0.34–0.39）与它要比较的效应同量级；
再加上测试集只有 2 个 basin，Au 上任何 A/B 对比目前都不具备判别力。

### 14.4 建议

1. Au/Pt 与 0910 的 benchmark 配置把 `cpu_threads` 设为 **1**（或全局打开
   `use_deterministic_algorithms`），否则重跑得到的不是同一个数。这是复现性的前提，先于任何模型改动。
2. 报告改为 **多个 seed 的分布**（至少 3 次），并显式给出测试集 basin 数；2 个 basin 的宏均值不要用于排名。
3. 若要继续用 Au 做框架对比，先固定线程再重新生成全部基线（zero-motion / Gaussian-only 是确定性的，
   模型臂不是）。
