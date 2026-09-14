# 核心 API 地图与入口清单（2026-09-11）

本文件回答三个问题：**程序从哪进**、**模块之间怎么调**、**哪些是核心、哪些是历史包袱**。
所有判断都附可核对的证据，不凭印象。

## 一、两个真实工作流（其余都是诊断或历史）

项目当前只有两条被实际运行的链路，configs 里 17 个配置**全部** `backend = painn`。

### 工作流 A：分子（Transition1x）R→P 产物生成

```
examples/train_transition1x_product_flow.py            <- 入口
  └─ basinflow.data.partitions.load_data_partitions    数据来源（外部三目录 / manifest / 比例）
  └─ basinflow.data.catalog.EventCatalog               领域对象：结构、事件、basin
  └─ basinflow.data.pyg.EventFlowDataset               PyG 视图：x_t、target_velocity、掩码
  └─ basinflow.models.factory.{model_spec,build_model} 后端选择与构造
        └─ basinflow.models.painn.PaiNN                 双几何 PaiNN 骨干
              ├─ painn/layers.py  张量层（径向基、包络、消息、更新、输出头）
              └─ painn/modules.py EventData 适配器 + 流时间映射
  └─ basinflow.training.product_flow.train_product_flow 训练循环（逐轮验证、检查点）
  └─ basinflow.evaluation.single_event_molecule.run_single_event_molecule_trials
                                                       采样 + 逐 trial 指标
  └─ basinflow.models.loss.flow_loss                    目标函数（两个后端共用）
```

### 工作流 B：周期体系（Au/Pt）basin 级提案

```
examples/benchmark_semantic_flow.py                     <- 入口
  └─ (同上) load_data_partitions / EventCatalog
  └─ basinflow.data.pyg.BasinDataset                    只含反应物的推理视图
  └─ basinflow.sampling.candidate_sampler.CandidateSampler   Euler 积分出多候选
  └─ basinflow.models.painn.PaiNN                       每一步重建自己的图
  └─ basinflow.evaluation.semantic_flow.{run_gaussian_trials,write_loss_artifacts}
```

**要点**：`BasinDataset` 与 `EventFlowDataset` 的分离是刻意的——前者禁止任何目标信息
进入推理路径，后者才有监督字段。改接口时不要打破这条线。

## 二、模块清单：职责与核心/历史

| 模块 | 职责 | 状态 |
|---|---|---|
| `geometry/mic.py` | 最小镜像、去活性原子、方向推导 | **核心** |
| `geometry/graph.py` | 结构级邻域图（ASE，PBC 正确性基准）+ torch 适配器 | **核心** |
| `geometry/radius_graph.py` | 张量级 radius graph（纯 torch，PaiNN 每步重建） | **核心** |
| `data/records.py` | `StructureRecord` / `EventRecord` / `BasinRecord` | **核心** |
| `data/catalog.py` | `EventCatalog`（领域对象 + `EventTarget`） | **核心** |
| `data/partitions.py` | 显式数据分区解析（外部目录 / manifest / 比例） | **核心** |
| `data/pyg.py` | `EventFlowDataset`（监督）+ `BasinDataset`（推理） | **核心** |
| `data/raw_events.py` | EON 目录读取 | 核心（Au/Pt 输入） |
| `seeds/records.py` | `EventSeed` / `SeedContext` | **核心** |
| `seeds/generators.py` | Zero / Gaussian / Directional / Product 四种种子 | Zero/Gaussian 核心；Product 仅诊断；Directional 仅 legacy 采样入口用 |
| `models/painn/` | 当前唯一在用的骨干 | **核心** |
| `models/loss.py` | `flow_loss` / `FlowLossWeights`（两后端共用） | **核心** |
| `models/factory.py` | 后端选择与检查点恢复 | **核心** |
| `models/egnn/` | 旧 EGNN 骨干（`layers.py` + `egnn.py`），按你的要求整理成与 `painn/` 同构的包 | **兼容路径** |
| `training/product_flow.py` | 训练循环 + 逐轮验证 | **核心** |
| `sampling/candidate_sampler.py` | basin → 多候选 | **核心** |
| `evaluation/basin_recall.py` | 召回、聚类、`movable_mic_rmsd` | **核心** |
| `evaluation/semantic_flow.py` | Au/Pt 试验编排 + 损失图 | 核心（周期体系） |
| `evaluation/single_event_molecule.py` | 分子单事件试验编排 | **核心**（分子体系） |
| `evaluation/product_flow_quality.py` | 旧 EGNN 的 pairwise 质量评测 | **历史** |
| `relaxation/` | 结构弛豫 | 可选；当前写死 EAM，见 §五 |

## 三、入口清单（真正该记的命令）

| 目的 | 命令 |
|---|---|
| 训练并评测 Transition1x（分子 R→P） | `python examples/train_transition1x_product_flow.py --config configs/0910/official_transition1x_aligned.ini` |
| 训练并评测 Au/Pt（周期体系） | `python examples/benchmark_semantic_flow.py --config configs/0910/official_pt_coverage_bounded.ini --system pt` |
| 汇总已有运行 | `python tools/summarize_benchmark.py --run-dir <RUN> --output <OUT.json>` |
| 训练稳定性诊断 | `python tools/diagnose_painn_stability.py --config <CFG> --stability-mode bounded` |
| 流场拟合 / oracle 诊断 | `python tools/diagnose_flow_fit.py --run-dir <RUN> --output <OUT.json>` |
| 数据转换（EON / Transition1x） | `python tools/eon_to_events.py …` / `python tools/transition1x_to_events.py …` |
| 显式分区导出 | `python tools/prepare_transition1x_explicit_split.py …` |

其余 `examples/*.py` 见 §四。

## 四、冗余盘点（含证据）

**(1) EGNN 后端：17 个配置全部使用 `painn`，无一使用 EGNN。**

- 证据：`grep backend configs/0910/*.ini` → 17/17 为 `painn`。
- 参考实现 `/Users/wx/Desktop/yyxwjq/egnn-pytorch` 的核心是 **454 行**，而 BasinFlow 的
  `egnn_product_flow.py` 在抽出共用损失后只有 **211 行**——**照参考重写会让代码变长**，
  与"更简洁"的目标相反。
- 它仍被 5 个测试文件覆盖（`test_candidate_sampler` / `test_e2e_stage3` /
  `test_pyg_datasets` / `test_short_names` / `test_training_product_flow`）。
- **建议**：EGNN 是纯粹的历史兼容路径。若要真正"放弃冗余"，应**整体删除 EGNN 后端
  及其 5 个测试文件**，同时删掉 `evaluation/product_flow_quality.py`、(可能) 
  `geometry/graph.py` 的 `torch_neighbor_graph_from_batch`（仅 EGNN 与 legacy 采样用）。
  **这会减少测试数量，属于破坏性改动，需你明确同意后我再做。**
  `AGENTS.md` 目前写着"保留 EGNN 兼容路径"，与"删冗余"直接冲突，需要你更新该条。

**(2) `examples/` 共 2174 行，其中约 1500 行是历史/演示脚本。**

| 脚本 | 行数 | 引用数 | 判断 |
|---|---:|---:|---|
| `train_transition1x_product_flow.py` | 235 | 4 | **核心入口 A** |
| `benchmark_semantic_flow.py` | 245 | 4 | **核心入口 B** |
| `train_product_flow.py` | 700 | 3 | 历史（EGNN pairwise 训练器） |
| `sample_product_flow.py` | 289 | 3 | 历史（EGNN 采样） |
| `demo_pipeline.py` | 143 | 4 | 演示（有测试依赖，保留） |

**已执行** ✅ 删除了三个只有文档引用的历史脚本（`rebuild_semantic_flow_output.py`、
`regenerate_semantic_flow.py`、`e2e_stage3.py`，合计 562 行）。`train_product_flow.py`
与 `sample_product_flow.py` **保留**：它们覆盖 EGNN 后端，而该后端按你的要求保留并整理。

## 五、已解决（本轮）

1. **`relaxation/` 已改为 calculator 无关** ✅
   - `relaxation/relax.py` 提供 `relax(structure, calculator, RelaxationConfig)`——
     **任何 ASE calculator 都能用**（EAM、EMT、DFT、机器学习势），EAM 只是一个便捷
     工厂 `eam_calculator(path, form)`。
   - `RelaxationConfig` 只管优化器（`fmax` / `steps` / `optimizer` ∈ FIRE/BFGS/LBFGS），
     不再出现 `eam_potential` 之类的字段。
   - `evaluation/semantic_flow.py` 的 `run_gaussian_trials` 改为接收
     `calculator_factory` + `relaxation_config`，不再耦合 EAM。
   - **能力没有退化**：工厂是惰性的，势文件缺失/损坏仍在**逐 trial** 记为
     `relax_status = "failed"` 而不会中止整轮 benchmark（这条由既有测试捕获——我第一版
     改错时它立刻失败，说明覆盖有效）。
   - 测试从 `test_eam_relaxation.py` 重写为 `test_relaxation.py`（4 条），其中一条专门
     用**任意对象**作 calculator 来证明解耦。

2. **总入口已建立** ✅ `basinflow/cli.py`，并通过 `[project.scripts]` 安装为
   `basinflow` 命令：
   ```
   basinflow train-transition1x --config <ini>
   basinflow benchmark --config <ini> [--system pt]
   ```
   编排已从脚本下沉到库里 `basinflow/workflows/`：
   - `workflows/config.py` —— `read_config(path, required)` + `parse_bool`
     （原先两个工作流各有一份**逐字节相同**的 `_read_config`，已合并）
   - `workflows/transition1x.py` —— `run_transition1x(config_path)`
   - `workflows/semantic.py` —— `run_semantic_benchmark(config_path, system=...)`
   - `seeds/from_config.py` —— `init_generators(config, seed)`（原先也是两份重复）
   - `examples/*.py` 变成 **27/28 行的薄包装**（原 235/245 行），CLI 与脚本调用同一函数，
     **不可能再漂移**。
   - 同时清掉了 7 处死导入。

   `device` 选择**故意没有合并**：`transition1x` 支持 `auto/cpu/mps`，`semantic` 明确
   只支持 CPU。这是能力差异而非重复，合并会掩盖它。

## 六、仍待你裁决 ☐

1. **EGNN 去留**（§四(1)）——破坏性：17 个配置全部使用 `painn`，EGNN 无任何配置在用，
   但被 5 个测试文件覆盖。删除它会**减少测试数量**，且与 `AGENTS.md` 现有的
   "保留 EGNN 兼容路径"一条直接冲突。需你更新该条并明确同意，我才会删。
2. **`examples/` 归档策略**（§四(2)）：`train_product_flow.py`(700 行)、
   `sample_product_flow.py`(289)、`rebuild_semantic_flow_output.py`(234, 0 引用)、
   `regenerate_semantic_flow.py`(113, 0 引用)、`e2e_stage3.py`(215)、
   `demo_pipeline.py`(143) 合计约 1500 行历史/演示代码。建议移入 `examples/archive/`
   或删除；删除同样会动到测试。
3. **MolGEN 严格对齐**的四个裁决点见 `docs/20_molgen_strict_parity.md` §七。其中
   **"任务用 R+P→TS 还是 R→P"决定后续全部工作**：MolGEN 是 R+P→TS，不换任务就无法
   与它的数字比较。
4. **`configs/` 组织**：建议按体系分组为 `configs/{au,pt,transition1x}/`（当前 17 个
   配置平铺在 `configs/0910/`）。
5. **提交基线**：工作树仍有约 98 项未提交变更，现有架构**没有提交历史**。建议先打
   基线提交再继续功能改动。


## 七、本轮追加完成 ✅

1. **EGNN 已整理成独立包**（按你的要求，与 `painn/` 同构）：
   ```
   models/egnn/{__init__.py, layers.py, egnn.py}
   models/painn/{__init__.py, layers.py, modules.py, painn.py}
   ```
   `layers.py` 放等变消息传递层，主模块负责组装嵌入/层/输出头——与 PaiNN 的组织方式一致。

2. **删除三个只有文档引用的历史脚本** ✅：`rebuild_semantic_flow_output.py`、
   `regenerate_semantic_flow.py`、`e2e_stage3.py`（合计 562 行）。`train_product_flow.py`
   与 `sample_product_flow.py` 保留——它们覆盖你要求保留的 EGNN 后端。

3. **新增 `tools/evaluate_checkpoint.py`** ✅：把同一套评测协议指向**任意中间 epoch
   检查点**，这样长训练不必"要么训到底、要么没有数字"。它绝不写入运行自己的
   `sampling/`（必须另给 `--output`），并带 `--max-basins` 供快速冒烟。

4. **为 R→P 目标补齐 MolGEN 对应的三个旋钮** ✅（此前缺少，导致无法做对齐实验）：
   - `[loss] norm = l1|mse`（MolGEN 默认 L1，`transport.py:392-393`）
   - `[init] time_distribution = uniform|beta` + `beta_alpha`（MolGEN 用 Beta(0.8,0.8)，
     `transport.py:265`，uniform 那行被注释掉）
   - AdamW 已支持（`training/product_flow.py` 早已接受 `optimizer_name="adamw"`）
   新增配置 `configs/0910/official_transition1x_molgen_style.ini`：256 特征 / 6 层 /
   96 径向基 / cutoff 12（MolGEN 有效容量）、L1、Beta(0.8,0.8)、AdamW lr 1e-4、
   grad clip 1.0、batch 64。
