# 源码布局、命名与接口设计（2026-09-11）

本文件规划 BasinFlow 的源码整洁度，并给出"键的变化条件"的接口设计。目标风格参照
`liflow`：**包名与项目同名、按概念（而非依赖）划分子包、产物目录独立并 gitignore、
配套 lint 工具链**。

本轮已完成的部分标注为 ✅，待办标注为 ☐。所有结论都基于实测，不改动已完成实验的
语义（`none`/`scaled` 前向逐位不变），也不覆盖 `/Users/wx/Desktop/benchmark/0910`
下任何产物。

## 一、现状诊断（证据）

| 问题 | 证据 |
|---|---|
| **包名与项目不符** | 仓库叫 BasinFlow，包叫 `fscgp`（遗留 "FS-CGP"），`pyproject.name = "fscgp"` |
| **死目录** | `src/fscgp/flow/` 只剩 `__pycache__`，无任何 `.py`；全仓库 0 处引用 |
| **子包按依赖而非概念切分** | `geometry/mic.py`（numpy）与 `graphs/*`（torch）本是同一件事，却因框架不同分成两个包 |
| **命名与领域词汇脱节** | `inits/` 里放的是 `EventSeed` 与种子生成器，而 AGENTS.md 的词汇是 "seed" |
| **`benchmark/` 名歧义** | 里面是"指标实现"，而"benchmark"在本项目又指 0910 那批实验运行 |
| **入口脚本重复自举** | `examples/*.py` 8 份、`tools/` 3 份各自 `sys.path.insert`；`tools/diagnose_painn_stability.py` **漏了**，第一次运行直接 `ModuleNotFoundError` |
| **未忽略的机器本地内容** | `runs/`（5.4 MB 实验产物）与 `.mcp.json` 未 ignored，直接污染 `git status` |
| **tests/ 下放非测试内容** | `tests/notebook2ppt/BasinFlow_Debug_Report.ipynb`（491 KB），无任何代码引用 |
| **无 lint/format 工具链** | `pyproject.toml` 没有 `[tool.ruff]`；无 `.pre-commit-config.yaml`；`ruff`/`pre-commit` 均未安装 |
| **工作树长期未提交** | 最后一次提交是 "feat: add Stage 3 minimal training loop"，当前 99 项未提交变更（63 新增、14 删除、22 修改）——**现有架构整体未入库** |

## 二、目标布局

```
BasinFlow/
├── src/basinflow/            # 包名与项目同名
│   ├── geometry/             # 坐标数学 + 图构造（合并原 graphs/）
│   ├── data/                 # catalog / records / partitions / pyg
│   ├── seeds/                # EventSeed 与初始化生成器（原 inits/）
│   ├── models/               # factory + 骨架
│   │   └── painn/
│   ├── training/             # 训练循环
│   ├── sampling/             # basin 级候选提案
│   ├── evaluation/           # 指标与汇总（原 benchmark/）
│   └── relaxation/           # 可选结构弛豫
├── configs/                  # 按体系分组
├── docs/                     # 保留编号（AGENTS.md 依赖）
├── examples/                 # 可运行入口
├── notebooks/                # 机器本地，gitignored
├── runs/                     # 机器本地产物，gitignored
├── tests/
├── tools/                    # 诊断与转换工具，去日期戳
└── pyproject.toml
```

子包从 12 个降到 8 个。判据是**概念**：`geometry` 统一"坐标 → 图"这一件事，
不因 numpy/torch 之别拆包；`evaluation` 明确是"评测指标"，与"benchmark 运行"区分。

## 三、本轮已完成的改动 ✅

**(1) 卫生层**

- `runs/`、`.mcp.json`、`notebooks/`、`.ruff_cache/`、`.mypy_cache/` 加入 `.gitignore`；
  实测 `git status` 中 `runs/` 与 `.mcp.json` 已消失。
- 删除死目录 `src/fscgp/flow/`（以及陈旧的 `src/fscgp.egg-info`）。
- `tests/notebook2ppt/BasinFlow_Debug_Report.ipynb` → `notebooks/`（对齐 liflow 的
  `notebooks/` 约定），`tests/` 只留测试。

**(2) 命名层**

- `src/fscgp/` → `src/basinflow/`；`pyproject.name = "basinflow"`；全仓库
  `fscgp` 引用 75 个文件全部更新，**无残留**（已 grep 验证）。
- `seeds/` ← `inits/`；`evaluation/` ← `benchmark/`；`graphs/*.py` 并入 `geometry/`。
- `basinflow/__init__.py` 重写：docstring 说明各子包职责，`__all__` 按流水线排序。

**(3) 顺带修掉的一个真实缺陷**

合并 `graphs/` 时，`graphs/__init__.py` 覆盖了 `geometry/__init__.py`，其**急切**
导入 `neighborlist` → `data.records` → `geometry.mic`，构成**循环导入**（10 个测试
收集失败）。修复：`geometry/__init__.py` 改为**只留 docstring、不做任何再导出**，
并说明原因（图构造依赖 `data`，急切再导出必然成环）。同时把唯一使用包根再导出的
`tests/test_torch_graph.py` 改为直接导入子模块。

另外发现并修正一处**被 sed 反向污染**的测试：`tests/test_short_names.py` 中
"断言遗留模块已删除"的检查被误改成新规范名，导致断言反转。现已恢复其本意
（断言 `basinflow.inits` 必须保持删除状态）。

**验证**：`python -m pytest -q` → **243 passed, 1 skipped**，与重构前一致。

## 四、本轮随后完成的部分 ✅

1. **`tools/` 命名统一** ✅ 已去日期戳、统一动词：

   | 旧 | 新 |
   |---|---|
   | `summarize_0910.py` | `summarize_benchmark.py` |
   | `eon2data.py` | `eon_to_events.py` |
   | `transition1x2data.py` | `transition1x_to_events.py` |
   | `debug_painn_training.py` | `diagnose_painn_stability.py` |
   | `prepare_transition1x_benchmark.py` | `prepare_transition1x_split.py` |
   | `prepare_explicit_transition1x.py` | `prepare_transition1x_explicit_split.py` |
   | `capture_benchmark_provenance.py` | `capture_provenance.py` |
   | `plot_total_loss.py` | `plot_loss.py` |

   全部引用（含 docs、测试内的 `subprocess` 调用）同步更新，grep 验证**无残留旧名**。

2. **入口自举去重** ✅ 已统一为**单一形式**（`REPO_ROOT / "src"`），删除了 8 个 example
   里各自定义的 `SRC_DIR` 变体。更重要的是修掉一个**真实缺陷**：
   `diagnose_painn_stability.py`（原 `debug_painn_training.py`）此前**没有**自举，首次
   运行即 `ModuleNotFoundError`。现已补齐，并新增
   `tests/test_entry_point_bootstrap.py` **两条守卫测试**：①凡导入 `basinflow` 的入口
   必须含规范自举；②不得再引入第二种写法。**这正是能捕获该缺陷的检查。**

   （可选后续：`pip install -e .` 后即可删除全部自举——这是 `src/` 布局的标准用法与
   liflow 的做法，但需要你先执行一次安装。）

3. **lint 工具链** ✅ 配置已加入：`pyproject.toml` 增加 `[tool.ruff]`
   （`extend-include = ["*.ipynb"]`，与 liflow 一致）与 `dev` 可选依赖（`pre-commit`、
   `ruff`），并新增 `.pre-commit-config.yaml`，hook 集合与 revision **照抄 liflow 的
   已知可用配置**（`pre-commit-hooks` v4.6.0、`ruff-pre-commit` v0.4.2）。本机
   **仍未安装**这两个工具，启用方式写在文件注释里：
   `pip install -e ".[dev]" && pre-commit install`。首次 `ruff format` 会产生较大 diff，
   建议**单独一个提交**便于 review。

**验证**：`python -m pytest -q` → **245 passed, 1 skipped**（较重构前 243 增加 2 条守卫
测试，无退化）；10 个工具与 8 个 example 的 `--help` 冒烟全部通过。

## 五、仍待你决定 ☐

1. **提交基线**：99 项未提交意味着现有架构**没有任何提交历史**。强烈建议先打一个
   基线提交（新增架构 + 删除旧文件）再动功能——否则后续重构无法回滚。我不会代替你
   决定提交信息与粒度。
2. **`docs/` 与 `configs/` 组织**（低风险、无阻塞）：`docs/` 建议**保持编号扁平**
   （AGENTS.md 直接引用 `docs/00_project_vision.md` 等路径），只补 `README.md` 索引、
   把 `11_code_review_2026-07-03.md` 的日期移入正文；`configs/` 建议按体系分组为
   `configs/{au,pt,transition1x}/`。
3. **第六节接口设计的四个决策**，尤其**决策 1（边集定义）**——它决定数据 schema，
   改起来代价最高，建议先拍板再做实现。

## 六、接口设计：键的变化条件

### 设计目标

实测结论（见 `docs/18` §十一）是 **t=0 生成步只解释 5.2%(val)/7.0%(train) 的位移
方差**，且 `train≈val` 证明这是**信息上限**：t=0 的输入里没有任何产物信息，而
近相同反应物的事件对化学位移分歧达 **1.605 Å**。因此要补的**不是几何细节，而是
"该做哪个反应"**。

键变化条件正是这个缺口，且**不给坐标**。它同时隐含键长与键角的变化——新键合模式
决定平衡长度与角度，具体数值仍须模型生成。

### 关键设计决策（需要你确认）

**决策 1：条件的定义域。** 键变化必须落在**明确定义的边集**上，否则模型无从对齐：

- 方案 A：定义在**反应物键图**上 → 只能表达"断裂"，表达不了"生成"。
- 方案 B：定义在**并集图**（R 或 P 中存在的键）上 → 两者都能表达，但并集天然依赖 P。
- **推荐方案 C**：边集 = `t=0 几何的距离截断图 ∪ 种子指定的候选新键`，每条边带
  三值标签 `{0: 不变, 1: 生成, 2: 断裂}`。种子负责给出候选新键，几何负责给出既有边。
  这样**推理时不需要 P**，而并集语义通过种子恢复。

**决策 2：条件的载体。** 建议放入 `EventSeed`（AGENTS.md 的语义正是"标量条件 +
等变条件 + 流初始几何"）：

```python
@dataclass(frozen=True)
class EventSeed:
    seed_id: str
    seed_type: str
    seed_displacement: np.ndarray        # [N, 3]
    seed_direction: np.ndarray           # [N, 3]
    active_prior: np.ndarray             # [N]
    movable_mask: np.ndarray             # [N]
    bond_change_pairs: np.ndarray        # [K, 2] int   —— 新增
    bond_change_labels: np.ndarray       # [K]    int8  —— 新增
    metadata: dict[str, Any] = field(default_factory=dict)
```

**决策 3：进入模型的位置。** 最小且显式的做法：`DualMessageBlock` 的 `filters` 目前由
径向基得到，把**边标签的 one-hot** 拼进滤波器的输入即可：

```python
# 现在
filters = self.reference_filter(reference_radial) * reference_cutoff[:, None]
# 改为
filters = self.reference_filter(torch.cat([reference_radial, bond_change_onehot], -1)) * reference_cutoff[:, None]
```

`bond_change_onehot` 形状 `[E, 3]`，与 `edge_index` 对齐。这是**边条件**，不进入
坐标通道，等变性不受影响（one-hot 是标量不变量）。

**决策 4：泄漏审计。** 必须沿用 `active_prior` / `target_active_mask` 的既有分离模式：

- 训练/诊断数据集从**目标**计算标签 → 写入种子；
- `BasinDataset`（无目标推理路径）**只读种子**，绝不读目标；
- 加一条测试，断言无目标路径下 `bond_change` 仅来自种子——与
  `tests/` 中既有的 active-prior 泄漏测试同构。

**并且必须避免 MolGEN 的泄漏**：我在其源码实测到 `fragments_idx` 是 `[B,3,L]`，
**含被生成的那一帧用自己的真值片段划分做掩码**。BasinFlow 的键变化条件**只允许描述
反应物侧拓扑变化**，不得包含产物的片段划分或任何坐标派生量。

### 与其它改动的耦合（必须分开消融）

由 `docs/18` §十一 的测量，条件增强**单独不足以**修复问题，因为学到的向量场只在
训练流形附近有效（oracle 起点反被推坏 6.22 Å）。因此第 2、3 项必须**配套**但不
**混同**：

| 项 | 内容 | 消融关系 |
|---|---|---|
| a | 输出头归一化 ✅ 已完成 | 已并入 `bounded`，`none`/`scaled` 不变 |
| b | 键变化条件（本节） | **单独消融** vs 基线 |
| c | 离路径增广/加噪训练 | **单独消融**，再与 b 组合 |
| d | 显式三体（角度）基 | **先不做**；实测 Transition1x 的 1-3 距离 100% 在 cutoff 内，角度已被距离与边向量交叉项双重编码，新增信息近乎为零；Pt 有 43% 在 cutoff 外，可能有边际收益 |

即：**不要同时改 b/c/d 再把改善归因于其中一个**（AGENTS.md 明确要求）。

### 实现状态与单项消融结果 ✅

按决策 1–4 已落地，实现位置：

| 决策 | 落地 |
|---|---|
| 1 边集 | 边集就是模型当前的双几何图（`dual_radius_graph`，`r_max`）。当前所有配置的 `r_max`（分子 12 Å、Au/Pt 5 Å）都大于成键/断键距离（约 1.3–2.5 Å），所以"距离截断图 ∪ 种子候选新键"里的并集部分被 cutoff 自动覆盖，无需额外构造并集图。若将来把 `r_max` 降到 3 Å 以下，必须显式补种子候选键。 |
| 2 载体 | `EventSeed.bond_change`（`int8 [N, N]`，取值 `{0,1,2}`），带形状/取值/固定原子三项校验 |
| 3 进入网络 | `DualMessageBlock` 的边滤波输入由 `num_radial_basis` 加宽到 `+3`，条件与参考构型 RBF 拼接 |
| 4 泄漏审计 | `EventFlowDataset(bond_change_source="oracle")` 从 R/P 差算标签；`BasinDataset` 只读种子，种子没有提议时取"全不变"；`tests/test_bond_change_condition.py` 断言无目标路径不产出标签张量 |

**决策 4 补充**：不采用 `fragments_idx` 式泄漏。条件矩阵 `[B, n_max, n_max]` 只有
离散标签、不含任何坐标，且 padding 区恒为 0，产物坐标不可能经由该路径进入模型。

**单项消融（b 项，`transition1x_direct_t0_9000` epoch 11，val 前 200 个 basin）**：

| 条件 | raw 均值 / 中位 | aligned 均值 / 中位 | `min(rmsd,1)` 均值 |
|---|---|---|---|
| oracle 键变化 | 1.209 / **0.724** | 1.091 / **0.705** | **0.716** |
| 全不变（等于无条件） | 3.549 / 0.990 | 3.548 / 0.978 | 0.888 |
| 什么都不做（R 当 P） | 4.214 / 1.081 | 4.214 / 1.081 | 0.900 |

即 b 项**单独**就把 raw 中位压低 33%、raw 均值压低 71%。但残余误差仍停在
0.7 Å 量级，原因见 `docs/22` §二（条件只覆盖 32% 的原子，且 `(R, 拓扑)` 一对多）。
c 项（离路径增广/加噪，对应 MolGEN 的 `x0std = 1.0`）与 d 项（角度基）本轮状态：
c 已开跑（`transition1x_bond_change_x0std1`），d 仍按计划不做。
