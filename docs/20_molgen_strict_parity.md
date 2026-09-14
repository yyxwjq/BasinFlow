# Transition1x 与 MolGEN 的严格对齐：规格、差距与阻塞（2026-09-11）

用户要求：**Transition1x benchmark 必须与 MolGEN 严格一致**——训练参数、数据集数量等。

本文件把"严格一致"逐项展开为可执行的对照表，并标出**无法在当前机器上完成的部分**。
MolGEN 一侧的证据来自本轮对 `/Users/wx/Desktop/yyxwjq/MolGEN` 的逐行源码审计（带
file:line）；凡未能核实的一律标注 **UNVERIFIED**，不猜测。

## 零、【更正】MolGEN 有第二个模式，就是 R→P

我在本文件第一节写的"MolGEN 只做 R+P→TS"是**错的**。
`mdgen/parsing.py:98-101` 有两个互斥开关，`mdgen/equivariant_wrapper.py:322-343`
定义它们的掩码：

| 模式 | 条件 | 监督/生成 | 任务 |
|---|---|---|---|
| `--tps_condition` | `cond_f` = 第 0 帧 (R)，`cond_r` = 第 2 帧 (P) | `cond_mask[:,1:-1] = 1` → 第 1 帧 = **TS** | R+P→TS |
| **`--sim_condition`** | **只有 `cond_f` = 第 0 帧 (R)** | `cond_mask[:,-1] = 1` → 第 2 帧 = **P** | **R→P** |

`Transition1x-inference.py` 末尾的断言也印证了这一点：`sim_condition` 分支要求
第 0 帧与参考一致、第 2 帧与参考**不一致**（即第 2 帧是生成出来的）。

**MolGEN 的 R→P 配方**：只给 R（不含任何产物派生信息），初值 `randn * x0std`
（`x0std = 1.0`），`x_t = (1−t)x0 + t·x1` 且 `t ~ Beta(0.8, 0.8)`，L1 损失，
50 步 ODE 采样，每个样本 30 次采样取最优，指标为 Kabsch 对齐 + 原子置换匹配 +
`min(rmsd, 1)`。

**此前两个臂各差一半，所以严格配方一直没跑**：
`transition1x_bond_change_x0std1` 有 σ=1.0 + Beta t + 50 步，但**额外给了键变化条件**
（来自产物，是 oracle）；`official_transition1x_molgen_style` 只给 R，但用了 σ=0.05。
严格配方现在跑在 `transition1x_molgen_simcond_9000`。

## 一、第一阻塞：任务本身不同

| | MolGEN（默认脚本） | BasinFlow 当前 |
|---|---|---|
| 生成对象 | **过渡态 TS** | 产物 P |
| 条件 | **R 与 P 都给**（`cond_f`=frame0, `cond_r`=frame2） | **只有 R** |

证据：`mdgen/dataset.py:734` 帧序 `[R, TS, P]`；`mdgen/equivariant_wrapper.py:452-455`
断言 `loss_mask[:,0]==0`、`[:,-1]==0`、`[:,1]==1`，即**只监督中间帧**，R/P 为 clamp 输入；
`:364-376` 构造 `cond_f`/`cond_r`。

**"严格一致"因此包含一次任务变更**：要么把 BasinFlow 改成 R+P→TS（与 AGENTS.md 当前
Stage 3 的 R-only 产物生成定位冲突，且用户最终目标"先生成 n 个有效末态再生成 TS"是
两段式），要么只能在"R+P→TS"这个 MolGEN 任务上做对齐实验、而不能声称 BasinFlow 的
R-only 产物模型与 MolGEN 同水平。**这是必须先由用户裁决的第一件事。**

## 二、第二阻塞：MolGEN 的预处理数据在本机不存在，且无法从仓库复现

审计结论（已核实）：

- 训练脚本硬编码读取 `data/Transition1x/tps_masked_train-fragmented_cutoffx1.5.pt`
  （`train-Transition1x-equivariant.py:31`）与 val（`:42`）；推理读
  `test-fragmented_cutoffx1.5.pt`（`Transition1x-inference.py:39`）。
- **本机不存在 `data/` 目录、不存在 `workdir/`、不存在任何 `*.ckpt`**（`.gitignore`
  排除这三者）；README 指向 Google Drive。
- **仓库内没有任何脚本会生成 `-fragmented_cutoffx1.5` 这些文件名**：notebook 只写
  `f"{data_dir}/tps_masked_{stage}.pt"`（`scripts/Transition1x/prep_data.ipynb:414`），
  `git log -S "fragmented_cutoffx1.5"` 显示该字符串**只出现在训练/推理脚本里**。
- notebook 产出的 dict 其 `v_mask` 为全 1（`mdgen/dataset.py:724-726`），**与 wrapper
  自己的断言（`equivariant_wrapper.py:452-454`）矛盾**。

**结论：MolGEN 的实际训练/测试产物既不在本机、也不能由本仓库脚本重建。** 因此：

- **"数据集数量严格一致"目前无法执行，甚至无法核对**——真实 train/val/test 条数
  必须从 Google Drive 数据或论文中取得；本机拿不到，我不会编造数字。
- 可选路径（需用户决定）：①下载官方预处理数据（需要网络与磁盘，且要核对授权）；
  ②用 `scripts/Transition1x/prep_data.ipynb` 自行处理 `transition1x.h5`（**但这条路径
  产出的文件名与训练脚本期望的不一致，需要改脚本，等于放弃"严格一致"**）。

## 三、逐项参数对照（MolGEN 实际值 vs BasinFlow 当前值）

MolGEN 侧标注来源；**"死参数"一栏尤其重要**——照抄 CLI 默认值会得到错误的配置。

| 项目 | MolGEN 实际 | 来源 | BasinFlow 当前 | 需改成 |
|---|---|---|---|---|
| 损失 | **L1** | `transport.py:392-393`，`--KL` 默认 `L1`（`parsing.py:77`） | 逐原子 MSE | L1 |
| 时间采样 | **Beta(0.8,0.8)** | `transport.py:265`（uniform 那行**被注释掉**）、`:144-147` | **均匀**（`pyg.py:130-135` 实测） | Beta(0.8,0.8) |
| 插值路径 | 默认 **GVP**（α=sin(πt/2), σ=cos(πt/2)）；README 用 **Linear** | `path.py:274-284`；`README` | 线性、**常数速度目标** | 需明确选 GVP 还是 Linear |
| 先验 | **原点各向同性 N(0, x0std²)**，x0std=1.0 | `transport.py:260-262`，`parsing.py:83` | **R + 0.05 Å 高斯** | 改为原点高斯（或见 §六） |
| 网络层数 | **6**（硬编码） | `equivariant_wrapper.py:168` | 4 | 6 |
| 宽度 | node/embed **256**，ff **768**，heads 8 | `:168,:173,:176` | 128 | 256 |
| 径向基 | **96** | `:147` | 32 | 96 |
| cutoff | **12** | `parsing.py:112`，README 用 12 | 5（分子） | 12 |
| 优化器 | **AdamW**；**实际 lr 3e-5**（无衰减）或 5e-5→1e-4→cosine 至 3e-5 over 610 ep | `wrapper.py:170`；`:173-197` | Adam, lr 1e-3 | AdamW + 3e-5 或 warmup/cosine |
| grad clip | 1.0 | `parsing.py:38` | 10.0 | 1.0 |
| batch | README 命令 batch 64 | README | 32 | 64 |
| 采样器 | **dopri5**，50 步 | `parsing.py:78`；`Transition1x-inference.py:46` | **Euler 16 步** | dopri5 / 50 |
| 候选数 | **30** | `Transition1x-inference.py:119` | 16 | 30 |
| 条件概率 | 默认 `ratio_conditonal=0.3`（README 用 1.0） | `parsing.py:103`，`equivariant_wrapper.py:342` | 恒为条件 | 需明确 |
| 预算 | README 2000 epochs | README | 10 epochs | 2000 epochs（或明确说明按步数对齐） |

**死参数警告**（照抄 CLI 会踩坑，均已核实）：

- `--lr`（默认 1e-4）**从未被读取**，只出现在注释块（`wrapper.py:203`）。
- `--num_layers`（5）、`--num_convs`（5）、`--num_heads`、`--ff_dim`、`--edge_dim`
  **均未被读取**，实际值硬编码为 6 层 / 256 / 8。
- `mdgen/equivariant_wrapper.py:69` 用 `np.math.factorial`，在 numpy 2.x 下不存在 →
  验证指标被裸 `except`（`:473`）吞掉变成 **NaN**。本机 numpy 为 2.2.6，已核实
  `hasattr(np,'math') == False`。

## 四、评价口径必须一起对齐，否则数值不可比

我已经用 BasinFlow 的旧 Transition1x 候选实测过口径差异（`docs/18` §六）：

| 口径 | MolGEN | BasinFlow 当前 |
|---|---|---|
| 先对齐 | Kabsch | 原始 + Kabsch 并列 |
| 允许镜像 | `ignore_chirality=True` | **禁用反射** |
| RMSD 截断 | **`min(rmsd, 1.0)`** | **不截断** |
| 原子置换 | `same_order=False`（搜索） | 固定映射 |
| 候选选择 | **30 候选取最低 DFT 能量（oracle）** | 无选择 |

同一批候选按参考口径重算：**1.6440 Å（raw）→ 1.0173（对齐）→ 0.9589（再允许镜像）**。
即换口径就能降约 42%，但 1.0 Å 截断会**饱和**——参考里任何接近 1.0 的数都是天花板。

**结论：要报出"与 MolGEN 同水平"的 RMSD，必须同时采用它的四项口径**，并且**并列**
报出未截断值，否则不可比也不可信。

## 五、必须先说清的一点：只对齐超参数达不到 0.1 Å

本轮实测（`docs/18` §十一）表明 BasinFlow 现在的瓶颈**不是超参数**：

| t | 方差解释率（val / train） |
|---:|---:|
| 0.00 | **5.2% / 7.0%** |
| 0.50 | 97.4% / 97.8% |

- 目标是**常数**（`target_velocity = P − x₀` 与 t 无关），但拟合率随 t 从 5% 涨到 97%，
  唯一原因是 `positions_2 = (1−t)x₀ + t·P` **泄露了产物**。模型在后半程"读出答案"。
- **train≈val** 证明这是**信息上限**，不是欠训练；而 `t<0.125` 区间已占总损失约 60%。
- **先验退化是关键**：x₀ = R + 0.05 Å 使 t=0 的输入里没有任何产物信息。MolGEN 的先验
  是原点高斯 σ=1，t=0 处**无可读出**，模型被迫真生成；且它的 R/P 是**显式条件输入**，
  不经 `x_t` 夹带。
- **oracle 起点反被推坏 6.22 Å**，说明学到的场只在训练流形附近有效。

因此严格对齐 MolGEN 的**有效配置**，实际上要求同时改：**任务（R+P→TS）+ 显式条件 +
原点高斯先验 + L1 + Beta 时间 + 6×256/96/12 网络 + dopri5/30 候选 + 口径**。
只改超参数（层数/宽度/学习率）不会把它带到 0.1 Å。

## 六、建议的对齐顺序（待用户确认后执行）

1. **裁决任务**：把 Transition1x 对齐实验定义为 **R+P→TS**（MolGEN 任务），与 BasinFlow
   的 R-only 产物生成**分开报告**。这是唯一能真正对上 MolGEN 数字的路径。
2. **取数据**：下载 MolGEN 官方预处理数据（或论文附录给出的 split 规模），记录 SHA-256
   与条数；在此之前不做任何"数据集数量一致"的声明。
3. **按 §三 表格逐项改**（网络/损失/时间/先验/优化器/采样器），**一次只改一类**并记录。
4. **按 §四 实现 MolGEN 口径**，并**同时**保留未截断、禁镜像、固定映射的严格口径。
5. **先验与条件是与超参数独立的两个变量**，必须分别消融（AGENTS.md 要求）。

## 七、需要用户裁决的问题

1. **任务**：Transition1x 对齐实验采用 **R+P→TS**（可与 MolGEN 比数），还是坚持 R→P
   （不可与 MolGEN 比数）？这决定后续全部工作。
2. **数据**：是否允许下载 MolGEN 官方预处理数据？若不允许，"数据集数量严格一致"
   无法执行——我不会用近似值冒充。
3. **路径/条件**：GVP 还是 Linear？`ratio_conditonal` 用 0.3 还是 1.0？（MolGEN 默认与
   README 命令不一致，必须二选一。）
4. **是否接受 oracle 候选选择**：MolGEN 报的数含"30 候选取最低 DFT 能量"。若照搬，
   必须明确标注为非无信息基线。


## 八、MolGEN 风格 R→P 实验的实测结果（2026-09-11）

按 §三 表格把**能对齐的都对齐了**，任务保持 R→P（用户裁决），数据用对齐后的单片段
分区（6733/392/391）：

`configs/0910/official_transition1x_molgen_style.ini`
256 特征 / 6 层 / 96 径向基 / cutoff 12（MolGEN 有效容量）、**L1**、
**t ~ Beta(0.8,0.8)**、AdamW lr 1e-4、grad clip 1.0、batch 64、`stability_mode=bounded`。

**结果（392 个 val basin、6272 个候选，epoch 14 / 1484 步）**：

| 指标 | 值 |
|---|---:|
| raw RMSD 均值 | **1.0370 Å** |
| 中位 | 0.9936 Å |
| 最小 | **0.1973 Å** |
| 最大 | 2.3877 Å |

对照：

| 实验 | raw RMSD 均值 |
|---|---:|
| 零移动基线（"不动"） | **0.9852 Å** |
| 对齐臂（128×4、cutoff 5、MSE、均匀 t、10 轮） | 1.1075 Å |
| **本实验（MolGEN 容量/损失/时间）** | **1.0370 Å** |

**三条结论，都必须如实陈述：**

1. **比 BasinFlow 此前最好结果改善约 6.4%**（1.1075 → 1.0370），但**仍然劣于"不动"
   基线**（0.9852）。目标 0.1–0.2 Å 未达成，差距约 5–10 倍。
2. **不是训练不足**：L1 损失自第 3 轮起就进入平台（0.1533 → 0.1125，第 7–15 轮一直在
   0.1125–0.1164 之间）。再加轮数不会有实质改善。
3. **不是积分器的错**：同一检查点上，**1 步直接预测（0.8755 Å，64 basin）不劣于
   16 步 Euler rollout（0.9477 Å）**。即多步积分没有带来增益，瓶颈在 **t≈0 的模型
   预测本身**——与 `docs/18` §十一 测得的 t=0 方差解释率仅 5.2%/7.0%（train≈val）
   完全一致。

**最小候选 0.1973 Å 说明表示能力是够的**（模型偶尔能生成接近目标的结构），
拉高均值的是"回归到条件均值"的收缩行为。

**因此：对齐 MolGEN 的超参数并不能迁移。** MolGEN 的数字来自**不同任务**
（R+P→TS，产物作为输入给出）与**不同口径**（对齐 + 允许镜像 + 截断 1.0 + 30 候选
取最低能量）。要在 R→P 上达到 0.1–0.2 Å，缺的是**"该做哪个事件"的信息**，
即 `docs/19` §六 的**键变化条件**，而不是容量或轮数。

## 九、对 §三 参数对照表的更正：先验尺度漏了一项（2026-09-11）

`docs/22` 的诊断（`tools/diagnose_flow_fit.py` 的模长比/余弦表）暴露了 §三 的一处
遗漏：**MolGEN 的初始噪声尺度是 `x0std = 1.0`，BasinFlow 一直用 `gaussian_scale = 0.05`。**

证据：`mdgen/parsing.py:83` `group.add_argument('--x0std', type=float, default=1.0)`；
`mdgen/transport/transport.py:262` `x0.append(th.randn_like(x1)*self.args.x0std)`；
MolGEN 自己的 checkpoint 命名也是 `x0std1.0`（`Transition1x-inference.py:6-14`）。

这不是一个无害的缩放差异。`x_0 = R + 0.05·ξ` 让 `x_0 ≈ R` 成为**已知量**，
于是插值位置 `x_t = (1-t)x_0 + t·x_1` 可以被代数反解
`x_1 = (x_t - (1-t)x_0)/t`。网络在 `t>0` 处学到的是这个反解而不是速度场：

| | σ=0.05（本轮之前全部 Transition1x 实验） | σ=1.0 |
|---|---|---|
| `t=0` 余弦 | 0.028（无方向） | 0.299 |
| `t=0.1` 余弦 | 0.995 | 0.843 |
| 从真实产物出发 rollout 16 步的终点 RMSD | **23.99 Å**（发散） | 见 `docs/22` |

`n=48` 事件、epoch 5/7 检查点，测量命令见 `docs/22` §一。

**结论**：§八 的三条结论里，第 2 条（"不是训练不足"）需要修正——
损失平台本身就是捷径解的产物，因为捷径在 `t>0` 处把损失压到接近 0。
真正的问题是 `t=0` 处的场从未被学到。修复后的两条路线
（`x0std=1.0`、以及 `flow_time=0` 直接回归）见 `docs/22` §一末表。
