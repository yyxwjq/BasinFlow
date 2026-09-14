# ReactOT/MolGEN 协议核查及下一轮运行

## 当前结论

用户确认自定义 Transition1x 分区为 train=9000、val=536、test=537，保留多片段。
原 64 个已评价反应及额外 8 个 Euler 诊断反应（共72个）固定留在验证集合。集合由外部工具一次生成并保存，训练
入口不再划分。这个协议可复现，但不是 MolGEN 原生 Transition1x 划分或
ReactOT 原始 1073 条评价集合的原样复现；新测试集也不能声称来自新的独立数据源。

当前主任务继续为 R-only 产品生成。ReactOT 的主结果是 R/P 条件 TS 生成，
MolGEN 默认脚本也主要生成 TS。将二者的小 TS RMSD 当成 R-only 产品任务的
数值验收目标不成立。可行做法是先复现参考原任务，再建立输入信息相同的
产品基线；不能通过喂入真实 TS、产品片段信息或 best-of-N 选择悄悄降低任务难度。

## 可核查的执行差异

| 项目 | BasinFlow 0910 | ReactOT 本地默认代码 | MolGEN 本地默认代码 |
|---|---|---|---|
| 主要任务 | R→P，多候选 | R+P→TS，确定性 | 默认脚本 R+P→TS；有 R→P 条件分支 |
| 初态 | R+0.05 Å Gaussian | (R+P)/2 | 原点 Gaussian，sigma=1 |
| 目标 | 常速度、uniform t、L2 | 时间表缩放 bridge 预测、MSE | Linear FM，Beta(0.8,0.8) 时间，L1；另有 KL 微调 |
| 网络 | 双几何 PaiNN 4×128、32 RBF | LEFTNet 6×196、96 RBF、cutoff10 | 等变消息+6×256 transformer processor、96 RBF、cutoff12 |
| 优化 | Adam lr1e-3、clip10 | AdamW lr1e-4、AMSGrad | AdamW；实现中无 decay 为3e-5，启用 decay 时 warmup/cosine |
| 预算 | 1000 updates | 上限3000×200 batches | README 2000 epochs、batch64 |
| 采样 | Euler16步、16候选 | midpoint ODE、单确定性输出 | Dopri5、30候选；50是输出时间点，非固定网络调用数 |
| RMSD | 同原子顺序、proper Kabsch、未截断诊断 | 同顺序、允许镜像、截断1Å | 可置换原子、允许镜像、截断1Å；独立分析另选30个中的最低能量候选 |

这些是本地默认代码或 README 中的设置，不能代替论文检查点实际训练全过程。
参考中的 clip/cap、候选选择和异常处理必须单列；即使提供兼容指标，也必须
保留未截断误差、几何失败率、原子映射和手性敏感指标。

ReactOT 证据：

- `react-ot/reactot/trainer/train_rpsb_ts1x.py:136`：mapping R+P→TS、RP 初态、sigma0。
- `react-ot/reactot/diffusion/en_sb.py:119`：R/P 中点；`:282`、`:310`、`:334`：训练时间、目标和 MSE。
- `react-ot/reactot/trainer/pl_trainer.py:877`：train_rpsb_all.pkl/valid_rpsb_all.pkl；`:930`：test.pkl。
- `react-ot/evaluation.py:38`、`:54`：实际 setup(fit)、val_dataloader；README 把1073条称为 set-aside test。
- `react-ot/reactot/trainer/train_rpsb_ts1x.py:68`：use_ind、包含多片段、不做 R/P 交换。
- `react-ot/reactot/trainer/train_rpsb_ts1x.py:265`：预算上限；动态 batch 使用原子数平方预算2800，见 `reactot/dataset/sampler.py:105`。
- `react-ot/reactot/analyze/rmsd.py:105`：同顺序、ignore_chirality=True、min(rmsd,1)。
- `react-ot/reactot/pre_process.py:34`：外部 XYZ 推理会将 P 对齐 R；训练 loader 只去质心，不能据此推断发布 pickle 上游没有对齐。

本地存在 `react-ot/reactot-pretrained.ckpt`（42,699,153字节），但缺少其
train_rpsb_all.pkl/valid_rpsb_all.pkl。检查点记录 epoch0/global_step200 与 lr0，
不足以反推论文完整训练史。README 数据来源为 https://zenodo.org/records/13131875 。
其 R→P 配置分支不能直接作为诚实基线：三对象数据可能仍保留真实 TS 条件，
两对象输入又与部分三对象采样索引不兼容，必须单独审计/修改并重训。

MolGEN 证据：

- `MolGEN/scripts/Transition1x/prep_data.ipynb:181`：原生 h5 datasplit；`:259`：各对象去质心，Kabsch代码被注释。
- `MolGEN/train-Transition1x-equivariant.py:31`：硬编码 tps_masked_train 数据路径，非仅依赖 data_dir。
- `MolGEN/mdgen/equivariant_wrapper.py:364`、`:377`：R-only 与 RP 条件分支。
- `MolGEN/mdgen/transport/transport.py:255`、`:392`：Gaussian 初态、L1；`:144`：Beta时间。
- `MolGEN/mdgen/wrapper.py:168`：实际优化器学习率，不应只抄 CLI --lr。
- `MolGEN/Transition1x-inference.py:22`、`:46`、`:119`：默认 TS 条件和30候选。
- `MolGEN/mdgen/equivariant_wrapper.py:452`：验证仍断言 TS-only masks、T=3。
- `MolGEN/mdgen/dataset.py:735` 和 `mdgen/model/equivariant_latent_model.py:403`：数据包含目标产品片段划分，object-aware 使用其屏蔽片段间边；产品任务需确认该信息来源。
- `MolGEN/scripts/calculate_err_pyscf_alltrials.py:147`、`:181`：30候选最低能量选择；`:403`：IRC端点邻接比较。

MolGEN 本地没有完整处理后数据与模型权重，只有测试夹具。README 提供外部
Google Drive 链接。因此当前可完成代码协议审计，尚无原始 MolGEN checkpoint
产品生成复现结果。片段条件、数据规模、原子排序及产品分支的可执行性需要
发布产物进一步确认。

## 对 BasinFlow 失败原因的证据等级

已确定：Pt 原先训练覆盖不足；原始 PaiNN 数值放大已复现并修复；Transition1x
旧划分/单片段子集/任务与参考不同；参考采样和 RMSD 口径不同；当前分子
候选内部距离和重叠严重恶化，增加 Euler 步数未解决。

待检验：原点 Gaussian 与 R 附近窄 Gaussian 的差异、L1/L2、时间采样、学习率、
足够训练预算、相对方向规范、网络容量。不能将多个差异同时调整后，把改善
归因于其中一个。独立 R/P 朝向对当前任务困难，但 MolGEN 也有仅去质心路径，
所以不能把 Kabsch 当作已经由参考验证的唯一答案。

建议顺序：先通过明确分区和多轮完整覆盖建立新基线；再固定数据/种子逐项
测试 prior、loss/time、坐标规范；用训练小子集过拟合、逐时间误差、rollout
几何三者区分容量/优化问题与采样分布偏移。TS 实验需独立条件和独立报告。

## 新工程入口

外部目录模式仅需：

```ini
[data]
train_events_dir = /absolute/train
val_events_dir = /absolute/val
test_events_dir = /absolute/test
active_threshold = 0.1
```

不需要 `[split]`。统一目录模式可指定 events_dir 和 `[split] manifest`，也不需
比例。两模式互斥。旧的显式比例自动划分仍兼容。跨目录 basin ID 或已知 rxn
标识重叠会报错；同名局部事件文件会被命名空间隔离，保留原始 event ID。
用户自行改名且无稳定身份的重复几何不在当前泄漏检测保证范围内。

转换单个分区可用 `--selection complement` 或 `--indices-json`，严格检查索引
整数、范围与重复。9000/536/537 的一次性准备命令为：

```bash
python tools/prepare_transition1x_explicit_split.py \
  /Users/wx/Desktop/yyxwjq/OAReactDiff/oa_reactdiff/data/transition1x/train_addprop.pkl \
  /Users/wx/Desktop/yyxwjq/OAReactDiff/oa_reactdiff/data/transition1x/valid_addprop.pkl \
  /Users/wx/Desktop/benchmark/0910/data/transition1x_explicit_9000 \
  --validation-count 536 --seed 20260910 \
  --prior-run /Users/wx/Desktop/benchmark/0910/official/transition1x_stable \
  --prior-plan /Users/wx/Desktop/benchmark/0910/official/diagnostics/transition1x_stable_euler/sampling_plan.json
```

固定预算配置：`configs/0910/official_pt_coverage.ini` 为 Pt 完整5轮、7260更新、
29035个样本曝光，不改变原lr/模型/初态；`official_transition1x_explicit.ini`
为新分区完整10轮，2820更新、90000样本曝光，先评价全部536验证，保留537测试。
它们仍是有界覆盖实验，不是宣称收敛或论文级训练预算。

逐轮验证使用固定 epoch0 噪声与时间的速度 MSE，不使用测试集合选参。
`save_each_epoch=true` 保存模型、优化器、Torch RNG、历史、完整轮/预算截断
标记和训练数据哈希。`resume_from` 仅接受完整 epoch 检查点，校验数据、
batch、loss与优化器；epochs/max_steps 表示累计目标。半轮状态不支持精确恢复。
旧0910只有模型权重的检查点不具备这种恢复能力。

每次训练使用新 output_dir，已有 config/training.log/checkpoint 会拒绝覆盖。
结果汇总支持外部目录并核对保存成员。测试和临时夹具位于测试临时目录，
不作为正式 benchmark 结果。

## 执行状态

代码和配置已实现。完整测试执行得到228 passed、1 skipped；随后新增诊断
样本固定测试，并将真实Au数据路径补入，运行转换/分区/真实Au入口组合得到
55 passed。完整epoch恢复与连续训练的权重逐位一致测试通过；半轮恢复和
数据变更被拒绝。两个实际CLI夹具已贯通外部目录、训练、验证、检查点和采样。

只读实际源数据核查得到9000/536/537，固定72个已评价反应，集合互不重叠；
三个集合中单片段数量分别为6733/419/364。成员已经可以由保存的命令确定，
但尚未写出真实分区目录，不能说数据导出已完成。

真实数据导出三次、Pt正式训练一次均在进程启动前被自动审批服务拦截，返回
原因均为 stream disconnected before completion；用户已明确授权重试，
再次尝试仍遇同一服务错误。目标目录仍不存在。尚未启动新的 Pt/Transition1x
正式训练，没有新的模型质量指标可报告。原0910结果保持不变。
