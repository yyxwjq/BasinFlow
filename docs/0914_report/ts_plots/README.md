# R + P → TS 结果图集（job 2474126）

数据源：`/Users/wx/Desktop/config_ts_2474126`（即超算 `runs/0914TS/config_ts_2474126` 的拷贝）。
所有图由 `tools/plot_transition_state_results.py` 与 `tools/plot_loss.py` 从该目录的
`loss_by_epoch.csv` / `sampling/trial_metrics.csv` 直接生成，数字未手工誊写。

## 文件清单

| 文件 | 内容 |
|---|---|
| `loss_by_epoch.png` | loss 随 epoch 的变化（40 轮） |
| `rmsd_distribution.png` | **全部 21440 次采样的 RMSD 分布**（左）＋累计精度（右） |
| `rmsd_by_event.png` | 536 个事件按平均 RMSD 排序，一眼看出长尾在哪 |
| `val_event_XXXXX_rmsd_40_trials.png` | **每个事件一张图**（共 536 张）：40 次采样的 RMSD 轨迹 + 直方图 |
| `plot_summary.json` | 图中统计量的机器可读版本 |

## 复现命令

```bash
python tools/plot_transition_state_results.py /Users/wx/Desktop/config_ts_2474126 \
    --per-event --output docs/0914_report/ts_plots
python tools/plot_loss.py /Users/wx/Desktop/config_ts_2474126/loss_by_epoch.csv \
    -o docs/0914_report/ts_plots/loss_by_epoch.png \
    --title "R + P -> TS training loss per epoch"
```

（`--per-event` 不加以则只出汇总图。`plot_loss.py` 原先只认空格分隔的
`training.log`，现已同时支持 `loss_by_epoch.csv` 这种逗号分隔表。）

## 关键统计

每次采样（n = 21440，**可部署口径**）：

| 阈值 | 占比 |
|---|---:|
| ≤ 0.10 Å | 16.8% |
| ≤ 0.20 Å | 44.0% |
| ≤ 0.30 Å | 64.0% |
| **≤ 0.487 Å（中点桥基线）** | **83.3%** |
| ≤ 1.00 Å | 97.6% |

每事件取最好一次（n = 536，**ORACLE 上界，不是成绩**）：

| 阈值 | 占比 |
|---|---:|
| ≤ 0.10 Å | 39.7% |
| ≤ 0.20 Å | 72.9% |
| ≤ 0.30 Å | 84.7% |
| ≤ 0.487 Å | 93.5% |
| ≤ 1.00 Å | 100.0% |

- 中位 0.2240 Å，均值 0.2975 Å
- **35 / 536 个事件（6.5%）的平均 RMSD 高于中点桥**，最差事件均值 0.987 Å —— 长尾集中在
  `rmsd_by_event.png` 的右端约 30 个事件

## 口径提醒

- 评测划分是 **val（536 事件）**，不是 test。`runs/0914TS` 的评测配置是
  `[evaluation] split = val`；`report.json` 里 `num_events = 536`、`aligned` 的 `count = 21440`
  都是对得上的。若要 test 口径需改 `split = test` 重跑评测。
- 这些是**未弛豫的几何 RMSD**：`report.json` 里 `relaxation` 与 `saddle_validation`
  均为 `not_run`，不是经过物理验证的事件召回率。
- 图里的 `aligned RMSD` 是 Kabsch 刚体对齐后的结果（不允许镜像），只衡量内部几何，
  不惩罚整体平移/旋转偏差。
- 每事件图中的橙色虚线是中点桥基线 0.487 Å，方便直接看出该事件是否被模型改善。
