# 26 — HPC deployment (SUSTech Slurm cluster)

This document records how the Stage 3 Transition1x workflow was deployed to and
run on the SUSTech HPC cluster, so the same run can be reproduced or moved to a
different cluster without rediscovering the environment traps.

## 1. Target

| Item | Value |
|------|-------|
| Login | `luojun26@210.75.252.4:3022` |
| Scheduler | Slurm 19.05 (`sbatch` / `squeue` / `sacct`) |
| Partition | `AMD2` — `amd[30-39]`, 2 x 48 cores, ~386 GB RAM, **no GPU** |
| Shared storage | `/public` (NFS, 153 TB) |
| BasinFlow source | `/public/home/luojun26/users/wangjq/apps/basinflow-src` |
| conda env | `/public/home/luojun26/software/miniconda3/envs/basinflow` |
| Project dir | `/public/home/luojun26/users/wangjq/transition1x` |

Home is a shared NFS mount, so the login node and every compute node see the
same conda env and the same data — no per-node staging is needed.

## 2. Environment construction

```bash
source /public/home/luojun26/software/miniconda3/etc/profile.d/conda.sh
conda create -y -n basinflow python=3.10
PY=/public/home/luojun26/software/miniconda3/envs/basinflow/bin/python
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy ase torch-geometric
$PY -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e /public/home/luojun26/users/wangjq/apps/basinflow-src
```

Resulting stack: python 3.10.18, numpy 2.2.6, scipy 1.15.3, ase 3.29.0,
torch 2.6.0+cpu, torch_geometric 2.8.0, matplotlib 3.10.9.

### Traps found (each cost a failed attempt)

1. **conda 4.9.2 mis-links `python=3.11`.** The transaction reports success but
   creates `lib/python3.1/site-packages` and never populates `bin/python`. Use
   `python=3.10`; the project requirement is only `>=3.10`.
2. **`conda activate` is a no-op in a non-interactive shell.** Sourcing
   `conda.sh` sets `CONDA_EXE` but `conda activate basinflow` can silently keep
   `/usr/bin/python` (2.7.5). Every script here uses the env's absolute
   interpreter path instead of relying on activation.
3. **`source conda.sh` with `set -u` aborts** on the unbound `PS1` reference in
   `conda.sh:55`. Do not combine `set -u` with conda's init script.
4. **`pillow>=11` has no cp310 wheel on the Tsinghua mirror** and its source
   build fails under the system `gcc` (defaults to C89). Pin `pillow<11`, which
   has a wheel. This only matters for `ase -> matplotlib -> pillow`.
5. **`pip install -e .` re-resolves `pillow`.** Install the editable package
   with `--no-deps` once the runtime dependencies are already present.
6. **`torch.backends.mps` / CUDA.** `device = auto` in
   `basinflow/workflows/transition1x.py` now resolves `cuda` when it is
   available, then `mps`, then `cpu`; `AMD2` therefore resolves to `cpu`.

## 3. Data

The frozen split (`tools/prepare_transition1x_explicit_split.py --selection
use_ind`, product aligned to reactant) was transferred as pre-converted
EON-style directories rather than re-running the converter on the cluster:

```text
.../transition1x/data/transition1x_aligned_9000/
  train/  9000 events + basin_table.csv
  val/     536 events + basin_table.csv
  test/    537 events + basin_table.csv
  split_manifest.json, source_partitions.json, {train,val,test}_indices.json
```

Because the three directories are explicit, `config.ini` must use
`train_events_dir` / `val_events_dir` / `test_events_dir` and must not set
`[split] manifest` — `load_data_partitions` rejects external directories
combined with a manifest.

`config_ts.ini` needs a second copy of the same split carrying TS targets:

```text
.../transition1x/data/transition1x_aligned_9000_ts/
  train/ 9000   val/ 536   test/ 537
```

Both datasets are on the cluster, so an R -> P number and a TS number from the
same frozen split are directly comparable.

## 4. Task input and submission

| File | Task | Backbone |
|------|------|----------|
| `config.ini` | R -> P | EGNN 128x4, `r_max` 6 |
| `config_ts.ini` | R + P -> TS (flow matching) | PaiNN 256x6, `r_max` 12 |
| `run_basinflow.sbatch` | production submission script | — |
| `smoke_test.sbatch` | 1 epoch / 2-layer / 2-event validation | — |

The scripts are deliberately thin. `run_basinflow.sbatch` activates the env, pins
the BLAS thread counts to the allocation, writes a per-job config, and calls the
CLI:

```bash
sbatch run_basinflow.sbatch                                 # R + P -> TS
sbatch run_basinflow.sbatch config.ini train-transition1x   # R -> P
squeue -u luojun26
tail -f slurm-<jobid>.out
```

Only two knobs are exposed at submit time, both via `--export=ALL,KEY=value`:
`EPOCHS` (overrides `[training] epochs`) and `CPU_THREADS` (overrides
`[runtime] cpu_threads` and the BLAS threads). Everything else belongs in the
task input file, not on the command line.

The script rewrites `output_dir` to `runs/<config-name>_<jobid>` in
`config_<jobid>.ini`, so every submission gets a fresh directory — the runner
refuses to start when the output directory already holds `training.log`,
`checkpoint.pt` or `config.resolved.ini`. An earlier revision of this script
printed package versions, counted event files and re-checked artifacts after the
run; that output was removed as noise, since the runner already writes
`config.resolved.ini`, `data_sources.json` and the training log.

## 5. Metric semantics

**R -> P** (`config.ini`, `eval_metrics.json`): contract
`pairwise_r_to_p_no_oracle` (no product or oracle topology is given to the
model) evaluated as `single_event_pairwise_no_relaxation`. The reported RMSDs are
**no-relaxation geometric candidate recall** against the recorded product.
`eval_metrics.json` states `"relaxation": "not_run"` and
`"saddle_validation": "not_run"`; these numbers are not validated KMC event
recall.

**R + P -> TS** (`config_ts.ini`, `report.json`): contract
`pairwise_rp_to_transition_state`, evaluated as `transition_state_flow_ode`.
Both endpoints are inputs by design — in this view the product frame is the task,
not the leak it is for R -> P. Sampling integrates the learned field from a
noisy source over `eval_num_steps` (16, per `docs/25` section 五之一 where 16 and
32 steps are indistinguishable), so it is a real ODE integration rather than one
Euler step. The comparison points from `docs/25` are the deterministic midpoint
bridge at 0.487 A aligned median and the pre-fix direct-regression arm at
0.230 A; a flow-matching arm below 0.487 A is the gate for extending the budget.

`[evaluation] trials_per_basin` is the number of samples drawn per test event (40
for TS) and `write_trajectories` stores each trial's integration path as a
multi-frame extxyz under `sampling/trajectories/`, with the relative path in the
`trajectory` column of `sampling/trial_metrics.csv`. Frame 0 is the drawn source,
the last frame is the proposal, and the frames between are the ODE path, so a
sample can be analysed by *how* it converged. The full val split at 40 trials is
21440 files at ~15.7 KB each, measured at **336 MB** on disk; note that
`time_distribution = fixed` (the legacy deterministic bridge) is mapped to a
uniform grid for sampling, since a pinned-time model has no time law to draw.

## 6. Backbone / loss-head constraint

`flow_loss` reads `active_logits` and `direction` whenever `active_weight` or
`direction_weight` is non-zero, but only the EGNN backbone exposes those heads.
Pairing `backend = painn` with a non-zero auxiliary weight raises
`KeyError: 'active_logits'` — this is exactly how the first deployment attempt
failed. Use `backend = egnn` for the auxiliary-head loss, or set both weights to
`0.0` for `painn`; `config_ts.ini` takes the second route.

The TS flow has a second, sharper constraint: `flow_time` must stay unset when
`time_distribution != "fixed"`. `TransitionStateDataset` rejects the combination,
because a pinned time collapses the supervision to a single point and removes the
field — that degeneracy is the root cause documented in `docs/25`.

## 7. Verification performed

* `sbatch smoke_test.sbatch` → `COMPLETED`, exit 0, 1:53 on `amd33`; artifacts
  `checkpoint.pt`, `config.resolved.ini`, `eval_metrics.json`, `history.json`,
  `loss_by_epoch.csv`, `loss_curves.png`, `sampling/`, `split_manifest.json`,
  `training.log`.
* `python -m pytest -q` on the deployed tree → 275 passed, 1 skipped,
  1 failed before the 2026-09-14 TS-flow sync; **283 passed** after it (the extra
  8 come from `tests/test_ts_flow_workflow.py`); **286 passed, 1 skipped** after
  adding the trajectory tests. The one failure is always
  `tests/test_benchmark_provenance.py`: the tree is transferred without `.git`,
  and `tools/capture_provenance.py` requires a git repository (`git diff HEAD` /
  `status` / `rev-parse`). Training and evaluation do not use git.
* Checkpoint reload: `load_model_checkpoint` restores the EGNN backbone
  (`model_backend = egnn`, `cutoff = 6.0`,
  `training_contract = transition1x_pairwise_r_to_p_no_oracle`) and
  `run_single_event_molecule_trials` samples from the reloaded model on the
  `test` partition, so a finished run is reusable for downstream inference
  without retraining.

## 8. First production R+P -> TS run (job 2474126)

`config_ts.ini` with `PaiNN 256x6 / r_max 12`, 40 epochs, `AMD2` / 32 cores /
64 GB. **COMPLETED in 3:16:05**, exit 0, on `amd33`.

| Quantity | Value |
|----------|-------|
| final training loss (L1) | 0.2062 |
| evaluation | full `val`, 536 events x 40 trials = 21440 samples, 16 ODE steps |
| aligned RMSD, one-shot | mean 0.2975 / **median 0.2240** / min 0.0102 / max 2.4289 A |
| aligned RMSD, best-of-40 | mean 0.1757 / **median 0.1236** / max 0.9868 A |
| source RMSD (drawn x0) | mean 2.9830 / median 1.7290 A |
| epoch checkpoints | 40 x ~85 MB, plus `checkpoint.pt` 28 MB; run total 3.6 GB |

Against the `docs/25` gate: the midpoint bridge is 0.487 A and the pre-fix
direct-regression arm is 0.230 A, so one-shot (0.2240) already clears both, and
best-of-40 (0.1236) is well beyond them. The source RMSD stays at its drawn value
while aligned RMSD halves, which is the behaviour of a field being learned rather
than a metric drifting.

The `--watch` scorer (`tools/evaluate_transition_state.py`) tracked the gate on a
fixed 64-event subset through training; the aligned median fell monotonically
from 0.4396 (epoch 1) to 0.2264 (epoch 40) while the source baseline held at
1.6352. Small subsets are optimistic: 8 events at epoch 3 read 0.2608 where 64
events read 0.3896, so quote the fixed-64 or full-val numbers.

Timing and sizing measured, not estimated: ~5 minutes/epoch (40 epochs ~3.2 h),
~85 MB per epoch checkpoint, and 21440 trajectory files at ~15.7 KB each
(336 MB). `save_each_epoch` is the knob for the checkpoint bulk.

## 9. Local access helpers

Kept out of version control in `.hpc/` (see `.gitignore`), because they embed
the cluster password:

| Script | Purpose |
|--------|---------|
| `.hpc/run.sh <cmd>` | run a command on the login node (`SSH_ASKPASS` based, no TTY needed) |
| `.hpc/deploy_src.sh` | `rsync --delete` the working tree into `apps/basinflow-src` (exact mirror) |
| `.hpc/deploy_data.sh` | tar the event partitions into `transition1x/data` |

`ssh` on macOS refuses to read a password without a TTY and `expect` cannot
allocate a PTY in the sandboxed agent shell, so non-interactive access uses
`SSH_ASKPASS` + `SSH_ASKPASS_REQUIRE=force`, which needs no PTY.

`deploy_src.sh` mirrors rather than merges, so a file deleted locally does not
stay importable on the cluster. Because the install is editable, updating code
needs only `deploy_src.sh`; a reinstall is required only when `pyproject.toml`
changes. Note that macOS ships `openrsync` (rsync 2.6.9 compatible), which does
not accept `--info=...`; use `--stats`.

An alternative to mirroring is to make the deployed tree a git repository and
`git pull` there, which would also fix the provenance test above.
