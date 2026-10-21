# 30 · ReactOT reproduction: ported components, deviations, remote runbook

Status: **leftnet_sb pipeline implemented and tested locally (CPU)**; the
training/evaluation runs themselves are executed on a remote V100 host.
Goal: reproduce ReactOT's reported accuracy (Transition1x 1073 set-aside
reactions, mean RMSD ≈ 0.103 Å) and put BasinFlow's own `leftnet_sb` numbers
next to the reference ones under the same metric.

---

## 1. What was changed

| Area | File | Notes |
| --- | --- | --- |
| Reference-protocol RMSD | `basinflow/evaluation/reactot_metrics.py` | mirror (`ignore_chirality=True`) + `min(rmsd, 1)`; strict column unchanged |
| Dual-protocol reporting | `basinflow/evaluation/transition_state.py` | new columns + `reactot_reference_protocol` block |
| Bridge schedule + algebra | `basinflow/models/sb.py` | `SBSchedule`, `q_sample`, `compute_label`, `compute_pred_x0`, `posterior_step`, `make_bridge_ode_field` |
| LEFTNet layers | `basinflow/models/leftnet/layers.py` | `torch_scatter` replaced by local reductions |
| LEFTNet backbone | `basinflow/models/leftnet/leftnet.py` | switch semantics preserved (`reflect_equiv`, `object_aware`, `legacy`, `update`, `single_layer_output`, `pos_grad`) |
| `EventData` adapter | `basinflow/models/leftnet/adapter.py` | `LeftNetBridge`: R/TS/P fragment layout, per-fragment 6->12->6 encoders, TS-block-only readout, per-sample mean removal |
| Reference checkpoint | `basinflow/models/reference.py` | maps `ddpm.dynamics.{model,encoders,decoders}` onto the port; strict 246/246 key match |
| Bridge dataset view | `basinflow/data/samples.py::BridgeFlowDataset` | emits `pos = x_t`, `target_scaled = (x_t - x0)/sigma_t`, `flow_time = step/T` |
| Bridge sampler | `basinflow/evaluation/bridge.py::run_bridge_trials` | `progress = 1 - t_sb` reparameterisation of the reference ODE |
| Objective selection | `basinflow/models/loss.py` | `"scaled"` key selects the bridge MSE/L1; velocity runs untouched |
| Backend registration | `basinflow/models/factory.py` | `leftnet`, `leftnet_sb` |
| EMA | `basinflow/training/loop.py` | optional; the reference callback inverts the coefficient (see §2) |
| Reference batching | `basinflow/training/batch_sampler.py` | `node^2`, `max_num = 2800`, seed 42 |
| Adaptive clipping | `basinflow/training/grad_clip.py` | `1.5*mean + 3*std`, queue 50 seeded with 3000 |
| TS loss scale | `basinflow/models/loss.py` | `scales[1] = 2` |
| Workflow | `basinflow/workflows/transition_state.py` | bridge branch: dataset, sampler, optional EMA scoring |
| Data conversion | `tools/transition1x_reactot_to_events.py` | official rpsb pickles → flat `events/` + `split.csv` |
| Reference shim | `tools/reactot_compat/torch_scatter.py` | PyG/torch implementation for the reference repo only |

Tests covering this path: `test_reactot_metrics.py`,
`test_reactot_data_conversion.py`, `test_leftnet_equivariance.py`,
`test_sb_bridge.py`, `test_bridge_dataset.py`, `test_bridge_sampler.py`,
`test_leftnet_sb_workflow.py`, `test_reference_training.py`. The suite targets the
ReactOT path only; the PaiNN/EGNN baseline code is kept but has no dedicated
workflow tests. Local suite: **94 passed**.

### 1.1 Verified data: what the two pickles actually contain

`tools/audit_reactot_pickles.py` reads both files and reports:

| | train file | valid file |
| --- | --- | --- |
| rows in every array | 10073 | 10073 |
| `use_ind` selection | 9000 | 1073 |
| `single_fragment` over all rows | 7516 true / 2557 false | same arrays |
| `single_fragment` inside `use_ind` | 6733 true / 2267 false | 783 true / 290 false |
| elements | H, C, N, O | same |
| max atoms | 23 | same |
| bad shapes / species mismatches | 0 / 0 | 0 / 0 |

The two files hold the **same 10073 rows**: every role/field matches row by row
(0 mismatches), and `use_ind(train) ∩ use_ind(valid) = ∅`,
`|union| = 10073`. So the training set is **9000**, not 10073; the 1073 set-aside
rows are the test set, and 10073 is only the size of the exported event directory.

The frames are **not** centred in the pickle (max |centroid| ≈ 0.61 A), which is
why `ProcessedTS1x(center=True)` and our converter subtract the per-frame centroid
at load time. Do not skip that step.

### 1.2 Parity check: score the released checkpoint with this port

The public `reactot-pretrained.ckpt` loads into `LeftNetBridge` with no missing or
unexpected keys (246/246). Scoring it on the first 20 test reactions with the
reference protocol gives:

```
count 20  mean 0.0982  median 0.0287  min 0.0077  max 0.5595
```

The reference reports mean 0.103 A / median 0.053 A over all 1073 reactions, so
the port starts in the right place; run all 1073 for the definitive number:

```bash
REACTOT_SRC=/path/to/react-ot python tools/evaluate_reactot_checkpoint.py \
  --events <case>/reactot_events \
  --checkpoint /path/to/reactot-pretrained.ckpt \
  --output <case>/runs/reference_port --steps 11
```

The checkpoint unpickles a `reactot.model.leftnet.LEFTNet` class out of its
hyper-parameters, which is why `REACTOT_SRC` (or `--reference-src`) has to point
at the reference checkout.

**Checkpoints written before the encoder port are incompatible** (the model grew
the per-fragment 6->12->6 encoders/decoders). `load_state_dict` fails loudly;
retrain rather than trying to resume them.

### 1.3 Paper vs released code

Two points in the paper do not match the released artifacts; the code and the
released pickles win, because they are what produced the checkpoint:

| Topic | Paper | Released code / data | This repo |
| --- | --- | --- | --- |
| Initialisation | fine-tuned from the OA-ReactDiff checkpoint, 200 extra epochs at lr 1e-4 | `train_rpsb_ts1x.py` trains from scratch, `max_epochs = 3000`, early stop on `val_ep_scaled_err`; the released ckpt records `epoch=0, global_step=200` | trains from scratch for 600k steps (3000 x 200); `init_checkpoint` can warm-start from a BasinFlow or ReactOT checkpoint, but the OA file is not distributed |
| R/P swap | not stated | training script `False`, released ckpt hyper-parameters `True` | `swap_reactant_product` option, default `False` (script value); validation and scoring always use the canonical order |
| R/P alignment | Kabsch-aligns R and P before building `(R+P)/2` | `ProcessedTS1x` only centers each frame; the released pickles carry arbitrary orientations | follows the pickles: per-frame centroid, no Kabsch. The reference metric is Kabsch-aligned, so orientation is absorbed at scoring time |
| R/P swap | not stated | `swapping_react_prod=False` in the training script, but `True` in the released ckpt hyper-parameters | `False` (script value); the option is not exposed yet |

None of these can be resolved from the paper alone. They are recorded so a
disagreement with 0.103 A can be attributed rather than guessed at.

One more code-level difference: **EMA semantics**. `reactot/trainer/ema.py`
passes `1 - decay` into `ModelEmaV2`, so the training script's
`ema_decay = 0.999` becomes an update rate of `0.001` (`ema <- 0.001*ema +
0.999*model`), i.e. almost the raw weights; the released checkpoint records
`ema: False`. This repo implements standard EMA (`ema <- decay*ema +
(1-decay)*model`), so the reference config template does **not** enable it; a
`[training] ema_decay` value is a deliberate strong-EMA arm, not a copy of the
reference.

## 2. Deviations register (quote these next to any number)

1. **In-sqrt epsilon in LEFTNet.** The reference normalises cross products as
   `sqrt(sum(v**2)) + EPS`. When a cross product is exactly zero — which happens
   for parallel atom pairs in the edge frame and for collinear `(a, b)` in the
   node frame — `sqrt` has an infinite derivative and the backward pass returns
   `nan`, zeroing the gradients of an otherwise well-conditioned batch. Measured
   locally: 4 of 12 edges and 1 of 6 nodes in one toy graph. The port adds
   `EPS_SQ = 1e-12` *inside* those square roots. Forward values move by ≤ ~1e-6
   relative; the model is **not bitwise equivalent** to
   `reactot/model/leftnet.py`.
2. **`torch-scatter` is not installed.** macOS arm64 has no wheel and the source
   build fails. The reference repo runs against `tools/reactot_compat/`
   (verified numerically, not bitwise, against the compiled kernels). The ported
   BasinFlow backend uses local reductions and needs no shim.
3. **`ase` pinned to 3.23.0** in the `reactot` env: 3.29 removed `ase.neb`
   (moved to `ase.mep`), which `reactot/diffusion/_utils.py` imports.
4. **No new BasinFlow dependencies.** The plan listed `torch-scatter`,
   `torch-sparse`, `torch-cluster`, `pytorch-lightning`, `pymatgen`, `networkx`,
   `lmdb`, `wandb`; none is needed after the port, so `pyproject.toml` is
   unchanged. They exist only in the reference environment.
5. **`norm = mse`** is the reference's `loss_type = "l2"`; this repo spells the
   L2 objective `mse` (`l1` is the alternative).
6. **`eval_num_steps = 11`** matches the reference's `nfe = 10`
   (`torchdiffeq` is given `nfe + 1` saved points).
7. **`is_bridge` has no velocity alias.** The bridge backend deliberately does
   not emit `{"velocity": ...}`, so `validate_each_epoch` is rejected for bridge
   runs; scoring always goes through the bridge sampler.

Reference behaviour that is now implemented (no longer listed as a deviation):

* **`node^2` batch sampler.** `basinflow/training/batch_sampler.py` packs a batch
  until `sum(n_atoms ** 2) > 2800`, drawing a fresh permutation per epoch from a
  generator seeded with 42 (`reactot/dataset/sampler.py:105`). The bridge workflow
  selects it by default; `[training] batch_sampler = samples` restores fixed
  batches for the PaiNN/EGNN arm.
* **Adaptive gradient clipping.** `basinflow/training/grad_clip.py` clips at
  `1.5 * mean + 3 * std` of a 50-entry queue seeded with `3000` and records
  `min(grad_norm, bound)` (`pl_trainer.py:455-485`). Default for the bridge;
  `[training] grad_clip = fixed` restores `max_grad_norm`.
* **Plain-MSE objective.** `SBModule.compute_loss` returns `EnSB.forward`'s
  `F.mse_loss(pred, label)`; the `scales = [1, 2, 1]` normalisation belongs to
  `DDPMModule` and does not apply to the bridge (`models/loss.py`).
* **Checkpoint cadence.** `save_each_epoch = true` combined with
  `save_every_n_epochs` (default 10 in the kit) writes a resumable checkpoint every
  N epochs; the training state is always written once at the end. Saving all 3000
  epochs would be several hundred GB.
* **200 batches per epoch.** `[training] limit_train_batches` defaults to 200 for
  the bridge, so 3000 epochs is exactly the reference's 600k steps and the sampler
  reshuffles on the reference's cadence.
* **Per-epoch validation and best-checkpoint selection.** `val_ep_scaled_err` is
  the per-batch `mean(|(pred-label)/max|label||)` averaged over the first
  `limit_val_batches = 20` validation batches, with the reference's `3 * 2800`
  node budget; the epoch with the lowest value is saved to `best.pt`, restored
  before scoring, and `early_stopping_patience` (default 2000 epochs) stops a
  plateauing run. The reference's `valid_rpsb_all` is our `test` split, so the
  default `validation_split` is `test` when `val` is empty.
* **R/P swap augmentation.** `[training] swap_reactant_product = true` swaps the
  two conditioning endpoints while keeping the midpoint prior and the TS target;
  validation and scoring stay in the canonical order.
* **Warm start.** `[training] init_checkpoint` loads a BasinFlow checkpoint or the
  released ReactOT `.ckpt` before training.

## 3. Remote runbook (V100 host)

### 3.1 Environments

Reference environment (for `evaluation.py` and the official baseline):

```bash
git clone https://github.com/deepprinciple/react-ot.git react-ot
cd react-ot
conda env create -f env.yaml            # python 3.10.14, torch 2.2.1, PL 2.4.0
conda activate reactot
pip install -e .
pip install ase==3.23.0                 # deviation 3
# torch-scatter has no wheel; put the shim first on PYTHONPATH (deviation 2)
export PYTHONPATH=<basinflow_repo>/tools/reactot_compat:$PYTHONPATH
mkdir -p reactot/data/transition1x
# download from https://zenodo.org/records/13131875 :
#   train_rpsb_all.pkl, valid_rpsb_all.pkl, sb-pretrained.ckpt
```

That block is the interactive form. `conda env create -f env.yaml` does not work
as written: `torch-scatter` / `torch-sparse` / `torch-cluster` ship no wheels, so
pip falls back to a source build (it also fails because the build environment has
no torch). Only `torch_scatter` is imported by the evaluation path, and the shim
covers it, so `run_reactot_reference.lsf` (§3.5) builds the environment by hand
from the same package list minus those three. `pip install -e .` is not needed
for the reference either: the runs start `evaluation.py` from the checkout root,
which is also what its hardcoded `datadir = "./reactot/data/transition1x"`
resolves against.

BasinFlow environment:

```bash
conda create -n basinflow python=3.10 -y
conda activate basinflow
# Install the CUDA-matched torch *first*. The `models` extra resolves torch from
# PyPI, i.e. the default CUDA-less wheel, so `pip install -e ".[models]"` on a GPU
# host can replace a CUDA build that was already there:
python -m pip install torch torch_geometric torchdiffeq
cd <basinflow_repo>
python -m pip install -e .              # the core needs only ase + numpy
python -m pytest -q                     # expect 94 passed
```

`torch`, `torch_geometric` and `torchdiffeq` live in the `models` extra of
`pyproject.toml`, which is why `pip install -e ".[models]"` is the one-liner for a
CPU-only box — that is how the local macOS environment is built. The two commands
above are equivalent on a GPU host as long as those three packages import; the
model, sampling and evaluation modules are the only ones that need them.

### 3.2 Data and the case directory

A run is described by a case directory that holds its own config and its own
data, the same shape as `~/Desktop/benchmark/au/train`:

```text
<case>/
  config.ini                    the task input; the ONLY place paths are set
  reactot_events/               the converted events directory (10073 traj + split.csv)
  runs/<name>/                  a fresh output directory per submission
```

The LSF submission scripts live in the separately distributed kit, not here.

Convert the official pickles into that directory:

```bash
python tools/transition1x_reactot_to_events.py \
  react-ot/reactot/data/transition1x/train_rpsb_all.pkl \
  react-ot/reactot/data/transition1x/valid_rpsb_all.pkl \
  <case>/reactot_events
# -> 10073 events; split.csv: train 9000, val 0, test 1073; per-frame centroid centred
```

`split.csv` is written next to the event files, so the config keeps
`auto_split = false` and `[data] events_dir` points at `<case>/reactot_events`.
The tool does **not** align product to reactant: ReactOT centres each frame and
nothing else, and the bridge initialises from `(R + P) / 2`, so an extra Kabsch
step would move the initial condition away from the reference's.

The local macOS case already exists:

```text
/Users/wx/Desktop/benchmark/0910/reactot/
  config.ini
  reactot_events -> /Users/wx/Desktop/benchmark/0910/data/reactot_transition1x_bridge/events
  runs/
```

On another host, recreate the same entries with that host's paths. There is no
repo-side config template: the config lives in the case directory, so no
machine's absolute paths enter version control. The canonical reference config
is:

```ini
[data]
events_dir = <case>/reactot_events
auto_split = false

[output]
output_dir = <case>/runs/leftnet_sb
log_to_stdout = true

[model]
backend = leftnet_sb
num_features = 196
num_layers = 6
num_radial = 96
cutoff = 10.0
num_elements = 5
reflect_equiv = true
legacy = true
update = true
single_layer_output = true
object_aware = true

[bridge]
timesteps = 3000
beta_max = 0.3
power = 0.5
inv_power = 1

[training]
epochs = 3000
max_steps = 600000
lr = 1.0e-4
batch_size = 14
optimizer = adamw
amsgrad = true
weight_decay = 0.0
save_each_epoch = true
save_every_n_epochs = 10
batch_sampler = node2
max_num = 2800
grad_clip = adaptive
limit_train_batches = 200
validate_each_epoch = true
validation_split = test
limit_val_batches = 20
val_max_num = 8400
early_stopping_patience = 2000
swap_reactant_product = false
# init_checkpoint = /path/to/reactot-pretrained.ckpt

[loss]
velocity_weight = 1.0
norm = mse

[runtime]
device = cuda
cpu_threads = 1

[evaluation]
eval_num_steps = 11
trials_per_event = 1
sampling_seed = 20260910
split = test
integrator = midpoint
solver_atol = 1.0e-2
solver_rtol = 1.0e-2
```

`batch_sampler`, `max_num`, `grad_clip`, `validate_each_epoch`, `validation_split`,
`limit_val_batches` and `val_max_num` default to the reference behaviour for
`leftnet_sb`; they are spelled out here so the run is self-describing.
`max_grad_norm` is only used by the fixed-clip arm. `init_checkpoint` is optional:
it accepts a BasinFlow checkpoint or the released ReactOT `.ckpt` (set
`REACTOT_SRC` so the class can be unpickled).

### 3.3 Reference baseline (the 0.103 Å number)

```bash
cd react-ot
python evaluation.py --checkpoint reactot-pretrained.ckpt --solver ode --nfe 10
# logs: mean=..., median=..., len(rmsds)=... over valid_rpsb_all (1073 reactions)
```

On the cluster this is `run_reactot_reference.lsf` (§3.5), which runs exactly that
command on the `8V100-32-sc` queue and moves the two artifacts `evaluation.py`
writes relative to its working directory (`.log/`, `ode-midpoint/nfe10/`) into
`<case>/runs/reference_<jobid>/`. `--dryrun` evaluates the first batch only and is
the cheap check that the checkpoint and data load at all.

### 3.4 BasinFlow bridge training + scoring

```bash
cd <basinflow_repo>
basinflow train-transition1x-ts --config <case>/config.ini
```

`<case>` is the directory from §3.2; its `config.ini` already sets
`[output] output_dir = <case>/runs/leftnet_sb`, so the run writes next to its own
data. Use a new `output_dir` per submission (see §3.2).

Outputs in `<output_dir>`:

```
checkpoint.pt          scored weights (best val_ep_scaled_err epoch)
best.pt                best epoch by val_ep_scaled_err (model state + metric)
training_state.pt      full state incl. optimizer, RNG and EMA copy (resume source)
config.resolved.ini    effective config, incl. the solver actually used
training.log           epoch/step/total_loss/velocity_loss
sampling/summary.json  strict + reactot_reference_protocol blocks
sampling/trial_metrics.csv
```

The config already sets `[evaluation] split = test` (all 1073 set-aside
reactions) and `integrator = midpoint`, `solver_atol/rtol = 1e-2`,
`eval_num_steps = 11`, `trials_per_event = 1`.

### 3.5 LSF submission

The LSF kit is distributed separately from this repository (`scripts/train.lsf`,
`check.lsf`, `score.lsf`, `reference.lsf`, `bootstrap.sh`, `convert.sh` and a
Chinese runbook). It defaults to queue `4a100-40`, one GPU, four CPU slots, a 24 h
wall time, `resume_from = auto` and per-epoch validation. See its `README_ZH.md`
for the exact commands; the case directory only holds `config.ini`, the events
and `runs/`.

## 4. How the result will be judged

1. The reference number is re-established by §3.3 on the same host, so the
   comparison is not against a remembered value.
2. BasinFlow's number is read from `sampling/summary.json`'s
   `reactot_reference_protocol.rmsd_angstrom` (mirror + `min(rmsd, 1)`), with the
   strict `aligned_rmsd_angstrom` reported next to it.
3. Any residual gap must be attributed with the deviations in §2 in hand —
   in particular the in-sqrt epsilon and the edge construction (the model is not
   bitwise equivalent to `reactot/model/leftnet.py`).

## 4.1 Cluster deployment

The LSF kit is distributed separately from this repository (it is not under
version control here). It contains `bootstrap.sh`, `convert.sh`, `check.lsf`,
`train.lsf`, `score.lsf`, `reference.lsf`, a `config.template.ini` and a Chinese
runbook. It targets one GPU, `python/anaconda3/2022.10`, `resume_from = auto` and
a 24 h wall time.

## 5. Open items

* **OA-ReactDiff warm start is unavailable.** The paper fine-tunes from that
  checkpoint; only the already-fine-tuned ReactOT `.ckpt` is distributed, so a
  from-scratch run cannot reproduce the paper's initialisation exactly. The
  `init_checkpoint` hook is in place for a future OA file.
* **`val_ep_scaled_err` is a sampled metric.** Each epoch samples new bridge times
  and uses the first 20 validation batches, exactly like the reference; two runs
  can therefore select different epochs even with identical weights.
* **EMA is a deliberate arm.** The reference callback's coefficient is effectively
  `0.001`; the default is raw weights. Do not enable a strong `ema_decay` when
  comparing against the reference number.
* **The full 1073-reaction parity check** for the released checkpoint has only
  been run on 20 reactions locally; run
  `tools/evaluate_reactot_checkpoint.py` on the GPU host for the definitive value.
