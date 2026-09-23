# 28 · ReactOT reproduction: ported components, deviations, remote runbook

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
| `EventData` adapter | `basinflow/models/leftnet/adapter.py` | `LeftNetBridge`: R/TS/P fragment layout, TS-block-only readout, per-sample mean removal |
| Bridge dataset view | `basinflow/data/samples.py::BridgeFlowDataset` | emits `pos = x_t`, `target_scaled = (x_t - x0)/sigma_t`, `flow_time = step/T` |
| Bridge sampler | `basinflow/evaluation/bridge.py::run_bridge_trials` | `progress = 1 - t_sb` reparameterisation of the reference ODE |
| Objective selection | `basinflow/models/loss.py` | `"scaled"` key selects the bridge MSE/L1; velocity runs untouched |
| Backend registration | `basinflow/models/factory.py` | `leftnet`, `leftnet_sb` |
| EMA | `basinflow/training/product_flow.py` | `ema_decay`, `training_state.pt` carries the EMA copy |
| Workflow | `basinflow/workflows/transition_state.py` | bridge branch: dataset, sampler, EMA scoring |
| Data conversion | `tools/transition1x_reactot_to_events.py` | official rpsb pickles → flat `events/` + `split.csv` |
| Reference shim | `tools/reactot_compat/torch_scatter.py` | PyG/torch implementation for the reference repo only |

Tests added: `test_reactot_metrics.py`, `test_reactot_data_conversion.py`,
`test_leftnet_equivariance.py`, `test_sb_bridge.py`, `test_bridge_dataset.py`,
`test_bridge_sampler.py`, `test_ema_training.py`, `test_leftnet_sb_workflow.py`.
Local suite: **339 passed, 1 failed** — the single failure is
`tests/test_stage3_au_scripts.py`, caused by `aten::unique_dim` being
unimplemented on the MPS device (torch 2.8); it passes with
`PYTORCH_ENABLE_MPS_FALLBACK=1` and is unrelated to this work.

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
5. **Fixed gradient clipping.** ReactOT clips adaptively at
   `1.5*mean + 3*std` of a rolling gradient-norm queue seeded with 3000
   (`pl_trainer.py:455-485`); the config uses a fixed `max_grad_norm = 1.0`, as
   the project plan specifies.
6. **`norm = mse`** is the reference's `loss_type = "l2"`; this repo spells the
   L2 objective `mse` (`l1` is the alternative).
7. **`eval_num_steps = 11`** matches the reference's `nfe = 10`
   (`torchdiffeq` is given `nfe + 1` saved points).
8. **Fixed `batch_size = 14`** instead of the reference's dynamic `node^2`
   sampler (`max_num = 2800`).
9. **`is_bridge` has no velocity alias.** The bridge backend deliberately does
   not emit `{"velocity": ...}`, so `validate_each_epoch` is rejected for bridge
   runs; scoring always goes through the bridge sampler.

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

BasinFlow environment:

```bash
conda create -n basinflow python=3.10 -y
conda activate basinflow
cd <basinflow_repo>
python -m pip install -e ".[models]"
python -m pytest -q                     # expect 339 passed, 1 failed (MPS-only, Linux: 340 passed)
```

### 3.2 Data

```bash
python tools/transition1x_reactot_to_events.py \
  react-ot/reactot/data/transition1x/train_rpsb_all.pkl \
  react-ot/reactot/data/transition1x/valid_rpsb_all.pkl \
  data/reactot_transition1x_bridge/events
# -> 10073 events, train=9000 val=0 test=1073, per-frame centroid centred
```

Then point `[data] events_dir` in the config at that `events/` directory
(`auto_split = false`, the tool writes `split.csv`).

### 3.3 Reference baseline (the 0.103 Å number)

```bash
cd react-ot
python evaluation.py --checkpoint reactot-pretrained.ckpt --solver ode --nfe 10
# logs: mean / median RMSD over valid_rpsb_all (1073 reactions)
```

### 3.4 BasinFlow bridge training + scoring

```bash
cd <basinflow_repo>
basinflow train-transition1x-ts --config configs/0910/reactot_ts_leftnet_sb.ini
```

Outputs in `<output_dir>`:

```
checkpoint.pt          scored weights (EMA when ema_decay is set)
training_state.pt      full state incl. optimizer, RNG and EMA copy (resume source)
config.resolved.ini    effective config, incl. the solver actually used
training.log           epoch/step/total_loss/velocity_loss
sampling/summary.json  strict + reactot_reference_protocol blocks
sampling/trial_metrics.csv
```

The config already sets `[evaluation] split = test` (all 1073 set-aside
reactions) and `integrator = midpoint`, `solver_atol/rtol = 1e-2`,
`eval_num_steps = 11`, `trials_per_event = 1`.

### 3.5 Resource notes for the LSF script

* 1 × V100 (16 GB) is enough: the reference config sizes the dynamic batch for
  16 GB, and this config uses a fixed batch of 14 molecules (4–23 atoms each).
* Set `[runtime] cpu_threads = 1` (already the config value). Multi-threaded CPU
  reductions are not bit-reproducible on this stack; with 1 thread two runs of
  the same seed agree.
* `[runtime] device = cuda`; the workflow moves batches to the device itself.
* Expect ~600k optimizer steps (`max_steps = 600000`, i.e. the reference's 3000
  epochs × `limit_train_batches = 200`). `save_each_epoch` is on, so the run can
  be scored while in flight.
* `wandb` is not used by BasinFlow, so the job needs no network access after the
  data download.

## 4. How the result will be judged

1. The reference number is re-established by §3.3 on the same host, so the
   comparison is not against a remembered value.
2. BasinFlow's number is read from `sampling/summary.json`'s
   `reactot_reference_protocol.rmsd_angstrom` (mirror + `min(rmsd, 1)`), with the
   strict `aligned_rmsd_angstrom` reported next to it.
3. Any residual gap must be attributed with the deviations in §2 in hand —
   in particular deviation 1 (the model is not the reference's exact arithmetic)
   and deviation 8 (a different batching rule).

## 5. Open items

* Adaptive gradient clipping (deviation 5) if the fixed clip proves limiting.
* The dynamic `node^2` batch sampler (deviation 8) if the fixed batch distorts
  the per-step node budget on the longer molecules.
* `tools/evaluate_bridge.py` (watch-style scoring of intermediate checkpoints)
  is not implemented; scoring happens once at the end of training. Intermediate
  `training_state.pt` files can be scored with a short script that calls
  `basinflow.evaluation.run_bridge_trials` directly.
