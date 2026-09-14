# BasinFlow

Learning-assisted KMC event proposal framework for reaction rate-table
construction.

> **Current stage**: Stage 3 Phase 2 (dual-geometry PaiNN product/event flow + candidate recall).
> Stage 1–3 data, initialization, flow targets, PaiNN, legacy EGNN, sampling, and no-relaxation recall contracts are available.

The 2026-09-10 configuration set is under `configs/0910/`. New runs use
velocity-only supervision, fresh epoch noise, target-free basin sampling and
dynamic pure-PyTorch PBC graphs. `models/painn/{painn,layers,modules}.py` separates
the tensor backbone, interaction layers and EventData adapter. Benchmark outputs
are organized under `/Users/wx/Desktop/benchmark/0910/{smoke,official}`.
These runs evaluate geometric candidates, not saddle-validated events. See
`docs/12_transition1x_split_audit.md` before comparing molecular experiments to
OAReactDiff; its source validation partition and R/P coordinate conventions
differ from the older random 70/15/15 subset experiment.

The recommended configurations are the `*_stable.ini` files (`stability_mode=scaled`).
The files without that suffix reproduce the initial unscaled diagnostics,
including the documented Pt numerical failure; they are not the recommended
starting point. For the prepared local data, run:

```bash
python examples/benchmark_semantic_flow.py --config configs/0910/official_au_stable.ini --system au
python examples/benchmark_semantic_flow.py --config configs/0910/official_pt_stable.ini --system pt
python examples/train_transition1x_product_flow.py --config configs/0910/official_transition1x_stable.ini
```

Choose new output directories before rerunning to retain existing experiment
artifacts. These configurations use 1000 optimizer updates, 16 Euler steps, and
16 candidates per evaluated basin. Au/Pt evaluate their complete saved test
splits; Transition1x evaluates a deterministic 64-reaction subset of the 783
provided validation reactions. This is a fixed-budget benchmark, not a claim of
convergence or an OAReactDiff reproduction. The experiment report is
`docs/13_0910_engineering_report.md`.

The follow-up supports explicit `[data] train_events_dir`, `val_events_dir`
and `test_events_dir` without a `[split]` section, or a shared `events_dir`
with `[split] manifest` without fractions. It validates disjoint basin/reaction
identities and records `data_sources.json`. Existing run directories are not
overwritten. `tools/prepare_transition1x_explicit_split.py` prepares the approved
custom 9000/536/537 partition with previously evaluated reactions pinned to val.
See `docs/17_reference_protocol_audit.md` for the exact export command, reference
protocol differences and current execution status.

`configs/0910/official_pt_coverage.ini` trains five complete Pt epochs;
`official_transition1x_explicit.ini` trains ten complete epochs and evaluates
the full custom validation set. These keep the previous model/prior/loss while
testing increased data coverage; they are not a MolGEN/ReactOT reproduction.
`validate_each_epoch` fixes validation noise/time, `save_each_epoch` saves full
training states, and `resume_from` restores a complete epoch into a new run
directory. Resume budgets are cumulative. Partial-epoch and old weight-only
checkpoints cannot be resumed as exact optimizer states.

The Transition1x R→P RMSD gap is diagnosed in
`docs/22_transition1x_rmsd_root_cause.md`: the σ=0.05 Å prior lets the network
recover the target algebraically from the interpolated position, so the fitted
field is an inversion rather than a velocity field and carries no direction at
`t = 0`. Two repair arms exist,
`configs/0910/conditioned_transition1x_bond_change_x0std1.ini` (MolGEN's
`x0std = 1.0`) and `configs/0910/direct_t0_transition1x_bond_change.ini` (direct
regression at `t = 0`). The same file quantifies why 0.1–0.2 Å is an `R+P→TS`,
Kabsch-aligned target rather than an `R→P` one. `tools/watch_rmsd_curve.py`
scores every epoch checkpoint while a run is in flight;
`tools/diagnose_bond_change_identifiability.py` and
`tools/diagnose_condition_ceiling.py` measure how much the bond-change condition
can buy.

`basinflow train-transition1x-ts` runs the R+P→TS task in parallel with R→P on
the same backbone and data split; see `docs/23_transition_state_task.md`. That
task conditions on both endpoints, and the endpoint geometry is ablatable
between an invariant channel (distances only) and an equivariant one (edge
directions in the vector message):
`configs/0910/transition1x_ts_{invariant,equivariant}.ini`.
`tools/evaluate_transition_state.py` scores its checkpoints.

## Quick Start

All commands below assume the project development environment is the
local conda environment `ifdiff`.

For running the Transition1x workflow on the SUSTech Slurm cluster (conda env
`basinflow`, partition `AMD2`, `sbatch run_basinflow.sbatch`), see
`docs/26_hpc_deployment.md`.

```bash
# Use the project development environment
conda activate ifdiff
python -m pip install -e .

# Optional model dependencies for Stage 3
python -m pip install -e ".[models]"

# Convert raw .con files → ASE trajectory events that preserve FixAtoms
python tools/eon_to_events.py <basin_root_dir>

# Convert an external Transition1x pickle once; all later commands use events_dir
python tools/transition1x_to_events.py <transition1x_train.pkl> <events_dir> --selection use_ind --center

# Run the full data pipeline demo
python examples/demo_pipeline.py <events_dir>

# Train and sample on the Au events folder
python examples/train_product_flow.py --events-dir /Users/wx/Desktop/benchmark/au/events
python examples/train_product_flow.py --events-dir /Users/wx/Desktop/benchmark/au/events --batch-size full
python examples/sample_product_flow.py --events-dir /Users/wx/Desktop/benchmark/au/events

# Config-driven training. The run directory receives:
# checkpoint.pt, split_manifest.json, training.log, config.resolved.ini, eval_metrics.json
python examples/train_product_flow.py --config configs/product_flow_pt.ini

# Direct-basin, no-oracle Stage 3 benchmark with 100 Gaussian trials per
# held-out basin and EAM/FIRE product-relaxation diagnostics.
python examples/benchmark_semantic_flow.py --config configs/semantic_flow_au.ini
python examples/benchmark_semantic_flow.py --config configs/semantic_flow_pt.ini

# Run tests
python -m pytest

# If your shell does not preserve conda activation, use:
conda run -n ifdiff python -m pytest

# Optional real-data Stage 2 integration check
BASINFLOW_EVENTS_DIR=/Users/wx/Desktop/benchmark/au/events \
python -m pytest tests/test_raw_events.py::test_stage2_real_events_dataset_when_env_is_set -q
```

## Implemented Capabilities

| Module | Status |
|--------|--------|
| `StructureRecord` — unified mol/crystal schema, ASE round-trip, canonical `movable_mask` (`True` = movable) | ✅ |
| `EventRecord` — three-frame model (R/P/TS), explicit activity labels or MIC-derived fallback | ✅ |
| `EventCatalog` + `BasinSplit` — read-only basin grouping and persisted basin-only splits (no leakage) | ✅ |
| `CandidateRecord` — schema for generated proposals (later stages) | ✅ |
| EON-style event reader plus EON / Transition1x conversion tools — standard event files → `EventCatalog` | ✅ |
| PBC-aware MIC displacement, active-atom derivation, event-direction extraction | ✅ |
| Dynamic cutoff-graph builder with cell offsets | ✅ |
| `EventFlowDataset` (training) + `BasinDataset` (no-oracle inference) — direct PyG samples | ✅ |
| Initialization/flow target contract with dummy product-event flow | ✅ |
| Minimal torch training loop smoke test (`loss.backward()` + toy overfit) | ✅ |
| Product-flow multi-graph batching with integer or `full` training batch size | ✅ |
| Seed-conditioned EGNN product/event flow with active/direction heads | ✅ |
| Multi-initialization candidate sampler producing `CandidateRecord` outputs | ✅ |
| No-relaxation candidate clustering and basin-level recall report | ✅ |
| Stage 3 train/test quality metrics (`eval_metrics.json`, rollout RMSD, oracle init RMSD) | ✅ |
| Minimal JSON split manifest save/load for reproducible basin splits | ✅ |
| Standard PyG `DataLoader` batching with graph-level `cell`, `pbc`, time, and seed fields | ✅ |
| Unit/integration tests plus optional real-data checks | ✅ |

## Direct-Basin Product-Flow Benchmark

`examples/benchmark_semantic_flow.py` is the reproducible Stage 3 evaluation
entry point for `configs/semantic_flow_au.ini` and
`configs/semantic_flow_pt.ini`.  It trains only with pairwise supervision and
then samples each held-out reactant basin from independent `GaussianInit`
noise.  The sampler receives reactant geometry, cell/PBC, `movable_mask`, and
the seed itself.  It never receives a known product, `target_active_mask`, or
`target_direction` as a model condition.

The run directory contains `checkpoint.pt`, `split_manifest.json`,
`config.resolved.ini`, `training.log`, `loss_by_epoch.csv`, and
`loss_curves.png`.  Its `sampling/` directory contains one CSV record per
trial, raw and EAM-relaxed trajectories, best/median/worst representative
trajectories, per-basin RMSD distributions, and a raw-versus-EAM RMSD parity
plot. `sampling/summary.json` reports raw and EAM-relaxed RMSD distributions,
the improvement fraction, nearest-event switch fraction, relaxation failures,
and wall time.

These measurements are geometric diagnostics against known products in the
same held-out basin.  EAM relaxation can change which known product is nearest;
that switch is recorded explicitly.  Neither a low RMSD nor a successful EAM
relaxation constitutes saddle validation or a validated KMC event.

## TODO

- [x] **Stage 2** — EON-style R/P/TS event dataset integration
- [x] **Geometry invariance tests** — translation, rotation, permutation (required before Stage 3)
- [x] **Stage 3 Phase 1** — minimal trainable product-flow loop
- [x] PyG `Data` bridge for collated batches
- [x] **Stage 3 Phase 2** — EGNN-style seed-conditioned product/event flow backbone
- [x] Minimal JSON metadata persistence for split manifests
- [ ] Improve Stage 3 basin-level initialization/proposal coverage beyond fixed smoke-test initializations
- [ ] **Stage 4** — React-OT-style TS flow
- [ ] **Stage 5** — Basin-level event proposal benchmark
- [ ] **Stage 6** — EON-side ML suggestion integration

See `docs/05_development_plan.md` for the full roadmap.
