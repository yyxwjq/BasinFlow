# BasinFlow — ReactOT reproduction

A minimal reimplementation of **ReactOT** ([paper](https://arxiv.org/abs/2404.13430),
[reference repo](https://github.com/deepprinciple/react-ot)) on top of an
`EventCatalog` data layer, plus the PaiNN/EGNN R+P→TS velocity-flow baseline that
was already in the project.

The task is `R + P -> TS`: both endpoints are inputs, the recorded transition
state is the target. The ReactOT backend (`leftnet_sb`) is a deterministic
Schrödinger bridge — prior `(R + P) / 2`, `sigma = 0`, `ot_ode = True` — scored by
the reference RMSD protocol (mirror allowed, capped at 1 Å).

## Data

Convert the official Zenodo pickles (`train_rpsb_all.pkl`, `valid_rpsb_all.pkl`)
into a flat event directory:

```bash
python tools/transition1x_reactot_to_events.py \
  react-ot/reactot/data/transition1x/train_rpsb_all.pkl \
  react-ot/reactot/data/transition1x/valid_rpsb_all.pkl \
  /path/to/case/reactot_events
```

This writes 10073 three-frame `event_<n>.traj` files (R, P, TS), a `split.csv`
with train 9000 / val 0 / test 1073, per-frame centroid centering, no Kabsch
alignment, `pbc = False`, and an all-True `movable_mask`. No basin table, no
fragment filtering: ReactOT trains and evaluates on all 10073 rows
(`single_frag_only = False` in the reference trainer).

An empty validation partition is legal — `split.csv` simply leaves the `val`
column empty, and the workflow only selects `val` when it is actually used.

## Training and scoring

```bash
basinflow train-transition1x-ts --config <case>/config.ini
```

The config lives in the case directory (no repo-side templates). The canonical
reference config — paths plus the 3000-epoch recipe — is in
`docs/30_reactot_reproduction.md` §3.2. For `leftnet_sb`, batching and clipping
already default to the reference: `sum(n_atoms ** 2) <= 2800` and adaptive
`1.5*mean + 3*std` clipping, with the TS loss scale of 2.

The run writes `checkpoint.pt` (raw weights by default; EMA when `ema_decay` is set),
`training_state.pt`, `config.resolved.ini`, `training.log`, and
`sampling/summary.json` + `sampling/trial_metrics.csv`. The summary carries both
protocols: `aligned_rmsd_angstrom` (strict, proper rotation) and
`reactot_reference_protocol` (mirror + `min(rmsd, 1)`) — compare the latter with
the reference numbers.

The PaiNN/EGNN baselines use the same workflow with `[model] backend = painn` or
`egnn`; they keep fixed batches and fixed clipping. Those runs optimize a
velocity field and are scored by `evaluation/transition_state.py`.

## Layout

```
basinflow/
  data/           Structure, Event/EventCatalog, split.csv, sample views
  geometry/       MIC, bond graphs, tensor neighbour graphs
  models/         LEFTNet (ReactOT), PaiNN, EGNN, bridge algebra, loss
  training/       shared training loop, EMA, loss plot
  evaluation/     bridge sampler, velocity-flow sampler, RMSD protocols
  workflows/      config loading, data resolution, R+P->TS orchestration
tools/            ReactOT pickle audit/converter, checkpoint scorers, reference shim
docs/30_reactot_reproduction.md   data audit, ports, deviations, remote runbook
tests/            contracts for the converter, bridge, sampler, scorer, workflow
```

## Tests

```bash
python -m pytest -q
```

The suite covers the ReactOT path only (94 tests). `torch`, `torch_geometric` and
`torchdiffeq` are needed for the model and workflow tests
(`pip install -e ".[models]"`).

## Checking parity with the released checkpoint

`basinflow/models/reference.py` maps the public `reactot-pretrained.ckpt` onto the
port (246/246 keys) and `tools/evaluate_reactot_checkpoint.py` scores it with the
reference protocol. On the first 20 test reactions it gives mean 0.098 A / median
0.029 A (reference: 0.103 A mean over all 1073).

```bash
REACTOT_SRC=/path/to/react-ot python tools/evaluate_reactot_checkpoint.py \
  --events <case>/reactot_events \
  --checkpoint /path/to/reactot-pretrained.ckpt \
  --output <case>/runs/reference_port
```

## Deviations from the reference

Recorded in `docs/30_reactot_reproduction.md` §2: in-sqrt epsilon in the LEFTNet
port, the `torch-scatter` shim for the reference repo, and the `nfe`/saved-point
convention. Batching, adaptive clipping, the plain-MSE bridge objective, per-epoch
`val_ep_scaled_err` selection and the R/P-swap option now match the reference.
The remaining gap is the paper's OA-ReactDiff initialisation (that checkpoint is
not distributed).
These make the run **not bitwise equivalent** to `react-ot/model/leftnet.py`;
the reference number is re-established on the same host before comparing.
