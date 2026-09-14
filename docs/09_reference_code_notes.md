# Reference Code Notes

## Explicit-partition and reference-protocol follow-up

`17_reference_protocol_audit.md` records the inspected ReactOT and MolGEN
executable paths. ReactOT's R/P midpoint-to-TS schedule, MolGEN's origin-Gaussian
prior and L1/Beta-time training, and their reflection/capped RMSD conventions
are distinct from the current R-only displacement flow. No reference code was
copied. In particular, reference TS/product-fragment conditions must not enter
ordinary BasinFlow reactant-only sampling. The custom external-partition loader
and epoch validation/checkpoint facilities improve reproducibility; they do not
claim reference-model reproduction.

## PaiNN engineering update (2026-09-10)

Inspected sources:

- `liflow/liflow/model/models.py`: DualPaiNN reference/current geometries.
- `liflow/liflow/model/layers.py`: scalar/vector interaction and update equations.
- `liflow/liflow/model/modules.py`: graph-time broadcasting and clean R conditioning.
- `AdsorbDiff/adsorbdiff/models/gemnet_oc/layers/radial_basis.py`: radial basis choices and the polynomial envelope.
- `AdsorbDiff/adsorbdiff/models/painn/painn_denoising.py`: periodic vector prediction.

Local implementation uses these mathematical/design ideas without importing a
reference module. `DualPaiNN` lives in `models/painn/painn.py`; `PaiNN` in
`modules.py` is the public BasinFlow adapter. Scalar state is `[N,F]`, vector
state `[N,3,F]`; learned vector linear maps act only on feature channels.
Readout gates are invariant and multiply a full Cartesian vector for each atom.
Two independent radial filters consume R and x_t distances. Cutoffs multiply
filter outputs, including biases, and gate each directional branch separately.
Bessel embeddings use the finite sinc limit at zero; polynomial envelopes have
zero first and second derivative at cutoff. No torch_scatter is required.

`graphs/tensor_neighbors.py` implements exact directed multi-image PBC topology
with bounded pair chunks. Reference/current edge unions avoid losing conditioning
when a contact leaves the current cutoff. Unit tests compare topology to ASE,
check strict componentwise O(3) equivariance, translations, permutations, mixed
batches, image shifts, gradients, empty graphs and checkpoint restoration.

The task does not use LiFlow's temperature or physical-lag conditions: it is
basin-event proposal, not finite-time MD prediction. Source Gaussian noise enters
only x_0; velocity-only loss and displacement-derived event semantics supersede
the previous persistent-noise/three-head default. Full framework details are in
`02_architecture.md`, source partition evidence in `12_transition1x_split_audit.md`.

Explicit backend/config checkpoint metadata is required for new experiments.
`models/factory.py` supports legacy EGNN loading, but the abandoned draft PaiNN
weights are not architecture-compatible. Never silently load them non-strictly.

The initial LiFlow-style unscaled multiplicative updates overflowed during real
Pt training on a valid long-displacement event. The current `stability_mode=scaled`
uses invariant LayerNorm on scalar message inputs and concatenated scalar/norm
update inputs, feature-count scaling of vector messages and scalar dot products,
and residual variance scaling. Vector Cartesian components are never normalized
independently. This extends AdsorbDiff's scale-control ideas; it is not a claim
of bitwise equivalence to its model. `stability_mode=none` is retained only for
reproducing the first-run diagnostics; missing mode in older PaiNN checkpoint
metadata explicitly resolves to `none`. Both behaviors are tested. See
`15_painn_numerical_audit.md` for the saved failing-batch analysis.

## Purpose

This document records concrete code-level lessons from the reference projects. It is intended for Codex, Claude, and future contributors who need to implement FS-CGP modules efficiently without rediscovering the same architecture details.

Reference code should be inspected and selectively adapted. Do not copy entire repositories into FS-CGP.

## OAReactDiff

Path:

```text
/Users/wx/Desktop/yyxwjq/OAReactDiff
```

Important files:

```text
oa_reactdiff/dataset/transition1x.py
oa_reactdiff/dataset/base_dataset.py
oa_reactdiff/diffusion/en_diffusion.py
oa_reactdiff/dynamics/egnn_dynamics.py
oa_reactdiff/dynamics/confidence.py
oa_reactdiff/model/egnn.py
oa_reactdiff/model/leftnet.py
oa_reactdiff/trainer/pl_trainer.py
oa_reactdiff/evaluate/evaluate_ts_w_rp.py
oa_reactdiff/evaluate/generate_confidence_sample.py
oa_reactdiff/utils/_graph_tools.py
oa_reactdiff/utils/sampling_tools.py
oa_reactdiff/analyze/rmsd.py
```

Reusable ideas:

- Pairwise reaction sample handling for reactant/product/transition-state data.
- EGNN/LEFTNet-style equivariant molecular dynamics modules.
- Diffusion training and sampling loop organization.
- Confidence model pattern for post-generation filtering.
- RMSD and product/TS evaluation utilities for molecular benchmarks.

FS-CGP mapping:

- `data/`: use OAReactDiff as a reference for pairwise event sample loaders, but adapt to multi-frame `event.extxyz` / `event.traj` raw inputs.
- `models/`: reuse the idea of an equivariant backbone, but keep the interface independent from OAReactDiff-specific molecular assumptions.
- `evaluation/`: use its RMSD-style utilities only for molecule cases; crystal/product-basin evaluation needs PBC-aware matching and relaxation.
- `validation_priority/`: the confidence model idea can become a future candidate-ranking or validation-priority head.

Do not copy directly:

- Static molecular graph assumptions.
- One-to-one reaction framing as the final benchmark.
- Direct product/TS RMSD metrics as the only success criterion.
- Dataset-specific Transition1x pickle assumptions.

Transition1x converter implemented in BasinFlow (2026-09-01):

- `tools/transition1x_to_events.py` is a standalone converter for OAReactDiff's
  pickle schema.  It validates the three columnar frame dictionaries and
  writes standard BasinFlow event files plus `basin_table.csv` from atomic
  numbers, coordinates, optional energies/forces, `use_ind`, and
  `single_fragment` metadata.
- No OAReactDiff source was copied.  The only reference used was the data
  layout exposed by `BaseDataset` and `ProcessedTS1x`.
- The converter has an explicit `--center` option.  It subtracts each frame's
  unweighted centroid before writing, preventing arbitrary global translation
  from becoming a product-flow target for multi-fragment rows.
- This molecular experiment uses one event per pseudo-basin and reports only
  pairwise, no-relaxation geometric diagnostics.  It must not be presented as
  a basin-level event-recall or KMC saddle-validation benchmark.

Key design warning:

OAReactDiff is a useful baseline for molecular reaction generation, but FS-CGP must be basin-level and periodic-aware.

## akmc-product-generation

Path:

```text
/Users/wx/Desktop/yyxwjq/akmc-product-generation
```

Former name:

```text
akmcgc
```

Important files:

```text
akmcpg/dataset/reaction_pair_dataset.py
akmcpg/utils/graph.py
akmcpg/denoising/denoiser.py
akmcpg/denoising/base_denoiser.py
akmcpg/denoising/confidence.py
akmcpg/diffusion/diff.py
akmcpg/diffusion/diff_sched.py
akmcpg/diffusion/diff_utils.py
akmcpg/diffusion/norm.py
akmcpg/model/egnn.py
akmcpg/model/leftnet.py
akmcpg/trainer/train_diff.py
akmcpg/trainer/train_conf.py
tests/data/h2o_react.extxyz
tests/data/h2o_product.extxyz
tests/data/nacl_crystal_react.extxyz
tests/data/nacl_crystal_product.extxyz
docs/_2026-05-11_rxn_dataset_batch_design.md
docs/_current_debug_overview_and_roadmap.md
```

Reusable ideas:

- Periodic adaptation of OAReactDiff-style joint reactant/product graph.
- ASE/extxyz input direction.
- Data fields for `cell`, `pbc`, and `cell_offsets`.
- PBC-aware distance handling experiments.
- `ReactionPairDataset` clearly defines a periodic reactant/product joint graph.
- Each item concatenates reactant and product nodes, with `fragment=0` for reactant and `fragment=1` for product.
- Graph edges are intra-fragment only: reactant graph and product graph are built separately, then concatenated.
- `collate_fn` distinguishes `fragment` from `mask`: `fragment` is local reactant/product membership, while `mask` is batch id.
- `cell` shape convention is explicit: `[2, 3, 3]` for one sample and `[B, 2, 3, 3]` for batched samples.
- `pbc`, `cell_offsets`, and `neighbors` are propagated through dataset, denoiser, and diffusion.
- `utils/graph.py` is a useful single source of truth for ASE neighbor-list construction, cell offsets, and PBC distance recovery.
- `Denoiser.forward()` shows how PBC tensors should pass into an EGNN/LEFTNet backbone.
- `Diffusion.forward()` separates diffusion math from neural denoising and uses `mask` for one timestep/noise level per reaction sample.
- Tests include both molecule-like H2O and periodic NaCl fixtures.

Limitations:

- Static graph and static cell-offset usage during sampling.
- No full dynamic periodic neighbor update.
- No basin-level multi-event event-set handling.
- Still close to one-to-one reaction/product generation.
- Raw input currently uses separate reactant and product files rather than FS-CGP's preferred one-event multi-frame file.
- Joint graph is useful for pairwise training, but FS-CGP's inference input should be reactant-only.

FS-CGP mapping:

- `data/raw_events.py`: adapt the reactant/product pair logic to read frame 0 and frame 1 from the same event file.
- `data/collate.py`: use the `fragment` versus `mask` distinction exactly; it avoids many batched PBC bugs.
- `graphs/pbc.py`: use the documented tensor contracts from `akmcpg/utils/graph.py` for `edge_index`, `cell_offsets`, `cell`, `fragment`, and `mask`.
- `models/denoiser.py`: use the denoiser wrapper pattern if building a diffusion baseline.
- `tests/fixtures/`: reuse the idea of molecule and NaCl periodic fixtures.

Do not copy directly:

- Any sampling loop that keeps reactant graph edges fixed while coordinates move.
- Any assumption that each reactant has exactly one product.
- The separate `react_file` / `product_file` raw dataset interface as the only supported format.

## AdsorbDiff

Path:

```text
/Users/wx/Desktop/yyxwjq/AdsorbDiff
```

Important files:

```text
adsorbdiff/trainers/sde_denoising_trainer.py
adsorbdiff/utils/atoms_to_graphs.py
adsorbdiff/datasets/lmdb_dataset.py
adsorbdiff/modules/scheduler.py
adsorbdiff/modules/loss.py
adsorbdiff/modules/transforms.py
adsorbdiff/relaxation/ml_relaxation.py
adsorbdiff/relaxation/calculator.py
adsorbdiff/placement/adsorbate.py
adsorbdiff/placement/slab.py
adsorbdiff/placement/adsorbate_slab_config.py
configs/denoising/painn_conditional.yml
configs/denoising/gemnet_so3.yml
docs/architecture.md
docs/placement.md
docs/workflows.md
```

Reusable ideas:

- Adsorbate/surface local generation rather than unconstrained full-cell generation.
- PBC graph conversion with `edge_index` and `cell_offsets` in `AtomsToGraphs`.
- Noise schedules specialized for adsorbate center-of-mass translation and rotation.
- Generation followed by ML relaxation.
- Placement utilities for adsorbate and slab structures.

FS-CGP mapping:

- `graphs/`: `AtomsToGraphs.convert()` is a useful reference for ASE-to-PyG conversion with `cell`, `pos`, `atomic_numbers`, `tags`, `edge_index`, and `cell_offsets`.
- `sampling/`: AdsorbDiff's local translation/rotation noise suggests FS-CGP seeds should be local event-direction seeds, not only global Gaussian noise.
- `relaxation/`: `ml_relaxation.py` and `calculator.py` are useful references for an ASE-backed relaxation interface.
- `surface_events/`: placement modules can inform adsorbate/surface event benchmarks.

Do not copy directly:

- Adsorbate-only assumptions such as tag-specific movement if implementing general bulk/defect events.
- LMDB as the required first data format. FS-CGP raw data should remain simple multi-frame event files.
- Relaxed adsorbate placement metrics as the only benchmark.

Key design warning:

AdsorbDiff demonstrates the correct pattern of generation plus relaxation, but FS-CGP must generalize from adsorption configurations to KMC event channels.

## liflow

Path:

```text
/Users/wx/Desktop/yyxwjq/liflow
```

Important files:

```text
liflow/utils/prior.py
liflow/utils/inference.py
liflow/utils/geometry.py
liflow/model/models.py
liflow/model/layers.py
liflow/data/dataset.py
liflow/experiment/train.py
liflow/experiment/test.py
liflow/config/train.yaml
liflow/config/test.yaml
```

Reusable ideas:

- Conditional flow/propagation framing.
- Prior classes: normal, uniform-scale normal, Maxwell-Boltzmann, and adaptive Maxwell-Boltzmann.
- Temperature-conditioned model inputs.
- Dual-geometry model design in `DualPaiNN`, where the model sees two coordinate states and predicts a vector field.
- Priors with physical meaning through temperature, mass, and atom type.

FS-CGP mapping:

- `inits/`: adapt `Prior` concepts into event seeds such as Gaussian displacement, Maxwell-Boltzmann pseudo-velocity, active-region seed, and event-library motif seed.
- `models/`: `DualPaiNN` is a strong reference for models that compare two coordinate states, useful for flow matching between seed/intermediate state and product.
- `configs/`: preserve a clean config-driven interface for training and inference.
- `data/pyg.py`: borrow only liflow's clear single-sample dataset boundary:
  the source catalog creates domain records, a view creates one PyG sample,
  and PyG owns batching.  This replaces BasinFlow's former repeated
  pairwise/flow/torch dictionary transformations.

Do not copy directly:

- Li-ion-specific assumptions from adaptive priors.
- Any target that assumes ordinary diffusion trajectory data rather than event samples.
- Static periodic graph construction or trajectory-time-delay semantics.

Key design warning:

liflow helps give `seed` a mathematical and physical role. For FS-CGP, a seed should represent an exploration direction or pseudo-dynamical initial condition, not a product label.

## TrajCast

Path:

```text
/Users/wx/Desktop/yyxwjq/trajcast
```

Important files:

```text
trajcast/data/trajectory.py
trajcast/data/atomic_graph.py
trajcast/data/dataset.py
trajcast/model/forecast.py
trajcast/model/models.py
trajcast/model/losses.py
trajcast/model/training.py
trajcast/model/forecast_tools.py
trajcast/nn/modules.py
trajcast/utils/atomic_computes.py
trajcast/validation/physical_behaviour.py
```

Reusable ideas:

- Multi-frame ASE trajectory ingestion in `ASETrajectory`.
- Automatic computation of displacement and next-frame fields.
- `AtomicGraph.from_atoms_dict()` for periodic and non-periodic graph construction.
- `AtomicGraph.update_edge_index()` for dynamic neighbor updates.
- Rollout engine pattern in `Forecast`.
- Position wrapping after each prediction step.
- Velocity-conditioned equivariant message passing.
- Momentum/temperature-style physical constraints for trajectory models.

FS-CGP mapping:

- `data/raw_events.py`: use the `ASETrajectory` pattern to read `event.extxyz` or `event.traj` with frame 0 as reactant and frame 1 as product.
- `geometry/mic.py`: use MIC displacement logic similar to `align_vectors_with_periodicity`, but implement and test project-local code.
- `graphs/atomic_graph.py`: use `AtomicGraph` as a reference for dynamic PBC graph updating.
- `sampling/candidate_sampler.py`: use `Forecast` as a reference for a config-driven rollout/sampling engine.
- `inits/`: velocity-conditioned design supports physically meaningful pseudo-velocity or local-perturbation seeds.

Do not copy directly:

- TrajCast's objective of learning ordinary MD time evolution as the main FS-CGP target.
- Full trajectory validation metrics as KMC event-discovery metrics.
- Any assumption that long MD rollout alone is sufficient to discover rare KMC events.

Key design warning:

TrajCast is a learned dynamics propagator. FS-CGP can borrow its rollout and dynamic graph patterns, but KMC event discovery still needs relaxation, clustering, and saddle validation.

## EGNN / egnn-pytorch

Paths:

```text
/Users/wx/Desktop/yyxwjq/egnn
/Users/wx/Desktop/yyxwjq/egnn-pytorch
```

Important files:

```text
egnn/models/egnn_clean/egnn_clean.py
egnn/models/gcl.py
egnn-pytorch/egnn_pytorch/egnn_pytorch_geometric.py
egnn-pytorch/tests/test_equivariance.py
```

Reusable ideas:

- `E_GCL` structure: edge MLP over source/target node features plus distance, node scatter aggregation, and coordinate/vector update by multiplying edge vectors with learned scalar weights.
- Explicit `edge_index` convention with shape `[2, E]`.
- Equivariance testing strategy for translated and rotated inputs.
- Sparse graph interface ideas from `egnn_pytorch_geometric.py`.

FS-CGP mapping:

- `models/egnn_product_flow.py`: implement a project-local EGNN core using plain PyTorch scatter operations and BasinFlow's PBC-aware `edge_vectors`.
- `graphs/torch_graph.py`: keep neighbor-list and PBC image handling outside the model.
- `tests/test_egnn_product_flow.py`: verify output shape, mask behavior, translation invariance, rotation equivariance, and toy overfit behavior.

Do not copy directly:

- Dense fully connected graph builders as the default sampling graph.
- PyG `MessagePassing` internals as the first production model path.
- Dataset-specific QM9/N-body assumptions.

## AMDEN

Path:

```text
/Users/wx/Desktop/yyxwjq/AMDEN-code
```

Important files:

```text
src/data.py
src/neighborlists.py
src/pipeline.py
src/utils.py
src/models/layers.py
src/models/modules/material_schedule.py
src/models/modules/noise_schedules.py
src/models/denoisers/egnn_deriv.py
```

Reusable ideas:

- `Sample` and `Batch` structure for material samples with positions, elements, lattice, PBC, and optional properties.
- ASE/extxyz dataset loading with optional JSON/YAML property conditioning.
- Dynamic neighborlist with cached rebuild logic in `Neighborlist`.
- `positions_into_cell()` for wrapping positions into the simulation cell.
- Diffusion inference with optional trajectory saving.
- Restart/repeat logic for invalid generated samples.
- HMC refinement inside denoising for low-energy relaxed amorphous structures.
- Property-null masking for classifier-free guidance style conditioning.

FS-CGP mapping:

- `data/records.py`: `Sample` is a useful reference for minimal structure object fields, but FS-CGP should keep event-specific records separate.
- `graphs/neighborlist.py`: implement cached neighbor rebuild inspired by `Neighborlist.get_edges()`, with tests for moving atoms and PBC offsets.
- `refinement/`: HMC denoising motivates a future `RefinementRunner` interface, but MVP can start with ASE relaxation.
- `sampling/`: restart invalid candidates can become a candidate-resampling policy.
- `conditioning/`: property-null masking suggests a future way to train conditional/unconditional event proposal heads in one model.

Do not copy directly:

- Ghost atom mechanism for MVP. FS-CGP event samples normally preserve atom identity and atom count.
- Full amorphous inverse-design objective.
- Treat generated structures as final without relaxation.

Key design warning:

AMDEN shows that standard denoising may fail to generate low-energy relaxed configurations. FS-CGP should define the output as relaxed and clustered product basins, not raw generated coordinates.

## Cross-Reference Implementation Guidance

### Geometry and MIC

Read:

```text
trajcast/utils/atomic_computes.py
AMDEN-code/src/utils.py
AdsorbDiff/adsorbdiff/trainers/sde_denoising_trainer.py
```

Implement in FS-CGP:

```text
src/basinflow/geometry/mic.py
tests/test_mic.py
```

Requirements:

- Molecular no-PBC case returns direct displacement.
- Periodic case returns MIC displacement.
- Boundary-crossing cases are tested.
- Non-orthogonal 3x3 cells should be supported where feasible.

### Dynamic PBC Graphs

Read:

```text
trajcast/data/atomic_graph.py
AMDEN-code/src/neighborlists.py
AdsorbDiff/adsorbdiff/utils/atoms_to_graphs.py
```

Implement in FS-CGP:

```text
src/basinflow/graphs/atomic_graph.py
src/basinflow/graphs/neighborlist.py
tests/test_graph_pbc.py
```

Requirements:

- Graphs must update during sampling when coordinates move.
- Store `edge_index`, `edge_vectors`, and `cell_offsets` or equivalent shifts.
- Consider cached neighborlist rebuild to avoid rebuilding every step.

### Raw Event Dataset

Read:

```text
trajcast/data/trajectory.py
AMDEN-code/src/data.py
OAReactDiff/oa_reactdiff/dataset/transition1x.py
```

Implement in FS-CGP:

```text
src/basinflow/data/raw_events.py
src/basinflow/data/records.py
tests/test_raw_events.py
```

Requirements:

- User-facing raw data is one multi-frame event file.
- Frame 0 is reactant.
- Frame 1 is product.
- Frame 2 may be transition state.
- ~~Frame 3+ may be path images~~ — **removed**.  The MVP adopts a
  strict three-frame model.  Path images belong to saddle validation
  (Stage 5), not to the event schema.
- Split generation is automatic and reproducible via `BasinSplit.create()`
  (basin-level shuffle, seed-controlled).

### Seed and Prior Design

Read:

```text
liflow/utils/prior.py
trajcast/model/forecast_tools.py
AdsorbDiff/adsorbdiff/trainers/sde_denoising_trainer.py
```

Implement in FS-CGP:

```text
src/basinflow/inits/
tests/test_seeds.py
```

Requirements:

- Gaussian displacement, active-region perturbation, pseudo-velocity, and event-library motifs are seed types for the neural proposer, not standalone Stage 2 baseline goals.
- Add pseudo-velocity / Maxwell-Boltzmann-style priors later.
- Keep seed semantics explicit: a seed is an exploration direction, not a product label.

### Candidate Sampling and Rollout

Read:

```text
trajcast/model/forecast.py
liflow/utils/inference.py
AMDEN-code/src/pipeline.py
```

Implement in FS-CGP:

```text
src/basinflow/sampling/candidate_sampler.py
tests/test_candidate_sampler.py
```

Requirements:

- Multiple calls generate multiple candidates.
- Each candidate records seed id, sampling config, checkpoint id, and graph update settings.
- Output raw generated candidates before relaxation.

### Relaxation and Refinement

Read:

```text
AdsorbDiff/adsorbdiff/relaxation/ml_relaxation.py
AdsorbDiff/adsorbdiff/relaxation/calculator.py
AMDEN-code/src/models/modules/material_schedule.py
```

Implement in FS-CGP:

```text
src/basinflow/relaxation/
src/basinflow/refinement/
tests/test_relaxation_interface.py
```

Requirements:

- ASE-backed relaxation interface first.
- HMC-like refinement is a future extension, not MVP.
- Generated coordinates become product candidates only after relaxation and clustering.

### Evaluation

Read:

```text
OAReactDiff/oa_reactdiff/analyze/rmsd.py
OAReactDiff/oa_reactdiff/evaluate/evaluate_ts_w_rp.py
trajcast/validation/physical_behaviour.py
AdsorbDiff/adsorbdiff/modules/evaluator.py
```

Implement in FS-CGP:

```text
src/basinflow/evaluation/
src/basinflow/clustering/
tests/test_clustering.py
```

Requirements:

- Molecule RMSD is only one metric.
- Periodic product-basin matching must be PBC-aware.
- Main metrics should be product-basin recall, duplicate rate, invalid rate, rate-weighted recall when barriers exist, and validation cost.
