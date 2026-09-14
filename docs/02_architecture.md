# Architecture

## Current PaiNN path (2026-09-10)

Stage 3 now offers a dual-geometry PaiNN backend alongside legacy EGNN checkpoint
compatibility. `models/painn/painn.py` contains the tensor backbone, `layers.py`
contains radial embeddings, cutoffs, messages, updates and vector readout, and
`modules.py` adapts `EventData` and graph-level time to the tensor API. The
backbone receives `positions_1=R`, `positions_2=x_t`, atomic numbers, Cartesian
image shifts, edge indices and atom-level time. It returns velocity and scalar
hidden features; it has no untrained active/direction heads.

The default `stability_mode=scaled` normalizes scalar message/update inputs and
scales vector messages, scalar dot products and residuals. These operations are
invariant scalars or act on feature channels, preserving O(3) equivariance.
The unscaled compatibility mode is retained for reproducing diagnostic runs;
it overflowed during Pt training and is not recommended for new training.

The adapter rebuilds the union of reference and current periodic radius graphs
at every velocity evaluation, using pure PyTorch and retaining all image ids.
Geometry is recomputed from differentiable coordinates and integer offsets.
The flow state uses a shared unwrapped lift relative to R; simultaneous per-atom
image shifts of R and x_t preserve the result. Independent wrapping of the two
states changes that lift and must not be performed inside integration.

The approved default objective is movable-atom component-mean velocity MSE.
Active atoms and unit directions are derived from the final MIC displacement,
with provenance recorded in each candidate. Gaussian noise affects x_0 and is
redrawn reproducibly by epoch; the PaiNN does not receive the original Gaussian
vector as an additional persistent condition. `active_prior` and `movable_mask`
remain scalar inputs. The older multi-head/seed-vector description below records
the legacy prototype, not the current default PaiNN training contract.

Sampling uses left-endpoint Euler with dynamic graphs. This integration rule is
explicitly recorded; it is not a midpoint or Heun solver. The exact tensor radius
graph has quadratic pair-search cost per structure, with chunked temporary
memory. CPU is tested; accelerator performance and large-system scaling need
separate qualification before deployment at that scale.

## Recommended Model Direction

The recommended generator is a seed-conditioned flow-matching event proposer over atomic displacements, active-region scores, candidate product coordinates, and optional transition-state coordinates.

Reasons:

- It gives a direct sampling trajectory from a seed structure to a candidate product.
- It is easier to condition on reactant geometry, local environment, and event seed than a pure denoising diffusion objective.
- It fits the idea of using different seeds to generate different exploration directions.
- It can be paired with a denoising or force-like corrector.
- It can expose interpretable proposal heads for active atoms and event direction.
- It can use a React-OT-style conditional flow for transition-state generation once a product candidate is known.

The framework should keep the model interface generic enough to support diffusion models later, but the first implementation should prioritize deterministic or low-variance conditional flow matching rather than OA-ReactDiff-style multi-sample stochastic diffusion.

## High-Level Pipeline

```text
EventCatalog
  ├─ EventFlowDataset (event-level pairwise training)
  │    └─ EGNNFlow
  └─ BasinDataset (basin-level inference)
       └─ EGNNFlow → CandidateSampler
                         ↓
                  DynamicPBCGraphBuilder
  -> RelaxationRunner
  -> ProductClusterer
  -> SaddleValidator
  -> EventLibrary
  -> KMCExporter
```

## Core Components

### EventCatalog and PyG Views

`EventCatalog` is the read-only domain boundary.  It owns
`StructureRecord`, `EventRecord`, and `BasinRecord`, validates their
cross-references and frame consistency, and supports basin-only subsets and
reproducible `BasinSplit` manifests.  It does not tensorize data, sample flow
times, construct graphs, or choose seeds.

`EventFlowDataset` is the Stage 3 pairwise training view.  One sample is a
known event plus an initializer and is emitted directly as an `EventData`
(PyG `Data`) object.  `BasinDataset` is the inference view.  One
sample is a basin plus an initializer and contains only reactant-derived
conditions; known products, true active labels, TS frames, and `target_*`
fields never enter it.

The first concrete dataset direction is an EON-style event table plus multi-frame event files:

```text
basin_table.csv
event_*.traj  # preferred when FixAtoms constraints must round-trip
event_*.extxyz
```

Each event file preserves reactant, product, and optional transition-state
frames.  `move_mask` becomes only the `movable_mask` constraint.  Activity
supervision uses an explicit `EventRecord.active_atoms` annotation when
available, otherwise it is derived from MIC reactant-to-product displacement.

### EventSeed and SeedGenerator

Produces initial proposal states or perturbation directions. The seed is not the answer. It is a mechanism for diversity. In BasinFlow, a seed should be interpreted as a proposed exploration direction or pseudo-dynamical perturbation, not as a hidden product label.

User-facing generator classes use `Init` names such as `ZeroInit`,
`GaussianInit`, and `ProductInit` to make this clearer: they generate
initial states or perturbation initializations for the flow, not final
products.  The lower-level `EventSeed` record remains the Stage 3 data
contract because it stores scalar conditions, vector conditions, and the
initial geometry in one object.

Seed types can include:

- Movable-atom or active-atom masks.
- Direction or pseudo-velocity fields.
- Product-displacement priors.
- Local-mode perturbations.
- Adsorbate/local-environment perturbations.
- Previous event-library motifs.

An event seed should enter the model in three forms:

```text
node scalar condition:
  active_prior, movable_mask, seed_type, local scores

node vector condition:
  seed_direction, seed_displacement, pseudo_velocity

flow initial geometry:
  x_0 = reactant + seed_displacement
```

This design follows the lessons from LiFlow and TrajCast: a physical seed should be a geometric state or equivariant vector condition, not just metadata appended to scalar node features.

### ProductEventFlow

Maps:

```text
reactant structure, seed state, flow time, optional event condition -> product velocity or score
```

The output is integrated into candidate product coordinates.

The first MVP outputs:

- active atom logits
- event displacement or direction field
- candidate product displacement or coordinates

### TSFlow

Maps:

```text
reactant structure, product structure, optional event seed, flow time -> TS velocity or score
```

This component should follow the React-OT lesson: when both reactant and product are available, transition-state generation should use a deterministic or low-variance R/P-conditioned flow path, such as interpolation from R/P to the TS target.

The first TS MVP outputs:

- transition-state displacement or coordinates.
- optional mode/direction field useful for saddle refinement.

Later heads may output:

- barrier or log-rate estimate
- uncertainty or validation-priority score

Raw generated coordinates are not final KMC events.

### DynamicPBCGraphBuilder

Builds molecular or periodic graphs during sampling. For crystals, graph edges must be updated as coordinates move. Static neighbor lists from the reactant are not sufficient when generated atoms cross cell boundaries or change local coordination.

### CandidateSampler

Runs multiple stochastic event proposals per basin and records trajectory metadata:

- seed id
- random seed
- number of steps
- graph update frequency
- model checkpoint
- sampling temperature
- active atoms
- event direction summary
- optional priority score

### RelaxationRunner

Relaxes candidates with a configured calculator:

- initial cheap ML potential
- optional DFT or higher-level validator
- constrained relaxation if needed

The generator should not be judged before relaxation.

### ProductClusterer

Groups relaxed candidates into local-minimum basins using structure matching.

For molecules:

- RMSD with atom mapping.
- Bond graph comparison.

For crystals:

- PBC-aware local environment fingerprints.
- ASE/pymatgen-compatible structure matching.
- Adsorbate or defect local-region matching when full-cell comparison is too strict.

### SaddleValidator

Checks whether a candidate product is connected to the reactant by a relevant path and saddle. This can start as an external interface to dimer, NEB, ARTn, or existing AKMC tools.

The validator may consume event-direction or TS-guess information as an initial search direction. The validator remains the authority for transition states and barriers.

### EventLibrary

Stores known and newly validated events, keyed by local environment and event signature. The library is essential for avoiding repeated validation of equivalent events.

### KMCExporter

Exports validated events into a rate-table format usable by downstream KMC code.

## Data Flow

Training flow:

```text
basin/event records -> pair sampler -> event seed + derived active/direction labels -> flow state -> product-event-flow loss
```

The first training objective should be simple and robust:

```text
L = L_product_displacement + lambda_active L_active + lambda_direction L_direction
```

All three targets derive from reactant/product pairs.  Explicit activity
annotations take precedence; otherwise activity derives from MIC displacement.
`movable_mask` remains a hard generation constraint and is never silently
substituted for a true activity label.

TS training flow:

```text
reactant/product/TS records -> R/P interpolation or seeded midpoint -> TS flow loss
```

TS targets are used only when TS or saddle-like frames are present. Barrier, rate, and priority losses are masked unless labels or validators are available.

Evaluation flow:

```text
held-out basin -> many seeds -> generated products + TS guesses -> matching/optional relaxation -> recall/cost metrics
```

Adaptive KMC flow:

```text
current state in EON -> optional library lookup -> BasinFlow model proposals -> EON saddle refinement/validation -> event table update
```

BasinFlow should not call EON during early model training. The intended integration is later EON-side support for an `ml_proposal` suggestion source that consumes BasinFlow product/TS/direction candidates.

## Why This Architecture

This architecture separates three questions that should not be mixed:

- Can the model propose diverse plausible end states?
- Can the model identify active atoms and plausible escape directions?
- Do the proposed structures relax to distinct local minima?
- Are those minima connected by valid transition states with useful rates?

Keeping these stages separate makes the project easier to benchmark and prevents overclaiming model accuracy.
