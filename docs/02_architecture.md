# Architecture

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
EventDataset
  -> BasinSampler
  -> SeedGenerator
  -> ProductEventFlow
  -> TSFlow
  -> DynamicPBCGraphBuilder
  -> CandidateSampler
  -> RelaxationRunner
  -> ProductClusterer
  -> SaddleValidator
  -> EventLibrary
  -> KMCExporter
```

## Core Components

### EventDataset

Loads basin-level and pairwise event samples. It should expose both:

- Pair records for training.
- Basin records for validation and sampling.

The first concrete dataset direction is an EON-style event table plus multi-frame event files:

```text
basin_table.csv
event_*.extxyz
```

Each event file should preserve reactant, product, and optional transition-state frames. Active labels should use `move_mask` when present, with MIC displacement-derived labels as a fallback.

### EventSeed and SeedGenerator

Produces initial proposal states or perturbation directions. The seed is not the answer. It is a mechanism for diversity. In BasinFlow, a seed should be interpreted as a proposed exploration direction or pseudo-dynamical perturbation, not as a hidden product label.

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

All three targets can be derived from reactant/product pairs, with `move_mask` preferred for active labels when present.

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
