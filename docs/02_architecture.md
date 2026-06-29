# Architecture

## Recommended Model Direction

The recommended generator is a conditional flow matching or rectified-flow event proposer over atomic displacements, active-region scores, and candidate product coordinates.

Reasons:

- It gives a direct sampling trajectory from a seed structure to a candidate product.
- It is easier to condition on reactant geometry, local environment, and event seed than a pure denoising diffusion objective.
- It fits the idea of using different seeds to generate different exploration directions.
- It can be paired with a denoising or force-like corrector.
- It can expose interpretable proposal heads for active atoms and event direction before adding harder TS or barrier targets.

The framework should keep the model interface generic enough to support diffusion models later, but the first implementation should prioritize conditional flow matching.

## High-Level Pipeline

```text
EventDataset
  -> BasinSampler
  -> SeedGenerator
  -> EventProposer
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

### SeedGenerator

Produces initial proposal states or perturbation directions. The seed is not the answer. It is a mechanism for diversity. In FS-CGP, a seed should be interpreted as a proposed exploration direction or pseudo-dynamical perturbation, not as a hidden product label.

Seed types can include:

- Gaussian displacement seeds.
- Local-mode perturbations.
- Bond-change proposals.
- Adsorbate/local-environment perturbations.
- Previous event-library motifs.

### EventProposer

Maps:

```text
reactant structure, seed state, time/noise level, optional event condition -> velocity or score
```

The output is integrated or iteratively denoised into candidate product coordinates.

The first MVP outputs:

- active atom logits
- event displacement or direction field
- candidate product displacement or coordinates

Later heads may output:

- transition-state guess
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
basin/event records -> pair sampler -> derived active/direction labels -> noised/interpolated state -> event-proposer loss
```

The first training objective should be simple and robust:

```text
L = L_product_displacement + lambda_active L_active + lambda_direction L_direction
```

All three targets can be derived from reactant/product pairs. TS, barrier, rate, and priority losses are masked unless labels or validators are available.

Evaluation flow:

```text
held-out basin -> many seeds -> generated candidates -> relaxation -> clustering -> recall/cost metrics
```

Adaptive KMC flow:

```text
current state -> library lookup -> model proposals -> cheap filtering -> selective saddle validation -> event table update
```

## Why This Architecture

This architecture separates three questions that should not be mixed:

- Can the model propose diverse plausible end states?
- Can the model identify active atoms and plausible escape directions?
- Do the proposed structures relax to distinct local minima?
- Are those minima connected by valid transition states with useful rates?

Keeping these stages separate makes the project easier to benchmark and prevents overclaiming model accuracy.
