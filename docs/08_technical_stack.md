# Technical Stack

## Purpose

This document defines the recommended technical stack for the first implementation. It should guide development without freezing every dependency version too early.

The first goal is a reproducible research codebase for molecular and periodic end-state candidate generation, not a production KMC engine.

## Language and Core Runtime

Recommended:

- Python 3.10 or 3.11.
- PyTorch as the main tensor and model framework.
- PyTorch Geometric or a compatible graph batching layer.
- ASE for structure I/O, periodic cells, trajectories, and relaxation interfaces.

Reasons:

- The reference frameworks are Python/PyTorch-oriented.
- ASE is the most practical common interface for molecules, surfaces, and crystals.
- PyTorch Geometric-style data objects fit variable-size atomic graphs.

## Geometry and Materials Tooling

Recommended:

- ASE for `Atoms`, calculators, extxyz, trajectory files, and neighbor lists.
- pymatgen as an optional dependency for structure matching and crystal analysis.
- spglib as an optional dependency for symmetry-related analysis.

Required project utilities:

- PBC-aware distance computation.
- Dynamic neighbor-list construction.
- Cell-offset edge representation.
- Structure canonicalization for clustering and event-library lookup.

## Neural Network Architecture

Recommended first architecture:

- Conditional flow matching or rectified-flow generator.
- Equivariant graph neural network backbone.
- Dynamic graph reconstruction during sampling.
- Optional corrector/denoiser stage.

Backbone candidates:

- EGNN-style model for simpler implementation and debugging.
- PaiNN/SchNet-style continuous-filter architecture for atomistic systems.
- e3nn/NequIP/MACE-style equivariant models when higher geometric fidelity is needed.

Initial recommendation:

- Start with an EGNN or PaiNN-like backbone for the MVP.
- Keep the interface compatible with stronger equivariant backbones later.

## Data and File Formats

Recommended formats:

- `*.xyz` or `*.extxyz` for structures and trajectories.
- `*.json` or `*.yaml` for basin, event, candidate, and split metadata.
- `*.pt` or `*.npz` only for cached tensors, not as the primary dataset source of truth.

Required records:

- `StructureRecord`
- `EventRecord`
- `BasinRecord`
- `CandidateRecord`

The source dataset should preserve basin-level grouping even if training uses pairwise samples.

## Relaxation and Validation Backends

Initial relaxation options:

- ASE calculators.
- ML potentials such as MACE, CHGNet, M3GNet, or locally available calculators.
- Classical potentials for simple benchmark systems when appropriate.

Future validation options:

- Dimer method.
- NEB or climbing-image NEB.
- ARTn-style saddle search.
- External AKMC event-search tools.

Design requirement:

- Relaxation and saddle validation must be external interfaces, not hard-coded inside the generator.

## Experiment Management

Recommended:

- YAML configuration files.
- Structured output directories per experiment.
- Checkpoint metadata containing model, dataset split, sampling settings, and git state when available.
- CSV/JSON benchmark summaries.

Optional:

- Weights & Biases, TensorBoard, or MLflow for experiment tracking.

Do not make online tracking mandatory.

## Testing Stack

Recommended:

- `pytest` for unit and integration tests.
- Small synthetic molecule and periodic fixtures.
- Geometry tests before model tests.

Required early tests:

- Structure I/O round trip.
- Molecule and crystal batching.
- PBC minimum-image distances.
- Dynamic neighbor-list update.
- Variable number of known events per basin.
- Candidate clustering behavior on simple examples.

## Package Layout Recommendation

Recommended initial layout:

```text
src/fscgp/
  data/
  geometry/
  graphs/
  models/
  sampling/
  relaxation/
  clustering/
  validation/
  event_library/
  export/
  workflows/
tests/
configs/
examples/
docs/
```

Each module should have a narrow responsibility. The generator should not own dataset parsing, relaxation, clustering, or KMC export logic.

## Dependency Policy

Start with a minimal dependency set:

- `torch`
- `numpy`
- `scipy`
- `ase`
- `pyyaml`
- `pytest`

Add graph and materials dependencies explicitly when the corresponding module is implemented:

- `torch-geometric`
- `pymatgen`
- `spglib`
- `e3nn`

Avoid adding heavy dependencies before there is a tested module that requires them.
