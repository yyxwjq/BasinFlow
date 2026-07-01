# Research Direction Memory

This note preserves the key decisions from the June 2026 planning discussion so future work does not drift back to an obsolete baseline-first plan.

## Current Project State

- Stage 1 geometry and data core is implemented in `src/fscgp/{data,geometry,graphs}`.
- The repository contains tests for records, raw event loading, MIC geometry, neighbor lists, and dataset collation.
- `docs/05_development_plan.md` records Stage 1 as complete with 20 passing tests.
- The first concrete dataset is `/Users/wx/Desktop/events`, an EON-style Au event set with `basin_table.csv` and `event_*.extxyz` files.
- Each event file is a multi-frame structure. The current interpretation is:
  - frame 0: reactant
  - frame 1: product
  - frame 2: transition-state or saddle-like structure, if present
- The Au example has a simple active region: one movable Au atom marked by `move_mask`, while other atoms are fixed.
- The working tree had a pre-existing modification in `tests/test_raw_events.py` during this discussion. Do not overwrite it without checking.

## Main Scientific Reframing

The project should not be framed as merely another transition-state generator. OA-ReactDiff and React-OT already cover important parts of that space.

The stronger project definition is:

```text
seed-conditioned basin-level event proposal for AKMC:
reactant basin -> multiple product + TS + direction candidates
```

The key gap is not `R + P -> TS`. The key gap is:

```text
R_basin -> {P_k, TS_k, active_atoms_k, direction_k}
```

This is the object AKMC needs: a distribution over escape-event channels from the current basin.

## Reference Lessons

### OAReactDiff

Path:

```text
/Users/wx/Desktop/yyxwjq/OAReactDiff
```

Confirmed from README and code:

- It jointly models reactant, transition state, and product as three objects/fragments.
- `ProcessedTS1x` reads `reactant`, `transition_state`, and `product`.
- The demo supports conditional inpainting where reactant and product are fixed and TS is generated.
- Its core contribution is object-aware SE(3) modeling for elementary reactions.

Project implication:

- Multi-object R/TS/P structure is a useful template.
- Pure OAReactDiff-style stochastic DDPM should not be the first choice for the new model because React-OT shows the SDE route is slower and less deterministic.

### React-OT

Local note/PDF:

```text
/Users/wx/Downloads/同步空间/obisidan/zotero/akmcgc-reference/2025 - Duan - Optimal transport for generating transition states in chemical reactions.md
/Users/wx/Downloads/同步空间/MyZotero/akmcgc/Optimal transport for generating transition states in chemical reactions  Nature Machine Intelligen.pdf
```

Key lesson:

- React-OT converts OAReactDiff's stochastic TS generation into deterministic flow matching / ODE transport.
- It uses an `R/P` interpolation as the TS initial state and learns a velocity field toward the true TS.

Project implication:

- For TS generation, prefer a React-OT-style conditional flow:

```text
R + P -> TS
```

- In BasinFlow, the novel part is producing good `P` candidates from `R_basin`; then TS generation can use the generated or known `P`.

### LiFlow

Path:

```text
/Users/wx/Desktop/yyxwjq/liflow
```

Confirmed from code:

- It learns coarse atomic displacement using conditional flow matching.
- Priors include Gaussian and Maxwell-Boltzmann variants.
- `TimeDelayedPairDataset` provides start and delayed-end positions.
- `DualPaiNN` uses the difference between two coordinate states as a vector feature.
- Inference samples a prior displacement, adds it to the current structure, and integrates the learned flow.

Project implication:

- Physical event seed should not be only metadata.
- Seed displacement or pseudo-velocity should become the flow initial state:

```text
x_0 = R + seed_displacement
x_1 = P
```

### TrajCast

Path:

```text
/Users/wx/Desktop/yyxwjq/trajcast
```

Confirmed from code and notes:

- It directly predicts displacement and next velocity without computing forces.
- It embeds velocity through norm features and spherical harmonics.
- It updates graph edges during rollout.

Project implication:

- Event direction and pseudo-velocity seed should be true equivariant vector channels, not ordinary scalar `x/y/z` features.
- Dynamic graph rebuilding remains important during sampling.

### EON

Path:

```text
/Users/wx/Desktop/yyxwjq/eon
```

Confirmed from code:

- EON already owns AKMC orchestration, process search, saddle refinement, minimization, barrier, and prefactor logic.
- EON `generate_displacement()` currently supports suggestion sources such as recycling, KDB, random, and dynamics.
- `process_search` consumes `pos.con`, optional `displacement.con`, and optional `direction.dat`.

Project implication:

- BasinFlow should not call EON during the early training chain.
- The correct integration path is later EON-side support for an `ml_proposal` suggestion source that consumes BasinFlow model outputs.

## Rejected Plan

The old Stage 2 plan was:

```text
random local displacement -> active-region perturbation -> hop-like heuristic baseline
```

This is no longer the main route. For the current single-movable-Au dataset it is likely to become system-specific boilerplate and does not address the main research contribution.

Minimal non-learning sanity checks can still exist, but they should not define the main development stage.

## New Development Logic

The project should proceed as:

```text
Stage 1: Geometry and Data Core
Stage 2: EON-style R/P/TS Event Dataset Integration
Stage 3: Seed-Conditioned Product/Event Flow
Stage 4: React-OT-style TS Flow
Stage 5: Basin-Level Event Proposal Benchmark
Stage 6: EON-Side ML Suggestion Integration
```

## Model Design Principle

An `EventSeed` should be represented in three ways:

```text
node scalar condition:
  active_prior, movable_mask, fixed_mask, seed_type, local scores

node vector condition:
  seed_direction, seed_displacement, pseudo_velocity

flow initial geometry:
  x_0 = reactant + seed_displacement
```

This avoids treating physical seed information as a weak prompt.

## Immediate Next Work

1. Reframe Stage 2 around `/Users/wx/Desktop/events`.
2. Build a clean EON-style event dataset path that preserves `move_mask`, product, and TS frames.
3. Add geometry invariance tests before model work.
4. Implement a seed-conditioned product flow before TS generation.
5. Implement React-OT-style TS flow using known/generated product conditions.
6. Keep EON runtime integration as a later EON-side plugin/interface task.
