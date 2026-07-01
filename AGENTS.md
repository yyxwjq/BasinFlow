# Project Agent Guide

## Project Goal

This project develops a learning-assisted KMC event proposal framework for KMC/AKMC rate table construction.

The target workflow is:

```text
reactant basin -> event proposals -> candidate products -> relaxation -> clustering -> saddle validation -> event library -> KMC export
```

The model is not expected to prove that an event is valid by itself. It proposes active atoms, escape directions, candidate product structures, and validation priorities that reduce the number of expensive blind saddle-search calls.

The current research direction is:

```text
seed-conditioned basin-level event proposal for AKMC:
reactant basin -> multiple product + TS + direction candidates
```

The main novelty should be framed as `R_basin -> {P_k, TS_k, active_atoms_k, direction_k}` rather than only `R + P -> TS`.

## Scientific Scope

Primary objective:

- Given a reactant structure or reactant basin, propose multiple candidate KMC escape events.

An event proposal may include active atoms, an event direction or displacement field, a candidate product/end-state structure, an optional transition-state guess, and an optional barrier/rate/uncertainty or validation-priority score.

Primary application:

- Build or expand reaction/event tables for KMC and adaptive KMC simulations.

Initial MVP:

- Benchmark event proposal on the initial EON-style Au event dataset and later broaden to small molecular and periodic systems.
- Evaluate candidates after relaxation and clustering.
- Measure event recall and computational cost reduction relative to brute-force saddle-search baselines.

Later extensions:

- Transition-state validation.
- Event-library reuse.
- Adaptive KMC integration.
- Active-learning loops with uncertainty and fallback to conventional saddle search.

## Non-Goals

Do not frame the first implementation as:

- A general crystal generator.
- A complete replacement for AKMC.
- A model that directly guarantees transition states or barriers.
- A one-to-one reactant-to-product predictor only.
- A pure product-coordinate generator with no active-region or event-direction semantics.
- A baseline-first project centered on random local displacement, hop-like, or site-specific heuristic proposers.
- A BasinFlow-side runtime wrapper around EON during early model training.

Pairwise training can be used, but evaluation must be basin-level.

## Reference Projects

Reference projects are kept outside this repository. Do not copy entire reference repositories into this project.

Known reference paths:

- `/Users/wx/Desktop/yyxwjq/OAReactDiff`
- `/Users/wx/Desktop/yyxwjq/akmc-product-generation`
- `/Users/wx/Desktop/yyxwjq/AdsorbDiff`
- `/Users/wx/Desktop/yyxwjq/liflow`
- `/Users/wx/Desktop/yyxwjq/trajcast`
- `/Users/wx/Desktop/yyxwjq/AMDEN-code`
- `/Users/wx/Desktop/yyxwjq/eon`

Important React-OT notes and PDFs are currently outside the repository:

- `/Users/wx/Downloads/同步空间/obisidan/zotero/akmcgc-reference`
- `/Users/wx/Downloads/同步空间/MyZotero/akmcgc`

Use these repositories for design inspection and selective porting only. If code is copied later, copy the minimum necessary module, preserve attribution, and document why it was imported.

## Design Principles

- Treat candidate generation and physical validation as separate stages.
- Treat product generation as one part of event proposal, not the whole project.
- Treat transition-state generation as a later `R/P -> TS` module that can use React-OT-style flow matching once product candidates exist.
- Use dynamic periodic graph construction during sampling for crystal systems.
- Support both molecules and periodic crystals through a shared structure interface.
- Organize data at basin level, even if training samples pairwise events.
- Prefer physically interpretable proposal targets: active atoms, event directions, product candidates, and validation priority.
- Represent an `EventSeed` as scalar conditions, equivariant/vector conditions, and a flow initial geometry, not only as metadata.
- Avoid global rewrites of imported framework ideas; implement clean project-local interfaces.
- Keep geometry, datasets, model, sampling, relaxation, clustering, validation, and KMC export as separate modules.
- Keep EON AKMC orchestration, saddle refinement, minimization, barriers, prefactors, and runtime scheduling inside EON. BasinFlow should export proposals for a later EON-side `ml_proposal` integration.

## Current Development Order

Follow `docs/05_development_plan.md` as the source of truth. The current staged direction is:

```text
Stage 1: Geometry and Data Core
Stage 2: EON-style R/P/TS Event Dataset Integration
Stage 3: Seed-Conditioned Product/Event Flow
Stage 4: React-OT-Style TS Flow
Stage 5: Basin-Level Event Proposal Benchmark
Stage 6: EON-Side ML Suggestion Integration
Stage 7: Event Library and Adaptive KMC Extensions
```

Do not resurrect the old Stage 2 heuristic-baseline plan as the main route. Gaussian/local perturbations, active-region perturbations, hop-like moves, and similar rules may be seed types or smoke-test utilities, but they should not define the research contribution.

## Expected Engineering Style

- Prefer PyTorch and PyTorch Geometric-style graph data where possible.
- Use ASE-compatible structure I/O for periodic systems.
- Store event/basin metadata in structured formats such as JSON/YAML plus trajectory files.
- Write tests for geometry invariance, PBC neighbor construction, dataset collation, and candidate clustering before large model training.
- Keep model code independent from workflow orchestration code.

## Agent Instructions

Before implementing new code:

1. Read `docs/00_project_vision.md`.
2. Read `docs/02_architecture.md`.
3. Read `docs/03_data_schema.md`.
4. Read `docs/08_technical_stack.md`.
5. Read `docs/10_research_direction_memory.md`.
6. If implementing a module inspired by a reference project, read `docs/09_reference_code_notes.md`.
7. Check `docs/05_development_plan.md` for the current stage.
8. If using ideas from a reference project, update `docs/06_reference_frameworks.md` and `docs/09_reference_code_notes.md`.

Do not assume one reactant has only one correct product. For this project, one reactant basin may have many valid events, and unknown generated events require validation rather than immediate rejection.
