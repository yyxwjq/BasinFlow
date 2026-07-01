# Development Plan

## Stage 0: Project Grounding

Deliverables:

- Formal project documents.
- Reference framework notes.
- Minimal package layout proposal.
- Data examples selected for molecule and crystal cases.

Exit criteria:

- The project goal, non-goals, and MVP benchmark are clear.
- Agents can understand the architecture from repository documents.

## Stage 1: Geometry and Data Core  ✅ COMPLETE

Implement:

- `StructureRecord`, `EventRecord`, `BasinRecord`, and `CandidateRecord`.
- ASE-compatible molecular and periodic I/O.
- PBC-aware distance and neighbor utilities.
- Dataset loader exposing pairwise and basin-level views.
- Batch collation for variable atom counts and variable event counts.

Tests:

- Molecule without cell.
- Periodic crystal with cell and PBC.
- Different numbers of events per basin.
- PBC minimum-image distance and cell-offset edge construction.

Status: complete.  See `examples/demo_pipeline.py` for an end-to-end
walkthrough from raw extxyz files to trainable batches.

## Stage 2: EON-Style Event Dataset Integration  ✅ CORE COMPLETE

Implement:

- Import EON-style event folders such as `/Users/wx/Desktop/events`.
- Parse `basin_table.csv` into basin-level event groups.
- Read multi-frame `event_*.extxyz` files with:
  - frame 0 as reactant.
  - frame 1 as product.
  - frame 2 as optional transition-state or saddle-like structure.
- Preserve `move_mask` semantics through canonical `movable_mask`, plus cell, PBC, event ids, and basin ids. Do not add explicit masses fields in Stage 2; masses remain available through ASE when later physical seed priors need them.
- Expose pairwise training records and basin-level evaluation records from the same dataset.
- Derive supervised labels:
  - active atoms from `move_mask` when present.
  - fallback active atoms from MIC reactant-product displacement.
  - product displacement.
  - event direction.
  - optional TS displacement / TS structure target.
- Add deterministic train/validation/test split utilities that split by basin, not by individual event.

Purpose:

- Make the existing EON-generated event examples directly usable for neural training and basin-level evaluation.
- Avoid spending the main development stage on system-specific heuristic proposal rules.
- Preserve the scientific distinction between pairwise training samples and basin-level event sets.

Tests:

- `/Users/wx/Desktop/events`-style table and event files are parsed into the expected number of basins and events.
- Multi-frame extxyz files preserve reactant, product, and optional TS frames.
- `move_mask` is converted into active labels.
- Basin-level view returns multiple events for basins that have multiple events.
- Splits do not put events from the same basin into different partitions.
- Derived product displacement and event direction are finite and respect PBC flags.

Non-goal:

- Do not implement random local displacement, hop-like, or site-specific heuristic proposers as the main Stage 2 deliverable. Minimal smoke-test proposers may be added later only if they help test a shared model interface.

## Stage 2.5: Seed and Product-Flow Target Contract  ✅ PROTOTYPE COMPLETE

Implemented:

- `EventSeed` with scalar/vector/initial-geometry roles.
- Seed generators for zero, Gaussian movable, and product-displacement seeds.
- Product-flow target construction for straight-line conditional flow matching.
- Dummy product-event flow and masked velocity loss for interface smoke tests.

Status:

- The data and contract layer is available for Stage 3.
- A trainable EGNN/PaiNN-style backbone, sampling loop, and benchmark runner
  are still Stage 3+ work.

## Stage 3: Seed-Conditioned Product/Event Flow

Implement:

- Conditional flow matching model interface.
- Reactant-conditioned graph encoder.
- `EventSeed` representation with scalar, vector, and initial-geometry roles.
- Active-atom prediction head.
- Event-direction or displacement-field head.
- Dynamic graph update during sampling.
- Sampling loop with multiple seeds per basin.
- Product candidate generation from:

```text
reactant + event seed -> candidate product displacement / coordinates
```

Training:

- Use pairwise event samples.
- Train on interpolation or flow states between a seed-initialized proposal state and the product.
- Derive active atoms from MIC product displacement.
- Prefer `move_mask` active labels when present.
- Train the MVP with product-displacement flow loss, active-atom loss, and event-direction loss.
- Treat TS/barrier/rate targets as masked or deferred to Stage 4 unless labels are explicitly available.
- Evaluate basin-level product/event proposal quality.

Design requirement:

- The event seed must not be only a metadata condition. It must enter as:
  - node scalar features such as movable mask, active prior, and seed type.
  - node vector features such as seed direction, pseudo-velocity, and seed displacement.
  - the flow initial state `x_0 = reactant + seed_displacement`.

Tests:

- Forward pass on molecule batch.
- Forward pass on periodic batch.
- Active-atom and direction targets can be derived from toy R/P data.
- Dynamic graph update changes edges when geometry changes.
- Sampling produces finite product candidates and valid records.
- Fixed atoms remain fixed or near-fixed when fixed masks are provided.
- Rotating/translating/permuting a toy system preserves equivariant/invariant behavior.

## Stage 4: React-OT-Style TS Flow

Implement:

- A transition-state flow model conditioned on reactant and product:

```text
reactant + product -> transition-state guess
```

- A deterministic React-OT-style flow path from an R/P interpolation or seeded midpoint to the TS frame.
- Optional conditioning on the Stage 3 event seed and active atoms.
- TS heads or output records compatible with later EON `displacement.con` / `direction.dat` generation.
- Losses for TS coordinate/displacement error and optional direction consistency.

Training:

- Use known product frames from Stage 2 first.
- Later support generated products from Stage 3 as product conditions.
- Keep barrier/rate losses masked unless labels are available.

Tests:

- TS flow forward pass works on toy molecule and periodic/event examples.
- R/P interpolation initialization produces finite TS candidates.
- Fixed atoms remain fixed or near-fixed when fixed masks are provided.
- TS loss is masked cleanly for events without TS frames.

## Stage 5: Basin-Level Event Proposal Benchmark

Implement:

- Basin-level benchmark runner.
- Multi-seed product and TS candidate generation.
- Known-event matching against Stage 2 products/TS labels.
- Metrics report for product, TS, active atom, and direction quality.
- Optional lightweight relaxation/clustering only when needed for a benchmark, not as a prerequisite to model training.

Outputs:

- Per-basin candidate table.
- Product and TS candidate table.
- Event recall and duplicate metrics.
- Active-region and direction metrics.
- TS coordinate/displacement metrics when TS labels exist.
- Product-basin recall.
- Rate-weighted recall when barrier/rate metadata exists.
- Failure-case structures for inspection.

Exit criteria:

- The model can be evaluated as a basin-conditioned event proposal model on held-out basins.
- The benchmark measures whether generated product/TS/direction candidates cover known events, not only one-to-one reconstruction error.

## Stage 6: EON-Side ML Suggestion Integration

Implement:

- Export model proposals in a format that EON can consume as a new suggestion source.
- Define candidate conversion to EON-compatible artifacts such as:
  - product guess.
  - TS or saddle guess.
  - displacement-like structure.
  - direction/mode field.
- Add or document an EON-side `ml_proposal` source after the standalone training chain is complete.
- Keep EON AKMC orchestration, saddle refinement, minimization, barriers, prefactors, and runtime scheduling inside EON.

Exit criteria:

- EON can consume BasinFlow-generated event proposals as saddle-search suggestions.
- BasinFlow does not need to call EON during model training.

## Stage 7: Event Library and Adaptive KMC Extensions

Implement:

- Event-library lookup by local environment.
- Model-guided event proposal at each KMC state.
- Validation-priority or uncertainty score for selecting which candidates to validate first.
- Confidence or coverage-based stopping rule.
- Fallback to conventional saddle search when needed.
- Rate-table export.

Exit criteria:

- The framework can run a small adaptive KMC loop with model proposals and physical validation.

## Immediate Next Tasks

1. ~~Choose the first molecule benchmark dataset.~~ → Au₁₀₁ cluster (periodic, 12 basins).
2. ~~Choose the first periodic benchmark dataset.~~ → same Au system, PBC-aware.
3. ~~Define the package layout.~~ → `src/fscgp/{data,geometry,graphs}`.
4. ~~Implement schema and I/O tests.~~ → core tests passing.
5. ~~Add geometry invariance tests (translation, rotation, permutation) before Stage 3.~~
6. ~~Implement Stage 2 EON-style event dataset integration for `/Users/wx/Desktop/events`.~~
7. ~~Define `EventSeed` and seed-derived product-flow training targets.~~
8. Implement the trainable Stage 3 product-event flow backbone and training loop.
9. Add the basin-level mini benchmark before full TS-flow work.
10. Select and ingest a second benchmark dataset with multi-atom or multi-element events.
