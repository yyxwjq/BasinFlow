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

Status: **20 tests passing**.  See `examples/demo_pipeline.py` for an
end-to-end walkthrough from raw extxyz files to trainable batches.

## Stage 2: Baseline Event Proposal Generator

Implement:

- Random local displacement seeds.
- Active-region perturbation seeds.
- Simple local-mode or bond-change seeds where available.
- Relaxation and clustering pipeline.
- Candidate proposal records with active atoms and event-direction metadata.

Purpose:

- Build the benchmark pipeline before training the neural model.
- Establish baseline costs and failure modes.

Tests:

- Candidate records are produced deterministically with fixed random seeds.
- Relaxed outputs are linked back to generated candidates.
- Duplicate clustering works on small examples.
- Active atom and direction metadata are recorded for each proposal.

## Stage 3: Neural Event Proposer MVP

Implement:

- Conditional flow matching model interface.
- Reactant-conditioned graph encoder.
- Seed/intermediate-state encoder.
- Active-atom prediction head.
- Event-direction or displacement-field head.
- Dynamic graph update during sampling.
- Sampling loop with multiple seeds per basin.

Training:

- Use pairwise event samples.
- Train on interpolation or noised states between seed/proposal state and product.
- Derive active atoms from MIC product displacement.
- Train the MVP with product-displacement, active-atom, and event-direction losses.
- Mask TS/barrier/rate targets unless labels are available.
- Evaluate only on basin-level candidate generation.

Tests:

- Forward pass on molecule batch.
- Forward pass on periodic batch.
- Active-atom and direction targets can be derived from toy R/P data.
- Dynamic graph update changes edges when geometry changes.
- Sampling produces finite coordinates and valid records.

## Stage 4: Benchmark and Analysis

Implement:

- Basin-level benchmark runner.
- Relaxation integration.
- Product clustering.
- Known-event matching.
- Metrics report.
- Active-atom precision/recall.
- Event-direction angular error on active atoms.

Outputs:

- Per-basin candidate table.
- Recall and duplicate metrics.
- Active-region and direction metrics.
- Cost comparison with baselines.
- Failure-case structures for inspection.

Exit criteria:

- The model can be compared fairly against non-learning baselines on small systems.

## Stage 5: Saddle Validation Interface

Implement:

- External interface for dimer, NEB, ARTn, or existing AKMC validation tools.
- Validation job records.
- Transition-state and barrier metadata.
- Validated event insertion into event library.
- Optional use of event direction or TS guess to initialize saddle search.

Exit criteria:

- Candidate products can be promoted to validated KMC events through a reproducible workflow.

## Stage 6: Adaptive KMC Integration

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
4. ~~Implement schema and I/O tests.~~ → 20 tests passing.
5. Add geometry invariance tests (translation, rotation, permutation) before Stage 3.
6. Implement baseline event proposal generation (Stage 2).
