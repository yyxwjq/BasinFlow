# Risks and Decisions

## Major Design Decisions

### Decision 1: New Clean Project Instead of Direct akmcgc Expansion

Rationale:

- `akmcgc` is valuable as a prototype but still inherits a one-to-one reaction-generation structure.
- The target project needs basin-level data, validation, and KMC integration from the beginning.
- A clean architecture reduces long-term coupling to molecular-only assumptions.

### Decision 2: Conditional Flow Matching as First Neural Event Proposer

Rationale:

- It naturally supports seed-driven diverse generation.
- It gives a direct path from proposal state to product-like candidate.
- It is compatible with a later corrector or denoiser.
- It can support interpretable heads for active atoms and event directions without requiring TS/barrier labels in the first version.

Fallback:

- Keep a generic generator interface so diffusion models can be tested later.

### Decision 3: Candidate Generation and Validation Are Separate

Rationale:

- A plausible product geometry is not automatically a valid KMC event.
- Physical validation is necessary for transition states, barriers, and rates.

### Decision 4: Product Candidate Is Only Part of Event Proposal

Rationale:

- KMC events are basin-escape channels, not only product structures.
- The first MVP should learn active atoms, event directions, and product candidates from R/P pairs.
- Barrier, rate, TS, committor, and priority heads should be added only when labels or validators exist.

### Decision 5: Basin-Level Evaluation

Rationale:

- One-to-one reconstruction does not measure event discovery.
- KMC needs coverage over important events from each basin.

## Technical Risks

### Risk: Generated Products Do Not Connect to Reactant

Mitigation:

- Add saddle/path validation stage.
- Track invalid and disconnected candidates explicitly.
- Use event-library feedback to improve seeds.

### Risk: Event-Direction Labels Are Noisy

Mitigation:

- Derive active atoms from MIC displacement with configurable thresholds.
- Treat active/direction heads as auxiliary targets, not the only source of truth.
- Evaluate whether they improve validation cost and basin recall before expanding the model.

### Risk: Dynamic PBC Graph Errors During Sampling

Mitigation:

- Build dedicated PBC geometry tests.
- Update neighbor lists during sampling.
- Store cell offsets and edge metadata for debugging.

### Risk: Dataset Has Incomplete Event Coverage

Mitigation:

- Use rate-weighted and validation-aware metrics.
- Treat unmatched generated candidates as unknown until validated.
- Compare under fixed validation budgets.

### Risk: Model Increases Cost Instead of Reducing It

Mitigation:

- Add cheap filters before saddle validation.
- Use duplicate clustering before expensive validation.
- Fall back to conventional AKMC search when model confidence is low.

### Risk: Molecule and Crystal Support Diverge

Mitigation:

- Use a shared `StructureRecord`.
- Keep PBC fields optional.
- Test molecular and periodic examples in every core module.

## Open Questions

- Which datasets will be used for the first molecular and periodic MVP benchmarks?
- Which relaxation backend will be used first?
- Which saddle-search validator is available locally?
- What KMC code or event-table format should the exporter target?
- How should local environments be canonicalized for event-library reuse?

## Current Recommended Next Decision

Choose the first two benchmark systems:

- One molecular reaction set.
- One small periodic event set.

This decision should happen before neural model implementation.
