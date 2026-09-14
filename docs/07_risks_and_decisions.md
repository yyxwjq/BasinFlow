# Risks and Decisions

## Major Design Decisions

### Decision 1: New Clean Project Instead of Direct akmcgc Expansion

Rationale:

- `akmcgc` is valuable as a prototype but still inherits a one-to-one reaction-generation structure.
- The target project needs basin-level data, validation, and KMC integration from the beginning.
- A clean architecture reduces long-term coupling to molecular-only assumptions.

### Decision 2: Seed-Conditioned Product/Event Flow as First Neural Event Proposer

Rationale:

- The target AKMC object is a basin-level event distribution:

```text
R_basin -> {P_k, TS_k, active_atoms_k, direction_k}
```

- Conditional flow matching naturally supports seed-driven diverse generation.
- It gives a direct path from a physical seed state to product-like candidates.
- It is compatible with a later corrector or denoiser.
- It can support interpretable heads for active atoms and event directions.
- It can train first from reactant/product pairs without requiring TS, barrier, or rate labels for every event.
- It makes physical seeds meaningful by using them as scalar conditions, vector conditions, and the flow initial geometry.

Fallback:

- Keep a generic generator interface so diffusion models can be tested later.

### Decision 3: TS Generation Should Be R/P-Conditioned and Secondary

Rationale:

- OAReactDiff and React-OT already show that transition-state generation is most reliable when reactant and product are known.
- BasinFlow's harder and more project-specific problem is proposing useful products and event directions from the reactant basin.
- Stage 4 should therefore use a React-OT-style `R + P -> TS` flow after Stage 3 can produce product candidates.

Implication:

- Do not frame BasinFlow primarily as another TS generator.
- TS guesses are valuable proposal artifacts, but the validator remains the authority for saddles and barriers.

### Decision 4: Candidate Generation and Validation Are Separate

Rationale:

- A plausible product geometry is not automatically a valid KMC event.
- Physical validation is necessary for transition states, barriers, and rates.

### Decision 5: Product Candidate Is Only Part of Event Proposal

Rationale:

- KMC events are basin-escape channels, not only product structures.
- The first MVP should learn active atoms, event directions, and product candidates from R/P pairs.
- Barrier, rate, TS, committor, and priority heads should be added only when labels or validators exist.

### Decision 6: Basin-Level Evaluation

Rationale:

- One-to-one reconstruction does not measure event discovery.
- KMC needs coverage over important events from each basin.

### Decision 7: EON Integration Belongs Later on the EON Side

Rationale:

- EON already owns AKMC orchestration, saddle search, minimization, barrier calculation, prefactor calculation, and process scheduling.
- Calling EON from BasinFlow during early dataset/model development would add integration work without improving the training chain.
- The cleaner boundary is for BasinFlow to export proposals that EON can consume later as an `ml_proposal` suggestion source.

Implication:

- Do not add an EON runtime dependency to BasinFlow Stage 2-5 work.
- Stage 6 should define proposal artifacts such as product guesses, TS/saddle guesses, displacement-like structures, and direction fields for EON-side consumption.

### Decision 8: Old Heuristic Baseline Stage Is Not Required as a Main Stage

Rationale:

- The former Stage 2 plan around random local displacement, active-region perturbation, hop-like moves, and bond/local-neighbor heuristics would become system-specific boilerplate for the current single-movable-Au dataset.
- It does not directly advance the central neural event-proposal problem.
- Such heuristics remain useful only as seed types, interface smoke tests, or simple sanity checks.

Implication:

- Stage 2 is now EON-style event dataset integration, not baseline proposer implementation.
- Non-learning baselines can be added later only when they answer a concrete benchmark question.

## Technical Risks

### Risk: Generated Products Do Not Connect to Reactant

Mitigation:

- Add saddle/path validation stage.
- Track invalid and disconnected candidates explicitly.
- Use event-library feedback to improve seeds.

### Risk: Event-Direction Labels Are Noisy

Mitigation:

- Derive active atoms from MIC displacement with configurable thresholds.
- Prefer explicit `move_mask` labels when present.
- Treat active/direction heads as auxiliary targets, not the only source of truth.
- Evaluate whether they improve validation cost and basin recall before expanding the model.

### Risk: Seed Conditioning Becomes Weak Metadata

Mitigation:

- Require every `EventSeed` to have scalar conditions, vector conditions, and a flow initial geometry.
- Treat seed displacement or pseudo-velocity as equivariant/vector information during message passing.
- Test that different seeds can produce different candidates for the same basin.

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

### Risk: Premature EON Coupling Slows Model Development

Mitigation:

- Keep early BasinFlow training and benchmark code independent of EON runtime calls.
- Use EON-style data files as input, but defer live AKMC integration to Stage 6.
- Implement or document the EON-side `ml_proposal` interface only after the standalone proposal benchmark is useful.

### Risk: Molecule and Crystal Support Diverge

Mitigation:

- Use a shared `StructureRecord`.
- Keep PBC fields optional.
- Test molecular and periodic examples in every core module.

## Open Questions

- How much of the current `/Users/wx/Desktop/benchmark/au/events` Au dataset should be used for the first train/validation/test split?
- Which additional molecular or periodic datasets should be added after the Au EON-style dataset?
- Which relaxation backend will be used first?
- Which saddle-search validator should be targeted first after the proposal benchmark exists?
- What exact EON-side proposal artifact format should Stage 6 export?
- How should local environments be canonicalized for event-library reuse?

## Current Recommended Next Decision

Implement Stage 2 around the existing EON-style Au dataset:

- Parse `basin_table.csv`.
- Preserve `move_mask`, fixed atoms, reactant/product/TS frames, cell, and PBC.
- Split by basin, not by individual event.
- Derive active-atom, product-displacement, event-direction, and optional TS targets.

This should happen before neural model implementation.
