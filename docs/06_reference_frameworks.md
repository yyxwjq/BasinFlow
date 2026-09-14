# Reference Frameworks

## Benchmark protocol audit (2026-09-10 follow-up)

The local ReactOT default is deterministic R/P-conditioned TS generation;
MolGEN's shipped entry points also primarily exercise TS generation despite
its product-conditioning branch. Neither default TS metric is an R-only
product-generation baseline. See `17_reference_protocol_audit.md` for verified
data selection, source-noise, objective, optimizer, solver, RMSD and candidate
selection differences, and the separately labeled custom 9000/536/537 split.

## 2026-09-10 implementation update

The new backend is a local, pure-PyTorch dual-geometry PaiNN, organized as
`models/painn/{painn,layers,modules}.py`. LiFlow informs the separate reference
and flow-state radial filters, scalar/vector messages, invariant gates, and
tensor/flow adapter boundary. AdsorbDiff informs Gaussian/Bessel filter choices
and smooth polynomial/cosine envelopes. The implementation follows the equations
independently; no reference repository or complete module was imported.

Gaussian initialization supplies the initial flow geometry; its original random
vector is not persistently fed to the PaiNN velocity field. Training redraws
reproducible noise each epoch. Velocity-only supervision is now the default;
active atoms and directions are derived from final candidate displacement.
Physical seed conditioning can be added explicitly when it is distinct from
the transport noise, rather than restoring a hidden source-noise shortcut.

See `12_transition1x_split_audit.md` for verified OA source partitions and the
independent R/P coordinate rotations. Source membership replication is distinct
from reproducing OA's object-aware joint diffusion objective or reverse-reaction
augmentation. The local validation file is not an independent test file.

## Policy

Reference frameworks should remain outside this repository unless a small, specific module is intentionally ported.

Do not copy entire repositories into this project. Record paths, lessons, and porting decisions here.

For code-level file references and implementation mapping, read `docs/09_reference_code_notes.md`.

## OAReactDiff

Path:

```text
/Users/wx/Desktop/yyxwjq/OAReactDiff
```

Useful ideas:

- Reaction-conditioned molecular diffusion framing.
- Joint handling of reactant and product structures.
- Pairwise reaction sample organization.
- Molecular graph generation and training-loop conventions.

Limitations for this project:

- Primarily molecular rather than periodic.
- Pairwise `R -> P` framing is insufficient for event-set discovery.
- Static molecular graph assumptions do not directly solve periodic neighbor updates.

Potential reuse:

- Dataset organization patterns.
- Conditional generation training loop ideas.
- Molecular baseline examples.

Transition1x conversion decision (2026-09-01):

- BasinFlow converts OAReactDiff's columnar `train.pkl` format once through
  `tools/transition1x_to_events.py`; the core package never reads the pickle.
- The `use_ind` selection is retained in source order.  Each selected reaction
  becomes a `transition1x_single_event_molecule` pseudo-basin so it can share
  the pairwise data contract while remaining explicitly distinct from a real
  multi-event KMC basin.
- R, P, and TS can be centroid-centered independently at conversion time to
  remove arbitrary whole-molecule translation.  Only R-to-P product-flow
  supervision is used in the Stage 3 experiment; TS remains in the standard
  event file for later modules, not as a Stage 3 target.

## akmc-product-generation

Path:

```text
/Users/wx/Desktop/yyxwjq/akmc-product-generation
```

Useful ideas:

- User-modified periodic support derived from OAReactDiff.
- ASE/extxyz-style input direction.
- Inclusion of cell, PBC, and cell-offset concepts.
- Reactant-product joint graph adaptation for periodic systems.
- Explicit tensor contracts for periodic joint graphs: `fragment`, `mask`, batched `cell`, `pbc`, and `cell_offsets`.

Known limitations:

- Periodic support is a prototype, not a complete architecture.
- Dynamic neighbor updates during sampling are not fully solved.
- Basin-level multi-event handling is absent.
- The framework is still close to one-to-one event generation.
- Raw input currently uses separate reactant/product files rather than FS-CGP's preferred one-event multi-frame event files.

Potential reuse:

- Existing understanding of where OAReactDiff needs modification.
- PBC fields and data-processing experiments.
- Lessons about what should be redesigned cleanly in the new project.
- Dataset collation conventions for periodic reactant/product joint graphs.

Recommendation:

- Do not continue by directly expanding `akmc-product-generation` as the main codebase.
- Use it as a reference for migration risks and PBC edge cases.

## EGNN and egnn-pytorch

Paths:

```text
/Users/wx/Desktop/yyxwjq/egnn
/Users/wx/Desktop/yyxwjq/egnn-pytorch
```

Useful ideas:

- Clean EGNN message-passing blocks using `edge_index`, scalar edge messages, and equivariant vector updates.
- Sparse/PyG-style graph input conventions without requiring a dense fully connected graph.
- Equivariance tests for translation and rotation behavior.

Limitations for this project:

- The reference implementations do not directly handle BasinFlow's basin/event records, `EventSeed` semantics, or PBC cell-offset edge vectors.
- The PyG implementation is useful as an interface reference, but BasinFlow's first EGNN core should remain plain PyTorch so PBC-aware `edge_vectors` stay explicit.

Potential reuse:

- Local EGNN layer structure.
- Scatter-based aggregation.
- Equivariance test patterns.

## AdsorbDiff

Path:

```text
/Users/wx/Desktop/yyxwjq/AdsorbDiff
```

Useful ideas:

- Conditional generation for adsorbate/surface final configurations.
- Local degree-of-freedom generation rather than full unconstrained crystal generation.
- On-the-fly periodic graph construction.
- Sampling followed by relaxation.

Limitations for this project:

- Focuses on adsorption configuration generation rather than general event discovery.
- Does not by itself solve transition-state validation or KMC event-library construction.

Potential reuse:

- Surface/adsorbate local-region representation.
- PBC graph construction patterns.
- Relaxation-after-generation workflow.

## liflow

Path:

```text
/Users/wx/Desktop/yyxwjq/liflow
```

Useful ideas:

- Conditional flow matching / flow-based trajectory learning.
- Prior or seed state as an exploration initializer.
- Propagator-corrector separation.
- Rollout from current structures.
- A narrow dataset boundary: raw loading and single-sample PyG construction
  are separate from framework batching, so model code receives one consistent
  graph object rather than nested intermediate dictionaries.

Limitations for this project:

- Needs adaptation to event-set generation and saddle-search validation.
- The meaning of seed/prior must be redefined for KMC event proposals.

Potential reuse:

- Flow matching objective design.
- Sampler structure.
- Corrector concept.
- `EventData` plus PyG `DataLoader` responsibility split.  BasinFlow borrows
  this interface discipline only; it does not import Li-ion trajectory-delay
  targets, Lightning lifecycle code, or static periodic graph assumptions.

## React-OT

Local notes and PDFs:

```text
/Users/wx/Downloads/同步空间/obisidan/zotero/akmcgc-reference
/Users/wx/Downloads/同步空间/MyZotero/akmcgc
```

Useful ideas:

- Deterministic or low-variance flow matching for transition-state generation.
- `R/P -> TS` conditioning through interpolation or optimal-transport-style paths.
- Replacement of slow stochastic TS sampling with ODE-style generation.

Limitations for this project:

- It assumes the product is already known.
- It does not solve basin-level discovery of multiple product/event channels from one reactant basin.
- It is a TS-generation reference, not a full AKMC event-table construction framework.

Potential reuse:

- Stage 4 TS flow design.
- R/P interpolation initialization.
- TS coordinate/displacement losses.

## TrajCast

Path:

```text
/Users/wx/Desktop/yyxwjq/trajcast
```

Useful ideas:

- Multi-frame ASE trajectory loading.
- Position and velocity conditioned equivariant prediction.
- Dynamic graph update during rollout.
- Force-free learned propagation with physically meaningful vector inputs.

Limitations for this project:

- It targets ordinary trajectory forecasting, not rare-event proposal.
- Long MD-like rollout is not enough to replace saddle validation.

Potential reuse:

- Pseudo-velocity or event-direction seed encoding.
- Dynamic PBC graph rebuild patterns.
- Rollout and sampling engine structure.

## AMDEN

Path:

```text
/Users/wx/Desktop/yyxwjq/AMDEN-code
```

Useful ideas:

- Material sample and batch records with lattice and PBC fields.
- Cached dynamic neighbor-list updates.
- Generated-sample validation and restart logic.
- Refinement-aware generation.

Limitations for this project:

- The objective is amorphous/material generation rather than AKMC event discovery.
- Ghost atoms and inverse-design assumptions are not required for the MVP.

Potential reuse:

- Cached neighbor-list design.
- Invalid-candidate resampling policies.
- Future refinement interface ideas.

## EON

Path:

```text
/Users/wx/Desktop/yyxwjq/eon
```

Useful ideas:

- Existing AKMC orchestration, saddle search, minimization, barrier, prefactor, and process scheduling.
- Suggestion-source pattern for generating saddle-search displacements.
- Search inputs such as position, displacement, and direction files.

Limitations for this project:

- BasinFlow should not call EON during early model training and benchmarking.
- Direct runtime integration is premature before the proposal model and standalone benchmark exist.

Potential reuse:

- Stage 2 EON-style event data format.
- Stage 6 EON-side `ml_proposal` interface design.
- Export artifacts for product guesses, TS/saddle guesses, displacement-like structures, and direction fields.

## FS-CGP Project Position

This repository is the new clean project that synthesizes selected ideas from the reference frameworks.

It should not be treated as another copied reference framework. Its role is to define the final architecture for learning-assisted KMC/AKMC end-state proposal:

```text
basin-level data -> conditional generation -> relaxation -> clustering -> saddle validation -> event library
```

## Cross-Framework Conclusion

The new project should not be a direct fork of any single framework.

Recommended synthesis:

- Use OAReactDiff for molecular conditional generation concepts.
- Use akmc-product-generation as a record of periodic adaptation issues and PBC tensor contracts.
- Use EGNN and egnn-pytorch for the first plain-PyTorch EGNN product/event-flow backbone and equivariance tests.
- Use AdsorbDiff for local periodic generation and relaxation-after-generation patterns.
- Use liflow for conditional flow matching and seed-driven diverse sampling.
- Use TrajCast for dynamic PBC graph rollout, multi-frame ASE trajectory handling, and physically meaningful velocity/perturbation seeds.
- Use AMDEN for cached neighborlist updates, refinement-aware generation, and the warning that raw generated coordinates are not relaxed basins.
- Add new project-specific layers for basin-level data, validation, event libraries, and KMC export.
