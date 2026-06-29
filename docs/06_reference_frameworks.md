# Reference Frameworks

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

Limitations for this project:

- Needs adaptation to event-set generation and saddle-search validation.
- The meaning of seed/prior must be redefined for KMC event proposals.

Potential reuse:

- Flow matching objective design.
- Sampler structure.
- Corrector concept.

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
- Use AdsorbDiff for local periodic generation and relaxation-after-generation patterns.
- Use liflow for conditional flow matching and seed-driven diverse sampling.
- Use TrajCast for dynamic PBC graph rollout, multi-frame ASE trajectory handling, and physically meaningful velocity/perturbation seeds.
- Use AMDEN for cached neighborlist updates, refinement-aware generation, and the warning that raw generated coordinates are not relaxed basins.
- Add new project-specific layers for basin-level data, validation, event libraries, and KMC export.
