# MVP Benchmark

## Purpose

The MVP benchmark should answer one question:

Can the model generate useful and diverse end-state candidates from a reactant basin at lower validation cost than blind local search?

It should not claim to solve full adaptive KMC in the first stage.

## Benchmark Levels

### Level 1: Pairwise Sanity

Input:

```text
R_i, seed -> P_i
```

Purpose:

- Confirm the model learns basic geometry transformation.
- Check molecular and periodic I/O.
- Verify equivariance, PBC handling, and relaxation.

This level does not prove event discovery.

### Level 2: Basin-Level Recall

Input:

```text
R_basin -> {P_1, P_2, ..., P_n}
```

Purpose:

- Generate many candidates from one reactant basin.
- Relax and cluster candidates.
- Compare clusters with known product basins.

This is the first meaningful benchmark for the project.

### Level 3: Validation-Aware Benchmark

Input:

```text
R_basin -> model proposals -> saddle/path validation
```

Purpose:

- Measure valid event discovery under a fixed expensive-validation budget.
- Compare model-guided proposals to random perturbation, high-temperature MD, dimer, or ARTn baselines.

## Recommended Metrics

Structure generation metrics:

- Relaxation success rate.
- Product cluster diversity.
- Duplicate rate.
- Invalid geometry rate.
- Known product match rate.

Event discovery metrics:

- Event recall.
- Low-barrier event recall.
- Rate-weighted recall.
- Novel validated event count.
- Expensive validation calls per valid event.
- Missed high-rate event count.

Cost metrics:

- Number of generated candidates.
- Number of relaxation calls.
- Number of saddle-search calls.
- Wall time per basin.
- Speedup at fixed recall.

## Suggested Initial Systems

Use small, interpretable systems first:

- Molecular rearrangement reactions from OAReactDiff-like data.
- Surface adsorbate diffusion or rotation events.
- Vacancy or interstitial hops in a simple periodic crystal.
- Small defect or local reconstruction events.

Avoid large multi-component crystals in the first benchmark. They make failure analysis too difficult.

## Baselines

Recommended baselines:

- Random local perturbation plus relaxation.
- Mode-following or local displacement heuristic.
- Existing AKMC/dimer/ARTn proposal budget.
- One-to-one model sampling without basin-level diversity control.

The model should be compared against cost-aware baselines, not only against reconstruction error.
