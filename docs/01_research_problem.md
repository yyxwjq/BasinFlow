# Research Problem

## Problem Statement

Given a reactant structure or reactant basin, propose multiple physically meaningful KMC escape-event candidates. Each event proposal may include active atoms, an escape direction, a product candidate, and later a transition-state guess or validation-priority score.

For a KMC event table, the relevant object is not a single product structure. It is a set of possible transitions from the current basin, each with a product basin, transition state, barrier, prefactor or rate, and validity domain.

Therefore, product generation is only one component of the research problem. The central object is an event channel:

```text
R_basin -> active region -> escape direction -> P_basin -> TS/barrier/rate
```

## Why One-to-One Data Is Insufficient

Many available datasets are stored as event samples:

```text
R_i -> P_i
```

This is useful for supervised training, but it does not directly evaluate event discovery. If each reactant appears with only one product, the model can be tested for reconstruction-like generation, but not for whether it discovers all relevant events from a basin.

For KMC/AKMC, the useful benchmark should be:

```text
R_basin -> {event_1, event_2, ..., event_n}
```

The model should be evaluated on whether it covers the important event set, especially low-barrier and high-rate events.

## Candidate Is Not Validation

A generated structure is only a proposal. It becomes a valid event only after:

1. Geometry sanity checks.
2. Local relaxation to a stable minimum.
3. Clustering against known and generated minima.
4. Optional path connection checks.
5. Saddle or transition-state validation.
6. Barrier and rate estimation.

Generated products not matching known references are not automatically wrong. They may be novel events, invalid structures, duplicates, or relaxation artifacts.

Likewise, a predicted active atom set or event direction is not proof of a valid event. It is a search hint that should reduce the cost of downstream relaxation and saddle validation.

## Benchmark Metrics

Recommended metrics:

- Product-basin recall.
- Low-barrier event recall.
- Rate-weighted coverage.
- Duplicate candidate rate.
- Invalid candidate rate.
- Novel validated event count.
- Cost per validated event.
- Number of expensive saddle-search calls saved.
- Active-atom precision/recall when displacement-derived labels are available.
- Event-direction angular error on active atoms.
- Validation-priority enrichment for low-barrier events when barrier labels exist.

For adaptive KMC use, rate-weighted coverage is more important than raw product count.

## Core Research Risk

The main risk is that a generative model may produce structures that look chemically plausible but do not connect to the reactant through a relevant saddle. This is why the architecture must include validation and fallback mechanisms from the beginning.
