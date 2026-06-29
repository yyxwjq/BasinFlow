# Project Vision

## Objective

Build a learning-assisted framework that proposes KMC escape events for KMC/AKMC event-table construction.

The intended input is a reactant structure or reactant basin. The intended output is a set of event proposals. Each proposal may contain active atoms, an event direction or displacement field, a candidate product structure, and an optional validation-priority score. Candidate products are relaxed, clustered, and validated through physical saddle-search or path-search methods.

## Why This Project Exists

Traditional adaptive KMC relies on expensive local event discovery, often through high-temperature MD, dimer, ARTn, NEB-assisted workflows, or other saddle-search procedures. These methods are physically grounded but can spend substantial compute exploring redundant, high-barrier, or unproductive directions.

The goal here is to use a generative model as a proposal mechanism:

```text
current basin -> event proposals -> candidate products -> cheap filters -> selective saddle validation
```

The model should reduce blind search cost, not eliminate physical validation.

## Scientific Framing

The correct learning target is not simply:

```text
reactant -> one product
```

The useful target is:

```text
reactant basin -> distribution over basin-escape event channels
```

Pairwise event samples can still be used for training, but the project should organize benchmark and evaluation around basin-level event sets.

An event channel is more than a product structure:

```text
active atoms + event direction + product basin + optional TS/barrier/rate
```

The first neural MVP should learn only the parts that can be supervised reliably from reactant/product pairs:

```text
MIC(product - reactant)
active atom mask derived from displacement
event direction derived from displacement
```

Transition-state, barrier, rate, and priority heads are later extensions when labels or validators are available.

## Initial Success Criteria

The MVP is successful if it can:

- Read molecular and periodic event samples into one unified data structure.
- Generate multiple event proposals from one reactant basin.
- Predict or derive active atoms and event directions for proposals.
- Relax and cluster generated structures into distinct local minima.
- Compare generated minima against known event products.
- Report event recall, duplicate rate, invalid rate, and cost per validated event.

## Long-Term Success Criteria

The full project is successful if it can:

- Increase low-barrier event recall under a fixed saddle-search budget.
- Prioritize candidates that are more likely to become useful KMC events.
- Reduce redundant validation attempts.
- Support adaptive KMC event-library growth.
- Reuse known events across similar local environments.
- Fall back to conventional saddle search when model confidence is low.
