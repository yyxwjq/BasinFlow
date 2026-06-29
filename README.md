# BasinFlow

Learning-assisted KMC event proposal framework for reaction rate-table
construction.

> **Current stage**: Stage 1 (Geometry & Data Core) — ✅ complete.
> Working toward Stage 2 (Baseline Event Proposal Generator).

## Quick Start

```bash
# Install (requires Python ≥3.10)
pip install -e .

# Convert raw .con files → extxyz events
PYTHONPATH=src python tools/eon2data.py <basin_root_dir>

# Run the full Stage 1 pipeline demo
PYTHONPATH=src python examples/demo_pipeline.py <events_dir>

# Run tests
PYTHONPATH=src pytest tests/
```

## Implemented (Stage 1)

| Module | Status |
|--------|--------|
| `StructureRecord` — unified mol/crystal schema, ASE round-trip, `constraints` (FixAtoms → bool mask) | ✅ |
| `EventRecord` — three-frame model (R/P/TS), auto-derived `active_atoms` + `event_direction` via MIC | ✅ |
| `BasinRecord` — multi-event basin grouping, runtime `split_basins()` (basin-level, no leakage) | ✅ |
| `CandidateRecord` — schema for generated proposals (Stage 2+) | ✅ |
| `read_events_directory()` — extxyz + `basin_table.csv` → `EventDataset` | ✅ |
| PBC-aware MIC displacement, active-atom derivation, event-direction extraction | ✅ |
| Dynamic cutoff-graph builder with cell offsets | ✅ |
| Pairwise & basin-level collation (variable atom counts, variable events/basin) | ✅ |
| 20 unit/integration tests | ✅ |

## TODO

- [ ] **Stage 2** — Baseline event proposal generator (random seeds, relaxation, clustering)
- [ ] **Geometry invariance tests** — translation, rotation, permutation (required before Stage 3)
- [ ] **Stage 3** — Neural event proposer MVP (conditional flow matching, equivariant GNN)
- [ ] PyG `Data` bridge for collated batches
- [ ] JSON/YAML metadata persistence for splits and basin metadata
- [ ] **Stage 4** — Benchmark & analysis pipeline
- [ ] **Stage 5** — Saddle validation interface
- [ ] **Stage 6** — Adaptive KMC integration

See `docs/05_development_plan.md` for the full roadmap.
