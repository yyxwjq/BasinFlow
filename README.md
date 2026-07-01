# BasinFlow

Learning-assisted KMC event proposal framework for reaction rate-table
construction.

> **Current stage**: Stage 2 (EON-style Event Dataset Integration).
> Stage 1 (Geometry & Data Core) is complete.

## Quick Start

```bash
# Use the project development environment
conda activate ifdiff
python -m pip install -e .

# Convert raw .con files → extxyz events
python tools/eon2data.py <basin_root_dir>

# Run the full data pipeline demo
python examples/demo_pipeline.py <events_dir>

# Run tests
python -m pytest

# Optional real-data Stage 2 integration check
BASINFLOW_EVENTS_DIR=/Users/wx/Desktop/events \
python -m pytest tests/test_raw_events.py::test_stage2_real_events_dataset_when_env_is_set -q
```

## Implemented (Stage 1)

| Module | Status |
|--------|--------|
| `StructureRecord` — unified mol/crystal schema, ASE round-trip, `constraints` (FixAtoms → bool mask) | ✅ |
| `EventRecord` — three-frame model (R/P/TS), auto-derived `active_atoms` + `event_direction` via MIC | ✅ |
| `BasinRecord` — multi-event basin grouping, runtime `split_basins()` (basin-level, no leakage) | ✅ |
| `CandidateRecord` — schema for generated proposals (later stages) | ✅ |
| `read_events_directory()` — extxyz + `basin_table.csv` → `EventDataset` | ✅ |
| PBC-aware MIC displacement, active-atom derivation, event-direction extraction | ✅ |
| Dynamic cutoff-graph builder with cell offsets | ✅ |
| Pairwise & basin-level collation (variable atom counts, variable events/basin) | ✅ |
| 20 unit/integration tests | ✅ |

## TODO

- [ ] **Stage 2** — EON-style R/P/TS event dataset integration
- [ ] **Geometry invariance tests** — translation, rotation, permutation (required before Stage 3)
- [ ] **Stage 3** — Seed-conditioned product/event flow
- [ ] PyG `Data` bridge for collated batches
- [ ] JSON/YAML metadata persistence for splits and basin metadata
- [ ] **Stage 4** — React-OT-style TS flow
- [ ] **Stage 5** — Basin-level event proposal benchmark
- [ ] **Stage 6** — EON-side ML suggestion integration

See `docs/05_development_plan.md` for the full roadmap.
