# Data Schema

## Design Goal

The schema supports both molecules and periodic crystals, and supports
pairwise training while preserving basin-level evaluation.  The current
MVP adopts a **three-frame event model** — reactant, product, and an
optional transition state.  Path images (NEB bands, etc.) are deferred
to the saddle-validation stage.

---

## StructureRecord

Represents one molecular or periodic structure.

```yaml
structure_id: string
species: [string]
positions: float[N, 3]
cell: float[3, 3] | null      # always normalized to (3,3); null → zero matrix
pbc: bool[3]                   # always normalized to (3,)
charge: int | null
constraints: bool[N] | null    # True = atom is fixed (structural skeleton)
tags: int[N] | null
energy: float | null
forces: float[N, 3] | null
metadata: object
```

Rules:

- `cell` and `pbc` are **normalized on construction**: `cell` is always
  a `(3, 3)` float array, `pbc` is always a `(3,)` bool array.
- `constraints` is a **per-atom boolean mask** where `True` marks atoms
  that are fixed (e.g. a bulk skeleton).  Only unconstrained atoms are
  displaced during generation.  When read from extxyz, ASE `FixAtoms`
  objects are automatically converted to this mask.
- `source_file` is stored in `metadata["source_file"]`, not as a
  top-level field.
- `spin` is deferred to `metadata` for future use.

---

## EventRecord

Represents a known event from one reactant basin to one product basin.

```yaml
event_id: string
reactant_structure_id: string
product_structure_id: string
basin_id: string
transition_state_structure_id: string | null
atom_mapping: [int] | null
active_atoms: [int] | null
event_direction: float[N, 3] | null
validation_status: string       # see recommended values below
metadata: object
```

Recommended `validation_status` values:

- `known_valid`
- `generated_unvalidated`
- `relaxed_candidate`
- `saddle_validated`
- `invalid`
- `duplicate`

### Derived labels

`active_atoms` and `event_direction` are **derived automatically** from
reactant/product coordinate pairs via MIC displacement when loading
events.  They can also be set manually for curated datasets:

```text
displacement_i = MIC(product_i - reactant_i)
active_i       = norm(displacement_i) > threshold
direction_i    = normalize(displacement_i) for active atoms
```

### Fields deferred to later stages

Barrier, rate, temperature, prefactor, and `delta_e` are **not** stored
on `EventRecord`.  They belong to the saddle-validation (Stage 5) and
KMC-export (Stage 6) stages and will live on a separate `ValidatedEvent`
record or in `metadata` when available.

---

## BasinRecord

Groups all known events that originate from the same reactant basin.

```yaml
basin_id: string
reactant_structure_id: string
known_event_ids: [string]
metadata: object
```

`system_type` and `split` are **not** stored on `BasinRecord`:

- `system_type` can be inferred from `reactant.pbc` (all-False =
  molecule; any-True = periodic).
- `split` is assigned at **training time** by `split_basins()`, which
  randomly shuffles basins (not events) into train/val/test groups.
  This prevents information leakage and lets users change ratios
  without editing data files.

---

## CandidateRecord

Stores generated candidates before and after relaxation (Stage 2+).

```yaml
candidate_id: string
basin_id: string
reactant_structure_id: string
initial_seed_id: string
generated_structure_id: string
model_checkpoint: string
sampling_config: object
relaxed_structure_id: string | null
cluster_id: string | null
relaxation_config: object | null
active_atom_scores: float[N] | null
predicted_active_atoms: [int] | null
event_direction: float[N, 3] | null
priority_score: float | null
uncertainty: float | null
status: string
metadata: object
```

Recommended `status` values:

- `generated`
- `relaxed`
- `clustered`
- `matched_known_event`
- `novel_candidate`
- `saddle_validated`
- `invalid`

---

## Raw Data Format

### Event files (`event_*.extxyz`)

One multi-frame ASE extxyz file per event:

- Frame 0 — reactant
- Frame 1 — product
- Frame 2 — transition state (optional)

The three-frame model is the only supported format for the MVP.
Path images are not stored in event files.

Extxyz comment-line properties such as `move_mask` are parsed
automatically by ASE and converted to `StructureRecord.constraints`.

### Basin table (`basin_table.csv`)

Produced by `tools/eon2data.py`.  Maps each event file to its basin:

```csv
global_event,local_event,basin,file
0,0,0,event_0.extxyz
1,1,0,event_1.extxyz
```

The `file` and `basin` columns are consumed by `read_events_directory()`;
other columns are informational.

### Directory layout

```text
events/
├── basin_table.csv
├── event_0.extxyz
├── event_1.extxyz
└── ...
```

---

## Dataset Loading

```python
from fscgp.data import read_events_directory, split_basins

ds = read_events_directory("path/to/events")
train, val, test = split_basins(ds, train=0.7, val=0.15, test=0.15, seed=42)
```

`read_events_directory` scans `event_*.extxyz` files, reads
`basin_table.csv` for basin mapping, and returns an `EventDataset`.
`split_basins` shuffles basins (not events) into reproducible splits.

---

## Training Views

**Pairwise view** — for conditional flow-matching / diffusion training:

```text
(reactant, product, displacement, active_mask, event_direction, metadata)
```

**Basin view** — for candidate-generation benchmarks:

```text
reactant basin, known event set
```

---

## Collation

Batching handles:

- Different atom counts (concatenated along atom axis).
- Different numbers of events per basin (kept as lists).
- Optional periodic cell / pbc fields (stacked per structure).
- `constraints` masks are **not** batched automatically — the model
  receives them per-structure.

The collated output is a plain `dict[str, ...]` of NumPy arrays and
lists, ready for conversion to PyTorch tensors or PyG `Data` objects.
