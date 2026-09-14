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
movable_mask: bool[N] | null   # True = atom may move; null → all atoms movable
tags: int[N] | null
energy: float | null
forces: float[N, 3] | null
metadata: object
```

Rules:

- `cell` and `pbc` are **normalized on construction**: `cell` is always
  a `(3, 3)` float array, `pbc` is always a `(3,)` bool array.
- `movable_mask` is the **canonical per-atom movement mask** where
  `True` marks atoms that may be displaced during generation, training,
  and relaxation.  Missing masks default to all atoms movable.
- Raw `move_mask` properties are treated as `True = movable` and are
  converted to `movable_mask` when reading ASE structures.  ASE
  `FixAtoms` constraints and legacy `constraints` inputs are accepted as
  compatibility paths and converted to `movable_mask` (`False` for fixed
  atoms).  Core APIs should not expose a separate `fixed_mask`; use
  `~movable_mask` when fixed atoms are needed locally.
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
- `split` is assigned at **training time** by a `BasinSplit`, which records
  train/validation/test basin ids and the random seed.  `EventCatalog.subset()`
  only accepts basin ids, preventing one basin's events from crossing splits.

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

### Event files (`event_*.traj` or `event_*.extxyz`)

One multi-frame ASE trajectory or extxyz file per event:

- Frame 0 — reactant
- Frame 1 — product
- Frame 2 — transition state (optional)

The three-frame model is the only supported format for the MVP.
Path images are not stored in event files.

ASE trajectory files (`*.traj`) are preferred for EON conversions because
they preserve `FixAtoms` constraints.  Extxyz atom properties such as
`move_mask` are parsed by ASE and converted to
`StructureRecord.movable_mask` when extxyz is used.

### Basin table (`basin_table.csv`)

Produced by `tools/eon_to_events.py`.  Maps each event file to its basin:

```csv
global_event,local_event,basin,file
0,0,0,event_0.extxyz
1,1,0,event_1.extxyz
```

The `file` and `basin` columns are consumed by `EventCatalog.from_eon_directory()`;
other columns are informational.

### Directory layout

```text
events/
├── basin_table.csv
├── event_0.traj
├── event_1.traj
└── ...
```

---

## Dataset Loading

```python
from basinflow.data import BasinSplit, EventCatalog

catalog = EventCatalog.from_eon_directory("path/to/events")
split = BasinSplit.create(catalog, train=0.7, val=0.15, test=0.15, seed=42)
train_catalog = split.select(catalog, "train")
```

`EventCatalog.from_eon_directory()` scans `event_*.extxyz` files and falls
back to `event_*.traj`, preserves `basin_table.csv` order, and accepts only
two-frame (R/P) or three-frame (R/P/TS) events.  The separate
`tools/transition1x_to_events.py` writes this same event-directory contract from a
Transition1x pickle.  It maps every selected reaction to a
`transition1x:<source_index>` pseudo-basin; this source is for pairwise
no-relaxation geometry diagnostics, not a KMC basin benchmark.

---

## Training Views

**`EventFlowDataset`** — pairwise conditional-flow training:

```text
EventData(z, pos=x_t, reactant_pos, source_pos, movable_mask, active_prior,
          seed_displacement, seed_direction, cell, pbc, flow_time, seed_type_id,
          target_pos, target_velocity, target_active_mask, target_direction)
```

`target_pos` is the MIC-unwrapped product geometry.  Fixed atoms have zero
target velocity and target direction.  TS data remains in `EventCatalog` for
the future R/P-to-TS Stage 4 view and is not zero-filled into Stage 3 batches.

**`BasinDataset`** — target-free candidate inference:

```text
EventData(z, pos=reactant_pos + seed_displacement, reactant_pos, source_pos,
          movable_mask, active_prior, seed_displacement, seed_direction,
          cell, pbc, flow_time, seed_type_id, basin_id)
```

It contains no product coordinates, `target_*` labels, true event direction,
or TS information.  Known events remain in the catalog for post-sampling
benchmark matching only.

---

## Collation

The standard PyG `DataLoader` batches `EventData`; no project-specific dict
collate stage exists.  Atom fields concatenate and `batch`/`ptr` identify the
graph.  `cell`, `pbc`, `flow_time`, and `seed_type_id` stay graph-level:

```text
pos:          [N_0 + ... + N_B, 3]
batch / ptr:  atom-to-graph assignment and graph offsets
cell:         [B, 3, 3]
pbc:          [B, 3]
flow_time:    [B, 1]
seed_type_id: [B, 1]
```

Models broadcast graph-level conditions with `batch` and rebuild periodic
neighbor graphs from the current `pos`; static edges are not stored as a data
contract.

Training accepts a positive integer batch size or `full`.  `full` means
all event-init flow items in one epoch are evaluated in one optimizer
update; `training.log` therefore records optimizer update steps rather
than raw event-init sample counts.
