# Data Schema

## Design Goal

The schema supports both molecules and periodic crystals, and supports
pairwise training while preserving basin-level evaluation.  The current
MVP adopts a **three-frame event model** — reactant, product, and an
optional transition state.  Path images (NEB bands, etc.) are deferred
to the saddle-validation stage.

---

## Structure

Represents one molecular or periodic frame. Five fields, nothing else.

```yaml
species: [string]
positions: float[N, 3]
cell: float[3, 3] | null      # always normalized to (3,3); null -> zero matrix
pbc: bool[3]                   # always normalized to (3,)
movable_mask: bool[N] | null   # True = atom may move; null -> all atoms movable
```

Rules:

- `cell` and `pbc` are **normalized on construction**: `cell` is always a
  `(3, 3)` float array, `pbc` is always a `(3,)` bool array. Molecular data
  carries a zero matrix and an all-`False` vector; the fields are never omitted,
  because `pbc` -- not a nonzero cell -- decides periodic behavior.
- `movable_mask` is a **given constraint**, not a learning target: an EON
  `move_mask`, ASE `FixAtoms`, or "everything moves" by default. Raw `move_mask`
  is treated as `True = movable`; `FixAtoms` is converted to `movable_mask`
  (`False` for fixed atoms). Disagreement between the two is an error rather
  than a silent precedence. Use `~movable_mask` when fixed atoms are needed.
- There is **no** `structure_id`, `metadata`, `charge`, `energy`, `forces`,
  `tags` or `constraints`: a `Structure` is geometry plus its movement
  constraint. Energy and forces are not consumed by any production path, and the
  event-level `metadata` lives on `Event` (see below).
- `to_ase()` rebuilds `FixAtoms` from `~movable_mask`, so an ASE-driven
  relaxation keeps fixed atoms fixed without a second constraint field.

---

## Event

Represents one known event from one reactant basin to one product basin.
The three frames are **inlined**, not referenced through a structure registry.

```yaml
event_id: string
basin_id: string
reactant: Structure
product: Structure
transition_state: Structure | null
metadata: object
```

`Event.__post_init__` validates the invariant that the three frames agree on
atom count, species order, cell and `pbc`. There is no `atom_mapping`,
`active_atoms`, `event_direction` or `validation_status`: the first was never
read, and the last three are either derived or owned by the stage that produces
them. Barrier, rate, prefactor and `delta_e` remain out of scope (Stage 5/6).

### The one supervised quantity: `event_target(event)`

```text
displacement     = MIC(product - reactant)
target_positions = reactant + displacement      # the lifted endpoint
```

The lifted endpoint is the reaction coordinate's endpoint, not the recorded
product geometry: for a periodic system the two differ by a lattice vector, and
only the lifted one crosses the cell boundary without a discontinuity.

Active-atom labels and event directions are **not** produced. Both are functions
of `displacement` (``active_i = norm(displacement_i) > threshold``,
``direction_i = normalize(displacement_i)``), and a derived label shipped next to
its source is only a second thing to keep in sync.

---

## Basin index

```yaml
basins: {basin_id: [event_id, ...]}
```

A basin is a plain mapping entry, not a record class: it has no state of its own.
`EventCatalog.__post_init__` checks that every event names a known basin, that
every listed event belongs to the basin that lists it, and that all events of one
basin share a reactant (same species, cell, `pbc` and `movable_mask`) -- otherwise
"the basin's reactant", which inference samples from, would be ambiguous.

`basin_reactant(catalog, basin_id)` returns it.

`system_type` is inferred from `reactant.pbc` (all-`False` = molecule, any-`True`
= periodic). Partition membership is assigned at training time by a `BasinSplit`,
which records train/validation/test basin ids and the seed. Selection is by whole
basin only, so no event of a basin can cross a split.

---

## Candidate (sampling layer, not a data-layer record)

`basinflow.sampling.Candidate` records one generated proposal:

```yaml
candidate_id: string
generated_structure_id: string
seed_id: string
seed_type: string
sampling_config: object
```

Relaxation status, clustering, scores and validation status belong to the stages
that produce them; the earlier data-layer record carried fields for all of them
that nothing ever wrote. Its `active_atom_scores`, `predicted_active_atoms` and
`event_direction` are gone with the heads that filled them.

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
`Structure.movable_mask` when extxyz is used.

### Basin table (`basin_table.csv`)

Produced by `tools/eon_to_events.py`.  Maps each event file to its basin:

```csv
global_event,local_event,basin,file
0,0,0,event_0.extxyz
1,1,0,event_1.extxyz
```

The `file` and `basin` columns are consumed by `load_events()`; other columns
(e.g. `dataset_kind`, `source_index`, `source_partition`) are informational and
land in `event.metadata`.

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

One mode: an event directory, plus an optional split manifest.

```python
from basinflow.data import find_split_manifest, load_events, load_split, split_basins

catalog = load_events("path/to/events")
manifest = find_split_manifest("path/to/events")     # <dir>/split_manifest.json or its parent
split = load_split(manifest) if manifest else split_basins(catalog, seed=42)
train_catalog = split.select(catalog, "train")
```

`load_events()` globs `event_*.traj` and `event_*.extxyz`, consumes
`basin_table.csv` when present (preserving its row order and using its `basin`
column), and accepts only two-frame (R/P) or three-frame (R/P/TS) events. Frame
`info` wins over the caller's metadata, so a converted dataset keeps naming the
pickle its `source_file` came from.

An empty partition is legal data -- the 0910 dataset ships `test: []` -- but
`split.select(catalog, "test")` raises on it rather than producing a silently
empty dataset. A config that still carries a removed key is rejected by
`basinflow.workflows.config.read_config`, so an old option cannot be silently
dropped from an experiment.

`tools/transition1x_to_events.py` writes this same event-directory contract from
a Transition1x pickle, mapping every selected reaction to a
`transition1x:<source_index>` pseudo-basin. This source is for pairwise
no-relaxation geometry diagnostics, not a KMC basin benchmark.
`tools/prepare_transition1x_explicit_split.py` writes one `events/` directory plus
a sibling `split_manifest.json`, and `tools/merge_event_partitions.py` converts an
older three-directory export into that layout.

---

## Training Views

**`EventFlowDataset`** — pairwise conditional-flow training:

```text
EventData(z, pos=x_t, reactant_pos, source_pos, movable_mask, active_prior,
          seed_displacement, seed_direction, cell, pbc, flow_time, seed_type_id,
          target_pos, target_velocity)
```

`pos` is the *current flow state* `x_t`, not a product geometry; the product is
supervision only. `target_pos` is the MIC-lifted product geometry. Fixed atoms
have zero target velocity. There is no `target_active_mask` or `target_direction`.

**`TransitionStateDataset`** — `R + P -> TS` (mode B, `docs/23`):

```text
EventData(z, pos=x_t, reactant_pos, endpoint_pos, source_pos, movable_mask,
          active_prior=all False, seed_displacement=0, seed_direction=0,
          cell, pbc, flow_time, seed_type_id, target_pos, target_velocity)
```

Here the product is a **conditioning input** (`endpoint_pos`), because the task is
defined to receive it; the TS is supervision. That is not the leak the same field
would be in mode A, and the two views are separate classes so it stays visible.

**`BasinDataset`** — target-free candidate inference:

```text
EventData(z, pos=reactant_pos + seed_displacement, reactant_pos, source_pos,
          movable_mask, active_prior, seed_displacement, seed_direction,
          cell, pbc, flow_time, seed_type_id, basin_id)
```

It constructs no `target_*` field at all, so an oracle cannot reach the model even
by accident. Known events remain in the catalog for post-sampling benchmark
matching only.

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
