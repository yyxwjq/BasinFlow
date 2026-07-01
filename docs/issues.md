# Open Design Questions

## 1. Basin folder naming — `isdigit()` restriction

**Status**: deferred.  **Affects**: `tools/eon2data.py:100`.

Current behaviour: only integer-named folders (`0/`, `1/`, …) are
recognised as basin directories.  A folder named `au_cluster_0/` is
silently skipped.

**Option A** — keep `isdigit()`:
- Pro: no code change; simple for small homogeneous datasets.
- Con: basin names carry no semantic information; "which system is
  basin 3?" requires an external lookup.

**Option B** — drop `isdigit()`, accept any folder name:
- Pro: self-documenting basin ids (`surface_defect_3`, `bulk_hop_1`);
  downstream is already string-based, no other code changes needed.
- Con: users must ensure folder names are meaningful and unique.

**When to re-evaluate**: before adding a second benchmark system
(Stage 2–3) where distinguishing `au_cluster_0` from `nacl_defect_0`
becomes important.

---

## 2. Event id source — filename stem vs CSV `global_event`

**Status**: decided (filename stem).  **Affects**: `raw_events.py`.

Event id currently derives from the extxyz filename (`event_0.extxyz`
→ `"event_0"`).  The `global_event` column in `basin_table.csv` is
redundant with the filename suffix and is not read.

If filenames ever diverge from the `event_<N>` convention (e.g. a user
renames files), the CSV `global_event` column could serve as an
alternative authority.  For now the filename convention is sufficient.

---

## 3. `local_event` / `global_event` metadata

**Status**: available but unused.  **Affects**: `raw_events.py`.

The CSV columns `global_event` and `local_event` are informational.
They could be injected into `EventRecord.metadata` so that downstream
code can query integer event ordinals without parsing filenames.

Low priority — no current use case.

---

## 4. Geometry invariance tests

**Status**: done.  **Affects**: `tests/test_geometry_invariance.py`.

Translation, rotation, permutation, and MIC-safe periodic translation
tests are present for pairwise geometry labels.

---

## 5. PyG `Data` bridge

**Status**: deferred.  **Affects**: `collate.py`.

Collated batches are plain dicts.  `AGENTS.md` says "Prefer PyTorch
Geometric-style graph data."  A `dict_to_pyg_data()` converter or
direct PyG output from collate should be added when PyTorch Geometric
enters the dependency list (Stage 3).

---

## 6. JSON/YAML metadata persistence

**Status**: missing.  **Affects**: `dataset.py`, workflow scripts.

No utilities exist to save/load split assignments, basin descriptions,
or dataset manifests.  `split_basins()` results are lost when the
process exits.

Needed before benchmark comparisons require exact split reproduction
(Stage 4).
