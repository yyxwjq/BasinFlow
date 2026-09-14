# Open Design Questions

## 1. Basin folder naming — `isdigit()` restriction

**Status**: deferred.  **Affects**: `tools/eon_to_events.py:100`.

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

Event id currently derives from the event filename (`event_0.traj` or
`event_0.extxyz` → `"event_0"`).  The `global_event` column in `basin_table.csv` is
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

## 5. Direct PyG datasets

**Status**: done.  **Affects**: `data/pyg.py`.

`EventFlowDataset` and `BasinDataset` emit `EventData` objects directly.
The former NumPy pairwise-collation bridge has been removed from the
delivery path.

---

## 6. JSON/YAML metadata persistence

**Status**: minimally done.  **Affects**: `dataset.py`, workflow scripts.

`BasinSplit.save()` and `BasinSplit.load()` persist ordered
train/validation/test basin ids and the split seed as JSON.  This is enough
to reproduce current basin splits.

Still missing: richer dataset manifests, basin descriptions, experiment
summaries, and model/sampling metadata.  These remain needed before full
benchmark comparisons require exact experiment reproduction (Stage 5).

---

## 7. Active prior versus active label leakage

**Status**: fixed at the Stage 3 flow boundary.  **Affects**:
`flow/targets.py`, `models/egnn_product_flow.py`, `sampling/candidate_sampler.py`.

Earlier Stage 3 batches used `active_mask` both as a model input and as
the supervised active-head label.  That made training/inference semantics
ambiguous because true event active labels are unavailable during
basin-level sampling.

Current behavior:

- `active_prior` is the model input condition from the seed.
- `target_active_mask` is the supervised label used only by losses.
- `CandidateSampler` builds inference batches with `active_prior` and no
  target active labels.

---

## 8. Stage 3 initialization/proposal quality gap

**Status**: open, now measurable.  **Affects**:
`examples/train_product_flow.py`, `examples/sample_product_flow.py`,
`sampling/candidate_sampler.py`, `benchmark/product_flow_quality.py`.

The Stage 3 EGNN can now be evaluated separately in two modes:

- pairwise product-flow rollout RMSD on train/val/test splits.
- basin-level candidate recall/RMSD from the inference sampler.

This separates two failure modes:

- poor `R + initialization -> P` flow fitting.
- poor basin-level initialization/proposal coverage.

The current Au quality run shows the second problem clearly: pairwise
test rollout RMSD is measurable below the raw basin sampling RMSD, while
basin-level recall remains zero with only two fixed initialization generators per
basin.  The next Stage 3 improvement should therefore prioritize richer
or learned basin-level initialization/event-channel proposal before moving to
Stage 4 TS flow.
