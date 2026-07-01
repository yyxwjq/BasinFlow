"""
End-to-end pipeline demo — from raw extxyz events to trainable batches.

Demonstrates Stage 1 (Geometry & Data Core) and Stage 2 (EON-style
Event Dataset Integration) capabilities using the Au₁₀₁ benchmark.

Usage::

    PYTHONPATH=src python examples/demo_pipeline.py [events_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1.  Load the full dataset
# ---------------------------------------------------------------------------
print("=" * 62)
print("  Stage 1+2  ·  Geometry & EON Event Dataset  ·  Pipeline Demo")
print("=" * 62)

events_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop/events"

from fscgp.data import read_events_directory

ds = read_events_directory(events_dir)
print(f"\n📦  Loaded from  {events_dir}")
print(f"    {len(ds.basin_ids):>4}  basins")
print(f"    {len(ds.event_ids):>4}  events")
print(f"    {len(ds.structures):>4}  structure records")

# ---------------------------------------------------------------------------
# 2.  Inspect a structure — constraints → skeleton / active region
# ---------------------------------------------------------------------------
rid = ds.basins["0"].reactant_structure_id
r = ds.structures[rid]
n_fixed = int(r.constraints.sum()) if r.constraints is not None else 0
n_free = r.n_atoms - n_fixed

print(f"\n🔬  Example structure  {rid}")
print(f"    {r.n_atoms} atoms  ·  {len(set(r.species))} species")
print(f"    cell     {r.cell[0,0]:.2f} × {r.cell[1,1]:.2f} × {r.cell[2,2]:.2f}  Å")
print(f"    pbc      {r.pbc.tolist()}")
print(f"    fixed    {n_fixed}  atoms  (structural skeleton)")
print(f"    mobile   {n_free}  atoms  (trainable region)")

# ---------------------------------------------------------------------------
# 3.  Basin-level train / val / test split
# ---------------------------------------------------------------------------
from fscgp.data import split_basins

train, val, test = split_basins(ds, train=0.7, val=0.15, test=0.15, seed=42)

print(f"\n📊  Basin-level split  (70 / 15 / 15,  seed=42)")
print(f"    train   {len(train.basin_ids):>4}  basins,  {len(train.event_ids):>4}  events")
print(f"    val     {len(val.basin_ids):>4}  basins,  {len(val.event_ids):>4}  events")
print(f"    test    {len(test.basin_ids):>4}  basins,  {len(test.event_ids):>4}  events")

all_ids = set(train.basin_ids) | set(val.basin_ids) | set(test.basin_ids)
assert all_ids == set(ds.basin_ids), "basin leakage!"
print("    ✓  no basin leakage between splits")

# ---------------------------------------------------------------------------
# 4.  Pairwise training view — MIC labels + constraints masks + TS data
# ---------------------------------------------------------------------------
item = train.pairwise_item(train.event_ids[0], active_threshold=0.1)

print(f"\n🎯  Pairwise training item  ·  {item['event_id']}")
print(f"    displacement    range  [{item['displacement'].min():.4f},  "
      f"{item['displacement'].max():.4f}]  Å")
print(f"    active atoms    {int(item['active_mask'].sum())}  /  {item['active_mask'].shape[0]}")
print(f"    fixed mask      {int(item['fixed_mask'].sum())}  atoms")
print(f"    movable mask    {int(item['movable_mask'].sum())}  atoms")
print(f"    has TS          {item['has_transition_state']}")
if item["has_transition_state"]:
    print(f"    ts_displacement range  [{item['ts_displacement'].min():.4f},  "
          f"{item['ts_displacement'].max():.4f}]  Å")
print(f"    event dir       shape  {item['event_direction'].shape}")
print(f"    atom mapping    {item['atom_mapping']}")

# Verify derived labels are consistent
assert item["fixed_mask"].sum() + item["movable_mask"].sum() == r.n_atoms
assert not (item["fixed_mask"] & item["movable_mask"]).any()
print("    ✓  fixed/movable masks are consistent")

# ---------------------------------------------------------------------------
# 5.  Verify active_atoms auto-populated from move_mask (Stage 2)
# ---------------------------------------------------------------------------
e0 = ds.events["event_0"]
print(f"\n🏷   EventRecord.active_atoms  ·  event_0")
print(f"    source          {'move_mask (constraints)' if e0.active_atoms else 'MIC fallback'}")
print(f"    active_atoms    {e0.active_atoms}")

# ---------------------------------------------------------------------------
# 6.  Basin view — multi-event groups with transition states
# ---------------------------------------------------------------------------
bitem = train.basin_item(train.basin_ids[0])

print(f"\n🗂   Basin view  ·  {bitem['basin_id']}")
print(f"    known events    {len(bitem['events'])}  →  {bitem['known_event_ids']}")
for ev, ts in zip(bitem["events"], bitem["transition_states"]):
    ts_id = ts.structure_id if ts else "None"
    print(f"      {ev.event_id}  →  product {ev.product_structure_id}  ·  TS {ts_id}")

# ---------------------------------------------------------------------------
# 7.  Collate a mini-batch (ready for model input)
# ---------------------------------------------------------------------------
from fscgp.data.collate import collate_pairwise

batch = collate_pairwise(
    [train.pairwise_item(eid) for eid in sorted(train.event_ids)[:4]]
)

print(f"\n📦  Collated batch  (4 events)")
for key in sorted(batch.keys()):
    val = batch[key]
    if hasattr(val, "shape"):
        print(f"    {key:<22s}  shape {str(val.shape):<16s}  dtype {str(val.dtype):<8s}")
    elif isinstance(val, list):
        print(f"    {key:<22s}  list[{len(val)}]")
    else:
        print(f"    {key:<22s}  {type(val).__name__}")

print(f"\n{'─' * 62}")
print("  ✅  Stage 1+2 pipeline complete — ready for Stage 3 model training.")
print(f"{'─' * 62}")
