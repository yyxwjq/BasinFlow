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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# 1.  Load the full dataset
# ---------------------------------------------------------------------------
print("=" * 62)
print("  Stage 1+2  ·  Geometry & EON Event Dataset  ·  Pipeline Demo")
print("=" * 62)

events_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Desktop/events"

from basinflow.data.catalog import BasinSplit, EventCatalog
from basinflow.data.pyg import EventFlowDataset
from basinflow.seeds import ZeroInit

ds = EventCatalog.from_eon_directory(events_dir)
print(f"\n📦  Loaded from  {events_dir}")
print(f"    {len(ds.basin_ids):>4}  basins")
print(f"    {len(ds.event_ids):>4}  events")
print(f"    {len(ds.structures):>4}  structure records")

# ---------------------------------------------------------------------------
# 2.  Inspect a structure — constraints → skeleton / active region
# ---------------------------------------------------------------------------
example_basin_id = ds.basin_ids[0]
example_event_id = ds.event_ids[0]
rid = ds.basins[example_basin_id].reactant_structure_id
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
split = BasinSplit.create(ds, train=0.7, val=0.15, test=0.15, seed=42)
train = split.select(ds, "train")
val = split.select(ds, "val")
test = split.select(ds, "test")

print(f"\n📊  Basin-level split  (70 / 15 / 15,  seed=42)")
print(f"    train   {len(train.basin_ids):>4}  basins,  {len(train.event_ids):>4}  events")
print(f"    val     {len(val.basin_ids):>4}  basins,  {len(val.event_ids):>4}  events")
print(f"    test    {len(test.basin_ids):>4}  basins,  {len(test.event_ids):>4}  events")

all_ids = set(train.basin_ids) | set(val.basin_ids) | set(test.basin_ids)
assert all_ids == set(ds.basin_ids), "basin leakage!"
print("    ✓  no basin leakage between splits")

# ---------------------------------------------------------------------------
# 4.  Pairwise training view — PyG flow sample with MIC labels
# ---------------------------------------------------------------------------
pairwise_source = train if train.event_ids else ds
target = pairwise_source.event_target(pairwise_source.event_ids[0], active_threshold=0.1)
flow_dataset = EventFlowDataset(pairwise_source, [ZeroInit()], flow_time=0.5)
item = flow_dataset[0]
fixed_mask = ~item.movable_mask

print(f"\n🎯  EventFlowDataset item  ·  {item.event_id}")
print(f"    displacement    range  [{target.displacement.min():.4f},  "
      f"{target.displacement.max():.4f}]  Å")
print(f"    active atoms    {int(item.target_active_mask.sum())}  /  {item.num_nodes}")
print(f"    fixed mask      {int(fixed_mask.sum())}  atoms")
print(f"    movable mask    {int(item.movable_mask.sum())}  atoms")
print(f"    has TS          {target.transition_state is not None}")
print(f"    event dir       shape  {tuple(item.target_direction.shape)}")
print(f"    source position shape  {tuple(item.source_pos.shape)}")

# Verify derived labels are consistent
assert fixed_mask.sum() + item.movable_mask.sum() == item.num_nodes
assert not (fixed_mask & item.movable_mask).any()
print("    ✓  fixed/movable masks are consistent")

# ---------------------------------------------------------------------------
# 5.  Verify active_atoms auto-populated from move_mask (Stage 2)
# ---------------------------------------------------------------------------
example_event = ds.events[example_event_id]
print(f"\n🏷   EventRecord.active_atoms  ·  {example_event_id}")
print(f"    source          {'move_mask (constraints)' if example_event.active_atoms else 'MIC fallback'}")
print(f"    active_atoms    {example_event.active_atoms}")

# ---------------------------------------------------------------------------
# 6.  Basin view — multi-event groups with transition states
# ---------------------------------------------------------------------------
basin_source = train if train.basin_ids else ds
basin = basin_source.basins[basin_source.basin_ids[0]]

print(f"\n🗂   Basin view  ·  {basin.basin_id}")
print(f"    known events    {len(basin.known_event_ids)}  →  {basin.known_event_ids}")
for event_id in basin.known_event_ids:
    event = basin_source.events[event_id]
    ts_id = event.transition_state_structure_id or "None"
    print(f"      {event.event_id}  →  product {event.product_structure_id}  ·  TS {ts_id}")

# ---------------------------------------------------------------------------
# 7.  PyG mini-batch (ready for model input)
# ---------------------------------------------------------------------------
from torch_geometric.loader import DataLoader

batch_event_ids = sorted(pairwise_source.event_ids)[:4]
batch_dataset = EventFlowDataset(
    pairwise_source.subset({pairwise_source.events[event_id].basin_id for event_id in batch_event_ids}),
    [ZeroInit()],
    flow_time=0.5,
)
batch = next(iter(DataLoader(batch_dataset, batch_size=len(batch_dataset), shuffle=False)))

print(f"\n📦  PyG batch  ({len(batch_dataset)} event/seed samples)")
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
