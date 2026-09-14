# Transition1x source-partition audit (2026-09-10)

This audit concerns Stage 3 pairwise product-flow training and reactant-only
candidate sampling. Each molecular reaction is one pseudo-basin with one known
event. Its metrics are single-event, no-relaxation geometric diagnostics, not
multi-event basin recall or saddle-validated KMC event recall.

## Local sources and exact membership

Source directory:
`/Users/wx/Desktop/yyxwjq/OAReactDiff/oa_reactdiff/data/transition1x`.
These three files are present; `test.pkl` is absent.

| File | Stored rows | `use_ind` rows | All single-fragment rows | `use_ind ∩ single_fragment` |
| --- | ---: | ---: | ---: | ---: |
| `train.pkl` | 10,073 | 9,000 | 7,516 | 6,733 |
| `train_addprop.pkl` | 10,073 | 9,000 | 7,516 | 6,733 |
| `valid_addprop.pkl` | 10,073 | 1,073 | 7,516 | 783 |

All `use_ind` lists contain unique, sorted, in-range indices. Train indices range
from 0 through 10,071; validation indices range from 7 through 10,072. These are
non-contiguous index selections, not prefix slices.

- `train.pkl.use_ind == train_addprop.pkl.use_ind`: 9,000 shared indices.
- Train/validation intersection: **0**; union: **10,073**, the entire row domain.
- Single-fragment train/validation intersection: **0**; union: **7,516**.
- Excluded multi-fragment records: 2,267 train and 290 validation, 2,557 total.
- Reactant `rxn` identifiers are unique across all 10,073 rows.

The full files contain both partition row arrays; reading every row from a file
named `train*.pkl` does not make every row a training example. Conversely, treating
the complement of the 9,000 training indices as a test set would misname the
provided validation partition.

## Hashes and ordered identity checks

SHA-256 of source file bytes:

| File | SHA-256 |
| --- | --- |
| `train.pkl` | `18e42152e83a9a2cec62be8d3515072f4a2a9f0ef5544ce2825a81178ebedd6f` |
| `train_addprop.pkl` | `ecc7d7f0eb6b0a98d7e7c6a004d67c45cb47bb32d392dfd08194ca750475b6c7` |
| `valid_addprop.pkl` | `85938e0dfa5a1b437bee30b6985a6269a1c43daf4ec0a8fcbe9df05e3e2f6915` |

SHA-256 of `numpy.asarray(use_ind, dtype='<i8').tobytes()`:

- Both train files: `a2cc08e774da2bc6813f68f9aa72c3c8f6bd7fc02867129321a4e114f0a8bc76`.
- Validation: `61950d5046022f269ced6593d193c4f269dfba0f7b90b874184ab3482d46e4e7`.

For every source row, in row-major then `reactant, product, transition_state`
order, the tuple `(str(rxn), str(formula), int(num_atoms), charges.tolist())` is
identical across all three files. SHA-256 of these records serialized with
`json.dumps(records, separators=(',', ':')).encode()` is:
`8ac72690b9008660268a7c264672c05e9e8db9dcdf0a102946d46d37179f608d`.

The preparation tool's required identity tuple omits optional `formula`; its
ordered identity hash is
`e668d1d62fb78f0237c8984765b2ceb840d30f19c0cb82279838f0a29ca70551`.
The identity hash deliberately excludes coordinates because the validation file
uses different coordinate frames. It verifies row identity/order, not geometric
equivalence. No OAReactDiff source code was copied.

## Coordinates are not interchangeable

`train.pkl` and `train_addprop.pkl` have exactly equal `num_atoms`, `charges`,
`positions`, `rxn`, and `formula` for every row in all three frame roles.
The added reactant properties include `smi`, `ediff`, and `uff_positions`.
The converter uses `positions`, not `uff_positions`.

In contrast, `valid_addprop.pkl` has different raw positions at **every row in
every frame role**, despite identical ordered reaction identities. Coordinate
arrays in both versions have `float64` dtype. Centering alone does not reconcile
them.

The following audit compares each same-index frame after independent centroid
subtraction and a proper Kabsch rotation, with atom identities fixed:

| Role | Maximum single-fragment Kabsch RMSD (Å) | Rows with changed pair distances above 1e-5 Å, all records |
| --- | ---: | ---: |
| Reactant | 3.054313e-7 | 13 |
| Product | 3.396373e-7 | 2,548 |
| Transition state | 3.143690e-7 | 0 |

All 7,516 single-fragment records preserve internal geometry to numerical
precision under separate frame rotations. Multi-fragment coordinates can also
change interfragment distances: maximum pair-distance differences are 6.393291 Å
for reactants and 14.810123 Å for products.

The rotations are **independent between R and P**. Fit the same-index reactant
rotation from training-file coordinates to validation-file coordinates, then
apply that rotation to the corresponding centered product. Every one of the
7,516 single-fragment rows has product RMSD above 1e-5 Å; median RMSD is
2.925719 Å and maximum is 5.703638 Å. Consequently, replacing validation
coordinates with same-index `train.pkl` coordinates changes the relative R/P
coordinate task. It is not an equivalent common rigid transformation.

OAReactDiff's object-aware setting accommodates separate object orientations.
A direct R-to-P displacement-flow comparison must explicitly state its molecular
alignment policy. Preserving OA source files reproduces source membership and
coordinates; it does not itself solve orientation ambiguity in a reactant-only
product-flow objective. Centroid removal is not Kabsch alignment. This audit
does not apply product/TS information to inference inputs.

### Optional product-to-reactant coordinate convention

Both conversion tools expose **opt-in** `--align-product-to-reactant`; the
default is false. This option is implemented for Stage 3 single-fragment,
nonperiodic molecules. It does not change source membership or atom mapping.
Selecting multi-fragment records with alignment enabled raises an error rather
than imposing a whole-complex alignment on disconnected fragments.

For row-vector coordinates, let `R_c = R - mean(R)` and `P_c = P - mean(P)`.
With `U, S, Vt = svd(P_c.T @ R_c)`, use
`Q = U @ diag(1, 1, det(U @ Vt)) @ Vt` and write
`P_aligned = P_c @ Q + mean(R)`. This is unweighted proper Kabsch alignment:
`det(Q) = +1`, no reflections or atom permutations, and all product internal
distances are preserved. When `--center` is also selected, as it always is in
the partition preparation tool, the reference centroid is zero. Without
centering, the product is translated into the unchanged reactant centroid.

This fixes a coordinate convention for the **supervised product target**. It
uses known R/P pairs during data preparation, which is part of pairwise training
target construction; the reactant is unchanged and no product-derived scalar,
vector, or initialization is introduced into `BasinDataset`. Active-displacement
labels and raw target-coordinate RMSD then refer to this aligned convention.
It is a different experimental target convention from the historical
centroid-only exports, so their RMSD numbers are not directly comparable.

Product forces, when available, are transformed with the same row-vector
rotation `F_aligned = F @ Q`; energies are unchanged. The product frame records
the applied rotation and `coordinate_convention=proper_kabsch_to_reactant`.
`source_partitions.json` records the alignment choice, preserved atom order,
proper-rotation constraint, and force transformation.

Only the product receives this Kabsch rotation. TS coordinates and forces remain
in their original frame, apart from the explicitly requested centroid removal
on coordinates. Aligned exports carry
`transition_state_usage=source_frame_not_an_aligned_path_guess`; their R/P/TS
triple must not be treated as a coherent path or a ready TS interpolation.
Aligning TS or preparing a physically meaningful path is outside this Stage 3
option. No physical relaxation or saddle validation is performed.

SVD handles planar inputs without division by small singular values. Fully
degenerate geometries can have non-unique optimal rotations; the tool does not
claim a unique orientation for those cases. Tests check nondegenerate independent
product rotations, planar finite behavior, internal-distance and chirality
preservation, force rotation, and unchanged target-free inference tensors after
modifying catalog product/TS targets. Implementing the option does not select it
for an experiment; an aligned benchmark export requires that coordinate
convention to be explicitly chosen.

The expanded converter/preparation suite passes all 16 tests in `ifdiff`,
including periodic-input rejection. This verification uses synthetic fixtures;
no real aligned benchmark dataset was exported as part of adding the option.

## What the OAReactDiff loader actually does

Verified code paths in the local reference repository:

- `oa_reactdiff/trainer/pl_trainer.py:160`: `setup('fit')` opens
  `train_addprop.pkl` and `valid_addprop.pkl`; the test branch requests the absent
  `test.pkl`.
- `oa_reactdiff/trainer/train_ts1x.py:78`: configuration sets `use_by_ind=True`,
  `single_frag_only=True`, `swapping_react_prod=True`, `reflection=False`, and
  `remove_h=False`.
- `oa_reactdiff/dataset/transition1x.py:21`: `ProcessedTS1x` defaults to
  `center=True`, intersects `use_ind` with `single_fragment == 1`, then optionally
  swaps R/P. Its set-intersection implementation need not preserve source order.

The unaugmented OA single-fragment counts are 6,733 train and 783 validation.
The displayed OA training configuration's R/P swapping doubles them to 13,466
and 1,566 dataset entries, respectively, because the same configuration is
passed to both loaders. BasinFlow preparation preserves source index order and
exports each forward reaction once; it does not silently add reverse reactions
or claim to reproduce OA's full training objective.

## Reproducible source-preserving export

From the BasinFlow repository, run with the `ifdiff` Python environment:

```bash
python tools/prepare_transition1x_split.py \
  /Users/wx/Desktop/yyxwjq/OAReactDiff/oa_reactdiff/data/transition1x/train_addprop.pkl \
  /Users/wx/Desktop/yyxwjq/OAReactDiff/oa_reactdiff/data/transition1x/valid_addprop.pkl \
  /Users/wx/Desktop/benchmark/0910/official/transition1x/data
```

The destination must be new. External benchmark writes need the environment's
filesystem authorization. Outputs are:

- `events/event_<source_index>.traj`: 7,516 three-frame R/P/TS trajectories;
  train rows use train source coordinates and validation rows use validation
  source coordinates, each frame centered at its own unweighted centroid.
- `events/basin_table.csv`: a combined catalog with collision-free source-index
  identities and explicit `source_partition` provenance.
- `split_manifest.json`: exact `use_ind ∩ single_fragment` membership, with
  **train=6,733, val=783, test=0**. `seed=0` is a schema placeholder; no random
  split is generated. Training and evaluation must load this manifest.
- `source_partitions.json`: source byte hashes, ordered identity hash, source
  row counts, selected index lists, coordinate convention, and absence of test.

Adding `--include-multifragment` selects 9,000 train and 1,073 validation records
using their respective source coordinates. This option changes the dataset
scope and must be recorded in reports. For a single-file conversion,
`tools/transition1x_to_events.py --single-fragment-only` explicitly intersects its
chosen `--selection` with the flags; the existing default remains unfiltered.

Exporting `train.pkl --selection all --center` and assigning the same train/val
index lists is a separate canonical-coordinate experiment. It must not be called
an exact OA-source-coordinate replication. Reusing an old 70/15/15 manifest
inside the 9,000 training indices is also a different benchmark. Evaluating the
provided held-out validation partition must retain the label **validation**;
an independent test score is unavailable from these local sources.

Initial source-partition verification on 2026-09-10: eight focused converter/partition tests passed in
`ifdiff`. A complete temporary export of the real source files produced exactly
7,516 trajectories and the 6,733/783/0 manifest; validation index 7's exported
R/P/TS coordinates equal the centered `valid_addprop.pkl` frames. Tests cover
selection intersection, source-index preservation, metadata length rejection,
partition overlap rejection, ordered identity mismatch rejection, and validation
coordinates coming from the actual validation source.

## Previous local benchmark results

These are historical artifacts under `/Users/wx/Desktop/benchmark`, not new
PaiNN results. Numbers below are copied from existing `sampling/summary.json`
files, rounded to six decimals. Different sample budgets and partitions prevent
treating these rows as a controlled architecture comparison.

| System / run suffix | Evaluation samples | Raw RMSD mean / median (Å) | EAM-relaxed RMSD mean / median (Å) |
| --- | ---: | --- | --- |
| Au / `semantic_flow_au_100trials` | 2 basins × 100 | 3.722565 / 5.120110 | 3.979123 / 5.459267 |
| Au / `semantic_flow_au_100trials_v2` | 2 basins × 100 | 3.716398 / 5.104789 | 3.981372 / 5.459228 |
| Pt / `semantic_flow_pt_100trials_v2` | 52 basins × 100 | 0.256300 / 0.244348 | 0.224952 / 0.214393 |
| Transition1x / `product_flow_use_ind_9000_100trials` | 100 pseudo-basins × 1 | 6.256426 / 2.366161 | Not run |
| Transition1x / `product_flow_use_ind_9000_40epochs_batch50_100trials` | 100 pseudo-basins × 1 | 5.070085 / 2.311308 | Not run |
| Transition1x / `product_flow_use_ind_9000_40epochs_batch500_100perbasin` | 1,350 pseudo-basins × 100 | 6.191493 / 3.428718 | Not run |

Locations follow `<system>/train/runs/<run>/sampling/summary.json`. The current
raw Au table has **24 events / 13 basins** at `au/events/basin_table.csv`; the Pt
table has **8,185 events / 350 basins** at `pt/pt-events/basin_table.csv`.
The existing v2 manifests use seed 42 with Au 9/2/2 train/val/test basins and Pt
245/53/52. All four historical Transition1x run manifests found use seed 42 with
6,300/1,350/1,350 pseudo-basins, resplit within `train.pkl.use_ind`. They do not
evaluate the provided 1,073-row validation selection.

Metric definitions from the existing runner code:

- RMSD is `sqrt(mean_i ||MIC(x_generated_i - x_reference_i)||²)` over movable
  atoms. Molecular data have no PBC, so this is direct coordinate RMSD; it is
  not an atom-permutation search or Kabsch-aligned RMSD.
- Au/Pt choose the closest known product within the source basin separately
  before and after EAM relaxation. Nearest-label assignment without a match
  threshold is not event recall.
- Au v2 reports zero relaxation failures and an RMSD improvement fraction of
  0.365; Pt v2 reports zero failures and an improvement fraction of 0.996154.
- Transition1x has one known reference per pseudo-basin and no relaxation or
  saddle validation. Mean active F1 is 0.017483 for the first 100-trial run and
  0.677988 for the 135,000-candidate batch-500 run.

The historical Transition1x batch-500 resolved configuration uses zero/Gaussian
initializers, Gaussian scale 0.05 Å, 40 epochs, hidden dimension 64, three layers,
cutoff 5 Å, batch size 500, and eight integration steps with graph updates every
two steps. These are pairwise training and single-event geometric sampling
diagnostics. No product-displacement oracle initializer is listed. A change to
source partitions or coordinate convention must be reported alongside any new
metrics; numerical differences cannot be attributed solely to the backbone.
