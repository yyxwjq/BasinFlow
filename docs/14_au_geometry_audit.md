# Au geometry audit — 2026-09-10

The Au zero-motion error near **3.45 Å is a real displacement of the one movable atom**, not evidence of global rotation, translation, or periodic wrapping. It should not be compared directly with Pt's reported movable-atom RMSD without reporting the number of movable atoms and displacement distribution.

## Inputs and contracts

- Audited all 24 `event_*.extxyz` files referenced by `/Users/wx/Desktop/benchmark/au/events/basin_table.csv`, covering 13 reactant basins. No input data were changed.
- Every event contains three frames, interpreted as reactant, product, and saddle by `src/basinflow/data/raw_events.py`; `tools/eon_to_events.py` writes that role order.
- Every frame contains 101 Au atoms. Exactly atom **47, zero-based**, is movable; the remaining 100 atoms are fixed. The source `move_mask` survives loading.
- All frames have the same diagonal cell, `(54.864514, 20.353349, 20.353349) Å`, and **PBC = `(False, False, False)`**. The current `pbc_override = auto` configuration therefore evaluates Au as nonperiodic. A nonzero cell does not make it periodic.
- Within each table-assigned basin, every reactant's coordinates agree exactly with that basin's catalog reactant: maximum absolute coordinate difference is `0.0 Å`.
- Reactant/product atom-symbol sequences agree in all events. Because every symbol is Au, species equality alone cannot establish atom identities. However, all 100 fixed atom coordinates are exactly unchanged at their original indices in every R/P pair. No permutation search was performed and no atom mapping was altered.

## Displacement and alignment results

RMSD is `sqrt(mean_i ||P_i - R_i||²)` on the stated atom set, with the original atom indices. MIC equals the raw displacement under the recorded nonperiodic flags. Proper Kabsch fits use centered coordinates, SVD, and a determinant correction enforcing a proper rotation, without reflections.

| Quantity across 24 events | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Raw / MIC movable-atom RMSD, Å | 2.636592 | 3.451077 | 3.490541 |
| Raw full-101-atom RMSD, Å | 0.262351 | 0.343395 | 0.347322 |
| Full-atom Kabsch fit, full-atom RMSD, Å | 0.261049 | 0.341691 | 0.345598 |
| Full-atom Kabsch fit, movable-atom RMSD, Å | 2.610488 | 3.416907 | 3.455981 |
| Fixed-100-atom Kabsch fit, movable-atom RMSD, Å | 2.636592 | 3.451077 | 3.490541 |
| Full-atom Kabsch rotation angle, degrees | 0.000000 | 0.000002 | 0.000016 |
| R/P centroid difference, Å | 0.026105 | 0.034169 | 0.034560 |
| Largest fixed-atom displacement, Å | 0.000000 | 0.000000 | 0.000000 |

Full-atom RMSD dilutes the one moving atom by `sqrt(101) ≈ 10.05`. Its small value does not mean that the product is a small displacement from the reactant on the active coordinate. The centroid shift is precisely the moving atom's displacement divided by 101, rather than a global translation of the structure. Removing that shift artificially moves the fixed scaffold.

Kabsch fitting **only the single movable atom** produces zero residual by translation for every event. Its covariance has rank zero and its rotation is undetermined, so this number is not a meaningful event reconstruction metric. Aligning on the fixed scaffold leaves the movable-atom RMSD unchanged. Full-structure rotation correction is negligible.

Across all `24 × 101 = 2424` atom displacements, quantiles at 0%, 25%, 50%, 75%, 95%, and 99% are all exactly zero; the 100% quantile is `3.490541 Å`. The movable-only distribution is the first table row. Every event moves atom 47 predominantly along the x-axis: absolute transverse components remain below `0.000444 Å`.

| Events | Basin IDs | Movable displacement magnitudes, Å |
|---|---|---|
| 0–1 | 0 | 3.450979, 3.451171 |
| 2–3 | 1 | 3.451083, 3.451116 |
| 4–5 | 2 | 3.450979, 3.451163 |
| 6–7 | 3 | 3.451076, 3.451118 |
| 8–9 | 4 | 3.451078, 3.450182 |
| 10–11 | 5 | 3.451083, 3.451103 |
| 12–13 | 6 | 3.451056, 3.450175 |
| 14–15 | 7 | 3.450113, 3.490541 |
| 16–17 | 8 | 3.450144, 3.490516 |
| 18–19 | 9 | 3.490474, 2.636601 |
| 20–21 | 10 | 3.490492, 2.636608 |
| 22 | 11 | 2.636608 |
| 23 | 12 | 2.636592 |

## Periodic-image and ordering checks

The largest absolute fractional R/P displacement component is `0.063621`, far below one half of a cell. As an audit-only counterfactual, enabling all three periodic flags changes any R/P displacement component by at most `4.44e-16 Å`. Therefore cell-image wrapping does not explain the approximately 3.45 Å distance. This counterfactual is not a recommendation to change PBC: doing so would also change neighbor construction.

The movable atom's nearest fixed-atom distance ranges from `2.618517` to `2.988738 Å` in reactants and from `2.618517` to `2.988739 Å` in products. The measured coordinates describe jumps within a fixed environment rather than a rigidly drifting structure. These geometry checks do not establish barrier heights, physical connectivity, or saddle validity.

## Held-out zero-motion baseline

The saved split at `/Users/wx/Desktop/benchmark/0910/smoke/au/split_manifest.json` uses test basins `1` and `5`.

| Test basin | Known products | R-to-product movable RMSDs, Å | Nearest-product RMSD, Å |
|---|---|---|---:|
| 1 | `event_2`, `event_3` | 3.451082820, 3.451115850 | 3.451082820 |
| 5 | `event_10`, `event_11` | 3.451082740, 3.451103244 | 3.451082740 |

The macro zero-motion nearest-product RMSD is **3.451082780 Å**. Its independent geometric event recall is zero at each of 0.1, 0.2, and 0.5 Å for all four known test events. Each held-out basin has products on opposing x directions; a prediction remaining near the reactant will miss both.

The two-update smoke model has a macro nearest-product RMSD of approximately `3.4140 Å`, while the paired Gaussian-only initialization is approximately `3.4150 Å`. The small difference supports only that this smoke run remains close to its initializer. The initialization scale is `0.05 Å` per movable Cartesian component, while these test products require approximately `3.45 Å` of x displacement. This scale contrast does not make a trained flow incapable of transporting the atom, but the smoke run is not evidence of learned event coverage.

## Reporting consequences

Keep the existing movable-atom, index-preserving evaluation and report its physical scale explicitly. Do not substitute full-atom or movable-only aligned RMSD to make the Au numbers smaller. Report Au as a nonperiodic constrained single-movable-atom dataset under its current files and configuration. Its 24 records do not demonstrate general multi-atom event generation. Compare learned proposals to the paired zero-motion and Gaussian-only baselines and report event-wise geometric recall independently of nearest-product distance.

Measurements are recorded for this session in `/tmp/au_geometry_audit_0910.json`; the read-only reproduction script is `/tmp/audit_au_geometry_0910.py`. The source files and metric formulas above remain the basis for reproducing the audit after temporary files are removed.
