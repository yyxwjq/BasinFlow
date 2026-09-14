# PaiNN numerical stability audit

The diagnostic state and traces are also preserved under
`/Users/wx/Desktop/benchmark/0910/official/pt/debug/` so the replay evidence does
not depend on temporary-file retention. Use that directory in place of
`/tmp/painn_pt_debug` in the commands below when reproducing later.

## Scope and reproducible evidence

This audit investigates pairwise velocity-only flow training on the full Pt
event catalog. It does not measure relaxed product recall or saddle-validated
event recall. Initializers use reactant-only Gaussian noise; oracle product
initialization is prohibited. The training target remains the MIC-unwrapped
product minus the sampled source geometry.

The original run used `configs/0910/official_pt.ini`: 128 features, four dual
PaiNN layers, 32 Bessel radial functions, polynomial envelope, 4.5 Å cutoff,
Adam learning rate 0.001, batch size four, Gaussian standard deviation 0.05 Å,
gradient clipping at 10, CPU with four threads, and training seed 42. It loaded
the existing manifest at
`/Users/wx/Desktop/benchmark/pt/train/runs/semantic_flow_pt_100trials_v2/split_manifest.json`.
No split was regenerated. The catalog is
`/Users/wx/Desktop/benchmark/pt/pt-events`, with PBC overridden to `[true, true, false]`.

The catalog convenience trainer constructs `EventFlowDataset` with its default
dataset seed **42**. This is separate from epoch numbering, which starts at one.
The initializers also use random seed 42. The diagnostic reproducer preserves
these settings and the shuffled DataLoader RNG sequence.

The original run recorded loss spikes at optimizer steps 152 (1,713,502.375)
and 158 (22,615.787), followed by a nonfinite loss before step 194. A separately
instrumented reproduction matched the first losses to approximately 1e-8,
then diverged numerically and failed its gradient-norm check at step 143.
This is a reproduction of the instability under the same protocol, not a
bitwise reproduction of the original step-194 state. The diagnostic makes no
optimizer update after the failed finite-gradient check.

The reproducer is `tools/diagnose_painn_stability.py`. It records per-event maxima
after every message and update block, and saves model weights, optimizer state,
the actual PyG batch, RNG state, geometry statistics, and targets before an
update whenever the loss exceeds 100 or becomes nonfinite. The diagnostic
artifacts for this investigation are local to `/tmp/painn_pt_debug`; they are
not committed fixtures or benchmark outputs.

## Failure mechanism

The decisive graph in the step-143 batch is `event_3490`. Its input geometry and
targets are finite, and its neighbor count is ordinary for this Pt system:

| Quantity | Observed value |
| --- | ---: |
| Atoms / movable atoms | 343 / 175 |
| Flow time | 0.858489 |
| Minimum reactant edge | 2.484639 Å |
| Minimum current edge | 1.872054 Å |
| Maximum neighbor degree / mean degree | 19 / 16.093294 |
| Maximum target displacement norm | 5.266872 Å |
| Maximum absolute target-velocity component | 5.315324 Å |
| Maximum absolute Gaussian-noise component | 0.161125 Å |

There is no near-zero-distance collision, excessive neighbor count, or
nonfinite target in this failing example. Its long but finite displacement
initializes a vector state whose multiplicative scalar/vector updates amplify
through the unnormalized network:

| Stage | Maximum absolute scalar | Maximum absolute vector component |
| --- | ---: | ---: |
| Message 0 | 1.084897 | 4.419783 |
| Update 0 | 146.639587 | 17.079933 |
| Update 1 | 243.205994 | 44.406284 |
| Update 2 | 1,506.568115 | 280.669006 |
| Message 3 | 1,486.946533 | 7,074.430176 |
| Update 3 | 268,646,688 | 435,326.593750 |

The final maximum velocity component reaches approximately 1.248e12, producing
batch loss 1.5185e21. The first strong amplification occurs in the update
block's unnormalized scalar gate times the vector dot product. Subsequent
messages and updates compound this growth.

Replaying the saved state in float64 produces the same loss scale, so the root
cause is not float32 rounding alone. All individual float32 gradient entries
remain finite in this particular saved batch, but their L2 norm is about
1.003e24; the float32 sum of squared gradients overflows. The trainer correctly
rejects that update. Merely computing clipping norms in higher precision would
leave the enormous forward amplification unresolved.

Before the implementation change, adding scalar LayerNorm only before the
message source MLP reduced this saved batch's loss to approximately 3.426e11
and maximum velocity to 2.682e7. This isolated intervention was insufficient.

## Correction and saved-batch verification

The new `stability_mode="scaled"` uses parameter-free scalar normalization,
dimension-aware vector and dot-product scaling, and scalar residual scaling.
The historical architecture remains available as `stability_mode="none"` for
loading and diagnosing earlier checkpoints. The model factory treats older
checkpoint metadata without a stability mode as historical behavior.

An explicit replay of the same saved model weights and identical input batch
under the two modes gives:

| Mode | Velocity loss | Maximum absolute velocity | Gradient L2 norm, evaluated in float64 |
| --- | ---: | ---: | ---: |
| `none` | 1.518503e21 | 1.247870e12 | 1.003084e24 |
| `scaled` | 0.009389319 | 2.666128 | 0.160863 |

The scaled mode also passes the ordinary float32 finite-gradient and clipping
check. This is an isolated numerical regression check with shared weights;
it is not evidence of trained proposal quality. Fresh training and complete
evaluation are required before claiming benchmark improvement.

## Commands

Reproduce the historical unnormalized training path, writing locally:

```bash
/Users/wx/miniconda3/envs/ifdiff/bin/python tools/diagnose_painn_stability.py \
  --config configs/0910/official_pt.ini \
  --stability-mode none --dataset-seed 42 \
  --output-dir /tmp/painn_pt_debug --max-steps 200
```

Replay the captured regression case without reloading the source catalog:

```bash
/Users/wx/miniconda3/envs/ifdiff/bin/python tools/diagnose_painn_stability.py \
  --replay /tmp/painn_pt_debug/pre_step_143.pt --stability-mode none
/Users/wx/miniconda3/envs/ifdiff/bin/python tools/diagnose_painn_stability.py \
  --replay /tmp/painn_pt_debug/pre_step_143.pt --stability-mode scaled
```

The exact failing step may vary with floating-point execution. The tool saves
each encountered high-loss batch under its actual step number. Local `/tmp`
files are temporary; retaining or publishing a trained binary fixture requires
an explicit dataset/artifact decision.

## Independent geometry checks

Before the production failure was observed, all 16 PaiNN and tensor-neighbor
tests passed. Independent randomized comparisons of 30 molecular, skew-cell,
and partial-PBC graphs against ASE also passed. Finite forward/backward checks
on initially random two/four-layer models using ordinary periodic FCC Au
geometry passed. These checks establish geometry correctness and ordinary
initial numerical behavior; they were insufficient to expose trained-state
amplification on long-displacement Pt events. A dense-intermediate-geometry
regression and saved-batch replay are therefore necessary additional evidence.
