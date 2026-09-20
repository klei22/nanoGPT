# Initial two-stream clock results

Measured on CPU with PyTorch 2.8.0, float32; seeds 0, 1 and 2; 2,000 updates each.

Default configuration: digits 0–7, ten numeric output slots (8 and 9 untargeted), letters a–e; aligned pairs; width 3, one layer, one head, GELU MLP width 12; batch 16, context 16; AdamW lr 0.003, weight decay 0.01. Each run consumes 512,000 paired timesteps. The common backbone and per-seed batch schedule are matched; table versus circle initialization and parameter counts differ.

Every value below uses the final weights and every joint-cycle start. Standard deviations are sample standard deviations over three seeds, not uncertainty estimates over independent natural-language datasets.

| Variant | Parameters | Mean CE ± SD (nats) | Digit accuracy | Letter accuracy | Joint accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free table | 162 | 0.7722 ± 0.3571 | 83.54% | 42.45% | 37.19% |
| Spherical table | 162 | 0.9285 ± 0.1480 | 52.45% | 68.18% | 29.69% |
| Great circle | 135 | 0.9900 ± 0.0526 | 52.60% | 71.41% | 47.29% |
| Small circle | 137 | 1.0002 ± 0.1027 | 47.97% | 78.39% | 46.15% |

The learned small circle is not the lowest-loss configuration in this initial run. It has higher mean joint accuracy than the two table controls here, but approximately the same mean joint accuracy as the great-circle control. The streams can trade off against each other; average loss alone does not show which task a run favors. No model solves both streams perfectly at this budget.

These are toy repeated-cycle measurements, not held-out generalization, statistical proof of an architectural advantage, or a converged optimization comparison. The free table can change logit scale by changing row norms; both constrained variants cannot do so that way (the shared final RMSNorm gain remains learnable).

## Fractional-input probes

No fractional targets were trained. At the final checkpoint, quarter/half/three-quarter offsets are fed through the same backbone. The continuous head chooses the phase maximizing the dot product. Error is shortest circular distance in turns; one turn equals ten numeric slots or five letter slots.

| Circle variant | Numeric circular phase MAE | Letter circular phase MAE |
| --- | ---: | ---: |
| Great circle | 0.1125 | 0.1058 |
| Small circle | 0.1158 | 0.0871 |

A smooth virtual-token map does not imply accurate continuous prediction. The current probe errors leave room for improvement. The default numeric cycle skips slots 8 and 9 at wraparound; the eight-slot/full-ten-digit presets test that choice separately.

## Verification

- 38 focused Python tests pass, including geometry, gradient flow, tying, shared final normalization, causality, legacy paths, package export, exact checkpoint continuation and both tiny-circle initializations.
- The normal train.py argument parser accepts the new optional settings.
- CPU smoke run covers three heads, eight numeric slots, independently phased streams and matched-circle table initialization.
- Chromium/Playwright checks real WebGL rendering, iteration and metric alignment, continuous point controls, summed inputs, run switching and a 390px layout. Browser checks use a local copy of the exact pinned Three.js npm release because esm.sh is unreachable in the execution environment.
- Every run includes iteration 0 plus 2,000 post-update geometry/loss frames. Generated trajectories and checkpoints are distributed separately from the source PR.

## Tiny starting circles

Six additional runs use the same configuration, seeds, common backbone and batch
schedule. Both token families start with circle radius 10% or 5% of the sphere
radius. Their frames and offsets are independent learned parameters. These
presets are optional and do not change the default four-variant sweep.

| Initial circle radius, both families | Parameters | Mean CE ± SD (nats) | Digit accuracy | Letter accuracy | Joint accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10% of R (`small_circle_r10`) | 137 | 0.9517 ± 0.0885 | 54.43% | 73.54% | 30.78% |
| 5% of R (`small_circle_r05`) | 137 | 1.1324 ± 0.3557 | 38.85% | 73.23% | 28.80% |

Measured final circle sizes, as percentages of the fixed sphere radius:

| Initial size | Seed | Final digit radius | Final letter radius |
| --- | ---: | ---: | ---: |
| 10% | 0 | 24.48% | 68.53% |
| 10% | 1 | 88.14% | 2.97% |
| 10% | 2 | 31.24% | 67.17% |
| 5% | 0 | 55.08% | 1.73% |
| 5% | 1 | 19.13% | 71.40% |
| 5% | 2 | 17.74% | 62.12% |

The families do not necessarily expand together. In two seeds the letter clock
shrinks while the numeric clock expands. The 5% seed-0 loss curve also develops
large fluctuations late in training. Small initialization alone does not ensure
better optimization or balanced accuracy. Across every snapshot in these six
runs, token norms stay within about 9e-7 of sqrt(3), while circle radii change.

Fractional-input circular phase MAE (digits / letters) is 0.0966 / 0.1216 turns
for the 10% preset and 0.1241 / 0.1307 for the 5% preset. The same probe limitations
described above apply.

The viewer now includes gentle wheel and middle-button-drag zoom, bounded camera
distance, +/− buttons, Reset view and the R shortcut. Focus digits/letters fits a
tiny clock and follows its center through training. Real Chromium checks cover
both zoom directions and limits, exact reset after panning, keyboard reset while
a button is focused, circle and table focus, run changes and a 390px layout.
The downloadable viewer contains all 18 runs and corresponding checkpoints.

See [DUAL_STREAM.md](DUAL_STREAM.md) for reproduction and integration.
