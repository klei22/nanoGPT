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

- 36 focused Python tests pass, including geometry, gradient flow, tying, shared final normalization, causality, legacy paths, package export and exact checkpoint continuation.
- The normal train.py argument parser accepts the new optional settings.
- CPU smoke run covers three heads, eight numeric slots, independently phased streams and matched-circle table initialization.
- Chromium/Playwright checks real WebGL rendering, iteration and metric alignment, continuous point controls, summed inputs, run switching and a 390px layout. Browser checks use a local copy of the exact pinned Three.js npm release because esm.sh is unreachable in the execution environment.
- Every run includes iteration 0 plus 2,000 post-update geometry/loss frames. Generated trajectories and checkpoints are distributed separately from the source PR.

See [DUAL_STREAM.md](DUAL_STREAM.md) for reproduction and integration.
