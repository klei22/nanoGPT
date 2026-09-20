# Two streams, one width-3 transformer

This optional addition to the trajectory experiment at
[`6cb931d`](https://github.com/ReaLLMASIC/ReaLLM-Forge/commit/6cb931dcccada904e523a15a18898e609bde1fb0)
uses this repository's actual `GPT` and existing `multicontext` path. Existing
single-stream training, viewing, schedules and defaults continue to work.

## Quick start

From the repository root, in your existing PyTorch environment:

```bash
python -m pip install torch numpy rich pytest
TASK_MODE=dual_stream MAX_ITERS=2000 SEEDS="0 1 2" \
  bash demos/digits_3d_trajectory_demo.sh
python -m http.server 8000
```

Open **http://localhost:8000/report/threejs/digits-3d/dual-stream.html**.
The existing `index.html` also links to this page. Three.js is pinned to 0.160.0
on esm.sh; the browser needs network access to that CDN. No browser-side training
or additional web server dependencies are required.

The default saves iteration 0 and **every optimizer update**, runs four variants,
and stores resumable checkpoints under `out/dual_stream_clock`. Existing results
are protected: add `--resume` to continue, or choose new output and checkpoint
directories for a different experiment. `--overwrite` explicitly replaces runs
with the same variant and seed.

```bash
# Extend all three seeds, preserving optimizer state and data RNG:
MAX_ITERS=10000 SEEDS="0 1 2" bash demos/dual_stream_clock_demo.sh --resume

# Add six runs with BOTH clocks starting at 10% or 5% of the sphere radius:
EMBEDDING_MODES="small_circle_r10 small_circle_r05" MAX_ITERS=2000 SEEDS="0 1 2" \
  bash demos/dual_stream_clock_demo.sh

# Exactly an eight-position numeric clock, 0–7:
NUM_DIGITS=8 DIGIT_SLOTS=8 DUAL_STREAM_DIR=report/threejs/digits-3d/dual-stream-8 \
  OUT_DIR=out/dual_stream_8 bash demos/dual_stream_clock_demo.sh

# Train all positions, 0–9; independent starts cover all 50 input pairs:
NUM_DIGITS=10 DIGIT_SLOTS=10 PAIRING=independent \
  DUAL_STREAM_DIR=report/threejs/digits-3d/dual-stream-10 OUT_DIR=out/dual_stream_10 \
  bash demos/dual_stream_clock_demo.sh

# Give the table controls exactly the same initial vectors as small_circle:
EMBEDDING_INIT=matched_circle EMBEDDING_MODES="table_sphere small_circle" \
  DUAL_STREAM_DIR=report/threejs/digits-3d/dual-stream-matched OUT_DIR=out/dual_stream_matched \
  bash demos/dual_stream_clock_demo.sh
```

For custom output, use, for example,
`dual-stream.html?manifest=dual-stream-8/manifest.json`. Run selections are saved
in the URL. Select two runs, scrub the synchronized iteration slider, choose
mean/digit/letter loss or joint accuracy, orbit both scenes, inspect trails and
summed inputs, and move virtual points continuously around each learned clock.
The virtual point controls visualize the mapping, not network predictions.

Use the wheel or hold the **middle mouse button and drag vertically** to zoom.
Zoom speed is reduced and camera distance is bounded. Each panel also has **+ / −**
buttons and **Reset view**; press **R** outside an input/select to reset both.
Reset restores the position, orbit target and zoom, including after panning.
**Focus digits** and **Focus letters** fit that family to the view and follow its
center as you scrub training. This makes very small starting circles inspectable;
Reset view returns to the whole sphere. Left-drag orbits, right-drag pans, and
touch supports pinch zoom. The percentage readout is relative to the fitted view.

## Precisely what is trained

Default streams are `01234567` and `abcde`, both advancing once per timestep:
`(0,a), (1,b), ..., (7,c), (0,d), ...`. Their joint period is 40, so every active
digit/letter combination appears. `NUM_DIGITS=8` sets the active numeric targets;
`DIGIT_SLOTS=10` sets output vocabulary and circle spacing. Thus 8 and 9 are
untargeted **competing output classes in every variant**, drawn faintly in the
viewer. They are not removed from softmax.

If the periods share a divisor, aligned streams cover only a subset of pairs
(10 and 5 cover 10 of 50). `PAIRING=independent` samples the starting phases
independently and evaluates every start pair.

```text
input = digit_embedding[d_t] + letter_embedding[l_t]
hidden = one existing GPT decoder block(input)
z = one shared final RMSNorm(hidden)
digit_logits = z @ digit_embedding.weight.T
letter_logits = z @ letter_embedding.weight.T
loss = (cross_entropy(digit_logits, next_digit)
        + cross_entropy(letter_logits, next_letter)) / 2
```

Width is 3, one attention head by default (`NUM_HEADS=3` also works), head
dimension is `3 / NUM_HEADS`, GELU MLP width is 12, and projections are bias-free.
The block uses the repo's pre-RMSNorm attention/MLP implementation, causal softmax
attention and ordinary residual addition. There are no position embeddings,
dropout, head-specific output norms, or added output projections. Only individual
token vectors are sphere-constrained; their input sum generally is **not**.

Each variant receives bit-identical backbone initialization and the same sampled
batches for a given seed. Free and spherical lookup tables also start with
identical vectors of norm R. Normal tables use Gaussian directions; clocks have
structured initialization, so initialization and parameter counts are reported
differences. Use `matched_circle` to match table vectors to the small-circle
variant as an extra control. The great-circle control has a different offset.

Defaults: AdamW, learning rate 0.003, weight decay 0.01, gradient clipping 1,
batch size 16, sequence length 16, radius sqrt(3), CPU float32, one CPU thread.
2,000 updates process 512,000 paired timesteps (1,024,000 channel tokens). Each
snapshot evaluates all joint cycle starts at sequence length 16. These repeated
cycle losses are **not a test of held-out generalization**. Minibatch loss is
exported separately as `pre_update_loss`; plotted metrics always match the
displayed post-update weights.

## Variants and the continuous equation

| Setting | Parameters per token family | Constraint |
| --- | --- | --- |
| `table_free` | Independent learned rows | Free after spherical initialization |
| `table_sphere` | Independent learned rows | Reproject rows after every update |
| `great_circle` | Learned orthonormal frame | Center offset fixed at zero |
| `small_circle` | Learned orthonormal frame and offset | Every virtual token has norm R |
| `small_circle_r10` | Same as `small_circle` | Both clocks initially have circle radius 0.10R |
| `small_circle_r05` | Same as `small_circle` | Both clocks initially have circle radius 0.05R |

The two tiny-start presets are opt-in; the default demo still runs the original
four variants. They initialize the center offset at `a = sqrt(1 - (r/R)^2)`:
about 0.9949874 for 10% and 0.9987492 for 5%. These are **initial** sizes, not
fixed circle radii. Digit and letter frames and offsets remain independently
learned, and every token stays on the sphere of radius R. The default
`small_circle` starts with offset 0.5, giving circle radius about 0.866R.
The two presets override `--circle-offset`; use `small_circle --circle-offset A`
for a custom initial offset. Exports record the effective initialization in
`circle_initialization` and display both measured radii at each iteration.

Each stream independently uses:

`x(t) = R [a*n + sqrt(1-a²) (u*cos(2πt) + v*sin(2πt))]`.

The frame `(n,u,v)` comes from sign-corrected reduced QR of a learned
dimension-by-3 matrix. It starts orthonormal; exactly rank-deficient frames are
singularities of QR, so training aborts on nonfinite gradients. The learned
offset is `a = 0.9999 * sigmoid(offset_logit)` to prevent exact loop collapse in
floating point. Sphere radius is fixed. Each stream has its own parameters.
Integer token k uses `t=k/vocab_size`. Centers, radii, bases and offsets are
exported at every frame, so the frontend computes exact virtual points rather
than connecting discrete tokens by straight chords. This is a small-circle
parameterization, not ordinary SLERP or line-segment interpolation.

Circle variants have no separate learned rows: input embedding and output logits
derive from the same parameters. The center contributes the same `z·center` to
every logit within a head, so it cancels in that softmax. It still affects the
summed input; offset also controls the radius of the discriminating part of the
vectors. A learned center is not an independent per-class logit bias.

The clocks are periodic: zero and a full turn coincide. The default 0–7 training
cycle on a ten-slot clock jumps from 7 back to 0, skipping 8 and 9. Choose the
eight-slot or full ten-digit preset if that discontinuity is unwanted.

## Ordinary optional model settings

`train.py` retains its categorical multicontext data flow and averaged loss.
These new model arguments select circles:

```bash
--training_mode multicontext --multicontext_datasets <digits_dataset> <letters_dataset> \
--multicontext_embedding_variant small_circle \
--circle_offset_init 0.5 --circle_learn_offset \
--wte_weight_tying --wte_fixed_norm_value 1.7320508075688772 \
--n_embd 3 --n_layer 1 --n_head 1 --n_kv_group 1 \
--norm_variant_output rmsnorm --no-use_abs_pos_embeddings
```

Ordinary `train.py` still needs its existing prepared datasets. The specialized
demo generates paired sequences directly and needs no `.bin` files. This first
integration rejects regression mode, untied heads, factored/quantized WTE,
imported WTE, and separate LM-head normalization with `small_circle`. Default
`table` behavior and old checkpoint schemas remain unchanged. Small circles
enforce their radius analytically, regardless of the table reprojection switch.

Floating input tensors contain continuous phases; integer tensors contain token
indices:

```python
# model is a circle multicontext GPT; shapes are [batch, time].
logits, _ = model(None, token_dict={
    'digits': torch.tensor([[0.125, 0.225, 0.325]]),
    'letters': torch.tensor([[0.10, 0.30, 0.50]]),
})
point = model.transformer.wte_0.embed_phase(torch.tensor(0.275))
```

Training targets remain discrete next-token cross-entropies for a controlled
comparison. The final `continuous_probe` feeds unseen quarter, half and
three-quarter fractional starts. It decodes the phase maximizing each continuous
dot-product head as `atan2(z·v,z·u)/(2π)` and reports circular phase MAE in turns.
It **does not** train a continuous target loss or establish interpolation
accuracy merely from a smooth embedding curve.

## Exports and verification

Run JSON contains all frames, both losses/accuracies, joint accuracy, norms,
circle parameters, configuration, initial backbone hash, token counts and the
fractional-input probe. `summary.csv` has one row per seed; `summary.json` reports
means and sample standard deviations within identical configurations. The
manifest updates atomically after each completed run. Checkpoints contain
model/config, optimizer, data RNG and trajectory history.

```bash
python -m pytest -q tests/test_small_circle_embeddings.py tests/test_dual_stream_clock.py \
  tests/test_wte_fixed_norm.py tests/test_numerical_multicontext_fp16.py \
  tests/test_export_3d_token_trajectories.py tests/test_package_3d_trajectory_site.py \
  tests/test_update_3d_sweep_manifest.py tests/test_digits_3d_dataset.py \
  tests/test_digits_3d_trajectory_demo.py
python analysis/package_3d_trajectory_site.py --output-dir dist/digits-3d-site
```

The packager includes existing runs plus the optional dual-stream viewer/results.
Serve or publish its output with the repo's existing workflow. Generated runs
and checkpoints remain ignored by git; source changes can be reviewed without
committing tens of megabytes of trajectories.
