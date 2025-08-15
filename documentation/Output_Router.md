# Output Router over MLP Residuals

This repository now supports an optional *output router* that chooses among the MLP outputs of each transformer layer before applying the final layer normalization.

## What it does
- Each transformer block exposes its MLP output.
- When `--use_output_router` is enabled, a small linear router scores these per-layer MLP outputs.
- During training, the top `--output_router_top_k` candidates are mixed according to a softmax over their scores.
- During evaluation and sampling, the top `--output_router_eval_top_k` candidates are used (set to 1 for deterministic routing).
- The selected vector is added to the residual stream prior to the final layer normalization.

## How to use

### Enabling the router during training
```bash
python train.py \
  --use_output_router \
  --output_router_top_k 2 \
  --output_router_eval_top_k 1 \
  --max_sample_tokens 128 \
  --colorize_output --colorize_mode router
```

### Visualizing router choices
The sampling utility (`sample.py` or the automatic sampling in `train.py`) now exposes a `router` colorization mode. When used with `--colorize_output`, tokens are tinted according to which layer's MLP output was selected (assuming `output_router_eval_top_k=1`). Lower layer indices appear red, while higher layers appear green.

```bash
python sample.py --out_dir out --colorize_output --colorize_mode router \
  --use_output_router --output_router_eval_top_k 1
```

This allows quick inspection of which layer contributes most to the final residual at each generated token.
