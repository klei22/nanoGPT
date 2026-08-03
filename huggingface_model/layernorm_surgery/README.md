# LayerNorm channel explorer and inference sweep

This directory connects two related experiments: inspecting normalization gains and measuring the accuracy/cost effect of pruning them. It supports Hugging Face causal language models whose LayerNorm or RMSNorm modules expose a one-dimensional `weight`.

## Interactive explorer

Install `torch transformers fastapi uvicorn`, then run from the repository root:

```bash
python -m huggingface_model.layernorm_surgery.webapp
```

The browser at `http://127.0.0.1:8000` loads a model only after **Load** is clicked. Choose any discovered norm, input/output embedding row, and token ID. The gain plot is sorted high-to-low; the embedding before and after the selected norm uses that exact channel permutation, making channel-wise comparisons aligned. “After” means applying that norm directly to the selected embedding vector; for an internal norm this is a probe, not the activation encountered at that model depth.

## Threshold, fold, quantize, benchmark

Install `lm-eval`, then run, for example:

```bash
python -m huggingface_model.layernorm_surgery.sweep \
  --model google/gemma-3-270m --targets final \
  --threshold-start 0 --threshold-stop 0.20 --threshold-step 0.02 \
  --bits 2,3,4,5,6,7,8 --tasks hellaswag,arc_easy,piqa \
  --output-dir runs/gemma-ln-sweep
```

`--targets final` (the default) selects the last discovered normalization. Exact comma-separated module names and `all` are also supported. Each threshold zeros effective gains satisfying `abs(gain) < threshold`. Gemma RMSNorm stores the multiplier minus one, so these tools translate its parameter to/from the effective gain rather than pruning the raw stored value. For the final norm, its gain is folded into LM-head columns, non-zero channel indices are placed first by a recorded permutation, zero columns are omitted from the reduced representation, and only remaining values are symmetrically fake-quantized. The ordinary evaluator receives a dense equivalent so no custom CUDA kernel is required.

For a vocabulary of `V` and `Z` zeroed final channels, the reduced final dot product shaves **`V × Z` multiplications and stored scalar parameters**, a fraction `Z / hidden_size` of the head. The sweep reports these estimates as `shaved_parameters` and `shaved_fraction` for every configuration. Internal norms are pruned but do not themselves reduce the final LM-head dot product; only the final norm is folded into that head.

Outputs are `settings.csv`, tidy `results.csv`, and an interactive Plotly `results.html`. The HTML uses the Plotly CDN, so opening it offline requires replacing the script URL with a local Plotly bundle.

> This is inference-time structured channel pruning/fake quantization. The zero columns are mathematically skippable only with a reduced/grouped dot-product kernel; the dense benchmark path estimates, rather than realizes, its latency and memory savings.
