# NanoGPT Android Benchmark

End-to-end pipeline: nanoGPT → ExecuTorch `.pte` → Android app measuring
TTFT, decode latency, memory, and energy.

---

## Quick start

### 1. Export the model

```bash
cd nanoGPT/exutorch

# Install ExecuTorch (if not already set up via setup_exutorch.sh)
pip install executorch==0.4.0 torch torchvision transformers

# FP32 + INT8 with XNNPack delegation
python export_nanogpt_android.py \
  --model_type gpt2 \
  --out_dir android/app/src/main/assets/

# Also export KV-cache models (prefill + decode)
python export_nanogpt_android.py --kvcache \
  --out_dir android/app/src/main/assets/
```

This creates (in `android/app/src/main/assets/`):
| File | Size | Notes |
|------|------|-------|
| `nanogpt_fp32.pte` | ~500 MB | FP32 weights, XNNPack ARM NEON kernels |
| `nanogpt_int8.pte` | ~130 MB | INT8 weight-only, XNNPack INT8 GEMM |
| `nanogpt_kvcache_prefill.pte` | ~500 MB | KV-cache prefill step |
| `nanogpt_kvcache_decode.pte`  | ~500 MB | KV-cache single-token decode |

### 2. Download tokenizer assets

```bash
cd android/app/src/main/assets/

# GPT-2 vocabulary
curl -fLO https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
mv vocab.json encoder.json

# BPE merge rules
curl -fLO https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt
mv merges.txt vocab.bpe
```

### 3. Build and install

Open the `android/` folder in **Android Studio** (Flamingo or newer) and press
**Run**. The app requires:
- Android 8.0+ (API 26) for `BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER`
- `arm64-v8a` device (phone / tablet) for XNNPack NEON acceleration

```bash
# Or from the command line
cd android/
./gradlew installDebug
```

---

## Measured metrics

| Metric | Source | Notes |
|--------|--------|-------|
| **TTFT** | `System.nanoTime()` | Time from start of first `forward()` call until first token available |
| **Decode latency** | Per-step `nanoTime()` delta | Min / median / mean / max + histogram |
| **Tokens / second** | 1000 / avg_decode_ms | Sustained decode throughput |
| **Memory (PSS)** | `Debug.getMemoryInfo()` | Proportional Set Size delta = memory attributable to this process |
| **Energy** | `BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER` | µWh consumed; falls back to charge-counter × voltage estimate |

---

## File structure

```
exutorch/
├── model.py                      # Self-contained GPT-2 (exutorch copy)
├── nanogpt_kvcache.py            # GPT-2 with KV-cache, torch.export compatible
├── export_nanogpt.py             # Original simple export (existing)
├── export_nanogpt_xnnpack.py     # XNNPack export (existing)
├── export_nanogpt_android.py     # NEW: FP32 + INT8 + KV-cache export CLI
│
└── android/
    ├── settings.gradle
    ├── build.gradle
    └── app/
        ├── build.gradle          # ExecuTorch AAR dependency
        └── src/main/
            ├── AndroidManifest.xml
            ├── assets/           # ← put .pte + encoder.json + vocab.bpe here
            └── java/com/example/nanogptbench/
                ├── MainActivity.kt      # UI + benchmark orchestration
                ├── NanoGPTRunner.kt     # ExecuTorch Module wrapper + sampling
                ├── BPETokenizer.kt      # Pure-Kotlin GPT-2 BPE tokenizer
                └── BenchmarkMetrics.kt  # TTFT / latency / memory / energy
```

---

## Model architecture recap (from train.py / model.py)

```
nanoGPT (GPT-2)
  Embedding  : wte (vocab×n_embd) + wpe (block_size×n_embd)
  Blocks × 12: LayerNorm → CausalSelfAttention → LayerNorm → MLP
               (each block: 12 heads, n_embd=768, MLP 4×768)
  Final norm  : RMSNorm / LayerNorm
  LM head     : Linear(n_embd → vocab), weight-tied with wte

Export wrapper:
  Input  : LongTensor (1, seq_len)  – token IDs
  Output : FloatTensor (1, 1, 50257) – logits for next token
  Dynamic shape: seq_len ∈ [1, block_size-1]
```

---

## Adding a custom checkpoint

```bash
# Train a model first:
#   python train.py --out_dir out/my_model ...

python export_nanogpt_android.py \
  --checkpoint ../out/my_model/ckpt.pt \
  --out_dir android/app/src/main/assets/
```

For checkpoints trained with the full `model.py` (custom attention variants,
MoE, etc.) the export script loads via `model.py` + `gpt_conf.py`.  Only ops
that lower cleanly through ExecuTorch's EXIR dialect will export; exotic
custom kernels may need to be replaced with standard equivalents first.
