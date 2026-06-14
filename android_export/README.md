# Android tokenizer export

The mobile model artifact (`.pte`, `.tflite`, or another exported runtime file) is not sufficient for correctness testing by itself. The Android app must feed the model the same token IDs that Python inference would feed for the same prompt, and it must decode generated IDs with the same tokenizer.

A tokenizer mismatch invalidates TTFT/TPOT correctness testing: timing may still be measured, but the app can be testing different prompts and generated IDs than Python inference because the token IDs differ.

## Export char-level tokenizer JSON

During training, nanoGPT copies `data/<dataset>/meta.pkl` into the checkpoint output directory. Export that file next to the Android app assets:

```bash
python android_export/export_tokenizer.py \
  --checkpoint-dir out \
  --output android_app/app/src/main/assets/tokenizer.json \
  --pretty
```

If `meta.pkl` lives somewhere else, pass it explicitly:

```bash
python android_export/export_tokenizer.py \
  --meta-path data/shakespeare_char/meta.pkl \
  --output android_app/app/src/main/assets/tokenizer.json \
  --pretty
```

For character-level or dataset-specific `meta.pkl` files that contain `stoi` and `itos`, the exported JSON has this shape:

```json
{
  "tokenizer_type": "char",
  "stoi": {"a": 0},
  "itos": {"0": "a"},
  "vocab_size": 1
}
```

The checked-in `android_app/app/src/main/assets/tokenizer.json` is a small ASCII char-level starter asset for Android parsing and encode/decode validation. Replace it with the JSON exported from the exact checkpoint used for the mobile model before running correctness or TTFT/TPOT comparisons.

## Android interface

Android tokenizers should implement:

```kotlin
interface Tokenizer {
    fun encode(text: String): IntArray
    fun decode(tokens: IntArray): String
}
```

`JsonCharTokenizer` is the starter implementation for char-level JSON. It loads `tokenizer.json`, validates `tokenizer_type`, `stoi`, `itos`, and `vocab_size`, then encodes each Unicode code point to the matching ID.

## GPT-2 / tiktoken models

If `meta.pkl` declares `tokenizer == "tiktoken"`, do **not** export a char-level `stoi`/`itos` JSON file. Android needs an implementation of the same byte-pair encoding used by Python `tiktoken`, including:

- the same encoding name, usually `gpt2` for GPT-2-compatible nanoGPT runs;
- byte-to-unicode handling that matches the Python tokenizer;
- the full BPE vocabulary/rank table;
- merge ranks or equivalent rank data;
- special-token IDs and allowed-special-token behavior.

Ship those tokenizer assets beside the model artifact and load them through a `Tokenizer` implementation with the same `encode(text: String): IntArray` and `decode(tokens: IntArray): String` contract. Validate Android by comparing token IDs and decoded text against Python for a fixed prompt before using TTFT/TPOT results for correctness claims.

## Validation checklist

1. Run the export script against the checkpoint directory that contains the model artifact's `meta.pkl`.
2. Copy the resulting `tokenizer.json` into `android_app/app/src/main/assets/`.
3. In Android instrumentation or unit tests, compare `encode(prompt)` output to Python for representative prompts.
4. Decode a known token sequence on both Android and Python and verify byte-for-byte matching text.
5. Only after tokenizer parity passes, use TTFT/TPOT measurements for correctness-sensitive comparisons.
