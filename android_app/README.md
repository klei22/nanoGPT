# nanoGPT Android benchmark app

This folder is a minimal Android Studio project for loading an exported nanoGPT model on an Android device and reporting baseline generation runtime stats.

## What is included

- Kotlin single-activity app.
- Prompt text input.
- Numeric `max_new_tokens` input.
- `Run benchmark` button.
- Runtime abstraction with:
  - `ModelRunner`
  - `OrtModelRunner` for `model.onnx` through ONNX Runtime Mobile
  - `ExecuTorchModelRunner` placeholder for `model.pte`
- Greedy `argmax` next-token decoding.
- Warmup run separated from the measured run.
- Runtime stats for TTFT, TPOT, total generation time, prompt/generated token counts, throughput, Java/Kotlin heap, native heap, and Android PSS.

## Copy the exported model and tokenizer

From the repository root, copy one exported model artifact into Android assets:

```bash
cp android_export/model.onnx android_app/app/src/main/assets/model.onnx
```

or, if you are adding ExecuTorch bindings later:

```bash
cp android_export/model.pte android_app/app/src/main/assets/model.pte
```

Then copy the tokenizer metadata that matches the exported checkpoint:

```bash
cp android_export/tokenizer.json android_app/app/src/main/assets/tokenizer.json
```

or:

```bash
cp android_export/meta.json android_app/app/src/main/assets/meta.json
```

A small placeholder `meta.json` is committed only so the first app build has an assets directory. Replace it before collecting real benchmark numbers.

## Expected ONNX signature

The first baseline expects an ONNX model with one integer input tensor shaped `[1, sequence_length]`. `OrtModelRunner` feeds the current token context to the first model input and reads the first output as logits. Supported logits shapes are:

- `[1, sequence_length, vocab_size]`
- `[sequence_length, vocab_size]`
- `[vocab_size]`

If your export uses additional inputs such as `attention_mask`, `position_ids`, KV-cache tensors, or fixed sequence lengths, extend `OrtModelRunner.runForFinalLogits` accordingly.

## Tokenizer format

The simple baseline tokenizer supports character-level nanoGPT metadata encoded as JSON:

```json
{
  "stoi": {"a": 0, "b": 1},
  "itos": {"0": "a", "1": "b"}
}
```

If `itos` is omitted, it is derived by reversing `stoi`. A tokenizer JSON with `model.vocab` is also accepted for simple vocab lookup, but this app intentionally does not implement full BPE merges yet. Keep the baseline greedy and simple until latency is validated.

## Open in Android Studio

1. Open `android_app/` in Android Studio.
2. Let Gradle sync dependencies.
3. Connect a device or start an emulator.
4. Run the `app` configuration.
5. Enter a prompt and token count, then tap **Run benchmark**.

## Metrics

- **TTFT ms**: elapsed time from the first measured model invocation start until the first generated token is available.
- **TPOT ms/token**: average elapsed time per generated token after the first generated token.
- **Total generation time**: elapsed wall-clock time for the measured generation loop.
- **tokens/sec**: generated tokens divided by total generation time.
- **Java/Kotlin heap MB**: `Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()`.
- **native heap MB**: `Debug.getNativeHeapAllocatedSize()`.
- **Android PSS MB**: `Debug.MemoryInfo.totalPss` when available from Android APIs.
