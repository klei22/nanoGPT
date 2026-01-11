# ExecuTorch Android Demo (nanoGPT)

This is a bare-bones Android app that loads a nanoGPT ExecuTorch PTE and
prints timing metrics for autoregressive decoding.

## What this app does

- Loads the exported `nanogpt_xnnpack.pte` from assets.
- Reads `manifest.json` for model configuration (e.g., `block_size`, `vocab_size`).
- Reads a comma-separated list of prompt token IDs from `prompt_tokens.txt`.
- Runs greedy decoding and reports:
  - **Time to first token** (TTFT)
  - **Time per decode token** (average after the first token)

## Export a PTE + tokenizer assets

From the repo root:

```bash
python exutorch/export_nanogpt_android.py \
  --checkpoint out/ckpt.pt \
  --out-dir exutorch/android_export
```

Copy the artifacts into the app assets:

```bash
cp exutorch/android_export/nanogpt_xnnpack.pte \
  exutorch/android_app/app/src/main/assets/
cp exutorch/android_export/manifest.json \
  exutorch/android_app/app/src/main/assets/
```

If you have tokenizer assets you want to bundle (for offline decoding), copy
them into `app/src/main/assets/tokenizer/` as needed.

## Provide prompt tokens

For a minimal demo, generate prompt token IDs on your dev machine and place them
into `prompt_tokens.txt` as a comma-separated list:

```
464, 3290, 318, 257, 1332
```

This avoids shipping a tokenizer in the app. If you want in-app tokenization,
port the tokenizer logic and use the exported metadata under `tokenizer/`.

## Add ExecuTorch Android runtime

This project expects an ExecuTorch Android AAR. You can:

1. Build ExecuTorch Android artifacts from the ExecuTorch repo.
2. Drop the resulting `executorch-android.aar` into `app/libs/`.

The `app/build.gradle` file already includes a `flatDir` repository and
`implementation(files("libs/executorch-android.aar"))` entry.

## Build and run

From this folder:

```bash
./gradlew :app:installDebug
```

Open the app on device and check Logcat:

- `TTFT ms: ...`
- `Avg decode ms/token: ...`

You can tweak `maxNewTokens` in `MainActivity.kt` to control decode length.
