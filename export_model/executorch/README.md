# ExecuTorch Export Utilities

These utilities transform nanoGPT training checkpoints into ExecuTorch
programs that the `EdgeAIApp-ExecuTorch` demo can execute on Android.  The
scripts cover every step from compilation to deployment.

## 1. Export a checkpoint to `.pte`

```bash
python export_to_executorch.py \
  --checkpoint path/to/ckpt.pt \
  --output-dir export_output \
  --model-name custom-nanogpt \
  --tokenizer path/to/tokenizer.json
```

Key options:

- `--checkpoint`: Path to the nanoGPT checkpoint produced by training.
- `--output-dir`: Directory where the ExecuTorch program and metadata will be
  written.  The directory is created if needed.
- `--model-name`: Name used for generated files (`<name>.pte`, metadata, etc.).
- `--block-size`: Override the context length if the Android app should use a
  smaller maximum sequence than the training checkpoint.
- `--dtype`: Convert weights to `float32` (default) or `float16` before export.
- `--tokenizer`: Optional tokenizer JSON/zip to copy next to the exported
  program.

The script emits:

- `<model-name>.pte`: ExecuTorch compiled model.
- `<model-name>.json`: Metadata describing configuration, vocabulary and input
  shapes.
- Optional tokenizer/config copies for convenient packaging.

## 2. Copy artifacts into the Android project

```bash
python copy_assets.py \
  --artifacts export_output \
  --app nanoGPT/EdgeAIApp-ExecuTorch \
  --model-name custom-nanogpt
```

This copies the `.pte` file, metadata, and any tokenizer/config assets into
`app/src/main/assets/models/<model-name>/`.  Use `--install-default-name` to
also copy the `.pte` as `llama_model.pte`, matching the file name expected by
the current Kotlin loaders.

## 3. Push artifacts to a connected device (optional)

```bash
python push_to_android.py \
  --artifacts export_output \
  --device-dir /sdcard/Android/data/com.example.edgeai/files/models
```

The script wraps common ADB commands to create the destination directory and
push every artifact so you can sideload models without rebuilding the app.
Override `--adb` if the binary is not on your `PATH`.

---

All scripts include `--dry-run` and verbose logging flags.  Run
`python <script> --help` for the full option list.
