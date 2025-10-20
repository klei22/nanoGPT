# Exporting nanoGPT Models for EdgeAI Android App

This directory contains tooling that bridges nanoGPT training checkpoints with the
`EdgeAIApp-ExecuTorch` Android application.  The utilities here create
ExecuTorch compatible artifacts and optionally deliver them to an Android
device so the mobile demo can execute your custom model.

## Workflow Overview

1. **Convert a checkpoint to ExecuTorch (.pte) format** using
   [`executorch/export_to_executorch.py`](executorch/export_to_executorch.py).
2. **Copy the generated assets into the Android project** with
   [`executorch/copy_assets.py`](executorch/copy_assets.py) when you want the
   files baked into the app bundle.
3. **(Optional) Push the artifacts directly to a connected device** via ADB
   with [`executorch/push_to_android.py`](executorch/push_to_android.py).

Each step is independent, allowing you to either ship artifacts with the app or
sideload them on demand.

Refer to the [`executorch` README](executorch/README.md) for detailed usage
instructions.

## End-to-end demo

Run `demos/executorch_android_export_demo.sh` for a lightweight example that
downloads the Shakespeare dataset, trains a compact checkpoint, exports it to
ExecuTorch, and copies the resulting assets into the Android demo project.  The
script also shows how to dry-run the optional ADB push step.
