# Android ExecuTorch nanoGPT template app

This Gradle project provides a minimal Android application that loads a nanoGPT ExecuTorch `.pte` artifact, runs autoregressive generation, and reports latency metrics that can be harvested from `logcat` or via `adb shell` invocations.

## Features

* Loads a `.pte` file and vocabulary JSON from the app `assets` directory.
* Exposes a broadcast receiver that accepts prompts and generation limits via `adb`.
* Spins up a foreground service that performs generation, computes TTFT/TPOT, and emits timing markers (`EXECUTORCH_METRICS_BEGIN` / `EXECUTORCH_METRICS_END`).
* Supports context-length sweeps by passing a comma-separated list to the broadcast intent.

## Project structure

```
template_app/
  build.gradle.kts            # Top-level Gradle configuration
  settings.gradle.kts         # Includes the `app` module
  app/
    build.gradle.kts          # Android application module configuration
    libs/                     # Drop ExecuTorch Android AARs here
    src/main/
      AndroidManifest.xml     # Declares main activity, service, and broadcast receiver
      assets/
        nanogpt.pte           # (Place exported ExecuTorch program here)
        vocab.json            # (Tokenizer vocabulary used by the checkpoint)
      java/com/nanogpt/executorch/template/
        MainActivity.kt
        GenerationBroadcastReceiver.kt
        NanoGptService.kt
        TokenSampler.kt
        VocabTokenizer.kt
      res/
        layout/activity_main.xml
        values/*.xml          # Basic theme + string resources
```

> **Note**
> The template references the ExecuTorch Android runtime AAR. Follow the ExecuTorch build instructions to produce `executorch-android.aar` and drop it into `app/libs/`. Update the `implementation` dependency in `app/build.gradle.kts` if your artifact path differs.

## Building

1. Copy your exported `nanogpt.pte` and matching `vocab.json` into `app/src/main/assets/`.
2. Place the ExecuTorch Android runtime AAR into `app/libs/` (create the directory if needed).
3. From this directory run:

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Ensure you have the Android SDK installed and environment variables configured (`ANDROID_HOME`, `JAVA_HOME`).

## Triggering generation via `adb`

The app listens for the `com.nanogpt.executorch.template.GENERATE` broadcast intent. Example invocation:

```bash
adb shell am broadcast \
  -a com.nanogpt.executorch.template.GENERATE \
  --es prompt "To be, or not to be" \
  --ei max_tokens 64 \
  --ei context_length 128
```

To sweep multiple context lengths in one call, supply a comma-separated list:

```bash
adb shell am broadcast \
  -a com.nanogpt.executorch.template.GENERATE \
  --es prompt "Hello" \
  --ei max_tokens 32 \
  --es context_sweep "32,64,128"
```

The app logs metrics for each context length using markers compatible with `profile_pte.py`. Inspect the output with:

```bash
adb logcat -s NanoGPTTemplate
```

or run the profiling helper in broadcast mode:

```bash
python hardware_targets/android/profile_pte.py \
  --runner adb:broadcast \
  --prompt "Hello" \
  --context-sweep "64,128,256" \
  --wait-seconds 8.0
```

## Metric format

Each generation emits JSON like the following between the ExecuTorch markers:

```json
{
  "ctx64_ttft": {"tokens": 1, "latency_ms": 145.1, "energy_mj": 0.0},
  "ctx64_decode": {"tokens": 32, "latency_ms": 812.4, "energy_mj": 0.0}
}
```

* `ctxXX_ttft` captures *time to first token* in milliseconds.
* `ctxXX_decode` captures aggregate decode latency; divide by `tokens` for TPOT.

The `profile_pte.py` helper calculates per-token latency automatically.

## Customisation tips

* Replace `VocabTokenizer` with a tokenizer compatible with your checkpoint (e.g. BPE or sentencepiece).
* Hook device-specific energy probes into `NanoGptService` and populate the `energy_mj` field.
* Extend `TokenSampler` to support nucleus sampling, temperature scaling, or other strategies.

Contributions that flesh out the UI, add device automation, or integrate power measurement APIs are welcome.
