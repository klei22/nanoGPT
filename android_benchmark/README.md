# NanoGPT Android Hardware Benchmark

This directory contains a minimal Android app source tree for strategic LLM
performance benchmarking on Android hardware. It does **not** commit an APK or a
model artifact. Instead, convert one of this repository's compatible `ckpt.pt`
files into `app/src/main/assets/nanogpt.bin`, build the app, and run the native
CPU benchmark on a device.

The app is intentionally small and scriptable so it works well with Android CLI
agent workflows:

- `android create` is not required because the Gradle project is already checked
  in.
- `android sdk install ...` can provision SDK packages in CI or on a workstation.
- `android run --apks=...` can install a built debug APK without Android Studio.
- `android screen capture` and `android layout` can collect benchmark evidence
  from a connected device.

## What is measured?

The included JNI implementation loads a converted fp32 checkpoint and performs a
simple greedy decode loop. It reports elapsed time and tokens per second for a
fixed 32-token generation run. This is a correctness-oriented baseline for
comparing CPU paths across devices. For production-grade mobile inference,
compare this against the existing ExecuTorch/XNNPACK path in `../exutorch/`.

Supported checkpoints are the repository's default GPT path:

- causal attention
- RMSNorm pre-norm and output norm
- absolute positional embeddings
- GELU MLP
- untied architectural experiments disabled (`use_moe=False`, no rotary, no
  factorized embeddings, no parallel/ASIC block variants)

The converter fails fast for unsupported variants so benchmark results are not
silently invalid.

## Convert a checkpoint

From the repository root:

```bash
python android_benchmark/scripts/convert_ckpt_to_nanogpt_bin.py \
  out/my_run/ckpt.pt \
  --out android_benchmark/app/src/main/assets/nanogpt.bin
```

The generated `nanogpt.bin` is ignored by git. Keep it out of commits and upload
it separately if you need to preserve a benchmark artifact.

## Build the APK

### Option A: Android CLI

Install/update Android CLI, initialize agent skills, and install SDK packages:

```bash
android update
android init
android sdk install platforms/android-35 build-tools/35.0.0 platform-tools
android sdk list 'cmake|ndk' --all
android sdk install '<cmake-package-name>' '<ndk-package-name>'
```

Build with Gradle from this directory:

```bash
cd android_benchmark
./gradlew :app:assembleDebug
```

If a Gradle wrapper is not present in your checkout, use a system Gradle install:

```bash
cd android_benchmark
gradle :app:assembleDebug
```

### Option B: Android Studio

Open `android_benchmark/` in Android Studio and run the `app` configuration. The
`studio` Android CLI commands from Android Studio Quail 2 Canary 1+ can inspect
or open files when Gemini in Android Studio is enabled.

## Install and run on a device

With Android CLI:

```bash
android run --apks=android_benchmark/app/build/outputs/apk/debug/app-debug.apk
```

Or with adb:

```bash
adb install -r android_benchmark/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.nanogpt.benchmark/.MainActivity
```

Tap **Run 32-token benchmark**. To collect evidence in an automated run:

```bash
android screen capture --output=benchmark.png
android layout --pretty --output=layout.json
```

## Notes for better benchmarking

1. Run on battery and plugged-in modes if thermal policy matters.
2. Reboot or cool the device between long runs.
3. Record model dimensions, Android build, SoC, governor state, and app version.
4. Repeat at least 5 times and compare median tokens/second.
5. Use the same `nanogpt.bin` for all devices in a comparison.
