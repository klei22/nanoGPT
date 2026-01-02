# Ubuntu Android Setup Guide (ExecuTorch nanoGPT demo)

This guide walks through setting up the Android build environment on Ubuntu,
building the demo app, and installing it on a virtual or real device.

## 1) Run setup scripts

All command sequences are grouped into numbered scripts under
`exutorch/android_app/scripts/`. Run them in order:

```bash
cd exutorch/android_app
bash scripts/01_install_deps.sh
```

Verify Java:

```bash
java -version
```

## 2) Install Android SDK command-line tools + env vars

```bash
cd exutorch/android_app
bash scripts/02_install_android_sdk.sh
```

Reload the shell after the script completes:

```bash
source ~/.bashrc
```

## 3) Install SDK packages

```bash
cd exutorch/android_app
bash scripts/03_install_sdk_packages.sh
```

## 4) Create an Android Virtual Device (AVD)

```bash
cd exutorch/android_app
bash scripts/04_create_avd.sh
```

Start the emulator:

```bash
cd exutorch/android_app
bash scripts/05_start_emulator.sh
```

## 5) Install the app on a virtual device

```bash
cd exutorch/android_app
bash scripts/06_install_app.sh
```

Open Logcat to see metrics:

```bash
cd exutorch/android_app
bash scripts/07_logcat.sh
```

You should see:
- `TTFT ms: ...`
- `Avg decode ms/token: ...`

## 6) Install on a real device

On the Android device:
- Enable **Developer options**.
- Enable **USB debugging**.

Then run:

```bash
cd exutorch/android_app
bash scripts/08_real_device.sh
```

## 7) Ensure assets are in place

Before building, copy your exported artifacts:

```bash
cd exutorch/android_app
bash scripts/00_copy_assets.sh
```

## Troubleshooting

- **Emulator fails to launch**: Ensure KVM is enabled and you’re in the `kvm` group.
  ```bash
  sudo apt install -y qemu-kvm
  sudo usermod -aG kvm $USER
  ```
  Log out/in, then retry.

- **Gradle can’t find SDK**: make sure `ANDROID_SDK_ROOT` is set in your shell and
  the path exports are loaded.
