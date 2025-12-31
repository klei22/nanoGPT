# Ubuntu Android Setup Guide (ExecuTorch nanoGPT demo)

This guide walks through setting up the Android build environment on Ubuntu,
building the demo app, and installing it on a virtual or real device.

## 1) Install system dependencies

```bash
sudo apt update
sudo apt install -y \
  unzip \
  zip \
  curl \
  wget \
  git \
  openjdk-17-jdk \
  libgl1 \
  libpulse0 \
  libxkbcommon0 \
  libxcomposite1 \
  libxdamage1 \
  libxrandr2 \
  libgtk-3-0
```

Verify Java:

```bash
java -version
```

## 2) Install Android SDK command-line tools

Set an SDK location and download the command-line tools:

```bash
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
mkdir -p "$ANDROID_SDK_ROOT/cmdline-tools"
cd /tmp
curl -fLO https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip commandlinetools-linux-11076708_latest.zip
mv cmdline-tools "$ANDROID_SDK_ROOT/cmdline-tools/latest"
```

Update your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export ANDROID_SDK_ROOT="$HOME/Android/Sdk"
export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/emulator:$PATH"
```

Reload the shell:

```bash
source ~/.bashrc
```

## 3) Install SDK packages

Accept licenses and install required components:

```bash
sdkmanager --licenses

sdkmanager \
  "platform-tools" \
  "platforms;android-34" \
  "build-tools;34.0.0" \
  "emulator" \
  "system-images;android-34;google_apis;x86_64"
```

## 4) Create an Android Virtual Device (AVD)

```bash
avdmanager create avd \
  --name nanogpt_avd \
  --package "system-images;android-34;google_apis;x86_64" \
  --device "pixel"
```

Start the emulator:

```bash
emulator -avd nanogpt_avd
```

If you need to run headless:

```bash
emulator -avd nanogpt_avd -no-window -no-audio
```

## 5) Install the app on a virtual device

From the repo root:

```bash
cd exutorch/android_app
./gradlew :app:installDebug
```

Open Logcat to see metrics:

```bash
adb logcat | grep NanoGPT
```

You should see:
- `TTFT ms: ...`
- `Avg decode ms/token: ...`

## 6) Install on a real device

On the Android device:
- Enable **Developer options**.
- Enable **USB debugging**.

Connect the device and verify:

```bash
adb devices
```

Then install:

```bash
cd exutorch/android_app
./gradlew :app:installDebug
```

Open Logcat:

```bash
adb logcat | grep NanoGPT
```

## 7) Ensure assets are in place

Before building, copy your exported artifacts:

```bash
cp ../android_export/nanogpt_xnnpack.pte app/src/main/assets/
cp ../android_export/manifest.json app/src/main/assets/
```

Update `prompt_tokens.txt` if desired:

```bash
cat > app/src/main/assets/prompt_tokens.txt <<'EOF'
464, 3290, 318, 257, 1332
EOF
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
