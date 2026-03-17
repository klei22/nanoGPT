#!/usr/bin/env python3
"""
USB IC Recorder Auto-Detection, File Download, and Transcription Pipeline.

Detects a Sony IC Recorder USB device, copies MP3 files organized by date,
and transcribes them using faster-whisper.

Usage:
    # Monitor mode - watch for USB device and process automatically:
    sudo python3 usb_recorder_pipeline.py monitor

    # Manual mode - process an already-mounted device:
    python3 usb_recorder_pipeline.py process /media/$USER/IC\ RECORDER

    # Transcribe only - transcribe already-downloaded files:
    python3 usb_recorder_pipeline.py transcribe ./recordings

Setup (one-time, requires root):
    sudo python3 usb_recorder_pipeline.py install-udev
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# --- Configuration ---
USB_VENDOR_ID = "054c"   # Sony
USB_PRODUCT_ID = "0d93"  # IC Recorder
DEVICE_LABEL = "IC RECORDER"
REC_FILE_SUBDIR = "REC_FILE"  # Path within device to recording folders

OUTPUT_BASE_DIR = Path.home() / "recordings" / "ic_recorder"
WHISPER_MODEL = "base"  # tiny, base, small, medium, large-v2, large-v3
WHISPER_LANGUAGE = "en"  # set to None for auto-detect

# File pattern: YYMMDD_HHMM.mp3 or YYMMDD_HHMM_NN.mp3
FILE_PATTERN = re.compile(
    r"^(\d{6})_(\d{4})(?:_\d+)?\.mp3$", re.IGNORECASE
)


# --- USB Detection via pyudev ---

def monitor_usb(callback):
    """Watch for USB device connections using pyudev."""
    try:
        import pyudev
    except ImportError:
        print("ERROR: pyudev not installed. Install with: pip install pyudev")
        print("Falling back to polling mode...")
        poll_for_device(callback)
        return

    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem="block", device_type="partition")

    print(f"Monitoring for Sony IC Recorder (vendor={USB_VENDOR_ID}, product={USB_PRODUCT_ID})...")
    print("Connect your device now. Press Ctrl+C to stop.\n")

    for device in iter(monitor.poll, None):
        if device.action == "add":
            # Walk up to find the USB device ancestor
            usb_device = device.find_parent("usb", "usb_device")
            if usb_device is None:
                continue

            vendor = usb_device.attributes.get("idVendor", b"").decode()
            product = usb_device.attributes.get("idProduct", b"").decode()

            if vendor == USB_VENDOR_ID and product == USB_PRODUCT_ID:
                dev_node = device.device_node  # e.g., /dev/sdc1
                print(f"IC Recorder detected at {dev_node}")
                mount_point = wait_for_mount(dev_node)
                if mount_point:
                    callback(mount_point)
                else:
                    print(f"  Device {dev_node} not auto-mounted. Attempting manual mount...")
                    mount_point = manual_mount(dev_node)
                    if mount_point:
                        callback(mount_point)
                        manual_unmount(mount_point)


def poll_for_device(callback, interval=3):
    """Fallback: poll /dev/disk/by-id for the device."""
    print(f"Polling for device every {interval}s... Press Ctrl+C to stop.\n")
    seen = set()

    while True:
        try:
            by_id = Path("/dev/disk/by-id")
            if by_id.exists():
                for entry in by_id.iterdir():
                    name = entry.name.lower()
                    if USB_VENDOR_ID in name and USB_PRODUCT_ID in name and "part" in name:
                        real = entry.resolve()
                        if real not in seen:
                            seen.add(real)
                            print(f"IC Recorder detected: {real}")
                            mount_point = wait_for_mount(str(real))
                            if mount_point:
                                callback(mount_point)
            time.sleep(interval)
        except KeyboardInterrupt:
            break


def wait_for_mount(dev_node, timeout=15):
    """Wait for the OS to auto-mount the device."""
    print(f"  Waiting up to {timeout}s for auto-mount of {dev_node}...")
    for _ in range(timeout):
        mount_point = find_mount_point(dev_node)
        if mount_point:
            print(f"  Mounted at: {mount_point}")
            return mount_point
        time.sleep(1)
    print(f"  Device {dev_node} was not auto-mounted within {timeout}s.")
    return None


def find_mount_point(dev_node):
    """Check /proc/mounts for where a device is mounted."""
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == dev_node:
                    return parts[1].replace("\\040", " ")
    except OSError:
        pass
    return None


def manual_mount(dev_node):
    """Attempt to mount the device manually."""
    mount_dir = f"/tmp/ic_recorder_{os.getpid()}"
    os.makedirs(mount_dir, exist_ok=True)
    try:
        subprocess.run(
            ["mount", "-o", "ro", dev_node, mount_dir],
            check=True, capture_output=True
        )
        print(f"  Manually mounted at {mount_dir}")
        return mount_dir
    except subprocess.CalledProcessError as e:
        print(f"  Mount failed: {e.stderr.decode().strip()}")
        os.rmdir(mount_dir)
        return None


def manual_unmount(mount_dir):
    """Unmount and clean up a manual mount."""
    try:
        subprocess.run(["umount", mount_dir], check=True, capture_output=True)
        os.rmdir(mount_dir)
        print(f"  Unmounted {mount_dir}")
    except subprocess.CalledProcessError:
        print(f"  Warning: could not unmount {mount_dir}")


# --- File Copy & Organization ---

def parse_filename(filename):
    """Extract date from filename like 240829_1941.mp3 -> (2024-08-29, 19:41)."""
    match = FILE_PATTERN.match(filename)
    if not match:
        return None, None
    date_str = match.group(1)  # YYMMDD
    time_str = match.group(2)  # HHMM
    try:
        dt = datetime.strptime(date_str + time_str, "%y%m%d%H%M")
        date_folder = dt.strftime("%Y-%m-%d")
        return date_folder, dt
    except ValueError:
        return None, None


def find_recording_files(mount_point):
    """Find all MP3 files on the device."""
    mount_path = Path(mount_point)
    mp3_files = []

    # Check standard IC Recorder paths
    rec_dir = mount_path / REC_FILE_SUBDIR
    if not rec_dir.exists():
        # Try searching the whole device
        rec_dir = mount_path

    for mp3 in rec_dir.rglob("*.mp3"):
        mp3_files.append(mp3)

    # Also look for .MP3 (case-insensitive on some filesystems)
    for mp3 in rec_dir.rglob("*.MP3"):
        if mp3 not in mp3_files:
            mp3_files.append(mp3)

    return sorted(mp3_files)


def copy_files(mount_point, output_base=None):
    """Copy recordings from device, organized into date folders.

    Returns dict mapping date_folder -> list of copied file paths.
    """
    if output_base is None:
        output_base = OUTPUT_BASE_DIR

    output_base = Path(output_base)
    mp3_files = find_recording_files(mount_point)

    if not mp3_files:
        print("No MP3 files found on device.")
        return {}

    print(f"\nFound {len(mp3_files)} MP3 file(s) on device.")
    files_by_date = {}
    copied = 0
    skipped = 0

    for src in mp3_files:
        fname = src.name
        date_folder, dt = parse_filename(fname)

        if date_folder is None:
            # Use file modification time as fallback
            mtime = datetime.fromtimestamp(src.stat().st_mtime)
            date_folder = mtime.strftime("%Y-%m-%d")
            print(f"  {fname}: no date in name, using mtime -> {date_folder}")

        dest_dir = output_base / date_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname

        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            skipped += 1
            # Still track for transcription
            files_by_date.setdefault(date_folder, []).append(dest)
            continue

        print(f"  Copying {fname} -> {dest_dir}/")
        shutil.copy2(str(src), str(dest))
        copied += 1
        files_by_date.setdefault(date_folder, []).append(dest)

    print(f"\nCopied {copied} file(s), skipped {skipped} existing file(s).")
    return files_by_date


# --- Transcription with faster-whisper ---

def transcribe_files(files_by_date, output_base=None):
    """Transcribe MP3 files using faster-whisper, one transcript per day."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("\nERROR: faster-whisper not installed.")
        print("Install with: pip install faster-whisper")
        return

    if output_base is None:
        output_base = OUTPUT_BASE_DIR
    output_base = Path(output_base)

    print(f"\nLoading Whisper model '{WHISPER_MODEL}'...")
    model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")

    for date_folder in sorted(files_by_date.keys()):
        files = sorted(files_by_date[date_folder])
        transcript_path = output_base / date_folder / f"transcript_{date_folder}.txt"

        print(f"\n--- Transcribing {len(files)} file(s) for {date_folder} ---")

        with open(transcript_path, "w") as out:
            out.write(f"Transcription for {date_folder}\n")
            out.write(f"{'=' * 40}\n\n")

            for mp3_path in files:
                print(f"  Transcribing: {mp3_path.name}")
                out.write(f"--- {mp3_path.name} ---\n")

                try:
                    segments, info = model.transcribe(
                        str(mp3_path),
                        language=WHISPER_LANGUAGE,
                        beam_size=5,
                    )

                    if WHISPER_LANGUAGE is None:
                        out.write(
                            f"Detected language: {info.language} "
                            f"(probability {info.language_probability:.2f})\n\n"
                        )

                    for segment in segments:
                        timestamp = f"[{segment.start:.1f}s -> {segment.end:.1f}s]"
                        out.write(f"{timestamp}  {segment.text.strip()}\n")

                except Exception as e:
                    msg = f"  ERROR transcribing {mp3_path.name}: {e}"
                    print(msg)
                    out.write(f"\n{msg}\n")

                out.write("\n")

        print(f"  Transcript saved: {transcript_path}")


def transcribe_from_directory(directory):
    """Build files_by_date from an existing directory of date-folders."""
    directory = Path(directory)
    files_by_date = {}

    for date_dir in sorted(directory.iterdir()):
        if date_dir.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
            mp3s = sorted(date_dir.glob("*.mp3")) + sorted(date_dir.glob("*.MP3"))
            if mp3s:
                files_by_date[date_dir.name] = mp3s

    if not files_by_date:
        # Maybe flat directory of mp3s
        mp3s = sorted(directory.glob("*.mp3")) + sorted(directory.glob("*.MP3"))
        for mp3 in mp3s:
            date_folder, _ = parse_filename(mp3.name)
            if date_folder is None:
                mtime = datetime.fromtimestamp(mp3.stat().st_mtime)
                date_folder = mtime.strftime("%Y-%m-%d")
            files_by_date.setdefault(date_folder, []).append(mp3)

    return files_by_date


# --- udev Rule Installation ---

UDEV_RULE = f"""\
# Auto-process Sony IC Recorder when connected
ACTION=="add", SUBSYSTEM=="block", \\
  ATTRS{{idVendor}}=="{USB_VENDOR_ID}", ATTRS{{idProduct}}=="{USB_PRODUCT_ID}", \\
  ENV{{DEVTYPE}}=="partition", \\
  RUN+="/bin/sh -c 'echo $devnode >> /tmp/ic_recorder_trigger'"
"""

SYSTEMD_SERVICE = """\
[Unit]
Description=Sony IC Recorder Auto-Processor
After=local-fs.target

[Service]
Type=simple
ExecStart={script} monitor
Restart=on-failure
User={user}
Environment=HOME={home}

[Install]
WantedBy=multi-user.target
""".format(
    script=os.path.abspath(__file__),
    user=os.environ.get("SUDO_USER", os.environ.get("USER", "root")),
    home=os.environ.get("HOME", "/root"),
)


def install_udev():
    """Install udev rule for automatic detection."""
    if os.geteuid() != 0:
        print("ERROR: Must run as root to install udev rules.")
        print(f"  sudo python3 {__file__} install-udev")
        sys.exit(1)

    # Write udev rule
    rule_path = "/etc/udev/rules.d/99-ic-recorder.rules"
    with open(rule_path, "w") as f:
        f.write(UDEV_RULE)
    print(f"Installed udev rule: {rule_path}")

    # Reload udev
    subprocess.run(["udevadm", "control", "--reload-rules"], check=True)
    subprocess.run(["udevadm", "trigger"], check=True)
    print("Reloaded udev rules.")

    # Optionally install systemd service
    svc_path = "/etc/systemd/system/ic-recorder-processor.service"
    with open(svc_path, "w") as f:
        f.write(SYSTEMD_SERVICE)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    print(f"Installed systemd service: {svc_path}")
    print("  Enable with: sudo systemctl enable --now ic-recorder-processor")


# --- Pipeline ---

def process_device(mount_point):
    """Full pipeline: copy files then transcribe."""
    print(f"\n{'=' * 50}")
    print(f"Processing device at: {mount_point}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"{'=' * 50}")

    files_by_date = copy_files(mount_point, OUTPUT_BASE_DIR)
    if files_by_date:
        transcribe_files(files_by_date, OUTPUT_BASE_DIR)

    print(f"\nDone! Files are in: {OUTPUT_BASE_DIR}")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Sony IC Recorder auto-detect, download, and transcription pipeline."
    )
    sub = parser.add_subparsers(dest="command")

    # Monitor for USB connection
    sub.add_parser("monitor", help="Watch for USB device and process automatically")

    # Process already-mounted device
    p_proc = sub.add_parser("process", help="Process an already-mounted device")
    p_proc.add_argument("mount_point", help="Path to mounted device")

    # Transcribe existing files
    p_trans = sub.add_parser("transcribe", help="Transcribe already-downloaded files")
    p_trans.add_argument("directory", help="Directory containing recordings")

    # Install udev rules
    sub.add_parser("install-udev", help="Install udev rule (requires root)")

    args = parser.parse_args()

    if args.command == "monitor":
        monitor_usb(process_device)

    elif args.command == "process":
        process_device(args.mount_point)

    elif args.command == "transcribe":
        directory = Path(args.directory)
        if not directory.exists():
            print(f"ERROR: Directory not found: {directory}")
            sys.exit(1)
        files_by_date = transcribe_from_directory(directory)
        if files_by_date:
            transcribe_files(files_by_date, directory)
        else:
            print("No MP3 files found.")

    elif args.command == "install-udev":
        install_udev()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
