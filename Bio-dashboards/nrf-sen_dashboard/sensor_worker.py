import asyncio
import csv
import datetime
import os
import sys
import json
import time
import logging
from bleak import BleakScanner, BleakClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_air")
os.makedirs(LOGS_DIR, exist_ok=True)

DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")

# Nordic UART Service UUIDs
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
csv_path = os.path.join(LOGS_DIR, f"sen69c_telemetry_{date_str}.csv")

csv_file = open(csv_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Timestamp_Epoch_ms", "PM1_0", "PM2_5", "PM4_0", "PM10_0", 
    "Humidity_RH", "Temp_C", "VOC_Index", "NOx_Index", "HCHO_ppb", "CO2_ppm"
])

data_buffer = ""
samples_written = 0
selected_target_mac = None
last_message_time = 0

MAX_BUFFER = 4096  # a line is ~60 bytes; anything this long means framing is lost

def handle_rx(sender, data):
    """Buffers UART chunks and writes EVERY complete line to the CSV.

    A single BLE notification can carry the end of one line and all of the next,
    so we must drain the whole buffer, not just the first line.
    """
    global data_buffer, samples_written, last_message_time
    last_message_time = time.time()

    data_buffer += data.decode("utf-8", errors="replace")
    if "\n" not in data_buffer:
        if len(data_buffer) > MAX_BUFFER:
            data_buffer = ""
        return

    *lines, data_buffer = data_buffer.split("\n")
    epoch_ms = int(time.time() * 1000)
    rows = []
    for line in lines:
        metrics = [m.strip() for m in line.strip().split(",")]
        if len(metrics) >= 10:
            rows.append([epoch_ms] + metrics[:10])

    if rows:
        csv_writer.writerows(rows)
        csv_file.flush()
        samples_written += len(rows)
        sys.stdout.write(f"\r[ 🌍 Air Monitor Connected ] | Samples Logged: {samples_written:5d} | CO2: {rows[-1][10]:>4s} ppm   ")
        sys.stdout.flush()

ble_device_cache = {}
NAME_HINTS = ("SuperMini", "SEN69", "Air", "Atmos")

def _read_command():
    if not os.path.exists(COMMAND_FILE):
        return None
    try:
        with open(COMMAND_FILE, "r") as f:
            cmd = json.load(f)
        os.remove(COMMAND_FILE)
        if cmd.get("action") == "connect":
            return cmd.get("address")
    except Exception:
        pass
    return None

async def scan_and_list_devices():
    """Scans for active environmental monitors and updates the UI state list."""
    global selected_target_mac

    logging.info("Radar active. Sweeping for air monitors...")
    found_devices = {}

    def detection_callback(device, adv_data):
        name = device.name or adv_data.local_name or ""
        uuids = [u.lower() for u in (adv_data.service_uuids or [])]
        # Name hints, or anything advertising Nordic UART -- so a board swap/rename still shows up
        if any(h in name for h in NAME_HINTS) or UART_SERVICE_UUID in uuids:
            ble_device_cache[device.address] = device
            found_devices[device.address] = {
                "name": name or f"UART device ({device.address[-5:]})",
                "address": device.address,
                "rssi": adv_data.rssi if adv_data.rssi else -100,
            }

    try:
        async with BleakScanner(detection_callback, passive=False):
            for _ in range(15):  # up to 3 s, exit early once the UI picks something
                await asyncio.sleep(0.2)
                if os.path.exists(COMMAND_FILE):
                    break
    except Exception as e:
        logging.error(f"Radar error: {e}")

    targets = sorted(found_devices.values(), key=lambda x: x["rssi"], reverse=True)
    with open(DEVICES_FILE + ".tmp", "w") as f:
        json.dump(targets, f)
    os.replace(DEVICES_FILE + ".tmp", DEVICES_FILE)

    mac = _read_command()
    if mac:
        selected_target_mac = mac
        logging.info(f"UI Command received! Target locked: {selected_target_mac}")

async def main():
    global selected_target_mac, last_message_time, data_buffer
    logging.info("Hardware Engine Online.")
    
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass

    while True:
        if not selected_target_mac:
            await scan_and_list_devices()
            continue

        try:
            # Reuse the BLEDevice from the scan instead of running a second full scan
            device = ble_device_cache.get(selected_target_mac) or \
                await BleakScanner.find_device_by_address(selected_target_mac, timeout=5.0)
            if not device:
                logging.warning("Target device vanished. Returning to radar scan.")
                selected_target_mac = None
                continue

            logging.info(f"Connecting to {device.name} [{device.address}]...")

            async with BleakClient(device) as client:
                data_buffer = ""
                logging.info("Connected! Subscribing to Nordic UART TX characteristic...")
                await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
                
                logging.info(f"!!! TELEMETRY ACTIVE: Logging live to {LOGS_DIR} !!!\n")
                
                # Watchdog Timer: Reset if no data arrives for 10 seconds
                last_message_time = time.time()
                while time.time() - last_message_time < 10.0:
                    await asyncio.sleep(1)
                    
                logging.warning("\nConnection Dropped (Data flow stopped). Returning to Scanner...")
                selected_target_mac = None
                
        except (asyncio.TimeoutError, TimeoutError):
            logging.warning("\n⚠️ BLE Timeout. Resetting radar...")
            selected_target_mac = None
            await asyncio.sleep(2)
        except Exception as e:
            logging.error(f"\n❌ Unexpected Error: {e}. Resetting radar...")
            selected_target_mac = None
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nHalting streaming contexts. Closing open file descriptors...")
        csv_file.close()
        logging.info("Logs closed cleanly. Safe to exit.")
