# ble_worker.py
import asyncio
import time
import csv
import datetime
import json
import logging
import os
import sys
from bleak import BleakClient, BleakScanner
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)]
)

# Known hardcoded fallback match
KNOWN_MAC = "F3:A0:A8:E3:F5:63"
HEALTH_SERVICE_UUID = "14839ac4-7d7e-415c-9a42-167340cf2339"
WRITE_CHAR_UUID = "8b00ace7-eb0a-49b0-b977-10a8d4d5e82f" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
DEVICES_FILE = os.path.join(BASE_DIR, "devices.json")
COMMAND_FILE = os.path.join(BASE_DIR, "command.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

current_metrics = {"spo2": 0, "hr": 0, "battery": 100, "status": "Scanning"}
selected_target_mac = None
ble_device_cache = {}

def save_to_json_dashboard():
    data = {
        "spo2": current_metrics["spo2"] if current_metrics["status"] == "Connected" else 0,
        "hr": current_metrics["hr"] if current_metrics["status"] == "Connected" else 0,
        "battery": current_metrics["battery"],
        "status": current_metrics["status"]
    }
    temp_file = DATA_FILE + ".tmp"
    try:
        with open(temp_file, "w") as f: json.dump(data, f)
        os.replace(temp_file, DATA_FILE)
    except Exception: pass

def append_to_csv_log():
    if current_metrics["status"] != "Connected": return
    now = datetime.datetime.now()
    csv_file_path = os.path.join(LOGS_DIR, f"health_log_{now.strftime('%Y-%m-%d')}.csv")
    file_exists = os.path.exists(csv_file_path)
    try:
        with open(csv_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "SpO2 (%)", "Heart Rate (BPM)", "Battery (%)"])
            writer.writerow([now.strftime("%H:%M:%S"), current_metrics["spo2"], current_metrics["hr"], current_metrics["battery"]])
    except Exception: pass

def parse_checkme_notification(sender, data):
    if len(data) == 0: return
    if len(data) == 1 and data[0] == 0xA5:
        if current_metrics["status"] != "Calibrating":
            current_metrics["status"] = "Calibrating"
            save_to_json_dashboard()
        return
    # Real-time frames start with 0x55; skip continuation fragments of longer frames
    if data[0] != 0x55:
        return
    try:
        data_dict = {}
        if len(data) >= 8: data_dict['spo2'] = int(data[7])
        if len(data) >= 9: data_dict['bpm'] = int(data[8])
        if len(data) >= 15: data_dict['battery'] = int(data[14])

        if data_dict and 40 <= data_dict.get('spo2', 0) <= 100:
            current_metrics["spo2"] = data_dict['spo2']
            current_metrics["hr"] = data_dict.get('bpm', current_metrics["hr"])
            current_metrics["battery"] = data_dict.get('battery', current_metrics["battery"])
            current_metrics["status"] = "Connected"
            print(f"🩸 LIVE BLE -> SpO2: {current_metrics['spo2']}% | HR: {current_metrics['hr']} BPM", flush=True)
            save_to_json_dashboard()
    except Exception: pass

async def log_timer_loop():
    while True:
        await asyncio.sleep(5)
        append_to_csv_log()

SEEN_TTL_S = 30  # keep a device listed this long after its last advertisement

async def scan_and_list_devices():
    """Scans CONTINUOUSLY until the UI picks a device.

    Wearables like the O2 Ultra advertise slowly or in bursts; short start/stop
    scan windows kept missing them. One long-lived scanner plus a last-seen TTL
    catches slow advertisers and keeps their card from flickering in and out.
    Returns once a connect command arrives.
    """
    global selected_target_mac
    current_metrics["status"] = "Scanning"
    save_to_json_dashboard()

    logging.info("Scanning for active Checkme wrist devices (continuous)...")
    seen = {}  # address -> (info, last_seen)

    def detection_callback(device: BLEDevice, adv_data: AdvertisementData):
        addr = device.address.upper()
        name = device.name or adv_data.local_name or ""
        uuids = [u.lower() for u in (adv_data.service_uuids or [])]

        # Identification: MAC matches, UUID matches, or text identifiers match
        is_match = (
            addr == KNOWN_MAC.upper() or
            HEALTH_SERVICE_UUID.lower() in uuids or
            any(x in name.upper() for x in ["O2", "CHECKME", "VIATOM", "BAND-WU"])
        )
        if not is_match:
            return

        ble_device_cache[device.address] = device
        prev = seen.get(device.address)
        info = {
            "name": name or (prev[0]["name"] if prev else f"Checkme Wrist Unit ({addr[-5:]})"),
            "address": device.address,
            "rssi": adv_data.rssi if adv_data.rssi else -100,
        }
        if not prev:
            logging.info(f"Spotted {info['name']} [{device.address}] at {info['rssi']} dBm")
        seen[device.address] = (info, time.time())

    last_count = -1
    try:
        async with BleakScanner(detection_callback, passive=False):
            while True:
                await asyncio.sleep(1.0)

                now = time.time()
                for a in [a for a, (_, t) in seen.items() if now - t > SEEN_TTL_S]:
                    logging.info(f"Lost {seen[a][0]['name']} [{a}] (no adverts for {SEEN_TTL_S}s)")
                    del seen[a]

                targets = sorted((i for i, _ in seen.values()), key=lambda x: x["rssi"], reverse=True)
                with open(DEVICES_FILE + ".tmp", "w") as f:
                    json.dump(targets, f)
                os.replace(DEVICES_FILE + ".tmp", DEVICES_FILE)

                if len(targets) != last_count:
                    logging.info(f"{len(targets)} matching wrist monitor(s) in range.")
                    last_count = len(targets)

                if os.path.exists(COMMAND_FILE):
                    try:
                        with open(COMMAND_FILE, 'r') as f:
                            cmd = json.load(f)
                        os.remove(COMMAND_FILE)
                        if cmd.get("action") == "connect":
                            selected_target_mac = cmd.get("address")
                            logging.info(f"UI Selection registered! Target locked: {selected_target_mac}")
                            return  # leaving the context stops the scan before we connect
                    except Exception:
                        pass
    except Exception as e:
        logging.error(f"Radar error: {e}")
        await asyncio.sleep(2)

async def run():
    global selected_target_mac
    log_task = asyncio.create_task(log_timer_loop())  # keep a reference so it isn't GC'd
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass

    while True:
        if not selected_target_mac:
            await scan_and_list_devices()
            continue
            
        logging.info(f"Connecting to user selected device: {selected_target_mac}")
        current_metrics["status"] = "Connecting"
        save_to_json_dashboard()
        
        try:
            device = ble_device_cache.get(selected_target_mac, selected_target_mac)
            async with BleakClient(device, timeout=10.0) as client:
                # Use a slightly softer service discovery check once connected
                services = client.services.get_service(HEALTH_SERVICE_UUID)
                
                # If the service cache lookup fails on this Linux build, extract characteristics directly
                chars = services.characteristics if services else []
                target_notify_uuid = next((c.uuid for c in chars if "notify" in c.properties), None)
                target_write_uuid = next((c.uuid for c in chars if "write" in c.properties or "write-without-response" in c.properties), WRITE_CHAR_UUID)
                if not target_notify_uuid:
                    # (the old fallback subscribed to the SERVICE uuid, which can never work)
                    raise RuntimeError("Viatom service/notify characteristic not found -- GATT cache stale?")

                await client.start_notify(target_notify_uuid, parse_checkme_notification)
                write_bytes = bytearray([0xAA, 0x17, 0xE8, 0x00, 0x00, 0x00, 0x00, 0x1B])
                
                use_response = True
                misses = 0
                while client.is_connected:
                    try:
                        await client.write_gatt_char(target_write_uuid, write_bytes, response=use_response)
                        misses = 0
                    except Exception:
                        # one failed poll shouldn't kill the link; flip write mode and retry
                        use_response = not use_response
                        misses += 1
                        if misses >= 3:
                            raise
                    await asyncio.sleep(2)
                    
                logging.warning("Connection dropped. Returning to radar scan.")
                selected_target_mac = None
                
        except Exception as e:
            logging.error(f"Link failed: {e}. Resetting target alignment.")
            selected_target_mac = None
            await asyncio.sleep(2)

if __name__ == "__main__":
    try: asyncio.run(run())
    except KeyboardInterrupt: logging.info("Clean shutdown.")
