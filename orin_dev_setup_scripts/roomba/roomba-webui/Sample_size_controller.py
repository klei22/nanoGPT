import cv2
import numpy as np
import threading
import time
import serial
import os
import random
import atexit
import sys
import signal
from datetime import datetime
from flask import Flask, Response, render_template_string, request, jsonify

# --- App Initialization & Globals ---
app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

current_frame = None
lock = threading.Lock()

shutdown_event = threading.Event()
camera_stop_event = threading.Event()
current_cap = None
camera_active = False
camera_thread = None

def check_opencv_gstreamer_support():
    build_info = cv2.getBuildInformation()
    if "GStreamer:" in build_info:
        lines = build_info.split('\n')
        for line in lines:
            if "GStreamer:" in line and "YES" in line:
                print("ENVIRONMENT CHECK: OpenCV variant with native GStreamer support verified.")
                return True
    print("CRITICAL ENVIRONMENT ERROR: Invalid OpenCV Variant Detected!")
    return False

if not check_opencv_gstreamer_support():
    sys.exit("Fatal: Invalid OpenCV binary variant. Please fix dependencies.")

def hardware_wake_roomba_via_cp2102():
    global roomba
    print("CP2102: Injecting physical hardware wake-up pulse via BRC/RTS pin line...")
    try:
        if 'roomba' in globals() and roomba is not None:
            try:
                roomba.close()
            except:
                pass
        ser = serial.Serial('/dev/ttyUSB0', baudrate=115200)
        ser.setRTS(True)
        time.sleep(0.6)
        ser.setRTS(False)
        time.sleep(0.5)
        ser.close()
        roomba = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0.1)
        roomba.setRTS(False)
        time.sleep(0.05)
        roomba.write(bytes([128, 131]))
        print("CP2102: Primary communication control handle bound and Safe Mode active.")
    except Exception as e:
        print(f"CP2102 Wake Error: {e}")

hardware_wake_roomba_via_cp2102()

PAN_TILT_PORT = "/dev/ttyACM0"
PAN_TILT_BAUD = 9600

pan_tilt_serial = None
current_pan_angle = 90
current_tilt_angle = 90
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 30, 150
PAN_TILT_STEP = 3  # small steps so held nudging reads as smooth motion, not snapping

def connect_pan_tilt():
    global pan_tilt_serial
    try:
        pan_tilt_serial = serial.Serial(PAN_TILT_PORT, PAN_TILT_BAUD, timeout=1)
        time.sleep(2)
        while pan_tilt_serial.in_waiting:
            pan_tilt_serial.readline()
        print("Pan/Tilt: Arduino connected successfully.")
    except Exception as e:
        pan_tilt_serial = None
        print(f"Pan/Tilt: not connected: {e}. Running in simulation mode.")

def _send_pan_tilt(cmd):
    global pan_tilt_serial
    if not pan_tilt_serial:
        return
    try:
        pan_tilt_serial.write((cmd + "\n").encode())
        time.sleep(0.02)
        while pan_tilt_serial.in_waiting:
            pan_tilt_serial.readline()
    except Exception as e:
        print(f"Pan/Tilt: lost connection mid-command ({e}). Attempting reconnect...")
        try:
            pan_tilt_serial.close()
        except:
            pass
        pan_tilt_serial = None
        threading.Thread(target=connect_pan_tilt, daemon=True).start()

def set_pan(angle):
    global current_pan_angle
    current_pan_angle = max(PAN_MIN, min(PAN_MAX, angle))
    _send_pan_tilt(f"P{current_pan_angle}")

def set_tilt(angle):
    global current_tilt_angle
    current_tilt_angle = max(TILT_MIN, min(TILT_MAX, angle))
    _send_pan_tilt(f"T{current_tilt_angle}")

def center_pan_tilt():
    set_pan(90)
    set_tilt(90)

connect_pan_tilt()

SAMPLE_SIZE = 50

ROAM_SPEED = 200
ROAM_WANDER_MIN_SEC = 4.0
ROAM_WANDER_MAX_SEC = 9.0

current_speed = 150
vacuum_on = False
last_keys_set = set()
auto_mode = False
roam_mode = False
roam_next_wander_time = 0.0
show_grayscale = False
current_action_label = ""
active_key_string = "idling"

frame_count = 0

last_heartbeat_time = time.time()

operational_mode = "manual"
movement_history = []
current_move_start = None
last_saved_keys = set()
playback_index = 0

total_distance = 0.0
current_heading = 0.0
last_fetched_angle = 0

live_left_speed_mms = 0.0
live_right_speed_mms = 0.0

live_speed_mms = 0.0
battery_pct = 100.0

output_dir = ""
csv_path = ""
csv_file = None

if 'roomba' not in globals() or roomba is None:
    try:
        roomba = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=0.1)
        roomba.setRTS(False)
        time.sleep(0.05)
        roomba.write(bytes([128, 131]))
        print("Roomba serial connected successfully.")
    except Exception as e:
        roomba = None
        print(f"Roomba not connected: {e}. Running in simulation mode.")

def drive_roomba(velocity, radius):
    if roomba:
        v_high, v_low = (velocity >> 8) & 0xFF, velocity & 0xFF
        r_high, r_low = (radius >> 8) & 0xFF, radius & 0xFF
        try:
            roomba.write(bytes([137, v_high, v_low, r_high, r_low]))
        except:
            pass

def set_vacuum(state):
    if roomba:
        val = 7 if state else 0
        try:
            roomba.write(bytes([138, val]))
        except:
            pass

def dock_roomba():
    if roomba:
        set_vacuum(False)
        try:
            roomba.write(bytes([143]))
        except:
            pass
        print("Roomba sent to Seek Dock (Command 143).")

def cancel_docking_sequence():
    if roomba:
        try:
            roomba.write(bytes([128, 131]))
            time.sleep(0.02)
            drive_roomba(0, 0)
        except:
            pass
        print("Control Restored: Passive Docking aborted.")

def keep_roomba_alive_worker():
    global last_heartbeat_time
    while True:
        try:
            if not auto_mode and current_action_label == "" and roomba and roomba.is_open:
                now = time.time()
                if now - last_heartbeat_time > 60.0:
                    with lock:
                        roomba.write(bytes([128, 131]))
                    last_heartbeat_time = now
        except:
            pass
        time.sleep(5)

def update_sensors_efficiently():
    global total_distance, current_heading, live_speed_mms, battery_pct, last_fetched_angle
    global live_left_speed_mms, live_right_speed_mms
    bump_left, bump_right, wheel_drop = 0, 0, 0
    if not roomba or not roomba.is_open: return 0, 0, 0
    try:
        roomba.flushInput()
        roomba.write(bytes([142, 0]))
        time.sleep(0.03)
        if roomba.inWaiting() >= 26:
            data = roomba.read(26)
            byte0 = data[0]
            bump_right = 1 if (byte0 & 0x01) else 0
            bump_left = 1 if (byte0 & 0x02) else 0
            wheel_drop = 1 if (byte0 & 0x04 or byte0 & 0x08) else 0
            delta_dist = int.from_bytes(data[12:14], byteorder='big', signed=True)
            total_distance += delta_dist
            last_fetched_angle = int.from_bytes(data[14:16], byteorder='big', signed=True)
            current_heading = (current_heading + last_fetched_angle) % 360
            charge = int.from_bytes(data[22:24], byteorder='big', signed=False)
            capacity = int.from_bytes(data[24:26], byteorder='big', signed=False)
            if capacity > 0:
                battery_pct = max(0.0, min(100.0, (charge / capacity) * 100.0))
            live_speed_mms = delta_dist / 0.055
        roomba.write(bytes([149, 2, 43, 44]))
        time.sleep(0.015)
        if roomba.inWaiting() >= 4:
            wheel_data = roomba.read(4)
            live_left_speed_mms = int.from_bytes(wheel_data[0:2], byteorder='big', signed=True)
            live_right_speed_mms = int.from_bytes(wheel_data[2:4], byteorder='big', signed=True)
    except:
        pass
    return bump_left, bump_right, wheel_drop

def draw_guidelines(img):
    h, w = img.shape[:2]
    left_start, left_end = (int(w * 0.10), h), (int(w * 0.38), int(h * 0.52))
    right_start, right_end = (int(w * 0.90), h), (int(w * 0.62), int(h * 0.52))
    cv2.line(img, left_start, left_end, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, right_start, right_end, (0, 255, 0), 2, cv2.LINE_AA)
    return img

csv_lock = threading.Lock()

def csv_header_line():
    total_pixels = SAMPLE_SIZE * SAMPLE_SIZE
    pixel_headers = ",".join([f"p{i}" for i in range(total_pixels)])
    return f"timestamp,frame_id,action,pressed_keys,bump_left,bump_right,wheel_dropped,total_distance_mm,heading_deg,speed_mm_s,left_wheel_v,right_wheel_v,battery_percent,{pixel_headers}\n"

def open_dataset_csv():
    documents_dir = os.path.expanduser("~/Documents")
    output_dir = os.path.join(documents_dir, "roomba_dataset")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"dataset_{SAMPLE_SIZE}x{SAMPLE_SIZE}.csv")
    is_new_or_empty = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    f = open(csv_path, 'a')
    if is_new_or_empty:
        f.write(csv_header_line())
        f.flush()
    return output_dir, csv_path, f

def start_new_csv_session():
    global csv_file, csv_path, output_dir, frame_count
    with csv_lock:
        if csv_file:
            try:
                csv_file.flush()
                csv_file.close()
            except Exception as e:
                print(f"Error closing CSV during rotation: {e}")
        archived_path = None
        if csv_path and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            base, ext = os.path.splitext(csv_path)
            archived_path = f"{base}_{timestamp}{ext}"
            counter = 2
            while os.path.exists(archived_path):
                archived_path = f"{base}_{timestamp}_{counter}{ext}"
                counter += 1
            os.rename(csv_path, archived_path)
        output_dir, csv_path, csv_file = open_dataset_csv()
        frame_count = 0
        print(f"New CSV session started at {csv_path}" + (f" (previous data archived to {archived_path})" if archived_path else ""))
        return archived_path, csv_path

def execution_playback_worker():
    global operational_mode, current_action_label, playback_index, active_key_string
    while playback_index < len(movement_history):
        if operational_mode != "playback":
            return
        saved_keys, duration, saved_speed = movement_history[playback_index]
        active_key_string = saved_keys[0] if saved_keys else "idling"
        if 'idling' in saved_keys or not saved_keys:
            drive_roomba(0, 0); current_action_label = ""
        elif 'space' in saved_keys:
            drive_roomba(0, 0); current_action_label = "stop"
        elif 'w' in saved_keys:
            drive_roomba(saved_speed, 32767); current_action_label = "forward"
        elif 's' in saved_keys:
            drive_roomba(-saved_speed, 32767); current_action_label = "backward"
        elif 'a' in saved_keys:
            drive_roomba(saved_speed, 1); current_action_label = "left"
        elif 'd' in saved_keys:
            drive_roomba(saved_speed, -1); current_action_label = "right"
        else:
            drive_roomba(0, 0); current_action_label = "stop"
        start_wait = time.time()
        while time.time() - start_wait < duration:
            if operational_mode != "playback":
                drive_roomba(0, 0)
                return
            time.sleep(0.02)
        playback_index += 1
    drive_roomba(0, 0)
    current_action_label = ""
    active_key_string = "idling"
    operational_mode = "manual"
    playback_index = 0

def execute_closed_loop_turn(target_degrees, direction_sign):
    global current_action_label, last_fetched_angle, active_key_string
    print(f"ENCODER CONTROL: Target turn initialized for {target_degrees} deg")
    turn_speed = 180
    accumulated_turn = 0.0
    active_key_string = "a" if direction_sign == 1 else "d"
    drive_roomba(turn_speed, direction_sign)
    current_action_label = f"encoder_turning_{target_degrees}"
    while abs(accumulated_turn) < (target_degrees - 4):
        update_sensors_efficiently()
        accumulated_turn += last_fetched_angle
        time.sleep(0.015)
    drive_roomba(0, 0)
    current_action_label = ""
    active_key_string = "idling"

def handle_roam_mode(b_left, b_right, now):
    global active_key_string, current_action_label, roam_next_wander_time
    if b_left == 1 or b_right == 1:
        active_key_string = "s"
        current_action_label = "roam_bump_backup"
        drive_roomba(-120, 32767)
        time.sleep(0.4)
        if b_left == 1 and b_right == 1:
            turn_dir = random.choice([1, -1])
            turn_deg = 140 + random.randint(-20, 20)
        elif b_left == 1:
            turn_dir = -1
            turn_deg = 45 + random.randint(0, 70)
        else:
            turn_dir = 1
            turn_deg = 45 + random.randint(0, 70)
        current_action_label = "roam_bump_turn"
        execute_closed_loop_turn(turn_deg, turn_dir)
        roam_next_wander_time = now + random.uniform(ROAM_WANDER_MIN_SEC, ROAM_WANDER_MAX_SEC)
        return
    if now >= roam_next_wander_time:
        current_action_label = "roam_wander_turn"
        turn_dir = random.choice([1, -1])
        turn_deg = random.randint(20, 90)
        execute_closed_loop_turn(turn_deg, turn_dir)
        roam_next_wander_time = now + random.uniform(ROAM_WANDER_MIN_SEC, ROAM_WANDER_MAX_SEC)
        return
    active_key_string = "w"
    current_action_label = "roam_forward"
    drive_roomba(ROAM_SPEED, 32767)

def camera_and_logic_loop():
    global current_frame, auto_mode, roam_mode, show_grayscale, current_action_label, csv_file, SAMPLE_SIZE, current_speed, active_key_string, frame_count
    global output_dir, csv_path, current_cap, camera_active

    pipeline = (
        f"nvarguscamerasrc sensor-id=0 "
        f"exposuretimerange='34000000 34000000' "
        f"gainrange='1 8' "
        f"wbmode=1 ! "
        f"video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width=640, height=360, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Camera failed to open - check the CSI ribbon cable and nvargus-daemon.")
        camera_active = False
        return
    current_cap = cap

    output_dir, csv_path, csv_file = open_dataset_csv()

    last_steering_error = 0.0
    last_time = time.time()

    blue_lower = np.array([95, 130, 20])
    blue_upper = np.array([125, 255, 255])

    wheel_drop_started = None
    hard_safety_triggered = False

    while True:
        if shutdown_event.is_set() or camera_stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret: break

        b_left, b_right, w_drop = update_sensors_efficiently()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(gray, (SAMPLE_SIZE, SAMPLE_SIZE), interpolation=cv2.INTER_AREA)
        pixels = small_frame.flatten()
        pixel_str = ",".join(map(str, pixels))

        synced_time = time.time()

        with csv_lock:
            csv_file.write(f"{synced_time},{frame_count},{current_action_label},{active_key_string},{b_left},{b_right},{w_drop},{total_distance:.1f},{current_heading:.1f},{live_speed_mms:.1f},{live_left_speed_mms},{live_right_speed_mms},{battery_pct:.1f},{pixel_str}\n")
        frame_count += 1

        if w_drop == 1:
            if wheel_drop_started is None: wheel_drop_started = synced_time
            if (synced_time - wheel_drop_started) >= 5.0: hard_safety_triggered = True
        else:
            wheel_drop_started = None
            hard_safety_triggered = False

        if hard_safety_triggered:
            drive_roomba(0, 0)
            current_action_label = ""
            active_key_string = "idling"
            auto_mode = False
            roam_mode = False

        h, w = frame.shape[:2]
        current_time = time.time()

        roi_y_start = 160
        roi_y_end = 360
        roi_x_start = 220
        roi_x_end = 420

        roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        img_yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized_roi = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        hsv_roi = cv2.cvtColor(equalized_roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, blue_lower, blue_upper)

        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tape_found = False
        cx = (roi_x_end - roi_x_start) // 2

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 250:
                tape_found = True
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cv2.drawContours(frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end], [largest_contour], -1, (0, 255, 0), 3)
                    cv2.circle(frame, (roi_x_start + cx, roi_y_start + int((roi_y_end - roi_y_start)/2)), 6, (255, 0, 0), -1)
            cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 120, 0), 2)

        if auto_mode:
            if b_left == 1 or b_right == 1:
                active_key_string = "s"
                drive_roomba(-120, 32767)
                time.sleep(0.4)
                execute_closed_loop_turn(180, random.choice([1, -1]))
            elif tape_found:
                dt = current_time - last_time
                if dt <= 0: dt = 0.033
                global_cx = roi_x_start + cx
                screen_center_x = w / 2
                steering_error = global_cx - screen_center_x
                error_derivative = (steering_error - last_steering_error) / dt
                Kp, Kd = 2.4, 0.16
                pd_correction = (steering_error * Kp) + (error_derivative * Kd)
                if abs(pd_correction) < 5:
                    oi_radius = 32767
                    active_key_string = "w"
                else:
                    calculated_radius = int(25000 / abs(pd_correction))
                    calculated_radius = max(150, min(2000, calculated_radius))
                    if pd_correction > 0:
                        oi_radius = -calculated_radius
                        active_key_string = "d"
                    else:
                        oi_radius = calculated_radius
                        active_key_string = "a"
                if abs(steering_error) > 40:
                    drive_roomba(110, oi_radius)
                else:
                    drive_roomba(current_speed, oi_radius)
                current_action_label = "line_following"
                last_steering_error = steering_error
                last_time = current_time
            else:
                current_action_label = "searching_tape"
                active_key_string = "s"
                drive_roomba(-100, 32767)
                time.sleep(0.4)
                execute_closed_loop_turn(180, 1)
                scan_directions = [-1, 1, -1]
                scan_durations = [1.2, 2.4, 1.2]
                tape_locked = False
                for sweep_dir, sweep_time in zip(scan_directions, scan_durations):
                    if tape_found:
                        tape_locked = True
                        break
                    active_key_string = "d" if sweep_dir == -1 else "a"
                    drive_roomba(110, sweep_dir)
                    start_sweep = time.time()
                    while time.time() - start_sweep < sweep_time:
                        if tape_found:
                            tape_locked = True
                            break
                        time.sleep(0.02)
                if tape_locked:
                    drive_roomba(0, 0)
                    active_key_string = "idling"
                    last_steering_error = 0.0
                    last_time = time.time()
                else:
                    drive_roomba(0, 0)
                    active_key_string = "idling"
                    auto_mode = False
                    current_action_label = ""
        elif roam_mode:
            handle_roam_mode(b_left, b_right, current_time)
            last_steering_error = 0.0
            last_time = current_time
        else:
            if current_action_label == "sticky_spin_left":
                drive_roomba(150, 1)
                active_key_string = "a"
            elif current_action_label == "sticky_spin_right":
                drive_roomba(150, -1)
                active_key_string = "d"
            elif current_action_label == "sticky_spin_random":
                if active_key_string not in ["a", "d"]:
                    active_key_string = random.choice(["a", "d"])
                chosen_radius = 1 if active_key_string == "a" else -1
                drive_roomba(150, chosen_radius)
            last_steering_error = 0.0
            last_time = current_time

        with lock:
            if show_grayscale:
                upscaled_ai_preview = cv2.resize(small_frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                current_frame = cv2.cvtColor(upscaled_ai_preview, cv2.COLOR_GRAY2BGR)
            else:
                hud_frame = frame.copy()
                current_frame = draw_guidelines(hud_frame)

    try:
        cap.release()
        print("Camera released cleanly.")
    except Exception as e:
        print(f"Error releasing camera: {e}")
    current_cap = None
    camera_active = False

    with csv_lock:
        if csv_file:
            try:
                csv_file.flush()
                csv_file.close()
                print(f"Dataset CSV saved to {csv_path}")
            except Exception as e:
                print(f"Error closing CSV: {e}")
            csv_file = None

@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>Roomba Command Deck</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
            <style>
                :root {
                    --bg-primary: #0c0d10;
                    --bg-surface: #15171c;
                    --bg-recessed: #0a0b0d;
                    --bg-glass: rgba(21, 23, 28, 0.82);
                    --line: #262a33;
                    --accent-cyan: #00e5c7;
                    --accent-blue: #4d8bff;
                    --accent-orange: #ff9d3d;
                    --accent-red: #ff5c5c;
                    --accent-green: #4ee08a;
                    --accent-violet: #b58aff;
                    --text-main: #eef0f3;
                    --text-muted: #7d8492;
                }
                * { box-sizing: border-box; }
                body { background: var(--bg-primary); color: var(--text-main); font-family: 'SF Mono', 'JetBrains Mono', ui-monospace, monospace; margin: 0; padding: 18px; user-select: none; }
                .deck { display: flex; flex-direction: column; align-items: center; max-width: 1000px; margin: 0 auto; gap: 14px; }
                .deck-header { width: 100%; max-width: 900px; display: flex; align-items: baseline; justify-content: space-between; padding: 2px 4px 6px; border-bottom: 1px solid var(--line); }
                .deck-header h1 { font-size: 15px; letter-spacing: 0.5px; margin: 0; color: var(--text-main); font-weight: 600; }
                .deck-header .deck-sub { font-size: 11px; color: var(--text-muted); }
                .viewport-row { display: flex; flex-direction: row; gap: 14px; width: 900px; max-width: 100%; align-items: stretch; justify-content: center; }
                @media(max-width: 920px) { .viewport-row { flex-direction: column; width: 640px; } }
                .video-hud-container { position: relative; width: 640px; aspect-ratio: 16/9; background: #000; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
                .video-hud-container img { width: 100%; height: 100%; display: block; object-fit: cover; }
                .side-column { width: 240px; display: flex; flex-direction: column; gap: 10px; }
                .logger-panel-box { background: var(--bg-surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; gap: 8px; box-sizing: border-box; flex: 1; }
                .log-data-card { background: var(--bg-recessed); border: 1px solid var(--line); border-radius: 7px; padding: 8px 10px; display: flex; flex-direction: column; gap: 3px; text-align: left; }
                .log-label { font-size: 9.5px; color: var(--text-muted); font-weight: 600; letter-spacing: 0.3px; }
                .log-value-keys { color: #ffd166; font-size: 14px; font-weight: 700; text-align: center; padding: 3px 0; }
                .log-value-action { color: var(--accent-cyan); font-size: 12px; font-weight: 600; overflow-wrap: break-word; word-break: break-all; min-height: 16px; }
                .log-value-frames { color: var(--accent-violet); font-size: 14px; font-weight: 700; text-align: center; padding: 1px 0; }
                .panel-title-sm { font-size: 10px; color: var(--text-muted); font-weight: 600; letter-spacing: 0.3px; border-bottom: 1px solid var(--line); padding-bottom: 5px; }
                .hud-telemetry-box { position: absolute; top: 10px; left: 10px; background: var(--bg-glass); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; text-align: left; font-size: 11px; backdrop-filter: blur(6px); pointer-events: none; z-index: 10; display: grid; gap: 4px; min-width: 152px; }
                .hud-telemetry-box div.row { display: flex; justify-content: space-between; gap: 10px; }
                .hud-status-text { color: var(--accent-cyan); font-weight: 700; }
                .hud-battery-text { color: var(--accent-orange); font-weight: 700; }
                .hud-speed-text { color: var(--accent-blue); }
                .hud-vac-off { color: var(--accent-red); font-weight: 700; }
                .hud-vac-on { color: var(--accent-green); font-weight: 700; }
                .speed-bar-track { height: 4px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; margin-top: 1px; }
                .speed-bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); transition: width 0.15s ease; }
                .dpad-overlay { position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%); display: grid; grid-template-columns: repeat(3, 46px); grid-template-rows: repeat(3, 42px); gap: 6px; pointer-events: none; z-index: 5; transition: opacity 0.2s ease; opacity: 0; }
                .dpad-visible { opacity: 1 !important; pointer-events: auto !important; }
                .touch-btn { background: rgba(12,13,16,0.85); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); border-radius: 6px; font-size: 15px; display: flex; align-items: center; justify-content: center; cursor: pointer; pointer-events: auto; backdrop-filter: blur(4px); }
                .touch-btn:active { background: var(--accent-cyan); color: #000; }
                .center-space { grid-column: 2; grid-row: 2; font-size: 8.5px; font-weight: 700; border-color: var(--text-muted); color: var(--text-muted); }
                .tab-shell { width: 900px; max-width: 100%; background: var(--bg-surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
                .tab-bar { display: grid; grid-template-columns: repeat(4, 1fr); border-bottom: 1px solid var(--line); }
                .tab-btn { background: transparent; border: none; color: var(--text-muted); font-family: inherit; font-size: 11px; font-weight: 600; letter-spacing: 0.2px; padding: 12px 8px; cursor: pointer; border-bottom: 2px solid transparent; transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease; }
                .tab-btn:hover { color: var(--text-main); background: rgba(255,255,255,0.02); }
                .tab-btn.tab-active { color: var(--accent-cyan); border-bottom-color: var(--accent-cyan); background: rgba(0,229,199,0.06); }
                .tab-panel { display: none; padding: 16px; }
                .tab-panel.tab-panel-active { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
                @media(max-width: 920px) { .tab-panel.tab-panel-active { grid-template-columns: 1fr; } }
                .cluster { display: flex; flex-direction: column; gap: 8px; }
                .cluster-title { font-size: 10px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; font-weight: 700; margin-bottom: 2px; }
                .cluster-note { font-size: 10px; color: var(--text-muted); margin: -4px 0 2px; line-height: 1.4; }
                .btn-group { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
                .btn-group-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
                .btn-group-full { display: flex; flex-direction: column; gap: 8px; }
                .dashboard-btn { background: #1b1e25; color: var(--text-main); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; font-size: 11px; font-weight: 600; font-family: inherit; cursor: pointer; transition: background 0.15s ease, border-color 0.15s ease; display: flex; align-items: center; justify-content: center; gap: 6px; min-height: 36px; box-sizing: border-box; text-decoration: none; }
                .dashboard-btn:hover { background: #23262f; border-color: var(--accent-blue); }
                .dashboard-btn:active { transform: scale(0.98); }
                .btn-cyan { border-color: var(--accent-cyan); color: var(--accent-cyan); }
                .btn-orange { border-color: var(--accent-orange); color: var(--accent-orange); }
                .btn-red { border-color: var(--accent-red); color: var(--accent-red); }
                .btn-green { border-color: var(--accent-green); color: var(--accent-green); }
                .active-tracking { background: rgba(0,229,199,0.12) !important; border-color: var(--accent-cyan) !important; color: var(--accent-cyan) !important; }
                .active-recording { background: rgba(255,92,92,0.12) !important; border-color: var(--accent-red) !important; color: var(--accent-red) !important; animation: pulse-border 1.5s infinite; }
                @keyframes pulse-border { 0% { box-shadow: 0 0 0 0 rgba(255,92,92,0.35); } 70% { box-shadow: 0 0 0 5px rgba(255,92,92,0); } 100% { box-shadow: 0 0 0 0 rgba(255,92,92,0); } }
                .angle-readout { display: flex; justify-content: space-around; font-size: 11px; color: var(--text-muted); margin-bottom: 2px; }
                .angle-readout span.val { color: var(--accent-cyan); font-weight: 700; }
                .instructions { color: var(--text-muted); font-size: 10.5px; line-height: 1.6; text-align: center; max-width: 900px; background: var(--bg-surface); border: 1px solid var(--line); padding: 10px; border-radius: 8px; width: 100%; box-sizing: border-box; }
                .key { display: inline-block; background: #22252d; padding: 2px 6px; border-radius: 4px; margin: 0 2px; color: #fff; font-weight: 700; border: 1px solid var(--line); }
            </style>
            <script>
                let keys = new Set();
                const validKeys = ['w','s','a','d','q','e','v','space'];

                document.addEventListener('keydown', (e) => {
                    let key = e.key.toLowerCase();
                    if (key === ' ') key = 'space';
                    if (validKeys.includes(key) && !keys.has(key)) { keys.add(key); sendState(); }
                });

                document.addEventListener('keyup', (e) => {
                    let key = e.key.toLowerCase();
                    if (key === ' ') key = 'space';
                    if (keys.has(key)) { keys.delete(key); sendState(); }
                });

                function sendState() { fetch('/keyboard_input?keys=' + Array.from(keys).join('-')); }

                const arrowKeyMap = {
                    'arrowleft':  ['pan', 'left'],
                    'arrowright': ['pan', 'right'],
                    'arrowup':    ['tilt', 'up'],
                    'arrowdown':  ['tilt', 'down']
                };
                const arrowIntervals = {};

                document.addEventListener('keydown', (e) => {
                    let key = e.key.toLowerCase();
                    if (arrowKeyMap[key]) {
                        e.preventDefault();
                        if (!arrowIntervals[key]) {
                            const axisDir = arrowKeyMap[key];
                            nudgePanTilt(axisDir[0], axisDir[1]);
                            arrowIntervals[key] = setInterval(() => nudgePanTilt(axisDir[0], axisDir[1]), 40);
                        }
                    }
                });

                document.addEventListener('keyup', (e) => {
                    let key = e.key.toLowerCase();
                    if (arrowKeyMap[key] && arrowIntervals[key]) {
                        clearInterval(arrowIntervals[key]);
                        delete arrowIntervals[key];
                    }
                });

                let gamepadIndex = null;
                let gamepadPanTiltInterval = null;
                let lastGamepadMoveKeys = new Set();
                let lastGamepadButtons = {};
                const GAMEPAD_DEADZONE = 0.25;

                window.addEventListener("gamepadconnected", (e) => {
                    gamepadIndex = e.gamepad.index;
                    const statusEl = document.getElementById("hud-gamepad-status");
                    if (statusEl) statusEl.innerText = "CONNECTED";
                    requestAnimationFrame(pollGamepad);
                });

                window.addEventListener("gamepaddisconnected", (e) => {
                    if (gamepadIndex === e.gamepad.index) {
                        gamepadIndex = null;
                        ['w', 'a', 's', 'd'].forEach(k => keys.delete(k));
                        sendState();
                        const statusEl = document.getElementById("hud-gamepad-status");
                        if (statusEl) statusEl.innerText = "NOT CONNECTED";
                    }
                });

                function handleGamepadButton(gp, index, onPress) {
                    const pressed = !!(gp.buttons[index] && gp.buttons[index].pressed);
                    const wasPressed = lastGamepadButtons[index] || false;
                    if (pressed && !wasPressed) onPress();
                    lastGamepadButtons[index] = pressed;
                }

                function pollGamepad() {
                    if (gamepadIndex === null) return;
                    const gp = navigator.getGamepads()[gamepadIndex];
                    if (!gp) { requestAnimationFrame(pollGamepad); return; }

                    const lx = gp.axes[0] || 0;
                    const ly = gp.axes[1] || 0;
                    let moveKeys = new Set();
                    if (ly < -GAMEPAD_DEADZONE) moveKeys.add('w');
                    else if (ly > GAMEPAD_DEADZONE) moveKeys.add('s');
                    if (lx < -GAMEPAD_DEADZONE) moveKeys.add('a');
                    else if (lx > GAMEPAD_DEADZONE) moveKeys.add('d');
                    if (gp.buttons[12] && gp.buttons[12].pressed) moveKeys.add('w');
                    if (gp.buttons[13] && gp.buttons[13].pressed) moveKeys.add('s');
                    if (gp.buttons[14] && gp.buttons[14].pressed) moveKeys.add('a');
                    if (gp.buttons[15] && gp.buttons[15].pressed) moveKeys.add('d');

                    const changed = moveKeys.size !== lastGamepadMoveKeys.size ||
                        [...moveKeys].some(k => !lastGamepadMoveKeys.has(k));
                    if (changed) {
                        keys = new Set([...keys].filter(k => !['w', 'a', 's', 'd'].includes(k)));
                        moveKeys.forEach(k => keys.add(k));
                        sendState();
                        lastGamepadMoveKeys = moveKeys;
                    }

                    const rx = gp.axes[2] || 0;
                    const ry = gp.axes[3] || 0;
                    if (Math.abs(rx) > GAMEPAD_DEADZONE || Math.abs(ry) > GAMEPAD_DEADZONE) {
                        if (!gamepadPanTiltInterval) {
                            gamepadPanTiltInterval = setInterval(() => {
                                const gp2 = navigator.getGamepads()[gamepadIndex];
                                if (!gp2) return;
                                const rx2 = gp2.axes[2] || 0;
                                const ry2 = gp2.axes[3] || 0;
                                if (Math.abs(rx2) > GAMEPAD_DEADZONE) nudgePanTilt('pan', rx2 > 0 ? 'right' : 'left');
                                if (Math.abs(ry2) > GAMEPAD_DEADZONE) nudgePanTilt('tilt', ry2 > 0 ? 'down' : 'up');
                            }, 40);
                        }
                    } else if (gamepadPanTiltInterval) {
                        clearInterval(gamepadPanTiltInterval);
                        gamepadPanTiltInterval = null;
                    }

                    handleGamepadButton(gp, 0, () => sendSingleAction('v'));
                    handleGamepadButton(gp, 1, () => sendSingleAction('space'));
                    handleGamepadButton(gp, 3, () => centerPanTilt());
                    handleGamepadButton(gp, 9, () => toggleAutoMode());
                    handleGamepadButton(gp, 4, () => sendSingleAction('q'));
                    handleGamepadButton(gp, 5, () => sendSingleAction('e'));

                    requestAnimationFrame(pollGamepad);
                }

                function showTab(name) {
                    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('tab-panel-active'));
                    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-active'));
                    document.getElementById('tab-panel-' + name).classList.add('tab-panel-active');
                    document.getElementById('tab-btn-' + name).classList.add('tab-active');
                }

                function toggleCamera() {
                    let btn = document.getElementById("hud-camera-btn");
                    btn.innerText = "Camera & Logging: ...";
                    fetch('/toggle_camera').then(res => res.json()).then(data => {
                        if (data.camera_active) {
                            btn.classList.add("active-tracking");
                            btn.innerText = "Camera & Logging: ON";
                        } else {
                            btn.classList.remove("active-tracking");
                            btn.innerText = "Camera & Logging: OFF";
                        }
                    });
                }

                function startNewCsv() {
                    let btn = document.getElementById("hud-newcsv-btn");
                    let original = btn.innerText;
                    btn.innerText = "Starting new CSV...";
                    fetch('/new_csv').then(res => res.json()).then(data => {
                        btn.innerText = "New CSV started";
                        setTimeout(() => { btn.innerText = original; }, 2000);
                    }).catch(() => {
                        btn.innerText = "Failed - check logs";
                        setTimeout(() => { btn.innerText = original; }, 2000);
                    });
                }

                function toggleAutoMode() {
                    fetch('/toggle_auto').then(res => res.text()).then(txt => {
                        let autoBtn = document.getElementById("hud-auto-btn");
                        let roamBtn = document.getElementById("hud-roam-btn");
                        if (txt.includes("ACTIVE")) { autoBtn.classList.add("active-tracking"); roamBtn.classList.remove("active-tracking"); }
                        else { autoBtn.classList.remove("active-tracking"); }
                    });
                }

                function toggleRoamMode() {
                    fetch('/toggle_roam').then(res => res.text()).then(txt => {
                        let roamBtn = document.getElementById("hud-roam-btn");
                        let autoBtn = document.getElementById("hud-auto-btn");
                        if (txt.includes("ACTIVE")) { roamBtn.classList.add("active-tracking"); autoBtn.classList.remove("active-tracking"); }
                        else { roamBtn.classList.remove("active-tracking"); }
                    });
                }

                function toggleGrayscaleView() {
                    fetch('/toggle_grayscale').then(res => res.json()).then(data => {
                        let btn = document.getElementById("hud-gray-btn");
                        if (data.grayscale) { btn.classList.add("active-tracking"); btn.innerText = "Standard HUD View"; }
                        else { btn.classList.remove("active-tracking"); btn.innerText = "AI Matrix View (Grayscale)"; }
                    });
                }

                function advanceMacroState() {
                    fetch('/advance_macro_mode').then(res => res.json()).then(data => {
                        let btn = document.getElementById("hud-macro-btn");
                        let pBtn = document.getElementById("hud-pause-btn");
                        document.getElementById("hud-mode-status").innerText = data.label.toUpperCase() + " MODE";
                        btn.className = "dashboard-btn " + data.css_class;
                        btn.innerText = data.button_text;
                        pBtn.style.display = data.show_pause ? "inline-block" : "none";
                        if(data.pause_text) pBtn.innerText = data.pause_text;
                    });
                }

                document.addEventListener('fullscreenchange', () => {
                    let fsBtn = document.getElementById("hud-fs-btn");
                    if (!fsBtn) return;
                    if (document.fullscreenElement) { fsBtn.innerText = "Minimize HUD"; }
                    else { fsBtn.innerText = "Full Screen HUD"; }
                });

                function togglePlayPause() {
                    fetch('/toggle_play_pause').then(res => res.json()).then(data => {
                        document.getElementById("hud-mode-status").innerText = data.label.toUpperCase() + " MODE";
                        document.getElementById("hud-pause-btn").innerText = data.pause_text;
                        document.getElementById("hud-macro-btn").className = "dashboard-btn " + data.css_class;
                    });
                }

                function toggleDpadVisibility() {
                    let dpad = document.getElementById("touch-dpad");
                    let btn = document.getElementById("hud-dpad-toggle");
                    if (dpad.classList.contains("dpad-visible")) {
                        dpad.classList.remove("dpad-visible");
                        btn.classList.remove("btn-cyan");
                    } else {
                        dpad.classList.add("dpad-visible");
                        btn.classList.add("btn-cyan");
                    }
                }

                function toggleFullScreen() {
                    let container = document.getElementById("hud-root");
                    if (!document.fullscreenElement) { container.requestFullscreen(); }
                    else { document.exitFullscreen(); }
                }

                function triggerHardwareWakePulse() {
                    let statusNode = document.getElementById("hud-mode-status");
                    statusNode.innerText = "WAKING...";
                    statusNode.style.color = "var(--accent-orange)";
                    fetch('/trigger_hardware_wake').then(res => res.text()).then(txt => {
                        statusNode.innerText = "MANUAL";
                        statusNode.style.color = "var(--accent-cyan)";
                    });
                }

                document.addEventListener('contextmenu', event => event.preventDefault());

                function triggerFullScreenMode(actionType) { fetch('/trigger_sticky_spin?mode=' + actionType); }

                function triggerDocking() {
                    fetch('/dock').then(res => res.text()).then(txt => {
                        document.getElementById("hud-mode-status").innerText = txt;
                        document.getElementById("hud-auto-btn").classList.remove("active-tracking");
                        document.getElementById("hud-dock-btn").style.display = "none";
                        document.getElementById("hud-cancel-dock-btn").style.display = "inline-block";
                    });
                }

                function triggerCancelDocking() {
                    fetch('/cancel_dock').then(res => res.text()).then(txt => {
                        document.getElementById("hud-mode-status").innerText = txt;
                        document.getElementById("hud-dock-btn").style.display = "inline-block";
                        document.getElementById("hud-cancel-dock-btn").style.display = "none";
                    });
                }

                function triggerFixedDegreeTurn(angle, dir) { fetch('/trigger_degree_turn?angle=' + angle + '&dir=' + dir); }
                function sendSingleAction(actionKey) { fetch('/keyboard_input?keys=' + actionKey); }
                function touchStartAction(actionKey) { keys.add(actionKey); sendState(); }
                function touchEndAction(actionKey) { keys.delete(actionKey); sendState(); }

                function nudgePanTilt(axis, direction) {
                    fetch('/pan_tilt/nudge?axis=' + axis + '&dir=' + direction)
                        .then(res => res.json())
                        .then(data => {
                            document.getElementById("hud-pan-val").innerText = data.pan + "°";
                            document.getElementById("hud-tilt-val").innerText = data.tilt + "°";
                        });
                }
                function centerPanTilt() {
                    fetch('/pan_tilt/center')
                        .then(res => res.json())
                        .then(data => {
                            document.getElementById("hud-pan-val").innerText = data.pan + "°";
                            document.getElementById("hud-tilt-val").innerText = data.tilt + "°";
                        });
                }

                let lastKnownSampleSize = null;
                function setSampleSize(size) {
                    let statusEl = document.getElementById("sample-size-status");
                    statusEl.innerText = "Applying...";
                    fetch('/set_sample_size?size=' + size)
                        .then(res => res.json().then(data => ({ ok: res.ok, data })))
                        .then(({ ok, data }) => {
                            if (ok && data.ok) {
                                lastKnownSampleSize = data.sample_size;
                                highlightSampleSize(data.sample_size);
                                statusEl.innerText = "Now logging at " + data.sample_size + "x" + data.sample_size + (data.archived_path ? " (previous CSV archived)" : "");
                            } else {
                                statusEl.innerText = "Couldn't change: " + (data.error || "unknown error");
                            }
                        })
                        .catch(() => { statusEl.innerText = "Request failed - check connection"; });
                }
                function highlightSampleSize(size) {
                    document.querySelectorAll('.sample-size-btn').forEach(btn => {
                        if (parseInt(btn.dataset.size) === size) btn.classList.add('active-tracking');
                        else btn.classList.remove('active-tracking');
                    });
                }

                setInterval(() => {
                    fetch('/robot_stats')
                        .then(res => res.json())
                        .then(data => {
                            document.getElementById("hud-battery").innerText = data.battery.toFixed(1) + "%";
                            document.getElementById("hud-live-speed").innerText = data.live_speed.toFixed(1) + " mm/s";
                            document.getElementById("hud-target-speed").innerText = data.target_speed + " mm/s";

                            let speedPct = Math.max(0, Math.min(100, ((data.target_speed - 50) / (600 - 50)) * 100));
                            document.getElementById("hud-speed-bar").style.width = speedPct + "%";

                            const driveModeLabels = {
                                "manual": ["MANUAL", "var(--text-muted)"],
                                "line_follower": ["LINE FOLLOWER", "var(--accent-cyan)"],
                                "auto_roam": ["AUTO ROAM", "var(--accent-orange)"]
                            };
                            let dm = driveModeLabels[data.drive_mode] || driveModeLabels["manual"];
                            let dmNode = document.getElementById("hud-drive-mode");
                            dmNode.innerText = dm[0];
                            dmNode.style.color = dm[1];

                            let camBtn = document.getElementById("hud-camera-btn");
                            if (!camBtn.innerText.includes("...")) {
                                if (data.camera_active) {
                                    camBtn.classList.add("active-tracking");
                                    camBtn.innerText = "Camera & Logging: ON";
                                } else {
                                    camBtn.classList.remove("active-tracking");
                                    camBtn.innerText = "Camera & Logging: OFF";
                                }
                            }

                            document.getElementById("live-logged-keys").innerText = data.logged_keys.toUpperCase();
                            document.getElementById("live-frame-count").innerText = data.frame_count;

                            if (lastKnownSampleSize !== data.sample_size) {
                                lastKnownSampleSize = data.sample_size;
                                highlightSampleSize(data.sample_size);
                            }

                            if (data.logged_action === "") {
                                document.getElementById("live-logged-action").innerHTML = "<i style='color:var(--text-muted); font-weight:normal;'>[empty_idle]</i>";
                            } else {
                                document.getElementById("live-logged-action").innerText = data.logged_action;
                            }

                            let vacLabel = document.getElementById("hud-vacuum");
                            let hudVacBtn = document.getElementById("hud-vac-btn");
                            if (data.vacuum) {
                                vacLabel.innerText = "ACTIVE"; vacLabel.className = "hud-vac-on";
                                hudVacBtn.classList.add("btn-green"); hudVacBtn.innerText = "Vacuum: ON";
                            } else {
                                vacLabel.innerText = "OFF"; vacLabel.className = "hud-vac-off";
                                hudVacBtn.classList.remove("btn-green"); hudVacBtn.innerText = "Vacuum: OFF";
                            }
                        });
                }, 200);
            </script>
        </head>
        <body>
            <div class="deck">
                <div class="deck-header">
                    <h1>ROOMBA COMMAND DECK</h1>
                    <span class="deck-sub">orin cluster / roomba rig</span>
                </div>

                <div class="viewport-row">
                    <div id="hud-root" class="video-hud-container">
                        <img src="/video_feed" />

                        <div class="hud-telemetry-box">
                            <div class="row"><span>System</span> <span id="hud-mode-status" class="hud-status-text">MANUAL</span></div>
                            <div class="row"><span>Autonomy</span> <span id="hud-drive-mode" class="hud-status-text">MANUAL</span></div>
                            <div class="row"><span>Battery</span> <span id="hud-battery" class="hud-battery-text">100.0%</span></div>
                            <div class="row"><span>Telemetry</span> <span id="hud-live-speed" class="hud-speed-text">0.0 mm/s</span></div>
                            <div class="row"><span>Baseline</span> <span id="hud-target-speed" style="color:var(--accent-violet);">150 mm/s</span></div>
                            <div class="speed-bar-track"><div id="hud-speed-bar" class="speed-bar-fill"></div></div>
                            <div class="row"><span>Vacuum</span> <span id="hud-vacuum" class="hud-vac-off">OFF</span></div>
                            <div class="row"><span>Controller</span> <span id="hud-gamepad-status" style="color:var(--text-muted); font-weight:700;">NOT CONNECTED</span></div>
                        </div>

                        <div id="touch-dpad" class="dpad-overlay">
                            <div class="touch-btn" style="grid-column: 2; grid-row: 1;" ontouchstart="touchStartAction('w')" ontouchend="touchEndAction('w')" onmousedown="touchStartAction('w')" onmouseup="touchEndAction('w')">^</div>
                            <div class="touch-btn" style="grid-column: 1; grid-row: 2;" ontouchstart="touchStartAction('a')" ontouchend="touchEndAction('a')" onmousedown="touchStartAction('a')" onmouseup="touchEndAction('a')">&lt;</div>
                            <div class="touch-btn center-space" style="grid-column: 2; grid-row: 2;" ontouchstart="touchStartAction('space')" ontouchend="touchEndAction('space')" onmousedown="touchStartAction('space')" onmouseup="touchEndAction('space')">STOP</div>
                            <div class="touch-btn" style="grid-column: 3; grid-row: 2;" ontouchstart="touchStartAction('d')" ontouchend="touchEndAction('d')" onmousedown="touchStartAction('d')" onmouseup="touchEndAction('d')">&gt;</div>
                            <div class="touch-btn" style="grid-column: 2; grid-row: 3;" ontouchstart="touchStartAction('s')" ontouchend="touchEndAction('s')" onmousedown="touchStartAction('s')" onmouseup="touchEndAction('s')">v</div>
                        </div>
                    </div>

                    <div class="side-column">
                        <div class="logger-panel-box">
                            <div class="panel-title-sm">Dataset Stream</div>
                            <div class="log-data-card">
                                <div class="log-label">Logged Key Value</div>
                                <div id="live-logged-keys" class="log-value-keys">IDLING</div>
                            </div>
                            <div class="log-data-card">
                                <div class="log-label">Current Action Matrix</div>
                                <div id="live-logged-action" class="log-value-action"></div>
                            </div>
                            <div class="log-data-card" style="flex-grow:1;">
                                <div class="log-label">Processed Frame ID</div>
                                <div id="live-frame-count" class="log-value-frames">0</div>
                            </div>
                            <button id="hud-newcsv-btn" class="dashboard-btn" onclick="startNewCsv()" title="Archives the current dataset CSV under a timestamped name and starts a fresh one">New CSV</button>
                        </div>
                    </div>
                </div>

                <div class="tab-shell">
                    <div class="tab-bar">
                        <button id="tab-btn-autonomy" class="tab-btn tab-active" onclick="showTab('autonomy')">Autonomy</button>
                        <button id="tab-btn-camera" class="tab-btn" onclick="showTab('camera')">Camera</button>
                        <button id="tab-btn-record" class="tab-btn" onclick="showTab('record')">Record &amp; Dock</button>
                        <button id="tab-btn-manual" class="tab-btn" onclick="showTab('manual')">Manual Drive</button>
                    </div>

                    <div id="tab-panel-autonomy" class="tab-panel tab-panel-active">
                        <div class="cluster">
                            <div class="cluster-title">Camera &amp; Logging</div>
                            <div class="cluster-note">Must be ON before any drive mode below will act.</div>
                            <button id="hud-camera-btn" class="dashboard-btn" onclick="toggleCamera()">Camera &amp; Logging: OFF</button>
                            <button id="hud-gray-btn" class="dashboard-btn" onclick="toggleGrayscaleView()">AI Matrix View (Grayscale)</button>
                        </div>
                        <div class="cluster">
                            <div class="cluster-title">Drive Modes</div>
                            <button id="hud-auto-btn" class="dashboard-btn" onclick="toggleAutoMode()">Blue Line Follower</button>
                            <button id="hud-roam-btn" class="dashboard-btn" onclick="toggleRoamMode()">Auto Roam (Stock-style)</button>
                            <button class="dashboard-btn btn-orange" onclick="triggerHardwareWakePulse()">Force Hardware Wake (BRC Pin)</button>
                        </div>
                    </div>

                    <div id="tab-panel-camera" class="tab-panel">
                        <div class="cluster">
                            <div class="cluster-title">Pan / Tilt Gimbal</div>
                            <div class="angle-readout">
                                <span>PAN: <span id="hud-pan-val" class="val">90°</span></span>
                                <span>TILT: <span id="hud-tilt-val" class="val">90°</span></span>
                            </div>
                            <div class="btn-group-3">
                                <span></span>
                                <button class="dashboard-btn" onclick="nudgePanTilt('tilt','up')">Tilt Up</button>
                                <span></span>
                                <button class="dashboard-btn" onclick="nudgePanTilt('pan','left')">Pan Left</button>
                                <button class="dashboard-btn btn-cyan" onclick="centerPanTilt()">Center</button>
                                <button class="dashboard-btn" onclick="nudgePanTilt('pan','right')">Pan Right</button>
                                <span></span>
                                <button class="dashboard-btn" onclick="nudgePanTilt('tilt','down')">Tilt Down</button>
                                <span></span>
                            </div>
                        </div>
                        <div class="cluster">
                            <div class="cluster-title">View</div>
                            <button id="hud-fs-btn" class="dashboard-btn" onclick="toggleFullScreen()">Full Screen HUD</button>
                            <button id="hud-dpad-toggle" class="dashboard-btn" onclick="toggleDpadVisibility()">Toggle Touch HUD</button>
                            <div class="cluster-note">Arrows or a controller's right stick also drive the gimbal - see the legend below.</div>
                        </div>
                        <div class="cluster" style="grid-column: 1 / -1;">
                            <div class="cluster-title">Dataset Resolution</div>
                            <div class="cluster-note">Camera must be OFF to change this - switching size starts a fresh CSV under the new grid.</div>
                            <div class="btn-group-3">
                                <button class="dashboard-btn sample-size-btn" data-size="16" onclick="setSampleSize(16)">16x16</button>
                                <button class="dashboard-btn sample-size-btn" data-size="25" onclick="setSampleSize(25)">25x25</button>
                                <button class="dashboard-btn sample-size-btn" data-size="32" onclick="setSampleSize(32)">32x32</button>
                                <button class="dashboard-btn sample-size-btn" data-size="50" onclick="setSampleSize(50)">50x50</button>
                                <button class="dashboard-btn sample-size-btn" data-size="64" onclick="setSampleSize(64)">64x64</button>
                                <button class="dashboard-btn sample-size-btn" data-size="96" onclick="setSampleSize(96)">96x96</button>
                            </div>
                            <div id="sample-size-status" class="cluster-note"></div>
                        </div>
                    </div>

                    <div id="tab-panel-record" class="tab-panel">
                        <div class="cluster">
                            <div class="cluster-title">Macro Recorder</div>
                            <div class="cluster-note">Record a driving sequence once, then auto-replay it.</div>
                            <button id="hud-macro-btn" class="dashboard-btn" onclick="advanceMacroState()">Start Recording</button>
                            <button id="hud-pause-btn" class="dashboard-btn btn-orange" style="display:none;" onclick="togglePlayPause()">Pause Playback</button>
                        </div>
                        <div class="cluster">
                            <div class="cluster-title">Docking</div>
                            <button id="hud-dock-btn" class="dashboard-btn btn-orange" onclick="triggerDocking()">Seek Base</button>
                            <button id="hud-cancel-dock-btn" class="dashboard-btn btn-red" style="display:none;" onclick="triggerCancelDocking()">Cancel Dock</button>
                        </div>
                    </div>

                    <div id="tab-panel-manual" class="tab-panel">
                        <div class="cluster">
                            <div class="cluster-title">Precision Turns</div>
                            <div class="btn-group">
                                <button class="dashboard-btn" onclick="triggerFixedDegreeTurn(90, 'left')">Left 90°</button>
                                <button class="dashboard-btn" onclick="triggerFixedDegreeTurn(90, 'right')">Right 90°</button>
                            </div>
                            <button class="dashboard-btn" style="border-color:var(--accent-violet); color:var(--accent-violet);" onclick="triggerFixedDegreeTurn(180, 'left')">Spin Around 180°</button>
                        </div>
                        <div class="cluster">
                            <div class="cluster-title">Continuous Spin</div>
                            <button class="dashboard-btn btn-orange" onclick="triggerFullScreenMode('sticky_spin_random')">Sticky Spin Random</button>
                            <div class="btn-group">
                                <button class="dashboard-btn btn-red" onclick="triggerFullScreenMode('sticky_spin_left')">Spin Left</button>
                                <button class="dashboard-btn btn-green" onclick="triggerFullScreenMode('sticky_spin_right')">Spin Right</button>
                            </div>
                        </div>
                        <div class="cluster" style="grid-column: 1 / -1;">
                            <div class="cluster-title">Vacuum &amp; Speed</div>
                            <div class="btn-group">
                                <button id="hud-vac-btn" class="dashboard-btn" onclick="sendSingleAction('v')">Vacuum Toggle</button>
                                <div class="btn-group">
                                    <button class="dashboard-btn" onclick="sendSingleAction('q')">Speed -</button>
                                    <button class="dashboard-btn" onclick="sendSingleAction('e')">Speed +</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="instructions">
                    <p><span class="key">W</span> / <span class="key">S</span> Forward/Back &nbsp; <span class="key">A</span> / <span class="key">D</span> Spin &nbsp; <span class="key">SPACE</span> Stop &nbsp; <span class="key">Q</span> / <span class="key">E</span> Speed &nbsp; <span class="key">V</span> Vacuum</p>
                    <p><span class="key">Up</span> <span class="key">Down</span> <span class="key">Left</span> <span class="key">Right</span> Camera Pan/Tilt (hold to keep nudging)</p>
                    <p>Controller - Left stick/D-pad: Drive &nbsp; Right stick: Camera &nbsp; A: Vacuum &nbsp; B: Stop &nbsp; Y: Center Camera &nbsp; Start: Auto Mode &nbsp; Bumpers: Speed</p>
                </div>
            </div>
        </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if current_frame is None:
                    frame_to_send = None
                else:
                    ret, jpeg = cv2.imencode('.jpg', current_frame)
                    frame_to_send = jpeg.tobytes() if ret else None
            if frame_to_send is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/trigger_hardware_wake')
def trigger_hardware_wake():
    hardware_wake_roomba_via_cp2102()
    return "WAKE COMPLETED", 200

@app.route('/pan_tilt/nudge')
def pan_tilt_nudge():
    axis = request.args.get('axis', 'pan')
    direction = request.args.get('dir', 'left')
    sign = 1 if direction in ('left', 'up') else -1
    if axis == 'pan':
        set_pan(current_pan_angle + sign * PAN_TILT_STEP)
    else:
        set_tilt(current_tilt_angle + sign * PAN_TILT_STEP)
    return jsonify({"pan": current_pan_angle, "tilt": current_tilt_angle})

@app.route('/pan_tilt/center')
def pan_tilt_center():
    center_pan_tilt()
    return jsonify({"pan": current_pan_angle, "tilt": current_tilt_angle})

@app.route('/toggle_camera')
def toggle_camera():
    global camera_active, camera_thread, current_frame
    if camera_active:
        camera_stop_event.set()
        camera_active = False
        current_frame = None
        return jsonify({"camera_active": False})
    else:
        camera_stop_event.clear()
        camera_active = True
        camera_thread = threading.Thread(target=camera_and_logic_loop, daemon=True)
        camera_thread.start()
        return jsonify({"camera_active": True})

@app.route('/new_csv')
def new_csv():
    archived_path, new_path = start_new_csv_session()
    return jsonify({
        "ok": True,
        "new_csv_path": new_path,
        "archived_path": archived_path
    })

@app.route('/set_sample_size')
def set_sample_size():
    """Changes the dataset's downsample grid (e.g. 25x25 -> 50x50). Only allowed
    while the camera is off, since the CSV header's pixel-column count is fixed
    per file - changing it mid-stream would corrupt the schema. Automatically
    rotates to a fresh CSV under the new size so the header stays consistent."""
    global SAMPLE_SIZE
    size = request.args.get('size', type=int)
    if not size or size < 8 or size > 128:
        return jsonify({"ok": False, "error": "size must be between 8 and 128"}), 400
    if camera_active:
        return jsonify({"ok": False, "error": "camera must be off to change sample size"}), 409
    SAMPLE_SIZE = size
    archived_path, new_path = start_new_csv_session()
    return jsonify({"ok": True, "sample_size": SAMPLE_SIZE, "new_csv_path": new_path, "archived_path": archived_path})

@app.route('/toggle_auto')
def toggle_auto():
    global auto_mode, roam_mode, current_action_label, operational_mode, active_key_string
    auto_mode = not auto_mode
    if auto_mode:
        roam_mode = False
    operational_mode = "manual"
    drive_roomba(0, 0)
    current_action_label = ""
    active_key_string = "idling"
    return "AUTO MODE ACTIVE" if auto_mode else "MANUAL MODE"

@app.route('/toggle_roam')
def toggle_roam():
    global roam_mode, auto_mode, current_action_label, operational_mode, active_key_string, roam_next_wander_time
    roam_mode = not roam_mode
    if roam_mode:
        auto_mode = False
        operational_mode = "manual"
        roam_next_wander_time = time.time() + random.uniform(ROAM_WANDER_MIN_SEC, ROAM_WANDER_MAX_SEC)
    drive_roomba(0, 0)
    current_action_label = ""
    active_key_string = "idling"
    return "ROAM MODE ACTIVE" if roam_mode else "MANUAL MODE"

@app.route('/toggle_grayscale')
def toggle_grayscale():
    global show_grayscale
    show_grayscale = not show_grayscale
    return jsonify({"grayscale": show_grayscale})

@app.route('/advance_macro_mode')
def advance_macro_mode():
    global operational_mode, movement_history, current_move_start, last_saved_keys, auto_mode, playback_index, active_key_string
    auto_mode = False
    if operational_mode in ["manual", "paused"]:
        operational_mode = "record"
        movement_history = []
        playback_index = 0
        last_saved_keys = set(['idling'])
        current_move_start = time.time()
        active_key_string = "idling"
        return jsonify({"label": "recording", "css_class": "btn-record active-recording", "button_text": "Stop & Playback", "show_pause": False})
    elif operational_mode == "record":
        if current_move_start is not None:
            duration = time.time() - current_move_start
            if duration > 0.05:
                movement_history.append(([active_key_string], duration, current_speed))
        drive_roomba(0, 0)
        active_key_string = "idling"
        if len(movement_history) > 0:
            operational_mode = "playback"
            playback_index = 0
            threading.Thread(target=execution_playback_worker, daemon=True).start()
            return jsonify({"label": "playback", "css_class": "btn-playback", "button_text": "Clear & Re-Record", "show_pause": True, "pause_text": "Pause Playback"})
        else:
            operational_mode = "manual"
            return jsonify({"label": "manual", "css_class": "btn-manual", "button_text": "Start Recording", "show_pause": False})
    else:
        operational_mode = "manual"
        playback_index = 0
        drive_roomba(0, 0)
        active_key_string = "idling"
        return jsonify({"label": "manual", "css_class": "btn-manual", "button_text": "Start Recording", "show_pause": False})

@app.route('/toggle_play_pause')
def toggle_play_pause():
    global operational_mode, active_key_string
    if operational_mode == "playback":
        operational_mode = "paused"
        drive_roomba(0, 0)
        active_key_string = "idling"
        return jsonify({"label": "paused", "css_class": "btn-paused", "pause_text": "Resume Playback"})
    elif operational_mode == "paused":
        operational_mode = "playback"
        threading.Thread(target=execution_playback_worker, daemon=True).start()
        return jsonify({"label": "playback", "css_class": "btn-playback", "pause_text": "Pause Playback"})
    return jsonify({"label": operational_mode, "css_class": "btn-manual", "pause_text": "Pause Playback"})

@app.route('/trigger_degree_turn')
def trigger_degree_turn():
    angle_param = request.args.get('angle', 90, type=int)
    direction_param = request.args.get('dir', 'left')
    direction_sign = 1 if direction_param == 'left' else -1
    threading.Thread(target=execute_closed_loop_turn, args=(angle_param, direction_sign,), daemon=True).start()
    return "OK", 200

@app.route('/trigger_sticky_spin')
def trigger_sticky_spin():
    global current_action_label, active_key_string, auto_mode
    auto_mode = False
    mode_param = request.args.get('mode', 'sticky_spin_random')
    current_action_label = mode_param
    active_key_string = "idling"
    return "OK", 200

@app.route('/dock')
def dock():
    global auto_mode, operational_mode, current_action_label, vacuum_on, active_key_string
    auto_mode = False
    operational_mode = "manual"
    vacuum_on = False
    current_action_label = "docking"
    active_key_string = "idling"
    dock_roomba()
    return "SEEKING HOME DOCK..."

@app.route('/cancel_dock')
def cancel_dock():
    global current_action_label, active_key_string
    current_action_label = ""
    active_key_string = "idling"
    cancel_docking_sequence()
    return "MANUAL CONTROL RESTORED"

@app.route('/robot_stats')
def robot_stats():
    if auto_mode:
        drive_mode = "line_follower"
    elif roam_mode:
        drive_mode = "auto_roam"
    else:
        drive_mode = "manual"
    return jsonify({
        "live_speed": abs(live_speed_mms),
        "target_speed": current_speed,
        "battery": battery_pct,
        "vacuum": vacuum_on,
        "logged_keys": active_key_string,
        "logged_action": current_action_label,
        "frame_count": frame_count,
        "drive_mode": drive_mode,
        "camera_active": camera_active,
        "sample_size": SAMPLE_SIZE
    })

@app.route('/keyboard_input')
def keyboard_input():
    global current_speed, vacuum_on, last_keys_set, auto_mode, roam_mode, current_action_label, operational_mode, current_move_start, last_saved_keys, active_key_string
    raw_keys = request.args.get('keys', '')
    current_keys = set(raw_keys.split('-')) if raw_keys else set()

    newly_pressed = current_keys - last_keys_set
    if 'q' in newly_pressed: current_speed = max(50, current_speed - 50)
    if 'e' in newly_pressed: current_speed = min(600, current_speed + 50)
    if 'v' in newly_pressed:
        vacuum_on = not vacuum_on
        set_vacuum(vacuum_on)

    last_keys_set = current_keys

    if (operational_mode in ["playback", "paused"] or current_action_label == "docking" or "sticky" in current_action_label or roam_mode) and len(current_keys) > 0 and 'space' not in current_keys:
        operational_mode = "manual"
        roam_mode = False
        cancel_docking_sequence()
        current_action_label = ""
        active_key_string = "idling"

    if not auto_mode and not roam_mode and operational_mode != "playback" and operational_mode != "paused" and current_action_label != "precision_turning" and "encoder_turning" not in current_action_label and current_action_label != "docking" and "sticky" not in current_action_label and "roam" not in current_action_label:
        filtered_move_keys = current_keys.intersection({'w', 'a', 's', 'd', 'space'})

        if not filtered_move_keys:
            active_key_string = "idling"
        elif 'space' in filtered_move_keys:
            active_key_string = "space"
        elif 'w' in filtered_move_keys:
            active_key_string = "w"
        elif 's' in filtered_move_keys:
            active_key_string = "s"
        elif 'a' in filtered_move_keys:
            active_key_string = "a"
        elif 'd' in filtered_move_keys:
            active_key_string = "d"

        if operational_mode == "record":
            if filtered_move_keys != last_saved_keys:
                now = time.time()
                if current_move_start is not None:
                    duration = now - current_move_start
                    if duration > 0.05:
                        movement_history.append(([active_key_string], duration, current_speed))
                last_saved_keys = filtered_move_keys
                current_move_start = now

        if not filtered_move_keys:
            drive_roomba(0, 0)
            current_action_label = ""
        elif 'space' in filtered_move_keys:
            drive_roomba(0, 0)
            current_action_label = "stop"
        elif 'w' in filtered_move_keys:
            drive_roomba(current_speed, 32767)
            current_action_label = "forward"
        elif 's' in filtered_move_keys:
            drive_roomba(-current_speed, 32767)
            current_action_label = "backward"
        elif 'a' in filtered_move_keys:
            drive_roomba(current_speed, 1)
            current_action_label = "left"
        elif 'd' in filtered_move_keys:
            drive_roomba(current_speed, -1)
            current_action_label = "right"

    return "OK", 200

def safely_close_local_session_on_exit():
    global csv_file
    drive_roomba(0, 0)
    if csv_file:
        try:
            csv_file.flush()
            csv_file.close()
            print("Dataset loop closed. Local telemetry CSV saved securely to your Documents folder.")
        except Exception as e:
            print(f"Error packing file resource: {e}")

atexit.register(safely_close_local_session_on_exit)

def handle_shutdown_signal(signum, frame):
    print(f"Received shutdown signal ({signum}) - releasing camera before exit...")
    shutdown_event.set()
    drive_roomba(0, 0)
    for _ in range(20):
        if current_cap is None:
            break
        time.sleep(0.1)
    else:
        print("Camera loop didn't confirm release in time - exiting anyway.")
    if csv_file:
        try:
            csv_file.flush()
            csv_file.close()
        except Exception as e:
            print(f"Error closing CSV: {e}")
    os._exit(0)

signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

if __name__ == '__main__':
    threading.Thread(target=keep_roomba_alive_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
