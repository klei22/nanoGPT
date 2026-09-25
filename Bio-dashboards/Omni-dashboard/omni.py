import asyncio
import csv
import datetime
import json
import logging
import os
import sys
import threading
import time
import math
from collections import deque
from itertools import islice
from flask import Flask, Response, jsonify, request, render_template_string
from bleak import BleakClient, BleakScanner
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice
from polar_python import PolarDevice

# ==========================================
# 1. CONFIGURATION & GLOBAL STATE
# ==========================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_omni")
os.makedirs(LOGS_DIR, exist_ok=True)

ECG_HZ = 130
ACC_HZ = 200

class SampleRing:
    """Thread-safe ring of recent samples with a running sequence number.

    The BLE thread appends; every SSE client keeps its own cursor and asks for
    everything since it last looked. Nothing is ever cleared out from under the
    writer, and N browser tabs each get the full stream.
    """
    def __init__(self, maxlen):
        self._buf = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()

    def extend(self, items):
        with self._lock:
            self._buf.extend(items)
            self._seq += len(items)

    def since(self, seq):
        with self._lock:
            n = min(self._seq - seq, len(self._buf))
            items = list(islice(self._buf, len(self._buf) - n, None)) if n > 0 else []
            return items, self._seq

    @property
    def seq(self):
        with self._lock:
            return self._seq

class StreamClock:
    """Sensor-clock -> wall-clock mapping for PMD frames (timestamp = last sample).
    Gives jitter-free sample spacing; re-anchors if drift exceeds 1 s."""
    def __init__(self, hz):
        self.period_ms = 1000.0 / hz
        self.offset_ms = None

    def reset(self):
        self.offset_ms = None

    def stamps(self, sensor_ts_ns, n):
        now_ms = time.time() * 1000.0
        if sensor_ts_ns:
            last_ms = sensor_ts_ns / 1e6
            if self.offset_ms is None or abs(last_ms + self.offset_ms - now_ms) > 1000.0:
                self.offset_ms = now_ms - last_ms
            end_ms = last_ms + self.offset_ms
        else:
            end_ms = now_ms
        return [int(end_ms - (n - 1 - i) * self.period_ms) for i in range(n)]

ecg_ring = SampleRing(ECG_HZ * 10)   # [ts_ms, mV]
acc_ring = SampleRing(ACC_HZ * 10)   # [ts_ms, x, y, z]
rr_ring = SampleRing(200)            # [ts_ms, rr_ms] -- one entry per beat
ecg_clock = StreamClock(ECG_HZ)
acc_clock = StreamClock(ACC_HZ)

# Shared Memory Matrix (scalars only; waveform data lives in the rings above)
omni_state = {
    "polar": {"status": "Disconnected", "hr": 0, "rr": 0, "rmssd": 0.0, "sdnn": 0.0},
    "viatom": {"status": "Disconnected", "spo2": 0, "hr": 0}
}

discovered_devices = {"polar": [], "viatom": []}
active_targets = {"polar": None, "viatom": None}

# DBus Cache
ble_device_cache = {}

polar_ppi_history = deque(maxlen=40)
polar_last_heartbeat = 0
viatom_link_up = False

# Hardware UUIDs
VIATOM_MAC = "F3:A0:A8:E3:F5:63"
VIATOM_SVC_UUID = "14839ac4-7d7e-415c-9a42-167340cf2339"
VIATOM_WRITE_UUID = "8b00ace7-eb0a-49b0-b977-10a8d4d5e82f"

# --- ADAPTER ROUTING ---
POLAR_ADAPTER = "hci0"    
GENERAL_ADAPTER = "hci1"  

# ==========================================
# 2. FLASK WEB DASHBOARD (STABILIZED UI)
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bio-Dash Advanced Lab</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.0.1/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.2.0/dist/chartjs-adapter-luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-streaming@2.0.0/dist/chartjs-plugin-streaming.min.js"></script>
</head>
<body class="bg-slate-900 text-white font-sans min-h-screen flex flex-col">

    <header class="p-4 bg-slate-800 border-b border-slate-700 shadow-md">
        <div class="container mx-auto flex justify-between items-center max-w-7xl">
            <div>
                <h1 class="text-xl font-bold text-sky-400 tracking-wide flex items-center gap-3">
                    <span class="relative flex h-3 w-3">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-3 w-3 bg-sky-500"></span>
                    </span>
                    Bio-Dash Omni-System
                </h1>
                <p class="text-xs text-slate-400 mt-1">Dual-Threaded Hardware Telemetry</p>
            </div>
            <div class="flex gap-2">
                <span id="stat-polar" class="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">POLAR: AWAITING</span>
                <span id="stat-viatom" class="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">O2: AWAITING</span>
            </div>
        </div>
    </header>

    <main class="container mx-auto px-4 py-6 flex-grow max-w-7xl flex flex-col gap-6">

        <div id="scanner-panel" class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl w-full transition-all duration-500">
            <h2 class="text-lg font-bold text-slate-200 mb-4">Radar Active: Scanning Airspace...</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div id="wrapper-polar">
                    <h3 class="text-sm font-bold text-rose-400 mb-2 uppercase tracking-wider">Polar H10</h3>
                    <div id="list-polar" class="flex flex-col gap-3">
                        <div class="text-slate-500 text-sm animate-pulse">Initializing Bluetooth Radar...</div>
                    </div>
                </div>
                <div id="wrapper-viatom">
                    <h3 class="text-sm font-bold text-indigo-400 mb-2 uppercase tracking-wider">Checkme O2</h3>
                    <div id="list-viatom" class="flex flex-col gap-3">
                        <div class="text-slate-500 text-sm animate-pulse">Initializing Bluetooth Radar...</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-6 lg:gap-8">
            
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 lg:p-5 shadow-xl flex flex-col justify-between items-center border-b-4 border-indigo-500 min-h-[140px]">
                <h2 class="text-[10px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest text-center w-full">O2 Saturation</h2>
                <div class="flex items-baseline justify-center w-full mt-2">
                    <div class="w-1/2 text-right"><span id="val-spo2" class="text-4xl lg:text-5xl font-extrabold text-indigo-400 tabular-nums tracking-tight">--</span></div>
                    <div class="w-1/2 text-left pl-1"><span class="text-sm lg:text-base font-medium text-slate-500">%</span></div>
                </div>
                <div class="h-4 mt-2 w-full"></div>
            </div>
            
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 lg:p-5 shadow-xl flex flex-col justify-between items-center border-b-4 border-cyan-500 min-h-[140px]">
                <h2 class="text-[10px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest text-center w-full">Ring HR</h2>
                <div class="flex items-baseline justify-center w-full mt-2">
                    <div class="w-1/2 text-right"><span id="val-viatom-hr" class="text-4xl lg:text-5xl font-extrabold text-cyan-400 tabular-nums tracking-tight">--</span></div>
                    <div class="w-1/2 text-left pl-1"><span class="text-sm lg:text-base font-medium text-slate-500">BPM</span></div>
                </div>
                <div class="h-4 mt-2 w-full"></div>
            </div>

            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 lg:p-5 shadow-xl flex flex-col justify-between items-center border-b-4 border-rose-500 min-h-[140px]">
                <h2 class="text-[10px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest text-center w-full">ECG HR</h2>
                <div class="flex items-baseline justify-center w-full mt-2">
                    <div class="w-1/2 text-right"><span id="val-polar-hr" class="text-4xl lg:text-5xl font-extrabold text-rose-500 tabular-nums tracking-tight">--</span></div>
                    <div class="w-1/2 text-left pl-1"><span class="text-sm lg:text-base font-medium text-slate-500">BPM</span></div>
                </div>
                <div class="h-4 mt-2 w-full"></div>
            </div>

            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 lg:p-5 shadow-xl flex flex-col justify-between items-center border-b-4 border-emerald-500 min-h-[140px]">
                <h2 class="text-[10px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest text-center w-full">RR Interval</h2>
                <div class="flex items-baseline justify-center w-full mt-2">
                    <div class="w-1/2 text-right"><span id="val-rr" class="text-4xl lg:text-5xl font-extrabold text-emerald-400 tabular-nums tracking-tight">--</span></div>
                    <div class="w-1/2 text-left pl-1"><span class="text-sm lg:text-base font-medium text-slate-500">ms</span></div>
                </div>
                <div class="h-4 mt-2 w-full"></div>
            </div>

            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 lg:p-5 shadow-xl flex flex-col justify-between items-center border-b-4 border-purple-500 min-h-[140px]">
                <h2 class="text-[10px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest text-center w-full">HRV (RMSSD)</h2>
                <div class="flex items-baseline justify-center w-full mt-2">
                    <div class="w-1/2 text-right"><span id="val-rmssd" class="text-4xl lg:text-5xl font-extrabold text-purple-400 tabular-nums tracking-tight">--</span></div>
                    <div class="w-1/2 text-left pl-1"><span class="text-sm lg:text-base font-medium text-slate-500">ms</span></div>
                </div>
                <div id="stress-label" class="text-[9px] sm:text-[10px] font-bold mt-2 text-slate-500 text-center w-full animate-pulse h-4 whitespace-nowrap">AWAITING BEATS...</div>
            </div>

            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 lg:p-5 shadow-xl flex flex-col justify-between items-center border-b-4 border-violet-500 min-h-[140px]">
                <h2 class="text-[10px] sm:text-xs font-bold text-slate-400 uppercase tracking-widest text-center w-full">Stress (SDNN)</h2>
                <div class="flex items-baseline justify-center w-full mt-2">
                    <div class="w-1/2 text-right"><span id="val-sdnn" class="text-4xl lg:text-5xl font-extrabold text-violet-400 tabular-nums tracking-tight">--</span></div>
                    <div class="w-1/2 text-left pl-1"><span class="text-sm lg:text-base font-medium text-slate-500">ms</span></div>
                </div>
                <div class="h-4 mt-2 w-full"></div>
            </div>
        </div>

        <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-xl w-full">
            <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Live PQRST Waveform (130Hz)</h2>
            <div class="relative h-48 w-full"><canvas id="ecgChart"></canvas></div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8">
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-xl w-full">
                <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Chest Kinematics (X,Y,Z)</h2>
                <div class="relative h-48 w-full"><canvas id="accChart"></canvas></div>
            </div>

            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-xl w-full">
                <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Heart Rate Variability (R-R)</h2>
                <div class="relative h-48 w-full"><canvas id="ppiChart"></canvas></div>
            </div>
        </div>
    </main>

    <script>
        const chartOptions = (delayTime) => ({
            responsive: true, maintainAspectRatio: false, animation: false,
            scales: {
                x: { type: 'realtime', realtime: { duration: 5000, refresh: 40, delay: delayTime } },
                y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { display: false } }
        });

        const ecgChart = new Chart(document.getElementById('ecgChart').getContext('2d'), {
            type: 'line', data: { datasets: [{ borderColor: '#f43f5e', borderWidth: 2, pointRadius: 0, data: [] }] },
            options: chartOptions(1000)
        });

        const accChart = new Chart(document.getElementById('accChart').getContext('2d'), {
            type: 'line', data: { datasets: [
                { label: 'X', borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0, data: [] },
                { label: 'Y', borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0, data: [] },
                { label: 'Z', borderColor: '#fbbf24', borderWidth: 1.5, pointRadius: 0, data: [] }
            ]},
            options: { ...chartOptions(1000), plugins: { legend: { display: true, labels: { color: '#94a3b8' } } } }
        });

        const ppiChart = new Chart(document.getElementById('ppiChart').getContext('2d'), {
            type: 'line', data: { datasets: [{ borderColor: '#818cf8', backgroundColor: '#818cf8', borderWidth: 0, pointRadius: 4, data: [] }] },
            options: chartOptions(1500)
        });

        const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

        async function fetchScanners() {
            try {
                const res = await fetch('/api/scanners');
                const state = await res.json();
                
                let anyScanning = false;

                ['polar', 'viatom'].forEach(type => {
                    const statusEl = document.getElementById(`stat-${type}`);
                    const wrapper = document.getElementById(`wrapper-${type}`);
                    const list = document.getElementById(`list-${type}`);
                    const deviceStatus = state.state[type].status;
                    
                    statusEl.innerText = `${type.toUpperCase()}: ${deviceStatus.toUpperCase()}`;
                    
                    if (deviceStatus === "Connected") {
                        statusEl.className = "px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
                        wrapper.style.display = "none";
                    } else if (deviceStatus === "Connecting" || deviceStatus === "Calibrating") {
                        statusEl.className = "px-3 py-1 rounded-full text-xs font-semibold bg-sky-500/20 text-sky-400 border border-sky-500/30";
                        wrapper.style.display = "none";
                    } else {
                        anyScanning = true;
                        wrapper.style.display = "block";
                        statusEl.className = "px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30";
                        
                        list.innerHTML = '';
                        if (state.devices[type].length === 0) {
                            list.innerHTML = `<div class="text-slate-500 text-sm italic py-2">Searching airspace...</div>`;
                        } else {
                            state.devices[type].forEach(d => {
                                const div = document.createElement('div');
                                div.className = 'bg-slate-700/50 p-4 rounded-xl border border-slate-600 flex justify-between items-center';
                                div.innerHTML = `
                                    <div>
                                        <div class="font-bold text-slate-200 text-sm">${esc(d.name)}</div>
                                        <div class="text-[10px] text-slate-400 font-mono mt-1">${esc(d.address)} · ${esc(d.rssi)} dBm</div>
                                    </div>
                                    <button class="bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold py-2 px-5 rounded transition-colors" 
                                            data-type="${type}" data-mac="${esc(d.address)}">
                                        LOCK
                                    </button>
                                `;
                                list.appendChild(div);
                            });
                        }
                    }
                });

                document.getElementById('scanner-panel').style.display = anyScanning ? "block" : "none";

            } catch(e) {}
        }
        setInterval(fetchScanners, 2000);

        document.getElementById('scanner-panel').addEventListener('click', (e) => {
            const b = e.target.closest('button[data-mac]');
            if (b) lockTarget(b.dataset.type, b.dataset.mac);
        });

        async function lockTarget(type, mac) {
            document.getElementById(`list-${type}`).innerHTML = `<div class="text-emerald-400 font-bold py-4 animate-pulse">Handshaking with ${esc(mac)}...</div>`;
            await fetch('/api/connect', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type: type, address: mac}) });
        }

        const $ = (id) => document.getElementById(id);
        const ecgData = ecgChart.data.datasets[0].data;
        const [accX, accY, accZ] = accChart.data.datasets.map(d => d.data);
        const ppiData = ppiChart.data.datasets[0].data;
        const labelEl = $('stress-label');
        const LABEL_BASE = 'text-[10px] sm:text-xs font-bold mt-2 text-center w-full h-4 ';

        // EventSource reconnects on its own if the server restarts
        const evtSource = new EventSource("/api/stream");
        evtSource.onmessage = (e) => {
            const { polar, viatom } = JSON.parse(e.data);

            if (polar.status === "Connected") {
                if (polar.hr > 0) $('val-polar-hr').textContent = polar.hr;
                if (polar.rr > 0) $('val-rr').textContent = polar.rr;

                // One chart point per actual beat (not one per SSE tick)
                for (const [ts, rr] of polar.rr_buffer) ppiData.push({ x: ts, y: rr });

                const rmssd = polar.rmssd;
                $('val-rmssd').textContent = rmssd.toFixed(1);
                if (rmssd > 0) {
                    const [text, color] = rmssd < 20 ? ['HIGH STRESS', 'text-rose-500']
                                        : rmssd < 50 ? ['BALANCED', 'text-amber-400']
                                        : ['RELAXED', 'text-emerald-400'];
                    labelEl.textContent = text;
                    labelEl.className = LABEL_BASE + color;
                }
                $('val-sdnn').textContent = polar.sdnn.toFixed(1);

                for (const [ts, mv] of polar.ecg_buffer) ecgData.push({ x: ts, y: mv });
                for (const [ts, x, y, z] of polar.acc_buffer) {
                    accX.push({ x: ts, y: x });
                    accY.push({ x: ts, y: y });
                    accZ.push({ x: ts, y: z });
                }
            }

            if (viatom.status === "Connected") {
                if (viatom.spo2 > 0) $('val-spo2').textContent = viatom.spo2;
                if (viatom.hr > 0) $('val-viatom-hr').textContent = viatom.hr;
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/scanners')
def get_scanners():
    return jsonify({"devices": discovered_devices, "state": omni_state})

@app.route('/api/connect', methods=['POST'])
def command_connect():
    req = request.json
    t_type = req.get('type')
    t_mac = req.get('address')
    if t_type in active_targets:
        active_targets[t_type] = t_mac
    return jsonify({"status": "locked", "target": t_mac})

@app.route('/api/stream')
def stream_data():
    def event_stream():
        # Start each client at "now" so it doesn't replay the ring on connect
        ecg_seq, acc_seq, rr_seq = ecg_ring.seq, acc_ring.seq, rr_ring.seq
        while True:
            ecg, ecg_seq = ecg_ring.since(ecg_seq)
            acc, acc_seq = acc_ring.since(acc_seq)
            rr, rr_seq = rr_ring.since(rr_seq)
            payload = {
                "polar": {**omni_state["polar"], "ecg_buffer": ecg, "acc_buffer": acc, "rr_buffer": rr},
                "viatom": omni_state["viatom"],
            }
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")


# ==========================================
# 3. BLUETOOTH ENGINE (ASYNCIO)
# ==========================================

def _first_attr(obj, names, default=None):
    for n in names:
        v = getattr(obj, n, None)
        if v:
            return v
    return default

def polar_hr_cb(data):
    # polar_python HRData: heartrate (int BPM), rr_intervals (list[float] ms)
    global polar_last_heartbeat
    polar_last_heartbeat = time.time()

    bpm = _first_attr(data, ("heartrate", "bpm", "heart_rate"))
    if bpm is not None:
        omni_state["polar"]["hr"] = int(bpm)

    rr_list = _first_attr(data, ("rr_intervals", "rrs", "rrs_ms"), [])
    new_rr = []
    now_ms = int(time.time() * 1000)
    for rr in rr_list:
        try:
            rr_val = int(round(float(rr)))
        except (ValueError, TypeError):
            continue
        if 200 < rr_val < 2000:
            polar_ppi_history.append(rr_val)
            new_rr.append([now_ms, rr_val])

    if not new_rr:
        return
    omni_state["polar"]["rr"] = new_rr[-1][1]
    rr_ring.extend(new_rr)

    h = polar_ppi_history
    if len(h) > 2:
        sq_diff = sum((h[i] - h[i - 1]) ** 2 for i in range(1, len(h)))
        omni_state["polar"]["rmssd"] = math.sqrt(sq_diff / (len(h) - 1))
        mean_rr = sum(h) / len(h)
        omni_state["polar"]["sdnn"] = math.sqrt(sum((x - mean_rr) ** 2 for x in h) / (len(h) - 1))

def _acc_xyz(val):
    if isinstance(val, (list, tuple)):
        return val[0], val[1], val[2]
    if isinstance(val, dict):
        return val.get('x', val.get('X', 0)), val.get('y', val.get('Y', 0)), val.get('z', val.get('Z', 0))
    return getattr(val, 'x', 0), getattr(val, 'y', 0), getattr(val, 'z', 0)

def polar_acc_cb(data):
    # polar_python ACCData: timestamp (ns, last sample), data (list[(x, y, z)] mG)
    samples = _first_attr(data, ("data", "samples", "acc"), [])
    if not samples: return
    stamps = acc_clock.stamps(getattr(data, "timestamp", None), len(samples))
    out = []
    for ts, val in zip(stamps, samples):
        try:
            x, y, z = _acc_xyz(val)
            out.append([ts, int(x), int(y), int(z)])
        except (ValueError, TypeError, IndexError):
            continue
    acc_ring.extend(out)

def polar_ecg_cb(data):
    # polar_python ECGData: timestamp (ns, last sample), data (list[int] µV)
    samples = _first_attr(data, ("data", "samples", "ecg", "voltages"), [])
    if not samples: return
    stamps = ecg_clock.stamps(getattr(data, "timestamp", None), len(samples))
    out = []
    for ts, val in zip(stamps, samples):
        try:
            out.append([ts, round(int(getattr(val, 'voltage', getattr(val, 'ecg_uv', val))) / 1000.0, 3)])
        except (ValueError, TypeError):
            continue
    if out:
        ecg_ring.extend(out)
        sys.stdout.write(f"\r[ POLAR ] ❤️ HR: {omni_state['polar']['hr']:3d} | ⚡ ECG: {out[-1][1]:>6.3f} mV    ")
        sys.stdout.flush()

def viatom_rx_cb(sender, data):
    if len(data) == 0: return
    
    # 0xA5 = sensor settling. This is a DISPLAY state only -- it must not touch the
    # link flag, or the write loop would exit and drop a perfectly good connection.
    if len(data) == 1 and data[0] == 0xA5:
        omni_state["viatom"]["status"] = "Calibrating"
        return

    # Real-time frames start with 0x55; continuation fragments of a longer frame
    # don't, and parsing them at fixed offsets produces garbage readings.
    if len(data) >= 9 and data[0] == 0x55:
        spo2_val, hr_val = int(data[7]), int(data[8])
        if 40 <= spo2_val <= 100:
            omni_state["viatom"]["spo2"] = spo2_val
            omni_state["viatom"]["hr"] = hr_val
            omni_state["viatom"]["status"] = "Connected"


async def omni_scanner():
    while True:
        # Scanning is paused while any link is being set up, and while the Polar is
        # streaming (active scans on the same adapter cause ECG/ACC drops).
        is_connecting = any(omni_state[k]["status"] == "Connecting" for k in omni_state)
        if is_connecting or omni_state["polar"]["status"] == "Connected":
            await asyncio.sleep(1)
            continue

        if any(not mac for mac in active_targets.values()):
            found = {"polar": {}, "viatom": {}}
            
            def scan_cb(device: BLEDevice, adv: AdvertisementData):
                ble_device_cache[device.address] = device
                name = (device.name or adv.local_name or "").upper()
                addr = device.address.upper()
                uuids = [u.lower() for u in (adv.service_uuids or [])]
                
                info = {"name": device.name or f"Device ({addr[-5:]})", "address": device.address, "rssi": adv.rssi or -100}
                if "POLAR H10" in name: found["polar"][addr] = info
                elif addr == VIATOM_MAC or VIATOM_SVC_UUID.lower() in uuids or any(x in name for x in ["O2", "CHECKME", "VIATOM", "BAND-WU"]): found["viatom"][addr] = info

            try:
                async with BleakScanner(scan_cb, passive=False, bluez={"adapter": GENERAL_ADAPTER}):
                    await asyncio.sleep(2.5) 
                for key in discovered_devices:
                    discovered_devices[key] = sorted(list(found[key].values()), key=lambda x: x["rssi"], reverse=True)
            except Exception as e:
                logging.error(f"Radar blocked by OS: {repr(e)}")
        await asyncio.sleep(2)


async def polar_worker():
    global polar_last_heartbeat
    while True:
        target_mac = active_targets["polar"]
        if not target_mac or omni_state["polar"]["status"] in ["Connected", "Connecting"]:
            await asyncio.sleep(1); continue
            
        omni_state["polar"]["status"] = "Connecting"
        try:
            device = ble_device_cache.get(target_mac)
            if device:
                async with PolarDevice(device) as p:
                    ecg_clock.reset()
                    acc_clock.reset()
                    omni_state["polar"]["status"] = "Connected"
                    polar_last_heartbeat = time.time()
                    
                    logging.info("Polar handshake complete. Activating Heart Rate Matrix...")
                    try: await p.start_hr_stream(polar_hr_cb)
                    except Exception as e: logging.error(f"[DIAGNOSTIC] HR Error: {e}")
                    await asyncio.sleep(1.5) 
                    
                    logging.info("Activating Kinematics...")
                    try:
                        try: await p.start_acc_stream(polar_acc_cb, 200, 16, 8)
                        except TypeError: await p.start_acc_stream(polar_acc_cb)
                    except Exception as e: logging.error(f"[DIAGNOSTIC] ACC Error: {e}")
                    await asyncio.sleep(1.5)

                    logging.info("Activating ECG...")
                    try:
                        try: await p.start_ecg_stream(polar_ecg_cb, 130, 14)
                        except TypeError: await p.start_ecg_stream(polar_ecg_cb)
                    except Exception as e: logging.error(f"[DIAGNOSTIC] ECG Error: {e}")
                    
                    while time.time() - polar_last_heartbeat < 10.0 and omni_state["polar"]["status"] == "Connected":
                        await asyncio.sleep(1)
                    logging.warning("\n⚠️ POLAR DROPPED: Watchdog timeout.")
            else:
                logging.warning(f"\n⚠️ Target not found in radar cache. Retrying...")
                
        except Exception as e: logging.error(f"Polar Error: {repr(e)}")
        
        omni_state["polar"]["status"] = "Disconnected"
        active_targets["polar"] = None
        sys.stdout.write("\n")
        await asyncio.sleep(2)


async def viatom_worker():
    global viatom_link_up

    def handle_disconnect(client):
        global viatom_link_up
        logging.warning("\n⚠️ VIATOM DROPPED: Hardware disconnect detected.")
        viatom_link_up = False

    write_bytes = bytearray([0xAA, 0x17, 0xE8, 0x00, 0x00, 0x00, 0x00, 0x1B])

    while True:
        target_mac = active_targets["viatom"]
        if not target_mac or omni_state["viatom"]["status"] != "Disconnected":
            await asyncio.sleep(1); continue

        omni_state["viatom"]["status"] = "Connecting"
        try:
            device = ble_device_cache.get(target_mac, target_mac)
            async with BleakClient(device, timeout=15.0, disconnected_callback=handle_disconnect) as client:
                viatom_link_up = True
                logging.info("Viatom connected. Waiting 3s for GATT table to boot...")
                await asyncio.sleep(3.0)

                svc = client.services.get_service(VIATOM_SVC_UUID)
                chars = svc.characteristics if svc else []
                notify_uuid = next((c.uuid for c in chars if "notify" in c.properties), None)
                write_uuid = next((c.uuid for c in chars if "write" in c.properties or "write-without-response" in c.properties), VIATOM_WRITE_UUID)
                if not notify_uuid:
                    raise RuntimeError("no notify characteristic in Viatom service (GATT not resolved)")

                await client.start_notify(notify_uuid, viatom_rx_cb)
                omni_state["viatom"]["status"] = "Connected"

                use_response = True
                while viatom_link_up and client.is_connected:
                    try:
                        await client.write_gatt_char(write_uuid, write_bytes, response=use_response)
                    except Exception:
                        use_response = not use_response  # some firmwares only accept one mode
                    await asyncio.sleep(2)
        except Exception as e: logging.error(f"Viatom Error: {repr(e)}")

        viatom_link_up = False
        omni_state["viatom"]["status"] = "Disconnected"
        active_targets["viatom"] = None
        await asyncio.sleep(2)

async def async_master():
    await asyncio.gather(omni_scanner(), polar_worker(), viatom_worker())

def start_ble_engine():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_master())

if __name__ == "__main__":
    ble_thread = threading.Thread(target=start_ble_engine, daemon=True)
    ble_thread.start()
    logging.info("OMNI-DASH LIVE: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
