import asyncio
import os
import glob
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_advanced")
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")
STATUS_FILE = os.path.join(LOGS_DIR, "status.json")

class ConnectRequest(BaseModel):
    address: str

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polar H10 Advanced Research Lab</title>
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
                <h1 class="text-xl font-bold text-rose-500 tracking-wide">Polar H10 Biometric Lab</h1>
                <p class="text-xs text-slate-400 mt-1">ECG (130Hz) | ACC (200Hz) | Live HRV</p>
            </div>
            <span id="status" class="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                Awaiting Data...
            </span>
        </div>
    </header>

    <main class="container mx-auto px-4 py-6 flex-grow max-w-7xl flex flex-col gap-6">
        
        <div id="scanner-panel" class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl w-full transition-all duration-500">
            <h2 class="text-lg font-bold text-slate-200 mb-4 flex items-center gap-3">
                <span class="relative flex h-3 w-3">
                  <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
                  <span class="relative inline-flex rounded-full h-3 w-3 bg-sky-500"></span>
                </span>
                Scanning for Polar Units...
            </h2>
            <div id="device-list" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                <div class="text-slate-500 text-sm animate-pulse">Initializing Bluetooth Radar...</div>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl text-center flex flex-col justify-center">
                <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Heart Rate</h2>
                <div class="text-6xl font-extrabold text-rose-500 my-2 tracking-tight">
                    <span id="hr-val">--</span><span class="text-2xl font-light text-slate-500 ml-2">BPM</span>
                </div>
            </div>
            
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl text-center flex flex-col justify-center">
                <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">ECG Potential</h2>
                <div class="text-6xl font-extrabold text-teal-400 my-2 tracking-tight">
                    <span id="ecg-val">0.00</span><span class="text-2xl font-light text-slate-500 ml-2">mV</span>
                </div>
            </div>

            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl text-center flex flex-col justify-center">
                <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Autonomic Stress (RMSSD)</h2>
                <div class="text-6xl font-extrabold text-indigo-400 my-2 tracking-tight">
                    <span id="rmssd-val">--</span><span class="text-2xl font-light text-slate-500 ml-2">ms</span>
                </div>
                <div id="stress-label" class="text-xs font-bold mt-1 text-slate-500 animate-pulse">GATHERING BEATS...</div>
            </div>
        </div>

        <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-xl w-full">
            <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Live PQRST Waveform</h2>
            <div class="relative h-48 w-full"><canvas id="ecgChart"></canvas></div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-xl w-full">
                <h2 class="text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">Chest Kinematics (mg)</h2>
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
                // PMD frames arrive ~every 0.5 s, so the delay must exceed that for a smooth scroll
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

        const $ = (id) => document.getElementById(id);
        const statusEl = $('status');
        const scannerPanel = $('scanner-panel');
        const hrEl = $('hr-val'), ecgEl = $('ecg-val'), rmssdEl = $('rmssd-val'), labelEl = $('stress-label');

        const BADGE = 'px-3 py-1 rounded-full text-xs font-semibold border ';
        function setStatus(text, tone) {
            statusEl.textContent = text;
            statusEl.className = BADGE + {
                ok: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
                bad: 'bg-red-500/20 text-red-400 border-red-500/30',
                wait: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
            }[tone];
        }

        // "Streaming" is driven by data freshness, not by the socket, so the scanner
        // comes back on its own when the strap drops and the worker returns to scanning.
        const STALE_MS = 5000;
        let uiIsConnected = false;
        let lastPacketAt = 0;

        function markLive() {
            lastPacketAt = Date.now();
            if (!uiIsConnected) {
                uiIsConnected = true;
                scannerPanel.style.display = 'none';
                setStatus('Telemetry Active 🟢', 'ok');
            }
        }
        function markIdle(text, tone) {
            uiIsConnected = false;
            scannerPanel.style.display = 'block';
            setStatus(text, tone);
            updateScannerList();
        }
        setInterval(() => {
            if (uiIsConnected && Date.now() - lastPacketAt > STALE_MS) markIdle('Signal Lost — Rescanning', 'wait');
        }, 1000);

        const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

        const deviceList = $('device-list');
        const note = (html, cls) => { deviceList.innerHTML = `<div class="${cls} col-span-3">${html}</div>`; };
        let lastListKey = '';

        // The worker reports scanning / connecting / streaming; the panel follows it,
        // so a poll mid-handshake can't redraw stale device cards over the progress message.
        async function updateScannerList() {
            if (uiIsConnected) return;
            try {
                const st = await (await fetch('/api/status')).json();
                if (st.state === 'connecting') {
                    lastListKey = '';
                    const tries = st.attempt > 1 ? ` (attempt ${st.attempt}/${st.max_attempts})` : '';
                    return note(`Handshaking with ${esc(st.address)}${tries}...`, 'text-emerald-400 font-bold py-4 animate-pulse');
                }
                if (st.state === 'streaming') {
                    lastListKey = '';
                    return note('Connected — starting ECG / ACC streams...', 'text-emerald-400 font-bold py-4 animate-pulse');
                }

                const devices = await (await fetch('/api/scan-results')).json();
                const key = JSON.stringify(devices.map(d => d.address));
                if (key === lastListKey) {
                    // same devices: just refresh RSSI in place, no DOM rebuild
                    devices.forEach(d => {
                        const el = deviceList.querySelector(`[data-rssi="${CSS.escape(d.address)}"]`);
                        if (el) el.textContent = `${d.rssi} dBm`;
                    });
                    return;
                }
                lastListKey = key;
                if (devices.length === 0) {
                    return note('No active straps detected... ensure pads are wet.', 'text-slate-500 text-sm animate-pulse');
                }
                deviceList.innerHTML = devices.map(d => `
                    <div class="bg-slate-700/50 p-4 rounded-xl border border-slate-600 flex flex-col gap-3 justify-between">
                        <div>
                            <div class="font-bold text-slate-200">${esc(d.name)}</div>
                            <div class="text-xs text-slate-400 font-mono">${esc(d.address)} · <span data-rssi="${esc(d.address)}">${esc(d.rssi)} dBm</span></div>
                        </div>
                        <button data-mac="${esc(d.address)}" class="bg-sky-600 hover:bg-sky-500 text-white text-sm font-bold py-2 px-4 rounded w-full transition-colors">
                            Connect Signal
                        </button>
                    </div>`).join('');
            } catch (err) {}
        }

        deviceList.addEventListener('click', async (e) => {
            const mac = e.target.closest('button[data-mac]')?.dataset.mac;
            if (!mac) return;
            lastListKey = '';
            note(`Handshaking with ${esc(mac)}...`, 'text-emerald-400 font-bold py-4 animate-pulse');
            await fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({address: mac})
            });
        });

        setInterval(updateScannerList, 1000);
        updateScannerList();

        // --- HRV ---
        const ppiHistory = [];
        function pushPpi(ppi) {
            ppiHistory.push(ppi);
            if (ppiHistory.length > 20) ppiHistory.shift();
            if (ppiHistory.length <= 2) return;

            let sum = 0;
            for (let i = 1; i < ppiHistory.length; i++) {
                const d = ppiHistory[i] - ppiHistory[i - 1];
                sum += d * d;
            }
            const rmssd = Math.sqrt(sum / (ppiHistory.length - 1));
            rmssdEl.textContent = rmssd.toFixed(1);

            const [text, color] = rmssd < 20 ? ['HIGH STRESS (SYMPATHETIC)', 'text-rose-500']
                                : rmssd < 50 ? ['MODERATE (BALANCED)', 'text-amber-400']
                                : ['RELAXED (PARASYMPATHETIC)', 'text-emerald-400'];
            labelEl.textContent = text;
            labelEl.className = 'text-xs font-bold mt-1 ' + color;
        }

        // --- Stream handling: each message is a batch {type, rows: [[...], ...]} ---
        const ecgData = ecgChart.data.datasets[0].data;
        const [accX, accY, accZ] = accChart.data.datasets.map(d => d.data);
        const ppiData = ppiChart.data.datasets[0].data;

        const handlers = {
            ecg(rows) {
                for (const [ts, mv] of rows) ecgData.push({ x: ts, y: mv });
                const [, mv, hr] = rows[rows.length - 1];
                ecgEl.textContent = mv.toFixed(2);
                if (hr > 0) hrEl.textContent = hr;
            },
            acc(rows) {
                for (const [ts, x, y, z] of rows) {
                    accX.push({ x: ts, y: x });
                    accY.push({ x: ts, y: y });
                    accZ.push({ x: ts, y: z });
                }
            },
            ppi(rows) {
                for (const [ts, ppi, hr] of rows) {
                    ppiData.push({ x: ts, y: ppi });
                    pushPpi(ppi);
                    if (hr > 0) hrEl.textContent = hr;
                }
            },
        };

        function connectWs() {
            const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
            ws.onopen = () => { if (!uiIsConnected) setStatus('System Online', 'wait'); };
            ws.onclose = () => {
                markIdle('Disconnected 🔴 — retrying', 'bad');
                setTimeout(connectWs, 2000);
            };
            ws.onmessage = (event) => {
                const packet = JSON.parse(event.data);
                const h = handlers[packet.type];
                if (h && packet.rows && packet.rows.length) {
                    markLive();
                    h(packet.rows);
                }
            };
        }
        connectWs();
    </script>
</body>
</html>
"""

class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: str):
        conns = list(self.active_connections)
        results = await asyncio.gather(*(c.send_text(message) for c in conns), return_exceptions=True)
        for c, r in zip(conns, results):
            if isinstance(r, Exception):
                self.disconnect(c)

manager = ConnectionManager()

# CSV row layouts written by advanced_worker.py -> (column count, per-column converters)
ROW_SCHEMAS = {
    "ecg": (3, (int, float, int)),   # ts_ms, mV, hr
    "acc": (4, (int, float, float, float)),  # ts_ms, x, y, z (mG)
    "ppi": (3, (int, int, int)),     # ts_ms, ppi_ms, hr
}

POLL_S = 0.02          # how often to check the file for new bytes
ROTATE_CHECK_S = 2.0   # how often to look for a newer log file (worker restart)

def get_latest_file(prefix):
    files = glob.glob(os.path.join(LOGS_DIR, f"{prefix}_*.csv"))
    return max(files, key=os.path.getmtime) if files else None

def parse_rows(lines, stream_type):
    ncols, conv = ROW_SCHEMAS[stream_type]
    rows = []
    for line in lines:
        parts = line.split(",")
        if len(parts) != ncols:
            continue
        try:
            rows.append([c(p) for c, p in zip(conv, parts)])
        except ValueError:
            continue  # header row or garbage
    return rows

async def tail_file_and_broadcast(prefix, stream_type):
    """Follows the newest `prefix_*.csv`, sending everything that arrived since the
    last poll as ONE websocket message instead of one message per sample."""
    loop = asyncio.get_running_loop()
    current, f, partial = None, None, ""
    next_rotate_check = 0.0
    try:
        while True:
            now = loop.time()
            if now >= next_rotate_check:
                next_rotate_check = now + ROTATE_CHECK_S
                latest = get_latest_file(prefix)
                if latest and latest != current:
                    first_lock = current is None
                    if f:
                        f.close()
                    f = open(latest, "r")
                    # On startup, skip past a file that already holds a lot (don't replay an old
                    # session), but read a fresh one from the top so nothing is dropped.
                    if first_lock and os.path.getsize(latest) > 64 * 1024:
                        f.seek(0, os.SEEK_END)
                    current, partial = latest, ""
                    print(f"📡 Router locked onto {stream_type.upper()}: {latest}")

            if f is None:
                await asyncio.sleep(1)
                continue

            chunk = f.read()
            if not chunk:
                await asyncio.sleep(POLL_S)
                continue

            # Keep any half-written trailing line for the next read
            data = partial + chunk
            lines = data.split("\n")
            partial = lines.pop()

            if manager.active_connections:
                rows = parse_rows((l.strip() for l in lines if l.strip()), stream_type)
                if rows:
                    await manager.broadcast(json.dumps({"type": stream_type, "rows": rows}, separators=(",", ":")))
    finally:
        if f:
            f.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass

    tasks = [
        asyncio.create_task(tail_file_and_broadcast("polar_ecg", "ecg")),
        asyncio.create_task(tail_file_and_broadcast("polar_acc", "acc")),
        asyncio.create_task(tail_file_and_broadcast("polar_ppi", "ppi")),
    ]
    yield
    for t in tasks:
        t.cancel()

# --- APP INITIALIZATION ---
app = FastAPI(lifespan=lifespan)

# --- ENDPOINTS ---
@app.get("/api/scan-results")
def get_scan_results():
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, "r") as f:
                return json.load(f)
        except Exception: pass
    return []

@app.get("/api/status")
def get_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "scanning"}

@app.post("/api/connect")
def connect_device(req: ConnectRequest):
    os.makedirs(LOGS_DIR, exist_ok=True)
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"action": "connect", "address": req.address}, f)
    os.replace(tmp, COMMAND_FILE)  # atomic, so the worker never reads a half-written command
    return {"status": "command_sent"}

@app.get("/")
async def get(): return HTMLResponse(HTML_TEMPLATE)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=False)
