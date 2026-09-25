import asyncio
import os
import glob
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs_air")
DEVICES_FILE = os.path.join(LOGS_DIR, "devices.json")
COMMAND_FILE = os.path.join(LOGS_DIR, "command.json")

class ConnectRequest(BaseModel):
    address: str

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperMini Air Monitor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 1rem;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans flex flex-col">

    <div class="max-w-4xl mx-auto px-4 py-8 flex-grow w-full">
        <header class="flex flex-col sm:flex-row justify-between items-center border-b border-slate-700 pb-6 mb-8 gap-4">
            <div>
                <h1 class="text-3xl font-bold tracking-tight text-emerald-400">SEN69C Environmental Monitor</h1>
                <p class="text-slate-400 text-sm mt-1">Local CSV Logging Dashboard</p>
            </div>
            <div id="status-badge" class="px-4 py-2 rounded-full text-sm font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">
                Awaiting Telemetry...
            </div>
        </header>

        <div id="scanner-panel" class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl w-full transition-all duration-500 mb-8">
            <h2 class="text-lg font-bold text-slate-200 mb-4 flex items-center gap-3">
                <span class="relative flex h-3 w-3">
                  <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                </span>
                Scanning for Air Monitors...
            </h2>
            <div id="device-list" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                <div class="text-slate-500 text-sm animate-pulse">Initializing Bluetooth Radar...</div>
            </div>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
            <div class="metric-card md:col-span-3 border-l-4 border-cyan-500">
                <h3 class="text-xs font-bold uppercase tracking-wider text-cyan-400 mb-4">Particulate Matter (µg/m³)</h3>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div><span class="text-slate-400 text-xs block">PM 1.0</span><span id="pm1" class="text-2xl font-semibold">--</span></div>
                    <div><span class="text-slate-400 text-xs block">PM 2.5</span><span id="pm25" class="text-2xl font-semibold text-cyan-300">--</span></div>
                    <div><span class="text-slate-400 text-xs block">PM 4.0</span><span id="pm4" class="text-2xl font-semibold">--</span></div>
                    <div><span class="text-slate-400 text-xs block">PM 10.0</span><span id="pm10" class="text-2xl font-semibold">--</span></div>
                </div>
            </div>

            <div class="metric-card border-l-4 border-amber-500">
                <span class="text-xs font-bold uppercase tracking-wider text-amber-400 block mb-1">Carbon Dioxide</span>
                <span id="co2" class="text-4xl font-extrabold block my-2">--</span>
                <span class="text-xs text-slate-400">PPM</span>
            </div>

            <div class="metric-card border-l-4 border-red-500">
                <span class="text-xs font-bold uppercase tracking-wider text-red-400 block mb-1">Formaldehyde (HCHO)</span>
                <span id="hcho" class="text-4xl font-extrabold block my-2">--</span>
                <span class="text-xs text-slate-400">PPB</span>
            </div>

            <div class="metric-card border-l-4 border-emerald-500">
                <span class="text-xs font-bold uppercase tracking-wider text-emerald-400 block mb-1">Temperature</span>
                <span id="temp" class="text-4xl font-extrabold block my-2">--</span>
                <span class="text-xs text-slate-400">°C</span>
            </div>

            <div class="metric-card border-l-4 border-blue-500">
                <span class="text-xs font-bold uppercase tracking-wider text-blue-400 block mb-1">Relative Humidity</span>
                <span id="humidity" class="text-4xl font-extrabold block my-2">--</span>
                <span class="text-xs text-slate-400">% RH</span>
            </div>

            <div class="metric-card border-l-4 border-purple-500">
                <span class="text-xs font-bold uppercase tracking-wider text-purple-400 block mb-1">VOC Index</span>
                <span id="voc" class="text-4xl font-extrabold block my-2">--</span>
                <span class="text-xs text-slate-400">1 - 500</span>
            </div>

            <div class="metric-card border-l-4 border-indigo-500">
                <span class="text-xs font-bold uppercase tracking-wider text-indigo-400 block mb-1">NOx Index</span>
                <span id="nox" class="text-4xl font-extrabold block my-2">--</span>
                <span class="text-xs text-slate-400">1 - 500</span>
            </div>
        </div>
    </div>

    <script>
        const $ = (id) => document.getElementById(id);
        const statusBadge = $('status-badge');
        const scannerPanel = $('scanner-panel');
        const BADGE = 'px-4 py-2 rounded-full text-sm font-semibold border ';
        const TONES = {
            ok: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
            info: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
            wait: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
            bad: 'bg-red-500/20 text-red-400 border-red-500/30',
        };
        const setStatus = (text, tone) => { statusBadge.textContent = text; statusBadge.className = BADGE + TONES[tone]; };
        const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

        // Sensor streams ~1 line/s; if nothing for 15 s the link is gone -> show the scanner again
        const STALE_MS = 15000;
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

        async function updateScannerList() {
            if (uiIsConnected) return;
            try {
                const devices = await (await fetch('/api/scan-results')).json();
                const container = $('device-list');
                if (devices.length === 0) {
                    container.innerHTML = '<div class="text-slate-500 text-sm animate-pulse col-span-3">No active environmental monitors detected...</div>';
                    return;
                }
                container.innerHTML = devices.map(d => `
                    <div class="bg-slate-700/50 p-4 rounded-xl border border-slate-600 flex flex-col gap-3 justify-between">
                        <div>
                            <div class="font-bold text-slate-200">${esc(d.name)}</div>
                            <div class="text-xs text-slate-400 font-mono">${esc(d.address)} · ${esc(d.rssi)} dBm</div>
                        </div>
                        <button data-mac="${esc(d.address)}" class="bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold py-2 px-4 rounded w-full transition-colors">
                            Connect Signal
                        </button>
                    </div>`).join('');
            } catch (err) {}
        }

        $('device-list').addEventListener('click', async (e) => {
            const mac = e.target.closest('button[data-mac]')?.dataset.mac;
            if (!mac) return;
            $('device-list').innerHTML = `<div class="text-emerald-400 font-bold py-4 col-span-3 animate-pulse">Handshaking with ${esc(mac)}...</div>`;
            await fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({address: mac})
            });
        });

        setInterval(updateScannerList, 3000);
        updateScannerList();

        // CSV column -> element id (column 0 is the timestamp)
        const FIELDS = ['pm1', 'pm25', 'pm4', 'pm10', 'humidity', 'temp', 'voc', 'nox', 'hcho', 'co2'];
        const els = FIELDS.map($);
        const co2Card = $('co2').parentElement;

        function render(line) {
            const parts = line.split(',');
            if (parts.length < 11) return;
            markLive();
            els.forEach((el, i) => { el.textContent = parts[i + 1]; });
            co2Card.style.borderColor = parseInt(parts[10]) > 1000 ? '#ef4444' : '#f59e0b';
        }

        function connectWs() {
            const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
            ws.onopen = () => { if (!uiIsConnected) setStatus('System Online', 'info'); };
            ws.onclose = () => {
                markIdle('Disconnected 🔴 — retrying', 'bad');
                setTimeout(connectWs, 2000);
            };
            // Server may batch several lines into one message; only the newest matters for the tiles
            ws.onmessage = (event) => {
                const lines = event.data.split('\n').filter(Boolean);
                if (lines.length) render(lines[lines.length - 1]);
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

POLL_S = 0.1
ROTATE_CHECK_S = 2.0

def get_latest_file():
    files = glob.glob(os.path.join(LOGS_DIR, 'sen69c_telemetry_*.csv'))
    return max(files, key=os.path.getmtime) if files else None

async def tail_file_and_broadcast():
    loop = asyncio.get_running_loop()
    current_file, f, partial = None, None, ""
    next_rotate_check = 0.0
    try:
        while True:
            # Only glob/stat the log dir every couple of seconds, not on every poll
            now = loop.time()
            if now >= next_rotate_check:
                next_rotate_check = now + ROTATE_CHECK_S
                latest_file = get_latest_file()
                if latest_file and latest_file != current_file:
                    first_lock = current_file is None
                    if f: f.close()
                    current_file, partial = latest_file, ""
                    print(f"📡 Router locked onto: {current_file}")
                    f = open(current_file, 'r')
                    if first_lock and os.path.getsize(current_file) > 16 * 1024:
                        f.seek(0, os.SEEK_END)

            if f is None:
                await asyncio.sleep(1)
                continue

            chunk = f.read()
            if not chunk:
                await asyncio.sleep(POLL_S)
                continue

            lines = (partial + chunk).split("\n")
            partial = lines.pop()
            good = [l.strip() for l in lines if l.strip() and not l.startswith("Timestamp")]
            if good and manager.active_connections:
                await manager.broadcast("\n".join(good))
    finally:
        if f: f.close()

# --- APP INITIALIZATION ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(COMMAND_FILE):
        try: os.remove(COMMAND_FILE)
        except Exception: pass
        
    tail_task = asyncio.create_task(tail_file_and_broadcast())
    yield
    tail_task.cancel()

app = FastAPI(lifespan=lifespan)

# --- ENDPOINTS ---
@app.get("/api/scan-results")
def get_scan_results():
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, 'r') as f: 
                return json.load(f)
        except Exception: pass
    return []

@app.post("/api/connect")
def connect_device(req: ConnectRequest):
    os.makedirs(LOGS_DIR, exist_ok=True)
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({"action": "connect", "address": req.address}, f)
    os.replace(tmp, COMMAND_FILE)
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
    uvicorn.run("app:app", host="0.0.0.0", port=5002, reload=False)
