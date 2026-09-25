# app.py
import json
import os
import time
from flask import Flask, Response, jsonify, request, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
DEVICES_FILE = os.path.join(BASE_DIR, "devices.json")
COMMAND_FILE = os.path.join(BASE_DIR, "command.json")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Bio-Dash Live Telemetry</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.0.1/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.2.0/dist/chartjs-adapter-luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-streaming@2.0.0/dist/chartjs-plugin-streaming.min.js"></script>

    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #121212; color: #e0e0e0; padding: 2rem;
            display: flex; flex-direction: column; align-items: center; min-height: 100vh; margin: 0;
        }
        .container { display: flex; flex-direction: row; gap: 2rem; max-width: 1100px; width: 100%; justify-content: center; flex-wrap: wrap; }
        .left-panel { display: flex; flex-direction: column; gap: 1.5rem; width: 340px; }
        .right-panel { flex: 1; min-width: 450px; display: flex; flex-direction: column; gap: 1.5rem; }
        
        .dashboard { background-color: #1e1e1e; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); text-align: center; }
        h1 { margin-top: 0; font-size: 1.8rem; color: #ffffff; }
        .metric-card { background: #252525; padding: 1.2rem; margin: 1rem 0; border-radius: 12px; border-left: 5px solid #444; }
        .metric-card.spo2 { border-left-color: #38bdf8; }
        .metric-card.hr { border-left-color: #f43f5e; }
        .value { font-size: 2.5rem; font-weight: bold; margin-top: 0.3rem; }
        .status { font-size: 0.9rem; color: #a3a3a3; margin-top: 1rem; }
        
        .graph-card { background-color: #1e1e1e; padding: 1.5rem; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); height: 220px; }
        .device-selector { background-color: #1e1e1e; padding: 1.5rem; border-radius: 16px; box-sizing: border-box; }
        h2 { font-size: 1.1rem; margin-top: 0; color: #ffffff; }
        .device-item { background: #252525; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; }
        .btn { background-color: #2563eb; color: white; border: none; padding: 0.4rem 0.8rem; border-radius: 6px; cursor: pointer; font-weight: bold; }
        .btn:hover { background-color: #1d4ed8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <div class="dashboard">
                <h1>Bio-Dash</h1>
                <div class="metric-card spo2">
                    <div style="color: #38bdf8; font-size: 0.85rem; font-weight: bold;">BLOOD OXYGEN</div>
                    <div class="value" id="spo2-val">--%</div>
                </div>
                <div class="metric-card hr">
                    <div style="color: #f43f5e; font-size: 0.85rem; font-weight: bold;">HEART RATE</div>
                    <div class="value" id="hr-val">-- BPM</div>
                </div>
                <div id="status-val" class="status">Device Status: Initializing...</div>
            </div>

            <div class="device-selector" id="selector-panel" style="display: none;">
                <h2>Select an O₂ Monitor</h2>
                <div id="device-list"></div>
            </div>
        </div>

        <div class="right-panel">
            <div class="graph-card">
                <canvas id="spo2Chart"></canvas>
            </div>
            <div class="graph-card">
                <canvas id="hrChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Common configuration options for fluid charting steps
        const chartOptions = (titleColor, gridColor) => ({
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { type: 'realtime', realtime: { duration: 30000, refresh: 100, delay: 500 } },
                y: { grid: { color: '#2d2d2d' }, ticks: { color: '#a3a3a3' } }
            },
            plugins: { legend: { display: false } }
        });

        // Initialize Oxygen Graph
        const ctxSpO2 = document.getElementById('spo2Chart').getContext('2d');
        const spo2Chart = new Chart(ctxSpO2, {
            type: 'line',
            data: { datasets: [{ label: 'SpO2', borderColor: '#38bdf8', borderWidth: 3, pointRadius: 0, data: [] }] },
            options: {
                ...chartOptions(),
                scales: { ...chartOptions().scales, y: { min: 80, max: 100, grid: { color: '#2d2d2d' }, ticks: { color: '#a3a3a3' } } }
            }
        });

        // Initialize Heart Rate Graph
        const ctxHR = document.getElementById('hrChart').getContext('2d');
        const hrChart = new Chart(ctxHR, {
            type: 'line',
            data: { datasets: [{ label: 'Heart Rate', borderColor: '#f43f5e', borderWidth: 3, pointRadius: 0, data: [] }] },
            options: {
                ...chartOptions(),
                scales: { ...chartOptions().scales, y: { min: 40, max: 120, grid: { color: '#2d2d2d' }, ticks: { color: '#a3a3a3' } } }
            }
        });

        const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

        // Handle incoming data packets inside the real-time SSE pipe
        const eventSource = new EventSource("/api/vitals-stream");
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const now = Date.now();
            
            document.getElementById('status-val').innerText = 'Device Status: ' + data.status;
            
            if (data.status === "Connected" && data.spo2 > 0) {
                document.getElementById('spo2-val').innerText = data.spo2 + '%';
                document.getElementById('hr-val').innerText = data.hr + ' BPM';
                
                // Push points directly into the active chart buffers instantly
                spo2Chart.data.datasets[0].data.push({ x: now, y: data.spo2 });
                hrChart.data.datasets[0].data.push({ x: now, y: data.hr });
            } else {
                document.getElementById('spo2-val').innerText = '--%';
                document.getElementById('hr-val').innerText = '-- BPM';
            }
            
            if (data.status === "Scanning") {
                document.getElementById('selector-panel').style.display = 'block';
            } else {
                document.getElementById('selector-panel').style.display = 'none';
            }
        };

        async function updateDeviceList() {
            try {
                const response = await fetch('/api/scan-results');
                const devices = await response.json();
                const container = document.getElementById('device-list');
                container.innerHTML = '';
                if (devices.length === 0) {
                    container.innerHTML = '<div style="text-align:center;color:#737373;padding:0.5rem;">Searching for wrist devices...</div>';
                    return;
                }
                devices.forEach(d => {
                    const div = document.createElement('div');
                    div.className = 'device-item';
                    div.innerHTML = `
                        <div><strong>${esc(d.name)}</strong><br><span style="font-size:0.75rem;color:#a3a3a3;">${esc(d.address)} · ${esc(d.rssi)} dBm</span></div>
                        <button class="btn" data-mac="${esc(d.address)}">Connect</button>
                    `;
                    container.appendChild(div);
                });
            } catch (err) {}
        }

        async function connectDevice(mac) {
            await fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({address: mac})
            });
        }

        document.getElementById('device-list').addEventListener('click', (e) => {
            const b = e.target.closest('button[data-mac]');
            if (b) connectDevice(b.dataset.mac);
        });

        setInterval(updateDeviceList, 3000);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/scan-results')
def scan_results():
    if os.path.exists(DEVICES_FILE):
        try:
            with open(DEVICES_FILE, 'r') as f: return jsonify(json.load(f))
        except Exception: pass
    return jsonify([])

@app.route('/api/connect', methods=['POST'])
def connect():
    target_mac = request.json.get('address')
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({"action": "connect", "address": target_mac}, f)
    os.replace(tmp, COMMAND_FILE)
    return jsonify({"status": "command_sent"})

@app.route('/api/vitals-stream')
def vitals_stream():
    def event_stream():
        last_mtime = 0
        while True:
            if os.path.exists(DATA_FILE):
                try:
                    current_mtime = os.path.getmtime(DATA_FILE)
                    if current_mtime != last_mtime:
                        last_mtime = current_mtime
                        with open(DATA_FILE, 'r') as f: yield f"data: {f.read()}\n\n"
                except Exception: pass
            time.sleep(0.05)
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
