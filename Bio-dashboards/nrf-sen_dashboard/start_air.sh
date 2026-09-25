#!/bin/bash

cd "$(dirname "$0")"

echo "============================================="
echo "   Starting SEN69C Air Quality Lab           "
echo "============================================="

echo "📦 Archiving previous telemetry logs..."
mkdir -p logs_air/archive
mv logs_air/*.csv logs_air/archive/ 2>/dev/null

echo "[BLE] Launching hardware engine (sensor_worker.py)..."
python3 sensor_worker.py &
WORKER_PID=$!

sleep 1.5

echo "[WEB] Launching dashboard interface (app.py)..."
python3 app.py &
WEB_PID=$!

cleanup() {
    echo -e "\n\n============================================="
    echo "   Shutting down Air Lab...                  "
    echo "============================================="
    kill $WEB_PID 2>/dev/null
    kill $WORKER_PID 2>/dev/null
    echo "✅ All processes terminated successfully."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "\n✅ System is LIVE!"
echo -e "🔗 Open your browser to: http://localhost:5002"
echo -e "⏳ Press [Ctrl + C] to stop both services.\n"

wait
