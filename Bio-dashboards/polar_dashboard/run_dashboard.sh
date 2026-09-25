#!/bin/bash

# Navigate to the script's directory to ensure relative paths stay intact
cd "$(dirname "$0")"

echo "============================================="
echo "   Starting Polar H10 Biometric Lab          "
echo "============================================="

# 1. The Archive System: Move old logs to prevent UI ghosting, but keep the data!
echo "📦 Archiving previous telemetry logs..."
mkdir -p logs_advanced/archive
mv logs_advanced/*.csv logs_advanced/archive/ 2>/dev/null

# 2. Start the Hardware Engine in the background
echo "[BLE] Launching hardware engine (advanced_worker.py)..."
python3 advanced_worker.py &
WORKER_PID=$!

# Give the Bluetooth engine a brief moment to initialize before booting the web layer
sleep 1.5

# 3. Start the Web Server in the background
echo "[WEB] Launching dashboard interface web server (app.py)..."
python3 app.py &
WEB_PID=$!

# Trap Ctrl+C (SIGINT) and exit signals to kill both background jobs cleanly
cleanup() {
    echo -e "\n\n============================================="
    echo "   Shutting down Polar H10 Lab...            "
    echo "============================================="
    
    echo "[WEB] Stopping web server (PID: $WEB_PID)..."
    kill $WEB_PID 2>/dev/null
    
    echo "[BLE] Stopping hardware engine (PID: $WORKER_PID)..."
    kill $WORKER_PID 2>/dev/null
    
    echo "✅ All processes terminated successfully."
    exit 0
}

# Assign the cleanup function to handle termination traps
trap cleanup SIGINT SIGTERM

echo -e "\n✅ System is LIVE!"
echo -e "🔗 Open your browser to: http://localhost:5001"
echo -e "⏳ Press [Ctrl + C] to stop both services.\n"

# Keep the script alive so it continues to intercept the trap signals
wait
