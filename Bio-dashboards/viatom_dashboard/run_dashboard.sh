#!/bin/bash

# Navigate to the script's directory to ensure relative paths stay intact
cd "$(dirname "$0")"

echo "============================================="
echo "   Starting Viatom Checkme O2 Ultra Dashboard "
echo "============================================="

# 1. Start the BLE Worker in the background
echo "[BLE] Launching biometric background worker..."
python3 ble_worker.py &
BLE_PID=$!

# Give the Bluetooth engine a brief moment to initialize before booting the web layer
sleep 1.5

# 2. Start the Flask Application in the foreground
echo "[WEB] Launching dashboard interface web server..."
python3 app.py &
WEB_PID=$!

# Trap Ctrl+C (SIGINT) and exit signals to kill both background jobs cleanly
cleanup() {
    echo -e "\n\n============================================="
    echo "   Shutting down Viatom Dashboard Engines...   "
    echo "============================================="
    
    echo "[WEB] Stopping web server (PID: $WEB_PID)..."
    kill $WEB_PID 2>/dev/null
    
    echo "[BLE] Stopping Bluetooth worker (PID: $BLE_PID)..."
    kill $BLE_PID 2>/dev/null
    
    echo "✅ All processes terminated successfully."
    exit 0
}

# Assign the cleanup function to handle termination traps
trap cleanup SIGINT SIGTERM

# Keep the script alive so it continues to intercept the trap signals
wait
