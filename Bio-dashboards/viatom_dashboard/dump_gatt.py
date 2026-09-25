# dump_gatt.py
import asyncio
from bleak import BleakClient

TARGET_MAC = "F3:A0:A8:E3:F5:63"

async def run():
    print(f"Connecting to {TARGET_MAC} to map services...")
    async with BleakClient(TARGET_MAC, timeout=15.0) as client:
        if client.is_connected:
            print("\n=== GATT PROFILE MAP DISCOVERED ===")
            for service in client.services:
                print(f"\n[Service] {service.uuid} ({service.description})")
                for char in service.characteristics:
                    print(f"  └── [Characteristic] {char.uuid} | Properties: {char.properties}")
            print("\n===================================")
        else:
            print("Failed to connect.")

if __name__ == "__main__":
    asyncio.run(run())
