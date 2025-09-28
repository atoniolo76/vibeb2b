#!/usr/bin/env python3
"""
Simple WebSocket test - just connect and send a basic message
"""

import websocket
import json
import time

def test_simple():
    def on_message(ws, message):
        print(f"📨 Received: {message}")

    def on_error(ws, error):
        print(f"❌ Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("🔌 Closed")

    def on_open(ws):
        print("✅ Connected!")
        # Send a simple test message
        ws.send(json.dumps({"test": "hello", "timestamp": time.time()}))
        print("📤 Sent simple test message")
        time.sleep(1)
        ws.close()

    try:
        ws = websocket.WebSocketApp(
            "ws://localhost:5003",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("🔍 Testing simple WebSocket connection...")
    test_simple()
