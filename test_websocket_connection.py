#!/usr/bin/env python3
"""
Test script to check if the WebSocket server is accessible
Run this while the integrated_emotion_server.py is running
"""

import websocket
import json
import time

def test_websocket_connection():
    """Test connection to the WebSocket server"""

    def on_message(ws, message):
        print(f"📨 Received: {message[:200]}...")

    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("🔌 WebSocket connection closed")

    def on_open(ws):
        print("✅ WebSocket connection opened!")
        print("🎯 Server is accessible. Your Google Meet bot should be able to connect.")

        # Send a test message
        test_msg = {
            "event": "test",
            "data": {"message": "Hello from test script"}
        }
        ws.send(json.dumps(test_msg))
        print("📤 Sent test message")

        # Close after a short time
        time.sleep(2)
        ws.close()

    # Test local connection
    try:
        print("🔍 Testing local WebSocket connection...")
        ws = websocket.WebSocketApp("ws://localhost:5003",
                                   on_open=on_open,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)

        ws.run_forever()
    except Exception as e:
        print(f"❌ Failed to connect to local WebSocket: {e}")

if __name__ == "__main__":
    print("🧪 Testing WebSocket Connection")
    print("Make sure integrated_emotion_server.py is running first!")
    print("=" * 50)

    test_websocket_connection()
