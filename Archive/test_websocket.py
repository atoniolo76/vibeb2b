#!/usr/bin/env python3
"""
Test WebSocket connection to our server using websocket-client
"""

import websocket
import json
import time

def test_connection():
    """Test connection and send a sample message"""
    try:
        # Create WebSocket connection
        ws = websocket.create_connection("ws://localhost:5003")
        print("✅ Connected to WebSocket server!")

        # Send a test message like Google Meet would
        test_message = {
            "event": "video_separate_png.data",
            "data": {
                "data": {
                    "buffer": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "timestamp": {"absolute": "2024-01-01T00:00:00Z"},
                    "participant": {"name": "Test User", "id": 123}
                },
                "recording": {"id": "test_recording"}
            }
        }

        ws.send(json.dumps(test_message))
        print("✅ Test message sent!")

        # Wait a bit for processing
        time.sleep(1)

        # Try to receive response (if any)
        try:
            result = ws.recv()
            print(f"📨 Received response: {result}")
        except:
            print("📨 No response received (normal)")

        ws.close()
        print("✅ Test completed!")

    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

if __name__ == '__main__':
    test_connection()
