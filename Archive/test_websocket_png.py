#!/usr/bin/env python3
"""
Test WebSocket connection to PNG test server
Run this to verify your WebSocket setup works
"""

import websocket
import json
import time
import base64
import numpy as np
import cv2

def create_test_png():
    """Create a small test PNG image"""
    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :] = [255, 0, 0]  # Red square

    # Add some text
    cv2.putText(test_image, "TEST", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Convert to PNG bytes
    success, png_buffer = cv2.imencode('.png', test_image)
    if success:
        png_bytes = png_buffer.tobytes()
        png_b64 = base64.b64encode(png_bytes).decode('utf-8')
        return png_b64
    return None

def test_websocket_connection():
    """Test WebSocket connection and send a test PNG"""

    def on_message(ws, message):
        print(f"📨 Server response: {message[:100]}...")

    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("🔌 Connection closed")

    def on_open(ws):
        print("✅ Connected to PNG test server!")
        print("📤 Sending test PNG frame...")

        # Create test PNG
        test_png = create_test_png()
        if test_png:
            # Send test message in correct Google Meet bot format
            test_message = {
                "event": "video_separate_png.data",
                "data": {
                    "data": {
                        "participant": {"name": "TestUser"},
                        "recording": {"id": "test_recording_123"},
                        "buffer": test_png
                    }
                }
            }

            ws.send(json.dumps(test_message))
            print("📤 Test PNG sent!")
            print("   Check server console for: '🖼️ PNG RECEIVED from: TestUser'")
        else:
            print("❌ Failed to create test PNG")

        # Close after sending
        time.sleep(2)
        ws.close()

    # Test connection
    try:
        print("🔍 Testing WebSocket connection to PNG test server...")
        print("Make sure png_test_server.py is running first!")
        print()

        ws = websocket.WebSocketApp(
            "ws://localhost:5003",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        ws.run_forever()

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure png_test_server.py is running on port 5003")

if __name__ == "__main__":
    print("🧪 TESTING WEBSOCKET CONNECTION TO PNG SERVER")
    print("=" * 50)
    print()
    test_websocket_connection()
    print()
    print("🎯 If successful, you should see:")
    print("   ✅ Connected to PNG test server!")
    print("   📤 Test PNG sent!")
    print("   🖼️ PNG RECEIVED from: TestUser (in server console)")
