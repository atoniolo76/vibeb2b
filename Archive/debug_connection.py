#!/usr/bin/env python3
"""
Connection debugger for Google Meet bot setup
"""

import requests
import socket
import json
import time

def check_local_server():
    """Check if local server is running"""
    print("🔍 Checking local server...")
    try:
        response = requests.get("http://localhost:5003", timeout=5)
        if response.status_code == 200:
            print("✅ Local server is running on port 5003")
            return True
        else:
            print(f"❌ Local server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to local server: {e}")
        return False

def check_ngrok():
    """Check if ngrok is running"""
    print("\n🔍 Checking ngrok...")
    try:
        response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5)
        if response.status_code == 200:
            tunnels = response.json()['tunnels']
            for tunnel in tunnels:
                if tunnel['proto'] == 'https':
                    public_url = tunnel['public_url']
                    ws_url = public_url.replace('https://', 'wss://')
                    print(f"✅ ngrok is running!")
                    print(f"   Public URL: {public_url}")
                    print(f"   WebSocket URL: {ws_url}")
                    return ws_url
        print("❌ ngrok API not accessible")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ ngrok not running: {e}")
        return None

def test_websocket_connection(ws_url):
    """Test WebSocket connection"""
    print(f"\n🔍 Testing WebSocket connection to {ws_url}...")
    try:
        import websocket
        ws = websocket.create_connection(ws_url, timeout=10)
        print("✅ WebSocket connection successful!")
        ws.close()
        return True
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False

def check_bot_config():
    """Check bot configuration format"""
    print("\n📋 Bot Configuration Checklist:")
    print("   1. ✅ meeting_url: Should be your Google Meet URL")
    print("   2. ✅ recording_config.video_separate_png: {} (empty object)")
    print("   3. ✅ realtime_endpoints[0].type: 'websocket'")
    print("   4. ✅ realtime_endpoints[0].url: Your ngrok wss:// URL")
    print("   5. ✅ realtime_endpoints[0].events: ['video_separate_png.data']")
    print("   6. ✅ NO variant field (PNG doesn't need web_4_core)")

def simulate_bot_message():
    """Send a test message to see if receiver works"""
    print("\n🧪 Sending test message to local server...")
    test_message = {
        "event": "video_separate_png.data",
        "data": {
            "data": {
                "buffer": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 transparent PNG
                "timestamp": {"absolute": "2024-01-01T00:00:00Z"},
                "participant": {"name": "Test User", "id": 123}
            },
            "recording": {"id": "test_recording"}
        }
    }

    try:
        import websocket
        ws = websocket.create_connection("ws://localhost:5003", timeout=5)
        ws.send(json.dumps(test_message))
        print("✅ Test message sent successfully!")
        ws.close()
        return True
    except Exception as e:
        print(f"❌ Test message failed: {e}")
        return False

def main():
    print("🚀 Google Meet Bot Connection Debugger")
    print("=" * 50)

    # Check local server
    local_ok = check_local_server()

    # Check ngrok
    ws_url = check_ngrok()

    if ws_url:
        # Test WebSocket connection
        ws_ok = test_websocket_connection(ws_url)

        # Show configuration
        check_bot_config()

        print("\n📤 Tell your friend to use this WebSocket URL:")
        print(f"   {ws_url}")
    else:
        print("\n❌ ngrok not running!")
        print("   Run: ngrok http 5003")

    # Test local connection
    if local_ok:
        simulate_bot_message()

    print("\n🔧 Troubleshooting:")
    print("   1. Make sure ngrok is running: ngrok http 5003")
    print("   2. Give your friend the wss:// URL from ngrok")
    print("   3. Friend checks his bot config matches the checklist above")
    print("   4. Friend starts Google Meet and bot joins")
    print("   5. You should see connection messages in this terminal")

if __name__ == '__main__':
    main()
