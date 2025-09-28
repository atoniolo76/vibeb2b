#!/usr/bin/env python3
"""
TEST RECEIVING - Simple frame display without processing
Shows received images with timestamps for debugging
"""

from flask import Flask, jsonify
from websocket_server import WebsocketServer
import json
import base64
import threading
import time
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

class SimpleFrameReceiver:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frames_received = 0
        self.last_timestamp = None

    def update_frame_png(self, png_data):
        """Handle PNG frame data"""
        try:
            png_bytes = base64.b64decode(png_data)
            nparr = np.frombuffer(png_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.frames_received += 1
                    self.last_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                return True
        except Exception as e:
            print(f"❌ PNG decode error: {e}")
        return False

    def update_frame_h264(self, h264_data):
        """Handle H264 frame data (simplified)"""
        try:
            h264_bytes = base64.b64decode(h264_data)

            # Write directly to temp file
            temp_file = f"temp_test_{self.frames_received % 10}.h264"
            with open(temp_file, 'wb') as f:
                f.write(h264_bytes)

            cap = cv2.VideoCapture(temp_file)
            ret, frame = cap.read()
            cap.release()

            # Clean up
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)

            if ret and frame is not None:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.frames_received += 1
                    self.last_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                return True
        except Exception as e:
            print(f"❌ H264 decode error: {e}")
        return False

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.last_timestamp
        return None, None

# Global receiver
receiver = SimpleFrameReceiver()

def display_worker():
    """Display received frames in a loop"""
    print("📺 Starting frame display...")

    while True:
        frame, timestamp = receiver.get_latest_frame()

        if frame is not None:
            # Add timestamp overlay
            h, w = frame.shape[:2]

            # Create overlay
            overlay = frame.copy()
            timestamp_text = f"Received: {timestamp}"
            frame_count_text = f"Frame #{receiver.frames_received}"

            # Draw timestamp
            cv2.putText(overlay, timestamp_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, timestamp_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            # Draw frame counter
            cv2.putText(overlay, frame_count_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Test Receiving - Raw Frames', overlay)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Display stopped by user")
                cv2.destroyAllWindows()
                break
        else:
            # No frame yet, show placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for frames...", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Test Receiving - Raw Frames', placeholder)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("👋 Display stopped by user")
                cv2.destroyAllWindows()
                break

        time.sleep(0.016)  # ~60 FPS display

# WebSocket handlers
def new_client(client, server):
    print(f"🎉 Client connected: {client}")

def client_left(client, server):
    print(f"❌ Client disconnected: {client}")

def message_received(client, server, message):
    try:
        ws_message = json.loads(message)

        if ws_message.get('event') == 'video_separate_png.data':
            participant = ws_message['data']['data'].get('participant', {})
            participant_name = participant.get('name', 'Unknown')

            print(f"🖼️ PNG from {participant_name}")

            png_buffer = ws_message['data']['data']['buffer']
            if receiver.update_frame_png(png_buffer):
                print(f"✅ Frame #{receiver.frames_received} received at {receiver.last_timestamp}")
            else:
                print("❌ Failed to decode PNG")

        elif ws_message.get('event') == 'video_separate_h264.data':
            participant = ws_message['data']['data'].get('participant', {})
            participant_name = participant.get('name', 'Unknown')

            print(f"🎬 H264 from {participant_name}")

            h264_buffer = ws_message['data']['data']['buffer']
            if receiver.update_frame_h264(h264_buffer):
                print(f"✅ Frame #{receiver.frames_received} received at {receiver.last_timestamp}")
            else:
                print("❌ Failed to decode H264")

        else:
            print(f"❓ Unknown event: {ws_message.get('event')}")

    except json.JSONDecodeError as e:
        print(f'❌ JSON parse error: {e}')
    except Exception as e:
        print(f'❌ WebSocket message error: {e}')

# Flask routes
@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'message': 'Test Receiving Server - Raw Frame Display',
        'frames_received': receiver.frames_received,
        'last_timestamp': receiver.last_timestamp
    })

@app.route('/stats')
def stats():
    return jsonify({
        'frames_received': receiver.frames_received,
        'last_timestamp': receiver.last_timestamp,
        'has_frame': receiver.latest_frame is not None
    })

def start_websocket_server():
    server = WebsocketServer(host='0.0.0.0', port=5003)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    print("🚀 WebSocket server started on port 5003")
    server.run_forever()

if __name__ == '__main__':
    print("🧪 TEST RECEIVING SERVER")
    print("Shows raw frames with timestamps - no processing")
    print("=" * 50)

    # Start WebSocket server
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    # Start display worker
    display_thread = threading.Thread(target=display_worker, daemon=True)
    display_thread.start()

    print("🚀 Starting Flask API server on port 5000...")
    print("📡 WebSocket: ws://localhost:5003")
    print("🌐 HTTP API: http://localhost:5000")
    print("📺 Preview window will show raw frames with timestamps")
    print("   Press 'q' in preview window to close")
    print("\n⚠️  Configure your bot to send PNG format for best results!")
    print("=" * 50)

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        cv2.destroyAllWindows()
