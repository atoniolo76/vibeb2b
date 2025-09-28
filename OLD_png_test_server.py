#!/usr/bin/env python3
"""
PNG TEST SERVER - Simple PNG reception without PyTorch
Test PNG frame reception and display before adding emotion processing
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

class PNGFrameManager:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frames_received = 0
        self.frames_processed = 0
        self.last_receive_time = 0
        self.receive_fps = 0
        self.frame_timestamps = []  # Track timing between frames
        self.duplicate_count = 0

    def update_frame_png(self, png_data):
        """Decode PNG frame data with duplicate filtering"""
        with self.frame_lock:
            try:
                # Decode PNG directly from base64
                png_bytes = base64.b64decode(png_data)
                nparr = np.frombuffer(png_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Check for duplicates before processing
                    frame_hash = hash(frame.tobytes())
                    if hasattr(self, 'last_frame_hash') and frame_hash == self.last_frame_hash:
                        self.duplicate_count += 1
                        # Still update timestamps for interval analysis
                        current_time = time.time()
                        self.frame_timestamps.append(current_time)
                        if len(self.frame_timestamps) > 10:
                            self.frame_timestamps.pop(0)
                        return True  # Don't count as new frame, but accept it

                    # New unique frame
                    self.latest_frame = frame.copy()
                    self.frames_received += 1
                    self.last_frame_hash = frame_hash

                    # Track timestamp for interval analysis
                    current_time = time.time()
                    self.frame_timestamps.append(current_time)

                    # Keep only last 10 timestamps for analysis
                    if len(self.frame_timestamps) > 10:
                        self.frame_timestamps.pop(0)

                    # Calculate receive FPS (only for unique frames)
                    if self.last_receive_time > 0:
                        time_diff = current_time - self.last_receive_time
                        if time_diff > 0:
                            self.receive_fps = 0.9 * self.receive_fps + 0.1 * (1.0 / time_diff)
                    self.last_receive_time = current_time

                    return True
                else:
                    print("❌ Failed to decode PNG frame")
                    return False

            except Exception as e:
                print(f"❌ PNG decode error: {e}")
                return False

    def get_latest_frame(self):
        """Get the most recent frame for processing"""
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                self.frames_processed += 1
                return frame
        return None

    def get_stats(self):
        """Get performance statistics"""
        with self.frame_lock:
            return {
                'frames_received': self.frames_received,
                'frames_processed': self.frames_processed,
                'duplicates_detected': self.duplicate_count,
                'unique_frames': self.frames_received,
                'receive_fps': round(self.receive_fps, 2),
                'has_frame': self.latest_frame is not None,
                'duplicate_ratio': round(self.duplicate_count / max(1, self.frames_received + self.duplicate_count) * 100, 1) if (self.frames_received + self.duplicate_count) > 0 else 0
            }

# Global frame manager
frame_manager = PNGFrameManager()

def draw_timestamp_overlay(frame):
    """Draw timestamp overlay on frame"""
    h, w = frame.shape[:2]

    # Create overlay
    overlay = frame.copy()

    # Add timestamp
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    frame_count = frame_manager.frames_received

    cv2.putText(overlay, f"Frame #{frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Time: {timestamp}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Add stats
    stats = frame_manager.get_stats()
    stats_text = f"FPS: {stats['receive_fps']:.1f} | Received: {stats['frames_received']}"
    cv2.putText(overlay, stats_text, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return overlay

def preview_display_worker():
    """Display received frames in a loop"""
    print("📺 Starting PNG frame preview...")

    while True:
        frame = frame_manager.get_latest_frame()

        if frame is not None:
            # Add timestamp overlay
            display_frame = draw_timestamp_overlay(frame)
            cv2.imshow('PNG Test Server - Raw Frames', display_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Preview stopped by user")
                cv2.destroyAllWindows()
                break
        else:
            # No frame yet, show placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for PNG frames...", (150, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, "Configure bot for PNG format", (120, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('PNG Test Server - Raw Frames', placeholder)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("👋 Preview stopped by user")
                cv2.destroyAllWindows()
                break

        time.sleep(0.016)  # ~60 FPS display

# WebSocket handlers
def new_client(client, server):
    print(f"🎉 PNG BOT CONNECTED: {client}")
    print(f"   ✅ WebSocket connection established!")
    print(f"   📡 Bot can now send PNG frames")

def client_left(client, server):
    print(f"❌ PNG BOT DISCONNECTED: {client}")
    print(f"   ❌ WebSocket connection lost")

def message_received(client, server, message):
    print(f"📨 RAW MESSAGE RECEIVED: {len(message)} chars")
    try:
        ws_message = json.loads(message)
        print(f"📨 PARSED EVENT: {ws_message.get('event', 'unknown')}")

        if ws_message.get('event') == 'video_separate_png.data':
            participant = ws_message['data']['data'].get('participant', {})
            participant_name = participant.get('name', 'Unknown')
            recording_id = ws_message['data']['recording']['id']

            # Get precise timestamp when frame arrives
            arrival_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            png_size = len(ws_message['data']['data']['buffer'])

            print(f"🖼️ PNG RECEIVED from: {participant_name} (Recording: {recording_id})")
            print(f"   📏 Size: {png_size} bytes | Arrival: {arrival_time}")

            if frame_manager.update_frame_png(ws_message['data']['data']['buffer']):
                stats = frame_manager.get_stats()
                print(f"✅ PNG PROCESSED | Total: {stats['frames_received']} | FPS: {stats['receive_fps']:.1f}")

                # Show duplicate count periodically
                if frame_manager.duplicate_count > 0 and frame_manager.frames_received % 5 == 0:
                    print(f"   📊 Stats: {frame_manager.duplicate_count} duplicates, {stats['frames_received']} unique frames")

                # Analyze timing between frames
                if len(frame_manager.frame_timestamps) >= 2:
                    intervals = []
                    for i in range(1, len(frame_manager.frame_timestamps)):
                        intervals.append(frame_manager.frame_timestamps[i] - frame_manager.frame_timestamps[i-1])

                    if intervals:
                        avg_interval = sum(intervals) / len(intervals)
                        min_interval = min(intervals)
                        max_interval = max(intervals)
                        print(f"   ⏱️  Frame intervals: Avg={avg_interval:.1f}s, Min={min_interval:.1f}s, Max={max_interval:.1f}s")

            else:
                print("❌ PNG decode failed")

        else:
            # Handle any other messages (like our test message)
            print(f"📨 OTHER MESSAGE: {ws_message}")
            if 'test' in ws_message:
                print(f"   🎯 Received test message: {ws_message['test']}")
            else:
                print(f"❓ Unknown event: {ws_message.get('event', 'no_event_field')}")

    except json.JSONDecodeError as e:
        print(f'❌ JSON parse error: {e}')
    except Exception as e:
        print(f'❌ WebSocket error: {e}')

# Flask routes
@app.route('/')
def index():
    stats = frame_manager.get_stats()
    return jsonify({
        'status': 'running',
        'message': 'PNG Test Server - Frame Reception Only',
        'websocket_port': 5003,
        'websocket_url': 'ws://localhost:5003',
        'frames_received': stats['frames_received'],
        'receive_fps': stats['receive_fps'],
        'instructions': 'Configure your bot to send PNG format to wss://YOUR_NGROK_URL:5003',
        'debug_steps': [
            '1. Run: ngrok http 5003',
            '2. Copy the HTTPS WebSocket URL',
            '3. Configure bot with PNG format',
            '4. Start Google Meet session',
            '5. Check this endpoint for connection status'
        ]
    })

@app.route('/websocket_status')
def websocket_status():
    """Check WebSocket server status"""
    return jsonify({
        'websocket_server_running': True,
        'port': 5003,
        'protocol': 'WebSocket',
        'expected_bot_events': ['video_separate_png.data'],
        'ngrok_command': 'ngrok http 5003',
        'troubleshooting': {
            'check_ngrok': 'Visit http://localhost:4040 for ngrok status',
            'test_connection': 'Use websocket client to test wss://your-url:5003',
            'bot_config': 'Ensure bot uses video_separate_png and video_separate_png.data'
        }
    })

@app.route('/stats')
def stats():
    return jsonify(frame_manager.get_stats())

@app.route('/timing')
def timing_analysis():
    """Analyze frame timing patterns"""
    with frame_manager.frame_lock:
        if len(frame_manager.frame_timestamps) < 2:
            return jsonify({
                'status': 'insufficient_data',
                'message': 'Need at least 2 frames for timing analysis'
            })

        # Calculate intervals
        intervals = []
        for i in range(1, len(frame_manager.frame_timestamps)):
            intervals.append(frame_manager.frame_timestamps[i] - frame_manager.frame_timestamps[i-1])

        if not intervals:
            return jsonify({'status': 'no_intervals'})

        # Calculate statistics
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)

        # Calculate effective FPS
        effective_fps = 1.0 / avg_interval if avg_interval > 0 else 0

        # Analyze for patterns
        large_gaps = sum(1 for i in intervals if i > 10)  # Gaps > 10 seconds
        small_gaps = sum(1 for i in intervals if i < 1)   # Gaps < 1 second

        # Analyze degradation trend
        first_half = intervals[:len(intervals)//2]
        second_half = intervals[len(intervals)//2:]

        degradation = "stable"
        if first_half and second_half:
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            if avg_second > avg_first * 1.5:  # 50% slower
                degradation = "getting_slower"
            elif avg_second < avg_first * 0.8:  # 20% faster
                degradation = "getting_faster"

        return jsonify({
            'status': 'analyzed',
            'frame_count': len(frame_manager.frame_timestamps),
            'timing_stats': {
                'avg_interval_seconds': round(avg_interval, 2),
                'min_interval_seconds': round(min_interval, 2),
                'max_interval_seconds': round(max_interval, 2),
                'effective_fps': round(effective_fps, 2)
            },
            'pattern_analysis': {
                'large_gaps_10s_plus': large_gaps,
                'small_gaps_1s_minus': small_gaps,
                'duplicates_detected': frame_manager.duplicate_count,
                'performance_trend': degradation
            },
            'interpretation': {
                'root_cause': 'google_meet_bot_limitations',
                'explanation': 'Google Meet bots are designed for recording, not real-time streaming. Frame rate degrades over time.',
                'recommendations': [
                    'Upgrade to paid Google Meet bot subscription',
                    'Use RTMP streaming instead of WebSocket',
                    'Use direct webcam capture for testing',
                    'Accept that Google Meet bots have inherent latency'
                ]
            }
        })

def start_websocket_server():
    server = WebsocketServer(host='0.0.0.0', port=5003)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    print("🚀 PNG WebSocket server started on port 5003")
    server.run_forever()

if __name__ == '__main__':
    print("🖼️ PNG TEST SERVER - FRAME RECEPTION ONLY")
    print("No PyTorch required - just test PNG reception")
    print("=" * 50)

    # Start WebSocket server
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    # Start preview display
    preview_thread = threading.Thread(target=preview_display_worker, daemon=True)
    preview_thread.start()

    print("🚀 Starting Flask API server on port 5000...")
    print("📡 WebSocket: ws://localhost:5003")
    print("🌐 HTTP API: http://localhost:5000")
    print("🌐 Make WebSocket public with: ngrok http 5003")
    print("\n🖼️ PNG-ONLY SERVER - TEST FRAME RECEPTION")
    print("📺 Preview shows raw PNG frames with timestamps")
    print("   Press 'q' to close preview window")
    print("\n⚠️  BOT CONFIGURATION REQUIRED:")
    print("   Use 'video_separate_png' in recording_config")
    print("   Use 'video_separate_png.data' in events array")
    print("=" * 50)

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down PNG test server...")
        cv2.destroyAllWindows()
