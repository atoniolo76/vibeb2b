from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import threading
import time
from datetime import datetime
import os
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", max_size=10*1024*1024, async_mode='threading')

class LatestOnlyFrameManager:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        self.last_receive_time = 0
        self.receive_fps = 0
        
    def update_frame(self, frame):
        """Replace current frame with latest - skip everything else"""
        with self.frame_lock:
            # Count statistics
            self.frames_received += 1
            if self.latest_frame is not None:
                self.frames_skipped += 1  # We're replacing an unprocessed frame
            
            # Always replace with latest
            self.latest_frame = frame.copy()
            
            # Calculate receive FPS
            current_time = time.time()
            if self.last_receive_time > 0:
                time_diff = current_time - self.last_receive_time
                if time_diff > 0:
                    self.receive_fps = 0.9 * self.receive_fps + 0.1 * (1.0 / time_diff)
            self.last_receive_time = current_time
    
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
                'frames_skipped': self.frames_skipped,
                'receive_fps': round(self.receive_fps, 2),
                'skip_ratio': round(self.frames_skipped / max(1, self.frames_received) * 100, 1),
                'has_frame': self.latest_frame is not None
            }

class SimpleH264Decoder:
    def __init__(self):
        self.temp_file = "temp_latest.h264"
        
    def decode_h264_chunk(self, h264_data):
        """Fast H264 decoding - get first frame only"""
        try:
            with open(self.temp_file, 'wb') as f:
                f.write(h264_data)
            
            cap = cv2.VideoCapture(self.temp_file)
            ret, frame = cap.read()
            cap.release()
            
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                
            return frame if ret else None
            
        except Exception as e:
            print(f"Decode error: {e}")
            return None

# Global instances
frame_manager = LatestOnlyFrameManager()
decoder = SimpleH264Decoder()

@app.route('/', methods=['POST'])
def index():    
    print("got smth")
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Realtime H264 Receiver</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .stats { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }
            .strategy { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 8px; }
            .good { color: green; font-weight: bold; }
            .warning { color: orange; font-weight: bold; }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
    </head>
    <body>
        <h1>Simple Realtime H264 Receiver</h1>
        
        <div class="strategy">
            <h3>📌 Strategy: Latest Frame Only</h3>
            <p>✅ Always uses most recent frame</p>
            <p>✅ Zero buffer buildup</p>
            <p>✅ Minimal latency</p>
            <p>⚠️ Skips frames (this is intentional!)</p>
        </div>
        
        <div class="stats">
            <h4>Performance Stats:</h4>
            <p>Frames Received: <span id="frames-received">0</span> 
               (Rate: <span id="receive-fps">0</span> FPS)</p>
            <p>Frames Processed: <span id="frames-processed">0</span></p>
            <p>Frames Skipped: <span id="frames-skipped">0</span> 
               (<span id="skip-ratio">0</span>% - <em>this is normal!</em>)</p>
            <p>Current Frame Available: <span id="has-frame">No</span></p>
        </div>
        
        <div>
            <h4>📡 API Endpoints:</h4>
            <ul>
                <li><code>GET /get_latest_frame</code> - Get current frame for emotion processing</li>
                <li><code>GET /stats</code> - Get performance statistics</li>
                <li><code>POST /reset_stats</code> - Reset counters</li>
            </ul>
        </div>
        
        <script>
            const socket = io();
            
            socket.on('stats_update', function(data) {
                document.getElementById('frames-received').textContent = data.frames_received;
                document.getElementById('frames-processed').textContent = data.frames_processed;
                document.getElementById('frames-skipped').textContent = data.frames_skipped;
                document.getElementById('receive-fps').textContent = data.receive_fps;
                document.getElementById('skip-ratio').textContent = data.skip_ratio;
                document.getElementById('has-frame').textContent = data.has_frame ? 'Yes' : 'No';
            });
            
            // Update stats every second
            setInterval(() => {
                fetch('/stats').then(r => r.json()).then(data => {
                    socket.emit('stats_update', data);
                });
            }, 1000);
        </script>
    </body>
    </html>
    '''

@socketio.on('connect')
def handle_connect():
    print(f"Client connected")
    emit('stats_update', frame_manager.get_stats())

@socketio.on('message')
def handle_google_meet_message(message):
    """Receive Google Meet WebSocket messages"""
    try:
        # Parse Google Meet message format
        if isinstance(message, str):
            ws_message = json.loads(message)
        else:
            ws_message = message
            
        # Handle H264 video data from Google Meet
        if ws_message.get('event') == 'video_separate_h264.data':
            buffer_data = ws_message['data']['data']['buffer']
            participant = ws_message['data']['data'].get('participant', {})
            
            print(f"Received H264 from participant: {participant.get('name', 'Unknown')}")
            
            # Decode H264 data
            h264_bytes = base64.b64decode(buffer_data)
            
            # Decode H264 to frame
            frame = decoder.decode_h264_chunk(h264_bytes)
            
            if frame is not None:
                # Update the latest frame
                frame_manager.update_frame(frame)
                print(f"📸 Frame updated! Shape: {frame.shape}")
                
                # Emit stats to connected clients
                emit('stats_update', frame_manager.get_stats(), broadcast=True)
            
        else:
            print(f"Unhandled message event: {ws_message.get('event', 'unknown')}")
            return
            
    except Exception as e:
        print(f"Error parsing Google Meet message: {e}")
        return

@socketio.on('h264_chunk')
def handle_h264_chunk(data):
    """Receive H264 chunk and update latest frame (for testing)"""
    try:
        # Decode data format
        if isinstance(data, dict) and 'data' in data:
            h264_bytes = base64.b64decode(data['data'])
        elif isinstance(data, str):
            h264_bytes = base64.b64decode(data)
        else:
            h264_bytes = data
        
        # Decode H264 to frame
        frame = decoder.decode_h264_chunk(h264_bytes)
        
        if frame is not None:
            # Update latest frame (this automatically skips old frames)
            frame_manager.update_frame(frame)
            
            # Emit stats update
            stats = frame_manager.get_stats()
            socketio.emit('stats_update', stats, broadcast=True)
            
    except Exception as e:
        print(f"Error processing H264: {e}")

@app.route('/get_latest_frame')
def get_latest_frame():
    """Get the most recent frame for emotion processing"""
    frame = frame_manager.get_latest_frame()
    
    if frame is not None:
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'success': True,
            'frame': frame_base64,
            'timestamp': datetime.now().isoformat(),
            'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
            'strategy': 'latest_only'
        }
    else:
        return {'success': False, 'message': 'No frame available'}, 404

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    return frame_manager.get_stats()

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Reset all statistics"""
    with frame_manager.frame_lock:
        frame_manager.frames_received = 0
        frame_manager.frames_processed = 0
        frame_manager.frames_skipped = 0
        frame_manager.receive_fps = 0
    return {'success': True, 'message': 'Statistics reset'}

@app.route('/health')
def health_check():
    """Health check"""
    stats = frame_manager.get_stats()
    return {
        'status': 'healthy',
        'has_frame': stats['has_frame'],
        'uptime': time.time()
    }

if __name__ == '__main__':
    print("🚀 Starting Simple Realtime H264 Receiver...")
    print("📡 WebSocket: ws://localhost:5003")
    print("🌐 Interface: http://localhost:5003")
    print("🎯 Latest Frame API: http://localhost:5003/get_latest_frame")
    print("\n📌 Strategy: LATEST FRAME ONLY")
    print("   ✅ Zero buffer buildup")
    print("   ✅ Minimal latency") 
    print("   ✅ Always most recent frame")
    print("   ⚠️  High skip ratio is NORMAL and GOOD!")
    print("\n🔗 Usage:")
    print("   Your friend sends H264 chunks to WebSocket event 'h264_chunk'")
    print("   You call GET /get_latest_frame whenever you want to process emotions")
    print("   Frame skipping happens automatically - no buffer issues!")
    
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True)
