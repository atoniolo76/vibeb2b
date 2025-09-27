#!/usr/bin/env python3
"""
PNG Realtime Receiver - Latest Frame Only Strategy
Receives PNG frames from Google Meet bot and keeps only the latest one
"""

from websocket_server import WebsocketServer
import json
import base64
import threading
import time
import cv2
import numpy as np
from datetime import datetime

class LatestPNGManager:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        self.last_receive_time = 0
        self.receive_fps = 0
        
    def update_frame(self, png_data):
        """Replace current frame with latest - skip everything else"""
        with self.frame_lock:
            # Count statistics
            self.frames_received += 1
            if self.latest_frame is not None:
                self.frames_skipped += 1  # We're replacing an unprocessed frame
            
            # Decode PNG directly to OpenCV format
            try:
                png_bytes = base64.b64decode(png_data)
                nparr = np.frombuffer(png_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Always replace with latest
                    self.latest_frame = frame.copy()
                    
                    # Calculate receive FPS
                    current_time = time.time()
                    if self.last_receive_time > 0:
                        time_diff = current_time - self.last_receive_time
                        if time_diff > 0:
                            self.receive_fps = 0.9 * self.receive_fps + 0.1 * (1.0 / time_diff)
                    self.last_receive_time = current_time
                    
                    return True
                    
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
                'frames_skipped': self.frames_skipped,
                'receive_fps': round(self.receive_fps, 2),
                'skip_ratio': round(self.frames_skipped / max(1, self.frames_received) * 100, 1),
                'has_frame': self.latest_frame is not None
            }

# Global frame manager
frame_manager = LatestPNGManager()

def new_client(client, server):
    """Called when a new client connects"""
    print("🎉 Google Meet bot connected!")

def client_left(client, server):
    """Called when a client disconnects"""
    print("❌ Google Meet bot disconnected!")

def message_received(client, server, message):
    """Called when a message is received"""
    try:
        ws_message = json.loads(message)

        if ws_message.get('event') == 'video_separate_png.data':
            participant = ws_message['data']['data'].get('participant', {})
            recording_id = ws_message['data']['recording']['id']
            
            # Get PNG data
            png_buffer = ws_message['data']['data']['buffer']
            
            # Update latest frame (this automatically handles buffering)
            if frame_manager.update_frame(png_buffer):
                stats = frame_manager.get_stats()
                print(f"📸 PNG from {participant.get('name', 'Unknown')} | "
                      f"Received: {stats['frames_received']} | "
                      f"Skipped: {stats['frames_skipped']} | "
                      f"FPS: {stats['receive_fps']}")
            
        elif ws_message.get('event') == 'video_separate_h264.data':
            print("⚠️ Received H264 - but we're configured for PNG!")
            
        else:
            print(f"❓ Unhandled event: {ws_message.get('event', 'unknown')}")

    except json.JSONDecodeError as e:
        print(f'❌ JSON parse error: {e}')
    except Exception as e:
        print(f'❌ Error: {e}')

def get_latest_frame_api():
    """API endpoint simulation - get latest frame"""
    frame = frame_manager.get_latest_frame()
    
    if frame is not None:
        # Convert to JPEG for API response
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
        return {'success': False, 'message': 'No frame available'}

def stats_monitor():
    """Print stats every 5 seconds"""
    while True:
        time.sleep(5)
        stats = frame_manager.get_stats()
        if stats['frames_received'] > 0:
            print(f"\n📊 STATS: Received {stats['frames_received']} | "
                  f"Processed {stats['frames_processed']} | "
                  f"Skipped {stats['frames_skipped']} ({stats['skip_ratio']}%) | "
                  f"FPS: {stats['receive_fps']}")

def main():
    """Start the WebSocket server"""
    # Start stats monitor in background
    stats_thread = threading.Thread(target=stats_monitor, daemon=True)
    stats_thread.start()
    
    server = WebsocketServer(host='0.0.0.0', port=5003)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)

    print("🚀 PNG Realtime Receiver Starting...")
    print("📡 WebSocket: ws://localhost:5003")
    print("🌐 Make public with: ngrok http 5003")
    print("\n📌 Strategy: LATEST PNG FRAME ONLY")
    print("   ✅ Receives 480x360 PNG frames at 2fps")
    print("   ✅ Zero buffer buildup - always latest frame")
    print("   ✅ Minimal latency for emotion processing")
    print("   ⚠️  High skip ratio is NORMAL and GOOD!")
    print("\n🔗 Usage:")
    print("   Google Meet bot → PNG frames → Latest frame buffer")
    print("   Your emotion API calls get_latest_frame_api()")
    print("   Frame skipping happens automatically!")
    print("\n" + "="*50)

    server.run_forever()

if __name__ == '__main__':
    main()
