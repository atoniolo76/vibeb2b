import socketio
import cv2
import base64
import time
import numpy as np

# This is a test client for your friend to understand how to send H264 data

class H264TestSender:
    def __init__(self, server_url='http://localhost:5001'):
        self.sio = socketio.Client()
        self.server_url = server_url
        self.setup_events()
        
    def setup_events(self):
        @self.sio.event
        def connect():
            print("✅ Connected to H264 receiver server")
            
        @self.sio.event
        def disconnect():
            print("❌ Disconnected from server")
            
        @self.sio.event
        def pong():
            print("🏓 Received pong from server")
    
    def connect_to_server(self):
        """Connect to the WebSocket server"""
        try:
            print(f"🔌 Connecting to {self.server_url}...")
            self.sio.connect(self.server_url)
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def send_h264_chunk(self, h264_data, frame_info=None):
        """Send H264 data chunk to server"""
        try:
            # Method 1: Send as base64 string
            h264_base64 = base64.b64encode(h264_data).decode('utf-8')
            self.sio.emit('h264_chunk', h264_base64)
            
            # Method 2: Send as JSON with metadata (alternative)
            # data_packet = {
            #     'data': h264_base64,
            #     'timestamp': time.time(),
            #     'frame_info': frame_info or {}
            # }
            # self.sio.emit('h264_chunk', data_packet)
            
            print(f"📤 Sent H264 chunk: {len(h264_data)} bytes")
            
        except Exception as e:
            print(f"❌ Error sending H264 chunk: {e}")
    
    def send_test_video(self, video_path=None):
        """Send test video as H264 chunks"""
        if video_path is None:
            # Create test video from webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Cannot open webcam")
                return
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video file: {video_path}")
                return
        
        # Set up H264 encoder
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        temp_file = 'temp_h264_stream.mp4'
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame for testing (optional)
                frame = cv2.resize(frame, (640, 480))
                
                # Create temporary H264 file for this frame
                out = cv2.VideoWriter(temp_file, fourcc, 30.0, (640, 480))
                out.write(frame)
                out.release()
                
                # Read the H264 data
                with open(temp_file, 'rb') as f:
                    h264_data = f.read()
                
                # Send to server
                frame_info = {
                    'frame_number': frame_count,
                    'resolution': '640x480',
                    'timestamp': time.time()
                }
                
                self.send_h264_chunk(h264_data, frame_info)
                
                frame_count += 1
                time.sleep(1/30)  # 30 FPS
                
                # Break after 100 frames for testing
                if frame_count >= 100:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Stopping video stream...")
        finally:
            cap.release()
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def disconnect(self):
        """Disconnect from server"""
        self.sio.disconnect()

def main():
    print("🧪 H264 Test Sender")
    print("This simulates how your friend should send H264 data")
    print("-" * 50)
    
    sender = H264TestSender()
    
    if sender.connect_to_server():
        print("📹 Starting test video stream...")
        print("Press Ctrl+C to stop")
        
        try:
            sender.send_test_video()  # Use webcam
            # sender.send_test_video('path/to/your/video.mp4')  # Use video file
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
        finally:
            sender.disconnect()
    else:
        print("❌ Could not connect to server")

if __name__ == '__main__':
    main()
