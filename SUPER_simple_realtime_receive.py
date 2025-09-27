#!/usr/bin/env python3
"""
SUPER SIMPLE WebSocket receiver - just to test if Google Meet bot can send ANYTHING
Based on the Google Meet docs example
"""

from flask import Flask
from flask_socketio import SocketIO
import json
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

frame_count = 0

@socketio.on('connect')
def handle_connect():
    print("🎉 CLIENT CONNECTED!")

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ CLIENT DISCONNECTED!")

@socketio.on('message')
def handle_message(message):
    """Handle any WebSocket message"""
    global frame_count
    
    print(f"\n📨 RAW MESSAGE RECEIVED:")
    print(f"Type: {type(message)}")
    print(f"Content: {str(message)[:200]}...")  # First 200 chars
    
    try:
        # Try to parse as JSON
        if isinstance(message, str):
            ws_message = json.loads(message)
        else:
            ws_message = message
            
        print(f"✅ PARSED JSON SUCCESS!")
        print(f"Event: {ws_message.get('event', 'NO EVENT FIELD')}")
        
        # Handle PNG frames
        if ws_message.get('event') == 'video_separate_png.data':
            recording_id = ws_message['data']['recording']['id']
            participant = ws_message['data']['data'].get('participant', {})
            
            print(f"🖼️  PNG FRAME #{frame_count}")
            print(f"   Recording ID: {recording_id}")
            print(f"   Participant: {participant.get('name', 'Unknown')}")
            
            # Save PNG to file
            filename = f"{recording_id}.{frame_count}.png"
            buffer_data = ws_message['data']['data']['buffer']
            
            with open(filename, 'wb') as f:
                png_data = base64.b64decode(buffer_data)
                f.write(png_data)
            
            print(f"   💾 Saved: {filename}")
            frame_count += 1
            
        # Handle H264 frames  
        elif ws_message.get('event') == 'video_separate_h264.data':
            recording_id = ws_message['data']['recording']['id']
            participant = ws_message['data']['data'].get('participant', {})
            
            print(f"🎥 H264 FRAME #{frame_count}")
            print(f"   Recording ID: {recording_id}")
            print(f"   Participant: {participant.get('name', 'Unknown')}")
            
            # Save H264 to file
            filename = f"{recording_id}.h264"
            buffer_data = ws_message['data']['data']['buffer']
            
            with open(filename, 'ab') as f:  # Append mode for H264
                h264_data = base64.b64decode(buffer_data)
                f.write(h264_data)
            
            print(f"   💾 Appended to: {filename}")
            frame_count += 1
            
        else:
            print(f"❓ UNHANDLED EVENT: {ws_message.get('event', 'unknown')}")
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON PARSE ERROR: {e}")
        print("Raw message was not valid JSON")
        
    except Exception as e:
        print(f"❌ GENERAL ERROR: {e}")

@socketio.on_error()
def error_handler(e):
    print(f"❌ WEBSOCKET ERROR: {e}")

if __name__ == '__main__':
    print("🚀 SUPER SIMPLE WebSocket Receiver Starting...")
    print("📡 Listening on: ws://localhost:5003")
    print("🌐 Web interface: http://localhost:5003")
    print("\n🎯 TESTING CHECKLIST:")
    print("   1. Start this server")
    print("   2. Use ngrok: ngrok http 5003")  
    print("   3. Give friend the wss://xxx.ngrok.io URL")
    print("   4. Friend configures Google Meet bot")
    print("   5. Watch for messages here!")
    print("\n" + "="*50)
    
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True)
