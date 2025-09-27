from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import sqlite3
import os

app = Flask(__name__)
CORS(app)


webhook_url = "https://briella-undecretive-eastward.ngrok-free.dev/webhook" # URL to this server, where transcript is received
# webhook_url = "http://localhost:3000/webhook" # URL to this flask server

websocket_url = "wss://697de5b2275d.ngrok-free.app/" # URL to donson's flask server, where video is received


# health check endpoint
@app.route('/health', methods=['GET'])
def health():
    print("Health check")
    return jsonify({"status": "healthy"}), 200


# call this endpoint with a body containing the meeting_url to start the bot
@app.route('/start_VB2B', methods=['POST'])
def create_bot():
    print("Creating bot")
    data = request.get_json()
    meeting_url = data.get('meeting_url', '')

    headers = {
        "Authorization": "05804d5b68e038ca6faf8890d8466fc390741dca",
        "accept": "application/json",
        "content-type": "application/json"
    }

    testing_payload = {
        "meeting_url": meeting_url,
        "bot_name": "VibeB2B",
        "recording_config": {"transcript": {"provider": {"meeting_captions": {}}}}
    }
    

    testing_transcript_payload = {
        "meeting_url": meeting_url,
        "bot_name": "VibeB2B",
        "recording_config": {
            "transcript": {
                "provider": {
                    "meeting_captions": {}
                }
            },
            "realtime_endpoints": [
                {
                    "type": "webhook",
                    "url": webhook_url,
                    "events": ["transcript.data", "transcript.partial_data"]
                }
            ]
        }
    }

    testing_video_payload = {
        "meeting_url": meeting_url,
        "bot_name": "VibeB2B",
        "recording_config": {
            "video_separate_png": {},
            "video_mixed_layout": "gallery_view_v2",
            "realtime_endpoints": [
            {
                "type": "websocket",
                "url": websocket_url,
                "events": ["video_separate_png.data"]
            }
            ]
        }
    }
    
    prod_payload = {
        "meeting_url": meeting_url,
        "bot_name": "VibeB2B",
        "recording_config": {
            "transcript": {
                "provider": {
                    "meeting_captions": {}
                }
            },
            "video_separate_png": {},
            "video_mixed_layout": "gallery_view_v2",
            "realtime_endpoints": [
                {
                    "type": "webhook",
                    "url": webhook_url,
                    "events": ["transcript.data", "transcript.partial_data"]
                },
                {
                    "type": "websocket",
                    "url": websocket_url,
                    "events": ["video_separate_png.data"]
                }
            ],

        }
    }
    
    
    response = requests.post("https://us-west-2.recall.ai/api/v1/bot", json=prod_payload, headers=headers)
    
    if response.status_code == 201:
        return jsonify(response.json()), 200
    else:
        return jsonify({"error": response.text}), response.status_code


# endpoint to receive the live transcript data
@app.route('/webhook', methods=['POST'])
def handle_webhook():
    print("Received webhook data")

    # Process the incoming webhook data
    data = request.json

    print(data)

    # Save transcript lines to SQLite database with TIMESTAMP, PARTICIPANTNAME, TEXT
    db_path = os.path.join(os.path.dirname(__file__), 'transcript.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS transcript_lines (
        TIMESTAMP TEXT,
        PARTICIPANTNAME TEXT,
        TEXT TEXT
    )''')

    if data.get('event') == 'transcript.data':
        webhook_data = data.get('data', {})
        words = webhook_data.get('words', [])
        participant = webhook_data.get('participant', {})
        part_name = participant.get('name', '')
        if words:
            text = ' '.join(word.get('text', '') for word in words)
            timestamp = words[0].get('start_timestamp', {}).get('absolute', '')
            cursor.execute('INSERT INTO transcript_lines (TIMESTAMP, PARTICIPANTNAME, TEXT) VALUES (?, ?, ?)',
                (timestamp, part_name, text))

    conn.commit()
    conn.close()

    # Perform actions based on the data
    return jsonify({"status": "received"}), 200  # Acknowledge receipt



if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 3000)