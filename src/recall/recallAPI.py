from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import sqlite3
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Use localhost for webhook URL (no external tunneling)
webhook_url = os.getenv("RECALL_WEBHOOK_URL", "http://localhost:3000/webhook")
websocket_url = os.getenv("RECALL_WEBSOCKET_URL", "")


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
        "Authorization": os.getenv("RECALL_API_KEY", ""),
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



def start_bot_for_meeting(meeting_url):
    """Start a bot for the given meeting URL"""
    api_key = os.getenv("RECALL_API_KEY", "")
    if not api_key:
        print("❌ Error: RECALL_API_KEY not found in environment variables")
        return False

    headers = {
        "Authorization": api_key,
        "accept": "application/json",
        "content-type": "application/json"
    }

    payload = {
        "meeting_url": meeting_url,
        "bot_name": "VibeB2B",
        "recording_config": {
            "transcript": {
                "provider": {
                    "meeting_captions": {}
                }
            },
            "video_separate_png": {},
            "video_mixed_layout": "gallery_view_v2"
        }
    }

    # Add realtime endpoints only if webhook_url is not localhost (for local testing)
    if webhook_url and not webhook_url.startswith("http://localhost"):
        payload["recording_config"]["realtime_endpoints"] = [
            {
                "type": "webhook",
                "url": webhook_url,
                "events": ["transcript.data", "transcript.partial_data"]
            }
        ]

        # Add websocket endpoint if configured
        if websocket_url:
            payload["recording_config"]["realtime_endpoints"].append({
                "type": "websocket",
                "url": websocket_url,
                "events": ["video_separate_png.data"]
            })

    try:
        print(f"🚀 Sending request to Recall.ai...")
        print(f"   Meeting URL: {meeting_url}")
        print(f"   Headers: Authorization present = {bool(headers.get('Authorization'))}")

        response = requests.post("https://us-west-2.recall.ai/api/v1/bot", json=payload, headers=headers, timeout=30)

        print(f"   Response status: {response.status_code}")

        if response.status_code == 201:
            result = response.json()
            print(f"✅ Bot started successfully!")
            print(f"   Bot ID: {result.get('id', 'Unknown')}")
            return True
        else:
            print(f"❌ Failed to start bot. Status: {response.status_code}")
            print(f"   Response headers: {dict(response.headers)}")

            # Try to get JSON error, fallback to text
            try:
                error_data = response.json()
                print(f"   Error JSON: {error_data}")
            except:
                print(f"   Error text: {response.text[:500]}...")

            return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out - Recall.ai might be slow to respond")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - can't reach Recall.ai")
        return False
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        return False

if __name__ == "__main__":
    # Check if a meeting URL was provided as command line argument
    if len(sys.argv) > 1:
        meeting_url = sys.argv[1]
        print(f"🎯 Starting bot for meeting: {meeting_url}")

        # Start the bot
        success = start_bot_for_meeting(meeting_url)
        if success:
            print("✅ Bot started successfully!")
        else:
            print("❌ Failed to start bot")
            sys.exit(1)
    else:
        # No arguments provided, start the Flask server normally
        print("🌐 Starting Flask server on port 3000...")
        print("📖 Use: npm run recall <meeting_url> to start a bot directly")
        app.run(host="0.0.0.0", port=3000)
