#!/bin/bash

# Setup script for VibeB2B Recall API integration
# This script sets up ngrok tunneling for the Flask server

echo "🚀 Setting up VibeB2B Recall API integration..."

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Installing ngrok..."
    brew install ngrok
fi

echo "✅ ngrok is installed"

# Function to cleanup background processes on exit
cleanup() {
    echo "🧹 Cleaning up..."
    kill $(jobs -p) 2>/dev/null
}

# Set trap to cleanup on script exit
trap cleanup EXIT

echo "📡 Starting Flask server on port 3000..."
# Start Flask server in background
npm run recall &
FLASK_PID=$!

# Wait a moment for Flask to start
sleep 3

echo "🌐 Starting ngrok tunnel to port 3000..."
# Start ngrok tunnel in background
ngrok http 3000 &
NGROK_PID=$!

# Wait for ngrok to start and get the URL
sleep 5

echo "🔗 Getting ngrok URL..."
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')

if [ -z "$NGROK_URL" ] || [ "$NGROK_URL" = "null" ]; then
    echo "❌ Failed to get ngrok URL. Make sure ngrok is running properly."
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "🌐 Your ngrok URL: $NGROK_URL"
echo "🔧 Update src/recall/recallAPI.py with this webhook URL:"
echo "   webhook_url = \"$NGROK_URL/webhook\""
echo ""
echo "📝 Next steps:"
echo "   1. Edit src/recall/recallAPI.py and replace the webhook_url"
echo "   2. Make sure you have a valid Recall.ai API key"
echo "   3. Test the /health endpoint: curl $NGROK_URL/health"
echo ""
echo "⚠️  Keep this script running to maintain the tunnel"
echo "   Press Ctrl+C to stop"

# Wait for user interrupt
wait
