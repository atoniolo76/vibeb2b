# Vibeb2b

A Mastra-based AI assistant for sales insights and weather planning.

## Setup

### Prerequisites

- Node.js >= 20.9.0
- npm or yarn

### Installation

```bash
npm install
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Slack Bot Setup

### 1. Create Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click "Create New App" → "From scratch"
3. Name it "vibeb2b" and select your workspace

### 2. Configure OAuth Permissions

In your app settings, go to **OAuth & Permissions**:

**Required Scopes:**
- `chat:write` - Send messages to channels

**Optional Scopes (for public channels):**
- `chat:write.public` - Send messages to public channels

### 3. Install App & Get Tokens

1. Click **Install to Workspace**
2. Copy the **Bot User OAuth Token** (starts with `xoxb-`)
3. Get your channel ID:
   - Right-click channel → Copy link
   - ID is the last part (e.g., `C1234567890`)

### 4. Configure Environment

Add to your `.env` file:

```env
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_CHANNEL_ID=C1234567890
SLACK_SIGNING_SECRET=your-signing-secret-here
```

### 5. Add Bot to Channel

Invite the bot to your desired channel: `@vibeb2b`

## Recall.ai Meeting Recording Setup

VibeB2B includes integration with [Recall.ai](https://recall.ai) for automated meeting recording and transcription.

### Prerequisites

- ngrok account and CLI (automatically installed via setup script)
- Recall.ai API key

### Installation & Setup

1. **Install Python dependencies**:
   ```bash
   pip3 install -r src/recall/requirements.txt
   ```

2. **Configure environment variables**:
   Copy `.env.example` to `.env` and fill in your settings:
   ```bash
   cp .env.example .env
   ```

   Required environment variables:
   ```env
   RECALL_API_KEY=your-recall-api-key-here
   NGROK_AUTHTOKEN=your-ngrok-authtoken-here  # Optional - enables auto-tunneling
   RECALL_WEBSOCKET_URL=wss://your-websocket-server-url  # Optional
   ```

3. **Get your ngrok authtoken** (optional but recommended):
   - Sign up at [ngrok.com](https://dashboard.ngrok.com/signup)
   - Get your authtoken from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
   - Add it to your `.env` file as `NGROK_AUTHTOKEN`

4. **Start the server**:
   ```bash
   npm run recall
   ```

   For local development, no external tunneling is needed. If `NGROK_AUTHTOKEN` is set, the server will:
   - Automatically authenticate with ngrok
   - Start an ngrok tunnel to port 3000
   - Use the tunnel URL for webhooks
   - Clean up the tunnel when the server stops

   Without `NGROK_AUTHTOKEN`, it runs in localhost mode (no external access).

### Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Terminal 1: Start Flask server
npm run recall

# Terminal 2: Start ngrok tunnel
ngrok http 3000
```

Copy the ngrok URL and update `src/recall/recallAPI.py` as described above.

### Testing

Test your setup by calling the health endpoint:
```bash
curl https://your-ngrok-url.ngrok-free.app/health
```

### Usage

#### Quick Start - Start a Bot Directly

Start recording a meeting directly from the command line:

```bash
npm run recall "https://zoom.us/j/123456789"
```

This will automatically create a bot and start recording the meeting.

#### API Endpoint Usage

Alternatively, start recording by sending a POST request to the `/start_VB2B` endpoint:

```bash
curl -X POST https://your-ngrok-url.ngrok-free.app/start_VB2B \
  -H "Content-Type: application/json" \
  -d '{"meeting_url": "https://zoom.us/j/123456789"}'
```

#### Meeting Requirements

**Supported Platforms:**
- Zoom
- Google Meet
- Microsoft Teams
- Webex
- And other major video conferencing platforms

**Requirements:**
- Meeting must be **publicly accessible** (no password protection)
- Meeting must allow **guests to join without registration**
- Bot needs to be able to join before the meeting starts (for best results)
- Meeting URL must be valid and accessible

**What Gets Recorded:**
- Audio transcripts (real-time)
- Video frames (if websocket URL is configured)
- Participant metadata

Transcripts are automatically saved to `src/recall/transcript.db` as SQLite database.

## Development

### Interactive Control Center

For an interactive experience managing all services:

```bash
npm run main
```

This launches the **VibeB2B Control Center** with commands like:
- `bot` - Create meeting recording bots
- `server` - Start/stop Recall Flask server
- `mastra` - Start/stop Mastra AI server
- `health` - Check service status
- `status` - Show process information

### Individual Services

Run individual services:

```bash
# Mastra AI development server
npm run dev

# Recall Flask server only
npm run recall

# Create bot directly
npm run recall "https://meeting-url.com"
```

## Build

```bash
npm run build
npm start
```
