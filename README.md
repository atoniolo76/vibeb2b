# VibeB2B

A Mastra-based AI assistant for sales insights and CRM management.

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

## AI Configuration

### Google AI Setup

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GOOGLE_GENERATIVE_AI_API_KEY`

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

## Attio CRM Setup

### 1. Get Your Attio API Token

1. Go to [Attio](https://attio.com) and sign in to your account
2. Navigate to **Settings** → **API Tokens**
3. Click **Create API Token**
4. Give it a name (e.g., "VibeB2B Integration")
5. Copy the generated token

### 2. Configure Environment

Add to your `.env` file:

```env
ATTIO_API_TOKEN=your-attio-api-token-here
```

### 3. Usage

The AI agents can now:
- Read people data from Attio CRM
- Add notes to client records
- Update CRM information based on insights

## Development

### GUI Application

For an interactive experience managing the Mastra server:

```bash
npm run electron-dev
```

This launches the **VibeB2B GUI** with controls to:
- Start/stop the Mastra AI server
- Monitor server health and status
- View real-time logs
- Manage environment variables

### Individual Services

Run individual services:

```bash
# Mastra AI development server only
npm run dev

# GUI application only
npm run electron-dev
```

## Build

```bash
npm run build
npm start
```
