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

## Development

```bash
npm run dev
```

## Build

```bash
npm run build
npm start
```
