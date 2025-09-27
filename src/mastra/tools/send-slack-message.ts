import { createTool } from '@mastra/core/tools';
import { WebClient } from '@slack/web-api';
import { z } from 'zod';

// Initialize Slack client with bot token from environment
const slackClient = new WebClient(process.env.SLACK_BOT_TOKEN);

export const sendSlackMessageTool = createTool({
  id: 'send-slack-message',
  description: 'Send a message to a configured Slack channel',
  inputSchema: z.object({
    message: z.string().describe('The message to send to Slack'),
  }),
  outputSchema: z.object({
    success: z.boolean(),
    messageId: z.string().optional(),
    error: z.string().optional(),
  }),
  execute: async ({ context }) => {
    try {
      const channelId = process.env.SLACK_CHANNEL_ID;

      if (!channelId) {
        throw new Error('SLACK_CHANNEL_ID environment variable is not set');
      }

      if (!process.env.SLACK_BOT_TOKEN) {
        throw new Error('SLACK_BOT_TOKEN environment variable is not set');
      }

      // Send message to Slack
      const result = await slackClient.chat.postMessage({
        channel: channelId,
        text: context.message,
      });

      return {
        success: true,
        messageId: result.ts,
      };
    } catch (error) {
      console.error('Error sending Slack message:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  },
});
