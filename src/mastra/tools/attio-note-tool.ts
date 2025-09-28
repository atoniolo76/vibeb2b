import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

interface AttioNoteResponse {
  data: {
    id: {
      object_id: string;
      record_id: string;
    };
    created_at: string;
    updated_at: string;
    title: string;
    content: string;
    format: string;
    parent_object: string;
    parent_record_id: string;
    meeting_id?: string;
  };
}

export const attioNoteTool = createTool({
  id: 'create-attio-note',
  description: 'Create a note in Attio and associate it with a person record',
  inputSchema: z.object({
    parent_record_id: z.string().describe('The UUID of the person record to associate the note with'),
    title: z.string().describe('Title of the note'),
    content: z.string().describe('Content of the note'),
    format: z.enum(['plaintext', 'markdown']).optional().default('markdown').describe('Format of the content (default: markdown)'),
    meeting_id: z.string().optional().describe('Optional meeting ID to associate with the note'),
    created_at: z.string().optional().describe('Optional ISO 8601 timestamp for when the note was created'),
  }),
  outputSchema: z.object({
    note_id: z.string(),
    created_at: z.string(),
    updated_at: z.string(),
    title: z.string(),
    content: z.string(),
    parent_record_id: z.string(),
  }),
  execute: async ({ context }) => {
    return await createAttioNote(context);
  },
});

const createAttioNote = async (params: {
  parent_record_id: string;
  title: string;
  content: string;
  format?: 'plaintext' | 'markdown';
  meeting_id?: string;
  created_at?: string;
}) => {
  const apiToken = process.env.ATTIO_API_TOKEN;

  if (!apiToken) {
    throw new Error('ATTIO_API_TOKEN environment variable is required');
  }

  const url = 'https://api.attio.com/v2/notes';

  const body = {
    data: {
      parent_object: 'people',
      parent_record_id: params.parent_record_id,
      title: params.title,
      format: params.format || 'markdown',
      content: params.content,
      ...(params.meeting_id && { meeting_id: params.meeting_id }),
      ...(params.created_at && { created_at: params.created_at }),
    },
  };

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Attio API error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  const data = (await response.json()) as AttioNoteResponse;

  return {
    note_id: data.data.id.record_id,
    created_at: data.data.created_at,
    updated_at: data.data.updated_at,
    title: data.data.title,
    content: data.data.content,
    parent_record_id: data.data.parent_record_id,
  };
};
