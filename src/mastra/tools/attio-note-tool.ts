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
  meeting_id?: string;
  created_at?: string;
}) => {
  const apiToken = process.env.ATTIO_API_TOKEN;

  if (!apiToken) {
    throw new Error('ATTIO_API_TOKEN environment variable is required');
  }

  // First, discover the notes object ID
  let notesObjectId = '';
  try {
    console.log('🔍 Discovering notes object from Attio...');
    const objectsResponse = await fetch('https://api.attio.com/v2/objects', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiToken}`,
        'Content-Type': 'application/json',
      },
    });

    if (objectsResponse.ok) {
      const objectsData = await objectsResponse.json();
      console.log('📊 Available objects:', objectsData.data?.map(obj => ({ id: obj.id, name: obj.api_slug })));

      // Find the notes object
      const notesObject = objectsData.data?.find(obj => obj.api_slug === 'notes');
      if (notesObject) {
        notesObjectId = notesObject.id;
        console.log(`✅ Found notes object: ${notesObjectId}`);
      }
    }
  } catch (error) {
    console.log(`❌ Failed to discover notes object: ${error.message}`);
  }

  // Try different possible endpoints for notes
  const possibleEndpoints = [];
  if (notesObjectId) {
    possibleEndpoints.push(`https://api.attio.com/v2/objects/${notesObjectId}/records`);
  }
  possibleEndpoints.push(
    'https://api.attio.com/v2/notes',
    'https://api.attio.com/v2/objects/notes/records'
  );

  let response: Response | null = null;
  let workingEndpoint = '';

  for (const endpoint of possibleEndpoints) {
    try {
      console.log(`Trying Attio notes endpoint: ${endpoint}`);

      const body = {
        data: {
          parent_object: 'people',
          parent_record_id: params.parent_record_id,
          title: params.title,
          format: 'plaintext',
          content: params.content,
          ...(params.meeting_id && { meeting_id: params.meeting_id }),
          ...(params.created_at && { created_at: params.created_at }),
        },
      };

      response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      console.log(`Notes endpoint ${endpoint} returned status: ${response.status}`);

      if (response.ok) {
        workingEndpoint = endpoint;
        console.log(`✅ Found working notes endpoint: ${endpoint}`);
        break;
      } else {
        const errorText = await response.text();
        console.log(`❌ Notes endpoint ${endpoint} failed: ${response.status} - ${errorText}`);
      }
    } catch (error) {
      console.log(`❌ Notes endpoint ${endpoint} error: ${error.message}`);
      continue;
    }
  }

  if (!response || !response.ok) {
    const errorMsg = await response?.text() || 'No response';
    let errorMessage = `Could not create note in Attio CRM. Status: ${response?.status || 'Unknown'}`;

    if (response?.status === 401) {
      errorMessage += '\n❌ Authentication failed. Please check your ATTIO_API_TOKEN in the .env file.';
    } else if (response?.status === 403) {
      errorMessage += '\n❌ Insufficient permissions. Your API token needs "Write access to records" scope.';
      errorMessage += '\n   Go to Attio Workspace Settings → Developers → [your integration] → Scopes';
    } else if (response?.status === 404) {
      errorMessage += '\n❌ Record not found. The parent_record_id may be invalid, or notes may not be available.';
    }

    throw new Error(`${errorMessage}\nResponse: ${errorMsg}`);
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
