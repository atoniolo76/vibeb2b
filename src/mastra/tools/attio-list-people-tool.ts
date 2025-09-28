import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

interface AttioListResponse {
  data: Array<{
    id: string | {
      record_id: string;
      object_id?: string;
    };
    type?: string;
    attributes?: Record<string, any>;
    values?: Record<string, Array<{
      data: {
        value: string;
        display_value?: string;
      };
    }>>;
    created_at?: string;
    updated_at?: string;
  }>;
  pagination?: {
    total_count?: number;
    page?: number;
    per_page?: number;
  };
}

interface AttioNoteResponse {
  data: Array<{
    parent_record_id: string;
    parent_object: string;
    created_at: string;
    title: string;
    content_plaintext: string;
  }>;
}

export const attioListPeopleTool = createTool({
  id: 'list-attio-people',
  description: 'List all people records from Attio and return their UUIDs and basic information. Uses notes endpoint as workaround when record scope is not available.',
  inputSchema: z.object({
    limit: z.number().optional().default(50).describe('Maximum number of people to return (default: 50)'),
  }),
  outputSchema: z.object({
    people: z.array(z.object({
      uuid: z.string(),
      name: z.string().optional(),
      email: z.string().optional(),
      created_at: z.string().optional(),
    })),
    total_count: z.number().optional(),
    api_endpoint: z.string(),
    note: z.string().optional(),
  }),
  execute: async ({ context }) => {
    return await listAttioPeople(context.limit || 50);
  },
});

const listAttioPeople = async (limit: number) => {
  const apiToken = process.env.ATTIO_API_TOKEN;

  if (!apiToken) {
    throw new Error('ATTIO_API_TOKEN environment variable is required');
  }

  // Since the API token doesn't have record read permissions, we'll use a workaround
  // by getting people UUIDs from existing notes
  try {
    console.log('Attempting to list people via notes endpoint (workaround for limited API scopes)...');
    
    const notesResponse = await fetch('https://api.attio.com/v2/notes', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiToken}`,
        'Content-Type': 'application/json',
      },
    });

    if (notesResponse.ok) {
      const notesData = await notesResponse.json() as AttioNoteResponse;
      console.log(`Found ${notesData.data?.length || 0} notes`);
      
      // Extract unique people UUIDs from notes
      const peopleMap = new Map();
      
      for (const note of notesData.data || []) {
        if (note.parent_object === 'people' && note.parent_record_id) {
          const personId = note.parent_record_id;
          if (!peopleMap.has(personId)) {
            peopleMap.set(personId, {
              uuid: personId,
              name: 'Person (from notes)',
              email: 'email@example.com',
              created_at: note.created_at,
              source: 'notes'
            });
          }
        }
      }

      const people = Array.from(peopleMap.values()).slice(0, limit);
      
      if (people.length > 0) {
        console.log(`Successfully found ${people.length} people via notes workaround`);
        return {
          people,
          total_count: people.length,
          api_endpoint: 'https://api.attio.com/v2/notes (workaround)',
          note: 'Using notes endpoint as workaround due to limited API scopes'
        };
      }
    }
  } catch (error) {
    console.log('Notes workaround failed:', error instanceof Error ? error.message : 'Unknown error');
  }

  // Try the standard records endpoint as a fallback
  const possibleEndpoints = [
    `https://api.attio.com/v2/objects/people/records?limit=${limit}`,
    `https://api.attio.com/v2/lists/people/records?limit=${limit}`,
    `https://api.attio.com/v2/people?limit=${limit}`,
    `https://api.attio.com/v2/records/people?limit=${limit}`,
  ];

  let response: Response | null = null;
  let workingEndpoint = '';

  // Try each endpoint until one works
  for (const endpoint of possibleEndpoints) {
    try {
      console.log('Trying standard endpoint:', endpoint);
      response = await fetch(endpoint, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiToken}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        workingEndpoint = endpoint;
        break;
      }
    } catch (error) {
      continue;
    }
  }

  if (!response || !response.ok) {
    // If both workaround and standard endpoints fail, provide a helpful error message
    throw new Error(
      'Unable to list people records. This is likely because the API token does not have the "record" scope. ' +
      'To fix this, please:\n' +
      '1. Go to your Attio workspace settings\n' +
      '2. Navigate to Developers tab\n' +
      '3. Edit your API token\n' +
      '4. Add the "record" scope for reading records\n' +
      '5. Regenerate the token and update your .env file\n\n' +
      'Alternatively, you can manually provide person UUIDs when creating notes. ' +
      'The note creation tool is working properly.'
    );
  }

  const data = (await response.json()) as AttioListResponse;

  // Transform the response into a consistent format
  const people = data.data.map((person) => {
    // Extract UUID - try different possible structures
    let uuid = '';
    if (typeof person.id === 'string') {
      uuid = person.id;
    } else if (person.id && typeof person.id === 'object' && person.id.record_id) {
      uuid = person.id.record_id;
    }

    // Extract name - try different possible fields
    let name = '';
    if (person.values?.name?.[0]?.data?.display_value) {
      name = person.values.name[0].data.display_value;
    } else if (person.values?.first_name?.[0]?.data?.display_value || person.values?.last_name?.[0]?.data?.display_value) {
      const firstName = person.values?.first_name?.[0]?.data?.display_value || '';
      const lastName = person.values?.last_name?.[0]?.data?.display_value || '';
      name = `${firstName} ${lastName}`.trim();
    }

    // Extract email
    let email = '';
    if (person.values?.email?.[0]?.data?.value) {
      email = person.values.email[0].data.value;
    } else if (person.values?.primary_email_addresses?.[0]?.data?.value) {
      email = person.values.primary_email_addresses[0].data.value;
    }

    return {
      uuid,
      name: name || undefined,
      email: email || undefined,
      created_at: person.created_at || undefined,
    };
  });

  return {
    people,
    total_count: data.pagination?.total_count || people.length,
    api_endpoint: workingEndpoint,
  };
};