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

export const attioListPeopleTool = createTool({
  id: 'list-attio-people',
  description: 'List all people records from Attio and return their UUIDs and basic information',
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

  // Try different possible endpoints for listing people
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
    throw new Error('Could not find a working endpoint to list people records. Please check your Attio API documentation.');
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
