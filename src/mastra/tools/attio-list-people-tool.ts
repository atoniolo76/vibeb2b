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

  // First, get the list of objects to find the people object ID
  try {
    console.log('🔍 Getting Attio objects to find people object...');
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

      // Find the people object
      const peopleObject = objectsData.data?.find(obj => obj.api_slug === 'people');
      if (peopleObject) {
        console.log(`✅ Found people object: ${peopleObject.id}`);

        // Use the correct object ID for listing people
        const endpoint = `https://api.attio.com/v2/objects/${peopleObject.id}/records?limit=${limit}`;
        console.log(`Trying Attio endpoint: ${endpoint}`);

        response = await fetch(endpoint, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${apiToken}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          workingEndpoint = endpoint;
          console.log(`✅ Found working endpoint: ${endpoint}`);
        }
      }
    }
  } catch (error) {
    console.log(`❌ Failed to discover objects: ${error.message}`);
  }

  // Fallback to trying common endpoints if object discovery failed
  if (!response || !response.ok) {
    console.log('🔄 Object discovery failed, trying common endpoints...');
    const fallbackEndpoints = [
      `https://api.attio.com/v2/objects/people/records?limit=${limit}`,
      `https://api.attio.com/v2/lists/people/entries?limit=${limit}`,
    ];

    for (const endpoint of fallbackEndpoints) {
      try {
        console.log(`Trying Attio endpoint: ${endpoint}`);
        response = await fetch(endpoint, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${apiToken}`,
            'Content-Type': 'application/json',
          },
        });

        console.log(`Endpoint ${endpoint} returned status: ${response.status}`);

        if (response.ok) {
          workingEndpoint = endpoint;
          console.log(`✅ Found working endpoint: ${endpoint}`);
          break;
        } else {
          const errorText = await response.text();
          console.log(`❌ Endpoint ${endpoint} failed: ${response.status} - ${errorText}`);
        }
      } catch (error) {
        console.log(`❌ Endpoint ${endpoint} error: ${error.message}`);
        continue;
      }
    }
  }

  if (!response || !response.ok) {
    const errorText = await response?.text() || 'No response';
    let errorMessage = `Could not access Attio API. Status: ${response?.status || 'Unknown'}`;

    if (response?.status === 401) {
      errorMessage += '\n❌ Authentication failed. Please check your ATTIO_API_TOKEN in the .env file.';
    } else if (response?.status === 403) {
      errorMessage += '\n❌ Insufficient permissions. Your API token needs "Read access to records" scope.';
      errorMessage += '\n   Go to Attio Workspace Settings → Developers → [your integration] → Scopes';
    } else if (response?.status === 404) {
      errorMessage += '\n❌ Endpoint not found. The Attio API structure may have changed.';
    }

    throw new Error(`${errorMessage}\nResponse: ${errorText}`);
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
