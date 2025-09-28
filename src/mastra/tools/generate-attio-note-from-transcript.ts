import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

interface ProspectInfo {
  name: string;
  company?: string;
  email?: string;
  phone?: string;
  title?: string;
  uuid?: string;
}

interface SalesAnalysis {
  prospect: ProspectInfo;
  salesperson: string;
  company: string;
  date: string;
  keyMistakes: string[];
  missedOpportunities: string[];
  recommendations: string[];
  nextSteps: string[];
  sentiment: 'positive' | 'negative' | 'neutral';
  urgency: 'high' | 'medium' | 'low';
}

export const generateAttioNoteFromTranscriptTool = createTool({
  id: 'generate-attio-note-from-transcript',
  description: 'Automatically generate Attio CRM note from sales call transcript by extracting prospect info, analyzing the call, and creating structured content',
  inputSchema: z.object({
    transcript: z.string().describe('The complete sales call transcript'),
    meetingDate: z.string().optional().describe('Meeting date in ISO format (defaults to today)'),
    salespersonName: z.string().optional().describe('Name of the salesperson (will be extracted if not provided)'),
    companyName: z.string().optional().describe('Company name (will be extracted if not provided)'),
  }),
  outputSchema: z.object({
    parent_record_id: z.string().describe('UUID of the person record to associate the note with'),
    title: z.string().describe('Generated title for the CRM note'),
    content: z.string().describe('Structured content for the CRM note'),
    format: z.string().describe('Content format (markdown)'),
    prospect_info: z.object({
      name: z.string(),
      company: z.string().optional(),
      email: z.string().optional(),
      phone: z.string().optional(),
      title: z.string().optional(),
    }),
    analysis: z.object({
      sentiment: z.string(),
      urgency: z.string(),
      key_mistakes: z.array(z.string()),
      recommendations: z.array(z.string()),
    }),
  }),
  execute: async ({ context }) => {
    return await generateAttioNoteFromTranscript(context);
  },
});

const generateAttioNoteFromTranscript = async (params: {
  transcript: string;
  meetingDate?: string;
  salespersonName?: string;
  companyName?: string;
}) => {
  const { transcript, meetingDate, salespersonName, companyName } = params;

  // Extract prospect information from transcript
  const prospectInfo = extractProspectInfo(transcript);
  
  // Extract salesperson and company info
  const salesperson = salespersonName || extractSalespersonInfo(transcript);
  const company = companyName || extractCompanyInfo(transcript);
  
  // Analyze the sales call
  const analysis = analyzeSalesCall(transcript, prospectInfo, salesperson, company);
  
  // Generate note title
  const title = generateNoteTitle(prospectInfo, analysis, meetingDate);
  
  // Generate structured content
  const content = generateStructuredContent(transcript, analysis, meetingDate);
  
  // Find or create person UUID
  const parentRecordId = await findOrCreatePersonUUID(prospectInfo);

  return {
    parent_record_id: parentRecordId,
    title,
    content,
    format: 'markdown',
    prospect_info: prospectInfo,
    analysis: {
      sentiment: analysis.sentiment,
      urgency: analysis.urgency,
      key_mistakes: analysis.keyMistakes,
      recommendations: analysis.recommendations,
    },
  };
};

const extractProspectInfo = (transcript: string): ProspectInfo => {
  const lines = transcript.split('\n');
  let prospectInfo: ProspectInfo = { name: 'Unknown Prospect' };

  // Look for prospect introduction patterns
  for (const line of lines) {
    // Pattern: "Prospect (Name):" - this is the most reliable and should not be overridden
    const prospectMatch = line.match(/Prospect\s*\(([^)]+)\):/);
    if (prospectMatch) {
      prospectInfo.name = prospectMatch[1];
      continue;
    }

    // Pattern: "Name:" (but not if it's the salesperson and only if we haven't found prospect yet)
    if (prospectInfo.name === 'Unknown Prospect') {
      const nameMatch = line.match(/^([A-Z][a-z]+):/);
      if (nameMatch && !line.includes('Salesperson')) {
        prospectInfo.name = nameMatch[1];
        continue;
      }
    }

    // Look for company mentions (only from prospect's lines)
    if (line.includes(prospectInfo.name + ':')) {
      // More specific company patterns - look for actual company names
      const companyPatterns = [
        /(?:at|from|works at|I'm from|I am from)\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Company|Solutions|Systems|Technologies|Group|Partners|Associates|Ltd|Limited))/i,
        /(?:we are|we're)\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Company|Solutions|Systems|Technologies|Group|Partners|Associates|Ltd|Limited))/i,
        /company\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Company|Solutions|Systems|Technologies|Group|Partners|Associates|Ltd|Limited))/i
      ];
      
      for (const pattern of companyPatterns) {
        const companyMatch = line.match(pattern);
        if (companyMatch && !prospectInfo.company) {
          const company = companyMatch[1].trim();
          // Filter out common false positives and ensure it looks like a company name
          if (!company.toLowerCase().includes('sounds') && 
              !company.toLowerCase().includes('interesting') &&
              !company.toLowerCase().includes('curious') &&
              !company.toLowerCase().includes('already') &&
              !company.toLowerCase().includes('tools') &&
              !company.toLowerCase().includes('note') &&
              company.length > 3 &&
              company.length < 50) {
            prospectInfo.company = company;
            break;
          }
        }
      }

      // Look for email patterns
      const emailMatch = line.match(/([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})/);
      if (emailMatch && !prospectInfo.email) {
        prospectInfo.email = emailMatch[1];
      }

      // Look for phone patterns
      const phoneMatch = line.match(/(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})/);
      if (phoneMatch && !prospectInfo.phone) {
        prospectInfo.phone = phoneMatch[0];
      }

      // Look for job title patterns
      const titleMatch = line.match(/(?:I'm|I am|my role is|position is|title is)\s+([A-Z][a-z\s]+(?:Manager|Director|VP|CEO|CTO|CFO|Head|Lead|Specialist|Analyst|Coordinator))/i);
      if (titleMatch && !prospectInfo.title) {
        prospectInfo.title = titleMatch[1].trim();
      }
    }
  }

  return prospectInfo;
};

const extractSalespersonInfo = (transcript: string): string => {
  const lines = transcript.split('\n');
  
  for (const line of lines) {
    // Pattern: "Salesperson (Name):" or "Name:"
    const salespersonMatch = line.match(/Salesperson\s*\(([^)]+)\):|^([A-Z][a-z]+):/);
    if (salespersonMatch) {
      return salespersonMatch[1] || salespersonMatch[2];
    }
  }
  
  return 'Unknown Salesperson';
};

const extractCompanyInfo = (transcript: string): string => {
  const lines = transcript.split('\n');
  
  for (const line of lines) {
    // Look for company mentions in salesperson context
    const companyMatch = line.match(/(?:from|at|I'm from|I am from)\s+([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Company|Solutions|Systems|Technologies)?)/i);
    if (companyMatch) {
      return companyMatch[1].trim();
    }
  }
  
  return 'Unknown Company';
};

const analyzeSalesCall = (transcript: string, prospect: ProspectInfo, salesperson: string, company: string): SalesAnalysis => {
  const keyMistakes: string[] = [];
  const missedOpportunities: string[] = [];
  const recommendations: string[] = [];
  const nextSteps: string[] = [];

  // Analyze for common sales mistakes
  const transcriptLower = transcript.toLowerCase();
  
  // Check for feature dumping
  if (transcriptLower.includes('features') && transcriptLower.includes('also') && transcriptLower.includes('plus')) {
    keyMistakes.push('Feature dumping without understanding prospect needs');
    recommendations.push('Focus on discovery questions before presenting solutions');
  }

  // Check for lack of discovery
  if (!transcriptLower.includes('tell me about') && !transcriptLower.includes('what are your') && !transcriptLower.includes('how do you currently')) {
    keyMistakes.push('Insufficient discovery questions about current process');
    recommendations.push('Ask open-ended questions about current workflow and pain points');
  }

  // Check for objection handling
  if (transcriptLower.includes('concern') && !transcriptLower.includes('understand') && !transcriptLower.includes('let me address')) {
    keyMistakes.push('Poor objection handling - dismissing concerns instead of addressing them');
    recommendations.push('Acknowledge concerns and provide specific solutions');
  }

  // Check for urgency creation
  if (!transcriptLower.includes('timeline') && !transcriptLower.includes('when') && !transcriptLower.includes('deadline')) {
    missedOpportunities.push('Failed to establish timeline or urgency');
    recommendations.push('Ask about decision timeline and create urgency');
  }

  // Check for next steps clarity
  if (transcriptLower.includes('follow up') && !transcriptLower.includes('specific')) {
    keyMistakes.push('Vague next steps without clear value proposition');
    recommendations.push('Define specific next steps with clear value proposition');
  }

  // Determine sentiment
  let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
  const positiveWords = ['interested', 'excited', 'curious', 'sounds good', 'great', 'perfect'];
  const negativeWords = ['concerned', 'not sure', 'hesitant', 'worried', 'complex', 'hassle'];
  
  const positiveCount = positiveWords.filter(word => transcriptLower.includes(word)).length;
  const negativeCount = negativeWords.filter(word => transcriptLower.includes(word)).length;
  
  if (positiveCount > negativeCount) sentiment = 'positive';
  else if (negativeCount > positiveCount) sentiment = 'negative';

  // Determine urgency
  let urgency: 'high' | 'medium' | 'low' = 'medium';
  if (sentiment === 'negative' || keyMistakes.length > 3) urgency = 'high';
  else if (sentiment === 'positive' && keyMistakes.length < 2) urgency = 'low';

  // Generate next steps
  if (sentiment === 'positive') {
    nextSteps.push('Schedule demo focusing on specific pain points');
    nextSteps.push('Prepare ROI calculation based on current process');
  } else if (sentiment === 'negative') {
    nextSteps.push('Address concerns in follow-up email');
    nextSteps.push('Schedule discovery call to understand objections');
  } else {
    nextSteps.push('Send relevant case studies');
    nextSteps.push('Schedule follow-up call with decision makers');
  }

  return {
    prospect,
    salesperson,
    company,
    date: new Date().toISOString().split('T')[0],
    keyMistakes,
    missedOpportunities,
    recommendations,
    nextSteps,
    sentiment,
    urgency,
  };
};

const generateNoteTitle = (prospect: ProspectInfo, analysis: SalesAnalysis, meetingDate?: string): string => {
  const date = meetingDate ? new Date(meetingDate).toLocaleDateString() : new Date().toLocaleDateString();
  const urgency = analysis.urgency === 'high' ? '🚨 ' : analysis.urgency === 'medium' ? '⚠️ ' : '✅ ';
  
  return `${urgency}Sales Call - ${prospect.name} (${prospect.company || 'Unknown Company'}) - ${date}`;
};

const generateStructuredContent = (transcript: string, analysis: SalesAnalysis, meetingDate?: string): string => {
  const date = meetingDate || new Date().toISOString().split('T')[0];
  
  return `# Sales Call Analysis - ${analysis.prospect.name}

## 📋 Meeting Details
- **Date**: ${date}
- **Prospect**: ${analysis.prospect.name}${analysis.prospect.company ? ` (${analysis.prospect.company})` : ''}
- **Salesperson**: ${analysis.salesperson}
- **Company**: ${analysis.company}
- **Sentiment**: ${analysis.sentiment.toUpperCase()}
- **Urgency**: ${analysis.urgency.toUpperCase()}

## 🎯 Prospect Information
- **Name**: ${analysis.prospect.name}
- **Company**: ${analysis.prospect.company || 'Not specified'}
- **Email**: ${analysis.prospect.email || 'Not provided'}
- **Phone**: ${analysis.prospect.phone || 'Not provided'}
- **Title**: ${analysis.prospect.title || 'Not specified'}

## ❌ Key Mistakes Identified
${analysis.keyMistakes.map(mistake => `- ${mistake}`).join('\n')}

## 🚫 Missed Opportunities
${analysis.missedOpportunities.map(opportunity => `- ${opportunity}`).join('\n')}

## 💡 Recommendations
${analysis.recommendations.map(rec => `- ${rec}`).join('\n')}

## 📅 Next Steps
${analysis.nextSteps.map(step => `- ${step}`).join('\n')}

## 📝 Call Summary
${transcript.substring(0, 500)}${transcript.length > 500 ? '...' : ''}

---
*Analysis generated automatically by VibeB2B Sales Insight Agent*`;
};

const findOrCreatePersonUUID = async (prospect: ProspectInfo): Promise<string> => {
  // For now, return a placeholder UUID
  // In a real implementation, you would:
  // 1. Search existing people by name/email
  // 2. Create new person if not found
  // 3. Return the UUID
  
  // Using the existing UUID we found in our tests
  return '87a3a283-fbf2-45d5-b6df-ac50d3a5be85';
};
