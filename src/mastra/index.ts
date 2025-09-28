
import 'dotenv/config';
import { Mastra } from '@mastra/core/mastra';
import { PinoLogger } from '@mastra/loggers';
import { LibSQLStore } from '@mastra/libsql';
// import { weatherWorkflow } from './workflows/weather-workflow'; // COMMENTED OUT: Weather workflow
import { insightAgent } from './agents/insight-agent';

export const mastra = new Mastra({
  // workflows: { weatherWorkflow }, // COMMENTED OUT: Weather workflow
  workflows: {},
  agents: { insightAgent },
  storage: new LibSQLStore({
    // stores telemetry, evals, ... into memory storage, if it needs to persist, change to file:../mastra.db
    url: ":memory:",
  }),
  logger: new PinoLogger({
    name: 'Mastra',
    level: 'info',
  }),
});
