const express = require('express');
const { config } = require('dotenv');

// Load environment variables
config();

const app = express();
const PORT = 3001;

// Middleware
app.use(express.json());

// Webhook endpoint for Recall.ai
app.post('/webhook', async (req, res) => {
  try {
    const { event, data } = req.body;

    console.log(`📨 Webhook received:`, { event, eventType: typeof event, data });

    if (event === 'recording.done') {
      // Bot recording completed
      const recordingId = data.recording.id;
      const title = data.recording.metadata?.title || 'Unknown';

      console.log(`🎯 Bot recording completed: ${recordingId} - "${title}"`);

      // Just acknowledge receipt - no processing needed yet
      res.status(200).json({
        status: 'received',
        recordingId,
        title,
        timestamp: new Date().toISOString()
      });
    } else if (event === 'sdk_upload.complete') {
      // Desktop SDK recording upload completed
      const recordingId = data.recording.id;
      const sdkUploadId = data.sdk_upload.id;
      const title = data.recording.metadata?.title || 'Unknown';

      console.log(`🎯 Desktop SDK recording completed: ${recordingId} - "${title}"`);
      console.log(`📤 SDK Upload ID: ${sdkUploadId}`);

      // This is when Desktop SDK recordings are ready for download
      // The recording should now have download URLs available
      res.status(200).json({
        status: 'desktop_recording_ready',
        recordingId,
        sdkUploadId,
        title,
        timestamp: new Date().toISOString()
      });
    } else if (event === 'sdk_upload.failed') {
      // Desktop SDK recording upload failed
      const recordingId = data.recording?.id || 'unknown';
      const sdkUploadId = data.sdk_upload?.id || 'unknown';

      console.error(`❌ Desktop SDK recording failed: ${recordingId}, Upload: ${sdkUploadId}`);

      res.status(200).json({
        status: 'desktop_recording_failed',
        recordingId,
        sdkUploadId,
        timestamp: new Date().toISOString()
      });
    } else {
      console.log(`📨 Other webhook event: ${event}`);
      res.status(200).json({ status: 'ignored', event });
    }
  } catch (error) {
    console.error('Webhook error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Test endpoint
app.post('/test', (req, res) => {
  console.log('🧪 Test endpoint called:', req.body);
  res.json({ status: 'test_received', body: req.body });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`🚀 Webhook server listening on port ${PORT}`);
  console.log(`📡 Webhook endpoint: http://localhost:${PORT}/webhook`);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('🛑 Shutting down webhook server...');
  server.close(() => {
    console.log('✅ Webhook server closed');
    process.exit(0);
  });
});

process.on('SIGTERM', () => {
  console.log('🛑 Shutting down webhook server...');
  server.close(() => {
    console.log('✅ Webhook server closed');
    process.exit(0);
  });
});
