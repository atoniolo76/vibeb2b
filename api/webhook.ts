import { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { event, data } = req.body;

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
}
