import { config } from 'dotenv';
import { SynchronousRecording } from './recall-ai/index';

// Load environment variables
config();
console.log('🔧 Environment loaded');

const RECALL_API_KEY = process.env.RECALL_AI_API_KEY;
console.log('🔑 API Key present:', RECALL_API_KEY ? 'YES' : 'NO');

if (!RECALL_API_KEY) {
  console.error('RECALL_AI_API_KEY not found in environment variables');
  process.exit(1);
}

// Synchronous recording test
async function testSynchronousRecording() {
  console.log('🎯 Starting Synchronous Desktop Recording Test');
  console.log('This will detect ONE meeting, record it, upload, and download automatically\n');

  const recorder = new SynchronousRecording();
  let currentWindowId: string | null = null;
  let hasProcessedMeeting = false;

  try {
    // Initialize SDK
    console.log('1️⃣ Initializing SDK...');
    await recorder.initialize();

    // Request permissions
    console.log('2️⃣ Requesting permissions...');
    await recorder.requestPermissions();

    console.log('✅ Setup complete! Listening for meeting detection...');
    console.log('📋 Start a Zoom or Google Meet meeting to begin recording\n');

    // Set up event listeners
    const sdk = (await import('@recallai/desktop-sdk')).default;

    // Meeting detected - start recording
    sdk.addEventListener('meeting-detected', async (evt: any) => {
      if (hasProcessedMeeting) return; // Only process one meeting

      console.log('🎯 Meeting detected!');
      console.log(`   Title: ${evt.window?.title}`);
      console.log(`   Platform: ${evt.meeting?.platform || 'unknown'}`);

      currentWindowId = evt.window?.id;
      if (!currentWindowId) {
        console.error('❌ No window ID found');
        return;
      }

      hasProcessedMeeting = true;
      console.log(`📺 Window ID: ${currentWindowId}`);

      try {
        // Start recording
        console.log('3️⃣ Starting recording...');
        await recorder.startRecording(currentWindowId);

        // Set up recording end listener
        sdk.addEventListener('recording-ended', async () => {
          console.log('🏁 Recording ended, processing...');

          try {
            // Stop recording and upload
            console.log('4️⃣ Stopping recording and uploading...');
            await recorder.stopRecording(currentWindowId!);

            // For Desktop SDK recordings, we should wait for the webhook instead of polling
            console.log('5️⃣ Waiting for sdk_upload.complete webhook...');
            console.log('📡 Webhook server should receive the completion event');
            console.log('⚠️ Since this is a test script, we\'ll wait a bit then try to download');

            // Wait for webhook (in a real app, this would be handled by the webhook endpoint)
            await new Promise(resolve => setTimeout(resolve, 60000)); // Wait 1 minute

            // Get the most recent recording (should be ours)
            const recordings = await recorder.getRecordings(1);
            if (recordings.length === 0) {
              throw new Error('No recordings found after waiting');
            }

            const recordingId = recordings[0].id;
            console.log(`📹 Found recording: ${recordingId}`);

            // Download to footage folder
            console.log('6️⃣ Downloading to footage folder...');
            const downloadPath = await recorder.downloadRecording(recordingId);

            console.log('🎉 Process complete!');
            console.log(`📁 Recording saved to: ${downloadPath}`);

            // Clean up and exit
            await recorder.cleanup();
            process.exit(0);

          } catch (error: any) {
            console.error('❌ Processing failed:', error.message);
            await recorder.cleanup();
            process.exit(1);
          }
        });

      } catch (error: any) {
        console.error('❌ Failed to start recording:', error.message);
        await recorder.cleanup();
        process.exit(1);
      }
    });

    // Handle permissions
    sdk.addEventListener('permissions-granted', () => {
      console.log('✅ Permissions granted');
    });

    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n🛑 Shutting down...');
      await recorder.cleanup();
      process.exit(0);
    });

    // Keep alive
    console.log('🔄 Listening for meetings... (Press Ctrl+C to exit)');

  } catch (error: any) {
    console.error('❌ Test failed:', error.message);
    await recorder.cleanup();
    process.exit(1);
  }
}


// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  testSynchronousRecording();
}
