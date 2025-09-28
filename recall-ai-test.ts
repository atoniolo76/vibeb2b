import * as fs from 'fs';
import * as path from 'path';
import { config } from 'dotenv';
import {
  RecallAIDesktopRecording,
  DesktopRecordingOptions,
  DesktopRecordingEvents,
  RecordingStatus
} from './src/recall-ai/index';

// Load environment variables
config();
console.log('🔧 Environment loaded');

const RECALL_API_KEY = process.env.RECALL_AI_API_KEY;
console.log('🔑 API Key present:', RECALL_API_KEY ? 'YES' : 'NO');

if (!RECALL_API_KEY) {
  console.error('RECALL_AI_API_KEY not found in environment variables');
  process.exit(1);
}

// Test class for Desktop Recording
class DesktopRecordingTest {
  private recording: RecallAIDesktopRecording | null = null;
  private currentWindowId: string | null = null;
  private isRecording = false;
  private outputDir: string;

  constructor(outputDir = './desktop-recordings') {
    this.outputDir = outputDir;
    this.ensureOutputDir();
  }

  private ensureOutputDir() {
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
      console.log(`Created output directory: ${this.outputDir}`);
    }
  }

  // Set up event handlers
  private setupEventHandlers(): DesktopRecordingEvents {
    return {
      onMeetingDetected: (meeting) => {
        console.log('🎯 Meeting detected:', meeting);
        console.log(`   Platform: ${meeting.platform}`);
        console.log(`   Title: ${meeting.title}`);
        console.log(`   URL: ${meeting.url || 'N/A'}`);

        // Extract window ID from the MeetingInfo object
        if (meeting.windowId) {
          this.currentWindowId = meeting.windowId;
          console.log(`📺 Set current window ID: ${this.currentWindowId}`);

          // Auto-start recording for detected meetings
          if (!this.isRecording) {
            console.log('🚀 Auto-starting recording for detected meeting...');
            this.startRecordingForWindow(this.currentWindowId);
          }
        } else {
          console.log('⚠️ No window ID found in meeting detection event');
        }
      },

      onRecordingStarted: (uploadToken) => {
        console.log('✅ Recording started with upload token:', uploadToken.substring(0, 20) + '...');
        this.isRecording = true;
      },

      onRecordingEnded: () => {
        console.log('🏁 Recording ended');
        this.isRecording = false;
      },

      onUploadProgress: (progress) => {
        console.log(`📊 Upload progress: ${progress}%`);
      },

      onRecordingCompleted: (recordingId) => {
        console.log('🎉 Recording completed! Recording ID:', recordingId);
        console.log('You can now access the recording via the Recall.ai dashboard or API');
      },

      onRecordingFailed: (error) => {
        console.error('❌ Recording failed:', error.message);
        this.isRecording = false;
      },

      onPermissionsGranted: () => {
        console.log('✅ All required permissions granted');
      },

      onPermissionsDenied: (deniedPermissions) => {
        console.error('❌ Permissions denied:', deniedPermissions);
        console.log('Please grant the required permissions and try again');
      },

      onError: (error) => {
        console.error('❌ Desktop recording error:', error.message);
      }
    };
  }

  // Initialize the desktop SDK
  async initialize(): Promise<void> {
    console.log('🔧 Initializing desktop recording...');

    const options: DesktopRecordingOptions = {
      outputDir: this.outputDir,
      transcriptProvider: 'assembly_ai_streaming',
      videoLayout: 'gallery_view_v2'
    };

    const events = this.setupEventHandlers();

    this.recording = new RecallAIDesktopRecording(options, events);

    try {
      await this.recording.initializeSDK();
      console.log('✅ Desktop recording initialized successfully');
      console.log('🎧 Listening for meeting detection events...');
    } catch (error: any) {
      console.error('❌ Failed to initialize desktop recording:', error.message);
      throw error;
    }
  }

  // Request permissions
  async requestPermissions(): Promise<void> {
    if (!this.recording) {
      throw new Error('Desktop recording not initialized');
    }

    console.log('🔐 Requesting permissions...');
    try {
      await this.recording.requestPermissions();
      console.log('✅ Permission request completed');
    } catch (error: any) {
      console.error('❌ Failed to request permissions:', error.message);
      throw error;
    }
  }

  // Start recording for a specific window
  async startRecordingForWindow(windowId: string): Promise<void> {
    if (!this.recording) {
      throw new Error('Desktop recording not initialized');
    }

    if (this.isRecording) {
      console.log('⚠️ Already recording, skipping...');
      return;
    }

    console.log(`🎬 Starting recording for window: ${windowId}`);
    try {
      await this.recording.startRecording(windowId);
      this.currentWindowId = windowId;
    } catch (error: any) {
      console.error('❌ Failed to start recording:', error.message);
      throw error;
    }
  }

  // Stop current recording
  async stopRecording(): Promise<void> {
    if (!this.recording || !this.currentWindowId) {
      console.log('⚠️ No active recording to stop');
      return;
    }

    console.log('🛑 Stopping recording...');
    try {
      await this.recording.stopRecording(this.currentWindowId);
    } catch (error: any) {
      console.error('❌ Failed to stop recording:', error.message);
      throw error;
    }
  }

  // Get recording status
  getStatus(): RecordingStatus {
    if (!this.recording) {
      return { state: 'failed', message: 'Not initialized' };
    }
    return this.recording.getRecordingStatus();
  }

  // Clean up
  async cleanup(): Promise<void> {
    if (this.recording) {
      console.log('🧹 Cleaning up desktop recording...');
      await this.recording.cleanup();
      this.recording = null;
    }
    this.isRecording = false;
    this.currentWindowId = null;
  }

  // Get available recordings
  async getRecordings(limit = 5): Promise<any[]> {
    if (!this.recording) {
      throw new Error('Desktop recording not initialized');
    }

    try {
      return await this.recording.getRecordings(limit);
    } catch (error: any) {
      console.error('Failed to get recordings:', error.message);
      throw error;
    }
  }

  // Download a recording
  async downloadRecording(recordingId: string, outputPath?: string): Promise<string> {
    if (!this.recording) {
      throw new Error('Desktop recording not initialized');
    }

    try {
      return await this.recording.downloadRecording(recordingId, outputPath);
    } catch (error: any) {
      console.error('Failed to download recording:', error.message);
      throw error;
    }
  }

  // Simulate meeting detection for testing
  simulateMeetingDetection(windowId: string, title: string = 'Test Meeting'): void {
    console.log(`🎭 Simulating meeting detection for testing...`);
    this.currentWindowId = windowId;

    // In a real scenario, this would be triggered by the SDK detecting a meeting
    // For testing, we'll manually trigger recording
    console.log('💡 For testing, you can now call startRecordingForWindow() manually');
    console.log(`   Window ID: ${windowId}`);
    console.log(`   Meeting Title: ${title}`);
  }
}

// Test function for desktop recording
async function testDesktopRecording() {
  console.log('🎯 Starting Desktop Recording Test');
  console.log('This test will initialize the Recall.ai Desktop SDK and listen for meetings');
  console.log('To test recording, start a Zoom or Google Meet meeting while this is running\n');

  const test = new DesktopRecordingTest('./desktop-test-recordings');

  try {
    // Initialize the desktop SDK
    console.log('1️⃣ Initializing Desktop SDK...');
    await test.initialize();

    // Request permissions
    console.log('2️⃣ Requesting permissions...');
    await test.requestPermissions();

    console.log('✅ Setup complete! The desktop SDK is now active.');
    console.log('📋 Next steps:');
    console.log('   - Start a Zoom or Google Meet meeting');
    console.log('   - The SDK will automatically detect the meeting');
    console.log('   - Recording will start automatically');
    console.log('   - When done, press Ctrl+C to exit\n');

    // For testing purposes, simulate a meeting detection
    // In production, this would happen automatically when the SDK detects a meeting
    if (process.argv[2] === '--simulate') {
      console.log('🎭 Simulating meeting detection for testing...');
      test.simulateMeetingDetection('test-window-123', 'Test Meeting Window');
    }

    // Keep the process running to listen for events
    console.log('🔄 Listening for meeting detection events...');
    console.log('   Press Ctrl+C to exit');

    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n🛑 Shutting down...');
      await test.cleanup();
      process.exit(0);
    });

    // Keep alive
    setInterval(() => {
      const status = test.getStatus();
      console.log(`📊 Status: ${status.state}`);
    }, 10000);

  } catch (error: any) {
    console.error('❌ Test failed:', error.message);
    await test.cleanup();
    process.exit(1);
  }
}

// Interactive test function
async function interactiveTest() {
  console.log('🎯 Interactive Desktop Recording Test');
  console.log('=====================================');

  const test = new DesktopRecordingTest('./interactive-test-recordings');

  try {
    // Initialize
    await test.initialize();
    console.log('✅ Initialized');

    // Request permissions
    await test.requestPermissions();
    console.log('✅ Permissions requested');

    console.log('\n🎮 Interactive Controls:');
    console.log('  start <windowId> - Start recording for a window');
    console.log('  stop             - Stop current recording');
    console.log('  status           - Show current status');
    console.log('  simulate <title> - Simulate meeting detection');
    console.log('  list             - List recent recordings');
    console.log('  download <id>    - Download recording by ID');
    console.log('  exit             - Exit\n');

    const readline = require('readline');
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    rl.on('line', async (input: string) => {
      const [command, ...args] = input.trim().split(' ');

      try {
        switch (command) {
          case 'start':
            if (args[0]) {
              await test.startRecordingForWindow(args[0]);
            } else {
              console.log('❌ Please provide a window ID');
            }
            break;

          case 'stop':
            await test.stopRecording();
            break;

          case 'status':
            const status = test.getStatus();
            console.log('📊 Status:', status);
            break;

          case 'simulate':
            const title = args.join(' ') || 'Simulated Meeting';
            test.simulateMeetingDetection(`sim-${Date.now()}`, title);
            break;

          case 'list':
            try {
              const recordings = await test.getRecordings();
              console.log('\n📋 Recent Recordings:');
              recordings.forEach((rec: any, i: number) => {
                const date = new Date(rec.created_at).toLocaleString();
                console.log(`${i+1}. ${rec.id} - ${date} - Status: ${rec.status?.code || 'unknown'}`);
              });
              if (recordings.length === 0) {
                console.log('No recordings found');
              }
            } catch (error: any) {
              console.log('❌ Failed to list recordings:', error.message);
            }
            break;

          case 'download':
            if (args[0]) {
              try {
                console.log(`Downloading recording: ${args[0]}`);
                const outputPath = await test.downloadRecording(args[0]);
                console.log(`✅ Downloaded to: ${outputPath}`);
              } catch (error: any) {
                console.log('❌ Failed to download recording:', error.message);
              }
            } else {
              console.log('❌ Please provide a recording ID');
            }
            break;

          case 'exit':
            console.log('👋 Exiting...');
            await test.cleanup();
            rl.close();
            process.exit(0);
            break;

          default:
            console.log('❓ Unknown command. Type "help" for available commands.');
        }
      } catch (error: any) {
        console.error('❌ Command failed:', error.message);
      }

      rl.prompt();
    });

    rl.on('close', async () => {
      await test.cleanup();
      process.exit(0);
    });

    rl.prompt();

  } catch (error: any) {
    console.error('❌ Interactive test failed:', error.message);
    await test.cleanup();
    process.exit(1);
  }
}

// Test function to download a specific recording
async function testDownloadRecording() {
  console.log('🎯 Testing Recording Download');

  const test = new DesktopRecordingTest('./out');

  try {
    await test.initialize();
    await test.requestPermissions();

    // First, list all recordings to see what's available
    console.log('📋 Listing all recordings first...');
    const recordings = await test.getRecordings(20);

    if (recordings.length === 0) {
      console.log('❌ No recordings found. You may need to create a recording first.');
      console.log('💡 Run the basic test to create a recording: npx tsx recall-ai-test.ts');
      return;
    }

    console.log('\n📋 Available recordings:');
    recordings.forEach((rec: any, i: number) => {
      const date = new Date(rec.created_at).toLocaleString();
      const status = rec.status?.code || 'unknown';
      console.log(`${i+1}. ${rec.id} - ${date} - Status: ${status}`);
      if (rec.media_shortcuts?.video_mixed) {
        console.log(`   📹 Video available: ${rec.media_shortcuts.video_mixed.data?.download_url ? 'YES' : 'NO'}`);
      }
    });

    // Use the recording ID from command line or the most recent one
    let recordingId = process.argv[3];

    if (!recordingId && recordings.length > 0) {
      // Use the most recent recording
      recordingId = recordings[0].id;
      console.log(`\n🎯 Using most recent recording: ${recordingId}`);
    }

    if (!recordingId) {
      console.log('❌ No recording ID specified and no recordings found');
      return;
    }

    console.log(`\n📥 Downloading recording: ${recordingId}`);
    const outputPath = await test.downloadRecording(recordingId);
    console.log(`✅ Successfully downloaded to: ${outputPath}`);

  } catch (error: any) {
    console.error('❌ Download test failed:', error.message);
  } finally {
    await test.cleanup();
  }
}

// Main execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const mode = process.argv[2];

  if (mode === '--interactive' || mode === '-i') {
    interactiveTest();
  } else if (mode === '--download' || mode === '-d') {
    testDownloadRecording();
  } else {
    testDesktopRecording();
  }
}

export { DesktopRecordingTest };
