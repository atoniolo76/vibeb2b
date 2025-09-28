import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

// Import the desktop SDK - this needs to be imported dynamically in Electron renderer
let RecallAiSdk: any = null;

async function loadDesktopSdk() {
  if (!RecallAiSdk) {
    try {
      RecallAiSdk = await import('@recallai/desktop-sdk');
    } catch (error) {
      console.error('Failed to load Recall.ai desktop SDK:', error);
      throw error;
    }
  }
  return RecallAiSdk;
}

// Recall AI API configuration - try different regions
const RECALL_REGIONS = [
  'https://us-east-1.recall.ai',
  'https://api.recall.ai',
  'https://us-west-2.recall.ai',
  'https://eu-central-1.recall.ai',
  'https://ap-northeast-1.recall.ai'
];

let RECALL_API_BASE = 'https://us-east-1.recall.ai';
let detectedRegion: string | null = null;

// Core interfaces
export interface MeetingInfo {
  url?: string;
  platform?: 'zoom' | 'google_meet' | 'teams' | 'slack';
  title?: string;
  windowId?: string;
}

export interface SDKUpload {
  id: string;
  upload_token: string;
}

// Simplified synchronous recording class
export class SynchronousRecording {
  private sdk: any = null;
  private currentUploadToken: string | null = null;
  private currentRecordingId: string | null = null;
  private isInitialized = false;

  // Detect and cache the correct API region
  private async detectRegion(): Promise<string> {
    if (detectedRegion) return detectedRegion;

    const apiKey = process.env.RECALL_AI_API_KEY;
    if (!apiKey) throw new Error('RECALL_AI_API_KEY not found');

    for (const regionUrl of RECALL_REGIONS) {
      try {
        await axios.get(`${regionUrl}/api/v1/bot/`, {
          headers: { 'Authorization': `Token ${apiKey}` },
          timeout: 5000
        });
        detectedRegion = regionUrl;
        RECALL_API_BASE = regionUrl;
        return regionUrl;
      } catch (error) {
        continue;
      }
    }
    throw new Error('Could not determine API region');
  }

  // Initialize SDK
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    const sdk = await loadDesktopSdk();
    await sdk.default.init({
      apiUrl: RECALL_API_BASE,
      acquirePermissionsOnStartup: ["accessibility", "screen-capture", "microphone"],
      restartOnError: true
    });

    this.sdk = sdk.default;
    this.isInitialized = true;
    console.log('✅ SDK initialized');
  }

  // Request permissions
  async requestPermissions(): Promise<void> {
    if (!this.sdk) throw new Error('SDK not initialized');

    for (const permission of ["accessibility", "screen-capture", "microphone"]) {
      await this.sdk.requestPermission(permission);
    }
    console.log('✅ Permissions granted');
  }

  // Create upload token
  async createUpload(): Promise<SDKUpload> {
    await this.detectRegion();
    const apiKey = process.env.RECALL_AI_API_KEY;

    const response = await axios.post(`${RECALL_API_BASE}/api/v1/sdk-upload/`, {}, {
      headers: {
        'Authorization': `Token ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    const upload = response.data;
    this.currentUploadToken = upload.upload_token;
    console.log(`✅ Upload created: ${upload.id}`);
    return upload;
  }

  // Start recording synchronously
  async startRecording(windowId: string): Promise<void> {
    if (!this.sdk) throw new Error('SDK not initialized');

    const upload = await this.createUpload();

    await this.sdk.startRecording({
      windowId: windowId,
      uploadToken: upload.upload_token
    });

    console.log('🎬 Recording started');
  }

  // Stop recording and trigger upload
  async stopRecording(windowId: string): Promise<void> {
    if (!this.sdk) throw new Error('SDK not initialized');

    await this.sdk.stopRecording({ windowId });
    console.log('🛑 Recording stopped, uploading...');

    // Trigger upload
    try {
      await this.sdk.uploadRecording({ windowId });
      console.log('✅ Upload triggered successfully');
    } catch (uploadError: any) {
      console.error('❌ Upload failed:', uploadError.message);
      throw uploadError;
    }
  }

  // Wait for recording completion
  async waitForCompletion(windowId: string, pollInterval = 10000, maxAttempts = 180): Promise<string> {
    console.log('⏳ Waiting for recording completion...');

    // Wait a bit for the upload to be processed before starting to poll
    console.log('⏳ Waiting 30 seconds for initial processing...');
    await new Promise(resolve => setTimeout(resolve, 30000));

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        // Get the most recent recording - it should be ours
        const recordings = await this.getRecordings(5);

        if (recordings.length > 0) {
          const latestRecording = recordings[0];
          const status = latestRecording.status?.code;

          // Log full status on first attempt to debug
          if (attempt === 0) {
            console.log('🔍 Recording details:', {
              id: latestRecording.id,
              status: latestRecording.status,
              created_at: latestRecording.created_at,
              duration: latestRecording.duration
            });
          }

          console.log(`📊 Status: ${status} (attempt ${attempt + 1}/${maxAttempts})`);

          // Check for various completion states
          if (status === 'done' || status === 'completed' || status === 'ready') {
            this.currentRecordingId = latestRecording.id;
            console.log(`✅ Recording completed: ${this.currentRecordingId}`);
            return this.currentRecordingId;
          } else if (status === 'failed' || status === 'error') {
            throw new Error('Recording failed to process');
          } else if (status === 'processing' && attempt > 10) {
            // If it's been processing for more than 10 attempts (2+ minutes), try to download anyway
            console.log('⚠️ Recording still processing, but attempting download...');
            this.currentRecordingId = latestRecording.id;
            return this.currentRecordingId;
          }
        } else {
          console.log(`⏳ No recordings found yet (attempt ${attempt + 1}/${maxAttempts})`);
        }

        await new Promise(resolve => setTimeout(resolve, pollInterval));
      } catch (error) {
        console.error('Error checking status:', error);
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }
    }

    throw new Error('Recording timed out');
  }

  // Download recording to footage folder
  async downloadRecording(recordingId: string): Promise<string> {
    const footageDir = path.join(process.cwd(), 'footage');
    if (!fs.existsSync(footageDir)) {
      fs.mkdirSync(footageDir, { recursive: true });
    }

    const recording = await this.getRecording(recordingId);
    const mediaShortcuts = recording.media_shortcuts || {};
    const videoUrl = mediaShortcuts.video_mixed?.data?.download_url;

    if (!videoUrl) throw new Error('No video download URL found');

    const outputPath = path.join(footageDir, `${recordingId}.mp4`);

    const response = await axios.get(videoUrl, { responseType: 'stream' });
    const writer = fs.createWriteStream(outputPath);
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', () => {
        console.log(`✅ Downloaded to: ${outputPath}`);
        resolve(outputPath);
      });
      writer.on('error', reject);
    });
  }

  // Get recording details
  private async getRecording(recordingId: string): Promise<any> {
    await this.detectRegion();
    const apiKey = process.env.RECALL_AI_API_KEY;

    const response = await axios.get(`${RECALL_API_BASE}/api/v1/recording/${recordingId}/`, {
      headers: { 'Authorization': `Token ${apiKey}` }
    });

    return response.data;
  }

  // Get recent recordings
  private async getRecordings(limit = 1): Promise<any[]> {
    await this.detectRegion();
    const apiKey = process.env.RECALL_AI_API_KEY;

    const response = await axios.get(`${RECALL_API_BASE}/api/v1/recording/`, {
      headers: { 'Authorization': `Token ${apiKey}` },
      params: { limit }
    });

    return response.data.results || response.data;
  }

  // Clean up
  async cleanup(): Promise<void> {
    if (this.sdk) {
      await this.sdk.shutdown();
    }
  }
}
