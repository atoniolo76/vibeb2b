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

  // Wait for Desktop SDK recording completion (different from bot recordings)
  async waitForCompletion(windowId: string, pollInterval = 15000, maxAttempts = 120): Promise<string> {
    console.log('⏳ Waiting for Desktop SDK recording completion...');
    console.log('📋 Note: Desktop SDK recordings require webhook confirmation, but we\'ll poll as fallback');

    // For Desktop SDK recordings, we should ideally wait for sdk_upload.complete webhook
    // But since this is a standalone script, we'll poll for the recording to appear
    console.log('⏳ Waiting 45 seconds for upload to complete...');
    await new Promise(resolve => setTimeout(resolve, 45000));

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        // Get recent recordings
        const recordings = await this.getRecordings(3);

        if (recordings.length > 0) {
          const latestRecording = recordings[0];

          // Log details on first few attempts for debugging
          if (attempt < 3) {
            console.log('🔍 Latest recording details:', {
              id: latestRecording.id,
              status: latestRecording.status?.code || 'unknown',
              created_at: latestRecording.created_at,
              duration: latestRecording.duration,
              media_shortcuts: {
                video_mixed: latestRecording.media_shortcuts?.video_mixed?.status?.code || 'not_available',
                transcript: latestRecording.media_shortcuts?.transcript?.status?.code || 'not_available',
                participant_events: latestRecording.media_shortcuts?.participant_events?.status?.code || 'not_available'
              }
            });
          }

          console.log(`📊 Status: ${latestRecording.status?.code || 'unknown'} (attempt ${attempt + 1}/${maxAttempts})`);

          // For Desktop SDK recordings, check if media shortcuts are available
          const mediaShortcuts = latestRecording.media_shortcuts || {};
          const videoShortcut = mediaShortcuts.video_mixed;

          if (videoShortcut) {
            const videoStatusCode = videoShortcut.status?.code || videoShortcut.status;
            console.log(`🎬 Video shortcut status check: ${videoStatusCode}`);

            // Check for completion status
            if (videoStatusCode === 'done' || videoStatusCode === 'completed' || videoStatusCode === 'ready') {
              this.currentRecordingId = latestRecording.id;
              console.log(`✅ Recording ready with video: ${this.currentRecordingId}`);
              return this.currentRecordingId;
            }
          }

          if (latestRecording.status?.code === 'failed') {
            throw new Error('Recording failed to process');
          } else if (attempt > 20) {
            // After 5+ minutes (20 attempts * 15s), try anyway if we have a recording
            console.log('⚠️ Taking too long, attempting download with current recording...');
            this.currentRecordingId = latestRecording.id;
            return this.currentRecordingId;
          }
        } else {
          console.log(`⏳ No recordings found yet (attempt ${attempt + 1}/${maxAttempts})`);
        }

        await new Promise(resolve => setTimeout(resolve, pollInterval));
      } catch (error) {
        console.error('Error checking recording status:', error);
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }
    }

    throw new Error('Desktop SDK recording timed out - webhook may be required for proper completion');
  }

  // Download recording to footage folder
  async downloadRecording(recordingId: string): Promise<string> {
    const footageDir = path.join(process.cwd(), 'footage');
    if (!fs.existsSync(footageDir)) {
      fs.mkdirSync(footageDir, { recursive: true });
    }

    console.log(`📥 Attempting to download recording: ${recordingId}`);

    const recording = await this.getRecording(recordingId);
    console.log('📊 Recording details:', {
      id: recording.id,
      status: recording.status?.code || 'unknown',
      media_shortcuts: {
        video_mixed: recording.media_shortcuts?.video_mixed?.status?.code || 'not_available',
        transcript: recording.media_shortcuts?.transcript?.status?.code || 'not_available',
        participant_events: recording.media_shortcuts?.participant_events?.status?.code || 'not_available'
      },
      created_at: recording.created_at
    });

    const mediaShortcuts = recording.media_shortcuts || {};
    const videoShortcut = mediaShortcuts.video_mixed;

    if (!videoShortcut) {
      throw new Error(`No video_mixed media shortcut found. Available shortcuts: ${Object.keys(mediaShortcuts).join(', ')}`);
    }

    console.log('🎬 Video shortcut full details:', JSON.stringify(videoShortcut, null, 2));

    // Check different possible status formats
    const statusCode = videoShortcut.status?.code || videoShortcut.status;
    console.log('🎬 Video shortcut status code:', statusCode);

    if (statusCode !== 'done' && statusCode !== 'completed' && statusCode !== 'ready') {
      throw new Error(`Video not ready for download. Status: ${statusCode || 'unknown'}`);
    }

    const videoUrl = videoShortcut.data?.download_url;
    if (!videoUrl) {
      throw new Error('Video shortcut marked as done but no download URL found');
    }

    console.log(`🔗 Download URL found: ${videoUrl}`);
    const outputPath = path.join(footageDir, `${recordingId}.mp4`);

    console.log(`💾 Downloading to: ${outputPath}`);
    const response = await axios.get(videoUrl, {
      responseType: 'stream',
      timeout: 300000 // 5 minutes timeout for large files
    });

    const writer = fs.createWriteStream(outputPath);
    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
      writer.on('finish', () => {
        console.log(`✅ Download completed: ${outputPath}`);
        resolve(outputPath);
      });
      writer.on('error', (error) => {
        console.error('❌ Download failed:', error);
        reject(error);
      });
    });
  }

  // Get recording details
  private async getRecording(recordingId: string): Promise<any> {
    await this.detectRegion();
    const apiKey = process.env.RECALL_AI_API_KEY;

    console.log(`🔍 Fetching recording details for ${recordingId} from ${RECALL_API_BASE}`);
    try {
      const response = await axios.get(`${RECALL_API_BASE}/api/v1/recording/${recordingId}/`, {
        headers: { 'Authorization': `Token ${apiKey}` }
      });

      console.log('📊 Raw recording response:', JSON.stringify(response.data, null, 2));
      return response.data;
    } catch (error: any) {
      console.error(`❌ Failed to get recording ${recordingId}:`, error.response?.data || error.message);
      throw error;
    }
  }

  // Get recent recordings
  async getRecordings(limit = 1): Promise<any[]> {
    await this.detectRegion();
    const apiKey = process.env.RECALL_AI_API_KEY;

    console.log(`🔍 Fetching ${limit} recent recordings from ${RECALL_API_BASE}`);
    const response = await axios.get(`${RECALL_API_BASE}/api/v1/recording/`, {
      headers: { 'Authorization': `Token ${apiKey}` },
      params: { limit }
    });

    const recordings = response.data.results || response.data;
    console.log(`📊 Found ${recordings.length} recordings`);
    return recordings;
  }

  // Clean up
  async cleanup(): Promise<void> {
    if (this.sdk) {
      await this.sdk.shutdown();
    }
  }
}
