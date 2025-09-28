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
  'https://api.recall.ai',  // Default/global
  'https://us-west-2.recall.ai',
  'https://eu-central-1.recall.ai',
  'https://ap-northeast-1.recall.ai'
];

let RECALL_API_BASE = 'https://us-east-1.recall.ai'; // Default starting region
let detectedRegion: string | null = null; // Cache detected region

// Types for Desktop SDK
export interface SDKUpload {
  id: string;
  upload_token: string;
}

export interface MeetingInfo {
  url?: string;
  platform?: 'zoom' | 'google_meet' | 'teams' | 'slack';
  title?: string;
  windowId?: string;
}

export interface DesktopRecordingOptions {
  outputDir?: string;
  transcriptProvider?: 'assembly_ai_streaming' | 'deepgram_streaming';
  videoLayout?: 'gallery_view_v2' | 'speaker_view';
}

export interface RecordingStatus {
  state: 'idle' | 'recording' | 'uploading' | 'completed' | 'failed';
  message?: string;
  uploadProgress?: number;
  recordingId?: string;
}

// Events emitted by the desktop recording system
export interface DesktopRecordingEvents {
  onMeetingDetected?: (meeting: MeetingInfo) => void;
  onRecordingStarted?: (uploadToken: string) => void;
  onRecordingEnded?: () => void;
  onUploadProgress?: (progress: number) => void;
  onRecordingCompleted?: (recordingId: string) => void;
  onRecordingFailed?: (error: Error) => void;
  onPermissionsGranted?: () => void;
  onPermissionsDenied?: (deniedPermissions: string[]) => void;
  onError?: (error: Error) => void;
}

// Main class for Recall AI desktop recording
export class RecallAIDesktopRecording {
  private outputDir: string;
  private isRecording = false;
  private currentUploadToken: string | null = null;
  private currentRecordingId: string | null = null;
  private options: Required<DesktopRecordingOptions>;
  private events: DesktopRecordingEvents;
  private sdkInitialized = false;

  constructor(options: DesktopRecordingOptions = {}, events: DesktopRecordingEvents = {}) {
    this.options = {
      outputDir: options.outputDir || path.join(process.cwd(), 'out'),
      transcriptProvider: options.transcriptProvider || 'assembly_ai_streaming',
      videoLayout: options.videoLayout || 'gallery_view_v2',
      ...options
    };

    this.outputDir = this.options.outputDir;
    this.events = events;
    this.ensureOutputDir();
  }

  private ensureOutputDir() {
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
      console.log(`Created output directory: ${this.outputDir}`);
    }
  }

  // Detect and cache the correct API region for the user's API key
  private async detectRegion(): Promise<string> {
    if (detectedRegion) {
      return detectedRegion;
    }

    const apiKey = process.env.RECALL_AI_API_KEY;
    if (!apiKey) {
      throw new Error('RECALL_AI_API_KEY not found in environment variables');
    }

    console.log('🔍 Detecting API region for your API key...');

    // Try each region to find the correct one
    for (const regionUrl of RECALL_REGIONS) {
      try {
        console.log(`Testing region: ${regionUrl}`);
        const response = await axios.get(`${regionUrl}/api/v1/bot/`, {
          headers: {
            'Authorization': `Token ${apiKey}`,
          },
          timeout: 5000 // Short timeout for testing
        });

        // If we get a successful response (even empty), this is the right region
        console.log(`✅ Found correct region: ${regionUrl}`);
        detectedRegion = regionUrl;
        RECALL_API_BASE = regionUrl;
        return regionUrl;
      } catch (error: any) {
        // Continue to next region if this one fails
        continue;
      }
    }

    throw new Error('Could not determine the correct API region for your API key. Please check your API key and try again.');
  }

  // Validate API key and ensure correct region is set
  private async validateApiKey(): Promise<void> {
    console.log('🔍 Checking API key and region...');
    const apiKey = process.env.RECALL_AI_API_KEY;
    console.log('process.env.RECALL_AI_API_KEY:', apiKey ? 'PRESENT' : 'MISSING');

    if (!apiKey) {
      const error = new Error('RECALL_AI_API_KEY not found in environment variables');
      this.events.onError?.(error);
      throw error;
    }

    // Ensure we have the correct region
    if (!detectedRegion) {
      await this.detectRegion();
    }
  }

  // Initialize the desktop SDK
  async initializeSDK(): Promise<void> {
    try {
      const sdk = await loadDesktopSdk();

      if (this.sdkInitialized) {
        console.log('Desktop SDK already initialized');
        return;
      }

      console.log('Initializing Recall.ai Desktop SDK...');

      // Initialize the SDK with configuration
      await sdk.default.init({
        apiUrl: RECALL_API_BASE,
        acquirePermissionsOnStartup: ["accessibility", "screen-capture", "microphone"],
        restartOnError: true
      });

      this.sdkInitialized = true;
      console.log('✅ Desktop SDK initialized successfully');

      // Set up event listeners
      this.setupEventListeners();

    } catch (error: any) {
      console.error('Failed to initialize desktop SDK:', error);
      const err = new Error(`Failed to initialize desktop SDK: ${error.message}`);
      this.events.onError?.(err);
      throw err;
    }
  }

  // Set up event listeners for the desktop SDK
  private setupEventListeners(): void {
    const sdk = RecallAiSdk?.default;

    if (!sdk) {
      console.error('SDK not loaded, cannot setup event listeners');
      return;
    }

    // Meeting detected event
    sdk.addEventListener('meeting-detected', async (evt: any) => {
      console.log('🎯 Meeting detected:', evt);
      const meeting: MeetingInfo = {
        url: evt.meeting?.url,
        platform: this.detectPlatform(evt.window?.title || ''),
        title: evt.window?.title,
        windowId: evt.window?.id
      };
      this.events.onMeetingDetected?.(meeting);
    });

    // Permissions granted event
    sdk.addEventListener('permissions-granted', async (evt: any) => {
      console.log('✅ Permissions granted');
      this.events.onPermissionsGranted?.();
    });

    // Permissions denied event
    sdk.addEventListener('permissions-denied', async (evt: any) => {
      console.log('❌ Permissions denied:', evt.deniedPermissions);
      this.events.onPermissionsDenied?.(evt.deniedPermissions);
    });

    // SDK state change event
    sdk.addEventListener('sdk-state-change', async (evt: any) => {
      const state = evt.sdk?.state?.code;
      console.log('🔄 SDK state changed:', state);

      switch (state) {
        case 'recording':
          console.log('🎬 Recording started');
          this.isRecording = true;
          break;
        case 'idle':
          console.log('⏸️ Recording stopped/idle');
          this.isRecording = false;
          this.events.onRecordingEnded?.();
          break;
        case 'uploading':
          console.log('📤 Upload started');
          break;
      }
    });

    // Recording ended event
    sdk.addEventListener('recording-ended', async (evt: any) => {
      console.log('🏁 Recording ended, starting upload...');
      try {
        await sdk.uploadRecording({ windowId: evt.window.id });
      } catch (error: any) {
        console.error('Failed to upload recording:', error);
        this.events.onError?.(error);
      }
    });

    // Upload progress event
    sdk.addEventListener('upload-progress', async (evt: any) => {
      const progress = evt.progress;
      console.log(`📊 Upload progress: ${progress}%`);
      this.events.onUploadProgress?.(progress);
    });

    // Recording uploaded event (this might be a custom event we need to handle)
    sdk.addEventListener('recording-uploaded', async (evt: any) => {
      console.log('✅ Recording uploaded successfully');
      // We would need to handle the webhook or poll for completion
    });

    console.log('🎧 Desktop SDK event listeners set up');
  }

  // Detect meeting platform from window title
  private detectPlatform(title: string): 'zoom' | 'google_meet' | 'teams' | 'slack' {
    const lowerTitle = title.toLowerCase();
    if (lowerTitle.includes('zoom')) return 'zoom';
    if (lowerTitle.includes('meet') || lowerTitle.includes('google')) return 'google_meet';
    if (lowerTitle.includes('teams') || lowerTitle.includes('microsoft')) return 'teams';
    if (lowerTitle.includes('slack') || lowerTitle.includes('huddle')) return 'slack';
    return 'zoom'; // default
  }

  // Create SDK upload for recording
  async createSDKUpload(): Promise<SDKUpload> {
    await this.validateApiKey();

    let lastError: any = null;

    // Try each region until one works
    for (const regionUrl of RECALL_REGIONS) {
      RECALL_API_BASE = regionUrl;

      try {
        console.log(`Trying to create SDK upload with region: ${regionUrl}`);

        const requestData = {
          // Start with minimal configuration - can add transcript later
        };

        const response = await axios.post(`${RECALL_API_BASE}/api/v1/sdk-upload/`, requestData, {
          headers: {
            'Authorization': `Token ${process.env.RECALL_AI_API_KEY}`,
            'Content-Type': 'application/json'
          }
        });

        const upload: SDKUpload = response.data;
        console.log(`✅ SDK upload created with ID: ${upload.id} using region: ${regionUrl}`);
        return upload;
      } catch (error: any) {
        lastError = error;
        const errorData = error.response?.data;
        console.log(`Failed with region ${regionUrl}:`, errorData?.detail || error.message);

        // If it's not an authentication error for this region, try the next one
        if (errorData?.detail?.includes('region') || errorData?.code === 'authentication_failed') {
          continue;
        }

        // If it's a different type of error, throw immediately
        throw error;
      }
    }

    // If we tried all regions and still failed
    console.error('Failed to create SDK upload with all regions:', lastError.response?.data || lastError.message);
    const err = new Error(`Failed to create SDK upload: ${lastError.message}`);
    this.events.onError?.(err);
    throw err;
  }

  // Start recording a detected meeting
  async startRecording(windowId: string): Promise<void> {
    try {
      if (!this.sdkInitialized) {
        await this.initializeSDK();
      }

      console.log('🎬 Starting desktop recording...');

      // Create upload token
      const upload = await this.createSDKUpload();
      this.currentUploadToken = upload.upload_token;

      const sdk = RecallAiSdk?.default;
      if (!sdk) {
        throw new Error('Desktop SDK not loaded');
      }

      // Start recording
      await sdk.startRecording({
        windowId: windowId,
        uploadToken: upload.upload_token
      });

      console.log('✅ Recording started successfully');
      this.events.onRecordingStarted?.(upload.upload_token);

    } catch (error: any) {
      console.error('Failed to start recording:', error);
      const err = new Error(`Failed to start recording: ${error.message}`);
      this.events.onRecordingFailed?.(err);
      throw err;
    }
  }

  // Stop current recording
  async stopRecording(windowId: string): Promise<void> {
    try {
      const sdk = RecallAiSdk?.default;
      if (!sdk) {
        throw new Error('Desktop SDK not loaded');
      }

      console.log('🛑 Stopping recording...');
      await sdk.stopRecording({ windowId });

    } catch (error: any) {
      console.error('Failed to stop recording:', error);
      this.events.onError?.(error);
    }
  }

  // Request permissions
  async requestPermissions(permissions: string[] = ["accessibility", "screen-capture", "microphone"]): Promise<void> {
    try {
      const sdk = RecallAiSdk?.default;
      if (!sdk) {
        throw new Error('Desktop SDK not loaded');
      }

      console.log('🔐 Requesting permissions:', permissions);

      for (const permission of permissions) {
        await sdk.requestPermission(permission);
      }

    } catch (error: any) {
      console.error('Failed to request permissions:', error);
      this.events.onError?.(error);
    }
  }

  // Get current recording status
  getRecordingStatus(): RecordingStatus {
    return {
      state: this.isRecording ? 'recording' : 'idle',
      recordingId: this.currentRecordingId || undefined,
      uploadProgress: undefined // Would need to track this separately
    };
  }

  // Shutdown the SDK
  async shutdown(): Promise<void> {
    try {
      const sdk = RecallAiSdk?.default;
      if (sdk) {
        console.log('Shutting down desktop SDK...');
        await sdk.shutdown();
      }
      this.sdkInitialized = false;
      this.isRecording = false;
    } catch (error: any) {
      console.error('Error shutting down SDK:', error);
      this.events.onError?.(error);
    }
  }

  // Clean up resources
  async cleanup(): Promise<void> {
    console.log('Cleaning up desktop recording...');
    await this.shutdown();
  }

  // Get recording details by ID
  async getRecording(recordingId: string): Promise<any> {
    await this.validateApiKey();

    try {
      console.log(`Getting recording details for ID: ${recordingId}`);

      const response = await axios.get(`${RECALL_API_BASE}/api/v1/recording/${recordingId}/`, {
        headers: {
          'Authorization': `Token ${process.env.RECALL_AI_API_KEY}`,
        }
      });

      const recording = response.data;
      console.log(`✅ Retrieved recording: ${recording.id}`);
      return recording;
    } catch (error: any) {
      console.error('Failed to get recording:', error.response?.data || error.message);
      const err = new Error(`Failed to get recording: ${error.message}`);
      this.events.onError?.(err);
      throw err;
    }
  }

  // Download recording file to local path
  async downloadRecording(recordingId: string, outputPath?: string): Promise<string> {
    try {
      // First get recording details
      const recording = await this.getRecording(recordingId);

      // Find the video download URL
      const mediaShortcuts = recording.media_shortcuts || {};
      const videoUrl = mediaShortcuts.video_mixed?.data?.download_url;

      if (!videoUrl) {
        throw new Error('No video download URL found for this recording');
      }

      console.log(`Downloading recording from: ${videoUrl}`);

      // Generate output path if not provided
      const actualOutputPath = outputPath || path.join(this.outputDir, `${recordingId}.mp4`);

      // Ensure output directory exists
      const outputDir = path.dirname(actualOutputPath);
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      // Download the file (S3 pre-signed URLs don't need auth headers)
      const response = await axios.get(videoUrl, {
        responseType: 'stream'
        // Note: No Authorization header needed for pre-signed S3 URLs
      });

      const writer = fs.createWriteStream(actualOutputPath);
      response.data.pipe(writer);

      return new Promise((resolve, reject) => {
        writer.on('finish', () => {
          console.log(`✅ Downloaded recording to: ${actualOutputPath}`);
          resolve(actualOutputPath);
        });
        writer.on('error', (error) => {
          console.error('Failed to write file:', error);
          reject(error);
        });
      });

    } catch (error: any) {
      console.error('Failed to download recording:', error.message);
      const err = new Error(`Failed to download recording: ${error.message}`);
      this.events.onError?.(err);
      throw err;
    }
  }

  // Get available recordings (list user's recordings)
  async getRecordings(limit = 10): Promise<any[]> {
    await this.validateApiKey();

    try {
      console.log(`Getting recent recordings (limit: ${limit})`);

      const response = await axios.get(`${RECALL_API_BASE}/api/v1/recording/`, {
        headers: {
          'Authorization': `Token ${process.env.RECALL_AI_API_KEY}`,
        },
        params: {
          limit: limit
        }
      });

      const recordings = response.data.results || response.data;
      console.log(`✅ Retrieved ${recordings.length} recordings`);
      return Array.isArray(recordings) ? recordings : [];
    } catch (error: any) {
      console.error('Failed to get recordings:', error.response?.data || error.message);
      const err = new Error(`Failed to get recordings: ${error.message}`);
      this.events.onError?.(err);
      throw err;
    }
  }
}

// Utility function to create a desktop recording instance with default settings
export function createDesktopRecording(
  options: DesktopRecordingOptions = {},
  events: DesktopRecordingEvents = {}
): RecallAIDesktopRecording {
  return new RecallAIDesktopRecording(options, events);
}

// Auto-initialize function for Electron apps
export async function initializeDesktopRecording(): Promise<RecallAIDesktopRecording> {
  const recording = createDesktopRecording();
  await recording.initializeSDK();
  return recording;
}
