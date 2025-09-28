#!/usr/bin/env python3
"""
INTEGRATED EMOTION SERVER - Everything in One File
Receives PNG frames from Google Meet bot, processes emotions, serves API results

IMPORTANT: This version stores ALL frames in MEMORY only.
No files are saved to disk to avoid computer clogging.

Debug output shows participant names and processing status.
Uses PNG for maximum reliability and no corruption.

CURRENT STATE SUMMARY:
=====================

WHAT IT DOES:
- Receives PNG frames from Google Meet bot via WebSocket (port 5003)
- Processes frames in real-time using PyTorch LSTM + ResNet50 models
- Analyzes facial emotions and calculates presentation metrics
- Serves emotion data via REST API (port 5000) in specified format
- Displays live preview window with emotion overlays and statistics
- Handles frame deduplication to prevent processing identical frames
- Stores all data in memory (no disk I/O) for maximum performance

CAPABILITIES:
- Real-time emotion detection from video streams
- Presentation metrics: Frustration (confusion) and Engagement
- Spike detection for sustained emotional changes
- MediaPipe face mesh analysis for brow furrow detection
- Thread-safe frame processing and preview display
- REST API endpoints: /get_metrics, /health, /websocket_status
- Live preview window with emotion bars and spike indicators
- Performance statistics and FPS monitoring
- Automatic model loading and error handling

LIMITATIONS:
- Requires PyTorch models: FER_dinamic_LSTM_Aff-Wild2.pt and FER_static_ResNet50_AffectNet.pt
- Only processes PNG format (no H264 support in current version)
- Single face detection per frame
- Emotion analysis limited to 7 basic emotions
- Spike detection requires 10+ frames of history
- Preview window must be closed manually (no auto-close)
- No authentication or rate limiting on API endpoints
- WebSocket connection drops require manual reconnection
- No persistent storage of emotion history
- Limited to local network without ngrok tunneling

TECHNICAL DETAILS:
- WebSocket server: ws://localhost:5003
- HTTP API server: http://localhost:5000
- Frame processing: ~5 FPS (limited by model inference time)
- Memory usage: ~100-200MB (depends on frame size and history)
- Dependencies: PyTorch, OpenCV, MediaPipe, Flask, websocket-server
- Supported platforms: Windows, Linux, macOS
- Python version: 3.8+

USAGE:
1. Start server: python integrated_emotion_server.py
2. Configure Google Meet bot to send PNG frames to WebSocket
3. Access emotion data via: curl http://localhost:5000/get_metrics
4. View live preview window for real-time emotion analysis
5. Use ngrok to expose WebSocket for remote bot connections

API RESPONSE FORMAT:
"TIMESTAMP, FRUS: 010, FRUS_HAS_SPIKE: TRUE, ENG: 005, ENG_HAS_SPIKE: FALSE"
"""


from flask import Flask, jsonify
from websocket_server import WebsocketServer
import json
import base64
import threading
import time
import cv2
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import mediapipe as mp
import math
import warnings
from PIL import Image

warnings.simplefilter("ignore", UserWarning)

# Flask app
app = Flask(__name__)

# Global state
pth_LSTM_model = None
pth_backbone_model = None
mp_face_mesh = None
face_mesh = None
lstm_features = []
metrics_history = []
latest_result = {
    "timestamp": "",
    "frustration": 0,
    "frustration_spike": False,
    "engagement": 0,
    "engagement_spike": False
}
latest_preview_frame = None
preview_lock = threading.Lock()

class LatestFrameManager:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Statistics
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        self.last_receive_time = 0
        self.receive_fps = 0

    def update_frame_png(self, png_data):
        """Decode PNG frame data"""
        with self.frame_lock:
            # Count statistics
            self.frames_received += 1
            if self.latest_frame is not None:
                self.frames_skipped += 1  # We're replacing an unprocessed frame

            try:
                # Decode PNG directly from base64
                png_bytes = base64.b64decode(png_data)
                nparr = np.frombuffer(png_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Always replace with latest
                    self.latest_frame = frame.copy()

                    # Calculate receive FPS
                    current_time = time.time()
                    if self.last_receive_time > 0:
                        time_diff = current_time - self.last_receive_time
                        if time_diff > 0:
                            self.receive_fps = 0.9 * self.receive_fps + 0.1 * (1.0 / time_diff)
                    self.last_receive_time = current_time

                    return True
                else:
                    print("❌ Failed to decode PNG frame")
                    return False

            except Exception as e:
                print(f"❌ PNG decode error: {e}")
                return False

    def get_latest_frame(self):
        """Get the most recent frame for processing"""
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                self.frames_processed += 1
                return frame
            return None

    def get_stats(self):
        """Get performance statistics"""
        with self.frame_lock:
            return {
                'frames_received': self.frames_received,
                'frames_processed': self.frames_processed,
                'frames_skipped': self.frames_skipped,
                'receive_fps': round(self.receive_fps, 2),
                'skip_ratio': round(self.frames_skipped / max(1, self.frames_received) * 100, 1),
                'has_frame': self.latest_frame is not None
            }

# Global frame manager
frame_manager = LatestFrameManager()

# PyTorch Model Classes
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion, eps=0.001, momentum=0.99)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv_layer_s2_same = Conv2dSame(num_channels, 64, 7, stride=2, groups=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*ResBlock.expansion, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def extract_features(self, x):
        x = self.relu(self.batch_norm1(self.conv_layer_s2_same(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride, bias=False, padding=0),
                nn.BatchNorm2d(planes*ResBlock.expansion, eps=0.001, momentum=0.99)
            )
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

class LSTMPyTorch(nn.Module):
    def __init__(self):
        super(LSTMPyTorch, self).__init__()
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x

def pth_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def __init__(self):
            super(PreprocessInput, self).__init__()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img):
        ttransform = transforms.Compose([
            transforms.PILToTensor(),
            PreprocessInput()
        ])
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        img = ttransform(img)
        img = torch.unsqueeze(img, 0)
        return img
    return get_img_torch(fp)

def norm_coordinates(normalized_x, normalized_y, image_width, image_height):
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = norm_coordinates(landmark.x, landmark.y, w, h)
        if landmark_px:
            idx_to_coors[idx] = landmark_px
    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:,0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:,1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:,0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:,1])
    (startX, startY) = (max(0, x_min), max(0, y_min))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    return startX, startY, endX, endY

def analyze_brow_furrow(fl, w, h):
    try:
        landmarks = {}
        for idx, landmark in enumerate(fl.landmark):
            landmarks[idx] = norm_coordinates(landmark.x, landmark.y, w, h)
        left_inner_brow = landmarks.get(55, (0,0))
        right_inner_brow = landmarks.get(285, (0,0))
        inner_brow_distance = abs(left_inner_brow[0] - right_inner_brow[0])
        left_brow_center = landmarks.get(70, (0,0))
        right_brow_center = landmarks.get(300, (0,0))
        left_eye_center = landmarks.get(159, (0,0))
        right_eye_center = landmarks.get(386, (0,0))
        face_width = abs(landmarks.get(454, (0,0))[0] - landmarks.get(234, (0,0))[0])
        if face_width > 0:
            left_brow_height = (left_eye_center[1] - left_brow_center[1]) / face_width
            right_brow_height = (right_eye_center[1] - right_brow_center[1]) / face_width
            avg_brow_height = (left_brow_height + right_brow_height) / 2
            normal_inner_distance = face_width * 0.15
            inner_brow_score = max(0, (normal_inner_distance - inner_brow_distance) / normal_inner_distance) * 50
            normal_brow_height = 0.08
            brow_height_score = max(0, (normal_brow_height - avg_brow_height) / normal_brow_height) * 50
            brow_furrow_score = min(100, inner_brow_score + brow_height_score)
        else:
            brow_furrow_score = 0
        return brow_furrow_score
    except Exception as e:
        return 0

def calculate_presentation_metrics(emotions, history_buffer=None, brow_furrow_score=0):
    global metrics_history

    neutral = emotions[0] * 100
    happy = emotions[1] * 100
    sad = emotions[2] * 100
    surprise = emotions[3] * 100
    fear = emotions[4] * 100
    disgust = emotions[5] * 100
    angry = emotions[6] * 100

    # Calculate frustration (confusion) and engagement
    emotion_confusion = (fear * 0.3 + surprise * 0.2 + disgust * 0.1 + sad * 0.1)
    frustration = (emotion_confusion * 0.4 + brow_furrow_score * 0.6)
    frustration = max(0, min(100, frustration))

    positive_engagement = (happy * 0.5 + surprise * 0.3)
    negative_disengagement = (sad * 0.3 + neutral * 0.2 + disgust * 0.2)
    engagement = positive_engagement - (negative_disengagement * 0.5)
    engagement = max(0, min(100, engagement * 2))

    # Calculate spikes
    emotion_confusion_spike = (fear * 0.3 + surprise * 0.2 + disgust * 0.2)
    frustration_spike = (emotion_confusion_spike * 0.2 + brow_furrow_score * 0.8)
    if brow_furrow_score > 30:
        frustration_spike = frustration_spike * (1 + (brow_furrow_score - 30) / 100)
    frustration_spike = max(0, min(100, frustration_spike * 2.0))

    excitement_spike = (happy * 0.4 + surprise * 0.6)
    excitement_spike = max(0, min(100, excitement_spike * 2))

    # Detect sustained spikes
    frustration_spike_detected = False
    engagement_spike_detected = False

    if history_buffer is not None and len(history_buffer) >= 10:
        if len(history_buffer) >= 15:
            baseline_confusion = [h['confusion'] for h in history_buffer[-15:-5]]
            baseline_excitement = [h['excitement'] for h in history_buffer[-15:-5]]
            recent_confusion = [h['confusion'] for h in history_buffer[-10:]]
            recent_excitement = [h['excitement'] for h in history_buffer[-10:]]
        else:
            mid_point = len(history_buffer) // 2
            baseline_confusion = [h['confusion'] for h in history_buffer[:mid_point]]
            baseline_excitement = [h['excitement'] for h in history_buffer[:mid_point]]
            recent_confusion = [h['confusion'] for h in history_buffer[mid_point:]]
            recent_excitement = [h['excitement'] for h in history_buffer[mid_point:]]

        if baseline_confusion and recent_confusion:
            avg_baseline_confusion = sum(baseline_confusion) / len(baseline_confusion)
            sustained_confusion_frames = sum(1 for val in recent_confusion if val > avg_baseline_confusion + 3)
            if sustained_confusion_frames >= len(recent_confusion) * 0.6:
                frustration_spike_detected = True

        if baseline_excitement and recent_excitement:
            avg_baseline_excitement = sum(baseline_excitement) / len(baseline_excitement)
            sustained_excitement_frames = sum(1 for val in recent_excitement if val > avg_baseline_excitement + 8)
            if sustained_excitement_frames >= len(recent_excitement) * 0.6:
                engagement_spike_detected = True

    return frustration, engagement, frustration_spike_detected, engagement_spike_detected, frustration_spike, excitement_spike

def process_frame(frame):
    """Process a single frame and return metrics"""
    global lstm_features, metrics_history, face_mesh

    try:
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                startX, startY, endX, endY = get_box(fl, w, h)
                cur_face = frame_rgb[startY:endY, startX:endX]
                brow_furrow_score = analyze_brow_furrow(fl, w, h)

                if cur_face.size > 0:
                    # Process with LSTM model
                    cur_face_pil = Image.fromarray(cur_face)
                    cur_face_processed = pth_processing(cur_face_pil)
                    features = torch.nn.functional.relu(pth_backbone_model.extract_features(cur_face_processed)).detach().numpy()

                    if len(lstm_features) == 0:
                        lstm_features = [features] * 10
                    else:
                        lstm_features = lstm_features[1:] + [features]

                    lstm_f = torch.from_numpy(np.vstack(lstm_features))
                    lstm_f = torch.unsqueeze(lstm_f, 0)
                    output = pth_LSTM_model(lstm_f).detach().numpy()
                    emotions_array = output[0]

                    # Calculate metrics
                    frustration, engagement, frustration_spike_detected, engagement_spike_detected, frustration_spike, excitement_spike = calculate_presentation_metrics(emotions_array, metrics_history, brow_furrow_score)

                    # Update history
                    current_metrics = {'confusion': frustration_spike, 'excitement': excitement_spike, 'timestamp': time.time()}
                    metrics_history.append(current_metrics)
                    if len(metrics_history) > 20:
                        metrics_history.pop(0)

                    return frustration, engagement, frustration_spike_detected, engagement_spike_detected

        # No face detected
        return 0, 0, False, False

    except Exception as e:
        print(f"Error processing frame: {e}")
        return 0, 0, False, False

def draw_preview_overlay(frame, frustration, engagement, frus_spike, eng_spike):
    """Draw emotion metrics overlay on preview frame"""
    h, w = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    
    # Create metrics text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_text = f"{timestamp}, FRUS: {int(frustration):03d}, ENG: {int(engagement):03d}"
    
    # Draw background rectangle for text
    text_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(overlay, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (text_size[0] + 20, text_size[1] + 30), (255, 255, 255), 2)
    
    # Draw metrics text
    cv2.putText(overlay, metrics_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw emotion bars
    bar_y = 60
    bar_width = 200
    bar_height = 20
    
    # Frustration bar
    frus_color = (0, 0, 255) if frus_spike else (0, 255, 255)
    cv2.rectangle(overlay, (10, bar_y), (10 + int(bar_width * frustration / 100), bar_y + bar_height), frus_color, -1)
    cv2.rectangle(overlay, (10, bar_y), (10 + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.putText(overlay, f"Frustration: {int(frustration)}", (220, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Engagement bar
    eng_color = (0, 255, 0) if eng_spike else (255, 255, 0)
    cv2.rectangle(overlay, (10, bar_y + 30), (10 + int(bar_width * engagement / 100), bar_y + 30 + bar_height), eng_color, -1)
    cv2.rectangle(overlay, (10, bar_y + 30), (10 + bar_width, bar_y + 30 + bar_height), (255, 255, 255), 2)
    cv2.putText(overlay, f"Engagement: {int(engagement)}", (220, bar_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Spike indicators
    if frus_spike:
        cv2.putText(overlay, "FRUSTRATION SPIKE!", (10, bar_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if eng_spike:
        cv2.putText(overlay, "ENGAGEMENT SPIKE!", (10, bar_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add stats info
    stats = frame_manager.get_stats()
    stats_text = f"Frames: {stats['frames_received']} | Processed: {stats['frames_processed']} | FPS: {stats['receive_fps']:.1f}"
    cv2.putText(overlay, stats_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return overlay

def process_frames_worker():
    """Background worker to process frames from memory"""
    global latest_result, latest_preview_frame

    print("🔄 Frame processing worker started - will process frames from memory")
    print("📺 Preview window will show processed frames with emotion stats")

    last_frame_hash = None

    while True:
        try:
            # Get latest frame from memory (no disk I/O)
            frame = frame_manager.get_latest_frame()

            if frame is not None:
                # Calculate frame hash to detect if it's actually a new frame
                frame_hash = hash(frame.tobytes())

                if frame_hash == last_frame_hash:
                    # Same frame as before, skip processing
                    print("⏭️  Skipping duplicate frame")
                    time.sleep(0.1)  # Short sleep before checking again
                    continue

                last_frame_hash = frame_hash
                print(f"🔍 Processing NEW frame from memory (shape: {frame.shape})")

                # Process immediately
                frustration, engagement, frus_spike, eng_spike = process_frame(frame)

                # Update global result
                latest_result = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "frustration": int(frustration),
                    "frustration_spike": frus_spike,
                    "engagement": int(engagement),
                    "engagement_spike": eng_spike
                }

                print(f"🎯 EMOTION RESULTS: FRUS:{int(frustration):3d} ENG:{int(engagement):3d} "
                      f"FRUS_SPIKE:{frus_spike} ENG_SPIKE:{eng_spike}")

                # Create preview with overlay and store it
                preview_frame = draw_preview_overlay(frame, frustration, engagement, frus_spike, eng_spike)
                with preview_lock:
                    latest_preview_frame = preview_frame.copy()

            else:
                print("⏳ No frames available to process yet...")

            time.sleep(0.2)  # Process faster but with deduplication

        except Exception as e:
            print(f"❌ Error in processing worker: {e}")
            time.sleep(1)

def load_models():
    """Load the AI models"""
    global pth_LSTM_model, pth_backbone_model, mp_face_mesh, face_mesh

    try:
        pth_LSTM_model = LSTMPyTorch()
        pth_LSTM_model.load_state_dict(torch.load('FER_dinamic_LSTM_Aff-Wild2.pt', map_location='cpu'))
        pth_LSTM_model.eval()

        pth_backbone_model = ResNet50(7, channels=3)
        pth_backbone_model.load_state_dict(torch.load('FER_static_ResNet50_AffectNet.pt', map_location='cpu'))
        pth_backbone_model.eval()

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        print("✅ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# WebSocket handlers
def new_client(client, server):
    """Called when a new client connects"""
    print(f"🎉 Google Meet bot connected! Client: {client}")

def client_left(client, server):
    """Called when a client disconnects"""
    print(f"❌ Google Meet bot disconnected! Client: {client}")

def message_received(client, server, message):
    """Called when a message is received"""
    try:
        ws_message = json.loads(message)

        if ws_message.get('event') == 'video_separate_png.data':
            participant = ws_message['data']['data'].get('participant', {})
            participant_name = participant.get('name', 'Unknown')
            recording_id = ws_message['data']['recording']['id']

            print(f"🖼️ RECEIVED PNG CHUNK from: {participant_name} (Recording: {recording_id})")

            # Get PNG data and update frame
            png_buffer = ws_message['data']['data']['buffer']

            if frame_manager.update_frame_png(png_buffer):
                stats = frame_manager.get_stats()
                print(f"✅ PNG FRAME PROCESSED | Total Received: {stats['frames_received']} | FPS: {stats['receive_fps']:.1f}")
            else:
                print("❌ Failed to decode PNG frame")

        else:
            print(f"❓ Unhandled WebSocket event: {ws_message.get('event', 'unknown')}")

    except json.JSONDecodeError as e:
        print(f'❌ JSON parse error: {e} | Raw message length: {len(message)}')
    except Exception as e:
        print(f'❌ WebSocket message error: {e}')

# Flask routes
@app.route('/')
def index():
    """Basic health check"""
    stats = frame_manager.get_stats()
    return jsonify({
        'status': 'running',
        'message': 'Integrated Emotion Server is running',
        'websocket_url': 'ws://localhost:5003',
        'frames_received': stats['frames_received'],
        'latest_result': latest_result
    })

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """API endpoint to get current metrics in the specified format"""
    # Format: "TIMESTAMP, FRUS: 010, FRUS_HAS_SPIKE: TRUE, ENG: 005, ENG_HAS_SPIKE: FALSE"
    result_string = f"{latest_result['timestamp']}, FRUS: {latest_result['frustration']:03d}, FRUS_HAS_SPIKE: {str(latest_result['frustration_spike']).upper()}, ENG: {latest_result['engagement']:03d}, ENG_HAS_SPIKE: {str(latest_result['engagement_spike']).upper()}"

    return result_string

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    stats = frame_manager.get_stats()
    return jsonify({
        "status": "healthy",
        "models_loaded": pth_LSTM_model is not None,
        "frames_received": stats['frames_received'],
        "frames_processed": stats['frames_processed'],
        "frames_skipped": stats['frames_skipped'],
        "latest_result": latest_result
    })

@app.route('/websocket_status', methods=['GET'])
def websocket_status():
    """Check WebSocket server status"""
    return jsonify({
        "websocket_port": 5003,
        "websocket_url": "ws://localhost:5003",
        "server_running": True,
        "instructions": "Configure your Google Meet bot to send video data to ws://YOUR_NGROK_URL:5003"
    })

def preview_display_worker():
    """Display preview window in main thread"""
    global latest_preview_frame
    
    print("📺 Preview display worker started")
    
    while True:
        try:
            with preview_lock:
                if latest_preview_frame is not None:
                    cv2.imshow('Emotion Analysis Preview', latest_preview_frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("👋 Preview window closed by user")
                        cv2.destroyAllWindows()
                        break
            
            time.sleep(0.033)  # ~30 FPS display rate
            
        except Exception as e:
            print(f"❌ Error in preview display: {e}")
            time.sleep(1)

def start_websocket_server():
    """Start WebSocket server in background thread"""
    server = WebsocketServer(host='0.0.0.0', port=5003)
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)

    print("🚀 WebSocket server started on port 5003")
    server.run_forever()

if __name__ == '__main__':
        print("🚀 Starting INTEGRATED Emotion Server...")
        print("Loading models...")

        if load_models():
            # Start WebSocket server in background
            ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
            ws_thread.start()

            # Start frame processing worker
            processing_thread = threading.Thread(target=process_frames_worker, daemon=True)
            processing_thread.start()

            # Start preview display worker
            preview_thread = threading.Thread(target=preview_display_worker, daemon=True)
            preview_thread.start()

            print("🚀 Starting Flask API server on port 5000...")
            print("📡 WebSocket: ws://localhost:5003")
            print("🌐 HTTP API: http://localhost:5000")
            print("🌐 Make WebSocket public with: ngrok http 5003")
            print("\n🎯 PNG-ONLY EMOTION SERVER - MAXIMUM RELIABILITY!")
            print("   PNG → Processing → Results (direct, no corruption)")
            print("   NO FILES SAVED - everything in memory!")
            print("📺 Preview window will show processed frames with emotion stats")
            print("   Press 'q' in preview window to close it")
            print("\n✅ CONFIGURE YOUR BOT FOR PNG FORMAT:")
            print("   Use 'video_separate_png' in recording_config")
            print("   Use 'video_separate_png.data' in events")
            print("\n" + "="*50)

            try:
                app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
                cv2.destroyAllWindows()
                print("✅ Server stopped cleanly")
        else:
            print("❌ Failed to load models. Exiting.")
