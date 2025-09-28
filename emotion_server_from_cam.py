#!/usr/bin/env python3
"""
EMOTION SERVER FROM CAM - Direct Webcam Input
Uses webcam (OBS Virtual Camera) for real-time emotion analysis
Displays preview with emotion metrics overlay

IMPORTANT: This version uses direct webcam input for face analysis.
No networking - just local webcam processing with visual preview.
"""

import cv2
import numpy as np
import time
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

# Global state
pth_LSTM_model = None
pth_backbone_model = None
mp_face_mesh = None
face_mesh = None
lstm_features = []
metrics_history = []

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

    # Calculate confusion - PRIMARILY driven by brow furrow
    # Brow furrow is the main indicator of confusion/frustration
    emotion_confusion = (fear * 0.2 + surprise * 0.1 + disgust * 0.1 + sad * 0.1)
    confusion = (emotion_confusion * 0.2 + brow_furrow_score * 0.8)  # 80% brow, 20% emotions
    confusion = max(0, min(100, confusion))

    positive_engagement = (happy * 0.5 + surprise * 0.3)
    negative_disengagement = (sad * 0.3 + neutral * 0.2 + disgust * 0.2)
    engagement = positive_engagement - (negative_disengagement * 0.5)
    engagement = max(0, min(100, engagement * 2))

    # Calculate spikes - confusion spike is almost entirely brow-driven
    emotion_confusion_spike = (fear * 0.1 + surprise * 0.1 + disgust * 0.1)
    confusion_spike = (emotion_confusion_spike * 0.1 + brow_furrow_score * 0.9)  # 90% brow
    if brow_furrow_score > 20:  # Lower threshold for brow detection
        confusion_spike = confusion_spike * (1 + (brow_furrow_score - 20) / 50)  # Amplify when brow is active
    confusion_spike = max(0, min(100, confusion_spike * 1.5))  # Scale up for visibility

    excitement_spike = (happy * 0.4 + surprise * 0.6)
    excitement_spike = max(0, min(100, excitement_spike * 2))

    # Detect sustained spikes
    confusion_spike_detected = False
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
            sustained_confusion_frames = sum(1 for val in recent_confusion if val > avg_baseline_confusion + 2)  # Lower threshold
            if sustained_confusion_frames >= len(recent_confusion) * 0.5:  # More sensitive
                confusion_spike_detected = True

        if baseline_excitement and recent_excitement:
            avg_baseline_excitement = sum(baseline_excitement) / len(baseline_excitement)
            sustained_excitement_frames = sum(1 for val in recent_excitement if val > avg_baseline_excitement + 8)
            if sustained_excitement_frames >= len(recent_excitement) * 0.6:
                engagement_spike_detected = True

    return confusion, engagement, confusion_spike_detected, engagement_spike_detected, confusion_spike, excitement_spike

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
                    confusion, engagement, confusion_spike_detected, engagement_spike_detected, confusion_spike, excitement_spike = calculate_presentation_metrics(emotions_array, metrics_history, brow_furrow_score)

                    # Update history
                    current_metrics = {'confusion': confusion_spike, 'excitement': excitement_spike, 'timestamp': time.time()}
                    metrics_history.append(current_metrics)
                    if len(metrics_history) > 20:
                        metrics_history.pop(0)

                    return confusion, engagement, confusion_spike_detected, engagement_spike_detected, startX, startY, endX, endY

        # No face detected
        return 0, 0, False, False, 0, 0, 0, 0

    except Exception as e:
        print(f"Error processing frame: {e}")
        return 0, 0, False, False, 0, 0, 0, 0

def draw_metrics_overlay(frame, confusion, engagement, conf_spike, eng_spike, startX, startY, endX, endY):
    """Draw emotion metrics overlay on frame"""
    h, w = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    
    # Draw face bounding box
    if startX > 0 and startY > 0 and endX > 0 and endY > 0:
        cv2.rectangle(overlay, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    # Create metrics text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_text = f"{timestamp}, CONF: {int(confusion):03d}, CONF_HAS_SPIKE: {str(conf_spike).upper()}, ENG: {int(engagement):03d}, ENG_HAS_SPIKE: {str(eng_spike).upper()}"
    
    # Draw background rectangle for text
    text_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(overlay, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (text_size[0] + 20, text_size[1] + 30), (255, 255, 255), 2)
    
    # Draw metrics text
    cv2.putText(overlay, metrics_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw individual emotion bars
    bar_y = 60
    bar_width = 200
    bar_height = 20
    
    # Confusion bar
    conf_color = (0, 0, 255) if conf_spike else (0, 255, 255)
    cv2.rectangle(overlay, (10, bar_y), (10 + int(bar_width * confusion / 100), bar_y + bar_height), conf_color, -1)
    cv2.rectangle(overlay, (10, bar_y), (10 + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.putText(overlay, f"Confusion: {int(confusion)}", (220, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Engagement bar
    eng_color = (0, 255, 0) if eng_spike else (255, 255, 0)
    cv2.rectangle(overlay, (10, bar_y + 30), (10 + int(bar_width * engagement / 100), bar_y + 30 + bar_height), eng_color, -1)
    cv2.rectangle(overlay, (10, bar_y + 30), (10 + bar_width, bar_y + 30 + bar_height), (255, 255, 255), 2)
    cv2.putText(overlay, f"Engagement: {int(engagement)}", (220, bar_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Spike indicators
    if conf_spike:
        cv2.putText(overlay, "CONFUSION SPIKE!", (10, bar_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if eng_spike:
        cv2.putText(overlay, "ENGAGEMENT SPIKE!", (10, bar_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return overlay

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

def main():
    """Main function to run webcam emotion analysis"""
    print("🚀 Starting Webcam Emotion Analysis...")
    print("Loading models...")

    if not load_models():
        print("❌ Failed to load models. Exiting.")
        return

    # Initialize webcam (try different camera indices)
    cap = None
    for camera_index in [1, 2]:  # Try different camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Webcam connected (camera index: {camera_index})")
            break
        else:
            cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌ Failed to connect to webcam. Make sure OBS Virtual Camera is running.")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("🎥 Webcam initialized - Press 'q' to quit")
    print("📊 Real-time emotion analysis with visual preview")
    print("=" * 60)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam")
            break

        frame_count += 1
        
        # Process frame for emotions
        confusion, engagement, conf_spike, eng_spike, startX, startY, endX, endY = process_frame(frame)
        
        # Draw metrics overlay
        display_frame = draw_metrics_overlay(frame, confusion, engagement, conf_spike, eng_spike, startX, startY, endX, endY)
        
        # Calculate and display FPS
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            # Get brow furrow score for debugging
            brow_score = 0
            if startX > 0:  # Face detected
                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        brow_score = analyze_brow_furrow(fl, w, h)
                        break
            print(f"🎯 FPS: {fps:.1f} | CONF: {int(confusion):3d} ENG: {int(engagement):3d} "
                  f"CONF_SPIKE: {conf_spike} ENG_SPIKE: {eng_spike} | BROW: {brow_score:.1f}")
        
        # Display frame
        cv2.imshow('Emotion Analysis - Webcam Input', display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam emotion analysis stopped")

if __name__ == '__main__':
    main()
