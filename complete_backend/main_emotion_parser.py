#!/usr/bin/env python3
"""
MAIN EMOTION PARSER
Processes MP4 video files from Upload folder, extracts every 5th frame,
runs emotion analysis using the FER model, and outputs JSON with timestamps and emotions.

This script integrates the emotion detection logic from integrated_emotion_server.py
to analyze video files offline and generate emotion analysis reports.
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
import math
import warnings
from PIL import Image
from datetime import datetime
import logging

warnings.simplefilter("ignore", UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyTorch Model Classes (from integrated_emotion_server.py)
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

class EmotionParser:
    def __init__(self, model_path="FER_static_ResNet50_AffectNet.pt"):
        """Initialize the emotion parser with the FER model."""
        self.model_path = model_path
        self.backbone_model = None
        self.lstm_model = None
        self.face_mesh = None
        self.mp_face_mesh = None
        self.lstm_features = []
        self.metrics_history = []
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load the AI models for emotion detection."""
        try:
            logger.info("Loading emotion detection models...")
            
            # Load LSTM model (we'll create a dummy one since we don't have the LSTM file)
            self.lstm_model = LSTMPyTorch()
            # Note: We don't have the LSTM model file, so we'll use a simplified approach
            
            # Load backbone ResNet model
            self.backbone_model = ResNet50(7, channels=3)
            self.backbone_model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.backbone_model.eval()
            
            # Initialize MediaPipe face mesh (if available)
            if MEDIAPIPE_AVAILABLE:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            else:
                logger.warning("MediaPipe not available. Face detection will be skipped.")
                self.mp_face_mesh = None
                self.face_mesh = None
            
            logger.info("✅ Models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def pth_processing(self, img):
        """Preprocess image for PyTorch model."""
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
        return get_img_torch(img)
    
    def norm_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        """Normalize coordinates."""
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    
    def get_box(self, fl, w, h):
        """Get bounding box from face landmarks."""
        idx_to_coors = {}
        for idx, landmark in enumerate(fl.landmark):
            landmark_px = self.norm_coordinates(landmark.x, landmark.y, w, h)
            if landmark_px:
                idx_to_coors[idx] = landmark_px
        x_min = np.min(np.asarray(list(idx_to_coors.values()))[:,0])
        y_min = np.min(np.asarray(list(idx_to_coors.values()))[:,1])
        endX = np.max(np.asarray(list(idx_to_coors.values()))[:,0])
        endY = np.max(np.asarray(list(idx_to_coors.values()))[:,1])
        (startX, startY) = (max(0, x_min), max(0, y_min))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        return startX, startY, endX, endY
    
    def analyze_brow_furrow(self, fl, w, h):
        """Analyze brow furrow for frustration detection."""
        try:
            landmarks = {}
            for idx, landmark in enumerate(fl.landmark):
                landmarks[idx] = self.norm_coordinates(landmark.x, landmark.y, w, h)
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
    
    def calculate_presentation_metrics(self, emotions, brow_furrow_score=0):
        """Calculate presentation metrics from emotion predictions."""
        neutral = emotions[0] * 100
        happy = emotions[1] * 100
        sad = emotions[2] * 100
        surprise = emotions[3] * 100
        fear = emotions[4] * 100
        disgust = emotions[5] * 100
        angry = emotions[6] * 100

        # Calculate frustration (confusion) - out of 100
        emotion_confusion = (fear * 0.3 + surprise * 0.2 + disgust * 0.1 + sad * 0.1)
        frustration = (emotion_confusion * 0.4 + brow_furrow_score * 0.6)
        frustration = max(0, min(100, frustration))

        # Engagement: Start with happiness, subtract negative emotions
        # All negative emotions (sad, fear, disgust, angry) reduce engagement
        # Neutral also reduces engagement slightly
        engagement = happy
        engagement -= (sad * 0.3)      # Sadness reduces engagement
        engagement -= (fear * 0.4)      # Fear reduces engagement  
        engagement -= (disgust * 0.3)   # Disgust reduces engagement
        engagement -= (angry * 0.4)     # Anger reduces engagement
        engagement -= (neutral * 0.1)   # Neutral slightly reduces engagement
        engagement = max(0, min(100, engagement))

        return {
            'neutral': round(neutral, 2),
            'happy': round(happy, 2),
            'sad': round(sad, 2),
            'surprise': round(surprise, 2),
            'fear': round(fear, 2),
            'disgust': round(disgust, 2),
            'angry': round(angry, 2),
            'frustration': round(frustration, 2),
            'engagement': round(engagement, 2),
            'brow_furrow_score': round(brow_furrow_score, 2)
        }
    
    def process_frame(self, frame, frame_number, timestamp):
        """Process a single frame and return emotion analysis."""
        try:
            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Check if MediaPipe is available
            if self.face_mesh is None:
                # No face detection available, return default neutral emotions
                return {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'face_detected': False,
                    'face_bbox': None,
                    'emotions': {
                        'neutral': 80.0,
                        'happy': 10.0,
                        'sad': 5.0,
                        'surprise': 2.0,
                        'fear': 1.0,
                        'disgust': 1.0,
                        'angry': 1.0,
                        'frustration': 0.0,
                        'engagement': 20.0,
                        'brow_furrow_score': 0.0
                    }
                }

            # Process with MediaPipe
            results = self.face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    startX, startY, endX, endY = self.get_box(fl, w, h)
                    cur_face = frame_rgb[startY:endY, startX:endX]
                    brow_furrow_score = self.analyze_brow_furrow(fl, w, h)

                    if cur_face.size > 0:
                        # Process with backbone model only (simplified approach)
                        cur_face_pil = Image.fromarray(cur_face)
                        cur_face_processed = self.pth_processing(cur_face_pil)
                        
                        # Get emotion predictions from backbone model
                        with torch.no_grad():
                            emotions_output = self.backbone_model(cur_face_processed)
                            emotions_array = torch.softmax(emotions_output, dim=1).numpy()[0]

                        # Calculate metrics
                        metrics = self.calculate_presentation_metrics(emotions_array, brow_furrow_score)
                        
                        return {
                            'frame_number': frame_number,
                            'timestamp': timestamp,
                            'face_detected': True,
                            'face_bbox': {
                                'startX': int(startX),
                                'startY': int(startY),
                                'endX': int(endX),
                                'endY': int(endY)
                            },
                            'emotions': metrics
                        }

            # No face detected
            return {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'face_detected': False,
                'face_bbox': None,
                'emotions': {
                    'neutral': 0,
                    'happy': 0,
                    'sad': 0,
                    'surprise': 0,
                    'fear': 0,
                    'disgust': 0,
                    'angry': 0,
                    'frustration': 0,
                    'engagement': 0,
                    'brow_furrow_score': 0
                }
            }

        except Exception as e:
            logger.error(f"Error processing frame {frame_number}: {e}")
            return {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'face_detected': False,
                'face_bbox': None,
                'emotions': {
                    'neutral': 0,
                    'happy': 0,
                    'sad': 0,
                    'surprise': 0,
                    'fear': 0,
                    'disgust': 0,
                    'angry': 0,
                    'frustration': 0,
                    'engagement': 0,
                    'brow_furrow_score': 0
                }
            }
    
    def process_video(self, video_path, output_path=None, frame_interval=5):
        """
        Process video file and extract emotions from every nth frame.
        
        Args:
            video_path: Path to the input MP4 file
            output_path: Path for the output JSON file (optional)
            frame_interval: Process every nth frame (default: 5)
            
        Returns:
            Dictionary containing emotion analysis results
        """
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
            
            # Process frames
            frame_results = []
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    
                    logger.info(f"Processing frame {frame_count}/{total_frames} (timestamp: {timestamp:.2f}s)")
                    
                    result = self.process_frame(frame, frame_count, timestamp)
                    frame_results.append(result)
                    processed_count += 1
                
                frame_count += 1
            
            cap.release()
            
            # Calculate summary statistics
            face_detected_frames = sum(1 for r in frame_results if r['face_detected'])
            avg_frustration = np.mean([r['emotions']['frustration'] for r in frame_results if r['face_detected']]) if face_detected_frames > 0 else 0
            avg_engagement = np.mean([r['emotions']['engagement'] for r in frame_results if r['face_detected']]) if face_detected_frames > 0 else 0
            
            # Create final result
            analysis_result = {
                'video_info': {
                    'file_path': video_path,
                    'total_frames': total_frames,
                    'processed_frames': processed_count,
                    'frame_interval': frame_interval,
                    'fps': fps,
                    'duration_seconds': duration,
                    'face_detected_frames': face_detected_frames,
                    'face_detection_rate': round(face_detected_frames / processed_count * 100, 2) if processed_count > 0 else 0
                },
                'summary_metrics': {
                    'average_frustration': round(avg_frustration, 2),
                    'average_engagement': round(avg_engagement, 2),
                    'max_frustration': round(max([r['emotions']['frustration'] for r in frame_results]), 2) if frame_results else 0,
                    'max_engagement': round(max([r['emotions']['engagement'] for r in frame_results]), 2) if frame_results else 0
                },
                'frame_analysis': frame_results,
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'model_used': 'FER_static_ResNet50_AffectNet',
                    'processing_version': '1.0'
                }
            }
            
            # Save to JSON file if output path is provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False)
                logger.info(f"Emotion analysis saved to {output_path}")
            
            logger.info(f"✅ Processing complete: {processed_count} frames analyzed, {face_detected_frames} with faces detected")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            raise

def process_upload_folder_emotions(upload_folder="Upload", output_folder="Output", frame_interval=5):
    """
    Process all MP4 files in the upload folder for emotion analysis.
    
    Args:
        upload_folder: Path to the folder containing MP4 files
        output_folder: Path to save the output JSON files
        frame_interval: Process every nth frame (default: 5)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize emotion parser
    parser = EmotionParser()
    
    # Find all MP4 files in upload folder
    mp4_files = [f for f in os.listdir(upload_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        logger.warning(f"No MP4 files found in {upload_folder}")
        return
    
    logger.info(f"Found {len(mp4_files)} MP4 file(s) to process for emotion analysis")
    
    for mp4_file in mp4_files:
        try:
            video_path = os.path.join(upload_folder, mp4_file)
            output_filename = mp4_file.replace('.mp4', '_emotion_analysis.json')
            output_path = os.path.join(output_folder, output_filename)
            
            logger.info(f"Processing emotions for: {mp4_file}")
            analysis_result = parser.process_video(video_path, output_path, frame_interval)
            
            # Print summary
            video_info = analysis_result['video_info']
            summary = analysis_result['summary_metrics']
            logger.info(f"Completed: {mp4_file}")
            logger.info(f"  - Processed {video_info['processed_frames']} frames (every {frame_interval}th frame)")
            logger.info(f"  - Face detection rate: {video_info['face_detection_rate']}%")
            logger.info(f"  - Average frustration: {summary['average_frustration']}")
            logger.info(f"  - Average engagement: {summary['average_engagement']}")
            
        except Exception as e:
            logger.error(f"Failed to process {mp4_file}: {e}")
            continue

if __name__ == "__main__":
    # Process all MP4 files in the Upload folder for emotion analysis
    process_upload_folder_emotions()
