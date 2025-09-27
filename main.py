import cv2
import mediapipe as mp
import math
import numpy as np
import time
import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

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

def display_EMO_PRED(img, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), line_width=2):
    lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)
    text2_color = (255, 0, 255)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, text2_color, thickness=lw, lineType=cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tf = max(lw - 1, 1)
    text_fond = (0, 0, 0)
    text_width_2, text_height_2 = cv2.getTextSize(label, font, lw / 3, tf)
    text_width_2 = text_width_2[0] + round(((p2[0] - p1[0]) * 10) / 360)
    center_face = p1[0] + round((p2[0] - p1[0]) / 2)
    cv2.putText(img, label, (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), font, lw / 3, text_fond, thickness=tf, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), font, lw / 3, text2_color, thickness=tf, lineType=cv2.LINE_AA)
    return img

def display_FPS(img, text, margin=1.0, box_scale=1.0):
    img_h, img_w, _ = img.shape
    line_width = int(min(img_h, img_w) * 0.001)
    thickness = max(int(line_width / 3), 1)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)
    font_scale = thickness / 1.5
    t_w, t_h = cv2.getTextSize(text, font_face, font_scale, None)[0]
    margin_n = int(t_h * margin)
    sub_img = img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale), img_w - t_w - margin_n - int(2 * t_h * box_scale): img_w - margin_n]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale), img_w - t_w - margin_n - int(2 * t_h * box_scale):img_w - margin_n] = cv2.addWeighted(sub_img, 0.5, white_rect, .5, 1.0)
    cv2.putText(img=img, text=text, org=(img_w - t_w - margin_n - int(2 * t_h * box_scale) // 2, 0 + margin_n + t_h + int(2 * t_h * box_scale) // 2), fontFace=font_face, fontScale=font_scale, color=font_color, thickness=thickness, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    return img

def display_emotion_metrics(img, emotions, focus, confusion, engagement, excitement_spike, confusion_spike, excitement_spike_detected, confusion_spike_detected, x, y):
    cv2.putText(img, f'Focus: {focus:.1f}', (x, y - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, f'Confusion: {confusion:.1f}', (x, y - 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, f'Engagement: {engagement:.1f}', (x, y - 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    excitement_color = (0, 255, 255) if excitement_spike_detected > 10 else (100, 200, 200)
    confusion_color = (0, 0, 255) if confusion_spike_detected > 10 else (100, 100, 200)
    cv2.putText(img, f'Excitement Spike: {excitement_spike:.1f}', (x, y - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, excitement_color, 2)
    cv2.putText(img, f'Confusion Spike: {confusion_spike:.1f}', (x, y - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, confusion_color, 2)
    if excitement_spike_detected > 10:
        cv2.putText(img, f'EXCITEMENT SPIKE! +{excitement_spike_detected:.0f}', (x, y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
    if confusion_spike_detected > 10:
        cv2.putText(img, f'CONFUSION SPIKE! +{confusion_spike_detected:.0f}', (x, y - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.putText(img, f'Neutral: {emotions[0]:.1f}%', (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
    cv2.putText(img, f'Happy: {emotions[1]:.1f}%', (x, y - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    cv2.putText(img, f'Sad: {emotions[2]:.1f}%', (x, y - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    cv2.putText(img, f'Surprise: {emotions[3]:.1f}%', (x, y - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 165, 0), 1)
    cv2.putText(img, f'Fear: {emotions[4]:.1f}%', (x, y - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 0, 128), 1)
    cv2.putText(img, f'Disgust: {emotions[5]:.1f}%', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (165, 42, 42), 1)
    cv2.putText(img, f'Anger: {emotions[6]:.1f}%', (x, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    return img

def calculate_presentation_metrics(emotions, history_buffer=None, brow_furrow_score=0):
    neutral = emotions[0] * 100
    happy = emotions[1] * 100
    sad = emotions[2] * 100
    surprise = emotions[3] * 100
    fear = emotions[4] * 100
    disgust = emotions[5] * 100
    angry = emotions[6] * 100
    
    emotional_intensity = (happy + sad + angry + fear + disgust + surprise)
    focus = neutral * (1 - emotional_intensity / 600)
    focus = max(0, min(100, focus * 1.2))
    
    emotion_confusion = (fear * 0.3 + surprise * 0.2 + disgust * 0.1 + sad * 0.1)
    confusion = (emotion_confusion * 0.4 + brow_furrow_score * 0.6)
    confusion = max(0, min(100, confusion))
    
    positive_engagement = (happy * 0.5 + surprise * 0.3)
    negative_disengagement = (sad * 0.3 + neutral * 0.2 + disgust * 0.2)
    engagement = positive_engagement - (negative_disengagement * 0.5)
    engagement = max(0, min(100, engagement * 2))
    
    excitement_spike = (happy * 0.4 + surprise * 0.6)
    excitement_spike = max(0, min(100, excitement_spike * 2))
    
    emotion_confusion_spike = (fear * 0.3 + surprise * 0.2 + disgust * 0.2)
    confusion_spike = (emotion_confusion_spike * 0.2 + brow_furrow_score * 0.8)
    if brow_furrow_score > 30:
        confusion_spike = confusion_spike * (1 + (brow_furrow_score - 30) / 100)
    confusion_spike = max(0, min(100, confusion_spike * 2.0))
    
    confusion_spike_detected = 0
    excitement_spike_detected = 0
    
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
                confusion_spike_detected = min(100, confusion_spike - avg_baseline_confusion)
                print(f"🔴 CONFUSION SPIKE DETECTED!")
        
        if baseline_excitement and recent_excitement:
            avg_baseline_excitement = sum(baseline_excitement) / len(baseline_excitement)
            sustained_excitement_frames = sum(1 for val in recent_excitement if val > avg_baseline_excitement + 8)
            if sustained_excitement_frames >= len(recent_excitement) * 0.6:
                excitement_spike_detected = min(100, excitement_spike - avg_baseline_excitement)
                print(f"🎉 EXCITEMENT SPIKE DETECTED!")
    
    return focus, confusion, engagement, excitement_spike, confusion_spike, excitement_spike_detected, confusion_spike_detected

if __name__ == "__main__":
    print("Loading models...")
    mp_face_mesh = mp.solutions.face_mesh
    
    try:
        pth_LSTM_model = LSTMPyTorch()
        pth_LSTM_model.load_state_dict(torch.load('FER_dinamic_LSTM_Aff-Wild2.pt', map_location='cpu'))
        pth_LSTM_model.eval()
        print("✅ LSTM model loaded successfully!")
        
        pth_backbone_model = ResNet50(7, channels=3)
        pth_backbone_model.load_state_dict(torch.load('FER_static_ResNet50_AffectNet.pt', map_location='cpu'))
        pth_backbone_model.eval()
        print("✅ ResNet50 backbone loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        exit(1)
    
    DICT_EMO = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
    
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    lstm_features = []
    metrics_history = []
    
    print("🎥 Starting webcam... Press 'q' to quit")
    
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            t1 = time.time()
            success, frame = cap.read()
            if frame is None: 
                break

            frame_copy = frame.copy()
            frame_copy.flags.writeable = False
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_copy)
            frame_copy.flags.writeable = True

            if results.multi_face_landmarks:
                for fl in results.multi_face_landmarks:
                    startX, startY, endX, endY = get_box(fl, w, h)
                    cur_face = frame_copy[startY:endY, startX: endX]
                    brow_furrow_score = analyze_brow_furrow(fl, w, h)
                    
                    if cur_face.size > 0:
                        try:
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

                            cl = np.argmax(output)
                            label = DICT_EMO[cl]
                            confidence = output[0][cl]
                            emotions_array = output[0]
                            
                            focus, confusion, engagement, excitement_spike, confusion_spike, excitement_spike_detected, confusion_spike_detected = calculate_presentation_metrics(emotions_array, metrics_history, brow_furrow_score)
                            
                            current_metrics = {'confusion': confusion_spike, 'excitement': excitement_spike, 'timestamp': time.time()}
                            metrics_history.append(current_metrics)
                            if len(metrics_history) > 20:
                                metrics_history.pop(0)
                            
                            frame = display_EMO_PRED(frame, (startX, startY, endX, endY), f'{label} {confidence:.1%}', line_width=3)
                            frame = display_emotion_metrics(frame, emotions_array * 100, focus, confusion, engagement, excitement_spike, confusion_spike, excitement_spike_detected, confusion_spike_detected, startX, startY)
                            cv2.putText(frame, f'Brow Furrow: {brow_furrow_score:.1f}', (startX, startY + endY - startY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            
                        except Exception as e:
                            print(f"Error processing face: {e}")

            t2 = time.time()
            frame = display_FPS(frame, f'FPS: {1 / (t2 - t1):.1f}', box_scale=.5)
            cv2.imshow('FER Dynamic LSTM Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Demo ended")