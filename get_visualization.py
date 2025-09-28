#!/usr/bin/env python3
"""
POSE PROCESSOR
Processes MP4 video files from Upload folder, applies MediaPipe pose landmarking,
and saves augmented video with pose overlays (blue dots with white outlines connected by white lines).
"""

import os
import cv2
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PoseProcessor:
    def __init__(self):
        """Initialize the pose processor with MediaPipe."""
        self.mp_pose = None
        self.pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None

        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available. Pose processing will be skipped.")
            return

        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            logger.info("✅ Pose processor initialized with MediaPipe")
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
            logger.warning("Pose processing will be skipped.")
            self.mp_pose = None
            self.pose = None
    
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get pose connections for drawing lines between landmarks."""
        return self.mp_pose.POSE_CONNECTIONS
    
    def draw_landmarks_and_connections(self, image, landmarks, connections):
        """Draw all pose landmarks as blue dots with white outlines and connect them with white lines."""
        h, w, _ = image.shape
        
        # Draw connections (white lines)
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                start_point = landmarks.landmark[start_idx]
                end_point = landmarks.landmark[end_idx]
                
                # Convert normalized coordinates to pixel coordinates
                start_x = int(start_point.x * w)
                start_y = int(start_point.y * h)
                end_x = int(end_point.x * w)
                end_y = int(end_point.y * h)
                
                # Draw white line
                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        
        # Draw all landmarks (blue dots with white outlines)
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # Draw white outline (larger circle)
            cv2.circle(image, (x, y), 6, (255, 255, 255), -1)
            # Draw blue center (smaller circle)
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
    
    def process_video(self, input_path: str, output_path: str):
        """
        Process video file and add pose overlays.

        Args:
            input_path: Path to input MP4 file
            output_path: Path to save output MP4 file
        """
        try:
            logger.info(f"Processing video: {input_path}")

            # Open video file
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {input_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                raise Exception(f"Could not create output video file: {output_path}")

            # Check if pose processing is available
            pose_available = self.pose is not None
            if pose_available:
                pose_connections = self.get_pose_connections()
                logger.info("Pose processing enabled - will add pose overlays")
            else:
                logger.info("Pose processing disabled - copying video without overlays")
            
            frame_count = 0
            poses_detected = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Process pose if available
                if pose_available:
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process frame with MediaPipe
                    results = self.pose.process(rgb_frame)

                    # Draw pose if detected
                    if results.pose_landmarks:
                        poses_detected += 1
                        self.draw_landmarks_and_connections(frame, results.pose_landmarks, pose_connections)

                # Write frame to output video
                out.write(frame)
                
                # Progress logging
                if frame_count % 30 == 0:  # Log every 30 frames
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - Poses detected: {poses_detected}")
            
            # Cleanup
            cap.release()
            out.release()
            
            # Calculate statistics
            if pose_available:
                pose_detection_rate = (poses_detected / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"✅ Video processing completed with pose overlays!")
                logger.info(f"   - Total frames processed: {frame_count}")
                logger.info(f"   - Frames with poses detected: {poses_detected}")
                logger.info(f"   - Pose detection rate: {pose_detection_rate:.1f}%")
            else:
                logger.info(f"✅ Video processing completed (no pose overlays)!")
                logger.info(f"   - Total frames processed: {frame_count}")
                logger.info(f"   - Pose processing was disabled")

            logger.info(f"   - Output saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            raise

def process_upload_folder(upload_folder: str = "Upload", output_folder: str = "Output"):
    """
    Process MP4 file in upload folder and save pose video.
    
    Args:
        upload_folder: Path to the folder containing MP4 files
        output_folder: Path to save the output video
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find MP4 files in upload folder
    mp4_files = [f for f in os.listdir(upload_folder) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        logger.warning(f"No MP4 files found in {upload_folder}")
        return
    
    if len(mp4_files) > 1:
        logger.warning(f"Multiple MP4 files found, processing the first one: {mp4_files[0]}")
    
    # Process the first (and should be only) MP4 file
    mp4_file = mp4_files[0]
    input_path = os.path.join(upload_folder, mp4_file)
    
    # Create output filename
    base_name = mp4_file.replace('.mp4', '')
    output_filename = f"{base_name}_pose.mp4"
    output_path = os.path.join(output_folder, output_filename)
    
    try:
        # Initialize processor
        processor = PoseProcessor()
        
        # Process video
        logger.info(f"Processing: {mp4_file}")
        processor.process_video(input_path, output_path)
        
        logger.info(f"✅ Pose video saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to process {mp4_file}: {e}")

if __name__ == "__main__":
    # Process MP4 file in Upload folder
    process_upload_folder()