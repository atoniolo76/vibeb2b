#!/usr/bin/env python3
"""
EMOTION SERVER WITH MOONDREAM AI - Direct Webcam Input
Uses webcam (OBS Virtual Camera) for real-time emotion analysis with Moondream AI
Displays preview with emotion metrics overlay

IMPORTANT: This version uses Moondream AI for emotion analysis.
No PyTorch models or brow analysis - just natural language AI.
"""

import cv2
import numpy as np
import time
from datetime import datetime
import moondream as md
from PIL import Image
import re
import warnings

warnings.simplefilter("ignore", UserWarning)

# Moondream AI configuration
MOONDREAM_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIyM2Q2ODE5Yy03NjM0LTRiOTEtOGRjZS0yMTY3OTQ5Njk5MzIiLCJvcmdfaWQiOiJOV1E5YlhudWdGdHVXdGJteXZTdUpYZFpvdzJzckxGZCIsImlhdCI6MTc1OTAxNjM1MCwidmVyIjoxfQ.g0DhZ8oh0kIiygatgS0LFNvb7UbXiYm9ecsXKAQZaSM"

# Global state
moondream_model = None
latest_category = "NEUTRAL"
metrics_history = []

def extract_category_from_text(text):
    """Extract single category from AI response text"""
    print(f"🔍 Extracting category from: '{text}'")
    
    text_upper = text.upper()
    
    # Look for the four main categories
    if 'NO_PERSON_FOUND' in text_upper:
        category = 'NO_PERSON_FOUND'
    elif 'SMILING' in text_upper:
        category = 'SMILING'
    elif 'CONFUSED' in text_upper:
        category = 'CONFUSED'
    elif 'NEUTRAL' in text_upper:
        category = 'NEUTRAL'
    else:
        # Fallback: look for keywords
        text_lower = text.lower()
        if 'smiling' in text_lower or 'smile' in text_lower or 'happy' in text_lower:
            category = 'SMILING'
        elif 'confused' in text_lower or 'puzzled' in text_lower or 'furrowed' in text_lower:
            category = 'CONFUSED'
        else:
            category = 'NEUTRAL'  # Default to neutral
    
    print(f"✅ Extracted category: {category}")
    return category

def analyze_emotions_with_moondream(frame):
    """Use Moondream AI to analyze emotions in the frame"""
    global moondream_model
    
    try:
        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize image for faster API processing (smaller = faster)
        pil_image = pil_image.resize((320, 240), Image.Resampling.LANCZOS)
        
        # Ask Moondream to categorize the person in the image (no test query for speed)
        query = """Look at this image. If you see a person, categorize their facial expression.

If NO PERSON is found in the image, respond with: "NO_PERSON_FOUND"

If a person IS found, choose ONE of these four categories:

NEUTRAL - The person has a normal, calm expression (not smiling, not confused)
SMILING - The person is clearly smiling or has a happy expression
CONFUSED - The person looks puzzled, confused, or frustrated (furrowed brows, puzzled look, squinting, frowning)

Respond with just ONE word: NEUTRAL, SMILING, CONFUSED, or NO_PERSON_FOUND"""
        
        print("🤖 Sending emotion analysis query...")
        result = moondream_model.query(pil_image, query)
        print(f"📡 Raw API response: {result}")
        
        if "answer" in result:
            answer = result["answer"]
            print(f"🤖 Moondream Response: {answer}")
            
            # Check for NO_PERSON_FOUND
            if "NO_PERSON_FOUND" in answer.upper():
                print("👤 No person detected in image")
                return "NO_PERSON_FOUND"
            
            # Extract single category from response
            category = extract_category_from_text(answer)
            
            print(f"📊 Parsed category: {category}")
            return category
        else:
            print(f"❌ No 'answer' field in response: {result}")
            return "NEUTRAL"
            
    except Exception as e:
        print(f"❌ Moondream analysis error: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return "NEUTRAL"  # Default value


def draw_metrics_overlay(frame, category):
    """Draw emotion metrics overlay on frame"""
    h, w = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    
    # Create metrics text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_text = f"{timestamp}, EMOTION: {category}"
    
    # Choose color based on category
    if category == "CONFUSED":
        color = (0, 0, 255)  # Red
    elif category == "SMILING":
        color = (0, 255, 0)  # Green
    elif category == "NO_PERSON_FOUND":
        color = (0, 0, 255)  # Red
    else:  # NEUTRAL
        color = (255, 255, 0)  # Yellow
    
    # Draw background rectangle for text
    text_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(overlay, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (text_size[0] + 20, text_size[1] + 30), (255, 255, 255), 2)
    
    # Draw metrics text
    cv2.putText(overlay, metrics_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw large category display
    category_text = f"STATE: {category}"
    category_size = cv2.getTextSize(category_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    category_x = (w - category_size[0]) // 2
    category_y = h - 50
    
    # Draw background for category
    cv2.rectangle(overlay, (category_x - 10, category_y - 40), (category_x + category_size[0] + 10, category_y + 10), (0, 0, 0), -1)
    cv2.rectangle(overlay, (category_x - 10, category_y - 40), (category_x + category_size[0] + 10, category_y + 10), (255, 255, 255), 3)
    
    # Draw category text
    cv2.putText(overlay, category_text, (category_x, category_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # Add AI indicator
    cv2.putText(overlay, "AI: Moondream", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return overlay

def load_moondream():
    """Load the Moondream AI model"""
    global moondream_model
    
    try:
        moondream_model = md.vl(api_key=MOONDREAM_API_KEY)
        print("✅ Moondream AI loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading Moondream AI: {e}")
        return False

def main():
    """Main function to run webcam emotion analysis with Moondream AI"""
    print("🚀 Starting Webcam Emotion Analysis with Moondream AI...")
    print("Loading Moondream AI...")

    if not load_moondream():
        print("❌ Failed to load Moondream AI. Exiting.")
        return

    # Initialize webcam (try different camera indices and backends)
    cap = None
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # Try different backends
    
    for backend in backends:
        for camera_index in [1, 2, 0]:  # Try OBS Virtual Camera first (index 1)
            print(f"🔍 Trying camera {camera_index} with backend {backend}")
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Webcam connected (camera index: {camera_index}, backend: {backend})")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        
        if cap and cap.isOpened():
            break
    
    if cap is None or not cap.isOpened():
        print("❌ Failed to connect to webcam. Make sure OBS Virtual Camera is running.")
        print("💡 Try running OBS Studio and starting Virtual Camera")
        return

    # Set camera properties (don't force specific resolution)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability

    print("🎥 Webcam initialized - Press 'q' to quit")
    print("🤖 Real-time emotion analysis with Moondream AI")
    print("📊 AI categorizes: NEUTRAL, SMILING, CONFUSED, or NO_PERSON_FOUND")
    print("=" * 60)

    frame_count = 0
    start_time = time.time()
    last_analysis_time = 0
    analysis_interval = 0.5  # Analyze every 0.5 seconds for minimal latency

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam - trying to reconnect...")
            # Try to reconnect
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try OBS Virtual Camera with DSHOW backend
            if not cap.isOpened():
                print("❌ Could not reconnect to webcam")
                break
            continue

        frame_count += 1
        current_time = time.time()
        
        # Analyze emotions with Moondream AI every few seconds
        if current_time - last_analysis_time >= analysis_interval:
            print("🤖 Analyzing emotions with Moondream AI...")
            category = analyze_emotions_with_moondream(frame)
            
            # Update global state
            latest_category = category
            
            last_analysis_time = current_time
        
        # Draw metrics overlay using latest analysis
        display_frame = draw_metrics_overlay(frame, latest_category)
        
        # Calculate and display FPS
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"🎯 FPS: {fps:.1f} | EMOTION: {latest_category}")
        
        # Display frame
        cv2.imshow('Emotion Analysis - Moondream AI', display_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam emotion analysis stopped")

if __name__ == '__main__':
    main()
