#!/usr/bin/env python3
"""
Test Moondream API connection and functionality
"""

import moondream as md
from PIL import Image
import numpy as np

# Moondream AI configuration
MOONDREAM_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIyM2Q2ODE5Yy03NjM0LTRiOTEtOGRjZS0yMTY3OTQ5Njk5MzIiLCJvcmdfaWQiOiJOV1E5YlhudWdGdHVXdGJteXZTdUpYZFpvdzJzckxGZCIsImlhdCI6MTc1OTAxNjM1MCwidmVyIjoxfQ.g0DhZ8oh0kIiygatgS0LFNvb7UbXiYm9ecsXKAQZaSM"

def test_moondream_api():
    print("🤖 Testing Moondream API...")
    
    try:
        # Initialize Moondream
        print("📡 Connecting to Moondream API...")
        model = md.vl(api_key=MOONDREAM_API_KEY)
        print("✅ Moondream API connected successfully!")
        
        # Create a test image (simple colored rectangle)
        print("🖼️ Creating test image...")
        test_image = Image.new('RGB', (640, 480), color='red')
        
        # Test simple query
        print("🔍 Testing simple query...")
        result = model.query(test_image, "What color is this image?")
        print(f"📡 API Response: {result}")
        
        if "answer" in result:
            print(f"✅ API working! Answer: {result['answer']}")
            
            # Test emotion analysis query
            print("\n🎭 Testing emotion analysis query...")
            emotion_result = model.query(test_image, "Rate the confusion and engagement levels from 1-10. Respond with: Confusion: X, Engagement: Y")
            print(f"📡 Emotion Response: {emotion_result}")
            
            if "answer" in emotion_result:
                print(f"✅ Emotion analysis working! Answer: {emotion_result['answer']}")
            else:
                print("❌ No answer in emotion response")
        else:
            print("❌ No 'answer' field in response")
            
    except Exception as e:
        print(f"❌ Moondream API error: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_moondream_api()
