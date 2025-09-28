#!/usr/bin/env python3
"""
Simple webcam test to check if camera is working
"""

import cv2
import time

def test_webcam():
    print("🔍 Testing webcam access...")
    
    # Try different backends and camera indices
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Any Backend")
    ]
    
    for backend, backend_name in backends:
        print(f"\n📹 Testing {backend_name} backend...")
        
        for camera_index in [1, 2, 0]:  # Try OBS Virtual Camera first (index 1)
            print(f"  🔍 Trying camera index {camera_index}...")
            
            cap = cv2.VideoCapture(camera_index, backend)
            
            if cap.isOpened():
                print(f"    ✅ Camera {camera_index} opened successfully")
                
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"    ✅ Successfully read frame: {frame.shape}")
                    
                    # Show frame for 3 seconds
                    print(f"    📺 Showing camera {camera_index} for 3 seconds...")
                    start_time = time.time()
                    
                    while time.time() - start_time < 3:
                        ret, frame = cap.read()
                        if ret:
                            cv2.imshow(f'Camera {camera_index} - {backend_name}', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            print("    ❌ Failed to read frame")
                            break
                    
                    cv2.destroyAllWindows()
                    cap.release()
                    print(f"    ✅ Camera {camera_index} working with {backend_name}")
                    return camera_index, backend
                else:
                    print(f"    ❌ Failed to read frame from camera {camera_index}")
                    cap.release()
            else:
                print(f"    ❌ Failed to open camera {camera_index}")
                if cap:
                    cap.release()
    
    print("\n❌ No working camera found!")
    print("💡 Make sure:")
    print("   - Your webcam is connected")
    print("   - OBS Virtual Camera is running (if using OBS)")
    print("   - No other applications are using the camera")
    return None, None

if __name__ == '__main__':
    camera_index, backend = test_webcam()
    if camera_index is not None:
        print(f"\n🎉 Working camera found: Index {camera_index} with backend {backend}")
    else:
        print("\n😞 No working camera found. Check your setup.")
