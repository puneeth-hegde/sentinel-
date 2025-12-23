"""
Face Capture Script - Optimized for TP-Link Tapo C310
Captures high-quality face images directly from your door camera
"""

import cv2
import os
import time
from datetime import datetime

# Configuration
RTSP_URL = "rtsp://Door_Camera:CREAKmyPASSWORD1219!!!@192.168.1.6:554/stream1"
OUTPUT_DIR = "dataset/puneeth"
TARGET_IMAGES = 40
NAME = "puneeth"

def capture_faces():
    """Capture face images from Tapo camera"""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print()
    print("=" * 60)
    print("  FACE CAPTURE - Tapo C310 Door Camera")
    print("=" * 60)
    print()
    print(f"📹 Connecting to door camera...")
    
    # Connect to camera
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot connect to camera!")
        print("Check:")
        print("  1. Camera is powered on")
        print("  2. Network connection")
        print("  3. RTSP URL is correct")
        return
    
    print(f"✅ Connected to Tapo C310")
    print()
    print("INSTRUCTIONS:")
    print("  • Stand 2-3 feet from door camera")
    print("  • Look at camera (green circle)")
    print("  • Press SPACE to capture")
    print("  • Vary angles and expressions")
    print("  • Press Q to quit when done")
    print()
    print(f"TARGET: {TARGET_IMAGES} images")
    print("=" * 60)
    print()
    
    # Capture counter
    captured = 0
    
    # Face detector (for guidance circle)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    while captured < TARGET_IMAGES:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Lost connection to camera")
            break
        
        # Resize for display (Tapo is 2304×1296, too big for screen)
        display_frame = cv2.resize(frame, (1280, 720))
        
        # Detect face for guidance
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw guidance
        h, w = display_frame.shape[:2]
        
        # Center circle for face positioning
        center_x, center_y = w // 2, h // 2
        cv2.circle(display_frame, (center_x, center_y), 150, (0, 255, 0), 3)
        cv2.putText(display_frame, "Position face here", 
                   (center_x - 100, center_y + 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show detected faces
        for (x, y, w_face, h_face) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 2)
        
        # Progress bar
        progress = int((captured / TARGET_IMAGES) * 100)
        cv2.rectangle(display_frame, (20, h - 50), (20 + progress * 10, h - 20), (0, 255, 0), -1)
        
        # Status text
        status_text = f"Captured: {captured}/{TARGET_IMAGES} | Press SPACE to capture, Q to quit"
        cv2.putText(display_frame, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Face Capture - Tapo C310", display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE - capture
            timestamp = int(time.time() * 1000)
            filename = f"{OUTPUT_DIR}/{NAME}_{timestamp}_{captured+1:03d}.jpg"
            
            # Save FULL RESOLUTION image (not display size)
            cv2.imwrite(filename, frame)
            
            captured += 1
            
            print(f"✅ [{captured}/{TARGET_IMAGES}] Captured: {os.path.basename(filename)}")
            
            # Flash effect
            flash = display_frame.copy()
            flash[:] = (255, 255, 255)
            cv2.imshow("Face Capture - Tapo C310", flash)
            cv2.waitKey(100)
        
        elif key == ord('q'):  # Q - quit
            print()
            print(f"⚠️  Quit early - captured {captured} images")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("=" * 60)
    print(f"✅ CAPTURE COMPLETE!")
    print(f"   Total images: {captured}")
    print(f"   Location: {OUTPUT_DIR}/")
    print()
    print("NEXT STEP:")
    print("   Run: python train_faces.py")
    print("=" * 60)
    print()

if __name__ == "__main__":
    try:
        capture_faces()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
