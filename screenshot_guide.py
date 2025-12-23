"""
Screenshot Capture Guide - For All 10 Required Screenshots
Automated helper to capture system screenshots for your project report
"""

import cv2
import time
import os
from datetime import datetime

SCREENSHOT_DIR = "screenshots_for_report"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

SCREENSHOTS_REQUIRED = [
    {
        "num": 1,
        "name": "known_person_recognized",
        "description": "Known person recognized at door",
        "demonstrates": "Face recognition working, welcome message",
        "setup": "Stand at door camera, wait for recognition",
        "audio_expected": "Welcome home, puneeth",
        "what_to_show": "Green box with name, 'RECOGNIZED' label"
    },
    {
        "num": 2,
        "name": "unknown_person_at_door",
        "description": "Unknown person at door",
        "demonstrates": "Unknown detection, prompt to identify",
        "setup": "Have friend stand at door camera",
        "audio_expected": "I don't recognize you. Please state your name",
        "what_to_show": "Red box with 'UNKNOWN' label"
    },
    {
        "num": 3,
        "name": "weapon_detection",
        "description": "Unknown person with weapon",
        "demonstrates": "Weapon detection + high threat alert",
        "setup": "Hold knife/gun replica at gate camera",
        "audio_expected": "Warning! Weapon detected!",
        "what_to_show": "Red weapon bounding box + CRITICAL alert"
    },
    {
        "num": 4,
        "name": "person_loitering",
        "description": "Person loitering at gate",
        "demonstrates": "Temporal analysis, loitering alert",
        "setup": "Stand at gate for 25+ seconds",
        "audio_expected": "You have been at the gate for X seconds",
        "what_to_show": "Timer overlay, loitering alert"
    },
    {
        "num": 5,
        "name": "cross_camera_tracking",
        "description": "Cross-camera tracking (gate → door)",
        "demonstrates": "Same person linked across cameras",
        "setup": "Walk from gate to door within 15 seconds",
        "audio_expected": "Welcome home (only once, not twice)",
        "what_to_show": "Same track ID in both views, 'MATCHED' label"
    },
    {
        "num": 6,
        "name": "disappearance_alert",
        "description": "Person at gate but not at door",
        "demonstrates": "Disappearance alert / break-in warning",
        "setup": "Stand at gate for 10s, then move away (not to door)",
        "audio_expected": "Break-in attempt detected!",
        "what_to_show": "CRITICAL alert in logs/display"
    },
    {
        "num": 7,
        "name": "multi_person_handling",
        "description": "Known + unknown person together",
        "demonstrates": "Multi-person handling",
        "setup": "You and friend stand at door together",
        "audio_expected": "Welcome + Unknown alert",
        "what_to_show": "Two bounding boxes, different colors/labels"
    },
    {
        "num": 8,
        "name": "direct_door_approach",
        "description": "Person at door but not at gate",
        "demonstrates": "Direct door approach detection",
        "setup": "Go directly to door without passing gate camera",
        "audio_expected": "Direct approach detected (if implemented)",
        "what_to_show": "Door detection without gate history"
    },
    {
        "num": 9,
        "name": "pose_skeleton_overlay",
        "description": "Pose skeleton overlay",
        "demonstrates": "MediaPipe visualization",
        "setup": "Stand at gate, pose should be visible",
        "audio_expected": "None (visual only)",
        "what_to_show": "Green skeleton overlay on person"
    },
    {
        "num": 10,
        "name": "terminal_log_output",
        "description": "Terminal/log showing alerts",
        "demonstrates": "Text-based alert output",
        "setup": "Capture terminal while system running",
        "audio_expected": "N/A",
        "what_to_show": "Log messages with timestamps and alert types"
    }
]

def print_guide():
    """Print comprehensive screenshot guide"""
    print()
    print("=" * 80)
    print(" " * 20 + "SCREENSHOT CAPTURE GUIDE")
    print("=" * 80)
    print()
    print("This guide will help you capture all 10 required screenshots for your project.")
    print()
    
    for screenshot in SCREENSHOTS_REQUIRED:
        print(f"📸 SCREENSHOT {screenshot['num']}: {screenshot['description'].upper()}")
        print(f"   {'─' * 74}")
        print(f"   What it demonstrates: {screenshot['demonstrates']}")
        print(f"   Setup: {screenshot['setup']}")
        print(f"   Expected audio: {screenshot['audio_expected']}")
        print(f"   What to show: {screenshot['what_to_show']}")
        print(f"   Save as: {SCREENSHOT_DIR}/{screenshot['name']}.png")
        print()
    
    print("=" * 80)
    print()
    print("HOW TO CAPTURE:")
    print()
    print("1. Run your system: python main.py")
    print("2. Follow setup instructions for each screenshot")
    print("3. Press WINDOWS+SHIFT+S to capture")
    print("4. Save to screenshots_for_report/ folder")
    print()
    print("TIPS:")
    print("  • Capture BOTH cameras view and terminal")
    print("  • Make sure text is readable")
    print("  • Show timestamp in logs")
    print("  • Highlight important parts (boxes, labels)")
    print()
    print("=" * 80)
    print()

def create_checklist():
    """Create a checklist file"""
    checklist_path = os.path.join(SCREENSHOT_DIR, "CHECKLIST.txt")
    
    with open(checklist_path, 'w') as f:
        f.write("SCREENSHOT CAPTURE CHECKLIST\n")
        f.write("=" * 60 + "\n\n")
        
        for screenshot in SCREENSHOTS_REQUIRED:
            f.write(f"[ ] Screenshot {screenshot['num']}: {screenshot['description']}\n")
            f.write(f"    File: {screenshot['name']}.png\n")
            f.write(f"    Setup: {screenshot['setup']}\n")
            f.write(f"    Shows: {screenshot['what_to_show']}\n")
            f.write("\n")
        
        f.write("\n")
        f.write("ESSENTIAL (Must have): Screenshots 1-6\n")
        f.write("GOOD TO HAVE: Screenshots 7-10\n")
        f.write("\n")
        f.write("Check off each screenshot as you capture it!\n")
    
    print(f"✅ Checklist created: {checklist_path}")

if __name__ == "__main__":
    print_guide()
    create_checklist()
    
    print()
    print("Ready to start capturing?")
    print()
    print("RECOMMENDED ORDER:")
    print("  1. Start system first (python main.py)")
    print("  2. Test face recognition (#1, #2)")
    print("  3. Test temporal features (#4, #6)")
    print("  4. Test cross-camera (#5)")
    print("  5. Test multi-person (#7)")
    print("  6. Capture terminal logs (#10)")
    print("  7. Capture pose overlay (#9)")
    print("  8. Test weapon (optional) (#3)")
    print()
