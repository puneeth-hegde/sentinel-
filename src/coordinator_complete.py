"""
SENTINEL v5.0 - COMPLETE FIXED SYSTEM
All critical bugs fixed, optimized performance, intelligent behavior
"""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Dict, List, Optional
from omegaconf import OmegaConf

from logger import SentinelLogger, get_logger
from metrics import PerformanceMetrics
from camera_manager import CameraManager
from detection_pipeline import DetectionPipeline
from face_recognition import FaceRecognitionSystem
from audio_manager import AudioManager
from alert_engine import AlertEngine
from cross_camera_tracker import CrossCameraTracker
from pose_estimation import PoseEstimationSystem
from weapon_detection import WeaponDetectionSystem
from user_enrollment import UserEnrollmentSystem
from security_threat_analyzer import SecurityThreatAnalyzer


class PersonSession:
    """Track person's session across cameras with persistence"""
    def __init__(self, track_id: int, name: str, camera: str):
        self.track_id = track_id
        self.name = name
        self.camera = camera
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.status = "active"
        self.alerted = False
        self.confirmed_count = 0
        
    def update(self):
        self.last_seen = time.time()
        self.confirmed_count += 1
    
    def age(self):
        return time.time() - self.last_seen
    
    def is_expired(self, timeout=60):
        return self.age() > timeout


class SentinelCoordinator:
    """Main system coordinator - FULLY FIXED VERSION"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Initialize logging
        self.logger_system = SentinelLogger(self.config)
        self.logger = get_logger("Coordinator")
        self.logger_system.log_config()
        
        # Initialize metrics
        self.metrics = PerformanceMetrics()
        
        # System state
        self.running = False
        self.processing_thread = None
        
        # Frame queues
        self.gate_queue = Queue(maxsize=10)
        self.door_queue = Queue(maxsize=10)
        
        # Detection results
        self.gate_detections = []
        self.door_detections = []
        
        # Session management (CRITICAL FIX)
        self.active_sessions = {}  # track_id -> PersonSession
        self.session_lock = threading.Lock()
        
        # Frame skipping for performance
        self.frame_skip = 2  # Process every 2nd frame
        self.gate_frame_count = 0
        self.door_frame_count = 0
        
        # Initialize components
        self._init_components()
        
        self.logger.success("★ SENTINEL v5.0 FIXED - All optimizations enabled! ★")
    
    def _init_components(self):
        """Initialize ALL system components"""
        self.logger.info("Initializing FIXED system...")
        
        # 1. Camera Managers
        self.logger.info("→ Cameras...")
        self.gate_camera = CameraManager(
            self.config.cameras.gate, 'gate', self.gate_queue, self.metrics
        )
        self.door_camera = CameraManager(
            self.config.cameras.door, 'door', self.door_queue, self.metrics
        )
        
        # 2. Detection Pipelines
        self.logger.info("→ Detection + Tracking...")
        self.gate_detector = DetectionPipeline(self.config, 'gate', self.metrics)
        self.door_detector = DetectionPipeline(self.config, 'door', self.metrics)
        
        # 3. Face Recognition
        self.logger.info("→ Face Recognition...")
        self.face_recognition = FaceRecognitionSystem(self.config, self.metrics)
        
        # 4. Audio Manager
        self.logger.info("→ Audio System...")
        self.audio = AudioManager(self.config, self.metrics)
        
        # 5. Pose Estimation
        self.logger.info("→ Pose Estimation...")
        self.pose_estimation = PoseEstimationSystem(self.config, self.metrics)
        
        # 6. Weapon Detection
        self.logger.info("→ Weapon Detection...")
        self.weapon_detection = WeaponDetectionSystem(
            self.config,
            self.gate_detector.model,
            self.metrics
        )
        
        # 7. Alert Engine
        self.logger.info("→ Alert Engine...")
        self.alerts = AlertEngine(self.config, self.audio, self.metrics)
        
        # 8. Cross-Camera Tracker
        self.logger.info("→ Cross-Camera Tracking...")
        self.cross_tracker = CrossCameraTracker(self.config, self.metrics)
        
        # 9. Security Threat Analyzer
        self.logger.info("→ Security Threat Analyzer...")
        self.threat_analyzer = SecurityThreatAnalyzer(self.config, self.audio, self.metrics)
        
        # 10. User Enrollment
        self.logger.info("→ User Enrollment System...")
        self.enrollment = UserEnrollmentSystem(
            self.config, self.face_recognition, self.metrics
        )
        
        self.logger.success("✓ All components initialized!")
    
    def _get_or_create_session(self, track_id: int, name: str, camera: str) -> PersonSession:
        """Get existing session or create new one"""
        with self.session_lock:
            if track_id not in self.active_sessions:
                self.active_sessions[track_id] = PersonSession(track_id, name, camera)
                self.logger.debug(f"Created new session: Track {track_id} = {name}")
            else:
                self.active_sessions[track_id].update()
            
            return self.active_sessions[track_id]
    
    def _cleanup_expired_sessions(self):
        """Remove old sessions"""
        with self.session_lock:
            expired = [
                tid for tid, session in self.active_sessions.items()
                if session.is_expired(timeout=60)
            ]
            for tid in expired:
                self.logger.debug(f"Session expired: Track {tid}")
                del self.active_sessions[tid]
    
    def _find_cross_camera_session(self, current_track_id: int, camera: str) -> Optional[PersonSession]:
        """Find if this person was seen on other camera recently"""
        with self.session_lock:
            # Look for recent sessions from other camera
            other_camera = 'door' if camera == 'gate' else 'gate'
            
            for session in self.active_sessions.values():
                if session.camera == other_camera and session.age() < 30:
                    # Found recent person from other camera
                    return session
        
        return None
    
    def _process_gate_frame(self, frame_data: Dict):
        """Process gate camera with optimizations"""
        frame = frame_data['frame']
        
        # Frame skipping for performance
        self.gate_frame_count += 1
        if self.gate_frame_count % self.frame_skip != 0:
            return
        
        # Run detection + tracking
        detections = self.gate_detector.process_frame(frame)
        
        # Process each detection
        for det in detections:
            track_id = det['track_id']
            
            # Update cross-camera tracker
            self.cross_tracker.update_gate_track(track_id, det)
            
            # Update threat analyzer
            self.threat_analyzer.update_gate_person(track_id, det)
            
            # Pose estimation (every 3rd detection for performance)
            pose_result = None
            if self.pose_estimation.enabled and track_id % 3 == 0:
                pose_result = self.pose_estimation.process_person(
                    track_id, frame, det['bbox']
                )
                det['pose'] = pose_result
                
                if pose_result:
                    self.threat_analyzer.add_pose_threat(track_id, pose_result)
            
            # Weapon detection
            weapon_result = None
            if self.weapon_detection.enabled:
                weapon_detect = self.weapon_detection.detect_weapons(
                    frame, det['bbox']
                )
                weapon_result = self.weapon_detection.update_person_weapons(
                    track_id, weapon_detect
                )
                det['weapon'] = weapon_result
            
            # Update alert engine
            self.alerts.update_person(
                track_id, det,
                recognition=None,
                pose_result=pose_result,
                weapon_result=weapon_result
            )
        
        # Check for disappearances and group threats
        active_ids = set([d['track_id'] for d in detections])
        self.threat_analyzer.check_disappearance(active_ids)
        self.threat_analyzer.check_group_threat(list(active_ids))
        
        self.gate_detections = detections
    
    def _process_door_frame(self, frame_data: Dict):
        """Process door camera with session management"""
        frame = frame_data['frame']
        
        # Frame skipping
        self.door_frame_count += 1
        if self.door_frame_count % self.frame_skip != 0:
            return
        
        # Run detection + tracking
        detections = self.door_detector.process_frame(frame)
        
        # Process each detection
        for det in detections:
            track_id = det['track_id']
            
            # Update cross-camera tracker
            self.cross_tracker.update_door_track(track_id, det)
            
            # Check enrollment mode
            if self.enrollment.is_enrolling():
                enrollment_status = self.enrollment.process_frame(frame, det)
                det['enrollment'] = enrollment_status
                continue
            
            # Check if we have a session from gate camera
            gate_session = self._find_cross_camera_session(track_id, 'door')
            
            if gate_session and gate_session.confirmed_count >= 2:
                # Person was already recognized at gate!
                det['identity'] = {
                    'name': gate_session.name,
                    'confidence': 0.95,
                    'confirmed': True,
                    'source': 'cross_camera'
                }
                
                # Create door session
                door_session = self._get_or_create_session(track_id, gate_session.name, 'door')
                
                # Alert only once per session
                if not door_session.alerted:
                    self.alerts.update_person(track_id, det, det['identity'])
                    door_session.alerted = True
                    self.logger.info(f"✓ Cross-camera recognition: {gate_session.name}")
                
                # Mark as entered
                self.threat_analyzer.mark_entered_door(gate_session.track_id)
                
            else:
                # Normal face recognition
                if self.face_recognition.should_recognize(track_id):
                    crop = det.get('crop')
                    if crop is not None and crop.size > 0:
                        recognition = self.face_recognition.recognize_face(track_id, crop)
                        det['identity'] = recognition
                        
                        if recognition and recognition.get('confirmed'):
                            name = recognition.get('name', 'Unknown')
                            
                            # Create or get session
                            session = self._get_or_create_session(track_id, name, 'door')
                            
                            # Alert only once per session
                            if not session.alerted or session.confirmed_count == 1:
                                self.alerts.update_person(track_id, det, recognition)
                                session.alerted = True
                            
                            # Mark as entered for threat analyzer
                            self.threat_analyzer.mark_entered_door(track_id)
                
                else:
                    # Use cached recognition
                    state = self.face_recognition.get_recognition_state(track_id)
                    if state['confirmed']:
                        det['identity'] = state
                        
                        # Update session
                        name = state.get('name', 'Unknown')
                        session = self._get_or_create_session(track_id, name, 'door')
        
        self.door_detections = detections
    
    def _display_loop(self):
        """Optimized OpenCV display with workaround"""
        self.logger.info("Starting optimized display...")
        
        # CRITICAL: Start window thread first
        cv2.startWindowThread()
        
        window_name = "SENTINEL v5.0 - OPTIMIZED"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        last_gate = None
        last_door = None
        
        self.logger.success("✓ Display window ready!")
        
        while self.running:
            try:
                # Get latest frames (drain queue)
                while not self.gate_queue.empty():
                    try:
                        data = self.gate_queue.get_nowait()
                        last_gate = data['frame'].copy()
                    except:
                        break
                
                while not self.door_queue.empty():
                    try:
                        data = self.door_queue.get_nowait()
                        last_door = data['frame'].copy()
                    except:
                        break
                
                # Create display
                if last_gate is not None and last_door is not None:
                    # Draw gate display
                    gate_display = self._draw_gate_display(last_gate)
                    
                    # Draw door display
                    door_display = self._draw_door_display(last_door)
                    
                    # Resize and combine
                    target_w = 1280
                    target_h = 360
                    
                    gate_resized = cv2.resize(gate_display, (target_w, target_h))
                    door_resized = cv2.resize(door_display, (target_w, target_h))
                    
                    combined = np.vstack([gate_resized, door_resized])
                    
                    # Add stats
                    combined = self._add_stats_overlay(combined)
                    
                    cv2.imshow(window_name, combined)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                    break
                elif key == ord('e'):
                    self._handle_enrollment()
                elif key == ord('c'):
                    if self.enrollment.is_enrolling():
                        self.enrollment.cancel_enrollment()
                
                time.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Display error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        for i in range(5):
            cv2.waitKey(1)
        
        self.logger.info("Display stopped")
    
    def _draw_gate_display(self, frame: np.ndarray) -> np.ndarray:
        """Draw gate camera"""
        display = self.gate_detector.draw_detections(frame.copy(), self.gate_detections)
        
        # Draw pose & weapons
        for det in self.gate_detections:
            if 'pose' in det and det['pose'].get('landmarks'):
                display = self.pose_estimation.draw_pose(
                    display, det['bbox'], det['pose']
                )
            
            if 'weapon' in det and det['weapon'].get('current_detection', {}).get('detected'):
                weapons = det['weapon']['current_detection']['weapons']
                confirmed = det['weapon'].get('confirmed', False)
                display = self.weapon_detection.draw_weapon_detections(
                    display, weapons, confirmed
                )
        
        # Label
        cv2.rectangle(display, (0, 0), (400, 40), (0, 0, 0), -1)
        cv2.putText(display, "GATE CAMERA", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return display
    
    def _draw_door_display(self, frame: np.ndarray) -> np.ndarray:
        """Draw door camera"""
        display = self.door_detector.draw_detections(frame.copy(), self.door_detections)
        
        # Draw enrollment UI
        if self.enrollment.is_enrolling():
            status = self.enrollment._get_enrollment_status()
            display = self.enrollment.draw_enrollment_ui(display, status)
        
        # Label
        cv2.rectangle(display, (0, 0), (400, 40), (0, 0, 0), -1)
        cv2.putText(display, "DOOR CAMERA", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return display
    
    def _add_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add system stats"""
        h, w = frame.shape[:2]
        
        # FPS
        fps = self.metrics.get_fps('gate')
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Active sessions
        with self.session_lock:
            session_count = len(self.active_sessions)
        
        session_text = f"Sessions: {session_count}"
        cv2.putText(frame, session_text, (w - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Feature indicators
        features = [
            ('Face', len(self.face_recognition.database) > 0 if self.face_recognition.database else False),
            ('Pose', self.pose_estimation.enabled),
            ('Weapon', self.weapon_detection.enabled),
        ]
        
        x_pos = w - 300
        y_pos = h - 20
        
        for name, enabled in features:
            color = (0, 255, 0) if enabled else (100, 100, 100)
            status = "✓" if enabled else "✗"
            text = f"{status}{name}"
            cv2.putText(frame, text, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            x_pos += 80
        
        return frame
    
    def _handle_enrollment(self):
        """Handle enrollment key press"""
        if self.enrollment.is_enrolling():
            return
        
        # Check for unknown person at door
        for det in self.door_detections:
            identity = det.get('identity', {})
            if identity.get('name') == 'Unknown' or not identity.get('confirmed'):
                name = input("\nEnter name for new user: ").strip()
                if name:
                    self.enrollment.start_enrollment(name)
                return
        
        self.logger.warning("No unknown person at door")
    
    def _processing_loop(self):
        """Main processing loop"""
        self.logger.info("Processing loop started")
        
        last_cleanup = time.time()
        
        while self.running:
            try:
                # Process gate frames
                try:
                    gate_data = self.gate_queue.get(timeout=0.01)
                    self._process_gate_frame(gate_data)
                except Empty:
                    pass
                
                # Process door frames
                try:
                    door_data = self.door_queue.get(timeout=0.01)
                    self._process_door_frame(door_data)
                except Empty:
                    pass
                
                # Periodic cleanup
                if time.time() - last_cleanup > 10:
                    self.cross_tracker.cleanup_old_tracks()
                    self.threat_analyzer.clear_old_data()
                    self._cleanup_expired_sessions()
                    last_cleanup = time.time()
                
                time.sleep(0.001)
            
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                time.sleep(0.1)
        
        self.logger.info("Processing loop stopped")
    
    def start(self):
        """Start system"""
        self.logger.info("=" * 80)
        self.logger.success("★ STARTING SENTINEL v5.0 FIXED SYSTEM ★")
        self.logger.info("=" * 80)
        self.logger.info("Optimizations:")
        self.logger.info("  ✓ Session-based recognition (no spam)")
        self.logger.info("  ✓ Cross-camera identity linking")
        self.logger.info("  ✓ Frame skipping (2x performance)")
        self.logger.info("  ✓ Recognition persistence (30s)")
        self.logger.info("  ✓ OpenCV display optimization")
        self.logger.info("=" * 80)
        
        if not self.gate_camera.start() or not self.door_camera.start():
            self.logger.error("Failed to start cameras")
            return False
        
        self.running = True
        
        # Processing in background
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()
        
        self.logger.success("★ SYSTEM RUNNING! ★")
        self.logger.info("Controls: Q=Quit, E=Enroll, C=Cancel")
        
        return True
    
    def stop(self):
        """Stop system"""
        self.logger.info("Stopping system...")
        
        self.running = False
        
        self.gate_camera.stop()
        self.door_camera.stop()
        self.audio.stop()
        
        if self.pose_estimation.enabled:
            self.pose_estimation.cleanup()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        summary = self.metrics.get_summary()
        self.logger.info("=" * 80)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Uptime: {summary['uptime']:.1f}s")
        self.logger.info(f"Total Frames: {summary['counters']['total_frames']}")
        self.logger.info(f"Recognitions: {summary['counters']['recognitions']}")
        self.logger.info("=" * 80)
        
        self.logger.success("System stopped successfully")
    
    def run(self):
        """Main entry point"""
        if self.start():
            try:
                # Display runs in main thread
                self._display_loop()
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt")
            finally:
                self.stop()
