"""
SENTINEL v5.0 - COMPLETE Main System Coordinator
★ WITH ALL ADVANCED FEATURES ★
- Body language analysis & threat detection
- Weapon detection (knife, gun)
- User enrollment system
- Loitering & face hidden detection
- Cross-camera tracking
- Audio alerts with priority
"""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Dict, List
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


class SentinelCoordinator:
    """Main system coordinator with COMPLETE feature set"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Initialize logging
        self.logger_system = SentinelLogger(self.config)
        self.logger = get_logger("Coordinator")
        self.logger_system.log_config()
        
        # Initialize metrics
        self.metrics = PerformanceMetrics()
        
        # System state (MUST BE BEFORE _init_components)
        self.running = False
        self.display_thread = None
        self.processing_thread = None
        
        # Frame queues (MUST BE BEFORE _init_components)
        self.gate_queue = Queue(maxsize=10)
        self.door_queue = Queue(maxsize=10)
        
        # Detection results
        self.gate_detections = []
        self.door_detections = []
        
        # Initialize components (NOW queues exist)
        self._init_components()
        
        self.logger.success("★ SENTINEL v5.0 COMPLETE - All features enabled! ★")
    
    def _init_components(self):
        """Initialize ALL system components"""
        self.logger.info("Initializing COMPLETE system...")
        
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
        
        # 5. Pose Estimation & Body Language (NEW!)
        self.logger.info("→ Pose Estimation & Body Language...")
        self.pose_estimation = PoseEstimationSystem(self.config, self.metrics)
        
        # 6. Weapon Detection (NEW!)
        self.logger.info("→ Weapon Detection...")
        self.weapon_detection = WeaponDetectionSystem(
            self.config,
            self.gate_detector.model,  # Share YOLO model
            self.metrics
        )
        
        # 7. Alert Engine
        self.logger.info("→ Alert Engine...")
        self.alerts = AlertEngine(self.config, self.audio, self.metrics)
        
        # 8. Cross-Camera Tracker
        self.logger.info("→ Cross-Camera Tracking...")
        self.cross_tracker = CrossCameraTracker(self.config, self.metrics)
        
        # 8.5. Security Threat Analyzer (REQUIRED!)
        self.logger.info("→ Security Threat Analyzer...")
        from security_threat_analyzer import SecurityThreatAnalyzer
        self.threat_analyzer = SecurityThreatAnalyzer(self.config, self.audio, self.metrics)
        
        # 9. User Enrollment System (NEW!)
        self.logger.info("→ User Enrollment System...")
        self.enrollment = UserEnrollmentSystem(
            self.config, self.face_recognition, self.metrics
        )
        
        self.logger.success("✓ All components initialized!")
    
    def _process_gate_frame(self, frame_data: Dict):
        """Process gate camera with pose & weapon detection"""
        frame = frame_data['frame']
        
        # Run detection + tracking
        detections = self.gate_detector.process_frame(frame)
        
        # Process each detection
        for det in detections:
            track_id = det['track_id']
            
            # Update cross-camera tracker
            self.cross_tracker.update_gate_track(track_id, det)
            
            # Pose estimation & body language analysis
            pose_result = None
            if self.pose_estimation.enabled:
                pose_result = self.pose_estimation.process_person(
                    track_id, frame, det['bbox']
                )
                det['pose'] = pose_result
            
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
            
            # Update alert engine with ALL data
            self.alerts.update_person(
                track_id, det,
                recognition=None,  # Gate doesn't do face recognition
                pose_result=pose_result,
                weapon_result=weapon_result
            )
        
        self.gate_detections = detections
    
    def _process_door_frame(self, frame_data: Dict):
        """Process door camera with face recognition & enrollment"""
        frame = frame_data['frame']
        
        # Run detection + tracking
        detections = self.door_detector.process_frame(frame)
        
        # Process each detection
        for det in detections:
            track_id = det['track_id']
            
            # Update cross-camera tracker
            self.cross_tracker.update_door_track(track_id, det)
            
            # Check if enrolling
            if self.enrollment.is_enrolling():
                enrollment_status = self.enrollment.process_frame(frame, det)
                det['enrollment'] = enrollment_status
            else:
                # Face recognition (if needed)
                if self.face_recognition.should_recognize(track_id):
                    crop = det.get('crop')
                    if crop is not None and crop.size > 0:
                        recognition = self.face_recognition.recognize_face(track_id, crop)
                        det['identity'] = recognition
                        
                        # Update alert engine
                        self.alerts.update_person(track_id, det, recognition)
                        
                        # Mark entered door (for threat analyzer)
                        if recognition and recognition.get('confirmed'):
                            # Try to find matching gate track
                            for gate_track_id in self.threat_analyzer.gate_persons.keys():
                                # Simple approach: mark most recent gate person as entered
                                self.threat_analyzer.mark_entered_door(gate_track_id)
                                break
                else:
                    # Get cached recognition state
                    state = self.face_recognition.get_recognition_state(track_id)
                    if state['confirmed']:
                        det['identity'] = state
                        self.alerts.update_person(track_id, det, state)
        
        self.door_detections = detections
    
    def _display_loop(self):
        """Enhanced display using matplotlib (OpenCV has Windows issues)"""
        self.logger.info("Starting matplotlib display...")
        
        import matplotlib
        matplotlib.use('TkAgg')  # Use Tkinter backend
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.canvas.manager.set_window_title('SENTINEL v5.0 - COMPLETE SYSTEM')
        
        # Keep last valid frames
        last_gate_frame = None
        last_door_frame = None
        
        def update_display(frame_num):
            nonlocal last_gate_frame, last_door_frame
            
            if not self.running:
                plt.close(fig)
                return
            
            # Get frames (non-blocking)
            try:
                if not self.gate_queue.empty():
                    gate_data = self.gate_queue.get_nowait()
                    last_gate_frame = gate_data['frame'].copy()
            except:
                pass
            
            try:
                if not self.door_queue.empty():
                    door_data = self.door_queue.get_nowait()
                    last_door_frame = door_data['frame'].copy()
            except:
                pass
            
            # Update gate display
            ax1.clear()
            if last_gate_frame is not None:
                try:
                    gate_display = self._draw_gate_display(last_gate_frame)
                    # Convert BGR to RGB for matplotlib
                    gate_display = cv2.cvtColor(gate_display, cv2.COLOR_BGR2RGB)
                    ax1.imshow(gate_display)
                    ax1.set_title('GATE CAMERA - Body Language & Weapons', fontsize=12, color='cyan')
                except Exception as e:
                    ax1.text(0.5, 0.5, f'Gate Error: {e}', ha='center', va='center')
            else:
                ax1.text(0.5, 0.5, 'GATE CAMERA STARTING...', ha='center', va='center', fontsize=14)
            ax1.axis('off')
            
            # Update door display
            ax2.clear()
            if last_door_frame is not None:
                try:
                    door_display = self._draw_door_display(last_door_frame)
                    # Convert BGR to RGB for matplotlib
                    door_display = cv2.cvtColor(door_display, cv2.COLOR_BGR2RGB)
                    ax2.imshow(door_display)
                    ax2.set_title('DOOR CAMERA - Face Recognition', fontsize=12, color='cyan')
                except Exception as e:
                    ax2.text(0.5, 0.5, f'Door Error: {e}', ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'DOOR CAMERA STARTING...', ha='center', va='center', fontsize=14)
            ax2.axis('off')
            
            plt.tight_layout()
        
        # Create animation (updates every 33ms = ~30 FPS)
        try:
            anim = FuncAnimation(fig, update_display, interval=33, blit=False)
            
            self.logger.success("Display started! Close window or press Ctrl+C to quit")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Display error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            self.logger.info("Display stopped")
    
    def _draw_gate_display(self, frame: np.ndarray) -> np.ndarray:
        """Draw gate camera with pose & weapon overlays"""
        display = frame.copy()
        
        # Draw detections
        display = self.gate_detector.draw_detections(display, self.gate_detections)
        
        # Draw pose & weapon for each detection
        for det in self.gate_detections:
            # Draw pose skeleton
            if 'pose' in det and det['pose'].get('landmarks'):
                display = self.pose_estimation.draw_pose(
                    display, det['bbox'], det['pose']
                )
            
            # Draw weapon detections
            if 'weapon' in det and det['weapon'].get('current_detection', {}).get('detected'):
                weapons = det['weapon']['current_detection']['weapons']
                confirmed = det['weapon'].get('confirmed', False)
                display = self.weapon_detection.draw_weapon_detections(
                    display, weapons, confirmed
                )
        
        # Add camera label
        display = self._add_overlay(display, 'GATE CAMERA - Body Language & Weapons')
        
        return display
    
    def _draw_door_display(self, frame: np.ndarray) -> np.ndarray:
        """Draw door camera with enrollment UI"""
        display = frame.copy()
        
        # Draw detections
        display = self.door_detector.draw_detections(display, self.door_detections)
        
        # Draw enrollment UI if active
        if self.enrollment.is_enrolling():
            enrollment_status = self.enrollment._get_enrollment_status()
            display = self.enrollment.draw_enrollment_ui(display, enrollment_status)
        
        # Add camera label
        label = 'DOOR CAMERA - Face Recognition'
        if self.enrollment.is_enrolling():
            label += ' [ENROLLING - Press C to cancel]'
        display = self._add_overlay(display, label)
        
        return display
    
    def _handle_enrollment_key(self):
        """Handle enrollment hotkey press"""
        if self.enrollment.is_enrolling():
            self.logger.info("Enrollment already in progress")
            return
        
        # Check if there's an unknown person at door
        unknown_person = None
        for det in self.door_detections:
            identity = det.get('identity', {})
            if identity.get('name') == 'Unknown' or not identity.get('confirmed'):
                unknown_person = det
                break
        
        if unknown_person is None:
            self.logger.warning("No unknown person at door to enroll")
            self.audio.audio.engine.say("No person detected for enrollment")
            self.audio.audio.engine.runAndWait()
            return
        
        # Ask for name
        self.logger.info("=" * 60)
        self.logger.info("ENROLLMENT MODE ACTIVATED")
        self.logger.info("=" * 60)
        
        # Pause system briefly for input
        name = input("Enter name for new user: ").strip()
        
        if name:
            self.enrollment.start_enrollment(name)
        else:
            self.logger.warning("Invalid name, enrollment cancelled")
    
    def _add_feature_status(self, frame: np.ndarray) -> np.ndarray:
        """Add feature status indicators"""
        h, w = frame.shape[:2]
        
        features = [
            ('Face Rec', self.face_recognition.database is not None and len(self.face_recognition.database) > 0),
            ('Pose', self.pose_estimation.enabled),
            ('Weapons', self.weapon_detection.enabled),
            ('Enroll', self.enrollment.enabled),
            ('Cross-Cam', self.cross_tracker.enabled),
        ]
        
        y_pos = h - 30
        x_pos = w - 400
        
        for name, enabled in features:
            color = (0, 255, 0) if enabled else (100, 100, 100)
            status = "✓" if enabled else "✗"
            text = f"{status} {name}"
            
            cv2.putText(
                frame, text,
                (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
            x_pos += 80
        
        return frame
    
    def _add_overlay(self, frame: np.ndarray, label: str) -> np.ndarray:
        """Add camera label overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (len(label) * 12, 50), (0, 0, 0), -1)
        cv2.putText(
            overlay, label,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    def _add_system_stats(self, frame: np.ndarray) -> np.ndarray:
        """Add system statistics"""
        self.metrics.update_system_stats()
        stats_text = self.metrics.get_display_text()
        
        lines = stats_text.split('\n')
        y_offset = 30
        
        for line in lines:
            cv2.putText(
                frame, line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
            y_offset += 25
        
        # Add weapon count if any
        weapon_count = self.weapon_detection.get_active_weapons()
        if weapon_count > 0:
            cv2.putText(
                frame, f"⚠ WEAPONS: {weapon_count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )
        
        return frame
    
    def _create_blank_frame(self, message: str) -> np.ndarray:
        """Create blank frame"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame, message,
            (100, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 255), 2
        )
        return frame
    
    def _processing_loop(self):
        """Main processing loop"""
        self.logger.info("Processing loop started")
        
        last_cleanup = time.time()
        
        while self.running:
            try:
                gate_data = self.gate_queue.get(timeout=0.01)
                self._process_gate_frame(gate_data)
            except Empty:
                pass
            
            try:
                door_data = self.door_queue.get(timeout=0.01)
                self._process_door_frame(door_data)
            except Empty:
                pass
            
            if time.time() - last_cleanup > 10:
                self.cross_tracker.cleanup_old_tracks()
                last_cleanup = time.time()
            
            time.sleep(0.001)
        
        self.logger.info("Processing loop stopped")
    
    def start(self):
        """Start COMPLETE system"""
        self.logger.info("=" * 80)
        self.logger.success("★ STARTING SENTINEL v5.0 COMPLETE SYSTEM ★")
        self.logger.info("=" * 80)
        self.logger.info("Features enabled:")
        self.logger.info("  ✓ Face Recognition")
        self.logger.info("  ✓ Cross-Camera Tracking")
        self.logger.info(f"  {'✓' if self.pose_estimation.enabled else '✗'} Body Language Analysis")
        self.logger.info(f"  {'✓' if self.weapon_detection.enabled else '✗'} Weapon Detection")
        self.logger.info(f"  {'✓' if self.enrollment.enabled else '✗'} User Enrollment")
        self.logger.info("=" * 80)
        
        if not self.gate_camera.start() or not self.door_camera.start():
            self.logger.error("Failed to start cameras")
            return False
        
        self.running = True
        
        # CRITICAL: Processing in background thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()
        
        # CRITICAL: Display runs in MAIN thread (not daemon!)
        # OpenCV GUI MUST be in main thread on Windows
        self.logger.success("★ SENTINEL v5.0 COMPLETE - RUNNING! ★")
        self.logger.info("Controls:")
        self.logger.info("  Q - Quit")
        self.logger.info("  E - Enroll new user")
        self.logger.info("  C - Cancel enrollment")
        
        return True
    
    def stop(self):
        """Stop system"""
        self.logger.info("Stopping SENTINEL v5.0 COMPLETE...")
        
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
        self.logger.info(f"Detections: {summary['counters']['detections']}")
        self.logger.info(f"Recognitions: {summary['counters']['recognitions']}")
        self.logger.info(f"Alerts: {summary['counters']['alerts']}")
        self.logger.info("=" * 80)
        
        self.logger.success("SENTINEL v5.0 COMPLETE stopped successfully")
    
    def run(self):
        """Main entry point - runs display in main thread"""
        if self.start():
            try:
                # CRITICAL: Run display in main thread (OpenCV requirement)
                self._display_loop()
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt")
            finally:
                self.stop()