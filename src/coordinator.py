"""
SENTINEL v5.0 - Main System Coordinator
Orchestrates all components: cameras, detection, recognition, alerts
"""

import cv2
import time
import threading
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


class SentinelCoordinator:
    """Main system coordinator"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        self.config = OmegaConf.load(config_path)
        
        # Initialize logging
        self.logger_system = SentinelLogger(self.config)
        self.logger = get_logger("Coordinator")
        self.logger_system.log_config()
        
        # Initialize metrics
        self.metrics = PerformanceMetrics()
        
        # Initialize components
        self._init_components()
        
        # System state
        self.running = False
        self.display_thread = None
        
        # Frame queues
        self.gate_queue = Queue(maxsize=10)
        self.door_queue = Queue(maxsize=10)
        
        # Detection results
        self.gate_detections = []
        self.door_detections = []
        
        self.logger.success("SENTINEL v5.0 initialized successfully")
    
    def _init_components(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")
        
        # 1. Camera Managers
        self.logger.info("Initializing cameras...")
        self.gate_camera = CameraManager(
            self.config.cameras.gate,
            'gate',
            self.gate_queue,
            self.metrics
        )
        self.door_camera = CameraManager(
            self.config.cameras.door,
            'door',
            self.door_queue,
            self.metrics
        )
        
        # 2. Detection Pipelines
        self.logger.info("Initializing detection pipelines...")
        self.gate_detector = DetectionPipeline(
            self.config,
            'gate',
            self.metrics
        )
        self.door_detector = DetectionPipeline(
            self.config,
            'door',
            self.metrics
        )
        
        # 3. Face Recognition
        self.logger.info("Initializing face recognition...")
        self.face_recognition = FaceRecognitionSystem(
            self.config,
            self.metrics
        )
        
        # 4. Audio Manager
        self.logger.info("Initializing audio system...")
        self.audio = AudioManager(
            self.config,
            self.metrics
        )
        
        # 5. Alert Engine
        self.logger.info("Initializing alert engine...")
        self.alerts = AlertEngine(
            self.config,
            self.audio,
            self.metrics
        )
        
        # 6. Cross-Camera Tracker
        self.logger.info("Initializing cross-camera tracking...")
        self.cross_tracker = CrossCameraTracker(
            self.config,
            self.metrics
        )
        
        self.logger.success("All components initialized")
    
    def _process_gate_frame(self, frame_data: Dict):
        """Process frame from gate camera"""
        frame = frame_data['frame']
        
        # Run detection + tracking
        detections = self.gate_detector.process_frame(frame)
        
        # Update cross-camera tracker
        for det in detections:
            self.cross_tracker.update_gate_track(det['track_id'], det)
        
        # Store detections
        self.gate_detections = detections
    
    def _process_door_frame(self, frame_data: Dict):
        """Process frame from door camera"""
        frame = frame_data['frame']
        
        # Run detection + tracking
        detections = self.door_detector.process_frame(frame)
        
        # Process each detection
        for det in detections:
            track_id = det['track_id']
            
            # Update cross-camera tracker
            self.cross_tracker.update_door_track(track_id, det)
            
            # Face recognition (if needed)
            if self.face_recognition.should_recognize(track_id):
                crop = det.get('crop')
                if crop is not None and crop.size > 0:
                    recognition = self.face_recognition.recognize_face(track_id, crop)
                    det['identity'] = recognition
                    
                    # Update alert engine
                    self.alerts.update_person(track_id, det, recognition)
            else:
                # Get cached recognition state
                state = self.face_recognition.get_recognition_state(track_id)
                if state['confirmed']:
                    det['identity'] = state
                    self.alerts.update_person(track_id, det, state)
        
        # Store detections
        self.door_detections = detections
    
    def _display_loop(self):
        """Display loop - shows camera feeds and overlay info"""
        self.logger.info("Starting display loop...")
        
        # Display window names
        window_name = "SENTINEL v5.0 - Security System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while self.running:
            try:
                # Get latest frames
                gate_frame = None
                door_frame = None
                
                try:
                    gate_data = self.gate_queue.get(timeout=0.1)
                    gate_frame = gate_data['frame'].copy()
                except Empty:
                    pass
                
                try:
                    door_data = self.door_queue.get(timeout=0.1)
                    door_frame = door_data['frame'].copy()
                except Empty:
                    pass
                
                # Process and draw detections
                if gate_frame is not None:
                    gate_display = self.gate_detector.draw_detections(
                        gate_frame, self.gate_detections
                    )
                    gate_display = self._add_overlay(gate_display, 'GATE CAMERA')
                else:
                    gate_display = self._create_blank_frame('GATE CAMERA OFFLINE')
                
                if door_frame is not None:
                    door_display = self.door_detector.draw_detections(
                        door_frame, self.door_detections
                    )
                    door_display = self._add_overlay(door_display, 'DOOR CAMERA')
                else:
                    door_display = self._create_blank_frame('DOOR CAMERA OFFLINE')
                
                # Resize frames
                target_h = self.config.performance.display_height // 2
                target_w = self.config.performance.display_width
                
                gate_display = cv2.resize(gate_display, (target_w, target_h))
                door_display = cv2.resize(door_display, (target_w, target_h))
                
                # Stack vertically
                combined = np.vstack([gate_display, door_display])
                
                # Add system stats
                combined = self._add_system_stats(combined)
                
                # Show display
                cv2.imshow(window_name, combined)
                
                # Record display FPS
                self.metrics.record_frame('display')
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit key pressed")
                    self.stop()
                    break
                elif key == ord('e'):
                    self.logger.info("Enrollment key pressed (not implemented yet)")
                
            except Exception as e:
                self.logger.error(f"Display error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        self.logger.info("Display loop stopped")
    
    def _add_overlay(self, frame: np.ndarray, label: str) -> np.ndarray:
        """Add camera label overlay"""
        import numpy as np
        overlay = frame.copy()
        
        # Add camera label
        cv2.rectangle(overlay, (0, 0), (300, 50), (0, 0, 0), -1)
        cv2.putText(
            overlay, label,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 255), 2
        )
        
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    def _add_system_stats(self, frame: np.ndarray) -> np.ndarray:
        """Add system statistics overlay"""
        import numpy as np
        
        # Update metrics
        self.metrics.update_system_stats()
        stats_text = self.metrics.get_display_text()
        
        # Draw stats box
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
        
        return frame
    
    def _create_blank_frame(self, message: str) -> 'np.ndarray':
        """Create blank frame with message"""
        import numpy as np
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
        self.logger.info("Starting processing loop...")
        
        last_cleanup = time.time()
        
        while self.running:
            # Process gate camera
            try:
                gate_data = self.gate_queue.get(timeout=0.01)
                self._process_gate_frame(gate_data)
            except Empty:
                pass
            
            # Process door camera
            try:
                door_data = self.door_queue.get(timeout=0.01)
                self._process_door_frame(door_data)
            except Empty:
                pass
            
            # Periodic cleanup
            if time.time() - last_cleanup > 10:
                self.cross_tracker.cleanup_old_tracks()
                last_cleanup = time.time()
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
        
        self.logger.info("Processing loop stopped")
    
    def start(self):
        """Start the SENTINEL system"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING SENTINEL v5.0")
        self.logger.info("=" * 80)
        
        # Start cameras
        self.logger.info("Starting cameras...")
        if not self.gate_camera.start():
            self.logger.error("Failed to start gate camera")
            return False
        
        if not self.door_camera.start():
            self.logger.error("Failed to start door camera")
            self.gate_camera.stop()
            return False
        
        # Start processing
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Start display thread
        self.display_thread = threading.Thread(
            target=self._display_loop,
            daemon=False
        )
        self.display_thread.start()
        
        self.logger.success("SENTINEL v5.0 is now running!")
        self.logger.info("Press 'Q' to quit")
        
        return True
    
    def stop(self):
        """Stop the SENTINEL system"""
        self.logger.info("=" * 80)
        self.logger.info("STOPPING SENTINEL v5.0")
        self.logger.info("=" * 80)
        
        self.running = False
        
        # Stop cameras
        self.gate_camera.stop()
        self.door_camera.stop()
        
        # Stop audio
        self.audio.stop()
        
        # Wait for threads
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        # Print final stats
        summary = self.metrics.get_summary()
        self.logger.info("=" * 80)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Uptime: {summary['uptime']:.1f} seconds")
        self.logger.info(f"Total Frames: {summary['counters']['total_frames']}")
        self.logger.info(f"Detections: {summary['counters']['detections']}")
        self.logger.info(f"Recognitions: {summary['counters']['recognitions']}")
        self.logger.info(f"Alerts: {summary['counters']['alerts']}")
        self.logger.info("=" * 80)
        
        self.logger.success("SENTINEL v5.0 stopped successfully")
    
    def run(self):
        """Main entry point - start and wait"""
        if self.start():
            try:
                # Wait for display thread to finish
                if self.display_thread:
                    self.display_thread.join()
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
            finally:
                self.stop()


# Import numpy for frame operations
import numpy as np
