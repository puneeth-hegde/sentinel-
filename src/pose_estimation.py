"""
SENTINEL v5.0 - Pose Estimation & Body Language Analysis
MediaPipe-based threat detection with aggressive posture, hands raised, weapon stances
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
import mediapipe as mp
from logger import get_logger


class PoseEstimationSystem:
    """Advanced body language and threat detection using MediaPipe"""
    
    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics
        self.logger = get_logger("PoseEstimation")
        
        self.enabled = config.pose.enabled
        
        if not self.enabled:
            self.logger.warning("Pose estimation disabled in config")
            return
        
        # Initialize MediaPipe Pose
        self.logger.info("Initializing MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.pose.confidence,
            min_tracking_confidence=config.pose.confidence
        )
        
        # FPS limiting
        self.fps_limit = config.pose.fps_limit
        self.last_process_time = {}  # track_id -> timestamp
        
        # Threat thresholds
        self.hands_raised_threshold = config.pose.hands_raised_threshold
        self.aggressive_threshold = config.pose.aggressive_threshold
        
        # State tracking
        self.pose_states = {}  # track_id -> pose state
        
        self.logger.success("Pose estimation initialized")
    
    def should_process(self, track_id: int) -> bool:
        """Check if we should process pose for this track (FPS limiting)"""
        if not self.enabled:
            return False
        
        current_time = time.time()
        
        if track_id not in self.last_process_time:
            self.last_process_time[track_id] = current_time
            return True
        
        elapsed = current_time - self.last_process_time[track_id]
        interval = 1.0 / self.fps_limit
        
        if elapsed >= interval:
            self.last_process_time[track_id] = current_time
            return True
        
        return False
    
    def process_person(self, track_id: int, frame: np.ndarray, bbox: List[int]) -> Dict:
        """
        Process person for pose estimation and threat detection
        
        Args:
            track_id: Person's tracking ID
            frame: Full frame
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Pose analysis dictionary
        """
        if not self.enabled or not self.should_process(track_id):
            return self._get_cached_state(track_id)
        
        start_time = time.time()
        
        try:
            # Extract person crop
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return self._create_result(track_id, None, "Empty crop")
            
            # Convert to RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb)
            
            if results.pose_landmarks is None:
                return self._create_result(track_id, None, "No pose detected")
            
            # Analyze pose
            analysis = self._analyze_pose(results.pose_landmarks, crop.shape)
            
            # Calculate threat score
            threat_score = self._calculate_threat_score(analysis)
            
            # Create result
            result = self._create_result(
                track_id=track_id,
                landmarks=results.pose_landmarks,
                analysis=analysis,
                threat_score=threat_score
            )
            
            # Cache state
            self.pose_states[track_id] = result
            
            # Record latency
            latency = time.time() - start_time
            self.metrics.record_latency('pose_estimation', latency)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pose processing error: {e}")
            return self._create_result(track_id, None, f"Error: {e}")
    
    def _analyze_pose(self, landmarks, crop_shape) -> Dict:
        """Analyze pose landmarks for body language"""
        h, w = crop_shape[:2]
        
        # Get key landmarks
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        analysis = {
            'hands_raised': False,
            'hands_raised_score': 0.0,
            'aggressive_stance': False,
            'aggressive_score': 0.0,
            'crouching': False,
            'crouching_score': 0.0,
            'arm_extended': False,
            'arm_extended_score': 0.0,
            'body_orientation': 'neutral'
        }
        
        # Check hands raised (surrender or threat)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        left_hand_raised = left_wrist.y < shoulder_y - 0.1
        right_hand_raised = right_wrist.y < shoulder_y - 0.1
        
        if left_hand_raised or right_hand_raised:
            analysis['hands_raised'] = True
            analysis['hands_raised_score'] = 0.5 if (left_hand_raised != right_hand_raised) else 1.0
        
        # Check aggressive stance (wide stance, forward lean)
        hip_width = abs(left_hip.x - right_hip.x)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        
        if hip_width > shoulder_width * 1.2:
            analysis['aggressive_stance'] = True
            analysis['aggressive_score'] = min(1.0, (hip_width / shoulder_width - 1.0))
        
        # Check crouching
        hip_y = (left_hip.y + right_hip.y) / 2
        if hip_y > 0.6:  # Hips in lower portion of frame
            analysis['crouching'] = True
            analysis['crouching_score'] = min(1.0, (hip_y - 0.5) * 2)
        
        # Check arm extended (pointing or weapon)
        left_arm_extended = self._is_arm_extended(left_shoulder, left_elbow, left_wrist)
        right_arm_extended = self._is_arm_extended(right_shoulder, right_elbow, right_wrist)
        
        if left_arm_extended or right_arm_extended:
            analysis['arm_extended'] = True
            analysis['arm_extended_score'] = 1.0 if (left_arm_extended and right_arm_extended) else 0.7
        
        # Body orientation
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        if shoulder_center_x < 0.4:
            analysis['body_orientation'] = 'left'
        elif shoulder_center_x > 0.6:
            analysis['body_orientation'] = 'right'
        else:
            analysis['body_orientation'] = 'center'
        
        return analysis
    
    def _is_arm_extended(self, shoulder, elbow, wrist) -> bool:
        """Check if arm is extended (straight)"""
        # Calculate arm segments
        upper_arm = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
        forearm = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        
        # Calculate angle between segments
        upper_norm = np.linalg.norm(upper_arm)
        fore_norm = np.linalg.norm(forearm)
        
        if upper_norm == 0 or fore_norm == 0:
            return False
        
        dot_product = np.dot(upper_arm, forearm)
        cos_angle = dot_product / (upper_norm * fore_norm)
        
        # If angle is close to 180° (straight), arm is extended
        return cos_angle > 0.5  # ~60° or less bend
    
    def _calculate_threat_score(self, analysis: Dict) -> float:
        """Calculate overall threat score from pose analysis"""
        threat_score = 0.0
        
        # Hands raised (can be surrender OR threat)
        if analysis['hands_raised']:
            threat_score += analysis['hands_raised_score'] * 0.3
        
        # Aggressive stance
        if analysis['aggressive_stance']:
            threat_score += analysis['aggressive_score'] * 0.4
        
        # Crouching (potentially hiding)
        if analysis['crouching']:
            threat_score += analysis['crouching_score'] * 0.2
        
        # Arm extended (pointing weapon)
        if analysis['arm_extended']:
            threat_score += analysis['arm_extended_score'] * 0.5
        
        return min(1.0, threat_score)
    
    def _create_result(
        self,
        track_id: int,
        landmarks,
        analysis: Optional[Dict] = None,
        threat_score: float = 0.0,
        error: Optional[str] = None
    ) -> Dict:
        """Create pose analysis result dictionary"""
        return {
            'track_id': track_id,
            'landmarks': landmarks,
            'analysis': analysis or {},
            'threat_score': threat_score,
            'threat_level': self._get_threat_level(threat_score),
            'error': error,
            'timestamp': time.time()
        }
    
    def _get_threat_level(self, score: float) -> str:
        """Convert threat score to level"""
        if score < 0.3:
            return 'none'
        elif score < 0.6:
            return 'low'
        elif score < 0.8:
            return 'medium'
        else:
            return 'high'
    
    def _get_cached_state(self, track_id: int) -> Dict:
        """Get cached pose state for a track"""
        if track_id in self.pose_states:
            return self.pose_states[track_id]
        return self._create_result(track_id, None, "Not processed yet")
    
    def draw_pose(self, frame: np.ndarray, bbox: List[int], result: Dict) -> np.ndarray:
        """Draw pose landmarks and threat indicators on frame"""
        if result.get('landmarks') is None:
            return frame
        
        x1, y1, x2, y2 = map(int, bbox)
        crop_h, crop_w = y2 - y1, x2 - x1
        
        # Draw landmarks on crop
        landmarks = result['landmarks']
        
        for landmark in landmarks.landmark:
            x = int(x1 + landmark.x * crop_w)
            y = int(y1 + landmark.y * crop_h)
            
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw connections
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start = landmarks.landmark[start_idx]
            end = landmarks.landmark[end_idx]
            
            start_x = int(x1 + start.x * crop_w)
            start_y = int(y1 + start.y * crop_h)
            end_x = int(x1 + end.x * crop_w)
            end_y = int(y1 + end.y * crop_h)
            
            if (0 <= start_x < frame.shape[1] and 0 <= start_y < frame.shape[0] and
                0 <= end_x < frame.shape[1] and 0 <= end_y < frame.shape[0]):
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
        
        # Draw threat indicator
        threat_level = result['threat_level']
        threat_score = result['threat_score']
        
        if threat_level != 'none':
            color = {
                'low': (0, 255, 255),    # Yellow
                'medium': (0, 165, 255),  # Orange
                'high': (0, 0, 255)       # Red
            }.get(threat_level, (255, 255, 255))
            
            # Draw threat bar
            bar_x = x2 + 5
            bar_y = y1
            bar_h = int((y2 - y1) * threat_score)
            
            cv2.rectangle(frame, (bar_x, y2), (bar_x + 10, y2 - bar_h), color, -1)
            cv2.rectangle(frame, (bar_x, y1), (bar_x + 10, y2), color, 2)
            
            # Draw threat label
            label = f"THREAT: {threat_level.upper()}"
            cv2.putText(
                frame, label,
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        
        return frame
    
    def clear_track(self, track_id: int):
        """Clear pose data for inactive track"""
        if track_id in self.pose_states:
            del self.pose_states[track_id]
        if track_id in self.last_process_time:
            del self.last_process_time[track_id]
    
    def cleanup(self):
        """Cleanup resources"""
        if self.enabled and self.pose:
            self.pose.close()
            self.logger.info("Pose estimation cleaned up")
