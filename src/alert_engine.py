"""
SENTINEL v5.0 - Alert Engine
Rule-based alert system for security events
NOW WITH: Weapon detection alerts, Body language threats, Complete event management
"""

import time
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from logger import get_logger


class AlertEngine:
    """Manages security alerts, weapon detection, and threat assessment"""
    
    def __init__(self, config, audio_manager, metrics):
        self.config = config
        self.audio = audio_manager
        self.metrics = metrics
        self.logger = get_logger("AlertEngine")
        
        # Alert tracking
        self.active_alerts = {}  # track_id -> alert state
        self.triggered_alerts = {}  # track_id -> set of triggered alert types
        
        # Snapshot management
        self.snapshot_dir = Path(config.system.snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Alert configurations
        self.face_hidden_duration = config.alerts.face_hidden.duration
        self.loitering_duration = config.alerts.loitering.duration
        
        self.logger.success("Alert engine initialized")
    
    def update_person(
        self,
        track_id: int,
        detection: Dict,
        recognition: Optional[Dict] = None,
        pose_result: Optional[Dict] = None,
        weapon_result: Optional[Dict] = None
    ):
        """
        Update alert state for a person (COMPLETE VERSION)
        
        Args:
            track_id: Person's tracking ID
            detection: Detection dictionary
            recognition: Recognition result (if available)
            pose_result: Pose/body language analysis (if available)
            weapon_result: Weapon detection result (if available)
        """
        camera = detection['camera']
        
        # Initialize alert state if new
        if track_id not in self.active_alerts:
            self.active_alerts[track_id] = {
                'first_seen': time.time(),
                'camera': camera,
                'face_hidden_start': None,
                'loitering_start': None,
                'recognized': False,
                'name': None,
                'threat_level': 'none',
                'weapon_detected': False
            }
            self.triggered_alerts[track_id] = set()
        
        state = self.active_alerts[track_id]
        
        # Update recognition state
        if recognition and recognition.get('confirmed'):
            state['recognized'] = True
            state['name'] = recognition.get('name')
            
            # Trigger welcome or unknown alert
            self._check_recognition_alert(track_id, state)
        
        # Check weapon detection (CRITICAL PRIORITY)
        if weapon_result and weapon_result.get('confirmed'):
            if not state['weapon_detected']:
                state['weapon_detected'] = True
                self._trigger_weapon_alert(track_id, state, weapon_result)
        
        # Check threat from pose/body language
        if pose_result and pose_result.get('threat_level') != 'none':
            threat_level = pose_result['threat_level']
            if threat_level != state['threat_level']:
                state['threat_level'] = threat_level
                if threat_level in ['medium', 'high']:
                    self._trigger_threat_alert(track_id, state, pose_result)
        
        # Check face hidden (door camera only)
        if camera == 'door' and self.config.alerts.face_hidden.enabled:
            self._check_face_hidden(track_id, state, detection)
        
        # Check loitering (gate camera only)
        if camera == 'gate' and self.config.alerts.loitering.enabled:
            self._check_loitering(track_id, state, detection)
    
    def _trigger_weapon_alert(self, track_id: int, state: Dict, weapon_result: Dict):
        """CRITICAL: Weapon detected alert"""
        if 'weapon' not in self.triggered_alerts[track_id]:
            weapon_type = weapon_result.get('weapon_type', 'weapon')
            
            self.logger.critical("=" * 80)
            self.logger.critical(f"⚠️  WEAPON DETECTED - Track {track_id}")
            self.logger.critical(f"   Type: {weapon_type.upper()}")
            self.logger.critical(f"   Detections: {weapon_result.get('detections_count', 0)}")
            self.logger.critical("=" * 80)
            
            # Play critical audio alert
            self.audio.play_weapon()
            
            self.triggered_alerts[track_id].add('weapon')
            self.metrics.increment_counter('alerts')
            
            # Save snapshot immediately
            detection = {'camera': state['camera'], 'frame': None, 'bbox': [0, 0, 0, 0]}
            self._save_snapshot(track_id, detection, f'WEAPON_{weapon_type}')
    
    def _trigger_threat_alert(self, track_id: int, state: Dict, pose_result: Dict):
        """Threat detected from body language"""
        threat_level = pose_result['threat_level']
        
        if f'threat_{threat_level}' not in self.triggered_alerts[track_id]:
            analysis = pose_result.get('analysis', {})
            
            threat_indicators = []
            if analysis.get('hands_raised'):
                threat_indicators.append("hands raised")
            if analysis.get('aggressive_stance'):
                threat_indicators.append("aggressive stance")
            if analysis.get('arm_extended'):
                threat_indicators.append("arm extended")
            
            self.logger.warning(
                f"Track {track_id}: {threat_level.upper()} threat "
                f"({', '.join(threat_indicators)})"
            )
            
            # Play threat audio for medium/high only
            if threat_level in ['medium', 'high']:
                self.audio.play_threat()
            
            self.triggered_alerts[track_id].add(f'threat_{threat_level}')
            self.metrics.increment_counter('alerts')

    
    def _check_recognition_alert(self, track_id: int, state: Dict):
        """Check if we should trigger recognition-based alert"""
        alert_type = 'recognized' if state['name'] != 'Unknown' else 'unknown'
        
        # Only trigger once per person
        if alert_type in self.triggered_alerts[track_id]:
            return
        
        if state['name'] and state['name'] != 'Unknown':
            # Authorized user
            self.logger.info(f"Track {track_id}: Authorized user '{state['name']}'")
            self.audio.play_welcome(state['name'])
            self.triggered_alerts[track_id].add('recognized')
            
        elif state['recognized']:
            # Unknown person
            self.logger.warning(f"Track {track_id}: Unknown person detected")
            self.audio.play_unknown()
            self.triggered_alerts[track_id].add('unknown')
            self.metrics.increment_counter('alerts')
    
    def _check_face_hidden(self, track_id: int, state: Dict, detection: Dict):
        """Check for face hidden condition"""
        # Simple heuristic: if person is at door but not recognized after threshold
        has_face = detection.get('crop') is not None
        
        if not has_face or not state['recognized']:
            # Face potentially hidden
            if state['face_hidden_start'] is None:
                state['face_hidden_start'] = time.time()
            else:
                duration = time.time() - state['face_hidden_start']
                
                if duration >= self.face_hidden_duration:
                    # Trigger alert
                    if 'face_hidden' not in self.triggered_alerts[track_id]:
                        self.logger.warning(
                            f"Track {track_id}: Face hidden for {duration:.0f}s"
                        )
                        self.audio.play_face_hidden()
                        self.triggered_alerts[track_id].add('face_hidden')
                        self.metrics.increment_counter('alerts')
                        
                        # Save snapshot
                        self._save_snapshot(track_id, detection, 'face_hidden')
        else:
            # Face visible
            state['face_hidden_start'] = None
    
    def _check_loitering(self, track_id: int, state: Dict, detection: Dict):
        """Check for loitering at gate"""
        duration = time.time() - state['first_seen']
        
        if duration >= self.loitering_duration:
            # Trigger alert
            if 'loitering' not in self.triggered_alerts[track_id]:
                self.logger.warning(
                    f"Track {track_id}: Loitering at gate for {duration:.0f}s"
                )
                self.audio.play_loitering(int(duration))
                self.triggered_alerts[track_id].add('loitering')
                self.metrics.increment_counter('alerts')
                
                # Save snapshot
                self._save_snapshot(track_id, detection, 'loitering')
    
    def _save_snapshot(self, track_id: int, detection: Dict, reason: str):
        """Save snapshot of detected event"""
        if not self.config.snapshots.enabled:
            return
        
        try:
            # Create directory structure
            date_str = datetime.now().strftime('%Y%m%d')
            camera_dir = self.snapshot_dir / detection['camera'] / date_str
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"{reason}_{track_id}_{timestamp}.jpg"
            filepath = camera_dir / filename
            
            # Save frame
            frame = detection.get('frame')
            if frame is not None:
                # Draw bounding box
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f"{reason.upper()} - Track {track_id}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
                
                # Save
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.logger.info(f"Snapshot saved: {filepath}")
        
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")
    
    def remove_person(self, track_id: int):
        """Clean up tracking for person who left"""
        if track_id in self.active_alerts:
            del self.active_alerts[track_id]
        if track_id in self.triggered_alerts:
            del self.triggered_alerts[track_id]
        
        self.logger.debug(f"Cleared alert state for track {track_id}")
    
    def get_alert_state(self, track_id: int) -> Optional[Dict]:
        """Get current alert state for a person"""
        return self.active_alerts.get(track_id)
    
    def get_active_alerts_count(self) -> int:
        """Get number of active alerts"""
        return len(self.active_alerts)
