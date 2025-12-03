"""
SENTINEL v5.0 - Advanced Security Threat Detection
Handles break-in attempts, suspicious behavior, and rapid movements
"""

import time
import numpy as np
from typing import Dict, List, Set
from logger import get_logger

class SecurityThreatAnalyzer:
    """
    Advanced threat detection for break-in scenarios
    Detects: suspicious loitering, rapid disappearance, group behavior, escape attempts
    """
    
    def __init__(self, config, audio_manager, metrics):
        self.config = config
        self.audio = audio_manager
        self.metrics = metrics
        self.logger = get_logger("SecurityThreat")
        
        # Track all persons at gate
        self.gate_persons = {}  # track_id -> person_state
        
        # Threat tracking
        self.active_threats = {}  # track_id -> threat_info
        self.disappeared_persons = {}  # track_id -> disappearance_info
        
        # Reference to tracker (Added for Fix)
        self.tracker = None
        
        # Thresholds
        self.suspicious_time = 15  # Seconds at gate before suspicious
        self.disappear_threshold = 8  # INCREASED: 5s -> 8s to allow walking time
        self.rapid_movement_threshold = 2.0  # Meters per second
        self.group_size_alert = 3  # Alert if 3+ people at gate
        
        # Break-in detection
        self.entry_timeout = 30  # If person doesn't enter door within 30s after gate = suspicious
        
        self.logger.success("Security Threat Analyzer initialized")

    def set_tracker(self, tracker):
        """Link the cross-camera tracker to check for matches"""
        self.tracker = tracker
        self.logger.info("Threat Analyzer linked to Cross-Camera Tracker")
    
    def update_gate_person(self, track_id: int, detection: Dict):
        """Update person state at gate camera"""
        current_time = time.time()
        
        if track_id not in self.gate_persons:
            # New person at gate
            self.gate_persons[track_id] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'positions': [detection['bbox']],
                'disappeared': False,
                'entered_door': False,
                'threat_level': 'none',
                'behaviors': set(),
                'bbox_history': [detection['bbox']],
                'pose_threats': []
            }
            
            self.logger.info(f"New person at gate: Track {track_id}")
        else:
            # Update existing person
            state = self.gate_persons[track_id]
            state['last_seen'] = current_time
            state['bbox_history'].append(detection['bbox'])
            
            # Keep only recent history (last 30 frames)
            if len(state['bbox_history']) > 30:
                state['bbox_history'] = state['bbox_history'][-30:]
        
        # Analyze behavior
        self._analyze_gate_behavior(track_id)
    
    def _analyze_gate_behavior(self, track_id: int):
        """Analyze person's behavior at gate"""
        state = self.gate_persons[track_id]
        current_time = time.time()
        time_at_gate = current_time - state['first_seen']
        
        behaviors = set()
        threat_level = 'none'
        
        # 1. SUSPICIOUS LOITERING
        if time_at_gate > self.suspicious_time and not state['entered_door']:
            behaviors.add('suspicious_loitering')
            threat_level = 'medium'
            
            if time_at_gate > 30:
                threat_level = 'high'
        
        # 2. ERRATIC MOVEMENT (back and forth)
        if len(state['bbox_history']) >= 10:
            movement_variance = self._calculate_movement_variance(state['bbox_history'])
            
            if movement_variance > 5000:  # High variance = erratic
                behaviors.add('erratic_movement')
                threat_level = max(threat_level, 'medium', key=self._threat_priority)
        
        # 3. RAPID APPROACH (moving very fast toward gate)
        if len(state['bbox_history']) >= 5:
            speed = self._calculate_speed(state['bbox_history'][-5:])
            
            if speed > self.rapid_movement_threshold:
                behaviors.add('rapid_approach')
                threat_level = max(threat_level, 'medium', key=self._threat_priority)
        
        # 4. PACING (moving back and forth)
        if len(state['bbox_history']) >= 20:
            is_pacing = self._detect_pacing(state['bbox_history'])
            
            if is_pacing:
                behaviors.add('pacing')
                threat_level = max(threat_level, 'medium', key=self._threat_priority)
        
        # Update state
        state['behaviors'] = behaviors
        state['threat_level'] = threat_level
        
        # Check if threat escalated
        if threat_level in ['medium', 'high'] and track_id not in self.active_threats:
            self._trigger_threat_alert(track_id, behaviors, threat_level)
    
    def check_disappearance(self, active_track_ids: Set[int]):
        """Check if persons disappeared from gate without entering door"""
        current_time = time.time()
        
        for track_id, state in list(self.gate_persons.items()):
            if track_id not in active_track_ids and not state['disappeared']:
                # Person no longer visible
                time_since_seen = current_time - state['last_seen']
                
                if time_since_seen > self.disappear_threshold:
                    
                    # === FIX: DOUBLE CHECK WITH TRACKER BEFORE ALERTING ===
                    if self.tracker and self.tracker.is_matched('gate', track_id):
                        self.mark_entered_door(track_id)
                        continue
                    # ======================================================

                    # Person disappeared
                    state['disappeared'] = True
                    
                    # Check if they entered door
                    if not state['entered_door']:
                        # CRITICAL: Person disappeared without entering = potential break-in
                        self._trigger_disappearance_alert(track_id, state)
    
    def mark_entered_door(self, gate_track_id: int):
        """Mark that person entered through door (legitimate entry)"""
        if gate_track_id in self.gate_persons:
            if not self.gate_persons[gate_track_id]['entered_door']:
                self.gate_persons[gate_track_id]['entered_door'] = True
                self.logger.success(f"Track {gate_track_id} confirmed entered door (Threat Cancelled)")
            
            # Cancel threat if person entered legitimately
            if gate_track_id in self.active_threats:
                del self.active_threats[gate_track_id]
    
    def check_group_threat(self, active_track_ids: List[int]):
        """Check if multiple people at gate (potential group break-in)"""
        if len(active_track_ids) >= self.group_size_alert:
            # Count how many are NOT recognized
            unknown_count = sum(
                1 for tid in active_track_ids 
                if self.gate_persons.get(tid, {}).get('threat_level') != 'none'
            )
            
            if len(active_track_ids) >= 3:
                self._trigger_group_alert(len(active_track_ids), unknown_count)
    
    def add_pose_threat(self, track_id: int, pose_result: Dict):
        """Add pose/body language threat data"""
        if track_id in self.gate_persons:
            threat_level = pose_result.get('threat_level', 'none')
            
            if threat_level in ['medium', 'high']:
                self.gate_persons[track_id]['pose_threats'].append({
                    'time': time.time(),
                    'level': threat_level,
                    'analysis': pose_result.get('analysis', {})
                })
                
                # Update overall threat level
                self.gate_persons[track_id]['threat_level'] = max(
                    self.gate_persons[track_id]['threat_level'],
                    threat_level,
                    key=self._threat_priority
                )
    
    def _trigger_threat_alert(self, track_id: int, behaviors: Set, threat_level: str):
        """Trigger security threat alert"""
        self.active_threats[track_id] = {
            'start_time': time.time(),
            'behaviors': behaviors,
            'level': threat_level
        }
        
        behavior_str = ', '.join(behaviors)
        
        self.logger.critical("=" * 80)
        self.logger.critical(f"圷 SECURITY THREAT DETECTED - Track {track_id}")
        self.logger.critical(f"   Level: {threat_level.upper()}")
        self.logger.critical(f"   Behaviors: {behavior_str}")
        self.logger.critical("=" * 80)
        
        # Audio alert
        if threat_level == 'high':
            self.audio.play_critical_threat()
        else:
            self.audio.play_threat()
        
        self.metrics.increment_counter('security_threats')
    
    def _trigger_disappearance_alert(self, track_id: int, state: Dict):
        """CRITICAL: Person disappeared without entering = break-in attempt"""
        time_at_gate = state['last_seen'] - state['first_seen']
        behaviors = state['behaviors']
        
        self.disappeared_persons[track_id] = {
            'disappear_time': time.time(),
            'time_at_gate': time_at_gate,
            'behaviors': behaviors,
            'threat_level': state['threat_level']
        }
        
        behavior_str = ', '.join(behaviors) if behaviors else 'normal approach'
        
        self.logger.critical("=" * 80)
        self.logger.critical(f"圷 BREAK-IN ATTEMPT DETECTED!")
        self.logger.critical(f"   Track {track_id} disappeared without door entry")
        self.logger.critical(f"   Time at gate: {time_at_gate:.1f}s")
        self.logger.critical(f"   Behaviors: {behavior_str}")
        self.logger.critical(f"   POSSIBLE BREAK-IN FROM ALTERNATE ENTRY!")
        self.logger.critical("=" * 80)
        
        # CRITICAL audio alert
        self.audio.play_break_in_alert()
        
        self.metrics.increment_counter('break_in_attempts')
    
    def _trigger_group_alert(self, total_count: int, suspicious_count: int):
        """Alert for group of people at gate"""
        # Don't spam - only alert once per minute
        current_time = time.time()
        
        if not hasattr(self, '_last_group_alert'):
            self._last_group_alert = 0
        
        if current_time - self._last_group_alert < 60:
            return
        
        self._last_group_alert = current_time
        
        self.logger.warning(f"則 GROUP AT GATE: {total_count} persons ({suspicious_count} suspicious)")
        
        if total_count >= 5:
            self.logger.critical("圷 LARGE GROUP DETECTED - Potential coordinated break-in")
            self.audio.play_critical_threat()
    
    def _calculate_movement_variance(self, bbox_history: List) -> float:
        """Calculate variance in movement (erratic = high variance)"""
        if len(bbox_history) < 2:
            return 0
        
        centers = []
        for bbox in bbox_history:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])
        
        centers = np.array(centers)
        variance = np.var(centers, axis=0).sum()
        
        return variance
    
    def _calculate_speed(self, bbox_history: List) -> float:
        """Calculate movement speed (pixels per second)"""
        if len(bbox_history) < 2:
            return 0
        
        # Calculate distance between first and last position
        first = bbox_history[0]
        last = bbox_history[-1]
        
        cx1 = (first[0] + first[2]) / 2
        cy1 = (first[1] + first[3]) / 2
        cx2 = (last[0] + last[2]) / 2
        cy2 = (last[1] + last[3]) / 2
        
        distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        
        # Assume 30 FPS, 5 frames = 0.166 seconds
        time_elapsed = len(bbox_history) / 30.0
        
        speed = distance / time_elapsed if time_elapsed > 0 else 0
        
        return speed
    
    def _detect_pacing(self, bbox_history: List) -> bool:
        """Detect pacing behavior (back and forth movement)"""
        if len(bbox_history) < 20:
            return False
        
        centers = []
        for bbox in bbox_history:
            cx = (bbox[0] + bbox[2]) / 2
            centers.append(cx)
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(centers) - 1):
            prev_dir = centers[i] - centers[i-1]
            curr_dir = centers[i+1] - centers[i]
            
            # Direction changed
            if prev_dir * curr_dir < 0:
                direction_changes += 1
        
        # If 3+ direction changes in 20 frames = pacing
        return direction_changes >= 3
    
    def _threat_priority(self, level: str) -> int:
        """Convert threat level to priority number"""
        levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        return levels.get(level, 0)
    
    def get_threat_summary(self) -> Dict:
        """Get current threat summary"""
        return {
            'active_threats': len(self.active_threats),
            'disappeared_persons': len(self.disappeared_persons),
            'persons_at_gate': len([p for p in self.gate_persons.values() if not p['disappeared']]),
            'high_threat_count': len([t for t in self.active_threats.values() if t['level'] == 'high'])
        }
    
    def clear_old_data(self):
        """Clean up old tracking data"""
        current_time = time.time()
        
        # Remove persons not seen in 60 seconds
        to_remove = []
        for track_id, state in self.gate_persons.items():
            if current_time - state['last_seen'] > 60:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.gate_persons[track_id]
            if track_id in self.active_threats:
                del self.active_threats[track_id]