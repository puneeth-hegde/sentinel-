"""
SENTINEL v5.0 - Cross-Camera Tracking
Match persons between gate and door cameras
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from logger import get_logger


class CrossCameraTracker:
    """Track persons across multiple cameras"""
    
    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics
        self.logger = get_logger("CrossCamera")
        
        self.enabled = config.tracking.enabled
        self.time_window = config.tracking.time_window
        self.max_time_gap = config.tracking.max_time_gap
        self.appearance_weight = config.tracking.appearance_weight
        self.distance_threshold = config.tracking.distance_threshold
        
        # Tracking state
        self.gate_tracks = {}  # gate_track_id -> track info
        self.door_tracks = {}  # door_track_id -> track info
        self.matches = {}      # door_track_id -> gate_track_id
        self.reverse_matches = {}  # gate_track_id -> door_track_id
        
        self.logger.success("Cross-camera tracker initialized")
    
    def update_gate_track(self, track_id: int, detection: Dict):
        """Update gate camera track"""
        if track_id not in self.gate_tracks:
            self.gate_tracks[track_id] = {
                'first_seen': time.time(),
                'last_seen': time.time(),
                'appearance': detection.get('appearance'),
                'center': detection['center'],
                'matched': False
            }
        else:
            self.gate_tracks[track_id]['last_seen'] = time.time()
            if detection.get('appearance') is not None:
                self.gate_tracks[track_id]['appearance'] = detection['appearance']
    
    def update_door_track(self, track_id: int, detection: Dict):
        """Update door camera track and attempt matching"""
        if track_id not in self.door_tracks:
            # New person at door - try to match with gate
            matched_gate_id = self._find_match(detection)
            
            self.door_tracks[track_id] = {
                'first_seen': time.time(),
                'last_seen': time.time(),
                'appearance': detection.get('appearance'),
                'center': detection['center'],
                'matched_gate_id': matched_gate_id
            }
            
            if matched_gate_id is not None:
                self.matches[track_id] = matched_gate_id
                self.reverse_matches[matched_gate_id] = track_id
                self.logger.success(
                    f"Match found: Gate {matched_gate_id} -> Door {track_id}"
                )
        else:
            self.door_tracks[track_id]['last_seen'] = time.time()
            if detection.get('appearance') is not None:
                self.door_tracks[track_id]['appearance'] = detection['appearance']
    
    def _find_match(self, door_detection: Dict) -> Optional[int]:
        """Find matching gate track for door detection"""
        if not self.enabled:
            return None
        
        current_time = time.time()
        best_match = None
        best_score = float('inf')
        
        for gate_id, gate_track in self.gate_tracks.items():
            # Check if already matched
            if gate_track['matched']:
                continue
            
            # Check time window
            time_since_gate = current_time - gate_track['last_seen']
            if time_since_gate > self.max_time_gap:
                continue
            
            # Calculate match score
            score = self._calculate_match_score(gate_track, door_detection, time_since_gate)
            
            if score < best_score and score < self.distance_threshold:
                best_score = score
                best_match = gate_id
        
        if best_match is not None:
            self.gate_tracks[best_match]['matched'] = True
        
        return best_match
    
    def _calculate_match_score(
        self,
        gate_track: Dict,
        door_detection: Dict,
        time_gap: float
    ) -> float:
        """Calculate similarity score between gate and door tracks"""
        scores = []
        
        # Appearance similarity (if available)
        if gate_track.get('appearance') is not None and door_detection.get('appearance') is not None:
            gate_app = gate_track['appearance']
            door_app = door_detection['appearance']
            
            # Cosine similarity
            cos_sim = np.dot(gate_app, door_app) / (
                np.linalg.norm(gate_app) * np.linalg.norm(door_app) + 1e-6
            )
            appearance_distance = 1 - cos_sim
            scores.append(appearance_distance * self.appearance_weight)
        
        # Temporal score (normalized by max time gap)
        temporal_score = time_gap / self.max_time_gap
        scores.append(temporal_score * (1 - self.appearance_weight))
        
        # Combined score
        if len(scores) > 0:
            return sum(scores) / len(scores)
        else:
            return 1.0  # No features available
    
    def get_unified_id(self, camera: str, track_id: int) -> str:
        """Get unified ID for a track across cameras"""
        if camera == 'gate':
            if track_id in self.reverse_matches:
                door_id = self.reverse_matches[track_id]
                return f"person_{track_id}_{door_id}"
            else:
                return f"gate_{track_id}"
        
        elif camera == 'door':
            if track_id in self.matches:
                gate_id = self.matches[track_id]
                return f"person_{gate_id}_{track_id}"
            else:
                return f"door_{track_id}"
        
        return f"{camera}_{track_id}"
    
    def get_match(self, camera: str, track_id: int) -> Optional[int]:
        """Get matched track ID from other camera"""
        if camera == 'gate':
            return self.reverse_matches.get(track_id)
        elif camera == 'door':
            return self.matches.get(track_id)
        return None
    
    def is_matched(self, camera: str, track_id: int) -> bool:
        """Check if track is matched with other camera"""
        if camera == 'gate':
            return track_id in self.reverse_matches
        elif camera == 'door':
            return track_id in self.matches
        return False
    
    def cleanup_old_tracks(self):
        """Remove old tracks that are no longer active"""
        current_time = time.time()
        timeout = 60  # 60 seconds timeout
        
        # Clean gate tracks
        to_remove = []
        for track_id, track in self.gate_tracks.items():
            if current_time - track['last_seen'] > timeout:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.gate_tracks[track_id]
            if track_id in self.reverse_matches:
                door_id = self.reverse_matches[track_id]
                del self.matches[door_id]
                del self.reverse_matches[track_id]
        
        # Clean door tracks
        to_remove = []
        for track_id, track in self.door_tracks.items():
            if current_time - track['last_seen'] > timeout:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.door_tracks[track_id]
            if track_id in self.matches:
                gate_id = self.matches[track_id]
                del self.reverse_matches[gate_id]
                del self.matches[track_id]
        
        if len(to_remove) > 0:
            self.logger.debug(f"Cleaned up {len(to_remove)} old tracks")
    
    def get_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'gate_tracks': len(self.gate_tracks),
            'door_tracks': len(self.door_tracks),
            'matches': len(self.matches)
        }