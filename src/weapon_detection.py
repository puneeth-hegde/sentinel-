"""
SENTINEL v5.0 - Weapon Detection System
Detects knives, guns, and weapons using YOLO classes
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional
from logger import get_logger


class WeaponDetectionSystem:
    """Detect weapons using YOLO"""
    
    # COCO dataset weapon classes
    WEAPON_CLASSES = {
        43: 'knife',
        44: 'gun',
        # Extended weapon detection (if using custom trained model)
        # 45: 'rifle',
        # 46: 'sword'
    }
    
    def __init__(self, config, yolo_model, metrics):
        self.config = config
        self.model = yolo_model
        self.metrics = metrics
        self.logger = get_logger("WeaponDetection")
        
        self.enabled = config.alerts.weapon.enabled
        self.weapon_classes = config.alerts.weapon.classes
        self.confidence_threshold = 0.5  # Higher confidence for weapons
        
        # Detection tracking
        self.weapon_detections = {}  # track_id -> weapon info
        self.detection_history = {}  # track_id -> list of detections
        self.confirmed_weapons = set()  # track_ids with confirmed weapons
        
        # Require multiple detections to confirm (reduce false positives)
        self.confirmations_required = 3
        
        if self.enabled:
            self.logger.success("Weapon detection enabled")
        else:
            self.logger.info("Weapon detection disabled")
    
    def detect_weapons(self, frame: np.ndarray, person_bbox: List[int]) -> Dict:
        """
        Detect weapons in person's area
        
        Args:
            frame: Full frame
            person_bbox: Person's bounding box [x1, y1, x2, y2]
        
        Returns:
            Weapon detection result
        """
        if not self.enabled:
            return {'detected': False, 'weapons': []}
        
        start_time = time.time()
        
        try:
            # Expand search area around person (weapons might be slightly outside bbox)
            x1, y1, x2, y2 = map(int, person_bbox)
            h, w = frame.shape[:2]
            
            # Expand by 20%
            expand_x = int((x2 - x1) * 0.2)
            expand_y = int((y2 - y1) * 0.2)
            
            search_x1 = max(0, x1 - expand_x)
            search_y1 = max(0, y1 - expand_y)
            search_x2 = min(w, x2 + expand_x)
            search_y2 = min(h, y2 + expand_y)
            
            search_area = frame[search_y1:search_y2, search_x1:search_x2]
            
            if search_area.size == 0:
                return {'detected': False, 'weapons': []}
            
            # Run YOLO detection on search area
            results = self.model(
                search_area,
                classes=self.weapon_classes,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            weapons = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    
                    if cls in self.WEAPON_CLASSES:
                        # Get bbox in search area coordinates
                        wx1, wy1, wx2, wy2 = boxes.xyxy[i].cpu().numpy()
                        
                        # Convert to full frame coordinates
                        weapon_bbox = [
                            int(search_x1 + wx1),
                            int(search_y1 + wy1),
                            int(search_x1 + wx2),
                            int(search_y1 + wy2)
                        ]
                        
                        weapons.append({
                            'type': self.WEAPON_CLASSES[cls],
                            'confidence': conf,
                            'bbox': weapon_bbox,
                            'class_id': cls
                        })
                        
                        self.logger.warning(
                            f"Weapon detected: {self.WEAPON_CLASSES[cls]} "
                            f"(confidence: {conf:.2f})"
                        )
            
            # Record latency
            latency = time.time() - start_time
            self.metrics.record_latency('weapon_detection', latency)
            
            return {
                'detected': len(weapons) > 0,
                'weapons': weapons,
                'count': len(weapons)
            }
            
        except Exception as e:
            self.logger.error(f"Weapon detection error: {e}")
            return {'detected': False, 'weapons': []}
    
    def update_person_weapons(self, track_id: int, detection_result: Dict) -> Dict:
        """
        Update weapon detection state for a person
        
        Args:
            track_id: Person's tracking ID
            detection_result: Result from detect_weapons()
        
        Returns:
            Confirmed weapon status
        """
        if track_id not in self.detection_history:
            self.detection_history[track_id] = []
        
        # Add to history
        self.detection_history[track_id].append({
            'timestamp': time.time(),
            'detected': detection_result['detected'],
            'weapons': detection_result['weapons']
        })
        
        # Keep only recent detections (last 10)
        self.detection_history[track_id] = self.detection_history[track_id][-10:]
        
        # Count recent positive detections
        recent_detections = self.detection_history[track_id][-5:]
        positive_count = sum(1 for d in recent_detections if d['detected'])
        
        # Confirm if enough detections
        if positive_count >= self.confirmations_required:
            if track_id not in self.confirmed_weapons:
                self.confirmed_weapons.add(track_id)
                self.logger.critical(
                    f"WEAPON CONFIRMED for track {track_id} "
                    f"({positive_count} detections)"
                )
                self.metrics.increment_counter('weapons_detected')
        
        # Get most common weapon type
        weapon_types = []
        for det in recent_detections:
            if det['detected']:
                for weapon in det['weapons']:
                    weapon_types.append(weapon['type'])
        
        most_common_weapon = None
        if weapon_types:
            from collections import Counter
            most_common_weapon = Counter(weapon_types).most_common(1)[0][0]
        
        return {
            'track_id': track_id,
            'confirmed': track_id in self.confirmed_weapons,
            'detections_count': positive_count,
            'weapon_type': most_common_weapon,
            'current_detection': detection_result
        }
    
    def is_weapon_confirmed(self, track_id: int) -> bool:
        """Check if weapon is confirmed for this track"""
        return track_id in self.confirmed_weapons
    
    def draw_weapon_detections(
        self,
        frame: np.ndarray,
        weapons: List[Dict],
        confirmed: bool = False
    ) -> np.ndarray:
        """Draw weapon detection bounding boxes"""
        for weapon in weapons:
            bbox = weapon['bbox']
            weapon_type = weapon['type']
            confidence = weapon['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Color: red for confirmed, orange for detection
            color = (0, 0, 255) if confirmed else (0, 165, 255)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"⚠ {weapon_type.upper()}"
            if confirmed:
                label = f"🚨 {weapon_type.upper()} CONFIRMED"
            
            # Label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )
            
            # Confidence
            conf_text = f"{confidence:.0%}"
            cv2.putText(
                frame, conf_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        
        return frame
    
    def clear_track(self, track_id: int):
        """Clear weapon detection data for inactive track"""
        if track_id in self.detection_history:
            del self.detection_history[track_id]
        if track_id in self.weapon_detections:
            del self.weapon_detections[track_id]
        if track_id in self.confirmed_weapons:
            self.confirmed_weapons.remove(track_id)
    
    def get_active_weapons(self) -> int:
        """Get count of active weapon detections"""
        return len(self.confirmed_weapons)
