"""
SENTINEL v5.0 - Detection Pipeline
YOLOv8 + ByteTrack for robust person detection and tracking
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from bytetrack import ByteTrack
from logger import get_logger
from typing import List, Dict, Tuple


class DetectionPipeline:
    """Person detection and tracking pipeline"""
    
    def __init__(self, config, camera_type: str, metrics):
        self.config = config
        self.camera_type = camera_type
        self.metrics = metrics
        self.logger = get_logger(f"Detection-{camera_type.upper()}")
        
        # Load YOLO model
        self.logger.info(f"Loading YOLO model: {config.detection.model}")
        self.model = YOLO(config.detection.model)
        self.model.to(config.detection.device)
        
        # Initialize ByteTrack
        self.tracker = ByteTrack(
            track_thresh=config.detection.track_thresh,
            match_thresh=config.detection.match_thresh,
            track_buffer=config.detection.track_buffer
        )
        
        # Detection settings
        self.confidence = config.detection.confidence
        self.classes = config.detection.classes
        self.min_height = config.detection.min_bbox_height[camera_type]
        self.min_width = config.detection.min_bbox_width[camera_type]
        self.imgsz = config.detection.imgsz
        
        # Tracking state
        self.tracked_objects = {}  # track_id -> object info
        self.active_tracks = set()
        
        self.logger.success(f"Detection pipeline initialized for {camera_type}")
        self.logger.info(f"Confidence: {self.confidence}, Min Height: {self.min_height}")
    
    def _filter_detections(self, results) -> List[np.ndarray]:
        """Filter YOLO detections by confidence, class, and size"""
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                
                # Check class and confidence
                if cls not in self.classes or conf < self.confidence:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # Check minimum size (CRITICAL for avoiding false alarms)
                if h < self.min_height or w < self.min_width:
                    continue
                
                detections.append(np.array([x1, y1, x2, y2, conf]))
        
        return detections
    
    def _create_detection_dict(self, bbox, track_id: int, frame: np.ndarray) -> Dict:
        """Create detection dictionary with metadata"""
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Extract person crop
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
        
        # Calculate appearance features (simple color histogram)
        appearance = None
        if crop is not None and crop.size > 0:
            try:
                crop_resized = cv2.resize(crop, (64, 128))
                hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                appearance = cv2.normalize(hist, hist).flatten()
            except:
                appearance = None
        
        return {
            'track_id': track_id,
            'camera': self.camera_type,
            'bbox': [x1, y1, x2, y2],
            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
            'width': x2 - x1,
            'height': y2 - y1,
            'crop': crop,
            'appearance': appearance,
            'timestamp': time.time(),
            'frame': frame
        }
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a frame: detect persons, track, and return detections
        
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            classes=self.classes,
            verbose=False
        )
        
        # Filter detections
        detections = self._filter_detections(results)
        
        # Update tracker
        tracked = self.tracker.update(detections)
        
        # Create detection dictionaries
        detections_list = []
        current_tracks = set()
        
        for track in tracked:
            track_id = int(track[4])
            current_tracks.add(track_id)
            
            det_dict = self._create_detection_dict(track, track_id, frame)
            detections_list.append(det_dict)
            
            # Update tracked objects
            self.tracked_objects[track_id] = det_dict
            
            # Increment detection counter
            if track_id not in self.active_tracks:
                self.metrics.increment_counter('detections')
        
        # Update active tracks
        self.active_tracks = current_tracks
        
        # Record latency
        latency = time.time() - start_time
        self.metrics.record_latency(f'detection_{self.camera_type}', latency)
        
        return detections_list
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        frame_display = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            
            # Default: yellow box for tracking
            color = (0, 255, 255)
            label = f"Tracking: {self.camera_type}_{track_id}"
            
            # Check if this person has recognition result
            if 'identity' in det:
                identity = det['identity']
                if identity['name'] != 'Unknown':
                    # Green for recognized
                    color = (0, 255, 0)
                    label = identity['name']
                else:
                    # Keep yellow for unknown
                    label = "Unknown"
            
            # Check for face hidden state
            if det.get('face_hidden', False):
                color = (0, 0, 255)  # Red
                label = "Face Hidden"
            
            # Draw bounding box
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame_display,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame_display,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame_display
    
    def get_track_info(self, track_id: int) -> Dict:
        """Get information about a tracked object"""
        return self.tracked_objects.get(track_id, None)
    
    def get_active_tracks(self) -> List[int]:
        """Get list of currently active track IDs"""
        return list(self.active_tracks)
