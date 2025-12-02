"""
SENTINEL v5.0 - ByteTrack Integration
Stable multi-object tracking optimized for person detection
"""

import numpy as np
from collections import OrderedDict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """Kalman Filter for tracking bounding boxes"""
    
    count = 0
    
    def __init__(self, bbox):
        """Initialize tracker with bounding box [x1, y1, x2, y2]"""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State: [x, y, s, r, vx, vy, vs] where s=area, r=aspect ratio
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        self.kf.R[2:,2:] *= 10.0
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def _convert_bbox_to_z(self, bbox):
        """Convert [x1,y1,x2,y2] to [x,y,s,r] format"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.0
        y = bbox[1] + h/2.0
        s = w * h
        r = w / float(h) if h != 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x):
        """Convert [x,y,s,r] to [x1,y1,x2,y2] format"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w if w != 0 else 0
        return np.array([
            x[0] - w/2.0,
            x[1] - h/2.0,
            x[0] + w/2.0,
            x[1] + h/2.0
        ]).reshape((1, 4))
    
    def update(self, bbox):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """Advance state and return predicted bounding box"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """Return current bounding box estimate"""
        return self._convert_x_to_bbox(self.kf.x)


class ByteTrack:
    """ByteTrack implementation for person tracking"""
    
    def __init__(self, track_thresh=0.5, match_thresh=0.8, track_buffer=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.trackers = []
        self.frame_count = 0
    
    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _associate(self, detections, trackers, iou_threshold):
        """Associate detections to trackers using Hungarian algorithm"""
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
        
        # Hungarian algorithm
        if iou_matrix.size > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.asarray(matched_indices).T
        else:
            matched_indices = np.empty((0, 2))
        
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, unmatched_detections, unmatched_trackers
    
    def update(self, detections):
        """
        Update trackers with new detections
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence] arrays
        
        Returns:
            List of [x1, y1, x2, y2, track_id] arrays
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.delete(trks, to_del, axis=0)
        
        # Separate high and low confidence detections
        if len(detections) > 0:
            det_bboxes = np.array([d[:4] for d in detections])
            det_scores = np.array([d[4] for d in detections])
            
            high_idx = det_scores > self.track_thresh
            low_idx = det_scores <= self.track_thresh
            
            high_dets = det_bboxes[high_idx]
            low_dets = det_bboxes[low_idx]
        else:
            high_dets = np.empty((0, 4))
            low_dets = np.empty((0, 4))
        
        # First round: match high confidence detections
        matched, unmatched_dets, unmatched_trks = self._associate(
            high_dets, trks, self.match_thresh
        )
        
        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(high_dets[m[0]])
        
        # Second round: match low confidence detections with unmatched trackers
        if len(low_dets) > 0 and len(unmatched_trks) > 0:
            unmatched_trks_bboxes = trks[unmatched_trks]
            matched_low, unmatched_dets_low, unmatched_trks_low = self._associate(
                low_dets, unmatched_trks_bboxes, 0.5
            )
            
            for m in matched_low:
                tracker_idx = unmatched_trks[m[1]]
                self.trackers[tracker_idx].update(low_dets[m[0]])
        
        # Create new trackers for unmatched high confidence detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(high_dets[i])
            self.trackers.append(trk)
        
        # Remove dead trackers
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hit_streak >= 1 or self.frame_count <= 1):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.track_buffer:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
