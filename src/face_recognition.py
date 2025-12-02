"""
SENTINEL v5.0 - Face Recognition System
InsightFace with GPU acceleration and intelligent matching
"""

import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from insightface.app import FaceAnalysis
from logger import get_logger


class FaceRecognitionSystem:
    """InsightFace-based face recognition with intelligent matching"""
    
    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics
        self.logger = get_logger("FaceRecognition")
        
        # Load InsightFace model
        self.logger.info(f"Loading InsightFace model: {config.face.model}")
        self.app = FaceAnalysis(
            name=config.face.model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0)
        
        # Load face database
        self.database = self._load_database()
        self.threshold = config.face.threshold
        
        # Recognition state
        self.recognition_cache = {}  # track_id -> recognition state
        self.last_recognition_time = {}  # track_id -> timestamp
        self.throttle_seconds = config.face.throttle_seconds
        
        # Quality filters
        self.min_face_size = config.face.min_face_size
        self.min_brightness = config.face.min_brightness
        self.max_brightness = config.face.max_brightness
        
        self.logger.success(f"Face recognition initialized with {len(self.database)} authorized users")
    
    def _load_database(self) -> Dict:
        """Load face embeddings database"""
        db_path = Path(self.config.face.database)
        
        if not db_path.exists():
            self.logger.warning(f"Database not found: {db_path}")
            self.logger.warning("Please run train_faces.py to create the database")
            return {}
        
        try:
            with open(db_path, 'rb') as f:
                database = pickle.load(f)
            
            self.logger.info(f"Loaded {len(database)} users from database")
            for name, embeddings in database.items():
                self.logger.info(f"  - {name}: {len(embeddings)} embeddings")
            
            return database
            
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
            return {}
    
    def _check_face_quality(self, face_crop: np.ndarray) -> Tuple[bool, str]:
        """Check if face image quality is good enough"""
        if face_crop is None or face_crop.size == 0:
            return False, "Empty crop"
        
        h, w = face_crop.shape[:2]
        
        # Check size
        if h < self.min_face_size or w < self.min_face_size:
            return False, f"Too small: {w}x{h}"
        
        # Check brightness
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < self.min_brightness:
            return False, f"Too dark: {brightness:.1f}"
        
        if brightness > self.max_brightness:
            return False, f"Too bright: {brightness:.1f}"
        
        # Check blur (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.config.face.min_blur_score:
            return False, f"Too blurry: {blur_score:.1f}"
        
        return True, "OK"
    
    def _extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from crop"""
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.app.get(rgb)
            
            if len(faces) == 0:
                return None
            
            # Return embedding of largest face
            largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            return largest_face.embedding
            
        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {e}")
            return None
    
    def _match_embedding(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Match embedding against database"""
        if len(self.database) == 0:
            return "Unknown", 1.0
        
        best_match = "Unknown"
        best_distance = float('inf')
        
        for name, db_embeddings in self.database.items():
            for db_emb in db_embeddings:
                # Cosine distance
                distance = 1 - np.dot(embedding, db_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(db_emb)
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = name
        
        # Check threshold
        if best_distance > self.threshold:
            return "Unknown", best_distance
        
        return best_match, best_distance
    
    def should_recognize(self, track_id: int) -> bool:
        """Check if we should attempt recognition for this track"""
        # Check throttling
        if track_id in self.last_recognition_time:
            elapsed = time.time() - self.last_recognition_time[track_id]
            if elapsed < self.throttle_seconds:
                return False
        
        # Check if already confirmed
        if track_id in self.recognition_cache:
            state = self.recognition_cache[track_id]
            if state['status'] == 'confirmed':
                return False
        
        return True
    
    def recognize_face(self, track_id: int, face_crop: np.ndarray) -> Dict:
        """
        Recognize a face from a person's crop
        
        Returns:
            Recognition result dictionary
        """
        start_time = time.time()
        
        # Check quality
        quality_ok, quality_reason = self._check_face_quality(face_crop)
        if not quality_ok:
            return {
                'track_id': track_id,
                'status': 'poor_quality',
                'reason': quality_reason,
                'name': None,
                'confidence': 0.0
            }
        
        # Extract embedding
        embedding = self._extract_embedding(face_crop)
        if embedding is None:
            return {
                'track_id': track_id,
                'status': 'no_face_detected',
                'reason': 'No face found in crop',
                'name': None,
                'confidence': 0.0
            }
        
        # Match against database
        name, distance = self._match_embedding(embedding)
        
        # Update cache
        if track_id not in self.recognition_cache:
            self.recognition_cache[track_id] = {
                'status': 'verifying',
                'matches': [],
                'match_count': {}
            }
        
        cache = self.recognition_cache[track_id]
        cache['matches'].append({'name': name, 'distance': distance})
        
        # Count matches
        cache['match_count'][name] = cache['match_count'].get(name, 0) + 1
        
        # Check for confirmation (need 2 matches)
        matches_required = self.config.face.matches_required
        if cache['match_count'].get(name, 0) >= matches_required:
            cache['status'] = 'confirmed'
            cache['confirmed_name'] = name
            cache['confirmed_distance'] = distance
            
            self.logger.success(
                f"Track {track_id} confirmed as '{name}' "
                f"(distance: {distance:.3f}, matches: {cache['match_count'][name]})"
            )
        
        # Update last recognition time
        self.last_recognition_time[track_id] = time.time()
        
        # Record metrics
        latency = time.time() - start_time
        self.metrics.record_latency('face_recognition', latency)
        self.metrics.increment_counter('recognitions')
        
        return {
            'track_id': track_id,
            'status': cache['status'],
            'name': name,
            'distance': distance,
            'confidence': 1.0 - distance,
            'matches_required': matches_required,
            'match_count': cache['match_count'].get(name, 0),
            'confirmed': cache['status'] == 'confirmed'
        }
    
    def get_recognition_state(self, track_id: int) -> Dict:
        """Get current recognition state for a track"""
        if track_id not in self.recognition_cache:
            return {
                'status': 'unknown',
                'name': None,
                'confirmed': False
            }
        
        cache = self.recognition_cache[track_id]
        
        return {
            'status': cache['status'],
            'name': cache.get('confirmed_name', None),
            'distance': cache.get('confirmed_distance', None),
            'confirmed': cache['status'] == 'confirmed'
        }
    
    def clear_track(self, track_id: int):
        """Clear recognition data for a track that's no longer active"""
        if track_id in self.recognition_cache:
            del self.recognition_cache[track_id]
        if track_id in self.last_recognition_time:
            del self.last_recognition_time[track_id]
    
    def enroll_new_user(self, name: str, face_crops: list) -> bool:
        """
        Enroll a new user into the database
        
        Args:
            name: User's name
            face_crops: List of face crop images
        
        Returns:
            Success status
        """
        self.logger.info(f"Enrolling new user: {name}")
        
        embeddings = []
        
        for i, crop in enumerate(face_crops):
            quality_ok, reason = self._check_face_quality(crop)
            if not quality_ok:
                self.logger.warning(f"Image {i+1} rejected: {reason}")
                continue
            
            embedding = self._extract_embedding(crop)
            if embedding is not None:
                embeddings.append(embedding)
                self.logger.info(f"Image {i+1} processed successfully")
        
        if len(embeddings) < self.config.users.enrollment.min_images:
            self.logger.error(
                f"Not enough valid images. "
                f"Got {len(embeddings)}, need {self.config.users.enrollment.min_images}"
            )
            return False
        
        # Add to database
        self.database[name] = embeddings
        
        # Save database
        try:
            db_path = Path(self.config.face.database)
            with open(db_path, 'wb') as f:
                pickle.dump(self.database, f)
            
            self.logger.success(
                f"User '{name}' enrolled successfully with {len(embeddings)} images"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save database: {e}")
            return False
