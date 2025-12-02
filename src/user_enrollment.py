"""
SENTINEL v5.0 - User Enrollment System
Capture face images and add new authorized users to database
"""

import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional
from logger import get_logger


class UserEnrollmentSystem:
    """Interactive user enrollment for adding authorized users"""
    
    def __init__(self, config, face_recognition_system, metrics):
        self.config = config
        self.face_recognition = face_recognition_system
        self.metrics = metrics
        self.logger = get_logger("Enrollment")
        
        self.enabled = config.users.enrollment.enabled
        self.min_images = config.users.enrollment.min_images
        self.max_images = config.users.enrollment.max_images
        self.quality_threshold = config.users.enrollment.quality_threshold
        
        # Enrollment state
        self.enrolling = False
        self.current_enrollment = None
        
        self.logger.success("User enrollment system initialized")
    
    def start_enrollment(self, user_name: str) -> bool:
        """
        Start enrollment process for a new user
        
        Args:
            user_name: Name of user to enroll
        
        Returns:
            Success status
        """
        if not self.enabled:
            self.logger.error("Enrollment disabled in config")
            return False
        
        if self.enrolling:
            self.logger.warning("Enrollment already in progress")
            return False
        
        if not user_name or len(user_name) < 2:
            self.logger.error("Invalid user name")
            return False
        
        # Check if user already exists
        if user_name in self.face_recognition.database:
            self.logger.warning(f"User '{user_name}' already exists")
            response = input(f"User '{user_name}' already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # Initialize enrollment
        self.current_enrollment = {
            'name': user_name,
            'images': [],
            'embeddings': [],
            'start_time': time.time(),
            'frames_processed': 0,
            'quality_rejections': 0
        }
        
        self.enrolling = True
        
        self.logger.info("=" * 60)
        self.logger.info(f"ENROLLMENT STARTED: {user_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"Need {self.min_images}-{self.max_images} good quality images")
        self.logger.info("Instructions:")
        self.logger.info("  - Look at camera")
        self.logger.info("  - Move head slightly (different angles)")
        self.logger.info("  - Keep face well-lit and visible")
        self.logger.info("  - System will capture automatically")
        self.logger.info("=" * 60)
        
        return True
    
    def process_frame(self, frame: np.ndarray, detection: Dict) -> Dict:
        """
        Process frame during enrollment
        
        Args:
            frame: Camera frame
            detection: Person detection dictionary
        
        Returns:
            Enrollment status dictionary
        """
        if not self.enrolling or self.current_enrollment is None:
            return {'enrolling': False}
        
        enrollment = self.current_enrollment
        enrollment['frames_processed'] += 1
        
        # Get face crop
        crop = detection.get('crop')
        if crop is None or crop.size == 0:
            return self._get_enrollment_status('Waiting for face...')
        
        # Check quality
        quality_ok, quality_reason = self.face_recognition._check_face_quality(crop)
        
        if not quality_ok:
            enrollment['quality_rejections'] += 1
            return self._get_enrollment_status(f'Quality issue: {quality_reason}')
        
        # Extract embedding
        embedding = self.face_recognition._extract_embedding(crop)
        
        if embedding is None:
            enrollment['quality_rejections'] += 1
            return self._get_enrollment_status('No face detected in crop')
        
        # Check for uniqueness (not too similar to existing captures)
        if len(enrollment['embeddings']) > 0:
            similarities = []
            for existing_emb in enrollment['embeddings']:
                similarity = np.dot(embedding, existing_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(existing_emb)
                )
                similarities.append(similarity)
            
            max_similarity = max(similarities)
            
            # If too similar, skip (want variety)
            if max_similarity > 0.95:
                return self._get_enrollment_status('Too similar to existing capture')
        
        # Add image and embedding
        enrollment['images'].append(crop.copy())
        enrollment['embeddings'].append(embedding)
        
        captured_count = len(enrollment['embeddings'])
        
        self.logger.success(
            f"Captured {captured_count}/{self.max_images} "
            f"(quality: {enrollment['frames_processed'] - enrollment['quality_rejections']}"
            f"/{enrollment['frames_processed']})"
        )
        
        # Check if done
        if captured_count >= self.max_images:
            return self._finalize_enrollment()
        
        return self._get_enrollment_status(
            f'Capturing... {captured_count}/{self.max_images}'
        )
    
    def _finalize_enrollment(self) -> Dict:
        """Finalize enrollment and save to database"""
        enrollment = self.current_enrollment
        name = enrollment['name']
        embeddings = enrollment['embeddings']
        
        if len(embeddings) < self.min_images:
            self.logger.error(
                f"Not enough quality images. "
                f"Got {len(embeddings)}, need {self.min_images}"
            )
            self.cancel_enrollment()
            return {'enrolling': False, 'error': 'Insufficient images'}
        
        try:
            # Add to face recognition database
            self.face_recognition.database[name] = embeddings
            
            # Save database
            db_path = Path(self.config.face.database)
            with open(db_path, 'wb') as f:
                pickle.dump(self.face_recognition.database, f)
            
            # Save images to dataset (optional)
            self._save_enrollment_images(name, enrollment['images'])
            
            duration = time.time() - enrollment['start_time']
            
            self.logger.info("=" * 60)
            self.logger.success(f"ENROLLMENT COMPLETE: {name}")
            self.logger.info("=" * 60)
            self.logger.info(f"Images captured: {len(embeddings)}")
            self.logger.info(f"Duration: {duration:.1f} seconds")
            quality_processed = enrollment['frames_processed'] - enrollment['quality_rejections']
            self.logger.info(f"Quality rate: {quality_processed} / {enrollment['frames_processed']}")
            self.logger.info("=" * 60)
            
            # Reset state
            self.enrolling = False
            self.current_enrollment = None
            
            return {
                'enrolling': False,
                'completed': True,
                'name': name,
                'images_captured': len(embeddings)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save enrollment: {e}")
            self.cancel_enrollment()
            return {'enrolling': False, 'error': str(e)}
    
    def _save_enrollment_images(self, name: str, images: List[np.ndarray]):
        """Save enrollment images to dataset directory"""
        try:
            dataset_dir = Path('dataset') / name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            
            for i, img in enumerate(images):
                filename = f"{name}_{timestamp}_{i+1}.jpg"
                filepath = dataset_dir / filename
                cv2.imwrite(str(filepath), img)
            
            self.logger.info(f"Saved {len(images)} images to {dataset_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save enrollment images: {e}")
    
    def cancel_enrollment(self):
        """Cancel current enrollment"""
        if self.enrolling and self.current_enrollment:
            name = self.current_enrollment['name']
            self.logger.warning(f"Enrollment cancelled for: {name}")
        
        self.enrolling = False
        self.current_enrollment = None
    
    def _get_enrollment_status(self, message: str = '') -> Dict:
        """Get current enrollment status"""
        if not self.enrolling or self.current_enrollment is None:
            return {'enrolling': False}
        
        enrollment = self.current_enrollment
        
        return {
            'enrolling': True,
            'name': enrollment['name'],
            'captured': len(enrollment['embeddings']),
            'required': self.min_images,
            'max': self.max_images,
            'progress': len(enrollment['embeddings']) / self.max_images,
            'message': message,
            'frames_processed': enrollment['frames_processed'],
            'quality_rejections': enrollment['quality_rejections']
        }
    
    def is_enrolling(self) -> bool:
        """Check if enrollment is in progress"""
        return self.enrolling
    
    def get_current_enrollment(self) -> Optional[Dict]:
        """Get current enrollment info"""
        if self.enrolling:
            return self.current_enrollment
        return None
    
    def draw_enrollment_ui(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """Draw enrollment UI overlay"""
        if not status.get('enrolling'):
            return frame
        
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        title = f"ENROLLING: {status['name'].upper()}"
        cv2.putText(
            frame, title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 255), 3
        )
        
        # Progress
        progress_text = f"Captured: {status['captured']}/{status['max']}"
        cv2.putText(
            frame, progress_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )
        
        # Progress bar
        bar_x = 20
        bar_y = 100
        bar_w = 400
        bar_h = 30
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
        
        progress_w = int(bar_w * status['progress'])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), (0, 255, 0), -1)
        
        # Status message
        if status.get('message'):
            cv2.putText(
                frame, status['message'],
                (bar_x + bar_w + 20, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )
        
        return frame
