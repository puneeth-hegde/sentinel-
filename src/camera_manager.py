"""
SENTINEL v5.0 - Camera Manager
Robust RTSP stream capture with auto-reconnect and frame buffering
"""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Full
from typing import Optional, Tuple
from logger import get_logger


class CameraManager:
    """Manages RTSP camera capture with reconnection logic"""
    
    def __init__(self, camera_config, camera_type: str, frame_queue: Queue, metrics):
        self.config = camera_config
        self.camera_type = camera_type
        self.frame_queue = frame_queue
        self.metrics = metrics
        self.logger = get_logger(f"Camera-{camera_type.upper()}")
        
        self.url = camera_config.url
        self.name = camera_config.name
        self.reconnect_attempts = camera_config.reconnect_attempts
        self.reconnect_delay = camera_config.reconnect_delay
        
        self.cap = None
        self.running = False
        self.thread = None
        self.last_frame = None
        self.frame_count = 0
        self.error_count = 0
        
        self.logger.info(f"Initialized {self.name}")
    
    def connect(self) -> bool:
        """Connect to RTSP stream"""
        try:
            self.logger.info(f"Connecting to {self.name}...")
            
            # OpenCV RTSP options for better performance
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to read a test frame
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.logger.success(f"Connected to {self.name}")
                    self.last_frame = frame
                    return True
            
            self.logger.error(f"Failed to connect to {self.name}")
            return False
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from stream"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.logger.info(f"Disconnected from {self.name}")
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to stream"""
        self.disconnect()
        
        for attempt in range(self.reconnect_attempts):
            self.logger.warning(f"Reconnect attempt {attempt + 1}/{self.reconnect_attempts}")
            
            if self.connect():
                self.error_count = 0
                return True
            
            time.sleep(self.reconnect_delay)
        
        self.logger.error(f"Failed to reconnect after {self.reconnect_attempts} attempts")
        return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame from the stream"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                self.last_frame = frame
                self.frame_count += 1
                self.error_count = 0
                return frame
            else:
                self.error_count += 1
                
                # Try reconnect after 5 consecutive errors
                if self.error_count >= 5:
                    self.logger.warning("Multiple read errors, attempting reconnect...")
                    if not self.reconnect():
                        return None
                
                # Return last known good frame
                return self.last_frame
                
        except Exception as e:
            self.logger.error(f"Frame read error: {e}")
            self.error_count += 1
            return self.last_frame
    
    def capture_loop(self):
        """Main capture loop (runs in separate thread)"""
        self.logger.info(f"Starting capture loop for {self.name}")
        
        while self.running:
            frame = self.read_frame()
            
            if frame is not None:
                # Add metadata
                frame_data = {
                    'frame': frame,
                    'camera': self.camera_type,
                    'timestamp': time.time(),
                    'frame_id': self.frame_count
                }
                
                # Try to put in queue (non-blocking)
                try:
                    self.frame_queue.put(frame_data, block=False)
                    self.metrics.record_frame(self.camera_type)
                except Full:
                    # Queue full, drop frame
                    pass
            else:
                # No frame, sleep a bit
                time.sleep(0.1)
        
        self.logger.info(f"Capture loop stopped for {self.name}")
    
    def start(self) -> bool:
        """Start the camera capture"""
        if self.running:
            self.logger.warning("Camera already running")
            return True
        
        # Connect first
        if not self.connect():
            return False
        
        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()
        
        self.logger.success(f"Camera {self.name} started")
        return True
    
    def stop(self):
        """Stop the camera capture"""
        self.logger.info(f"Stopping {self.name}...")
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=5)
        
        self.disconnect()
        self.logger.success(f"Camera {self.name} stopped")
    
    def is_healthy(self) -> bool:
        """Check if camera is operating normally"""
        return (
            self.running and
            self.cap is not None and
            self.cap.isOpened() and
            self.error_count < 10
        )
    
    def get_stats(self) -> dict:
        """Get camera statistics"""
        return {
            'name': self.name,
            'type': self.camera_type,
            'running': self.running,
            'healthy': self.is_healthy(),
            'frames_captured': self.frame_count,
            'error_count': self.error_count,
            'connected': self.cap is not None and self.cap.isOpened()
        }
