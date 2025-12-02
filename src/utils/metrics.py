"""
SENTINEL v5.0 - Performance Metrics Tracker
Real-time FPS, latency, and system health monitoring
"""

import time
import psutil
from collections import deque
from typing import Dict, Optional
import threading


class PerformanceMetrics:
    """Track system performance metrics"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        
        # FPS tracking
        self.frame_times = {
            'gate': deque(maxlen=window_size),
            'door': deque(maxlen=window_size),
            'display': deque(maxlen=window_size)
        }
        
        # Component latencies
        self.latencies = {
            'detection_gate': deque(maxlen=window_size),
            'detection_door': deque(maxlen=window_size),
            'face_recognition': deque(maxlen=window_size),
            'tracking': deque(maxlen=window_size)
        }
        
        # Counters
        self.counters = {
            'total_frames': 0,
            'detections': 0,
            'recognitions': 0,
            'alerts': 0,
            'errors': 0
        }
        
        # System stats
        self.cpu_percent = 0
        self.memory_percent = 0
        self.gpu_memory = 0
        
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def record_frame(self, camera: str):
        """Record a frame timestamp"""
        with self.lock:
            self.frame_times[camera].append(time.time())
            self.counters['total_frames'] += 1
    
    def record_latency(self, component: str, duration: float):
        """Record component processing latency"""
        with self.lock:
            if component in self.latencies:
                self.latencies[component].append(duration)
    
    def increment_counter(self, counter: str):
        """Increment a counter"""
        with self.lock:
            if counter in self.counters:
                self.counters[counter] += 1
    
    def get_fps(self, camera: str) -> float:
        """Calculate FPS for a camera"""
        with self.lock:
            times = self.frame_times.get(camera, deque())
            if len(times) < 2:
                return 0.0
            
            time_diff = times[-1] - times[0]
            if time_diff == 0:
                return 0.0
            
            return (len(times) - 1) / time_diff
    
    def get_avg_latency(self, component: str) -> float:
        """Get average latency for a component"""
        with self.lock:
            lats = self.latencies.get(component, deque())
            if not lats:
                return 0.0
            return sum(lats) / len(lats)
    
    def update_system_stats(self):
        """Update CPU/memory stats"""
        try:
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            self.memory_percent = psutil.virtual_memory().percent
            
            # Try to get GPU memory (if nvidia-smi available)
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            except:
                pass
        except Exception:
            pass
    
    def get_summary(self) -> Dict:
        """Get complete metrics summary"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                'uptime': uptime,
                'fps': {
                    'gate': self.get_fps('gate'),
                    'door': self.get_fps('door'),
                    'display': self.get_fps('display')
                },
                'latency': {
                    'detection_gate': self.get_avg_latency('detection_gate'),
                    'detection_door': self.get_avg_latency('detection_door'),
                    'face_recognition': self.get_avg_latency('face_recognition'),
                    'tracking': self.get_avg_latency('tracking')
                },
                'counters': self.counters.copy(),
                'system': {
                    'cpu': self.cpu_percent,
                    'memory': self.memory_percent,
                    'gpu_memory': self.gpu_memory
                }
            }
    
    def get_display_text(self) -> str:
        """Get formatted text for display overlay"""
        summary = self.get_summary()
        
        text = [
            f"FPS: Gate={summary['fps']['gate']:.1f} Door={summary['fps']['door']:.1f}",
            f"CPU: {summary['system']['cpu']:.1f}% | RAM: {summary['system']['memory']:.1f}%",
            f"Detections: {summary['counters']['detections']} | Alerts: {summary['counters']['alerts']}"
        ]
        
        if summary['system']['gpu_memory'] > 0:
            text.insert(1, f"GPU: {summary['system']['gpu_memory']:.2f} GB")
        
        return "\n".join(text)


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, metrics: PerformanceMetrics, component: str):
        self.metrics = metrics
        self.component = component
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start
        self.metrics.record_latency(self.component, duration)
