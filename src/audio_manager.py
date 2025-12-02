"""
SENTINEL v5.0 - Audio Alert System
Text-to-speech with priority queue and cooldowns
"""

import time
import threading
from queue import PriorityQueue
from typing import Dict, Optional
import pyttsx3
from logger import get_logger


class AudioManager:
    """Manages text-to-speech audio alerts with priority and cooldowns"""
    
    # Priority levels (lower number = higher priority)
    PRIORITY_CRITICAL = 0  # Weapon detected
    PRIORITY_HIGH = 1      # Threat, face hidden
    PRIORITY_NORMAL = 2    # Welcome, unknown person
    PRIORITY_LOW = 3       # Loitering
    
    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics
        self.logger = get_logger("Audio")
        
        self.enabled = config.audio.enabled
        
        if not self.enabled:
            self.logger.warning("Audio system disabled in config")
            return
        
        # Initialize TTS engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', config.audio.rate)
            self.engine.setProperty('volume', config.audio.volume)
            self.logger.success("Audio engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize audio engine: {e}")
            self.enabled = False
            return
        
        # Message templates
        self.messages = config.audio.messages
        
        # Cooldown tracking
        self.cooldowns = config.audio.cooldowns
        self.last_played = {}  # message_type -> timestamp
        
        # Priority queue for alerts
        self.queue = PriorityQueue()
        self.playing = False
        self.running = False
        
        # Start worker thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
        self.logger.success("Audio manager started")
    
    def _worker(self):
        """Worker thread that processes audio queue"""
        self.running = True
        
        while self.running:
            try:
                # Get next alert (blocks until available)
                priority, timestamp, message_type, message_data = self.queue.get(timeout=1)
                
                # Check cooldown
                if not self._check_cooldown(message_type):
                    self.logger.debug(f"Skipping '{message_type}' - still in cooldown")
                    continue
                
                # Format message
                message = self._format_message(message_type, message_data)
                
                if message:
                    self.logger.info(f"Playing: '{message}' (priority: {priority})")
                    self._speak(message)
                    
                    # Update last played time
                    self.last_played[message_type] = time.time()
                    
            except Exception as e:
                # Silently ignore queue timeout (normal operation)
                if "Empty" not in str(type(e).__name__):
                    if self.running:  # Only log real errors
                        self.logger.error(f"Audio worker error: {e}")
            
            time.sleep(0.1)
    
    def _check_cooldown(self, message_type: str) -> bool:
        """Check if message type is still in cooldown"""
        if message_type not in self.last_played:
            return True
        
        cooldown = self.cooldowns.get(message_type, 0)
        elapsed = time.time() - self.last_played[message_type]
        
        return elapsed >= cooldown
    
    def _format_message(self, message_type: str, data: Dict) -> Optional[str]:
        """Format message with data"""
        template = self.messages.get(message_type)
        
        if not template:
            self.logger.warning(f"No template for message type: {message_type}")
            return None
        
        try:
            return template.format(**data)
        except KeyError as e:
            self.logger.error(f"Missing key in message data: {e}")
            return template
    
    def _speak(self, text: str):
        """Speak the text using TTS engine"""
        if not self.enabled:
            return
        
        try:
            self.playing = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.playing = False
        except Exception as e:
            self.logger.error(f"Speech error: {e}")
            self.playing = False
    
    def play_welcome(self, name: str):
        """Play welcome message for authorized user"""
        self.queue.put((
            self.PRIORITY_NORMAL,
            time.time(),
            'welcome',
            {'name': name}
        ))
    
    def play_unknown(self):
        """Play message for unknown person"""
        self.queue.put((
            self.PRIORITY_NORMAL,
            time.time(),
            'unknown',
            {}
        ))
    
    def play_face_hidden(self):
        """Request person to show face"""
        self.queue.put((
            self.PRIORITY_HIGH,
            time.time(),
            'face_hidden',
            {}
        ))
    
    def play_loitering(self, duration: int):
        """Alert about loitering"""
        self.queue.put((
            self.PRIORITY_LOW,
            time.time(),
            'loitering',
            {'duration': duration}
        ))
    
    def play_threat(self):
        """Alert about threat detected"""
        self.queue.put((
            self.PRIORITY_HIGH,
            time.time(),
            'threat_detected',
            {}
        ))
    
    def play_weapon(self):
        """Critical alert for weapon detection"""
        self.queue.put((
            self.PRIORITY_CRITICAL,
            time.time(),
            'weapon_detected',
            {}
        ))
    
    def play_break_in_alert(self):
        """CRITICAL: Break-in attempt detected"""
        self.queue.put((
            self.PRIORITY_CRITICAL,
            time.time(),
            'break_in_attempt',
            {}
        ))
    
    def play_critical_threat(self):
        """High-level security threat"""
        self.queue.put((
            self.PRIORITY_CRITICAL,
            time.time(),
            'critical_threat',
            {}
        ))
    
    def play_group_alert(self):
        """Group of people alert"""
        self.queue.put((
            self.PRIORITY_HIGH,
            time.time(),
            'group_detected',
            {}
        ))
    
    def play_rapid_movement(self):
        """Rapid suspicious movement"""
        self.queue.put((
            self.PRIORITY_HIGH,
            time.time(),
            'rapid_movement',
            {}
        ))
    
    def stop(self):
        """Stop the audio manager"""
        self.logger.info("Stopping audio manager...")
        self.running = False
        
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        
        if self.enabled and self.engine:
            try:
                self.engine.stop()
            except:
                pass
        
        self.logger.success("Audio manager stopped")
    
    def is_playing(self) -> bool:
        """Check if currently playing audio"""
        return self.playing
    
    def get_queue_size(self) -> int:
        """Get number of pending alerts"""
        return self.queue.qsize()
