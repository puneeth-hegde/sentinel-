"""
SENTINEL v5.0 - Centralized Logging System
Uses Loguru for structured, colored, file-based logging
"""

import sys
from pathlib import Path
from loguru import logger
from datetime import datetime


class SentinelLogger:
    """Centralized logging system for all SENTINEL components"""
    
    def __init__(self, config):
        self.config = config
        self.log_dir = Path(config.system.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add console logger with colors
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
            level=config.system.log_level,
            colorize=True
        )
        
        # Add file logger (detailed)
        log_file = self.log_dir / f"sentinel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="7 days",
            compression="zip"
        )
        
        # Add error-only log file
        error_log = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            error_log,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="50 MB",
            retention="30 days"
        )
        
        self.logger = logger
        
    def get_logger(self, component_name: str):
        """Get a logger for a specific component"""
        return self.logger.bind(name=component_name)
    
    def log_config(self):
        """Log the current configuration"""
        self.logger.info("=" * 80)
        self.logger.info(f"SENTINEL v{self.config.system.version} - STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"Log Level: {self.config.system.log_level}")
        self.logger.info(f"GPU Device: {self.config.detection.device}")
        self.logger.info(f"Face Recognition Threshold: {self.config.face.threshold}")
        self.logger.info(f"Detection Confidence: {self.config.detection.confidence}")
        self.logger.info(f"Target FPS: {self.config.performance.target_fps}")
        self.logger.info(f"Authorized Users: {', '.join(self.config.users.authorized)}")
        self.logger.info("=" * 80)


def get_logger(name: str):
    """Quick helper to get a logger by name"""
    return logger.bind(name=name)
