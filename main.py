"""
SENTINEL v5.0 - Main Entry Point
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'utils'))

from coordinator_complete import SentinelCoordinator


def print_banner():
    """Print system banner"""
    print()
    print("    ╔══════════════════════════════════════════════════════════════╗")
    print("    ║                                                              ║")
    print("    ║              SENTINEL v5.0 - AI Security System              ║")
    print("    ║                         FIXED VERSION                        ║")
    print("    ║          Face Recognition | Person Tracking | Alerts         ║")
    print("    ║                                                              ║")
    print("    ╚══════════════════════════════════════════════════════════════╝")
    print()


def main():
    """Main entry point"""
    print_banner()
    
    try:
        coordinator = SentinelCoordinator(config_path="config/config.yaml")
        coordinator.run()
    
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()