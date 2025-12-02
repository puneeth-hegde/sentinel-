"""
SENTINEL v5.0 - Main Entry Point
Start the complete security system
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.coordinator_complete import SentinelCoordinator


def main():
    """Main entry point"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              SENTINEL v5.0 - AI Security System              ║
    ║                                                              ║
    ║          Face Recognition | Person Tracking | Alerts         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Create coordinator
        coordinator = SentinelCoordinator(config_path="config/config.yaml")
        
        # Run system
        coordinator.run()
        
    except KeyboardInterrupt:
        print("\n\n[!] Shutting down...")
    except Exception as e:
        print(f"\n\n[ERROR] System error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
