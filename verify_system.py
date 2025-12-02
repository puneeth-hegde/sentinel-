"""
SENTINEL v5.0 - System Verification Script
Checks that everything is installed and configured correctly
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("\n[1/10] Checking Python version...")
    major, minor = sys.version_info[:2]
    
    if major >= 3 and minor >= 10:
        print(f"  ✓ Python {major}.{minor} (OK)")
        return True
    else:
        print(f"  ✗ Python {major}.{minor} (Need 3.10+)")
        return False


def check_imports():
    """Check critical imports"""
    print("\n[2/10] Checking critical imports...")
    
    modules = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'insightface': 'InsightFace',
        'ultralytics': 'YOLOv8',
        'pyttsx3': 'Text-to-Speech',
        'loguru': 'Logging',
        'omegaconf': 'Configuration',
        'scipy': 'SciPy',
        'numpy': 'NumPy'
    }
    
    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability"""
    print("\n[3/10] Checking CUDA/GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available")
            print(f"  ✓ GPU: {gpu_name}")
            return True
        else:
            print("  ✗ CUDA not available")
            print("  → System will run on CPU (SLOW!)")
            return False
            
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def check_project_structure():
    """Check project directory structure"""
    print("\n[4/10] Checking project structure...")
    
    required_dirs = [
        'config',
        'src',
        'src/utils',
        'dataset',
        'logs',
        'snapshots'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            all_ok = False
    
    return all_ok


def check_config_file():
    """Check configuration file"""
    print("\n[5/10] Checking configuration...")
    
    config_path = Path('config/config.yaml')
    
    if not config_path.exists():
        print("  ✗ config/config.yaml - NOT FOUND")
        return False
    
    print("  ✓ config/config.yaml exists")
    
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        
        # Check critical settings
        checks = [
            ('cameras.gate.url', "Gate camera URL"),
            ('cameras.door.url', "Door camera URL"),
            ('detection.confidence', "Detection confidence"),
            ('face.threshold', "Face threshold"),
            ('users.authorized', "Authorized users")
        ]
        
        all_ok = True
        for key, name in checks:
            try:
                value = OmegaConf.select(config, key)
                if value is not None:
                    print(f"  ✓ {name}: {value}")
                else:
                    print(f"  ✗ {name} - NOT SET")
                    all_ok = False
            except Exception:
                print(f"  ✗ {name} - ERROR")
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def check_source_files():
    """Check source files exist"""
    print("\n[6/10] Checking source files...")
    
    required_files = [
        'src/coordinator.py',
        'src/camera_manager.py',
        'src/detection_pipeline.py',
        'src/face_recognition.py',
        'src/audio_manager.py',
        'src/alert_engine.py',
        'src/cross_camera_tracker.py',
        'src/bytetrack.py',
        'src/utils/logger.py',
        'src/utils/metrics.py'
    ]
    
    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_ok = False
    
    return all_ok


def check_dataset():
    """Check dataset directory"""
    print("\n[7/10] Checking dataset...")
    
    dataset_path = Path('dataset')
    
    if not dataset_path.exists():
        print("  ✗ dataset/ directory not found")
        return False
    
    # Check for person directories
    person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(person_dirs) == 0:
        print("  ✗ No person directories in dataset/")
        print("  → Add your face images: dataset/your_name/*.jpg")
        return False
    
    print(f"  ✓ Found {len(person_dirs)} person(s):")
    for person_dir in person_dirs:
        image_count = len(list(person_dir.glob('*')))
        print(f"    - {person_dir.name}: {image_count} images")
    
    return True


def check_face_database():
    """Check face database"""
    print("\n[8/10] Checking face database...")
    
    db_path = Path('face_database.pkl')
    
    if not db_path.exists():
        print("  ✗ face_database.pkl not found")
        print("  → Run: python train_faces.py")
        return False
    
    try:
        import pickle
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
        
        print(f"  ✓ face_database.pkl exists")
        print(f"  ✓ Contains {len(database)} person(s):")
        for name, embeddings in database.items():
            print(f"    - {name}: {len(embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading database: {e}")
        return False


def check_yolo_model():
    """Check YOLO model"""
    print("\n[9/10] Checking YOLO model...")
    
    try:
        from ultralytics import YOLO
        
        print("  → Downloading YOLOv8n model (first time only)...")
        model = YOLO('yolov8n.pt')
        
        print("  ✓ YOLOv8n model loaded")
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading YOLO: {e}")
        return False


def check_insightface_model():
    """Check InsightFace model"""
    print("\n[10/10] Checking InsightFace model...")
    
    try:
        from insightface.app import FaceAnalysis
        
        print("  → Loading buffalo_l model (first time may take a minute)...")
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0)
        
        print("  ✓ InsightFace buffalo_l model loaded")
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading InsightFace: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 80)
    print("SENTINEL v5.0 - SYSTEM VERIFICATION")
    print("=" * 80)
    print("\nThis will check if everything is installed and configured correctly.")
    print("Please wait, this may take 1-2 minutes...")
    
    results = []
    
    results.append(("Python version", check_python_version()))
    results.append(("Required packages", check_imports()))
    results.append(("CUDA/GPU", check_cuda()))
    results.append(("Project structure", check_project_structure()))
    results.append(("Configuration", check_config_file()))
    results.append(("Source files", check_source_files()))
    results.append(("Dataset", check_dataset()))
    results.append(("Face database", check_face_database()))
    results.append(("YOLO model", check_yolo_model()))
    results.append(("InsightFace model", check_insightface_model()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED! System is ready to run!")
        print("\nNext steps:")
        print("  1. Review config/config.yaml (camera URLs, authorized users)")
        print("  2. Run: python main.py")
        print("  3. Test with demo scenarios")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED!")
        print("\nPlease fix the failed checks before running the system.")
        print("See INSTALLATION_GUIDE.md for help.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
