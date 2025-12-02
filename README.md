# 🛡️ SENTINEL v5.0 - AI Security System

**Complete dual-camera security system with face recognition, body language analysis, and weapon detection**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green)

## 🎯 Features

### Core Features
- ✅ Face Recognition (InsightFace, <3 sec)
- ✅ Person Detection & Tracking (YOLOv8 + ByteTrack)
- ✅ Cross-Camera Tracking
- ✅ Audio Alerts (TTS)
- ✅ Unknown Person Detection
- ✅ Loitering Detection
- ✅ Face Hidden Detection

### Advanced Features
- ✅ Body Language Analysis (MediaPipe)
- ✅ Weapon Detection (YOLO)
- ✅ User Enrollment (Press 'E')
- ✅ Threat Scoring
- ✅ Snapshot Management

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU (GTX 1650 or better)
- CUDA 11.8
- Python 3.10
- 16GB RAM

### Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/SENTINEL_v5.git
cd SENTINEL_v5

# 2. Create environment
conda create -n sentinel python=3.10 -y
conda activate sentinel

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. Verify installation
python verify_system.py
```

### Configuration

Edit `config/config.yaml`:
```yaml
cameras:
  gate:
    url: "YOUR_GATE_CAMERA_RTSP_URL"
  door:
    url: "YOUR_DOOR_CAMERA_RTSP_URL"

users:
  authorized:
    - your_name_here
```

### Train Face Database

```bash
# Copy your face images to dataset/your_name/
python train_faces.py
```

### Run System

```bash
python main.py
```

## 🎮 Controls

- **Q** - Quit
- **E** - Enroll new user
- **C** - Cancel enrollment

## 📊 Performance

- FPS: 12-15 (GTX 1650)
- Face Recognition: <3 seconds
- CPU: 60-75%
- GPU Memory: 2.5-3.0 GB

## 📖 Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Advanced Features](docs/ADVANCED_FEATURES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## 🎓 Demo

```bash
# Test scenarios
1. Walk gate → door (authorized user)
2. Unknown person at door
3. Press 'E' to enroll new user
4. Test body language detection
5. Test weapon detection
```

## 🐛 Troubleshooting

### CUDA not available
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Low FPS
Edit `config/config.yaml`:
```yaml
performance:
  process_width: 480
  process_height: 270
```

### Camera won't connect
Test RTSP URL in VLC player first.

## 📄 License

Educational/Academic Use - College Major Project

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- InsightFace
- MediaPipe by Google
- ByteTrack

---

**Built for emergency 1-day deployment | Optimized for GTX 1650**
