# 🚀 SENTINEL v5.0 - Installation Guide

## ⚡ Quick Install (30 minutes)

### Step 1: System Requirements

**Hardware:**
- NVIDIA GPU (GTX 1650 or better)
- 16GB RAM
- 50GB free disk space

**Software:**
- Windows 10/11
- Anaconda/Miniconda
- CUDA 11.8
- Git

### Step 2: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SENTINEL_v5.git
cd SENTINEL_v5
```

### Step 3: Create Environment

```bash
conda create -n sentinel python=3.10 -y
conda activate sentinel
```

### Step 4: Install PyTorch (CRITICAL!)

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

**Verify CUDA:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
Must print: `CUDA: True`

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
python verify_system.py
```

All checks should pass!

### Step 7: Configure Cameras

Edit `config/config.yaml`:
```yaml
cameras:
  gate:
    url: "rtsp://YOUR_GATE_CAMERA_URL"
  door:
    url: "rtsp://YOUR_DOOR_CAMERA_URL"

users:
  authorized:
    - your_name  # Change this!
```

### Step 8: Prepare Dataset

```bash
# Create your face image folder
mkdir dataset\your_name

# Copy 10-20 face images to dataset\your_name\
```

### Step 9: Train Face Database

```bash
python train_faces.py
```

Should create `face_database.pkl`.

### Step 10: Run System

```bash
python main.py
```

## 🎯 Expected Output

```
╔══════════════════════════════════════╗
║   SENTINEL v5.0 - AI Security System ║
╚══════════════════════════════════════╝

[SUCCESS] Connected to Gate Camera
[SUCCESS] Connected to Door Camera
[INFO] Loading YOLO model...
[INFO] Loading InsightFace...
[SUCCESS] ★ SENTINEL v5.0 COMPLETE - All features enabled! ★
[SUCCESS] SENTINEL v5.0 is now running!

Controls:
  Q - Quit
  E - Enroll new user
  C - Cancel enrollment
```

## 🐛 Troubleshooting

### CUDA not available
```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### ImportError
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Camera won't connect
1. Test RTSP URL in VLC player
2. Check camera is on same network
3. Verify credentials in URL

### Low FPS
Edit `config/config.yaml`:
```yaml
performance:
  process_width: 480
  process_height: 270
```

## ✅ Installation Complete!

Test with:
```bash
# Scenario 1: Walk gate → door (you should be recognized)
# Scenario 2: Have friend approach (should show "Unknown")
# Scenario 3: Press 'E' to enroll new user
```

---

**Total time: ~30 minutes**
