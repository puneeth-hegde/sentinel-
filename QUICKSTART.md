# ⚡ SENTINEL v5.0 - Quick Start Guide

## 🎯 Complete Setup in 10 Commands

### 1️⃣ Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/SENTINEL_v5.git
cd SENTINEL_v5
```

### 2️⃣ Run Auto-Install (Windows)
```bash
install.bat
```

**OR Manual Install:**
```bash
conda create -n sentinel python=3.10 -y
conda activate sentinel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python verify_system.py
```

### 3️⃣ Configure Your Cameras
```bash
notepad config\config.yaml
```
Change:
- Line 17: Gate camera URL
- Line 24: Door camera URL
- Line 177: Your name

### 4️⃣ Add Your Face Images
```bash
mkdir dataset\your_name
# Copy 10-20 photos of your face to dataset\your_name\
```

### 5️⃣ Train Face Database
```bash
python train_faces.py
```

### 6️⃣ Run System
```bash
python main.py
```

## ✅ Verification Checklist

- [ ] CUDA shows True in verify_system.py
- [ ] Both cameras appear in display
- [ ] FPS shows 12-15
- [ ] You are recognized (green box)
- [ ] Audio plays when you appear
- [ ] Unknown person shows yellow box

## 🎮 Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| E | Enroll new user |
| C | Cancel enrollment |

## 🎬 Test Scenarios

### Scenario 1: Authorized User (YOU)
1. Walk past gate camera
2. Walk to door camera
3. **Expected:** Green box + "Welcome home, [name]"

### Scenario 2: Unknown Person
1. Have friend approach door
2. **Expected:** Yellow "Unknown" + alert audio

### Scenario 3: Enroll New User
1. Unknown person at door
2. Press 'E' key
3. Enter name
4. System captures 10 images
5. **Expected:** User enrolled, recognized next time

## 🐛 Quick Fixes

### Low FPS?
Edit config/config.yaml:
```yaml
performance:
  process_width: 480
  process_height: 270
```

### Not Recognizing You?
Edit config/config.yaml:
```yaml
face:
  threshold: 0.40  # Was 0.35
```

### Camera Won't Connect?
1. Test URL in VLC player
2. Check network
3. Increase timeouts in config

## 📊 What Success Looks Like

```
╔══════════════════════════════════════╗
║   SENTINEL v5.0 - AI Security System ║
╚══════════════════════════════════════╝

[SUCCESS] Connected to Gate Camera
[SUCCESS] Connected to Door Camera
[SUCCESS] ★ SENTINEL v5.0 COMPLETE - All features enabled! ★

FPS: Gate=15.2 Door=16.1
CPU: 62% | RAM: 4.1GB | GPU: 2.3GB
Detections: 2 | Alerts: 1

✓ Face Rec ✓ Pose ✓ Weapons ✓ Enroll ✓ Cross-Cam
```

## 🎯 Total Time

- **Install:** 15-20 minutes
- **Configure:** 5 minutes
- **Train:** 2 minutes
- **Test:** 5 minutes
- **TOTAL:** ~30 minutes

## 🆘 Need Help?

1. Check `logs/sentinel_*.log`
2. Run `python verify_system.py`
3. Read `INSTALL.md`
4. Check camera URLs in config

---

**You're ready! Start building!** 🚀
