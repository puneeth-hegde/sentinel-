# 📦 How to Upload SENTINEL to GitHub

## Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click "+" → "New repository"
3. Name: `SENTINEL_v5`
4. Description: "AI Security System with Face Recognition"
5. **Public** or **Private** (your choice)
6. ✅ Check "Add README" (we'll replace it)
7. Click "Create repository"

## Step 2: Download This Folder

You have this complete folder structure:
```
SENTINEL_v5_GITHUB/
├── config/
│   └── config.yaml
├── src/
│   ├── (13 Python files)
│   └── utils/
│       └── (2 Python files)
├── dataset/
├── logs/
├── snapshots/
├── .gitignore
├── README.md
├── INSTALL.md
├── QUICKSTART.md
├── requirements.txt
├── main.py
├── train_faces.py
├── verify_system.py
└── install.bat
```

## Step 3: Upload to GitHub

### Method A: GitHub Web Interface (Easiest)

1. Open your repository on GitHub
2. Click "Add file" → "Upload files"
3. Drag ALL files from SENTINEL_v5_GITHUB folder
4. Commit message: "Initial commit - Complete SENTINEL v5.0 system"
5. Click "Commit changes"

### Method B: Git Command Line

```bash
cd C:\projects\Major_Project
# (Assuming you saved files here)

# Initialize git
git init
git add .
git commit -m "Initial commit - Complete SENTINEL v5.0 system"

# Connect to GitHub
git remote add origin https://github.com/YOUR_USERNAME/SENTINEL_v5.git
git branch -M main
git push -u origin main
```

## Step 4: Clone to Your Machine

```bash
cd C:\projects\Major_Project
git clone https://github.com/YOUR_USERNAME/SENTINEL_v5.git
cd SENTINEL_v5
```

## Step 5: Install and Run

```bash
# Option 1: Auto install (Windows)
install.bat

# Option 2: Manual
conda create -n sentinel python=3.10 -y
conda activate sentinel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## ✅ Done!

Your repository is live and you can:
- Clone on any machine
- Share with professors
- Version control your changes
- Backup your work

---

**Repository URL Format:**
`https://github.com/YOUR_USERNAME/SENTINEL_v5`

Replace `YOUR_USERNAME` with your GitHub username!
