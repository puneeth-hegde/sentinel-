@echo off
echo ========================================
echo SENTINEL v5.0 - Quick Install
echo ========================================
echo.

echo [1/5] Creating conda environment...
call conda create -n sentinel python=3.10 -y
if errorlevel 1 goto error

echo.
echo [2/5] Activating environment...
call conda activate sentinel
if errorlevel 1 goto error

echo.
echo [3/5] Installing PyTorch with CUDA...
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto error

echo.
echo [4/5] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [5/5] Verifying installation...
python verify_system.py
if errorlevel 1 goto error

echo.
echo ========================================
echo ✓ Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit config/config.yaml with your camera URLs
echo 2. Copy face images to dataset/your_name/
echo 3. Run: python train_faces.py
echo 4. Run: python main.py
echo.
pause
exit /b 0

:error
echo.
echo ✗ Installation failed!
echo Please check the error messages above.
pause
exit /b 1
