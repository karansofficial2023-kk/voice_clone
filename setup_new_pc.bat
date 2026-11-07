@echo off
echo ============================================================
echo   CosyVoice 2 & Dhee Pravega-TTS - 1-Click Setup for New PC
echo ============================================================
echo.

cd /d "%~dp0"

:: Check if conda is installed
call conda --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda is not found in PATH!
    echo Please install Miniforge or Miniconda first:
    echo   https://github.com/conda-forge/miniforge/releases
    pause
    exit /b 1
)

echo [1/5] Creating Conda environment: cosyvoice2 (Python 3.10)...
call conda create -n cosyvoice2 python=3.10 -y

echo.
echo [2/5] Activating cosyvoice2 environment...
call conda activate cosyvoice2

echo.
echo [3/5] Installing PyTorch with CUDA 12.1 support...
call pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4/5] Installing dependencies from requirements.txt...
call pip install -r requirements.txt

echo.
echo [5/5] Downloading models (CosyVoice 2-0.5B + Dhee Pravega-TTS Indic Engine)...
python download_all_models.py

echo.
echo ============================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo   Launching WebUI on http://localhost:50000 ...
echo ============================================================
python webui.py --port 50000
pause
