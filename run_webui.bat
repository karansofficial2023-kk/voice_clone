@echo off
echo ============================================================
echo   Launching CosyVoice 2 WebUI on http://127.0.0.1:50000
echo ============================================================
echo.

cd /d "%~dp0"
set PYTHONPATH=third_party\Matcha-TTS
set PYTHONIOENCODING=utf-8

:: Try activating cosyvoice2 environment
call conda activate cosyvoice2 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    :: Try direct path if conda not in PATH
    if exist "%USERPROFILE%\miniforge3\envs\cosyvoice2\python.exe" (
        "%USERPROFILE%\miniforge3\envs\cosyvoice2\python.exe" webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
        goto end
    )
    if exist "%USERPROFILE%\miniconda3\envs\cosyvoice2\python.exe" (
        "%USERPROFILE%\miniconda3\envs\cosyvoice2\python.exe" webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
        goto end
    )
    if exist "C:\miniforge3\envs\cosyvoice2\python.exe" (
        "C:\miniforge3\envs\cosyvoice2\python.exe" webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
        goto end
    )
)

python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B

:end
pause

