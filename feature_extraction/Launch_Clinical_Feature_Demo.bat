@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo The feature_extraction virtual environment was not found.
  echo Create it and install the package before launching this demo.
  pause
  exit /b 1
)

echo Starting ECG Cascade Clinical Feature Demo...
echo The browser should open automatically at http://127.0.0.1:8765/
echo Keep this window open while presenting. Press Ctrl+C here to stop.
echo.

".venv\Scripts\python.exe" -m ecg_cascade.clinical_demo_web
if errorlevel 1 pause
