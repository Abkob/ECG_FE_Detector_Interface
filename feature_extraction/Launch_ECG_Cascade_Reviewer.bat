@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "REVIEWER_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%REVIEWER_PYTHON%" (
  echo The project Python environment was not found:
  echo %REVIEWER_PYTHON%
  echo.
  echo Create it with the setup commands in README.md, then run this launcher again.
  pause
  exit /b 1
)

cd /d "%PROJECT_DIR%"
"%REVIEWER_PYTHON%" -m ecg_cascade.rpeak_viewer %*
if errorlevel 1 (
  echo.
  echo The ECG Cascade Branch Reviewer closed after an error.
  pause
)
endlocal
