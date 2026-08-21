@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "VIEWER_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%VIEWER_PYTHON%" (
  echo The project Python environment was not found:
  echo %VIEWER_PYTHON%
  echo.
  echo Create it with the setup commands in README.md, then run this launcher again.
  pause
  exit /b 1
)

cd /d "%PROJECT_DIR%"
"%VIEWER_PYTHON%" -m ecg_cascade.rpeak_viewer %*
if errorlevel 1 (
  echo.
  echo The reviewer closed after an error.
  pause
)
endlocal
