@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title BUT QDB selective downloader
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Download_BUTQDB_Selected_Records.ps1" %*
set "BUTQ_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%BUTQ_EXIT_CODE%"=="0" (
    echo BUT QDB download finished successfully.
) else (
    echo BUT QDB download stopped with error code %BUTQ_EXIT_CODE%.
)
echo.
pause
exit /b %BUTQ_EXIT_CODE%
