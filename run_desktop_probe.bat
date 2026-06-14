@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================
echo     AISpeechApp - Desktop Demo Probe
echo ============================================
echo.

set PYTHON=%~dp0.venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    echo.
    echo Create the project virtual environment first:
    echo   powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
    echo.
    pause
    exit /b 1
)

echo Running AISpeechApp desktop probe...
echo.
"%PYTHON%" -m aispeechapp.desktop_demo_probe --output-dir outputs\desktop_demo %*

if errorlevel 1 (
    echo.
    echo Probe finished with failures. Check output above.
    pause
)
endlocal
