@echo off
setlocal
cd /d "%~dp0"

echo.
echo ============================================
echo     AISpeechApp - Visible Desktop Demo
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

echo Launching AISpeechApp visible demo...
echo.
"%PYTHON%" -m aispeechapp.gui --demo %*

if errorlevel 1 (
    echo.
    echo Demo exited with an error. Check messages above.
    pause
)
endlocal
