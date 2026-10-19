@echo off
title AI Handwritten Digit Recognizer
echo ======================================================
echo          AI Handwritten Digit Recognizer
echo ======================================================
echo.

:: Determine Python 3.13 launcher
py -3.13 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=py -3.13
) else (
    set PY_CMD=python
)

:: Fast check: Only run setup if model is missing
if not exist "model.keras" (
    if not exist "model.h5" (
        echo [First-time setup: installing dependencies...]
        %PY_CMD% -m pip install -r requirements.txt
        echo [Training neural network on MNIST dataset...]
        %PY_CMD% train.py
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Setup failed.
            pause
            exit /b 1
        )
    )
)

echo [Launching application in dedicated window...]
echo ======================================================
echo  * Opening in its own separate window.
echo  * Your other browser tabs will NOT be affected.
echo  * Click 'Stop Session' in the app to close cleanly.
echo ======================================================
echo.

%PY_CMD% launch.py
