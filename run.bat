@echo off
title Meme Pattern Detector

echo Starting Meme Pattern Detector...
echo =========================================

REM Check if the virtual environment exists by looking for the activate script
if not exist "venv\Scripts\activate.bat" (
    echo First time run detected! 
    echo Setting up the application for you...
    echo This might take a minute or two.
    echo.
    
    REM 1. Create the virtual environment
    python -m venv venv
    
    REM 2. Activate the environment
    call venv\Scripts\activate
    
    REM 3. Upgrade pip to ensure smooth installation
    python -m pip install --quiet --upgrade pip
    
    REM 4. Install the required libraries from the text file
    python -m pip install -r requirements.txt
    
    echo.
    echo Setup complete!
) else (
    REM If the environment already exists, just activate it
    call venv\Scripts\activate
)

echo =========================================
echo Launching the application...
python main.py

echo.
echo =========================================
echo Application closed.
pause