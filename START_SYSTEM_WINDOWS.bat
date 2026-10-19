@echo off
title ANN Inventory Control System
cd /d "%~dp0"
echo ==================================================
echo   ANN Inventory Control System - Starting
echo ==================================================
where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python was not found on this computer.
  echo Install Python 3.10 or newer from https://www.python.org/downloads
  echo and tick "Add python.exe to PATH" during installation. Then run this file again.
  echo.
  pause
  exit /b
)
if not exist venv\Scripts\python.exe (
  echo Creating a private Python environment for the system...
  python -m venv venv
)
if not exist venv\installed.txt (
  echo Installing the required libraries. This happens only once and needs internet...
  venv\Scripts\python.exe -m pip install --upgrade pip
  venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installation of the libraries failed. Check your internet connection and run this file again.
    pause
    exit /b
  )
  echo done> venv\installed.txt
)
venv\Scripts\python.exe app.py
pause
