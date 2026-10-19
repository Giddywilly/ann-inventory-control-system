#!/bin/bash
cd "$(dirname "$0")"
echo "=================================================="
echo "  ANN Inventory Control System - Starting"
echo "=================================================="
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10 or newer from https://www.python.org/downloads and run this file again."
  exit 1
fi
if [ ! -x venv/bin/python ]; then
  echo "Creating a private Python environment for the system..."
  python3 -m venv venv || { echo "Could not create the environment. On Ubuntu run: sudo apt install python3-venv"; exit 1; }
fi
if [ ! -f venv/installed.txt ]; then
  echo "Installing the required libraries. This happens only once and needs internet..."
  venv/bin/python -m pip install --upgrade pip
  venv/bin/python -m pip install -r requirements.txt || { echo "Installation failed. Check your internet connection and run again."; exit 1; }
  echo done > venv/installed.txt
fi
venv/bin/python app.py
