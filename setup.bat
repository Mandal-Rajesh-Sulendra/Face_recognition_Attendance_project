@echo off
echo =============================================
echo  Face Recognition Attendance System - Setup
echo =============================================
echo.

echo [1/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/5] Installing cmake (needed for dlib)...
pip install cmake

echo.
echo [3/5] Installing dlib (pre-compiled wheel)...
pip install dlib

echo.
echo [4/5] Installing face_recognition and other dependencies...
pip install face_recognition opencv-python numpy pandas openpyxl Pillow

echo.
echo [5/5] Creating project folders...
if not exist dataset mkdir dataset
if not exist encodings mkdir encodings
if not exist attendance mkdir attendance

echo.
echo =============================================
echo  Setup Complete! Run: python main.py
echo =============================================
pause
