@echo off
REM FloorMind Backend Startup Script for Windows
REM This script starts the FloorMind backend server with your fine-tuned SDXL model

echo 🏗️  Starting FloorMind Backend with SDXL Model
echo ==============================================
echo.

REM Check if .env exists, if not create from example
if not exist .env (
    echo 📝 Creating .env file from .env.example...
    copy .env.example .env
    echo ✅ .env file created. Please edit it if needed.
    echo.
)

REM Check if virtual environment exists
if not exist venv (
    echo ⚠️  Virtual environment not found.
    echo Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created.
    echo.
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo 📦 Installing dependencies...
pip install -q -r requirements.txt

echo.
echo 🚀 Starting FloorMind Backend Server...
echo    Model: .\models\floormind_sdxl_finetuned
echo    Port: 5001
echo    GPU: Enabled (if available)
echo.

REM Start the backend
cd backend
python api\app.py
