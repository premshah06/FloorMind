#!/bin/bash

# FloorMind Backend Startup Script
# This script starts the FloorMind backend server with your fine-tuned SDXL model

echo "🏗️  Starting FloorMind Backend with SDXL Model"
echo "=============================================="
echo ""

# Check if .env exists, if not create from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it if needed."
    echo ""
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found."
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created."
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "🚀 Starting FloorMind Backend Server..."
echo "   Model: ./models/floormind_sdxl_finetuned"
echo "   Port: 5001"
echo "   GPU: Enabled (if available)"
echo ""

# Start the backend
cd backend
python api/app.py
