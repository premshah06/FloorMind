#!/bin/bash

# FloorMind Backend Startup Script

echo "🏗️  Starting FloorMind Backend..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Start the server
echo ""
echo "🚀 Starting server..."
python backend/api/app.py
