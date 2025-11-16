#!/bin/bash
# Comprehensive Backend Startup Script
# Handles dependencies, paths, and starts the API server

set -e  # Exit on error

echo "🚀 Starting Chatbot Backend"
echo "=========================="
echo ""

# Change to project root
cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install missing dependency if needed
echo "🔍 Checking dependencies..."
if ! python3 -c "import multipart" 2>/dev/null; then
    echo "   Installing python-multipart..."
    pip install python-multipart --quiet
    echo "   ✅ python-multipart installed"
else
    echo "   ✅ All dependencies present"
fi

# Check if database exists
DB_PATH="databases/vector/chroma_db"
if [ ! -d "$DB_PATH" ]; then
    echo ""
    echo "⚠️  Warning: Database not found at $DB_PATH"
    echo "   You may need to upload data via the UI first."
fi

echo ""
echo "🌐 Starting API server on http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

# Run API server from backend/core
cd backend/core
python3 api.py
