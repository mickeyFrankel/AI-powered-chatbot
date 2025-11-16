#!/bin/bash
# Start the complete API server

cd "$(dirname "$0")"

echo "🚀 Starting Backend API Server..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r ../requirements.txt
else
    source .venv/bin/activate
fi

# Start the server
python api_complete.py
