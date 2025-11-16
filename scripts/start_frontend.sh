#!/bin/bash
# Start the React frontend

cd "$(dirname "$0")"

echo "🎨 Starting Frontend..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the dev server
npm run dev
