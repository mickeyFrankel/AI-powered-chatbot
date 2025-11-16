#!/bin/bash
# Start backend and frontend

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Chatbot Servers...${NC}\n"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Check if backend dependencies installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
fi

# Check if frontend dependencies installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Start backend
echo -e "\n${BLUE}Starting Backend (FastAPI on :8000)...${NC}"
python3 api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting Frontend (React on :3000)...${NC}\n"
cd frontend && npm run dev &
FRONTEND_PID=$!

# Trap Ctrl+C to kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

echo -e "\n${GREEN}✅ Servers running:${NC}"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo -e "\nPress Ctrl+C to stop both servers\n"

# Wait for both processes
wait
