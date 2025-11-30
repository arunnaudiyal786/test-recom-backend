#!/bin/bash

# Development startup script for EIP Dashboard
# Starts both the Python FastAPI backend and Next.js frontend

set -e

echo "🚀 Starting EIP Intelligent Ticket Management Dashboard"
echo "================================================"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create a .env file with your OpenAI API key"
    echo "Example: cp .env.example .env"
    exit 1
fi

# Check if FAISS index exists
if [ ! -f data/faiss_index/tickets.index ]; then
    echo "⚠️  Warning: FAISS index not found"
    echo "You may need to run: python3 scripts/setup_vectorstore.py"
    echo ""
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    echo "Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# Check if Python dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check if Node dependencies are installed
if [ ! -d frontend/node_modules ]; then
    echo "⚠️  Installing Node.js dependencies..."
    cd frontend && npm install && cd ..
fi

echo "✅ Pre-flight checks complete"
echo ""
echo "Starting services..."
echo "  📡 Backend API: http://localhost:8000"
echo "  🎨 Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Python backend in background
echo "Starting Python FastAPI backend..."
python3 api_server.py &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Start Next.js frontend in background
echo "Starting Next.js frontend..."
cd frontend && npm run dev &
FRONTEND_PID=$!

# Wait for both processes
wait
