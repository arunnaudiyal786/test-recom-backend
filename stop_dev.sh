#!/bin/bash

# Stop script for Test Recommendation Backend
# Kills the Python FastAPI backend server on port 8000

echo "========================================"
echo "  Stopping Backend Server (Port 8000)  "
echo "========================================"
echo ""

# Find and kill processes on port 8000
PIDS=$(lsof -ti:8000 2>/dev/null)

if [ -n "$PIDS" ]; then
    echo "Stopping backend processes: $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null
    echo "Backend server stopped successfully"
else
    echo "No backend server running on port 8000"
fi

echo ""
