#!/bin/bash

# Stop script for EIP Dashboard
# Kills both the Python FastAPI backend and Next.js frontend

echo "🛑 Stopping EIP Intelligent Ticket Management Dashboard"
echo "================================================"
echo ""

# Kill processes on ports 8000 and 3000
echo "Stopping backend (port 8000) and frontend (port 3000)..."
lsof -ti:8000 -ti:3000 | xargs kill -9 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ All services stopped successfully"
else
    echo "ℹ️  No services were running"
fi

echo ""
