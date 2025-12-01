#!/bin/bash

# Development startup script for Test Recommendation Backend
# Starts the Python FastAPI backend server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Test Recommendation Backend Server   "
echo "========================================"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
    if [ -f .env.example ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo "Please edit .env and add your OPENAI_API_KEY"
        echo ""
    else
        echo "Please create a .env file with your OpenAI API key"
        exit 1
    fi
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.11+ from https://python.org"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Check if FAISS index exists (warning only)
if [ ! -f data/faiss_index/tickets.index ]; then
    echo ""
    echo "Warning: FAISS index not found"
    echo "You may need to run: python3 scripts/setup_vectorstore.py"
    echo ""
fi

# Check if Python dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null || ! python3 -c "import pydantic_settings" 2>/dev/null; then
    echo ""
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
fi

echo ""
echo "Starting Backend API server..."
echo ""
echo "  Main API (LangGraph Orchestrated):"
echo "    /api/process-ticket    Full pipeline with streaming"
echo "    /api/preview-search    Search tuning preview"
echo "    /api/output            Get last result"
echo "    /api/health            Health check"
echo ""
echo "  Component Endpoints (v2):"
echo "    /v2/embedding/*        Embedding generation"
echo "    /v2/retrieval/*        Similar ticket search"
echo "    /v2/classification/*   Domain classification"
echo "    /v2/labeling/*         Label assignment"
echo ""
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the backend server
python3 api_server.py
