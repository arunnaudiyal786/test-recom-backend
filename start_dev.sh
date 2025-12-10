#!/bin/bash

# Development startup script for Test Recommendation Backend
# Starts the Python FastAPI backend server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Virtual environment is in parent directory (Code/.venv)
VENV_PYTHON="$(dirname "$SCRIPT_DIR")/.venv/bin/python3"
VENV_PIP="$(dirname "$SCRIPT_DIR")/.venv/bin/pip"

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

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at $(dirname "$SCRIPT_DIR")/.venv"
    echo "Please create it with: python3 -m venv $(dirname "$SCRIPT_DIR")/.venv"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION (using .venv)"

# Check if Python dependencies are installed
if ! "$VENV_PYTHON" -c "import fastapi" 2>/dev/null || ! "$VENV_PYTHON" -c "import pydantic_settings" 2>/dev/null; then
    echo ""
    echo "Installing Python dependencies..."
    "$VENV_PIP" install -r requirements.txt
fi

# Function to check and generate embeddings if needed
check_and_generate_embeddings() {
    local FAISS_INDEX="$SCRIPT_DIR/data/faiss_index/tickets.index"
    local FAISS_METADATA="$SCRIPT_DIR/data/faiss_index/metadata.json"
    local CATEGORY_EMBEDDINGS="$SCRIPT_DIR/data/metadata/category_embeddings.json"
    local HISTORICAL_CSV="$SCRIPT_DIR/data/raw/test_plan_historical.csv"

    local NEEDS_FAISS=false
    local NEEDS_CATEGORY=false

    # Check FAISS index (need both files)
    if [ ! -f "$FAISS_INDEX" ] || [ ! -f "$FAISS_METADATA" ]; then
        NEEDS_FAISS=true
    fi

    # Check category embeddings
    if [ ! -f "$CATEGORY_EMBEDDINGS" ]; then
        NEEDS_CATEGORY=true
    fi

    # If nothing needs to be generated, return early
    if [ "$NEEDS_FAISS" = false ] && [ "$NEEDS_CATEGORY" = false ]; then
        echo "  All embedding files present"
        return 0
    fi

    # Show first-time setup message
    echo ""
    echo "========================================"
    echo "  First-Time Setup: Generating Embeddings"
    echo "========================================"
    echo ""
    echo "  This is a one-time operation that creates vector stores"
    echo "  for the AI-powered ticket matching system."
    echo ""
    echo "  Estimated time: 1-2 minutes"
    echo "  Estimated cost: ~\$0.03 (OpenAI API)"
    echo ""

    # Generate FAISS index if needed
    if [ "$NEEDS_FAISS" = true ]; then
        # Check if historical data exists
        if [ ! -f "$HISTORICAL_CSV" ]; then
            echo "  Error: Historical tickets CSV not found at:"
            echo "         $HISTORICAL_CSV"
            echo ""
            echo "  Please ensure historical data exists before starting."
            exit 1
        fi

        echo "  [1/2] Generating FAISS vector store (historical embeddings)..."
        "$VENV_PYTHON" scripts/setup_vectorstore.py
        echo ""
    else
        echo "  [1/2] FAISS vector store already exists - skipping"
    fi

    # Generate category embeddings if needed
    if [ "$NEEDS_CATEGORY" = true ]; then
        echo "  [2/2] Generating category embeddings..."
        "$VENV_PYTHON" scripts/generate_category_embeddings.py
        echo ""
    else
        echo "  [2/2] Category embeddings already exist - skipping"
    fi

    echo "  Embedding generation complete!"
    echo ""
}

# Check and generate embeddings if needed
echo ""
echo "Checking embedding files..."
check_and_generate_embeddings

# Fix OpenMP duplicate library conflict (FAISS + NumPy on macOS)
export KMP_DUPLICATE_LIB_OK=TRUE

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
"$VENV_PYTHON" api_server.py
