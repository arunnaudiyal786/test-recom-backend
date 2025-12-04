# Backend Server Knowledge Transfer

This document provides everything you need to set up, run, and operate the **Test Recommendation Backend** - a FastAPI server powered by LangGraph that orchestrates a multi-agent pipeline for intelligent ticket processing.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [First-Time Setup](#first-time-setup)
3. [Starting the Backend Server](#starting-the-backend-server)
4. [Health & Status Checks](#health--status-checks)
5. [Processing Tickets](#processing-tickets)
6. [API Endpoints Reference](#api-endpoints-reference)
7. [Common Operations](#common-operations)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

If the environment is already set up, run these commands:

```bash
# Navigate to backend directory
cd test-recom-backend

# Start the server
python3 api_server.py

# Server will be available at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
```

---

## First-Time Setup

### 1. Install Dependencies

```bash
cd test-recom-backend
pip install -r requirements.txt
```

**Key dependencies:**
- `fastapi` + `uvicorn` - API server
- `langgraph` + `langchain` - Agent orchestration
- `openai` - LLM integration
- `faiss-cpu` - Vector similarity search
- `pandas` - Data processing

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-actual-key-here
```

**Environment variables explained:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `CLASSIFICATION_MODEL` | `gpt-4o-mini` | Model for domain classification |
| `RESOLUTION_MODEL` | `gpt-4o` | Model for resolution generation |
| `TOP_K_SIMILAR_TICKETS` | `20` | Number of similar tickets to retrieve |
| `CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for classification |
| `LABEL_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for label assignment |
| `API_PORT` | `8000` | Backend server port |

### 3. Generate Sample Historical Data

```bash
# Generate 100 sample historical tickets in CSV format
python3 scripts/generate_sample_csv_data.py

# Output: data/raw/historical_tickets.csv
```

### 4. Build the FAISS Vector Index

```bash
# Create embeddings and build the FAISS index
python3 scripts/setup_vectorstore.py

# Output files:
#   data/faiss/tickets.index (vector index)
#   data/faiss/metadata.json (ticket metadata)
```

**Note:** This step costs approximately $0.02 for 100 tickets (embedding API calls).

---

## Starting the Backend Server

### Option 1: Direct Python Execution

```bash
cd test-recom-backend
python3 api_server.py
```

### Option 2: Using Root Start Script (Includes Frontend)

```bash
# From project root
./start_dev.sh
```

### Option 3: Using Uvicorn Directly

```bash
cd test-recom-backend
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
🚀 Starting RRE API Server...
📡 Frontend: http://localhost:3000
🔧 API Docs: http://localhost:8000/docs
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Stopping the Server

```bash
# From project root
./stop_dev.sh

# Or manually kill the process
lsof -ti:8000 | xargs kill -9
```

---

## Health & Status Checks

### Check Server Health

```bash
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-04T10:30:00.123456",
  "service": "RRE Ticket Processing API"
}
```

### Check API Root (Available Endpoints)

```bash
curl http://localhost:8000/
```

### Check Component Health

```bash
# Embedding service
curl http://localhost:8000/v2/embedding/health

# Retrieval service (FAISS)
curl http://localhost:8000/v2/retrieval/health

# Classification service
curl http://localhost:8000/v2/classification/health

# Labeling service
curl http://localhost:8000/v2/labeling/health
```

### Get Current Configuration

```bash
curl http://localhost:8000/api/config
```

**Response shows active agents:**
```json
{
  "skip_domain_classification": true,
  "active_agents": [
    "Pattern Recognition Agent",
    "Label Assignment Agent",
    "Resolution Generation Agent"
  ]
}
```

### Check Vector Store Stats

```bash
curl http://localhost:8000/v2/retrieval/stats
```

---

## Processing Tickets

### Method 1: CLI Processing (Direct File)

Process a ticket from `input/current_ticket.json`:

```bash
cd test-recom-backend
python3 main.py
```

**Input file format (`input/current_ticket.json`):**
```json
{
  "ticket_id": "JIRA-TEST-001",
  "title": "Brief ticket title",
  "description": "Detailed description of the issue...",
  "priority": "High",
  "metadata": {
    "reported_by": "user@example.com",
    "environment": "production"
  }
}
```

**Output:**
- Session directory: `output/YYYYMMDD_HHMMSS_xxxxx/`
- JSON result: `output/latest/ticket_resolution.json`
- CSV export: `output/latest/ticket_results.csv`

### Method 2: API Processing (SSE Stream)

```bash
curl -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "JIRA-TEST-001",
    "title": "Database connection errors in production",
    "description": "Users experiencing intermittent connection failures to the main database. Error: Connection pool exhausted.",
    "priority": "High",
    "metadata": {"environment": "production", "affected_users": 150}
  }'
```

This returns a **Server-Sent Events (SSE)** stream with real-time agent updates.

### Method 3: Load Sample Ticket

```bash
# Load the sample ticket from input/current_ticket.json
curl http://localhost:8000/api/load-sample
```

### Get Processing Results

```bash
# Get the latest processed output
curl http://localhost:8000/api/output

# Download as CSV
curl http://localhost:8000/api/download-csv -o results.csv
```

---

## API Endpoints Reference

### Core Pipeline Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process-ticket` | Process ticket through full pipeline (SSE stream) |
| `GET` | `/api/load-sample` | Load sample ticket from file |
| `GET` | `/api/output` | Get latest processing results (JSON) |
| `GET` | `/api/download-csv` | Download results as CSV |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/config` | Get current configuration |
| `GET` | `/api/prompts` | Get LLM prompt templates |

### Search Tuning Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/preview-search` | Preview search with custom config |
| `POST` | `/api/save-search-config` | Save search configuration |
| `GET` | `/api/load-search-config` | Load saved search configuration |

### Session Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/sessions` | List all processing sessions |
| `GET` | `/api/sessions/{id}` | Get specific session details |
| `GET` | `/api/sessions/{id}/output` | Get session's final output |
| `GET` | `/api/sessions/{id}/agents/{name}` | Get specific agent output |
| `GET` | `/api/sessions/{id}/csv` | Download session's CSV |

### Component Endpoints (v2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v2/embedding/generate` | Generate embeddings for text |
| `POST` | `/v2/retrieval/search` | Search similar tickets |
| `GET` | `/v2/retrieval/stats` | Get vector store statistics |
| `POST` | `/v2/classification/classify` | Classify ticket domain |
| `POST` | `/v2/labeling/assign` | Assign labels to ticket |

### Interactive API Documentation

Open in browser: **http://localhost:8000/docs**

This provides:
- Swagger UI with all endpoints
- Try-it-out functionality
- Request/response schemas

---

## Common Operations

### Rebuild FAISS Index After Data Changes

```bash
cd test-recom-backend

# After modifying data/raw/historical_tickets.csv
python3 scripts/setup_vectorstore.py
```

### Run Data Ingestion Directly

```bash
python3 -m src.vectorstore.data_ingestion
```

### View Recent Sessions

```bash
curl http://localhost:8000/api/sessions?limit=10
```

### Reload Schema Configuration (Without Restart)

```bash
curl -X POST http://localhost:8000/api/reload-schema-config
```

### Test Classification Only

```bash
curl -X POST http://localhost:8000/v2/classification/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database error in production",
    "description": "Connection timeout issues affecting users"
  }'
```

### Test Similar Ticket Search

```bash
curl -X POST http://localhost:8000/v2/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database connection issues",
    "description": "Connection pool exhaustion during peak hours",
    "top_k": 5
  }'
```

---

## Configuration

### File Locations

| File | Purpose |
|------|---------|
| `.env` | Environment variables |
| `config/schema_config.yaml` | Domain/label definitions for UI |
| `config/search_config.json` | Saved search parameters |
| `input/current_ticket.json` | Sample input ticket |
| `data/raw/historical_tickets.csv` | Historical ticket data |
| `data/faiss/tickets.index` | FAISS vector index |
| `data/faiss/metadata.json` | Ticket metadata for retrieval |

### Agent Pipeline

The backend runs agents **sequentially** in this order:

```
1. Pattern Recognition Agent (FAISS search)
       ↓
2. Label Assignment Agent (binary classifiers)
       ↓
3. Resolution Generation Agent (Chain-of-Thought)
```

**Note:** Domain Classification Agent is disabled by default. Enable it in `src/orchestrator/workflow.py`:
```python
SKIP_DOMAIN_CLASSIFICATION = False
```

---

## Troubleshooting

### "FAISS index not found"

```bash
python3 scripts/setup_vectorstore.py
```

### "OpenAI API key not set"

```bash
# Ensure .env exists and has the key
cat .env | grep OPENAI_API_KEY
```

### "Port 8000 already in use"

```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use stop script from project root
./stop_dev.sh
```

### "No similar tickets found"

- Verify FAISS index exists: `ls data/faiss/`
- Check historical data: `wc -l data/raw/historical_tickets.csv`
- Rebuild index: `python3 scripts/setup_vectorstore.py`

### "Low classification confidence"

Check domain definitions in `src/prompts/classification_prompts.py` and ensure they match your ticket content.

### View Server Logs

The server logs to stdout by default. Look for:
- `📁 Session ID:` - New processing session started
- `💾 Saved output to:` - Results saved successfully
- `📊 Exported results to CSV:` - CSV export completed

### Check if Server is Running

```bash
lsof -i:8000
# or
curl -s http://localhost:8000/api/health
```

---

## Performance Notes

- **Processing time:** 8-12 seconds per ticket
- **Cost per ticket:** ~$0.07 (mostly resolution generation)
- **Embedding cost:** ~$0.0002 per ticket for index building

---

## Quick Reference Card

```bash
# Start server
cd test-recom-backend && python3 api_server.py

# Health check
curl http://localhost:8000/api/health

# Process ticket via API
curl -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_id":"T001","title":"Issue","description":"Details","priority":"High","metadata":{}}'

# Process ticket via CLI
python3 main.py

# Get results
curl http://localhost:8000/api/output

# Rebuild vector index
python3 scripts/setup_vectorstore.py

# API docs
open http://localhost:8000/docs
```
