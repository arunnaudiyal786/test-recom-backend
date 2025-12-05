# Backend Startup Guide

Quick reference for running the Test Recommendation Backend.

---

## Quick Start (TL;DR)

```bash
# First time only
pip install -r requirements.txt
cp .env.example .env          # Add your OPENAI_API_KEY
python3 scripts/setup_vectorstore.py

# Run the server
python3 api_server.py
```

Server: http://localhost:8000 | Docs: http://localhost:8000/docs

---

## 1. First-Time Setup

### Step 1: Install Dependencies

```bash
cd test-recom-backend
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your API key:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Build Vector Index

The system needs a FAISS index built from historical ticket data:

```bash
python3 scripts/setup_vectorstore.py
```

This reads from `data/raw/historical_tickets.csv` and creates the index in `data/faiss_index/`.

---

## 2. Running the Backend Server

### Start the Server

```bash
python3 api_server.py
```

Output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Verify It's Running

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{"status": "healthy", "service": "RRE Ticket Processing API"}
```

### Stop the Server

Press `Ctrl+C` or:
```bash
lsof -ti:8000 | xargs kill -9
```

---

## 3. Input Ticket Location

### Default Input File

Place your input ticket JSON at:
```
input/current_ticket.json
```

### Input File Format

```json
{
  "ticket_id": "JIRA-TEST-001",
  "title": "Brief title describing the issue",
  "description": "Detailed description of the issue, symptoms, and context",
  "priority": "High",
  "metadata": {
    "reported_by": "user@example.com",
    "environment": "production"
  }
}
```

**Required Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Unique identifier (e.g., "JIRA-123") |
| `title` | string | Short summary (1 line) |
| `description` | string | Detailed description |
| `priority` | string | "Low", "Medium", "High", or "Critical" |
| `metadata` | object | Optional additional context |

---

## 4. Processing Tickets

### Method A: CLI (Non-Streaming)

Process the default ticket file:
```bash
python3 main.py
```

This reads `input/current_ticket.json` and saves results to `output/ticket_resolution.json`.

### Method B: API (Streaming)

First start the server, then send a POST request:

```bash
# Start server in background
python3 api_server.py &

# Process a ticket via API
curl -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TEST-001",
    "title": "Database connection timeout in MM service",
    "description": "Users experiencing intermittent connection timeouts when accessing MM_ALDER module during peak hours.",
    "priority": "High",
    "metadata": {}
  }'
```

### Method C: Process from JSON File

```bash
curl -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d @input/current_ticket.json
```

---

## 5. Output Location

| Output | Location | Format |
|--------|----------|--------|
| Full results | `output/ticket_resolution.json` | JSON |
| CSV export | `output/ticket_results.csv` | CSV |
| Workflow graph | `output/workflow_graph.png` | PNG |

### Retrieve Output via API

```bash
# Get JSON output
curl http://localhost:8000/api/output | python3 -m json.tool

# Download CSV
curl -O http://localhost:8000/api/download-csv
```

---

## 6. Historical Data

### Location

Historical tickets for similarity search:
```
data/raw/historical_tickets.csv
```

### CSV Format

Required columns:
| Column | Description |
|--------|-------------|
| `key` | Ticket ID (e.g., "JIRA-MM-001") |
| `Summary` | Ticket title |
| `Description` | Full description |
| `Issue Priority` | Low/Medium/High/Critical |
| `Labels` | Comma-separated labels |
| `Resolution` | Newline-separated resolution steps |
| `created` | Created timestamp |
| `closed date` | Closed timestamp |

### Rebuild Index After Changes

When you modify historical data:
```bash
python3 scripts/setup_vectorstore.py
```

---

## 7. Useful Commands

### Health & Status

```bash
# Health check
curl http://localhost:8000/api/health

# Current configuration
curl http://localhost:8000/api/config

# Schema configuration (domains, labels)
curl http://localhost:8000/api/schema-config | python3 -m json.tool
```

### Load Sample Ticket

```bash
curl http://localhost:8000/api/load-sample | python3 -m json.tool
```

### Reload Configuration (No Restart)

```bash
curl -X POST http://localhost:8000/api/reload-schema-config
```

---

## 8. Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API keys, model settings, thresholds |
| `config/schema_config.yaml` | Domains, labels, colors, column mappings |
| `config/search_config.json` | Saved search tuning parameters |
| `input/current_ticket.json` | Default input ticket |

### Key Environment Variables

```bash
# Models
CLASSIFICATION_MODEL=gpt-4o-mini   # For classification/labeling
RESOLUTION_MODEL=gpt-4o           # For resolution generation

# Thresholds
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
LABEL_CONFIDENCE_THRESHOLD=0.7
TOP_K_SIMILAR_TICKETS=20
```

---

## 9. Troubleshooting

### Port 8000 Already in Use

```bash
lsof -ti:8000 | xargs kill -9
python3 api_server.py
```

### FAISS Index Not Found

```bash
python3 scripts/setup_vectorstore.py
```

### Module Not Found

```bash
pip install -r requirements.txt
```

### OpenAI API Errors

Check your `.env` file has a valid `OPENAI_API_KEY`.

### Low Similarity Scores

Rebuild the FAISS index:
```bash
python3 scripts/setup_vectorstore.py
```

---

## 10. Example: Complete Workflow

```bash
# 1. Set up (first time only)
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key

# 2. Build vector index
python3 scripts/setup_vectorstore.py

# 3. Start server
python3 api_server.py &
sleep 3

# 4. Process a ticket
curl -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "DEMO-001",
    "title": "EDI 834 enrollment feed validation failure",
    "description": "The EDI 834 feed is failing validation for family plan enrollments.",
    "priority": "Critical",
    "metadata": {}
  }' | grep -E '"status"|"agent"'

# 5. Get full results
curl http://localhost:8000/api/output | python3 -m json.tool | head -50

# 6. Stop server
lsof -ti:8000 | xargs kill -9
```

---

## 11. API Endpoints Reference

### Main Pipeline

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process-ticket` | POST | Process ticket (SSE streaming) |
| `/api/output` | GET | Get final JSON output |
| `/api/download-csv` | GET | Download CSV results |
| `/api/load-sample` | GET | Load sample ticket |

### Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET | Current config |
| `/api/schema-config` | GET | Schema configuration |
| `/api/reload-schema-config` | POST | Reload config without restart |

### Search Tuning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/preview-search` | POST | Preview search with custom config |
| `/api/save-search-config` | POST | Save search configuration |
| `/api/load-search-config` | GET | Load saved search config |

### Session Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | List all processing sessions |
| `/api/sessions/{session_id}` | GET | Get details for a specific session |
| `/api/sessions/{session_id}/output` | GET | Get full output for session |
| `/api/sessions/{session_id}/agents/{agent_name}` | GET | Get specific agent output |
| `/api/sessions/{session_id}/csv` | GET | Download CSV for session |

### Prompt Transparency

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/prompts` | GET | Get prompts from last processing run |

### Component APIs (v2)

| Endpoint | Description |
|----------|-------------|
| `/v2/embedding/*` | Embedding generation |
| `/v2/retrieval/*` | Similar ticket search |
| `/v2/classification/*` | Domain classification |
| `/v2/labeling/*` | Label assignment |

Full API documentation: http://localhost:8000/docs

---

## 12. Session Management

Each ticket processing run creates a unique session with ID format: `YYYYMMDD_HHMMSS_xxxxx`.

### View All Sessions

```bash
curl http://localhost:8000/api/sessions | python3 -m json.tool
```

### Get Session Details

```bash
# Replace with your session ID
curl http://localhost:8000/api/sessions/20241205_142030_a1b2c | python3 -m json.tool
```

### Get Specific Agent Output

```bash
# Get resolution output for a session
curl http://localhost:8000/api/sessions/SESSION_ID/agents/resolution
```

Sessions are stored in `output/sessions/` directory with full state at each agent step.
