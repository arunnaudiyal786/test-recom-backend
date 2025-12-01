# Startup Guide

Quick reference for running the Test Recommendation Backend.

---

## 1. First-Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (if not already done)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Build FAISS vector index from historical data
python3 scripts/setup_vectorstore.py
```

---

## 2. Start the Backend Server

```bash
# Start the API server (runs on port 8000)
python3 api_server.py
```

Server will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 3. Process a Ticket via Terminal

### Option A: Using curl (Streaming Output)

```bash
curl -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TEST-001",
    "title": "Your ticket title here",
    "description": "Detailed description of the issue or test case",
    "priority": "High",
    "metadata": {"environment": "production"}
  }'
```

### Option B: Using a JSON File

1. Create a ticket file:
```bash
cat > /tmp/my_ticket.json << 'EOF'
{
  "ticket_id": "TEST-001",
  "title": "AWD Payment Processing Failure",
  "description": "Verify AWD payment processing when member enrolls in family plan. Setup includes primary member and 2 dependents with combined premium.",
  "priority": "High",
  "metadata": {
    "environment": "test",
    "reported_by": "qa-team@example.com"
  }
}
EOF
```

2. Submit it:
```bash
curl -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d @/tmp/my_ticket.json
```

### Option C: Using the CLI Script (Non-Streaming)

```bash
# Process the default ticket from input/current_ticket.json
python3 main.py

# Output will be saved to output/ticket_resolution.json
```

---

## 4. View Results

### Get Full JSON Output
```bash
curl -s http://localhost:8000/api/output | python3 -m json.tool
```

### Download CSV Results
```bash
curl -O http://localhost:8000/api/download-csv
# Downloads: ticket_results.csv
```

### View Output File Directly
```bash
cat output/ticket_resolution.json | python3 -m json.tool
```

---

## 5. Quick Test Commands

```bash
# Health check
curl http://localhost:8000/api/health

# Get current configuration
curl http://localhost:8000/api/config

# Get schema configuration (domains, labels, etc.)
curl http://localhost:8000/api/schema-config | python3 -m json.tool

# Load sample ticket
curl http://localhost:8000/api/load-sample
```

---

## 6. Configuration Files

| File | Purpose |
|------|---------|
| `.env` | API keys, model settings, thresholds |
| `config/schema_config.yaml` | Column mappings, domains, labels, UI settings |
| `config/search_config.json` | Saved search tuning parameters |
| `input/current_ticket.json` | Default input ticket for CLI |

---

## 7. Rebuild After Data Changes

If you modify `data/raw/test_plan_historical.csv`:

```bash
# Rebuild the FAISS vector index
python3 scripts/setup_vectorstore.py

# Reload schema config (if server is running)
curl -X POST http://localhost:8000/api/reload-schema-config
```

---

## 8. Common Issues

**Port 8000 already in use:**
```bash
lsof -ti:8000 | xargs kill -9
python3 api_server.py
```

**FAISS index not found:**
```bash
python3 scripts/setup_vectorstore.py
```

**Module not found errors:**
```bash
pip install -r requirements.txt
```

---

## 9. Example: Full Workflow

```bash
# 1. Start server in background
python3 api_server.py &

# 2. Wait for startup
sleep 3

# 3. Process a ticket
curl -s -X POST http://localhost:8000/api/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "DEMO-001",
    "title": "EDI 834 enrollment feed validation failure",
    "description": "The EDI 834 feed is failing validation for family plan enrollments with multiple dependents.",
    "priority": "Critical",
    "metadata": {}
  }' | grep -E '"status"|"agent"'

# 4. Get the full output
curl -s http://localhost:8000/api/output | python3 -m json.tool | head -100

# 5. Stop server
lsof -ti:8000 | xargs kill -9
```
