# EIP Dashboard Quick Start Guide

This guide helps you quickly start and stop the EIP Intelligent Ticket Management Dashboard.

## 🚀 Starting the Dashboard

### Option 1: Using the Start Script (Recommended)

```bash
./start_dev.sh
```

This will:
- ✅ Check all prerequisites (.env file, FAISS index, dependencies)
- 🚀 Start the FastAPI backend on http://localhost:8000
- 🎨 Start the Next.js frontend on http://localhost:3000
- 📊 Display status messages for both services

**The script will run both servers and keep them active until you press `Ctrl+C`.**

### Option 2: Manual Start (Separate Terminals)

**Terminal 1 - Backend:**
```bash
python3 api_server.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## 🛑 Stopping the Dashboard

### Option 1: Using the Stop Script

```bash
./stop_dev.sh
```

This will cleanly kill all running backend and frontend processes.

### Option 2: If Using start_dev.sh

Press `Ctrl+C` in the terminal where `start_dev.sh` is running.

### Option 3: Manual Stop

```bash
# Kill processes on ports 8000 and 3000
lsof -ti:8000 -ti:3000 | xargs kill -9
```

## 📍 Access Points

Once started, you can access:

- **Frontend Dashboard:** http://localhost:3000/pattern-recognition
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health

## 🧪 Testing the System

1. Open http://localhost:3000/pattern-recognition in your browser
2. Click "Load Sample Ticket" to load a pre-configured test ticket
3. Watch as the 4 agents process the ticket sequentially:
   - Domain Classification Agent
   - Pattern Recognition Agent
   - Label Assignment Agent
   - Resolution Generation Agent
4. The complete resolution will be saved to `output/ticket_resolution.json`

## 📊 Expected Output

**Backend Logs (Terminal):**
```
🚀 Starting EIP API Server...
📡 Frontend: http://localhost:3000
🔧 API Docs: http://localhost:8000/docs

🔍 Classification Agent - Analyzing ticket: JIRA-NEW-001
   ✅ Classified as: MM (confidence: 100.00%)

🔎 Pattern Recognition Agent - Finding similar MM tickets
   ✅ Found 20 similar tickets

🏷️  Label Assignment Agent - Analyzing candidate labels
   ✅ Assigned labels: #MM_ALDER, Configuration Fix

📝 Resolution Generation Agent - Creating resolution plan
   ✅ Generated resolution plan
```

**Frontend (Browser):**
- Clean UI with 4 agent cards
- Real-time status updates as agents work
- Final resolution displayed with detailed steps

## ⚠️ Troubleshooting

### Port Already in Use

If you see "Port 3000 is in use" or "Port 8000 is in use":

```bash
./stop_dev.sh
./start_dev.sh
```

### FAISS Index Not Found

If you see a warning about missing FAISS index:

```bash
python3 scripts/setup_vectorstore.py
```

### Missing Dependencies

**Python:**
```bash
pip install -r requirements.txt
```

**Node.js:**
```bash
cd frontend
npm install
```

### Frontend Build Errors

If the frontend shows errors, ensure autoprefixer is installed:

```bash
cd frontend
npm install autoprefixer
```

## 📁 Important Files

- `start_dev.sh` - Start both servers
- `stop_dev.sh` - Stop both servers
- `api_server.py` - Backend FastAPI server
- `frontend/` - Next.js frontend application
- `output/ticket_resolution.json` - Generated resolution plans
- `.env` - OpenAI API key configuration

## 🔧 Configuration

All configuration is in `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
CLASSIFICATION_MODEL=gpt-4o
RESOLUTION_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
TOP_K_SIMILAR_TICKETS=20
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
```

## 💡 Tips

1. **Keep both servers running** during development for hot-reload support
2. **Check backend logs** to see detailed agent execution information
3. **Sample tickets** are loaded from `input/current_ticket.json`
4. **Resolution outputs** are saved to `output/ticket_resolution.json`
5. **Cost per ticket:** Approximately $0.07 per ticket processed

## 🎯 Next Steps

- Customize sample tickets in `input/current_ticket.json`
- Add more historical tickets to improve pattern recognition
- Modify agent prompts in `src/prompts/`
- Adjust confidence thresholds in `.env`

---

**Questions?** Check `CLAUDE.md` for detailed system documentation.
