# my_agentic_app

An intelligent multi-agent system built with LangGraph.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## Project Structure

```
├── src/
│   ├── agents/          # Agent implementations
│   ├── graph/           # LangGraph workflow
│   ├── models/          # Data schemas
│   ├── prompts/         # LLM prompts
│   ├── utils/           # Utilities
│   └── vectorstore/     # FAISS operations
├── scripts/             # Setup & utility scripts
├── data/                # Data storage
└── config/              # Configuration files
```
