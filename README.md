# Intelligent Ticket Management System

A multi-agent ticket management system built with LangGraph that automatically classifies, analyzes, and generates resolutions for technical support tickets using OpenAI LLMs and FAISS vector database.

## Overview

This system processes individual Jira tickets through a hierarchical four-agent pipeline:

1. **Classification Agent** → Classifies ticket domain (MM, CIW, Specialty) using binary classifiers
2. **Pattern Recognition Agent** → Retrieves top 20 similar historical tickets from FAISS
3. **Label Assignment Agent** → Assigns labels based on similar tickets' labels
4. **Resolution Generation Agent** → Generates comprehensive resolution steps

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Generate sample data
python3 scripts/generate_sample_data.py

# 4. Build FAISS index
python3 scripts/setup_vectorstore.py

# 5. Process a ticket
python3 main.py
```

See full documentation in the README for detailed setup and usage instructions.
