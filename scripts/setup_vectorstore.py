"""
One-time setup script to build FAISS index from historical tickets.

Run this script after generating sample data to create the vector database.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.data_ingestion import main

if __name__ == "__main__":
    print("🔧 Setting up FAISS Vector Store...")
    print("   This will generate embeddings for 100 tickets using OpenAI API")
    print("   Estimated cost: ~$0.02 for embeddings\n")

    asyncio.run(main())
