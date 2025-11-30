#!/usr/bin/env python3
"""
Script to visualize the LangGraph workflow.

Usage:
    python3 scripts/visualize_graph.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.graph.workflow import visualize_workflow


if __name__ == "__main__":
    print("Generating workflow visualization...")
    visualize_workflow()
    print("\nYou can now view the graph at: output/workflow_graph.png")
