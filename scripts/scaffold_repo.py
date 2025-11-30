#!/usr/bin/env python3
"""
Repository Scaffolding Tool for Agentic Applications

This script generates the folder and file structure for a LangGraph-based
multi-agent application backend. It creates files directly in the target
directory with optional starter templates for essential configuration files.

Usage:
    # Create structure in current directory
    python scripts/scaffold_repo.py .

    # Create structure in a specific directory
    python scripts/scaffold_repo.py /path/to/new/repo

    # Set project name for templates (README, package.json, etc.)
    python scripts/scaffold_repo.py . --name my_ticket_system

    # Preview what would be created
    python scripts/scaffold_repo.py ./new_repo --dry-run

    # Create empty files (no template content)
    python scripts/scaffold_repo.py . --no-templates
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# =============================================================================
# REPOSITORY STRUCTURE DEFINITION
# =============================================================================
# This is the canonical structure for agentic application backends.
# Modify this dict to customize the structure for your needs.

REPO_STRUCTURE: Dict[str, Union[List[str], Dict]] = {
    # Root-level files
    "_root_files": [
        "main.py",
        "api_server.py",
        "requirements.txt",
        "README.md",
        "CLAUDE.md",
        ".env.example",
        ".gitignore",
        "start_dev.sh",
        "stop_dev.sh",
    ],

    # Configuration directory (empty)
    "config": [".gitkeep"],

    # Data directories (empty)
    "data": {
        "raw": [".gitkeep"],
        "processed": [".gitkeep"],
        "faiss_index": [".gitkeep"],
    },

    # Documentation (empty)
    "docs": [".gitkeep"],

    # Input/Output directories (empty)
    "input": [".gitkeep"],
    "output": [".gitkeep"],

    # Scripts directory (empty)
    "scripts": [".gitkeep"],

    # Main source code (empty subdirs)
    "src": {
        "agents": [".gitkeep"],
        "graph": [".gitkeep"],
        "models": [".gitkeep"],
        "prompts": [".gitkeep"],
        "utils": [".gitkeep"],
        "vectorstore": [".gitkeep"],
    },
}


# =============================================================================
# FILE TEMPLATES
# =============================================================================
# Starter content for root-level files. Subdirectories only have .gitkeep.

FILE_TEMPLATES: Dict[str, str] = {
    ".gitkeep": "# This file exists to keep this empty directory in git\n",

    ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
.eggs/
*.egg-info/
dist/
build/

# Environment
.env
.env.local

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/faiss_index/*.index
data/faiss_index/metadata.json
data/processed/*.csv
data/processed/*.json
output/*.json
output/*.csv
output/*.png

# Logs
*.log
logs/
""",

    ".env.example": """# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here

# Model Selection
CLASSIFICATION_MODEL=gpt-4o-mini
RESOLUTION_MODEL=gpt-4o

# Thresholds
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
LABEL_CONFIDENCE_THRESHOLD=0.7
TOP_K_SIMILAR_TICKETS=20

# Server Configuration
API_HOST=localhost
API_PORT=8000
""",

    "requirements.txt": """# Core dependencies
langchain>=0.1.0
langgraph>=0.0.20
openai>=1.0.0
faiss-cpu>=1.7.4
numpy>=1.24.0
pydantic>=2.0.0

# API Server
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6

# Utilities
python-dotenv>=1.0.0
httpx>=0.24.0
pandas>=2.0.0

# Development
pytest>=7.0.0
pytest-asyncio>=0.21.0
""",

    "README.md": """# {project_name}

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
""",

    "CLAUDE.md": """# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Multi-agent system built with LangGraph.

## Essential Commands

```bash
pip install -r requirements.txt
python main.py
python api_server.py
```

## Key Directories

- `src/agents/` - Agent implementations
- `src/graph/` - LangGraph workflow
- `src/prompts/` - LLM prompt templates
""",

    "main.py": '''#!/usr/bin/env python3
"""Main entry point."""

import asyncio


async def main():
    """Run the main workflow."""
    print("Hello from {project_name}")
    # TODO: Implement your workflow here


if __name__ == "__main__":
    asyncio.run(main())
''',

    "api_server.py": '''#!/usr/bin/env python3
"""FastAPI server."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="{project_name} API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
''',

    "start_dev.sh": """#!/bin/bash
echo "Starting API server..."
python api_server.py
""",

    "stop_dev.sh": """#!/bin/bash
pkill -f api_server.py
echo "Server stopped"
""",
}


# =============================================================================
# SCAFFOLDING LOGIC
# =============================================================================

def create_structure(
    base_path: Path,
    structure: Dict[str, Union[List[str], Dict]],
    project_name: str,
    with_templates: bool = True,
    dry_run: bool = False,
) -> List[str]:
    """
    Recursively create the directory and file structure.

    Args:
        base_path: Base directory to create structure in
        structure: Dictionary defining the structure
        project_name: Name of the project (for template substitution)
        with_templates: Whether to add template content to files
        dry_run: If True, only print what would be created

    Returns:
        List of created paths
    """
    created_paths = []

    for key, value in structure.items():
        if key == "_root_files":
            # Handle root-level files in current directory
            for filename in value:
                file_path = base_path / filename
                created_paths.extend(
                    create_file(file_path, filename, project_name, with_templates, dry_run)
                )
        elif isinstance(value, list):
            # It's a directory with files
            dir_path = base_path / key
            if not dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            created_paths.append(str(dir_path))

            for filename in value:
                file_path = dir_path / filename
                created_paths.extend(
                    create_file(file_path, filename, project_name, with_templates, dry_run, key)
                )
        elif isinstance(value, dict):
            # It's a nested directory structure
            dir_path = base_path / key
            if not dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            created_paths.append(str(dir_path))

            # Recursively create nested structure
            created_paths.extend(
                create_structure(dir_path, value, project_name, with_templates, dry_run)
            )

    return created_paths


def create_file(
    file_path: Path,
    filename: str,
    project_name: str,
    with_templates: bool,
    dry_run: bool,
    parent_dir: str = "",
) -> List[str]:
    """Create a single file with optional template content."""
    created = []

    if dry_run:
        print(f"  Would create: {file_path}")
        return [str(file_path)]

    if file_path.exists():
        print(f"  Skipping (exists): {file_path}")
        return []

    # Determine template key
    template_key = filename
    if parent_dir:
        # Check for path-specific template (e.g., "frontend/package.json")
        path_key = f"{parent_dir}/{filename}"
        if path_key in FILE_TEMPLATES:
            template_key = path_key
        # Also check relative path from file
        rel_path = str(file_path.relative_to(file_path.parents[len(file_path.parts) - 3]))
        if rel_path in FILE_TEMPLATES:
            template_key = rel_path

    # Get template content
    content = ""
    if with_templates:
        # Try to find matching template
        for key in [str(file_path).split("/")[-2:], template_key, filename]:
            if isinstance(key, list):
                key = "/".join(key)
            if key in FILE_TEMPLATES:
                content = FILE_TEMPLATES[key].replace("{project_name}", project_name)
                break

    # Create file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)

    # Make shell scripts executable
    if filename.endswith(".sh"):
        os.chmod(file_path, 0o755)

    print(f"  Created: {file_path}")
    created.append(str(file_path))

    return created


def scaffold_repository(
    output_dir: Optional[str] = None,
    project_name: str = "my_agentic_app",
    with_templates: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Main function to scaffold a new repository.

    Args:
        output_dir: Directory to create files in (defaults to current dir)
        project_name: Name for template substitution (README, package.json, etc.)
        with_templates: Whether to include template file content
        dry_run: If True, only show what would be created
    """
    # Files are created directly in output_dir (no project subdirectory)
    if output_dir:
        base_path = Path(output_dir)
    else:
        base_path = Path.cwd()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Scaffolding agentic app structure")
    print(f"Location: {base_path}")
    print(f"Project name (for templates): {project_name}\n")

    if not dry_run:
        base_path.mkdir(parents=True, exist_ok=True)

    # Create the structure
    created = create_structure(base_path, REPO_STRUCTURE, project_name, with_templates, dry_run)

    # Summary
    print(f"\n{'Would create' if dry_run else 'Created'} {len(created)} items")

    if not dry_run:
        print(f"\nNext steps:")
        print(f"  cd {base_path}")
        print(f"  python -m venv .venv && source .venv/bin/activate")
        print(f"  pip install -r requirements.txt")
        print(f"  cp .env.example .env  # Add your API keys")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scaffold agentic application structure directly into a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create structure in current directory
  %(prog)s .

  # Create structure in a new directory
  %(prog)s ~/projects/my_new_repo

  # Create with custom project name for templates
  %(prog)s . --name my_ticket_system

  # Preview what would be created
  %(prog)s ./new_repo --dry-run

  # Create without template content (empty files)
  %(prog)s . --no-templates
        """,
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Directory to create files in (default: current directory)",
    )
    parser.add_argument(
        "--name", "-n",
        dest="project_name",
        default="my_agentic_app",
        help="Project name for template substitution (README, package.json, etc.)",
    )
    parser.add_argument(
        "--no-templates",
        action="store_true",
        help="Create empty files without template content",
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be created without actually creating files",
    )

    args = parser.parse_args()

    scaffold_repository(
        output_dir=args.output_dir,
        project_name=args.project_name,
        with_templates=not args.no_templates,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
