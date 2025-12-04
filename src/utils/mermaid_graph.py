"""
Mermaid Graph Generator for LangGraph Workflows.

Provides utilities to generate and save Mermaid diagram representations
of the LangGraph workflow for visualization and documentation purposes.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph import CompiledGraph


def generate_mermaid_graph(
    workflow: "CompiledGraph",
    xray: bool = True
) -> str:
    """
    Generate Mermaid diagram syntax from a LangGraph workflow.

    Args:
        workflow: Compiled LangGraph workflow instance
        xray: If True, shows internal subgraph details (default: True)

    Returns:
        Mermaid diagram syntax as a string
    """
    try:
        graph = workflow.get_graph(xray=xray)
        return graph.draw_mermaid()
    except Exception as e:
        return f"Error generating Mermaid graph: {str(e)}"


def save_mermaid_graph(
    workflow: "CompiledGraph",
    output_dir: Path,
    filename: str = "workflow_graph",
    save_png: bool = True,
    save_mermaid: bool = True,
    xray: bool = True
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save LangGraph workflow as both Mermaid text and PNG image.

    Args:
        workflow: Compiled LangGraph workflow instance
        output_dir: Directory to save the files
        filename: Base filename without extension (default: "workflow_graph")
        save_png: Whether to save PNG image (default: True)
        save_mermaid: Whether to save Mermaid text file (default: True)
        xray: If True, shows internal subgraph details (default: True)

    Returns:
        Tuple of (mermaid_path, png_path) - paths to saved files, None if not saved
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mermaid_path = None
    png_path = None

    try:
        graph = workflow.get_graph(xray=xray)

        # Save Mermaid text file (.md for better rendering in viewers)
        if save_mermaid:
            mermaid_content = graph.draw_mermaid()
            mermaid_path = output_dir / f"{filename}.mermaid"

            # Wrap in markdown code block for easy viewing
            markdown_content = f"""# Workflow Graph

This file contains the Mermaid diagram representation of the LangGraph workflow.

## Diagram

```mermaid
{mermaid_content}
```

## Raw Mermaid Syntax

```
{mermaid_content}
```
"""
            with open(mermaid_path, 'w') as f:
                f.write(markdown_content)

            print(f"   📊 Mermaid graph saved to: {mermaid_path}")

        # Save PNG image
        if save_png:
            try:
                png_data = graph.draw_mermaid_png()
                png_path = output_dir / f"{filename}.png"

                with open(png_path, 'wb') as f:
                    f.write(png_data)

                print(f"   🖼️  PNG graph saved to: {png_path}")

            except Exception as png_error:
                print(f"   ⚠️  Could not generate PNG (graphviz required): {png_error}")
                png_path = None

    except Exception as e:
        print(f"   ❌ Error generating workflow graph: {e}")

    return mermaid_path, png_path


def save_session_workflow_graph(
    session_dir: Path,
    workflow: Optional["CompiledGraph"] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save workflow graph for a specific session.

    This is a convenience function that handles workflow retrieval
    and saves graphs to the session's output directory.

    Args:
        session_dir: Path to the session output directory
        workflow: Optional workflow instance. If None, gets the singleton workflow.

    Returns:
        Tuple of (mermaid_path, png_path) - paths to saved files
    """
    # Import here to avoid circular imports
    if workflow is None:
        from src.orchestrator.workflow import get_workflow
        workflow = get_workflow()

    return save_mermaid_graph(
        workflow=workflow,
        output_dir=session_dir,
        filename="workflow_graph",
        save_png=True,
        save_mermaid=True,
        xray=True
    )


def get_workflow_mermaid_string() -> str:
    """
    Get the Mermaid diagram string for the current workflow.

    Convenience function for getting the Mermaid representation
    without saving to a file.

    Returns:
        Mermaid diagram syntax string
    """
    from src.orchestrator.workflow import get_workflow
    workflow = get_workflow()
    return generate_mermaid_graph(workflow)
