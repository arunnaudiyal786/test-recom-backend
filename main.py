"""
Main entry point for the Intelligent Ticket Management System.

Processes a single ticket through the LangGraph workflow.
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Use new LangGraph orchestrator
from src.orchestrator.workflow import get_workflow, visualize_workflow
from src.orchestrator.state import TicketWorkflowState
from src.models.ticket_schema import IncomingTicket, FinalTicketOutput
from src.utils.config import Config
from src.utils.csv_exporter import export_ticket_results_to_csv


def load_input_ticket(file_path: Path) -> Dict[str, Any]:
    """
    Load ticket from JSON file.

    Args:
        file_path: Path to input JSON file

    Returns:
        Ticket dict
    """
    with open(file_path, 'r') as f:
        ticket_data = json.load(f)

    # Validate using Pydantic
    ticket = IncomingTicket(**ticket_data)
    return ticket.model_dump()


def prepare_initial_state(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare initial state for LangGraph workflow.

    Args:
        ticket: Ticket dict from input

    Returns:
        Initial state dict
    """
    return {
        "ticket_id": ticket["ticket_id"],
        "title": ticket["title"],
        "description": ticket["description"],
        "priority": ticket["priority"],
        "metadata": ticket.get("metadata", {}),
        "processing_stage": "start",
        "status": "processing",
        "error_message": None,
        "current_agent": "start",
        "messages": [],
        "overall_confidence": 0.0
    }


def calculate_overall_confidence(state: Dict[str, Any]) -> float:
    """
    Calculate overall confidence from all agent confidences.

    Args:
        state: Final workflow state

    Returns:
        Overall confidence score 0-1
    """
    confidences = []

    # Classification confidence
    if "classification_confidence" in state and state["classification_confidence"]:
        confidences.append(state["classification_confidence"])

    # Pattern matching (use avg similarity as proxy)
    search_meta = state.get("search_metadata", {})
    if "avg_similarity" in search_meta:
        confidences.append(search_meta["avg_similarity"])

    # Resolution confidence
    if "resolution_confidence" in state and state["resolution_confidence"]:
        confidences.append(state["resolution_confidence"])

    # Average all available confidences
    if confidences:
        return sum(confidences) / len(confidences)
    return 0.0


def format_final_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format final output for saving to JSON.

    Args:
        state: Final workflow state

    Returns:
        Formatted output dict
    """
    overall_conf = calculate_overall_confidence(state)

    output = {
        "ticket_id": state["ticket_id"],
        "classified_domain": state.get("classified_domain", "Unknown"),
        "classification_confidence": state.get("classification_confidence", 0.0),
        "assigned_labels": state.get("assigned_labels", []),
        "label_confidence": state.get("label_confidence", {}),
        "resolution_plan": state.get("resolution_plan", {}),
        "overall_confidence": overall_conf,
        "processing_metadata": {
            "status": state.get("status", "unknown"),
            "error_message": state.get("error_message"),
            "similar_tickets_count": len(state.get("similar_tickets", [])),
            "avg_similarity": state.get("search_metadata", {}).get("avg_similarity", 0.0)
        }
    }

    return output


def save_output(output: Dict[str, Any], file_path: Path):
    """
    Save output to JSON file.

    Args:
        output: Output dict to save
        file_path: Path to save file
    """
    # Create output directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved output to: {file_path}")


async def process_ticket(input_file: Path, output_file: Path):
    """
    Process a single ticket through the workflow.

    Args:
        input_file: Path to input ticket JSON
        output_file: Path to save output JSON
    """
    print("=" * 80)
    print("🎫 INTELLIGENT TICKET MANAGEMENT SYSTEM")
    print("=" * 80)

    start_time = time.time()

    # Load input ticket
    print(f"\n📂 Loading ticket from: {input_file}")
    ticket = load_input_ticket(input_file)
    print(f"   Ticket ID: {ticket['ticket_id']}")
    print(f"   Title: {ticket['title']}")
    print(f"   Priority: {ticket['priority']}")

    # Prepare initial state
    initial_state = prepare_initial_state(ticket)

    # Generate workflow visualization
    print(f"\n📊 Generating workflow graph...")
    graph_path = Config.PROJECT_ROOT / "output" / "workflow_graph.png"
    try:
        visualize_workflow(str(graph_path))
    except Exception as e:
        print(f"   ⚠️  Could not generate graph visualization: {e}")
        print(f"   (Install graphviz: brew install graphviz)")

    # Get workflow
    print(f"\n🚀 Starting LangGraph workflow...")
    workflow = get_workflow()

    # Execute workflow
    try:
        final_state = await workflow.ainvoke(initial_state)
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {str(e)}")
        sys.exit(1)

    # Calculate processing time
    processing_time = time.time() - start_time

    # Format and save output
    output = format_final_output(final_state)
    save_output(output, output_file)

    # Export to CSV (includes labels and similar tickets)
    csv_path = export_ticket_results_to_csv(final_state)
    print(f"📊 Exported results to CSV: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)
    print(f"   Domain: {output['classified_domain']}")
    print(f"   Classification Confidence: {output['classification_confidence']:.2%}")
    print(f"   Labels: {', '.join(output['assigned_labels']) if output['assigned_labels'] else 'None'}")
    print(f"   Resolution Steps: {len(output['resolution_plan'].get('resolution_steps', []))}")
    print(f"   Estimated Resolution Time: {output['resolution_plan'].get('total_estimated_time_hours', 0)} hours")
    print(f"   Overall Confidence: {output['overall_confidence']:.2%}")
    print(f"   Processing Time: {processing_time:.2f} seconds")
    print(f"\n   📊 Workflow graph: output/workflow_graph.png")
    print(f"   📄 Full results: {output_file}")
    print(f"   📋 CSV export: {csv_path}")
    print("=" * 80)


async def main():
    """Main entry point."""
    # Default paths
    input_file = Config.PROJECT_ROOT / "input" / "current_ticket.json"
    output_file = Config.PROJECT_ROOT / "output" / "ticket_resolution.json"

    # Check if input file exists
    if not input_file.exists():
        print(f"❌ Error: Input file not found at {input_file}")
        print("   Create a ticket JSON file in the input/ directory")
        sys.exit(1)

    # Process the ticket
    await process_ticket(input_file, output_file)


if __name__ == "__main__":
    asyncio.run(main())
