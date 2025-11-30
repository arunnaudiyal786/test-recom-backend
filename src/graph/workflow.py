"""
LangGraph workflow definition for ticket processing.

Implements the four-agent pipeline:
Classification → Pattern Recognition → Label Assignment → Resolution Generation
"""
from langgraph.graph import StateGraph, END
from src.models.state_schema import TicketState
from src.agents.classification_agent import classification_agent
from src.agents.pattern_recognition_agent import pattern_recognition_agent
from src.agents.label_assignment_agent import label_assignment_agent
from src.agents.resolution_generation_agent import resolution_generation_agent
from src.graph.state_manager import (
    route_after_classification,
    route_after_pattern_recognition,
    route_after_label_assignment,
    route_after_resolution_generation,
    route_after_error_handler
)


def error_handler_node(state: TicketState) -> dict:
    """
    Handle errors by escalating to manual review.

    Args:
        state: Current workflow state

    Returns:
        State update dict with manual escalation plan
    """
    error_message = state.get("error_message", "Unknown error")
    current_agent = state.get("current_agent", "unknown")

    print(f"\n⚠️  Error Handler")
    print(f"   Agent: {current_agent}")
    print(f"   Error: {error_message}")
    print(f"   ❌ Workflow failed. Escalating to manual review.")

    return {
        "status": "failed",
        "error_message": f"Workflow failed at {current_agent}: {error_message}",
        "resolution_plan": {
            "summary": "Automatic processing failed. Manual review required.",
            "diagnostic_steps": [],
            "resolution_steps": [{
                "step_number": 1,
                "description": "Escalate to human agent for manual processing",
                "commands": [],
                "validation": "N/A",
                "estimated_time_minutes": 0,
                "risk_level": "low",
                "rollback_procedure": None
            }],
            "additional_considerations": [
                f"Automatic processing failed at {current_agent} stage",
                f"Error: {error_message}"
            ],
            "references": [],
            "total_estimated_time_hours": 0,
            "confidence": 0.0,
            "alternative_approaches": []
        }
    }


def build_ticket_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for ticket processing.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create workflow with TicketState schema
    workflow = StateGraph(TicketState)

    # Add nodes (agents) with descriptive names
    workflow.add_node("Domain Classification Agent", classification_agent)
    workflow.add_node("Pattern Recognition Agent", pattern_recognition_agent)
    workflow.add_node("Label Assignment Agent", label_assignment_agent)
    workflow.add_node("Resolution Generation Agent", resolution_generation_agent)
    workflow.add_node("Error Handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("Domain Classification Agent")

    # Add conditional edges with routing functions
    workflow.add_conditional_edges(
        "Domain Classification Agent",
        route_after_classification,
        {
            "pattern_recognition": "Pattern Recognition Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Pattern Recognition Agent",
        route_after_pattern_recognition,
        {
            "label_assignment": "Label Assignment Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Label Assignment Agent",
        route_after_label_assignment,
        {
            "resolution_generation": "Resolution Generation Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Resolution Generation Agent",
        route_after_resolution_generation,
        {
            "end": END,
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Error Handler",
        route_after_error_handler,
        {
            "end": END
        }
    )

    # Compile the workflow
    app = workflow.compile()

    return app


def visualize_workflow(output_path: str = "output/workflow_graph.png") -> None:
    """
    Visualize the workflow graph and save as PNG.

    Args:
        output_path: Path where the PNG will be saved
    """
    import os

    # Build the workflow
    app = build_ticket_workflow()

    # Generate Mermaid PNG
    try:
        png_data = app.get_graph().draw_mermaid_png()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save PNG to file
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"✅ Workflow graph saved to: {output_path}")

    except Exception as e:
        print(f"❌ Failed to generate workflow graph: {e}")
        print("   Note: Requires graphviz to be installed: brew install graphviz")


# Create global workflow instance
def get_workflow():
    """Get the compiled workflow instance."""
    return build_ticket_workflow()
