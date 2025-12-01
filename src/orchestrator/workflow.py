"""
LangGraph Workflow - Orchestrates agent components into a pipeline.

Implements the four-agent pipeline:
Classification -> Retrieval -> Labeling -> Resolution

Each agent is a component from the components/ folder, composed here
using LangGraph's StateGraph for workflow orchestration.

Configuration:
- SKIP_DOMAIN_CLASSIFICATION: When True, skips the Domain Classification Agent
  and starts directly from Pattern Recognition Agent. Set to False to re-enable.
"""

from langgraph.graph import StateGraph, END

from src.orchestrator.state import TicketWorkflowState

# Import agent nodes from components
from components.classification.agent import classification_node
from components.retrieval.agent import retrieval_node
from components.labeling.agent import labeling_node
from components.resolution.agent import resolution_node

# ============================================================================
# CONFIGURATION FLAG: Set to True to skip Domain Classification Agent
# Set back to False to re-enable the classification step
# ============================================================================
SKIP_DOMAIN_CLASSIFICATION = True


def route_after_classification(state: TicketWorkflowState) -> str:
    """
    Route after classification based on success/error.

    Args:
        state: Current workflow state

    Returns:
        Next node name: "retrieval" or "error_handler"
    """
    if state.get("status") == "error":
        return "error_handler"
    return "retrieval"


def route_after_retrieval(state: TicketWorkflowState) -> str:
    """
    Route after retrieval based on success/error.

    Args:
        state: Current workflow state

    Returns:
        Next node name: "labeling" or "error_handler"
    """
    if state.get("status") == "error":
        return "error_handler"
    return "labeling"


def route_after_labeling(state: TicketWorkflowState) -> str:
    """
    Route after labeling based on success/error.

    Args:
        state: Current workflow state

    Returns:
        Next node name: "resolution" or "error_handler"
    """
    if state.get("status") == "error":
        return "error_handler"
    return "resolution"


def route_after_resolution(state: TicketWorkflowState) -> str:
    """
    Route after resolution based on success/error.

    Args:
        state: Current workflow state

    Returns:
        Next node name: "end" or "error_handler"
    """
    if state.get("status") == "error":
        return "error_handler"
    return "end"


def route_after_error_handler(state: TicketWorkflowState) -> str:
    """
    Route after error handler (always ends).

    Args:
        state: Current workflow state

    Returns:
        "end" to terminate workflow
    """
    return "end"


def error_handler_node(state: TicketWorkflowState) -> dict:
    """
    Handle errors by escalating to manual review.

    Generates a fallback resolution plan when any agent fails.

    Args:
        state: Current workflow state

    Returns:
        State update dict with manual escalation plan
    """
    error_message = state.get("error_message", "Unknown error")
    current_agent = state.get("current_agent", "unknown")

    print(f"\n Warning: Error Handler")
    print(f"   Agent: {current_agent}")
    print(f"   Error: {error_message}")
    print(f"   Workflow failed. Escalating to manual review.")

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
        },
        "messages": [{
            "role": "assistant",
            "content": f"Workflow failed at {current_agent}: {error_message}"
        }]
    }


def build_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for ticket processing.

    Creates a StateGraph with:
    - Four agent nodes (classification, retrieval, labeling, resolution)
    - Error handler node for graceful degradation
    - Conditional routing based on agent status

    When SKIP_DOMAIN_CLASSIFICATION is True:
    - Entry point is Pattern Recognition Agent (skips classification)
    - Domain Classification Agent node is still added but not used

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create workflow with TicketWorkflowState schema
    workflow = StateGraph(TicketWorkflowState)

    # Add agent nodes from components
    # Note: Classification node is added but may not be used if SKIP_DOMAIN_CLASSIFICATION is True
    if not SKIP_DOMAIN_CLASSIFICATION:
        workflow.add_node("Domain Classification Agent", classification_node)
    workflow.add_node("Pattern Recognition Agent", retrieval_node)
    workflow.add_node("Label Assignment Agent", labeling_node)
    workflow.add_node("Resolution Generation Agent", resolution_node)
    workflow.add_node("Error Handler", error_handler_node)

    # Set entry point based on configuration
    if SKIP_DOMAIN_CLASSIFICATION:
        # Skip classification - start directly from Pattern Recognition
        workflow.set_entry_point("Pattern Recognition Agent")
        print("⚡ Domain Classification Agent DISABLED - starting from Pattern Recognition")
    else:
        # Normal flow - start from Domain Classification
        workflow.set_entry_point("Domain Classification Agent")
        # Add routing from classification to retrieval
        workflow.add_conditional_edges(
            "Domain Classification Agent",
            route_after_classification,
            {
                "retrieval": "Pattern Recognition Agent",
                "error_handler": "Error Handler"
            }
        )

    # Add conditional edges for Pattern Recognition -> Label Assignment
    workflow.add_conditional_edges(
        "Pattern Recognition Agent",
        route_after_retrieval,
        {
            "labeling": "Label Assignment Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Label Assignment Agent",
        route_after_labeling,
        {
            "resolution": "Resolution Generation Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Resolution Generation Agent",
        route_after_resolution,
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

    # Compile and return
    return workflow.compile()


def visualize_workflow(output_path: str = "output/workflow_graph.png") -> None:
    """
    Visualize the workflow graph and save as PNG.

    Args:
        output_path: Path where the PNG will be saved
    """
    import os

    app = build_workflow()

    try:
        png_data = app.get_graph().draw_mermaid_png()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"Workflow graph saved to: {output_path}")

    except Exception as e:
        print(f"Failed to generate workflow graph: {e}")
        print("   Note: Requires graphviz to be installed: brew install graphviz")


# Singleton workflow instance
_workflow = None


def get_workflow():
    """
    Get the compiled workflow instance (singleton).

    Returns:
        Compiled LangGraph workflow
    """
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow
