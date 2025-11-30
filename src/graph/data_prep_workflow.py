"""
LangGraph workflow definition for data preparation pipeline.

Implements the three-agent pipeline:
Data Validator → Data Preprocessor → Data Summarizer
"""
from langgraph.graph import StateGraph, END
from src.models.data_prep_state import DataPrepState, DataPrepRoutingDecision
from src.agents.data_validator_agent import data_validator_agent
from src.agents.data_preprocessor_agent import data_preprocessor_agent
from src.agents.data_summarizer_agent import data_summarizer_agent


# ========== Routing Functions ==========

def route_after_validation(state: DataPrepState) -> DataPrepRoutingDecision:
    """
    Route after data validator agent completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    status = state.get("status", "error")

    if status == "error":
        return "error_handler"

    # Success - proceed to preprocessing
    return "data_preprocessor"


def route_after_preprocessing(state: DataPrepState) -> DataPrepRoutingDecision:
    """
    Route after data preprocessor agent completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    status = state.get("status", "error")

    if status == "error":
        return "error_handler"

    # Success - proceed to summarization
    return "data_summarizer"


def route_after_summarization(state: DataPrepState) -> DataPrepRoutingDecision:
    """
    Route after data summarizer agent completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    status = state.get("status", "error")

    if status == "error":
        return "error_handler"

    # Success - workflow complete
    return "end"


def route_after_error_handler(state: DataPrepState) -> DataPrepRoutingDecision:
    """
    Route after error handler completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    # Error handler always ends the workflow
    return "end"


# ========== Error Handler Node ==========

def error_handler_node(state: DataPrepState) -> dict:
    """
    Handle errors by logging and gracefully terminating.

    Args:
        state: Current workflow state

    Returns:
        State update dict with error information
    """
    error_message = state.get("error_message", "Unknown error")
    current_agent = state.get("current_agent", "unknown")
    processing_stage = state.get("processing_stage", "unknown")

    print(f"\n⚠️  Data Preparation Error Handler")
    print(f"   Agent: {current_agent}")
    print(f"   Stage: {processing_stage}")
    print(f"   Error: {error_message}")
    print(f"   ❌ Data preparation pipeline failed.")

    return {
        "status": "failed",
        "error_message": f"Pipeline failed at {current_agent} ({processing_stage}): {error_message}",
        "messages": [{
            "role": "assistant",
            "content": (
                f"Data preparation failed at {current_agent} stage. "
                f"Error: {error_message}. "
                f"Please check the input data and try again."
            )
        }]
    }


# ========== Workflow Builder ==========

def build_data_prep_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for data preparation.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create workflow with DataPrepState schema
    workflow = StateGraph(DataPrepState)

    # Add nodes (agents) with descriptive names
    workflow.add_node("Data Validator Agent", data_validator_agent)
    workflow.add_node("Data Preprocessor Agent", data_preprocessor_agent)
    workflow.add_node("Data Summarizer Agent", data_summarizer_agent)
    workflow.add_node("Error Handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("Data Validator Agent")

    # Add conditional edges with routing functions
    workflow.add_conditional_edges(
        "Data Validator Agent",
        route_after_validation,
        {
            "data_preprocessor": "Data Preprocessor Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Data Preprocessor Agent",
        route_after_preprocessing,
        {
            "data_summarizer": "Data Summarizer Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Data Summarizer Agent",
        route_after_summarization,
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


def get_data_prep_workflow():
    """Get the compiled data preparation workflow instance."""
    return build_data_prep_workflow()


def visualize_data_prep_workflow(output_path: str = "output/data_prep_workflow_graph.png") -> None:
    """
    Visualize the data preparation workflow graph and save as PNG.

    Args:
        output_path: Path where the PNG will be saved
    """
    import os

    # Build the workflow
    app = build_data_prep_workflow()

    # Generate Mermaid PNG
    try:
        png_data = app.get_graph().draw_mermaid_png()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save PNG to file
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"✅ Data prep workflow graph saved to: {output_path}")

    except Exception as e:
        print(f"❌ Failed to generate workflow graph: {e}")
        print("   Note: Requires graphviz to be installed: brew install graphviz")
