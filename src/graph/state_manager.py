"""
State management and routing functions for LangGraph workflow.
"""
from typing import Literal
from src.models.state_schema import TicketState, RoutingDecision


# Routing functions for conditional edges

def route_after_classification(state: TicketState) -> RoutingDecision:
    """
    Route after classification agent completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    status = state.get("status", "error")

    if status == "error":
        return "error_handler"

    # Success - proceed to pattern recognition
    return "pattern_recognition"


def route_after_pattern_recognition(state: TicketState) -> RoutingDecision:
    """
    Route after pattern recognition agent completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    status = state.get("status", "error")

    if status == "error":
        return "error_handler"

    # Success - proceed to label assignment
    return "label_assignment"


def route_after_label_assignment(state: TicketState) -> RoutingDecision:
    """
    Route after label assignment agent completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    status = state.get("status", "error")

    if status == "error":
        return "error_handler"

    # Success - proceed to resolution generation
    return "resolution_generation"


def route_after_resolution_generation(state: TicketState) -> RoutingDecision:
    """
    Route after resolution generation agent completes.

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


def route_after_error_handler(state: TicketState) -> RoutingDecision:
    """
    Route after error handler completes.

    Args:
        state: Current workflow state

    Returns:
        Next node to execute
    """
    # Error handler always ends the workflow
    return "end"
