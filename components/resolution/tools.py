"""
Resolution Tools - LangChain @tool decorated functions for resolution generation.

These tools extract resolution steps from similar historical tickets and generate
comprehensive resolution plans with summary and considerations.

Uses LangChain's create_agent for agent-based resolution generation as LangGraph nodes.
"""

import json
from typing import Dict, Any, List

from langchain_core.tools import tool
from langchain.agents import create_agent

from config.config import Config
from components.resolution.models import (
    ResolutionStep,
    ResolutionPlan
)
from components.resolution.prompts import (
    RESOLUTION_SYSTEM_PROMPT,
    get_resolution_user_prompt,
    format_historical_context
)


# ============================================================================
# TOOLS FOR RESOLUTION AGENT
# ============================================================================

@tool
def analyze_similar_resolutions(similar_tickets: List[Dict[str, Any]]) -> str:
    """
    Analyze resolution patterns from similar historical tickets.

    Extracts and formats resolution information from similar tickets
    to provide context for resolution generation.

    Args:
        similar_tickets: List of similar ticket dicts with resolution info

    Returns:
        Formatted string of historical resolution patterns
    """
    return format_historical_context(similar_tickets)


def _extract_resolution_steps_from_tickets(
    similar_tickets: List[Dict[str, Any]],
    max_tickets: int = 5
) -> List[Dict[str, Any]]:
    """
    Extract and format resolution steps from similar historical tickets.

    Args:
        similar_tickets: List of similar tickets with resolution_steps
        max_tickets: Maximum number of tickets to use (default 5)

    Returns:
        List of formatted resolution steps with source references
    """
    steps = []
    step_number = 1

    for ticket in similar_tickets[:max_tickets]:
        ticket_id = ticket.get("ticket_id", "Unknown")
        similarity = ticket.get("similarity_score", 0)
        resolution = ticket.get("resolution_steps", ticket.get("resolution", []))

        # Handle both string and list formats
        if isinstance(resolution, str):
            resolution_lines = [r.strip() for r in resolution.split('\n') if r.strip()]
        elif isinstance(resolution, list):
            resolution_lines = [str(r).strip() for r in resolution if str(r).strip()]
        else:
            resolution_lines = []

        for step_desc in resolution_lines:
            if step_desc:
                steps.append({
                    "step_number": step_number,
                    "description": step_desc,
                    "expected_result": f"Step completed successfully as per {ticket_id}",
                    "commands": [],
                    "validation": f"Verify as per ticket {ticket_id}",
                    "estimated_time_minutes": 10,
                    "risk_level": "low",
                    "rollback_procedure": f"Refer to rollback procedures in {ticket_id}",
                    "source_ticket": ticket_id,
                    "source_similarity": round(similarity * 100, 1)
                })
                step_number += 1

    return steps


# ============================================================================
# RESPONSE TOOL FOR RESOLUTION AGENT
# ============================================================================

def generate_test_plan_response(
    summary: str,
    test_plan: List[Dict[str, Any]],
    additional_considerations: List[str],
    references: List[Dict[str, Any]],
    confidence: float
) -> str:
    """
    Generate the final test plan response in JSON format.

    This tool is used by the Resolution Agent to output the synthesized test plan.
    The agent analyzes similar tickets and calls this tool to structure its response.

    Args:
        summary: Executive summary of the recommended test approach (2-3 sentences)
        test_plan: List of test steps with step_number, description, expected_result,
                   validation, source_tickets, and estimated_time_minutes
        additional_considerations: List of additional considerations for the QA/test team
        references: List of reference dicts with ticket_id, similarity, and note
        confidence: Confidence score based on similar case similarity (0.0-1.0)

    Returns:
        JSON string of the complete test plan
    """
    result = {
        "summary": summary,
        "test_plan": test_plan[:5],  # Limit to 5 steps
        "additional_considerations": additional_considerations,
        "references": references,
        "confidence": confidence
    }
    return json.dumps(result)


# Create the tool using @tool decorator
@tool
def submit_test_plan(
    summary: str,
    test_plan: str,
    additional_considerations: str,
    references: str,
    confidence: float
) -> str:
    """
    Submit the final test plan response.

    The agent should call this tool after analyzing similar tickets to output
    the synthesized test plan in the required JSON format.

    Args:
        summary: Executive summary of the recommended test approach (2-3 sentences)
        test_plan: JSON string of test steps array
        additional_considerations: JSON string of considerations array
        references: JSON string of references array
        confidence: Confidence score (0.0-1.0)

    Returns:
        Confirmation message
    """
    return json.dumps({
        "summary": summary,
        "test_plan": json.loads(test_plan) if test_plan else [],
        "additional_considerations": json.loads(additional_considerations) if additional_considerations else [],
        "references": json.loads(references) if references else [],
        "confidence": confidence
    })


# ============================================================================
# RESOLUTION AGENT CREATION
# ============================================================================

def create_resolution_agent():
    """
    Create a LangChain agent for resolution generation.

    Uses langchain.agents.create_agent to build a graph-based agent runtime.
    The agent uses the submit_test_plan tool to output its synthesized test plan.

    Returns:
        CompiledStateGraph: A LangChain agent configured for test plan generation
    """
    # Use model name string format for create_agent
    # Format: "provider:model" or just "model" for OpenAI
    model_name = Config.RESOLUTION_MODEL  # e.g., "gpt-4o" or "gpt-4o-mini"

    # Create the agent with system prompt and response tool
    agent = create_agent(
        model=model_name,
        tools=[submit_test_plan],  # Tool for outputting the test plan
        system_prompt=RESOLUTION_SYSTEM_PROMPT
    )

    return agent


# Singleton agent instance (created lazily)
_resolution_agent = None


def get_resolution_agent():
    """Get or create the resolution agent singleton."""
    global _resolution_agent
    if _resolution_agent is None:
        _resolution_agent = create_resolution_agent()
    return _resolution_agent


# ============================================================================
# MAIN RESOLUTION GENERATION FUNCTION
# ============================================================================

@tool
async def generate_resolution_plan(
    title: str,
    description: str,
    domain: str,
    priority: str,
    labels: List[str],
    similar_tickets: List[Dict[str, Any]],
    avg_similarity: float
) -> Dict[str, Any]:
    """
    Generate a comprehensive resolution plan using the resolution agent.

    Uses LangChain's create_agent to analyze historical tickets and
    generate a synthesized test plan. The agent runs as a LangGraph node.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        labels: Assigned labels
        similar_tickets: List of similar historical tickets with resolution steps
        avg_similarity: Average similarity to historical tickets

    Returns:
        Dict containing:
        - resolution_plan: Complete resolution plan dict
        - confidence: Confidence score (0-1)
        - actual_prompt: The prompt sent to the agent
    """
    # Extract resolution steps from similar tickets (as fallback)
    extracted_steps = _extract_resolution_steps_from_tickets(similar_tickets, max_tickets=5)

    # Build historical context for the agent
    historical_context = format_historical_context(similar_tickets)

    # Generate the user prompt
    user_prompt = get_resolution_user_prompt(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        labels=labels,
        historical_context=historical_context,
        avg_similarity=avg_similarity
    )

    try:
        # Get the resolution agent
        agent = get_resolution_agent()

        # Invoke the agent with the user prompt
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        # Extract the response from tool messages
        # create_agent returns tool call results in the messages
        response_content = None
        for msg in reversed(result.get("messages", [])):
            # Check for tool message content
            if hasattr(msg, "content") and msg.content:
                try:
                    # Try to parse as JSON
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and ("summary" in data or "test_plan" in data):
                        response_content = msg.content
                        break
                except (json.JSONDecodeError, TypeError):
                    continue
            # Also check for structured response
            if hasattr(msg, "structured_response"):
                response_content = json.dumps(msg.structured_response)
                break

        if not response_content:
            # Fallback: try the last message
            response_content = result["messages"][-1].content if result.get("messages") else "{}"

        # Parse the JSON response
        data = json.loads(response_content) if response_content else {}

        # Build resolution plan with extracted steps
        resolution_plan = _build_resolution_plan(data, extracted_steps)

        return {
            "resolution_plan": resolution_plan,
            "confidence": resolution_plan.get("confidence", 0.5),
            "actual_prompt": user_prompt
        }

    except Exception as e:
        # Return fallback test plan on error - limit to 5 steps max
        fallback_steps = extracted_steps[:5] if extracted_steps else [{
            "step_number": 1,
            "description": "Escalate to senior engineer for manual test plan creation",
            "expected_result": "Test plan created by senior engineer",
            "commands": [],
            "validation": "N/A",
            "estimated_time_minutes": 0,
            "risk_level": "low",
            "rollback_procedure": None,
            "source_ticket": None,
            "source_similarity": None
        }]

        fallback = {
            "summary": f"Test plan based on {len(fallback_steps)} steps from similar tickets. (Agent error: {str(e)})",
            "diagnostic_steps": [],
            "resolution_steps": fallback_steps,
            "additional_considerations": ["Review similar tickets for additional context"],
            "references": [],
            "total_estimated_time_hours": round(len(fallback_steps) * 15 / 60, 2),
            "confidence": avg_similarity if avg_similarity > 0 else 0.5,
            "alternative_approaches": []
        }

        return {
            "resolution_plan": fallback,
            "confidence": fallback["confidence"],
            "actual_prompt": user_prompt
        }


def _build_resolution_plan(
    data: Dict[str, Any],
    extracted_steps: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build and validate resolution plan from agent response data.

    Args:
        data: Raw JSON data from agent response (summary, test_plan, considerations, etc.)
        extracted_steps: Pre-extracted resolution steps (used as fallback only)

    Returns:
        Validated resolution plan dict
    """
    # Use agent-generated test_plan (synthesized steps) - limit to 5 max
    raw_test_plan = data.get("test_plan", [])[:5]

    # Convert test_plan to resolution_steps format for frontend compatibility
    resolution_steps = []
    for i, step in enumerate(raw_test_plan, 1):
        source_tickets = step.get("source_tickets", [])
        resolution_steps.append({
            "step_number": step.get("step_number", i),
            "description": step.get("description", ""),
            "expected_result": step.get("expected_result", ""),
            "commands": [],  # Test plan steps don't have commands
            "validation": step.get("validation", "Verify step completed"),
            "estimated_time_minutes": step.get("estimated_time_minutes", 15),
            "risk_level": "low",
            "rollback_procedure": None,
            "source_ticket": ", ".join(source_tickets) if source_tickets else None,
            "source_similarity": None
        })

    # Fallback to extracted steps if agent didn't generate test_plan
    if not resolution_steps and extracted_steps:
        resolution_steps = extracted_steps[:5]  # Also limit fallback to 5

    # Calculate total time from resolution steps
    total_minutes = sum(s.get("estimated_time_minutes", 15) for s in resolution_steps)
    total_time = round(total_minutes / 60, 2)

    # Process references from agent response
    references = []
    raw_references = data.get("references", [])
    for ref in raw_references:
        if isinstance(ref, dict):
            references.append({
                "ticket_id": ref.get("ticket_id", "Unknown"),
                "similarity": ref.get("similarity", 0.0),
                "note": ref.get("note", "")
            })
        elif isinstance(ref, str):
            references.append({
                "ticket_id": ref,
                "similarity": 0.0,
                "note": ""
            })

    return {
        "summary": data.get("summary", "Test plan based on similar historical tickets"),
        "diagnostic_steps": [],  # No longer generating diagnostic steps
        "resolution_steps": resolution_steps,  # Now contains synthesized test plan steps
        "additional_considerations": data.get("additional_considerations", []),
        "references": references,
        "total_estimated_time_hours": total_time,
        "confidence": data.get("confidence", 0.5),
        "alternative_approaches": []  # Removed alternative_approaches for test plan focus
    }
