"""
Resolution Tools - LangChain @tool decorated functions for resolution generation.

These tools analyze similar tickets and generate comprehensive resolution plans
using Chain-of-Thought reasoning.
"""

import json
from typing import Dict, Any, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.utils.config import Config
from components.resolution.models import (
    DiagnosticStep,
    ResolutionStep,
    ResolutionPlan
)


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
    if not similar_tickets:
        return "No similar historical tickets available."

    context_parts = []
    for i, ticket in enumerate(similar_tickets[:5], 1):
        resolution = ticket.get("resolution", "No resolution recorded")
        context_parts.append(f"""
--- Historical Ticket {i} ---
ID: {ticket.get('ticket_id', 'N/A')}
Title: {ticket.get('title', 'N/A')}
Similarity: {ticket.get('similarity_score', 0):.2%}
Labels: {', '.join(ticket.get('labels', []))}
Resolution Time: {ticket.get('resolution_time_hours', 'N/A')} hours
Resolution:
{resolution}
""")

    return "\n".join(context_parts)


@tool
async def generate_resolution_plan(
    title: str,
    description: str,
    domain: str,
    priority: str,
    labels: List[str],
    historical_context: str,
    avg_similarity: float
) -> Dict[str, Any]:
    """
    Generate a comprehensive resolution plan using Chain-of-Thought reasoning.

    Analyzes the ticket context and similar historical resolutions to generate
    detailed diagnostic and resolution steps.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        labels: Assigned labels
        historical_context: Formatted string of similar ticket resolutions
        avg_similarity: Average similarity to historical tickets

    Returns:
        Dict containing:
        - resolution_plan: Complete resolution plan dict
        - confidence: Confidence score (0-1)
    """
    llm = ChatOpenAI(
        model=Config.RESOLUTION_MODEL,
        temperature=0.6,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    prompt = f"""You are an expert technical support engineer with deep knowledge of healthcare IT systems.

Generate a comprehensive resolution plan for the following ticket using Chain-of-Thought reasoning.

=== CURRENT TICKET ===
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}
Labels: {', '.join(labels)}
Average similarity to historical tickets: {avg_similarity:.2%}

=== SIMILAR HISTORICAL TICKETS ===
{historical_context}

=== CHAIN-OF-THOUGHT PROCESS ===
1. Analyze the current ticket's symptoms and context
2. Review patterns from similar historical tickets
3. Identify likely root causes
4. Plan diagnostic steps to confirm the issue
5. Develop resolution steps with proper validation
6. Consider risks and rollback procedures
7. Estimate time and assign confidence level

=== OUTPUT FORMAT (JSON) ===
{{
  "summary": "Executive summary of the resolution approach (2-3 sentences)",
  "diagnostic_steps": [
    {{
      "step_number": 1,
      "description": "What to check/verify",
      "commands": ["command1", "command2"],
      "expected_output": "What to expect",
      "estimated_time_minutes": 5
    }}
  ],
  "resolution_steps": [
    {{
      "step_number": 1,
      "description": "What action to take",
      "commands": ["command1", "command2"],
      "validation": "How to verify the step worked",
      "estimated_time_minutes": 10,
      "risk_level": "low|medium|high",
      "rollback_procedure": "How to rollback if needed"
    }}
  ],
  "additional_considerations": ["consideration1", "consideration2"],
  "references": ["reference to docs/tickets"],
  "total_estimated_time_hours": 2.5,
  "confidence": 0.85,
  "alternative_approaches": ["Alternative approach 1", "Alternative approach 2"]
}}

Generate a detailed, actionable resolution plan. Respond ONLY with valid JSON."""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are an expert technical support engineer. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ])

        data = json.loads(response.content)

        # Validate and build resolution plan
        resolution_plan = _build_resolution_plan(data)

        return {
            "resolution_plan": resolution_plan,
            "confidence": resolution_plan.get("confidence", 0.5)
        }

    except Exception as e:
        # Return fallback plan on error
        fallback = {
            "summary": f"Error generating resolution plan: {str(e)}",
            "diagnostic_steps": [{
                "step_number": 1,
                "description": "Manual review required",
                "commands": [],
                "expected_output": "N/A",
                "estimated_time_minutes": 0
            }],
            "resolution_steps": [{
                "step_number": 1,
                "description": "Escalate to senior engineer for manual resolution",
                "commands": [],
                "validation": "N/A",
                "estimated_time_minutes": 0,
                "risk_level": "low",
                "rollback_procedure": None
            }],
            "additional_considerations": ["Automatic resolution generation failed"],
            "references": [],
            "total_estimated_time_hours": 0,
            "confidence": 0.0,
            "alternative_approaches": []
        }

        return {
            "resolution_plan": fallback,
            "confidence": 0.0
        }


def _build_resolution_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build and validate resolution plan from raw data.

    Args:
        data: Raw JSON data from LLM response

    Returns:
        Validated resolution plan dict
    """
    # Defaults
    defaults = {
        "summary": "Resolution plan generated",
        "diagnostic_steps": [],
        "resolution_steps": [],
        "additional_considerations": [],
        "references": [],
        "total_estimated_time_hours": 0.0,
        "confidence": 0.5,
        "alternative_approaches": []
    }

    for key, default_value in defaults.items():
        if key not in data:
            data[key] = default_value

    # Convert diagnostic steps
    diagnostic_steps = []
    for i, step in enumerate(data.get("diagnostic_steps", []), 1):
        diagnostic_steps.append({
            "step_number": step.get("step_number", i),
            "description": step.get("description", ""),
            "commands": step.get("commands", []),
            "expected_output": step.get("expected_output", ""),
            "estimated_time_minutes": step.get("estimated_time_minutes", 5)
        })

    # Convert resolution steps
    resolution_steps = []
    for i, step in enumerate(data.get("resolution_steps", []), 1):
        resolution_steps.append({
            "step_number": step.get("step_number", i),
            "description": step.get("description", ""),
            "commands": step.get("commands", []),
            "validation": step.get("validation", "Verify step completed"),
            "estimated_time_minutes": step.get("estimated_time_minutes", 10),
            "risk_level": step.get("risk_level", "low"),
            "rollback_procedure": step.get("rollback_procedure")
        })

    # Calculate total time if not provided
    total_time = data.get("total_estimated_time_hours", 0)
    if total_time == 0:
        total_minutes = sum(s["estimated_time_minutes"] for s in diagnostic_steps)
        total_minutes += sum(s["estimated_time_minutes"] for s in resolution_steps)
        total_time = round(total_minutes / 60, 2)

    return {
        "summary": data.get("summary", ""),
        "diagnostic_steps": diagnostic_steps,
        "resolution_steps": resolution_steps,
        "additional_considerations": data.get("additional_considerations", []),
        "references": data.get("references", []),
        "total_estimated_time_hours": total_time,
        "confidence": data.get("confidence", 0.5),
        "alternative_approaches": data.get("alternative_approaches", [])
    }
