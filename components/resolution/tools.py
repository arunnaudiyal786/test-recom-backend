"""
Resolution Tools - LangChain @tool decorated functions for resolution generation.

These tools extract resolution steps from similar historical tickets and generate
comprehensive resolution plans with summary and considerations.
"""

import json
from typing import Dict, Any, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import Config
from components.resolution.models import (
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
    Generate a comprehensive resolution plan by extracting steps from similar tickets.

    Extracts resolution steps from similar historical tickets and uses LLM to
    generate a summary and additional considerations.

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
    """
    # Extract resolution steps from similar tickets
    extracted_steps = _extract_resolution_steps_from_tickets(similar_tickets, max_tickets=5)

    # Build historical context for LLM summary generation
    historical_context = analyze_similar_resolutions.invoke({"similar_tickets": similar_tickets})

    llm = ChatOpenAI(
        model=Config.RESOLUTION_MODEL,
        temperature=0.6,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Simplified prompt - only generate summary and considerations
    user_prompt = f"""You are an expert technical support engineer with deep knowledge of healthcare IT systems.

Generate a summary and additional considerations for resolving the following ticket based on similar historical cases.

=== CURRENT TICKET ===
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}
Labels: {', '.join(labels)}
Average similarity to historical tickets: {avg_similarity:.2%}

=== SIMILAR HISTORICAL TICKETS ===
{historical_context}

=== TASK ===
Based on the similar historical tickets, provide:
1. An executive summary of the recommended resolution approach
2. Additional considerations for the support team
3. References to the most relevant historical tickets
4. Alternative approaches if the primary resolution doesn't work
5. A confidence score based on how similar the historical cases are

NOTE: Resolution steps will be extracted directly from similar tickets. Focus on synthesizing the approach.

=== OUTPUT FORMAT (JSON) ===
{{
  "summary": "Executive summary of the resolution approach (2-3 sentences)",
  "additional_considerations": ["consideration1", "consideration2"],
  "references": [
    {{"ticket_id": "TICKET-123", "similarity": 0.95, "note": "Similar issue with same root cause"}}
  ],
  "confidence": 0.85,
  "alternative_approaches": ["Alternative approach 1", "Alternative approach 2"]
}}

Respond ONLY with valid JSON."""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are an expert technical support engineer. Respond only with valid JSON."},
            {"role": "user", "content": user_prompt}
        ])

        data = json.loads(response.content)

        # Build resolution plan with extracted steps
        resolution_plan = _build_resolution_plan(data, extracted_steps)

        return {
            "resolution_plan": resolution_plan,
            "confidence": resolution_plan.get("confidence", 0.5),
            "actual_prompt": user_prompt
        }

    except Exception as e:
        # Return fallback plan on error with extracted steps
        fallback = {
            "summary": f"Resolution plan based on {len(extracted_steps)} steps from similar tickets. (LLM summary failed: {str(e)})",
            "diagnostic_steps": [],
            "resolution_steps": extracted_steps if extracted_steps else [{
                "step_number": 1,
                "description": "Escalate to senior engineer for manual resolution",
                "commands": [],
                "validation": "N/A",
                "estimated_time_minutes": 0,
                "risk_level": "low",
                "rollback_procedure": None,
                "source_ticket": None,
                "source_similarity": None
            }],
            "additional_considerations": ["Review similar tickets for additional context"],
            "references": [],
            "total_estimated_time_hours": round(len(extracted_steps) * 10 / 60, 2),
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
    Build and validate resolution plan from LLM data and extracted steps.

    Args:
        data: Raw JSON data from LLM response (summary, considerations, etc.)
        extracted_steps: Pre-extracted resolution steps from similar tickets

    Returns:
        Validated resolution plan dict
    """
    # Use extracted steps or empty list
    resolution_steps = extracted_steps if extracted_steps else []

    # Calculate total time from resolution steps
    total_minutes = sum(s.get("estimated_time_minutes", 10) for s in resolution_steps)
    total_time = round(total_minutes / 60, 2)

    # Process references from LLM response
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
        "summary": data.get("summary", "Resolution plan based on similar historical tickets"),
        "diagnostic_steps": [],  # No longer generating diagnostic steps
        "resolution_steps": resolution_steps,
        "additional_considerations": data.get("additional_considerations", []),
        "references": references,
        "total_estimated_time_hours": total_time,
        "confidence": data.get("confidence", 0.5),
        "alternative_approaches": data.get("alternative_approaches", [])
    }
