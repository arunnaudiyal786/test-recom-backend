"""
Resolution Tools - LangChain @tool decorated functions for resolution generation.

These tools extract resolution steps from similar historical tickets and generate
comprehensive resolution plans with summary and considerations.
"""

import json
from typing import Dict, Any, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Fix Pydantic V2 forward reference issue with ChatOpenAI
ChatOpenAI.model_rebuild()

from config.config import Config
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

    # Prompt for synthesized Test Plan generation
    user_prompt = f"""You are an expert test engineer with deep knowledge of healthcare IT systems.

Analyze the similar historical tickets and synthesize a Test Plan for the current ticket.

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
Based on the resolution patterns from similar historical tickets, synthesize a **Test Plan** with AT MOST 5 steps.

Guidelines for synthesizing the Test Plan:
1. Consolidate similar resolution approaches from multiple tickets into cohesive test steps
2. Prioritize the most impactful and commonly successful testing actions
3. Each step should be actionable and specific
4. Reference which historical tickets informed each step
5. Keep total steps to 5 or fewer for clarity

Also provide:
- An executive summary of the recommended test approach
- Additional considerations for the QA/test team
- A confidence score based on how similar the historical cases are

=== OUTPUT FORMAT (JSON) ===
{{
  "summary": "Executive summary of the recommended test approach (2-3 sentences)",
  "test_plan": [
    {{
      "step_number": 1,
      "description": "Clear, actionable test step synthesized from historical tickets",
      "validation": "How to verify this step was successful",
      "source_tickets": ["TICKET-123", "TICKET-456"],
      "estimated_time_minutes": 15
    }}
  ],
  "additional_considerations": ["consideration1", "consideration2"],
  "references": [
    {{"ticket_id": "TICKET-123", "similarity": 0.95, "note": "Similar issue with same root cause"}}
  ],
  "confidence": 0.85
}}

IMPORTANT: The test_plan array must have AT MOST 5 steps. Synthesize and consolidate related steps.

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
        # Return fallback test plan on error - limit to 5 steps max
        fallback_steps = extracted_steps[:5] if extracted_steps else [{
            "step_number": 1,
            "description": "Escalate to senior engineer for manual test plan creation",
            "commands": [],
            "validation": "N/A",
            "estimated_time_minutes": 0,
            "risk_level": "low",
            "rollback_procedure": None,
            "source_ticket": None,
            "source_similarity": None
        }]

        fallback = {
            "summary": f"Test plan based on {len(fallback_steps)} steps from similar tickets. (LLM synthesis failed: {str(e)})",
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
    Build and validate resolution plan from LLM data with synthesized test plan.

    Args:
        data: Raw JSON data from LLM response (summary, test_plan, considerations, etc.)
        extracted_steps: Pre-extracted resolution steps (used as fallback only)

    Returns:
        Validated resolution plan dict
    """
    # Use LLM-generated test_plan (synthesized steps) - limit to 5 max
    raw_test_plan = data.get("test_plan", [])[:5]

    # Convert test_plan to resolution_steps format for frontend compatibility
    resolution_steps = []
    for i, step in enumerate(raw_test_plan, 1):
        source_tickets = step.get("source_tickets", [])
        resolution_steps.append({
            "step_number": step.get("step_number", i),
            "description": step.get("description", ""),
            "commands": [],  # Test plan steps don't have commands
            "validation": step.get("validation", "Verify step completed"),
            "estimated_time_minutes": step.get("estimated_time_minutes", 15),
            "risk_level": "low",
            "rollback_procedure": None,
            "source_ticket": ", ".join(source_tickets) if source_tickets else None,
            "source_similarity": None
        })

    # Fallback to extracted steps if LLM didn't generate test_plan
    if not resolution_steps and extracted_steps:
        resolution_steps = extracted_steps[:5]  # Also limit fallback to 5

    # Calculate total time from resolution steps
    total_minutes = sum(s.get("estimated_time_minutes", 15) for s in resolution_steps)
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
        "summary": data.get("summary", "Test plan based on similar historical tickets"),
        "diagnostic_steps": [],  # No longer generating diagnostic steps
        "resolution_steps": resolution_steps,  # Now contains synthesized test plan steps
        "additional_considerations": data.get("additional_considerations", []),
        "references": references,
        "total_estimated_time_hours": total_time,
        "confidence": data.get("confidence", 0.5),
        "alternative_approaches": []  # Removed alternative_approaches for test plan focus
    }
