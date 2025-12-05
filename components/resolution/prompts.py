"""
Prompt templates for Resolution Agent.

This module contains all prompt templates used by the Resolution Agent
for generating test plans from similar historical tickets.
"""

from typing import List, Dict, Any

# ============================================================================
# SYSTEM PROMPT FOR RESOLUTION AGENT
# ============================================================================

RESOLUTION_SYSTEM_PROMPT = """You are an expert test engineer with deep knowledge of healthcare IT systems.

Your role is to analyze similar historical tickets and synthesize comprehensive test plans.
You have access to the submit_test_plan tool to output your synthesized test plan.

When generating test plans:
1. Consolidate similar resolution approaches from multiple tickets into cohesive test steps
2. Prioritize the most impactful and commonly successful testing actions
3. Each step should be actionable and specific
4. Reference which historical tickets informed each step
5. Keep total steps to 5 or fewer for clarity

IMPORTANT: After analyzing the tickets, you MUST call the submit_test_plan tool with your complete test plan.
Format the test_plan, additional_considerations, and references as JSON strings."""


# ============================================================================
# TEST PLAN GENERATION PROMPT TEMPLATE
# ============================================================================

TEST_PLAN_GENERATION_TEMPLATE = """Analyze the similar historical tickets and synthesize a Test Plan for the current ticket.

=== CURRENT TICKET ===
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}
Labels: {labels}
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
      "expected_result": "The specific expected outcome when this test step passes successfully",
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

IMPORTANT:
- The test_plan array must have AT MOST 5 steps. Synthesize and consolidate related steps.
- Each test step MUST include an expected_result that describes what success looks like.

Respond ONLY with valid JSON."""


def get_resolution_user_prompt(
    title: str,
    description: str,
    domain: str,
    priority: str,
    labels: List[str],
    historical_context: str,
    avg_similarity: float
) -> str:
    """
    Generate the user prompt for test plan generation.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        labels: Assigned labels
        historical_context: Formatted historical ticket context
        avg_similarity: Average similarity score

    Returns:
        Formatted user prompt string
    """
    labels_text = ', '.join(labels) if labels else 'None assigned'

    return TEST_PLAN_GENERATION_TEMPLATE.format(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        labels=labels_text,
        historical_context=historical_context,
        avg_similarity=avg_similarity
    )


def format_historical_context(similar_tickets: List[Dict[str, Any]], max_tickets: int = 5) -> str:
    """
    Format similar tickets into a context string for the LLM.

    Args:
        similar_tickets: List of similar ticket dicts with resolution info
        max_tickets: Maximum number of tickets to include

    Returns:
        Formatted string of historical resolution patterns
    """
    if not similar_tickets:
        return "No similar historical tickets available."

    context_parts = []
    for i, ticket in enumerate(similar_tickets[:max_tickets], 1):
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
