"""
Prompt templates for resolution generation using Chain-of-Thought.
"""
from typing import List, Dict

RESOLUTION_GENERATION_PROMPT = """You are an expert technical support engineer generating resolution steps for a support ticket.

Ticket Context:
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}
Assigned Labels: {labels}

Historical Analysis:
- Found {num_similar} similar resolved tickets
- Average similarity: {avg_similarity:.2%}
- Common patterns identified in similar tickets

Top 5 Most Similar Resolved Tickets:
{top_similar_tickets}

Task: Generate a comprehensive, actionable resolution plan for this ticket.

Requirements:
1. **Diagnostic Steps**: Start with 2-4 diagnostic steps to confirm the issue
   - Each step should include commands/queries to run
   - Specify expected output
   - Estimate time in minutes

2. **Resolution Steps**: Provide 3-7 resolution steps that fix the issue
   - Step-by-step instructions with specific commands
   - Include validation method for each step
   - Specify risk level (low/medium/high)
   - Provide rollback procedure for risky steps
   - Estimate time in minutes for each step

3. **Additional Considerations**: List 2-3 important notes
   - Post-resolution monitoring
   - Preventive measures
   - Related documentation

4. **References**: Link to 2-3 most relevant similar tickets
   - Include ticket ID, similarity score, and brief note

5. **Total Time Estimate**: Sum of all step times in hours

6. **Confidence**: Your confidence in this resolution plan (0.0-1.0)

7. **Alternative Approaches**: If main approach doesn't work, provide 1-2 alternatives

Guidelines:
- Be specific with commands, file paths, and configurations
- Use actual values from similar tickets when applicable
- Prioritize actions that worked in similar historical cases
- Consider the domain ({domain}) context for technical details
- Adapt resolution complexity to priority level ({priority})

Output Format (JSON only):
{{
  "summary": "Brief 2-3 sentence overview of the resolution approach",
  "diagnostic_steps": [
    {{
      "step_number": 1,
      "description": "Clear description of diagnostic action",
      "commands": ["specific command to run"],
      "expected_output": "What success looks like",
      "estimated_time_minutes": 5
    }}
  ],
  "resolution_steps": [
    {{
      "step_number": 1,
      "description": "Clear description of resolution action",
      "commands": ["specific command or action"],
      "validation": "How to verify this step succeeded",
      "estimated_time_minutes": 10,
      "risk_level": "low",
      "rollback_procedure": "How to undo if needed (or null if not applicable)"
    }}
  ],
  "additional_considerations": [
    "Important note 1",
    "Important note 2"
  ],
  "references": [
    {{
      "ticket_id": "JIRA-XXX",
      "similarity": 0.XX,
      "note": "Brief note on relevance"
    }}
  ],
  "total_estimated_time_hours": 0.0,
  "confidence": 0.0 to 1.0,
  "alternative_approaches": [
    "Alternative 1 if main approach doesn't work",
    "Alternative 2"
  ]
}}

Respond ONLY with valid JSON. Do not include any other text."""


def get_resolution_generation_prompt(
    title: str,
    description: str,
    domain: str,
    priority: str,
    labels: List[str],
    similar_tickets: List[Dict],
    avg_similarity: float
) -> str:
    """
    Generate resolution generation prompt with context.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        labels: Assigned labels
        similar_tickets: List of similar tickets (use top 5)
        avg_similarity: Average similarity score

    Returns:
        Formatted prompt string
    """
    from src.utils.helpers import truncate_text

    # Format top 5 similar tickets
    top_5 = similar_tickets[:5]
    formatted_tickets = []

    for i, ticket in enumerate(top_5, 1):
        # Format resolution steps
        res_steps = ticket.get('resolution_steps', [])
        formatted_steps = '\n   '.join([f"- {step}" for step in res_steps[:5]])

        formatted_tickets.append(f"""
Ticket {i}: {ticket.get('ticket_id', 'Unknown')} (Similarity: {ticket.get('similarity_score', 0):.2%})
Title: {ticket.get('title', 'N/A')}
Description: {truncate_text(ticket.get('description', 'N/A'), max_tokens=100)}
Labels: {', '.join(ticket.get('labels', []))}
Priority: {ticket.get('priority', 'N/A')}
Resolution Time: {ticket.get('resolution_time_hours', 0)} hours
Resolution Steps:
   {formatted_steps}
""")

    similar_tickets_text = "\n".join(formatted_tickets)

    # Format labels
    labels_text = ', '.join(labels) if labels else 'None assigned'

    # Generate prompt
    return RESOLUTION_GENERATION_PROMPT.format(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        labels=labels_text,
        num_similar=len(similar_tickets),
        avg_similarity=avg_similarity,
        top_similar_tickets=similar_tickets_text
    )
