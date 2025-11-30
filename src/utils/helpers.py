"""
Helper functions for text processing and data manipulation.
"""
import re
from typing import List, Dict, Any
import html


def clean_text(text: str) -> str:
    """
    Clean and normalize text for embedding generation.

    Args:
        text: Raw input text

    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)

    # Lowercase for consistency
    text = text.lower().strip()

    return text


def combine_ticket_text(title: str, description: str) -> str:
    """
    Combine ticket title and description for embedding.

    Args:
        title: Ticket title
        description: Ticket description

    Returns:
        Combined text optimized for embedding
    """
    cleaned_title = clean_text(title)
    cleaned_desc = clean_text(description)

    # Format: "Title. Description"
    return f"{cleaned_title}. {cleaned_desc}"


def truncate_text(text: str, max_tokens: int = 8000) -> str:
    """
    Truncate text to approximate token limit.

    Rough estimation: 1 token ≈ 4 characters for English text.

    Args:
        text: Input text
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from text that may contain markdown code blocks.

    Args:
        text: Text potentially containing JSON in code blocks

    Returns:
        Extracted JSON string
    """
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find raw JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return text


def format_similar_tickets_for_prompt(similar_tickets: List[Dict[str, Any]], limit: int = 5) -> str:
    """
    Format similar tickets for inclusion in LLM prompts.

    Args:
        similar_tickets: List of similar ticket dicts
        limit: Max number to include in formatted output

    Returns:
        Formatted string representation
    """
    formatted = []

    for i, ticket in enumerate(similar_tickets[:limit], 1):
        formatted.append(f"""
Ticket {i}: {ticket.get('ticket_id', 'Unknown')} (Similarity: {ticket.get('similarity_score', 0):.2f})
Title: {ticket.get('title', 'N/A')}
Description: {truncate_text(ticket.get('description', 'N/A'), max_tokens=100)}
Labels: {', '.join(ticket.get('labels', []))}
Resolution Steps:
{format_resolution_steps(ticket.get('resolution_steps', []))}
""")

    return "\n".join(formatted)


def format_resolution_steps(steps: List[str], max_steps: int = 5) -> str:
    """
    Format resolution steps for display.

    Args:
        steps: List of resolution step strings
        max_steps: Maximum number of steps to display

    Returns:
        Formatted steps string
    """
    if not steps:
        return "  No resolution steps available"

    formatted_steps = []
    for i, step in enumerate(steps[:max_steps], 1):
        formatted_steps.append(f"  {step}")

    if len(steps) > max_steps:
        formatted_steps.append(f"  ... and {len(steps) - max_steps} more steps")

    return "\n".join(formatted_steps)


def calculate_label_distribution(similar_tickets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate label frequency distribution from similar tickets.

    Args:
        similar_tickets: List of similar ticket dicts

    Returns:
        Dict mapping label name to frequency info
    """
    label_counts: Dict[str, int] = {}
    total_tickets = len(similar_tickets)

    # Count label occurrences
    for ticket in similar_tickets:
        labels = ticket.get('labels', [])
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Calculate percentages and format
    distribution = {}
    for label, count in label_counts.items():
        percentage = (count / total_tickets * 100) if total_tickets > 0 else 0
        distribution[label] = {
            "count": count,
            "total": total_tickets,
            "percentage": percentage,
            "formatted": f"{count}/{total_tickets} ({percentage:.0f}%)"
        }

    return distribution


def format_confidence_scores(scores: Dict[str, float]) -> str:
    """
    Format confidence scores for display.

    Args:
        scores: Dict mapping category/label to confidence score

    Returns:
        Formatted string
    """
    formatted = []
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        formatted.append(f"  {name}: {score:.2%}")

    return "\n".join(formatted)


def validate_domain(domain: str) -> bool:
    """
    Validate that domain is one of the allowed values.

    Args:
        domain: Domain string to validate

    Returns:
        True if valid, False otherwise
    """
    valid_domains = {"MM", "CIW", "Specialty"}
    return domain in valid_domains


def validate_priority(priority: str) -> bool:
    """
    Validate that priority is one of the allowed values.

    Args:
        priority: Priority string to validate

    Returns:
        True if valid, False otherwise
    """
    valid_priorities = {"Low", "Medium", "High", "Critical"}
    return priority in valid_priorities
