"""
Prompt templates for Labeling Agent.

This module contains all prompt templates used by the Labeling Agent
for category classification and label generation.
"""

from typing import List, Dict, Any

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _format_similar_tickets_context(
    similar_tickets: List[Dict[str, Any]],
    max_tickets: int = 5
) -> str:
    """
    Format similar tickets for prompt context.

    Provides historical context from Historical Match Agent output
    to help inform business and technical label generation.

    Args:
        similar_tickets: List of similar ticket dicts from retrieval agent
        max_tickets: Maximum number of tickets to include (default 5)

    Returns:
        Formatted string for prompt context
    """
    if not similar_tickets:
        return "No similar historical tickets found."

    lines = []
    for i, ticket in enumerate(similar_tickets[:max_tickets], 1):
        # Handle both "title" and "summary" field names
        title = ticket.get("title", ticket.get("summary", "N/A"))

        # Extract labels (may be list or comma-separated string)
        labels = ticket.get("labels", [])
        if isinstance(labels, str):
            labels_str = labels
        elif isinstance(labels, list):
            labels_str = ", ".join(labels) if labels else "None"
        else:
            labels_str = "None"

        # Get resolution and truncate for brevity
        resolution = ticket.get("resolution", "")
        if resolution:
            resolution = resolution[:200].replace("\n", " ").strip()
            if len(ticket.get("resolution", "")) > 200:
                resolution += "..."
        else:
            resolution = "N/A"

        # Get similarity score if available
        similarity = ticket.get("similarity_score", ticket.get("score", None))
        similarity_str = f" (similarity: {similarity:.2f})" if similarity else ""

        lines.append(
            f"Ticket {i}{similarity_str}:\n"
            f"  Title: {title}\n"
            f"  Labels: {labels_str}\n"
            f"  Resolution: {resolution}"
        )

    return "\n\n".join(lines)


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

BINARY_CLASSIFICATION_SYSTEM_PROMPT = """You are a ticket classification specialist for healthcare IT systems.

Your role is to determine if a ticket belongs to a specific category based on:
- Category description and keywords
- Example scenarios for the category
- The ticket's title and description

Be precise. Only return true if you are confident the category is appropriate.

IMPORTANT: After your analysis, you MUST call the submit_classification_result tool with:
- decision: true or false
- confidence: 0.0-1.0
- reasoning: brief explanation"""


BUSINESS_LABEL_SYSTEM_PROMPT = """You are a business analyst expert in IT service management.

Your role is to generate business-oriented labels for tickets from a business impact perspective.
Focus on:
- Impact: Customer-facing, Internal, Revenue-impacting
- Urgency: Time-sensitive, Compliance-related, SLA-bound
- Process: Workflow-blocking, Data-quality, Integration-issue

IMPORTANT: After your analysis, you MUST call the submit_business_labels tool with a JSON string containing an array of labels."""


TECHNICAL_LABEL_SYSTEM_PROMPT = """You are a senior software engineer expert in system diagnostics.

Your role is to generate technical labels for tickets from a technical/root-cause perspective.
Focus on:
- Component: Database, API, UI, Integration, Batch
- Issue Type: Performance, Error, Configuration, Data
- Root Cause: Connection, Timeout, Memory, Logic, External

IMPORTANT: After your analysis, you MUST call the submit_technical_labels tool with a JSON string containing an array of labels."""


# ============================================================================
# BINARY CATEGORY CLASSIFICATION TEMPLATE
# ============================================================================

BINARY_CLASSIFICATION_TEMPLATE = """Your task is to determine if this ticket belongs to ONE SPECIFIC category.

=== TICKET ===
Title: {title}
Description: {description}
Priority: {priority}

=== CATEGORY TO EVALUATE ===
ID: {category_id}
Name: {category_name}
Description: {category_description}
Keywords: {category_keywords}
Examples: {category_examples}

=== DECISION CRITERIA ===
Answer: Does this ticket belong to the "{category_name}" category?

Consider:
1. Does the ticket content align with the category description?
2. Are relevant keywords present in the ticket?
3. Would this ticket be similar to the example scenarios?
4. Is this the RIGHT category, not just a tangentially related one?

Output JSON:
{{
  "decision": true or false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this ticket does or doesn't match this category"
}}

Be precise. Only return true if you are confident this category is appropriate.
Respond ONLY with valid JSON."""


# ============================================================================
# BUSINESS LABEL GENERATION TEMPLATE
# ============================================================================

BUSINESS_LABEL_TEMPLATE = """Generate business-oriented labels for this ticket from a business impact perspective.

=== CURRENT TICKET ===
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {existing_labels}

=== SIMILAR HISTORICAL TICKETS (for reference) ===
{similar_tickets_context}

Use the patterns from similar tickets to inform your label generation.
Pay attention to what business impact labels were historically assigned to similar issues.

=== BUSINESS LABEL CATEGORIES ===
- Impact: Customer-facing, Internal, Revenue-impacting
- Urgency: Time-sensitive, Compliance-related, SLA-bound
- Process: Workflow-blocking, Data-quality, Integration-issue

Output JSON with up to {max_labels} labels:
{{
  "business_labels": [
    {{
      "label": "label name",
      "confidence": 0.0-1.0,
      "category": "Impact/Urgency/Process",
      "reasoning": "brief explanation"
    }}
  ]
}}"""


# ============================================================================
# TECHNICAL LABEL GENERATION TEMPLATE
# ============================================================================

TECHNICAL_LABEL_TEMPLATE = """Generate technical labels for this ticket from a technical/root-cause perspective.

=== CURRENT TICKET ===
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {existing_labels}

=== SIMILAR HISTORICAL TICKETS (for reference) ===
{similar_tickets_context}

Use the patterns from similar tickets to inform your label generation.
Pay attention to what technical labels and root causes were historically identified for similar issues.

=== TECHNICAL LABEL CATEGORIES ===
- Component: Database, API, UI, Integration, Batch
- Issue Type: Performance, Error, Configuration, Data
- Root Cause: Connection, Timeout, Memory, Logic, External

Output JSON with up to {max_labels} labels:
{{
  "technical_labels": [
    {{
      "label": "label name",
      "confidence": 0.0-1.0,
      "category": "Component/Issue Type/Root Cause",
      "reasoning": "brief explanation"
    }}
  ]
}}"""


# ============================================================================
# PROMPT GENERATION FUNCTIONS
# ============================================================================

def get_binary_classification_prompt(
    title: str,
    description: str,
    priority: str,
    category_id: str,
    category_name: str,
    category_description: str,
    category_keywords: str,
    category_examples: str
) -> str:
    """
    Generate binary classification prompt for a single category.

    Args:
        title: Ticket title
        description: Ticket description
        priority: Ticket priority
        category_id: Category ID to evaluate
        category_name: Category name
        category_description: Category description
        category_keywords: Comma-separated keywords
        category_examples: Comma-separated example scenarios

    Returns:
        Formatted prompt for binary classification
    """
    return BINARY_CLASSIFICATION_TEMPLATE.format(
        title=title,
        description=description,
        priority=priority,
        category_id=category_id,
        category_name=category_name,
        category_description=category_description,
        category_keywords=category_keywords,
        category_examples=category_examples
    )


def get_business_label_prompt(
    title: str,
    description: str,
    domain: str,
    priority: str,
    existing_labels: List[str],
    max_labels: int = 5,
    similar_tickets: List[Dict[str, Any]] = None
) -> str:
    """
    Generate prompt for business label generation.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate
        similar_tickets: List of similar tickets for historical context (top 5)

    Returns:
        Formatted prompt for business label generation
    """
    existing_labels_text = ', '.join(existing_labels) if existing_labels else 'None'
    similar_tickets_context = _format_similar_tickets_context(similar_tickets or [])

    return BUSINESS_LABEL_TEMPLATE.format(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        existing_labels=existing_labels_text,
        max_labels=max_labels,
        similar_tickets_context=similar_tickets_context
    )


def get_technical_label_prompt(
    title: str,
    description: str,
    domain: str,
    priority: str,
    existing_labels: List[str],
    max_labels: int = 5,
    similar_tickets: List[Dict[str, Any]] = None
) -> str:
    """
    Generate prompt for technical label generation.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate
        similar_tickets: List of similar tickets for historical context (top 5)

    Returns:
        Formatted prompt for technical label generation
    """
    existing_labels_text = ', '.join(existing_labels) if existing_labels else 'None'
    similar_tickets_context = _format_similar_tickets_context(similar_tickets or [])

    return TECHNICAL_LABEL_TEMPLATE.format(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        existing_labels=existing_labels_text,
        max_labels=max_labels,
        similar_tickets_context=similar_tickets_context
    )
