"""
Prompt templates for ticket summarization in the Data Preparation Pipeline.

These prompts guide the LLM to generate concise, structured summaries
of Jira tickets highlighting key aspects for downstream processing.
"""


def get_ticket_summarization_prompt(
    summary: str,
    description: str,
    priority: str = "",
    issue_type: str = "",
    labels: str = "",
    resolution: str = ""
) -> str:
    """
    Generate a prompt for summarizing a single ticket.

    TODO: You can customize this prompt to highlight specific aspects
    that are important for your use case.

    Args:
        summary: Ticket summary/title
        description: Full ticket description
        priority: Issue priority (Low/Medium/High/Critical)
        issue_type: Type of issue (Bug, Feature Request, etc.)
        labels: Associated labels
        resolution: Resolution steps (if available)

    Returns:
        Formatted prompt string
    """
    # ========== YOUR IMPLEMENTATION HERE ==========
    # Customize what aspects the summarizer should focus on.
    #
    # Consider:
    # - Technical keywords to extract
    # - Problem severity indicators
    # - Component/system mentions
    # - Root cause indicators
    # - Impact scope
    #
    # Current implementation focuses on:
    # 1. Core issue identification
    # 2. Technical components involved
    # 3. Business impact
    # 4. Resolution approach

    prompt = f"""You are an AI technical ticket summarizer for a healthcare IT support system.

Your task is to create a concise, structured summary of the ticket that captures the essential information for downstream AI processing and human review.

Ticket Information:
- Title: {summary}
- Description: {description}
- Priority: {priority if priority else "Not specified"}
- Issue Type: {issue_type if issue_type else "Not specified"}
- Labels: {labels if labels else "None"}
- Resolution: {resolution if resolution else "Not provided"}

Summarization Guidelines:
1. Extract the CORE PROBLEM in 1-2 sentences
2. Identify KEY TECHNICAL COMPONENTS (services, systems, databases)
3. Note the BUSINESS IMPACT (users affected, severity)
4. Highlight any ROOT CAUSE indicators
5. Summarize the RESOLUTION APPROACH (if resolution provided)
6. Keep total summary under 150 words

Output Format (JSON only):
{{
  "core_problem": "Brief description of the main issue",
  "technical_components": ["component1", "component2"],
  "business_impact": "Impact on users/systems",
  "root_cause_indicators": "Any hints about root cause",
  "resolution_summary": "Brief resolution approach if available",
  "severity_assessment": "Low|Medium|High|Critical based on content",
  "key_keywords": ["keyword1", "keyword2", "keyword3"],
  "combined_summary": "A single paragraph combining all aspects above (100-150 words)"
}}

Important:
- Be objective and technical
- Focus on facts, not opinions
- Use domain-specific terminology appropriately
- The combined_summary should be suitable for vector embedding

Respond ONLY with valid JSON. Do not include any other text."""

    return prompt

    # ========== END OF YOUR IMPLEMENTATION ==========


BATCH_SUMMARIZATION_SYSTEM_PROMPT = """You are an expert technical ticket summarizer for healthcare IT support systems.

Your role is to:
1. Analyze technical support tickets
2. Extract key information efficiently
3. Create concise, actionable summaries
4. Identify patterns and common issues
5. Maintain consistency in summarization format

You excel at:
- Understanding healthcare IT terminology (MM, CIW, claims processing, member management)
- Identifying technical components and their interactions
- Assessing business impact and severity
- Extracting root cause indicators from symptoms
- Summarizing resolutions in actionable formats

Always output valid JSON following the specified schema."""


def get_batch_summary_prompt(tickets_batch: list) -> str:
    """
    Generate a prompt for batch summarization (multiple tickets).

    Args:
        tickets_batch: List of ticket dictionaries

    Returns:
        Formatted prompt for batch processing
    """
    tickets_text = ""
    for idx, ticket in enumerate(tickets_batch):
        tickets_text += f"""
---
Ticket {idx + 1}:
Title: {ticket.get('Summary', ticket.get('summary', 'N/A'))}
Description: {ticket.get('Description', ticket.get('description', 'N/A'))}
Priority: {ticket.get('Issue Priority', ticket.get('priority', 'N/A'))}
Issue Type: {ticket.get('issue type', 'N/A')}
Labels: {ticket.get('Labels', 'N/A')}
Resolution: {ticket.get('Resolution', 'N/A')}
---
"""

    prompt = f"""Summarize the following {len(tickets_batch)} technical support tickets.

For each ticket, provide a JSON object with the summarization.

Tickets:
{tickets_text}

Output Format (JSON array):
[
  {{
    "ticket_index": 0,
    "core_problem": "Brief description",
    "technical_components": ["comp1", "comp2"],
    "business_impact": "Impact description",
    "root_cause_indicators": "Indicators",
    "resolution_summary": "Resolution if available",
    "severity_assessment": "Low|Medium|High|Critical",
    "key_keywords": ["kw1", "kw2"],
    "combined_summary": "100-150 word summary paragraph"
  }},
  ...
]

Respond ONLY with a valid JSON array containing {len(tickets_batch)} objects.
Do not include any other text."""

    return prompt
