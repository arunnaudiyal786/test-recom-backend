"""
Prompt templates for label assignment using binary classifiers.
Includes templates for:
- Historical label assignment (from similar tickets)
- Business label generation (AI-generated)
- Technical label generation (AI-generated)
"""

# ============================================================================
# HISTORICAL LABEL ASSIGNMENT TEMPLATE (Existing)
# ============================================================================

LABEL_ASSIGNMENT_TEMPLATE = """You are a label assignment specialist for technical support tickets.

Task: Determine if the ticket should be assigned the label "{label_name}"

Context from Historical Analysis:
- {total_similar} similar tickets found in the {domain} domain
- {label_frequency_formatted} of similar tickets have this label
- This indicates {strength} historical evidence for this label

Current Ticket:
Title: {title}
Description: {description}
Domain: {domain}

Top Similar Tickets with "{label_name}" label:
{similar_tickets_with_label}

Decision Criteria for "{label_name}":
{label_criteria}

Instructions:
1. Analyze the ticket content for label-specific indicators
2. Compare with historical pattern distribution
3. Evaluate confidence based on semantic similarity and historical evidence
4. Consider domain-specific conventions

Guidelines:
- If >= 70% of similar tickets have this label AND ticket shows relevant indicators → likely assign
- If < 30% of similar tickets have this label AND no strong indicators → likely reject
- Middle ground (30-70%) requires careful analysis of ticket content

Output Format (JSON only):
{{
  "assign_label": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation of your decision",
  "supporting_evidence": ["specific evidence from ticket that supports (or contradicts) this label"]
}}

Respond ONLY with valid JSON. Do not include any other text."""


# Label-specific criteria

LABEL_CRITERIA = {
    "Code Fix": """
Indicators for "Code Fix":
- Mentions of code changes, software modifications, or deployment issues
- Stack traces, error messages, or exception handling
- Performance problems requiring code optimization
- Logic errors or bugs in application code
- Configuration changes in code files
- Service restarts or application redeployments
Threshold: >= 0.70 confidence to assign
""",

    "Data Fix": """
Indicators for "Data Fix":
- Data corruption, incorrect records, or inconsistent data
- Database issues, SQL errors, or data integrity problems
- Missing or duplicate data entries
- Data migration or synchronization issues
- Record cleanup or data validation errors
- Direct database manipulation required
Threshold: >= 0.70 confidence to assign
""",

    "Configuration Fix": """
Indicators for "Configuration Fix":
- Configuration file changes or settings adjustments
- Environment variable modifications
- Timeout settings, connection pool configurations
- System parameters or feature flags
- Infrastructure configuration (nginx, apache, etc.)
- Does NOT require code changes, only config adjustments
Threshold: >= 0.70 confidence to assign
""",

    "#MM_ALDER": """
Indicators for "#MM_ALDER":
- Explicitly mentions MM_ALDER service or module
- Issues in MM domain involving ALDER component
- Database connections, memory leaks, or API issues in MM_ALDER
- Must be in MM domain
- ALDER-specific logs or error messages
Threshold: >= 0.75 confidence to assign
""",

    "#MMALDR": """
Indicators for "#MMALDR":
- Mentions MMALDR component or service
- MM domain issues with MMALDR-specific problems
- Often appears alongside #MM_ALDER for ALDER-related issues
- Must be in MM domain
Threshold: >= 0.75 confidence to assign
""",

    "#CIW_INTEGRATION": """
Indicators for "#CIW_INTEGRATION":
- Integration issues with external systems
- Data synchronization or API connection problems in CIW
- Claims processing integration failures
- Must be in CIW domain
- Mentions integration endpoints, sync jobs, or external system connectivity
Threshold: >= 0.75 confidence to assign
""",

    "#SPECIALTY_CUSTOM": """
Indicators for "#SPECIALTY_CUSTOM":
- Custom modules, workflows, or specialized features
- Specialty-specific reporting or analytics
- Custom UI components or frontend issues
- Must be in Specialty domain
- Mentions custom implementations or specialty-specific features
Threshold: >= 0.75 confidence to assign
"""
}


def get_label_assignment_prompt(
    label_name: str,
    title: str,
    description: str,
    domain: str,
    similar_tickets: list,
    label_frequency: dict
) -> str:
    """
    Generate label assignment prompt for a specific label.

    Args:
        label_name: Label to evaluate
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        similar_tickets: List of similar tickets
        label_frequency: Label frequency info from similar tickets

    Returns:
        Formatted prompt
    """
    # Get label criteria
    criteria = LABEL_CRITERIA.get(label_name, "No specific criteria defined.")

    # Calculate frequency info
    total_similar = len(similar_tickets)
    label_freq_info = label_frequency.get(label_name, {"count": 0, "total": total_similar, "percentage": 0})

    # Format strength of evidence
    percentage = label_freq_info['percentage']
    if percentage >= 70:
        strength = "STRONG"
    elif percentage >= 40:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    # Get similar tickets that have this label
    tickets_with_label = [
        t for t in similar_tickets
        if label_name in t.get('labels', [])
    ][:3]  # Top 3

    # Format similar tickets
    if tickets_with_label:
        formatted_tickets = []
        for i, ticket in enumerate(tickets_with_label, 1):
            formatted_tickets.append(f"""
Ticket {i}: {ticket['ticket_id']} (Similarity: {ticket.get('similarity_score', 0):.2f})
Title: {ticket['title']}
Description: {ticket['description'][:200]}...
Resolution: {' '.join(ticket.get('resolution_steps', [])[:2])}
""")
        similar_tickets_text = "\n".join(formatted_tickets)
    else:
        similar_tickets_text = f"No similar tickets found with the '{label_name}' label."

    # Format the prompt
    return LABEL_ASSIGNMENT_TEMPLATE.format(
        label_name=label_name,
        total_similar=total_similar,
        domain=domain,
        label_frequency_formatted=label_freq_info['formatted'],
        strength=strength,
        title=title,
        description=description,
        similar_tickets_with_label=similar_tickets_text,
        label_criteria=criteria
    )


# ============================================================================
# BUSINESS LABEL GENERATION TEMPLATE (AI-Generated)
# ============================================================================

BUSINESS_LABEL_GENERATION_TEMPLATE = """You are a business analyst specializing in IT support ticket categorization.

Task: Generate business-oriented labels for the given ticket based on analysis of similar historical tickets.

Business Label Categories to Consider:
1. **Process Impact**: Which business processes are affected?
   - Examples: "Claims Processing", "Member Enrollment", "Provider Management", "Authorization Workflow"

2. **Customer Impact**: Who is affected and how severely?
   - Examples: "Member-Facing", "Provider-Facing", "Internal Operations", "Partner Integration"

3. **Business Priority Indicators**: What business urgency signals exist?
   - Examples: "SLA Critical", "Revenue Impact", "Compliance Risk", "Audit Finding"

4. **Functional Area**: Which business function owns this?
   - Examples: "Benefits Administration", "Network Management", "Care Management", "Financial Operations"

5. **Service Category**: What type of service is impacted?
   - Examples: "Real-time Services", "Batch Processing", "Reporting", "Data Exchange"

Current Ticket:
- Title: {title}
- Description: {description}
- Domain: {domain}
- Priority: {priority}

Similar Historical Tickets ({total_similar} found):
{similar_tickets_summary}

Existing Labels in Similar Tickets:
{existing_labels}

Instructions:
1. Analyze the ticket from a BUSINESS perspective (not technical)
2. Identify business processes, stakeholders, and impact areas
3. Generate labels that would help business stakeholders understand and prioritize
4. Each label should be actionable for business decision-making
5. Avoid technical jargon - focus on business terminology

Output Format (JSON only):
{{
  "business_labels": [
    {{
      "label": "Label Name",
      "confidence": 0.0 to 1.0,
      "category": "Process Impact|Customer Impact|Business Priority|Functional Area|Service Category",
      "reasoning": "Why this label applies from a business perspective"
    }}
  ],
  "business_summary": "Brief business impact summary (1-2 sentences)"
}}

Generate up to {max_labels} most relevant business labels. Only include labels with >= 0.7 confidence.
Respond ONLY with valid JSON."""


TECHNICAL_LABEL_GENERATION_TEMPLATE = """You are a senior software engineer specializing in IT infrastructure and application support.

Task: Generate technical labels for the given ticket based on analysis of similar historical tickets.

Technical Label Categories to Consider:
1. **System Component**: Which technical component is affected?
   - Examples: "Database Layer", "API Gateway", "Message Queue", "Cache System", "Load Balancer"

2. **Failure Mode**: What type of technical failure is this?
   - Examples: "Memory Leak", "Connection Pool Exhaustion", "Deadlock", "Race Condition", "Timeout"

3. **Technology Stack**: Which technologies are involved?
   - Examples: "Java/Spring", "Python/FastAPI", "PostgreSQL", "Redis", "Kafka", "Kubernetes"

4. **Infrastructure Layer**: Where in the stack is the issue?
   - Examples: "Application Layer", "Data Layer", "Network Layer", "Infrastructure Layer", "Security Layer"

5. **Resolution Type**: What kind of fix is likely needed?
   - Examples: "Code Change", "Config Update", "Schema Migration", "Scaling Required", "Hotfix"

6. **Observability**: How was/should this be detected?
   - Examples: "Alerting Gap", "Log Analysis", "Metrics Anomaly", "Trace Required", "Health Check"

Current Ticket:
- Title: {title}
- Description: {description}
- Domain: {domain}
- Priority: {priority}

Similar Historical Tickets ({total_similar} found):
{similar_tickets_summary}

Existing Labels in Similar Tickets:
{existing_labels}

Technical Patterns Observed in Similar Tickets:
{technical_patterns}

Instructions:
1. Analyze the ticket from a TECHNICAL perspective
2. Identify system components, failure modes, and technology stack
3. Generate labels that would help engineers route, diagnose, and resolve
4. Each label should be actionable for technical decision-making
5. Be specific - prefer "PostgreSQL Connection Pool" over generic "Database Issue"

Output Format (JSON only):
{{
  "technical_labels": [
    {{
      "label": "Label Name",
      "confidence": 0.0 to 1.0,
      "category": "System Component|Failure Mode|Technology Stack|Infrastructure Layer|Resolution Type|Observability",
      "reasoning": "Technical justification for this label"
    }}
  ],
  "root_cause_hypothesis": "Brief technical hypothesis about root cause (1-2 sentences)"
}}

Generate up to {max_labels} most relevant technical labels. Only include labels with >= 0.7 confidence.
Respond ONLY with valid JSON."""


# ============================================================================
# PROMPT GENERATION FUNCTIONS FOR AI-GENERATED LABELS
# ============================================================================

def get_business_label_generation_prompt(
    title: str,
    description: str,
    domain: str,
    priority: str,
    similar_tickets: list,
    existing_labels: set,
    max_labels: int = 5
) -> str:
    """
    Generate prompt for business label generation.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        similar_tickets: List of similar tickets
        existing_labels: Set of existing labels from similar tickets
        max_labels: Maximum number of labels to generate

    Returns:
        Formatted prompt
    """
    total_similar = len(similar_tickets)

    # Format similar tickets summary (business-focused)
    if similar_tickets:
        tickets_summary = []
        for i, ticket in enumerate(similar_tickets[:5], 1):
            tickets_summary.append(
                f"{i}. {ticket.get('ticket_id', 'N/A')} - {ticket.get('title', 'N/A')[:80]}\n"
                f"   Labels: {', '.join(ticket.get('labels', []))}\n"
                f"   Priority: {ticket.get('priority', 'N/A')}"
            )
        similar_tickets_text = "\n".join(tickets_summary)
    else:
        similar_tickets_text = "No similar tickets found."

    # Format existing labels
    existing_labels_text = ", ".join(sorted(existing_labels)) if existing_labels else "None"

    return BUSINESS_LABEL_GENERATION_TEMPLATE.format(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        total_similar=total_similar,
        similar_tickets_summary=similar_tickets_text,
        existing_labels=existing_labels_text,
        max_labels=max_labels
    )


def get_technical_label_generation_prompt(
    title: str,
    description: str,
    domain: str,
    priority: str,
    similar_tickets: list,
    existing_labels: set,
    max_labels: int = 5
) -> str:
    """
    Generate prompt for technical label generation.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        similar_tickets: List of similar tickets
        existing_labels: Set of existing labels from similar tickets
        max_labels: Maximum number of labels to generate

    Returns:
        Formatted prompt
    """
    total_similar = len(similar_tickets)

    # Format similar tickets summary (technical-focused)
    if similar_tickets:
        tickets_summary = []
        for i, ticket in enumerate(similar_tickets[:5], 1):
            resolution_preview = ""
            if ticket.get('resolution_steps'):
                resolution_preview = ticket['resolution_steps'][0][:100] if ticket['resolution_steps'] else ""
            tickets_summary.append(
                f"{i}. {ticket.get('ticket_id', 'N/A')} - {ticket.get('title', 'N/A')[:80]}\n"
                f"   Labels: {', '.join(ticket.get('labels', []))}\n"
                f"   Resolution hint: {resolution_preview}..."
            )
        similar_tickets_text = "\n".join(tickets_summary)
    else:
        similar_tickets_text = "No similar tickets found."

    # Extract technical patterns from similar tickets
    technical_patterns = _extract_technical_patterns(similar_tickets)

    # Format existing labels
    existing_labels_text = ", ".join(sorted(existing_labels)) if existing_labels else "None"

    return TECHNICAL_LABEL_GENERATION_TEMPLATE.format(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        total_similar=total_similar,
        similar_tickets_summary=similar_tickets_text,
        existing_labels=existing_labels_text,
        technical_patterns=technical_patterns,
        max_labels=max_labels
    )


def _extract_technical_patterns(similar_tickets: list) -> str:
    """
    Extract common technical patterns from similar tickets.

    Args:
        similar_tickets: List of similar ticket dicts

    Returns:
        Formatted string of technical patterns
    """
    if not similar_tickets:
        return "No patterns available."

    # Collect resolution patterns
    resolution_keywords = {}
    label_patterns = {}

    for ticket in similar_tickets:
        # Analyze labels for technical patterns
        for label in ticket.get('labels', []):
            label_patterns[label] = label_patterns.get(label, 0) + 1

        # Analyze resolution steps for keywords
        for step in ticket.get('resolution_steps', []):
            step_lower = step.lower()
            tech_keywords = [
                'database', 'api', 'cache', 'memory', 'connection', 'timeout',
                'deploy', 'config', 'restart', 'scale', 'index', 'query',
                'log', 'monitor', 'alert', 'heap', 'thread', 'pool'
            ]
            for keyword in tech_keywords:
                if keyword in step_lower:
                    resolution_keywords[keyword] = resolution_keywords.get(keyword, 0) + 1

    # Format patterns
    patterns = []

    if label_patterns:
        top_labels = sorted(label_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        patterns.append(f"Common labels: {', '.join([f'{l[0]} ({l[1]}x)' for l in top_labels])}")

    if resolution_keywords:
        top_keywords = sorted(resolution_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        patterns.append(f"Resolution keywords: {', '.join([f'{k[0]} ({k[1]}x)' for k in top_keywords])}")

    return "\n".join(patterns) if patterns else "No clear patterns identified."
