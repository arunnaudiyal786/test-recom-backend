"""
Labeling Tools - LangChain @tool decorated functions for label assignment.

These tools handle historical label extraction, evaluation, and AI-generated labels.
"""

import asyncio
import json
from typing import Dict, Any, List, Set

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.utils.config import Config


@tool
def extract_candidate_labels(similar_tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract unique labels and their frequency from similar historical tickets.

    Args:
        similar_tickets: List of similar ticket dicts with 'labels' field

    Returns:
        Dict containing:
        - candidate_labels: Set of unique labels
        - label_distribution: Dict mapping label to {count, percentage, formatted}
    """
    labels = set()
    label_counts = {}
    total = len(similar_tickets)

    for ticket in similar_tickets:
        ticket_labels = ticket.get("labels", [])
        if isinstance(ticket_labels, list):
            labels.update(ticket_labels)
            for label in ticket_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

    distribution = {
        label: {
            "count": count,
            "percentage": count / total if total > 0 else 0,
            "formatted": f"{count}/{total}"
        }
        for label, count in label_counts.items()
    }

    return {
        "candidate_labels": list(labels),
        "label_distribution": distribution,
        "total_tickets": total
    }


async def _evaluate_single_label(
    label_name: str,
    title: str,
    description: str,
    domain: str,
    frequency: str,
    llm: ChatOpenAI
) -> Dict[str, Any]:
    """
    Internal function to evaluate a single label using binary classifier.

    Args:
        label_name: Label to evaluate
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        frequency: Label frequency string (e.g., "5/20")
        llm: ChatOpenAI instance

    Returns:
        Evaluation result dict
    """
    system_prompt = "You are a label classification expert. Respond only with valid JSON."
    user_prompt = f"""You are a label validation expert for technical support tickets.

Evaluate whether the label "{label_name}" should be assigned to this ticket.

Historical frequency: This label appears in {frequency} similar historical tickets.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}

Consider:
1. Does the ticket content match the label semantics?
2. Is the historical frequency a strong indicator?
3. What is your confidence level?

Output JSON:
{{
  "assign_label": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        result = json.loads(response.content)
        return {
            "label": label_name,
            "assign": result.get("assign_label", False),
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "actual_prompt": user_prompt
        }

    except Exception as e:
        return {
            "label": label_name,
            "assign": False,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "actual_prompt": user_prompt
        }


@tool
async def evaluate_historical_labels(
    title: str,
    description: str,
    domain: str,
    candidate_labels: List[str],
    label_distribution: Dict[str, Dict],
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Evaluate historical labels using parallel binary classifiers.

    Runs a binary classifier for each candidate label to determine
    if it should be assigned to the current ticket.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        candidate_labels: List of candidate labels to evaluate
        label_distribution: Dict mapping label to frequency info
        confidence_threshold: Minimum confidence to assign label (default 0.7)

    Returns:
        Dict containing:
        - assigned_labels: List of labels that passed threshold
        - label_confidence: Dict mapping label to confidence score
        - all_evaluations: Full evaluation results for all labels
    """
    if not candidate_labels:
        return {
            "assigned_labels": [],
            "label_confidence": {},
            "all_evaluations": [],
            "sample_prompt": "[No historical labels to evaluate - no similar tickets found with labels]"
        }

    llm = ChatOpenAI(
        model=Config.CLASSIFICATION_MODEL,
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Evaluate all labels in parallel
    tasks = [
        _evaluate_single_label(
            label,
            title,
            description,
            domain,
            label_distribution.get(label, {}).get("formatted", "0/0"),
            llm
        )
        for label in candidate_labels
    ]
    results = await asyncio.gather(*tasks)

    # Filter by threshold
    assigned_labels = []
    label_confidence = {}
    sample_prompt = None

    for result in results:
        label_confidence[result["label"]] = result["confidence"]
        if result["assign"] and result["confidence"] >= confidence_threshold:
            assigned_labels.append(result["label"])
        # Capture the first prompt as a sample
        if sample_prompt is None and result.get("actual_prompt"):
            sample_prompt = result["actual_prompt"]

    return {
        "assigned_labels": assigned_labels,
        "label_confidence": label_confidence,
        "all_evaluations": results,
        "sample_prompt": sample_prompt
    }


@tool
async def generate_business_labels(
    title: str,
    description: str,
    domain: str,
    priority: str,
    existing_labels: List[str],
    max_labels: int = 5,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Generate business-oriented labels using AI analysis.

    Analyzes the ticket from a business impact perspective and generates
    relevant business labels.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate (default 5)
        confidence_threshold: Minimum confidence to include (default 0.7)

    Returns:
        Dict with labels list and actual_prompt
    """
    llm = ChatOpenAI(
        model=Config.CLASSIFICATION_MODEL,
        temperature=0.4,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    user_prompt = f"""You are a business analyst expert in IT service management.

Generate business-oriented labels for this ticket from a business impact perspective.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {existing_labels}

Business label categories to consider:
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

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are a business analyst expert. Respond only with valid JSON."},
            {"role": "user", "content": user_prompt}
        ])

        result = json.loads(response.content)
        labels = []

        for item in result.get("business_labels", []):
            if item.get("confidence", 0) >= confidence_threshold:
                labels.append({
                    "label": item.get("label", ""),
                    "confidence": item.get("confidence", 0.0),
                    "category": "business",
                    "reasoning": item.get("reasoning", "")
                })

        return {"labels": labels, "actual_prompt": user_prompt}

    except Exception as e:
        return {"labels": [], "actual_prompt": user_prompt}


@tool
async def generate_technical_labels(
    title: str,
    description: str,
    domain: str,
    priority: str,
    existing_labels: List[str],
    max_labels: int = 5,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Generate technical labels using AI analysis.

    Analyzes the ticket from a technical/root-cause perspective and generates
    relevant technical labels.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate (default 5)
        confidence_threshold: Minimum confidence to include (default 0.7)

    Returns:
        Dict with labels list and actual_prompt
    """
    llm = ChatOpenAI(
        model=Config.CLASSIFICATION_MODEL,
        temperature=0.3,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    user_prompt = f"""You are a senior software engineer expert in system diagnostics.

Generate technical labels for this ticket from a technical/root-cause perspective.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {existing_labels}

Technical label categories to consider:
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

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are a software engineer expert. Respond only with valid JSON."},
            {"role": "user", "content": user_prompt}
        ])

        result = json.loads(response.content)
        labels = []

        for item in result.get("technical_labels", []):
            if item.get("confidence", 0) >= confidence_threshold:
                labels.append({
                    "label": item.get("label", ""),
                    "confidence": item.get("confidence", 0.0),
                    "category": "technical",
                    "reasoning": item.get("reasoning", "")
                })

        return {"labels": labels, "actual_prompt": user_prompt}

    except Exception as e:
        return {"labels": [], "actual_prompt": user_prompt}
