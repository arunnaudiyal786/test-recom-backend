"""
Labeling Agent - LangGraph node for label assignment.

This agent assigns labels to tickets using three methods:
1. Historical labels from similar tickets (validated by AI)
2. AI-generated business labels
3. AI-generated technical labels
"""

import asyncio
from typing import Dict, Any, List

from components.labeling.tools import (
    extract_candidate_labels,
    evaluate_historical_labels,
    generate_business_labels,
    generate_technical_labels
)


async def labeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for label assignment.

    Assigns labels using three-tier approach:
    1. Historical labels from similar tickets
    2. Business labels (AI-generated)
    3. Technical labels (AI-generated)

    Args:
        state: Current workflow state with ticket info and similar tickets

    Returns:
        Partial state update with assigned labels
    """
    try:
        title = state.get("title", "")
        description = state.get("description", "")
        # Handle None domain (when classification is skipped)
        domain = state.get("classified_domain") or "Unknown"
        priority = state.get("priority") or "Medium"
        similar_tickets = state.get("similar_tickets", [])
        ticket_id = state.get("ticket_id", "N/A")

        print(f"\n🏷️  Labeling Agent - Assigning labels: {ticket_id}")

        # Step 1: Extract candidate labels from similar tickets
        candidates = extract_candidate_labels.invoke({
            "similar_tickets": similar_tickets
        })

        candidate_labels = candidates.get("candidate_labels", [])
        label_distribution = candidates.get("label_distribution", {})

        print(f"   📋 Found {len(candidate_labels)} candidate labels from history")

        # Step 2: Run all three label methods in parallel
        historical_task = evaluate_historical_labels.ainvoke({
            "title": title,
            "description": description,
            "domain": domain,
            "candidate_labels": candidate_labels,
            "label_distribution": label_distribution,
            "confidence_threshold": 0.7
        })

        # Get existing labels to avoid duplicates
        existing_labels = list(candidate_labels)

        business_task = generate_business_labels.ainvoke({
            "title": title,
            "description": description,
            "domain": domain,
            "priority": priority,
            "existing_labels": existing_labels,
            "max_labels": 5,
            "confidence_threshold": 0.7
        })

        technical_task = generate_technical_labels.ainvoke({
            "title": title,
            "description": description,
            "domain": domain,
            "priority": priority,
            "existing_labels": existing_labels,
            "max_labels": 5,
            "confidence_threshold": 0.7
        })

        # Wait for all tasks
        historical_result, business_labels, technical_labels = await asyncio.gather(
            historical_task, business_task, technical_task
        )

        # Extract historical labels
        historical_labels = historical_result.get("assigned_labels", [])
        historical_confidence = historical_result.get("label_confidence", {})

        # Format label distribution for output
        distribution_formatted = {
            label: info.get("formatted", "0/0")
            for label, info in label_distribution.items()
        }

        # Combine all labels
        all_labels = set(historical_labels)
        for label_info in business_labels:
            all_labels.add(f"[BIZ] {label_info.get('label', '')}")
        for label_info in technical_labels:
            all_labels.add(f"[TECH] {label_info.get('label', '')}")

        print(f"   ✅ Assigned {len(historical_labels)} historical labels")
        print(f"   ✅ Generated {len(business_labels)} business labels")
        print(f"   ✅ Generated {len(technical_labels)} technical labels")

        return {
            # Historical labels
            "historical_labels": historical_labels,
            "historical_label_confidence": historical_confidence,
            "historical_label_distribution": distribution_formatted,

            # AI-generated labels
            "business_labels": business_labels,
            "technical_labels": technical_labels,

            # Combined (backward compatibility)
            "assigned_labels": list(all_labels),
            "label_confidence": historical_confidence,
            "label_distribution": distribution_formatted,

            "status": "success",
            "current_agent": "labeling",
            "messages": [{
                "role": "assistant",
                "content": f"Assigned {len(all_labels)} total labels: {len(historical_labels)} historical, {len(business_labels)} business, {len(technical_labels)} technical"
            }]
        }

    except Exception as e:
        print(f"   ❌ Labeling error: {str(e)}")
        return {
            "historical_labels": [],
            "historical_label_confidence": {},
            "historical_label_distribution": {},
            "business_labels": [],
            "technical_labels": [],
            "assigned_labels": [],
            "label_confidence": {},
            "label_distribution": {},
            "status": "error",
            "current_agent": "labeling",
            "error_message": f"Labeling failed: {str(e)}",
            "messages": [{
                "role": "assistant",
                "content": f"Labeling failed: {str(e)}"
            }]
        }


# For backward compatibility - callable class wrapper
class LabelAssignmentAgent:
    """Callable wrapper for labeling_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await labeling_node(state)


# Singleton instance
label_assignment_agent = LabelAssignmentAgent()
