"""
Labeling Agent - LangGraph node for label assignment.

This agent assigns labels to tickets using three methods:
1. Category labels from predefined taxonomy (categories.json)
2. AI-generated business labels
3. AI-generated technical labels
"""

import asyncio
from typing import Dict, Any, List

from components.labeling.tools import (
    classify_ticket_categories,
    generate_business_labels,
    generate_technical_labels
)
from config.config import Config


async def labeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for label assignment.

    Assigns labels using three-tier approach:
    1. Category labels from predefined taxonomy (categories.json)
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
        ticket_id = state.get("ticket_id", "N/A")

        print(f"\n🏷️  Labeling Agent - Assigning labels: {ticket_id}")

        # Run all three label methods in parallel
        # 1. Category classification (replaces historical labels)
        category_task = classify_ticket_categories.ainvoke({
            "title": title,
            "description": description,
            "priority": priority
        })

        # 2. Business labels (uses Config for max count)
        business_task = generate_business_labels.ainvoke({
            "title": title,
            "description": description,
            "domain": domain,
            "priority": priority,
            "existing_labels": [],  # No historical labels to avoid
            "max_labels": Config.BUSINESS_LABEL_MAX_COUNT,
            "confidence_threshold": Config.GENERATED_LABEL_CONFIDENCE_THRESHOLD
        })

        # 3. Technical labels (uses Config for max count)
        technical_task = generate_technical_labels.ainvoke({
            "title": title,
            "description": description,
            "domain": domain,
            "priority": priority,
            "existing_labels": [],  # No historical labels to avoid
            "max_labels": Config.TECHNICAL_LABEL_MAX_COUNT,
            "confidence_threshold": Config.GENERATED_LABEL_CONFIDENCE_THRESHOLD
        })

        # Wait for all tasks
        category_result, business_result, technical_result = await asyncio.gather(
            category_task, business_task, technical_task
        )

        # Extract category labels (replaces historical labels)
        category_labels = category_result.get("assigned_categories", [])
        novelty_detected = category_result.get("novelty_detected", False)
        novelty_reasoning = category_result.get("novelty_reasoning")

        # Extract embedding data for novelty detection agent
        ticket_embedding = category_result.get("ticket_embedding", [])
        all_similarity_scores = category_result.get("all_similarity_scores", [])

        # Extract business and technical labels
        business_labels = business_result.get("labels", [])
        technical_labels = technical_result.get("labels", [])

        # Collect actual prompts for transparency
        actual_prompts = {
            "category": category_result.get("actual_prompt", ""),
            "business": business_result.get("actual_prompt", ""),
            "technical": technical_result.get("actual_prompt", "")
        }

        # Combine all labels for backward compatibility
        all_labels = set()
        for cat in category_labels:
            all_labels.add(f"[CAT] {cat.get('name', '')}")
        for label_info in business_labels:
            all_labels.add(f"[BIZ] {label_info.get('label', '')}")
        for label_info in technical_labels:
            all_labels.add(f"[TECH] {label_info.get('label', '')}")

        print(f"   ✅ Assigned {len(category_labels)} category labels")
        print(f"   ✅ Generated {len(business_labels)} business labels")
        print(f"   ✅ Generated {len(technical_labels)} technical labels")
        if novelty_detected:
            print(f"   ⚠️  Novelty detected: {novelty_reasoning}")

        return {
            # Category labels (replaces historical_labels)
            "category_labels": category_labels,

            # AI-generated labels (unchanged)
            "business_labels": business_labels,
            "technical_labels": technical_labels,

            # Combined (backward compatibility)
            "assigned_labels": list(all_labels),

            # Embedding data for novelty detection agent
            "ticket_embedding": ticket_embedding,
            "all_category_scores": all_similarity_scores,

            # Actual prompts sent to LLM
            "label_assignment_prompts": actual_prompts,

            "status": "success",
            "current_agent": "labeling",
            "messages": [{
                "role": "assistant",
                "content": f"Assigned {len(all_labels)} total labels: {len(category_labels)} category, {len(business_labels)} business, {len(technical_labels)} technical"
            }]
        }

    except Exception as e:
        print(f"   ❌ Labeling error: {str(e)}")
        return {
            "category_labels": [],
            "business_labels": [],
            "technical_labels": [],
            "assigned_labels": [],
            "ticket_embedding": [],
            "all_category_scores": [],
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
