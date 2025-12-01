"""
Classification Agent - LangGraph node for domain classification.

This agent classifies tickets into domains (MM, CIW, Specialty)
and is designed to be used as a node in a LangGraph workflow.
"""

from typing import Dict, Any

from components.classification.tools import classify_ticket_domain


async def classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for domain classification.

    Extracts title and description from state, runs classification,
    and returns partial state update with classification results.

    Args:
        state: Current workflow state containing ticket info

    Returns:
        Partial state update with classification results
    """
    try:
        title = state.get("title", "")
        description = state.get("description", "")
        ticket_id = state.get("ticket_id", "N/A")

        print(f"\n🔍 Classification Agent - Analyzing ticket: {ticket_id}")

        # Run classification tool
        result = await classify_ticket_domain.ainvoke({
            "title": title,
            "description": description
        })

        domain = result.get("classified_domain", "Unknown")
        confidence = result.get("confidence", 0.0)

        print(f"   ✅ Classified as: {domain} (confidence: {confidence:.2%})")

        # Return partial state update
        return {
            "classified_domain": domain,
            "classification_confidence": confidence,
            "classification_reasoning": result.get("reasoning", ""),
            "classification_scores": result.get("domain_scores", {}),
            "extracted_keywords": result.get("extracted_keywords", []),
            "status": "success",
            "current_agent": "classification",
            "messages": [{
                "role": "assistant",
                "content": f"Classified ticket as {domain} domain with {confidence:.2%} confidence"
            }]
        }

    except Exception as e:
        print(f"   ❌ Classification error: {str(e)}")
        return {
            "status": "error",
            "current_agent": "classification",
            "error_message": f"Classification failed: {str(e)}",
            "messages": [{
                "role": "assistant",
                "content": f"Classification failed: {str(e)}"
            }]
        }


# For backward compatibility - callable class wrapper
class ClassificationAgent:
    """Callable wrapper for classification_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await classification_node(state)


# Singleton instance for direct use
classification_agent = ClassificationAgent()
