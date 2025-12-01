"""
Resolution Agent - LangGraph node for resolution generation.

This agent generates comprehensive resolution plans using Chain-of-Thought
reasoning based on ticket context and similar historical resolutions.
"""

from typing import Dict, Any

from components.resolution.tools import (
    analyze_similar_resolutions,
    generate_resolution_plan
)


async def resolution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for resolution generation.

    Analyzes similar ticket resolutions and generates a comprehensive
    resolution plan for the current ticket.

    Args:
        state: Current workflow state with ticket info, labels, and similar tickets

    Returns:
        Partial state update with resolution plan
    """
    try:
        title = state.get("title", "")
        description = state.get("description", "")
        # Handle None domain (when classification is skipped)
        domain = state.get("classified_domain") or "Unknown"
        priority = state.get("priority") or "Medium"
        similar_tickets = state.get("similar_tickets", [])
        ticket_id = state.get("ticket_id", "N/A")

        # Get all labels (combined)
        all_labels = state.get("assigned_labels", [])
        if not all_labels:
            # Fallback to individual label types
            historical = state.get("historical_labels", [])
            business = [f"[BIZ] {l.get('label', '')}" for l in state.get("business_labels", [])]
            technical = [f"[TECH] {l.get('label', '')}" for l in state.get("technical_labels", [])]
            all_labels = historical + business + technical

        # Get average similarity from search metadata
        search_metadata = state.get("search_metadata", {})
        avg_similarity = search_metadata.get("avg_similarity", 0.0)

        print(f"\n📝 Resolution Agent - Generating plan: {ticket_id}")

        # Analyze similar resolutions
        historical_context = analyze_similar_resolutions.invoke({
            "similar_tickets": similar_tickets
        })

        print(f"   📊 Analyzed {min(5, len(similar_tickets))} similar resolutions")

        # Generate resolution plan
        result = await generate_resolution_plan.ainvoke({
            "title": title,
            "description": description,
            "domain": domain,
            "priority": priority,
            "labels": all_labels,
            "historical_context": historical_context,
            "avg_similarity": avg_similarity
        })

        resolution_plan = result.get("resolution_plan", {})
        confidence = result.get("confidence", 0.0)

        print(f"   ✅ Generated resolution plan with confidence {confidence:.2%}")
        print(f"   📋 {len(resolution_plan.get('diagnostic_steps', []))} diagnostic steps")
        print(f"   📋 {len(resolution_plan.get('resolution_steps', []))} resolution steps")

        return {
            "resolution_plan": resolution_plan,
            "resolution_confidence": confidence,
            "status": "success",
            "current_agent": "resolution",
            "messages": [{
                "role": "assistant",
                "content": f"Generated resolution plan with {confidence:.2%} confidence"
            }]
        }

    except Exception as e:
        print(f"   ❌ Resolution error: {str(e)}")

        # Return fallback plan
        fallback_plan = {
            "summary": "Automatic processing failed. Manual review required.",
            "diagnostic_steps": [],
            "resolution_steps": [{
                "step_number": 1,
                "description": "Escalate to human agent for manual processing",
                "commands": [],
                "validation": "N/A",
                "estimated_time_minutes": 0,
                "risk_level": "low",
                "rollback_procedure": None
            }],
            "additional_considerations": [f"Resolution generation failed: {str(e)}"],
            "references": [],
            "total_estimated_time_hours": 0,
            "confidence": 0.0,
            "alternative_approaches": []
        }

        return {
            "resolution_plan": fallback_plan,
            "resolution_confidence": 0.0,
            "status": "error",
            "current_agent": "resolution",
            "error_message": f"Resolution generation failed: {str(e)}",
            "messages": [{
                "role": "assistant",
                "content": f"Resolution generation failed: {str(e)}"
            }]
        }


# For backward compatibility - callable class wrapper
class ResolutionGenerationAgent:
    """Callable wrapper for resolution_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await resolution_node(state)


# Singleton instance
resolution_agent = ResolutionGenerationAgent()
