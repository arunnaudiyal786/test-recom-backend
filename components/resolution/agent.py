"""
Resolution Agent - LangGraph node for resolution generation.

This agent extracts resolution steps from similar historical tickets and
generates a comprehensive resolution plan with summary and considerations.

Uses LangGraph's create_react_agent internally via the generate_resolution_plan tool.
Prompts are sourced from components/resolution/prompts.py.
"""

from typing import Dict, Any

from components.resolution.tools import generate_resolution_plan


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

        print(f"\n📝 Resolution Agent - Generating Test Plan: {ticket_id}")
        print(f"   📊 Synthesizing test plan from {min(5, len(similar_tickets))} similar tickets")

        # Generate resolution plan (now extracts steps directly from similar tickets)
        result = await generate_resolution_plan.ainvoke({
            "title": title,
            "description": description,
            "domain": domain,
            "priority": priority,
            "labels": all_labels,
            "similar_tickets": similar_tickets,
            "avg_similarity": avg_similarity
        })

        resolution_plan = result.get("resolution_plan", {})
        confidence = result.get("confidence", 0.0)
        actual_prompt = result.get("actual_prompt", "")

        # Check for novelty detection results and add warnings if needed
        novelty_detected = state.get("novelty_detected", False)
        novelty_score = state.get("novelty_score", 0.0)
        novelty_recommendation = state.get("novelty_recommendation", "proceed")

        if novelty_detected:
            # Add novelty warning to resolution plan
            if "warnings" not in resolution_plan:
                resolution_plan["warnings"] = []

            resolution_plan["warnings"].append({
                "type": "novelty_detected",
                "severity": "high" if novelty_score > 0.8 else "medium",
                "score": novelty_score,
                "recommendation": novelty_recommendation,
                "message": f"This ticket may represent a novel category (score: {novelty_score:.2f}). Consider reviewing the category taxonomy."
            })

            # Also add to additional_considerations
            if "additional_considerations" not in resolution_plan:
                resolution_plan["additional_considerations"] = []

            resolution_plan["additional_considerations"].append(
                f"NOVELTY WARNING: Ticket may not fit existing categories (novelty score: {novelty_score:.2f}). "
                f"Recommendation: {novelty_recommendation}. Consider adding a new category to the taxonomy if this pattern recurs."
            )

            print(f"   ⚠️  Novelty warning added (score: {novelty_score:.2f})")

        print(f"   ✅ Generated test plan with confidence {confidence:.2%}")
        print(f"   📋 {len(resolution_plan.get('resolution_steps', []))} test steps (synthesized from similar tickets)")

        return {
            "resolution_plan": resolution_plan,
            "resolution_confidence": confidence,
            "resolution_generation_prompt": actual_prompt,
            "status": "success",
            "current_agent": "resolution",
            "messages": [{
                "role": "assistant",
                "content": f"Generated test plan with {confidence:.2%} confidence" +
                           (f" (NOVELTY DETECTED: {novelty_score:.2f})" if novelty_detected else "")
            }]
        }

    except Exception as e:
        print(f"   ❌ Resolution error: {str(e)}")

        # Return fallback test plan
        fallback_plan = {
            "summary": "Automatic test plan generation failed. Manual review required.",
            "diagnostic_steps": [],
            "resolution_steps": [{
                "step_number": 1,
                "description": "Escalate to test engineer for manual test plan creation",
                "commands": [],
                "validation": "N/A",
                "estimated_time_minutes": 0,
                "risk_level": "low",
                "rollback_procedure": None
            }],
            "additional_considerations": [f"Test plan generation failed: {str(e)}"],
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
            "error_message": f"Test plan generation failed: {str(e)}",
            "messages": [{
                "role": "assistant",
                "content": f"Test plan generation failed: {str(e)}"
            }]
        }


# For backward compatibility - callable class wrapper
class ResolutionGenerationAgent:
    """Callable wrapper for resolution_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await resolution_node(state)


# Singleton instance
resolution_agent = ResolutionGenerationAgent()
