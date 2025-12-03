"""
Novelty Detection Agent - LangGraph node for novelty detection.

This agent runs after the labeling agent to analyze whether the ticket
represents a novel category that doesn't exist in the current taxonomy.

It uses data from the labeling step:
- ticket_embedding: The ticket's semantic embedding
- all_category_scores: Similarity scores for all categories
- category_labels: Assigned categories with confidence scores
"""

from typing import Dict, Any, List, Tuple

from components.novelty.tools import detect_novelty
from config import Config


async def novelty_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for novelty detection.

    Analyzes whether a ticket represents a novel category using three signals:
    1. Maximum confidence score (is the best match weak?)
    2. Confidence distribution entropy (is uncertainty spread evenly?)
    3. Embedding distance to centroids (is ticket semantically far from all categories?)

    Args:
        state: Current workflow state with labeling results

    Returns:
        Partial state update with novelty detection results
    """
    try:
        ticket_id = state.get("ticket_id", "N/A")
        print(f"\n🔍 Novelty Detection Agent - Analyzing ticket: {ticket_id}")

        # ================================================================
        # Extract required data from state
        # ================================================================

        # Get ticket embedding (from labeling step)
        ticket_embedding = state.get("ticket_embedding", [])
        if not ticket_embedding:
            print("   ⚠️  No ticket embedding available - skipping novelty detection")
            return {
                "novelty_detected": False,
                "novelty_score": 0.0,
                "novelty_signals": {},
                "novelty_recommendation": "proceed",
                "status": "success",
                "current_agent": "novelty",
                "messages": [{
                    "role": "assistant",
                    "content": "Novelty detection skipped - no ticket embedding available"
                }]
            }

        # Get all category scores (from labeling step)
        all_category_scores = state.get("all_category_scores", [])

        # Convert to list of tuples if needed
        if all_category_scores and isinstance(all_category_scores[0], dict):
            # Format: [{"category_id": "x", "score": 0.5}, ...]
            category_confidence_scores = [
                (item.get("category_id", ""), item.get("score", 0.0))
                for item in all_category_scores
            ]
        elif all_category_scores and isinstance(all_category_scores[0], (list, tuple)):
            # Format: [("category_id", 0.5), ...]
            category_confidence_scores = [(str(item[0]), float(item[1])) for item in all_category_scores]
        else:
            category_confidence_scores = []

        # Get max confidence from assigned categories or from labeling result
        category_labels = state.get("category_labels", [])
        if category_labels:
            max_confidence = max(
                cat.get("confidence", 0.0) for cat in category_labels
            )
        elif category_confidence_scores:
            max_confidence = max(score for _, score in category_confidence_scores)
        else:
            max_confidence = 0.0

        # If we still don't have category scores, use category_labels
        if not category_confidence_scores and category_labels:
            category_confidence_scores = [
                (cat.get("id", ""), cat.get("confidence", 0.0))
                for cat in category_labels
            ]

        # ================================================================
        # Get configuration thresholds
        # ================================================================
        signal1_threshold = getattr(Config, 'NOVELTY_SIGNAL1_THRESHOLD', 0.5)
        signal2_threshold = getattr(Config, 'NOVELTY_SIGNAL2_THRESHOLD', 0.7)
        signal3_threshold = getattr(Config, 'NOVELTY_SIGNAL3_THRESHOLD', 0.4)
        signal1_weight = getattr(Config, 'NOVELTY_SIGNAL1_WEIGHT', 0.4)
        signal2_weight = getattr(Config, 'NOVELTY_SIGNAL2_WEIGHT', 0.3)
        signal3_weight = getattr(Config, 'NOVELTY_SIGNAL3_WEIGHT', 0.3)
        novelty_score_threshold = getattr(Config, 'NOVELTY_SCORE_THRESHOLD', 0.6)

        # ================================================================
        # Run novelty detection
        # ================================================================
        result = await detect_novelty.ainvoke({
            "ticket_embedding": ticket_embedding,
            "category_confidence_scores": category_confidence_scores,
            "max_confidence": max_confidence,
            "signal1_threshold": signal1_threshold,
            "signal2_threshold": signal2_threshold,
            "signal3_threshold": signal3_threshold,
            "signal1_weight": signal1_weight,
            "signal2_weight": signal2_weight,
            "signal3_weight": signal3_weight,
            "novelty_score_threshold": novelty_score_threshold
        })

        # ================================================================
        # Log results
        # ================================================================
        is_novel = result.get("is_novel", False)
        novelty_score = result.get("novelty_score", 0.0)
        signals_fired = result.get("signals_fired", 0)
        recommendation = result.get("recommendation", "proceed")

        if is_novel:
            print(f"   ⚠️  NOVELTY DETECTED (score: {novelty_score:.3f}, signals: {signals_fired}/3)")
            print(f"   📋 Recommendation: {recommendation}")
            print(f"   💡 {result.get('reasoning', '')}")
        else:
            print(f"   ✅ No novelty detected (score: {novelty_score:.3f})")
            print(f"   📍 Nearest category: {result.get('nearest_category', 'N/A')}")

        # ================================================================
        # Return state update
        # ================================================================
        return {
            "novelty_detected": is_novel,
            "novelty_score": novelty_score,
            "novelty_signals": result.get("signals", {}),
            "novelty_recommendation": recommendation,
            "novelty_reasoning": result.get("reasoning", ""),
            "novelty_details": {
                "signals_fired": signals_fired,
                "decision_factors": result.get("decision_factors", {}),
                "recommendation_reason": result.get("recommendation_reason", ""),
                "nearest_category": result.get("nearest_category", "")
            },
            "status": "success",
            "current_agent": "novelty",
            "messages": [{
                "role": "assistant",
                "content": f"Novelty detection complete. Score: {novelty_score:.3f}, Novel: {is_novel}, Recommendation: {recommendation}"
            }]
        }

    except Exception as e:
        print(f"   ❌ Novelty detection error: {str(e)}")
        return {
            "novelty_detected": False,
            "novelty_score": 0.0,
            "novelty_signals": {},
            "novelty_recommendation": "proceed",
            "novelty_reasoning": f"Error during detection: {str(e)}",
            "status": "success",  # Don't fail the pipeline for novelty errors
            "current_agent": "novelty",
            "error_message": f"Novelty detection failed: {str(e)}",
            "messages": [{
                "role": "assistant",
                "content": f"Novelty detection failed (non-blocking): {str(e)}"
            }]
        }


# For backward compatibility - callable class wrapper
class NoveltyDetectionAgent:
    """Callable wrapper for novelty_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await novelty_node(state)


# Singleton instance
novelty_detection_agent = NoveltyDetectionAgent()
