"""
Novelty Detection Tools - Multi-signal detection for novel tickets.

Implements three complementary signals to detect when a ticket doesn't
match any known category in the taxonomy:

1. Signal 1 (Weight 0.4): Maximum Confidence Score
   - If the best category match has low confidence, the ticket likely doesn't fit

2. Signal 2 (Weight 0.3): Confidence Distribution Entropy
   - High entropy means uncertainty is spread across categories (fits nowhere well)

3. Signal 3 (Weight 0.3): Embedding Distance to Centroids
   - Large distance from all category centroids indicates semantic novelty

Final Decision:
    novelty_score = (0.4 * signal_1) + (0.3 * signal_2) + (0.3 * signal_3)
    is_novel = (max_confidence < 0.5) OR (novelty_score > 0.6)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from langchain_core.tools import tool

from components.labeling.category_embeddings import CategoryEmbeddings
from config import Config


def calculate_normalized_entropy(confidence_scores: List[float]) -> float:
    """
    Calculate normalized Shannon entropy of confidence distribution.

    Entropy measures how spread out the confidence is across categories.
    High entropy = uncertainty spread evenly = ticket doesn't fit any category well.
    Low entropy = confidence concentrated = ticket fits specific categories.

    Args:
        confidence_scores: List of confidence values (0-1) for each category

    Returns:
        Normalized entropy in range [0, 1]
        - 0: All confidence in one category (certain)
        - 1: Equal confidence across all categories (maximum uncertainty)
    """
    if not confidence_scores or len(confidence_scores) < 2:
        return 0.0

    # Convert to numpy array and ensure non-negative
    scores = np.array(confidence_scores, dtype=np.float64)
    scores = np.maximum(scores, 0.0)

    # Normalize to probability distribution
    total = scores.sum()
    if total <= 0:
        return 0.0

    probs = scores / total

    # Filter out zero probabilities (log(0) is undefined)
    probs = probs[probs > 0]

    if len(probs) < 2:
        return 0.0

    # Calculate Shannon entropy: H = -sum(p * log2(p))
    raw_entropy = -np.sum(probs * np.log2(probs))

    # Normalize by maximum possible entropy (uniform distribution)
    # Max entropy = log2(n) where n is number of categories
    max_entropy = np.log2(len(confidence_scores))

    if max_entropy <= 0:
        return 0.0

    normalized_entropy = raw_entropy / max_entropy

    return float(np.clip(normalized_entropy, 0.0, 1.0))


def calculate_min_centroid_distance(
    ticket_embedding: List[float],
    category_embeddings: CategoryEmbeddings
) -> Tuple[float, str]:
    """
    Calculate minimum distance from ticket to any category centroid.

    Uses cosine distance: distance = 1 - cosine_similarity
    Range: [0, 2] but typically [0, 1] for normalized embeddings.

    Args:
        ticket_embedding: Ticket's embedding vector
        category_embeddings: CategoryEmbeddings singleton with pre-computed centroids

    Returns:
        Tuple of (min_distance, nearest_category_id)
    """
    if not category_embeddings.is_loaded():
        return 1.0, "unknown"

    if not ticket_embedding:
        return 1.0, "unknown"

    # Get similarities (sorted descending by similarity)
    similarities = category_embeddings.compute_similarities(ticket_embedding)

    if not similarities:
        return 1.0, "unknown"

    # Most similar category is first
    nearest_category_id, max_similarity = similarities[0]

    # Convert similarity to distance
    # Cosine similarity range: [-1, 1], so distance range: [0, 2]
    min_distance = 1.0 - max_similarity

    return float(min_distance), nearest_category_id


def compute_signal_1(max_confidence: float, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Signal 1: Maximum Confidence Score Analysis.

    If the highest category confidence is below threshold, the ticket
    likely doesn't belong to any known category.

    Args:
        max_confidence: Highest confidence score from category classification
        threshold: Confidence below this triggers the signal (default: 0.5)

    Returns:
        Dict with signal details
    """
    fires = max_confidence < threshold
    value = 1.0 if fires else 0.0

    return {
        "name": "max_confidence",
        "fires": fires,
        "value": value,
        "threshold": threshold,
        "actual": max_confidence,
        "reasoning": f"Max confidence {max_confidence:.3f} {'<' if fires else '>='} threshold {threshold}"
    }


def compute_signal_2(
    confidence_scores: List[float],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Signal 2: Confidence Distribution Entropy.

    High entropy indicates uncertainty is spread across all categories,
    suggesting the ticket doesn't fit any specific category well.

    Args:
        confidence_scores: All category confidence scores
        threshold: Entropy above this triggers the signal (default: 0.7)

    Returns:
        Dict with signal details
    """
    normalized_entropy = calculate_normalized_entropy(confidence_scores)
    fires = normalized_entropy > threshold
    value = 1.0 if fires else 0.0

    return {
        "name": "entropy",
        "fires": fires,
        "value": value,
        "threshold": threshold,
        "actual": normalized_entropy,
        "num_categories": len(confidence_scores),
        "reasoning": f"Normalized entropy {normalized_entropy:.3f} {'>' if fires else '<='} threshold {threshold}"
    }


def compute_signal_3(
    ticket_embedding: List[float],
    category_embeddings: CategoryEmbeddings,
    threshold: float = 0.4
) -> Dict[str, Any]:
    """
    Signal 3: Embedding Distance to Nearest Category Centroid.

    Large distance from all category centroids indicates the ticket
    is semantically different from all known categories.

    Args:
        ticket_embedding: Ticket's embedding vector
        category_embeddings: Pre-computed category embeddings
        threshold: Distance above this triggers the signal (default: 0.4)

    Returns:
        Dict with signal details
    """
    min_distance, nearest_category = calculate_min_centroid_distance(
        ticket_embedding, category_embeddings
    )

    fires = min_distance > threshold
    value = 1.0 if fires else 0.0

    return {
        "name": "centroid_distance",
        "fires": fires,
        "value": value,
        "threshold": threshold,
        "actual": min_distance,
        "nearest_category": nearest_category,
        "reasoning": f"Min distance {min_distance:.3f} {'>' if fires else '<='} threshold {threshold}"
    }


@tool
async def detect_novelty(
    ticket_embedding: List[float],
    category_confidence_scores: List[Tuple[str, float]],
    max_confidence: float,
    signal1_threshold: float = 0.5,
    signal2_threshold: float = 0.7,
    signal3_threshold: float = 0.4,
    signal1_weight: float = 0.4,
    signal2_weight: float = 0.3,
    signal3_weight: float = 0.3,
    novelty_score_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Multi-signal novelty detection for tickets.

    Combines three signals to determine if a ticket represents a novel
    category that doesn't exist in the current taxonomy.

    Pipeline:
    1. Compute Signal 1: Check if max confidence is below threshold
    2. Compute Signal 2: Calculate entropy of confidence distribution
    3. Compute Signal 3: Measure embedding distance to category centroids
    4. Combine signals with weighted average
    5. Apply decision logic: is_novel = (max_conf < 0.5) OR (score > 0.6)

    Args:
        ticket_embedding: The ticket's embedding vector (3072 dimensions)
        category_confidence_scores: List of (category_id, confidence) tuples
        max_confidence: Highest confidence from any category
        signal1_threshold: Max confidence threshold (default: 0.5)
        signal2_threshold: Entropy threshold (default: 0.7)
        signal3_threshold: Distance threshold (default: 0.4)
        signal1_weight: Weight for signal 1 (default: 0.4)
        signal2_weight: Weight for signal 2 (default: 0.3)
        signal3_weight: Weight for signal 3 (default: 0.3)
        novelty_score_threshold: Score above this = novel (default: 0.6)

    Returns:
        Dict containing:
        - is_novel: bool - Final novelty decision
        - novelty_score: float - Combined score (0-1)
        - signals: Dict - Individual signal details
        - recommendation: str - Suggested action
        - reasoning: str - Human-readable explanation
    """
    # Get category embeddings singleton
    category_embeddings = CategoryEmbeddings.get_instance()

    # Extract just the confidence values for entropy calculation
    confidence_values = [score for _, score in category_confidence_scores] if category_confidence_scores else []

    # ================================================================
    # Compute all three signals
    # ================================================================
    signal_1 = compute_signal_1(max_confidence, signal1_threshold)
    signal_2 = compute_signal_2(confidence_values, signal2_threshold)
    signal_3 = compute_signal_3(ticket_embedding, category_embeddings, signal3_threshold)

    # ================================================================
    # Calculate weighted novelty score
    # ================================================================
    novelty_score = (
        signal1_weight * signal_1["value"] +
        signal2_weight * signal_2["value"] +
        signal3_weight * signal_3["value"]
    )

    # ================================================================
    # Final decision logic
    # is_novel = (max_confidence < 0.5) OR (novelty_score > 0.6)
    # ================================================================
    is_novel_by_confidence = max_confidence < signal1_threshold
    is_novel_by_score = novelty_score > novelty_score_threshold
    is_novel = is_novel_by_confidence or is_novel_by_score

    # ================================================================
    # Determine recommendation
    # ================================================================
    if is_novel:
        if novelty_score > 0.8:
            recommendation = "escalate"
            recommendation_reason = "Strong novelty indicators - requires immediate taxonomy review"
        elif is_novel_by_confidence and not is_novel_by_score:
            recommendation = "flag_for_review"
            recommendation_reason = "Low confidence match - may need new category"
        else:
            recommendation = "flag_for_review"
            recommendation_reason = "Multiple novelty signals detected - review category taxonomy"
    else:
        recommendation = "proceed"
        recommendation_reason = "Ticket matches existing categories sufficiently"

    # ================================================================
    # Build reasoning string
    # ================================================================
    signals_fired = sum([signal_1["fires"], signal_2["fires"], signal_3["fires"]])
    reasoning_parts = []

    if signal_1["fires"]:
        reasoning_parts.append(f"Low max confidence ({max_confidence:.2f} < {signal1_threshold})")
    if signal_2["fires"]:
        reasoning_parts.append(f"High entropy ({signal_2['actual']:.2f} > {signal2_threshold})")
    if signal_3["fires"]:
        reasoning_parts.append(f"Far from centroids (distance {signal_3['actual']:.2f} > {signal3_threshold})")

    if reasoning_parts:
        reasoning = f"Novelty detected: {'; '.join(reasoning_parts)}. Score: {novelty_score:.2f}"
    else:
        reasoning = f"No novelty indicators. Score: {novelty_score:.2f}. Nearest category: {signal_3['nearest_category']}"

    return {
        "is_novel": is_novel,
        "novelty_score": round(novelty_score, 4),
        "signals_fired": signals_fired,
        "signals": {
            "signal_1_max_confidence": signal_1,
            "signal_2_entropy": signal_2,
            "signal_3_centroid_distance": signal_3
        },
        "decision_factors": {
            "is_novel_by_confidence": is_novel_by_confidence,
            "is_novel_by_score": is_novel_by_score,
            "novelty_score_threshold": novelty_score_threshold
        },
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "reasoning": reasoning,
        "nearest_category": signal_3["nearest_category"]
    }


def detect_novelty_sync(
    ticket_embedding: List[float],
    category_confidence_scores: List[Tuple[str, float]],
    max_confidence: float
) -> Dict[str, Any]:
    """
    Synchronous wrapper for detect_novelty tool.

    Uses default thresholds from Config.

    Args:
        ticket_embedding: The ticket's embedding vector
        category_confidence_scores: List of (category_id, confidence) tuples
        max_confidence: Highest confidence from any category

    Returns:
        Novelty detection result dict
    """
    import asyncio

    return asyncio.run(detect_novelty.ainvoke({
        "ticket_embedding": ticket_embedding,
        "category_confidence_scores": category_confidence_scores,
        "max_confidence": max_confidence,
        "signal1_threshold": getattr(Config, 'NOVELTY_SIGNAL1_THRESHOLD', 0.5),
        "signal2_threshold": getattr(Config, 'NOVELTY_SIGNAL2_THRESHOLD', 0.7),
        "signal3_threshold": getattr(Config, 'NOVELTY_SIGNAL3_THRESHOLD', 0.4),
        "signal1_weight": getattr(Config, 'NOVELTY_SIGNAL1_WEIGHT', 0.4),
        "signal2_weight": getattr(Config, 'NOVELTY_SIGNAL2_WEIGHT', 0.3),
        "signal3_weight": getattr(Config, 'NOVELTY_SIGNAL3_WEIGHT', 0.3),
        "novelty_score_threshold": getattr(Config, 'NOVELTY_SCORE_THRESHOLD', 0.6)
    }))
