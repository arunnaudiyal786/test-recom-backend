# Novelty Detection Implementation Plan

## High-Level Summary

This plan implements **multi-signal novelty detection** for the ticket labeling system. The goal is to accurately detect when a ticket doesn't fit into any predefined category by combining three complementary signals:

1. **Maximum Confidence Score** - Detects weak best-match (weight: 0.4)
2. **Confidence Distribution Entropy** - Detects uncertainty spread (weight: 0.3)
3. **Embedding Distance to Centroids** - Detects semantic distance (weight: 0.3)

### Architecture Decision: Merge with Labeling Component

**Recommendation: MERGE into existing labeling component** rather than creating a new component.

**Rationale:**
1. **Data Dependencies** - Novelty detection requires:
   - Binary classification results (already in `tools.py`)
   - Ticket embedding (already generated in `tools.py`)
   - Category embeddings/centroids (already loaded in `category_embeddings.py`)

2. **Execution Timing** - Novelty detection runs AFTER category classification completes, within the same pipeline step

3. **Single Responsibility** - The labeling component already handles `novelty_detected` and `novelty_reasoning` fields

4. **Avoids Code Duplication** - Creating a separate component would require duplicating embedding generation and classification logic

5. **Minimal State Changes** - Only needs to enrich existing `novelty_detected` boolean with a richer `novelty_score` and signal breakdown

---

## Implementation Plan

### Step 1: Add Novelty Detection Configuration to Config
**File:** `src/utils/config.py`
**Lines:** After line 90 (after ENSEMBLE_LLM_WEIGHT)

Add new configuration constants:
```python
# ========== NOVELTY DETECTION CONFIGURATION ==========
# Signal 1: Maximum confidence threshold
NOVELTY_MAX_CONFIDENCE_THRESHOLD = 0.5  # Below this triggers signal 1
NOVELTY_SIGNAL_1_WEIGHT = 0.4           # Weight for max confidence signal

# Signal 2: Entropy threshold
NOVELTY_ENTROPY_THRESHOLD = 0.7         # Normalized entropy above this triggers signal 2
NOVELTY_SIGNAL_2_WEIGHT = 0.3           # Weight for entropy signal

# Signal 3: Centroid distance threshold
NOVELTY_CENTROID_DISTANCE_THRESHOLD = 0.4  # Distance above this triggers signal 3
NOVELTY_SIGNAL_3_WEIGHT = 0.3              # Weight for centroid distance signal

# Final decision threshold
NOVELTY_SCORE_THRESHOLD = 0.6           # Combined score above this = novel
```

---

### Step 2: Create Novelty Detector Module
**File:** `components/labeling/novelty_detector.py` (NEW FILE)

Create a dedicated module for novelty detection logic with clear separation of concerns:

```python
"""
Novelty Detector - Multi-signal novelty detection for category classification.

Implements three complementary signals to detect when a ticket doesn't fit
into any predefined category:

Signal 1: Maximum Confidence Score (weight: 0.4)
- If max_confidence < 0.5 → signal fires
- Intuition: If best match is weak, ticket doesn't fit

Signal 2: Confidence Distribution Entropy (weight: 0.3)
- Calculate entropy of confidence scores
- If normalized_entropy > 0.7 → signal fires
- Intuition: Uncertain across all categories = doesn't fit any

Signal 3: Embedding Distance to Centroids (weight: 0.3)
- Calculate distance from ticket to nearest category centroid
- If min_distance > 0.4 → signal fires
- Intuition: Semantically far from all categories

Final Decision:
novelty_score = (0.4 × signal_1) + (0.3 × signal_2) + (0.3 × signal_3)
is_novel = (max_confidence < 0.5) OR (novelty_score > 0.6)
"""
```

**Key Classes/Functions:**
- `NoveltySignals` - Dataclass for signal values
- `NoveltyResult` - Dataclass for detection result
- `compute_entropy_signal()` - Calculate normalized entropy
- `compute_centroid_distance_signal()` - Calculate min distance to category centroids
- `detect_novelty()` - Main function combining all signals

---

### Step 3: Extend CategoryEmbeddings for Centroid Distance
**File:** `components/labeling/category_embeddings.py`
**Lines:** After `get_top_k_candidates()` method (around line 197)

Add method for computing minimum distance to category centroids:

```python
def compute_min_centroid_distance(
    self,
    ticket_embedding: List[float]
) -> float:
    """
    Compute minimum distance from ticket embedding to any category centroid.

    For single-embedding-per-category (current design), each category embedding
    IS the centroid. Distance = 1 - cosine_similarity.

    Returns:
        Minimum distance (0 = identical, 2 = opposite)
    """
```

**Note:** Since we have one embedding per category, each category's embedding effectively IS its centroid. Distance = 1 - similarity (for cosine similarity).

---

### Step 4: Integrate Novelty Detection into Tools
**File:** `components/labeling/tools.py`
**Location:** Inside `classify_ticket_categories()` function

**4a. Import novelty detector (line ~25):**
```python
from components.labeling.novelty_detector import detect_novelty, NoveltyResult
```

**4b. After Step 5 (ensemble scoring) at line ~271, add novelty detection:**
```python
# ================================================================
# STEP 5.5: Enhanced Novelty Detection (Multi-Signal)
# ================================================================
pipeline_info["step5_5_novelty_detection"] = "in_progress"

# Collect all confidence scores from binary classifiers
all_confidences = [
    result["confidence"] for result in binary_results
    if result.get("confidence") is not None
]

# Compute novelty using multi-signal approach
novelty_result = detect_novelty(
    classification_confidences=all_confidences,
    ticket_embedding=ticket_embedding,
    category_embeddings=category_embeddings
)

pipeline_info["step5_5_novelty_detection"] = f"completed (score: {novelty_result.novelty_score:.3f})"
```

**4c. Update return statement to include rich novelty data (lines ~304-316):**
```python
return {
    "assigned_categories": assigned_categories,

    # Enhanced novelty detection output
    "novelty_detected": novelty_result.is_novel,
    "novelty_score": novelty_result.novelty_score,
    "novelty_reasoning": novelty_result.reasoning,
    "novelty_signals": {
        "signal_1_max_confidence": novelty_result.signals.signal_1_max_confidence,
        "signal_2_entropy": novelty_result.signals.signal_2_entropy,
        "signal_3_centroid_distance": novelty_result.signals.signal_3_centroid_distance,
        "max_confidence": novelty_result.signals.max_confidence,
        "normalized_entropy": novelty_result.signals.normalized_entropy,
        "min_centroid_distance": novelty_result.signals.min_centroid_distance
    },

    "semantic_candidates": semantic_candidates,
    "pipeline_info": pipeline_info
}
```

---

### Step 5: Update Pydantic Models
**File:** `components/labeling/models.py`
**Lines:** After line 97

Add new models for novelty detection:

```python
class NoveltySignalDetails(BaseModel):
    """Detailed breakdown of novelty detection signals."""

    signal_1_max_confidence: float = Field(
        description="Signal 1 value (0 or 1): 1 if max confidence < threshold"
    )
    signal_2_entropy: float = Field(
        description="Signal 2 value (0 or 1): 1 if entropy > threshold"
    )
    signal_3_centroid_distance: float = Field(
        description="Signal 3 value (0 or 1): 1 if min distance > threshold"
    )
    max_confidence: float = Field(
        description="Actual maximum confidence from classifiers"
    )
    normalized_entropy: float = Field(
        description="Actual normalized entropy of confidence distribution"
    )
    min_centroid_distance: float = Field(
        description="Actual minimum distance to category centroids"
    )


class LabelingResponse(BaseModel):
    # ... existing fields ...

    # Enhanced novelty detection
    novelty_score: Optional[float] = Field(
        default=None,
        description="Combined novelty score (0-1), higher = more likely novel"
    )
    novelty_signals: Optional[NoveltySignalDetails] = Field(
        default=None,
        description="Breakdown of individual novelty signals"
    )
```

---

### Step 6: Update Workflow State Schema
**File:** `src/orchestrator/state.py`
**Lines:** Around line 46 (after `novelty_reasoning`)

Add new state fields:

```python
# Enhanced Novelty Detection
novelty_score: Optional[float]  # Combined novelty score (0-1)
novelty_signals: Optional[Dict]  # Signal breakdown for debugging
```

---

### Step 7: Update Agent to Pass Through New Fields
**File:** `components/labeling/agent.py`
**Lines:** Around line 82 (after `novelty_reasoning`)

```python
novelty_score = category_result.get("novelty_score", 0.0)
novelty_signals = category_result.get("novelty_signals", {})
```

And add to return dict (line ~111):

```python
return {
    "category_labels": category_labels,
    "novelty_detected": novelty_detected,
    "novelty_reasoning": novelty_reasoning,
    "novelty_score": novelty_score,      # NEW
    "novelty_signals": novelty_signals,  # NEW
    # ... rest unchanged
}
```

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/utils/config.py` | MODIFY | Add novelty detection thresholds |
| `components/labeling/novelty_detector.py` | CREATE | Core novelty detection logic |
| `components/labeling/category_embeddings.py` | MODIFY | Add centroid distance method |
| `components/labeling/tools.py` | MODIFY | Integrate novelty detection into pipeline |
| `components/labeling/models.py` | MODIFY | Add Pydantic models for novelty signals |
| `src/orchestrator/state.py` | MODIFY | Add state fields for novelty data |
| `components/labeling/agent.py` | MODIFY | Pass through new novelty fields |

---

## Algorithm Details

### Signal 1: Maximum Confidence Score

```python
def compute_max_confidence_signal(confidences: List[float]) -> Tuple[float, float]:
    """
    Signal fires if best classification confidence is weak.

    Returns:
        (signal_value, max_confidence)
        signal_value: 1.0 if max_confidence < 0.5, else 0.0
    """
    max_conf = max(confidences) if confidences else 0.0
    signal = 1.0 if max_conf < 0.5 else 0.0
    return signal, max_conf
```

### Signal 2: Confidence Distribution Entropy

```python
import math

def compute_entropy_signal(confidences: List[float]) -> Tuple[float, float]:
    """
    Signal fires if confidence is spread uniformly across categories.
    High entropy = uncertain across all categories = doesn't fit any.

    Returns:
        (signal_value, normalized_entropy)
        signal_value: 1.0 if normalized_entropy > 0.7, else 0.0
    """
    if not confidences or len(confidences) < 2:
        return 0.0, 0.0

    # Normalize to probability distribution
    total = sum(confidences)
    if total == 0:
        return 1.0, 1.0  # Uniform zero = maximum entropy

    probs = [c / total for c in confidences]

    # Calculate entropy: H = -sum(p * log(p))
    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)

    # Normalize by maximum entropy (uniform distribution)
    max_entropy = math.log(len(confidences))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    signal = 1.0 if normalized > 0.7 else 0.0
    return signal, normalized
```

### Signal 3: Embedding Distance to Centroids

```python
def compute_centroid_distance_signal(
    ticket_embedding: List[float],
    category_embeddings: CategoryEmbeddings
) -> Tuple[float, float]:
    """
    Signal fires if ticket is semantically far from all category centroids.

    Distance = 1 - cosine_similarity
    - 0.0 = identical (similarity = 1.0)
    - 2.0 = opposite (similarity = -1.0)

    Returns:
        (signal_value, min_distance)
        signal_value: 1.0 if min_distance > 0.4, else 0.0
    """
    similarities = category_embeddings.compute_similarities(ticket_embedding)
    if not similarities:
        return 1.0, 2.0  # No categories = maximum distance

    max_similarity = similarities[0][1]  # Already sorted descending
    min_distance = 1.0 - max_similarity

    signal = 1.0 if min_distance > 0.4 else 0.0
    return signal, min_distance
```

### Final Novelty Decision

```python
def detect_novelty(
    classification_confidences: List[float],
    ticket_embedding: List[float],
    category_embeddings: CategoryEmbeddings
) -> NoveltyResult:
    """
    Combine all signals for final novelty decision.

    Formula:
    novelty_score = (0.4 × signal_1) + (0.3 × signal_2) + (0.3 × signal_3)
    is_novel = (max_confidence < 0.5) OR (novelty_score > 0.6)
    """
    # Compute individual signals
    signal_1, max_conf = compute_max_confidence_signal(classification_confidences)
    signal_2, entropy = compute_entropy_signal(classification_confidences)
    signal_3, distance = compute_centroid_distance_signal(ticket_embedding, category_embeddings)

    # Weighted combination
    novelty_score = (0.4 * signal_1) + (0.3 * signal_2) + (0.3 * signal_3)

    # Decision: immediate trigger on weak max confidence OR high combined score
    is_novel = (max_conf < 0.5) or (novelty_score > 0.6)

    # Generate reasoning
    reasons = []
    if signal_1 > 0:
        reasons.append(f"Weak best match (confidence: {max_conf:.2f})")
    if signal_2 > 0:
        reasons.append(f"High uncertainty across categories (entropy: {entropy:.2f})")
    if signal_3 > 0:
        reasons.append(f"Semantically distant from all categories (distance: {distance:.2f})")

    reasoning = "; ".join(reasons) if reasons else "Ticket fits within known categories"

    return NoveltyResult(
        is_novel=is_novel,
        novelty_score=novelty_score,
        reasoning=reasoning,
        signals=NoveltySignals(
            signal_1_max_confidence=signal_1,
            signal_2_entropy=signal_2,
            signal_3_centroid_distance=signal_3,
            max_confidence=max_conf,
            normalized_entropy=entropy,
            min_centroid_distance=distance
        )
    )
```

---

## Potential Challenges

1. **Empty Classification Results** - When no categories pass the semantic pre-filter, we won't have binary classification confidences. Solution: Use semantic similarity scores as proxy confidences.

2. **Single-Category Entropy** - Entropy is undefined for single-category case. Solution: Return entropy=0 (certain) for single category.

3. **Category Centroids** - Current design has one embedding per category (single representative text). For true centroids, we'd need multiple historical ticket embeddings per category. Solution: Use the single category embedding as a surrogate centroid.

4. **Threshold Tuning** - The 0.5, 0.7, 0.4 thresholds are heuristics that may need tuning based on real ticket data. Solution: Make all thresholds configurable via Config.

---

## Testing Strategy

1. **Unit Tests for Novelty Detector:**
   - Test entropy calculation with known distributions
   - Test distance calculation with mock embeddings
   - Test final decision logic with edge cases

2. **Integration Test:**
   - Run sample ticket through full pipeline
   - Verify novelty signals appear in output JSON
   - Verify frontend can display signal breakdown

3. **Manual Testing:**
   - Submit tickets that should be novel (IT ops issues vs healthcare categories)
   - Submit tickets that should match (healthcare test plan tickets)
   - Verify detection accuracy
