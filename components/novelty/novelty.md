# Novelty Detection Component

The Novelty Detection component detects when tickets don't match any known category in the taxonomy using a **multi-signal analysis** approach. It runs after the Label Assignment Agent to flag potentially novel ticket types that may require taxonomy expansion.

## Overview

This component uses three complementary signals to detect novel tickets:
1. **Signal 1: Maximum Confidence Score** - Is the best category match weak?
2. **Signal 2: Confidence Distribution Entropy** - Is uncertainty spread evenly across categories?
3. **Signal 3: Embedding Distance to Centroids** - Is the ticket semantically far from all category centroids?

## Architecture

```
novelty/
├── __init__.py          # Public API exports
├── agent.py             # LangGraph node wrapper (novelty_node)
├── tools.py             # Multi-signal detection logic
└── novelty.md           # This documentation
```

## Multi-Signal Detection Pipeline

```
                     Ticket Embedding + Category Scores
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│   Signal 1   │           │   Signal 2   │           │   Signal 3   │
│    Max       │           │   Entropy    │           │  Centroid    │
│  Confidence  │           │ Distribution │           │  Distance    │
│ (Weight 0.4) │           │ (Weight 0.3) │           │ (Weight 0.3) │
└──────────────┘           └──────────────┘           └──────────────┘
       │                           │                           │
       │    fires if < 0.5         │    fires if > 0.7         │    fires if > 0.4
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Weighted Novelty   │
                        │        Score         │
                        │                      │
                        │ score = 0.4×S1 +     │
                        │         0.3×S2 +     │
                        │         0.3×S3       │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Decision Logic     │
                        │                      │
                        │ is_novel =           │
                        │   (max_conf < 0.5)   │
                        │   OR                 │
                        │   (score > 0.6)      │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Recommendation     │
                        │                      │
                        │ - proceed            │
                        │ - flag_for_review    │
                        │ - escalate           │
                        └──────────────────────┘
```

---

## Components

### Agent (`agent.py`)

LangGraph node for novelty detection.

#### novelty_node

```python
async def novelty_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for novelty detection.

    Analyzes whether a ticket represents a novel category using three signals:
    1. Maximum confidence score (is the best match weak?)
    2. Confidence distribution entropy (is uncertainty spread evenly?)
    3. Embedding distance to centroids (is ticket semantically far from all categories?)

    State Requirements:
        - ticket_embedding: The ticket's semantic embedding (from labeling step)
        - all_category_scores: Similarity scores for all categories
        - category_labels: Assigned categories with confidence scores

    Returns partial state update:
        - novelty_detected: bool - Final novelty decision
        - novelty_score: float - Combined score (0-1)
        - novelty_signals: Dict - Individual signal details
        - novelty_recommendation: str - Suggested action
        - novelty_reasoning: str - Human-readable explanation
        - novelty_details: Dict - Additional details
        - status: "success" or "error"
        - current_agent: "novelty"
        - messages: Status message
    """
```

#### NoveltyDetectionAgent

Callable wrapper class for backward compatibility.

```python
class NoveltyDetectionAgent:
    """Callable wrapper for novelty_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await novelty_node(state)

# Singleton instance
novelty_detection_agent = NoveltyDetectionAgent()
```

---

### Tools (`tools.py`)

LangChain `@tool` decorated functions for novelty detection.

#### detect_novelty

```python
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
        - signals_fired: int - Number of signals that fired (0-3)
        - recommendation: str - Suggested action
        - recommendation_reason: str - Explanation for recommendation
        - reasoning: str - Human-readable explanation
        - decision_factors: Dict - Factors that led to decision
        - nearest_category: str - Closest category by embedding distance
    """
```

#### Helper Functions

```python
def calculate_normalized_entropy(confidence_scores: List[float]) -> float:
    """
    Calculate normalized Shannon entropy of confidence distribution.

    High entropy = uncertainty spread evenly = ticket doesn't fit any category well.
    Low entropy = confidence concentrated = ticket fits specific categories.

    Returns:
        Normalized entropy in range [0, 1]
        - 0: All confidence in one category (certain)
        - 1: Equal confidence across all categories (maximum uncertainty)
    """

def calculate_min_centroid_distance(
    ticket_embedding: List[float],
    category_embeddings: CategoryEmbeddings
) -> Tuple[float, str]:
    """
    Calculate minimum distance from ticket to any category centroid.
    Uses cosine distance: distance = 1 - cosine_similarity

    Returns:
        Tuple of (min_distance, nearest_category_id)
    """

def compute_signal_1(max_confidence: float, threshold: float = 0.5) -> Dict[str, Any]:
    """Signal 1: Maximum Confidence Score Analysis."""

def compute_signal_2(confidence_scores: List[float], threshold: float = 0.7) -> Dict[str, Any]:
    """Signal 2: Confidence Distribution Entropy."""

def compute_signal_3(
    ticket_embedding: List[float],
    category_embeddings: CategoryEmbeddings,
    threshold: float = 0.4
) -> Dict[str, Any]:
    """Signal 3: Embedding Distance to Nearest Category Centroid."""

def detect_novelty_sync(...) -> Dict[str, Any]:
    """Synchronous wrapper for detect_novelty tool."""
```

---

## The Three Signals Explained

### Signal 1: Maximum Confidence Score (Weight: 40%)

**Purpose**: Detects when the best category match is weak.

**Logic**:
- If `max_confidence < threshold (0.5)`, the signal fires
- When fired, contributes `1.0` to the weighted score; otherwise `0.0`

**Reasoning**: If the LLM's highest confidence for any category is below 50%, the ticket likely doesn't fit any known category well.

| max_confidence | threshold | fires | value |
|----------------|-----------|-------|-------|
| 0.35           | 0.5       | Yes   | 1.0   |
| 0.55           | 0.5       | No    | 0.0   |
| 0.80           | 0.5       | No    | 0.0   |

### Signal 2: Confidence Distribution Entropy (Weight: 30%)

**Purpose**: Detects when uncertainty is spread evenly across all categories.

**Logic**:
- Calculates normalized Shannon entropy of confidence scores
- If `entropy > threshold (0.7)`, the signal fires
- Entropy = 1.0 means equal probability across all categories (maximum uncertainty)
- Entropy = 0.0 means all confidence in one category

**Formula**:
```
H = -Σ(p * log2(p))
normalized_entropy = H / log2(n)  # n = number of categories
```

**Reasoning**: High entropy means the ticket doesn't strongly match any single category - it's equally (un)likely to be any of them.

| confidence_distribution | entropy | fires | interpretation |
|-------------------------|---------|-------|----------------|
| [0.9, 0.05, 0.05]       | Low     | No    | Clear match to one category |
| [0.33, 0.33, 0.34]      | High    | Yes   | Equally uncertain about all |
| [0.5, 0.3, 0.2]         | Medium  | Maybe | Some preference but not strong |

### Signal 3: Embedding Distance to Centroids (Weight: 30%)

**Purpose**: Detects when the ticket is semantically far from all known category centroids.

**Logic**:
- Calculates cosine distance from ticket embedding to each category centroid
- Uses pre-computed category embeddings (via `CategoryEmbeddings` singleton)
- If `min_distance > threshold (0.4)`, the signal fires
- Distance = 1 - cosine_similarity

**Reasoning**: Even if an LLM assigns a category with moderate confidence, if the ticket's embedding is far from all category centroids, it may represent a new topic not captured by existing categories.

| min_distance | threshold | fires | interpretation |
|--------------|-----------|-------|----------------|
| 0.25         | 0.4       | No    | Close to known categories |
| 0.45         | 0.4       | Yes   | Semantically different |
| 0.70         | 0.4       | Yes   | Very far from all categories |

---

## Decision Logic

### Novelty Score Calculation

```python
novelty_score = (0.4 * signal_1_value) + (0.3 * signal_2_value) + (0.3 * signal_3_value)
```

### Final Decision

```python
is_novel = (max_confidence < 0.5) OR (novelty_score > 0.6)
```

**Two paths to being marked novel**:
1. **Direct path**: Max confidence is below 0.5 (even if other signals don't fire)
2. **Combined path**: Weighted novelty score exceeds 0.6 (multiple weak signals combine)

### Recommendations

| Condition | Recommendation | Reason |
|-----------|----------------|--------|
| `novelty_score > 0.8` | `escalate` | Strong novelty indicators - requires immediate taxonomy review |
| `is_novel_by_confidence` only | `flag_for_review` | Low confidence match - may need new category |
| `is_novel` by score | `flag_for_review` | Multiple novelty signals detected - review category taxonomy |
| Not novel | `proceed` | Ticket matches existing categories sufficiently |

---

## Configuration

### Environment Variables / Config

Thresholds can be configured via `Config` class (in `config/config.py`):

```python
NOVELTY_SIGNAL1_THRESHOLD = 0.5       # Max confidence threshold
NOVELTY_SIGNAL2_THRESHOLD = 0.7       # Entropy threshold
NOVELTY_SIGNAL3_THRESHOLD = 0.4       # Distance threshold
NOVELTY_SIGNAL1_WEIGHT = 0.4          # Weight for signal 1
NOVELTY_SIGNAL2_WEIGHT = 0.3          # Weight for signal 2
NOVELTY_SIGNAL3_WEIGHT = 0.3          # Weight for signal 3
NOVELTY_SCORE_THRESHOLD = 0.6         # Final score threshold
```

---

## Usage Examples

### LangGraph Workflow

```python
from components.novelty.agent import novelty_node

state = {
    "ticket_id": "T-123",
    "title": "Unknown issue type",
    "description": "This ticket describes a completely new problem...",
    "ticket_embedding": [...],  # 3072-dim vector from labeling
    "all_category_scores": [
        {"category_id": "batch_enrollment", "score": 0.35},
        {"category_id": "database_performance", "score": 0.32},
        {"category_id": "api_integration", "score": 0.33}
    ],
    "category_labels": [
        {"id": "batch_enrollment", "confidence": 0.35}
    ]
}

result = await novelty_node(state)

print(f"Novel: {result['novelty_detected']}")        # True
print(f"Score: {result['novelty_score']}")           # 0.85
print(f"Recommendation: {result['novelty_recommendation']}")  # "escalate"
print(f"Reasoning: {result['novelty_reasoning']}")
# "Novelty detected: Low max confidence (0.35 < 0.5); High entropy (0.99 > 0.7). Score: 0.85"
```

### Direct Tool Usage

```python
from components.novelty.tools import detect_novelty

result = await detect_novelty.ainvoke({
    "ticket_embedding": [...],  # 3072-dim vector
    "category_confidence_scores": [
        ("batch_enrollment", 0.35),
        ("database_performance", 0.32),
        ("api_integration", 0.33)
    ],
    "max_confidence": 0.35,
    "signal1_threshold": 0.5,
    "signal2_threshold": 0.7,
    "signal3_threshold": 0.4,
    "novelty_score_threshold": 0.6
})

print(result["is_novel"])       # True
print(result["signals_fired"])  # 2 or 3
print(result["nearest_category"])  # "batch_enrollment"
```

---

## Integration with Workflow

The Novelty Detection Agent runs **after Label Assignment** and **before Resolution Generation**:

```
Pattern Recognition Agent
        ↓
Label Assignment Agent
        ↓ (provides ticket_embedding, all_category_scores, category_labels)
Novelty Detection Agent  ← THIS COMPONENT
        ↓ (adds novelty_detected, novelty_score, novelty_recommendation)
Resolution Generation Agent
        ↓
Final Output
```

### State Dependencies

**Input from Label Assignment**:
- `ticket_embedding`: Semantic embedding of the ticket (3072 dimensions)
- `all_category_scores`: Confidence scores for all categories
- `category_labels`: Assigned categories with confidence

**Output to Resolution Generation**:
- `novelty_detected`: Boolean flag for novel ticket
- `novelty_score`: Numerical score (0-1)
- `novelty_recommendation`: Action suggestion
- `novelty_reasoning`: Human-readable explanation

---

## Output Fields

The novelty node adds the following fields to state:

```python
{
    "novelty_detected": True,           # Boolean: is this a novel ticket?
    "novelty_score": 0.85,              # Float: combined weighted score (0-1)
    "novelty_signals": {                # Individual signal details
        "signal_1_max_confidence": {
            "name": "max_confidence",
            "fires": True,
            "value": 1.0,
            "threshold": 0.5,
            "actual": 0.35,
            "reasoning": "Max confidence 0.350 < threshold 0.5"
        },
        "signal_2_entropy": {
            "name": "entropy",
            "fires": True,
            "value": 1.0,
            "threshold": 0.7,
            "actual": 0.99,
            "num_categories": 3,
            "reasoning": "Normalized entropy 0.990 > threshold 0.7"
        },
        "signal_3_centroid_distance": {
            "name": "centroid_distance",
            "fires": False,
            "value": 0.0,
            "threshold": 0.4,
            "actual": 0.25,
            "nearest_category": "batch_enrollment",
            "reasoning": "Min distance 0.250 <= threshold 0.4"
        }
    },
    "novelty_recommendation": "escalate",  # "proceed" | "flag_for_review" | "escalate"
    "novelty_reasoning": "Novelty detected: Low max confidence (0.35 < 0.5); High entropy (0.99 > 0.7). Score: 0.85",
    "novelty_details": {
        "signals_fired": 2,
        "decision_factors": {
            "is_novel_by_confidence": True,
            "is_novel_by_score": True,
            "novelty_score_threshold": 0.6
        },
        "recommendation_reason": "Strong novelty indicators - requires immediate taxonomy review",
        "nearest_category": "batch_enrollment"
    },
    "status": "success",
    "current_agent": "novelty"
}
```

---

## Error Handling

Novelty detection is **non-blocking** - errors don't fail the pipeline:

```python
try:
    result = await novelty_node(state)
except Exception as e:
    # Return safe defaults - don't block pipeline
    return {
        "novelty_detected": False,
        "novelty_score": 0.0,
        "novelty_signals": {},
        "novelty_recommendation": "proceed",
        "novelty_reasoning": f"Error during detection: {str(e)}",
        "status": "success",  # Still "success" to not block pipeline
        "error_message": f"Novelty detection failed: {str(e)}"
    }
```

**Rationale**: Novelty detection is an enhancement, not a critical step. If it fails, the pipeline should continue with resolution generation.

---

## Performance

| Metric | Value |
|--------|-------|
| Dependencies | ticket_embedding, category_scores from labeling |
| Computation | Signal calculations are local (no API calls) |
| Time | <50ms (if category embeddings are cached) |
| Cost | $0 (no LLM calls, pure computation) |

---

## Public Exports (`__init__.py`)

```python
from components.novelty.agent import novelty_node
from components.novelty.tools import detect_novelty

__all__ = ["novelty_node", "detect_novelty"]
```

---

## When Novelty Detection Matters

1. **New Product Features**: When users report issues with newly launched features not yet in taxonomy
2. **Emerging Issue Patterns**: When novel bug types or integration issues appear
3. **Taxonomy Maintenance**: Identifies when category definitions need expansion
4. **Quality Control**: Prevents mis-categorization of truly novel issues
5. **Escalation Routing**: Helps route genuinely unknown issues to specialists

---

## Tuning Guide

### If Too Many False Positives (marking normal tickets as novel):
- Increase `NOVELTY_SIGNAL1_THRESHOLD` (e.g., 0.4 → 0.3)
- Decrease `NOVELTY_SCORE_THRESHOLD` (e.g., 0.6 → 0.7)
- Decrease signal weights that fire too often

### If Too Many False Negatives (missing truly novel tickets):
- Decrease `NOVELTY_SIGNAL1_THRESHOLD` (e.g., 0.5 → 0.6)
- Increase `NOVELTY_SCORE_THRESHOLD` (e.g., 0.6 → 0.5)
- Increase weights on more sensitive signals

### Recommended Baseline:
```python
NOVELTY_SIGNAL1_THRESHOLD = 0.5   # Conservative - only fire if really uncertain
NOVELTY_SIGNAL2_THRESHOLD = 0.7   # High bar for entropy (near-uniform distribution)
NOVELTY_SIGNAL3_THRESHOLD = 0.4   # Moderate distance threshold
NOVELTY_SCORE_THRESHOLD = 0.6     # Require multiple signals to combine
```
