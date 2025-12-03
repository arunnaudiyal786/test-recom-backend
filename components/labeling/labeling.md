# Labeling Component

The Labeling component assigns labels to tickets using a **three-tier approach**: category labels from a predefined taxonomy, AI-generated business labels, and AI-generated technical labels.

## Overview

This component uses intelligent label assignment that:
1. Classifies tickets into predefined categories from `categories.json` (25 categories)
2. Generates business-oriented labels from an impact perspective
3. Generates technical labels from a root-cause perspective

All three methods run **in parallel** for optimal performance.

## File Structure

```
labeling/
├── __init__.py     # Public API exports
├── agent.py        # LangGraph node (labeling_node, LabelAssignmentAgent)
├── models.py       # Pydantic models (CategoryLabel, LabelWithConfidence, etc.)
├── service.py      # CategoryTaxonomy singleton + LabelingService
├── tools.py        # LangChain @tool functions
├── router.py       # FastAPI HTTP endpoints (/v2/labeling/*)
└── labeling.md     # This documentation
```

## Three-Tier Labeling Flow

```
Input Ticket
      │
      ├───────────────────────┬───────────────────────┐
      │                       │                       │
      ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  Category    │       │  Business    │       │  Technical   │
│  Labels      │       │  Labels      │       │  Labels      │
│ (taxonomy)   │       │ (AI-gen)     │       │ (AI-gen)     │
└──────────────┘       └──────────────┘       └──────────────┘
      │                       │                       │
   [CAT]                   [BIZ]                  [TECH]
      │                       │                       │
      └───────────────────────┴───────────────────────┘
                              │
                              ▼
                    Combined Unique Labels
```

| Type | Source | Prefix | Example |
|------|--------|--------|---------|
| **Category** | Predefined taxonomy | `[CAT]` | `[CAT] Batch Enrollment Maintenance` |
| **Business** | AI analysis | `[BIZ]` | `[BIZ] Customer-facing` |
| **Technical** | AI analysis | `[TECH]` | `[TECH] Database-connection` |

---

## Models (`models.py`)

### CategoryLabel

Category assignment from the predefined taxonomy.

```python
class CategoryLabel(BaseModel):
    id: str           # Category ID (e.g., "batch_enrollment_maintenance")
    name: str         # Human-readable name
    confidence: float # Confidence score (0-1)
    reasoning: Optional[str] = None
```

### LabelWithConfidence

AI-generated label with confidence.

```python
class LabelWithConfidence(BaseModel):
    label: str        # Label name
    confidence: float # Confidence score (0-1)
    category: str     # "category", "business", or "technical"
    reasoning: Optional[str] = None
```

### SimilarTicketInput

Simplified similar ticket for labeling input.

```python
class SimilarTicketInput(BaseModel):
    ticket_id: str
    title: str
    description: str
    labels: List[str] = []
    priority: str = "Medium"
    resolution: Optional[str] = None
```

### LabelingRequest

Input for label assignment.

```python
class LabelingRequest(BaseModel):
    title: str
    description: str
    domain: str           # MM, CIW, Specialty
    priority: str = "Medium"
    similar_tickets: List[Dict]
```

### LabelingResponse

Output from label assignment.

```python
class LabelingResponse(BaseModel):
    category_labels: List[CategoryLabel]        # From taxonomy
    business_labels: List[LabelWithConfidence]  # AI-generated
    technical_labels: List[LabelWithConfidence] # AI-generated
    all_labels: List[str]                       # Combined with prefixes
    novelty_detected: bool = False              # Novel category flag
    novelty_reasoning: Optional[str] = None
```

---

## Service (`service.py`)

Contains two classes: **CategoryTaxonomy** (singleton) and **LabelingService**.

### CategoryTaxonomy

Singleton cache for category taxonomy from `categories.json`.

```python
class CategoryTaxonomy:
    """Loads categories once at first access."""

    @classmethod
    def get_instance(cls) -> "CategoryTaxonomy"

    @classmethod
    def reset_instance(cls) -> None  # For testing

    def get_all_categories(self) -> List[Dict[str, Any]]
    def get_category_by_id(self, category_id: str) -> Optional[Dict]
    def get_category_ids(self) -> List[str]
    def get_confidence_threshold(self, category_id: str) -> float
    def get_max_labels_per_ticket(self) -> int
    def get_novelty_threshold(self) -> float
    def format_categories_for_prompt(self) -> str
    def format_categories_compact(self) -> str
    def validate_category_assignment(self, category_id: str, confidence: float) -> bool
    def get_category_count(self) -> int
    def get_settings(self) -> Dict[str, Any]
```

### LabelingConfig

Configuration for LabelingService.

```python
class LabelingConfig(ComponentConfig):
    labeling_model: str = "gpt-4o"
    labeling_temperature: float = 0.2
    label_confidence_threshold: float = 0.7
    generated_label_confidence_threshold: float = 0.7
    enable_ai_labels: bool = True
    max_business_labels: int = 5
    max_technical_labels: int = 5

    class Config:
        env_prefix = "LABELING_"
```

### LabelingService

Main service for HTTP API usage.

```python
class LabelingService(BaseComponent[LabelingRequest, LabelingResponse]):
    def __init__(self, config: Optional[LabelingConfig] = None)

    async def process(self, request: LabelingRequest) -> LabelingResponse
    async def health_check(self) -> Dict[str, Any]

    # Internal methods
    async def _assign_category_labels(title, description, priority)
    async def _generate_business_labels(title, description, domain, priority, existing_labels)
    async def _generate_technical_labels(title, description, domain, priority, existing_labels)
```

---

## Tools (`tools.py`)

LangChain `@tool` decorated functions for LangGraph integration.

### classify_ticket_categories

```python
@tool
async def classify_ticket_categories(
    title: str,
    description: str,
    priority: str
) -> Dict[str, Any]:
    """
    Classify a ticket into categories from the predefined taxonomy.

    Uses thresholds from Config:
    - Config.CATEGORY_MAX_LABELS_PER_TICKET (default: 3)
    - Config.CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD (default: 0.7)
    - Config.CATEGORY_NOVELTY_DETECTION_THRESHOLD (default: 0.5)

    Returns:
        - assigned_categories: List of {id, name, confidence, reasoning}
        - novelty_detected: Boolean
        - novelty_reasoning: Explanation if novelty detected
        - actual_prompt: The prompt sent to LLM
    """
```

### generate_business_labels

```python
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
    Generate business-oriented labels.

    Categories: Impact, Urgency, Process

    Returns:
        - labels: List of {label, confidence, category, reasoning}
        - actual_prompt: The prompt sent to LLM
    """
```

### generate_technical_labels

```python
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
    Generate technical labels.

    Categories: Component, Issue Type, Root Cause

    Returns:
        - labels: List of {label, confidence, category, reasoning}
        - actual_prompt: The prompt sent to LLM
    """
```

---

## Agent (`agent.py`)

LangGraph node for the workflow.

### labeling_node

```python
async def labeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for label assignment.

    Runs all three labeling methods in parallel using asyncio.gather().

    State Requirements:
        - title: Ticket title
        - description: Ticket description
        - classified_domain: Domain (optional, defaults to "Unknown")
        - priority: Priority (optional, defaults to "Medium")

    Returns partial state update:
        - category_labels: Assigned categories
        - novelty_detected: Boolean
        - novelty_reasoning: Explanation
        - business_labels: AI-generated business labels
        - technical_labels: AI-generated technical labels
        - assigned_labels: Combined with prefixes (backward compat)
        - label_assignment_prompts: {category, business, technical}
        - status: "success" or "error"
        - current_agent: "labeling"
        - messages: Status message
    """
```

### LabelAssignmentAgent

Callable wrapper for backward compatibility.

```python
class LabelAssignmentAgent:
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await labeling_node(state)

label_assignment_agent = LabelAssignmentAgent()  # Singleton
```

---

## Router (`router.py`)

FastAPI HTTP endpoints.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v2/labeling/assign` | Assign labels to a ticket |
| `GET` | `/v2/labeling/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/v2/labeling/assign \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database connection timeout",
    "description": "MM_ALDER experiencing connection pool exhaustion",
    "domain": "MM",
    "priority": "High",
    "similar_tickets": []
  }'
```

### Example Response

```json
{
  "category_labels": [
    {
      "id": "database_performance",
      "name": "Database Performance",
      "confidence": 0.88,
      "reasoning": "Connection pool exhaustion is a database performance issue"
    }
  ],
  "business_labels": [
    {
      "label": "Customer-facing",
      "confidence": 0.78,
      "category": "business",
      "reasoning": "Service affects member-facing operations"
    }
  ],
  "technical_labels": [
    {
      "label": "Database-connection",
      "confidence": 0.88,
      "category": "technical",
      "reasoning": "Connection pool exhaustion is DB-related"
    }
  ],
  "all_labels": [
    "[CAT] Database Performance",
    "[BIZ] Customer-facing",
    "[TECH] Database-connection"
  ],
  "novelty_detected": false,
  "novelty_reasoning": null
}
```

---

## Public Exports (`__init__.py`)

```python
from components.labeling import (
    # LangGraph node (primary)
    labeling_node,
    label_assignment_agent,

    # LangChain tools
    classify_ticket_categories,
    generate_business_labels,
    generate_technical_labels,

    # Category taxonomy
    CategoryTaxonomy,

    # Models
    LabelingRequest,
    LabelingResponse,
    LabelWithConfidence,
    CategoryLabel,
    SimilarTicketInput,

    # Service
    LabelingService,
    LabelingConfig,

    # Router
    router,
)
```

---

## Category Taxonomy

Categories are defined in `data/metadata/categories.json`.

### Structure

```json
{
  "categories": [
    {
      "id": "batch_enrollment_maintenance",
      "name": "Batch Enrollment Maintenance",
      "description": "Issues related to enrollment batch processing",
      "keywords": ["enrollment", "batch", "maintenance"],
      "examples": ["Enrollment file failed to process"],
      "confidence_threshold": 0.75
    }
  ],
  "settings": {}
}
```

### Per-Category Thresholds

Each category can have a custom `confidence_threshold`. If not specified, uses `Config.CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD` (0.7).

---

## Novelty Detection

If a ticket doesn't match any category with sufficient confidence:

```python
if not category_labels and not novelty_detected:
    novelty_detected = True
    novelty_reasoning = "No categories matched with sufficient confidence"
```

This flags tickets for potential taxonomy expansion.

---

## Configuration

### Centralized Config (`src/utils/config.py`)

```python
CATEGORIES_JSON_PATH = PROJECT_ROOT / "data" / "metadata" / "categories.json"
CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD = 0.7
CATEGORY_MAX_LABELS_PER_TICKET = 3
CATEGORY_NOVELTY_DETECTION_THRESHOLD = 0.5
CATEGORY_CLASSIFICATION_MODEL = "gpt-4o"
CATEGORY_CLASSIFICATION_TEMPERATURE = 0.2
```

### Environment Variables

```bash
LABELING_MODEL=gpt-4o
LABELING_TEMPERATURE=0.2
LABELING_CONFIDENCE_THRESHOLD=0.7
LABELING_ENABLE_AI_LABELS=true
LABELING_MAX_BUSINESS_LABELS=5
LABELING_MAX_TECHNICAL_LABELS=5
```

---

## Usage Examples

### LangGraph Workflow

```python
from components.labeling import labeling_node

state = {
    "title": "Login failure",
    "description": "Users unable to authenticate",
    "classified_domain": "MM",
    "priority": "Critical",
}

result = await labeling_node(state)
print(f"Labels: {result['assigned_labels']}")
print(f"Novelty: {result['novelty_detected']}")
```

### Direct Tool Usage

```python
from components.labeling import classify_ticket_categories

result = await classify_ticket_categories.ainvoke({
    "title": "DB timeout",
    "description": "Connection pool exhausted",
    "priority": "High"
})

print(result["assigned_categories"])
```

### CategoryTaxonomy

```python
from components.labeling import CategoryTaxonomy

taxonomy = CategoryTaxonomy.get_instance()
print(f"Categories: {taxonomy.get_category_count()}")
print(f"IDs: {taxonomy.get_category_ids()[:3]}")
```

### HTTP Service

```python
from components.labeling import LabelingService, LabelingRequest

service = LabelingService()
response = await service.process(
    LabelingRequest(
        title="API timeout",
        description="External API calls timing out",
        domain="CIW",
        priority="High",
        similar_tickets=[]
    )
)
```

---

## Performance

| Metric | Value |
|--------|-------|
| Parallel execution | 3 concurrent API calls |
| Total time | ~2-3 seconds |
| API calls | 3 (category + business + technical) |
| Cost per ticket | ~$0.02 |

---

## Error Handling

```python
try:
    result = await labeling_node(state)
except Exception as e:
    return {
        "category_labels": [],
        "business_labels": [],
        "technical_labels": [],
        "assigned_labels": [],
        "novelty_detected": True,
        "novelty_reasoning": f"Labeling failed: {str(e)}",
        "status": "error",
        "current_agent": "labeling",
        "error_message": f"Labeling failed: {str(e)}"
    }
```

---

## Prompt Transparency

The component returns actual prompts for debugging:

```python
result = await labeling_node(state)
print(result["label_assignment_prompts"]["category"])
print(result["label_assignment_prompts"]["business"])
print(result["label_assignment_prompts"]["technical"])
```
