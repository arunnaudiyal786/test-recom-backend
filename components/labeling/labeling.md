# Labeling Component

The Labeling component assigns labels to tickets using a **three-tier approach**: historical labels from similar tickets (validated by AI), AI-generated business labels, and AI-generated technical labels.

## Overview

Unlike simple tag extraction, this component uses intelligent label assignment that:
1. Validates historical labels against the current ticket context
2. Generates new business-oriented labels from an impact perspective
3. Generates new technical labels from a root-cause perspective

All three methods run **in parallel** for optimal performance.

## Architecture

```
labeling/
├── __init__.py          # Public API exports
├── agent.py             # LangGraph node wrapper (LabelAssignmentAgent)
├── models.py            # Pydantic request/response models
├── service.py           # LabelingService (full-featured)
├── tools.py             # LangChain @tool decorated functions
├── router.py            # FastAPI HTTP endpoints
└── README.md            # This file
```

## Three-Tier Labeling Approach

```
Input Ticket + Similar Tickets
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│Historic│ │Business│ │Technic-│
│Labels  │ │Labels  │ │al      │
│(AI-val)│ │(AI-gen)│ │Labels  │
└────────┘ └────────┘ └────────┘
    │         │         │
    │    (prefix: [BIZ]) │(prefix: [TECH])
    │         │         │
    └─────────┼─────────┘
              ▼
    Combined Unique Labels
```

### Label Categories

| Category | Source | Prefix | Example |
|----------|--------|--------|---------|
| **Historical** | Similar tickets | None | `Code Fix`, `#MM_ALDER` |
| **Business** | AI analysis | `[BIZ]` | `[BIZ] Customer-facing` |
| **Technical** | AI analysis | `[TECH]` | `[TECH] Database-issue` |

## Components

### Models (`models.py`)

#### LabelWithConfidence

A label with its confidence score and metadata.

```python
class LabelWithConfidence(BaseModel):
    label: str               # Label name
    confidence: float        # Confidence score (0-1)
    category: str            # 'historical', 'business', or 'technical'
    reasoning: Optional[str] # Explanation for assignment
```

#### LabelingRequest

Input for label assignment.

```python
class LabelingRequest(BaseModel):
    title: str                    # Ticket title
    description: str              # Ticket description
    domain: str                   # Classified domain (MM, CIW, Specialty)
    priority: str = "Medium"      # Ticket priority
    similar_tickets: List[Dict]   # Similar tickets from retrieval
```

#### LabelingResponse

Output from label assignment.

```python
class LabelingResponse(BaseModel):
    historical_labels: List[LabelWithConfidence]  # From similar tickets
    business_labels: List[LabelWithConfidence]    # AI-generated business
    technical_labels: List[LabelWithConfidence]   # AI-generated technical
    all_labels: List[str]         # Combined unique labels
    label_distribution: Dict[str, str]  # Distribution in similar tickets
```

---

### Tools (`tools.py`)

LangChain `@tool` decorated functions for LangGraph integration.

#### `extract_candidate_labels`

Extracts unique labels and their frequency from similar tickets.

```python
@tool
def extract_candidate_labels(
    similar_tickets: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract unique labels and their frequency from similar tickets.

    Args:
        similar_tickets: List of similar ticket dicts with 'labels' field

    Returns:
        Dict containing:
        - candidate_labels: List of unique labels
        - label_distribution: Dict mapping label to {count, percentage, formatted}
        - total_tickets: Number of tickets analyzed
    """
```

**Example Output**:
```python
{
    "candidate_labels": ["Code Fix", "#MM_ALDER", "Configuration Fix"],
    "label_distribution": {
        "Code Fix": {"count": 14, "percentage": 0.7, "formatted": "14/20"},
        "#MM_ALDER": {"count": 8, "percentage": 0.4, "formatted": "8/20"}
    },
    "total_tickets": 20
}
```

#### `evaluate_historical_labels`

Validates historical labels using parallel binary classifiers.

```python
@tool
async def evaluate_historical_labels(
    title: str,
    description: str,
    domain: str,
    candidate_labels: List[str],
    label_distribution: Dict[str, Dict],
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Evaluate historical labels using parallel binary classifiers.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        candidate_labels: List of candidate labels to evaluate
        label_distribution: Dict mapping label to frequency info
        confidence_threshold: Minimum confidence to assign (default 0.7)

    Returns:
        Dict containing:
        - assigned_labels: Labels that passed threshold
        - label_confidence: Dict mapping label to confidence
        - all_evaluations: Full results for all labels
        - sample_prompt: Sample prompt used for transparency
    """
```

**Binary Classifier Prompt**:
```
You are a label validation expert for technical support tickets.

Evaluate whether the label "{label_name}" should be assigned.

Historical frequency: This label appears in {frequency} similar tickets.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}

Consider:
1. Does the ticket content match the label semantics?
2. Is the historical frequency a strong indicator?
3. What is your confidence level?

Output JSON:
{
  "assign_label": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}
```

#### `generate_business_labels`

Generates business-oriented labels from an impact perspective.

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
    Generate business-oriented labels using AI analysis.

    Business label categories:
    - Impact: Customer-facing, Internal, Revenue-impacting
    - Urgency: Time-sensitive, Compliance-related, SLA-bound
    - Process: Workflow-blocking, Data-quality, Integration-issue

    Returns:
        Dict with:
        - labels: List of label dicts with confidence
        - actual_prompt: The prompt sent to LLM
    """
```

#### `generate_technical_labels`

Generates technical labels from a root-cause perspective.

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
    Generate technical labels using AI analysis.

    Technical label categories:
    - Component: Database, API, UI, Integration, Batch
    - Issue Type: Performance, Error, Configuration, Data
    - Root Cause: Connection, Timeout, Memory, Logic, External

    Returns:
        Dict with:
        - labels: List of label dicts with confidence
        - actual_prompt: The prompt sent to LLM
    """
```

---

### Agent (`agent.py`)

LangGraph node wrapper that orchestrates all three labeling methods.

#### `labeling_node`

```python
async def labeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for label assignment.

    Runs three methods in parallel:
    1. Historical label evaluation
    2. Business label generation
    3. Technical label generation

    Args:
        state: Current workflow state with ticket info and similar tickets

    Returns:
        Partial state update with:
        - historical_labels: Validated historical labels
        - historical_label_confidence: Confidence per label
        - historical_label_distribution: Frequency distribution
        - business_labels: AI-generated business labels
        - technical_labels: AI-generated technical labels
        - assigned_labels: Combined unique labels (backward compat)
        - label_assignment_prompts: Actual prompts used
        - status: "success" or "error"
        - current_agent: "labeling"
    """
```

**State Requirements**:
- `title`: Ticket title
- `description`: Ticket description
- `classified_domain`: Domain from classification (optional, defaults to "Unknown")
- `priority`: Ticket priority (optional, defaults to "Medium")
- `similar_tickets`: List of similar tickets from retrieval

#### LabelAssignmentAgent

Callable wrapper class.

```python
class LabelAssignmentAgent:
    """Callable wrapper for labeling_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await labeling_node(state)

# Singleton instance
label_assignment_agent = LabelAssignmentAgent()
```

---

### Service (`service.py`)

Full-featured service class for HTTP API usage.

#### LabelingConfig

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

| Setting | Environment Variable | Default |
|---------|---------------------|---------|
| Model | `LABELING_MODEL` | `gpt-4o` |
| Temperature | `LABELING_TEMPERATURE` | `0.2` |
| Confidence Threshold | `LABELING_CONFIDENCE_THRESHOLD` | `0.7` |
| Enable AI Labels | `LABELING_ENABLE_AI_LABELS` | `True` |

#### LabelingService

```python
class LabelingService(BaseComponent[LabelingRequest, LabelingResponse]):
    """Service for assigning labels to tickets."""

    async def process(self, request: LabelingRequest) -> LabelingResponse:
        """Assign labels to a ticket."""

    async def health_check(self) -> Dict[str, Any]:
        """Check if labeling service is healthy."""
```

**Key Methods**:

| Method | Description |
|--------|-------------|
| `_extract_candidate_labels()` | Get unique labels from similar tickets |
| `_calculate_label_distribution()` | Calculate label frequency |
| `_evaluate_historical_label()` | Binary classifier for one label |
| `_assign_historical_labels()` | Validate all historical labels |
| `_generate_business_labels()` | Generate business labels |
| `_generate_technical_labels()` | Generate technical labels |

---

### Router (`router.py`)

FastAPI HTTP endpoints.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v2/labeling/assign` | Assign labels to a ticket |
| `GET` | `/v2/labeling/health` | Health check |

#### Example Request

```bash
curl -X POST http://localhost:8000/v2/labeling/assign \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database connection timeout",
    "description": "MM_ALDER experiencing connection pool exhaustion",
    "domain": "MM",
    "priority": "High",
    "similar_tickets": [
      {
        "ticket_id": "JIRA-MM-001",
        "title": "Similar DB issue",
        "labels": ["Code Fix", "#MM_ALDER"]
      }
    ]
  }'
```

#### Example Response

```json
{
  "historical_labels": [
    {
      "label": "Code Fix",
      "confidence": 0.85,
      "category": "historical",
      "reasoning": "Ticket describes code-related issue requiring fix"
    },
    {
      "label": "#MM_ALDER",
      "confidence": 0.92,
      "category": "historical",
      "reasoning": "Explicitly mentions MM_ALDER service"
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
    "Code Fix",
    "#MM_ALDER",
    "[BIZ] Customer-facing",
    "[TECH] Database-connection"
  ],
  "label_distribution": {
    "Code Fix": "14/20",
    "#MM_ALDER": "8/20"
  }
}
```

## Historical Label Validation Logic

Each historical label goes through binary classification:

```python
async def _evaluate_single_label(label_name, title, description, domain, frequency, llm):
    """
    Binary classifier for a single label.

    Uses historical frequency as context:
    "This label appears in 14/20 similar tickets"

    Returns:
        {
            "label": label_name,
            "assign": bool,
            "confidence": float,
            "reasoning": str
        }
    """
```

Labels are only assigned if:
- `assign == True` (positive decision)
- `confidence >= 0.7` (meets threshold)

## Parallel Execution

All three methods run concurrently:

```python
historical_task = evaluate_historical_labels.ainvoke(...)
business_task = generate_business_labels.ainvoke(...)
technical_task = generate_technical_labels.ainvoke(...)

historical_result, business_result, technical_result = await asyncio.gather(
    historical_task, business_task, technical_task
)
```

This reduces total latency from ~9s (sequential) to ~3s (parallel).

## Usage Examples

### Direct Tool Usage

```python
from components.labeling.tools import (
    extract_candidate_labels,
    evaluate_historical_labels,
    generate_business_labels
)

# Step 1: Extract candidates
candidates = extract_candidate_labels.invoke({
    "similar_tickets": similar_tickets
})

# Step 2: Evaluate historical labels
historical = await evaluate_historical_labels.ainvoke({
    "title": "DB timeout",
    "description": "Connection pool exhausted",
    "domain": "MM",
    "candidate_labels": candidates["candidate_labels"],
    "label_distribution": candidates["label_distribution"],
    "confidence_threshold": 0.7
})

print(historical["assigned_labels"])
```

### Service Usage

```python
from components.labeling.service import LabelingService
from components.labeling.models import LabelingRequest

service = LabelingService()
response = await service.process(
    LabelingRequest(
        title="API timeout error",
        description="External API calls timing out",
        domain="CIW",
        priority="High",
        similar_tickets=[...]
    )
)

print(f"Historical: {[l.label for l in response.historical_labels]}")
print(f"Business: {[l.label for l in response.business_labels]}")
print(f"Technical: {[l.label for l in response.technical_labels]}")
```

### LangGraph Workflow

```python
from components.labeling.agent import labeling_node

state = {
    "title": "Login failure",
    "description": "Users unable to authenticate",
    "classified_domain": "MM",
    "priority": "Critical",
    "similar_tickets": [...]
}

result = await labeling_node(state)
print(f"Assigned: {result['assigned_labels']}")
```

## Label Categories

### Business Labels

Generated from a business analyst perspective:

| Category | Examples |
|----------|----------|
| **Impact** | Customer-facing, Internal, Revenue-impacting |
| **Urgency** | Time-sensitive, Compliance-related, SLA-bound |
| **Process** | Workflow-blocking, Data-quality, Integration-issue |

### Technical Labels

Generated from an engineer perspective:

| Category | Examples |
|----------|----------|
| **Component** | Database, API, UI, Integration, Batch |
| **Issue Type** | Performance, Error, Configuration, Data |
| **Root Cause** | Connection, Timeout, Memory, Logic, External |

## Performance

- **Parallel Execution**: All 3 methods run simultaneously
- **Historical Evaluation**: 1 API call per candidate label (parallel)
- **Business Generation**: 1 API call
- **Technical Generation**: 1 API call
- **Total Time**: ~2-3 seconds (limited by slowest call)
- **Cost**: ~$0.02 per labeling (depends on label count)

## Error Handling

```python
try:
    result = await labeling_node(state)
except Exception as e:
    return {
        "historical_labels": [],
        "business_labels": [],
        "technical_labels": [],
        "assigned_labels": [],
        "status": "error",
        "error_message": f"Labeling failed: {str(e)}"
    }
```

## Configuration

### Environment Variables

```bash
LABELING_MODEL=gpt-4o
LABELING_TEMPERATURE=0.2
LABELING_CONFIDENCE_THRESHOLD=0.7
LABELING_ENABLE_AI_LABELS=true
LABELING_MAX_BUSINESS_LABELS=5
LABELING_MAX_TECHNICAL_LABELS=5
```

### Disabling AI-Generated Labels

Set `enable_ai_labels: False` in config to only use historical labels:

```python
config = LabelingConfig(enable_ai_labels=False)
service = LabelingService(config)
# Only returns historical_labels, no business/technical
```

## Prompt Transparency

The component returns the actual prompts used for debugging:

```python
result = await labeling_node(state)
print(result["label_assignment_prompts"]["historical"])
print(result["label_assignment_prompts"]["business"])
print(result["label_assignment_prompts"]["technical"])
```

This enables prompt engineering and debugging of label decisions.
