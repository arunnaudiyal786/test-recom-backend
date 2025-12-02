# Classification Component

The Classification component classifies support tickets into domains (MM, CIW, Specialty) using a **Multi-Task Classification with LLM (MTC-LLM)** approach with parallel binary classifiers.

## Overview

Instead of using a single multi-class classifier, this component runs **3 parallel binary classifiers** (one per domain). Each classifier independently evaluates whether the ticket belongs to its domain, returning a decision and confidence score. The final domain is determined by the classifier with the highest confidence among positive decisions.

## Architecture

```
classification/
├── __init__.py          # Public API exports
├── agent.py             # LangGraph node wrapper
├── models.py            # Pydantic request/response models
├── service.py           # ClassificationService (legacy, full-featured)
├── tools.py             # LangChain @tool decorated functions
├── router.py            # FastAPI HTTP endpoints
└── README.md            # This file
```

## Classification Logic

### MTC-LLM Approach

```
Input Ticket
     │
     ├──► MM Classifier ──────► {decision: true, confidence: 0.85}
     │
     ├──► CIW Classifier ─────► {decision: false, confidence: 0.60}
     │
     └──► Specialty Classifier ► {decision: false, confidence: 0.45}
                                        │
                                        ▼
                            Final: MM (confidence: 0.85)
```

### Why Binary Classifiers?

1. **Independence**: Each classifier focuses solely on its domain characteristics
2. **Extensibility**: Adding a new domain only requires adding one new classifier
3. **Transparency**: Each domain provides its own reasoning and confidence
4. **Parallelism**: All classifiers run concurrently via `asyncio.gather()`

### Domain Definitions

Domains are loaded from `config/schema_config.yaml`:

| Domain | Description | Key Indicators |
|--------|-------------|----------------|
| **MM** | Member Management | MM_ALDER, MMALDR, member eligibility, enrollment |
| **CIW** | Claims Integration Workflow | Claims processing, provider lookup, eligibility verification |
| **Specialty** | Custom/Specialty Modules | Custom workflows, specialty reporting, dashboards |

## Components

### Models (`models.py`)

#### DomainScore

Individual classifier result for one domain.

```python
class DomainScore(BaseModel):
    domain: str          # Domain name (MM, CIW, Specialty)
    decision: bool       # Whether this domain matches
    confidence: float    # Confidence score (0-1)
    reasoning: str       # Explanation for the decision
    keywords: List[str]  # Extracted domain-specific keywords
```

#### ClassificationRequest

Input for classification.

```python
class ClassificationRequest(BaseModel):
    title: str        # Ticket title/summary (min 1 char)
    description: str  # Ticket description with full context (min 1 char)
```

#### ClassificationResponse

Output from classification.

```python
class ClassificationResponse(BaseModel):
    classified_domain: str           # Final domain (MM, CIW, Specialty)
    confidence: float                # Final confidence score (0-1)
    reasoning: str                   # Combined reasoning
    domain_scores: Dict[str, DomainScore]  # Individual scores per domain
    extracted_keywords: List[str]    # All unique keywords found
```

---

### Tools (`tools.py`)

LangChain `@tool` decorated functions used by the LangGraph workflow.

#### `classify_ticket_domain`

Main classification tool that orchestrates parallel binary classifiers.

```python
@tool
async def classify_ticket_domain(title: str, description: str) -> Dict[str, Any]:
    """
    Classify a ticket into one of the configured domains.

    Args:
        title: The ticket title
        description: The ticket description

    Returns:
        Dict containing:
        - classified_domain: Final domain
        - confidence: Confidence score (0-1)
        - reasoning: Combined reasoning from all classifiers
        - domain_scores: Individual scores for each domain
        - extracted_keywords: Keywords found in the ticket
    """
```

**Internal Flow**:
1. Loads domains from `config/schema_config.yaml`
2. Builds prompts from configuration
3. Runs `_classify_single_domain()` for each domain in parallel
4. Determines final domain from positive decisions with highest confidence
5. Returns combined result

#### `_classify_single_domain`

Internal function for single domain classification.

```python
async def _classify_single_domain(
    domain: str,
    title: str,
    description: str,
    llm: ChatOpenAI,
    domain_prompts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Classify a single domain using binary classifier.

    Returns:
        {
            "decision": bool,
            "confidence": float,
            "reasoning": str,
            "extracted_keywords": List[str]
        }
    """
```

#### Helper Functions

| Function | Description |
|----------|-------------|
| `_build_domain_prompt()` | Builds full prompt from domain template |
| `_get_domain_prompts()` | Gets prompts from schema config |
| `_get_configured_domains()` | Gets domain list from schema config |
| `classify_ticket_domain_sync()` | Synchronous wrapper for non-async contexts |

---

### Agent (`agent.py`)

LangGraph node wrapper for the classification tools.

#### `classification_node`

```python
async def classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for domain classification.

    Args:
        state: Current workflow state containing ticket info

    Returns:
        Partial state update with:
        - classified_domain: Final domain
        - classification_confidence: Confidence score
        - classification_reasoning: Explanation
        - classification_scores: Per-domain scores
        - extracted_keywords: Found keywords
        - status: "success" or "error"
        - current_agent: "classification"
        - messages: Status message for workflow
    """
```

**State Requirements**:
- `title`: Ticket title
- `description`: Ticket description
- `ticket_id`: Optional ticket identifier

#### ClassificationAgent

Callable wrapper for backward compatibility.

```python
class ClassificationAgent:
    """Callable wrapper for classification_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await classification_node(state)

# Singleton instance
classification_agent = ClassificationAgent()
```

---

### Service (`service.py`)

Full-featured service class for HTTP API usage.

#### ClassificationConfig

```python
class ClassificationConfig(ComponentConfig):
    classification_model: str = "gpt-4o"
    classification_temperature: float = 0.2
    domains: List[str] = ["MM", "CIW", "Specialty"]
    confidence_threshold: float = 0.7

    class Config:
        env_prefix = "CLASSIFICATION_"
```

| Setting | Environment Variable | Default |
|---------|---------------------|---------|
| Model | `CLASSIFICATION_MODEL` | `gpt-4o` |
| Temperature | `CLASSIFICATION_TEMPERATURE` | `0.2` |
| Confidence Threshold | `CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.7` |

#### ClassificationService

```python
class ClassificationService(BaseComponent[ClassificationRequest, ClassificationResponse]):
    """
    Service for classifying tickets into domains.

    Uses parallel binary classifiers (MTC-LLM approach).
    """

    async def process(self, request: ClassificationRequest) -> ClassificationResponse:
        """Classify a ticket into a domain."""

    async def health_check(self) -> Dict[str, Any]:
        """Check if classification service is healthy."""
```

**Key Methods**:

| Method | Description |
|--------|-------------|
| `_classify_domain()` | Run binary classifier for one domain |
| `_classify_all_domains()` | Run all classifiers in parallel |
| `_determine_final_domain()` | Select final domain from results |

---

### Router (`router.py`)

FastAPI HTTP endpoints.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v2/classification/classify` | Classify a ticket |
| `GET` | `/v2/classification/health` | Health check |

#### Example Request

```bash
curl -X POST http://localhost:8000/v2/classification/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "MM_ALDER service connection timeout",
    "description": "The MM_ALDER service is experiencing connection timeouts when trying to connect to the member database during peak hours."
  }'
```

#### Example Response

```json
{
  "classified_domain": "MM",
  "confidence": 0.92,
  "reasoning": "Selected MM with confidence 0.92.\n✓ MM (0.92): Contains MM_ALDER keyword...\n✗ CIW (0.35): No claims-related terms...\n✗ Specialty (0.28): No specialty indicators...",
  "domain_scores": {
    "MM": {
      "domain": "MM",
      "decision": true,
      "confidence": 0.92,
      "reasoning": "Contains MM_ALDER keyword, member database reference",
      "keywords": ["MM_ALDER", "member database"]
    },
    "CIW": { ... },
    "Specialty": { ... }
  },
  "extracted_keywords": ["MM_ALDER", "member database", "connection timeout"]
}
```

## Domain Selection Logic

```python
def _determine_final_domain(self, classifications: Dict[str, Dict]) -> Tuple[str, float, str]:
    """
    Determine final domain from classifier results.

    Logic:
    - Prefer domains with decision=True
    - Select highest confidence among positive decisions
    - If no positive decisions, use highest raw confidence * 0.3
    """
    domain_scores = {}

    for domain, result in classifications.items():
        if result.get("decision", False):
            domain_scores[domain] = result.get("confidence", 0.0)
        else:
            # Penalize negative decisions
            domain_scores[domain] = result.get("confidence", 0.0) * 0.3

    final_domain = max(domain_scores, key=domain_scores.get)
    final_confidence = domain_scores[final_domain]

    return final_domain, final_confidence, combined_reasoning
```

## Prompt Engineering

Each domain has a Chain-of-Thought prompt structure:

```
1. Domain definition with specific indicators
2. Chain-of-Thought process steps
3. Ticket content (title + description)
4. JSON output format specification
```

**Key Prompt Features**:
- **JSON Mode**: Uses `response_format={"type": "json_object"}`
- **Low Temperature**: `0.2` for deterministic classification
- **Keyword Extraction**: Each classifier extracts domain-specific keywords
- **Reasoning Required**: Classifiers must explain their decisions

## Usage Examples

### Direct Tool Usage

```python
from components.classification.tools import classify_ticket_domain

result = await classify_ticket_domain.ainvoke({
    "title": "Claims validation error",
    "description": "CIW integration rejecting claims with error CIW-5001"
})
print(result["classified_domain"])  # "CIW"
```

### Service Usage

```python
from components.classification.service import ClassificationService
from components.classification.models import ClassificationRequest

service = ClassificationService()
response = await service.process(
    ClassificationRequest(
        title="Custom workflow engine error",
        description="Specialty reporting dashboard not loading"
    )
)
print(response.classified_domain)  # "Specialty"
```

### LangGraph Workflow

```python
from components.classification.agent import classification_node

state = {
    "ticket_id": "JIRA-123",
    "title": "MM_ALDER timeout",
    "description": "Connection issues in member database"
}

result = await classification_node(state)
print(result["classified_domain"])  # "MM"
```

## Error Handling

The component handles errors gracefully:

```python
try:
    result = await classify_ticket_domain(...)
except Exception as e:
    return {
        "status": "error",
        "error_message": f"Classification failed: {str(e)}",
        "current_agent": "classification"
    }
```

**Retry Logic**: API calls use exponential backoff (2s, 4s, 8s) for transient failures.

## Performance

- **Parallel Execution**: All 3 classifiers run simultaneously
- **Total Time**: ~2-3 seconds (limited by slowest API call)
- **API Calls**: 3 calls per classification (one per domain)
- **Cost**: ~$0.01 per classification (gpt-4o)

## Configuration

### Environment Variables

```bash
CLASSIFICATION_MODEL=gpt-4o
CLASSIFICATION_TEMPERATURE=0.2
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
```

### Schema Config

Domains and prompts are defined in `config/schema_config.yaml`:

```yaml
domains:
  MM:
    name: "Member Management"
    prompt: |
      You are an AI classifier for MM domain tickets...
  CIW:
    name: "Claims Integration Workflow"
    prompt: |
      You are an AI classifier for CIW domain tickets...
```

## Adding a New Domain

1. Add domain to `config/schema_config.yaml` with name and prompt
2. Add historical tickets for the new domain to CSV
3. Rebuild FAISS index: `python3 scripts/setup_vectorstore.py`

No code changes required - the system automatically loads new domains from config.
