# Resolution Component

The Resolution component generates comprehensive resolution plans using **Chain-of-Thought (CoT) reasoning** based on ticket context and similar historical resolutions. This is the final agent in the pipeline, producing actionable diagnostic and resolution steps.

## Overview

This component synthesizes all prior analysis (classification, retrieval, labeling) to generate:
- Executive summary of the resolution approach
- Diagnostic steps to confirm the issue
- Resolution steps with commands, validation, and rollback procedures
- Time estimates and confidence scores
- Alternative approaches

## Architecture

```
resolution/
├── __init__.py          # Public API exports
├── agent.py             # LangGraph node wrapper (ResolutionGenerationAgent)
├── models.py            # Pydantic request/response models
├── tools.py             # LangChain @tool decorated functions
└── README.md            # This file
```

## Resolution Generation Pipeline

```
Ticket Context + Similar Tickets + Labels
                    │
                    ▼
        ┌───────────────────────┐
        │ Analyze Historical    │
        │ Resolutions (Top 5)   │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Chain-of-Thought      │
        │ Reasoning             │
        │                       │
        │ 1. Analyze symptoms   │
        │ 2. Review patterns    │
        │ 3. Identify root cause│
        │ 4. Plan diagnostics   │
        │ 5. Develop resolution │
        │ 6. Consider risks     │
        │ 7. Estimate time      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Structured            │
        │ Resolution Plan       │
        └───────────────────────┘
```

## Components

### Models (`models.py`)

#### DiagnosticStep

A single diagnostic step to confirm the issue.

```python
class DiagnosticStep(BaseModel):
    step_number: int              # Step sequence number
    description: str              # What to check/verify
    commands: List[str]           # Commands to run
    expected_output: str          # What to expect
    estimated_time_minutes: int   # Time estimate (default: 5)
```

#### ResolutionStep

A single resolution step with validation and rollback.

```python
class ResolutionStep(BaseModel):
    step_number: int              # Step sequence number
    description: str              # What action to take
    commands: List[str]           # Commands to execute
    validation: str               # How to verify success
    estimated_time_minutes: int   # Time estimate (default: 10)
    risk_level: str               # "low", "medium", "high"
    rollback_procedure: Optional[str]  # How to rollback
```

#### ResolutionPlan

Complete resolution plan structure.

```python
class ResolutionPlan(BaseModel):
    summary: str                  # Executive summary (2-3 sentences)
    diagnostic_steps: List[DiagnosticStep]
    resolution_steps: List[ResolutionStep]
    additional_considerations: List[str]  # Extra notes
    references: List[str]         # Doc/ticket references
    total_estimated_time_hours: float
    confidence: float             # Confidence score (0-1)
    alternative_approaches: List[str]  # Other options
```

#### ResolutionRequest

Input for resolution generation.

```python
class ResolutionRequest(BaseModel):
    title: str                    # Ticket title
    description: str              # Ticket description
    domain: str                   # Classified domain
    priority: str = "Medium"      # Ticket priority
    labels: List[str] = []        # Assigned labels
    similar_tickets: List[dict]   # Similar tickets from retrieval
    avg_similarity: float = 0.0   # Average similarity score
```

#### ResolutionResponse

Output from resolution generation.

```python
class ResolutionResponse(BaseModel):
    resolution_plan: ResolutionPlan
    confidence: float             # Overall confidence
```

---

### Tools (`tools.py`)

LangChain `@tool` decorated functions for LangGraph integration.

#### `analyze_similar_resolutions`

Formats historical resolutions for context.

```python
@tool
def analyze_similar_resolutions(
    similar_tickets: List[Dict[str, Any]]
) -> str:
    """
    Analyze resolution patterns from similar historical tickets.

    Extracts and formats resolution information from top 5 similar tickets
    to provide context for resolution generation.

    Args:
        similar_tickets: List of similar ticket dicts with resolution info

    Returns:
        Formatted string of historical resolution patterns
    """
```

**Example Output**:
```
--- Historical Ticket 1 ---
ID: JIRA-MM-042
Title: MM_ALDER connection pool exhaustion
Similarity: 87.50%
Labels: Code Fix, #MM_ALDER
Resolution Time: 4.5 hours
Resolution:
1. Increased connection pool size to 50
2. Added monitoring alerts for pool usage
3. Implemented connection timeout of 30s
...
```

#### `generate_resolution_plan`

Main resolution generation function using Chain-of-Thought.

```python
@tool
async def generate_resolution_plan(
    title: str,
    description: str,
    domain: str,
    priority: str,
    labels: List[str],
    historical_context: str,
    avg_similarity: float
) -> Dict[str, Any]:
    """
    Generate a comprehensive resolution plan using Chain-of-Thought reasoning.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        labels: Assigned labels
        historical_context: Formatted string of similar resolutions
        avg_similarity: Average similarity to historical tickets

    Returns:
        Dict containing:
        - resolution_plan: Complete resolution plan dict
        - confidence: Confidence score (0-1)
        - actual_prompt: The prompt sent to LLM
    """
```

#### `_build_resolution_plan`

Helper function to validate and build the resolution plan.

```python
def _build_resolution_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build and validate resolution plan from raw LLM data.

    - Applies defaults for missing fields
    - Validates step structure
    - Calculates total time if not provided

    Returns:
        Validated resolution plan dict
    """
```

---

### Agent (`agent.py`)

LangGraph node wrapper that orchestrates resolution generation.

#### `resolution_node`

```python
async def resolution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for resolution generation.

    Analyzes similar ticket resolutions and generates a comprehensive
    resolution plan for the current ticket.

    Args:
        state: Current workflow state with ticket info, labels, and similar tickets

    Returns:
        Partial state update with:
        - resolution_plan: Complete resolution plan dict
        - resolution_confidence: Confidence score
        - resolution_generation_prompt: Actual prompt used
        - status: "success" or "error"
        - current_agent: "resolution"
        - messages: Status message
    """
```

**State Requirements**:
- `title`: Ticket title
- `description`: Ticket description
- `classified_domain`: Domain from classification (optional)
- `priority`: Ticket priority (optional)
- `similar_tickets`: Similar tickets from retrieval
- `assigned_labels`: Combined labels from labeling (optional)
- `historical_labels`, `business_labels`, `technical_labels`: Alternative label sources
- `search_metadata`: Search metadata with avg_similarity

#### ResolutionGenerationAgent

Callable wrapper class.

```python
class ResolutionGenerationAgent:
    """Callable wrapper for resolution_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await resolution_node(state)

# Singleton instance
resolution_agent = ResolutionGenerationAgent()
```

---

## Chain-of-Thought Prompt Structure

The resolution prompt uses a structured CoT approach:

```
=== CURRENT TICKET ===
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}
Labels: {labels}
Average similarity to historical tickets: {avg_similarity}

=== SIMILAR HISTORICAL TICKETS ===
{historical_context}

=== CHAIN-OF-THOUGHT PROCESS ===
1. Analyze the current ticket's symptoms and context
2. Review patterns from similar historical tickets
3. Identify likely root causes
4. Plan diagnostic steps to confirm the issue
5. Develop resolution steps with proper validation
6. Consider risks and rollback procedures
7. Estimate time and assign confidence level

=== OUTPUT FORMAT (JSON) ===
{resolution_plan_schema}
```

## Output JSON Schema

```json
{
  "summary": "Executive summary of the resolution approach (2-3 sentences)",
  "diagnostic_steps": [
    {
      "step_number": 1,
      "description": "What to check/verify",
      "commands": ["command1", "command2"],
      "expected_output": "What to expect",
      "estimated_time_minutes": 5
    }
  ],
  "resolution_steps": [
    {
      "step_number": 1,
      "description": "What action to take",
      "commands": ["command1", "command2"],
      "validation": "How to verify the step worked",
      "estimated_time_minutes": 10,
      "risk_level": "low|medium|high",
      "rollback_procedure": "How to rollback if needed"
    }
  ],
  "additional_considerations": ["consideration1", "consideration2"],
  "references": ["reference to docs/tickets"],
  "total_estimated_time_hours": 2.5,
  "confidence": 0.85,
  "alternative_approaches": ["Alternative 1", "Alternative 2"]
}
```

## Usage Examples

### Direct Tool Usage

```python
from components.resolution.tools import (
    analyze_similar_resolutions,
    generate_resolution_plan
)

# Step 1: Analyze historical resolutions
context = analyze_similar_resolutions.invoke({
    "similar_tickets": similar_tickets
})

# Step 2: Generate resolution plan
result = await generate_resolution_plan.ainvoke({
    "title": "Database connection timeout",
    "description": "Users experiencing timeouts during peak hours",
    "domain": "MM",
    "priority": "High",
    "labels": ["Code Fix", "#MM_ALDER"],
    "historical_context": context,
    "avg_similarity": 0.85
})

print(result["resolution_plan"]["summary"])
```

### LangGraph Workflow

```python
from components.resolution.agent import resolution_node

state = {
    "title": "API timeout error",
    "description": "External API calls timing out",
    "classified_domain": "CIW",
    "priority": "Critical",
    "similar_tickets": [...],
    "assigned_labels": ["Integration-issue", "[TECH] Timeout"],
    "search_metadata": {"avg_similarity": 0.78}
}

result = await resolution_node(state)

plan = result["resolution_plan"]
print(f"Summary: {plan['summary']}")
print(f"Diagnostic Steps: {len(plan['diagnostic_steps'])}")
print(f"Resolution Steps: {len(plan['resolution_steps'])}")
print(f"Confidence: {result['resolution_confidence']:.0%}")
```

## Fallback Plan

When resolution generation fails, a fallback plan is returned:

```python
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
    "additional_considerations": ["Resolution generation failed: {error}"],
    "references": [],
    "total_estimated_time_hours": 0,
    "confidence": 0.0,
    "alternative_approaches": []
}
```

## Configuration

The resolution generation uses different settings than classification:

| Setting | Value | Reason |
|---------|-------|--------|
| Model | `RESOLUTION_MODEL` (gpt-4o) | Better reasoning |
| Temperature | `0.6` | Balanced creativity |
| Max Tokens | Large (8K+) | Complex output |
| Response Format | JSON mode | Structured output |

### Environment Variables

```bash
RESOLUTION_MODEL=gpt-4o
```

## Risk Levels

Steps are categorized by risk:

| Risk Level | Description | Example |
|------------|-------------|---------|
| **low** | Safe, easily reversible | Check logs, restart service |
| **medium** | Moderate impact, reversible | Update config, clear cache |
| **high** | Significant impact, careful rollback needed | Schema change, data migration |

## Performance

- **Historical Analysis**: Instant (string formatting)
- **Resolution Generation**: ~3-5 seconds (complex LLM reasoning)
- **Total Time**: ~4-6 seconds per resolution
- **Cost**: ~$0.05 per resolution (gpt-4o with large context)

## Confidence Scoring

The confidence score reflects:
- **High (>0.8)**: Strong historical patterns, similar symptoms
- **Medium (0.5-0.8)**: Some patterns found, moderate confidence
- **Low (<0.5)**: Limited historical data, novel issue

Factors influencing confidence:
- Average similarity to historical tickets
- Consistency of resolution patterns
- Clarity of symptoms
- Complexity of the issue

## Error Handling

```python
try:
    result = await resolution_node(state)
except Exception as e:
    # Return fallback plan with error context
    return {
        "resolution_plan": fallback_plan,
        "resolution_confidence": 0.0,
        "status": "error",
        "error_message": f"Resolution generation failed: {str(e)}"
    }
```

## Integration with Workflow

The resolution agent is the final step in the pipeline:

```
Classification → Retrieval → Labeling → Resolution
                                            │
                                            ▼
                                    Final Output JSON
```

The resolution plan is saved to `output/ticket_resolution.json` and includes:
- All classification results
- Similar tickets found
- Labels assigned
- Complete resolution plan

## Best Practices

1. **Context Quality**: More similar tickets = better resolution
2. **Label Accuracy**: Correct labels improve resolution relevance
3. **Priority Alignment**: High-priority tickets get faster resolution steps
4. **Validation Steps**: Always include validation for each resolution step
5. **Rollback Procedures**: Include rollback for medium/high risk steps

## Example Output

```json
{
  "summary": "Database connection pool exhaustion requiring configuration increase and monitoring setup. Based on 3 similar historical tickets with 85% average similarity.",
  "diagnostic_steps": [
    {
      "step_number": 1,
      "description": "Check current connection pool status",
      "commands": ["kubectl exec -it mm-alder-pod -- cat /proc/pool_status"],
      "expected_output": "Active connections near max limit",
      "estimated_time_minutes": 5
    }
  ],
  "resolution_steps": [
    {
      "step_number": 1,
      "description": "Increase connection pool size",
      "commands": ["kubectl set env deployment/mm-alder MAX_POOL_SIZE=50"],
      "validation": "Verify pool size increased in pod logs",
      "estimated_time_minutes": 10,
      "risk_level": "medium",
      "rollback_procedure": "kubectl set env deployment/mm-alder MAX_POOL_SIZE=20"
    }
  ],
  "additional_considerations": [
    "Monitor pool usage after change for 24 hours",
    "Consider horizontal scaling if issue persists"
  ],
  "references": ["JIRA-MM-042", "Runbook: MM-ALDER-POOL"],
  "total_estimated_time_hours": 1.5,
  "confidence": 0.85,
  "alternative_approaches": [
    "Implement connection pooler like PgBouncer",
    "Add read replicas for query distribution"
  ]
}
```
