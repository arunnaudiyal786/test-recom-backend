# Backend Architecture Flow - Knowledge Transfer Guide

This document explains the complete data flow and architecture of the Intelligent Ticket Management System backend. It's designed for new developers joining the team.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Directory Structure & Purpose](#2-directory-structure--purpose)
3. [The Two Architectural Patterns](#3-the-two-architectural-patterns)
4. [LangGraph Orchestration](#4-langgraph-orchestration)
5. [Component Anatomy](#5-component-anatomy)
6. [State Management](#6-state-management)
7. [Complete Data Flow](#7-complete-data-flow)
   - [FAISS Search Details](#faiss-search-details)
   - [Hybrid Scoring Mechanism (Deep Dive)](#hybrid-scoring-mechanism-deep-dive)
8. [API Layer](#8-api-layer)
9. [Key Concepts Explained](#9-key-concepts-explained)
10. [Common Modification Patterns](#10-common-modification-patterns)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Next.js)                             │
│                              localhost:3000                                 │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ HTTP/SSE
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           api_server.py (FastAPI)                           │
│                              localhost:8000                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Main Pipeline:     POST /api/process-ticket (SSE streaming)           │ │
│  │  Component APIs:    /v2/retrieval/*, /v2/labeling/*, etc.              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    src/orchestrator/workflow.py                             │
│                        LangGraph StateGraph                                 │
│                                                                             │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│   │ Pattern      │───▶│ Label        │───▶│ Resolution   │───▶│   END    │ │
│   │ Recognition  │    │ Assignment   │    │ Generation   │    │          │ │
│   │ Agent        │    │ Agent        │    │ Agent        │    │          │ │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│         │                   │                   │                          │
│         └───────────────────┴───────────────────┼──────────────────────────┘
│                                                 │ On Error
│                                                 ▼
│                                         ┌──────────────┐
│                                         │ Error Handler│
│                                         │ (Manual      │
│                                         │  Escalation) │
│                                         └──────────────┘
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌──────────┐   ┌──────────┐   ┌──────────┐
             │components│   │  src/    │   │  FAISS   │
             │  folder  │   │vectorstore│   │  Index   │
             └──────────┘   └──────────┘   └──────────┘
```

**Key Insight**: This is a **sequential pipeline**, NOT a conversational agent system. Each agent must complete before the next starts.

---

## 2. Directory Structure & Purpose

```
test-recom-backend/
│
├── api_server.py              # FastAPI app - HTTP entry point
├── main.py                    # CLI entry point for batch processing
│
├── components/                # NEW ARCHITECTURE - LangChain-style agents
│   ├── base/                  # Abstract base classes
│   │   ├── component.py       # BaseComponent ABC
│   │   ├── config.py          # ComponentConfig (Pydantic Settings)
│   │   └── exceptions.py      # Custom exception classes
│   │
│   ├── classification/        # Domain Classification Agent
│   ├── retrieval/             # Pattern Recognition Agent (FAISS search)
│   ├── labeling/              # Label Assignment Agent
│   ├── resolution/            # Resolution Generation Agent
│   └── embedding/             # Utility service (not an agent)
│
├── src/
│   ├── agents/                # OLD ARCHITECTURE - class-based agents (legacy)
│   │   ├── pattern_recognition_agent.py
│   │   ├── label_assignment_agent.py
│   │   └── resolution_generation_agent.py
│   │
│   ├── orchestrator/          # LangGraph workflow definition
│   │   ├── state.py           # TicketWorkflowState TypedDict
│   │   └── workflow.py        # StateGraph construction
│   │
│   ├── prompts/               # LLM prompt templates
│   │   ├── classification_prompts.py
│   │   ├── label_assignment_prompts.py
│   │   └── resolution_generation_prompts.py
│   │
│   ├── vectorstore/           # FAISS index management
│   │   ├── faiss_manager.py   # Index CRUD operations
│   │   ├── embedding_generator.py
│   │   └── data_ingestion.py  # CSV to embeddings to FAISS
│   │
│   ├── models/                # Pydantic models
│   └── utils/                 # Config, helpers, OpenAI client
│
├── data/
│   ├── raw/historical_tickets.csv     # Source data
│   └── faiss_index/                   # Built index + metadata
│       ├── tickets.index              # FAISS binary index
│       └── metadata.json              # Ticket metadata
│
├── input/current_ticket.json  # Sample input for testing
├── output/                    # Processing results
└── scripts/                   # Setup and utility scripts
```

---

## 3. The Two Architectural Patterns

### Why Two Patterns Exist

The codebase has evolved through two architectural approaches:

| Aspect | `src/agents/` (Legacy) | `components/` (Current) |
|--------|----------------------|------------------------|
| Style | Class-based with `__call__` | Functional with `@tool` decorators |
| Framework | Custom implementation | LangChain tools pattern |
| State | Passed via method args | Passed via LangGraph state dict |
| Usage | Direct instantiation | Via LangGraph nodes |
| HTTP Access | Not exposed | Optional router.py |

### The Legacy Pattern (`src/agents/`)

```python
# src/agents/pattern_recognition_agent.py
class PatternRecognitionAgent:
    async def __call__(self, state: TicketState) -> AgentOutput:
        # Direct implementation
        similar_tickets = await self.find_similar_tickets(...)
        return {"similar_tickets": similar_tickets, "status": "success"}

# Usage: Singleton instance
pattern_recognition_agent = PatternRecognitionAgent()
```

### The Current Pattern (`components/`)

```python
# components/retrieval/tools.py
@tool
async def search_similar_tickets(title: str, description: str, ...) -> Dict:
    """Search FAISS for similar tickets."""
    # Implementation
    return {"similar_tickets": results}

# components/retrieval/agent.py
async def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node wrapper."""
    result = await search_similar_tickets.ainvoke({...})
    return {"similar_tickets": result["similar_tickets"], "status": "success"}
```

**Key Insight**: The `components/` architecture separates concerns:
- `tools.py` = Pure business logic (LangChain `@tool` decorated)
- `agent.py` = LangGraph node wrapper (state in, partial state out)

---

## 4. LangGraph Orchestration

### What is LangGraph?

LangGraph is a library for building stateful, multi-step agent workflows. It provides:
- **StateGraph**: A directed graph where nodes are agents
- **Conditional Edges**: Dynamic routing based on state
- **State Reducers**: How state fields get merged

### Workflow Definition (`src/orchestrator/workflow.py`)

```python
from langgraph.graph import StateGraph, END

def build_workflow() -> StateGraph:
    # 1. Create graph with state schema
    workflow = StateGraph(TicketWorkflowState)

    # 2. Add nodes (each node is an agent function)
    workflow.add_node("Pattern Recognition Agent", retrieval_node)
    workflow.add_node("Label Assignment Agent", labeling_node)
    workflow.add_node("Resolution Generation Agent", resolution_node)
    workflow.add_node("Error Handler", error_handler_node)

    # 3. Set entry point
    workflow.set_entry_point("Pattern Recognition Agent")

    # 4. Add conditional edges (routing logic)
    workflow.add_conditional_edges(
        "Pattern Recognition Agent",
        route_after_retrieval,           # Routing function
        {
            "labeling": "Label Assignment Agent",
            "error_handler": "Error Handler"
        }
    )

    # 5. Compile and return
    return workflow.compile()
```

### Routing Functions

```python
def route_after_retrieval(state: TicketWorkflowState) -> str:
    """Route based on success/error status."""
    if state.get("status") == "error":
        return "error_handler"
    return "labeling"
```

**Key Insight**: Routing functions inspect the state and return a string key that maps to the next node. This enables conditional branching.

---

## 5. Component Anatomy

Each component in `components/` follows a standard structure:

```
components/retrieval/
├── __init__.py        # Public exports
├── agent.py           # LangGraph node function (THE WRAPPER)
├── tools.py           # @tool decorated functions (THE LOGIC)
├── service.py         # Alternative class-based interface (optional)
├── router.py          # FastAPI endpoints (optional)
└── models.py          # Pydantic request/response models
```

### The Wrapper Pattern Explained

The **wrapper** is the `agent.py` file. It wraps the tools to create a LangGraph-compatible node:

```python
# components/retrieval/agent.py

async def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for pattern recognition / retrieval.

    THIS IS THE WRAPPER - it:
    1. Extracts needed data from state
    2. Calls the tools (business logic)
    3. Returns a PARTIAL state update
    """
    try:
        # 1. Extract from state
        title = state.get("title", "")
        description = state.get("description", "")

        # 2. Call the tool (defined in tools.py)
        search_result = await search_similar_tickets.ainvoke({
            "title": title,
            "description": description
        })

        # 3. Return PARTIAL state update
        return {
            "similar_tickets": search_result["similar_tickets"],
            "status": "success",
            "current_agent": "retrieval",
            "messages": [{"role": "assistant", "content": "Found X tickets"}]
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }
```

### Tools vs Service vs Agent

| File | Purpose | When to Use |
|------|---------|-------------|
| `tools.py` | Pure business logic with `@tool` decorator | Called by agent.py or directly |
| `service.py` | Class-based interface (inherits BaseComponent) | For HTTP endpoints, dependency injection |
| `agent.py` | LangGraph node wrapper | Called by LangGraph workflow |
| `router.py` | FastAPI endpoints | When component needs HTTP access |

**Key Insight**: `tools.py` contains the actual logic. `agent.py` is just a thin wrapper that handles state management.

---

## 6. State Management

### TicketWorkflowState (`src/orchestrator/state.py`)

```python
class TicketWorkflowState(TypedDict, total=False):
    # Input Fields
    ticket_id: str
    title: str
    description: str
    priority: str
    metadata: Dict

    # Classification Output
    classified_domain: Optional[str]

    # Retrieval Output
    similar_tickets: Optional[List[Dict]]

    # Labeling Output
    assigned_labels: Optional[List[str]]

    # Resolution Output
    resolution_plan: Optional[Dict]

    # Workflow Control
    status: Literal["processing", "success", "error", "failed"]
    current_agent: str

    # Message Accumulation (uses reducer)
    messages: Annotated[List[Dict], operator.add]
```

### Key State Patterns

1. **Partial Updates**: Agents return only the fields they modify, not the entire state
2. **`total=False`**: All fields are optional (allows partial updates)
3. **`Annotated[..., operator.add]`**: Messages accumulate across agents

### State Flow Example

```
Initial State:
{
  "ticket_id": "T-123",
  "title": "DB Connection Error",
  "description": "...",
  "status": "processing"
}
     │
     ▼ Pattern Recognition Agent
{
  "similar_tickets": [...],      # Added
  "status": "success",           # Updated
  "current_agent": "retrieval"   # Updated
}
     │
     ▼ Label Assignment Agent
{
  "assigned_labels": ["DB", "Connection"],  # Added
  "status": "success"                       # Confirmed
}
     │
     ▼ Resolution Generation Agent
{
  "resolution_plan": {...},      # Added
  "status": "success"            # Final
}
```

---

## 7. Complete Data Flow

### Entry Point: HTTP Request

```
POST /api/process-ticket
{
  "ticket_id": "T-123",
  "title": "Database connection timeouts",
  "description": "Users experiencing slow queries...",
  "priority": "High",
  "metadata": {}
}
```

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. api_server.py receives POST request                                      │
│    └─▶ Creates initial state dict                                           │
│    └─▶ Calls get_workflow() to get compiled LangGraph                       │
│    └─▶ Starts async iteration: workflow.astream(initial_state)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Pattern Recognition Agent (components/retrieval/agent.py)                │
│    └─▶ retrieval_node(state) called by LangGraph                            │
│    └─▶ Calls search_similar_tickets tool (components/retrieval/tools.py)    │
│        └─▶ Generates embedding via OpenAI                                   │
│        └─▶ Searches FAISS index (src/vectorstore/faiss_manager.py)          │
│        └─▶ Returns top K similar tickets                                    │
│    └─▶ Calls apply_hybrid_scoring tool                                      │
│        └─▶ Combines vector similarity (70%) + metadata (30%)                │
│    └─▶ Returns partial state: {similar_tickets: [...], status: "success"}   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (state merged)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Label Assignment Agent (components/labeling/agent.py)                    │
│    └─▶ labeling_node(state) called by LangGraph                             │
│    └─▶ Calls extract_candidate_labels tool                                  │
│        └─▶ Extracts unique labels from similar_tickets                      │
│    └─▶ Runs 3 PARALLEL tasks via asyncio.gather:                            │
│        ├─▶ check_historical_labels (binary classifier per label)            │
│        ├─▶ generate_business_labels (AI-generated)                          │
│        └─▶ generate_technical_labels (AI-generated)                         │
│    └─▶ Returns partial state: {assigned_labels: [...], status: "success"}   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (state merged)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Resolution Generation Agent (components/resolution/agent.py)             │
│    └─▶ resolution_node(state) called by LangGraph                           │
│    └─▶ Calls generate_resolution_plan tool                                  │
│        └─▶ Constructs prompt with ticket + similar tickets + labels         │
│        └─▶ Calls OpenAI with Chain-of-Thought prompt                        │
│        └─▶ Returns structured JSON resolution plan                          │
│    └─▶ Returns partial state: {resolution_plan: {...}, status: "success"}   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (state merged)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. Workflow Complete                                                        │
│    └─▶ Final state saved to output/ticket_resolution.json                   │
│    └─▶ CSV export created                                                   │
│    └─▶ SSE stream sends completion event to frontend                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FAISS Search Details

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          FAISS SIMILARITY SEARCH                             │
│                                                                              │
│  Query Ticket                     FAISS Index                                │
│  ┌─────────────┐                 ┌─────────────────────────────┐            │
│  │ Title +     │  ─▶ Embedding ─▶│ Historical Tickets (100+)   │            │
│  │ Description │     Generator   │ Each with 3072-dim vector   │            │
│  └─────────────┘                 └─────────────────────────────┘            │
│        │                                    │                                │
│        ▼                                    ▼                                │
│  [3072-dim vector]              [Cosine Similarity Search]                   │
│        │                                    │                                │
│        └────────────────┬───────────────────┘                                │
│                         ▼                                                    │
│                 ┌───────────────┐                                            │
│                 │ Top K Results │  (filtered by domain if classified)       │
│                 │ + Metadata    │                                            │
│                 └───────────────┘                                            │
│                         │                                                    │
│                         ▼                                                    │
│                 ┌───────────────┐                                            │
│                 │ Hybrid Scoring│                                            │
│                 │ 70% vector    │                                            │
│                 │ 30% metadata  │                                            │
│                 └───────────────┘                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Scoring Mechanism (Deep Dive)

The hybrid scoring mechanism is **critical** to understanding how the system ranks similar tickets. It combines semantic similarity (from vector embeddings) with business relevance (from ticket metadata) to produce a final ranking that better matches real-world usefulness.

#### Why Hybrid Scoring?

Pure vector similarity finds semantically similar text, but doesn't account for:
- **Priority alignment** - A Critical-priority similar ticket is more relevant than a Low-priority one
- **Resolution efficiency** - Tickets resolved quickly likely had clearer solutions
- **Business context** - Not all similar text means similar business impact

Hybrid scoring addresses this by blending semantic and metadata signals.

#### The Formula

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID SCORING FORMULA                              │
│                                                                             │
│   hybrid_score = (vector_weight × vector_similarity)                        │
│                + (metadata_weight × metadata_score)                         │
│                                                                             │
│   Default: hybrid_score = (0.7 × vector_similarity) + (0.3 × metadata_score)│
│                                                                             │
│   Where:                                                                    │
│     metadata_score = (priority_score × 0.6) + (time_score × 0.4)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Component Breakdown

**1. Vector Similarity (70% weight by default)**

```
Vector similarity comes directly from FAISS cosine similarity search.

- Range: 0.0 to 1.0 (normalized via L2 normalization)
- Source: Inner product of query embedding vs stored embeddings
- Embedding model: text-embedding-3-large (3072 dimensions)

Higher score = More semantically similar ticket text
```

**2. Metadata Score (30% weight by default)**

The metadata score is itself a weighted combination:

```
metadata_score = (priority_score × 0.6) + (time_score × 0.4)
                       ↓                        ↓
               60% weight               40% weight
```

**2a. Priority Score (60% of metadata)**

Maps ticket priority to a numerical value:

| Priority | Score | Reasoning |
|----------|-------|-----------|
| Critical | 1.0   | Highest urgency, most business-critical |
| High     | 0.8   | Important but not emergency |
| Medium   | 0.5   | Standard priority (baseline) |
| Low      | 0.3   | Lower urgency, less impactful |

```python
# From components/retrieval/tools.py
priority_scores = {
    "Critical": 1.0,
    "High": 0.8,
    "Medium": 0.5,
    "Low": 0.3
}
```

**Why prioritize Critical tickets?** A similar ticket that was treated as Critical likely had more thorough investigation, better documentation, and more senior engineer involvement—making its resolution more valuable as a reference.

**2b. Time Score (40% of metadata)**

Rewards tickets that were resolved quickly:

```
time_score = max(0, 1 - (resolution_time_hours / normalization_hours))

Default normalization_hours = 100

Examples:
- Resolved in 2 hours:  1 - (2/100)   = 0.98 (excellent)
- Resolved in 24 hours: 1 - (24/100)  = 0.76 (good)
- Resolved in 50 hours: 1 - (50/100)  = 0.50 (average)
- Resolved in 100+ hours: max(0, ...) = 0.00 (no bonus)
```

**Why favor fast resolutions?** Quickly resolved tickets often had:
- Clear root cause
- Well-documented solution
- Reusable fix patterns

#### Visual Score Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE SCORE CALCULATION                                │
│                                                                             │
│  Ticket A: "Database connection timeout"                                    │
│  ├── vector_similarity = 0.85 (high semantic match)                        │
│  ├── priority = "High" → priority_score = 0.8                              │
│  └── resolution_time = 8 hours → time_score = 1 - (8/100) = 0.92           │
│                                                                             │
│  Step 1: Calculate metadata_score                                           │
│          = (0.8 × 0.6) + (0.92 × 0.4)                                      │
│          = 0.48 + 0.368                                                     │
│          = 0.848                                                            │
│                                                                             │
│  Step 2: Calculate hybrid_score                                             │
│          = (0.7 × 0.85) + (0.3 × 0.848)                                    │
│          = 0.595 + 0.254                                                    │
│          = 0.849                                                            │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Ticket B: "DB timeout during peak load"                                    │
│  ├── vector_similarity = 0.90 (very high semantic match)                   │
│  ├── priority = "Low" → priority_score = 0.3                               │
│  └── resolution_time = 72 hours → time_score = 1 - (72/100) = 0.28         │
│                                                                             │
│  Step 1: metadata_score = (0.3 × 0.6) + (0.28 × 0.4) = 0.292               │
│  Step 2: hybrid_score = (0.7 × 0.90) + (0.3 × 0.292) = 0.718               │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Result: Ticket A (0.849) ranks HIGHER than Ticket B (0.718)                │
│          despite Ticket B having higher vector similarity!                  │
│                                                                             │
│  This is the power of hybrid scoring: it prevents slow, low-priority       │
│  tickets from outranking faster, higher-priority ones just because         │
│  of slightly better text matching.                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Configuration Options

The system supports tunable parameters via `RetrievalConfig` (`src/models/retrieval_config.py`):

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `vector_weight` | 0.7 | 0.0-1.0 | Weight for semantic similarity |
| `metadata_weight` | 0.3 | 0.0-1.0 | Weight for metadata relevance |
| `top_k` | 20 | 5-50 | Number of similar tickets to retrieve |
| `time_normalization_hours` | 100.0 | 1-500 | Reference hours for time scoring |
| `priority_weights.Critical` | 1.0 | 0.0-1.0 | Score for Critical priority |
| `priority_weights.High` | 0.8 | 0.0-1.0 | Score for High priority |
| `priority_weights.Medium` | 0.5 | 0.0-1.0 | Score for Medium priority |
| `priority_weights.Low` | 0.3 | 0.0-1.0 | Score for Low priority |

**Tuning via UI:** The frontend's "Retrieval Engine" page allows real-time adjustment of these weights with preview functionality.

#### When to Adjust Weights

| Scenario | Recommendation |
|----------|----------------|
| Text matching is most important | Increase `vector_weight` to 0.85+ |
| Business priority matters most | Decrease `vector_weight` to 0.5, increase Critical/High weights |
| Fast resolutions are critical | Increase time portion in metadata (requires code change) |
| Domain has many similar tickets | Increase `top_k` to cast wider net |

#### Code Locations

| Purpose | File Path |
|---------|-----------|
| Tool implementation | `components/retrieval/tools.py:141` - `apply_hybrid_scoring()` |
| Legacy class method | `src/agents/pattern_recognition_agent.py:223` - `apply_hybrid_scoring()` |
| Configurable version | `src/agents/pattern_recognition_agent.py:115` - `apply_hybrid_scoring_with_config()` |
| Configuration model | `src/models/retrieval_config.py` - `RetrievalConfig` class |

#### Output Fields Added to Tickets

After hybrid scoring, each ticket in the results contains:

```python
{
    "ticket_id": "JIRA-MM-042",
    "title": "...",
    "description": "...",
    # Scoring fields added by hybrid scoring:
    "similarity_score": 0.849,      # Final hybrid score (used for ranking)
    "vector_similarity": 0.85,      # Raw FAISS cosine similarity
    "metadata_score": 0.848,        # Combined priority + time score
    # Original metadata:
    "priority": "High",
    "resolution_time_hours": 8,
    ...
}
```

---

## 8. API Layer

### Main Pipeline Endpoint

```python
# api_server.py
@app.post("/api/process-ticket")
async def process_ticket(ticket: TicketInput):
    """SSE stream of agent processing."""
    return StreamingResponse(
        stream_agent_updates(ticket),
        media_type="text/event-stream"
    )
```

### Component Endpoints (v2)

Each component can expose its own HTTP endpoints:

```python
# components/retrieval/router.py
router = APIRouter(prefix="/retrieval", tags=["Retrieval"])

@router.post("/search")
async def search_similar_tickets(request: RetrievalRequest):
    service = get_service()
    return await service.process(request)
```

**Mounted in api_server.py:**
```python
app.include_router(retrieval_router, prefix="/v2")
# Creates: POST /v2/retrieval/search
```

### Endpoint Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/process-ticket` | POST | Full pipeline (SSE streaming) |
| `/api/output` | GET | Retrieve last result |
| `/api/preview-search` | POST | Test search params |
| `/v2/retrieval/search` | POST | Direct FAISS search |
| `/v2/labeling/assign` | POST | Direct label assignment |
| `/v2/classification/classify` | POST | Direct classification |

---

## 9. Key Concepts Explained

### What is a "Wrapper"?

A **wrapper** is code that adapts one interface to another. In this codebase:

```
LangGraph expects:  async def node(state: Dict) -> Dict

Tools provide:      @tool decorated functions with specific args

Wrapper bridges:    Extracts args from state -> calls tool -> returns partial state
```

### What is `@tool`?

LangChain's `@tool` decorator marks a function as a reusable tool:

```python
from langchain_core.tools import tool

@tool
async def search_similar_tickets(title: str, description: str) -> Dict:
    """Search FAISS for similar tickets."""  # Docstring = tool description
    # Implementation
    return {"similar_tickets": results}

# Usage:
result = await search_similar_tickets.ainvoke({"title": "...", "description": "..."})
```

### What is `asyncio.gather`?

Runs multiple async tasks **in parallel**:

```python
# Instead of:
result1 = await task1()
result2 = await task2()  # Waits for task1 first

# Do this:
result1, result2 = await asyncio.gather(task1(), task2())  # Run simultaneously
```

Used extensively in label assignment for parallel LLM calls.

### What is Singleton Pattern?

A **singleton** ensures only one instance exists:

```python
_workflow = None

def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()  # Created once
    return _workflow
```

Used for FAISS manager, workflow, and service instances to avoid re-initialization.

---

## 10. Common Modification Patterns

### Adding a New Agent

1. Create component folder:
   ```
   components/my_agent/
   ├── __init__.py
   ├── tools.py      # Business logic with @tool
   ├── agent.py      # LangGraph node wrapper
   └── models.py     # Pydantic models
   ```

2. Add node to workflow (`src/orchestrator/workflow.py`):
   ```python
   from components.my_agent.agent import my_agent_node

   workflow.add_node("My Agent", my_agent_node)
   workflow.add_conditional_edges(...)
   ```

3. Update state if needed (`src/orchestrator/state.py`):
   ```python
   class TicketWorkflowState(TypedDict, total=False):
       my_new_field: Optional[str]
   ```

### Adding a New Tool to Existing Agent

1. Add function to `tools.py`:
   ```python
   @tool
   def my_new_tool(param: str) -> Dict:
       """Tool description."""
       return {"result": "..."}
   ```

2. Call from `agent.py`:
   ```python
   result = await my_new_tool.ainvoke({"param": state["some_field"]})
   ```

### Modifying Hybrid Scoring Weights

Edit `components/retrieval/tools.py`:

```python
@tool
def apply_hybrid_scoring(
    similar_tickets: List[Dict],
    vector_weight: float = 0.7,   # Change here
    metadata_weight: float = 0.3  # Change here
) -> List[Dict]:
```

### Adding New Labels

1. Edit `src/prompts/label_assignment_prompts.py`
2. Add to `LABEL_CRITERIA` dict
3. Rebuild FAISS if labels exist in historical data

---

## Quick Reference

### Running the System

```bash
# Start both frontend + backend
./start_dev.sh

# Backend only
cd test-recom-backend
python3 -m uvicorn api_server:app --reload --port 8000

# Process single ticket (CLI)
python3 main.py
```

### Key Files to Know

| File | Purpose |
|------|---------|
| `api_server.py` | HTTP entry point |
| `src/orchestrator/workflow.py` | LangGraph definition |
| `src/orchestrator/state.py` | State schema |
| `components/*/agent.py` | Agent wrappers |
| `components/*/tools.py` | Business logic |
| `src/vectorstore/faiss_manager.py` | FAISS operations |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for all LLM calls |
| `TOP_K_SIMILAR_TICKETS` | Number of similar tickets (default: 20) |
| `CLASSIFICATION_MODEL` | Model for classification |
| `RESOLUTION_MODEL` | Model for resolution generation |

---

## Summary

1. **Components (`components/`)** contain the modern LangChain-style agents
2. **Wrappers (`agent.py`)** bridge LangGraph state and tool parameters
3. **Tools (`tools.py`)** contain actual business logic with `@tool` decorators
4. **State** flows through the pipeline, each agent adding its output
5. **LangGraph** orchestrates the sequential execution with error handling
6. **FAISS** provides fast vector similarity search for historical tickets

The system is designed for **maintainability**: add new agents by creating component folders, modify behavior by editing tools, adjust flow by updating the workflow graph.
