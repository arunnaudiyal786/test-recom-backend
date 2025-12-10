# Backend Architecture Flow - Knowledge Transfer Guide

This document explains the complete data flow and architecture of the Intelligent Ticket Management System backend. It's designed for new developers joining the team.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Directory Structure & Purpose](#2-directory-structure--purpose)
3. [Component Architecture Pattern](#3-component-architecture-pattern)
4. [LangGraph Orchestration](#4-langgraph-orchestration)
5. [Component Anatomy](#5-component-anatomy)
6. [State Management](#6-state-management)
7. [Complete Data Flow](#7-complete-data-flow)
   - [FAISS Search Details](#faiss-search-details)
   - [Hybrid Scoring Mechanism (Deep Dive)](#hybrid-scoring-mechanism-deep-dive)
   - [Novelty Detection Details](#novelty-detection-details)
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
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│   │ Historical   │──▶│ Label        │──▶│ Novelty      │──▶│ Resolution │  │
│   │ Match        │   │ Assignment   │   │ Detection    │   │ Generation │  │
│   │ Agent        │   │ Agent        │   │ Agent        │   │ Agent      │  │
│   └──────────────┘   └──────────────┘   └──────────────┘   └────────────┘  │
│         │                   │                  │                  │         │
│         └───────────────────┴──────────────────┴──────────────────┼─────────┘
│                                                                   │ On Error
│                                                                   ▼
│                                                          ┌──────────────┐
│                                                          │ Error Handler│
│                                                          │ (Manual      │
│                                                          │  Escalation) │
│                                                          └──────────────┘
│                                                                   │
│                                                                   ▼
│                                                          ┌──────────────┐
│                                                          │     END      │
│                                                          └──────────────┘
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

**Note**: Domain Classification Agent can be enabled via `SKIP_DOMAIN_CLASSIFICATION = False` in `src/orchestrator/workflow.py`. When enabled, it runs before Historical Match.

---

## 2. Directory Structure & Purpose

```
test-recom-backend/
│
├── api_server.py              # FastAPI app - HTTP entry point
├── main.py                    # CLI entry point for batch processing
│
├── config/                    # Configuration files
│   ├── config.py              # Config class with all settings
│   ├── schema_config.yaml     # Domain/label definitions for UI
│   └── search_config.json     # Saved search parameters (auto-generated)
│
├── components/                # LangChain-style agent components
│   ├── base/                  # Abstract base classes
│   │   ├── component.py       # BaseComponent ABC
│   │   ├── config.py          # ComponentConfig (Pydantic Settings)
│   │   └── exceptions.py      # Custom exception classes
│   │
│   ├── classification/        # Domain Classification Agent (optional, skipped by default)
│   ├── retrieval/             # Historical Match Agent (FAISS search)
│   ├── labeling/              # Label Assignment Agent (hybrid semantic + LLM)
│   │   ├── agent.py           # LangGraph node wrapper
│   │   ├── tools.py           # @tool functions with create_agent for classification
│   │   ├── prompts.py         # Classification and label generation prompts
│   │   ├── service.py         # CategoryTaxonomy for taxonomy management
│   │   └── category_embeddings.py  # Pre-computed category embeddings
│   ├── novelty/               # Novelty Detection Agent (multi-signal analysis)
│   │   ├── agent.py           # LangGraph node wrapper
│   │   ├── tools.py           # Multi-signal novelty detection logic
│   │   └── novelty.md         # Comprehensive documentation
│   ├── resolution/            # Resolution Generation Agent
│   └── embedding/             # Utility service (not an agent)
│
├── src/
│   ├── orchestrator/          # LangGraph workflow definition
│   │   ├── state.py           # TicketWorkflowState TypedDict
│   │   └── workflow.py        # StateGraph construction
│   │
│   ├── vectorstore/           # FAISS index management
│   │   ├── faiss_manager.py   # Index CRUD operations
│   │   ├── embedding_generator.py
│   │   └── data_ingestion.py  # CSV to embeddings to FAISS
│   │
│   ├── models/                # Pydantic models
│   └── utils/                 # Helpers, OpenAI client, session manager
│
├── data/
│   ├── raw/test_plan_historical.csv   # Source data (default CSV file)
│   ├── faiss_index/                   # Built index + metadata
│   │   ├── tickets.index              # FAISS binary index
│   │   └── metadata.json              # Ticket metadata
│   └── metadata/                      # Category taxonomy & embeddings
│       ├── categories.json            # Category definitions
│       └── category_embeddings.json   # Pre-computed embeddings for semantic search
│
├── input/current_ticket.json  # Sample input for testing
├── output/                    # Processing results (session-based)
│   ├── latest/                # Symlink to most recent session
│   └── DDMMYYYYHHMM_xxxxx/    # Individual session directories
└── scripts/                   # Setup and utility scripts
```

---

## 3. Component Architecture Pattern

### LangChain Tools + LangGraph Nodes

The codebase uses a clean separation of concerns:

| File | Purpose | Description |
|------|---------|-------------|
| `tools.py` | Business logic | Pure functions with `@tool` decorator |
| `agent.py` | State management | LangGraph node wrapper (state in, partial state out) |
| `service.py` | HTTP interface | Optional class-based interface for API endpoints |
| `router.py` | API endpoints | Optional FastAPI router |

### The Pattern

```python
# components/retrieval/tools.py - Business Logic
@tool
async def search_similar_tickets(title: str, description: str, ...) -> Dict:
    """Search FAISS for similar tickets."""
    # Implementation
    return {"similar_tickets": results}

# components/retrieval/agent.py - LangGraph Node Wrapper
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
    # Note: Classification node is optional (controlled by SKIP_DOMAIN_CLASSIFICATION)
    if not SKIP_DOMAIN_CLASSIFICATION:
        workflow.add_node("Domain Classification Agent", classification_node)
    workflow.add_node("Historical Match Agent", retrieval_node)
    workflow.add_node("Label Assignment Agent", labeling_node)
    workflow.add_node("Novelty Detection Agent", novelty_node)
    workflow.add_node("Resolution Generation Agent", resolution_node)
    workflow.add_node("Error Handler", error_handler_node)

    # 3. Set entry point (skips classification by default)
    if SKIP_DOMAIN_CLASSIFICATION:
        workflow.set_entry_point("Historical Match Agent")
    else:
        workflow.set_entry_point("Domain Classification Agent")

    # 4. Add conditional edges (routing logic)
    workflow.add_conditional_edges(
        "Historical Match Agent",
        route_after_retrieval,
        {"labeling": "Label Assignment Agent", "error_handler": "Error Handler"}
    )

    workflow.add_conditional_edges(
        "Label Assignment Agent",
        route_after_labeling,
        {"novelty": "Novelty Detection Agent", "error_handler": "Error Handler"}
    )

    workflow.add_conditional_edges(
        "Novelty Detection Agent",
        route_after_novelty,
        {"resolution": "Resolution Generation Agent", "error_handler": "Error Handler"}
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
    LangGraph node for historical matching / retrieval.

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

    # Session Management
    session_id: Optional[str]  # Unique session ID (YYYYMMDD_HHMMSS_xxxxx)
    search_config: Optional[Dict]  # Custom search config from UI

    # Classification Output (optional - can be skipped)
    classified_domain: Optional[str]
    classification_confidence: Optional[float]
    classification_reasoning: Optional[str]
    extracted_keywords: Optional[List[str]]

    # Retrieval Output
    similar_tickets: Optional[List[Dict]]
    similarity_scores: Optional[List[float]]
    search_metadata: Optional[Dict]

    # Labeling Output (three-tier system)
    category_labels: Optional[List[Dict]]   # From predefined taxonomy
    business_labels: Optional[List[Dict]]   # AI-generated business labels
    technical_labels: Optional[List[Dict]]  # AI-generated technical labels
    assigned_labels: Optional[List[str]]    # Combined labels for display
    ticket_embedding: Optional[List[float]] # For novelty detection
    all_category_scores: Optional[List[Dict]]  # For entropy calculation

    # Novelty Detection Output
    novelty_detected: Optional[bool]
    novelty_score: Optional[float]
    novelty_signals: Optional[Dict]
    novelty_recommendation: Optional[str]  # "proceed" | "flag_for_review" | "escalate"
    novelty_reasoning: Optional[str]

    # Resolution Output
    resolution_plan: Optional[Dict]
    resolution_confidence: Optional[float]

    # Prompt Transparency (for UI display)
    label_assignment_prompts: Optional[Dict[str, str]]
    resolution_generation_prompt: Optional[str]

    # Workflow Control
    status: Literal["processing", "success", "error", "failed"]
    current_agent: str
    error_message: Optional[str]

    # Message Accumulation (uses reducer)
    messages: Annotated[List[Dict], operator.add]

    # Overall Metrics
    overall_confidence: Optional[float]
    processing_time_seconds: Optional[float]
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
     ▼ Historical Match Agent
{
  "similar_tickets": [...],      # Added
  "status": "success",           # Updated
  "current_agent": "retrieval"   # Updated
}
     │
     ▼ Label Assignment Agent (uses similar_tickets[:5] for context)
{
  "category_labels": [...],                 # Added (from taxonomy)
  "business_labels": [...],                 # Added (AI-generated with similar tickets context)
  "technical_labels": [...],                # Added (AI-generated with similar tickets context)
  "assigned_labels": ["[CAT]...", "[BIZ]...", "[TECH]..."],  # Combined for display
  "ticket_embedding": [...],                # Added (passed to novelty detection)
  "all_category_scores": [...],             # Added (passed to novelty detection)
  "status": "success"
}
     │
     ▼ Novelty Detection Agent
{
  "novelty_detected": false,     # Added
  "novelty_score": 0.25,         # Added
  "novelty_recommendation": "proceed",  # Added
  "novelty_signals": {...},      # Added (signal details)
  "status": "success"
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
│ 2. Historical Match Agent (components/retrieval/agent.py)                   │
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
│    └─▶ Extracts top 5 similar_tickets from state (from Historical Match)    │
│    └─▶ Runs HYBRID SEMANTIC + LLM classification pipeline:                  │
│                                                                             │
│    STEP 1: Generate ticket embedding (1 OpenAI API call)                    │
│        └─▶ Uses text-embedding-3-large (3072 dimensions)                    │
│                                                                             │
│    STEP 2: Semantic pre-filtering (no API call)                             │
│        └─▶ Computes cosine similarity to ALL category embeddings            │
│        └─▶ Pre-computed embeddings in data/metadata/category_embeddings.json│
│                                                                             │
│    STEP 3: Select TOP-5 candidates above similarity threshold (0.3)         │
│                                                                             │
│    STEP 4: Run PARALLEL binary classifiers using langchain.agents.create_agent│
│        └─▶ Uses submit_classification_result tool for structured output     │
│        └─▶ 5 parallel agent invocations for top candidates                  │
│                                                                             │
│    STEP 5: Compute ENSEMBLE scores: 40% semantic + 60% LLM confidence       │
│                                                                             │
│    STEP 6: Filter by threshold and limit to max 3 categories                │
│                                                                             │
│    └─▶ Also runs in PARALLEL via asyncio.gather:                            │
│        ├─▶ generate_business_labels (AI-generated, using create_agent)      │
│        │     └─▶ Includes top 5 similar tickets for historical context      │
│        └─▶ generate_technical_labels (AI-generated, using create_agent)     │
│              └─▶ Includes top 5 similar tickets for historical context      │
│                                                                             │
│    └─▶ Passes ticket_embedding and all_category_scores to novelty detection │
│    └─▶ Returns partial state: {category_labels, business_labels,            │
│                                technical_labels, ticket_embedding, ...}     │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (state merged)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Novelty Detection Agent (components/novelty/agent.py)                    │
│    └─▶ novelty_node(state) called by LangGraph                              │
│    └─▶ Extracts ticket_embedding and category scores from state             │
│    └─▶ Calls detect_novelty tool (no LLM calls - pure computation)          │
│        └─▶ Signal 1: Check if max confidence < 0.5 (40% weight)             │
│        └─▶ Signal 2: Calculate entropy of confidence distribution (30%)     │
│        └─▶ Signal 3: Measure embedding distance to centroids (30%)          │
│        └─▶ Combine signals: novelty_score = weighted average                │
│        └─▶ Decision: is_novel = (max_conf < 0.5) OR (score > 0.6)           │
│    └─▶ Returns partial state: {novelty_detected: bool, novelty_score: ...}  │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (state merged)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. Resolution Generation Agent (components/resolution/agent.py)             │
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
│ 6. Workflow Complete                                                        │
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
| Tool implementation | `components/retrieval/tools.py` - `apply_hybrid_scoring()` |
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

### Novelty Detection Details

The Novelty Detection Agent analyzes whether a ticket represents a **novel category** that doesn't exist in the current taxonomy. It runs after Label Assignment and before Resolution Generation.

#### Why Novelty Detection?

- **Taxonomy Evolution**: New types of issues emerge over time
- **Prevent Misclassification**: Avoid forcing tickets into wrong categories
- **Alert for Review**: Flag tickets that need human attention
- **Improve Recommendations**: When novelty is detected, resolution can be adjusted

#### Three-Signal Detection System

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       NOVELTY DETECTION SIGNALS                              │
│                                                                              │
│  Signal 1: Maximum Confidence Score (40% weight)                             │
│  ├── If best category match confidence < 0.5, fires                          │
│  └── Detects: Weak category matches                                          │
│                                                                              │
│  Signal 2: Confidence Distribution Entropy (30% weight)                      │
│  ├── If normalized Shannon entropy > 0.7, fires                              │
│  └── Detects: Uncertainty spread evenly across all categories                │
│                                                                              │
│  Signal 3: Embedding Distance to Centroids (30% weight)                      │
│  ├── If min cosine distance to any centroid > 0.4, fires                     │
│  └── Detects: Semantic novelty (ticket far from all known categories)        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Decision Logic

```python
novelty_score = (0.4 * signal_1) + (0.3 * signal_2) + (0.3 * signal_3)
is_novel = (max_confidence < 0.5) OR (novelty_score > 0.6)
```

#### Recommendations

| Condition | Recommendation | Action |
|-----------|----------------|--------|
| `novelty_score > 0.8` | `"escalate"` | Requires immediate taxonomy review |
| Low confidence only | `"flag_for_review"` | May need new category |
| Multiple signals | `"flag_for_review"` | Review category taxonomy |
| Not novel | `"proceed"` | Continue to resolution |

#### Data Flow

```
From Label Assignment Agent:
├── ticket_embedding (3072-dim vector)
├── all_category_scores ([{category_id, score}, ...])
└── category_labels ([{id, name, confidence}, ...])
          │
          ▼
Novelty Detection Agent (NO LLM CALLS - pure computation)
├── Compute Signal 1: max_confidence analysis
├── Compute Signal 2: entropy calculation
├── Compute Signal 3: centroid distance (uses pre-computed embeddings)
├── Calculate weighted novelty_score
└── Determine recommendation
          │
          ▼
To Resolution Generation Agent:
├── novelty_detected: bool
├── novelty_score: float (0-1)
├── novelty_signals: {...signal details...}
├── novelty_recommendation: "proceed" | "flag_for_review" | "escalate"
└── novelty_reasoning: "Human-readable explanation"
```

#### Configuration (`config/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NOVELTY_SIGNAL1_THRESHOLD` | 0.5 | Max confidence below this = signal fires |
| `NOVELTY_SIGNAL2_THRESHOLD` | 0.7 | Entropy above this = signal fires |
| `NOVELTY_SIGNAL3_THRESHOLD` | 0.4 | Distance above this = signal fires |
| `NOVELTY_SIGNAL1_WEIGHT` | 0.4 | Weight for Signal 1 (40%) |
| `NOVELTY_SIGNAL2_WEIGHT` | 0.3 | Weight for Signal 2 (30%) |
| `NOVELTY_SIGNAL3_WEIGHT` | 0.3 | Weight for Signal 3 (30%) |
| `NOVELTY_SCORE_THRESHOLD` | 0.6 | Score above this = novel |

#### Code Locations

| Purpose | File Path |
|---------|-----------|
| LangGraph node | `components/novelty/agent.py` - `novelty_node()` |
| Detection logic | `components/novelty/tools.py` - `detect_novelty()` |
| Category embeddings | `components/labeling/category_embeddings.py` - `CategoryEmbeddings` |
| Pre-computed embeddings | `data/metadata/category_embeddings.json` |

#### Key Characteristic: Zero LLM Calls

The Novelty Detection Agent performs **pure mathematical computation**:
- No OpenAI API calls
- Processing time < 50ms
- Uses pre-computed category embeddings loaded once at startup
- Non-blocking errors (failures default to no novelty, don't fail pipeline)

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

1. Edit `components/labeling/prompts.py`
2. Add to label criteria or category definitions
3. If adding categories, update `data/metadata/categories.json`
4. Regenerate category embeddings: `python scripts/generate_category_embeddings.py`
5. Rebuild FAISS if labels exist in historical data: `python scripts/setup_vectorstore.py`

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

### Configuration (config/config.py)

**Note**: Configuration is now centralized in `config/config.py` as a class with attributes, NOT environment variables. Only `OPENAI_API_KEY` comes from environment.

| Attribute | Default | Purpose |
|-----------|---------|---------|
| `OPENAI_API_KEY` | (env) | Required for all LLM calls |
| `CLASSIFICATION_MODEL` | `gpt-4o` | Model for classification |
| `RESOLUTION_MODEL` | `gpt-4o` | Model for resolution generation |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Model for embeddings (3072 dims) |
| `TOP_K_SIMILAR_TICKETS` | `10` | Number of similar tickets to retrieve |
| `CLASSIFICATION_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for classification |
| `LABEL_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for label assignment |
| `SEMANTIC_TOP_K_CANDIDATES` | `5` | Candidates from semantic pre-filtering |
| `ENSEMBLE_SEMANTIC_WEIGHT` | `0.4` | Weight for semantic similarity in labeling |
| `ENSEMBLE_LLM_WEIGHT` | `0.6` | Weight for LLM confidence in labeling |
| `LABEL_SIMILAR_TICKETS_COUNT` | `5` | Similar tickets to include in label prompts |
| `NOVELTY_SCORE_THRESHOLD` | `0.6` | Threshold for novelty detection |

---

## Summary

1. **Components (`components/`)** contain the modern LangChain-style agents
2. **Wrappers (`agent.py`)** bridge LangGraph state and tool parameters
3. **Tools (`tools.py`)** contain actual business logic with `@tool` decorators
4. **Agents** use `langchain.agents.create_agent` for LLM-powered classification
5. **State** flows through the pipeline, each agent adding its output
6. **LangGraph** orchestrates the sequential execution with error handling
7. **FAISS** provides fast vector similarity search for historical tickets

### Agent Pipeline (Default Configuration)

```
Historical Match → Label Assignment → Novelty Detection → Resolution Generation
    (FAISS)         (hybrid semantic    (multi-signal,     (Chain-of-Thought)
                     + LLM ensemble)     no LLM calls)
```

| Agent | Purpose | LLM Calls | Uses Similar Tickets |
|-------|---------|-----------|---------------------|
| Historical Match | FAISS similarity search + hybrid scoring | 1 (embedding only) | Produces them |
| Label Assignment | Hybrid semantic + LLM category classification | 5-8 (parallel agents) | Top 5 for business/technical labels |
| Novelty Detection | Multi-signal analysis (confidence, entropy, distance) | 0 (pure computation) | No |
| Resolution Generation | Chain-of-Thought resolution plan | 1 (gpt-4o) | Yes (all retrieved) |

**Note**: Domain Classification Agent can be optionally enabled by setting `SKIP_DOMAIN_CLASSIFICATION = False` in `src/orchestrator/workflow.py`, which adds it before Historical Match.

### Key Architectural Pattern: `langchain.agents.create_agent`

The Label Assignment Agent uses `langchain.agents.create_agent` to create LangChain agents that:
- Have a system prompt defining their role
- Use specific tools (`submit_classification_result`, `submit_business_labels`, etc.)
- Return structured JSON output via tool calls

```python
# Example from components/labeling/tools.py
agent = create_agent(
    model="gpt-4o",
    tools=[submit_classification_result],  # Tool for structured output
    system_prompt=BINARY_CLASSIFICATION_SYSTEM_PROMPT
)
```

The system is designed for **maintainability**: add new agents by creating component folders, modify behavior by editing tools, adjust flow by updating the workflow graph.
