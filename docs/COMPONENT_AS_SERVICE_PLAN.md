# Component as a Service (CaaS) Architecture Plan

> **⚠️ NOTE: This document describes the original planning phase.**
>
> The architecture has been **refactored to use LangChain agents with LangGraph orchestration**.
> See `CLAUDE.md` for the current architecture documentation.
>
> **Key Changes from this Plan:**
> - Components now use LangChain `@tool` decorator pattern
> - Orchestration moved to `src/orchestrator/` using LangGraph StateGraph
> - `augmentation` component merged into `resolution` component
> - `orchestrator` component replaced by `src/orchestrator/workflow.py`

---

## Executive Summary (Original Plan)

This plan transforms the current monolithic RAG-based ticket management system into a **modular, service-oriented architecture** where each component (Embedding, Retrieval, Classification, Label Assignment, Resolution Generation) can operate:

1. **As standalone microservices** (AWS Lambda)
2. **As unified FastAPI endpoints** within a single service (local development)
3. **As importable Python modules** for direct code integration

**Local Development**: Run `uvicorn api_server:app --reload` to start all components in one process.

---

## Current Architecture Analysis

### Current Structure
```
test-recom-backend/
├── api_server.py                    # FastAPI with tightly coupled endpoints
├── main.py                          # CLI entry point
├── src/
│   ├── agents/                      # LangGraph agents (tightly coupled)
│   │   ├── classification_agent.py  # Domain classification
│   │   ├── pattern_recognition_agent.py  # FAISS similarity search
│   │   ├── label_assignment_agent.py  # Label assignment
│   │   └── resolution_generation_agent.py  # Resolution generation
│   ├── vectorstore/
│   │   ├── embedding_generator.py   # OpenAI embeddings
│   │   ├── faiss_manager.py         # FAISS index operations
│   │   └── data_ingestion.py        # Data loading
│   ├── graph/
│   │   ├── workflow.py              # LangGraph workflow definition
│   │   └── state_manager.py         # State routing
│   ├── models/                      # Pydantic/TypedDict schemas
│   ├── prompts/                     # LLM prompts
│   └── utils/                       # Config, OpenAI client, helpers
```

### Key Issues with Current Design
1. **Tight Coupling**: Agents depend on LangGraph state and each other
2. **Singleton Pattern**: Global instances prevent independent instantiation
3. **No Service Interface**: No standardized request/response contracts
4. **Config Coupling**: `Config` class loaded at import time
5. **No Health/Metrics**: Components lack observability hooks

---

## Target Architecture: Component as a Service

### Design Principles

1. **Dependency Injection**: No global singletons; dependencies passed explicitly
2. **Interface Contracts**: Each component has clear input/output Pydantic models
3. **Configuration Isolation**: Each component can be configured independently
4. **Transport Agnostic**: Core logic separated from HTTP handlers
5. **Zero LangGraph Dependency**: Components work without LangGraph workflow

### New Directory Structure

```
test-recom-backend/
├── api_server.py                    # Unified FastAPI server (keeps existing endpoints)
├── main.py                          # CLI entry point (unchanged)
│
├── components/                      # NEW: Standalone service components
│   ├── __init__.py
│   │
│   ├── base/                        # Base classes and interfaces
│   │   ├── __init__.py
│   │   ├── component.py             # BaseComponent abstract class
│   │   ├── config.py                # ComponentConfig base
│   │   └── exceptions.py            # Component-specific exceptions
│   │
│   ├── embedding/                   # Embedding Service Component
│   │   ├── __init__.py
│   │   ├── service.py               # EmbeddingService class
│   │   ├── models.py                # EmbeddingRequest, EmbeddingResponse
│   │   └── router.py                # FastAPI router for /embedding/*
│   │
│   ├── retrieval/                   # Retrieval Service Component
│   │   ├── __init__.py
│   │   ├── service.py               # RetrievalService class
│   │   ├── models.py                # RetrievalRequest, RetrievalResponse
│   │   └── router.py                # FastAPI router for /retrieval/*
│   │
│   ├── classification/              # Classification Service Component
│   │   ├── __init__.py
│   │   ├── service.py               # ClassificationService class
│   │   ├── models.py                # ClassificationRequest, ClassificationResponse
│   │   └── router.py                # FastAPI router for /classification/*
│   │
│   ├── labeling/                    # Label Assignment Service Component
│   │   ├── __init__.py
│   │   ├── service.py               # LabelingService class
│   │   ├── models.py                # LabelingRequest, LabelingResponse
│   │   └── router.py                # FastAPI router for /labeling/*
│   │
│   ├── augmentation/                # Resolution/Augmentation Service Component
│   │   ├── __init__.py
│   │   ├── service.py               # AugmentationService class
│   │   ├── models.py                # AugmentationRequest, AugmentationResponse
│   │   └── router.py                # FastAPI router for /augmentation/*
│   │
│   └── orchestrator/                # Pipeline Orchestrator
│       ├── __init__.py
│       ├── service.py               # Orchestrates component calls
│       ├── models.py                # Full pipeline request/response
│       └── router.py                # FastAPI router for /pipeline/*
│
├── src/                             # EXISTING: Keep for backward compatibility
│   ├── agents/                      # Wrap new components for LangGraph
│   ├── vectorstore/                 # Keep FAISS logic (used by retrieval component)
│   ├── graph/                       # LangGraph workflow (uses new components)
│   ├── models/                      # Existing schemas
│   ├── prompts/                     # LLM prompts (used by components)
│   └── utils/                       # Shared utilities
│
└── tests/                           # NEW: Component tests
    └── components/
        ├── test_embedding.py
        ├── test_retrieval.py
        └── ...
```

---

## Detailed Component Specifications

### 1. Base Component Framework

**File: `components/base/component.py`**

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel

TRequest = TypeVar('TRequest', bound=BaseModel)
TResponse = TypeVar('TResponse', bound=BaseModel)

class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """Base class for all service components."""

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """Process a request and return response."""
        pass

    @abstractmethod
    async def health_check(self) -> dict:
        """Return component health status."""
        pass

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Return component identifier."""
        pass
```

**File: `components/base/config.py`**

```python
from pydantic_settings import BaseSettings
from typing import Optional

class ComponentConfig(BaseSettings):
    """Base configuration for all components."""

    # OpenAI settings (can be overridden per component)
    openai_api_key: str

    # Component-specific settings loaded from env vars
    log_level: str = "INFO"
    enable_metrics: bool = True

    class Config:
        env_prefix = ""  # Will be overridden by subclasses
        env_file = ".env"
```

---

### 2. Embedding Component

**Purpose**: Generate embeddings for ticket text using OpenAI

**File: `components/embedding/models.py`**

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    text: str = Field(..., description="Text to embed")
    # OR structured ticket input
    title: Optional[str] = None
    description: Optional[str] = None

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embedding: List[float]
    model: str
    dimensions: int
    token_count: int

class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding."""
    texts: List[str]
    batch_size: int = 10

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    total_tokens: int
```

**File: `components/embedding/service.py`**

```python
from components.base.component import BaseComponent
from components.embedding.models import EmbeddingRequest, EmbeddingResponse
from components.embedding.config import EmbeddingConfig

class EmbeddingService(BaseComponent[EmbeddingRequest, EmbeddingResponse]):
    """Embedding service component."""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self._client = None  # Lazy initialization

    async def process(self, request: EmbeddingRequest) -> EmbeddingResponse:
        # Implementation here
        pass

    async def health_check(self) -> dict:
        return {"status": "healthy", "component": "embedding"}

    @property
    def component_name(self) -> str:
        return "embedding"
```

---

### 3. Retrieval Component

**Purpose**: Search FAISS index for similar tickets

**File: `components/retrieval/models.py`**

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SimilarTicket(BaseModel):
    """A similar ticket from the vector store."""
    ticket_id: str
    title: str
    description: str
    similarity_score: float
    vector_similarity: float
    metadata_score: float
    priority: str
    labels: List[str]
    resolution_time_hours: float
    domain: str
    resolution: Optional[str] = None

class RetrievalRequest(BaseModel):
    """Request model for retrieval."""
    # Query can be text OR embedding
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None

    # OR structured ticket input
    title: Optional[str] = None
    description: Optional[str] = None

    # Search parameters
    top_k: int = Field(default=20, ge=1, le=100)
    domain_filter: Optional[str] = None

    # Scoring weights
    vector_weight: float = Field(default=0.7, ge=0, le=1)
    metadata_weight: float = Field(default=0.3, ge=0, le=1)

class RetrievalResponse(BaseModel):
    """Response model for retrieval."""
    similar_tickets: List[SimilarTicket]
    search_metadata: Dict[str, Any]
    query_domain: Optional[str]
    total_found: int
    avg_similarity: float
```

---

### 4. Classification Component

**Purpose**: Classify tickets into domains (MM, CIW, Specialty)

**File: `components/classification/models.py`**

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ClassificationRequest(BaseModel):
    """Request model for classification."""
    title: str
    description: str

class DomainScore(BaseModel):
    """Score for a single domain."""
    domain: str
    confidence: float
    decision: bool
    reasoning: str
    keywords: List[str]

class ClassificationResponse(BaseModel):
    """Response model for classification."""
    classified_domain: str
    confidence: float
    reasoning: str
    domain_scores: Dict[str, DomainScore]
    extracted_keywords: List[str]
```

---

### 5. Labeling Component

**Purpose**: Assign labels based on similar tickets and AI analysis

**File: `components/labeling/models.py`**

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class LabelWithConfidence(BaseModel):
    """A label with its confidence score."""
    label: str
    confidence: float
    category: str  # "historical", "business", "technical"
    reasoning: Optional[str] = None

class LabelingRequest(BaseModel):
    """Request model for labeling."""
    title: str
    description: str
    domain: str
    priority: str = "Medium"
    similar_tickets: List[Dict]  # From retrieval component

class LabelingResponse(BaseModel):
    """Response model for labeling."""
    historical_labels: List[LabelWithConfidence]
    business_labels: List[LabelWithConfidence]
    technical_labels: List[LabelWithConfidence]
    all_labels: List[str]  # Combined list for convenience
```

---

### 6. Augmentation Component (Resolution Generation)

**Purpose**: Generate resolution plans based on context

**File: `components/augmentation/models.py`**

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ResolutionStep(BaseModel):
    """A single resolution step."""
    step_number: int
    description: str
    commands: List[str]
    validation: str
    estimated_time_minutes: int
    risk_level: str
    rollback_procedure: Optional[str]

class AugmentationRequest(BaseModel):
    """Request model for augmentation/resolution."""
    title: str
    description: str
    domain: str
    priority: str
    labels: List[str]
    similar_tickets: List[Dict]
    avg_similarity: float

class AugmentationResponse(BaseModel):
    """Response model for augmentation/resolution."""
    summary: str
    diagnostic_steps: List[ResolutionStep]
    resolution_steps: List[ResolutionStep]
    additional_considerations: List[str]
    references: List[str]
    total_estimated_time_hours: float
    confidence: float
    alternative_approaches: List[str]
```

---

## Implementation Plan

### Phase 1: Create Base Framework (Files 1-4)

1. **Create `components/base/` directory and files**
   - `component.py` - BaseComponent abstract class
   - `config.py` - ComponentConfig base class
   - `exceptions.py` - Custom exceptions
   - `__init__.py` - Exports

2. **Key Design Decisions**:
   - Use Pydantic v2 for all models
   - Use `pydantic-settings` for configuration
   - Support both sync and async operations
   - Include health check and metrics interfaces

### Phase 2: Implement Embedding Component (Files 5-9)

1. **Create `components/embedding/` directory**
   - Extract embedding logic from `src/vectorstore/embedding_generator.py`
   - Create clean interface without global state
   - Add FastAPI router
   - Add Lambda handler

2. **Migration Strategy**:
   - Keep existing `embedding_generator.py` as wrapper
   - New code uses `EmbeddingService` directly
   - LangGraph agents use wrapper for backward compatibility

### Phase 3: Implement Retrieval Component (Files 10-14)

1. **Create `components/retrieval/` directory**
   - Extract FAISS logic from `src/vectorstore/faiss_manager.py`
   - Extract hybrid scoring from `pattern_recognition_agent.py`
   - Support both embedding input and text input (will call embedding component)

2. **Dependencies**:
   - Can optionally depend on Embedding component
   - FAISS index path configurable

### Phase 4: Implement Classification Component (Files 15-19)

1. **Create `components/classification/` directory**
   - Extract classification logic from `src/agents/classification_agent.py`
   - Keep prompt templates in `src/prompts/`
   - Remove LangGraph state dependency

### Phase 5: Implement Labeling Component (Files 20-24)

1. **Create `components/labeling/` directory**
   - Extract from `src/agents/label_assignment_agent.py`
   - Requires similar tickets as input (from retrieval)
   - Keep three-tier label logic

### Phase 6: Implement Augmentation Component (Files 25-29)

1. **Create `components/augmentation/` directory**
   - Extract from `src/agents/resolution_generation_agent.py`
   - This is the "generation" part of RAG

### Phase 7: Create Orchestrator Component (Files 30-32)

1. **Create `components/orchestrator/` directory**
   - Chains all components together
   - Provides full RAG pipeline as single call
   - Replaces LangGraph workflow for simple use cases

### Phase 8: Update API Server (File 33)

1. **Modify `api_server.py`**
   - Mount all component routers
   - Keep existing endpoints for backward compatibility
   - Add new `/v2/*` endpoints using components

### Phase 9: Update LangGraph Integration (Files 34-36)

1. **Update `src/agents/` to use new components**
   - Each agent becomes a thin wrapper
   - Maintains LangGraph state interface
   - Delegates to component services


---

## API Endpoints After Implementation

### FastAPI Server Endpoints

```
# Existing endpoints (backward compatible)
POST /api/process-ticket          # Full pipeline with streaming
GET  /api/health                  # Health check
POST /api/preview-search          # Search tuning UI

# NEW: Component endpoints (v2)
POST /v2/embedding/generate       # Generate embedding
POST /v2/embedding/batch          # Batch embeddings

POST /v2/retrieval/search         # Search similar tickets
POST /v2/retrieval/search-by-embedding  # Search with pre-computed embedding

POST /v2/classification/classify  # Classify ticket domain

POST /v2/labeling/assign          # Assign labels

POST /v2/augmentation/generate    # Generate resolution plan

POST /v2/pipeline/process         # Full pipeline (non-streaming)
POST /v2/pipeline/stream          # Full pipeline (SSE streaming)
```

---

## Usage Examples

### 1. Direct Python Import (Library Mode)

```python
from components.embedding import EmbeddingService, EmbeddingRequest
from components.retrieval import RetrievalService, RetrievalRequest

# Initialize with custom config
embedding_svc = EmbeddingService(config=EmbeddingConfig(
    openai_api_key="sk-...",
    model="text-embedding-3-large"
))

# Use directly
embedding_response = await embedding_svc.process(
    EmbeddingRequest(title="DB Error", description="Connection timeout...")
)

# Chain with retrieval
retrieval_svc = RetrievalService(config=RetrievalConfig(
    faiss_index_path="/path/to/index"
))

results = await retrieval_svc.process(
    RetrievalRequest(
        query_embedding=embedding_response.embedding,
        top_k=10,
        domain_filter="MM"
    )
)
```

### 2. FastAPI Endpoints (HTTP Mode)

```bash
# Generate embedding
curl -X POST http://localhost:8000/v2/embedding/generate \
  -H "Content-Type: application/json" \
  -d '{"title": "DB Error", "description": "Connection timeout..."}'

# Search similar tickets
curl -X POST http://localhost:8000/v2/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{"title": "DB Error", "description": "...", "top_k": 10}'
```

---

## Files to Create/Modify

### New Files to Create (29 files)

| # | File Path | Purpose |
|---|-----------|---------|
| 1 | `components/__init__.py` | Package init |
| 2 | `components/base/__init__.py` | Base package |
| 3 | `components/base/component.py` | BaseComponent ABC |
| 4 | `components/base/config.py` | ComponentConfig base |
| 5 | `components/base/exceptions.py` | Custom exceptions |
| 6 | `components/embedding/__init__.py` | Embedding package |
| 7 | `components/embedding/service.py` | EmbeddingService |
| 8 | `components/embedding/models.py` | Request/Response models |
| 9 | `components/embedding/router.py` | FastAPI router |
| 10 | `components/retrieval/__init__.py` | Retrieval package |
| 11 | `components/retrieval/service.py` | RetrievalService |
| 12 | `components/retrieval/models.py` | Request/Response models |
| 13 | `components/retrieval/router.py` | FastAPI router |
| 14 | `components/classification/__init__.py` | Classification package |
| 15 | `components/classification/service.py` | ClassificationService |
| 16 | `components/classification/models.py` | Request/Response models |
| 17 | `components/classification/router.py` | FastAPI router |
| 18 | `components/labeling/__init__.py` | Labeling package |
| 19 | `components/labeling/service.py` | LabelingService |
| 20 | `components/labeling/models.py` | Request/Response models |
| 21 | `components/labeling/router.py` | FastAPI router |
| 22 | `components/augmentation/__init__.py` | Augmentation package |
| 23 | `components/augmentation/service.py` | AugmentationService |
| 24 | `components/augmentation/models.py` | Request/Response models |
| 25 | `components/augmentation/router.py` | FastAPI router |
| 26 | `components/orchestrator/__init__.py` | Orchestrator package |
| 27 | `components/orchestrator/service.py` | OrchestratorService |
| 28 | `components/orchestrator/models.py` | Pipeline models |
| 29 | `components/orchestrator/router.py` | FastAPI router |

### Files to Modify (4 files)

| # | File Path | Changes |
|---|-----------|---------|
| 1 | `api_server.py` | Mount component routers, add /v2 endpoints |
| 2 | `requirements.txt` | Add pydantic-settings |
| 3 | `src/agents/classification_agent.py` | Wrap new ClassificationService |
| 4 | `src/agents/pattern_recognition_agent.py` | Wrap new RetrievalService |

---

## Success Criteria

1. **Independence**: Each component can be imported and used without others
2. **Testability**: Each component has unit tests with mocked dependencies
3. **Local Development**: Run `uvicorn api_server:app --reload` to start all components
4. **Backward Compatibility**: Existing `api_server.py` endpoints work unchanged
5. **Documentation**: Each component has OpenAPI docs via FastAPI
6. **Configuration**: Each component loads config from environment variables

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing API | High | Keep all existing endpoints, add new /v2 prefix |
| Performance overhead | Medium | Use connection pooling, lazy initialization |
| Config complexity | Medium | Use pydantic-settings with clear env var naming |
| Testing gaps | Medium | Create comprehensive component tests first |

---

## Next Steps

1. Review and approve this plan
2. Start with Phase 1 (Base Framework)
3. Implement components in order (Embedding → Retrieval → Classification → Labeling → Augmentation)
4. Update API server last to ensure components are stable
5. Create deployment configs once all components work
