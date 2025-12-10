# Component-as-a-Service Architecture

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Concepts](#core-concepts)
4. [The Base Infrastructure](#the-base-infrastructure)
5. [Component Patterns](#component-patterns)
6. [File Structure Convention](#file-structure-convention)
7. [Building a Complete Component](#building-a-complete-component)
8. [Integration with LangGraph](#integration-with-langgraph)
9. [HTTP API Exposure](#http-api-exposure)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

This document explains the **Component-as-a-Service (CaaS)** architecture used in our backend system. This architecture allows each piece of functionality to be:

1. **Self-contained** - Each component has its own models, logic, and API
2. **Independently deployable** - Components can be used standalone via HTTP
3. **Composable** - Components integrate into LangGraph workflows
4. **Testable** - Clear boundaries make unit testing straightforward

### Why This Architecture?

Traditional monolithic backends tightly couple business logic, making changes risky and testing difficult. Our CaaS approach solves this by:

- **Separation of Concerns**: Each component owns one responsibility
- **Multiple Access Paths**: Use via HTTP API, Python import, or LangGraph workflow
- **Type Safety**: Pydantic models enforce contracts at every boundary
- **Graceful Degradation**: Errors in one component don't crash the entire system

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                          │
│                         (api_server.py)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│   │  /v2/embedding  │  │ /v2/retrieval   │  │/v2/classification│   │
│   │     Router      │  │     Router      │  │     Router       │   │
│   └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘   │
│            │                     │                     │            │
│   ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼─────────┐   │
│   │ EmbeddingService│  │ RetrievalService│  │ClassificationSvc │   │
│   │  (BaseComponent)│  │  (BaseComponent)│  │  (BaseComponent) │   │
│   └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘   │
│            │                     │                     │            │
│   ┌────────▼────────┐  ┌────────▼────────┐  ┌────────▼─────────┐   │
│   │   @tool funcs   │  │   @tool funcs   │  │   @tool funcs    │   │
│   │   (LangChain)   │  │   (LangChain)   │  │   (LangChain)    │   │
│   └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘   │
│            │                     │                     │            │
│            └─────────────────────┼─────────────────────┘            │
│                                  │                                   │
│                    ┌─────────────▼─────────────┐                    │
│                    │   LangGraph Workflow      │                    │
│                    │   (src/orchestrator/)     │                    │
│                    └───────────────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### The Three Access Paths

Every component can be accessed three ways:

```python
# Path 1: Direct Python Import
from components.embedding import EmbeddingService
service = EmbeddingService()
response = await service.process(EmbeddingRequest(text="hello"))

# Path 2: HTTP API
# POST /v2/embedding/generate
# {"text": "hello"}

# Path 3: LangGraph Workflow Node
from components.retrieval import retrieval_node
workflow.add_node("retrieval", retrieval_node)
```

---

## Core Concepts

### 1. BaseComponent Abstract Class

Every service component extends `BaseComponent`, which defines the contract:

```python
class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """
    Abstract base class for all service components.

    Type Parameters:
        TRequest: Pydantic model for input validation
        TResponse: Pydantic model for output structure
    """

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """Main business logic - process a request."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return health status for monitoring."""
        pass

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Unique identifier for logging/metrics."""
        pass
```

### 2. Component Configuration

Each component has a configuration class extending `ComponentConfig`:

```python
class ComponentConfig(BaseSettings):
    """
    Base configuration using pydantic-settings.

    Settings are loaded from:
    1. Environment variables
    2. .env file
    3. Default values
    """

    openai_api_key: str = Field(default=None)
    log_level: str = Field(default="INFO")
    max_retries: int = Field(default=3)
    retry_delay_seconds: int = Field(default=2)
```

### 3. Request/Response Models

Every component defines Pydantic models for type-safe contracts:

```python
class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    text: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embedding: List[float]
    model: str
    dimensions: int
```

### 4. Exception Hierarchy

Custom exceptions for consistent error handling:

```python
class ComponentError(Exception):
    """Base exception for all component errors."""

    def __init__(self, message: str, component: str = None, details: dict = None):
        self.message = message
        self.component = component
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert to API-friendly format."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "details": self.details,
        }

class ConfigurationError(ComponentError):
    """Raised when configuration is invalid."""
    pass

class ProcessingError(ComponentError):
    """Raised when processing fails."""
    pass

class ValidationError(ComponentError):
    """Raised when input validation fails."""
    pass
```

---

## The Base Infrastructure

The `components/base/` directory provides shared infrastructure:

```
components/base/
├── __init__.py      # Public exports
├── component.py     # BaseComponent abstract class
├── config.py        # ComponentConfig base settings
└── exceptions.py    # Custom exception classes
```

### base/component.py

```python
"""
Base component abstract class.

All service components inherit from this class to ensure
consistent interface across the system.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any
from pydantic import BaseModel

# Generic type variables for request/response
TRequest = TypeVar("TRequest", bound=BaseModel)
TResponse = TypeVar("TResponse", bound=BaseModel)


class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """
    Abstract base class for all service components.

    Each component must implement:
    - process(): Main processing logic
    - health_check(): Health status check
    - component_name: Unique identifier

    Usage:
        class MyService(BaseComponent[MyRequest, MyResponse]):
            async def process(self, request: MyRequest) -> MyResponse:
                # Implementation
                pass
    """

    @abstractmethod
    async def process(self, request: TRequest) -> TResponse:
        """
        Process a request and return a response.

        This is the main entry point for the component.
        All business logic should be implemented here.

        Args:
            request: Pydantic model containing input data

        Returns:
            Pydantic model containing output data

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input validation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check component health status.

        Returns:
            Dict with at least:
            - status: "healthy" | "unhealthy" | "degraded"
            - component: Component name
            - details: Optional additional info
        """
        pass

    @property
    @abstractmethod
    def component_name(self) -> str:
        """
        Return unique component identifier.

        This is used for logging, metrics, and error reporting.
        """
        pass

    async def __call__(self, request: TRequest) -> TResponse:
        """
        Allow component to be called directly.

        Makes the component callable: response = await component(request)
        """
        return await self.process(request)
```

### base/config.py

```python
"""
Base configuration class for all components.

Uses pydantic-settings for environment variable loading.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ComponentConfig(BaseSettings):
    """
    Base configuration for all component services.

    Settings are loaded from:
    1. Environment variables
    2. .env file (if present)
    3. Default values

    Subclasses should override model_config to set env_prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )

    # OpenAI Configuration (shared across components)
    openai_api_key: str = Field(
        default=None,
        description="OpenAI API key for LLM and embedding calls",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # Retry configuration
    max_retries: int = Field(default=3)
    retry_delay_seconds: int = Field(default=2)

    def __init__(self, **kwargs):
        # Allow openai_api_key from OPENAI_API_KEY env var
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        super().__init__(**kwargs)

    def validate_required(self, *fields: str) -> None:
        """Validate that required fields are set."""
        from components.base.exceptions import ConfigurationError

        missing = []
        for field in fields:
            value = getattr(self, field, None)
            if value is None or value == "":
                missing.append(field)

        if missing:
            raise ConfigurationError(
                f"Missing required configuration: {', '.join(missing)}",
                component=self.__class__.__name__,
                missing_keys=missing,
            )
```

---

## Component Patterns

Our architecture supports three component patterns:

### Pattern 1: Full Service Component

**Used by:** Embedding, Retrieval, Classification, Labeling

This is the most complete pattern with all layers:

```
components/embedding/
├── __init__.py        # Public exports
├── models.py          # Pydantic request/response models
├── service.py         # BaseComponent implementation
├── router.py          # FastAPI HTTP endpoints
└── embedding.md       # Component documentation
```

### Pattern 2: Agent-Only Component

**Used by:** Novelty, Resolution

Components that only participate in LangGraph workflows:

```
components/novelty/
├── __init__.py        # Public exports
├── tools.py           # LangChain @tool functions
├── agent.py           # LangGraph node function
└── novelty.md         # Component documentation
```

### Pattern 3: Full Agent Component

**Used by:** Retrieval, Classification, Labeling

The most comprehensive pattern with all layers including LangChain tools:

```
components/retrieval/
├── __init__.py        # Public exports
├── models.py          # Pydantic models
├── tools.py           # LangChain @tool functions
├── agent.py           # LangGraph node function
├── service.py         # BaseComponent implementation
├── router.py          # FastAPI endpoints
└── retrieval.md       # Documentation
```

---

## File Structure Convention

### models.py - Data Contracts

Defines the input/output contracts:

```python
"""
Pydantic models for the Embedding component.

Defines request/response contracts for embedding generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for single embedding generation."""

    # Option 1: Raw text
    text: Optional[str] = Field(
        default=None,
        description="Raw text to embed. Use this OR title+description.",
    )

    # Option 2: Structured ticket input
    title: Optional[str] = Field(
        default=None,
        description="Ticket title",
    )
    description: Optional[str] = Field(
        default=None,
        description="Ticket description",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"text": "Database connection timeout error"},
                {"title": "DB Error", "description": "Timeout when connecting"},
            ]
        }


class EmbeddingResponse(BaseModel):
    """Response model for single embedding generation."""

    embedding: List[float] = Field(description="Embedding vector")
    model: str = Field(description="Model used")
    dimensions: int = Field(description="Vector dimensionality")
    input_text: str = Field(description="Text that was embedded")
```

### service.py - Business Logic

Implements the `BaseComponent` interface:

```python
"""
Embedding Service Component.

Generates embeddings for text using OpenAI's embedding models.
"""

import asyncio
from typing import Dict, Any, Optional

from openai import AsyncOpenAI
from openai import RateLimitError, APIError

from components.base import BaseComponent, ComponentConfig
from components.base.exceptions import ProcessingError, ConfigurationError
from components.embedding.models import EmbeddingRequest, EmbeddingResponse


class EmbeddingConfig(ComponentConfig):
    """Configuration for Embedding Service."""

    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    class Config:
        env_prefix = "EMBEDDING_"  # EMBEDDING_MODEL, etc.


class EmbeddingService(BaseComponent[EmbeddingRequest, EmbeddingResponse]):
    """
    Service for generating text embeddings using OpenAI.

    Usage:
        service = EmbeddingService()
        response = await service.process(EmbeddingRequest(text="hello"))
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not self.config.openai_api_key:
                raise ConfigurationError(
                    "OpenAI API key is required",
                    component=self.component_name,
                )
            self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
        return self._client

    @property
    def component_name(self) -> str:
        return "embedding"

    async def process(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding for the input text."""
        input_text = self._get_input_text(request)
        embedding = await self._generate_embedding(input_text)

        return EmbeddingResponse(
            embedding=embedding,
            model=self.config.embedding_model,
            dimensions=len(embedding),
            input_text=input_text[:200] + "..." if len(input_text) > 200 else input_text,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check if embedding service is healthy."""
        try:
            # Verify connectivity with a test embedding
            await self.client.embeddings.create(
                model=self.config.embedding_model,
                input="health check",
            )
            return {
                "status": "healthy",
                "component": self.component_name,
                "model": self.config.embedding_model,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": self.component_name,
                "error": str(e),
            }

    def _get_input_text(self, request: EmbeddingRequest) -> str:
        """Extract text from request."""
        if request.text:
            return request.text
        if request.title or request.description:
            parts = []
            if request.title:
                parts.append(f"Title: {request.title}")
            if request.description:
                parts.append(f"Description: {request.description}")
            return "\n".join(parts)
        raise ProcessingError(
            "Either 'text' or 'title/description' required",
            component=self.component_name,
        )

    async def _generate_embedding(self, text: str) -> list:
        """Generate embedding with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=text,
                )
                return response.data[0].embedding
            except (RateLimitError, APIError) as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
                else:
                    raise ProcessingError(
                        f"Max retries exceeded: {e}",
                        component=self.component_name,
                        original_error=e,
                    )
```

### tools.py - LangChain Tools

Defines `@tool` decorated functions for LangChain integration:

```python
"""
Retrieval Tools - LangChain @tool decorated functions for FAISS search.

These tools handle vector search, hybrid scoring, and similar ticket retrieval.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import faiss
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings


# Module-level cache for FAISS index
_faiss_index: Optional[faiss.Index] = None
_metadata: List[Dict[str, Any]] = []
_index_loaded = False


@tool
async def search_similar_tickets(
    title: str,
    description: str,
    domain_filter: Optional[str] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Search FAISS index for similar historical tickets.

    Generates an embedding for the query text and searches the FAISS index
    for the most similar historical tickets. Optionally filters by domain.

    Args:
        title: Ticket title
        description: Ticket description
        domain_filter: Optional domain to filter results (MM, CIW, Specialty)
        top_k: Number of results to return (default 20)

    Returns:
        Dict containing:
        - similar_tickets: List of similar ticket dicts with raw scores
        - query_embedding: The generated embedding (for reuse)
        - total_searched: Number of tickets searched
    """
    # Generate embedding for query
    query_text = f"{title}\n\n{description}"
    query_embedding = await _generate_embedding(query_text)

    # Ensure index is loaded
    index, metadata = _ensure_index_loaded()

    # Normalize for cosine similarity
    query_array = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_array)

    # Search with optional domain filtering
    search_k = top_k * 3 if domain_filter else top_k
    distances, indices = index.search(query_array, search_k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:  # FAISS padding
            continue

        ticket_data = metadata[idx].copy()

        # Apply domain filter
        if domain_filter and ticket_data.get("domain") != domain_filter:
            continue

        ticket_data["vector_similarity"] = float(distance)
        results.append(ticket_data)

        if len(results) >= top_k:
            break

    return {
        "similar_tickets": results,
        "query_embedding": query_embedding,
        "total_searched": index.ntotal
    }


@tool
def apply_hybrid_scoring(
    similar_tickets: List[Dict[str, Any]],
    vector_weight: float = 0.7,
    metadata_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Apply hybrid scoring to search results.

    Hybrid score = (vector_weight * vector_similarity) + (metadata_weight * metadata_score)

    Args:
        similar_tickets: List of tickets with vector_similarity scores
        vector_weight: Weight for vector similarity (default 0.7)
        metadata_weight: Weight for metadata relevance (default 0.3)

    Returns:
        List of tickets with similarity_score, sorted by score descending
    """
    priority_scores = {"Critical": 1.0, "High": 0.8, "Medium": 0.5, "Low": 0.3}

    scored_tickets = []
    for ticket in similar_tickets:
        vector_score = ticket.get("vector_similarity", 0.0)

        # Priority factor
        priority = ticket.get("priority", "Medium")
        priority_score = priority_scores.get(priority, 0.5)

        # Resolution time factor
        res_time = ticket.get("resolution_time_hours", 24)
        time_score = max(0, 1 - (res_time / 100.0))

        # Combined scores
        metadata_score = (priority_score * 0.6) + (time_score * 0.4)
        hybrid_score = (vector_weight * vector_score) + (metadata_weight * metadata_score)

        ticket_with_scores = ticket.copy()
        ticket_with_scores["similarity_score"] = hybrid_score
        ticket_with_scores["metadata_score"] = metadata_score
        scored_tickets.append(ticket_with_scores)

    scored_tickets.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored_tickets
```

### agent.py - LangGraph Node

Wraps tools into a LangGraph-compatible node function:

```python
"""
Retrieval Agent - LangGraph node for historical matching.

This agent searches for similar historical tickets using FAISS
and applies hybrid scoring for relevance ranking.
"""

from typing import Dict, Any
from components.retrieval.tools import search_similar_tickets, apply_hybrid_scoring


async def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for historical matching / retrieval.

    IMPORTANT: Returns a PARTIAL state dict that gets merged into
    the full workflow state. Do NOT return a complete state object.

    Args:
        state: Current workflow state with ticket info and classified domain

    Returns:
        Partial state update with similar tickets and search metadata
    """
    try:
        title = state.get("title", "")
        description = state.get("description", "")
        domain = state.get("classified_domain", None)
        ticket_id = state.get("ticket_id", "N/A")

        # Get search config from state if available
        search_config = state.get("search_config", {}) or {}
        top_k = search_config.get("top_k", 20)
        vector_weight = search_config.get("vector_weight", 0.7)
        metadata_weight = search_config.get("metadata_weight", 0.3)

        print(f"\n🔎 Retrieval Agent - Searching: {ticket_id}")
        print(f"   Domain filter: {domain}")

        # Search for similar tickets using LangChain tool
        search_result = await search_similar_tickets.ainvoke({
            "title": title,
            "description": description,
            "domain_filter": domain,
            "top_k": top_k
        })

        raw_tickets = search_result.get("similar_tickets", [])

        # Apply hybrid scoring
        scored_tickets = apply_hybrid_scoring.invoke({
            "similar_tickets": raw_tickets,
            "vector_weight": vector_weight,
            "metadata_weight": metadata_weight
        })

        # Calculate statistics
        total_found = len(scored_tickets)
        avg_similarity = (
            sum(t["similarity_score"] for t in scored_tickets) / total_found
            if total_found > 0 else 0
        )

        print(f"   ✅ Found {total_found} similar tickets")

        # Return PARTIAL state update
        return {
            "similar_tickets": scored_tickets,
            "similarity_scores": [t["similarity_score"] for t in scored_tickets],
            "search_metadata": {
                "query_domain": domain,
                "total_found": total_found,
                "avg_similarity": avg_similarity,
            },
            "status": "success",
            "current_agent": "retrieval",
            "messages": [{
                "role": "assistant",
                "content": f"Found {total_found} similar tickets"
            }]
        }

    except Exception as e:
        print(f"   ❌ Retrieval error: {str(e)}")
        return {
            "similar_tickets": [],
            "similarity_scores": [],
            "search_metadata": {},
            "status": "error",
            "current_agent": "retrieval",
            "error_message": f"Retrieval failed: {str(e)}",
            "messages": [{
                "role": "assistant",
                "content": f"Retrieval failed: {str(e)}"
            }]
        }
```

### router.py - FastAPI Endpoints

Exposes the service via HTTP:

```python
"""
FastAPI router for Embedding Service.

Provides HTTP endpoints for embedding generation.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from components.embedding.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)
from components.embedding.service import EmbeddingService
from components.base.exceptions import ComponentError


# Create router with prefix and tags
router = APIRouter(
    prefix="/embedding",
    tags=["Embedding"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
    },
)

# Singleton service instance (lazy initialization)
_service: Optional[EmbeddingService] = None


def get_service() -> EmbeddingService:
    """Dependency injection for embedding service."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


@router.post(
    "/generate",
    response_model=EmbeddingResponse,
    summary="Generate embedding for text",
)
async def generate_embedding(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_service),
) -> EmbeddingResponse:
    """Generate embedding for a single text."""
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/batch",
    response_model=BatchEmbeddingResponse,
    summary="Generate embeddings for multiple texts",
)
async def generate_batch_embeddings(
    request: BatchEmbeddingRequest,
    service: EmbeddingService = Depends(get_service),
) -> BatchEmbeddingResponse:
    """Generate embeddings for multiple texts."""
    try:
        return await service.process_batch(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/health",
    summary="Check embedding service health",
)
async def health_check(
    service: EmbeddingService = Depends(get_service),
) -> dict:
    """Check embedding service health."""
    return await service.health_check()
```

### __init__.py - Public Exports

Defines the component's public API:

```python
"""
Embedding Component

Provides text embedding generation using OpenAI models.

Usage:
    # Import service
    from components.embedding import EmbeddingService, EmbeddingConfig

    # Import models
    from components.embedding import EmbeddingRequest, EmbeddingResponse

    # Import router for FastAPI
    from components.embedding import router
"""

from components.embedding.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)
from components.embedding.service import (
    EmbeddingService,
    EmbeddingConfig,
)
from components.embedding.router import router

__all__ = [
    # Models
    "EmbeddingRequest",
    "EmbeddingResponse",
    "BatchEmbeddingRequest",
    "BatchEmbeddingResponse",
    # Service
    "EmbeddingService",
    "EmbeddingConfig",
    # Router
    "router",
]
```

---

## Building a Complete Component

Let's walk through building a new component step-by-step.

### Step 1: Create Directory Structure

```bash
mkdir -p components/my_component
touch components/my_component/__init__.py
touch components/my_component/models.py
touch components/my_component/service.py
touch components/my_component/router.py
touch components/my_component/tools.py   # Optional: for LangGraph
touch components/my_component/agent.py   # Optional: for LangGraph
```

### Step 2: Define Models (models.py)

```python
"""Pydantic models for MyComponent."""

from pydantic import BaseModel, Field
from typing import Optional


class MyRequest(BaseModel):
    """Request model."""
    input_data: str = Field(description="Input data to process")
    option: Optional[str] = Field(default="default", description="Processing option")


class MyResponse(BaseModel):
    """Response model."""
    result: str = Field(description="Processing result")
    processed_at: str = Field(description="Timestamp")
```

### Step 3: Implement Service (service.py)

```python
"""MyComponent Service."""

from datetime import datetime
from typing import Dict, Any, Optional

from components.base import BaseComponent, ComponentConfig
from components.base.exceptions import ProcessingError
from components.my_component.models import MyRequest, MyResponse


class MyComponentConfig(ComponentConfig):
    """Configuration for MyComponent."""

    custom_setting: str = "default_value"

    class Config:
        env_prefix = "MY_COMPONENT_"


class MyComponentService(BaseComponent[MyRequest, MyResponse]):
    """Service for my custom processing."""

    def __init__(self, config: Optional[MyComponentConfig] = None):
        self.config = config or MyComponentConfig()

    @property
    def component_name(self) -> str:
        return "my_component"

    async def process(self, request: MyRequest) -> MyResponse:
        """Process the request."""
        try:
            result = f"Processed: {request.input_data} with {request.option}"

            return MyResponse(
                result=result,
                processed_at=datetime.now().isoformat(),
            )
        except Exception as e:
            raise ProcessingError(
                f"Processing failed: {e}",
                component=self.component_name,
                original_error=e,
            )

    async def health_check(self) -> Dict[str, Any]:
        """Check health."""
        return {
            "status": "healthy",
            "component": self.component_name,
        }
```

### Step 4: Create Router (router.py)

```python
"""FastAPI router for MyComponent."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from components.my_component.models import MyRequest, MyResponse
from components.my_component.service import MyComponentService
from components.base.exceptions import ComponentError


router = APIRouter(
    prefix="/my-component",
    tags=["MyComponent"],
)

_service: Optional[MyComponentService] = None


def get_service() -> MyComponentService:
    global _service
    if _service is None:
        _service = MyComponentService()
    return _service


@router.post("/process", response_model=MyResponse)
async def process(
    request: MyRequest,
    service: MyComponentService = Depends(get_service),
) -> MyResponse:
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())


@router.get("/health")
async def health_check(
    service: MyComponentService = Depends(get_service),
) -> dict:
    return await service.health_check()
```

### Step 5: Export Public API (__init__.py)

```python
"""MyComponent - Custom processing service."""

from components.my_component.models import MyRequest, MyResponse
from components.my_component.service import MyComponentService, MyComponentConfig
from components.my_component.router import router

__all__ = [
    "MyRequest",
    "MyResponse",
    "MyComponentService",
    "MyComponentConfig",
    "router",
]
```

### Step 6: Mount Router in FastAPI

```python
# In api_server.py
from components.my_component import router as my_component_router

app.include_router(my_component_router, prefix="/v2")
```

---

## Integration with LangGraph

### Workflow State

The workflow uses a TypedDict with `total=False` for partial updates:

```python
# src/orchestrator/state.py
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class TicketWorkflowState(TypedDict, total=False):
    """
    State that flows through the LangGraph workflow.

    Using total=False allows agents to return partial updates
    that get merged into the full state.
    """

    # Input fields
    ticket_id: str
    title: str
    description: str
    priority: str

    # Classification output
    classified_domain: Optional[str]
    classification_confidence: float

    # Retrieval output
    similar_tickets: List[Dict[str, Any]]
    similarity_scores: List[float]

    # Labeling output
    assigned_labels: List[str]
    label_confidence: Dict[str, float]

    # Resolution output
    resolution_plan: Dict[str, Any]

    # Status tracking
    status: str  # "processing" | "success" | "error"
    current_agent: str
    error_message: Optional[str]

    # Messages accumulate across agents
    messages: Annotated[List[Dict], operator.add]
```

### Building the Workflow

```python
# src/orchestrator/workflow.py
from langgraph.graph import StateGraph, END
from src.orchestrator.state import TicketWorkflowState

# Import agent nodes from components
from components.classification.agent import classification_node
from components.retrieval.agent import retrieval_node
from components.labeling.agent import labeling_node
from components.resolution.agent import resolution_node


def route_after_agent(state: TicketWorkflowState) -> str:
    """Route based on agent status."""
    if state.get("status") == "error":
        return "error_handler"
    return "next_agent"  # Name varies per routing function


def build_workflow() -> StateGraph:
    """Build the LangGraph workflow."""

    workflow = StateGraph(TicketWorkflowState)

    # Add agent nodes from components
    workflow.add_node("Historical Match Agent", retrieval_node)
    workflow.add_node("Label Assignment Agent", labeling_node)
    workflow.add_node("Resolution Generation Agent", resolution_node)
    workflow.add_node("Error Handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("Historical Match Agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "Historical Match Agent",
        route_after_retrieval,
        {
            "labeling": "Label Assignment Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Label Assignment Agent",
        route_after_labeling,
        {
            "resolution": "Resolution Generation Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Resolution Generation Agent",
        route_after_resolution,
        {
            "end": END,
            "error_handler": "Error Handler"
        }
    )

    return workflow.compile()
```

### Partial State Updates

**Critical Pattern**: Agent nodes return **partial dictionaries**, not complete state objects:

```python
# ✅ CORRECT - Returns only new/updated fields
async def my_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    result = await process_something(state)

    return {
        "my_result": result,
        "status": "success",
        "current_agent": "my_agent",
        "messages": [{"role": "assistant", "content": "Done"}]
    }

# ❌ WRONG - Never return complete state
async def my_agent_node(state: Dict[str, Any]) -> TicketWorkflowState:
    # DON'T DO THIS
    return TicketWorkflowState(
        ticket_id=state["ticket_id"],
        title=state["title"],
        # ... copying all fields
    )
```

---

## HTTP API Exposure

### Router Mounting

All component routers are mounted under `/v2/` prefix:

```python
# api_server.py

from fastapi import FastAPI
from components.embedding import router as embedding_router
from components.retrieval import router as retrieval_router
from components.classification import router as classification_router
from components.labeling import router as labeling_router

app = FastAPI(title="Ticket Processing API")

# Mount component routers
app.include_router(embedding_router, prefix="/v2")
app.include_router(retrieval_router, prefix="/v2")
app.include_router(classification_router, prefix="/v2")
app.include_router(labeling_router, prefix="/v2")
```

### Resulting API Structure

```
/v2/embedding/
├── POST /generate     - Generate single embedding
├── POST /batch        - Generate batch embeddings
└── GET  /health       - Health check

/v2/retrieval/
├── POST /search       - Search similar tickets
├── GET  /stats        - Get index statistics
└── GET  /health       - Health check

/v2/classification/
├── POST /classify     - Classify ticket domain
└── GET  /health       - Health check

/v2/labeling/
├── POST /assign       - Assign labels to ticket
└── GET  /health       - Health check
```

### Dependency Injection Pattern

Services use FastAPI's dependency injection with lazy initialization:

```python
# Singleton instance (module-level)
_service: Optional[MyService] = None


def get_service() -> MyService:
    """
    Dependency injection function.

    Creates service on first call, reuses thereafter.
    This pattern ensures:
    - Service is created only when needed
    - Same instance is reused across requests
    - Easy to mock in tests
    """
    global _service
    if _service is None:
        _service = MyService()
    return _service


@router.post("/process")
async def process(
    request: MyRequest,
    service: MyService = Depends(get_service),  # Injected
) -> MyResponse:
    return await service.process(request)
```

---

## Best Practices

### 1. Always Validate Inputs

Use Pydantic models to validate all inputs at the boundary:

```python
class MyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    count: int = Field(ge=1, le=100)
```

### 2. Use Lazy Client Initialization

Don't create clients in `__init__`, use lazy loading:

```python
class MyService(BaseComponent):
    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client
```

### 3. Implement Health Checks

Every service should implement meaningful health checks:

```python
async def health_check(self) -> Dict[str, Any]:
    try:
        # Actually verify connectivity
        await self.client.models.list()
        return {"status": "healthy", "component": self.component_name}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### 4. Use Consistent Error Handling

Always catch exceptions and convert to ComponentError:

```python
async def process(self, request: MyRequest) -> MyResponse:
    try:
        result = await self._do_work(request)
        return MyResponse(result=result)
    except ValidationError as e:
        raise ValidationError(str(e), component=self.component_name)
    except Exception as e:
        raise ProcessingError(
            f"Processing failed: {e}",
            component=self.component_name,
            original_error=e,
        )
```

### 5. Return Partial State from Agent Nodes

Never return complete workflow state from agent nodes:

```python
# ✅ Good
return {
    "result": data,
    "status": "success",
    "current_agent": "my_agent"
}

# ❌ Bad - don't copy all state fields
```

### 6. Use Parallel Execution Where Possible

When operations are independent, use `asyncio.gather()`:

```python
# Process 3 classifiers in parallel
results = await asyncio.gather(
    classify_domain("MM", text),
    classify_domain("CIW", text),
    classify_domain("Specialty", text),
)
```

### 7. Document Your Components

Each component should have a markdown file explaining its purpose:

```markdown
# Embedding Component

## Purpose
Generates vector embeddings for text using OpenAI's text-embedding-3-large model.

## API
- `POST /v2/embedding/generate` - Single embedding
- `POST /v2/embedding/batch` - Batch embeddings

## Configuration
- `EMBEDDING_MODEL` - Model to use (default: text-embedding-3-large)
- `EMBEDDING_DIMENSIONS` - Vector dimensions (default: 3072)
```

---

## Troubleshooting

### Common Issues

**1. "Component X not found"**

Check that the component is exported in `__init__.py`:

```python
# components/my_component/__init__.py
from components.my_component.service import MyService
__all__ = ["MyService"]  # Must include the class
```

**2. "OpenAI API key not configured"**

Verify `.env` file exists and contains the key:

```bash
cat .env | grep OPENAI
# Should show: OPENAI_API_KEY=sk-...
```

**3. "State field not updating in workflow"**

Ensure you're returning a partial dict, not a full state:

```python
# ✅ Partial update
return {"my_field": value}

# ❌ Full state (wrong)
return TicketWorkflowState(**all_fields)
```

**4. "Health check returns unhealthy"**

Check the service logs for the actual error:

```python
# The health_check should log the error
async def health_check(self):
    try:
        await self.verify_connectivity()
        return {"status": "healthy"}
    except Exception as e:
        print(f"Health check failed: {e}")  # Add logging
        return {"status": "unhealthy", "error": str(e)}
```

**5. "Router endpoint not found"**

Verify the router is mounted in `api_server.py`:

```python
app.include_router(my_router, prefix="/v2")
```

And check the prefix in the router itself:

```python
router = APIRouter(prefix="/my-component")  # Results in /v2/my-component/...
```

---

## Summary

The Component-as-a-Service architecture provides:

1. **Modularity** - Each component is self-contained
2. **Flexibility** - Use via HTTP, Python import, or LangGraph
3. **Type Safety** - Pydantic models at every boundary
4. **Testability** - Clear interfaces make mocking easy
5. **Scalability** - Add new components without modifying existing code

Follow the patterns in this guide to create consistent, maintainable components that integrate seamlessly into the larger system.
