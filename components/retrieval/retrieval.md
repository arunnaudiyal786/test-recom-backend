# Retrieval Component

The Retrieval component (also known as Pattern Recognition) searches for similar historical tickets using **FAISS vector search** and applies **hybrid scoring** that combines vector similarity with metadata relevance.

## Overview

This component is the core of the RAG (Retrieval-Augmented Generation) pipeline. It finds historical tickets similar to the current one, providing context for downstream label assignment and resolution generation.

## Architecture

```
retrieval/
├── __init__.py          # Public API exports
├── agent.py             # LangGraph node wrapper (PatternRecognitionAgent)
├── models.py            # Pydantic request/response models
├── service.py           # RetrievalService (full-featured)
├── tools.py             # LangChain @tool decorated functions
├── router.py            # FastAPI HTTP endpoints
└── README.md            # This file
```

## Search Pipeline

```
Input Query (title + description)
           │
           ▼
    ┌─────────────────┐
    │ Generate        │
    │ Embedding       │  ◄── text-embedding-3-large (3072 dims)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ FAISS Search    │  ◄── IndexFlatIP (exact inner product)
    │ (Cosine Sim)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Domain Filter   │  ◄── Optional: filter by classified domain
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Hybrid Scoring  │  ◄── 70% vector + 30% metadata
    └────────┬────────┘
             │
             ▼
     Top K Similar Tickets
```

## Hybrid Scoring Formula

The component combines vector similarity with metadata relevance:

```
hybrid_score = (vector_weight × vector_similarity) + (metadata_weight × metadata_score)

metadata_score = (priority_score × 0.6) + (time_score × 0.4)

time_score = max(0, 1 - (resolution_time_hours / normalization_hours))
```

### Priority Weights (Default)

| Priority | Weight |
|----------|--------|
| Critical | 1.0 |
| High | 0.8 |
| Medium | 0.5 |
| Low | 0.3 |

### Default Scoring Weights

| Weight | Default | Description |
|--------|---------|-------------|
| `vector_weight` | 0.7 | Weight for vector similarity |
| `metadata_weight` | 0.3 | Weight for metadata score |
| `time_normalization_hours` | 100.0 | Max hours for time score calculation |

## Components

### Models (`models.py`)

#### PriorityWeights

Configurable priority weights for hybrid scoring.

```python
class PriorityWeights(BaseModel):
    Critical: float = 1.0
    High: float = 0.8
    Medium: float = 0.5
    Low: float = 0.3
```

#### SimilarTicket

A similar ticket from the vector store with all scoring details.

```python
class SimilarTicket(BaseModel):
    ticket_id: str               # Unique ticket identifier
    title: str                   # Ticket title/summary
    description: str             # Ticket description
    similarity_score: float      # Final hybrid similarity score (0-1)
    vector_similarity: float     # Raw vector similarity score
    metadata_score: float        # Metadata relevance score
    priority: str                # Ticket priority level
    labels: List[str]            # Assigned labels
    resolution_time_hours: float # Time to resolution
    domain: str                  # Ticket domain (MM, CIW, Specialty)
    resolution: Optional[str]    # Resolution summary if available
```

#### RetrievalRequest

Search request with all configurable parameters.

```python
class RetrievalRequest(BaseModel):
    # Query input (one of these required)
    query_text: Optional[str]           # Raw text to search
    query_embedding: Optional[List[float]]  # Pre-computed embedding
    title: Optional[str]                # Ticket title
    description: Optional[str]          # Ticket description

    # Search parameters
    top_k: int = 20                     # Number of results (1-100)
    domain_filter: Optional[str]        # Filter by domain

    # Scoring weights
    vector_weight: float = 0.7          # Vector similarity weight
    metadata_weight: float = 0.3        # Metadata weight
    priority_weights: Optional[PriorityWeights]  # Custom priority weights
    time_normalization_hours: float = 100.0
```

#### SearchMetadata

Metadata about the search operation.

```python
class SearchMetadata(BaseModel):
    query_domain: Optional[str]  # Domain filter applied
    total_found: int             # Number of results returned
    avg_similarity: float        # Average similarity score
    top_similarity: float        # Highest similarity score
    index_total: int             # Total vectors in index
```

#### RetrievalResponse

Complete search response.

```python
class RetrievalResponse(BaseModel):
    similar_tickets: List[SimilarTicket]  # Ranked by similarity
    search_metadata: SearchMetadata
    config_used: Dict[str, Any]           # Parameters used
```

---

### Tools (`tools.py`)

LangChain `@tool` decorated functions for LangGraph integration.

#### `search_similar_tickets`

Main search function that queries FAISS and returns raw results.

```python
@tool
async def search_similar_tickets(
    title: str,
    description: str,
    domain_filter: Optional[str] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Search FAISS index for similar historical tickets.

    Args:
        title: Ticket title
        description: Ticket description
        domain_filter: Optional domain to filter results
        top_k: Number of results to return

    Returns:
        Dict containing:
        - similar_tickets: List of ticket dicts with raw scores
        - query_embedding: The generated embedding
        - total_searched: Number of tickets searched
    """
```

**Implementation Details**:
1. Combines title and description into single query text
2. Generates embedding using OpenAI `text-embedding-3-large`
3. Normalizes vector for cosine similarity
4. Searches FAISS index (requests 3x results if domain filtering)
5. Applies domain filter post-search
6. Returns raw vector similarity scores

#### `apply_hybrid_scoring`

Applies hybrid scoring to search results.

```python
@tool
def apply_hybrid_scoring(
    similar_tickets: List[Dict[str, Any]],
    vector_weight: float = 0.7,
    metadata_weight: float = 0.3,
    time_normalization_hours: float = 100.0
) -> List[Dict[str, Any]]:
    """
    Apply hybrid scoring combining vector similarity and metadata.

    Args:
        similar_tickets: List of tickets with vector_similarity scores
        vector_weight: Weight for vector similarity (default 0.7)
        metadata_weight: Weight for metadata (default 0.3)
        time_normalization_hours: Max hours for time normalization

    Returns:
        List of tickets with added similarity_score and metadata_score,
        sorted by hybrid score descending
    """
```

#### `get_index_stats`

Returns statistics about the FAISS index.

```python
@tool
def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the FAISS index.

    Returns:
        Dict containing:
        - total_vectors: Number of vectors in index
        - dimension: Vector dimension (3072)
        - domain_distribution: Count per domain
        - metadata_entries: Number of metadata entries
    """
```

#### Helper Functions

| Function | Description |
|----------|-------------|
| `_get_project_root()` | Get project root directory |
| `_ensure_index_loaded()` | Lazy load FAISS index and metadata |
| `_generate_embedding()` | Generate embedding using OpenAI |

---

### Agent (`agent.py`)

LangGraph node wrapper that orchestrates the search pipeline.

#### `retrieval_node`

```python
async def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for pattern recognition / retrieval.

    Args:
        state: Current workflow state with ticket info and classified domain

    Returns:
        Partial state update with:
        - similar_tickets: List of similar tickets with scores
        - similarity_scores: List of similarity scores
        - search_metadata: Search operation metadata
        - status: "success" or "error"
        - current_agent: "retrieval"
        - messages: Status message
    """
```

**State Requirements**:
- `title`: Ticket title
- `description`: Ticket description
- `classified_domain`: Domain from classification (optional)
- `search_config`: Optional search configuration

**Search Config Options**:
```python
search_config = {
    "top_k": 20,
    "vector_weight": 0.7,
    "metadata_weight": 0.3
}
```

#### PatternRecognitionAgent

Callable wrapper with preview functionality.

```python
class PatternRecognitionAgent:
    """Callable wrapper for retrieval_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await retrieval_node(state)

    async def preview_search(
        self,
        title: str,
        description: str,
        domain: str,
        config: Any
    ) -> Dict[str, Any]:
        """Preview search with custom config (for UI tuning)."""

# Singleton instance
pattern_recognition_agent = PatternRecognitionAgent()
```

---

### Service (`service.py`)

Full-featured service class for HTTP API usage.

#### RetrievalConfig

```python
class RetrievalConfig(ComponentConfig):
    faiss_index_path: str = "data/faiss_index/tickets.index"
    faiss_metadata_path: str = "data/faiss_index/metadata.json"
    default_top_k: int = 20
    default_vector_weight: float = 0.7
    default_metadata_weight: float = 0.3
    default_time_normalization_hours: float = 100.0
    embedding_dimensions: int = 3072

    class Config:
        env_prefix = "RETRIEVAL_"
```

#### RetrievalService

```python
class RetrievalService(BaseComponent[RetrievalRequest, RetrievalResponse]):
    """Service for finding similar tickets using FAISS vector search."""

    async def process(self, request: RetrievalRequest) -> RetrievalResponse:
        """Search for similar tickets."""

    async def health_check(self) -> Dict[str, Any]:
        """Check if retrieval service is healthy."""

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
```

**Key Methods**:

| Method | Description |
|--------|-------------|
| `_ensure_index_loaded()` | Lazy load FAISS index |
| `_get_query_embedding()` | Get/generate query embedding |
| `_search_faiss()` | Execute FAISS search with domain filter |
| `_apply_hybrid_scoring()` | Apply hybrid scoring formula |

---

### Router (`router.py`)

FastAPI HTTP endpoints.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v2/retrieval/search` | Search for similar tickets |
| `GET` | `/v2/retrieval/health` | Health check |
| `GET` | `/v2/retrieval/stats` | Index statistics |

#### Example Request

```bash
curl -X POST http://localhost:8000/v2/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Database connection timeout",
    "description": "Users experiencing timeouts during peak hours",
    "domain_filter": "MM",
    "top_k": 10,
    "vector_weight": 0.8,
    "metadata_weight": 0.2
  }'
```

#### Example Response

```json
{
  "similar_tickets": [
    {
      "ticket_id": "JIRA-MM-042",
      "title": "MM_ALDER connection pool exhaustion",
      "description": "Service hitting max DB connections...",
      "similarity_score": 0.87,
      "vector_similarity": 0.92,
      "metadata_score": 0.75,
      "priority": "High",
      "labels": ["Code Fix", "#MM_ALDER"],
      "resolution_time_hours": 4.5,
      "domain": "MM",
      "resolution": "1. Increased pool size\n2. Added monitoring"
    }
  ],
  "search_metadata": {
    "query_domain": "MM",
    "total_found": 10,
    "avg_similarity": 0.72,
    "top_similarity": 0.87,
    "index_total": 300
  },
  "config_used": {
    "top_k": 10,
    "domain_filter": "MM",
    "vector_weight": 0.8,
    "metadata_weight": 0.2
  }
}
```

## FAISS Index Details

### Index Type

- **IndexFlatIP**: Exact inner product (brute-force search)
- **Normalized vectors**: L2-normalized for cosine similarity
- **Dimension**: 3072 (text-embedding-3-large)

### Files

| File | Description |
|------|-------------|
| `data/faiss_index/tickets.index` | FAISS binary index |
| `data/faiss_index/metadata.json` | Ticket metadata (title, labels, etc.) |

### Rebuilding Index

```bash
# After modifying historical_tickets.csv
python3 scripts/setup_vectorstore.py
```

## Domain Filtering

When `domain_filter` is specified:

1. **Over-fetch**: Requests `top_k × 3` results from FAISS
2. **Post-filter**: Removes tickets not matching the domain
3. **Truncate**: Returns first `top_k` matching results

This ensures sufficient results even when most tickets are filtered out.

## Usage Examples

### Direct Tool Usage

```python
from components.retrieval.tools import search_similar_tickets, apply_hybrid_scoring

# Search
search_result = await search_similar_tickets.ainvoke({
    "title": "API timeout error",
    "description": "External API calls timing out",
    "domain_filter": "CIW",
    "top_k": 15
})

# Apply scoring
scored_tickets = apply_hybrid_scoring.invoke({
    "similar_tickets": search_result["similar_tickets"],
    "vector_weight": 0.8,
    "metadata_weight": 0.2
})
```

### Service Usage

```python
from components.retrieval.service import RetrievalService
from components.retrieval.models import RetrievalRequest

service = RetrievalService()
response = await service.process(
    RetrievalRequest(
        title="Login authentication failure",
        description="Users unable to authenticate",
        top_k=20,
        domain_filter="MM"
    )
)

for ticket in response.similar_tickets[:5]:
    print(f"{ticket.ticket_id}: {ticket.similarity_score:.2%}")
```

### LangGraph Workflow

```python
from components.retrieval.agent import retrieval_node

state = {
    "ticket_id": "NEW-001",
    "title": "Database connection error",
    "description": "Timeout connecting to member DB",
    "classified_domain": "MM",
    "search_config": {"top_k": 20}
}

result = await retrieval_node(state)
print(f"Found {len(result['similar_tickets'])} similar tickets")
```

## Performance

- **Index Load Time**: ~100ms (lazy loading, cached)
- **Search Time**: <10ms (FAISS brute-force)
- **Embedding Generation**: ~500ms (API call)
- **Total Pipeline**: ~600ms per search

## Error Handling

```python
# Index not found
ConfigurationError(
    "FAISS index not found at: path. Run scripts/setup_vectorstore.py"
)

# No query input
ProcessingError(
    "Either query_embedding, query_text, or title/description required"
)
```

## Configuration

### Environment Variables

```bash
RETRIEVAL_FAISS_INDEX_PATH=data/faiss_index/tickets.index
RETRIEVAL_FAISS_METADATA_PATH=data/faiss_index/metadata.json
RETRIEVAL_DEFAULT_TOP_K=20
```

### Tuning Hybrid Scoring

For different use cases, adjust the weights:

| Use Case | Vector Weight | Metadata Weight |
|----------|---------------|-----------------|
| **Semantic Match** | 0.9 | 0.1 |
| **Balanced** (default) | 0.7 | 0.3 |
| **Priority-Focused** | 0.5 | 0.5 |
| **Fast Resolution** | 0.6 | 0.4 |

## Lazy Loading

The FAISS index is loaded on first use:

```python
# Module-level cache
_faiss_index: Optional[faiss.Index] = None
_metadata: List[Dict[str, Any]] = []
_index_loaded = False

def _ensure_index_loaded():
    """Lazy load - only loads once."""
    global _faiss_index, _metadata, _index_loaded
    if _index_loaded:
        return _faiss_index, _metadata
    # Load from disk...
    _index_loaded = True
```

This prevents slow startup and unnecessary memory usage when the component isn't used.
