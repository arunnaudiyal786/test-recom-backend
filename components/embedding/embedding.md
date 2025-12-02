# Embedding Component

The Embedding component generates text embeddings using OpenAI's embedding models. It serves as a foundational utility used by the Retrieval component and can also be used standalone via HTTP API.

## Overview

This component provides:
- Single text embedding generation
- Batch embedding generation with concurrent processing
- Support for both raw text and structured (title + description) input
- Built-in retry logic with exponential backoff

## Architecture

```
embedding/
├── __init__.py          # Public API exports
├── models.py            # Pydantic request/response models
├── service.py           # EmbeddingService
├── router.py            # FastAPI HTTP endpoints
└── README.md            # This file
```

## Embedding Model

| Property | Value |
|----------|-------|
| Model | `text-embedding-3-large` |
| Dimensions | 3072 |
| Max Input | 8191 tokens |
| Encoding | Float |

## Components

### Models (`models.py`)

#### EmbeddingRequest

Request for single embedding generation.

```python
class EmbeddingRequest(BaseModel):
    # Option 1: Raw text
    text: Optional[str] = None

    # Option 2: Structured ticket input
    title: Optional[str] = None
    description: Optional[str] = None
```

**Usage**:
- Provide `text` for raw text embedding
- Provide `title` and/or `description` for structured input (combined as "Title: {title}\nDescription: {description}")

#### EmbeddingResponse

Response with generated embedding.

```python
class EmbeddingResponse(BaseModel):
    embedding: List[float]    # Embedding vector (3072 dimensions)
    model: str                # Model used
    dimensions: int           # Vector dimensionality
    input_text: str           # Text that was embedded (truncated)
```

#### BatchEmbeddingRequest

Request for batch embedding generation.

```python
class BatchEmbeddingRequest(BaseModel):
    texts: List[str]          # List of texts (max 100)
    batch_size: int = 10      # Concurrent processing batch size (1-50)
```

#### BatchEmbeddingResponse

Response with batch embeddings.

```python
class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]  # List of embedding vectors
    model: str                     # Model used
    dimensions: int                # Vector dimensionality
    count: int                     # Number of embeddings generated
```

---

### Service (`service.py`)

The main service class that handles embedding generation.

#### EmbeddingConfig

```python
class EmbeddingConfig(ComponentConfig):
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    class Config:
        env_prefix = "EMBEDDING_"
```

| Setting | Environment Variable | Default |
|---------|---------------------|---------|
| Model | `EMBEDDING_MODEL` | `text-embedding-3-large` |
| Dimensions | `EMBEDDING_DIMENSIONS` | `3072` |

#### EmbeddingService

```python
class EmbeddingService(BaseComponent[EmbeddingRequest, EmbeddingResponse]):
    """Service for generating text embeddings using OpenAI."""

    async def process(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embedding for a single text."""

    async def process_batch(self, request: BatchEmbeddingRequest) -> BatchEmbeddingResponse:
        """Generate embeddings for multiple texts."""

    async def health_check(self) -> Dict[str, Any]:
        """Check if embedding service is healthy."""
```

**Key Methods**:

| Method | Description |
|--------|-------------|
| `_combine_text()` | Combine title and description into single text |
| `_clean_text()` | Remove excess whitespace from text |
| `_get_input_text()` | Extract text from request (handles both formats) |
| `_generate_embedding_with_retry()` | Generate embedding with exponential backoff |

---

### Router (`router.py`)

FastAPI HTTP endpoints.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v2/embedding/generate` | Generate single embedding |
| `POST` | `/v2/embedding/batch` | Generate batch embeddings |
| `GET` | `/v2/embedding/health` | Health check |

#### Single Embedding Request

```bash
curl -X POST http://localhost:8000/v2/embedding/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Database connection timeout error in production"
  }'
```

#### Structured Input Request

```bash
curl -X POST http://localhost:8000/v2/embedding/generate \
  -H "Content-Type: application/json" \
  -d '{
    "title": "DB Connection Error",
    "description": "Users experiencing timeout when connecting to MySQL database"
  }'
```

#### Batch Request

```bash
curl -X POST http://localhost:8000/v2/embedding/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First ticket description",
      "Second ticket description",
      "Third ticket description"
    ],
    "batch_size": 10
  }'
```

#### Single Embedding Response

```json
{
  "embedding": [0.0123, -0.0456, 0.0789, ...],
  "model": "text-embedding-3-large",
  "dimensions": 3072,
  "input_text": "Database connection timeout error in production"
}
```

#### Batch Response

```json
{
  "embeddings": [
    [0.0123, -0.0456, ...],
    [0.0789, -0.0123, ...],
    [0.0456, -0.0789, ...]
  ],
  "model": "text-embedding-3-large",
  "dimensions": 3072,
  "count": 3
}
```

## Usage Examples

### Basic Usage

```python
from components.embedding import EmbeddingService, EmbeddingRequest

# Create service
service = EmbeddingService()

# Generate embedding from text
response = await service.process(
    EmbeddingRequest(text="Database connection error")
)

print(f"Vector dimensions: {response.dimensions}")  # 3072
print(f"First 5 values: {response.embedding[:5]}")
```

### Structured Input

```python
# Generate embedding from title + description
response = await service.process(
    EmbeddingRequest(
        title="MM_ALDER timeout",
        description="Connection pool exhaustion during peak hours"
    )
)

# The input is combined as:
# "Title: MM_ALDER timeout\nDescription: Connection pool exhaustion..."
```

### Batch Processing

```python
from components.embedding.models import BatchEmbeddingRequest

# Generate multiple embeddings
texts = [
    "First ticket: Login failure",
    "Second ticket: API timeout",
    "Third ticket: Database error"
]

batch_response = await service.process_batch(
    BatchEmbeddingRequest(texts=texts, batch_size=10)
)

print(f"Generated {batch_response.count} embeddings")
for i, embedding in enumerate(batch_response.embeddings):
    print(f"  Text {i+1}: {len(embedding)} dimensions")
```

### With Custom Config

```python
from components.embedding.service import EmbeddingService, EmbeddingConfig

# Use smaller model (faster, cheaper)
config = EmbeddingConfig(
    embedding_model="text-embedding-3-small"
)
service = EmbeddingService(config)
```

## Retry Logic

The service implements exponential backoff for transient failures:

```python
async def _generate_embedding_with_retry(self, text: str) -> List[float]:
    for attempt in range(self.config.max_retries):  # Default: 3
        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding

        except (RateLimitError, APITimeoutError) as e:
            if attempt < self.config.max_retries - 1:
                wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                # Wait: 2s, 4s, 8s...
                await asyncio.sleep(wait_time)
            else:
                raise ProcessingError(f"Max retries exceeded: {e}")
```

## Batch Processing Implementation

Batch embedding uses concurrent processing for efficiency:

```python
async def process_batch(self, request: BatchEmbeddingRequest) -> BatchEmbeddingResponse:
    embeddings = []

    for i in range(0, len(request.texts), request.batch_size):
        batch = request.texts[i : i + request.batch_size]

        # Process batch concurrently
        batch_embeddings = await asyncio.gather(
            *[
                self._generate_embedding_with_retry(self._clean_text(text))
                for text in batch
            ]
        )

        embeddings.extend(batch_embeddings)

    return BatchEmbeddingResponse(...)
```

**Performance**:
- Processes `batch_size` texts concurrently
- Default batch size: 10
- Maximum batch size: 50
- Maximum total texts: 100

## Integration with Retrieval

The Retrieval component uses EmbeddingService internally:

```python
from components.embedding import EmbeddingService, EmbeddingRequest

class RetrievalService:
    def __init__(self):
        self._embedding_service = None

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy initialization of embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    async def _get_query_embedding(self, request):
        # Use pre-computed embedding if provided
        if request.query_embedding:
            return request.query_embedding

        # Otherwise, generate embedding
        embed_request = EmbeddingRequest(
            title=request.title,
            description=request.description
        )
        response = await self.embedding_service.process(embed_request)
        return response.embedding
```

## Text Cleaning

Input text is cleaned before embedding:

```python
def _clean_text(self, text: str) -> str:
    """Clean text for embedding (remove excess whitespace)."""
    return " ".join(text.split())
```

This normalizes whitespace and removes leading/trailing spaces.

## Text Combination

For structured input, title and description are combined:

```python
def _combine_text(self, title: Optional[str], description: Optional[str]) -> str:
    """Combine title and description into single text."""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    return "\n".join(parts)
```

**Output format**:
```
Title: MM_ALDER timeout
Description: Connection pool exhaustion during peak hours
```

## Health Check

The health check verifies OpenAI connectivity:

```python
async def health_check(self) -> Dict[str, Any]:
    try:
        # Test embedding generation
        test_response = await self.client.embeddings.create(
            model=self.config.embedding_model,
            input="health check",
            encoding_format="float",
        )

        return {
            "status": "healthy",
            "component": "embedding",
            "model": self.config.embedding_model,
            "dimensions": len(test_response.data[0].embedding),
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "component": "embedding",
            "error": str(e),
        }
```

## Performance

| Operation | Time | Cost |
|-----------|------|------|
| Single embedding | ~500ms | ~$0.0001 |
| Batch (10 texts) | ~1-2s | ~$0.001 |
| Batch (100 texts) | ~10-20s | ~$0.01 |

## Error Handling

```python
from components.base.exceptions import ProcessingError, ConfigurationError

# Missing API key
ConfigurationError(
    "OpenAI API key is required",
    component="embedding",
    missing_keys=["openai_api_key"]
)

# No input provided
ProcessingError(
    "Either 'text' or 'title/description' must be provided",
    component="embedding",
    stage="input_validation"
)

# API failure after retries
ProcessingError(
    "Max retries exceeded: Rate limit error",
    component="embedding",
    stage="api_call"
)
```

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
```

### Alternative Models

| Model | Dimensions | Speed | Cost |
|-------|------------|-------|------|
| `text-embedding-3-large` | 3072 | Standard | Higher |
| `text-embedding-3-small` | 1536 | Faster | Lower |
| `text-embedding-ada-002` | 1536 | Fast | Lowest |

**Note**: FAISS index must be rebuilt if embedding model changes.

## Lazy Initialization

The OpenAI client is initialized on first use:

```python
@property
def client(self) -> AsyncOpenAI:
    """Lazy initialization of OpenAI client."""
    if self._client is None:
        if not self.config.openai_api_key:
            raise ConfigurationError("OpenAI API key is required")
        self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
    return self._client
```

This prevents errors at import time and reduces startup overhead.

## Public API

The `__init__.py` exports:

```python
from components.embedding.service import EmbeddingService, EmbeddingConfig
from components.embedding.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingConfig",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "BatchEmbeddingRequest",
    "BatchEmbeddingResponse",
]
```

## Best Practices

1. **Reuse Service Instance**: Create one `EmbeddingService` and reuse it
2. **Use Batch for Multiple**: Use `process_batch()` for 5+ texts
3. **Provide Pre-computed Embeddings**: Cache embeddings when possible
4. **Handle Rate Limits**: The retry logic handles transient failures
5. **Monitor Costs**: Each embedding call has a small cost
