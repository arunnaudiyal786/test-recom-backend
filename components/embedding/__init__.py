"""
Embedding Service Component.

Generates text embeddings using OpenAI's embedding models.

Usage:
    # As a Python module
    from components.embedding import EmbeddingService, EmbeddingRequest

    service = EmbeddingService()
    response = await service.process(
        EmbeddingRequest(text="Database connection error")
    )
    print(response.embedding)  # [0.1, 0.2, ...]

    # As FastAPI router
    from components.embedding import router
    app.include_router(router, prefix="/v2")
"""

from components.embedding.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)
from components.embedding.service import EmbeddingService, EmbeddingConfig
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
