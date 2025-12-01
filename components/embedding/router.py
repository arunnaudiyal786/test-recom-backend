"""
FastAPI router for Embedding Service.

Provides HTTP endpoints for embedding generation.
Mount this router on your FastAPI app to expose embedding APIs.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from components.embedding.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)
from components.embedding.service import EmbeddingService, EmbeddingConfig
from components.base.exceptions import ComponentError

# Create router
router = APIRouter(
    prefix="/embedding",
    tags=["Embedding"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
    },
)

# Singleton service instance (created on first request)
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
    description="Generate an embedding vector for the provided text or ticket title/description.",
)
async def generate_embedding(
    request: EmbeddingRequest,
    service: EmbeddingService = Depends(get_service),
) -> EmbeddingResponse:
    """
    Generate embedding for a single text.

    Accepts either:
    - `text`: Raw text string
    - `title` + `description`: Ticket-style input (will be combined)
    """
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
    description="Generate embedding vectors for a batch of texts (max 100).",
)
async def generate_batch_embeddings(
    request: BatchEmbeddingRequest,
    service: EmbeddingService = Depends(get_service),
) -> BatchEmbeddingResponse:
    """
    Generate embeddings for multiple texts.

    Texts are processed in batches for efficiency.
    """
    try:
        return await service.process_batch(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/health",
    summary="Check embedding service health",
    description="Verify the embedding service is operational and can connect to OpenAI.",
)
async def health_check(
    service: EmbeddingService = Depends(get_service),
) -> dict:
    """Check embedding service health."""
    return await service.health_check()
