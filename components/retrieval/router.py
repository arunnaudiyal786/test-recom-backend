"""
FastAPI router for Retrieval Service.

Provides HTTP endpoints for similarity search.
Mount this router on your FastAPI app to expose retrieval APIs.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any

from components.retrieval.models import RetrievalRequest, RetrievalResponse
from components.retrieval.service import RetrievalService, RetrievalConfig
from components.base.exceptions import ComponentError

# Create router
router = APIRouter(
    prefix="/retrieval",
    tags=["Retrieval"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
    },
)

# Singleton service instance
_service: Optional[RetrievalService] = None


def get_service() -> RetrievalService:
    """Dependency injection for retrieval service."""
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="Search for similar tickets",
    description="Search FAISS index for similar historical tickets using hybrid scoring.",
)
async def search_similar_tickets(
    request: RetrievalRequest,
    service: RetrievalService = Depends(get_service),
) -> RetrievalResponse:
    """
    Search for similar tickets.

    Query can be provided as:
    - `query_text`: Raw text string
    - `query_embedding`: Pre-computed embedding vector
    - `title` + `description`: Structured ticket input

    Search can be filtered by domain and customized with scoring weights.
    """
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/health",
    summary="Check retrieval service health",
    description="Verify the retrieval service is operational and index is loaded.",
)
async def health_check(
    service: RetrievalService = Depends(get_service),
) -> Dict[str, Any]:
    """Check retrieval service health."""
    return await service.health_check()


@router.get(
    "/stats",
    summary="Get index statistics",
    description="Get statistics about the FAISS index (vector count, domain distribution).",
)
async def get_stats(
    service: RetrievalService = Depends(get_service),
) -> Dict[str, Any]:
    """Get FAISS index statistics."""
    try:
        return service.get_index_stats()
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
