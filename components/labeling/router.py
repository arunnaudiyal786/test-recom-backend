"""
FastAPI router for Labeling Service.

Provides HTTP endpoints for label assignment.
Mount this router on your FastAPI app to expose labeling APIs.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any

from components.labeling.models import LabelingRequest, LabelingResponse
from components.labeling.service import LabelingService, LabelingConfig
from components.base.exceptions import ComponentError

# Create router
router = APIRouter(
    prefix="/labeling",
    tags=["Labeling"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
    },
)

# Singleton service instance
_service: Optional[LabelingService] = None


def get_service() -> LabelingService:
    """Dependency injection for labeling service."""
    global _service
    if _service is None:
        _service = LabelingService()
    return _service


@router.post(
    "/assign",
    response_model=LabelingResponse,
    summary="Assign labels to a ticket",
    description="Assign labels using historical patterns and AI-generated business/technical labels.",
)
async def assign_labels(
    request: LabelingRequest,
    service: LabelingService = Depends(get_service),
) -> LabelingResponse:
    """
    Assign labels to a ticket.

    Uses three-tier labeling:
    1. Historical labels from similar tickets (validated)
    2. Business labels (AI-generated)
    3. Technical labels (AI-generated)

    Requires similar_tickets from the retrieval component.
    """
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/health",
    summary="Check labeling service health",
    description="Verify the labeling service is operational.",
)
async def health_check(
    service: LabelingService = Depends(get_service),
) -> Dict[str, Any]:
    """Check labeling service health."""
    return await service.health_check()
