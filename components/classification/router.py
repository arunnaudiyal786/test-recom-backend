"""
FastAPI router for Classification Service.

Provides HTTP endpoints for ticket domain classification.
Mount this router on your FastAPI app to expose classification APIs.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any

from components.classification.models import (
    ClassificationRequest,
    ClassificationResponse,
)
from components.classification.service import ClassificationService, ClassificationConfig
from components.base.exceptions import ComponentError

# Create router
router = APIRouter(
    prefix="/classification",
    tags=["Classification"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
    },
)

# Singleton service instance
_service: Optional[ClassificationService] = None


def get_service() -> ClassificationService:
    """Dependency injection for classification service."""
    global _service
    if _service is None:
        _service = ClassificationService()
    return _service


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify ticket domain",
    description="Classify a ticket into a domain (MM, CIW, or Specialty) using parallel binary classifiers.",
)
async def classify_ticket(
    request: ClassificationRequest,
    service: ClassificationService = Depends(get_service),
) -> ClassificationResponse:
    """
    Classify a ticket's domain.

    Uses MTC-LLM approach with parallel binary classifiers:
    - MM (Member Management)
    - CIW (Claims Integration Workflow)
    - Specialty (Custom modules)

    Returns the domain with highest confidence.
    """
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/health",
    summary="Check classification service health",
    description="Verify the classification service is operational.",
)
async def health_check(
    service: ClassificationService = Depends(get_service),
) -> Dict[str, Any]:
    """Check classification service health."""
    return await service.health_check()
