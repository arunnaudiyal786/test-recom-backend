"""
Pydantic models for the Labeling component.

Defines request/response contracts for label assignment.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LabelWithConfidence(BaseModel):
    """A label with its confidence score and metadata."""

    label: str = Field(description="Label name")
    confidence: float = Field(description="Confidence score (0-1)")
    category: str = Field(
        description="Label category: 'category', 'business', or 'technical'"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Reasoning for this label assignment"
    )


class CategoryLabel(BaseModel):
    """A category assignment from the predefined taxonomy."""

    id: str = Field(description="Category ID from taxonomy (e.g., 'batch_enrollment_maintenance')")
    name: str = Field(description="Human-readable category name")
    confidence: float = Field(description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(default=None, description="Why this category applies")


class SimilarTicketInput(BaseModel):
    """Simplified similar ticket for labeling input."""

    ticket_id: str
    title: str
    description: str
    labels: List[str] = Field(default_factory=list)
    priority: str = "Medium"
    resolution: Optional[str] = None


class LabelingRequest(BaseModel):
    """Request model for label assignment."""

    title: str = Field(description="Ticket title")
    description: str = Field(description="Ticket description")
    domain: str = Field(description="Classified domain (MM, CIW, Specialty)")
    priority: str = Field(default="Medium", description="Ticket priority")
    similar_tickets: List[Dict] = Field(
        description="Similar tickets from retrieval (used for historical labels)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "title": "Database connection timeout",
                    "description": "MM_ALDER experiencing timeouts",
                    "domain": "MM",
                    "priority": "High",
                    "similar_tickets": [
                        {
                            "ticket_id": "JIRA-MM-001",
                            "title": "Similar DB issue",
                            "labels": ["Code Fix", "#MM_ALDER"],
                        }
                    ],
                }
            ]
        }


class LabelingResponse(BaseModel):
    """Response model for label assignment."""

    category_labels: List[CategoryLabel] = Field(
        description="Categories from predefined taxonomy (replaces historical_labels)"
    )
    business_labels: List[LabelWithConfidence] = Field(
        description="AI-generated business-oriented labels"
    )
    technical_labels: List[LabelWithConfidence] = Field(
        description="AI-generated technical labels"
    )
    all_labels: List[str] = Field(
        description="Combined unique labels for convenience"
    )
    novelty_detected: bool = Field(
        default=False,
        description="True if ticket doesn't fit well into any category"
    )
    novelty_reasoning: Optional[str] = Field(
        default=None,
        description="Explanation if novelty was detected"
    )
