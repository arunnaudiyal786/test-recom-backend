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
        description="Label category: 'historical', 'business', or 'technical'"
    )
    reasoning: Optional[str] = Field(
        default=None, description="Reasoning for this label assignment"
    )


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

    historical_labels: List[LabelWithConfidence] = Field(
        description="Labels derived from similar historical tickets"
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
    label_distribution: Dict[str, str] = Field(
        description="Distribution of labels in similar tickets (e.g., '14/20')"
    )
