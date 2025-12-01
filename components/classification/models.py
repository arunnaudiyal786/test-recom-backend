"""
Pydantic models for the Classification component.

Defines request/response contracts for domain classification.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DomainScore(BaseModel):
    """Classification score for a single domain."""

    domain: str = Field(description="Domain name (MM, CIW, Specialty)")
    decision: bool = Field(description="Whether this domain matches (binary decision)")
    confidence: float = Field(description="Confidence score (0-1)")
    reasoning: str = Field(description="Explanation for the classification decision")
    keywords: List[str] = Field(
        default_factory=list, description="Keywords extracted for this domain"
    )


class ClassificationRequest(BaseModel):
    """Request model for ticket classification."""

    title: str = Field(
        min_length=1,
        description="Ticket title/summary",
    )
    description: str = Field(
        min_length=1,
        description="Ticket description with full context",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "title": "MM_ALDER service connection timeout",
                    "description": "The MM_ALDER service is experiencing connection timeouts "
                    "when trying to connect to the member database during peak hours.",
                },
                {
                    "title": "Claims submission validation errors",
                    "description": "CIW integration is rejecting claims with error code CIW-5001 "
                    "during provider eligibility verification.",
                },
            ]
        }


class ClassificationResponse(BaseModel):
    """Response model for ticket classification."""

    classified_domain: str = Field(
        description="Final classified domain (MM, CIW, or Specialty)"
    )
    confidence: float = Field(description="Confidence in the classification (0-1)")
    reasoning: str = Field(
        description="Combined reasoning explaining the classification decision"
    )
    domain_scores: Dict[str, DomainScore] = Field(
        description="Individual scores for each domain classifier"
    )
    extracted_keywords: List[str] = Field(
        description="All unique keywords extracted during classification"
    )
