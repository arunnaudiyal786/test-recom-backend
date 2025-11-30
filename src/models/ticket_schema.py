"""
Pydantic models for ticket data structures.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class HistoricalTicket(BaseModel):
    """Schema for historical tickets in the database."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    title: str = Field(..., description="Ticket title/summary")
    description: str = Field(..., description="Detailed ticket description")
    domain: str = Field(..., description="Ticket domain: MM, CIW, or Specialty")
    labels: List[str] = Field(default_factory=list, description="Assigned labels")
    resolution_steps: List[str] = Field(default_factory=list, description="Resolution steps")
    priority: str = Field(..., description="Priority level: Low, Medium, High, Critical")
    created_date: str = Field(..., description="Creation date in YYYY-MM-DD format")
    resolution_time_hours: float = Field(..., description="Time to resolution in hours")

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "JIRA-001",
                "title": "Database connection timeout in MM module",
                "description": "Users experiencing timeouts when connecting to MM database...",
                "domain": "MM",
                "labels": ["Code Fix", "#MM_ALDER"],
                "resolution_steps": [
                    "1. Check database connection pool settings",
                    "2. Increase timeout configuration to 30s",
                    "3. Restart MM service"
                ],
                "priority": "High",
                "created_date": "2024-01-15",
                "resolution_time_hours": 4.0
            }
        }


class IncomingTicket(BaseModel):
    """Schema for incoming tickets to be processed."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    title: str = Field(..., description="Ticket title/summary")
    description: str = Field(..., description="Detailed ticket description")
    priority: str = Field(..., description="Priority level: Low, Medium, High, Critical")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "JIRA-NEW-001",
                "title": "MM_ALDER service failing with connection timeout",
                "description": "The MM_ALDER service is experiencing intermittent connection timeouts...",
                "priority": "High",
                "metadata": {}
            }
        }


class ClassificationResult(BaseModel):
    """Classification agent output."""

    classified_domain: str = Field(..., description="Classified domain: MM, CIW, or Specialty")
    confidence_scores: dict[str, float] = Field(..., description="Confidence scores per domain")
    reasoning: str = Field(..., description="Step-by-step classification reasoning")
    extracted_keywords: List[str] = Field(default_factory=list, description="Domain-specific keywords")


class SimilarTicket(BaseModel):
    """Schema for similar ticket returned from FAISS."""

    ticket_id: str
    title: str
    description: str
    domain: str
    labels: List[str]
    resolution_steps: List[str]
    similarity_score: float = Field(..., description="Cosine similarity score 0-1")


class PatternRecognitionResult(BaseModel):
    """Pattern recognition agent output."""

    similar_tickets: List[SimilarTicket] = Field(..., description="Top K similar tickets")
    search_metadata: dict = Field(..., description="Search statistics and metadata")


class LabelAssignmentResult(BaseModel):
    """Label assignment agent output."""

    assigned_labels: List[str] = Field(..., description="Labels assigned to ticket")
    label_confidence: dict[str, float] = Field(..., description="Confidence score per label")
    label_distribution_in_similar_tickets: dict[str, str] = Field(
        ..., description="Label frequency in similar tickets"
    )


class ResolutionStep(BaseModel):
    """Individual resolution step."""

    step_number: int
    description: str
    commands: Optional[List[str]] = Field(default_factory=list)
    validation: Optional[str] = None
    expected_output: Optional[str] = None
    estimated_time_minutes: int
    risk_level: str = Field(default="low", description="low, medium, high")
    rollback_procedure: Optional[str] = None


class TicketReference(BaseModel):
    """Reference to a similar historical ticket."""

    ticket_id: str
    similarity: float
    note: str


class ResolutionPlan(BaseModel):
    """Complete resolution plan."""

    summary: str = Field(..., description="Brief overview of the resolution approach")
    diagnostic_steps: List[ResolutionStep] = Field(default_factory=list)
    resolution_steps: List[ResolutionStep] = Field(default_factory=list)
    additional_considerations: List[str] = Field(default_factory=list)
    references: List[TicketReference] = Field(default_factory=list)
    total_estimated_time_hours: float
    confidence: float = Field(..., description="Overall confidence in resolution plan 0-1")
    alternative_approaches: List[str] = Field(default_factory=list)


class ResolutionGenerationResult(BaseModel):
    """Resolution generation agent output."""

    resolution_plan: ResolutionPlan


class FinalTicketOutput(BaseModel):
    """Final output for a processed ticket."""

    ticket_id: str
    classified_domain: str
    classification_confidence: float
    assigned_labels: List[str]
    label_confidence: dict[str, float]
    resolution_plan: ResolutionPlan
    overall_confidence: float
    processing_metadata: Optional[dict] = Field(default_factory=dict)
