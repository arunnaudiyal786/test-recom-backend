"""
Resolution Component Models.

Pydantic models for resolution plan generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DiagnosticStep(BaseModel):
    """A single diagnostic step."""
    step_number: int
    description: str
    commands: List[str] = Field(default_factory=list)
    expected_output: str = ""
    estimated_time_minutes: int = 5


class ResolutionStep(BaseModel):
    """A single resolution step extracted from similar historical tickets."""
    step_number: int
    description: str
    commands: List[str] = Field(default_factory=list)
    validation: str = "Verify step completed"
    estimated_time_minutes: int = 10
    risk_level: str = "low"
    rollback_procedure: Optional[str] = None
    source_ticket: Optional[str] = Field(default=None, description="Source ticket ID this step was extracted from")
    source_similarity: Optional[float] = Field(default=None, description="Similarity percentage of the source ticket")


class ResolutionPlan(BaseModel):
    """Complete resolution plan."""
    summary: str
    diagnostic_steps: List[DiagnosticStep] = Field(default_factory=list)
    resolution_steps: List[ResolutionStep] = Field(default_factory=list)
    additional_considerations: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    total_estimated_time_hours: float = 0.0
    confidence: float = 0.5
    alternative_approaches: List[str] = Field(default_factory=list)


class ResolutionRequest(BaseModel):
    """Request model for resolution generation."""
    title: str
    description: str
    domain: str
    priority: str = "Medium"
    labels: List[str] = Field(default_factory=list)
    similar_tickets: List[dict] = Field(default_factory=list)
    avg_similarity: float = 0.0


class ResolutionResponse(BaseModel):
    """Response model for resolution generation."""
    resolution_plan: ResolutionPlan
    confidence: float
