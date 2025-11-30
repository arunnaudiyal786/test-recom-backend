"""
Structured output schemas for LLM responses.
Used with JsonOutputParser for type-safe LLM outputs.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class BinaryClassifierOutput(BaseModel):
    """Output for binary classification decisions (Yes/No)."""

    decision: bool = Field(..., description="True if condition met, False otherwise")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(..., description="Step-by-step explanation of decision")
    extracted_keywords: List[str] = Field(
        default_factory=list,
        description="Key terms that influenced the decision"
    )


class DomainClassificationOutput(BaseModel):
    """Output for domain classification with all three domains."""

    mm_decision: bool = Field(..., description="Is this an MM domain ticket?")
    mm_confidence: float = Field(..., ge=0.0, le=1.0)
    mm_reasoning: str

    ciw_decision: bool = Field(..., description="Is this a CIW domain ticket?")
    ciw_confidence: float = Field(..., ge=0.0, le=1.0)
    ciw_reasoning: str

    specialty_decision: bool = Field(..., description="Is this a Specialty domain ticket?")
    specialty_confidence: float = Field(..., ge=0.0, le=1.0)
    specialty_reasoning: str

    extracted_keywords: List[str] = Field(default_factory=list)


class LabelClassifierOutput(BaseModel):
    """Output for individual label classification."""

    assign_label: bool = Field(..., description="Should this label be assigned?")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation for label assignment decision")
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Specific evidence from ticket supporting this label"
    )


class PatternAnalysisOutput(BaseModel):
    """Output for pattern recognition analysis."""

    common_patterns: List[str] = Field(
        ...,
        description="Common patterns identified across similar tickets"
    )
    root_causes: List[str] = Field(
        ...,
        description="Potential root causes based on historical analysis"
    )
    recurring_themes: List[str] = Field(
        ...,
        description="Recurring themes in similar tickets"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class DiagnosticStepOutput(BaseModel):
    """Individual diagnostic step in resolution plan."""

    step_number: int
    description: str
    commands: List[str] = Field(default_factory=list)
    expected_output: str
    estimated_time_minutes: int


class ResolutionStepOutput(BaseModel):
    """Individual resolution step in resolution plan."""

    step_number: int
    description: str
    commands: List[str] = Field(default_factory=list)
    validation: str = Field(..., description="How to validate this step succeeded")
    estimated_time_minutes: int
    risk_level: str = Field(default="low", pattern="^(low|medium|high)$")
    rollback_procedure: Optional[str] = None


class TicketReferenceOutput(BaseModel):
    """Reference to similar historical ticket."""

    ticket_id: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    note: str = Field(..., description="Brief note on why this ticket is relevant")


class ResolutionPlanOutput(BaseModel):
    """Complete structured resolution plan."""

    summary: str = Field(..., description="Brief overview (2-3 sentences) of approach")
    diagnostic_steps: List[DiagnosticStepOutput]
    resolution_steps: List[ResolutionStepOutput]
    additional_considerations: List[str] = Field(default_factory=list)
    references: List[TicketReferenceOutput]
    total_estimated_time_hours: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternative_approaches: List[str] = Field(default_factory=list)
