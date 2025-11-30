"""
Retrieval configuration models for tuning FAISS search parameters.

These models define the configurable parameters for the Pattern Recognition Agent,
allowing users to tune retrieval behavior through the UI before processing.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict


class PriorityWeights(BaseModel):
    """Weights for priority-based scoring in hybrid retrieval."""
    Critical: float = Field(default=1.0, ge=0.0, le=1.0, description="Score multiplier for Critical priority")
    High: float = Field(default=0.8, ge=0.0, le=1.0, description="Score multiplier for High priority")
    Medium: float = Field(default=0.5, ge=0.0, le=1.0, description="Score multiplier for Medium priority")
    Low: float = Field(default=0.3, ge=0.0, le=1.0, description="Score multiplier for Low priority")


class RetrievalConfig(BaseModel):
    """
    Configuration for retrieval parameters.

    Controls how the Pattern Recognition Agent searches for and scores similar tickets.
    The hybrid scoring formula is:
        hybrid_score = (vector_weight * vector_similarity) + (metadata_weight * metadata_score)

    Where metadata_score = (priority_score * 0.6) + (time_score * 0.4)
    """
    top_k: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of similar tickets to retrieve"
    )
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid scoring (0-1)"
    )
    metadata_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for metadata relevance in hybrid scoring (0-1)"
    )
    priority_weights: PriorityWeights = Field(
        default_factory=PriorityWeights,
        description="Score multipliers for each priority level"
    )
    time_normalization_hours: float = Field(
        default=100.0,
        ge=1.0,
        le=500.0,
        description="Reference hours for resolution time normalization"
    )
    domain_filter: Optional[str] = Field(
        default=None,
        description="Force specific domain filter (MM, CIW, Specialty) or None for auto-classification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "top_k": 20,
                "vector_weight": 0.7,
                "metadata_weight": 0.3,
                "priority_weights": {
                    "Critical": 1.0,
                    "High": 0.8,
                    "Medium": 0.5,
                    "Low": 0.3
                },
                "time_normalization_hours": 100.0,
                "domain_filter": None
            }
        }


class RetrievalPreviewRequest(BaseModel):
    """Request schema for retrieval preview endpoint."""
    title: str = Field(..., min_length=1, description="Ticket title")
    description: str = Field(..., min_length=1, description="Ticket description")
    config: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="Retrieval configuration parameters"
    )


class SimilarTicketPreview(BaseModel):
    """Preview of a similar ticket with scoring details."""
    ticket_id: str
    title: str
    description: str
    similarity_score: float = Field(description="Final hybrid score (0-1)")
    vector_similarity: float = Field(description="Raw vector similarity (0-1)")
    metadata_score: float = Field(description="Metadata relevance score (0-1)")
    priority: str
    labels: list[str]
    resolution_time_hours: float
    domain: str
    resolution: Optional[str] = Field(default=None, description="Resolution steps if available")


class SearchMetadata(BaseModel):
    """Statistics about the retrieval search."""
    query_domain: str = Field(description="Domain used for filtering")
    total_found: int = Field(description="Total tickets found")
    avg_similarity: float = Field(description="Average hybrid score")
    top_similarity: float = Field(description="Highest hybrid score")
    classification_confidence: Optional[float] = Field(
        default=None,
        description="Confidence of domain classification (if auto-classified)"
    )


class RetrievalPreviewResponse(BaseModel):
    """Response schema for retrieval preview endpoint."""
    similar_tickets: list[SimilarTicketPreview]
    search_metadata: SearchMetadata
    config_used: RetrievalConfig
