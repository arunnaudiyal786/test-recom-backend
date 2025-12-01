"""
Pydantic models for the Retrieval component.

Defines request/response contracts for similarity search.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PriorityWeights(BaseModel):
    """Priority weights for hybrid scoring."""

    Critical: float = Field(default=1.0, ge=0, le=1)
    High: float = Field(default=0.8, ge=0, le=1)
    Medium: float = Field(default=0.5, ge=0, le=1)
    Low: float = Field(default=0.3, ge=0, le=1)


class SimilarTicket(BaseModel):
    """A similar ticket from the vector store."""

    ticket_id: str = Field(description="Unique ticket identifier")
    title: str = Field(description="Ticket title/summary")
    description: str = Field(description="Ticket description")
    similarity_score: float = Field(description="Final hybrid similarity score (0-1)")
    vector_similarity: float = Field(description="Raw vector similarity score")
    metadata_score: float = Field(description="Metadata relevance score")
    priority: str = Field(description="Ticket priority level")
    labels: List[str] = Field(default_factory=list, description="Assigned labels")
    resolution_time_hours: float = Field(
        default=24.0, description="Time to resolution in hours"
    )
    domain: str = Field(description="Ticket domain (MM, CIW, Specialty)")
    resolution: Optional[str] = Field(
        default=None, description="Resolution summary if available"
    )


class RetrievalRequest(BaseModel):
    """Request model for similarity search."""

    # Query input (one of these is required)
    query_text: Optional[str] = Field(
        default=None,
        description="Raw text to search (will be embedded)",
    )
    query_embedding: Optional[List[float]] = Field(
        default=None,
        description="Pre-computed embedding vector",
    )

    # OR structured ticket input
    title: Optional[str] = Field(
        default=None,
        description="Ticket title (combined with description for search)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Ticket description (combined with title for search)",
    )

    # Search parameters
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of similar tickets to return",
    )
    domain_filter: Optional[str] = Field(
        default=None,
        description="Filter by domain (MM, CIW, Specialty). None = all domains.",
    )

    # Scoring weights
    vector_weight: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Weight for vector similarity (0-1)",
    )
    metadata_weight: float = Field(
        default=0.3,
        ge=0,
        le=1,
        description="Weight for metadata score (0-1)",
    )

    # Priority weights for metadata scoring
    priority_weights: Optional[PriorityWeights] = Field(
        default=None,
        description="Custom priority weights. Uses defaults if not provided.",
    )

    # Time normalization
    time_normalization_hours: float = Field(
        default=100.0,
        gt=0,
        description="Max hours for time score normalization",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "title": "Database Connection Error",
                    "description": "Users experiencing timeout errors",
                    "top_k": 10,
                    "domain_filter": "MM",
                },
                {
                    "query_text": "Login authentication failures in production",
                    "top_k": 20,
                    "vector_weight": 0.8,
                    "metadata_weight": 0.2,
                },
            ]
        }


class SearchMetadata(BaseModel):
    """Metadata about the search operation."""

    query_domain: Optional[str] = Field(description="Domain filter applied")
    total_found: int = Field(description="Number of results returned")
    avg_similarity: float = Field(description="Average similarity score")
    top_similarity: float = Field(description="Highest similarity score")
    index_total: int = Field(description="Total vectors in index")


class RetrievalResponse(BaseModel):
    """Response model for similarity search."""

    similar_tickets: List[SimilarTicket] = Field(
        description="List of similar tickets ranked by similarity"
    )
    search_metadata: SearchMetadata = Field(
        description="Search operation metadata"
    )
    config_used: Dict[str, Any] = Field(
        description="Configuration parameters used for this search"
    )
