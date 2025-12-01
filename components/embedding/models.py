"""
Pydantic models for the Embedding component.

Defines request/response contracts for embedding generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for single embedding generation."""

    # Option 1: Raw text
    text: Optional[str] = Field(
        default=None,
        description="Raw text to embed. Use this OR title+description.",
    )

    # Option 2: Structured ticket input
    title: Optional[str] = Field(
        default=None,
        description="Ticket title (combined with description if both provided)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Ticket description (combined with title if both provided)",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "text": "Database connection timeout error in production",
                },
                {
                    "title": "DB Connection Error",
                    "description": "Users experiencing timeout when connecting to MySQL database",
                },
            ]
        }


class EmbeddingResponse(BaseModel):
    """Response model for single embedding generation."""

    embedding: List[float] = Field(
        description="Embedding vector as list of floats",
    )
    model: str = Field(
        description="Model used for embedding generation",
    )
    dimensions: int = Field(
        description="Dimensionality of the embedding vector",
    )
    input_text: str = Field(
        description="The text that was embedded (for verification)",
    )


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding generation."""

    texts: List[str] = Field(
        min_length=1,
        max_length=100,
        description="List of texts to embed (max 100)",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of texts to process concurrently",
    )


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""

    embeddings: List[List[float]] = Field(
        description="List of embedding vectors",
    )
    model: str = Field(
        description="Model used for embedding generation",
    )
    dimensions: int = Field(
        description="Dimensionality of each embedding vector",
    )
    count: int = Field(
        description="Number of embeddings generated",
    )
