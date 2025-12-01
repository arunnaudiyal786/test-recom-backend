"""
Retrieval Service Component.

Searches FAISS vector index for similar historical tickets
using hybrid scoring (vector similarity + metadata relevance).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss

from components.base import BaseComponent, ComponentConfig
from components.base.exceptions import ProcessingError, ConfigurationError
from components.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
    SimilarTicket,
    SearchMetadata,
    PriorityWeights,
)
from components.embedding import EmbeddingService, EmbeddingRequest


class RetrievalConfig(ComponentConfig):
    """Configuration for Retrieval Service."""

    # FAISS index paths
    faiss_index_path: str = "data/faiss_index/tickets.index"
    faiss_metadata_path: str = "data/faiss_index/metadata.json"

    # Default search parameters
    default_top_k: int = 20
    default_vector_weight: float = 0.7
    default_metadata_weight: float = 0.3
    default_time_normalization_hours: float = 100.0

    # Embedding dimensions (must match index)
    embedding_dimensions: int = 3072

    class Config:
        env_prefix = "RETRIEVAL_"


class RetrievalService(BaseComponent[RetrievalRequest, RetrievalResponse]):
    """
    Service for finding similar tickets using FAISS vector search.

    Features:
    - Vector similarity search using FAISS
    - Domain filtering
    - Hybrid scoring (vector + metadata)
    - Configurable weights and parameters

    Usage:
        service = RetrievalService()
        response = await service.process(
            RetrievalRequest(
                title="Database error",
                description="Connection timeout",
                domain_filter="MM",
                top_k=10
            )
        )
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize the retrieval service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
            embedding_service: Optional embedding service for text-based queries.
                              Created automatically if not provided.
        """
        self.config = config or RetrievalConfig()
        self._embedding_service = embedding_service
        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []
        self._index_loaded = False

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy initialization of embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def component_name(self) -> str:
        return "retrieval"

    def _get_index_path(self) -> Path:
        """Get resolved index path."""
        path = Path(self.config.faiss_index_path)
        if not path.is_absolute():
            # Relative to project root
            path = Path(__file__).parent.parent.parent / path
        return path

    def _get_metadata_path(self) -> Path:
        """Get resolved metadata path."""
        path = Path(self.config.faiss_metadata_path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / path
        return path

    def _ensure_index_loaded(self) -> None:
        """Ensure FAISS index is loaded (lazy loading)."""
        if self._index_loaded:
            return

        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()

        if not index_path.exists():
            raise ConfigurationError(
                f"FAISS index not found at: {index_path}. "
                "Run scripts/setup_vectorstore.py to create the index.",
                component=self.component_name,
            )

        if not metadata_path.exists():
            raise ConfigurationError(
                f"Metadata file not found at: {metadata_path}",
                component=self.component_name,
            )

        # Load FAISS index
        self._index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, "r") as f:
            self._metadata = json.load(f)

        self._index_loaded = True

    async def _get_query_embedding(self, request: RetrievalRequest) -> List[float]:
        """Get or generate embedding from request."""
        # If embedding provided, use it directly
        if request.query_embedding:
            return request.query_embedding

        # Generate embedding from text
        if request.query_text:
            embed_request = EmbeddingRequest(text=request.query_text)
        elif request.title or request.description:
            embed_request = EmbeddingRequest(
                title=request.title,
                description=request.description,
            )
        else:
            raise ProcessingError(
                "Either query_embedding, query_text, or title/description required",
                component=self.component_name,
                stage="input_validation",
            )

        response = await self.embedding_service.process(embed_request)
        return response.embedding

    def _search_faiss(
        self,
        query_embedding: List[float],
        k: int,
        domain_filter: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search FAISS index for similar tickets.

        Args:
            query_embedding: Query vector
            k: Number of results
            domain_filter: Optional domain filter

        Returns:
            Tuple of (ticket_metadata_list, similarity_scores)
        """
        self._ensure_index_loaded()

        # Normalize query vector for cosine similarity
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Request more results if filtering by domain
        search_k = k * 3 if domain_filter else k
        distances, indices = self._index.search(query_array, search_k)

        results = []
        scores = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS padding
                continue

            ticket_data = self._metadata[idx].copy()

            # Apply domain filter
            if domain_filter and ticket_data.get("domain") != domain_filter:
                continue

            results.append(ticket_data)
            scores.append(float(distance))

            if len(results) >= k:
                break

        return results, scores

    def _apply_hybrid_scoring(
        self,
        tickets: List[Dict[str, Any]],
        scores: List[float],
        request: RetrievalRequest,
    ) -> List[SimilarTicket]:
        """
        Apply hybrid scoring and return SimilarTicket models.

        Hybrid score = (vector_weight * vector_similarity) + (metadata_weight * metadata_score)
        Metadata score = (priority_score * 0.6) + (time_score * 0.4)
        """
        # Get priority weights
        priority_weights = request.priority_weights or PriorityWeights()
        priority_scores = {
            "Critical": priority_weights.Critical,
            "High": priority_weights.High,
            "Medium": priority_weights.Medium,
            "Low": priority_weights.Low,
        }

        results = []

        for i, ticket in enumerate(tickets):
            vector_score = scores[i]

            # Metadata factors
            priority = ticket.get("priority", "Medium")
            priority_score = priority_scores.get(priority, 0.5)

            # Resolution time factor (faster = better)
            res_time = ticket.get("resolution_time_hours", 24)
            time_score = max(0, 1 - (res_time / request.time_normalization_hours))

            # Combined metadata score (60% priority, 40% time)
            metadata_score = (priority_score * 0.6) + (time_score * 0.4)

            # Hybrid score
            hybrid_score = (request.vector_weight * vector_score) + (
                request.metadata_weight * metadata_score
            )

            # Create SimilarTicket model
            results.append(
                SimilarTicket(
                    ticket_id=ticket.get("ticket_id", ""),
                    title=ticket.get("title", ""),
                    description=ticket.get("description", ""),
                    similarity_score=hybrid_score,
                    vector_similarity=vector_score,
                    metadata_score=metadata_score,
                    priority=priority,
                    labels=ticket.get("labels", []),
                    resolution_time_hours=res_time,
                    domain=ticket.get("domain", ""),
                    resolution=ticket.get("resolution"),
                )
            )

        # Sort by hybrid score
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        return results

    async def process(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Search for similar tickets.

        Args:
            request: RetrievalRequest with query and search parameters

        Returns:
            RetrievalResponse with similar tickets and metadata
        """
        # Ensure index is loaded
        self._ensure_index_loaded()

        # Get query embedding
        query_embedding = await self._get_query_embedding(request)

        # Search FAISS
        tickets, scores = self._search_faiss(
            query_embedding=query_embedding,
            k=request.top_k,
            domain_filter=request.domain_filter,
        )

        # Apply hybrid scoring
        similar_tickets = self._apply_hybrid_scoring(tickets, scores, request)

        # Calculate metadata
        avg_similarity = (
            sum(t.similarity_score for t in similar_tickets) / len(similar_tickets)
            if similar_tickets
            else 0
        )
        top_similarity = similar_tickets[0].similarity_score if similar_tickets else 0

        search_metadata = SearchMetadata(
            query_domain=request.domain_filter,
            total_found=len(similar_tickets),
            avg_similarity=avg_similarity,
            top_similarity=top_similarity,
            index_total=self._index.ntotal if self._index else 0,
        )

        config_used = {
            "top_k": request.top_k,
            "domain_filter": request.domain_filter,
            "vector_weight": request.vector_weight,
            "metadata_weight": request.metadata_weight,
            "time_normalization_hours": request.time_normalization_hours,
        }

        return RetrievalResponse(
            similar_tickets=similar_tickets,
            search_metadata=search_metadata,
            config_used=config_used,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check if retrieval service is healthy."""
        try:
            self._ensure_index_loaded()

            return {
                "status": "healthy",
                "component": self.component_name,
                "index_loaded": self._index_loaded,
                "total_vectors": self._index.ntotal if self._index else 0,
                "metadata_count": len(self._metadata),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "component": self.component_name,
                "error": str(e),
            }

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        self._ensure_index_loaded()

        domain_counts = {}
        for ticket in self._metadata:
            domain = ticket.get("domain", "Unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "dimension": self.config.embedding_dimensions,
            "domain_distribution": domain_counts,
            "metadata_entries": len(self._metadata),
        }
