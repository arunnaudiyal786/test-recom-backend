"""
Retrieval Component - LangChain agent for pattern recognition.

Searches FAISS vector index for similar historical tickets
using hybrid scoring (vector similarity + metadata relevance).

Usage:
    # As LangGraph node (preferred)
    from components.retrieval import retrieval_node
    workflow.add_node("retrieval", retrieval_node)

    # As LangChain tools
    from components.retrieval import search_similar_tickets, apply_hybrid_scoring
    result = await search_similar_tickets.ainvoke({...})

    # Legacy service (backward compatibility)
    from components.retrieval import RetrievalService, RetrievalRequest
    service = RetrievalService()
    response = await service.process(RetrievalRequest(...))
"""

from components.retrieval.models import (
    RetrievalRequest,
    RetrievalResponse,
    SimilarTicket,
    SearchMetadata,
    PriorityWeights,
)
from components.retrieval.service import RetrievalService, RetrievalConfig
from components.retrieval.router import router

# New LangChain agent and tools
from components.retrieval.agent import retrieval_node, pattern_recognition_agent
from components.retrieval.tools import search_similar_tickets, apply_hybrid_scoring, get_index_stats

__all__ = [
    # LangGraph node (primary)
    "retrieval_node",
    "pattern_recognition_agent",
    # LangChain tools
    "search_similar_tickets",
    "apply_hybrid_scoring",
    "get_index_stats",
    # Models
    "RetrievalRequest",
    "RetrievalResponse",
    "SimilarTicket",
    "SearchMetadata",
    "PriorityWeights",
    # Legacy service
    "RetrievalService",
    "RetrievalConfig",
    # Router
    "router",
]
