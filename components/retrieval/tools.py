"""
Retrieval Tools - LangChain @tool decorated functions for FAISS search.

These tools handle vector search, hybrid scoring, and similar ticket retrieval.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from config.config import Config


# Module-level cache for FAISS index and metadata
_faiss_index: Optional[faiss.Index] = None
_metadata: List[Dict[str, Any]] = []
_index_loaded = False


def _get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def _ensure_index_loaded() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Ensure FAISS index is loaded (lazy loading).

    Returns:
        Tuple of (FAISS index, metadata list)
    """
    global _faiss_index, _metadata, _index_loaded

    if _index_loaded:
        return _faiss_index, _metadata

    project_root = _get_project_root()
    index_path = project_root / "data" / "faiss_index" / "tickets.index"
    metadata_path = project_root / "data" / "faiss_index" / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at: {index_path}. "
            "Run scripts/setup_vectorstore.py to create the index."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    # Load FAISS index
    _faiss_index = faiss.read_index(str(index_path))

    # Load metadata
    with open(metadata_path, "r") as f:
        _metadata = json.load(f)

    _index_loaded = True
    return _faiss_index, _metadata


async def _generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    return await embeddings.aembed_query(text)


@tool
async def search_similar_tickets(
    title: str,
    description: str,
    domain_filter: Optional[str] = None,
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Search FAISS index for similar historical tickets.

    Generates an embedding for the query text and searches the FAISS index
    for the most similar historical tickets. Optionally filters by domain.

    Args:
        title: Ticket title
        description: Ticket description
        domain_filter: Optional domain to filter results (MM, CIW, Specialty)
        top_k: Number of results to return (default 20)

    Returns:
        Dict containing:
        - similar_tickets: List of similar ticket dicts with raw scores
        - query_embedding: The generated embedding (for reuse)
        - total_searched: Number of tickets searched
    """
    # Generate embedding for query
    query_text = f"{title}\n\n{description}"
    query_embedding = await _generate_embedding(query_text)

    # Ensure index is loaded
    index, metadata = _ensure_index_loaded()

    # Normalize query vector for cosine similarity
    query_array = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_array)

    # Request more results if filtering by domain
    search_k = top_k * 3 if domain_filter else top_k
    distances, indices = index.search(query_array, search_k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:  # FAISS padding
            continue

        ticket_data = metadata[idx].copy()

        # Apply domain filter
        if domain_filter and ticket_data.get("domain") != domain_filter:
            continue

        # Add raw vector similarity score
        ticket_data["vector_similarity"] = float(distance)
        results.append(ticket_data)

        if len(results) >= top_k:
            break

    return {
        "similar_tickets": results,
        "query_embedding": query_embedding,
        "total_searched": index.ntotal
    }


@tool
def apply_hybrid_scoring(
    similar_tickets: List[Dict[str, Any]],
    vector_weight: float = 0.7,
    metadata_weight: float = 0.3,
    time_normalization_hours: float = 100.0
) -> List[Dict[str, Any]]:
    """
    Apply hybrid scoring to search results combining vector similarity and metadata.

    Hybrid score = (vector_weight * vector_similarity) + (metadata_weight * metadata_score)
    Metadata score = (priority_score * 0.6) + (time_score * 0.4)

    Args:
        similar_tickets: List of tickets with vector_similarity scores
        vector_weight: Weight for vector similarity (default 0.7)
        metadata_weight: Weight for metadata relevance (default 0.3)
        time_normalization_hours: Hours for time score normalization (default 100)

    Returns:
        List of tickets with added similarity_score, metadata_score fields, sorted by score
    """
    priority_scores = {
        "Critical": 1.0,
        "High": 0.8,
        "Medium": 0.5,
        "Low": 0.3
    }

    scored_tickets = []
    for ticket in similar_tickets:
        vector_score = ticket.get("vector_similarity", 0.0)

        # Priority factor
        priority = ticket.get("priority", "Medium")
        priority_score = priority_scores.get(priority, 0.5)

        # Resolution time factor (faster = better)
        res_time = ticket.get("resolution_time_hours", 24)
        time_score = max(0, 1 - (res_time / time_normalization_hours))

        # Combined metadata score
        metadata_score = (priority_score * 0.6) + (time_score * 0.4)

        # Hybrid score
        hybrid_score = (vector_weight * vector_score) + (metadata_weight * metadata_score)

        # Add scores to ticket
        ticket_with_scores = ticket.copy()
        ticket_with_scores["similarity_score"] = hybrid_score
        ticket_with_scores["metadata_score"] = metadata_score

        scored_tickets.append(ticket_with_scores)

    # Sort by hybrid score descending
    scored_tickets.sort(key=lambda x: x["similarity_score"], reverse=True)

    return scored_tickets


@tool
def get_index_stats() -> Dict[str, Any]:
    """
    Get statistics about the FAISS index.

    Returns:
        Dict containing:
        - total_vectors: Number of vectors in index
        - domain_distribution: Count of tickets per domain
        - metadata_entries: Number of metadata entries
    """
    index, metadata = _ensure_index_loaded()

    domain_counts = {}
    for ticket in metadata:
        domain = ticket.get("domain", "Unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    return {
        "total_vectors": index.ntotal,
        "dimension": 3072,
        "domain_distribution": domain_counts,
        "metadata_entries": len(metadata)
    }
