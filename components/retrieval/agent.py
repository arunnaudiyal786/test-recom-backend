"""
Retrieval Agent - LangGraph node for pattern recognition.

This agent searches for similar historical tickets using FAISS
and applies hybrid scoring for relevance ranking.
"""

from typing import Dict, Any

from components.retrieval.tools import search_similar_tickets, apply_hybrid_scoring


async def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for pattern recognition / retrieval.

    Searches for similar historical tickets based on the classified domain,
    applies hybrid scoring, and returns the top matches.

    Args:
        state: Current workflow state with ticket info and classified domain

    Returns:
        Partial state update with similar tickets and search metadata
    """
    try:
        title = state.get("title", "")
        description = state.get("description", "")
        domain = state.get("classified_domain", None)
        ticket_id = state.get("ticket_id", "N/A")

        # Get search config from state if available
        search_config = state.get("search_config", {}) or {}
        top_k = search_config.get("top_k", 20)
        vector_weight = search_config.get("vector_weight", 0.7)
        metadata_weight = search_config.get("metadata_weight", 0.3)

        print(f"\n🔎 Retrieval Agent - Searching for similar tickets: {ticket_id}")
        print(f"   Domain filter: {domain}")

        # Search for similar tickets
        search_result = await search_similar_tickets.ainvoke({
            "title": title,
            "description": description,
            "domain_filter": domain,
            "top_k": top_k
        })

        raw_tickets = search_result.get("similar_tickets", [])

        # Apply hybrid scoring
        scored_tickets = apply_hybrid_scoring.invoke({
            "similar_tickets": raw_tickets,
            "vector_weight": vector_weight,
            "metadata_weight": metadata_weight
        })

        # Calculate statistics
        total_found = len(scored_tickets)
        avg_similarity = (
            sum(t["similarity_score"] for t in scored_tickets) / total_found
            if total_found > 0 else 0
        )
        top_similarity = scored_tickets[0]["similarity_score"] if scored_tickets else 0

        print(f"   ✅ Found {total_found} similar tickets")
        print(f"   📊 Avg similarity: {avg_similarity:.2%}, Top: {top_similarity:.2%}")

        # Build search metadata
        search_metadata = {
            "query_domain": domain,
            "total_found": total_found,
            "avg_similarity": avg_similarity,
            "top_similarity": top_similarity,
            "config_used": {
                "top_k": top_k,
                "vector_weight": vector_weight,
                "metadata_weight": metadata_weight
            }
        }

        return {
            "similar_tickets": scored_tickets,
            "similarity_scores": [t["similarity_score"] for t in scored_tickets],
            "search_metadata": search_metadata,
            "status": "success",
            "current_agent": "retrieval",
            "messages": [{
                "role": "assistant",
                "content": f"Found {total_found} similar tickets with avg similarity {avg_similarity:.2%}"
            }]
        }

    except Exception as e:
        print(f"   ❌ Retrieval error: {str(e)}")
        return {
            "similar_tickets": [],
            "similarity_scores": [],
            "search_metadata": {},
            "status": "error",
            "current_agent": "retrieval",
            "error_message": f"Retrieval failed: {str(e)}",
            "messages": [{
                "role": "assistant",
                "content": f"Retrieval failed: {str(e)}"
            }]
        }


# For backward compatibility - callable class wrapper
class PatternRecognitionAgent:
    """Callable wrapper for retrieval_node."""

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await retrieval_node(state)

    async def preview_search(
        self,
        title: str,
        description: str,
        domain: str,
        config: Any
    ) -> Dict[str, Any]:
        """Preview search with custom config (for UI tuning)."""
        search_result = await search_similar_tickets.ainvoke({
            "title": title,
            "description": description,
            "domain_filter": config.domain_filter or domain,
            "top_k": config.top_k
        })

        raw_tickets = search_result.get("similar_tickets", [])

        scored_tickets = apply_hybrid_scoring.invoke({
            "similar_tickets": raw_tickets,
            "vector_weight": config.vector_weight,
            "metadata_weight": config.metadata_weight
        })

        total_found = len(scored_tickets)
        avg_similarity = (
            sum(t["similarity_score"] for t in scored_tickets) / total_found
            if total_found > 0 else 0
        )

        return {
            "similar_tickets": scored_tickets,
            "search_metadata": {
                "query_domain": config.domain_filter or domain,
                "total_found": total_found,
                "avg_similarity": avg_similarity,
                "top_similarity": scored_tickets[0]["similarity_score"] if scored_tickets else 0
            }
        }


# Singleton instance
pattern_recognition_agent = PatternRecognitionAgent()
