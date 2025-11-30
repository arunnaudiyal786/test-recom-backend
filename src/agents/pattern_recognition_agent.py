"""
Pattern recognition agent using FAISS similarity search.

Retrieves top K similar historical tickets filtered by domain.
Supports configurable retrieval parameters for tuning.
"""
from typing import Dict, Any, List, Optional
from src.vectorstore.faiss_manager import get_faiss_manager
from src.vectorstore.embedding_generator import get_embedding_generator
from src.utils.config import Config
from src.utils.helpers import combine_ticket_text
from src.models.state_schema import TicketState, AgentOutput
from src.models.retrieval_config import RetrievalConfig, PriorityWeights


class PatternRecognitionAgent:
    """Agent for finding similar historical tickets using FAISS."""

    def __init__(self):
        self.faiss_manager = get_faiss_manager()
        self.embedding_generator = get_embedding_generator()
        self.top_k = Config.TOP_K_SIMILAR_TICKETS

        # Load FAISS index (lazy loading)
        self._index_loaded = False

    async def _ensure_index_loaded(self):
        """Ensure FAISS index is loaded (lazy loading)."""
        if not self._index_loaded:
            try:
                self.faiss_manager.load()
                self._index_loaded = True
            except FileNotFoundError as e:
                raise Exception(
                    f"FAISS index not found. Run scripts/setup_vectorstore.py first. Error: {str(e)}"
                )

    async def find_similar_tickets(
        self,
        title: str,
        description: str,
        domain: str,
        k: int = None
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """
        Find top K similar tickets filtered by domain.

        Args:
            title: Ticket title
            description: Ticket description
            domain: Domain to filter by ('MM', 'CIW', 'Specialty')
            k: Number of similar tickets to return (default: from config)

        Returns:
            Tuple of (similar_tickets, similarity_scores)
        """
        k = k or self.top_k

        # Ensure index is loaded
        await self._ensure_index_loaded()

        # Generate embedding for current ticket
        query_embedding = await self.embedding_generator.generate_ticket_embedding(
            title, description
        )

        # Search FAISS with domain filter
        similar_tickets, scores = self.faiss_manager.search(
            query_embedding=query_embedding,
            k=k,
            domain_filter=domain
        )

        return similar_tickets, scores

    async def find_similar_with_config(
        self,
        title: str,
        description: str,
        domain: str,
        config: RetrievalConfig
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """
        Find similar tickets using custom configuration.

        This method allows fine-tuning of search parameters for experimentation.

        Args:
            title: Ticket title
            description: Ticket description
            domain: Domain to filter by ('MM', 'CIW', 'Specialty')
            config: RetrievalConfig with custom parameters

        Returns:
            Tuple of (similar_tickets, similarity_scores)
        """
        # Ensure index is loaded
        await self._ensure_index_loaded()

        # Generate embedding for current ticket
        query_embedding = await self.embedding_generator.generate_ticket_embedding(
            title, description
        )

        # Search FAISS with domain filter (use config's domain_filter if provided)
        effective_domain = config.domain_filter if config.domain_filter else domain
        similar_tickets, scores = self.faiss_manager.search(
            query_embedding=query_embedding,
            k=config.top_k,
            domain_filter=effective_domain
        )

        return similar_tickets, scores

    def apply_hybrid_scoring_with_config(
        self,
        similar_tickets: List[Dict[str, Any]],
        similarity_scores: List[float],
        config: RetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid scoring with configurable weights.

        The hybrid scoring formula is:
            hybrid_score = (vector_weight * vector_similarity) + (metadata_weight * metadata_score)

        Where metadata_score = (priority_score * 0.6) + (time_score * 0.4)

        Args:
            similar_tickets: List of ticket dicts
            similarity_scores: List of cosine similarity scores
            config: RetrievalConfig with custom weights

        Returns:
            Re-ranked list of tickets with updated similarity_score
        """
        # Build priority scores dict from config
        priority_scores = {
            'Critical': config.priority_weights.Critical,
            'High': config.priority_weights.High,
            'Medium': config.priority_weights.Medium,
            'Low': config.priority_weights.Low
        }

        # Calculate hybrid scores
        for i, ticket in enumerate(similar_tickets):
            vector_score = similarity_scores[i]

            # Metadata factors
            priority_score = priority_scores.get(ticket.get('priority', 'Medium'), 0.5)

            # Resolution time factor (faster = better, normalize to 0-1)
            res_time = ticket.get('resolution_time_hours', 24)
            time_score = max(0, 1 - (res_time / config.time_normalization_hours))

            # Combine metadata scores (60% priority, 40% time)
            metadata_score = (priority_score * 0.6) + (time_score * 0.4)

            # Hybrid score using configurable weights
            hybrid_score = (config.vector_weight * vector_score) + (config.metadata_weight * metadata_score)

            # Update ticket with hybrid score
            ticket['similarity_score'] = hybrid_score
            ticket['vector_similarity'] = vector_score
            ticket['metadata_score'] = metadata_score

        # Re-sort by hybrid score
        similar_tickets.sort(key=lambda x: x['similarity_score'], reverse=True)

        return similar_tickets

    async def preview_search(
        self,
        title: str,
        description: str,
        domain: str,
        config: RetrievalConfig
    ) -> Dict[str, Any]:
        """
        Preview search results with custom configuration.

        Used by the UI for tuning parameters before running full pipeline.

        Args:
            title: Ticket title
            description: Ticket description
            domain: Domain to filter by (from classification or config override)
            config: RetrievalConfig with custom parameters

        Returns:
            Dict with similar_tickets, search_metadata, and config_used
        """
        # Find similar tickets with config
        similar_tickets, similarity_scores = await self.find_similar_with_config(
            title, description, domain, config
        )

        # Apply hybrid scoring with config
        similar_tickets = self.apply_hybrid_scoring_with_config(
            similar_tickets, similarity_scores, config
        )

        # Calculate search metadata
        effective_domain = config.domain_filter if config.domain_filter else domain
        avg_similarity = (
            sum(t['similarity_score'] for t in similar_tickets) / len(similar_tickets)
            if similar_tickets else 0
        )

        search_metadata = {
            'query_domain': effective_domain,
            'total_found': len(similar_tickets),
            'avg_similarity': avg_similarity,
            'top_similarity': similar_tickets[0]['similarity_score'] if similar_tickets else 0
        }

        return {
            'similar_tickets': similar_tickets,
            'search_metadata': search_metadata,
            'config_used': config.model_dump()
        }

    def apply_hybrid_scoring(
        self,
        similar_tickets: List[Dict[str, Any]],
        similarity_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid scoring: 70% vector similarity + 30% metadata relevance.

        Metadata relevance factors:
        - Priority (Critical > High > Medium > Low)
        - Recent tickets score higher
        - Faster resolution time scores higher

        Args:
            similar_tickets: List of ticket dicts
            similarity_scores: List of cosine similarity scores

        Returns:
            Re-ranked list of tickets with updated similarity_score
        """
        priority_scores = {
            'Critical': 1.0,
            'High': 0.8,
            'Medium': 0.5,
            'Low': 0.3
        }

        # Calculate hybrid scores
        for i, ticket in enumerate(similar_tickets):
            vector_score = similarity_scores[i]

            # Metadata factors
            priority_score = priority_scores.get(ticket.get('priority', 'Medium'), 0.5)

            # Resolution time factor (faster = better, normalize to 0-1)
            res_time = ticket.get('resolution_time_hours', 24)
            time_score = max(0, 1 - (res_time / 100))  # Normalize assuming max 100 hours

            # Combine metadata scores
            metadata_score = (priority_score * 0.6) + (time_score * 0.4)

            # Hybrid score: 70% vector + 30% metadata
            hybrid_score = (0.7 * vector_score) + (0.3 * metadata_score)

            # Update ticket with hybrid score
            ticket['similarity_score'] = hybrid_score
            ticket['vector_similarity'] = vector_score
            ticket['metadata_score'] = metadata_score

        # Re-sort by hybrid score
        similar_tickets.sort(key=lambda x: x['similarity_score'], reverse=True)

        return similar_tickets

    async def __call__(self, state: TicketState) -> AgentOutput:
        """
        Main agent execution function for LangGraph.

        Args:
            state: Current ticket state

        Returns:
            Partial state update with similar tickets
        """
        try:
            title = state['title']
            description = state['description']
            domain = state.get('classified_domain')

            if not domain:
                raise ValueError("Domain not classified. Classification agent must run first.")

            print(f"\n🔎 Pattern Recognition Agent - Finding similar {domain} tickets")

            # Check if custom search config is provided in state
            search_config_dict = state.get('search_config')

            if search_config_dict:
                # Use custom config from UI tuning
                print("   📐 Using custom search configuration from UI")
                config = RetrievalConfig(**search_config_dict)
                similar_tickets, similarity_scores = await self.find_similar_with_config(
                    title, description, domain, config
                )
                similar_tickets = self.apply_hybrid_scoring_with_config(
                    similar_tickets, similarity_scores, config
                )
            else:
                # Use default config
                similar_tickets, similarity_scores = await self.find_similar_tickets(
                    title, description, domain
                )
                similar_tickets = self.apply_hybrid_scoring(similar_tickets, similarity_scores)

            # Calculate search metadata
            avg_similarity = (
                sum(t['similarity_score'] for t in similar_tickets) / len(similar_tickets)
                if similar_tickets else 0
            )

            search_metadata = {
                'query_domain': domain,
                'total_found': len(similar_tickets),
                'avg_similarity': avg_similarity,
                'top_similarity': similar_tickets[0]['similarity_score'] if similar_tickets else 0
            }

            print(f"   ✅ Found {len(similar_tickets)} similar tickets")
            print(f"   📊 Average similarity: {avg_similarity:.2%}")
            if similar_tickets:
                print(f"   🔝 Top match: {similar_tickets[0]['ticket_id']} (score: {similar_tickets[0]['similarity_score']:.2%})")

            # Return state update
            return {
                "similar_tickets": similar_tickets,
                "similarity_scores": [t['similarity_score'] for t in similar_tickets],
                "search_metadata": search_metadata,
                "status": "success",
                "current_agent": "Pattern Recognition Agent",
                "messages": [{
                    "role": "assistant",
                    "content": f"Found {len(similar_tickets)} similar {domain} tickets with avg similarity {avg_similarity:.2%}"
                }]
            }

        except Exception as e:
            print(f"   ❌ Pattern recognition error: {str(e)}")
            return {
                "status": "error",
                "current_agent": "Pattern Recognition Agent",
                "error_message": f"Pattern recognition failed: {str(e)}"
            }


# Create singleton instance for use in LangGraph
pattern_recognition_agent = PatternRecognitionAgent()
