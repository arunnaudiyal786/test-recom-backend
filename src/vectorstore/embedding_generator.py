"""
OpenAI embedding generation for ticket text.
"""
import asyncio
from typing import List
from src.utils.openai_client import get_openai_client
from src.utils.helpers import combine_ticket_text, clean_text
from config import Config


class EmbeddingGenerator:
    """Generate embeddings for ticket text using OpenAI."""

    def __init__(self):
        self.client = get_openai_client()
        self.model = Config.EMBEDDING_MODEL

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        cleaned_text = clean_text(text)
        return await self.client.generate_embedding(cleaned_text, model=self.model)

    async def generate_ticket_embedding(self, title: str, description: str) -> List[float]:
        """
        Generate embedding for a ticket by combining title and description.

        Args:
            title: Ticket title
            description: Ticket description

        Returns:
            Embedding vector
        """
        combined_text = combine_ticket_text(title, description)
        return await self.generate_embedding(combined_text)

    async def generate_batch_embeddings(
        self, texts: List[str], batch_size: int = 10, show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process concurrently
            show_progress: Whether to print progress messages

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Process batch concurrently
            batch_embeddings = await asyncio.gather(
                *[self.generate_embedding(text) for text in batch]
            )

            embeddings.extend(batch_embeddings)

            if show_progress:
                progress = min(i + batch_size, len(texts))
                print(f"  Generated embeddings: {progress}/{len(texts)}")

        return embeddings


# Global instance
_embedding_generator: EmbeddingGenerator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the singleton embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
