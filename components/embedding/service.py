"""
Embedding Service Component.

Generates embeddings for text using OpenAI's embedding models.
Can be used standalone or as part of the RAG pipeline.
"""

import asyncio
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI
from openai import RateLimitError, APIError, APITimeoutError

from components.base import BaseComponent, ComponentConfig
from components.base.exceptions import ProcessingError, ConfigurationError
from components.embedding.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
)


class EmbeddingConfig(ComponentConfig):
    """Configuration for Embedding Service."""

    # Model settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072  # Default for text-embedding-3-large

    class Config:
        env_prefix = "EMBEDDING_"


class EmbeddingService(BaseComponent[EmbeddingRequest, EmbeddingResponse]):
    """
    Service for generating text embeddings using OpenAI.

    Usage:
        # Direct instantiation
        service = EmbeddingService()
        response = await service.process(EmbeddingRequest(text="hello world"))

        # With custom config
        config = EmbeddingConfig(embedding_model="text-embedding-3-small")
        service = EmbeddingService(config)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            if not self.config.openai_api_key:
                raise ConfigurationError(
                    "OpenAI API key is required",
                    component=self.component_name,
                    missing_keys=["openai_api_key"],
                )
            self._client = AsyncOpenAI(api_key=self.config.openai_api_key)
        return self._client

    @property
    def component_name(self) -> str:
        return "embedding"

    def _combine_text(self, title: Optional[str], description: Optional[str]) -> str:
        """Combine title and description into single text."""
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if description:
            parts.append(f"Description: {description}")
        return "\n".join(parts)

    def _clean_text(self, text: str) -> str:
        """Clean text for embedding (remove excess whitespace)."""
        return " ".join(text.split())

    def _get_input_text(self, request: EmbeddingRequest) -> str:
        """Extract text from request (handles both text and title/description)."""
        if request.text:
            return self._clean_text(request.text)

        if request.title or request.description:
            combined = self._combine_text(request.title, request.description)
            return self._clean_text(combined)

        raise ProcessingError(
            "Either 'text' or 'title/description' must be provided",
            component=self.component_name,
            stage="input_validation",
        )

    async def _generate_embedding_with_retry(self, text: str) -> List[float]:
        """Generate embedding with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=text,
                    encoding_format="float",
                )
                return response.data[0].embedding

            except (RateLimitError, APITimeoutError) as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay_seconds * (2**attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise ProcessingError(
                        f"Max retries exceeded: {str(e)}",
                        component=self.component_name,
                        stage="api_call",
                        original_error=e,
                    )

            except APIError as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay_seconds * (2**attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise ProcessingError(
                        f"OpenAI API error: {str(e)}",
                        component=self.component_name,
                        stage="api_call",
                        original_error=e,
                    )

        raise ProcessingError(
            "Failed to generate embedding after all retries",
            component=self.component_name,
            stage="api_call",
        )

    async def process(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embedding for the input text.

        Args:
            request: EmbeddingRequest with text or title/description

        Returns:
            EmbeddingResponse with embedding vector
        """
        # Get input text
        input_text = self._get_input_text(request)

        # Generate embedding
        embedding = await self._generate_embedding_with_retry(input_text)

        return EmbeddingResponse(
            embedding=embedding,
            model=self.config.embedding_model,
            dimensions=len(embedding),
            input_text=input_text[:200] + "..." if len(input_text) > 200 else input_text,
        )

    async def process_batch(
        self, request: BatchEmbeddingRequest
    ) -> BatchEmbeddingResponse:
        """
        Generate embeddings for multiple texts.

        Args:
            request: BatchEmbeddingRequest with list of texts

        Returns:
            BatchEmbeddingResponse with list of embeddings
        """
        embeddings = []

        for i in range(0, len(request.texts), request.batch_size):
            batch = request.texts[i : i + request.batch_size]

            # Process batch concurrently
            batch_embeddings = await asyncio.gather(
                *[
                    self._generate_embedding_with_retry(self._clean_text(text))
                    for text in batch
                ]
            )

            embeddings.extend(batch_embeddings)

        return BatchEmbeddingResponse(
            embeddings=embeddings,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions,
            count=len(embeddings),
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check if embedding service is healthy."""
        try:
            # Verify API key is configured
            if not self.config.openai_api_key:
                return {
                    "status": "unhealthy",
                    "component": self.component_name,
                    "error": "OpenAI API key not configured",
                }

            # Try a simple embedding to verify connectivity
            test_response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input="health check",
                encoding_format="float",
            )

            return {
                "status": "healthy",
                "component": self.component_name,
                "model": self.config.embedding_model,
                "dimensions": len(test_response.data[0].embedding),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "component": self.component_name,
                "error": str(e),
            }
