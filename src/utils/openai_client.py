"""
OpenAI client wrapper with async support and retry logic.
"""
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from openai import RateLimitError, APIError, APITimeoutError
from config.config import Config


class OpenAIClient:
    """
    Async OpenAI client with built-in retry logic and error handling.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = Config.RETRY_DELAY_SECONDS

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = Config.CLASSIFICATION_MODEL,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Call OpenAI Chat Completion API with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature 0-2
            max_tokens: Max tokens to generate
            response_format: Optional format specification (e.g., {"type": "json_object"})

        Returns:
            String content from the completion

        Raises:
            Exception: If all retries are exhausted
        """
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                if response_format:
                    kwargs["response_format"] = response_format

                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except (RateLimitError, APITimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"Rate limit/timeout error. Retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Max retries exceeded: {str(e)}")

            except APIError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    print(
                        f"API error. Retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"OpenAI API error after {self.max_retries} attempts: {str(e)}")

            except Exception as e:
                raise Exception(f"Unexpected error in chat completion: {str(e)}")

        raise Exception("Failed to complete request after all retries")

    async def generate_embedding(self, text: str, model: str = Config.EMBEDDING_MODEL) -> List[float]:
        """
        Generate embedding vector for text using OpenAI embeddings API.

        Args:
            text: Input text to embed
            model: Embedding model name

        Returns:
            List of floats representing the embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        for attempt in range(self.max_retries):
            try:
                # Clean text (remove newlines and excessive whitespace)
                cleaned_text = " ".join(text.split())

                response = await self.client.embeddings.create(
                    model=model,
                    input=cleaned_text,
                    encoding_format="float"
                )

                return response.data[0].embedding

            except (RateLimitError, APITimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    print(
                        f"Rate limit/timeout in embedding. Retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Max retries exceeded for embedding: {str(e)}")

            except Exception as e:
                raise Exception(f"Unexpected error generating embedding: {str(e)}")

        raise Exception("Failed to generate embedding after all retries")

    async def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        model: str = Config.CLASSIFICATION_MODEL,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call OpenAI with JSON response format.

        Args:
            messages: List of message dicts
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Parsed JSON dict

        Raises:
            Exception: If response is not valid JSON
        """
        import json

        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {str(e)}\nResponse: {response}")


# Global client instance
_client_instance: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create the singleton OpenAI client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenAIClient()
    return _client_instance
