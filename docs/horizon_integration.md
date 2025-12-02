# Horizon LLM Integration Guide

This document explains how to replace the OpenAI integration with the Horizon LLM client. It identifies all integration points and provides step-by-step instructions.

---

## Overview

The application currently uses OpenAI for two purposes:
1. **Chat Completions** - Classification, labeling, and resolution generation
2. **Embeddings** - Vector embeddings for FAISS similarity search

Both need to be replaced with Horizon equivalents.

---

## Integration Points Summary

| File | Current Dependency | Purpose | Priority |
|------|-------------------|---------|----------|
| `src/utils/config.py` | `OPENAI_API_KEY` | API key configuration | **HIGH** |
| `src/utils/openai_client.py` | `openai.AsyncOpenAI` | Chat completions & embeddings wrapper | **HIGH** |
| `components/classification/tools.py` | `langchain_openai.ChatOpenAI` | Domain classification | **HIGH** |
| `components/labeling/tools.py` | `langchain_openai.ChatOpenAI` | Label assignment | **HIGH** |
| `components/resolution/tools.py` | `langchain_openai.ChatOpenAI` | Resolution generation | **HIGH** |
| `components/retrieval/tools.py` | `langchain_openai.OpenAIEmbeddings` | Embedding generation | **HIGH** |
| `src/vectorstore/embedding_generator.py` | `get_openai_client()` | Embedding for FAISS indexing | **MEDIUM** |

---

## Step 1: Update Configuration (`src/utils/config.py`)

### Current Code (Lines 15-18)
```python
# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
```

### Replace With
```python
# Horizon Configuration
HORIZON_API_KEY = os.getenv("HORIZON_API_KEY")
HORIZON_BASE_URL = os.getenv("HORIZON_BASE_URL", "https://api.horizon.example.com/v1")

if not HORIZON_API_KEY:
    raise ValueError("HORIZON_API_KEY environment variable is required")
```

### Also Update Model Validation (Lines 73-81)
Remove or update the model validation to match Horizon's available models:
```python
# Update to Horizon model names
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "horizon-chat-v1")
RESOLUTION_MODEL = os.getenv("RESOLUTION_MODEL", "horizon-chat-v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "horizon-embed-v1")
```

---

## Step 2: Create Horizon Client (`src/utils/horizon_client.py`)

Create a new file to replace `openai_client.py`:

```python
"""
Horizon LLM client wrapper with async support and retry logic.
"""
import asyncio
from typing import List, Dict, Any, Optional

# Import your Horizon SDK here
# from horizon import AsyncHorizonClient  # Example import

from src.utils.config import Config


class HorizonClient:
    """
    Async Horizon client with built-in retry logic and error handling.
    """

    def __init__(self):
        # Initialize Horizon client with your SDK
        # self.client = AsyncHorizonClient(
        #     api_key=Config.HORIZON_API_KEY,
        #     base_url=Config.HORIZON_BASE_URL
        # )
        self.max_retries = Config.MAX_RETRIES
        self.retry_delay = Config.RETRY_DELAY_SECONDS

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Call Horizon Chat Completion API with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature 0-2
            max_tokens: Max tokens to generate
            response_format: Optional format specification (e.g., {"type": "json_object"})

        Returns:
            String content from the completion
        """
        model = model or Config.CLASSIFICATION_MODEL

        for attempt in range(self.max_retries):
            try:
                # TODO: Replace with Horizon API call
                # response = await self.client.chat.completions.create(
                #     model=model,
                #     messages=messages,
                #     temperature=temperature,
                #     max_tokens=max_tokens,
                #     response_format=response_format
                # )
                # return response.choices[0].message.content

                raise NotImplementedError("Implement Horizon chat completion")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Error. Retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Horizon API error after {self.max_retries} attempts: {str(e)}")

        raise Exception("Failed to complete request after all retries")

    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Generate embedding vector for text using Horizon embeddings API.

        Args:
            text: Input text to embed
            model: Embedding model name

        Returns:
            List of floats representing the embedding vector
        """
        model = model or Config.EMBEDDING_MODEL

        for attempt in range(self.max_retries):
            try:
                cleaned_text = " ".join(text.split())

                # TODO: Replace with Horizon API call
                # response = await self.client.embeddings.create(
                #     model=model,
                #     input=cleaned_text
                # )
                # return response.data[0].embedding

                raise NotImplementedError("Implement Horizon embedding generation")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception(f"Horizon embedding error: {str(e)}")

        raise Exception("Failed to generate embedding after all retries")

    async def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call Horizon with JSON response format.
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
_client_instance: Optional[HorizonClient] = None


def get_horizon_client() -> HorizonClient:
    """Get or create the singleton Horizon client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = HorizonClient()
    return _client_instance


# Backward compatibility alias
get_openai_client = get_horizon_client
```

---

## Step 3: Update LangChain Components

The components use LangChain's `ChatOpenAI` and `OpenAIEmbeddings`. You have two options:

### Option A: Use Custom LangChain Integration (Recommended)

If Horizon has OpenAI-compatible API, you can use LangChain's `ChatOpenAI` with a custom base URL:

```python
# In components/classification/tools.py (Line 158)
# BEFORE:
llm = ChatOpenAI(
    model=Config.CLASSIFICATION_MODEL,
    temperature=0.2
)

# AFTER:
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=Config.CLASSIFICATION_MODEL,
    temperature=0.2,
    openai_api_key=Config.HORIZON_API_KEY,
    openai_api_base=Config.HORIZON_BASE_URL  # Point to Horizon endpoint
)
```

### Option B: Create Custom LangChain LLM Class

If Horizon has a different API format:

```python
# Create: components/base/horizon_llm.py
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from typing import List, Optional, Any
from src.utils.config import Config


class HorizonChatModel(BaseChatModel):
    """Custom LangChain chat model for Horizon."""

    model_name: str = "horizon-chat-v1"
    temperature: float = 0.2

    def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
        # Implement synchronous generation
        raise NotImplementedError("Use async version")

    async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> Any:
        from src.utils.horizon_client import get_horizon_client

        client = get_horizon_client()
        formatted_messages = [
            {"role": m.type, "content": m.content}
            for m in messages
        ]

        response = await client.chat_completion(
            messages=formatted_messages,
            model=self.model_name,
            temperature=self.temperature
        )

        return AIMessage(content=response)

    @property
    def _llm_type(self) -> str:
        return "horizon"
```

---

## Step 4: Update Embedding Generation

### File: `components/retrieval/tools.py` (Lines 68-72)

```python
# BEFORE:
async def _generate_embedding(text: str) -> List[float]:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072
    )
    return await embeddings.aembed_query(text)

# AFTER (Option A - OpenAI Compatible):
async def _generate_embedding(text: str) -> List[float]:
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        dimensions=Config.EMBEDDING_DIMENSIONS,
        openai_api_key=Config.HORIZON_API_KEY,
        openai_api_base=Config.HORIZON_BASE_URL
    )
    return await embeddings.aembed_query(text)

# AFTER (Option B - Custom Client):
async def _generate_embedding(text: str) -> List[float]:
    from src.utils.horizon_client import get_horizon_client

    client = get_horizon_client()
    return await client.generate_embedding(text)
```

---

## Step 5: Update Environment Variables

### File: `.env`

```bash
# Horizon Configuration (replaces OpenAI)
HORIZON_API_KEY=your-horizon-api-key-here
HORIZON_BASE_URL=https://api.horizon.example.com/v1

# Model Selection (update to Horizon model names)
CLASSIFICATION_MODEL=horizon-chat-v1
RESOLUTION_MODEL=horizon-chat-v1
EMBEDDING_MODEL=horizon-embed-v1

# Thresholds (unchanged)
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
LABEL_CONFIDENCE_THRESHOLD=0.7
TOP_K_SIMILAR_TICKETS=20
```

---

## Files to Modify (Complete List)

| Priority | File | Changes Required |
|----------|------|------------------|
| 1 | `src/utils/config.py` | Replace `OPENAI_API_KEY` with `HORIZON_API_KEY` + `HORIZON_BASE_URL` |
| 2 | `src/utils/openai_client.py` | Rename to `horizon_client.py`, update client class |
| 3 | `components/classification/tools.py:158` | Update `ChatOpenAI` instantiation |
| 4 | `components/labeling/tools.py:165,234,321` | Update `ChatOpenAI` instantiation (3 places) |
| 5 | `components/resolution/tools.py:86` | Update `ChatOpenAI` instantiation |
| 6 | `components/retrieval/tools.py:68` | Update `OpenAIEmbeddings` instantiation |
| 7 | `src/vectorstore/embedding_generator.py:15` | Update to use `get_horizon_client()` |
| 8 | `.env.example` | Update environment variable names |
| 9 | `requirements.txt` | Add Horizon SDK, optionally remove `openai` |

---

## Testing the Integration

After making changes:

```bash
# 1. Update .env with Horizon credentials
cp .env.example .env
# Edit .env with your HORIZON_API_KEY

# 2. Test configuration loads
python3 -c "from src.utils.config import Config; print('Config OK')"

# 3. Test client initialization
python3 -c "from src.utils.horizon_client import get_horizon_client; print('Client OK')"

# 4. Rebuild FAISS index (uses new embedding client)
python3 scripts/setup_vectorstore.py

# 5. Run full pipeline test
python3 main.py
```

---

## Rollback Plan

If issues occur, the OpenAI integration can be restored by:

1. Reverting `config.py` to use `OPENAI_API_KEY`
2. Keeping `openai_client.py` as-is
3. Removing `openai_api_base` parameter from LangChain calls

---

## Notes

- **Embedding Dimensions**: Verify Horizon's embedding dimensions match the current 3072. Update `Config.EMBEDDING_DIMENSIONS` if different.
- **JSON Mode**: Ensure Horizon supports `response_format={"type": "json_object"}` for structured outputs.
- **Rate Limits**: The retry logic with exponential backoff should work, but adjust `MAX_RETRIES` and `RETRY_DELAY_SECONDS` based on Horizon's rate limits.
- **Model Names**: Update all model name references to match Horizon's available models.
