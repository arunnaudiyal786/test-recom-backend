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
| `config/config.py` | `OPENAI_API_KEY` | API key configuration (class-based Config) | **HIGH** |
| `components/classification/tools.py` | `langchain_openai.ChatOpenAI` | Domain classification | **HIGH** |
| `components/labeling/tools.py` | `langchain.agents.create_agent` + `openai.OpenAI` | Label assignment (hybrid semantic + LLM) | **HIGH** |
| `components/resolution/tools.py` | `langchain_openai.ChatOpenAI` | Resolution generation | **HIGH** |
| `components/retrieval/tools.py` | `openai.OpenAI` embeddings | Embedding generation for search | **HIGH** |
| `components/labeling/category_embeddings.py` | `openai.OpenAI` embeddings | Pre-computed category embeddings | **MEDIUM** |
| `src/vectorstore/embedding_generator.py` | `openai.OpenAI` | Embedding for FAISS indexing | **MEDIUM** |

---

## Step 1: Update Configuration (`config/config.py`)

**Note**: Configuration is now a class-based system in `config/config.py`, not environment variables.

### Current Code (Lines 12-18)
```python
class Config:
    """Configuration class for application settings."""

    # OpenAI Configuration (only API key from environment)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
```

### Replace With
```python
class Config:
    """Configuration class for application settings."""

    # Horizon Configuration
    HORIZON_API_KEY = os.getenv("HORIZON_API_KEY")
    HORIZON_BASE_URL = os.getenv("HORIZON_BASE_URL", "https://api.horizon.example.com/v1")

    if not HORIZON_API_KEY:
        raise ValueError("HORIZON_API_KEY environment variable is required")
```

### Also Update Model Settings (Lines 21-23)
Update the model attributes to match Horizon's available models:
```python
# ========== MODEL CONFIGURATION ==========
CLASSIFICATION_MODEL = "horizon-chat-v1"
RESOLUTION_MODEL = "horizon-chat-v1"
EMBEDDING_MODEL = "horizon-embed-v1"
```

### Update Model Validation (Lines 112-120)
```python
@classmethod
def validate(cls):
    """Validate configuration settings."""
    if not cls.HORIZON_API_KEY:
        raise ValueError("HORIZON_API_KEY is required")

    # Update to Horizon model names
    valid_chat_models = ["horizon-chat-v1", "horizon-chat-v2"]
    valid_embed_models = ["horizon-embed-v1"]

    if cls.CLASSIFICATION_MODEL not in valid_chat_models:
        raise ValueError(f"Invalid CLASSIFICATION_MODEL: {cls.CLASSIFICATION_MODEL}")

    if cls.EMBEDDING_MODEL not in valid_embed_models:
        raise ValueError(f"Invalid EMBEDDING_MODEL: {cls.EMBEDDING_MODEL}")

    return True
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

from config import Config


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

The components use multiple patterns for LLM integration:
1. **Direct OpenAI SDK**: `openai.OpenAI` for embeddings
2. **LangChain ChatOpenAI**: For classification and resolution
3. **LangChain create_agent**: For label assignment (uses model name string)

### Pattern 1: Direct OpenAI SDK (Embeddings)

Used in `components/labeling/tools.py` and `components/retrieval/tools.py`:

```python
# BEFORE:
from openai import OpenAI
client = OpenAI(api_key=Config.OPENAI_API_KEY)
response = client.embeddings.create(model=Config.EMBEDDING_MODEL, input=text)

# AFTER (Option A - OpenAI-compatible API):
from openai import OpenAI
client = OpenAI(
    api_key=Config.HORIZON_API_KEY,
    base_url=Config.HORIZON_BASE_URL
)
response = client.embeddings.create(model=Config.EMBEDDING_MODEL, input=text)
```

### Pattern 2: LangChain ChatOpenAI (Classification/Resolution)

If Horizon has OpenAI-compatible API, use custom base URL:

```python
# In components/classification/tools.py
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

### Pattern 3: LangChain create_agent (Label Assignment)

The labeling component uses `langchain.agents.create_agent` which takes a model name string:

```python
# In components/labeling/tools.py (Lines 122-131)
# BEFORE:
from langchain.agents import create_agent

agent = create_agent(
    model=Config.CATEGORY_CLASSIFICATION_MODEL,  # "gpt-4o"
    tools=[submit_classification_result],
    system_prompt=BINARY_CLASSIFICATION_SYSTEM_PROMPT
)

# AFTER (requires custom model provider or LangChain Horizon integration):
# Option A: If Horizon supports OpenAI-compatible model names
agent = create_agent(
    model="horizon-chat-v1",  # Use Horizon model name
    tools=[submit_classification_result],
    system_prompt=BINARY_CLASSIFICATION_SYSTEM_PROMPT
)

# Option B: Create custom model binding
# This may require implementing a custom LangChain model class
# See LangChain documentation for custom chat model integration
```

### Option B: Create Custom LangChain LLM Class

If Horizon has a different API format, create a custom chat model:

```python
# Create: components/base/horizon_llm.py
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from typing import List, Optional, Any
from config.config import Config


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
| 1 | `config/config.py` | Replace `OPENAI_API_KEY` with `HORIZON_API_KEY` + `HORIZON_BASE_URL` |
| 2 | `components/labeling/tools.py` | Update `create_agent` model names and `OpenAI` embeddings client (Lines 21, 123, 145, 167, 223) |
| 3 | `components/classification/tools.py` | Update `ChatOpenAI` instantiation |
| 4 | `components/resolution/tools.py` | Update `ChatOpenAI` instantiation |
| 5 | `components/retrieval/tools.py` | Update `OpenAI` embeddings client |
| 6 | `components/labeling/category_embeddings.py` | Update `OpenAI` embeddings client |
| 7 | `src/vectorstore/embedding_generator.py` | Update to use Horizon embedding client |
| 8 | `.env` | Update environment variable: `HORIZON_API_KEY` instead of `OPENAI_API_KEY` |
| 9 | `requirements.txt` | Add Horizon SDK if needed |

### Additional Considerations for create_agent

The `langchain.agents.create_agent` function in `components/labeling/tools.py` takes a model name as a string. If Horizon requires a custom model binding, you may need to:

1. Register a custom model provider with LangChain
2. Or modify the `create_agent` calls to use a pre-configured LLM instance instead of a model name string

Check LangChain documentation for `create_agent` with custom models.

---

## Testing the Integration

After making changes:

```bash
# 1. Update .env with Horizon credentials
cp .env.example .env
# Edit .env with your HORIZON_API_KEY

# 2. Test configuration loads
python3 -c "from config import Config; print('Config OK')"

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
