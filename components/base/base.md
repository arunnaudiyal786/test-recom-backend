# Base Component

The `base` module provides foundational abstractions and utilities that all other components inherit from. It establishes consistent patterns for configuration, error handling, and service interfaces across the entire system.

## Overview

This module contains three core building blocks:
- **BaseComponent**: Abstract base class defining the service interface
- **ComponentConfig**: Configuration management with environment variable support
- **Exceptions**: Structured exception hierarchy for consistent error handling

## Architecture

```
base/
├── __init__.py          # Public API exports
├── component.py         # BaseComponent abstract class
├── config.py            # ComponentConfig base settings
├── exceptions.py        # Custom exception classes
└── README.md            # This file
```

## Components

### BaseComponent (`component.py`)

An abstract base class that all service components must inherit from. Uses Python generics to enforce type-safe request/response contracts.

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from pydantic import BaseModel

TRequest = TypeVar("TRequest", bound=BaseModel)
TResponse = TypeVar("TResponse", bound=BaseModel)

class BaseComponent(ABC, Generic[TRequest, TResponse]):
    """Abstract base class for all service components."""
```

#### Required Methods

| Method | Description | Return Type |
|--------|-------------|-------------|
| `process(request)` | Main processing logic | `TResponse` |
| `health_check()` | Health status check | `Dict[str, Any]` |
| `component_name` | Unique identifier (property) | `str` |

#### Usage Example

```python
from components.base import BaseComponent
from pydantic import BaseModel

class MyRequest(BaseModel):
    input_data: str

class MyResponse(BaseModel):
    result: str

class MyService(BaseComponent[MyRequest, MyResponse]):
    @property
    def component_name(self) -> str:
        return "my-service"

    async def process(self, request: MyRequest) -> MyResponse:
        # Your business logic here
        return MyResponse(result=f"Processed: {request.input_data}")

    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "component": self.component_name}
```

#### Callable Interface

Components can be called directly like functions:

```python
service = MyService()
response = await service(MyRequest(input_data="test"))
# Equivalent to: await service.process(MyRequest(input_data="test"))
```

---

### ComponentConfig (`config.py`)

Base configuration class using `pydantic-settings` for environment variable loading.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class ComponentConfig(BaseSettings):
    """Base configuration for all component services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
```

#### Built-in Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `openai_api_key` | `str` | `None` | OpenAI API key (from `OPENAI_API_KEY`) |
| `log_level` | `str` | `"INFO"` | Logging level |
| `max_retries` | `int` | `3` | Max retry attempts for API calls |
| `retry_delay_seconds` | `int` | `2` | Base delay between retries |

#### Creating Component-Specific Config

```python
from components.base import ComponentConfig

class ClassificationConfig(ComponentConfig):
    """Configuration for Classification Service."""

    classification_model: str = "gpt-4o"
    classification_temperature: float = 0.2
    confidence_threshold: float = 0.7

    class Config:
        env_prefix = "CLASSIFICATION_"  # CLASSIFICATION_MODEL, etc.
```

#### Validation Method

```python
def validate_required(self, *fields: str) -> None:
    """
    Validate that required fields are set.

    Raises:
        ConfigurationError: If any required field is missing
    """
```

---

### Exceptions (`exceptions.py`)

Structured exception hierarchy for consistent error handling and API responses.

#### Exception Hierarchy

```
ComponentError (base)
├── ConfigurationError    # Invalid/missing configuration
├── ProcessingError       # Processing failures
├── ValidationError       # Input validation failures
└── ExternalServiceError  # External service failures
```

#### ComponentError (Base)

```python
class ComponentError(Exception):
    """Base exception for all component errors."""

    def __init__(
        self,
        message: str,
        component: str = None,
        details: dict = None
    ):
        self.message = message
        self.component = component
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "component": self.component,
            "details": self.details,
        }
```

#### ConfigurationError

Raised when component configuration is invalid or missing.

```python
raise ConfigurationError(
    "Missing required configuration: openai_api_key",
    component="classification",
    missing_keys=["openai_api_key"]
)
```

#### ProcessingError

Raised when component processing fails.

```python
raise ProcessingError(
    "Failed to classify ticket",
    component="classification",
    stage="llm_call",
    original_error=e
)
```

#### ValidationError

Raised when input validation fails.

```python
raise ValidationError(
    "Invalid domain value",
    component="classification",
    field="domain",
    value="InvalidDomain"
)
```

#### ExternalServiceError

Raised when external services (OpenAI, FAISS) fail.

```python
raise ExternalServiceError(
    "OpenAI API rate limited",
    component="embedding",
    service="OpenAI",
    status_code=429
)
```

## Public API

The `__init__.py` exports all public classes:

```python
from components.base import (
    BaseComponent,
    ComponentConfig,
    ComponentError,
    ConfigurationError,
    ProcessingError,
    ValidationError,
)
```

## Design Principles

1. **Type Safety**: Uses Pydantic models and Python generics for compile-time type checking
2. **Lazy Initialization**: OpenAI clients and expensive resources are initialized on first use
3. **Environment-Driven**: Configuration loaded from environment variables with `.env` support
4. **Consistent Error Handling**: All exceptions serialize to JSON for API responses
5. **Retry Logic**: Built-in exponential backoff for transient failures

## Integration with FastAPI

Exceptions automatically convert to HTTP responses:

```python
from fastapi import HTTPException
from components.base.exceptions import ComponentError

@router.post("/classify")
async def classify(request: ClassificationRequest):
    try:
        return await service.process(request)
    except ComponentError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
```

## Best Practices

1. **Always inherit from BaseComponent** for new services
2. **Use ComponentConfig subclasses** with `env_prefix` for component-specific settings
3. **Raise specific exceptions** (ConfigurationError, ProcessingError, etc.)
4. **Implement health_check()** to return meaningful diagnostics
5. **Use `component_name`** consistently in logs and error messages
