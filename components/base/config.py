"""
Base configuration class for all components.

Uses pydantic-settings for environment variable loading.
Each component extends this with its own settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ComponentConfig(BaseSettings):
    """
    Base configuration for all component services.

    Settings are loaded from:
    1. Environment variables
    2. .env file (if present)
    3. Default values

    Subclasses should override model_config to set env_prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )

    # OpenAI Configuration (shared across components)
    openai_api_key: str = Field(
        default=None,
        description="OpenAI API key for LLM and embedding calls",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for external API calls",
    )
    retry_delay_seconds: int = Field(
        default=2,
        description="Base delay between retries (uses exponential backoff)",
    )

    def __init__(self, **kwargs):
        # Allow openai_api_key to be loaded from OPENAI_API_KEY env var
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        super().__init__(**kwargs)

    def validate_required(self, *fields: str) -> None:
        """
        Validate that required fields are set.

        Args:
            *fields: Field names to validate

        Raises:
            ConfigurationError: If any required field is missing
        """
        from components.base.exceptions import ConfigurationError

        missing = []
        for field in fields:
            value = getattr(self, field, None)
            if value is None or value == "":
                missing.append(field)

        if missing:
            raise ConfigurationError(
                f"Missing required configuration: {', '.join(missing)}",
                component=self.__class__.__name__,
                missing_keys=missing,
            )
