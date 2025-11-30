"""
Configuration management using environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for application settings."""

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Model Configuration
    CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o")
    RESOLUTION_MODEL = os.getenv("RESOLUTION_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # Temperature settings
    CLASSIFICATION_TEMPERATURE = float(os.getenv("CLASSIFICATION_TEMPERATURE", "0.2"))
    RESOLUTION_TEMPERATURE = float(os.getenv("RESOLUTION_TEMPERATURE", "0.6"))

    # Token limits
    MAX_RESOLUTION_TOKENS = int(os.getenv("MAX_RESOLUTION_TOKENS", "8000"))

    # Vector Database Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    FAISS_INDEX_PATH = PROJECT_ROOT / os.getenv(
        "FAISS_INDEX_PATH", "data/faiss_index/tickets.index"
    )
    FAISS_METADATA_PATH = PROJECT_ROOT / os.getenv(
        "FAISS_METADATA_PATH", "data/faiss_index/metadata.json"
    )

    # Processing Configuration
    TOP_K_SIMILAR_TICKETS = int(os.getenv("TOP_K_SIMILAR_TICKETS", "20"))
    CLASSIFICATION_CONFIDENCE_THRESHOLD = float(
        os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.7")
    )
    LABEL_CONFIDENCE_THRESHOLD = float(os.getenv("LABEL_CONFIDENCE_THRESHOLD", "0.7"))

    # AI Label Generation Configuration
    LABEL_GENERATION_ENABLED = os.getenv("LABEL_GENERATION_ENABLED", "true").lower() == "true"
    BUSINESS_LABEL_MAX_COUNT = int(os.getenv("BUSINESS_LABEL_MAX_COUNT", "5"))
    TECHNICAL_LABEL_MAX_COUNT = int(os.getenv("TECHNICAL_LABEL_MAX_COUNT", "5"))
    GENERATED_LABEL_CONFIDENCE_THRESHOLD = float(
        os.getenv("GENERATED_LABEL_CONFIDENCE_THRESHOLD", "0.7")
    )

    # Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "2"))

    # Embedding Configuration
    EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large dimensions

    @classmethod
    def validate(cls):
        """Validate configuration settings."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

        if cls.CLASSIFICATION_MODEL not in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
            raise ValueError(f"Invalid CLASSIFICATION_MODEL: {cls.CLASSIFICATION_MODEL}")

        if cls.EMBEDDING_MODEL not in [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ]:
            raise ValueError(f"Invalid EMBEDDING_MODEL: {cls.EMBEDDING_MODEL}")

        return True


# Validate configuration on import
Config.validate()
