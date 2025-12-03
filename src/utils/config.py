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

    # Data Source Configuration
    HISTORICAL_TICKETS_CSV = os.getenv("HISTORICAL_TICKETS_CSV", "historical_tickets.csv")
    HISTORICAL_TICKETS_PATH = PROJECT_ROOT / "data" / "raw" / HISTORICAL_TICKETS_CSV

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

    # ========== CATEGORY LABELING CONFIGURATION ==========
    # Path to categories taxonomy file
    CATEGORIES_JSON_PATH = PROJECT_ROOT / "data" / "metadata" / "categories.json"

    # Category assignment thresholds (centralized - not from .env)
    CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # Used when category has no specific threshold
    CATEGORY_MAX_LABELS_PER_TICKET = 3           # Maximum categories to assign
    CATEGORY_NOVELTY_DETECTION_THRESHOLD = 0.5   # Below this = potential novel category

    # Category classification model settings
    CATEGORY_CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "gpt-4o")
    CATEGORY_CLASSIFICATION_TEMPERATURE = 0.2  # Low temperature for deterministic classification

    # ========== HYBRID SEMANTIC + LLM CLASSIFICATION ==========
    # Pre-computed category embeddings path (JSON format for safe serialization)
    CATEGORY_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "metadata" / "category_embeddings.json"

    # Semantic pre-filtering thresholds
    SEMANTIC_TOP_K_CANDIDATES = 5        # Number of candidates from semantic search
    SEMANTIC_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to consider a candidate

    # Ensemble scoring weights (must sum to 1.0)
    ENSEMBLE_SEMANTIC_WEIGHT = 0.4       # 40% weight for semantic similarity
    ENSEMBLE_LLM_WEIGHT = 0.6            # 60% weight for LLM binary classifier confidence

    # ========== NOVELTY DETECTION CONFIGURATION ==========
    # Signal 1: Maximum Confidence Score
    # If best category match has confidence below this, Signal 1 fires
    NOVELTY_SIGNAL1_THRESHOLD = float(os.getenv("NOVELTY_SIGNAL1_THRESHOLD", "0.5"))

    # Signal 2: Confidence Distribution Entropy
    # If normalized entropy exceeds this, Signal 2 fires (high uncertainty)
    NOVELTY_SIGNAL2_THRESHOLD = float(os.getenv("NOVELTY_SIGNAL2_THRESHOLD", "0.7"))

    # Signal 3: Embedding Distance to Centroids
    # If min distance to any category centroid exceeds this, Signal 3 fires
    NOVELTY_SIGNAL3_THRESHOLD = float(os.getenv("NOVELTY_SIGNAL3_THRESHOLD", "0.4"))

    # Signal weights (must sum to 1.0)
    NOVELTY_SIGNAL1_WEIGHT = float(os.getenv("NOVELTY_SIGNAL1_WEIGHT", "0.4"))
    NOVELTY_SIGNAL2_WEIGHT = float(os.getenv("NOVELTY_SIGNAL2_WEIGHT", "0.3"))
    NOVELTY_SIGNAL3_WEIGHT = float(os.getenv("NOVELTY_SIGNAL3_WEIGHT", "0.3"))

    # Final novelty decision threshold
    # is_novel = (max_confidence < SIGNAL1_THRESHOLD) OR (novelty_score > NOVELTY_SCORE_THRESHOLD)
    NOVELTY_SCORE_THRESHOLD = float(os.getenv("NOVELTY_SCORE_THRESHOLD", "0.6"))

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
