"""
Labeling Service Component.

Assigns labels to tickets using three methods:
1. Category Labels - from predefined taxonomy (categories.json)
2. Business Labels - AI-generated from business perspective
3. Technical Labels - AI-generated from technical perspective

Also includes CategoryTaxonomy - a singleton cache for category data.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from openai import AsyncOpenAI
from openai import RateLimitError, APIError, APITimeoutError

from components.base import BaseComponent, ComponentConfig
from components.base.exceptions import ProcessingError, ConfigurationError
from components.labeling.models import (
    LabelingRequest,
    LabelingResponse,
    LabelWithConfidence,
    CategoryLabel,
)
from src.utils.config import Config
from src.prompts.label_assignment_prompts import get_category_classification_prompt


# =============================================================================
# CATEGORY TAXONOMY (Singleton Cache)
# =============================================================================

class CategoryTaxonomy:
    """
    Singleton cache for category taxonomy from categories.json.

    Loads categories once at first access and provides lookup functions.
    Uses thresholds from centralized Config.

    Usage:
        taxonomy = CategoryTaxonomy.get_instance()
        categories = taxonomy.get_all_categories()
        threshold = taxonomy.get_confidence_threshold("batch_enrollment")
    """

    _instance: Optional["CategoryTaxonomy"] = None
    _categories: List[Dict[str, Any]] = []
    _categories_by_id: Dict[str, Dict[str, Any]] = {}
    _settings: Dict[str, Any] = {}

    def __new__(cls) -> "CategoryTaxonomy":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_categories()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "CategoryTaxonomy":
        """Get singleton instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None
        cls._categories = []
        cls._categories_by_id = {}
        cls._settings = {}

    def _load_categories(self) -> None:
        """Load categories from JSON file."""
        categories_path = Config.CATEGORIES_JSON_PATH

        if not categories_path.exists():
            raise FileNotFoundError(
                f"Categories file not found: {categories_path}. "
                "Please ensure data/metadata/categories.json exists."
            )

        with open(categories_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._categories = data.get("categories", [])
        self._settings = data.get("settings", {})

        # Build lookup index by ID
        self._categories_by_id = {
            cat["id"]: cat for cat in self._categories
        }

        print(f"   [CategoryTaxonomy] Loaded {len(self._categories)} categories")

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Return all categories with metadata."""
        return self._categories

    def get_category_by_id(self, category_id: str) -> Optional[Dict[str, Any]]:
        """Lookup category by ID."""
        return self._categories_by_id.get(category_id)

    def get_category_ids(self) -> List[str]:
        """Get list of all category IDs."""
        return list(self._categories_by_id.keys())

    def get_confidence_threshold(self, category_id: str) -> float:
        """
        Get confidence threshold for a category.

        Uses per-category threshold from JSON if available,
        otherwise falls back to Config.CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD.
        """
        category = self.get_category_by_id(category_id)
        if category and "confidence_threshold" in category:
            return category["confidence_threshold"]
        return Config.CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD

    def get_max_labels_per_ticket(self) -> int:
        """Get maximum labels per ticket from Config."""
        return Config.CATEGORY_MAX_LABELS_PER_TICKET

    def get_novelty_threshold(self) -> float:
        """Get novelty detection threshold from Config."""
        return Config.CATEGORY_NOVELTY_DETECTION_THRESHOLD

    def format_categories_for_prompt(self) -> str:
        """Format all categories for LLM prompt context."""
        lines = []
        for cat in self._categories:
            keywords_str = ", ".join(cat.get("keywords", [])[:5])
            examples_str = " | ".join(cat.get("examples", [])[:2])

            lines.append(
                f"**{cat['id']}** - {cat['name']}\n"
                f"  Description: {cat['description']}\n"
                f"  Keywords: {keywords_str}\n"
                f"  Examples: {examples_str}"
            )

        return "\n\n".join(lines)

    def format_categories_compact(self) -> str:
        """Format categories in a compact format for shorter prompts."""
        lines = []
        for cat in self._categories:
            keywords_str = ", ".join(cat.get("keywords", [])[:3])
            lines.append(f"- **{cat['id']}**: {cat['name']} ({keywords_str})")

        return "\n".join(lines)

    def validate_category_assignment(
        self,
        category_id: str,
        confidence: float
    ) -> bool:
        """Validate if a category assignment meets its threshold."""
        threshold = self.get_confidence_threshold(category_id)
        return confidence >= threshold

    def get_category_count(self) -> int:
        """Get total number of categories."""
        return len(self._categories)

    def get_settings(self) -> Dict[str, Any]:
        """Get settings from categories.json."""
        return self._settings


# =============================================================================
# LABELING SERVICE
# =============================================================================

class LabelingConfig(ComponentConfig):
    """Configuration for Labeling Service."""

    # Model settings
    labeling_model: str = "gpt-4o"
    labeling_temperature: float = 0.2

    # Confidence thresholds
    label_confidence_threshold: float = 0.7
    generated_label_confidence_threshold: float = 0.7

    # Label generation settings
    enable_ai_labels: bool = True
    max_business_labels: int = 5
    max_technical_labels: int = 5

    class Config:
        env_prefix = "LABELING_"


class LabelingService(BaseComponent[LabelingRequest, LabelingResponse]):
    """
    Service for assigning labels to tickets.

    Uses three-tier labeling:
    1. Category labels from predefined taxonomy
    2. Business labels (AI-generated)
    3. Technical labels (AI-generated)

    Usage:
        service = LabelingService()
        response = await service.process(
            LabelingRequest(
                title="Database timeout",
                description="Connection errors in MM_ALDER",
                domain="MM",
                similar_tickets=[...]
            )
        )
    """

    def __init__(self, config: Optional[LabelingConfig] = None):
        """Initialize the labeling service."""
        self.config = config or LabelingConfig()
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
        return "labeling"

    async def _assign_category_labels(
        self,
        title: str,
        description: str,
        priority: str,
    ) -> tuple[List[CategoryLabel], bool, Optional[str]]:
        """
        Classify ticket into predefined categories.

        Returns:
            Tuple of (category_labels, novelty_detected, novelty_reasoning)
        """
        # Get taxonomy singleton
        taxonomy = CategoryTaxonomy.get_instance()

        # Get thresholds from centralized config
        max_categories = Config.CATEGORY_MAX_LABELS_PER_TICKET
        confidence_threshold = Config.CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD

        # Format categories for prompt
        category_definitions = taxonomy.format_categories_for_prompt()
        total_categories = taxonomy.get_category_count()

        # Build prompt
        prompt = get_category_classification_prompt(
            title=title,
            description=description,
            priority=priority,
            category_definitions=category_definitions,
            total_categories=total_categories,
            max_categories=max_categories,
            confidence_threshold=confidence_threshold
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.config.labeling_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a ticket classification specialist. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.labeling_temperature,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Process categories - filter by per-category threshold
            category_labels = []
            for cat in result.get("categories", []):
                cat_id = cat.get("id", "")
                confidence = cat.get("confidence", 0.0)

                # Use per-category threshold if available
                threshold = taxonomy.get_confidence_threshold(cat_id)

                if confidence >= threshold:
                    # Validate category exists in taxonomy
                    category_info = taxonomy.get_category_by_id(cat_id)
                    if category_info:
                        category_labels.append(
                            CategoryLabel(
                                id=cat_id,
                                name=category_info.get("name", cat.get("name", "")),
                                confidence=confidence,
                                reasoning=cat.get("reasoning", "")
                            )
                        )

            # Limit to max categories
            category_labels = category_labels[:max_categories]

            # Check for novelty
            novelty_detected = result.get("novelty_detected", False)
            novelty_reasoning = result.get("novelty_reasoning")

            # If no categories assigned and novelty not flagged, detect novelty
            if not category_labels and not novelty_detected:
                novelty_detected = True
                novelty_reasoning = "No categories matched with sufficient confidence"

            return category_labels, novelty_detected, novelty_reasoning

        except Exception as e:
            return [], True, f"Classification error: {str(e)}"

    async def _generate_business_labels(
        self,
        title: str,
        description: str,
        domain: str,
        priority: str,
        existing_labels: Set[str],
    ) -> List[LabelWithConfidence]:
        """Generate business-oriented labels using AI."""
        prompt = f"""You are a business analyst expert in IT service management.

Generate business-oriented labels for this ticket from a business impact perspective.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {list(existing_labels)}

Business label categories to consider:
- Impact: Customer-facing, Internal, Revenue-impacting
- Urgency: Time-sensitive, Compliance-related, SLA-bound
- Process: Workflow-blocking, Data-quality, Integration-issue

Output JSON with up to {self.config.max_business_labels} labels:
{{
  "business_labels": [
    {{
      "label": "label name",
      "confidence": 0.0-1.0,
      "category": "Impact/Urgency/Process",
      "reasoning": "brief explanation"
    }}
  ],
  "business_summary": "one-line business impact summary"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.labeling_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a business analyst expert in IT service management.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            labels = []
            for item in result.get("business_labels", []):
                if (
                    item.get("confidence", 0)
                    >= self.config.generated_label_confidence_threshold
                ):
                    labels.append(
                        LabelWithConfidence(
                            label=item.get("label", ""),
                            confidence=item.get("confidence", 0.0),
                            category="business",
                            reasoning=item.get("reasoning", ""),
                        )
                    )

            return labels

        except Exception as e:
            return []

    async def _generate_technical_labels(
        self,
        title: str,
        description: str,
        domain: str,
        priority: str,
        existing_labels: Set[str],
    ) -> List[LabelWithConfidence]:
        """Generate technical labels using AI."""
        prompt = f"""You are a senior software engineer expert in system diagnostics.

Generate technical labels for this ticket from a technical/root-cause perspective.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {list(existing_labels)}

Technical label categories to consider:
- Component: Database, API, UI, Integration, Batch
- Issue Type: Performance, Error, Configuration, Data
- Root Cause: Connection, Timeout, Memory, Logic, External

Output JSON with up to {self.config.max_technical_labels} labels:
{{
  "technical_labels": [
    {{
      "label": "label name",
      "confidence": 0.0-1.0,
      "category": "Component/Issue Type/Root Cause",
      "reasoning": "brief explanation"
    }}
  ],
  "root_cause_hypothesis": "likely root cause summary"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.labeling_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior software engineer expert in system diagnostics.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            labels = []
            for item in result.get("technical_labels", []):
                if (
                    item.get("confidence", 0)
                    >= self.config.generated_label_confidence_threshold
                ):
                    labels.append(
                        LabelWithConfidence(
                            label=item.get("label", ""),
                            confidence=item.get("confidence", 0.0),
                            category="technical",
                            reasoning=item.get("reasoning", ""),
                        )
                    )

            return labels

        except Exception as e:
            return []

    async def process(self, request: LabelingRequest) -> LabelingResponse:
        """Assign labels to a ticket."""
        if self.config.enable_ai_labels:
            # Run all label methods in parallel
            category_task = self._assign_category_labels(
                request.title,
                request.description,
                request.priority,
            )
            business_task = self._generate_business_labels(
                request.title,
                request.description,
                request.domain,
                request.priority,
                set(),
            )
            technical_task = self._generate_technical_labels(
                request.title,
                request.description,
                request.domain,
                request.priority,
                set(),
            )

            (category_labels, novelty_detected, novelty_reasoning), business_labels, technical_labels = (
                await asyncio.gather(category_task, business_task, technical_task)
            )
        else:
            category_labels, novelty_detected, novelty_reasoning = await self._assign_category_labels(
                request.title,
                request.description,
                request.priority,
            )
            business_labels = []
            technical_labels = []

        # Combine all unique labels
        all_labels = set()
        for label in category_labels:
            all_labels.add(f"[CAT] {label.name}")
        for label in business_labels:
            all_labels.add(f"[BIZ] {label.label}")
        for label in technical_labels:
            all_labels.add(f"[TECH] {label.label}")

        return LabelingResponse(
            category_labels=category_labels,
            business_labels=business_labels,
            technical_labels=technical_labels,
            all_labels=list(all_labels),
            novelty_detected=novelty_detected,
            novelty_reasoning=novelty_reasoning,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check if labeling service is healthy."""
        try:
            if not self.config.openai_api_key:
                return {
                    "status": "unhealthy",
                    "component": self.component_name,
                    "error": "OpenAI API key not configured",
                }

            return {
                "status": "healthy",
                "component": self.component_name,
                "model": self.config.labeling_model,
                "ai_labels_enabled": self.config.enable_ai_labels,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "component": self.component_name,
                "error": str(e),
            }
