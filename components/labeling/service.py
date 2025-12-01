"""
Labeling Service Component.

Assigns labels to tickets using three methods:
1. Historical Labels - from similar historical tickets
2. Business Labels - AI-generated from business perspective
3. Technical Labels - AI-generated from technical perspective
"""

import asyncio
from typing import Dict, Any, List, Optional, Set

from openai import AsyncOpenAI
from openai import RateLimitError, APIError, APITimeoutError

from components.base import BaseComponent, ComponentConfig
from components.base.exceptions import ProcessingError, ConfigurationError
from components.labeling.models import (
    LabelingRequest,
    LabelingResponse,
    LabelWithConfidence,
)


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
    1. Historical labels from similar tickets (validated by AI)
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
        """
        Initialize the labeling service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
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

    def _extract_candidate_labels(
        self, similar_tickets: List[Dict[str, Any]]
    ) -> Set[str]:
        """Extract unique labels from similar tickets."""
        labels = set()
        for ticket in similar_tickets:
            ticket_labels = ticket.get("labels", [])
            if isinstance(ticket_labels, list):
                labels.update(ticket_labels)
        return labels

    def _calculate_label_distribution(
        self, similar_tickets: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate label frequency distribution."""
        label_counts = {}
        total = len(similar_tickets)

        for ticket in similar_tickets:
            for label in ticket.get("labels", []):
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

        return {
            label: {
                "count": count,
                "percentage": count / total if total > 0 else 0,
                "formatted": f"{count}/{total}",
            }
            for label, count in label_counts.items()
        }

    async def _evaluate_historical_label(
        self,
        label_name: str,
        title: str,
        description: str,
        domain: str,
        label_distribution: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate a single historical label using binary classifier."""
        frequency_info = label_distribution.get(label_name, {})
        frequency = frequency_info.get("formatted", "0/0")

        prompt = f"""You are a label validation expert for technical support tickets.

Evaluate whether the label "{label_name}" should be assigned to this ticket.

Historical frequency: This label appears in {frequency} similar historical tickets.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}

Consider:
1. Does the ticket content match the label semantics?
2. Is the historical frequency a strong indicator?
3. What is your confidence level?

Output JSON:
{{
  "assign_label": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.labeling_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a label classification expert for technical support systems.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.labeling_temperature,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            import json
            result = json.loads(response.choices[0].message.content)

            return {
                "label": label_name,
                "assign": result.get("assign_label", False),
                "confidence": result.get("confidence", 0.0),
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            return {
                "label": label_name,
                "assign": False,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
            }

    async def _assign_historical_labels(
        self,
        title: str,
        description: str,
        domain: str,
        similar_tickets: List[Dict[str, Any]],
    ) -> tuple[List[LabelWithConfidence], Dict[str, str]]:
        """Assign labels based on similar historical tickets."""
        candidate_labels = self._extract_candidate_labels(similar_tickets)
        if not candidate_labels:
            return [], {}

        label_distribution = self._calculate_label_distribution(similar_tickets)

        # Evaluate all labels in parallel
        tasks = [
            self._evaluate_historical_label(
                label, title, description, domain, label_distribution
            )
            for label in candidate_labels
        ]
        results = await asyncio.gather(*tasks)

        # Filter by confidence threshold
        historical_labels = []
        for result in results:
            if (
                result["assign"]
                and result["confidence"] >= self.config.label_confidence_threshold
            ):
                historical_labels.append(
                    LabelWithConfidence(
                        label=result["label"],
                        confidence=result["confidence"],
                        category="historical",
                        reasoning=result["reasoning"],
                    )
                )

        # Format distribution for output
        distribution = {
            label: info["formatted"]
            for label, info in label_distribution.items()
        }

        return historical_labels, distribution

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

            import json
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

            import json
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
        """
        Assign labels to a ticket.

        Args:
            request: LabelingRequest with ticket details and similar tickets

        Returns:
            LabelingResponse with all label categories
        """
        # Get existing labels from similar tickets
        existing_labels = self._extract_candidate_labels(request.similar_tickets)

        if self.config.enable_ai_labels:
            # Run all label methods in parallel
            historical_task = self._assign_historical_labels(
                request.title,
                request.description,
                request.domain,
                request.similar_tickets,
            )
            business_task = self._generate_business_labels(
                request.title,
                request.description,
                request.domain,
                request.priority,
                existing_labels,
            )
            technical_task = self._generate_technical_labels(
                request.title,
                request.description,
                request.domain,
                request.priority,
                existing_labels,
            )

            (historical_labels, distribution), business_labels, technical_labels = (
                await asyncio.gather(historical_task, business_task, technical_task)
            )
        else:
            historical_labels, distribution = await self._assign_historical_labels(
                request.title,
                request.description,
                request.domain,
                request.similar_tickets,
            )
            business_labels = []
            technical_labels = []

        # Combine all unique labels
        all_labels = set()
        for label in historical_labels:
            all_labels.add(label.label)
        for label in business_labels:
            all_labels.add(f"[BIZ] {label.label}")
        for label in technical_labels:
            all_labels.add(f"[TECH] {label.label}")

        return LabelingResponse(
            historical_labels=historical_labels,
            business_labels=business_labels,
            technical_labels=technical_labels,
            all_labels=list(all_labels),
            label_distribution=distribution,
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
