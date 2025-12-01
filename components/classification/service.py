"""
Classification Service Component.

Classifies tickets into domains (MM, CIW, Specialty) using
parallel binary classifiers (MTC-LLM approach).
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple

from openai import AsyncOpenAI
from openai import RateLimitError, APIError, APITimeoutError

from components.base import BaseComponent, ComponentConfig
from components.base.exceptions import ProcessingError, ConfigurationError
from components.classification.models import (
    ClassificationRequest,
    ClassificationResponse,
    DomainScore,
)


# Domain-specific prompts
DOMAIN_PROMPTS = {
    "MM": """You are an AI classifier for MM (Member Management) domain tickets in a healthcare technical support system.

Definition: An MM domain ticket involves:
- MM_ALDER service or module issues
- MM core service, integration layer, or data processor problems
- Database connection issues specific to MM components
- Batch jobs, APIs, or services prefixed with "MM"
- Member management, eligibility, or enrollment system issues
- MMALDR (MM ALDER) specific problems

Chain-of-Thought Process:
1. Extract technical terms and component names from ticket
2. Identify MM-specific keywords (MM_ALDER, MMALDR, mm service, member management, etc.)
3. Check for database/service/API patterns related to member systems
4. Evaluate if the issue is clearly within MM domain
5. Determine confidence level based on keyword matches and context clarity

Ticket:
Title: {title}
Description: {description}

Analyze this ticket and determine if it belongs to the MM domain.

Output Format (JSON only):
{{
  "decision": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation of the classification decision",
  "extracted_keywords": ["keyword1", "keyword2", ...]
}}

Respond ONLY with valid JSON. Do not include any other text.""",
    "CIW": """You are an AI classifier for CIW (Claims Integration Workflow) domain tickets in a healthcare technical support system.

Definition: A CIW domain ticket involves:
- CIW integration service issues
- Claims processing, submission, or validation problems
- Provider lookup, eligibility verification, or authorization systems
- Data synchronization from upstream claims systems
- CIW-specific modules, APIs, or batch processes
- Integration failures with external claims systems

Chain-of-Thought Process:
1. Extract technical terms and component names from ticket
2. Identify CIW-specific keywords (CIW, claims, provider, eligibility, authorization, etc.)
3. Check for integration, sync, or claims processing patterns
4. Evaluate if the issue is clearly within CIW domain
5. Determine confidence level based on keyword matches and context clarity

Ticket:
Title: {title}
Description: {description}

Analyze this ticket and determine if it belongs to the CIW domain.

Output Format (JSON only):
{{
  "decision": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation of the classification decision",
  "extracted_keywords": ["keyword1", "keyword2", ...]
}}

Respond ONLY with valid JSON. Do not include any other text.""",
    "Specialty": """You are an AI classifier for Specialty domain tickets in a healthcare technical support system.

Definition: A Specialty domain ticket involves:
- Custom specialty modules or applications
- Specialized workflow engines or custom workflows
- Specialty reporting, analytics, or dashboard issues
- Custom form builders or UI components
- Specialty-specific features not part of MM or CIW
- Frontend/UI issues in specialty applications
- Custom integrations or extensions

Chain-of-Thought Process:
1. Extract technical terms and component names from ticket
2. Identify specialty-specific keywords (custom, specialty, workflow, report, dashboard, etc.)
3. Check for frontend/UI patterns, custom modules, or specialized features
4. Evaluate if the issue is clearly within Specialty domain
5. Determine confidence level based on keyword matches and context clarity

Ticket:
Title: {title}
Description: {description}

Analyze this ticket and determine if it belongs to the Specialty domain.

Output Format (JSON only):
{{
  "decision": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation of the classification decision",
  "extracted_keywords": ["keyword1", "keyword2", ...]
}}

Respond ONLY with valid JSON. Do not include any other text.""",
}


class ClassificationConfig(ComponentConfig):
    """Configuration for Classification Service."""

    # Model settings
    classification_model: str = "gpt-4o"
    classification_temperature: float = 0.2

    # Available domains
    domains: List[str] = ["MM", "CIW", "Specialty"]

    # Confidence threshold for classification
    confidence_threshold: float = 0.7

    class Config:
        env_prefix = "CLASSIFICATION_"


class ClassificationService(
    BaseComponent[ClassificationRequest, ClassificationResponse]
):
    """
    Service for classifying tickets into domains.

    Uses parallel binary classifiers (MTC-LLM approach):
    - One classifier per domain (MM, CIW, Specialty)
    - Classifiers run in parallel
    - Final domain = highest confidence with decision=True

    Usage:
        service = ClassificationService()
        response = await service.process(
            ClassificationRequest(
                title="MM_ALDER timeout",
                description="Connection timeout in member database"
            )
        )
        print(response.classified_domain)  # "MM"
    """

    def __init__(self, config: Optional[ClassificationConfig] = None):
        """
        Initialize the classification service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ClassificationConfig()
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
        return "classification"

    async def _classify_domain(
        self, domain: str, title: str, description: str
    ) -> Dict[str, Any]:
        """
        Run binary classifier for a specific domain.

        Args:
            domain: Domain to classify for (MM, CIW, Specialty)
            title: Ticket title
            description: Ticket description

        Returns:
            Classification result dict
        """
        if domain not in DOMAIN_PROMPTS:
            raise ProcessingError(
                f"Unknown domain: {domain}",
                component=self.component_name,
                stage="prompt_generation",
            )

        prompt = DOMAIN_PROMPTS[domain].format(title=title, description=description)

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.classification_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a domain classification expert for healthcare ticketing systems.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.classification_temperature,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )

                import json

                return json.loads(response.choices[0].message.content)

            except (RateLimitError, APITimeoutError) as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay_seconds * (2**attempt)
                    await asyncio.sleep(wait_time)
                else:
                    return {
                        "decision": False,
                        "confidence": 0.0,
                        "reasoning": f"API error after retries: {str(e)}",
                        "extracted_keywords": [],
                    }

            except Exception as e:
                return {
                    "decision": False,
                    "confidence": 0.0,
                    "reasoning": f"Classification error: {str(e)}",
                    "extracted_keywords": [],
                }

        return {
            "decision": False,
            "confidence": 0.0,
            "reasoning": "Failed after all retries",
            "extracted_keywords": [],
        }

    async def _classify_all_domains(
        self, title: str, description: str
    ) -> Dict[str, Dict[str, Any]]:
        """Run all binary classifiers in parallel."""
        results = await asyncio.gather(
            *[
                self._classify_domain(domain, title, description)
                for domain in self.config.domains
            ]
        )

        return {domain: result for domain, result in zip(self.config.domains, results)}

    def _determine_final_domain(
        self, classifications: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, float, str]:
        """
        Determine final domain from classifier results.

        Logic:
        - Prefer domains with decision=True
        - Select highest confidence among those
        - If no positive decisions, use highest raw confidence * 0.3

        Returns:
            Tuple of (domain, confidence, combined_reasoning)
        """
        domain_scores = {}

        for domain, result in classifications.items():
            if result.get("decision", False):
                domain_scores[domain] = result.get("confidence", 0.0)
            else:
                # Lower confidence for negative decisions
                domain_scores[domain] = result.get("confidence", 0.0) * 0.3

        final_domain = max(domain_scores, key=domain_scores.get)
        final_confidence = domain_scores[final_domain]

        # Build combined reasoning
        reasonings = []
        for domain, result in classifications.items():
            decision_text = "✓" if result.get("decision", False) else "✗"
            conf = result.get("confidence", 0.0)
            reason = result.get("reasoning", "No reasoning")[:100]
            reasonings.append(f"{decision_text} {domain} ({conf:.2f}): {reason}")

        combined_reasoning = (
            f"Selected {final_domain} with confidence {final_confidence:.2f}.\n"
            + "\n".join(reasonings)
        )

        return final_domain, final_confidence, combined_reasoning

    async def process(
        self, request: ClassificationRequest
    ) -> ClassificationResponse:
        """
        Classify a ticket into a domain.

        Args:
            request: ClassificationRequest with title and description

        Returns:
            ClassificationResponse with domain and scores
        """
        # Run all classifiers in parallel
        classifications = await self._classify_all_domains(
            request.title, request.description
        )

        # Determine final domain
        domain, confidence, reasoning = self._determine_final_domain(classifications)

        # Build domain scores
        domain_scores = {}
        all_keywords = []

        for d, result in classifications.items():
            keywords = result.get("extracted_keywords", [])
            all_keywords.extend(keywords)

            domain_scores[d] = DomainScore(
                domain=d,
                decision=result.get("decision", False),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ""),
                keywords=keywords,
            )

        return ClassificationResponse(
            classified_domain=domain,
            confidence=confidence,
            reasoning=reasoning,
            domain_scores=domain_scores,
            extracted_keywords=list(set(all_keywords)),
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check if classification service is healthy."""
        try:
            if not self.config.openai_api_key:
                return {
                    "status": "unhealthy",
                    "component": self.component_name,
                    "error": "OpenAI API key not configured",
                }

            # Quick test classification
            test_result = await self._classify_domain(
                "MM", "test ticket", "health check test"
            )

            return {
                "status": "healthy",
                "component": self.component_name,
                "model": self.config.classification_model,
                "domains": self.config.domains,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "component": self.component_name,
                "error": str(e),
            }
