"""
Classification agent using hierarchical binary classifiers (MTC-LLM approach).

Uses 3 parallel binary classifiers to determine domain: MM, CIW, or Specialty.
"""
import asyncio
import json
from typing import Dict, Any, List
from src.utils.openai_client import get_openai_client
from src.utils.config import Config
from src.prompts.classification_prompts import get_classification_prompt
from src.models.state_schema import TicketState, AgentOutput


class ClassificationAgent:
    """Agent for classifying tickets into domains using binary classifiers."""

    def __init__(self):
        self.client = get_openai_client()
        self.model = Config.CLASSIFICATION_MODEL
        self.temperature = Config.CLASSIFICATION_TEMPERATURE
        self.domains = ['MM', 'CIW', 'Specialty']

    async def classify_domain(self, domain: str, title: str, description: str) -> Dict[str, Any]:
        """
        Run a single binary classifier for a specific domain.

        Args:
            domain: Domain name to classify for
            title: Ticket title
            description: Ticket description

        Returns:
            Dict with decision, confidence, reasoning, and keywords
        """
        try:
            # Get domain-specific prompt
            prompt = get_classification_prompt(domain, title, description)

            # Call OpenAI with JSON mode
            response = await self.client.chat_completion_json(
                messages=[
                    {"role": "system", "content": "You are a domain classification expert for healthcare ticketing systems."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=500
            )

            return response

        except Exception as e:
            print(f"⚠️ Error in {domain} classifier: {str(e)}")
            # Return default response on error
            return {
                "decision": False,
                "confidence": 0.0,
                "reasoning": f"Error during classification: {str(e)}",
                "extracted_keywords": []
            }

    async def classify_all_domains(self, title: str, description: str) -> Dict[str, Dict[str, Any]]:
        """
        Run all 3 binary classifiers in parallel.

        Args:
            title: Ticket title
            description: Ticket description

        Returns:
            Dict mapping domain name to classification result
        """
        # Run all classifiers concurrently
        results = await asyncio.gather(
            self.classify_domain('MM', title, description),
            self.classify_domain('CIW', title, description),
            self.classify_domain('Specialty', title, description)
        )

        return {
            'MM': results[0],
            'CIW': results[1],
            'Specialty': results[2]
        }

    def determine_final_domain(self, classifications: Dict[str, Dict[str, Any]]) -> tuple[str, float, str]:
        """
        Determine final domain based on binary classifier results.

        Uses the domain with highest confidence score.
        If multiple domains have high confidence, prioritize based on decision=True.

        Args:
            classifications: Results from all binary classifiers

        Returns:
            Tuple of (domain_name, confidence, reasoning)
        """
        # Extract confidence scores for domains that returned decision=True
        domain_scores = {}
        for domain, result in classifications.items():
            if result['decision']:
                domain_scores[domain] = result['confidence']
            else:
                # Lower confidence for negative decisions
                domain_scores[domain] = result['confidence'] * 0.3

        # Find domain with highest confidence
        final_domain = max(domain_scores, key=domain_scores.get)
        final_confidence = domain_scores[final_domain]

        # Build reasoning from all classifiers
        reasonings = []
        for domain, result in classifications.items():
            decision_text = "✓" if result['decision'] else "✗"
            reasonings.append(
                f"{decision_text} {domain} ({result['confidence']:.2f}): {result['reasoning'][:100]}"
            )

        combined_reasoning = f"Selected {final_domain} with confidence {final_confidence:.2f}.\n" + "\n".join(reasonings)

        return final_domain, final_confidence, combined_reasoning

    async def __call__(self, state: TicketState) -> AgentOutput:
        """
        Main agent execution function for LangGraph.

        Args:
            state: Current ticket state

        Returns:
            Partial state update with classification results
        """
        try:
            title = state['title']
            description = state['description']

            print(f"\n🔍 Classification Agent - Analyzing ticket: {state.get('ticket_id', 'N/A')}")

            # Run all binary classifiers
            classifications = await self.classify_all_domains(title, description)

            # Determine final domain
            domain, confidence, reasoning = self.determine_final_domain(classifications)

            # Extract all keywords
            all_keywords = []
            for result in classifications.values():
                all_keywords.extend(result.get('extracted_keywords', []))

            # Build confidence scores dict
            confidence_scores = {
                d: classifications[d]['confidence']
                for d in self.domains
            }

            print(f"   ✅ Classified as: {domain} (confidence: {confidence:.2%})")
            print(f"   📊 Scores: {', '.join([f'{d}={v:.2f}' for d, v in confidence_scores.items()])}")

            # Return state update
            return {
                "classified_domain": domain,
                "classification_confidence": confidence,
                "classification_reasoning": reasoning,
                "classification_scores": confidence_scores,
                "extracted_keywords": list(set(all_keywords)),  # Remove duplicates
                "status": "success",
                "current_agent": "Domain Classification Agent",
                "messages": [{
                    "role": "assistant",
                    "content": f"Classified ticket as {domain} domain with {confidence:.2%} confidence"
                }]
            }

        except Exception as e:
            print(f"   ❌ Classification error: {str(e)}")
            return {
                "status": "error",
                "current_agent": "Domain Classification Agent",
                "error_message": f"Classification failed: {str(e)}"
            }


# Create singleton instance for use in LangGraph
classification_agent = ClassificationAgent()
