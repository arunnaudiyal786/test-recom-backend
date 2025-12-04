"""
Classification Tools - LangChain @tool decorated functions.

These tools perform domain classification using parallel binary classifiers.
Each tool is independently testable and can be used by any LangChain agent.

Domain definitions are loaded from config/schema_config.yaml for flexibility.
"""

import asyncio
import json
from typing import Dict, Any, List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config.config import Config
from src.utils.schema_config import get_schema_config


def _build_domain_prompt(domain: str, prompt_template: str) -> str:
    """
    Build a full classification prompt from the domain's template.

    Args:
        domain: Domain name
        prompt_template: Base prompt from config

    Returns:
        Full prompt with JSON output format
    """
    return f"""{prompt_template}

Ticket:
Title: {{title}}
Description: {{description}}

Analyze this ticket and determine if it belongs to the {domain} domain.

Output Format (JSON only):
{{{{
  "decision": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "step-by-step explanation",
  "extracted_keywords": ["keyword1", "keyword2"]
}}}}"""


def _get_domain_prompts() -> Dict[str, str]:
    """
    Get domain classification prompts from schema config.

    Returns:
        Dict mapping domain name to full classification prompt
    """
    schema_config = get_schema_config()
    classification_config = schema_config.get_classification_config()

    prompts = {}
    for domain, prompt_template in classification_config['prompts'].items():
        if prompt_template:
            prompts[domain] = _build_domain_prompt(domain, prompt_template)

    return prompts


def _get_configured_domains() -> List[str]:
    """Get list of configured domains from schema config."""
    schema_config = get_schema_config()
    return schema_config.domain_names


async def _classify_single_domain(
    domain: str,
    title: str,
    description: str,
    llm: ChatOpenAI,
    domain_prompts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Internal function to classify a single domain.

    Args:
        domain: Domain name (loaded from config)
        title: Ticket title
        description: Ticket description
        llm: ChatOpenAI instance
        domain_prompts: Dict of domain prompts from config

    Returns:
        Classification result dict
    """
    if domain not in domain_prompts:
        return {
            "decision": False,
            "confidence": 0.0,
            "reasoning": f"No prompt configured for domain: {domain}",
            "extracted_keywords": []
        }

    prompt = domain_prompts[domain].format(title=title, description=description)

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": "You are a domain classification expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        return json.loads(response.content)

    except Exception as e:
        return {
            "decision": False,
            "confidence": 0.0,
            "reasoning": f"Classification error: {str(e)}",
            "extracted_keywords": []
        }


@tool
async def classify_ticket_domain(title: str, description: str) -> Dict[str, Any]:
    """
    Classify a ticket into one of the configured domains.

    Runs parallel binary classifiers (one per domain from config) and determines
    the final domain based on the highest confidence score among positive decisions.

    Domains are loaded from config/schema_config.yaml.

    Args:
        title: The ticket title
        description: The ticket description

    Returns:
        Dict containing:
        - classified_domain: Final domain (from config)
        - confidence: Confidence score (0-1)
        - reasoning: Combined reasoning from all classifiers
        - domain_scores: Individual scores for each domain
        - extracted_keywords: Keywords found in the ticket
    """
    # Load domains and prompts from config
    domains = _get_configured_domains()
    domain_prompts = _get_domain_prompts()

    if not domains:
        return {
            "classified_domain": "Unknown",
            "confidence": 0.0,
            "reasoning": "No domains configured in schema_config.yaml",
            "domain_scores": {},
            "extracted_keywords": []
        }

    # Create LLM instance with JSON mode
    llm = ChatOpenAI(
        model=Config.CLASSIFICATION_MODEL,
        temperature=Config.CLASSIFICATION_TEMPERATURE,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # Run all binary classifiers in parallel
    results = await asyncio.gather(
        *[_classify_single_domain(d, title, description, llm, domain_prompts) for d in domains]
    )

    classifications = dict(zip(domains, results))

    # Determine final domain based on scores
    domain_scores = {}
    for domain, result in classifications.items():
        if result.get("decision", False):
            domain_scores[domain] = result.get("confidence", 0.0)
        else:
            # Lower score for negative decisions
            domain_scores[domain] = result.get("confidence", 0.0) * 0.3

    final_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "Unknown"
    final_confidence = domain_scores.get(final_domain, 0.0)

    # Build combined reasoning
    reasonings = []
    all_keywords = []
    for domain, result in classifications.items():
        decision_text = "✓" if result.get("decision", False) else "✗"
        conf = result.get("confidence", 0.0)
        reason = result.get("reasoning", "No reasoning")[:100]
        reasonings.append(f"{decision_text} {domain} ({conf:.2f}): {reason}")
        all_keywords.extend(result.get("extracted_keywords", []))

    combined_reasoning = (
        f"Selected {final_domain} with confidence {final_confidence:.2f}.\n"
        + "\n".join(reasonings)
    )

    return {
        "classified_domain": final_domain,
        "confidence": final_confidence,
        "reasoning": combined_reasoning,
        "domain_scores": {
            d: {
                "decision": classifications[d].get("decision", False),
                "confidence": classifications[d].get("confidence", 0.0),
                "reasoning": classifications[d].get("reasoning", "")
            }
            for d in domains
        },
        "extracted_keywords": list(set(all_keywords))
    }


# Synchronous wrapper for non-async contexts
def classify_ticket_domain_sync(title: str, description: str) -> Dict[str, Any]:
    """Synchronous wrapper for classify_ticket_domain."""
    return asyncio.run(classify_ticket_domain.ainvoke({"title": title, "description": description}))
