"""
Labeling Tools - LangChain @tool decorated functions for label assignment.

These tools handle category classification using a hybrid semantic + LLM approach:
1. Semantic pre-filtering: Compute cosine similarity between ticket and category embeddings
2. LLM binary classification: Run parallel binary classifiers for top-K candidates
3. Ensemble scoring: Combine 40% semantic + 60% LLM confidence
4. Filter and output: Apply threshold and limit to max categories
"""

import json
import asyncio
from typing import Dict, Any, List, Tuple

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.utils.config import Config
from src.prompts.label_assignment_prompts import (
    get_category_classification_prompt,
    get_binary_classification_prompt
)
from components.labeling.service import CategoryTaxonomy
from components.labeling.category_embeddings import CategoryEmbeddings


async def _generate_ticket_embedding(title: str, description: str) -> List[float]:
    """
    Generate embedding for a ticket using OpenAI's embedding model.

    Args:
        title: Ticket title
        description: Ticket description

    Returns:
        List of floats representing the embedding vector
    """
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Combine title and description for embedding
    text = f"{title} {description}"

    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


async def _binary_classify_category(
    title: str,
    description: str,
    priority: str,
    category_id: str,
    category_info: Dict[str, Any],
    llm: ChatOpenAI
) -> Dict[str, Any]:
    """
    Run binary classification for a single category.

    Args:
        title: Ticket title
        description: Ticket description
        priority: Ticket priority
        category_id: Category ID to evaluate
        category_info: Category metadata from taxonomy
        llm: ChatOpenAI instance

    Returns:
        Dict with decision, confidence, reasoning
    """
    # Format category details for prompt
    keywords = ", ".join(category_info.get("keywords", []))
    examples = ", ".join(category_info.get("examples", []))

    prompt = get_binary_classification_prompt(
        title=title,
        description=description,
        priority=priority,
        category_id=category_id,
        category_name=category_info.get("name", ""),
        category_description=category_info.get("description", ""),
        category_keywords=keywords,
        category_examples=examples
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are a ticket classification specialist. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ])

        result = json.loads(response.content)
        return {
            "category_id": category_id,
            "decision": result.get("decision", False),
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", "")
        }

    except Exception as e:
        return {
            "category_id": category_id,
            "decision": False,
            "confidence": 0.0,
            "reasoning": f"Classification error: {str(e)}"
        }


@tool
async def classify_ticket_categories(
    title: str,
    description: str,
    priority: str
) -> Dict[str, Any]:
    """
    Classify a ticket into categories using hybrid semantic + LLM approach.

    Pipeline:
    STEP 1: Generate ticket embedding (1 API call)
    STEP 2: Compute cosine similarity to all 25 category embeddings (no API call)
    STEP 3: Select TOP-5 candidates above similarity threshold (0.3)
    STEP 4: Run parallel binary classifiers for TOP-5 (5 API calls)
    STEP 5: Compute ensemble scores: 0.4 * semantic + 0.6 * LLM confidence
    STEP 6: Filter by threshold and limit to max 3 categories

    Args:
        title: Ticket title
        description: Ticket description
        priority: Ticket priority

    Returns:
        Dict containing:
        - assigned_categories: List of category dicts with id, name, confidence, reasoning
        - novelty_detected: Boolean if ticket might be a novel category
        - novelty_reasoning: Explanation if novelty detected
        - semantic_candidates: Top-K candidates from semantic search (for debugging)
        - pipeline_info: Execution info about each step
    """
    taxonomy = CategoryTaxonomy.get_instance()
    category_embeddings = CategoryEmbeddings.get_instance()

    # Config values
    top_k = Config.SEMANTIC_TOP_K_CANDIDATES
    semantic_threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD
    semantic_weight = Config.ENSEMBLE_SEMANTIC_WEIGHT
    llm_weight = Config.ENSEMBLE_LLM_WEIGHT
    max_categories = Config.CATEGORY_MAX_LABELS_PER_TICKET

    pipeline_info = {
        "step1_embedding": "pending",
        "step2_similarity": "pending",
        "step3_candidates": "pending",
        "step4_binary_classifiers": "pending",
        "step5_ensemble": "pending",
        "step6_filtering": "pending"
    }

    try:
        # ================================================================
        # STEP 1: Generate ticket embedding
        # ================================================================
        pipeline_info["step1_embedding"] = "in_progress"
        ticket_embedding = await _generate_ticket_embedding(title, description)
        pipeline_info["step1_embedding"] = "completed"

        # ================================================================
        # STEP 2: Compute cosine similarity to all category embeddings
        # ================================================================
        pipeline_info["step2_similarity"] = "in_progress"

        if not category_embeddings.is_loaded():
            return {
                "assigned_categories": [],
                "novelty_detected": True,
                "novelty_reasoning": "Category embeddings not loaded. Run: python scripts/generate_category_embeddings.py",
                "semantic_candidates": [],
                "pipeline_info": pipeline_info
            }

        all_similarities = category_embeddings.compute_similarities(ticket_embedding)
        pipeline_info["step2_similarity"] = f"completed ({len(all_similarities)} categories)"

        # ================================================================
        # STEP 3: Select TOP-K candidates above threshold
        # ================================================================
        pipeline_info["step3_candidates"] = "in_progress"
        candidates = category_embeddings.get_top_k_candidates(
            ticket_embedding,
            top_k=top_k,
            threshold=semantic_threshold
        )
        pipeline_info["step3_candidates"] = f"completed ({len(candidates)} candidates)"

        # Store semantic candidates for debugging
        semantic_candidates = [
            {"category_id": cat_id, "semantic_score": score}
            for cat_id, score in candidates
        ]

        # If no candidates above threshold, detect novelty
        if not candidates:
            pipeline_info["step4_binary_classifiers"] = "skipped (no candidates)"
            pipeline_info["step5_ensemble"] = "skipped"
            pipeline_info["step6_filtering"] = "skipped"

            return {
                "assigned_categories": [],
                "novelty_detected": True,
                "novelty_reasoning": f"No categories matched with semantic similarity >= {semantic_threshold}",
                "semantic_candidates": semantic_candidates,
                "pipeline_info": pipeline_info
            }

        # ================================================================
        # STEP 4: Run parallel binary classifiers for TOP-K candidates
        # ================================================================
        pipeline_info["step4_binary_classifiers"] = "in_progress"

        llm = ChatOpenAI(
            model=Config.CATEGORY_CLASSIFICATION_MODEL,
            temperature=Config.CATEGORY_CLASSIFICATION_TEMPERATURE,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        # Create tasks for parallel execution
        binary_tasks = []
        for cat_id, semantic_score in candidates:
            category_info = taxonomy.get_category_by_id(cat_id)
            if category_info:
                binary_tasks.append(
                    _binary_classify_category(
                        title, description, priority,
                        cat_id, category_info, llm
                    )
                )

        # Run all binary classifiers in parallel
        binary_results = await asyncio.gather(*binary_tasks)
        pipeline_info["step4_binary_classifiers"] = f"completed ({len(binary_results)} classifiers)"

        # ================================================================
        # STEP 5: Compute ensemble scores
        # ================================================================
        pipeline_info["step5_ensemble"] = "in_progress"

        # Create lookup for semantic scores
        semantic_scores = {cat_id: score for cat_id, score in candidates}

        # Compute ensemble scores for categories where decision=True
        ensemble_results = []
        for result in binary_results:
            if result["decision"]:
                cat_id = result["category_id"]
                semantic_score = semantic_scores.get(cat_id, 0.0)
                llm_confidence = result["confidence"]

                # Ensemble: 40% semantic + 60% LLM
                ensemble_score = (semantic_weight * semantic_score) + (llm_weight * llm_confidence)

                ensemble_results.append({
                    "category_id": cat_id,
                    "ensemble_score": ensemble_score,
                    "semantic_score": semantic_score,
                    "llm_confidence": llm_confidence,
                    "reasoning": result["reasoning"]
                })

        pipeline_info["step5_ensemble"] = f"completed ({len(ensemble_results)} positive decisions)"

        # ================================================================
        # STEP 6: Filter by threshold and limit to max categories
        # ================================================================
        pipeline_info["step6_filtering"] = "in_progress"

        # Sort by ensemble score descending
        ensemble_results.sort(key=lambda x: x["ensemble_score"], reverse=True)

        assigned_categories = []
        for result in ensemble_results:
            cat_id = result["category_id"]

            # Use per-category threshold if available
            threshold = taxonomy.get_confidence_threshold(cat_id)

            if result["ensemble_score"] >= threshold:
                category_info = taxonomy.get_category_by_id(cat_id)
                if category_info:
                    assigned_categories.append({
                        "id": cat_id,
                        "name": category_info.get("name", ""),
                        "confidence": round(result["ensemble_score"], 3),
                        "reasoning": result["reasoning"],
                        "semantic_score": round(result["semantic_score"], 3),
                        "llm_confidence": round(result["llm_confidence"], 3)
                    })

        # Limit to max categories
        assigned_categories = assigned_categories[:max_categories]
        pipeline_info["step6_filtering"] = f"completed ({len(assigned_categories)} categories assigned)"

        # Detect novelty if no categories assigned
        novelty_detected = len(assigned_categories) == 0
        novelty_reasoning = None
        if novelty_detected:
            novelty_reasoning = "No categories passed ensemble threshold after binary classification"

        # Format all similarity scores for novelty detection
        all_similarity_scores = [
            {"category_id": cat_id, "score": score}
            for cat_id, score in all_similarities
        ]

        return {
            "assigned_categories": assigned_categories,
            "novelty_detected": novelty_detected,
            "novelty_reasoning": novelty_reasoning,
            "semantic_candidates": semantic_candidates,
            "pipeline_info": pipeline_info,
            # NEW: Pass through for novelty detection
            "ticket_embedding": ticket_embedding,
            "all_similarity_scores": all_similarity_scores
        }

    except Exception as e:
        pipeline_info["error"] = str(e)
        return {
            "assigned_categories": [],
            "novelty_detected": True,
            "novelty_reasoning": f"Classification error: {str(e)}",
            "semantic_candidates": [],
            "pipeline_info": pipeline_info,
            "ticket_embedding": [],
            "all_similarity_scores": []
        }


@tool
async def generate_business_labels(
    title: str,
    description: str,
    domain: str,
    priority: str,
    existing_labels: List[str],
    max_labels: int = 5,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Generate business-oriented labels using AI analysis.

    Analyzes the ticket from a business impact perspective and generates
    relevant business labels.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate (default 5)
        confidence_threshold: Minimum confidence to include (default 0.7)

    Returns:
        Dict with labels list and actual_prompt
    """
    llm = ChatOpenAI(
        model=Config.CLASSIFICATION_MODEL,
        temperature=0.4,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    user_prompt = f"""You are a business analyst expert in IT service management.

Generate business-oriented labels for this ticket from a business impact perspective.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {existing_labels}

Business label categories to consider:
- Impact: Customer-facing, Internal, Revenue-impacting
- Urgency: Time-sensitive, Compliance-related, SLA-bound
- Process: Workflow-blocking, Data-quality, Integration-issue

Output JSON with up to {max_labels} labels:
{{
  "business_labels": [
    {{
      "label": "label name",
      "confidence": 0.0-1.0,
      "category": "Impact/Urgency/Process",
      "reasoning": "brief explanation"
    }}
  ]
}}"""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are a business analyst expert. Respond only with valid JSON."},
            {"role": "user", "content": user_prompt}
        ])

        result = json.loads(response.content)
        labels = []

        for item in result.get("business_labels", []):
            if item.get("confidence", 0) >= confidence_threshold:
                labels.append({
                    "label": item.get("label", ""),
                    "confidence": item.get("confidence", 0.0),
                    "category": "business",
                    "reasoning": item.get("reasoning", "")
                })

        return {"labels": labels, "actual_prompt": user_prompt}

    except Exception as e:
        return {"labels": [], "actual_prompt": user_prompt}


@tool
async def generate_technical_labels(
    title: str,
    description: str,
    domain: str,
    priority: str,
    existing_labels: List[str],
    max_labels: int = 5,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Generate technical labels using AI analysis.

    Analyzes the ticket from a technical/root-cause perspective and generates
    relevant technical labels.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate (default 5)
        confidence_threshold: Minimum confidence to include (default 0.7)

    Returns:
        Dict with labels list and actual_prompt
    """
    llm = ChatOpenAI(
        model=Config.CLASSIFICATION_MODEL,
        temperature=0.3,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    user_prompt = f"""You are a senior software engineer expert in system diagnostics.

Generate technical labels for this ticket from a technical/root-cause perspective.

Ticket:
Title: {title}
Description: {description}
Domain: {domain}
Priority: {priority}

Existing labels to avoid duplicating: {existing_labels}

Technical label categories to consider:
- Component: Database, API, UI, Integration, Batch
- Issue Type: Performance, Error, Configuration, Data
- Root Cause: Connection, Timeout, Memory, Logic, External

Output JSON with up to {max_labels} labels:
{{
  "technical_labels": [
    {{
      "label": "label name",
      "confidence": 0.0-1.0,
      "category": "Component/Issue Type/Root Cause",
      "reasoning": "brief explanation"
    }}
  ]
}}"""

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": "You are a software engineer expert. Respond only with valid JSON."},
            {"role": "user", "content": user_prompt}
        ])

        result = json.loads(response.content)
        labels = []

        for item in result.get("technical_labels", []):
            if item.get("confidence", 0) >= confidence_threshold:
                labels.append({
                    "label": item.get("label", ""),
                    "confidence": item.get("confidence", 0.0),
                    "category": "technical",
                    "reasoning": item.get("reasoning", "")
                })

        return {"labels": labels, "actual_prompt": user_prompt}

    except Exception as e:
        return {"labels": [], "actual_prompt": user_prompt}
