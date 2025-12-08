"""
Labeling Tools - LangChain @tool decorated functions for label assignment.

These tools handle category classification using a hybrid semantic + LLM approach:
1. Semantic pre-filtering: Compute cosine similarity between ticket and category embeddings
2. LLM binary classification: Run parallel binary classifiers for top-K candidates
3. Ensemble scoring: Combine 40% semantic + 60% LLM confidence
4. Filter and output: Apply threshold and limit to max categories

Uses LangChain's create_agent for agent-based label generation as LangGraph nodes.
Prompts are sourced from components/labeling/prompts.py.
"""

import json
import asyncio
from typing import Dict, Any, List, Tuple

from langchain_core.tools import tool
from langchain.agents import create_agent
from openai import OpenAI

from config.config import Config
from components.labeling.prompts import (
    BINARY_CLASSIFICATION_SYSTEM_PROMPT,
    BUSINESS_LABEL_SYSTEM_PROMPT,
    TECHNICAL_LABEL_SYSTEM_PROMPT,
    get_binary_classification_prompt,
    get_business_label_prompt,
    get_technical_label_prompt
)
from components.labeling.service import CategoryTaxonomy
from components.labeling.category_embeddings import CategoryEmbeddings


# ============================================================================
# RESPONSE TOOLS FOR LABELING AGENTS
# ============================================================================

@tool
def submit_classification_result(
    decision: bool,
    confidence: float,
    reasoning: str
) -> str:
    """
    Submit the binary classification result.

    The agent should call this tool after analyzing whether the ticket
    belongs to a specific category.

    Args:
        decision: True if the ticket belongs to this category, False otherwise
        confidence: Confidence score (0.0-1.0)
        reasoning: Brief explanation of the classification decision

    Returns:
        JSON string of the classification result
    """
    return json.dumps({
        "decision": decision,
        "confidence": confidence,
        "reasoning": reasoning
    })


@tool
def submit_business_labels(labels_json: str) -> str:
    """
    Submit the generated business labels.

    The agent should call this tool after analyzing the ticket from a
    business impact perspective.

    Args:
        labels_json: JSON string of business_labels array with label, confidence, category, reasoning

    Returns:
        JSON string of the business labels result
    """
    try:
        parsed = json.loads(labels_json) if labels_json else []
        # Handle both formats: array or object with business_labels key
        if isinstance(parsed, dict) and "business_labels" in parsed:
            labels = parsed["business_labels"]
        elif isinstance(parsed, list):
            labels = parsed
        else:
            labels = []
    except json.JSONDecodeError:
        labels = []
    return json.dumps({"business_labels": labels})


@tool
def submit_technical_labels(labels_json: str) -> str:
    """
    Submit the generated technical labels.

    The agent should call this tool after analyzing the ticket from a
    technical/root-cause perspective.

    Args:
        labels_json: JSON string of technical_labels array with label, confidence, category, reasoning

    Returns:
        JSON string of the technical labels result
    """
    try:
        parsed = json.loads(labels_json) if labels_json else []
        # Handle both formats: array or object with technical_labels key
        if isinstance(parsed, dict) and "technical_labels" in parsed:
            labels = parsed["technical_labels"]
        elif isinstance(parsed, list):
            labels = parsed
        else:
            labels = []
    except json.JSONDecodeError:
        labels = []
    return json.dumps({"technical_labels": labels})


# ============================================================================
# AGENT CREATION FUNCTIONS
# ============================================================================

def create_binary_classification_agent():
    """
    Create a LangChain agent for binary category classification.

    Uses langchain.agents.create_agent to build a graph-based agent runtime.
    The agent uses submit_classification_result tool to output its decision.

    Returns:
        CompiledStateGraph: A LangChain agent configured for binary classification
    """
    # Use model name string format for create_agent
    model_name = Config.CATEGORY_CLASSIFICATION_MODEL

    agent = create_agent(
        model=model_name,
        tools=[submit_classification_result],  # Tool for outputting classification
        system_prompt=BINARY_CLASSIFICATION_SYSTEM_PROMPT
    )

    return agent


def create_business_label_agent():
    """
    Create a LangChain agent for business label generation.

    Uses langchain.agents.create_agent to build a graph-based agent runtime.
    The agent uses submit_business_labels tool to output its generated labels.

    Returns:
        CompiledStateGraph: A LangChain agent configured for business label generation
    """
    # Use model name string format for create_agent
    model_name = Config.CLASSIFICATION_MODEL

    agent = create_agent(
        model=model_name,
        tools=[submit_business_labels],  # Tool for outputting business labels
        system_prompt=BUSINESS_LABEL_SYSTEM_PROMPT
    )

    return agent


def create_technical_label_agent():
    """
    Create a LangChain agent for technical label generation.

    Uses langchain.agents.create_agent to build a graph-based agent runtime.
    The agent uses submit_technical_labels tool to output its generated labels.

    Returns:
        CompiledStateGraph: A LangChain agent configured for technical label generation
    """
    # Use model name string format for create_agent
    model_name = Config.CLASSIFICATION_MODEL

    agent = create_agent(
        model=model_name,
        tools=[submit_technical_labels],  # Tool for outputting technical labels
        system_prompt=TECHNICAL_LABEL_SYSTEM_PROMPT
    )

    return agent


# Singleton agent instances (created lazily)
_binary_classification_agent = None
_business_label_agent = None
_technical_label_agent = None


def get_binary_classification_agent():
    """Get or create the binary classification agent singleton."""
    global _binary_classification_agent
    if _binary_classification_agent is None:
        _binary_classification_agent = create_binary_classification_agent()
    return _binary_classification_agent


def get_business_label_agent():
    """Get or create the business label agent singleton."""
    global _business_label_agent
    if _business_label_agent is None:
        _business_label_agent = create_business_label_agent()
    return _business_label_agent


def get_technical_label_agent():
    """Get or create the technical label agent singleton."""
    global _technical_label_agent
    if _technical_label_agent is None:
        _technical_label_agent = create_technical_label_agent()
    return _technical_label_agent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    category_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run binary classification for a single category using the agent.

    Args:
        title: Ticket title
        description: Ticket description
        priority: Ticket priority
        category_id: Category ID to evaluate
        category_info: Category metadata from taxonomy

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
        # Get the binary classification agent
        agent = get_binary_classification_agent()

        # Invoke the agent
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": prompt}]
        })

        # Extract the response from tool messages
        result_data = {}
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and "decision" in data:
                        result_data = data
                        break
                except (json.JSONDecodeError, TypeError):
                    continue

        return {
            "category_id": category_id,
            "decision": result_data.get("decision", False),
            "confidence": result_data.get("confidence", 0.0),
            "reasoning": result_data.get("reasoning", "")
        }

    except Exception as e:
        return {
            "category_id": category_id,
            "decision": False,
            "confidence": 0.0,
            "reasoning": f"Classification error: {str(e)}"
        }


# ============================================================================
# MAIN TOOL FUNCTIONS
# ============================================================================

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
    STEP 4: Run parallel binary classifiers using agents for TOP-5 (5 agent calls)
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
        # STEP 4: Run parallel binary classifiers using agents for TOP-K
        # ================================================================
        pipeline_info["step4_binary_classifiers"] = "in_progress"

        # Create tasks for parallel execution
        binary_tasks = []
        for cat_id, semantic_score in candidates:
            category_info = taxonomy.get_category_by_id(cat_id)
            if category_info:
                binary_tasks.append(
                    _binary_classify_category(
                        title, description, priority,
                        cat_id, category_info
                    )
                )

        # Run all binary classifiers in parallel
        binary_results = await asyncio.gather(*binary_tasks)
        pipeline_info["step4_binary_classifiers"] = f"completed ({len(binary_results)} agent calls)"

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
            # Pass through for novelty detection
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
    confidence_threshold: float = 0.7,
    similar_tickets: List[Dict] = None
) -> Dict[str, Any]:
    """
    Generate business-oriented labels using the business label agent.

    Analyzes the ticket from a business impact perspective and generates
    relevant business labels. Uses similar historical tickets for context.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate (default 5)
        confidence_threshold: Minimum confidence to include (default 0.7)
        similar_tickets: Top 5 similar tickets for historical context

    Returns:
        Dict with labels list and actual_prompt
    """
    # Generate the user prompt with similar tickets context
    user_prompt = get_business_label_prompt(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        existing_labels=existing_labels,
        max_labels=max_labels,
        similar_tickets=similar_tickets or []
    )

    try:
        # Get the business label agent
        agent = get_business_label_agent()

        # Invoke the agent
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        # Extract the response from tool messages
        result_data = {}
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and "business_labels" in data:
                        result_data = data
                        break
                except (json.JSONDecodeError, TypeError):
                    continue

        labels = []
        for item in result_data.get("business_labels", []):
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
    confidence_threshold: float = 0.7,
    similar_tickets: List[Dict] = None
) -> Dict[str, Any]:
    """
    Generate technical labels using the technical label agent.

    Analyzes the ticket from a technical/root-cause perspective and generates
    relevant technical labels. Uses similar historical tickets for context.

    Args:
        title: Ticket title
        description: Ticket description
        domain: Classified domain
        priority: Ticket priority
        existing_labels: Labels to avoid duplicating
        max_labels: Maximum labels to generate (default 5)
        confidence_threshold: Minimum confidence to include (default 0.7)
        similar_tickets: Top 5 similar tickets for historical context

    Returns:
        Dict with labels list and actual_prompt
    """
    # Generate the user prompt with similar tickets context
    user_prompt = get_technical_label_prompt(
        title=title,
        description=description,
        domain=domain,
        priority=priority,
        existing_labels=existing_labels,
        max_labels=max_labels,
        similar_tickets=similar_tickets or []
    )

    try:
        # Get the technical label agent
        agent = get_technical_label_agent()

        # Invoke the agent
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        # Extract the response from tool messages
        result_data = {}
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and "technical_labels" in data:
                        result_data = data
                        break
                except (json.JSONDecodeError, TypeError):
                    continue

        labels = []
        for item in result_data.get("technical_labels", []):
            if item.get("confidence", 0) >= confidence_threshold:
                labels.append({
                    "label": item.get("label", ""),
                    "confidence": item.get("confidence", 0.0),
                    "category": "technical",
                    "reasoning": item.get("reasoning", "")
                })

        return {"labels": labels, "actual_prompt": user_prompt}

    except Exception as e:
        print(f"   ⚠️  Technical label generation error: {str(e)}")
        return {"labels": [], "actual_prompt": user_prompt, "error": str(e)}
