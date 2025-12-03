"""
Workflow State Schema - TypedDict for LangGraph workflow state.

Defines the shared state structure that flows through all agents
in the ticket processing pipeline.
"""

from typing import TypedDict, List, Dict, Optional, Literal, Annotated
import operator


class TicketWorkflowState(TypedDict, total=False):
    """
    State schema for the LangGraph ticket processing workflow.

    Uses TypedDict with total=False to allow partial state updates.
    Fields marked with Annotated use reducers for state accumulation.
    """

    # ========== Input Fields ==========
    ticket_id: str
    title: str
    description: str
    priority: str
    metadata: Dict

    # ========== Session Management ==========
    session_id: Optional[str]  # Unique session ID for output storage (format: ddmmyyyyhhmm)

    # ========== Search Configuration (optional, from UI tuning) ==========
    search_config: Optional[Dict]

    # ========== Classification Output ==========
    classified_domain: Optional[str]  # "MM", "CIW", or "Specialty"
    classification_confidence: Optional[float]  # 0-1
    classification_reasoning: Optional[str]
    classification_scores: Optional[Dict[str, Dict]]  # Detailed scores per domain
    extracted_keywords: Optional[List[str]]

    # ========== Retrieval Output ==========
    similar_tickets: Optional[List[Dict]]  # List of similar ticket dicts
    similarity_scores: Optional[List[float]]
    search_metadata: Optional[Dict]

    # ========== Labeling Output ==========
    # Category Labels (from predefined taxonomy - replaces historical_labels)
    category_labels: Optional[List[Dict]]  # [{id, name, confidence, reasoning}]

    # AI-Generated Business Labels
    business_labels: Optional[List[Dict]]  # [{label, confidence, reasoning}]

    # AI-Generated Technical Labels
    technical_labels: Optional[List[Dict]]  # [{label, confidence, reasoning}]

    # Combined (for backward compatibility)
    assigned_labels: Optional[List[str]]

    # Ticket embedding (passed through for novelty detection)
    ticket_embedding: Optional[List[float]]  # 3072-dimensional embedding vector

    # All category similarity scores (for novelty detection entropy calculation)
    all_category_scores: Optional[List[Dict]]  # [{category_id, score}, ...]

    # ========== Novelty Detection Output ==========
    # Multi-signal novelty detection results
    novelty_detected: Optional[bool]  # True if ticket doesn't fit any category well
    novelty_score: Optional[float]  # Combined novelty score (0-1)
    novelty_signals: Optional[Dict]  # Individual signal details
    novelty_recommendation: Optional[str]  # "proceed" | "flag_for_review" | "escalate"
    novelty_reasoning: Optional[str]  # Human-readable explanation
    novelty_details: Optional[Dict]  # Additional details (signals_fired, nearest_category, etc.)

    # ========== Resolution Output ==========
    resolution_plan: Optional[Dict]
    resolution_confidence: Optional[float]

    # ========== Captured Prompts for UI Transparency ==========
    label_assignment_prompts: Optional[Dict[str, str]]  # {historical, business, technical}
    resolution_generation_prompt: Optional[str]  # Full resolution prompt sent to LLM

    # ========== Workflow Control ==========
    status: Literal["processing", "success", "error", "failed"]
    error_message: Optional[str]
    current_agent: str  # Name of current/last processing agent

    # ========== Accumulated Messages (uses reducer) ==========
    # Messages accumulate across agents for audit trail
    messages: Annotated[List[Dict], operator.add]

    # ========== Overall Metrics ==========
    overall_confidence: Optional[float]
    processing_time_seconds: Optional[float]


# Type alias for routing decisions
RoutingDecision = Literal[
    "retrieval",
    "labeling",
    "resolution",
    "error_handler",
    "end"
]
