"""
LangGraph state schema using TypedDict.
"""
from typing import TypedDict, List, Dict, Optional, Literal, Annotated
import operator


class TicketState(TypedDict, total=False):
    """
    State schema for LangGraph workflow.

    Uses TypedDict with total=False to allow partial state updates.
    Fields marked with Annotated use reducers for state accumulation.
    """

    # ========== Input Fields ==========
    ticket_id: str
    title: str
    description: str
    priority: str
    metadata: Dict

    # ========== Search Configuration (optional, from UI tuning) ==========
    search_config: Optional[Dict]  # Custom search parameters from UI

    # ========== Classification Output ==========
    classified_domain: Optional[str]  # "MM", "CIW", or "Specialty"
    classification_confidence: Optional[float]  # 0-1
    classification_reasoning: Optional[str]
    classification_scores: Optional[Dict[str, float]]  # Scores for all domains
    extracted_keywords: Optional[List[str]]

    # ========== Pattern Recognition Output ==========
    similar_tickets: Optional[List[Dict]]  # List of similar ticket dictionaries
    similarity_scores: Optional[List[float]]
    search_metadata: Optional[Dict]

    # ========== Label Assignment Output ==========
    # Historical Labels (from similar tickets)
    historical_labels: Optional[List[str]]
    historical_label_confidence: Optional[Dict[str, float]]
    historical_label_distribution: Optional[Dict[str, str]]

    # AI-Generated Business Labels
    business_labels: Optional[List[Dict]]  # [{label, confidence, reasoning}]

    # AI-Generated Technical Labels
    technical_labels: Optional[List[Dict]]  # [{label, confidence, reasoning}]

    # Combined (for backward compatibility)
    assigned_labels: Optional[List[str]]
    label_confidence: Optional[Dict[str, float]]
    label_distribution: Optional[Dict[str, str]]

    # ========== Resolution Generation Output ==========
    resolution_plan: Optional[Dict]  # Complete resolution plan
    resolution_confidence: Optional[float]

    # ========== Workflow Control ==========
    processing_stage: str  # Current stage in workflow
    status: Literal["processing", "success", "error", "failed"]
    error_message: Optional[str]
    current_agent: str  # Name of agent currently/last processing

    # ========== Accumulated Messages (uses reducer) ==========
    # Messages accumulate across agents for audit trail
    messages: Annotated[List[Dict], operator.add]

    # ========== Overall Metrics ==========
    overall_confidence: Optional[float]
    processing_time_seconds: Optional[float]


class AgentOutput(TypedDict, total=False):
    """
    Standard output format for all agents.

    Each agent returns a partial state update with only the fields they modify.
    """

    # Required fields (all agents must provide)
    status: Literal["success", "error"]
    current_agent: str

    # Error handling (required if status == "error")
    error_message: Optional[str]

    # Agent-specific outputs (optional, depends on agent)
    classified_domain: Optional[str]
    classification_confidence: Optional[float]
    classification_reasoning: Optional[str]
    classification_scores: Optional[Dict[str, float]]
    extracted_keywords: Optional[List[str]]

    similar_tickets: Optional[List[Dict]]
    similarity_scores: Optional[List[float]]
    search_metadata: Optional[Dict]

    # Historical labels
    historical_labels: Optional[List[str]]
    historical_label_confidence: Optional[Dict[str, float]]
    historical_label_distribution: Optional[Dict[str, str]]

    # AI-Generated labels
    business_labels: Optional[List[Dict]]
    technical_labels: Optional[List[Dict]]

    # Combined (backward compatibility)
    assigned_labels: Optional[List[str]]
    label_confidence: Optional[Dict[str, float]]
    label_distribution: Optional[Dict[str, str]]

    resolution_plan: Optional[Dict]
    resolution_confidence: Optional[float]

    # Messages to add to audit trail
    messages: Optional[List[Dict]]


# Type aliases for routing decisions
RoutingDecision = Literal[
    "pattern_recognition",
    "label_assignment",
    "resolution_generation",
    "error_handler",
    "end"
]
