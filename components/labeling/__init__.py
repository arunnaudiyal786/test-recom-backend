"""
Labeling Component - LangChain agent for label assignment.

Assigns labels to tickets using three methods:
1. Historical Labels - from similar historical tickets
2. Business Labels - AI-generated from business perspective
3. Technical Labels - AI-generated from technical perspective

Usage:
    # As LangGraph node (preferred)
    from components.labeling import labeling_node
    workflow.add_node("labeling", labeling_node)

    # As LangChain tools
    from components.labeling import evaluate_historical_labels, generate_business_labels
    result = await evaluate_historical_labels.ainvoke({...})

    # Legacy service (backward compatibility)
    from components.labeling import LabelingService, LabelingRequest
    service = LabelingService()
    response = await service.process(LabelingRequest(...))
"""

from components.labeling.models import (
    LabelingRequest,
    LabelingResponse,
    LabelWithConfidence,
    SimilarTicketInput,
)
from components.labeling.service import LabelingService, LabelingConfig
from components.labeling.router import router

# New LangChain agent and tools
from components.labeling.agent import labeling_node, label_assignment_agent
from components.labeling.tools import (
    extract_candidate_labels,
    evaluate_historical_labels,
    generate_business_labels,
    generate_technical_labels
)

__all__ = [
    # LangGraph node (primary)
    "labeling_node",
    "label_assignment_agent",
    # LangChain tools
    "extract_candidate_labels",
    "evaluate_historical_labels",
    "generate_business_labels",
    "generate_technical_labels",
    # Models
    "LabelingRequest",
    "LabelingResponse",
    "LabelWithConfidence",
    "SimilarTicketInput",
    # Legacy service
    "LabelingService",
    "LabelingConfig",
    # Router
    "router",
]
