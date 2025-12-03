"""
Labeling Component - LangChain agent for label assignment.

Assigns labels to tickets using three methods:
1. Category Labels - from predefined taxonomy (categories.json)
2. Business Labels - AI-generated from business perspective
3. Technical Labels - AI-generated from technical perspective

Usage:
    # As LangGraph node (preferred)
    from components.labeling import labeling_node
    workflow.add_node("labeling", labeling_node)

    # As LangChain tools
    from components.labeling import classify_ticket_categories, generate_business_labels
    result = await classify_ticket_categories.ainvoke({...})

    # Legacy service (backward compatibility)
    from components.labeling import LabelingService, LabelingRequest
    service = LabelingService()
    response = await service.process(LabelingRequest(...))
"""

from components.labeling.models import (
    LabelingRequest,
    LabelingResponse,
    LabelWithConfidence,
    CategoryLabel,
    SimilarTicketInput,
)
from components.labeling.service import LabelingService, LabelingConfig, CategoryTaxonomy
from components.labeling.category_embeddings import CategoryEmbeddings
from components.labeling.router import router

# New LangChain agent and tools
from components.labeling.agent import labeling_node, label_assignment_agent
from components.labeling.tools import (
    classify_ticket_categories,
    generate_business_labels,
    generate_technical_labels
)

__all__ = [
    # LangGraph node (primary)
    "labeling_node",
    "label_assignment_agent",
    # LangChain tools
    "classify_ticket_categories",
    "generate_business_labels",
    "generate_technical_labels",
    # Category taxonomy singleton
    "CategoryTaxonomy",
    # Category embeddings singleton (for hybrid classification)
    "CategoryEmbeddings",
    # Models
    "LabelingRequest",
    "LabelingResponse",
    "LabelWithConfidence",
    "CategoryLabel",
    "SimilarTicketInput",
    # Legacy service
    "LabelingService",
    "LabelingConfig",
    # Router
    "router",
]
