"""
Classification Component - LangChain agent for domain classification.

Classifies tickets into domains (MM, CIW, Specialty) using
parallel binary classifiers (MTC-LLM approach).

Usage:
    # As LangGraph node (preferred)
    from components.classification import classification_node
    workflow.add_node("classification", classification_node)

    # As LangChain tool
    from components.classification import classify_ticket_domain
    result = await classify_ticket_domain.ainvoke({"title": "...", "description": "..."})

    # Legacy service (backward compatibility)
    from components.classification import ClassificationService, ClassificationRequest
    service = ClassificationService()
    response = await service.process(ClassificationRequest(...))
"""

from components.classification.models import (
    ClassificationRequest,
    ClassificationResponse,
    DomainScore,
)
from components.classification.service import ClassificationService, ClassificationConfig
from components.classification.router import router

# New LangChain agent and tools
from components.classification.agent import classification_node, classification_agent
from components.classification.tools import classify_ticket_domain

__all__ = [
    # LangGraph node (primary)
    "classification_node",
    "classification_agent",
    # LangChain tools
    "classify_ticket_domain",
    # Models
    "ClassificationRequest",
    "ClassificationResponse",
    "DomainScore",
    # Legacy service
    "ClassificationService",
    "ClassificationConfig",
    # Router
    "router",
]
