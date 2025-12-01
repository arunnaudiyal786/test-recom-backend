"""
Components as a Service (CaaS) - Modular RAG Components.

Each component can be:
1. Imported directly as a Python module
2. Exposed as FastAPI endpoints
3. Used independently or chained together

Usage:
    from components.embedding import EmbeddingService, EmbeddingRequest
    from components.retrieval import RetrievalService
    from components.classification import ClassificationService
    from components.labeling import LabelingService
    from components.augmentation import AugmentationService
    from components.orchestrator import OrchestratorService
"""

from components.base import BaseComponent, ComponentConfig, ComponentError

__all__ = [
    "BaseComponent",
    "ComponentConfig",
    "ComponentError",
]
