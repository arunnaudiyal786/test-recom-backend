"""
Resolution Component - Generates resolution plans for tickets.

This component replaces the augmentation service and provides
LangChain-style tools and agent for resolution generation.
"""

from components.resolution.agent import resolution_node, resolution_agent
from components.resolution.tools import (
    generate_resolution_plan,
    analyze_similar_resolutions
)

__all__ = [
    "resolution_node",
    "resolution_agent",
    "generate_resolution_plan",
    "analyze_similar_resolutions"
]
