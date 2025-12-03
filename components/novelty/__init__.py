"""
Novelty Detection Component - Detects when tickets don't match known categories.

This component implements multi-signal novelty detection:
1. Signal 1: Maximum confidence score analysis
2. Signal 2: Confidence distribution entropy
3. Signal 3: Embedding distance to category centroids

Usage:
    from components.novelty.agent import novelty_node
    from components.novelty.tools import detect_novelty
"""

from components.novelty.agent import novelty_node
from components.novelty.tools import detect_novelty

__all__ = ["novelty_node", "detect_novelty"]
