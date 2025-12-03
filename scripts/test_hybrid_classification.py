"""
Test script for hybrid semantic + LLM category classification.

Usage:
    cd test-recom-backend
    python scripts/test_hybrid_classification.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from components.labeling.tools import classify_ticket_categories
from components.labeling.category_embeddings import CategoryEmbeddings


async def test_hybrid_classification():
    """Test the hybrid classification pipeline with sample tickets."""

    print("\n" + "=" * 70)
    print("Hybrid Semantic + LLM Category Classification Test")
    print("=" * 70)

    # Check if embeddings are loaded
    embeddings = CategoryEmbeddings.get_instance()
    print(f"\n1. Category Embeddings:")
    print(f"   Loaded: {embeddings.is_loaded()}")
    print(f"   Categories: {len(embeddings)}")

    if not embeddings.is_loaded():
        print("   ERROR: Embeddings not loaded! Run: python scripts/generate_category_embeddings.py")
        return False

    # Sample tickets to test
    test_tickets = [
        {
            "title": "Member enrollment batch job failed with connection timeout",
            "description": "The nightly batch enrollment job ENRL_BATCH_001 failed at 2:30 AM with a connection timeout error. The job was processing 5000 member records when it encountered a database connection pool exhaustion. Error: 'Connection timeout after 30s'. This is blocking member enrollment updates.",
            "priority": "High"
        },
        {
            "title": "Need to add new route code for California region",
            "description": "We need to implement a new route code RC_CA_001 for the California region. This involves updating the routing tables and configuring the routing logic for proper claim routing. Please add this as part of the next release.",
            "priority": "Medium"
        },
        {
            "title": "ID card not printing correct member name",
            "description": "Members are reporting that their ID cards show incorrect names. The first name and last name appear to be swapped. This affects members with IDs starting with 'CA-2024'. Need to investigate the ID card generation process.",
            "priority": "High"
        }
    ]

    print("\n2. Testing Classification Pipeline:")
    print("-" * 70)

    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n[Ticket {i}]")
        print(f"Title: {ticket['title'][:60]}...")
        print(f"Priority: {ticket['priority']}")

        # Run classification
        result = await classify_ticket_categories.ainvoke({
            "title": ticket["title"],
            "description": ticket["description"],
            "priority": ticket["priority"]
        })

        # Display results
        print(f"\nPipeline Steps:")
        for step, status in result.get("pipeline_info", {}).items():
            print(f"   {step}: {status}")

        print(f"\nSemantic Candidates (Top-K):")
        for candidate in result.get("semantic_candidates", []):
            print(f"   - {candidate['category_id']}: {candidate['semantic_score']:.3f}")

        print(f"\nAssigned Categories:")
        if result.get("assigned_categories"):
            for cat in result["assigned_categories"]:
                print(f"   - {cat['id']} ({cat['name']})")
                print(f"     Confidence: {cat['confidence']:.3f} (sem: {cat.get('semantic_score', 0):.3f}, llm: {cat.get('llm_confidence', 0):.3f})")
                print(f"     Reasoning: {cat['reasoning'][:80]}...")
        else:
            print("   (none)")

        if result.get("novelty_detected"):
            print(f"\nNovelty Detected: {result['novelty_reasoning']}")

        print("-" * 70)

    print("\n3. Test Complete!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_hybrid_classification())
    sys.exit(0 if success else 1)
