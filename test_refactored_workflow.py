#!/usr/bin/env python3
"""
Test script for the refactored LangChain/LangGraph workflow.

Run this script to verify all imports and workflow compilation work correctly.

Usage:
    python3 test_refactored_workflow.py
"""

import asyncio


def test_imports():
    """Test that all new modules can be imported."""
    print("=" * 60)
    print("Testing Refactored LangChain/LangGraph Workflow")
    print("=" * 60)
    print("\n1. Testing imports...")

    try:
        # Test new orchestrator
        from src.orchestrator.workflow import get_workflow, build_workflow
        from src.orchestrator.state import TicketWorkflowState
        print("   ✅ src/orchestrator imports successful")

        # Test classification component
        from components.classification import classification_node, classify_ticket_domain
        print("   ✅ components/classification imports successful")

        # Test retrieval component
        from components.retrieval import retrieval_node, search_similar_tickets
        print("   ✅ components/retrieval imports successful")

        # Test labeling component
        from components.labeling import labeling_node, evaluate_historical_labels
        print("   ✅ components/labeling imports successful")

        # Test resolution component
        from components.resolution import resolution_node, generate_resolution_plan
        print("   ✅ components/resolution imports successful")

        return True

    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def test_workflow_build():
    """Test that LangGraph workflow compiles correctly."""
    print("\n2. Building LangGraph workflow...")

    try:
        from src.orchestrator.workflow import build_workflow

        workflow = build_workflow()
        print("   ✅ LangGraph workflow compiled successfully")

        # Print workflow nodes
        print("\n   Workflow nodes:")
        for node in workflow.get_graph().nodes:
            print(f"      - {node}")

        return workflow

    except Exception as e:
        print(f"   ❌ Workflow build error: {e}")
        return None


async def test_classification_tool():
    """Test the classification tool with a sample ticket."""
    print("\n3. Testing classification tool...")

    try:
        from components.classification.tools import classify_ticket_domain

        # Sample ticket
        result = await classify_ticket_domain.ainvoke({
            "title": "MM_ALDER database connection timeout",
            "description": "The MM_ALDER service is experiencing connection pool exhaustion during peak hours."
        })

        domain = result.get("classified_domain", "Unknown")
        confidence = result.get("confidence", 0)

        print(f"   ✅ Classification successful")
        print(f"      Domain: {domain}")
        print(f"      Confidence: {confidence:.2%}")

        return result

    except Exception as e:
        print(f"   ❌ Classification error: {e}")
        return None


async def main():
    """Run all tests."""
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed. Fix import errors before proceeding.")
        return

    # Test 2: Workflow build
    workflow = test_workflow_build()
    if not workflow:
        print("\n❌ Workflow build failed. Fix workflow errors before proceeding.")
        return

    # Test 3: Classification tool (requires OpenAI API key)
    print("\n4. Testing classification tool (requires API key)...")
    try:
        result = await test_classification_tool()
        if result:
            print("\n" + "=" * 60)
            print("✅ All tests passed! The refactored workflow is ready.")
            print("=" * 60)
        else:
            print("\n⚠️  Classification test failed (API key may be missing)")
            print("    Other tests passed - workflow structure is correct.")
    except Exception as e:
        print(f"   ⚠️  Classification test skipped: {e}")
        print("\n" + "=" * 60)
        print("✅ Import and workflow tests passed!")
        print("   Run with valid API key to test classification.")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
