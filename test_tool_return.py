"""Quick test to check what LangChain tools return."""
import asyncio
from components.labeling.tools import generate_business_labels

async def main():
    result = await generate_business_labels.ainvoke({
        "title": "Test ticket",
        "description": "This is a test ticket for database connection pool exhaustion.",
        "domain": "MM",
        "priority": "High",
        "existing_labels": [],
        "max_labels": 3,
        "confidence_threshold": 0.7
    })

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
    print(f"actual_prompt present: {'actual_prompt' in result if isinstance(result, dict) else 'N/A'}")
    print(f"actual_prompt length: {len(result.get('actual_prompt', '')) if isinstance(result, dict) else 'N/A'}")

if __name__ == "__main__":
    asyncio.run(main())
