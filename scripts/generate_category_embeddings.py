"""
Generate pre-computed embeddings for all categories.

This script creates embeddings for each category by combining its
description and keywords, then saves them to a JSON file for fast
semantic pre-filtering during ticket classification.

Usage:
    cd test-recom-backend
    source .venv/bin/activate
    python scripts/generate_category_embeddings.py

Output:
    - data/metadata/category_embeddings.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from config import Config


def generate_category_embeddings():
    """Generate and save category embeddings to JSON file."""

    print("\n" + "=" * 60)
    print("Category Embeddings Generator")
    print("=" * 60)

    # Load categories
    categories_path = Config.CATEGORIES_JSON_PATH
    print(f"\n1. Loading categories from: {categories_path}")

    if not categories_path.exists():
        print(f"   ERROR: Categories file not found!")
        return False

    with open(categories_path, "r", encoding="utf-8") as f:
        categories_data = json.load(f)

    categories = categories_data.get("categories", [])
    print(f"   Found {len(categories)} categories")

    # Initialize OpenAI client
    print(f"\n2. Initializing OpenAI client...")
    print(f"   Model: {Config.EMBEDDING_MODEL}")
    print(f"   Dimension: {Config.EMBEDDING_DIMENSIONS}")

    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Generate embeddings for each category
    print(f"\n3. Generating embeddings...")
    embeddings = {}
    failed = []

    for i, category in enumerate(categories):
        cat_id = category["id"]
        cat_name = category["name"]

        # Combine description + keywords for rich embedding
        keywords_text = " ".join(category.get("keywords", []))
        examples_text = " ".join(category.get("examples", []))

        # Create embedding text: description + keywords + examples
        embedding_text = f"{category['description']} {keywords_text} {examples_text}"

        try:
            response = client.embeddings.create(
                model=Config.EMBEDDING_MODEL,
                input=embedding_text
            )

            embeddings[cat_id] = response.data[0].embedding
            print(f"   [{i+1:2d}/{len(categories)}] {cat_id}")

        except Exception as e:
            print(f"   [{i+1:2d}/{len(categories)}] {cat_id} - FAILED: {e}")
            failed.append(cat_id)

    # Create output data structure
    output_data = {
        "metadata": {
            "model": Config.EMBEDDING_MODEL,
            "dimension": Config.EMBEDDING_DIMENSIONS,
            "category_count": len(embeddings),
            "category_ids": list(embeddings.keys()),
            "generated_at": datetime.now().isoformat(),
            "source_file": str(categories_path)
        },
        "embeddings": embeddings
    }

    # Save to JSON file
    output_path = Config.CATEGORY_EMBEDDINGS_PATH
    print(f"\n4. Saving embeddings to: {output_path}")

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f)

    # Calculate file size
    file_size = output_path.stat().st_size / 1024  # KB

    print(f"\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"   Categories processed: {len(categories)}")
    print(f"   Embeddings generated: {len(embeddings)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Output file: {output_path}")
    print(f"   File size: {file_size:.1f} KB")
    print(f"   Embedding dimension: {Config.EMBEDDING_DIMENSIONS}")

    if failed:
        print(f"\n   Failed categories: {failed}")
        return False

    print(f"\n   Done!")
    return True


if __name__ == "__main__":
    success = generate_category_embeddings()
    sys.exit(0 if success else 1)
