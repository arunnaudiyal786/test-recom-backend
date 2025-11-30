"""
FAISS index manager for similarity search.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.utils.config import Config


class FAISSManager:
    """Manage FAISS index for ticket similarity search."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = Config.EMBEDDING_DIMENSIONS

    def create_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """
        Create a new FAISS index from embeddings.

        Uses IndexFlatIP for cosine similarity search (inner product after normalization).

        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dicts (one per embedding)
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) must have same length"
            )

        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity using inner product
        faiss.normalize_L2(embeddings_array)

        # Create index - IndexFlatIP for exact inner product search
        self.index = faiss.IndexFlatIP(self.dimension)

        # Add vectors to index
        self.index.add(embeddings_array)

        # Store metadata
        self.metadata = metadata

        print(f"✅ Created FAISS index with {self.index.ntotal} vectors")

    def save(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        """
        Save FAISS index and metadata to disk.

        Args:
            index_path: Path to save index file (default: from config)
            metadata_path: Path to save metadata JSON (default: from config)
        """
        if self.index is None:
            raise ValueError("No index to save. Create index first.")

        index_path = index_path or Config.FAISS_INDEX_PATH
        metadata_path = metadata_path or Config.FAISS_METADATA_PATH

        # Create directories if they don't exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self.index, str(index_path))
        print(f"💾 Saved FAISS index to: {index_path}")

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Saved metadata to: {metadata_path}")

    def load(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to index file (default: from config)
            metadata_path: Path to metadata JSON (default: from config)
        """
        index_path = index_path or Config.FAISS_INDEX_PATH
        metadata_path = metadata_path or Config.FAISS_METADATA_PATH

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load index
        self.index = faiss.read_index(str(index_path))
        print(f"📂 Loaded FAISS index from: {index_path} ({self.index.ntotal} vectors)")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"📂 Loaded metadata: {len(self.metadata)} entries")

    def search(
        self,
        query_embedding: List[float],
        k: int = 20,
        domain_filter: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar tickets using FAISS.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            domain_filter: Optional domain to filter by ('MM', 'CIW', 'Specialty')

        Returns:
            Tuple of (similar_tickets, similarity_scores)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() first.")

        # Normalize query vector
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Search FAISS index
        # Request more results if filtering by domain
        search_k = k * 3 if domain_filter else k
        distances, indices = self.index.search(query_array, search_k)

        # Get results
        results = []
        scores = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for padding
                continue

            ticket_data = self.metadata[idx].copy()

            # Apply domain filter if specified
            if domain_filter and ticket_data.get('domain') != domain_filter:
                continue

            ticket_data['similarity_score'] = float(distance)
            results.append(ticket_data)
            scores.append(float(distance))

            # Stop when we have enough results
            if len(results) >= k:
                break

        return results, scores

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {"status": "No index loaded"}

        domain_counts = {}
        for ticket in self.metadata:
            domain = ticket.get('domain', 'Unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "domain_distribution": domain_counts,
            "total_metadata_entries": len(self.metadata)
        }


# Global instance
_faiss_manager: Optional[FAISSManager] = None


def get_faiss_manager() -> FAISSManager:
    """Get or create the singleton FAISS manager instance."""
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSManager()
    return _faiss_manager
