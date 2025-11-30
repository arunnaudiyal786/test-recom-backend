"""
Data ingestion pipeline for historical tickets into FAISS.
"""
import csv
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm

from src.vectorstore.embedding_generator import get_embedding_generator
from src.vectorstore.faiss_manager import get_faiss_manager
from src.utils.helpers import combine_ticket_text
from src.utils.config import Config


class DataIngestionPipeline:
    """Pipeline to ingest historical tickets into FAISS index."""

    def __init__(self):
        self.embedding_generator = get_embedding_generator()
        self.faiss_manager = get_faiss_manager()

    def _extract_domain_from_key(self, key: str) -> str:
        """
        Extract domain from ticket key (e.g., JIRA-MM-001 -> MM).

        Args:
            key: Ticket key in format JIRA-XX-NNN

        Returns:
            Domain name (MM, CI for CIW, or SP for Specialty)
        """
        parts = key.split('-')
        if len(parts) >= 2:
            domain_code = parts[1]
            # Map 2-letter codes to full domain names
            domain_map = {
                'MM': 'MM',
                'CI': 'CIW',
                'SP': 'Specialty'
            }
            return domain_map.get(domain_code, 'Unknown')
        return 'Unknown'

    def _calculate_resolution_time(self, created: str, closed: str) -> float:
        """
        Calculate resolution time in hours from created and closed dates.

        Args:
            created: Created datetime string
            closed: Closed datetime string

        Returns:
            Resolution time in hours
        """
        try:
            created_dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
            closed_dt = datetime.strptime(closed, "%Y-%m-%d %H:%M:%S")
            delta = closed_dt - created_dt
            return round(delta.total_seconds() / 3600, 2)
        except Exception:
            return 0.0

    async def load_historical_tickets(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load historical tickets from CSV file.

        Args:
            file_path: Path to historical_tickets.csv

        Returns:
            List of ticket dicts with normalized structure
        """
        print(f"📂 Loading historical tickets from: {file_path}")

        tickets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert CSV row to normalized ticket structure
                ticket = {
                    'ticket_id': row['key'],
                    'title': row['Summary'],
                    'description': row['Description'],
                    'domain': self._extract_domain_from_key(row['key']),
                    'labels': row['Labels'].split(',') if row['Labels'] else [],
                    'resolution_steps': row['Resolution'].split('\n') if row['Resolution'] else [],
                    'priority': row['Issue Priority'],
                    'created_date': row['created'].split(' ')[0],  # Extract date only
                    'resolution_time_hours': self._calculate_resolution_time(
                        row['created'], row['closed date']
                    ),
                    # Additional CSV-specific fields (stored as metadata)
                    'issue_type': row['issue type'],
                    'assignee': row['assignee'],
                    'it_team': row['IT Team'],
                    'reporter': row['Reporter'],
                }
                tickets.append(ticket)

        print(f"✅ Loaded {len(tickets)} historical tickets")
        return tickets

    async def generate_embeddings_for_tickets(
        self, tickets: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """
        Generate embeddings for all tickets.

        Args:
            tickets: List of ticket dicts with 'title' and 'description'

        Returns:
            List of embedding vectors
        """
        print(f"\n🔄 Generating embeddings for {len(tickets)} tickets...")

        # Combine title and description for each ticket
        texts = [
            combine_ticket_text(ticket['title'], ticket['description'])
            for ticket in tickets
        ]

        # Generate embeddings in batches with progress
        embeddings = await self.embedding_generator.generate_batch_embeddings(
            texts, batch_size=10, show_progress=True
        )

        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings

    def prepare_metadata(self, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare metadata for FAISS index.

        Metadata includes all ticket fields needed for retrieval.

        Args:
            tickets: List of ticket dicts

        Returns:
            List of metadata dicts
        """
        metadata = []

        for ticket in tickets:
            meta = {
                'ticket_id': ticket['ticket_id'],
                'title': ticket['title'],
                'description': ticket['description'],
                'domain': ticket['domain'],
                'labels': ticket['labels'],
                'resolution_steps': ticket['resolution_steps'],
                'priority': ticket['priority'],
                'created_date': ticket['created_date'],
                'resolution_time_hours': ticket['resolution_time_hours'],
                # Additional CSV-specific fields
                'issue_type': ticket.get('issue_type', ''),
                'assignee': ticket.get('assignee', ''),
                'it_team': ticket.get('it_team', ''),
                'reporter': ticket.get('reporter', ''),
            }
            metadata.append(meta)

        return metadata

    async def build_index(self, tickets: List[Dict[str, Any]]):
        """
        Build FAISS index from tickets.

        Args:
            tickets: List of ticket dicts
        """
        print("\n🔨 Building FAISS index...")

        # Generate embeddings
        embeddings = await self.generate_embeddings_for_tickets(tickets)

        # Prepare metadata
        metadata = self.prepare_metadata(tickets)

        # Create index
        self.faiss_manager.create_index(embeddings, metadata)

        # Print statistics
        stats = self.faiss_manager.get_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"  Total vectors: {stats['total_vectors']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Domain distribution:")
        for domain, count in stats['domain_distribution'].items():
            print(f"    - {domain}: {count}")

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        print("\n💾 Saving FAISS index...")
        self.faiss_manager.save()

    async def run_pipeline(self, input_file: Path):
        """
        Run the complete data ingestion pipeline.

        Args:
            input_file: Path to historical_tickets.json
        """
        print("=" * 60)
        print("🚀 Starting Data Ingestion Pipeline")
        print("=" * 60)

        # Load tickets
        tickets = await self.load_historical_tickets(input_file)

        # Build index
        await self.build_index(tickets)

        # Save to disk
        self.save_index()

        print("\n" + "=" * 60)
        print("✅ Data Ingestion Pipeline Complete!")
        print("=" * 60)


async def main():
    """Main entry point for data ingestion."""
    # Input file path (now CSV instead of JSON)
    input_file = Config.PROJECT_ROOT / "data" / "raw" / "historical_tickets.csv"

    if not input_file.exists():
        print(f"❌ Error: Historical tickets file not found at {input_file}")
        print("   Run scripts/generate_sample_csv_data.py first")
        return

    # Run pipeline
    pipeline = DataIngestionPipeline()
    await pipeline.run_pipeline(input_file)


if __name__ == "__main__":
    asyncio.run(main())
