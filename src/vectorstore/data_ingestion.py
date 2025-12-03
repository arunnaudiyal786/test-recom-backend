"""
Data ingestion pipeline for historical tickets into FAISS.

Uses schema_config.yaml for dynamic column mappings and domain definitions.
"""
import csv
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm

from src.vectorstore.embedding_generator import get_embedding_generator
from src.vectorstore.faiss_manager import get_faiss_manager
from src.utils.helpers import combine_ticket_text
from config import Config
from src.utils.schema_config import get_schema_config


class DataIngestionPipeline:
    """Pipeline to ingest historical tickets into FAISS index."""

    def __init__(self):
        self.embedding_generator = get_embedding_generator()
        self.faiss_manager = get_faiss_manager()
        self.schema_config = get_schema_config()

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

    def _extract_domain_from_column(self, value: str) -> str:
        """
        Extract domain from configured column value using schema config.

        Args:
            value: Value from the domain extraction column

        Returns:
            Domain name from config mapping
        """
        return self.schema_config.extract_domain_from_value(value)

    def _story_points_to_priority(self, story_points: str) -> str:
        """
        Convert story points to priority level using schema config.

        Args:
            story_points: Story points value

        Returns:
            Priority level (Low, Medium, High, Critical)
        """
        return self.schema_config.derive_priority_from_story_points(story_points)

    def _detect_csv_schema(self, headers: List[str]) -> str:
        """
        Detect which CSV schema is being used.

        Args:
            headers: List of column headers

        Returns:
            Schema type: 'historical_tickets' or 'test_plan'
        """
        if 'Issue_Key' in headers and 'Test_Step_Description' in headers:
            return 'test_plan'
        elif 'key' in headers and 'Resolution' in headers:
            return 'historical_tickets'
        else:
            return 'unknown'

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

    def _parse_historical_tickets_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Parse a row from historical_tickets.csv schema."""
        return {
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
            'issue_type': row['issue type'],
            'assignee': row['assignee'],
            'it_team': row['IT Team'],
            'reporter': row['Reporter'],
        }

    def _parse_test_plan_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Parse a row from test_plan_historical.csv schema using config."""
        # Get label columns from schema config (excludes story ID column)
        label_columns = self.schema_config.label_columns
        excluded_columns = self.schema_config.excluded_label_columns

        labels = []
        for col in label_columns:
            if col not in excluded_columns and row.get(col):
                labels.append(row[col])

        # Get column mappings from config
        col_map = self.schema_config.column_mappings

        # Parse resolution steps from configured column
        resolution_col = col_map.get('resolution_steps', 'Expected_Result')
        resolution_text = row.get(resolution_col, '') if resolution_col else ''
        resolution_steps = []
        if resolution_text:
            # Split by numbered pattern like "1. " or just newlines
            steps = re.split(r'\d+\.\s*', resolution_text)
            resolution_steps = [s.strip() for s in steps if s.strip()]

        # Extract domain using configured column
        domain_col = self.schema_config.domain_extraction_column
        domain_value = row.get(domain_col, '')
        domain = self._extract_domain_from_column(domain_value)

        # Get priority - either from column or derived from story points
        priority_col = col_map.get('priority')
        if priority_col and row.get(priority_col):
            priority = row[priority_col]
        else:
            story_points_col = col_map.get('story_points', 'Story_Points')
            priority = self._story_points_to_priority(row.get(story_points_col, ''))

        return {
            'ticket_id': row.get(col_map.get('ticket_id', 'Issue_Key'), ''),
            'title': row.get(col_map.get('title', 'Summary'), ''),
            'description': row.get(col_map.get('description', 'Description'), ''),
            'domain': domain,
            'labels': labels,
            'resolution_steps': resolution_steps,
            'priority': priority,
            'created_date': row.get(col_map.get('created_date', ''), '') if col_map.get('created_date') else '',
            'resolution_time_hours': 0.0,  # Calculate if dates available
            'issue_type': row.get(col_map.get('issue_type', 'Type'), ''),
            'assignee': row.get(col_map.get('assignee', ''), ''),
            'it_team': row.get(col_map.get('team', ''), ''),
            'reporter': row.get(col_map.get('reporter', ''), '') if col_map.get('reporter') else '',
            # Test plan specific fields
            'precondition': row.get(col_map.get('precondition', 'Precondition'), ''),
            'test_steps': row.get(col_map.get('test_steps', 'Test_Step_Description'), ''),
            'expected_result': row.get(col_map.get('expected_result', 'Expected_Result'), ''),
            'story_points': row.get(col_map.get('story_points', 'Story_Points'), ''),
            'status': row.get(col_map.get('status', 'Status'), ''),
        }

    async def load_historical_tickets(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load historical tickets from CSV file.

        Supports two schemas:
        - historical_tickets.csv: Original JIRA ticket format
        - test_plan_historical.csv: Test plan format with different columns

        Args:
            file_path: Path to CSV file

        Returns:
            List of ticket dicts with normalized structure
        """
        print(f"📂 Loading historical tickets from: {file_path}")

        tickets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Detect schema type
            schema_type = self._detect_csv_schema(headers)
            print(f"   Detected schema: {schema_type}")

            if schema_type == 'unknown':
                print(f"⚠️  Warning: Unknown CSV schema. Expected columns not found.")
                print(f"   Headers found: {headers}")
                return []

            for row in reader:
                try:
                    if schema_type == 'historical_tickets':
                        ticket = self._parse_historical_tickets_row(row)
                    else:  # test_plan
                        ticket = self._parse_test_plan_row(row)
                    tickets.append(ticket)
                except KeyError as e:
                    print(f"⚠️  Skipping row due to missing column: {e}")
                    continue

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
                'created_date': ticket.get('created_date', ''),
                'resolution_time_hours': ticket.get('resolution_time_hours', 0.0),
                # Additional CSV-specific fields
                'issue_type': ticket.get('issue_type', ''),
                'assignee': ticket.get('assignee', ''),
                'it_team': ticket.get('it_team', ''),
                'reporter': ticket.get('reporter', ''),
                # Test plan specific fields (will be empty for historical_tickets schema)
                'precondition': ticket.get('precondition', ''),
                'test_steps': ticket.get('test_steps', ''),
                'expected_result': ticket.get('expected_result', ''),
                'story_points': ticket.get('story_points', ''),
                'status': ticket.get('status', ''),
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
    # Input file path from config (configurable via .env HISTORICAL_TICKETS_CSV)
    input_file = Config.HISTORICAL_TICKETS_PATH

    if not input_file.exists():
        print(f"❌ Error: Historical tickets file not found at {input_file}")
        print(f"   Check HISTORICAL_TICKETS_CSV in .env (current: {Config.HISTORICAL_TICKETS_CSV})")
        print("   Or run scripts/generate_sample_csv_data.py to create default data")
        return

    # Run pipeline
    pipeline = DataIngestionPipeline()
    await pipeline.run_pipeline(input_file)


if __name__ == "__main__":
    asyncio.run(main())
