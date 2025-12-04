"""
Data ingestion pipeline for historical tickets into FAISS.

FULLY DYNAMIC: All column mappings are read from config/schema_config.yaml.
No hardcoded column names - change the YAML to support any CSV schema.
"""
import csv
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

from src.vectorstore.embedding_generator import get_embedding_generator
from src.vectorstore.faiss_manager import get_faiss_manager
from src.utils.helpers import combine_ticket_text, combine_text_for_embedding
from config.config import Config
from src.utils.schema_config import get_schema_config


class DataIngestionPipeline:
    """
    Pipeline to ingest historical tickets into FAISS index.

    All column mappings are read from schema_config.yaml - no hardcoded
    column names in this class. To support a new CSV format, update the
    YAML configuration file only.
    """

    def __init__(self):
        self.embedding_generator = get_embedding_generator()
        self.faiss_manager = get_faiss_manager()
        self.schema_config = get_schema_config()

    def _get_csv_value(self, row: Dict[str, str], internal_field: str) -> str:
        """
        Get value from CSV row using the configured column mapping.

        Args:
            row: CSV row as dict
            internal_field: Internal field name (e.g., 'ticket_id', 'description')

        Returns:
            Value from CSV or empty string if not found/mapped
        """
        csv_column = self.schema_config.column_mappings.get(internal_field)
        if csv_column is None:
            return ''
        return row.get(csv_column, '')

    def _extract_domain(self, row: Dict[str, str], ticket_id: str) -> str:
        """
        Extract domain using the configured extraction mode.

        Supports two modes (configured in schema_config.yaml):
        - 'from_column': Extract from a specific column using domain_map
        - 'from_ticket_key': Extract from ticket_id using regex pattern

        Args:
            row: CSV row as dict
            ticket_id: The ticket ID value

        Returns:
            Domain name or 'Unknown'
        """
        mode = self.schema_config.domain_extraction_mode

        if mode == 'from_ticket_key':
            # Extract domain code from ticket key using configured regex
            pattern = self.schema_config.ticket_key_domain_pattern
            domain_map = self.schema_config.ticket_key_domain_map
            match = re.match(pattern, ticket_id)
            if match and match.group(1):
                domain_code = match.group(1)
                return domain_map.get(domain_code, 'Unknown')
            return 'Unknown'
        else:
            # Extract from configured column using domain_map
            domain_col = self.schema_config.domain_extraction_column
            domain_value = row.get(domain_col, '')
            return self.schema_config.extract_domain_from_value(domain_value)

    def _parse_resolution_steps(self, resolution_text: str) -> List[str]:
        """
        Parse resolution steps using configured split pattern.

        Args:
            resolution_text: Raw resolution text from CSV

        Returns:
            List of resolution step strings
        """
        if not resolution_text:
            return []

        pattern = self.schema_config.resolution_split_pattern
        steps = re.split(pattern, resolution_text)
        return [s.strip() for s in steps if s.strip()]

    def _parse_labels(self, row: Dict[str, str]) -> List[str]:
        """
        Parse labels from configured label columns.

        Supports multiple label columns and excludes configured columns.

        Args:
            row: CSV row as dict

        Returns:
            List of label strings
        """
        label_columns = self.schema_config.label_columns or []
        excluded_columns = self.schema_config.excluded_label_columns or []
        delimiter = self.schema_config.labels_delimiter or ','

        labels = []
        for col in label_columns:
            if col in excluded_columns:
                continue
            value = row.get(col, '')
            if value:
                # Split by delimiter if present, otherwise add as single label
                if delimiter in value:
                    labels.extend([l.strip() for l in value.split(delimiter) if l.strip()])
                else:
                    labels.append(value.strip())

        return labels

    def _derive_priority(self, row: Dict[str, str]) -> str:
        """
        Get priority from column or derive from story points.

        Args:
            row: CSV row as dict

        Returns:
            Priority string (Low, Medium, High, Critical)
        """
        # First try direct priority column
        priority_value = self._get_csv_value(row, 'priority')
        if priority_value:
            return priority_value

        # Fall back to deriving from story points
        story_points = self._get_csv_value(row, 'story_points')
        return self.schema_config.derive_priority_from_story_points(story_points)

    def _calculate_resolution_time(self, created: str, closed: str) -> float:
        """
        Calculate resolution time in hours from created and closed dates.

        Uses date format from schema_config.yaml.

        Args:
            created: Created datetime string
            closed: Closed datetime string

        Returns:
            Resolution time in hours, or 0.0 if dates invalid
        """
        if not created or not closed:
            return 0.0

        date_format = self.schema_config.date_format
        try:
            created_dt = datetime.strptime(created, date_format)
            closed_dt = datetime.strptime(closed, date_format)
            delta = closed_dt - created_dt
            return round(delta.total_seconds() / 3600, 2)
        except (ValueError, TypeError):
            return 0.0

    def _parse_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse a CSV row into a normalized ticket dict.

        ALL column mappings come from schema_config.yaml - no hardcoded
        column names. This single method handles any CSV schema.

        Args:
            row: CSV row as dict

        Returns:
            Normalized ticket dict with standard internal field names
        """
        # Core fields
        ticket_id = self._get_csv_value(row, 'ticket_id')
        title = self._get_csv_value(row, 'title')
        description = self._get_csv_value(row, 'description')

        # Domain extraction (mode-aware)
        domain = self._extract_domain(row, ticket_id)

        # Labels (from multiple columns if configured)
        labels = self._parse_labels(row)

        # Resolution steps (parsed using configured pattern)
        resolution_text = self._get_csv_value(row, 'resolution_steps')
        resolution_steps = self._parse_resolution_steps(resolution_text)

        # Priority (from column or derived from story points)
        priority = self._derive_priority(row)

        # Date fields
        created_date = self._get_csv_value(row, 'created_date')
        closed_date = self._get_csv_value(row, 'closed_date')

        # Calculate resolution time if both dates present
        resolution_time_hours = self._calculate_resolution_time(created_date, closed_date)

        # Extract just the date portion if datetime string
        if created_date and ' ' in created_date:
            created_date = created_date.split(' ')[0]

        return {
            # Core fields
            'ticket_id': ticket_id,
            'title': title,
            'description': description,
            'domain': domain,
            'labels': labels,
            'resolution_steps': resolution_steps,
            'priority': priority,
            # Date/time fields
            'created_date': created_date,
            'resolution_time_hours': resolution_time_hours,
            # Metadata fields
            'issue_type': self._get_csv_value(row, 'issue_type'),
            'assignee': self._get_csv_value(row, 'assignee'),
            'it_team': self._get_csv_value(row, 'team'),
            'reporter': self._get_csv_value(row, 'reporter'),
            # Extended fields (may be empty depending on CSV schema)
            'precondition': self._get_csv_value(row, 'precondition'),
            'test_steps': self._get_csv_value(row, 'test_steps'),
            'expected_result': self._get_csv_value(row, 'expected_result'),
            'story_points': self._get_csv_value(row, 'story_points'),
            'status': self._get_csv_value(row, 'status'),
            'category_labels': self._get_csv_value(row, 'category_labels'),
        }

    def _validate_csv_headers(self, headers: List[str]) -> List[str]:
        """
        Validate CSV headers against configured column mappings.

        Args:
            headers: List of CSV column headers

        Returns:
            List of warning messages for missing required columns
        """
        warnings = []
        col_map = self.schema_config.column_mappings

        # Check required fields
        required_fields = ['ticket_id', 'title', 'description']
        for field in required_fields:
            csv_col = col_map.get(field)
            if csv_col and csv_col not in headers:
                warnings.append(f"Required column '{csv_col}' (for {field}) not found in CSV")

        return warnings

    async def load_historical_tickets(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load historical tickets from CSV file.

        Uses schema_config.yaml for ALL column mappings - no hardcoded
        column names. Any CSV format can be supported by updating the
        YAML configuration.

        Args:
            file_path: Path to CSV file

        Returns:
            List of ticket dicts with normalized structure
        """
        print(f"📂 Loading historical tickets from: {file_path}")
        print(f"   Using column mappings from schema_config.yaml")

        tickets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            print(f"   CSV columns found: {len(headers)}")

            # Validate headers against config
            warnings = self._validate_csv_headers(headers)
            for warning in warnings:
                print(f"⚠️  {warning}")

            if warnings:
                print(f"   Available headers: {headers}")
                print(f"   Check column_mappings in config/schema_config.yaml")

            # Parse each row using unified parser
            row_count = 0
            error_count = 0
            for row in reader:
                row_count += 1
                try:
                    ticket = self._parse_row(row)
                    # Skip rows without ticket_id
                    if ticket['ticket_id']:
                        tickets.append(ticket)
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:  # Only show first 3 errors
                        print(f"⚠️  Error parsing row {row_count}: {e}")

            if error_count > 3:
                print(f"⚠️  ... and {error_count - 3} more parsing errors")

        print(f"✅ Loaded {len(tickets)} historical tickets (from {row_count} rows)")
        return tickets

    async def generate_embeddings_for_tickets(
        self, tickets: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """
        Generate embeddings for all tickets using configured columns.

        Uses vectorization settings from schema_config.yaml to determine
        which columns to include in the embedding text.

        Args:
            tickets: List of ticket dicts

        Returns:
            List of embedding vectors
        """
        # Get vectorization settings from config
        columns = self.schema_config.vectorization_columns
        separator = self.schema_config.vectorization_separator
        should_clean = self.schema_config.vectorization_clean_text
        max_tokens = self.schema_config.vectorization_max_tokens

        print(f"\n🔄 Generating embeddings for {len(tickets)} tickets...")
        print(f"   Using columns: {columns}")

        # Combine configured columns for each ticket
        texts = [
            combine_text_for_embedding(
                ticket=ticket,
                columns=columns,
                separator=separator,
                should_clean=should_clean,
                max_tokens=max_tokens
            )
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

        Stores all ticket fields for retrieval and display.
        Fields are populated based on what's available from the CSV.

        Args:
            tickets: List of ticket dicts

        Returns:
            List of metadata dicts
        """
        metadata = []

        for ticket in tickets:
            meta = {
                # Core fields (always present)
                'ticket_id': ticket['ticket_id'],
                'title': ticket['title'],
                'description': ticket['description'],
                'domain': ticket['domain'],
                'labels': ticket['labels'],
                'resolution_steps': ticket['resolution_steps'],
                'priority': ticket['priority'],
                # Date/time fields
                'created_date': ticket.get('created_date', ''),
                'resolution_time_hours': ticket.get('resolution_time_hours', 0.0),
                # Metadata fields
                'issue_type': ticket.get('issue_type', ''),
                'assignee': ticket.get('assignee', ''),
                'it_team': ticket.get('it_team', ''),
                'reporter': ticket.get('reporter', ''),
                # Extended fields (populated based on CSV schema)
                'precondition': ticket.get('precondition', ''),
                'test_steps': ticket.get('test_steps', ''),
                'expected_result': ticket.get('expected_result', ''),
                'story_points': ticket.get('story_points', ''),
                'status': ticket.get('status', ''),
                'category_labels': ticket.get('category_labels', ''),
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
