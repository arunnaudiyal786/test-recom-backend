"""
Data Preprocessor Agent for the Data Preparation Pipeline.

Cleans and normalizes text data by removing HTML tags, special characters,
and normalizing text format while preserving original data.
"""
import re
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import html

from src.models.data_prep_state import DataPrepState, DataPrepAgentOutput


class DataPreprocessorAgent:
    """
    Agent responsible for cleaning and normalizing ticket text data.

    Responsibilities:
    - Remove HTML tags from text fields
    - Remove or normalize special characters
    - Normalize text (lowercase for processing, preserve original)
    - Add cleaned columns alongside original data
    """

    def __init__(self):
        """Initialize the Data Preprocessor Agent."""
        # Fields to clean
        self.text_fields_to_clean = ["Summary", "Description", "Resolution"]

        # Characters to preserve (beyond alphanumeric)
        self.preserved_punctuation = set(".,!?;:'\"()-_/\\@#$%&*+=<>[]{}|`~")

    async def __call__(self, state: DataPrepState) -> DataPrepAgentOutput:
        """
        Main entry point for LangGraph workflow.

        Args:
            state: Current workflow state containing raw_data from validator

        Returns:
            Partial state update with preprocessed data
        """
        try:
            print(f"[Data Preprocessor Agent] Starting preprocessing...")

            # Get raw data from previous agent
            raw_data = state.get("raw_data", [])
            if not raw_data:
                raise ValueError("raw_data not found in state. Data Validator must run first.")

            total_rows = len(raw_data)
            print(f"[Data Preprocessor Agent] Processing {total_rows} rows...")

            # Track preprocessing statistics
            stats = {
                "html_tags_removed": 0,
                "special_chars_cleaned": 0,
                "rows_modified": 0,
                "fields_normalized": []
            }

            # Process each row
            preprocessed_data = []
            for idx, row in enumerate(raw_data):
                processed_row, row_stats = await self._preprocess_row(row.copy(), idx)
                preprocessed_data.append(processed_row)

                # Accumulate statistics
                if row_stats["was_modified"]:
                    stats["rows_modified"] += 1
                stats["html_tags_removed"] += row_stats["html_removed"]
                stats["special_chars_cleaned"] += row_stats["special_chars_cleaned"]

                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"[Data Preprocessor Agent] Processed {idx + 1}/{total_rows} rows...")

            # Track which fields were normalized
            stats["fields_normalized"] = [
                f"{field}_cleaned" for field in self.text_fields_to_clean
            ]

            print(
                f"[Data Preprocessor Agent] Preprocessing complete. "
                f"Modified {stats['rows_modified']} rows. "
                f"Removed {stats['html_tags_removed']} HTML tags."
            )

            return {
                "status": "success",
                "current_agent": "Data Preprocessor Agent",
                "processing_stage": "preprocessing",
                "preprocessed_data": preprocessed_data,
                "normalization_report": stats,
                "messages": [{
                    "role": "assistant",
                    "content": (
                        f"Data Preprocessor Agent completed. "
                        f"Processed {total_rows} rows. "
                        f"Modified {stats['rows_modified']} rows. "
                        f"Removed {stats['html_tags_removed']} HTML tags and "
                        f"cleaned {stats['special_chars_cleaned']} special character sequences."
                    )
                }]
            }

        except Exception as e:
            print(f"[Data Preprocessor Agent] ERROR: {str(e)}")
            return {
                "status": "error",
                "current_agent": "Data Preprocessor Agent",
                "processing_stage": "preprocessing",
                "error_message": f"Data preprocessing failed: {str(e)}",
                "messages": [{
                    "role": "assistant",
                    "content": f"Data Preprocessor Agent failed with error: {str(e)}"
                }]
            }

    async def _preprocess_row(
        self,
        row: Dict[str, Any],
        row_idx: int
    ) -> tuple[Dict[str, Any], Dict[str, int]]:
        """
        Preprocess a single row of data.

        Args:
            row: Single row dictionary
            row_idx: Row index for tracking

        Returns:
            Tuple of (processed_row, statistics)
        """
        row_stats = {
            "was_modified": False,
            "html_removed": 0,
            "special_chars_cleaned": 0
        }

        # Process each text field
        for field in self.text_fields_to_clean:
            if field in row:
                original_value = row[field]

                # Skip if value is None or NaN
                if original_value is None or (
                    isinstance(original_value, float) and str(original_value) == 'nan'
                ):
                    row[f"{field}_cleaned"] = ""
                    row[f"{field}_normalized"] = ""
                    continue

                # Convert to string
                text = str(original_value)

                # Step 1: Remove HTML tags
                cleaned_text, html_count = self._remove_html_tags(text)
                if html_count > 0:
                    row_stats["html_removed"] += html_count
                    row_stats["was_modified"] = True

                # Step 2: Clean special characters
                cleaned_text, special_count = self._clean_special_characters(cleaned_text)
                if special_count > 0:
                    row_stats["special_chars_cleaned"] += special_count
                    row_stats["was_modified"] = True

                # Step 3: Normalize whitespace
                cleaned_text = self._normalize_whitespace(cleaned_text)

                # Store cleaned version (preserves case)
                row[f"{field}_cleaned"] = cleaned_text.strip()

                # Store normalized version (lowercase for processing)
                row[f"{field}_normalized"] = cleaned_text.lower().strip()

            else:
                # Field doesn't exist, add empty cleaned versions
                row[f"{field}_cleaned"] = ""
                row[f"{field}_normalized"] = ""

        # Add data quality score from validation (if available)
        # This helps downstream processes know which rows are more reliable
        row["_preprocessing_row_index"] = row_idx

        return row, row_stats

    def _remove_html_tags(self, text: str) -> tuple[str, int]:
        """
        Remove HTML tags from text using BeautifulSoup.

        Args:
            text: Input text potentially containing HTML

        Returns:
            Tuple of (cleaned_text, number_of_tags_removed)
        """
        if not text or not isinstance(text, str):
            return text, 0

        # Count HTML tags before cleaning
        html_tag_pattern = r'<[^>]+>'
        tags_found = len(re.findall(html_tag_pattern, text))

        if tags_found == 0:
            # No HTML tags found, return as-is
            return text, 0

        # Use BeautifulSoup to parse and extract text
        try:
            # Decode HTML entities first
            text = html.unescape(text)

            # Parse with BeautifulSoup
            soup = BeautifulSoup(text, 'html5lib')

            # Get text content
            cleaned = soup.get_text(separator=' ')

            return cleaned, tags_found

        except Exception:
            # If parsing fails, use regex fallback
            cleaned = re.sub(html_tag_pattern, ' ', text)
            return cleaned, tags_found

    def _clean_special_characters(self, text: str) -> tuple[str, int]:
        """
        Clean special characters while preserving meaningful punctuation.

        Args:
            text: Input text

        Returns:
            Tuple of (cleaned_text, number_of_special_chars_cleaned)
        """
        if not text or not isinstance(text, str):
            return text, 0

        original_length = len(text)
        cleaned_chars = []
        special_count = 0

        for char in text:
            if char.isalnum() or char.isspace():
                # Keep alphanumeric and whitespace
                cleaned_chars.append(char)
            elif char in self.preserved_punctuation:
                # Keep common punctuation
                cleaned_chars.append(char)
            elif ord(char) > 127:
                # Handle non-ASCII characters
                # Keep common accented characters, replace others
                if char in 'áéíóúÁÉÍÓÚñÑüÜ':
                    cleaned_chars.append(char)
                else:
                    # Replace with space to maintain word boundaries
                    cleaned_chars.append(' ')
                    special_count += 1
            else:
                # Replace control characters and other special chars with space
                cleaned_chars.append(' ')
                special_count += 1

        cleaned_text = ''.join(cleaned_chars)

        return cleaned_text, special_count

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.

        - Replace multiple spaces with single space
        - Replace tabs and newlines with spaces (for single-line representation)
        - Strip leading/trailing whitespace

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        if not text or not isinstance(text, str):
            return text

        # Replace newlines and tabs with spaces
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


# Create singleton instance for LangGraph workflow
data_preprocessor_agent = DataPreprocessorAgent()
