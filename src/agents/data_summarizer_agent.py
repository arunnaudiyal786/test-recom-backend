"""
Data Summarizer Agent for the Data Preparation Pipeline.

Uses LLM to generate AI summaries for each ticket, highlighting key aspects
like core problem, technical components, and business impact.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from src.models.data_prep_state import DataPrepState, DataPrepAgentOutput
from src.utils.openai_client import get_openai_client
from config import Config
from src.prompts.summarization_prompts import get_ticket_summarization_prompt


class DataSummarizerAgent:
    """
    Agent responsible for generating AI summaries of ticket data.

    Responsibilities:
    - Generate structured summaries for each ticket using LLM
    - Extract key aspects (core problem, technical components, impact)
    - Combine all outputs into final CSV
    - Handle rate limiting and batch processing
    """

    def __init__(self):
        """Initialize the Data Summarizer Agent."""
        self.client = get_openai_client()
        self.model = Config.CLASSIFICATION_MODEL  # Use cost-efficient model
        self.temperature = 0.3  # Slightly higher for creative summarization
        self.max_tokens = 1000  # Enough for detailed JSON summary

        # Batch processing settings
        self.batch_size = 5  # Process 5 tickets concurrently
        self.delay_between_batches = 1.0  # Seconds to wait between batches

    async def __call__(self, state: DataPrepState) -> DataPrepAgentOutput:
        """
        Main entry point for LangGraph workflow.

        Args:
            state: Current workflow state containing preprocessed_data

        Returns:
            Partial state update with summaries and final CSV
        """
        try:
            start_time = time.time()
            print(f"[Data Summarizer Agent] Starting summarization...")

            # Get preprocessed data from previous agent
            preprocessed_data = state.get("preprocessed_data", [])
            if not preprocessed_data:
                raise ValueError(
                    "preprocessed_data not found in state. "
                    "Data Preprocessor must run first."
                )

            total_rows = len(preprocessed_data)
            print(f"[Data Summarizer Agent] Summarizing {total_rows} tickets...")

            # Generate summaries for all tickets
            summaries, stats = await self._generate_all_summaries(preprocessed_data)

            print(
                f"[Data Summarizer Agent] Summarization complete. "
                f"Successfully summarized {stats['total_summarized']}/{total_rows} tickets."
            )

            # Combine preprocessed data with summaries to create final output
            final_data = self._create_final_dataset(preprocessed_data, summaries)

            # Save to CSV
            output_file_path = state.get("output_file_path", "")
            if not output_file_path:
                # Default output path
                output_file_path = str(
                    Config.PROJECT_ROOT / "data" / "processed" / "prepared_tickets.csv"
                )

            await self._save_to_csv(final_data, output_file_path)
            print(f"[Data Summarizer Agent] Saved final CSV to: {output_file_path}")

            # Calculate processing time
            end_time = time.time()
            total_time = end_time - start_time

            return {
                "status": "success",
                "current_agent": "Data Summarizer Agent",
                "processing_stage": "summarization",
                "summaries": summaries,
                "summarization_stats": stats,
                "final_data": final_data,
                "output_file_path": output_file_path,
                "processing_end_time": end_time,
                "total_processing_time_seconds": total_time,
                "messages": [{
                    "role": "assistant",
                    "content": (
                        f"Data Summarizer Agent completed. "
                        f"Generated {stats['total_summarized']} AI summaries. "
                        f"Failed summaries: {stats['failed_summaries']}. "
                        f"Average summary length: {stats['avg_summary_length']:.0f} chars. "
                        f"Final CSV saved to: {output_file_path}. "
                        f"Total processing time: {total_time:.2f} seconds."
                    )
                }]
            }

        except Exception as e:
            print(f"[Data Summarizer Agent] ERROR: {str(e)}")
            return {
                "status": "error",
                "current_agent": "Data Summarizer Agent",
                "processing_stage": "summarization",
                "error_message": f"Summarization failed: {str(e)}",
                "messages": [{
                    "role": "assistant",
                    "content": f"Data Summarizer Agent failed with error: {str(e)}"
                }]
            }

    async def _generate_all_summaries(
        self,
        preprocessed_data: List[Dict[str, Any]]
    ) -> tuple[List[str], Dict[str, Any]]:
        """
        Generate AI summaries for all tickets.

        Uses batch processing with rate limiting to avoid API throttling.

        Args:
            preprocessed_data: List of preprocessed ticket dictionaries

        Returns:
            Tuple of (list of summary strings, statistics dict)
        """
        summaries = []
        failed_count = 0
        total_length = 0

        # Process in batches
        total_batches = (len(preprocessed_data) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(preprocessed_data))
            batch = preprocessed_data[start_idx:end_idx]

            print(
                f"[Data Summarizer Agent] Processing batch {batch_idx + 1}/{total_batches} "
                f"(tickets {start_idx + 1}-{end_idx})"
            )

            # Process batch concurrently
            tasks = [
                self._summarize_single_ticket(ticket, idx)
                for idx, ticket in enumerate(batch, start=start_idx)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    # Handle failed summarization
                    summaries.append(f"ERROR: Failed to generate summary - {str(result)}")
                    failed_count += 1
                else:
                    summaries.append(result)
                    total_length += len(result)

            # Rate limiting delay between batches
            if batch_idx < total_batches - 1:
                await asyncio.sleep(self.delay_between_batches)

        # Calculate statistics
        successful_count = len(summaries) - failed_count
        avg_length = total_length / successful_count if successful_count > 0 else 0

        stats = {
            "total_summarized": successful_count,
            "failed_summaries": failed_count,
            "avg_summary_length": avg_length,
            "total_tickets": len(preprocessed_data)
        }

        return summaries, stats

    async def _summarize_single_ticket(
        self,
        ticket: Dict[str, Any],
        ticket_idx: int
    ) -> str:
        """
        Generate AI summary for a single ticket.

        Args:
            ticket: Preprocessed ticket dictionary
            ticket_idx: Index for logging

        Returns:
            Combined summary string
        """
        try:
            # Extract relevant fields (use cleaned versions if available)
            summary_text = ticket.get("Summary_cleaned", ticket.get("Summary", "N/A"))
            description_text = ticket.get(
                "Description_cleaned",
                ticket.get("Description", "N/A")
            )
            priority = ticket.get("Issue Priority", "")
            issue_type = ticket.get("issue type", "")
            labels = ticket.get("Labels", "")
            resolution = ticket.get("Resolution_cleaned", ticket.get("Resolution", ""))

            # Generate prompt
            prompt = get_ticket_summarization_prompt(
                summary=summary_text,
                description=description_text,
                priority=priority,
                issue_type=issue_type,
                labels=labels,
                resolution=resolution
            )

            # Call OpenAI with JSON response
            messages = [{"role": "user", "content": prompt}]

            response_json = await self.client.chat_completion_json(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract the combined summary
            combined_summary = response_json.get("combined_summary", "")

            # If combined_summary is missing, construct from parts
            if not combined_summary:
                parts = []
                if response_json.get("core_problem"):
                    parts.append(f"Issue: {response_json['core_problem']}")
                if response_json.get("technical_components"):
                    parts.append(
                        f"Components: {', '.join(response_json['technical_components'])}"
                    )
                if response_json.get("business_impact"):
                    parts.append(f"Impact: {response_json['business_impact']}")
                if response_json.get("severity_assessment"):
                    parts.append(f"Severity: {response_json['severity_assessment']}")
                if response_json.get("resolution_summary"):
                    parts.append(f"Resolution: {response_json['resolution_summary']}")

                combined_summary = " | ".join(parts)

            # Store the full JSON response as well for potential downstream use
            # This allows access to structured fields like key_keywords, etc.
            # We'll just return the combined summary for the CSV column
            return combined_summary

        except Exception as e:
            # Log error but don't fail the entire batch
            print(f"[Data Summarizer Agent] Warning: Failed to summarize ticket {ticket_idx}: {str(e)}")
            raise e

    def _create_final_dataset(
        self,
        preprocessed_data: List[Dict[str, Any]],
        summaries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Combine preprocessed data with AI summaries.

        Args:
            preprocessed_data: List of preprocessed ticket dictionaries
            summaries: List of AI-generated summaries

        Returns:
            Final dataset with all columns including AI_Summary
        """
        final_data = []

        for idx, ticket in enumerate(preprocessed_data):
            # Create a copy of the preprocessed ticket
            final_ticket = ticket.copy()

            # Add the AI summary as the last column
            if idx < len(summaries):
                final_ticket["AI_Summary"] = summaries[idx]
            else:
                final_ticket["AI_Summary"] = "ERROR: Summary not generated"

            # Remove internal processing fields (clean up)
            if "_preprocessing_row_index" in final_ticket:
                del final_ticket["_preprocessing_row_index"]

            final_data.append(final_ticket)

        return final_data

    async def _save_to_csv(
        self,
        final_data: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """
        Save final dataset to CSV file.

        Args:
            final_data: List of dictionaries to save
            output_path: Path to save CSV file
        """
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame and save
        df = pd.DataFrame(final_data)

        # Reorder columns to put AI_Summary last
        cols = [c for c in df.columns if c != "AI_Summary"]
        if "AI_Summary" in df.columns:
            cols.append("AI_Summary")
        df = df[cols]

        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')


# Create singleton instance for LangGraph workflow
data_summarizer_agent = DataSummarizerAgent()
