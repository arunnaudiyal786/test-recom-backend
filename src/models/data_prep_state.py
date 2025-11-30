"""
LangGraph state schema for Data Preparation Pipeline.

This state manages batch processing of CSV ticket data through
validation, preprocessing, and summarization agents.
"""
from typing import TypedDict, List, Dict, Optional, Literal, Annotated, Any
import operator


class DataPrepState(TypedDict, total=False):
    """
    State schema for data preparation workflow.

    Uses TypedDict with total=False to allow partial state updates.
    Processes entire CSV files rather than single tickets.
    """

    # ========== Input Configuration ==========
    input_file_path: str  # Path to input CSV file
    output_file_path: str  # Path where processed CSV will be saved

    # ========== Raw Data ==========
    # Stored as list of dicts for JSON serialization (DataFrame.to_dict('records'))
    raw_data: List[Dict[str, Any]]
    total_rows: int
    column_names: List[str]

    # ========== Validation Outputs ==========
    validation_report: Dict[str, Any]  # Overall quality metrics
    # Structure: {
    #   "overall_completeness_score": float,
    #   "total_rows": int,
    #   "valid_rows": int,
    #   "rows_with_issues": int,
    #   "column_null_percentages": Dict[str, float],
    #   "critical_fields_missing": List[str]
    # }

    per_row_validation: List[Dict[str, Any]]  # Per-row validation results
    # Each dict: {
    #   "row_index": int,
    #   "is_valid": bool,
    #   "missing_fields": List[str],
    #   "completeness_score": float,
    #   "warnings": List[str]
    # }

    # ========== Preprocessing Outputs ==========
    preprocessed_data: List[Dict[str, Any]]  # Cleaned data records
    normalization_report: Dict[str, Any]  # What was cleaned
    # Structure: {
    #   "html_tags_removed": int,
    #   "special_chars_cleaned": int,
    #   "rows_modified": int,
    #   "fields_normalized": List[str]
    # }

    # ========== Summarization Outputs ==========
    summaries: List[str]  # AI-generated summaries for each ticket
    summarization_stats: Dict[str, Any]  # Stats about summarization
    # Structure: {
    #   "total_summarized": int,
    #   "avg_summary_length": float,
    #   "failed_summaries": int
    # }

    # ========== Final Output ==========
    final_data: List[Dict[str, Any]]  # Complete processed data with all new columns

    # ========== Workflow Control ==========
    processing_stage: str  # Current stage: "validation", "preprocessing", "summarization"
    status: Literal["processing", "success", "error", "failed"]
    current_agent: str
    error_message: Optional[str]

    # ========== Accumulated Messages (uses reducer) ==========
    messages: Annotated[List[Dict[str, str]], operator.add]

    # ========== Processing Metrics ==========
    processing_start_time: Optional[float]  # Unix timestamp
    processing_end_time: Optional[float]
    total_processing_time_seconds: Optional[float]


class DataPrepAgentOutput(TypedDict, total=False):
    """
    Standard output format for data preparation agents.

    Each agent returns a partial state update.
    """

    # Required fields
    status: Literal["success", "error"]
    current_agent: str
    processing_stage: str

    # Error handling
    error_message: Optional[str]

    # Data Validator outputs
    validation_report: Optional[Dict[str, Any]]
    per_row_validation: Optional[List[Dict[str, Any]]]
    raw_data: Optional[List[Dict[str, Any]]]
    total_rows: Optional[int]
    column_names: Optional[List[str]]

    # Data Preprocessor outputs
    preprocessed_data: Optional[List[Dict[str, Any]]]
    normalization_report: Optional[Dict[str, Any]]

    # Data Summarizer outputs
    summaries: Optional[List[str]]
    summarization_stats: Optional[Dict[str, Any]]
    final_data: Optional[List[Dict[str, Any]]]
    output_file_path: Optional[str]

    # Processing metrics
    processing_end_time: Optional[float]
    total_processing_time_seconds: Optional[float]

    # Messages to add to audit trail
    messages: Optional[List[Dict[str, str]]]


# Type aliases for routing decisions in data prep workflow
DataPrepRoutingDecision = Literal[
    "data_preprocessor",
    "data_summarizer",
    "error_handler",
    "end"
]
