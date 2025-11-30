#!/usr/bin/env python3
"""
Data Preparation Pipeline Entry Point.

This script runs the complete data preparation workflow:
1. Data Validation - Check for missing fields, generate quality report
2. Data Preprocessing - Remove HTML tags, clean special characters, normalize text
3. Data Summarization - Generate AI summaries for each ticket

Usage:
    python3 scripts/run_data_preparation.py --input data/raw/your_tickets.csv
    python3 scripts/run_data_preparation.py --input data/raw/tickets.csv --output data/processed/prepared.csv

Output:
    - Processed CSV file with original columns + cleaned columns + AI_Summary
    - Console output with data quality reports and processing statistics
"""
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.data_prep_workflow import get_data_prep_workflow
from src.utils.config import Config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run data preparation pipeline on Jira ticket CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/run_data_preparation.py --input data/raw/historical_tickets.csv
  python3 scripts/run_data_preparation.py --input my_tickets.csv --output prepared_data.csv
  python3 scripts/run_data_preparation.py --input data/raw/tickets.csv --visualize
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file containing Jira ticket data"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to output CSV file (default: data/processed/prepared_tickets_TIMESTAMP.csv)"
        )
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate workflow visualization graph"
    )

    return parser.parse_args()


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("   🔧 Data Preparation Pipeline")
    print("   Intelligent Ticket Management System")
    print("=" * 60)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


async def run_data_preparation(input_path: str, output_path: str) -> dict:
    """
    Run the complete data preparation workflow.

    Args:
        input_path: Path to input CSV file
        output_path: Path for output CSV file

    Returns:
        Final workflow state dictionary
    """
    print_section("Initializing Workflow")

    # Build the workflow
    workflow = get_data_prep_workflow()
    print("✅ LangGraph workflow compiled successfully")

    # Create initial state
    start_time = time.time()
    initial_state = {
        "input_file_path": str(Path(input_path).resolve()),
        "output_file_path": str(Path(output_path).resolve()),
        "processing_stage": "start",
        "status": "processing",
        "current_agent": "start",
        "error_message": None,
        "messages": [],
        "processing_start_time": start_time
    }

    print(f"📂 Input file: {initial_state['input_file_path']}")
    print(f"📂 Output file: {initial_state['output_file_path']}")

    print_section("Running Pipeline")

    # Execute the workflow
    final_state = await workflow.ainvoke(initial_state)

    return final_state


def print_validation_report(validation_report: dict):
    """Print the validation report in a readable format."""
    print_section("Data Quality Report")

    print(f"  Total Rows: {validation_report.get('total_rows', 0)}")
    print(f"  Valid Rows: {validation_report.get('valid_rows', 0)}")
    print(f"  Rows with Issues: {validation_report.get('rows_with_issues', 0)}")
    print(
        f"  Overall Completeness Score: "
        f"{validation_report.get('overall_completeness_score', 0):.2%}"
    )

    # Print column null percentages
    null_percentages = validation_report.get("column_null_percentages", {})
    if null_percentages:
        print("\n  Column Null Percentages:")
        for col, pct in sorted(null_percentages.items(), key=lambda x: x[1], reverse=True):
            if pct > 0:
                print(f"    • {col}: {pct:.1f}%")

    # Print critical fields with issues
    critical_issues = validation_report.get("critical_fields_with_issues", [])
    if critical_issues:
        print("\n  ⚠️  Critical Field Issues:")
        for issue in critical_issues:
            print(f"    • {issue}")

    # Print missing columns
    missing_cols = validation_report.get("columns_missing", [])
    if missing_cols:
        print("\n  📋 Expected columns not found:")
        for col in missing_cols:
            print(f"    • {col}")


def print_normalization_report(normalization_report: dict):
    """Print preprocessing statistics."""
    print_section("Preprocessing Report")

    print(f"  Rows Modified: {normalization_report.get('rows_modified', 0)}")
    print(f"  HTML Tags Removed: {normalization_report.get('html_tags_removed', 0)}")
    print(
        f"  Special Characters Cleaned: "
        f"{normalization_report.get('special_chars_cleaned', 0)}"
    )

    fields = normalization_report.get("fields_normalized", [])
    if fields:
        print(f"  New Columns Added: {', '.join(fields)}")


def print_summarization_stats(summarization_stats: dict):
    """Print summarization statistics."""
    print_section("AI Summarization Report")

    total = summarization_stats.get("total_tickets", 0)
    success = summarization_stats.get("total_summarized", 0)
    failed = summarization_stats.get("failed_summaries", 0)
    avg_length = summarization_stats.get("avg_summary_length", 0)

    print(f"  Total Tickets: {total}")
    print(f"  Successfully Summarized: {success}")
    print(f"  Failed Summaries: {failed}")
    print(f"  Success Rate: {(success / total * 100) if total > 0 else 0:.1f}%")
    print(f"  Average Summary Length: {avg_length:.0f} characters")


def print_final_summary(final_state: dict):
    """Print final pipeline summary."""
    print_section("Pipeline Summary")

    status = final_state.get("status", "unknown")
    total_time = final_state.get("total_processing_time_seconds", 0)
    output_path = final_state.get("output_file_path", "")

    if status == "success":
        print("  ✅ Pipeline completed successfully!")
    elif status == "failed":
        print("  ❌ Pipeline failed!")
        error = final_state.get("error_message", "Unknown error")
        print(f"  Error: {error}")
    else:
        print(f"  ⚠️  Pipeline ended with status: {status}")

    print(f"  Total Processing Time: {total_time:.2f} seconds")

    if output_path and status == "success":
        print(f"  Output File: {output_path}")

    # Print message audit trail
    messages = final_state.get("messages", [])
    if messages:
        print("\n  📝 Processing Log:")
        for i, msg in enumerate(messages, 1):
            content = msg.get("content", "")
            # Truncate long messages
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"    {i}. {content}")


async def main():
    """Main entry point."""
    print_banner()

    # Parse arguments
    args = parse_arguments()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)

    if not input_path.suffix.lower() == ".csv":
        print(f"❌ Error: Input file must be a CSV file, got: {input_path.suffix}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Generate timestamped output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Config.PROJECT_ROOT / "data" / "processed" / f"prepared_tickets_{timestamp}.csv"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally visualize workflow
    if args.visualize:
        print("\n📊 Generating workflow visualization...")
        from src.graph.data_prep_workflow import visualize_data_prep_workflow
        visualize_data_prep_workflow()

    try:
        # Run the pipeline
        final_state = await run_data_preparation(str(input_path), str(output_path))

        # Print reports
        if "validation_report" in final_state:
            print_validation_report(final_state["validation_report"])

        if "normalization_report" in final_state:
            print_normalization_report(final_state["normalization_report"])

        if "summarization_stats" in final_state:
            print_summarization_stats(final_state["summarization_stats"])

        # Print final summary
        print_final_summary(final_state)

        # Exit with appropriate code
        if final_state.get("status") == "success":
            print("\n" + "=" * 60)
            print("   🎉 Data preparation complete!")
            print("=" * 60 + "\n")
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
