"""
CSV Exporter for ticket processing results.

Exports domain labels, historical/business/technical labels,
and top similar tickets to CSV format.
"""
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def extract_label_names(labels: List[Dict[str, Any]]) -> List[str]:
    """
    Extract label names from a list of label dicts.

    Args:
        labels: List of label dicts with 'label' key

    Returns:
        List of label name strings
    """
    if not labels:
        return []
    return [label.get('label', '') for label in labels if label.get('label')]


def format_similar_ticket(ticket: Dict[str, Any]) -> str:
    """
    Format a similar ticket for CSV output.

    Args:
        ticket: Similar ticket dict

    Returns:
        Formatted string with ticket ID and title
    """
    ticket_id = ticket.get('ticket_id', 'Unknown')
    title = ticket.get('title', 'No title')
    score = ticket.get('similarity_score', 0)
    return f"{ticket_id}: {title} ({score:.1%})"


def export_ticket_results_to_csv(
    state: Dict[str, Any],
    output_path: Optional[Path] = None,
    append: bool = True
) -> Path:
    """
    Export ticket processing results to CSV.

    Creates a CSV row with:
    - ticket_id: The processed ticket ID
    - domain_label: Classified domain (MM, CIW, Specialty)
    - historical_labels: Comma-separated historical labels
    - business_labels: Comma-separated AI-generated business labels
    - technical_labels: Comma-separated AI-generated technical labels
    - top_1_similar_ticket through top_5_similar_ticket: Top 5 similar tickets

    Args:
        state: Final workflow state dict
        output_path: Path for CSV file (default: output/ticket_results.csv)
        append: If True, append to existing file; if False, overwrite

    Returns:
        Path to the created/updated CSV file
    """
    # Default output path
    if output_path is None:
        from config import Config
        output_path = Config.PROJECT_ROOT / "output" / "ticket_results.csv"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract data from state
    ticket_id = state.get('ticket_id', 'Unknown')
    domain_label = state.get('classified_domain', '')

    # Historical labels (already a list of strings)
    historical_labels = state.get('historical_labels', [])
    historical_labels_str = ', '.join(historical_labels) if historical_labels else ''

    # Business labels (list of dicts with 'label' key)
    business_labels = state.get('business_labels', [])
    business_labels_str = ', '.join(extract_label_names(business_labels))

    # Technical labels (list of dicts with 'label' key)
    technical_labels = state.get('technical_labels', [])
    technical_labels_str = ', '.join(extract_label_names(technical_labels))

    # Similar tickets (top 5)
    similar_tickets = state.get('similar_tickets', [])
    top_similar = {}
    for i in range(5):
        if i < len(similar_tickets):
            top_similar[f'top_{i+1}_similar_ticket'] = format_similar_ticket(similar_tickets[i])
        else:
            top_similar[f'top_{i+1}_similar_ticket'] = ''

    # Build the row
    row = {
        'ticket_id': ticket_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'domain_label': domain_label,
        'classification_confidence': f"{state.get('classification_confidence', 0):.2%}",
        'historical_labels': historical_labels_str,
        'business_labels': business_labels_str,
        'technical_labels': technical_labels_str,
        **top_similar
    }

    # CSV column order
    fieldnames = [
        'ticket_id',
        'timestamp',
        'domain_label',
        'classification_confidence',
        'historical_labels',
        'business_labels',
        'technical_labels',
        'top_1_similar_ticket',
        'top_2_similar_ticket',
        'top_3_similar_ticket',
        'top_4_similar_ticket',
        'top_5_similar_ticket'
    ]

    # Check if file exists to determine if we need to write header
    file_exists = output_path.exists()
    write_header = not file_exists or not append

    # Write mode
    mode = 'a' if append and file_exists else 'w'

    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return output_path


def export_batch_results_to_csv(
    results: List[Dict[str, Any]],
    output_path: Optional[Path] = None
) -> Path:
    """
    Export multiple ticket results to CSV (batch mode).

    Args:
        results: List of final state dicts from multiple tickets
        output_path: Path for CSV file

    Returns:
        Path to the created CSV file
    """
    if not results:
        raise ValueError("No results to export")

    # Default output path with timestamp
    if output_path is None:
        from config import Config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Config.PROJECT_ROOT / "output" / f"ticket_results_batch_{timestamp}.csv"

    # Export first result (creates file with header)
    export_ticket_results_to_csv(results[0], output_path, append=False)

    # Append remaining results
    for state in results[1:]:
        export_ticket_results_to_csv(state, output_path, append=True)

    return output_path
