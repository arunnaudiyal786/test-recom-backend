#!/bin/bash
#
# Data Preparation Pipeline Runner
#
# This script provides a convenient way to run the data preparation pipeline
# with sensible defaults and helpful error messages.
#
# Usage:
#   ./scripts/run_data_prep.sh <input_csv>
#   ./scripts/run_data_prep.sh <input_csv> <output_csv>
#   ./scripts/run_data_prep.sh --help
#
# Examples:
#   ./scripts/run_data_prep.sh data/raw/historical_tickets.csv
#   ./scripts/run_data_prep.sh data/raw/tickets.csv data/processed/prepared.csv
#   ./scripts/run_data_prep.sh --visualize data/raw/tickets.csv

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default values
INPUT_FILE=""
OUTPUT_FILE=""
VISUALIZE=false
VERBOSE=false

# Print usage information
print_usage() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           Data Preparation Pipeline Runner                   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "USAGE:"
    echo "  $0 [OPTIONS] <input_csv> [output_csv]"
    echo ""
    echo "ARGUMENTS:"
    echo "  input_csv     Required. Path to input CSV file containing Jira tickets"
    echo "  output_csv    Optional. Path for output CSV file"
    echo "                Default: data/processed/prepared_tickets_TIMESTAMP.csv"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help       Show this help message"
    echo "  -v, --visualize  Generate workflow visualization graph"
    echo "  --verbose        Enable verbose output"
    echo ""
    echo "EXAMPLES:"
    echo "  # Basic usage with auto-generated output filename"
    echo "  $0 data/raw/historical_tickets.csv"
    echo ""
    echo "  # Specify custom output path"
    echo "  $0 data/raw/tickets.csv data/processed/prepared.csv"
    echo ""
    echo "  # Generate workflow visualization"
    echo "  $0 --visualize data/raw/tickets.csv"
    echo ""
    echo "REQUIREMENTS:"
    echo "  - Python 3.11+"
    echo "  - OpenAI API key configured in .env file"
    echo "  - Required packages: pandas, beautifulsoup4, langgraph, openai"
    echo ""
    echo "OUTPUT:"
    echo "  The pipeline will generate a CSV file with:"
    echo "  - Original columns from input"
    echo "  - {Field}_cleaned columns (HTML/special chars removed)"
    echo "  - {Field}_normalized columns (lowercase for ML)"
    echo "  - AI_Summary column (100-150 word semantic summary)"
    echo ""
}

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    REQUIRED_VERSION="3.11"

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_error "Python 3.11+ is required. Found: Python $PYTHON_VERSION"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION found"

    # Check for .env file
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_warning ".env file not found in project root"
        print_info "Creating .env from .env.example..."
        if [ -f "$PROJECT_ROOT/.env.example" ]; then
            cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
            print_warning "Please edit .env and add your OPENAI_API_KEY"
        else
            print_error "No .env.example found. Please create .env with OPENAI_API_KEY"
            exit 1
        fi
    fi

    # Check for required Python packages
    print_info "Checking Python packages..."
    python3 -c "import pandas" 2>/dev/null || {
        print_error "pandas not installed. Run: pip install pandas"
        exit 1
    }
    python3 -c "import bs4" 2>/dev/null || {
        print_error "beautifulsoup4 not installed. Run: pip install beautifulsoup4 html5lib"
        exit 1
    }
    python3 -c "import langgraph" 2>/dev/null || {
        print_error "langgraph not installed. Run: pip install -r requirements.txt"
        exit 1
    }
    python3 -c "import openai" 2>/dev/null || {
        print_error "openai not installed. Run: pip install openai"
        exit 1
    }

    print_success "All required packages found"
}

# Validate input file
validate_input() {
    if [ -z "$INPUT_FILE" ]; then
        print_error "No input file specified"
        print_usage
        exit 1
    fi

    # Convert to absolute path if relative
    if [[ ! "$INPUT_FILE" = /* ]]; then
        INPUT_FILE="$PROJECT_ROOT/$INPUT_FILE"
    fi

    if [ ! -f "$INPUT_FILE" ]; then
        print_error "Input file not found: $INPUT_FILE"
        exit 1
    fi

    # Check file extension
    if [[ ! "$INPUT_FILE" =~ \.csv$ ]]; then
        print_error "Input file must be a CSV file: $INPUT_FILE"
        exit 1
    fi

    # Check file is not empty
    if [ ! -s "$INPUT_FILE" ]; then
        print_error "Input file is empty: $INPUT_FILE"
        exit 1
    fi

    print_success "Input file validated: $INPUT_FILE"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -v|--visualize)
                VISUALIZE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                if [ -z "$INPUT_FILE" ]; then
                    INPUT_FILE="$1"
                elif [ -z "$OUTPUT_FILE" ]; then
                    OUTPUT_FILE="$1"
                else
                    print_error "Too many arguments"
                    print_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
}

# Run the pipeline
run_pipeline() {
    print_info "Starting Data Preparation Pipeline..."
    echo ""

    # Build command arguments
    local ARGS=()
    ARGS+=("--input" "$INPUT_FILE")

    if [ -n "$OUTPUT_FILE" ]; then
        # Convert to absolute path if relative
        if [[ ! "$OUTPUT_FILE" = /* ]]; then
            OUTPUT_FILE="$PROJECT_ROOT/$OUTPUT_FILE"
        fi
        ARGS+=("--output" "$OUTPUT_FILE")
    fi

    if [ "$VISUALIZE" = true ]; then
        ARGS+=("--visualize")
    fi

    # Run the pipeline
    cd "$PROJECT_ROOT"

    if [ "$VERBOSE" = true ]; then
        print_info "Running: python3 scripts/run_data_preparation.py ${ARGS[*]}"
    fi

    # Execute with proper quoting
    python3 "$PROJECT_ROOT/scripts/run_data_preparation.py" "${ARGS[@]}"
    RESULT=$?

    return $RESULT
}

# Main execution
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           Data Preparation Pipeline                          ║"
    echo "║           Intelligent Ticket Management System               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""

    # Parse arguments
    parse_args "$@"

    # Check prerequisites
    check_prerequisites

    # Validate input
    validate_input

    # Run pipeline
    run_pipeline
    RESULT=$?

    if [ $RESULT -eq 0 ]; then
        echo ""
        print_success "Pipeline completed successfully!"
        echo ""
    else
        echo ""
        print_error "Pipeline failed with exit code: $RESULT"
        echo ""
        exit $RESULT
    fi
}

# Execute main function
main "$@"
