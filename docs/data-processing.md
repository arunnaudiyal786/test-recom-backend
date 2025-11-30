# Data Preparation Pipeline - Developer Guide

This document provides a comprehensive guide for developers to understand, maintain, and extend the Data Preparation Pipeline for the Intelligent Ticket Management System.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Components Deep Dive](#components-deep-dive)
5. [State Management](#state-management)
6. [Agent Implementation Details](#agent-implementation-details)
7. [Workflow Orchestration](#workflow-orchestration)
8. [Configuration & Customization](#configuration--customization)
9. [Error Handling](#error-handling)
10. [Testing](#testing)
11. [Performance Considerations](#performance-considerations)
12. [Extending the Pipeline](#extending-the-pipeline)
13. [Troubleshooting](#troubleshooting)

---

## Overview

The Data Preparation Pipeline is a **LangGraph-based multi-agent system** that processes raw Jira ticket CSV data through three sequential stages:

1. **Data Validation** - Quality assessment and missing field identification
2. **Data Preprocessing** - HTML removal, text normalization, special character cleaning
3. **Data Summarization** - AI-generated semantic summaries using OpenAI

### Key Features

- Sequential multi-agent workflow with error routing
- Batch CSV processing (not single-ticket)
- Comprehensive data quality reporting
- Automatic HTML and special character cleaning
- AI-powered ticket summarization for downstream ML tasks
- Full audit trail of processing steps

### Output

The pipeline transforms:
```
Input CSV (12 columns) → Output CSV (19 columns)
```

New columns added:
- `Summary_cleaned`, `Summary_normalized`
- `Description_cleaned`, `Description_normalized`
- `Resolution_cleaned`, `Resolution_normalized`
- `AI_Summary` (100-150 word semantic summary)

---

## Architecture

### Pipeline Flow

```
┌─────────────────────┐
│   Input CSV File    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data Validator     │──────┐
│  Agent              │      │ (on error)
└──────────┬──────────┘      │
           │                 │
           ▼                 ▼
┌─────────────────────┐   ┌─────────────┐
│  Data Preprocessor  │   │   Error     │
│  Agent              │──▶│   Handler   │
└──────────┬──────────┘   └──────┬──────┘
           │                      │
           ▼                      │
┌─────────────────────┐          │
│  Data Summarizer    │──────────┘
│  Agent              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Output CSV File    │
└─────────────────────┘
```

### Directory Structure

```
src/
├── agents/
│   ├── data_validator_agent.py      # Stage 1: Validation
│   ├── data_preprocessor_agent.py   # Stage 2: Cleaning
│   └── data_summarizer_agent.py     # Stage 3: AI Summarization
│
├── models/
│   └── data_prep_state.py           # TypedDict state schema
│
├── prompts/
│   └── summarization_prompts.py     # LLM prompt templates
│
├── graph/
│   └── data_prep_workflow.py        # LangGraph workflow definition
│
└── utils/
    ├── config.py                    # Environment configuration
    └── openai_client.py             # OpenAI API wrapper

scripts/
└── run_data_preparation.py          # CLI entry point

data/
├── raw/                             # Input CSV files
└── processed/                       # Output CSV files
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install beautifulsoup4 html5lib
```

### 2. Configure Environment

Ensure your `.env` file has the OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Pipeline

Basic usage:
```bash
python3 scripts/run_data_preparation.py --input data/raw/historical_tickets.csv
```

With custom output:
```bash
python3 scripts/run_data_preparation.py \
  --input data/raw/tickets.csv \
  --output data/processed/prepared_tickets.csv
```

Or use the convenience script:
```bash
./scripts/run_data_prep.sh data/raw/historical_tickets.csv
```

### 4. Check Output

```python
import pandas as pd
df = pd.read_csv('data/processed/prepared_tickets.csv')
print(df.columns.tolist())
print(df['AI_Summary'].iloc[0])
```

---

## Components Deep Dive

### 1. Data Validator Agent

**Location**: `src/agents/data_validator_agent.py`

**Responsibilities**:
- Load CSV file using pandas
- Identify missing critical fields per row
- Calculate completeness scores
- Generate data quality report

**Key Code Snippet - Loading CSV**:
```python
async def _load_csv(self, file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding='utf-8',
        on_bad_lines='warn',
        na_values=['', 'NA', 'N/A', 'null', 'None'],
        keep_default_na=True,
        quotechar='"',           # Handle quoted multi-line fields
        escapechar='\\',
        doublequote=True,
        engine='python'          # Better multi-line support
    )
    return df
```

**Key Code Snippet - Row Validation**:
```python
async def _validate_rows(self, df: pd.DataFrame) -> List[Dict]:
    per_row_results = []

    for idx, row in df.iterrows():
        row_validation = {
            "row_index": int(idx),
            "is_valid": True,
            "missing_fields": [],
            "completeness_score": 0.0,
            "warnings": []
        }

        # Check critical fields
        for field in self.critical_fields:
            if field in df.columns:
                value = row.get(field)
                if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                    row_validation["missing_fields"].append(field)
                    row_validation["is_valid"] = False

        # Calculate completeness score
        row_validation["completeness_score"] = self._calculate_row_completeness(row, df.columns)
        per_row_results.append(row_validation)

    return per_row_results
```

**Output State Fields**:
- `raw_data` - List of ticket dictionaries
- `validation_report` - Overall quality metrics
- `per_row_validation` - Per-row validation results

---

### 2. Data Preprocessor Agent

**Location**: `src/agents/data_preprocessor_agent.py`

**Responsibilities**:
- Remove HTML tags using BeautifulSoup
- Clean special characters
- Normalize whitespace
- Create cleaned and normalized versions of text fields

**Key Code Snippet - HTML Removal**:
```python
def _remove_html_tags(self, text: str) -> tuple[str, int]:
    html_tag_pattern = r'<[^>]+>'
    tags_found = len(re.findall(html_tag_pattern, text))

    if tags_found == 0:
        return text, 0

    # Decode HTML entities
    text = html.unescape(text)

    # Parse with BeautifulSoup
    soup = BeautifulSoup(text, 'html5lib')
    cleaned = soup.get_text(separator=' ')

    return cleaned, tags_found
```

**Key Code Snippet - Special Character Cleaning**:
```python
def _clean_special_characters(self, text: str) -> tuple[str, int]:
    preserved_punctuation = set(".,!?;:'\"()-_/\\@#$%&*+=<>[]{}|`~")
    cleaned_chars = []
    special_count = 0

    for char in text:
        if char.isalnum() or char.isspace():
            cleaned_chars.append(char)
        elif char in preserved_punctuation:
            cleaned_chars.append(char)
        elif ord(char) > 127:
            # Handle non-ASCII
            if char in 'áéíóúÁÉÍÓÚñÑüÜ':
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(' ')
                special_count += 1
        else:
            cleaned_chars.append(' ')
            special_count += 1

    return ''.join(cleaned_chars), special_count
```

**Output State Fields**:
- `preprocessed_data` - Cleaned ticket dictionaries with new columns
- `normalization_report` - Statistics on what was cleaned

**New Columns Added**:
- `{Field}_cleaned` - HTML/special chars removed, original case preserved
- `{Field}_normalized` - Lowercase version for ML processing

---

### 3. Data Summarizer Agent

**Location**: `src/agents/data_summarizer_agent.py`

**Responsibilities**:
- Generate AI summaries for each ticket using OpenAI
- Batch processing with rate limiting
- Combine all data into final CSV
- Save output file

**Key Code Snippet - Batch Processing**:
```python
async def _generate_all_summaries(self, preprocessed_data: List[Dict]) -> tuple[List[str], Dict]:
    summaries = []

    # Process in batches of 5
    total_batches = (len(preprocessed_data) + self.batch_size - 1) // self.batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(preprocessed_data))
        batch = preprocessed_data[start_idx:end_idx]

        # Process batch concurrently
        tasks = [
            self._summarize_single_ticket(ticket, idx)
            for idx, ticket in enumerate(batch, start=start_idx)
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        for result in batch_results:
            if isinstance(result, Exception):
                summaries.append(f"ERROR: Failed to generate summary - {str(result)}")
            else:
                summaries.append(result)

        # Rate limiting delay between batches
        if batch_idx < total_batches - 1:
            await asyncio.sleep(self.delay_between_batches)

    return summaries, stats
```

**Key Code Snippet - Single Ticket Summarization**:
```python
async def _summarize_single_ticket(self, ticket: Dict, ticket_idx: int) -> str:
    # Extract fields (use cleaned versions)
    summary_text = ticket.get("Summary_cleaned", ticket.get("Summary", "N/A"))
    description_text = ticket.get("Description_cleaned", ticket.get("Description", "N/A"))

    # Generate prompt
    prompt = get_ticket_summarization_prompt(
        summary=summary_text,
        description=description_text,
        priority=ticket.get("Issue Priority", ""),
        issue_type=ticket.get("issue type", ""),
        labels=ticket.get("Labels", ""),
        resolution=ticket.get("Resolution_cleaned", "")
    )

    # Call OpenAI with JSON response
    messages = [{"role": "user", "content": prompt}]
    response_json = await self.client.chat_completion_json(
        messages=messages,
        model=self.model,
        temperature=0.3,
        max_tokens=1000
    )

    # Extract combined summary
    return response_json.get("combined_summary", "")
```

**Output State Fields**:
- `summaries` - List of AI-generated summary strings
- `summarization_stats` - Statistics about summarization
- `final_data` - Complete dataset with all columns
- `output_file_path` - Path where CSV was saved

---

## State Management

### DataPrepState Schema

**Location**: `src/models/data_prep_state.py`

The state uses **TypedDict with `total=False`** to allow partial updates from agents:

```python
class DataPrepState(TypedDict, total=False):
    # Input
    input_file_path: str
    output_file_path: str

    # Raw data
    raw_data: List[Dict[str, Any]]
    total_rows: int
    column_names: List[str]

    # Validation outputs
    validation_report: Dict[str, Any]
    per_row_validation: List[Dict[str, Any]]

    # Preprocessing outputs
    preprocessed_data: List[Dict[str, Any]]
    normalization_report: Dict[str, Any]

    # Summarization outputs
    summaries: List[str]
    summarization_stats: Dict[str, Any]
    final_data: List[Dict[str, Any]]

    # Workflow control
    status: Literal["processing", "success", "error", "failed"]
    current_agent: str
    error_message: Optional[str]
    processing_stage: str

    # Audit trail (uses reducer for accumulation)
    messages: Annotated[List[Dict[str, str]], operator.add]
```

### Agent Output Pattern

Each agent returns a **partial state update**:

```python
class DataPrepAgentOutput(TypedDict, total=False):
    status: Literal["success", "error"]
    current_agent: str
    processing_stage: str
    error_message: Optional[str]

    # Agent-specific fields...
    validation_report: Optional[Dict]
    preprocessed_data: Optional[List[Dict]]
    summaries: Optional[List[str]]

    # Audit messages
    messages: Optional[List[Dict[str, str]]]
```

**Important**: Agents return partial dicts, not complete DataPrepState objects. LangGraph merges these into the growing state.

---

## Workflow Orchestration

### Building the Workflow

**Location**: `src/graph/data_prep_workflow.py`

```python
def build_data_prep_workflow() -> StateGraph:
    workflow = StateGraph(DataPrepState)

    # Add agent nodes
    workflow.add_node("Data Validator Agent", data_validator_agent)
    workflow.add_node("Data Preprocessor Agent", data_preprocessor_agent)
    workflow.add_node("Data Summarizer Agent", data_summarizer_agent)
    workflow.add_node("Error Handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("Data Validator Agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "Data Validator Agent",
        route_after_validation,
        {
            "data_preprocessor": "Data Preprocessor Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Data Preprocessor Agent",
        route_after_preprocessing,
        {
            "data_summarizer": "Data Summarizer Agent",
            "error_handler": "Error Handler"
        }
    )

    workflow.add_conditional_edges(
        "Data Summarizer Agent",
        route_after_summarization,
        {
            "end": END,
            "error_handler": "Error Handler"
        }
    )

    return workflow.compile()
```

### Routing Functions

```python
def route_after_validation(state: DataPrepState) -> DataPrepRoutingDecision:
    status = state.get("status", "error")

    if status == "error":
        return "error_handler"

    return "data_preprocessor"
```

The pattern is simple: check status, route to error handler on failure, otherwise proceed to next agent.

---

## Configuration & Customization

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional (with defaults)
CLASSIFICATION_MODEL=gpt-4o      # Used for summarization
CLASSIFICATION_TEMPERATURE=0.2
MAX_RETRIES=3
RETRY_DELAY_SECONDS=2
```

### Customizable Scoring Logic

**Location**: `src/agents/data_validator_agent.py:_calculate_row_completeness()`

You can adjust field weights to match your data quality standards:

```python
def _calculate_row_completeness(self, row: pd.Series, columns: pd.Index) -> float:
    # Critical field weights (80% of total)
    critical_weights = {
        "Summary": 0.25,           # Most important
        "Description": 0.25,       # Essential for context
        "Resolution": 0.20,        # Key for training
        "Issue Priority": 0.10,    # Important metadata
        "Labels": 0.10             # Classification target
    }

    # Important field weights (20% of total)
    important_weights = {
        "key": 0.02,
        "created": 0.02,
        "closed date": 0.02,
        "issue type": 0.02,
        "assignee": 0.01,
        "IT Team": 0.005,
        "Reporter": 0.005
    }

    # Calculate weighted score...
```

### Customizable Summarization Prompt

**Location**: `src/prompts/summarization_prompts.py:get_ticket_summarization_prompt()`

Adjust what aspects the AI focuses on:

```python
def get_ticket_summarization_prompt(summary, description, priority, issue_type, labels, resolution):
    prompt = f"""You are an AI technical ticket summarizer...

Summarization Guidelines:
1. Extract the CORE PROBLEM in 1-2 sentences
2. Identify KEY TECHNICAL COMPONENTS
3. Note the BUSINESS IMPACT
4. Highlight ROOT CAUSE indicators
5. Summarize RESOLUTION APPROACH
6. Keep total summary under 150 words

Output Format (JSON only):
{{
  "core_problem": "Brief description",
  "technical_components": ["comp1", "comp2"],
  "business_impact": "Impact description",
  "root_cause_indicators": "Indicators",
  "resolution_summary": "Resolution approach",
  "severity_assessment": "Low|Medium|High|Critical",
  "key_keywords": ["keyword1", "keyword2"],
  "combined_summary": "100-150 word summary"
}}
"""
    return prompt
```

---

## Error Handling

### Agent-Level Error Handling

Every agent follows this pattern:

```python
async def __call__(self, state: DataPrepState) -> DataPrepAgentOutput:
    try:
        # Process data...
        return {
            "status": "success",
            "current_agent": "Agent Name",
            # ... results
        }
    except Exception as e:
        return {
            "status": "error",
            "current_agent": "Agent Name",
            "error_message": f"Descriptive error: {str(e)}"
        }
```

### Workflow-Level Error Routing

When any agent returns `status: "error"`, the routing function directs to the Error Handler:

```python
def error_handler_node(state: DataPrepState) -> dict:
    error_message = state.get("error_message", "Unknown error")
    current_agent = state.get("current_agent", "unknown")

    return {
        "status": "failed",
        "error_message": f"Pipeline failed at {current_agent}: {error_message}",
        "messages": [{
            "role": "assistant",
            "content": f"Data preparation failed at {current_agent} stage."
        }]
    }
```

**Philosophy**: Fail fast and escalate. No retry logic - errors immediately end the workflow with detailed diagnostics.

---

## Testing

### Unit Testing Individual Agents

```python
import asyncio
from src.agents.data_validator_agent import DataValidatorAgent

async def test_validator():
    agent = DataValidatorAgent()

    state = {
        "input_file_path": "data/raw/test_tickets.csv"
    }

    result = await agent(state)

    assert result["status"] == "success"
    assert "validation_report" in result
    assert len(result["raw_data"]) > 0

asyncio.run(test_validator())
```

### End-to-End Testing

```bash
# Create small test dataset
python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/historical_tickets.csv', engine='python')
df.head(3).to_csv('data/raw/test_tickets.csv', index=False)
"

# Run pipeline
python3 scripts/run_data_preparation.py \
  --input data/raw/test_tickets.csv \
  --output data/processed/test_output.csv

# Verify output
python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/test_output.csv')
print('Columns:', df.columns.tolist())
print('Rows:', len(df))
print('Sample Summary:', df['AI_Summary'].iloc[0][:200])
"
```

### Testing Error Handling

```python
# Test with invalid file path
python3 scripts/run_data_preparation.py --input /nonexistent/file.csv

# Test with malformed CSV (create one manually)
echo "col1,col2" > bad.csv
echo "val1" >> bad.csv  # Missing second column
python3 scripts/run_data_preparation.py --input bad.csv
```

---

## Performance Considerations

### Processing Time Estimates

| Dataset Size | Validation | Preprocessing | Summarization | Total |
|-------------|------------|---------------|---------------|-------|
| 10 tickets  | <1s        | <1s           | ~10s          | ~12s  |
| 100 tickets | <1s        | ~2s           | ~60s          | ~65s  |
| 1000 tickets| ~5s        | ~20s          | ~10min        | ~11min|

### Cost Estimates (OpenAI API)

- **Summarization**: ~$0.002-0.005 per ticket (gpt-4o)
- **100 tickets**: ~$0.50-1.00
- **1000 tickets**: ~$5-10

### Optimization Tips

1. **Increase batch size** (if API limits allow):
   ```python
   # In data_summarizer_agent.py
   self.batch_size = 10  # Up from 5
   self.delay_between_batches = 0.5  # Reduce delay
   ```

2. **Use cheaper model** for summarization:
   ```python
   self.model = "gpt-4o-mini"  # Cheaper but less capable
   ```

3. **Skip summarization** for low-quality rows:
   ```python
   # Only summarize rows with high completeness
   if row_completeness > 0.7:
       await self._summarize_single_ticket(ticket, idx)
   ```

---

## Extending the Pipeline

### Adding a New Agent

1. **Create the agent class**:
   ```python
   # src/agents/new_agent.py
   from src.models.data_prep_state import DataPrepState, DataPrepAgentOutput

   class NewAgent:
       def __init__(self):
           # Initialize dependencies
           pass

       async def __call__(self, state: DataPrepState) -> DataPrepAgentOutput:
           try:
               # Process data
               return {
                   "status": "success",
                   "current_agent": "New Agent",
                   "processing_stage": "new_stage",
                   # ... outputs
               }
           except Exception as e:
               return {
                   "status": "error",
                   "current_agent": "New Agent",
                   "error_message": str(e)
               }

   new_agent = NewAgent()
   ```

2. **Update state schema** (if new fields needed):
   ```python
   # In data_prep_state.py
   class DataPrepState(TypedDict, total=False):
       # ... existing fields
       new_agent_output: Optional[Dict[str, Any]]
   ```

3. **Add to workflow**:
   ```python
   # In data_prep_workflow.py
   from src.agents.new_agent import new_agent

   workflow.add_node("New Agent", new_agent)

   # Add routing
   workflow.add_conditional_edges(
       "Data Summarizer Agent",
       route_after_summarization,
       {
           "new_agent": "New Agent",  # Changed from "end"
           "error_handler": "Error Handler"
       }
   )
   ```

4. **Add routing function**:
   ```python
   def route_after_new_agent(state):
       if state.get("status") == "error":
           return "error_handler"
       return "end"
   ```

### Adding New Validation Rules

```python
# In data_validator_agent.py

def _validate_rows(self, df):
    # ... existing validation

    # Add custom rule: Check for minimum description length
    if len(str(row.get("Description", ""))) < 50:
        row_validation["warnings"].append("Description too short (<50 chars)")

    # Add custom rule: Validate priority values
    valid_priorities = ["Low", "Medium", "High", "Critical"]
    if row.get("Issue Priority") not in valid_priorities:
        row_validation["warnings"].append(f"Invalid priority: {row.get('Issue Priority')}")
```

---

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   ```bash
   # Check .env file exists and has key
   cat .env | grep OPENAI_API_KEY

   # Ensure python-dotenv is installed
   pip install python-dotenv
   ```

2. **"CSV parsing error: EOF inside string"**
   ```bash
   # CSV has multi-line fields not properly quoted
   # Use pandas to re-export properly:
   python3 -c "
   import pandas as pd
   df = pd.read_csv('data/raw/file.csv', engine='python', quotechar='\"')
   df.to_csv('data/raw/file_fixed.csv', index=False)
   "
   ```

3. **"Rate limit exceeded"**
   ```python
   # Increase delay in data_summarizer_agent.py
   self.delay_between_batches = 2.0  # Up from 1.0
   self.batch_size = 3  # Down from 5
   ```

4. **"No module named 'bs4'"**
   ```bash
   uv pip install beautifulsoup4 html5lib
   # or
   pip install beautifulsoup4 html5lib
   ```

5. **Low completeness scores**
   - Check that your CSV columns match expected names (case-sensitive)
   - Verify critical fields aren't empty strings
   - Adjust weights in `_calculate_row_completeness()` if needed

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   # Add to agent methods
   print(f"[Agent Name] Processing row {idx}: {row.get('key')}")
   ```

2. **Inspect intermediate state**:
   ```python
   # In run_data_preparation.py, after workflow execution
   import json
   print(json.dumps(final_state['validation_report'], indent=2))
   ```

3. **Test single agent**:
   ```python
   from src.agents.data_validator_agent import data_validator_agent
   import asyncio

   state = {"input_file_path": "data/raw/test.csv"}
   result = asyncio.run(data_validator_agent(state))
   print(result)
   ```

---

## Summary

The Data Preparation Pipeline provides a robust, extensible framework for processing Jira ticket CSV data. Key takeaways:

1. **Sequential Agent Architecture** - Each agent builds on the previous one's output
2. **TypedDict State Management** - Partial state updates merged by LangGraph
3. **Graceful Error Handling** - Fail fast with detailed diagnostics
4. **Batch Processing** - Rate-limited API calls prevent throttling
5. **Comprehensive Reporting** - Quality metrics, preprocessing stats, and audit trails

For questions or issues, consult the main `CLAUDE.md` file or contact the development team.
