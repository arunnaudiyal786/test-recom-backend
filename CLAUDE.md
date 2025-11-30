# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Intelligent Ticket Management System** built with LangGraph - a multi-agent pipeline that automatically classifies technical support tickets, finds similar historical cases, assigns labels, and generates detailed resolution plans using OpenAI LLMs and FAISS vector search.

**Key characteristic**: This is a **sequential multi-agent workflow** where each agent depends on the output of the previous one, not a conversational system.

## Essential Setup Commands

### First-Time Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure OpenAI API key
cp .env.example .env
# Then edit .env and add: OPENAI_API_KEY=sk-your-key-here

# 3. Generate sample historical data in CSV format (100 tickets)
python3 scripts/generate_sample_csv_data.py

# 4. Build FAISS vector index from CSV (costs ~$0.02)
python3 scripts/setup_vectorstore.py
```

### Running the System
```bash
# Process a single ticket (reads from input/current_ticket.json)
python3 main.py

# Run data ingestion pipeline directly (processes CSV data)
python3 -m src.vectorstore.data_ingestion

# Generate new sample historical data (CSV format)
python3 scripts/generate_sample_csv_data.py
```

### After Modifying Historical Data
```bash
# Rebuild FAISS index after adding/modifying historical tickets CSV
python3 scripts/setup_vectorstore.py
```

## Architecture: Multi-Agent Pipeline

### Core Workflow (Sequential, Not Parallel)

The system implements a **strict sequential pipeline** via LangGraph:

```
Classification Agent
    ↓ (domain: MM/CIW/Specialty)
Pattern Recognition Agent
    ↓ (top 20 similar tickets from FAISS)
Label Assignment Agent
    ↓ (assigned labels based on historical patterns)
Resolution Generation Agent
    ↓
Final Output (JSON)
```

**Critical**: Each agent MUST complete successfully before the next starts. The workflow uses **conditional routing** - on error, it routes directly to the error handler for manual escalation.

### State Management Pattern

This system uses **TypedDict-based state** (not Pydantic models) for LangGraph:

- **TicketState** (`src/models/state_schema.py`): Main workflow state with `total=False` for partial updates
- **AgentOutput**: Standard return type for all agents (partial state dict)
- **Annotated fields**: `messages` field uses `Annotated[List[Dict], operator.add]` for accumulation across agents

**Key insight**: Agents return **partial state dicts** that get merged into the full state, NOT complete TicketState objects.

### Agent Implementation Pattern

All agents follow this structure:

```python
class SomeAgent:
    async def __call__(self, state: TicketState) -> AgentOutput:
        try:
            # 1. Extract inputs from state
            # 2. Process with OpenAI/FAISS
            # 3. Return partial state update
            return {
                "status": "success",
                "current_agent": "agent_name",
                "agent_specific_field": result,
                "messages": [{"role": "assistant", "content": "..."}]
            }
        except Exception as e:
            # 4. Return error state for manual escalation
            return {
                "status": "error",
                "current_agent": "agent_name",
                "error_message": str(e)
            }
```

**Global singleton pattern**: Each agent is instantiated once at module level (e.g., `classification_agent = ClassificationAgent()`)

## Critical System Components

### 1. MTC-LLM Binary Classification (Not Multi-Class)

The Classification Agent uses **3 parallel binary classifiers** (MM, CIW, Specialty), NOT a single multi-class classifier:

- Each classifier returns: `{decision: bool, confidence: float, reasoning: str}`
- Final domain = classifier with highest confidence
- Runs in parallel via `asyncio.gather()`

**Why this matters**: When adding new domains or modifying classification logic, you must create separate binary classifier prompts in `src/prompts/classification_prompts.py`, not modify a single prompt.

### 2. FAISS Vector Store Architecture

- **Index type**: `IndexFlatIP` (exact inner product search)
- **Embeddings**: L2-normalized for cosine similarity
- **Dimension**: 3072 (text-embedding-3-large)
- **Storage**: Separate files for index (`tickets.index`) and metadata (`metadata.json`)

**Key operation**: Pattern Recognition Agent uses **domain filtering** - searches FAISS then filters results by classified domain.

**Lazy loading**: FAISS index loads on first use in Pattern Recognition Agent, not at startup.

### 3. Hybrid Scoring System

Pattern Recognition combines:
- 70% vector similarity (from FAISS)
- 30% metadata relevance (priority, resolution_time_hours)

Formula in `pattern_recognition_agent.py:apply_hybrid_scoring()`:
```python
hybrid_score = (0.7 * vector_score) + (0.3 * metadata_score)
```

### 4. Label Assignment Logic

The Label Assignment Agent:

1. Extracts **candidate labels** from top 20 similar tickets
2. Runs **binary classifier per label** (parallel via asyncio.gather)
3. Applies 0.7 confidence threshold
4. Considers historical frequency (e.g., "14/20 tickets have this label")

**Available labels** (defined in `src/prompts/label_assignment_prompts.py`):
- Code Fix, Data Fix, Configuration Fix
- #MM_ALDER, #MMALDR, #CIW_INTEGRATION, #SPECIALTY_CUSTOM

### 5. Error Handling Pattern

Every agent has:
- Try-catch block that returns `{"status": "error", "error_message": "..."}`
- Routing functions check status and route to "error_handler" on first error
- No retry logic - errors immediately escalate to manual review
- Error handler returns graceful degradation (manual escalation plan)

## Configuration Deep Dive

Environment variables in `.env`:

**Model Selection**:
- `CLASSIFICATION_MODEL`: Used for all binary classifiers (cheap, deterministic)
- `RESOLUTION_MODEL`: Used for resolution generation (more expensive, better reasoning)
- Temperature: 0.2 for classification (deterministic), 0.6 for resolution (balanced)

**Thresholds**:
- `CLASSIFICATION_CONFIDENCE_THRESHOLD`: 0.7 (domain must have 70%+ confidence)
- `LABEL_CONFIDENCE_THRESHOLD`: 0.7 (label must have 70%+ confidence)
- `TOP_K_SIMILAR_TICKETS`: 20 (retrieval count, feeds into label assignment & resolution)

**Why these matter**: Changing TOP_K affects downstream agents' context size. Lower thresholds increase false positives.

## Working with Historical Data

Historical tickets are stored in **CSV format** (`data/raw/historical_tickets.csv`) with the following columns:

**Required CSV Columns**:
- `key`: Ticket identifier (e.g., "JIRA-MM-001", "JIRA-CI-042", "JIRA-SP-015")
  - Format: JIRA-{DOMAIN_CODE}-{NUMBER}
  - Domain codes: MM (MM), CI (CIW), SP (Specialty)
- `Summary`: Short ticket title
- `Description`: Detailed ticket description
- `Issue Priority`: Low | Medium | High | Critical
- `Labels`: Comma-separated labels (e.g., "Code Fix,#MM_ALDER,#MMALDR")
- `Resolution`: Newline-separated resolution steps
- `created`: Created timestamp (YYYY-MM-DD HH:MM:SS)
- `closed date`: Closed timestamp (YYYY-MM-DD HH:MM:SS)
- `issue type`: Bug | Feature Request | Performance | Integration Issue | Data Issue | Configuration
- `assignee`: Email of assignee
- `IT Team`: Team name (e.g., "MM Core Team", "CIW Integration")
- `Reporter`: Email of reporter

**Example CSV Row**:
```csv
key,closed date,issue type,assignee,created,IT Team,Issue Priority,Labels,Reporter,Resolution,Summary,Description
JIRA-MM-001,2024-11-17 01:22:00,Performance,john.doe@company.com,2024-11-15 17:42:05,MM Infrastructure,Medium,"Configuration Fix,#MM_ALDER",ops.team@company.com,"1. Increase connection pool size
2. Add monitoring alerts",Database connection pool exhaustion,Service hitting max DB connections during peak hours...
```

**To add custom historical data**:
1. Edit `data/raw/historical_tickets.csv` (or replace with your data)
2. Ensure `key` column follows format: JIRA-{MM|CI|SP}-{NUMBER}
3. Use comma-separated values for `Labels` field
4. Use newline-separated steps for `Resolution` field
5. Run `python3 scripts/setup_vectorstore.py` to rebuild FAISS index
6. Cost: ~$0.0002 per ticket for embeddings

**Domain Extraction**: The system automatically extracts domain from the ticket key:
- `JIRA-MM-*` → MM domain
- `JIRA-CI-*` → CIW domain
- `JIRA-SP-*` → Specialty domain

## Prompt Engineering Guidelines

Prompts are in `src/prompts/`:

- **Classification prompts**: Define domain characteristics as bullet points
- **Label assignment prompts**: Include historical frequency context + specific criteria
- **Resolution prompts**: Use Chain-of-Thought format with structured JSON output

**All LLM calls use JSON mode** (`response_format={"type": "json_object"}`). Never remove JSON schema from prompts.

**Prompt caching**: Prompts are structured with static system context first, dynamic content last (for OpenAI prompt caching).

## Async/Await Patterns

This codebase uses async extensively:

- **Parallel operations**: `asyncio.gather()` for binary classifiers, batch embeddings
- **Sequential workflow**: LangGraph calls agents with `await workflow.ainvoke()`
- **OpenAI client**: All calls via `await client.chat_completion()` or `await client.generate_embedding()`

**Critical**: Main entry point uses `asyncio.run(main())`. All agent `__call__` methods are `async def`.

## Common Modification Patterns

### Adding a New Domain
1. Add binary classifier prompt in `src/prompts/classification_prompts.py`
2. Add domain to `self.domains` list in `classification_agent.py`
3. Update `determine_final_domain()` logic if needed
4. Generate historical tickets with new domain
5. Rebuild FAISS index

### Adding a New Label
1. Add label criteria to `LABEL_CRITERIA` dict in `src/prompts/label_assignment_prompts.py`
2. Include new label in historical tickets' `labels` array
3. Rebuild FAISS index

### Modifying Similarity Search
1. Edit `hybrid_scoring` weights in `pattern_recognition_agent.py`
2. Current: 70% vector / 30% metadata
3. Metadata factors: priority (60%) + resolution_time (40%)

### Changing Agent Pipeline
1. Modify workflow in `src/graph/workflow.py:build_ticket_workflow()`
2. Add new nodes with `workflow.add_node()`
3. Update conditional edges and routing functions
4. Create new routing function in `src/graph/state_manager.py`

## Testing Strategy

No formal test suite exists yet. To test changes:

1. **End-to-end test**: `python3 main.py` with sample ticket
2. **Individual agents**: Import and call directly with mock state
3. **FAISS operations**: Run `scripts/setup_vectorstore.py` and check index stats
4. **Data generation**: Run `scripts/generate_sample_data.py` and inspect JSON

## Performance Characteristics

- **Total processing time**: 8-12 seconds per ticket
- **Classification**: 2-3s (3 parallel API calls)
- **Pattern recognition**: 0.5-1s (FAISS search <1ms + 1 embedding API call)
- **Label assignment**: 2-3s (4-6 parallel API calls)
- **Resolution generation**: 3-5s (1 API call with 8K max tokens)

**Cost per ticket**: ~$0.07 (mostly resolution generation)

## Troubleshooting Common Issues

**"FAISS index not found"**: Run `python3 scripts/setup_vectorstore.py`

**Low classification confidence**: Check domain definitions in classification prompts match your ticket content

**No similar tickets found**: Verify domain filter matches classified domain, check FAISS index has tickets in that domain

**Agent timeout**: Check OpenAI API status or network connectivity

**State not updating**: Ensure agent returns correct `AgentOutput` format (partial dict, not full TicketState)

## Data Flow Summary

```
Input: input/current_ticket.json
  ↓
main.py → load_input_ticket()
  ↓
LangGraph workflow.ainvoke(initial_state)
  ↓
4 agents (each updates state)
  ↓
format_final_output()
  ↓
Output: output/ticket_resolution.json
```

**Input Format**: JSON file with the following structure:
```json
{
  "ticket_id": "JIRA-NEW-001",
  "title": "Brief ticket description",
  "description": "Detailed ticket description with context, symptoms, and impact",
  "priority": "High",
  "metadata": {
    "reported_by": "ops-team@example.com",
    "affected_users": 150,
    "environment": "production"
  }
}
```

**Historical Data Format**: CSV file (see "Working with Historical Data" section)

State flows through workflow as a single dict that gets progressively enriched by each agent.

## Important Constraints

- **Python 3.11+** required (uses modern type hints)
- **OpenAI API key** required (no fallback)
- **English-language tickets** (prompts optimized for English)
- **Sequential execution** (agents cannot run in parallel, each needs previous output)
- **Local FAISS** (no external vector database, index persisted to disk)
