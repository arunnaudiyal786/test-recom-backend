# Configuration Directory Documentation

This document provides comprehensive documentation for all configuration files in the `config/` directory of the Test Recommendation System backend.

---

## Table of Contents

1. [Overview](#overview)
2. [config.py - Python Configuration](#configpy---python-configuration)
3. [schema_config.yaml - Schema Configuration](#schema_configyaml---schema-configuration)
4. [search_config.json - Search Configuration](#search_configjson---search-configuration)
5. [Configuration Relationships](#configuration-relationships)

---

## Overview

The configuration directory contains three main files that control different aspects of the system:

| File | Format | Purpose |
|------|--------|---------|
| `config.py` | Python | Runtime configuration, API keys, model settings, thresholds |
| `schema_config.yaml` | YAML | Data schema mapping, domain definitions, UI settings |
| `search_config.json` | JSON | Vector search and hybrid scoring parameters |

---

## config.py - Python Configuration

**Location**: `test-recom-backend/config/config.py`

This is the main Python configuration class that manages all runtime settings. It uses environment variables for sensitive data and hardcoded values for stable configuration.

### OpenAI Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | `str` | *Required* | OpenAI API key loaded from `.env` file. System will raise `ValueError` if not set. |

### Model Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CLASSIFICATION_MODEL` | `str` | `"gpt-4o"` | Model used for domain classification and label assignment. Use cheaper models for cost savings. |
| `RESOLUTION_MODEL` | `str` | `"gpt-4o"` | Model used for resolution generation. Typically needs better reasoning capabilities. |
| `EMBEDDING_MODEL` | `str` | `"text-embedding-3-large"` | Model for generating text embeddings. Determines embedding dimensions. |

**Valid Model Options**:
- Classification/Resolution: `"gpt-4o"`, `"gpt-4o-mini"`, `"gpt-4-turbo"`
- Embedding: `"text-embedding-3-large"`, `"text-embedding-3-small"`, `"text-embedding-ada-002"`

### Temperature Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CLASSIFICATION_TEMPERATURE` | `float` | `0.2` | Low temperature for deterministic, consistent classification results. |
| `RESOLUTION_TEMPERATURE` | `float` | `0.6` | Higher temperature for more creative, varied resolution suggestions. |

**Temperature Guidelines**:
- `0.0 - 0.3`: Deterministic, consistent outputs (good for classification)
- `0.4 - 0.7`: Balanced creativity and consistency (good for generation)
- `0.8 - 1.0`: High creativity, more varied outputs

### Token Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_RESOLUTION_TOKENS` | `int` | `8000` | Maximum tokens for resolution generation output. Controls response length. |

### Vector Database Paths

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PROJECT_ROOT` | `Path` | Auto-detected | Base path for the backend project (parent of config directory). |
| `FAISS_INDEX_PATH` | `Path` | `data/faiss_index/tickets.index` | Path to the FAISS vector index file. |
| `FAISS_METADATA_PATH` | `Path` | `data/faiss_index/metadata.json` | Path to the FAISS metadata JSON file. |

### Data Source Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HISTORICAL_TICKETS_CSV` | `str` | `"test_plan_historical.csv"` | Filename of the historical tickets CSV data. |
| `HISTORICAL_TICKETS_PATH` | `Path` | `data/raw/{filename}` | Full path to the historical tickets CSV file. |

### Processing Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TOP_K_SIMILAR_TICKETS` | `int` | `10` | Number of similar tickets to retrieve from FAISS search. Affects context size for downstream agents. |
| `CLASSIFICATION_CONFIDENCE_THRESHOLD` | `float` | `0.7` | Minimum confidence (0-1) required to accept a domain classification. |
| `LABEL_CONFIDENCE_THRESHOLD` | `float` | `0.7` | Minimum confidence (0-1) required to assign a label. |

### AI Label Generation Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LABEL_GENERATION_ENABLED` | `bool` | `True` | Enable/disable AI-generated labels beyond predefined categories. |
| `BUSINESS_LABEL_MAX_COUNT` | `int` | `3` | Maximum number of business labels to generate per ticket. |
| `TECHNICAL_LABEL_MAX_COUNT` | `int` | `3` | Maximum number of technical labels to generate per ticket. |
| `GENERATED_LABEL_CONFIDENCE_THRESHOLD` | `float` | `0.7` | Minimum confidence for accepting AI-generated labels. |

### Retry Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_RETRIES` | `int` | `3` | Maximum number of retries for failed API calls. |
| `RETRY_DELAY_SECONDS` | `int` | `2` | Delay between retry attempts in seconds. |

### Embedding Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBEDDING_DIMENSIONS` | `int` | `3072` | Dimensionality of embeddings. Must match the embedding model (text-embedding-3-large = 3072). |

### Category Labeling Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CATEGORIES_JSON_PATH` | `Path` | `data/metadata/categories.json` | Path to the category taxonomy JSON file. |
| `CATEGORY_DEFAULT_CONFIDENCE_THRESHOLD` | `float` | `0.7` | Default confidence threshold when category has no specific threshold. |
| `CATEGORY_MAX_LABELS_PER_TICKET` | `int` | `3` | Maximum number of categories to assign per ticket. |
| `CATEGORY_NOVELTY_DETECTION_THRESHOLD` | `float` | `0.5` | Confidence below this triggers potential novel category detection. |
| `CATEGORY_CLASSIFICATION_MODEL` | `str` | `"gpt-4o"` | Model used specifically for category classification. |
| `CATEGORY_CLASSIFICATION_TEMPERATURE` | `float` | `0.2` | Low temperature for deterministic category classification. |

### Hybrid Semantic + LLM Classification

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CATEGORY_EMBEDDINGS_PATH` | `Path` | `data/metadata/category_embeddings.json` | Path to pre-computed category embeddings (JSON format). |
| `SEMANTIC_TOP_K_CANDIDATES` | `int` | `5` | Number of candidate categories from semantic search before LLM refinement. |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `float` | `0.3` | Minimum similarity score (0-1) to consider a category as candidate. |
| `ENSEMBLE_SEMANTIC_WEIGHT` | `float` | `0.4` | Weight for semantic similarity in ensemble scoring (40%). |
| `ENSEMBLE_LLM_WEIGHT` | `float` | `0.6` | Weight for LLM classifier confidence in ensemble scoring (60%). |

**Note**: `ENSEMBLE_SEMANTIC_WEIGHT + ENSEMBLE_LLM_WEIGHT` must equal `1.0`.

### Novelty Detection Configuration

The novelty detection system uses three signals to determine if a ticket represents a new, uncategorized type.

#### Signal 1: Maximum Confidence Score

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NOVELTY_SIGNAL1_THRESHOLD` | `float` | `0.5` | If best category match confidence is below this, Signal 1 fires. |

#### Signal 2: Confidence Distribution Entropy

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NOVELTY_SIGNAL2_THRESHOLD` | `float` | `0.7` | If normalized entropy exceeds this, Signal 2 fires (high uncertainty across categories). |

#### Signal 3: Embedding Distance to Centroids

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NOVELTY_SIGNAL3_THRESHOLD` | `float` | `0.4` | If minimum distance to any category centroid exceeds this, Signal 3 fires. |

#### Signal Weights

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NOVELTY_SIGNAL1_WEIGHT` | `float` | `0.4` | Weight for Signal 1 (max confidence) in novelty score. |
| `NOVELTY_SIGNAL2_WEIGHT` | `float` | `0.3` | Weight for Signal 2 (entropy) in novelty score. |
| `NOVELTY_SIGNAL3_WEIGHT` | `float` | `0.3` | Weight for Signal 3 (embedding distance) in novelty score. |

**Note**: Signal weights must sum to `1.0`.

#### Final Novelty Decision

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NOVELTY_SCORE_THRESHOLD` | `float` | `0.6` | Final threshold for novelty decision. `is_novel = (max_confidence < SIGNAL1_THRESHOLD) OR (novelty_score > NOVELTY_SCORE_THRESHOLD)` |

### Validation Method

```python
@classmethod
def validate(cls):
    """Validates configuration on import."""
```

The `validate()` method is called automatically when the module is imported. It checks:
1. `OPENAI_API_KEY` is set
2. `CLASSIFICATION_MODEL` is a valid option
3. `EMBEDDING_MODEL` is a valid option

---

## schema_config.yaml - Schema Configuration

**Location**: `test-recom-backend/config/schema_config.yaml`

This YAML file defines how CSV data is mapped to the system's internal model, domain definitions, and UI settings.

### Data Source Type

| Key | Type | Options | Description |
|-----|------|---------|-------------|
| `data_source_type` | `str` | `historical_tickets`, `test_plan` | Determines which parsing logic to use for CSV data. |

### Vectorization Configuration

Controls which columns are used to generate embeddings.

```yaml
vectorization:
  columns: [list]        # Columns to include in embedding text
  separator: str         # Separator between column values
  clean_text: bool       # Whether to normalize text before embedding
  max_tokens: int        # Maximum tokens per embedding
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `columns` | `list[str]` | See below | Internal field names to concatenate for embedding. |
| `separator` | `str` | `". "` | String used to join column values. |
| `clean_text` | `bool` | `true` | Enable text cleaning/normalization before embedding. |
| `max_tokens` | `int` | `8000` | Maximum tokens per embedding (text-embedding-3-large supports 8191). |

**Default Vectorization Columns**:
- `description` - Core content
- `test_steps` - Test step details
- `expected_result` - Expected outcomes
- `precondition` - Setup requirements
- `category_labels` - Category information

### Column Mappings

Maps CSV column names to internal field names.

#### Required Fields

| Internal Field | CSV Column | Description |
|----------------|------------|-------------|
| `ticket_id` | `Issue_Key` | Unique identifier for each record. |
| `title` | `Summary` | Short title/summary. |
| `description` | `Description` | Detailed description. |

#### Optional Fields

| Internal Field | CSV Column | Description |
|----------------|------------|-------------|
| `priority` | `null` | Priority level (derived from story_points if null). |
| `resolution_steps` | `Expected_Result` | Steps for resolution. |
| `created_date` | `null` | Created date column. |
| `closed_date` | `null` | Closed date column. |
| `issue_type` | `Type` | Type of issue. |
| `assignee` | `Labels6` | Person assigned. |
| `team` | `Labels3` | Team name. |
| `reporter` | `null` | Reporter email. |

#### Test Plan Specific Fields

| Internal Field | CSV Column | Description |
|----------------|------------|-------------|
| `precondition` | `Precondition` | Test preconditions. |
| `test_steps` | `Test_Step_Description` | Test step descriptions. |
| `expected_result` | `Expected_Result` | Expected test results. |
| `story_points` | `Story_Points` | Story point estimate. |
| `status` | `Status` | Current status. |

### Labels Configuration

Defines which columns contain label values.

```yaml
labels:
  columns: [list]           # Columns containing label values
  exclude_columns: [list]   # Columns to exclude from labels
```

| Column | Purpose |
|--------|---------|
| `Labels2` | Domain code (BL-CORE, EN-ENROLL, etc.) |
| `Labels3` | Team name (BillingTeam, etc.) |
| `Labels4` | DoR status |
| `Labels5` | Component name (CoreBilling, etc.) |
| `Labels6` | QA Lead |
| `Labels7` | Architect |
| `Labels8` | Environment (PROD, STAGE) |
| `Labels9` | Refinement status |

**Excluded**: `Labels` (contains story IDs, not labels)

### Domains Configuration

Defines business domains for classification.

#### Domain Extraction

| Key | Type | Value | Description |
|-----|------|-------|-------------|
| `extraction_column` | `str` | `Labels2` | CSV column containing domain codes. |

#### Domain Map

Maps CSV values to domain names:

| CSV Value | Domain Name |
|-----------|-------------|
| `BL-CORE` | Billing |
| `EN-ENROLL` | Enrollment |
| `CL-CLAIMS` | Claims |
| `PR-PREMIUM` | Premium |
| `RN-RENEWAL` | Renewal |
| `IN-INTEGRATION` | Integration |
| `RP-REPORTING` | Reporting |
| `CS-CUSTSERV` | CustomerService |
| `SEC-SECURITY` | Security |
| `PF-PERFORMANCE` | Performance |

#### Domain Definitions

Each domain has the following structure:

```yaml
DomainName:
  full_name: str            # Display name
  description: str          # Brief description
  color_scheme: str         # UI color scheme reference
  classification_prompt: |  # Multi-line LLM prompt
    ...
```

**Domain Details**:

| Domain | Full Name | Color | Description |
|--------|-----------|-------|-------------|
| `Billing` | Core Billing Services | blue | Billing operations, invoicing, payment processing, AWD setup |
| `Enrollment` | Member Enrollment Services | green | Member enrollment, family plans, COBRA, SEP, dependent management |
| `Claims` | Claims Processing | purple | Claims submission, adjudication, COB, denials, EOB generation |
| `Premium` | Premium Calculation | amber | Premium rates, subsidies, rate changes, APTC calculations |
| `Renewal` | Renewal Processing | orange | Annual renewals, plan migrations, AWD continuation |
| `Integration` | System Integration | cyan | EDI transactions, payment gateway, external system interfaces |
| `Reporting` | Analytics & Reporting | indigo | Dashboards, reconciliation reports, data analytics |
| `CustomerService` | Customer Service | pink | CSR portal, member support, account updates |
| `Security` | Security & Compliance | red | Data encryption, RBAC, PCI-DSS compliance, audit trails |
| `Performance` | Performance Testing | slate | Load testing, batch processing SLAs, throughput benchmarks |

### Priority Configuration

Defines how priority is determined from story points.

```yaml
priority:
  derive_from: story_points
  story_points_map:
    low_max: 8              # <= 8 points = Low
    medium_max: 21          # <= 21 points = Medium
    high_max: 34            # <= 34 points = High
                            # > 34 points = Critical
  valid_values: [Low, Medium, High, Critical]
```

| Story Points | Priority |
|--------------|----------|
| 0 - 8 | Low |
| 9 - 21 | Medium |
| 22 - 34 | High |
| 35+ | Critical |

### UI Configuration

Frontend display settings including color schemes.

#### Color Schemes

Each color scheme defines Tailwind CSS classes for:

```yaml
color_name:
  bg: str      # Background color classes
  text: str    # Text color classes
  border: str  # Border color classes
  icon: str    # Icon color classes
```

**Available Color Schemes**: `blue`, `green`, `purple`, `amber`, `orange`, `cyan`, `indigo`, `pink`, `red`, `slate`

Each includes both light and dark mode variants (e.g., `bg-blue-50 dark:bg-blue-950`).

#### Sample Ticket Placeholder

```yaml
ui:
  sample_ticket_placeholder: |
    Enter test case description here...

    Example:
    Verify AWD payment processing when member enrolls in family plan.
    Setup includes primary member and 2 dependents with combined premium.
```

---

## search_config.json - Search Configuration

**Location**: `test-recom-backend/config/search_config.json`

This JSON file controls the vector search and hybrid scoring parameters used by the Pattern Recognition Agent.

### Full Configuration

```json
{
  "top_k": 30,
  "vector_weight": 0.7,
  "metadata_weight": 0.3,
  "priority_weights": {
    "Critical": 1.0,
    "High": 0.8,
    "Medium": 0.5,
    "Low": 0.3
  },
  "time_normalization_hours": 100.0,
  "domain_filter": null
}
```

### Variables

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `top_k` | `int` | `30` | Maximum number of results to retrieve from FAISS before filtering. |
| `vector_weight` | `float` | `0.7` | Weight for vector similarity in hybrid scoring (70%). |
| `metadata_weight` | `float` | `0.3` | Weight for metadata relevance in hybrid scoring (30%). |
| `priority_weights` | `object` | See below | Importance multipliers for each priority level. |
| `time_normalization_hours` | `float` | `100.0` | Normalization factor for resolution time (hours). Used in metadata scoring. |
| `domain_filter` | `str|null` | `null` | Optional domain to filter results. `null` means no filtering. |

### Priority Weights

| Priority | Weight | Description |
|----------|--------|-------------|
| `Critical` | `1.0` | Highest importance, full weight |
| `High` | `0.8` | High importance |
| `Medium` | `0.5` | Moderate importance |
| `Low` | `0.3` | Lower importance |

### Hybrid Scoring Formula

```
hybrid_score = (vector_weight × vector_similarity) + (metadata_weight × metadata_score)
```

Where:
- `vector_similarity` = Cosine similarity from FAISS (0-1)
- `metadata_score` = Combination of priority weight and time factor

---

## Configuration Relationships

### How Files Interact

```
┌─────────────────────────────────────────────────────────────────┐
│                        config.py                                 │
│  (Runtime settings, API keys, model selection, thresholds)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     schema_config.yaml                           │
│  (Data mapping, domain definitions, UI colors)                   │
│                                                                  │
│  • Defines which CSV columns map to which fields                 │
│  • Defines domains referenced by classification                  │
│  • Provides prompts used by config.py model settings             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     search_config.json                           │
│  (Search tuning parameters)                                      │
│                                                                  │
│  • Uses TOP_K_SIMILAR_TICKETS from config.py as fallback         │
│  • Applied during pattern recognition retrieval                  │
└─────────────────────────────────────────────────────────────────┘
```

### Common Modification Scenarios

| Scenario | Files to Modify |
|----------|-----------------|
| Change OpenAI model | `config.py` |
| Add new domain | `schema_config.yaml` (domains section) |
| Adjust search results count | `search_config.json` (top_k) or `config.py` (TOP_K_SIMILAR_TICKETS) |
| Change confidence thresholds | `config.py` |
| Map new CSV columns | `schema_config.yaml` (column_mappings) |
| Adjust hybrid scoring weights | `search_config.json` |
| Add new color scheme | `schema_config.yaml` (ui.color_schemes) |
| Change novelty detection sensitivity | `config.py` (NOVELTY_* variables) |

### Environment Variables

Only `OPENAI_API_KEY` is loaded from the `.env` file. All other configuration is managed through these config files.

```bash
# .env file
OPENAI_API_KEY=sk-your-api-key-here
```

---

## Best Practices

1. **Never commit API keys**: Keep `OPENAI_API_KEY` in `.env` file (gitignored)
2. **Weights must sum correctly**:
   - `vector_weight + metadata_weight = 1.0`
   - `ENSEMBLE_SEMANTIC_WEIGHT + ENSEMBLE_LLM_WEIGHT = 1.0`
   - `NOVELTY_SIGNAL*_WEIGHT` must sum to `1.0`
3. **Rebuild FAISS after schema changes**: Run `python3 scripts/setup_vectorstore.py`
4. **Test threshold changes**: Lower thresholds increase false positives, higher thresholds increase false negatives
5. **Match embedding dimensions**: `EMBEDDING_DIMENSIONS` must match the chosen `EMBEDDING_MODEL`
