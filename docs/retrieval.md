# Retrieval System Documentation

This document provides an in-depth explanation of the retrieval logic used in the Intelligent Ticket Management System's RAG (Retrieval-Augmented Generation) pipeline.

---

## Quick Answer: How Does Retrieval Work?

### Q: Are we first embedding the input query and then doing a similarity search?

**Yes, exactly!** The retrieval follows this sequence:

```
Input Query (title + description)
         │
         ▼
┌─────────────────────────────┐
│  1. TEXT PREPROCESSING      │
│  combine_ticket_text()      │
│  "{title}. {description}"   │
│  + clean_text() (lowercase, │
│    remove HTML, normalize)  │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  2. EMBEDDING GENERATION    │
│  OpenAI text-embedding-3-   │
│  large API call             │
│  Output: 3072-dim vector    │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  3. L2 NORMALIZATION        │
│  faiss.normalize_L2()       │
│  (required for cosine sim)  │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  4. FAISS SIMILARITY SEARCH │
│  index.search(query, k*3)   │
│  Returns: indices, scores   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  5. DOMAIN FILTERING        │
│  Keep only tickets matching │
│  classified domain          │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  6. HYBRID RE-RANKING       │
│  70% vector + 30% metadata  │
│  Sort by final score        │
└─────────────────────────────┘
         │
         ▼
    Top K Results
```

### Q: What exactly is being searched/indexed?

**The FAISS index does NOT search specific "columns"** - it searches **a single embedding vector** that represents the **combined title + description** of each ticket.

#### What Gets Embedded (Indexed):

```python
# When building the index (data_ingestion.py):
text_to_embed = f"{ticket['title']}. {ticket['description']}"
# Example: "database connection timeout. the mm_alder service is experiencing
#          intermittent connection timeout errors when trying to connect..."
```

#### What Gets Stored as Metadata (NOT Searched, but Returned):

The FAISS index only stores vectors. All other ticket data is stored **separately** in `metadata.json`:

| Field | Searched? | Used For |
|-------|-----------|----------|
| `ticket_id` | ❌ | Display |
| `title` | ✅ (in embedding) | Display + Embedding |
| `description` | ✅ (in embedding) | Display + Embedding |
| `domain` | ❌ | Post-search filtering |
| `priority` | ❌ | Hybrid scoring |
| `labels` | ❌ | Display |
| `resolution_steps` | ❌ | Display |
| `resolution_time_hours` | ❌ | Hybrid scoring |

### Q: How is similarity calculated?

1. **Vector Similarity (Cosine)**: FAISS computes inner product between L2-normalized vectors
   - Score range: 0.0 to 1.0 (higher = more similar)
   - This is pure semantic similarity based on text content

2. **Hybrid Score** (what you see in the UI):
   ```
   hybrid_score = (0.7 × vector_similarity) + (0.3 × metadata_score)

   where:
   metadata_score = (0.6 × priority_weight) + (0.4 × time_score)
   ```

### Summary Table

| Step | What Happens | Code Location |
|------|--------------|---------------|
| 1. Combine text | `"{title}. {description}"` | `helpers.py:combine_ticket_text()` |
| 2. Clean text | Lowercase, remove HTML, normalize whitespace | `helpers.py:clean_text()` |
| 3. Generate embedding | OpenAI API → 3072-dim vector | `embedding_generator.py` |
| 4. Normalize | L2 normalization for cosine similarity | `faiss_manager.py:search()` |
| 5. FAISS search | Find k×3 nearest vectors | `faiss_manager.py:search()` |
| 6. Filter by domain | Remove non-matching domain tickets | `faiss_manager.py:search()` |
| 7. Hybrid re-rank | Apply priority + time weights | `pattern_recognition_agent.py` |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Vector Store: FAISS](#vector-store-faiss)
3. [Embedding Generation](#embedding-generation)
4. [Similarity Search](#similarity-search)
5. [Hybrid Scoring Algorithm](#hybrid-scoring-algorithm)
6. [Configurable Parameters](#configurable-parameters)
7. [API Endpoints](#api-endpoints)
8. [Data Flow Diagram](#data-flow-diagram)

---

## Architecture Overview

The retrieval system is a critical component of the Pattern Recognition Agent. It finds historically similar tickets to inform label assignment and resolution generation. The system uses a **two-stage approach**:

1. **Vector Similarity Search**: FAISS performs approximate nearest neighbor search on embedding vectors
2. **Hybrid Re-ranking**: Results are re-scored using a weighted combination of vector similarity and metadata relevance

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Input Ticket   │───▶│   Embedding     │───▶│  FAISS Search   │
│  (title + desc) │    │   Generation    │    │  (Top K × 3)    │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                       ┌─────────────────┐    ┌────────▼────────┐
                       │  Final Top K    │◀───│  Domain Filter  │
                       │    Results      │    │  + Hybrid Score │
                       └─────────────────┘    └─────────────────┘
```

---

## Vector Store: FAISS

FAISS (Facebook AI Similarity Search) is used for efficient vector similarity search. The system uses `IndexFlatIP` for exact inner product search, which becomes cosine similarity after L2 normalization.

### Index Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Index Type | `IndexFlatIP` | Exact inner product search |
| Dimension | 3072 | OpenAI text-embedding-3-large output dimension |
| Normalization | L2 | Converts inner product to cosine similarity |
| Storage | Disk-based | Separate files for index and metadata |

### FAISSManager Implementation

```python
# src/vectorstore/faiss_manager.py

class FAISSManager:
    """Manage FAISS index for ticket similarity search."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = Config.EMBEDDING_DIMENSIONS  # 3072

    def create_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """
        Create a new FAISS index from embeddings.
        Uses IndexFlatIP for cosine similarity search (inner product after normalization).
        """
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity using inner product
        faiss.normalize_L2(embeddings_array)

        # Create index - IndexFlatIP for exact inner product search
        self.index = faiss.IndexFlatIP(self.dimension)

        # Add vectors to index
        self.index.add(embeddings_array)

        # Store metadata separately (FAISS only stores vectors)
        self.metadata = metadata
```

### Why IndexFlatIP?

- **Exact search**: No approximation error, perfect for our scale (~1000s of tickets)
- **Cosine similarity**: After L2 normalization, inner product equals cosine similarity
- **Simplicity**: No training required, straightforward to update

### Persistence

The index and metadata are stored separately:

```python
def save(self, index_path: Path, metadata_path: Path):
    """Save FAISS index and metadata to disk."""
    # Save FAISS index (binary format)
    faiss.write_index(self.index, str(index_path))

    # Save metadata as JSON (ticket_id, title, labels, priority, etc.)
    with open(metadata_path, 'w') as f:
        json.dump(self.metadata, f, indent=2)

def load(self, index_path: Path, metadata_path: Path):
    """Load FAISS index and metadata from disk."""
    self.index = faiss.read_index(str(index_path))

    with open(metadata_path, 'r') as f:
        self.metadata = json.load(f)
```

**File Locations:**
- Index: `data/faiss_index/tickets.index`
- Metadata: `data/faiss_index/metadata.json`

---

## Embedding Generation

Embeddings are generated using OpenAI's `text-embedding-3-large` model, which produces 3072-dimensional vectors.

### EmbeddingGenerator Implementation

```python
# src/vectorstore/embedding_generator.py

class EmbeddingGenerator:
    """Generate embeddings for ticket text using OpenAI."""

    def __init__(self):
        self.client = get_openai_client()
        self.model = Config.EMBEDDING_MODEL  # "text-embedding-3-large"

    async def generate_ticket_embedding(self, title: str, description: str) -> List[float]:
        """
        Generate embedding for a ticket by combining title and description.
        """
        combined_text = combine_ticket_text(title, description)
        return await self.generate_embedding(combined_text)

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        cleaned_text = clean_text(text)
        return await self.client.generate_embedding(cleaned_text, model=self.model)
```

### Text Preprocessing

Before embedding, ticket text is combined and cleaned:

```python
# src/utils/helpers.py

def combine_ticket_text(title: str, description: str) -> str:
    """Combine title and description with separator."""
    return f"{title}\n\n{description}"

def clean_text(text: str) -> str:
    """Clean text for embedding: normalize whitespace, remove special chars."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    return text
```

### Batch Processing

For efficiency during index creation, embeddings are generated in parallel batches:

```python
async def generate_batch_embeddings(
    self, texts: List[str], batch_size: int = 10
) -> List[List[float]]:
    """Generate embeddings for multiple texts in batches."""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Process batch concurrently using asyncio.gather
        batch_embeddings = await asyncio.gather(
            *[self.generate_embedding(text) for text in batch]
        )

        embeddings.extend(batch_embeddings)

    return embeddings
```

---

## Similarity Search

The search process involves querying the FAISS index and optionally filtering by domain.

### Search Implementation

```python
# src/vectorstore/faiss_manager.py

def search(
    self,
    query_embedding: List[float],
    k: int = 20,
    domain_filter: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Search for similar tickets using FAISS.

    Args:
        query_embedding: Query vector (3072 dimensions)
        k: Number of results to return
        domain_filter: Optional domain to filter by ('MM', 'CIW', 'Specialty')

    Returns:
        Tuple of (similar_tickets, similarity_scores)
    """
    # Normalize query vector (required for cosine similarity)
    query_array = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_array)

    # Request more results if filtering by domain (some will be filtered out)
    search_k = k * 3 if domain_filter else k
    distances, indices = self.index.search(query_array, search_k)

    # Build results with domain filtering
    results = []
    scores = []

    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:  # FAISS returns -1 for padding
            continue

        ticket_data = self.metadata[idx].copy()

        # Apply domain filter if specified
        if domain_filter and ticket_data.get('domain') != domain_filter:
            continue

        ticket_data['similarity_score'] = float(distance)
        results.append(ticket_data)
        scores.append(float(distance))

        # Stop when we have enough results
        if len(results) >= k:
            break

    return results, scores
```

### Domain Filtering Strategy

When a domain filter is specified:

1. **Over-fetch**: Request `k × 3` results from FAISS
2. **Post-filter**: Remove tickets not matching the target domain
3. **Truncate**: Return only the top `k` results

This ensures we get enough domain-specific results even if the domain distribution is uneven.

---

## Hybrid Scoring Algorithm

The core innovation is the **hybrid scoring system** that combines semantic similarity with operational metadata.

### Formula

```
hybrid_score = (vector_weight × vector_similarity) + (metadata_weight × metadata_score)

where:
    metadata_score = (priority_score × 0.6) + (time_score × 0.4)
    time_score = max(0, 1 - (resolution_time / normalization_hours))
```

### The Problem Hybrid Scoring Solves

Consider two tickets returned from FAISS with **identical vector similarity** (72%):

| Ticket | Vector Similarity | Priority | Resolution Time |
|--------|------------------|----------|-----------------|
| JIRA-MM-009 | 72% | High | 4 hours |
| JIRA-MM-023 | 72% | Low | 150 hours |

**Pure vector search would rank them equally.** But operationally, JIRA-MM-009 is far more valuable:
- Higher priority means it was a more important issue
- 4-hour resolution means the solution was effective and quick

Hybrid scoring captures this operational context.

---

### Worked Example: Step-by-Step Calculation

**Configuration (using defaults):**
```python
vector_weight = 0.7           # 70% weight to semantic similarity
metadata_weight = 0.3         # 30% weight to operational factors
priority_weights = {
    "Critical": 1.0,
    "High": 0.8,
    "Medium": 0.5,
    "Low": 0.3
}
time_normalization_hours = 100  # Reference point for time scoring
```

---

#### Ticket A: JIRA-MM-009 (High priority, 4 hours resolution)

**Step 1: Vector similarity (from FAISS)**
```
vector_similarity = 0.72
```

**Step 2: Priority score (lookup from config)**
```
priority_score = priority_weights["High"] = 0.8
```

**Step 3: Time score (faster resolution = higher score)**
```
time_score = max(0, 1 - (resolution_time / normalization_hours))
time_score = max(0, 1 - (4 / 100))
time_score = max(0, 1 - 0.04)
time_score = 0.96  ← Fast resolution rewards with high score!
```

**Step 4: Metadata score (combine priority + time)**
```
metadata_score = (priority_score × 0.6) + (time_score × 0.4)
metadata_score = (0.8 × 0.6) + (0.96 × 0.4)
metadata_score = 0.48 + 0.384
metadata_score = 0.864
```

**Step 5: Final hybrid score**
```
hybrid_score = (vector_weight × vector_similarity) + (metadata_weight × metadata_score)
hybrid_score = (0.7 × 0.72) + (0.3 × 0.864)
hybrid_score = 0.504 + 0.2592
hybrid_score = 0.7632 → 76.3%
```

---

#### Ticket B: JIRA-MM-023 (Low priority, 150 hours resolution)

**Step 1: Vector similarity**
```
vector_similarity = 0.72  (identical to Ticket A)
```

**Step 2: Priority score**
```
priority_score = priority_weights["Low"] = 0.3
```

**Step 3: Time score**
```
time_score = max(0, 1 - (150 / 100))
time_score = max(0, 1 - 1.5)
time_score = max(0, -0.5)
time_score = 0  ← Exceeds normalization, clamped to zero!
```

**Step 4: Metadata score**
```
metadata_score = (0.3 × 0.6) + (0 × 0.4)
metadata_score = 0.18 + 0
metadata_score = 0.18
```

**Step 5: Final hybrid score**
```
hybrid_score = (0.7 × 0.72) + (0.3 × 0.18)
hybrid_score = 0.504 + 0.054
hybrid_score = 0.558 → 55.8%
```

---

### Result Comparison

| Ticket | Vector | Priority | Time | Metadata | **Hybrid** | Rank |
|--------|--------|----------|------|----------|------------|------|
| JIRA-MM-009 | 72% | 0.8 | 0.96 | 86.4% | **76.3%** | #1 |
| JIRA-MM-023 | 72% | 0.3 | 0.00 | 18.0% | **55.8%** | #2 |

**Impact: 20+ percentage point difference** despite identical semantic similarity!

### Visual Score Breakdown

```
JIRA-MM-009 (High priority, 4h resolution)
┌─────────────────────────────────────────────────────┐
│ HYBRID SCORE: 76.3%                                 │
├─────────────────────────────────────────────────────┤
│ ██████████████████████████████████░░░░░░░░░░░░░░░░░ │
│ Vector Contribution: 50.4%  (0.7 × 0.72)            │
├─────────────────────────────────────────────────────┤
│ █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Metadata Contribution: 25.9%  (0.3 × 0.864)         │
│   └─ Priority: 0.8 × 0.6 = 0.48                     │
│   └─ Time:     0.96 × 0.4 = 0.384                   │
└─────────────────────────────────────────────────────┘

JIRA-MM-023 (Low priority, 150h resolution)
┌─────────────────────────────────────────────────────┐
│ HYBRID SCORE: 55.8%                                 │
├─────────────────────────────────────────────────────┤
│ ██████████████████████████████████░░░░░░░░░░░░░░░░░ │
│ Vector Contribution: 50.4%  (0.7 × 0.72)            │
├─────────────────────────────────────────────────────┤
│ ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ Metadata Contribution: 5.4%  (0.3 × 0.18)           │
│   └─ Priority: 0.3 × 0.6 = 0.18                     │
│   └─ Time:     0.0 × 0.4 = 0.00                     │
└─────────────────────────────────────────────────────┘
```

### Edge Cases

| Scenario | Time Score | Explanation |
|----------|------------|-------------|
| Resolved in 0 hours | 1.0 | Perfect score (instant fix) |
| Resolved in 50 hours | 0.5 | Midpoint (50/100) |
| Resolved in 100 hours | 0.0 | At normalization threshold |
| Resolved in 150+ hours | 0.0 | Clamped to zero (max function) |

---

### Implementation

```python
# src/agents/pattern_recognition_agent.py

def apply_hybrid_scoring_with_config(
    self,
    similar_tickets: List[Dict[str, Any]],
    similarity_scores: List[float],
    config: RetrievalConfig
) -> List[Dict[str, Any]]:
    """
    Apply hybrid scoring with configurable weights.

    The hybrid scoring formula is:
        hybrid_score = (vector_weight * vector_similarity) + (metadata_weight * metadata_score)

    Where metadata_score = (priority_score * 0.6) + (time_score * 0.4)
    """
    # Build priority scores dict from config
    priority_scores = {
        'Critical': config.priority_weights.Critical,  # default: 1.0
        'High': config.priority_weights.High,          # default: 0.8
        'Medium': config.priority_weights.Medium,      # default: 0.5
        'Low': config.priority_weights.Low             # default: 0.3
    }

    # Calculate hybrid scores for each ticket
    for i, ticket in enumerate(similar_tickets):
        vector_score = similarity_scores[i]

        # Priority factor
        priority_score = priority_scores.get(ticket.get('priority', 'Medium'), 0.5)

        # Resolution time factor (faster = better, normalize to 0-1)
        res_time = ticket.get('resolution_time_hours', 24)
        time_score = max(0, 1 - (res_time / config.time_normalization_hours))

        # Combine metadata scores (60% priority, 40% time)
        metadata_score = (priority_score * 0.6) + (time_score * 0.4)

        # Hybrid score using configurable weights
        hybrid_score = (config.vector_weight * vector_score) + \
                       (config.metadata_weight * metadata_score)

        # Store all scores for transparency
        ticket['similarity_score'] = hybrid_score
        ticket['vector_similarity'] = vector_score
        ticket['metadata_score'] = metadata_score

    # Re-sort by hybrid score (descending)
    similar_tickets.sort(key=lambda x: x['similarity_score'], reverse=True)

    return similar_tickets
```

### Scoring Components Explained

#### 1. Vector Similarity (default weight: 70%)

- Raw cosine similarity from FAISS
- Measures **semantic similarity** between ticket texts
- Range: 0.0 to 1.0 (higher = more similar)

#### 2. Priority Score (60% of metadata)

| Priority | Default Score | Rationale |
|----------|---------------|-----------|
| Critical | 1.0 | Highest urgency, most valuable patterns |
| High | 0.8 | Important issues with proven solutions |
| Medium | 0.5 | Standard baseline |
| Low | 0.3 | Less critical, may have less refined solutions |

#### 3. Time Score (40% of metadata)

Rewards tickets that were resolved quickly:

```python
time_score = max(0, 1 - (resolution_time / normalization_hours))
```

Example with `normalization_hours = 100`:
- 0 hours → score 1.0 (instant resolution)
- 50 hours → score 0.5 (moderate)
- 100+ hours → score 0.0 (slow resolution)

### Why Hybrid Scoring?

Pure vector similarity can miss important operational context:

| Scenario | Vector Only | Hybrid |
|----------|-------------|--------|
| Two equally similar tickets, one Critical, one Low | Same rank | Critical ranked higher |
| Similar ticket resolved in 2 hours vs 200 hours | Same rank | Fast resolution ranked higher |
| Perfect semantic match but Low priority, slow fix | #1 result | May be outranked by relevant High priority tickets |

---

## Configurable Parameters

All retrieval parameters can be tuned through the UI or API.

### RetrievalConfig Schema

```python
# src/models/retrieval_config.py

class PriorityWeights(BaseModel):
    """Weights for priority-based scoring in hybrid retrieval."""
    Critical: float = Field(default=1.0, ge=0.0, le=1.0)
    High: float = Field(default=0.8, ge=0.0, le=1.0)
    Medium: float = Field(default=0.5, ge=0.0, le=1.0)
    Low: float = Field(default=0.3, ge=0.0, le=1.0)


class RetrievalConfig(BaseModel):
    """
    Configuration for retrieval parameters.

    Controls how the Pattern Recognition Agent searches for and scores similar tickets.
    """
    top_k: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of similar tickets to retrieve"
    )
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid scoring (0-1)"
    )
    metadata_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for metadata relevance in hybrid scoring (0-1)"
    )
    priority_weights: PriorityWeights = Field(
        default_factory=PriorityWeights,
        description="Score multipliers for each priority level"
    )
    time_normalization_hours: float = Field(
        default=100.0,
        ge=1.0,
        le=500.0,
        description="Reference hours for resolution time normalization"
    )
    domain_filter: Optional[str] = Field(
        default=None,
        description="Force specific domain filter (MM, CIW, Specialty) or None for auto"
    )
```

### Parameter Tuning Guide

| Parameter | Effect of Increasing | When to Adjust |
|-----------|---------------------|----------------|
| `top_k` | More tickets for context | Increase for complex issues needing more examples |
| `vector_weight` | Prioritize semantic similarity | When ticket content is highly specialized |
| `metadata_weight` | Prioritize operational factors | When priority/resolution time are strong signals |
| `priority_weights.Critical` | Boost Critical tickets | When Critical tickets have best solutions |
| `time_normalization_hours` | Extend time scoring range | When your tickets have longer resolution times |

### Configuration Persistence

Configurations are saved to `config/search_config.json`:

```json
{
  "top_k": 20,
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

---

## API Endpoints

### Preview Search

Test retrieval with custom configuration before processing.

```http
POST /api/preview-search
Content-Type: application/json

{
  "title": "MM_ALDER service connection timeout",
  "description": "Service experiencing intermittent connection timeouts...",
  "config": {
    "top_k": 20,
    "vector_weight": 0.7,
    "metadata_weight": 0.3,
    "domain_filter": null
  }
}
```

**Response:**

```json
{
  "similar_tickets": [
    {
      "ticket_id": "JIRA-MM-009",
      "title": "MM service database connection pool exhaustion",
      "similarity_score": 0.763,
      "vector_similarity": 0.72,
      "metadata_score": 0.87,
      "priority": "High",
      "labels": ["Configuration Fix", "#MM_ALDER"],
      "resolution_time_hours": 4.5,
      "domain": "MM"
    }
  ],
  "search_metadata": {
    "query_domain": "MM",
    "total_found": 20,
    "avg_similarity": 0.675,
    "top_similarity": 0.763,
    "classification_confidence": 1.0
  },
  "config_used": { ... }
}
```

### Save Configuration

Persist configuration for use in pipeline processing.

```http
POST /api/save-search-config
Content-Type: application/json

{
  "top_k": 25,
  "vector_weight": 0.6,
  "metadata_weight": 0.4,
  ...
}
```

### Load Configuration

Retrieve saved configuration.

```http
GET /api/load-search-config
```

---

## Data Flow Diagram

### Complete Retrieval Flow

```
                          ┌──────────────────────────────────────────┐
                          │           User Interface                  │
                          │  ┌────────────────────────────────────┐  │
                          │  │     Search Tuning Panel            │  │
                          │  │  • top_k slider (5-50)             │  │
                          │  │  • vector_weight slider (0-100%)   │  │
                          │  │  • domain filter dropdown          │  │
                          │  │  • priority weights (advanced)     │  │
                          │  └────────────────────────────────────┘  │
                          └────────────────────┬─────────────────────┘
                                               │
                                               ▼
                          ┌──────────────────────────────────────────┐
                          │        API: /api/preview-search          │
                          │  Input: title, description, config       │
                          └────────────────────┬─────────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          │
     ┌──────────────────────┐   ┌──────────────────────┐                  │
     │ Domain Classification │   │  Embedding Generator │                  │
     │  (if auto-detect)     │   │  text-embedding-3-   │                  │
     │                       │   │  large (3072-dim)    │                  │
     └───────────┬───────────┘   └───────────┬──────────┘                  │
                 │                           │                             │
                 │              ┌────────────▼────────────┐                │
                 │              │      FAISS Manager      │                │
                 │              │  ┌──────────────────┐   │                │
                 │              │  │ 1. L2 Normalize  │   │                │
                 │              │  │ 2. IndexFlatIP   │   │                │
                 │              │  │    search(k×3)   │   │                │
                 │              │  │ 3. Get metadata  │   │                │
                 │              │  └──────────────────┘   │                │
                 │              └────────────┬────────────┘                │
                 │                           │                             │
                 └───────────────┬───────────┘                             │
                                 │                                         │
                                 ▼                                         │
                  ┌──────────────────────────────┐                         │
                  │       Domain Filtering        │                        │
                  │  Remove non-matching domain   │                        │
                  │  tickets from results         │                        │
                  └──────────────┬───────────────┘                         │
                                 │                                         │
                                 ▼                                         │
                  ┌──────────────────────────────┐                         │
                  │      Hybrid Scoring           │                        │
                  │  ┌────────────────────────┐   │                        │
                  │  │ For each ticket:       │   │                        │
                  │  │                        │   │                        │
                  │  │ priority_score =       │   │                        │
                  │  │   weights[priority]    │   │                        │
                  │  │                        │   │                        │
                  │  │ time_score =           │   │                        │
                  │  │   1 - (time/norm)      │   │                        │
                  │  │                        │   │                        │
                  │  │ metadata =             │   │                        │
                  │  │   0.6×priority +       │   │                        │
                  │  │   0.4×time             │   │                        │
                  │  │                        │   │                        │
                  │  │ hybrid =               │   │                        │
                  │  │   w_vec×vector +       │   │                        │
                  │  │   w_meta×metadata      │   │                        │
                  │  └────────────────────────┘   │                        │
                  └──────────────┬───────────────┘                         │
                                 │                                         │
                                 ▼                                         │
                  ┌──────────────────────────────┐                         │
                  │     Sort by Hybrid Score      │                        │
                  │     Return Top K Results      │◀────────────────────────┘
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │       Response Payload        │
                  │  • similar_tickets[]          │
                  │  • search_metadata            │
                  │  • config_used                │
                  └──────────────────────────────┘
```

### Integration with Pipeline

When processing a ticket through the full pipeline, saved configuration is automatically loaded:

```python
# api_server.py - Pipeline integration

async def stream_agent_updates(ticket_data: dict, ...):
    # Load saved search config if it exists
    search_config = None
    config_path = Path("config/search_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            search_config = json.load(f)

    # Create initial state with search config
    initial_state = {
        "ticket_id": ticket_data.get("ticket_id"),
        "title": title,
        "description": description,
        "priority": ticket_data.get("priority", "Medium"),
        "metadata": ticket_data.get("metadata", {}),
        "search_config": search_config,  # <-- Passed to Pattern Recognition Agent
        "processing_stage": "Starting pipeline",
        "status": "processing",
        "messages": []
    }
```

The Pattern Recognition Agent then uses this config:

```python
# src/agents/pattern_recognition_agent.py

async def __call__(self, state: TicketState) -> AgentOutput:
    # Check if custom search config is provided in state
    search_config_dict = state.get('search_config')

    if search_config_dict:
        # Use custom config from UI tuning
        config = RetrievalConfig(**search_config_dict)
        similar_tickets, similarity_scores = await self.find_similar_with_config(
            title, description, domain, config
        )
        similar_tickets = self.apply_hybrid_scoring_with_config(
            similar_tickets, similarity_scores, config
        )
    else:
        # Use default config
        similar_tickets, similarity_scores = await self.find_similar_tickets(
            title, description, domain
        )
        similar_tickets = self.apply_hybrid_scoring(similar_tickets, similarity_scores)
```

---

## Performance Characteristics

| Operation | Typical Duration | Notes |
|-----------|------------------|-------|
| Embedding generation | 200-500ms | Single API call to OpenAI |
| FAISS search | < 1ms | In-memory, exact search |
| Domain filtering | < 1ms | Simple string comparison |
| Hybrid scoring | < 1ms | Pure computation |
| **Total retrieval** | **~500ms** | Dominated by embedding API call |

### Memory Usage

- FAISS index: ~12 MB per 1000 tickets (3072 dimensions × 4 bytes × 1000)
- Metadata: ~1-5 MB per 1000 tickets (depending on content)

---

## Troubleshooting

### "FAISS index not found"

```bash
# Rebuild the index from CSV data
python3 scripts/setup_vectorstore.py
```

### Low similarity scores

1. Check domain filter matches ticket content
2. Verify historical data covers similar scenarios
3. Try increasing `top_k` to see more results

### Results not matching expectations

1. Use the Search Tuning panel to preview results
2. Adjust `vector_weight` vs `metadata_weight`
3. Check priority weights if operational factors should matter more

---

## References

- [FAISS Documentation](https://faiss.ai/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- Source code: `src/vectorstore/`, `src/agents/pattern_recognition_agent.py`
