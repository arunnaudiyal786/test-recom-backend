# 🎫 **Complete Codebase Analysis: Intelligent Ticket Management System**

## **Table of Contents**
1. [System Overview](#system-overview)
2. [Backend Architecture](#backend-architecture)
3. [Frontend Architecture](#frontend-architecture)
4. [Data Flow Explanation](#data-flow)
5. [Code Examples for Developers](#code-examples)

---

## **System Overview**

### **What Does This System Do?**

Imagine you're a support engineer receiving hundreds of technical tickets daily. This system **automatically**:
1. **Classifies** the ticket into the right domain (MM, CIW, or Specialty)
2. **Finds** similar tickets from history using AI
3. **Assigns** appropriate labels
4. **Generates** a complete step-by-step resolution plan

**Think of it as**: A smart assistant that reads a ticket, remembers similar problems from the past, and suggests how to fix it - all in 8-12 seconds.

---

`★ Insight ─────────────────────────────────────`
**Why This Architecture?**
- **Sequential Pipeline**: Each agent builds on the previous one's output (classification → pattern matching → labeling → resolution)
- **Vector Search**: Uses FAISS (Facebook AI Similarity Search) to find similar tickets by *meaning*, not just keywords
- **Multi-Agent Design**: Each agent is a specialist - like having 4 experts collaborate on one ticket
`─────────────────────────────────────────────────`

---

## **Backend Architecture**

### **1. Entry Point: `main.py`**

This is where everything starts. Let me explain it step-by-step:

```python
# main.py:141-183
async def process_ticket(input_file: Path, output_file: Path):
    """Main orchestrator function"""

    # Step 1: Load the ticket from JSON file
    ticket = load_input_ticket(input_file)
    # Example: {"ticket_id": "JIRA-001", "title": "DB timeout", ...}

    # Step 2: Create initial state (this is the "memory" that flows through agents)
    initial_state = prepare_initial_state(ticket)
    # State = {"ticket_id": ..., "title": ..., "status": "processing", ...}

    # Step 3: Get the LangGraph workflow (the 4-agent pipeline)
    workflow = get_workflow()

    # Step 4: Execute the workflow asynchronously
    final_state = await workflow.ainvoke(initial_state)
    # This runs: Classification → Pattern Recognition → Labeling → Resolution

    # Step 5: Format and save the output
    output = format_final_output(final_state)
    save_output(output, output_file)
```

**Key Concept**: The `state` is like a shared notebook that each agent reads from and writes to. Each agent adds their findings to this notebook.

---

### **2. The LangGraph Workflow: `src/orchestrator/workflow.py`**

LangGraph is a framework for building **sequential workflows** where each step depends on the previous one.

```python
# src/orchestrator/workflow.py
def build_workflow() -> StateGraph:
    """Build the multi-agent pipeline"""

    # Create a workflow graph with TicketWorkflowState as the schema
    workflow = StateGraph(TicketWorkflowState)

    # Add agent nodes (think of these as processing stations)
    # Note: Classification is optional (controlled by SKIP_DOMAIN_CLASSIFICATION)
    if not SKIP_DOMAIN_CLASSIFICATION:
        workflow.add_node("Domain Classification Agent", classification_node)
    workflow.add_node("Pattern Recognition Agent", retrieval_node)
    workflow.add_node("Label Assignment Agent", labeling_node)
    workflow.add_node("Novelty Detection Agent", novelty_node)
    workflow.add_node("Resolution Generation Agent", resolution_node)
    workflow.add_node("Error Handler", error_handler_node)

    # Set the starting point (skips classification by default)
    if SKIP_DOMAIN_CLASSIFICATION:
        workflow.set_entry_point("Pattern Recognition Agent")
    else:
        workflow.set_entry_point("Domain Classification Agent")

    # Define the flow with conditional edges
    # Each checks if the previous agent succeeded or failed
    workflow.add_conditional_edges(
        "Pattern Recognition Agent",
        route_after_retrieval,
        {"labeling": "Label Assignment Agent", "error_handler": "Error Handler"}
    )

    # ... similar edges for Label → Novelty → Resolution ...

    return workflow.compile()
```

**Analogy**: Think of this as an assembly line (default configuration):
- **Station 1**: Pattern Recognition (finds similar tickets via FAISS)
- **Station 2**: Label Assignment (adds category, business, and technical labels)
- **Station 3**: Novelty Detection (checks if ticket is truly novel)
- **Station 4**: Resolution Generation (creates fix plan)
- If any station fails, the item goes to **Error Handler** (manual review)

**Note**: Domain Classification Agent can be enabled as Station 0 by setting `SKIP_DOMAIN_CLASSIFICATION = False`.

---

### **3. State Management: `src/models/state_schema.py`**

The state is a **TypedDict** (not a class) that holds all ticket information as it flows through agents.

```python
# src/models/state_schema.py:8-57
class TicketState(TypedDict, total=False):
    """
    The shared state that flows through all 4 agents.
    total=False means agents can update only their fields.
    """

    # INPUT FIELDS (from user)
    ticket_id: str
    title: str
    description: str
    priority: str
    metadata: Dict

    # AGENT 1 OUTPUT (Classification Agent)
    classified_domain: Optional[str]  # "MM", "CIW", or "Specialty"
    classification_confidence: Optional[float]  # 0.0 to 1.0
    classification_reasoning: Optional[str]
    extracted_keywords: Optional[List[str]]

    # AGENT 2 OUTPUT (Pattern Recognition Agent)
    similar_tickets: Optional[List[Dict]]  # Top 20 similar tickets
    similarity_scores: Optional[List[float]]
    search_metadata: Optional[Dict]

    # AGENT 3 OUTPUT (Label Assignment Agent)
    assigned_labels: Optional[List[str]]  # ["Code Fix", "#MM_ALDER"]
    label_confidence: Optional[Dict[str, float]]

    # AGENT 4 OUTPUT (Resolution Generation Agent)
    resolution_plan: Optional[Dict]  # Complete fix plan with steps
    resolution_confidence: Optional[float]

    # WORKFLOW CONTROL
    status: Literal["processing", "success", "error", "failed"]
    error_message: Optional[str]
    current_agent: str

    # AUDIT TRAIL (accumulates across agents)
    messages: Annotated[List[Dict], operator.add]
```

**Key Insight**: `total=False` allows **partial updates**. Each agent only sets their fields, not the entire state.

```python
# Example: Classification Agent returns only what it computed
return {
    "classified_domain": "MM",
    "classification_confidence": 0.92,
    "status": "success",
    "current_agent": "Domain Classification Agent"
}
# Other fields remain unchanged!
```

---

`★ Insight ─────────────────────────────────────`
**Why TypedDict instead of Pydantic Models?**
- **LangGraph Requirement**: LangGraph state must be TypedDict for efficient merging
- **Partial Updates**: Agents return small dicts, not full state objects
- **Memory Efficiency**: Only changed fields are transmitted between agents
`─────────────────────────────────────────────────`

---

### **4. Agent 1: Classification Agent**

This agent determines which domain the ticket belongs to.

```python
# src/agents/classification_agent.py:125-183
async def __call__(self, state: TicketState) -> AgentOutput:
    """
    Main execution function - LangGraph calls this automatically
    """

    title = state['title']
    description = state['description']

    # STEP 1: Run 3 binary classifiers in parallel
    # Instead of asking "Which domain?", we ask 3 yes/no questions:
    # - Is this MM? (Yes/No)
    # - Is this CIW? (Yes/No)
    # - Is this Specialty? (Yes/No)
    classifications = await self.classify_all_domains(title, description)
    # Returns: {"MM": {...}, "CIW": {...}, "Specialty": {...}}

    # STEP 2: Pick the domain with highest confidence
    domain, confidence, reasoning = self.determine_final_domain(classifications)
    # Example: domain="MM", confidence=0.92

    # STEP 3: Return partial state update
    return {
        "classified_domain": domain,
        "classification_confidence": confidence,
        "classification_reasoning": reasoning,
        "status": "success",
        "current_agent": "Domain Classification Agent",
        "messages": [{"role": "assistant", "content": f"Classified as {domain}"}]
    }
```

**The Binary Classifier Approach**:
```python
# src/agents/classification_agent.py:63-86
async def classify_all_domains(self, title, description):
    """Run 3 classifiers in parallel using asyncio.gather"""

    # Launch 3 async calls simultaneously
    results = await asyncio.gather(
        self.classify_domain('MM', title, description),
        self.classify_domain('CIW', title, description),
        self.classify_domain('Specialty', title, description)
    )

    return {
        'MM': results[0],       # {decision: True, confidence: 0.92, ...}
        'CIW': results[1],      # {decision: False, confidence: 0.3, ...}
        'Specialty': results[2] # {decision: False, confidence: 0.25, ...}
    }
```

**Why 3 Binary Classifiers Instead of 1 Multi-Class?**
- **Better Accuracy**: Each classifier specializes in detecting one domain
- **Confidence Scores**: We get separate confidence for each domain
- **Parallelization**: All 3 run simultaneously (faster!)

---

### **5. Agent 2: Pattern Recognition Agent**

This agent uses **FAISS vector search** to find similar historical tickets.

```python
# src/agents/pattern_recognition_agent.py:128-193
async def __call__(self, state: TicketState) -> AgentOutput:
    """Find similar tickets using AI embeddings"""

    title = state['title']
    description = state['description']
    domain = state.get('classified_domain')  # From Agent 1

    # STEP 1: Generate embedding for current ticket
    # This converts text into a 3072-dimensional vector
    query_embedding = await self.embedding_generator.generate_ticket_embedding(
        title, description
    )
    # Example: [0.012, -0.043, 0.091, ...] (3072 numbers)

    # STEP 2: Search FAISS index for similar vectors in same domain
    similar_tickets, scores = self.faiss_manager.search(
        query_embedding=query_embedding,
        k=20,  # Get top 20
        domain_filter=domain  # Only search MM tickets if classified as MM
    )

    # STEP 3: Apply hybrid scoring (70% vector + 30% metadata)
    similar_tickets = self.apply_hybrid_scoring(similar_tickets, scores)

    # STEP 4: Return results
    return {
        "similar_tickets": similar_tickets,
        "similarity_scores": [t['similarity_score'] for t in similar_tickets],
        "search_metadata": {...},
        "status": "success"
    }
```

**What is Vector Search?**

Imagine tickets as points in 3D space (actually 3072D). Similar tickets cluster together:

```
         MM Domain Tickets
              •
         •  Current    •
            • Ticket
              •

         CIW Domain      Specialty Domain
           •                  •
         •   •              •   •
```

FAISS finds the 20 nearest neighbors (most similar tickets).

---

### **6. FAISS Vector Store: `src/vectorstore/faiss_manager.py`**

FAISS (Facebook AI Similarity Search) is a library for fast vector search.

```python
# src/vectorstore/faiss_manager.py:104-155
def search(self, query_embedding, k=20, domain_filter=None):
    """
    Search for similar tickets using cosine similarity
    """

    # STEP 1: Normalize the query vector for cosine similarity
    query_array = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_array)  # Unit length = 1

    # STEP 2: Search the index
    # If filtering by domain, request 3x more results (we'll filter later)
    search_k = k * 3 if domain_filter else k
    distances, indices = self.index.search(query_array, search_k)
    # distances = similarity scores (higher = more similar)
    # indices = positions in the metadata array

    # STEP 3: Filter by domain and collect results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx == -1:  # Padding value
            continue

        ticket_data = self.metadata[idx].copy()

        # Apply domain filter
        if domain_filter and ticket_data.get('domain') != domain_filter:
            continue  # Skip tickets from other domains

        ticket_data['similarity_score'] = float(distance)
        results.append(ticket_data)

        if len(results) >= k:  # Stop at 20 results
            break

    return results, scores
```

**How FAISS Index is Built**:
```python
# src/vectorstore/faiss_manager.py:20-50
def create_index(self, embeddings, metadata):
    """Build the searchable index from historical tickets"""

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)

    # Create index (IndexFlatIP = exact inner product search)
    self.index = faiss.IndexFlatIP(3072)  # 3072 dimensions

    # Add all vectors to index
    self.index.add(embeddings_array)

    # Store metadata separately
    self.metadata = metadata
```

---

`★ Insight ─────────────────────────────────────`
**Why Cosine Similarity Instead of Euclidean Distance?**
- **Direction Matters More Than Magnitude**: Two tickets about "database timeout" should match even if one is longer
- **Normalized Vectors**: L2 normalization ensures all vectors have length 1
- **Inner Product After Normalization = Cosine Similarity**: FAISS uses IndexFlatIP (Inner Product) on normalized vectors
`─────────────────────────────────────────────────`

---

### **7. Agent 3: Label Assignment Agent**

This agent assigns labels based on patterns in similar tickets.

```python
# src/agents/label_assignment_agent.py:125-195
async def __call__(self, state: TicketState) -> AgentOutput:
    """Assign labels based on historical patterns"""

    similar_tickets = state.get('similar_tickets', [])

    # STEP 1: Extract candidate labels from similar tickets
    candidate_labels = self.extract_candidate_labels(similar_tickets)
    # Example: ["Code Fix", "#MM_ALDER", "Configuration Fix", "Performance"]

    # STEP 2: Calculate label frequency in similar tickets
    label_distribution = self.calculate_label_distribution(
        candidate_labels, similar_tickets
    )
    # Example: {"Code Fix": "14/20", "#MM_ALDER": "18/20"}

    # STEP 3: Run binary classifier for each label in parallel
    label_decisions = await asyncio.gather(
        *[self.evaluate_label(label, state, label_distribution)
          for label in candidate_labels]
    )
    # Each call asks: "Should this ticket have label X?" (Yes/No + confidence)

    # STEP 4: Filter labels above confidence threshold (0.7)
    final_labels = []
    label_confidences = {}
    for label, result in zip(candidate_labels, label_decisions):
        if result['should_assign'] and result['confidence'] >= 0.7:
            final_labels.append(label)
            label_confidences[label] = result['confidence']

    return {
        "assigned_labels": final_labels,
        "label_confidence": label_confidences,
        "label_distribution": label_distribution,
        "status": "success"
    }
```

**Label Frequency Context**:
```python
# If 18 out of 20 similar tickets have label "#MM_ALDER"
# The LLM prompt includes this context:
"Historical frequency: 18/20 similar tickets have this label"
# This helps the model decide if the current ticket should have it too
```

---

### **8. Agent 4: Resolution Generation Agent**

This agent creates a detailed fix plan with steps, commands, and time estimates.

```python
# src/agents/resolution_generation_agent.py:80-150
async def __call__(self, state: TicketState) -> AgentOutput:
    """Generate resolution plan using Chain-of-Thought"""

    # STEP 1: Build context from previous agents' outputs
    ticket_info = {
        "title": state['title'],
        "description": state['description'],
        "domain": state.get('classified_domain'),
        "labels": state.get('assigned_labels', []),
        "similar_tickets": state.get('similar_tickets', [])[:5]  # Top 5
    }

    # STEP 2: Create prompt with CoT (Chain-of-Thought) format
    prompt = create_resolution_prompt(ticket_info)
    # Prompt structure:
    # "Analyze the ticket, review similar cases, synthesize patterns,
    #  create diagnostic steps, create resolution steps, estimate time..."

    # STEP 3: Call OpenAI with JSON mode
    response = await self.client.chat_completion_json(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",  # More powerful model for complex reasoning
        temperature=0.6,  # Balanced creativity
        max_tokens=8000   # Allow long responses
    )

    # STEP 4: Validate and return
    resolution_plan = response  # Already in JSON format

    return {
        "resolution_plan": resolution_plan,
        "resolution_confidence": resolution_plan.get('confidence', 0.0),
        "status": "success"
    }
```

**Example Resolution Plan Output**:
```json
{
  "summary": "Increase database connection pool size and add monitoring",
  "diagnostic_steps": [
    {
      "step_number": 1,
      "description": "Check current connection pool settings",
      "commands": ["cat /etc/db/config.yaml | grep pool_size"],
      "estimated_time_minutes": 5
    }
  ],
  "resolution_steps": [
    {
      "step_number": 1,
      "description": "Increase connection pool from 100 to 200",
      "commands": ["sed -i 's/pool_size: 100/pool_size: 200/' /etc/db/config.yaml"],
      "validation": "Verify pool_size=200 in config",
      "estimated_time_minutes": 10,
      "risk_level": "low",
      "rollback_procedure": "Revert to pool_size=100 if issues occur"
    }
  ],
  "total_estimated_time_hours": 2.0,
  "confidence": 0.88
}
```

---

## **Frontend Architecture**

### **Tech Stack**
- **Framework**: Next.js 15 (React 19 with App Router)
- **UI Library**: shadcn/ui (Tailwind CSS components)
- **Styling**: Tailwind CSS
- **Communication**: Server-Sent Events (SSE) for real-time streaming

---

### **1. Main Page: `frontend/app/pattern-recognition/page.tsx`**

This is the UI that users interact with.

```typescript
// frontend/app/pattern-recognition/page.tsx:85-230
const handleSubmit = async (ticket: TicketData) => {
  // STEP 1: Reset all agent states to idle
  setIsProcessing(true)
  Object.keys(agents).forEach((key) => {
    updateAgentState(key, { status: "idle", progress: 0, ... })
  })

  // STEP 2: Call Python FastAPI backend with SSE streaming
  const response = await fetch("http://localhost:8000/api/process-ticket", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(ticket)
  })

  // STEP 3: Process the SSE stream
  const reader = response.body?.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    const chunk = decoder.decode(value)
    const lines = chunk.split("\n\n")

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = JSON.parse(line.slice(6))

        // Update agent UI based on event
        if (data.agent) {
          if (data.status === "processing") {
            updateAgentState(data.agent, {
              status: "processing",
              progress: data.progress
            })
          } else if (data.status === "complete") {
            updateAgentState(data.agent, {
              status: "complete",
              progress: 100,
              output: formatAgentOutput(data.agent, data.data)
            })
          }
        }
      }
    }
  }
}
```

**What is Server-Sent Events (SSE)?**

SSE allows the server to push updates to the client in real-time:

```
Client                          Server
  |                               |
  |--- POST /api/process-ticket --|
  |                               |
  |<-- data: {agent: "classification", status: "processing"} --|
  |<-- data: {agent: "classification", status: "complete"} ----|
  |<-- data: {agent: "patternRecognition", status: "processing"} --|
  |<-- data: {agent: "patternRecognition", status: "complete"} ----|
  |<-- data: {status: "workflow_complete"} ----|
```

---

### **2. Agent Card Component: `frontend/components/agent-card.tsx`**

This displays the status of each agent.

```typescript
// frontend/components/agent-card.tsx:63-190
export function AgentCard({
  name,          // "Domain Classification Agent"
  description,   // "Classifies tickets into MM, CIW, or Specialty"
  icon,          // <Brain /> icon
  status,        // "idle" | "processing" | "streaming" | "complete" | "error"
  progress,      // 0-100
  output,        // Final output text
  streamingText, // Text being streamed in real-time
}) {
  return (
    <Card className={statusColor}>
      <CardHeader>
        <div className="flex items-center gap-3">
          {icon}
          <div>
            <CardTitle>{name}</CardTitle>
            <p>{description}</p>
          </div>
        </div>
        <Badge>{status}</Badge>
      </CardHeader>

      <CardContent>
        {/* Progress Bar */}
        {status === "processing" && <Progress value={progress} />}

        {/* Streaming Text with Typing Effect */}
        {status === "streaming" && (
          <span className="animate-pulse">{streamingText}</span>
        )}

        {/* Final Output */}
        {status === "complete" && <pre>{output}</pre>}
      </CardContent>
    </Card>
  )
}
```

---

### **3. FastAPI Backend: `api_server.py`**

This Python server connects the Next.js frontend to the LangGraph backend.

```python
# api_server.py:57-187
async def stream_agent_updates(ticket: TicketInput):
    """
    Stream agent progress updates as SSE events
    """

    # Get LangGraph workflow
    workflow = get_workflow()

    # Create initial state
    initial_state = {
        "ticket_id": ticket.ticket_id,
        "title": ticket.title,
        ...
    }

    # Process through workflow with streaming
    async for event in workflow.astream(initial_state):
        # LangGraph yields events as {node_name: state}
        for node_name, node_state in event.items():
            current_agent = node_state.get("current_agent")
            status = node_state.get("status")

            # Send SSE event to frontend
            if status == "processing":
                update = AgentUpdate(
                    agent=_agent_key(current_agent),
                    status="processing",
                    message=f"Starting {current_agent}...",
                    progress=0
                )
                yield f"data: {update.model_dump_json()}\n\n"

            elif status == "success":
                agent_data = _extract_agent_data(current_agent, node_state)
                update = AgentUpdate(
                    agent=_agent_key(current_agent),
                    status="complete",
                    data=agent_data,
                    progress=100
                )
                yield f"data: {update.model_dump_json()}\n\n"

    # Send completion event
    yield f'data: {{"status": "workflow_complete"}}\n\n'
```

**LangGraph Streaming**:
```python
# workflow.astream() yields events as they happen:
async for event in workflow.astream(initial_state):
    # Event structure: {"Domain Classification Agent": {...state...}}
    # We extract the state and send it to the frontend as SSE
```

---

## **Data Flow Explanation**

Let me trace a complete ticket from submission to resolution:

### **Step-by-Step Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER SUBMITS TICKET via Frontend                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
{
  "ticket_id": "JIRA-NEW-001",
  "title": "MM_ALDER database timeout",
  "description": "Connection pool exhausted during peak hours...",
  "priority": "High"
}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Next.js sends to FastAPI Backend                        │
│    POST http://localhost:8000/api/process-ticket            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FastAPI calls LangGraph Workflow                         │
│    workflow.astream(initial_state)                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. AGENT 1: Classification Agent                            │
│    - Runs 3 binary classifiers in parallel                  │
│    - Result: domain="MM", confidence=0.92                   │
│    - Updates state with classification                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
        SSE Event to Frontend ──→ data: {"agent": "classification", "status": "complete", "data": {...}}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. AGENT 2: Pattern Recognition Agent                       │
│    - Generates embedding for ticket (3072D vector)          │
│    - Searches FAISS index for similar MM tickets            │
│    - Applies hybrid scoring (70% vector + 30% metadata)     │
│    - Result: 20 similar tickets with scores                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
        SSE Event to Frontend ──→ data: {"agent": "patternRecognition", "status": "complete", ...}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. AGENT 2: Label Assignment Agent                          │
│    - Classifies into categories from predefined taxonomy    │
│    - Generates business-oriented labels (AI)                │
│    - Generates technical labels (AI)                        │
│    - All 3 run in PARALLEL via asyncio.gather               │
│    - Result: [CAT], [BIZ], [TECH] prefixed labels           │
└─────────────────────────────────────────────────────────────┘
                            ↓
        SSE Event to Frontend ──→ data: {"agent": "labelAssignment", "status": "complete", ...}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. AGENT 3: Novelty Detection Agent                         │
│    - Analyzes if ticket is truly novel (no LLM calls)       │
│    - Signal 1: Max confidence check (40% weight)            │
│    - Signal 2: Entropy analysis (30% weight)                │
│    - Signal 3: Centroid distance (30% weight)               │
│    - Result: novelty_detected, novelty_score, recommendation│
└─────────────────────────────────────────────────────────────┘
                            ↓
        SSE Event to Frontend ──→ data: {"agent": "noveltyDetection", "status": "complete", ...}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. AGENT 4: Resolution Generation Agent                     │
│    - Builds context from all previous agents                │
│    - Calls GPT-4 with Chain-of-Thought prompt               │
│    - Generates diagnostic + resolution steps                │
│    - Result: Complete resolution plan with time estimates   │
└─────────────────────────────────────────────────────────────┘
                            ↓
        SSE Event to Frontend ──→ data: {"agent": "resolutionGeneration", "status": "complete", ...}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. Workflow Complete                                        │
│    - FastAPI saves final state to output/ticket_resolution.json │
│    - Sends completion event to frontend                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
        SSE Event ──→ data: {"status": "workflow_complete", "output_available": true}
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. Frontend fetches final output                            │
│    GET http://localhost:8000/api/output                     │
│    - Displays JSON with Download button                     │
└─────────────────────────────────────────────────────────────┘
```

---

## **Code Examples for Developers**

### **Example 1: Adding a New Domain**

Let's say you want to add a "Security" domain:

**Step 1**: Add binary classifier prompt
```python
# src/prompts/classification_prompts.py
SECURITY_CLASSIFIER_PROMPT = """
Is this ticket related to Security domain?

Security domain characteristics:
- Authentication/authorization issues
- Vulnerability reports
- Security patch requests
- Access control problems
- Encryption concerns

Ticket: {title}
Description: {description}

Return JSON:
{{
  "decision": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "explanation",
  "extracted_keywords": ["keyword1", "keyword2"]
}}
"""
```

**Step 2**: Update Classification Agent
```python
# src/agents/classification_agent.py:22
def __init__(self):
    self.domains = ['MM', 'CIW', 'Specialty', 'Security']  # Add Security

# Update classify_all_domains method:
async def classify_all_domains(self, title, description):
    results = await asyncio.gather(
        self.classify_domain('MM', title, description),
        self.classify_domain('CIW', title, description),
        self.classify_domain('Specialty', title, description),
        self.classify_domain('Security', title, description)  # Add this
    )
    return {
        'MM': results[0],
        'CIW': results[1],
        'Specialty': results[2],
        'Security': results[3]  # Add this
    }
```

**Step 3**: Generate historical Security tickets
```python
# Add to data/raw/historical_tickets.csv with keys like JIRA-SE-001
```

**Step 4**: Rebuild FAISS index
```bash
python3 scripts/setup_vectorstore.py
```

---

### **Example 2: Customizing Hybrid Scoring**

Want to prioritize recent tickets over old ones?

```python
# src/agents/pattern_recognition_agent.py:73-126
def apply_hybrid_scoring(self, similar_tickets, similarity_scores):
    """Modified to include recency factor"""

    from datetime import datetime

    for i, ticket in enumerate(similar_tickets):
        vector_score = similarity_scores[i]

        # Priority factor
        priority_score = priority_scores.get(ticket.get('priority', 'Medium'), 0.5)

        # Resolution time factor
        res_time = ticket.get('resolution_time_hours', 24)
        time_score = max(0, 1 - (res_time / 100))

        # NEW: Recency factor (tickets from last 30 days score higher)
        created_date = datetime.fromisoformat(ticket.get('created_date', '2020-01-01'))
        days_ago = (datetime.now() - created_date).days
        recency_score = max(0, 1 - (days_ago / 365))  # Decay over 1 year

        # Combine: 60% vector + 20% priority + 10% time + 10% recency
        hybrid_score = (
            (0.6 * vector_score) +
            (0.2 * priority_score) +
            (0.1 * time_score) +
            (0.1 * recency_score)  # NEW
        )

        ticket['similarity_score'] = hybrid_score

    return sorted(similar_tickets, key=lambda x: x['similarity_score'], reverse=True)
```

---

### **Example 3: Adding Real-Time Updates to Frontend**

Want to show intermediate LLM responses?

**Backend** (`api_server.py`):
```python
async def stream_agent_updates(ticket: TicketInput):
    # Track intermediate messages
    previous_message_count = 0

    async for event in workflow.astream(initial_state):
        for node_name, node_state in event.items:
            # NEW: Stream intermediate messages
            current_messages = node_state.get("messages", [])
            if len(current_messages) > previous_message_count:
                new_messages = current_messages[previous_message_count:]
                previous_message_count = len(current_messages)

                for msg in new_messages:
                    if msg.get("role") == "assistant":
                        update = AgentUpdate(
                            agent=_agent_key(current_agent),
                            status="streaming",  # NEW status
                            message=msg.get("content"),
                            progress=50
                        )
                        yield f"data: {update.model_dump_json()}\n\n"
                        await asyncio.sleep(0.05)  # Small delay for readability
```

**Frontend** (`page.tsx`):
```typescript
// Handle streaming status
if (data.status === "streaming") {
  setAgents((prev) => ({
    ...prev,
    [data.agent]: {
      ...prev[data.agent],
      status: "streaming",
      streamingText: prev[data.agent].streamingText + "\n" + data.message,
      progress: 50
    }
  }))
}
```

**UI Component** (`agent-card.tsx`):
```typescript
{/* Display streaming text with typing cursor */}
{status === "streaming" && streamingText && (
  <div className="font-mono text-xs">
    <span className="animate-pulse">{streamingText}</span>
    <span className="inline-block h-4 w-1 animate-pulse bg-current ml-0.5" />
  </div>
)}
```

---

`★ Insight ─────────────────────────────────────`
**Understanding the TypeScript Frontend**
- **React State Management**: Uses `useState` to track each agent's status, progress, and output
- **SSE Streaming**: `ReadableStream` API reads server events in chunks
- **Real-Time Updates**: Each SSE event immediately updates the UI without page refresh
- **Component Communication**: Parent component (`page.tsx`) manages state, child components (`AgentCard`) display it
`─────────────────────────────────────────────────`

---

## **Key Takeaways for Developers**

### **Backend (Python)**
1. **LangGraph**: Think of it as a state machine where each node (agent) transforms the state
2. **TypedDict State**: Use partial updates to avoid re-sending unchanged data
3. **Async/Await**: All I/O operations (OpenAI API, FAISS search) are async for performance
4. **FAISS**: Pre-built index allows <1ms similarity search across thousands of tickets
5. **Binary Classifiers**: More accurate than multi-class for domain/label classification

### **Frontend (TypeScript/Next.js)**
1. **Server-Sent Events (SSE)**: Real-time updates without WebSockets
2. **Component State**: Each agent card manages its own display state
3. **Tailwind CSS**: Utility-first styling for rapid UI development
4. **TypeScript**: Type safety prevents bugs in data flow

### **Data Flow**
1. **Sequential Pipeline**: Each agent depends on previous outputs (no parallelization)
2. **Error Handling**: First error routes to manual escalation (no retries)
3. **State Accumulation**: Messages field uses `operator.add` to append, not replace
4. **Confidence Thresholds**: Classification (0.7), Labels (0.7) filter low-confidence results

---

## **Architecture Diagrams**

### **Component Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  pattern-recognition/page.tsx (Main UI)                  │  │
│  │  - Ticket Submission Form                                │  │
│  │  - Agent Status Cards (4 cards)                          │  │
│  │  - Final Output Display                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓↑ SSE                             │
└─────────────────────────────────────────────────────────────────┘
                                ↓↑
┌─────────────────────────────────────────────────────────────────┐
│                   API SERVER (FastAPI)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  api_server.py                                           │  │
│  │  - /api/process-ticket (POST + SSE streaming)            │  │
│  │  - /api/output (GET final result)                        │  │
│  │  - /api/load-sample (GET sample ticket)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓↑                                 │
└─────────────────────────────────────────────────────────────────┘
                                ↓↑
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND (LangGraph Pipeline)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  workflow.py (Sequential State Machine)                  │  │
│  │                                                           │  │
│  │  Agent 1: Classification                                 │  │
│  │     ↓                                                     │  │
│  │  Agent 2: Pattern Recognition ←─┐                        │  │
│  │     ↓                            │ FAISS Vector Search   │  │
│  │  Agent 3: Label Assignment       │                       │  │
│  │     ↓                            │                       │  │
│  │  Agent 4: Resolution Generation  │                       │  │
│  │                                  │                       │  │
│  │  ┌────────────────────────────┐ │                       │  │
│  │  │  FAISS Manager             │◄┘                       │  │
│  │  │  - Vector Index (3072D)    │                         │  │
│  │  │  - Metadata Store          │                         │  │
│  │  │  - Domain Filtering        │                         │  │
│  │  └────────────────────────────┘                         │  │
│  │                                                           │  │
│  │  ┌────────────────────────────┐                         │  │
│  │  │  OpenAI Client             │                         │  │
│  │  │  - Embeddings (text-3)     │                         │  │
│  │  │  - Chat Completions (GPT-4)│                         │  │
│  │  └────────────────────────────┘                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### **State Flow Through Agents**

```
Initial State                Agent 1 Output              Agent 2 Output
┌─────────────┐             ┌─────────────┐             ┌─────────────┐
│ ticket_id   │             │ + domain    │             │ + similar   │
│ title       │  ──────→    │ + confidence│  ──────→    │   tickets   │
│ description │             │ + keywords  │             │ + scores    │
│ priority    │             │             │             │             │
└─────────────┘             └─────────────┘             └─────────────┘
                                                                ↓
Agent 4 Output              Agent 3 Output                     │
┌─────────────┐             ┌─────────────┐                    │
│ + resolution│  ←──────    │ + labels    │  ←─────────────────┘
│   plan      │             │ + label_conf│
│ + confidence│             │             │
└─────────────┘             └─────────────┘
```

---

## **File Structure Reference**

```
test-recom-backend/                 # Python Backend
│
├── api_server.py                   # FastAPI SSE server
├── main.py                         # CLI entry point
│
├── components/                     # NEW ARCHITECTURE - LangChain agents
│   ├── base/                       # Abstract base classes
│   │   ├── component.py            # BaseComponent ABC
│   │   ├── config.py               # ComponentConfig (Pydantic Settings)
│   │   └── exceptions.py           # Custom exception classes
│   ├── classification/             # Domain Classification Agent (optional)
│   │   ├── agent.py                # LangGraph node wrapper
│   │   ├── tools.py                # @tool decorated functions
│   │   └── classification.md       # Component documentation
│   ├── retrieval/                  # Pattern Recognition Agent
│   │   ├── agent.py
│   │   ├── tools.py
│   │   └── retrieval.md
│   ├── labeling/                   # Label Assignment Agent
│   │   ├── agent.py
│   │   ├── tools.py
│   │   ├── service.py
│   │   └── labeling.md
│   ├── novelty/                    # Novelty Detection Agent
│   │   ├── agent.py
│   │   ├── tools.py
│   │   └── novelty.md
│   ├── resolution/                 # Resolution Generation Agent
│   │   ├── agent.py
│   │   ├── tools.py
│   │   └── resolution.md
│   └── embedding/                  # Utility service (not an agent)
│       ├── service.py
│       └── embedding.md
│
├── src/
│   ├── agents/                     # OLD ARCHITECTURE - legacy (for reference)
│   ├── orchestrator/               # LangGraph workflow definition
│   │   ├── workflow.py             # StateGraph construction
│   │   └── state.py                # TicketWorkflowState TypedDict
│   ├── prompts/                    # LLM prompt templates
│   ├── vectorstore/                # FAISS index management
│   │   ├── faiss_manager.py        # Index CRUD operations
│   │   └── data_ingestion.py       # CSV → embeddings → FAISS
│   ├── models/                     # Pydantic models
│   └── utils/                      # Config, helpers, OpenAI client
│
├── data/
│   ├── raw/
│   │   └── historical_tickets.csv  # Historical data (CSV format)
│   ├── faiss/                      # FAISS index files
│   │   ├── tickets.index           # Binary index file
│   │   └── metadata.json           # Ticket metadata
│   └── metadata/
│       └── categories.json         # Category taxonomy
│
├── input/
│   └── current_ticket.json         # Sample input ticket
│
├── output/
│   └── ticket_resolution.json      # Processing results
│
├── scripts/
│   ├── setup_vectorstore.py        # Build FAISS index
│   └── generate_sample_csv_data.py # Generate sample data
│
├── docs/                           # Documentation
│   ├── CODEBASE.md                 # This file
│   ├── flow.md                     # Architecture flow guide
│   ├── STARTUP.md                  # Setup guide
│   └── horizon_integration.md      # Horizon LLM migration guide
│
├── config/
│   └── schema_config.yaml          # Domain/label/color configuration
│
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
└── CLAUDE.md                       # Project instructions
```

**Frontend Structure** (in `test-recom-frontend/`):

```
test-recom-frontend/                # Next.js 15 Frontend
├── app/
│   ├── page.tsx                    # Root redirect
│   ├── layout.tsx                  # App layout
│   ├── pattern-recognition/        # Main ticket processing UI
│   └── retrieval-engine/           # Search tuning interface
├── components/                     # React components
│   ├── ui/                         # shadcn/ui components
│   └── ...
└── package.json
```

---

## **Environment Configuration**

### **Required Environment Variables** (`.env`)

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-key-here

# Model Selection
CLASSIFICATION_MODEL=gpt-4o-mini           # Cheap, fast for classification
RESOLUTION_MODEL=gpt-4o                    # Powerful for resolution generation

# Model Parameters
CLASSIFICATION_TEMPERATURE=0.2             # Low = deterministic
RESOLUTION_TEMPERATURE=0.6                 # Medium = balanced creativity

# Confidence Thresholds
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7    # Minimum domain confidence
LABEL_CONFIDENCE_THRESHOLD=0.7             # Minimum label confidence

# Retrieval Settings
TOP_K_SIMILAR_TICKETS=20                   # Number of similar tickets to fetch

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-large     # 3072 dimensions
EMBEDDING_DIMENSIONS=3072
```

### **How Configuration Affects Behavior**

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `CLASSIFICATION_MODEL` | Accuracy vs cost trade-off | Use gpt-4o-mini for production |
| `CLASSIFICATION_TEMPERATURE` | Consistency of classifications | Keep at 0.2 for deterministic results |
| `CLASSIFICATION_CONFIDENCE_THRESHOLD` | False positive vs false negative rate | Lower = more permissive, higher = more strict |
| `TOP_K_SIMILAR_TICKETS` | Context richness vs processing time | 20 is optimal for GPT-4 context window |
| `EMBEDDING_DIMENSIONS` | Search accuracy vs storage | 3072 (text-embedding-3-large) for best accuracy |

---

## **Performance Characteristics**

### **Timing Breakdown** (per ticket)

| Component | Time | Explanation |
|-----------|------|-------------|
| Classification Agent | 2-3s | 3 parallel API calls to OpenAI |
| Pattern Recognition | 0.5-1s | 1 embedding call + <1ms FAISS search |
| Label Assignment | 2-3s | 4-6 parallel API calls (one per label) |
| Resolution Generation | 3-5s | 1 GPT-4 call with 8K max tokens |
| **Total** | **8-12s** | End-to-end processing time |

### **Cost Breakdown** (per ticket)

| Component | Tokens | Cost (approx) |
|-----------|--------|---------------|
| Classification (3 calls) | ~1,500 | $0.002 |
| Embedding (1 call) | ~500 | $0.0001 |
| Label Assignment (5 calls) | ~2,000 | $0.003 |
| Resolution Generation | ~6,000 | $0.06 |
| **Total** | **~10,000** | **~$0.07** |

### **FAISS Index Stats**

- **Index Build Time**: ~2 minutes for 100 tickets
- **Index Build Cost**: ~$0.02 (100 embedding calls)
- **Search Latency**: <1ms per query
- **Storage Size**: ~1.2MB per 100 tickets (index + metadata)

---

## **Common Troubleshooting**

### **Issue: "FAISS index not found"**

**Cause**: Vector store hasn't been built yet

**Solution**:
```bash
# Generate sample data
python3 scripts/generate_sample_csv_data.py

# Build FAISS index
python3 scripts/setup_vectorstore.py
```

---

### **Issue: Low classification confidence (<0.7)**

**Cause**: Ticket description doesn't match domain characteristics in prompts

**Solution**:
1. Check domain definitions in `src/prompts/classification_prompts.py`
2. Add more domain-specific keywords to prompts
3. Lower `CLASSIFICATION_CONFIDENCE_THRESHOLD` in `.env` (not recommended)

---

### **Issue: No similar tickets found**

**Cause**: Domain filter mismatch or empty index

**Solution**:
1. Verify domain classification is correct
2. Check FAISS index has tickets in that domain:
   ```python
   from src.vectorstore.faiss_manager import get_faiss_manager
   manager = get_faiss_manager()
   manager.load()
   stats = manager.get_index_stats()
   print(stats)  # Shows domain_distribution
   ```
3. Regenerate historical data with more domain diversity

---

### **Issue: Frontend shows "Network Error"**

**Cause**: FastAPI server not running

**Solution**:
```bash
# Terminal 1: Start FastAPI backend
python3 api_server.py

# Terminal 2: Start Next.js frontend
cd frontend
npm run dev
```

---

### **Issue: Agent stuck in "processing" state**

**Cause**: OpenAI API timeout or error

**Solution**:
1. Check API key is valid: `echo $OPENAI_API_KEY`
2. Check OpenAI status: https://status.openai.com
3. Review backend logs for error messages
4. Increase timeout in `src/utils/openai_client.py`

---

## **Testing & Development**

### **Running the CLI Version**

```bash
# 1. Create input ticket
cat > input/current_ticket.json <<EOF
{
  "ticket_id": "TEST-001",
  "title": "Database connection timeout",
  "description": "MM_ALDER service failing with connection pool exhausted errors",
  "priority": "High",
  "metadata": {}
}
EOF

# 2. Run the workflow
python3 main.py

# 3. Check output
cat output/ticket_resolution.json
```

### **Running the Full Stack**

```bash
# Terminal 1: Start FastAPI backend
python3 api_server.py
# Server runs at http://localhost:8000

# Terminal 2: Start Next.js frontend
cd frontend
npm run dev
# Frontend runs at http://localhost:3000

# Open browser to http://localhost:3000/pattern-recognition
```

### **Testing Individual Agents**

```python
# Test Classification Agent in isolation
import asyncio
from src.agents.classification_agent import classification_agent

async def test():
    state = {
        "ticket_id": "TEST-001",
        "title": "MM_ALDER connection timeout",
        "description": "Database connection pool exhausted during peak hours",
        "priority": "High",
        "metadata": {}
    }

    result = await classification_agent(state)
    print(result)

asyncio.run(test())
```

---

## **Deployment Considerations**

### **Production Checklist**

- [ ] Set `OPENAI_API_KEY` in production environment
- [ ] Configure CORS in `api_server.py` for production domain
- [ ] Build FAISS index from production historical data
- [ ] Set up logging and monitoring
- [ ] Configure rate limiting for API endpoints
- [ ] Add authentication to FastAPI endpoints
- [ ] Build Next.js for production: `npm run build`
- [ ] Set up reverse proxy (nginx) for API and frontend
- [ ] Configure persistent storage for FAISS index
- [ ] Set up automated FAISS index rebuilds (cron job)

### **Scaling Considerations**

**For High Volume** (>100 tickets/hour):
- Use Redis for caching classification results
- Batch similar tickets together for label assignment
- Implement queue system (Celery) for async processing
- Use FAISS IVF index for faster search (>10K tickets)

**For Large Historical Dataset** (>10K tickets):
- Switch to `faiss.IndexIVFFlat` (approximate search)
- Implement pagination for similar tickets
- Consider separate FAISS indices per domain
- Use disk-based storage for metadata (SQLite/PostgreSQL)

---

## **Summary**

This **Intelligent Ticket Management System** demonstrates a production-ready implementation of:

1. **Multi-Agent LLM Pipeline**: 4 specialized agents working sequentially (5 with optional Classification)
2. **Vector Similarity Search**: FAISS for sub-millisecond semantic search
3. **Multi-Signal Novelty Detection**: Detects unknown ticket types without LLM calls
4. **Three-Tier Labeling**: Category (taxonomy) + Business (AI) + Technical (AI) labels
5. **Real-Time UI Updates**: SSE streaming from backend to frontend
6. **Modern Tech Stack**: Python (LangGraph/FastAPI) + TypeScript (Next.js 15)
7. **Scalable Architecture**: Async operations, efficient state management

**Default Pipeline**:
```
Pattern Recognition → Label Assignment → Novelty Detection → Resolution Generation
```

**Key Innovations**:
- Three-tier labeling (predefined categories + AI-generated labels)
- Multi-signal novelty detection (confidence, entropy, centroid distance)
- Hybrid scoring (70% vector + 30% metadata relevance)

**Use Cases**: Technical support automation, incident triage, knowledge base recommendations, automated troubleshooting, novel issue detection.

---

**Document Version**: 2.0
**Last Updated**: 2025-12-03
**Authors**: Codebase Analysis Team
