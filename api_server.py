"""
FastAPI server for streaming LangGraph agent updates to the Next.js frontend.

This server provides SSE (Server-Sent Events) streaming of agent progress
as tickets are processed through the LangGraph workflow.
"""
import asyncio
import json
from typing import AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# Use new LangGraph orchestrator from src/orchestrator
from src.orchestrator.workflow import get_workflow, SKIP_DOMAIN_CLASSIFICATION
from src.orchestrator.state import TicketWorkflowState as TicketState
from src.utils.csv_exporter import export_ticket_results_to_csv
from src.utils.session_manager import SessionManager
from src.utils.mermaid_graph import save_session_workflow_graph
from src.models.retrieval_config import (
    RetrievalConfig,
    RetrievalPreviewRequest,
    RetrievalPreviewResponse,
    SimilarTicketPreview,
    SearchMetadata
)

# Import agents from components (LangChain-style)
from components.retrieval.agent import pattern_recognition_agent as historical_match_agent
from components.classification.agent import classification_agent
from components.classification.tools import classify_ticket_domain

# Import v2 component routers (for individual component HTTP access)
from components.embedding import router as embedding_router
from components.retrieval import router as retrieval_router
from components.classification import router as classification_router
from components.labeling import router as labeling_router
# Note: augmentation is now part of resolution, orchestrator is in src/orchestrator


app = FastAPI(
    title="RRE Ticket Processing API",
    description="API for processing support tickets through the LangGraph multi-agent pipeline",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Mount v2 Component Routers
# ============================================================================
# Each component is self-contained and can be used:
# 1. Via HTTP endpoints (mounted here)
# 2. As Python modules (import from components.*)
#
# API Structure:
#   /v2/embedding/*      - Embedding generation
#   /v2/retrieval/*      - Similar ticket search
#   /v2/classification/* - Domain classification
#   /v2/labeling/*       - Label assignment
#
# Note: Full pipeline orchestration uses /api/process-ticket (LangGraph-based)
#       Resolution generation is integrated into the pipeline workflow

app.include_router(embedding_router, prefix="/v2")
app.include_router(retrieval_router, prefix="/v2")
app.include_router(classification_router, prefix="/v2")
app.include_router(labeling_router, prefix="/v2")


class TicketInput(BaseModel):
    """Input schema for ticket processing."""
    ticket_id: str
    title: str
    description: str
    priority: str
    metadata: dict


class AgentUpdate(BaseModel):
    """Schema for agent status updates."""
    agent: str
    status: str  # "processing", "streaming", "complete", "error"
    message: str | None = None
    data: dict | None = None
    progress: int | None = None
    tool_calls: list[dict] | None = None  # Track tool invocations
    tool_outputs: list[dict] | None = None  # Track tool results


async def stream_agent_updates(ticket: TicketInput) -> AsyncGenerator[str, None]:
    """
    Stream agent updates as they process the ticket.

    Yields SSE-formatted messages with agent progress updates.
    """
    try:
        # Initialize session manager and generate unique session ID
        session_manager = SessionManager()
        session_id = session_manager.generate_session_id()
        print(f"📁 Session ID: {session_id}")

        # Get compiled workflow
        workflow = get_workflow()

        # Load saved search config if available
        from pathlib import Path
        search_config = None
        config_path = Path("config/search_config.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    search_config = json.load(f)
                print("📐 Using saved search configuration")
            except Exception as e:
                print(f"Warning: Could not load search config: {e}")

        # Create initial state - adjust based on whether classification is skipped
        if SKIP_DOMAIN_CLASSIFICATION:
            initial_state: TicketState = {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "description": ticket.description,
                "priority": ticket.priority,
                "metadata": ticket.metadata,
                "session_id": session_id,  # Unique session for output storage
                "search_config": search_config,  # Include custom search config if saved
                "processing_stage": "retrieval",
                "status": "processing",
                "current_agent": "Historical Match Agent",
                "classified_domain": None,  # No domain classification - will search all domains
                "messages": [],
            }
            # Agent names for progress tracking (without classification)
            agents = [
                "Historical Match Agent",
                "Label Assignment Agent",
                "Novelty Detection Agent",
                "Resolution Generation Agent"
            ]
        else:
            initial_state: TicketState = {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "description": ticket.description,
                "priority": ticket.priority,
                "metadata": ticket.metadata,
                "session_id": session_id,  # Unique session for output storage
                "search_config": search_config,  # Include custom search config if saved
                "processing_stage": "classification",
                "status": "processing",
                "current_agent": "Domain Classification Agent",
                "messages": [],
            }
            # Agent names for progress tracking (with classification)
            agents = [
                "Domain Classification Agent",
                "Historical Match Agent",
                "Label Assignment Agent",
                "Novelty Detection Agent",
                "Resolution Generation Agent"
            ]

        agent_index = 0
        # Accumulate full state across all agents (not just last partial update)
        accumulated_state = dict(initial_state)
        previous_message_count = 0  # Track messages we've already sent

        # Save session metadata at start
        session_manager.save_session_metadata(session_id, {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "description": ticket.description,
            "priority": ticket.priority,
            "metadata": ticket.metadata,
        })

        # Process through workflow with streaming updates
        async for event in workflow.astream(initial_state):
            # Extract the current state from the event
            # LangGraph astream yields dict with node names as keys
            for node_name, node_state in event.items():
                # Merge partial state into accumulated state
                for key, value in node_state.items():
                    if key == "messages" and isinstance(value, list):
                        # Accumulate messages (they use operator.add reducer)
                        existing = accumulated_state.get("messages", [])
                        accumulated_state["messages"] = existing + value
                    else:
                        accumulated_state[key] = value
                current_agent = node_state.get("current_agent", "unknown")
                status = node_state.get("status", "processing")

                # Send agent start notification
                if agent_index < len(agents) and current_agent == agents[agent_index]:
                    update = AgentUpdate(
                        agent=_agent_key(current_agent),
                        status="processing",
                        message=f"Starting {current_agent}...",
                        progress=0
                    )
                    yield f"data: {update.model_dump_json()}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for UI

                # Stream any new messages that were added to state
                current_messages = node_state.get("messages", [])
                if len(current_messages) > previous_message_count:
                    # New messages have been added - stream them
                    new_messages = current_messages[previous_message_count:]
                    previous_message_count = len(current_messages)

                    for msg in new_messages:
                        if msg.get("role") == "assistant":
                            update = AgentUpdate(
                                agent=_agent_key(current_agent),
                                status="streaming",
                                message=msg.get("content", ""),
                                progress=50
                            )
                            yield f"data: {update.model_dump_json()}\n\n"
                            await asyncio.sleep(0.05)


                # Send completion notification
                if status == "success":
                    # Extract agent-specific data
                    agent_data = _extract_agent_data(current_agent, node_state)

                    # Save individual agent output to session directory
                    try:
                        session_manager.save_agent_output(
                            session_id=session_id,
                            agent_name=_agent_key(current_agent),
                            data=node_state
                        )
                    except Exception as e:
                        print(f"Warning: Could not save agent output: {e}")

                    update = AgentUpdate(
                        agent=_agent_key(current_agent),
                        status="complete",
                        message=f"{current_agent} completed successfully",
                        data=agent_data,
                        progress=100
                    )
                    yield f"data: {update.model_dump_json()}\n\n"
                    agent_index += 1
                    await asyncio.sleep(0.1)

                # Handle errors
                elif status == "error":
                    error_msg = node_state.get("error_message", "Unknown error")
                    update = AgentUpdate(
                        agent=_agent_key(current_agent),
                        status="error",
                        message=error_msg,
                        progress=0
                    )
                    yield f"data: {update.model_dump_json()}\n\n"
                    return

        # Send workflow completion with final state
        # Save the final state to session directory
        csv_path = None
        session_dir = None
        try:
            import json as json_lib

            if accumulated_state:
                # Get session output directory
                session_dir = session_manager.get_session_output_dir(session_id)

                # Save JSON output (full accumulated state) to session directory
                output_file = session_manager.save_final_output(
                    session_id=session_id,
                    data=accumulated_state,
                    filename="ticket_resolution.json"
                )
                print(f"💾 Saved output to: {output_file}")

                # Export to CSV in session directory
                csv_path = export_ticket_results_to_csv(
                    accumulated_state,
                    output_path=session_dir / "ticket_results.csv",
                    append=False  # Each session gets its own CSV
                )
                print(f"📊 Exported results to CSV: {csv_path}")

                # Save workflow graph (Mermaid + PNG) in session directory
                mermaid_path, png_path = save_session_workflow_graph(session_dir)

                # Update 'latest' symlink to point to this session
                session_manager.update_latest_symlink(session_id)

        except Exception as e:
            print(f"Error saving output file: {e}")
            mermaid_path = None
            png_path = None

        completion = {
            "status": "workflow_complete",
            "message": "All agents completed successfully",
            "session_id": session_id,
            "session_path": str(session_dir) if session_dir else None,
            "output_available": True,
            "csv_exported": csv_path is not None,
            "csv_path": str(csv_path) if csv_path else None,
            "mermaid_graph_path": str(mermaid_path) if mermaid_path else None,
            "png_graph_path": str(png_path) if png_path else None
        }
        yield f"data: {json.dumps(completion)}\n\n"

    except Exception as e:
        error_update = {
            "status": "error",
            "message": f"Workflow error: {str(e)}"
        }
        yield f"data: {json.dumps(error_update)}\n\n"


def _agent_key(agent_name: str) -> str:
    """Convert agent name to frontend key.

    Handles both full agent names and short names from node state.
    """
    # Mapping for both full names and short names
    mapping = {
        # Full names
        "Domain Classification Agent": "classification",
        "Historical Match Agent": "historicalMatch",
        "Label Assignment Agent": "labelAssignment",
        "Novelty Detection Agent": "noveltyDetection",
        "Resolution Generation Agent": "resolutionGeneration",
        # Short names (from node state current_agent field)
        "classification": "classification",
        "retrieval": "historicalMatch",
        "labeling": "labelAssignment",
        "novelty": "noveltyDetection",
        "resolution": "resolutionGeneration",
    }
    return mapping.get(agent_name, agent_name.lower().replace(" ", "_"))


def _extract_agent_data(agent_name: str, state: dict) -> dict:
    """Extract relevant data from state for each agent.

    Handles both full agent names (e.g., "Domain Classification Agent")
    and short names from node state (e.g., "classification").
    """
    # Normalize agent name to handle both short and full names
    agent_key = agent_name.lower()

    # DEBUG: Log state keys for label assignment agent
    if agent_key in ("label assignment agent", "labeling"):
        print(f"   🔍 DEBUG _extract: state keys: {list(state.keys())}")
        print(f"   🔍 DEBUG _extract: label_assignment_prompts present: {'label_assignment_prompts' in state}")
        if 'label_assignment_prompts' in state:
            prompts = state.get('label_assignment_prompts', {})
            print(f"   🔍 DEBUG _extract: prompts keys: {list(prompts.keys()) if isinstance(prompts, dict) else type(prompts)}")

    if agent_key in ("domain classification agent", "classification"):
        return {
            "classified_domain": state.get("classified_domain"),
            "confidence": state.get("classification_confidence"),
            "reasoning": state.get("classification_reasoning"),
            "keywords": state.get("extracted_keywords", []),
        }
    elif agent_key in ("historical match agent", "retrieval"):
        similar_tickets = state.get("similar_tickets", [])
        # Get top 5 similar tickets for display
        top_tickets = similar_tickets[:5] if similar_tickets else []
        search_metadata = state.get("search_metadata", {})

        return {
            "similar_tickets_count": len(similar_tickets),
            "top_similarity": state.get("similarity_scores", [0])[0] if state.get("similarity_scores") else 0,
            "domain_filter": state.get("classified_domain"),
            "avg_similarity": search_metadata.get("avg_similarity", 0),
            "top_tickets": [
                {
                    "ticket_id": ticket.get("ticket_id", "Unknown"),
                    "title": ticket.get("title", ""),
                    "similarity_score": ticket.get("similarity_score", 0),
                    "vector_similarity": ticket.get("vector_similarity", 0),
                    "metadata_score": ticket.get("metadata_score", 0),
                    "priority": ticket.get("priority", "Medium"),
                    "labels": ticket.get("labels", []),
                    "resolution_time_hours": ticket.get("resolution_time_hours", 0),
                }
                for ticket in top_tickets
            ],
        }
    elif agent_key in ("label assignment agent", "labeling"):
        # Category labels (from predefined taxonomy - the new three-tier system)
        category_labels = state.get("category_labels", [])

        # Historical labels (from similar tickets - legacy, kept for backward compat)
        historical_labels = state.get("historical_labels", [])
        historical_confidence = state.get("historical_label_confidence", {})
        historical_distribution = state.get("historical_label_distribution", {})

        # AI-Generated labels
        business_labels = state.get("business_labels", [])
        technical_labels = state.get("technical_labels", [])

        # Combined labels (backward compatibility)
        assigned_labels = state.get("assigned_labels", [])
        all_confidence = state.get("label_confidence", {})
        label_distribution = state.get("label_distribution", {})

        # Actual prompts sent to LLM
        actual_prompts = state.get("label_assignment_prompts", {})

        # Build historical labels with details (for rejected calculation)
        assigned_with_details = [
            {
                "label": label,
                "confidence": historical_confidence.get(label, all_confidence.get(label, 0)),
                "distribution": historical_distribution.get(label, label_distribution.get(label, "N/A")),
                "assigned": True,
            }
            for label in historical_labels
        ]

        # Get rejected labels (historical labels that were considered but not assigned)
        rejected_labels = [
            {
                "label": label,
                "confidence": conf,
                "distribution": historical_distribution.get(label, label_distribution.get(label, "N/A")),
                "assigned": False,
            }
            for label, conf in historical_confidence.items()
            if label not in historical_labels and not label.startswith("[BIZ]") and not label.startswith("[TECH]")
        ]

        # Sort rejected by confidence (descending)
        rejected_labels.sort(key=lambda x: x["confidence"], reverse=True)

        # Calculate total candidates (only historical labels were candidates)
        total_candidates = len(historical_confidence) if historical_confidence else len(all_confidence)

        return {
            # Three-tier labels
            "category_labels": category_labels,  # From predefined taxonomy
            "historical_labels": historical_labels,
            "historical_label_confidence": historical_confidence,
            "historical_label_distribution": historical_distribution,
            "business_labels": business_labels,
            "technical_labels": technical_labels,

            # Combined (backward compatibility)
            "assigned_labels": assigned_labels,
            "label_count": len(assigned_labels),
            "confidence": all_confidence,
            "assigned_with_details": assigned_with_details,
            "rejected_labels": rejected_labels,
            "total_candidates": total_candidates,

            # Actual prompts for transparency
            "actual_prompts": actual_prompts,
        }
    elif agent_key in ("novelty detection agent", "novelty"):
        return {
            "novelty_detected": state.get("novelty_detected", False),
            "novelty_score": state.get("novelty_score", 0),
            "novelty_signals": state.get("novelty_signals", {}),
            "novelty_recommendation": state.get("novelty_recommendation", "proceed"),
            "novelty_reasoning": state.get("novelty_reasoning", ""),
            "novelty_details": state.get("novelty_details", {}),
        }
    elif agent_key in ("resolution generation agent", "resolution"):
        resolution = state.get("resolution_plan", {})
        return {
            "summary": resolution.get("summary", ""),
            "total_steps": len(resolution.get("resolution_steps", [])),
            "estimated_hours": resolution.get("total_estimated_time_hours", 0),
            "confidence": resolution.get("confidence", 0),
            # Actual prompt for transparency
            "actual_prompt": state.get("resolution_generation_prompt", ""),
        }
    return {}


@app.post("/api/process-ticket")
async def process_ticket(ticket: TicketInput):
    """
    Process a ticket through the LangGraph workflow with streaming updates.

    Returns Server-Sent Events (SSE) stream of agent progress.
    """
    return StreamingResponse(
        stream_agent_updates(ticket),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/load-sample")
async def load_sample_ticket():
    """Load sample ticket from input/current_ticket.json."""
    import json
    from pathlib import Path

    try:
        sample_path = Path("input/current_ticket.json")
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Sample ticket file not found")

        with open(sample_path, "r") as f:
            sample_ticket = json.load(f)

        return sample_ticket
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample ticket: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "RRE Ticket Processing API"
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration settings."""
    return {
        "skip_domain_classification": SKIP_DOMAIN_CLASSIFICATION,
        "active_agents": [
            "Historical Match Agent",
            "Label Assignment Agent",
            "Resolution Generation Agent"
        ] if SKIP_DOMAIN_CLASSIFICATION else [
            "Domain Classification Agent",
            "Historical Match Agent",
            "Label Assignment Agent",
            "Resolution Generation Agent"
        ]
    }


@app.get("/api/schema-config")
async def get_schema_config_endpoint():
    """
    Get schema configuration for the frontend.

    Returns domain definitions, colors, and UI settings that allow
    the frontend to dynamically adapt to different data schemas.
    """
    from src.utils.schema_config import get_schema_config

    try:
        schema_config = get_schema_config()
        return schema_config.get_frontend_config()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load schema config: {str(e)}"
        )


@app.post("/api/reload-schema-config")
async def reload_schema_config_endpoint():
    """
    Reload schema configuration from file.

    Use this after editing config/schema_config.yaml to apply changes
    without restarting the server.
    """
    from src.utils.schema_config import reload_schema_config

    try:
        reload_schema_config()
        return {
            "status": "success",
            "message": "Schema configuration reloaded"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload schema config: {str(e)}"
        )


@app.get("/api/output")
async def get_output():
    """Retrieve the final ticket resolution JSON output from latest session."""
    from pathlib import Path

    try:
        # Try latest symlink first (session-based)
        output_path = Path("output/latest/ticket_resolution.json")
        if not output_path.exists():
            # Fallback to legacy flat path for backward compatibility
            output_path = Path("output/ticket_resolution.json")

        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found. Process a ticket first.")

        with open(output_path, "r") as f:
            output_data = json.load(f)

        return output_data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error loading output: {str(e)}")


@app.get("/api/download-csv")
async def download_csv():
    """Download the ticket results CSV file from latest session."""
    from pathlib import Path

    try:
        # Try latest symlink first (session-based)
        csv_path = Path("output/latest/ticket_results.csv")
        if not csv_path.exists():
            # Fallback to legacy flat path for backward compatibility
            csv_path = Path("output/ticket_results.csv")

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="CSV file not found. Process a ticket first."
            )

        return FileResponse(
            path=csv_path,
            filename="ticket_results.csv",
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=ticket_results.csv"
            }
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error downloading CSV: {str(e)}")


@app.get("/api/prompts")
async def get_prompts():
    """
    Return prompt templates used by agents for UI transparency.

    This endpoint exposes the actual prompts used by the Label Assignment
    and Resolution Generation agents, allowing users to understand
    how the AI is making decisions.
    """
    from src.prompts.label_assignment_prompts import (
        LABEL_ASSIGNMENT_TEMPLATE,
        BUSINESS_LABEL_GENERATION_TEMPLATE,
        TECHNICAL_LABEL_GENERATION_TEMPLATE,
        LABEL_CRITERIA
    )
    from src.prompts.resolution_generation_prompts import (
        RESOLUTION_GENERATION_PROMPT
    )

    return {
        "label_assignment": {
            "historical": {
                "template": LABEL_ASSIGNMENT_TEMPLATE,
                "description": "Binary classifier prompt for evaluating historical labels from similar tickets",
                "label_criteria": LABEL_CRITERIA
            },
            "business": {
                "template": BUSINESS_LABEL_GENERATION_TEMPLATE,
                "description": "AI-generated business-oriented labels based on impact and stakeholders"
            },
            "technical": {
                "template": TECHNICAL_LABEL_GENERATION_TEMPLATE,
                "description": "AI-generated technical labels for system components and failure modes"
            }
        },
        "resolution_generation": {
            "template": RESOLUTION_GENERATION_PROMPT,
            "description": "Chain-of-Thought prompt for generating comprehensive resolution plans"
        }
    }


# ============================================================================
# Search Tuning Endpoints
# ============================================================================

@app.post("/api/preview-search", response_model=RetrievalPreviewResponse)
async def preview_search(request: RetrievalPreviewRequest):
    """
    Preview search results with custom configuration.

    This endpoint allows testing different search parameters before
    running the full pipeline. It performs:
    1. Domain classification (if domain_filter not provided)
    2. FAISS vector search with custom top_k
    3. Hybrid scoring with custom weights

    Returns similar tickets with detailed scoring breakdown.
    """
    try:
        # Determine domain - either from config or via classification
        if request.config.domain_filter:
            domain = request.config.domain_filter
            classification_confidence = None
        else:
            # Run classification to get domain using LangChain tool
            classification_result = await classify_ticket_domain.ainvoke({
                "title": request.title,
                "description": request.description
            })
            domain = classification_result.get("classified_domain", "Unknown")
            classification_confidence = classification_result.get("confidence")

        # Run preview search with config
        result = await historical_match_agent.preview_search(
            title=request.title,
            description=request.description,
            domain=domain,
            config=request.config
        )

        # Build response
        similar_tickets = [
            SimilarTicketPreview(
                ticket_id=t.get('ticket_id', ''),
                title=t.get('title', ''),
                description=t.get('description', ''),  # Full description for Retrieval Engine
                similarity_score=t.get('similarity_score', 0),
                vector_similarity=t.get('vector_similarity', 0),
                metadata_score=t.get('metadata_score', 0),
                priority=t.get('priority', 'Medium'),
                labels=t.get('labels', []),
                resolution_time_hours=t.get('resolution_time_hours', 0),
                domain=t.get('domain', ''),
                resolution=t.get('resolution', None)  # Include resolution steps
            )
            for t in result['similar_tickets']
        ]

        search_metadata = SearchMetadata(
            query_domain=result['search_metadata']['query_domain'],
            total_found=result['search_metadata']['total_found'],
            avg_similarity=result['search_metadata']['avg_similarity'],
            top_similarity=result['search_metadata']['top_similarity'],
            classification_confidence=classification_confidence
        )

        return RetrievalPreviewResponse(
            similar_tickets=similar_tickets,
            search_metadata=search_metadata,
            config_used=request.config
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preview search failed: {str(e)}"
        )


@app.post("/api/save-search-config")
async def save_search_config(config: RetrievalConfig):
    """
    Save search configuration for use in the pipeline.

    The saved config will be used by downstream agents when processing tickets.
    """
    from pathlib import Path

    try:
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "search_config.json"
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

        return {
            "status": "success",
            "message": "Search configuration saved",
            "config_path": str(config_path)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save config: {str(e)}"
        )


@app.get("/api/load-search-config", response_model=RetrievalConfig)
async def load_search_config():
    """
    Load saved search configuration.

    Returns the previously saved config or defaults if none exists.
    """
    from pathlib import Path

    try:
        config_path = Path("config/search_config.json")

        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return RetrievalConfig(**config_data)
        else:
            # Return defaults
            return RetrievalConfig()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load config: {str(e)}"
        )


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.get("/api/sessions")
async def list_sessions(limit: int = 50):
    """
    List all processing sessions.

    Returns a list of sessions with metadata, sorted by date (newest first).
    """
    try:
        session_manager = SessionManager()
        sessions = session_manager.list_sessions(limit=limit)
        return {
            "sessions": sessions,
            "total": len(sessions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get detailed information about a specific session.
    """
    try:
        session_manager = SessionManager()
        session = session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found"
            )

        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session: {str(e)}"
        )


@app.get("/api/sessions/{session_id}/output")
async def get_session_output(session_id: str):
    """
    Get the final output from a specific session.
    """
    try:
        session_manager = SessionManager()
        output = session_manager.get_session_output(session_id)

        if not output:
            raise HTTPException(
                status_code=404,
                detail=f"Output not found for session '{session_id}'"
            )

        return output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session output: {str(e)}"
        )


@app.get("/api/sessions/{session_id}/agents/{agent_name}")
async def get_session_agent_output(session_id: str, agent_name: str):
    """
    Get a specific agent's output from a session.

    Agent names: classification, historicalMatch, labelAssignment,
                 novelty, resolutionGeneration
    """
    try:
        session_manager = SessionManager()
        output = session_manager.get_agent_output(session_id, agent_name)

        if not output:
            raise HTTPException(
                status_code=404,
                detail=f"Agent output '{agent_name}' not found for session '{session_id}'"
            )

        return output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent output: {str(e)}"
        )


@app.get("/api/sessions/{session_id}/csv")
async def download_session_csv(session_id: str):
    """
    Download the CSV file from a specific session.
    """
    from pathlib import Path

    try:
        session_manager = SessionManager()
        session_dir = session_manager.get_session_output_dir(session_id)
        csv_path = session_dir / "ticket_results.csv"

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"CSV file not found for session '{session_id}'"
            )

        return FileResponse(
            path=csv_path,
            filename=f"ticket_results_{session_id}.csv",
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=ticket_results_{session_id}.csv"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download CSV: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RRE Ticket Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            # v1 endpoints (LangGraph-based)
            "process_ticket": "/api/process-ticket",
            "preview_search": "/api/preview-search",
            "save_search_config": "/api/save-search-config",
            "load_search_config": "/api/load-search-config",
            "get_output": "/api/output",
            "download_csv": "/api/download-csv",
            "health": "/api/health",
        },
        "session_endpoints": {
            # Session management endpoints
            "list_sessions": "/api/sessions",
            "get_session": "/api/sessions/{session_id}",
            "get_session_output": "/api/sessions/{session_id}/output",
            "get_agent_output": "/api/sessions/{session_id}/agents/{agent_name}",
            "download_session_csv": "/api/sessions/{session_id}/csv",
        },
        "v2_endpoints": {
            # Component endpoints (individual component access)
            "embedding": {
                "generate": "/v2/embedding/generate",
                "batch": "/v2/embedding/batch",
                "health": "/v2/embedding/health",
            },
            "retrieval": {
                "search": "/v2/retrieval/search",
                "stats": "/v2/retrieval/stats",
                "health": "/v2/retrieval/health",
            },
            "classification": {
                "classify": "/v2/classification/classify",
                "health": "/v2/classification/health",
            },
            "labeling": {
                "assign": "/v2/labeling/assign",
                "health": "/v2/labeling/health",
            },
            # Note: Full pipeline orchestration uses /api/process-ticket (LangGraph-based)
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting RRE API Server...")
    print("📡 Frontend: http://localhost:3000")
    print("🔧 API Docs: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
