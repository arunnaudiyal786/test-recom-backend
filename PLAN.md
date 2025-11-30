# Implementation Plan: Retrieval Engine Page

## High-Level Summary

This plan creates a new **"Retrieval Engine"** page in the sidebar that consolidates search/retrieval functionality from the Pattern Recognition page. The new page will:

1. Add a new sidebar navigation item for "Retrieval Engine"
2. Move the Search Tuning functionality to this dedicated page
3. Add detailed retrieval visualization showing exactly what gets retrieved when a query is submitted

The approach leverages existing components (`SearchTuningPanel`) and backend APIs (`/api/preview-search`) while enhancing them to show comprehensive retrieval results.

---

## Architecture Overview

```
SIDEBAR                           MAIN CONTENT AREA
┌─────────────────┐              ┌─────────────────────────────────────────┐
│ Data Preprocess │              │ Retrieval Engine                         │
├─────────────────┤              │ Test and visualize FAISS vector search  │
│ Retrieval       │◄─────────────┼─────────────────────────────────────────│
│ Engine (NEW)    │              │ ┌───────────────────────────────────────┐│
├─────────────────┤              │ │ Query Input                           ││
│ Pattern Recog.  │              │ │ [Textarea]  [Load Sample] [Search]   ││
├─────────────────┤              │ └───────────────────────────────────────┘│
│ Test Recommend. │              │ ┌─────────────────┬─────────────────────┐│
│ Code Fix        │              │ │ Search Config   │ Query Analysis      ││
└─────────────────┘              │ │ - Top K         │ - Domain: MM        ││
                                 │ │ - Vector %      │ - Confidence: 94%   ││
                                 │ │ - Domain Filter │ - Keywords: [...]   ││
                                 │ │ - Advanced...   │                     ││
                                 │ └─────────────────┴─────────────────────┘│
                                 │ ┌───────────────────────────────────────┐│
                                 │ │ Retrieved Tickets (20 found)          ││
                                 │ │ ┌─────────────────────────────────────┐││
                                 │ │ │ #1 JIRA-MM-042 [94.2%]             │││
                                 │ │ │ Title: Database connection timeout │││
                                 │ │ │ ▼ Expand to see full details      │││
                                 │ │ └─────────────────────────────────────┘││
                                 │ │ ┌─────────────────────────────────────┐││
                                 │ │ │ #2 JIRA-MM-015 [89.1%]             │││
                                 │ │ │ ...                                 │││
                                 │ │ └─────────────────────────────────────┘││
                                 │ └───────────────────────────────────────┘│
                                 └─────────────────────────────────────────┘
```

---

## Step-by-Step Implementation Plan

### Step 1: Add New Navigation Item to Sidebar

**File:** `frontend/components/sidebar.tsx`

**Location:** Lines 11-36 (navigationItems array)

**Changes:**
- Import the `SearchCheck` icon from `lucide-react` (represents search/retrieval)
- Add new navigation item between "Data Preprocessing" and "Pattern Recognition":

```typescript
{
  name: "Retrieval Engine",
  href: "/retrieval-engine",
  icon: SearchCheck,  // Search with checkmark icon
  status: "active" as const,
}
```

**Rationale:** The Retrieval Engine is a core feature that deserves its own sidebar entry. Placing it after Data Preprocessing follows the logical workflow (data must exist before searching it).

---

### Step 2: Create New Page Route

**File:** `frontend/app/retrieval-engine/page.tsx` (NEW FILE)

**Structure:**
```
frontend/app/
├── retrieval-engine/
│   └── page.tsx        # New Retrieval Engine page
├── pattern-recognition/
│   └── page.tsx        # Existing (will be modified)
...
```

**Page Components:**
1. **Page Header** - Title and description
2. **Query Input Section** - Ticket/query input form (simplified)
3. **Search Configuration Panel** - Adapted from `SearchTuningPanel` (always expanded)
4. **Query Analysis Section** - Shows domain classification and keywords
5. **Retrieval Results Section** - Detailed view of retrieved tickets

**Rationale:** Following Next.js App Router conventions, each route needs its own directory with a `page.tsx` file.

---

### Step 3: Create Enhanced Retrieval Results Component

**File:** `frontend/components/retrieval-results.tsx` (NEW FILE)

**Purpose:** Display detailed retrieval results including:
- Full list of retrieved tickets (up to top_k)
- Expandable ticket cards showing:
  - Ticket ID, title
  - Full description (not truncated)
  - Vector similarity score
  - Metadata score breakdown
  - Priority and labels
  - Resolution steps (if available)
  - Resolution time

**Key Features:**
- Collapsible ticket cards (click to expand/collapse)
- Score breakdown visualization (blue bar for vector, green for metadata)
- Color-coded similarity badges (green >80%, yellow 60-80%, gray <60%)
- Export retrieved results to JSON

**Dependencies:**
- Uses existing types from `@/types/search-config.ts`
- Uses existing UI components (`Card`, `Badge`, `ScrollArea`, `Collapsible`)

**Rationale:** The current `SearchTuningPanel` shows limited preview results (truncated descriptions). A dedicated component provides the comprehensive view requested.

---

### Step 4: Extend Types for Full Retrieval Data

**File:** `frontend/types/search-config.ts`

**Additions:**
```typescript
export interface RetrievalResult extends SimilarTicketPreview {
  // Full description (not truncated) - already available in API
  // Resolution steps if available
  resolution?: string
}

export interface QueryAnalysis {
  detected_domain: string
  classification_confidence: number | null
  keywords?: string[]
}
```

**Rationale:** Need types to display query analysis alongside results.

---

### Step 5: Add Backend Endpoint for Full Retrieval Details (Optional Enhancement)

**File:** `api_server.py`

**Option A - Minimal Change (Recommended):**
The existing `/api/preview-search` endpoint already returns full data. We just need to:
- Ensure description is not truncated (currently truncated to 500 chars at line 483)
- Include resolution text in response

**Changes at line 483:**
```python
description=t.get('description', ''),  # Remove [:500] truncation
resolution=t.get('resolution', ''),    # Add resolution field
```

**Option B - New Endpoint:**
Create `/api/full-retrieval` that returns complete ticket data including resolution steps. This keeps preview-search fast and adds a separate detailed endpoint.

**Recommendation:** Option A is simpler and sufficient for this use case.

---

### Step 6: Create the Retrieval Engine Page Content

**File:** `frontend/app/retrieval-engine/page.tsx`

**Page Layout:**
```tsx
"use client"

export default function RetrievalEnginePage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Retrieval Engine</h1>
        <p className="text-muted-foreground">
          Test and visualize the FAISS vector search retrieval system
        </p>
      </div>

      {/* Query Input */}
      <Card>
        <CardHeader>
          <CardTitle>Query Input</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Textarea for query/ticket */}
          {/* Load Sample and Search buttons */}
        </CardContent>
      </Card>

      {/* Two-column layout: Config + Analysis */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Search Configuration - extracted from SearchTuningPanel */}
        <Card>...</Card>

        {/* Query Analysis - shows domain, confidence, keywords */}
        <Card>...</Card>
      </div>

      {/* Retrieval Results - full detailed list */}
      <RetrievalResults results={...} config={...} />
    </div>
  )
}
```

**State Management:**
- `queryText: string` - User input
- `searchConfig: SearchConfig` - Current configuration
- `results: SimilarTicketPreview[]` - Retrieved tickets
- `metadata: SearchMetadata | null` - Search statistics
- `isSearching: boolean` - Loading state
- `expandedTickets: Set<string>` - Track which tickets are expanded

---

### Step 7: Remove Search Tuning from Pattern Recognition Page

**File:** `frontend/app/pattern-recognition/page.tsx`

**Changes:**
- Remove `SearchTuningPanel` import (line 7)
- Remove `SearchTuningPanel` component usage (lines 276-282)
- Remove `savedSearchConfig` state (line 29)
- Remove `SearchConfig` import (line 12) if no longer needed
- Optionally add a note or link directing users to the Retrieval Engine page

**Before (lines 276-282):**
```tsx
{/* Search Tuning Panel - Test and tune similarity search before processing */}
<SearchTuningPanel
  ticketTitle={currentTicketText.split("\n")[0] || ""}
  ticketDescription={currentTicketText}
  onConfigSaved={setSavedSearchConfig}
  disabled={isProcessing}
/>
```

**After:**
```tsx
{/* Search configuration is managed in Retrieval Engine page */}
```

**Rationale:** Search tuning is now in the dedicated Retrieval Engine page. The Pattern Recognition page focuses on the full pipeline workflow. Config is still persisted to file and read by the backend.

---

### Step 8: Update SimilarTicketPreview Type (Backend)

**File:** `src/models/retrieval_config.py`

**Changes:**
Add `resolution` field to `SimilarTicketPreview`:
```python
class SimilarTicketPreview(BaseModel):
    ticket_id: str
    title: str
    description: str
    similarity_score: float
    vector_similarity: float
    metadata_score: float
    priority: str
    labels: List[str]
    resolution_time_hours: float
    domain: str
    resolution: Optional[str] = None  # ADD THIS FIELD
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `frontend/components/sidebar.tsx` | MODIFY | Add "Retrieval Engine" nav item |
| `frontend/app/retrieval-engine/page.tsx` | CREATE | New page for retrieval testing |
| `frontend/components/retrieval-results.tsx` | CREATE | Component for detailed results display |
| `frontend/types/search-config.ts` | MODIFY | Add QueryAnalysis type, resolution field |
| `frontend/app/pattern-recognition/page.tsx` | MODIFY | Remove SearchTuningPanel |
| `api_server.py` | MODIFY | Include resolution in preview response, remove truncation |
| `src/models/retrieval_config.py` | MODIFY | Add resolution field to SimilarTicketPreview |

---

## Potential Challenges & Solutions

### Challenge 1: Large Response Sizes
**Issue:** Full ticket descriptions could make API responses large.
**Solution:** Keep descriptions in response but implement virtual scrolling in UI. Only render visible tickets.

### Challenge 2: Component Extraction
**Issue:** `SearchTuningPanel` combines config controls and preview results.
**Solution:** For the new page, we can either:
- Reuse `SearchTuningPanel` as-is (simpler)
- Extract config controls into separate component (cleaner architecture)

**Recommendation:** Reuse `SearchTuningPanel` initially, refactor later if needed.

### Challenge 3: State Synchronization
**Issue:** Search config changes in Retrieval Engine should apply to Pattern Recognition workflow.
**Solution:** Config is already persisted to `config/search_config.json`. Both pages read from the same source. The "Save Config" button in Retrieval Engine persists for the Pattern Recognition workflow.

---

## Dependencies

- No new npm packages required
- Uses existing Radix UI components
- Uses existing backend infrastructure
- Uses existing FAISS vector store

---

## Testing Checklist

1. [ ] Sidebar shows new "Retrieval Engine" link
2. [ ] Clicking link navigates to `/retrieval-engine`
3. [ ] Query input accepts text
4. [ ] "Load Sample" loads example ticket
5. [ ] Search returns results with full details
6. [ ] Expanding ticket shows description and resolution
7. [ ] Config changes affect search results
8. [ ] Saving config persists to file
9. [ ] Pattern Recognition page works without SearchTuningPanel
10. [ ] Config saved in Retrieval Engine applies to Pattern Recognition workflow

---

## Order of Implementation

1. **Sidebar modification** - Add navigation item (quick, enables navigation)
2. **Create basic page** - Route exists with placeholder content
3. **Add resolution to backend** - Modify API response
4. **Create RetrievalResults component** - Enhanced display
5. **Build full page** - Integrate query input, config, and results
6. **Remove from Pattern Recognition** - Cleanup
7. **Test end-to-end** - Verify everything works

---

## Estimated Effort

| Task | Complexity | Files Changed |
|------|-----------|---------------|
| Sidebar nav item | Low | 1 |
| New page route | Medium | 1 |
| RetrievalResults component | Medium | 1 |
| Backend resolution field | Low | 2 |
| Pattern Recognition cleanup | Low | 1 |
| Testing | Medium | - |

**Total: 6 files modified/created**
