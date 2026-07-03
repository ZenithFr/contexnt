# Contex'nt 2.0 - Implementation Plan

This document outlines the step-by-step technical implementation to build the 6-Layer Cognitive Architecture for Contex'nt.

## Tech Stack
- **Language:** Python 3.10+
- **MCP Framework:** `mcp` (FastMCP for rapid tool exposure)
- **Agent Orchestration:** `langgraph` (Manager and Librarian routing)
- **Data Layer:** `aiosqlite` / `sqlite3` (built-in, WAL mode) and local filesystem (Obsidian).

---

## Technical Specifications & Edge Cases Resolved

#### 1. LangGraph Native State & Thread Config
We delegate all conversational state persistence to LangGraph's native `SqliteSaver` attached to `vault.db`. The incoming `session_id` is mapped to LangGraph's thread config: `{"configurable": {"thread_id": session_id}}`.

#### 2. Subgraph Isolation (Preventing State Key Collision)
We define a distinct `LibrarianState` containing `search_messages`, `session_id`, `librarian_index`, and `search_result: str`. The Manager passes query and indexing context in, and maps only the final string output to the `search_result: str` field in the parent `AgentState`, ensuring absolute isolation.

#### 3. Token Checking via Conditional Edges & Hard Caps
LangGraph lacks middleware. We use conditional edges (`route_token_check`). If tokens exceed `MAX_LIMIT`, it routes to a `rollover_node`. 
Crucially, the `rollover_node` enforces hard caps (Max 5 Managers, Max 3 Librarians). If exceeded, it raises `RuntimeError("Cognitive limits exceeded")`.

#### 4. The Handoff Message (Super Subs Context Retention)
When a Super Sub rollover occurs, the `rollover_node` creates a "handoff message" (*"You are agent-2. Agent-1 ran out of context and was archived. Resume from: [last state]"*) as the first message for the fresh agent.

#### 5. Context Retention & Explicit Turn Counting
We add a `turn_count: int` and `summary: str` field to `AgentState`. On the 4th turn, a `summarize_history` node updates the summary and resets the pocket. The Manager node dynamically compiles its system prompt using `state.get('summary', '')`.

#### 6. Thread Safety, Async Lifecycles & FastMCP Context Managers
FastMCP uses async context managers, not decorators. The daemon and indexing logic run inside `mcp_lifespan(app: FastMCP)`. This manages the background `overseer_loop()` task and ensures graceful teardown via `shutdown_event = asyncio.Event()`. All Overseer DB operations use `aiosqlite`.

#### 7. SQLite Connection Locking Conflicts & Indexing
Execute `PRAGMA busy_timeout = 10000;` on all SQLite connections to enqueue write locks. Crucial indexing commands (`idx_logs_session_timestamp` and `idx_metadata_last_active`) are applied.

#### 8. Namespace-Based Hotbar Routing
Map bundles directly to namespaces in the `hotbar_cache` table for O(1) lookup speed.

#### 9. Overseer Idle-Detection & File Cleanup
FastMCP tool updates `session_metadata`. The Overseer processes sessions idle > 600 seconds. After parsing the session, the Overseer explicitly deletes the temporary `active_context_{session_id}.md` to prevent Obsidian vault bloating.

#### 10. Payload Handoff to Obsidian
Large payloads write to `agentic-zen/contexnt/active_context_{session_id}.md`. The primary agent receives only `{ "path": str, "summary": str, "session_id": str }` so it can thread subsequent calls if the MCP server generated a new UUID session.

#### 11. Nested Git Repository Protection & Bound Traversal
The Obsidian adapter skips local `git init` and appends to parent `.gitignore` if a root `.git` exists. The traversal search stops strictly at `~/` to prevent falsely detecting system-wide git directories. It also uses `mkdir(parents=True, exist_ok=True)` on first initialization to prevent `FileNotFoundError`.

#### 12. Two-Way Skill Sync
The `mcp_lifespan` cold start performs a two-way sync: it upserts existing `SKILL.md` files and executes a `DELETE` for any skills in `vault.db` where the filepath no longer exists on disk.

---

## State & Database Schemas

### LangGraph Agent States
```python
class AgentState(TypedDict):
    goal: str
    summary: str  
    search_result: str  # Output mapping from Librarian subgraph
    messages: list[BaseMessage]
    turn_count: int
    manager_index: int
    librarian_index: int
    session_id: str

class LibrarianState(TypedDict):
    search_query: str
    search_messages: list[BaseMessage]
    search_result: str
    session_id: str
    librarian_index: int
```

### SQLite Database (vault.db)
```sql
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    namespace TEXT NOT NULL,
    filepath TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skill_usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_logs_session_timestamp ON skill_usage_logs(session_id, timestamp);

CREATE TABLE IF NOT EXISTS hotbar_cache (
    namespace TEXT PRIMARY KEY,
    skills_list TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_metadata (
    session_id TEXT PRIMARY KEY,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_metadata_last_active ON session_metadata(last_active);
```

---

## Implementation Phases

### Phase 1: Foundation, The Vault, & Lifespans
1. **Workspace:** Setup `venv`, install `mcp`, `langgraph`, `aiosqlite`, and `pytest`.
2. **Obsidian Adapter:** IO module with directory initialization (`mkdir(parents=True)`) and bounded `detect_parent_git(path)`.
3. **Database Setup:** Initialize `vault.db` (`WAL mode`, `busy_timeout=10000`), schemas, indices, and `SqliteSaver`.
4. **FastMCP Lifespan:** Implement `mcp_lifespan` to perform Two-Way Skill Sync (`index_skills_on_startup()`) and spawn `overseer_loop()`.

### Phase 2: Internal Agents (Manager & Librarian Subgraph)
1. **State Definition:** Implement mapped `AgentState` and `LibrarianState`.
2. **The Librarian Subgraph:** A nested graph handling `scan_shelf` mapping `search_result`. 
3. **The Manager Graph:** Orchestrator node routing namespaces. Injects `summary` into system prompt and reads `search_result`.

### Phase 3: Hot Memory Lifecycles & Handoffs
1. **3-Turn Pocket:** Increment `turn_count`. Route to `summarize_history` on turn 4, reset counter.
2. **Super Subs Handoff:** 
   - `route_token_check` enforces caps (`RuntimeError` if > 5 Managers or > 3 Librarians).
   - Generates handoff message, writes to Obsidian, resets `messages`.

### Phase 4: MCP Exposure
1. **Server Instantiation:** `mcp = FastMCP("Contexnt", lifespan=mcp_lifespan)`.
2. **Tool Registration:** `@mcp.tool() async def consult_contexnt(prompt: str, session_id: str | None = None)`. Generate UUID if `None`.
3. **Thread Config:** Maps `{"configurable": {"thread_id": session_id}}`.
4. **Metadata Update:** Run `INSERT OR REPLACE INTO session_metadata`.
5. **Payload Handoff:** Write payload to `active_context_{session_id}.md`, return `{path, summary, session_id}`.

### Phase 5: The Overseer & Caching
1. **In-Process Daemon:** Runs strictly async via `mcp_lifespan`. 
2. **Idle Processing & Cleanup:** Analyzes sessions idle > 600s. Deletes abandoned `active_context_{session_id}.md` files.
3. **Ranking & Bundling:** Build the O(1) `hotbar_cache`.

### Phase 6: Verification & Testing
1. **Pytest Suite:** Create `tests/` using `pytest` and `pytest-asyncio`.
2. **Core Test Coverage:**
   - **Git Traversal:** Verify bounded search stops at `~` and does *not* write to parent `.gitignore`.
   - **Token Rollover:** Mock LLM outputs to exceed limits, assert graph routes to `rollover_node` and throws `RuntimeError` at caps.
   - **3-Turn Pocket:** Assert that `turn_count == 3` triggers the `summarize_history` node and flushes the message list.
   - **DB Concurrency:** Mock concurrent operations to verify WAL mode and `busy_timeout` prevent locking conflicts.
