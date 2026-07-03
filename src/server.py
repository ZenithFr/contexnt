import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
import aiosqlite

from src.db import init_db
from src.obsidian import init_obsidian_vault, write_context_payload
from src.skills import index_skills_on_startup
from src.overseer import overseer_loop
from src.agents import compile_manager_graph, AgentState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core Paths
VAULT_PATH = Path.home() / "Documents" / "agentic-zen" / "contextnt"
DB_PATH = str(VAULT_PATH / "vault.db")

# Global Shutdown Event
shutdown_event = asyncio.Event()
overseer_task = None

@asynccontextmanager
async def mcp_lifespan(app: FastMCP):
    """Lifecycle hook for Phase 1: Foundation and The Vault."""
    global overseer_task
    
    logger.info("Initializing Context'nt FastMCP Server...")
    
    # Phase 1.2: Obsidian Adapter Initialization
    init_obsidian_vault(VAULT_PATH)
    
    # Phase 1.3: Database Setup
    await init_db(DB_PATH)
    
    # Phase 1.4: Two-Way Skill Sync
    await index_skills_on_startup(DB_PATH, VAULT_PATH)
    
    # Phase 5.1: Launch Overseer Task
    overseer_task = asyncio.create_task(overseer_loop(DB_PATH, VAULT_PATH, shutdown_event))
    
    logger.info("Server started. Handing control to FastMCP.")
    yield {}
    
    # Teardown
    logger.info("Initiating server shutdown...")
    shutdown_event.set()
    if overseer_task:
        await overseer_task
    logger.info("Server shutdown complete.")

# Instantiation
mcp = FastMCP("Contextnt", lifespan=mcp_lifespan)

# Phase 4.2: FastMCP Tool Registration
@mcp.tool()
async def consult_contextnt(prompt: str, session_id: Optional[str] = None) -> dict:
    """
    Consults Context'nt to retrieve context payloads for a given prompt/session.
    If session_id is not provided, one is generated automatically.
    
    Returns a dictionary containing:
      - path: Path to the generated active context file in the Obsidian vault.
      - summary: A summary of the conversation up to this point.
      - session_id: The session ID used for this request.
    """
    # 1. Generate session_id if missing
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # 2. Update session metadata for Overseer idle-detection
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO session_metadata (session_id, last_active) VALUES (?, CURRENT_TIMESTAMP)",
            (session_id,)
        )
        await db.commit()
        
    # 3. Invoke Manager Graph with Checkpointer
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = compile_manager_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        
        # Check current turn_count and index counts
        state_snapshot = await graph.aget_state(config)
        if state_snapshot.values:
            current_turn = state_snapshot.values.get("turn_count", 0)
            next_turn = current_turn + 1
            manager_index = state_snapshot.values.get("manager_index", 1)
            librarian_index = state_snapshot.values.get("librarian_index", 1)
            goal = state_snapshot.values.get("goal") or prompt
        else:
            next_turn = 1
            manager_index = 1
            librarian_index = 1
            goal = prompt
            
        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "turn_count": next_turn,
            "manager_index": manager_index,
            "librarian_index": librarian_index,
            "goal": goal,
            "session_id": session_id
        }
        
        outputs = await graph.ainvoke(inputs, config)
        
    # 4. Extract final response message
    output_messages = outputs.get("messages") or []
    final_payload = ""
    if output_messages:
        final_payload = output_messages[-1].content
        
    # 5. Write payload to Obsidian
    payload_path = await write_context_payload(VAULT_PATH, session_id, final_payload)
    
    # 6. Extract summary
    summary = outputs.get("summary") or ""
    
    return {
        "path": str(payload_path.resolve()),
        "summary": summary,
        "session_id": session_id
    }

if __name__ == "__main__":
    mcp.run()
