import os
import shutil
import pytest
import asyncio
import aiosqlite
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.obsidian import detect_parent_git, init_obsidian_vault
from src.agents import compile_manager_graph, count_tokens, AgentState
from src.db import init_db

# 1. Mock LLM class for testing LangGraph without API keys
class MockChatOpenAI:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        
    def bind_tools(self, tools):
        return self
        
    async def ainvoke(self, messages, *args, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
        else:
            resp = AIMessage(content="Default mock response")
        return resp

# 2. Test Git Traversal and Safety
def test_detect_parent_git(tmp_path):
    # Setup mock home directory
    mock_home = tmp_path / "home_user"
    mock_home.mkdir()
    
    # Setup mock vault path
    vault_path = mock_home / "Documents" / "agentic-zen" / "contextnt"
    vault_path.mkdir(parents=True)
    
    # Patch Path.home() to point to our mock_home
    with patch("pathlib.Path.home", return_value=mock_home):
        # Scenario A: No parent Git repo
        assert detect_parent_git(vault_path) is None
        
        # Scenario B: Git repo at mock_home
        (mock_home / ".git").mkdir()
        assert detect_parent_git(vault_path) == mock_home
        
        # Scenario C: Git repo at system root or above home (should stop at home)
        # We delete home-level .git and create parent-level .git
        shutil.rmtree(mock_home / ".git")
        (tmp_path / ".git").mkdir()
        # Traversal should stop at mock_home and return None
        assert detect_parent_git(vault_path) is None

def test_init_obsidian_vault_home_safety(tmp_path):
    mock_home = tmp_path / "home_user"
    mock_home.mkdir()
    vault_path = mock_home / "Documents" / "agentic-zen" / "contextnt"
    
    # Git at home folder
    (mock_home / ".git").mkdir()
    
    with patch("pathlib.Path.home", return_value=mock_home):
        # Run init
        init_obsidian_vault(vault_path)
        # Verify .gitignore at home was NOT created or modified
        assert not (mock_home / ".gitignore").exists()

# 3. Test 3-Turn Pocket Summarization
@pytest.mark.asyncio
async def test_3_turn_pocket_summarize(tmp_path):
    db_path = tmp_path / "vault.db"
    await init_db(str(db_path))
    
    # Mock LLM to return a summary response
    mock_summary = AIMessage(content="This is the summarized history.")
    mock_manager_resp = AIMessage(content="Manager response to prompt.")
    llm_mock = MockChatOpenAI(responses=[mock_summary, mock_manager_resp])
    
    with patch("src.agents.get_llm", return_value=llm_mock):
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        graph = compile_manager_graph(checkpointer=memory)
        
        # Scenario: turn_count = 3 (this incoming human prompt will trigger turn 4)
        config = {"configurable": {"thread_id": "test_session"}}
        inputs = {
            "messages": [
                HumanMessage(content="old turn 1"),
                AIMessage(content="resp 1"),
                HumanMessage(content="old turn 2"),
                AIMessage(content="resp 2"),
                HumanMessage(content="old turn 3"),
                AIMessage(content="resp 3"),
                HumanMessage(content="new turn 4 prompt")
            ],
            "turn_count": 4,
            "manager_index": 1,
            "librarian_index": 1,
            "goal": "Test summarization flow",
            "session_id": "test_session"
        }
        
        outputs = await graph.ainvoke(inputs, config)
        
        # Verify messages flushed: should contain only new prompt + manager response
        messages = outputs["messages"]
        assert len(messages) == 2
        assert messages[0].content == "new turn 4 prompt"
        
        # Verify summary populated
        assert outputs["summary"] == "This is the summarized history."
        # Verify turn_count reset
        assert outputs["turn_count"] == 0

# 4. Test Token Rollover & Hard Caps
@pytest.mark.asyncio
async def test_manager_token_rollover_and_caps(tmp_path):
    db_path = tmp_path / "vault.db"
    await init_db(str(db_path))
    
    # Setup mock home to write handoff context files to Obsidian
    mock_home = tmp_path / "home_user"
    mock_home.mkdir()
    
    # Mock LLM to return standard responses
    llm_mock = MockChatOpenAI(responses=[AIMessage(content="Mocked Response")])
    
    # Lower MAX_LIMIT to 150 tokens for testing to easily trigger rollover
    with patch("src.agents.MAX_LIMIT", 150), \
         patch("pathlib.Path.home", return_value=mock_home), \
         patch("src.agents.get_llm", return_value=llm_mock):
         
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        graph = compile_manager_graph(checkpointer=memory)
        
        # 1. Normal rollover trigger
        config = {"configurable": {"thread_id": "test_rollover"}}
        inputs = {
            "messages": [HumanMessage(content="This prompt contains many tokens. " * 30)],
            "turn_count": 1,
            "manager_index": 1,
            "librarian_index": 1,
            "goal": "Test rollover",
            "session_id": "test_rollover"
        }
        
        outputs = await graph.ainvoke(inputs, config)
        
        # Verifications
        # Message list reset to contains system + handoff + manager response
        assert outputs["manager_index"] == 2
        assert len(outputs["messages"]) == 3
        assert isinstance(outputs["messages"][0], SystemMessage)
        assert isinstance(outputs["messages"][1], HumanMessage)
        assert "archived" in outputs["messages"][1].content
        
        # File written in mock Obsidian
        obsidian_dir = mock_home / "Documents" / "agentic-zen" / "contextnt"
        dump_file = obsidian_dir / "manager_test_rollover_1_context.md"
        assert dump_file.exists()
        assert "Manager Handoff Context" in dump_file.read_text()
        
        # 2. Assert exception is raised when limit (5) is exceeded
        # We manually invoke the rollover node with manager_index=5
        from src.agents import manager_rollover
        state = {
            "messages": [HumanMessage(content="excessive tokens")],
            "manager_index": 5,
            "session_id": "test_cap",
            "goal": "Test Cap"
        }
        with pytest.raises(RuntimeError, match="Cognitive limits exceeded"):
            await manager_rollover(state)

@pytest.mark.asyncio
async def test_librarian_token_rollover_caps(tmp_path):
    from src.agents import librarian_rollover
    
    mock_home = tmp_path / "home_user"
    mock_home.mkdir()
    
    with patch("pathlib.Path.home", return_value=mock_home):
        state = {
            "search_messages": [HumanMessage(content="excessive tokens")],
            "librarian_index": 3,
            "session_id": "test_lib_cap",
            "search_query": "Test query"
        }
        with pytest.raises(RuntimeError, match="Cognitive limits exceeded"):
            await librarian_rollover(state)

# 5. Test SQLite WAL Concurrency
@pytest.mark.asyncio
async def test_sqlite_wal_concurrency(tmp_path):
    db_path = str(tmp_path / "vault.db")
    await init_db(db_path)
    
    # Define parallel tasks to read and write database to prove WAL & timeout work without locking
    async def db_writer(task_id):
        async with aiosqlite.connect(db_path, timeout=10.0) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA busy_timeout=10000;")
            for i in range(50):
                await db.execute(
                    "INSERT INTO skill_usage_logs (session_id, skill_name) VALUES (?, ?)",
                    (f"session_{task_id}", f"skill_{i}")
                )
                await db.commit()
                await asyncio.sleep(0.001)
                
    async def db_reader():
        async with aiosqlite.connect(db_path, timeout=10.0) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA busy_timeout=10000;")
            for _ in range(50):
                async with db.execute("SELECT COUNT(*) FROM skill_usage_logs") as cursor:
                    await cursor.fetchone()
                await asyncio.sleep(0.001)
                
    # Run 5 concurrent writers and 3 readers
    tasks = [db_writer(i) for i in range(5)] + [db_reader() for _ in range(3)]
    # All tasks should execute without raising sqlite3.OperationalError (Database Locked)
    await asyncio.gather(*tasks)
