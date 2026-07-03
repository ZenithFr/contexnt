import os
import logging
import asyncio
from pathlib import Path
from typing import TypedDict, Optional
import tiktoken

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

# Constants
MAX_LIMIT = int(os.getenv("CONTEXNT_MAX_LIMIT", 4000))
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# 1. State Definitions
class AgentState(TypedDict):
    goal: str
    summary: str  
    search_result: str  
    messages: list[BaseMessage]
    turn_count: int
    manager_index: int
    librarian_index: int
    session_id: str

class LibrarianState(TypedDict):
    search_query: str
    search_messages: list[BaseMessage]
    session_id: str
    librarian_index: int
    search_result: str

# 2. Helper Functions
def count_tokens(messages: list[BaseMessage]) -> int:
    num_tokens = 3  # Overhead per completion
    for message in messages:
        num_tokens += 3  # Overhead per message
        if message.content:
            if isinstance(message.content, str):
                num_tokens += len(TOKENIZER.encode(message.content))
            elif isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and "text" in item:
                        num_tokens += len(TOKENIZER.encode(item["text"]))
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                if "args" in tc:
                    num_tokens += len(TOKENIZER.encode(str(tc["args"])))
                if "name" in tc:
                    num_tokens += len(TOKENIZER.encode(tc["name"]))
    return num_tokens

_llm_cache = {}

def get_llm(temperature: float = 0.3):
    if temperature not in _llm_cache:
        api_key = os.getenv("OPENAI_API_KEY", "freellmapi-1e53eb0c02156d62c5660ddcd5fd7e8d674b6bcc77c6dc38")
        api_base = os.getenv("OPENAI_API_BASE", "http://10.0.0.4:5001/v1")
        model_name = os.getenv("OPENAI_MODEL_NAME", "llama-3.3-70b-versatile")
        _llm_cache[temperature] = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature
        )
    return _llm_cache[temperature]

# 3. Librarian Subgraph Tools and Nodes
def create_scan_shelf_tool(db_path: str):
    from langchain_core.runnables import RunnableConfig
    import json
    
    @tool
    async def scan_shelf(namespace: str, keyword: Optional[str] = None, *, config: RunnableConfig = None) -> str:
        """
        Scans a specific bookshelf namespace (e.g. 'coding/python') for skills.
        Optionally filters matching skills containing the keyword.
        """
        from src.db import get_db
        
        # 1. Extract session_id (thread_id) from config
        session_id = "default"
        if config and "configurable" in config:
            session_id = config["configurable"].get("thread_id", "default")
            
        filepaths = []
        db = get_db()
        if not db:
            return "Error: Database connection not found."
            
        # 2. Check hotbar_cache first (O(1) hit)
        async with db.execute("SELECT skills_list FROM hotbar_cache WHERE namespace = ?", (namespace,)) as cursor:
            row = await cursor.fetchone()
        if row:
            try:
                filepaths = json.loads(row[0])
                logger.info(f"scan_shelf: O(1) Hotbar Cache Hit for namespace '{namespace}'")
            except Exception as e:
                logger.error(f"Failed to parse hotbar cache for namespace '{namespace}': {e}")
                
        # 3. Fallback to querying the skills table if cache miss
        if not filepaths:
            async with db.execute("SELECT filepath FROM skills WHERE namespace = ?", (namespace,)) as cursor:
                rows = await cursor.fetchall()
            filepaths = [r[0] for r in rows]
            
        if not filepaths:
            return f"No skills found in namespace: '{namespace}'"
            
        results = []
        for filepath in filepaths:
            path = Path(filepath)
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                    skill_name = path.parent.name
                    
                    # Log skill usage in database
                    await db.execute(
                        "INSERT INTO skill_usage_logs (session_id, skill_name) VALUES (?, ?)",
                        (session_id, skill_name)
                    )
                    
                    if keyword:
                        if keyword.lower() in content.lower():
                            results.append(f"--- Skill from {path.name} (Namespace: {namespace}) ---\n{content}")
                    else:
                        results.append(f"--- Skill from {path.name} (Namespace: {namespace}) ---\n{content}")
                except Exception as e:
                    results.append(f"Error reading skill {path.name}: {e}")
            else:
                results.append(f"Skill file not found on disk: {filepath}")
        await db.commit()
            
        if not results:
            if keyword:
                return f"No skills on shelf '{namespace}' matched keyword '{keyword}'."
            return f"No readable skill files found on shelf '{namespace}'."
            
        return "\n\n".join(results)

async def librarian_rollover(state: LibrarianState):
    current_index = state.get("librarian_index") or 1
    session_id = state.get("session_id", "default")
    
    if current_index >= 3:
        raise RuntimeError("Cognitive limits exceeded: Max Librarian Super Subs reached")
        
    vault_path = Path.home() / "Documents" / "agentic-zen" / "contexnt"
    vault_path.mkdir(parents=True, exist_ok=True)
    
    dump_content = f"# Librarian Handoff Context - Session {session_id} - Index {current_index}\n\n"
    for msg in state["search_messages"]:
        dump_content += f"### {type(msg).__name__}\n{msg.content}\n\n"
        
    filename = f"librarian_{session_id}_{current_index}_context.md"
    file_path = vault_path / filename
    file_path.write_text(dump_content, encoding="utf-8")
    logger.info(f"Librarian: Dumped context to {file_path}")
    
    last_state_summary = f"Librarian-{current_index} completed up to this point. Search query was: {state['search_query']}."
    handoff_content = (
        f"You are librarian-{current_index + 1}. Librarian-{current_index} ran out of context and was archived. "
        f"Resume searching from this state. Handoff details: {last_state_summary}"
    )
    
    system_prompt = (
        "You are the Librarian, an internal search specialist. "
        "Your task is to retrieve skills or documentation from the vault to answer the search query.\n"
        "Use the `scan_shelf` tool to search a specific namespace (e.g. 'coding/python').\n"
        "Once you have retrieved the necessary skills, summarize them clearly and answer the search query.\n"
        "If no skills are found, clearly state that."
    )
    
    new_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=handoff_content)
    ]
    
    return {
        "search_messages": new_messages,
        "librarian_index": current_index + 1
    }

# 4. Compile Librarian Graph
def compile_librarian_graph(db_path: str):
    scan_shelf_tool = create_scan_shelf_tool(db_path)
    
    async def call_librarian_tools(state: LibrarianState):
        messages = list(state.get("search_messages") or [])
        last_message = messages[-1]
        
        new_tool_messages = []
        if last_message.tool_calls:
            for tc in last_message.tool_calls:
                if tc["name"] == "scan_shelf":
                    from langchain_core.runnables import RunnableConfig
                    config = RunnableConfig(configurable={"thread_id": state.get("session_id", "default")})
                    result = await scan_shelf_tool.ainvoke(tc["args"], config=config)
                    
                    tool_msg = ToolMessage(
                        content=str(result),
                        name="scan_shelf",
                        tool_call_id=tc["id"]
                    )
                    new_tool_messages.append(tool_msg)
                    
        return {"search_messages": messages + new_tool_messages}
        
    llm = get_llm().bind_tools([scan_shelf_tool])
    
    async def call_llm(state: LibrarianState):
        messages = list(state.get("search_messages") or [])
        if not messages:
            system_prompt = (
                "You are the Librarian, an internal search specialist. "
                "Your task is to retrieve skills or documentation from the vault to answer the search query.\n"
                "Use the `scan_shelf` tool to search a specific namespace (e.g. 'coding/python').\n"
                "Once you have retrieved the necessary skills, summarize them clearly and answer the search query.\n"
                "If no skills are found, clearly state that."
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Search Query: {state['search_query']}")
            ]
        
        response = await llm.ainvoke(messages)
        return {"search_messages": messages + [response]}
        
    def should_continue(state: LibrarianState):
        messages = state.get("search_messages") or []
        last_message = messages[-1]
        
        if last_message.tool_calls:
            # Loop guard: Max 3 tool execution attempts
            tool_call_msgs = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
            if len(tool_call_msgs) >= 3:
                logger.info("Librarian: Loop guard triggered. Forcing end of search.")
                return END
            return "tools"
        return END
        
    def check_tokens_and_continue(state: LibrarianState):
        messages = state.get("search_messages") or []
        if count_tokens(messages) > MAX_LIMIT:
            return "rollover"
        return should_continue(state)
        
    async def retrieve_result(state: LibrarianState):
        messages = state.get("search_messages") or []
        if not messages:
            return {"search_result": "No results."}
        last_message = messages[-1]
        
        # If the last message has tool calls (indicating loop guard or exit without tool run),
        # combine the contents of all tool messages retrieved so far.
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_results = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tool_results.append(msg.content)
            if tool_results:
                return {"search_result": "\n\n".join(tool_results)}
                
        return {"search_result": last_message.content or "No results found."}
        
    workflow = StateGraph(LibrarianState)
    workflow.add_node("agent", call_llm)
    workflow.add_node("tools", call_librarian_tools)
    workflow.add_node("rollover", librarian_rollover)
    workflow.add_node("retrieve", retrieve_result)
    
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        check_tokens_and_continue,
        {
            "tools": "tools",
            "rollover": "rollover",
            END: "retrieve"
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.add_edge("rollover", "agent")
    workflow.add_edge("retrieve", END)
    
    return workflow.compile()

# 5. Manager Subgraph Tools and Nodes
@tool
def consult_librarian(search_query: str) -> str:
    """
    Invokes the Librarian agent to search the vault shelves for skills or context
    matching the query.
    """
    return ""

async def summarize_history(state: AgentState):
    messages = state.get("messages") or []
    if len(messages) <= 1:
        return {"turn_count": 0}
        
    history_to_summarize = messages[:-1]
    new_prompt_msg = messages[-1]
    
    llm = get_llm()
    prompt = (
        "Write a concise summary of the following conversation history, preserving "
        "all key facts, tasks, and retrieved context:\n\n"
    )
    for msg in history_to_summarize:
        prompt += f"{type(msg).__name__}: {msg.content}\n"
        
    summary_response = await llm.ainvoke([HumanMessage(content=prompt)])
    new_summary = summary_response.content
    
    old_summary = state.get("summary") or ""
    combined_summary = f"{old_summary}\n\n{new_summary}".strip()
    
    logger.info(f"3-Turn Pocket Limit reached. Summarized history. Turn count reset.")
    
    return {
        "summary": combined_summary,
        "messages": [new_prompt_msg],
        "turn_count": 0
    }

async def call_manager_agent(state: AgentState):
    summary = state.get("summary") or ""
    system_prompt = (
        "You are the Manager, the primary orchestrator of the Contex'nt system.\n"
        "Your task is to coordinate with the Librarian to retrieve appropriate skills, "
        "format the context, and answer the user's prompt.\n"
    )
    if summary:
        system_prompt += f"\nSummary of prior conversation:\n{summary}\n"
        
    system_prompt += (
        "\nIf you need to retrieve skills or context from the vault, call the tool "
        "`consult_librarian(search_query)`. Do not guess skill contents.\n"
        "Once you have all the necessary context, formulate your final response to the user."
    )
    
    messages = list(state.get("messages") or [])
    llm_messages = [SystemMessage(content=system_prompt)] + [m for m in messages if not isinstance(m, SystemMessage)]
    
    tools = [consult_librarian]
    llm = get_llm().bind_tools(tools)
    
    response = await llm.ainvoke(llm_messages)
    return {"messages": messages + [response]}

async def manager_rollover(state: AgentState):
    current_index = state.get("manager_index") or 1
    session_id = state.get("session_id", "default")
    
    if current_index >= 5:
        raise RuntimeError("Cognitive limits exceeded: Max Manager Super Subs reached")
        
    vault_path = Path.home() / "Documents" / "agentic-zen" / "contexnt"
    vault_path.mkdir(parents=True, exist_ok=True)
    
    dump_content = f"# Manager Handoff Context - Session {session_id} - Index {current_index}\n\n"
    for msg in state["messages"]:
        dump_content += f"### {type(msg).__name__}\n{msg.content}\n\n"
        
    filename = f"manager_{session_id}_{current_index}_context.md"
    file_path = vault_path / filename
    file_path.write_text(dump_content, encoding="utf-8")
    logger.info(f"Manager: Dumped context to {file_path}")
    
    last_state_summary = f"Manager-{current_index} completed up to this point. Goal was: {state['goal']}."
    handoff_content = (
        f"You are manager-{current_index + 1}. Manager-{current_index} ran out of context and was archived. "
        f"Resume execution from this state. Handoff details: {last_state_summary}"
    )
    
    system_prompt = (
        "You are the Manager, the primary orchestrator of the Contex'nt system.\n"
        "Your task is to coordinate with the Librarian to retrieve appropriate skills, "
        "format the context, and answer the user's prompt."
    )
    
    new_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=handoff_content)
    ]
    
    return {
        "messages": new_messages,
        "manager_index": current_index + 1
    }

_librarian_graph_cache = {}

def get_librarian_graph(db_path: str):
    if db_path not in _librarian_graph_cache:
        _librarian_graph_cache[db_path] = compile_librarian_graph(db_path)
    return _librarian_graph_cache[db_path]

async def call_librarian_subgraph(state: AgentState):
    messages = state.get("messages") or []
    last_message = messages[-1]
    
    search_query = ""
    tool_call_id = None
    if last_message.tool_calls:
        for tc in last_message.tool_calls:
            if tc["name"] == "consult_librarian":
                search_query = tc["args"].get("search_query", "")
                tool_call_id = tc["id"]
                break
                
    if not tool_call_id:
        return {}
        
    vault_path = Path.home() / "Documents" / "agentic-zen" / "contexnt"
    db_path = str(vault_path / "vault.db")
    
    librarian_graph = get_librarian_graph(db_path)
    
    librarian_input = {
        "search_query": search_query,
        "search_messages": [],
        "session_id": state.get("session_id", "default"),
        "librarian_index": state.get("librarian_index") or 1,
        "search_result": ""
    }
    
    librarian_output = await librarian_graph.ainvoke(librarian_input)
    
    search_result = librarian_output.get("search_result") or "No results found."
    new_librarian_index = librarian_output.get("librarian_index") or 1
    
    tool_message = ToolMessage(
        content=search_result,
        name="consult_librarian",
        tool_call_id=tool_call_id
    )
    
    return {
        "messages": messages + [tool_message],
        "search_result": search_result,
        "librarian_index": new_librarian_index
    }

async def loop_guard_fallback(state: AgentState):
    messages = state.get("messages") or []
    # Find all ToolMessages to see what the Librarian returned
    retrieved_info = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            retrieved_info.append(msg.content)
            
    llm = get_llm()
    fallback_prompt = (
        "You are the Manager. You have searched the vault multiple times but hit the safety execution limit. "
        "Based on the retrieved context below, please write a final response to the user's prompt: "
        f"'{state.get('goal')}'\n\n"
        "Retrieved Context:\n" + "\n\n".join(retrieved_info)
    )
    
    response = await llm.ainvoke([
        SystemMessage(content="You must formulate a final response based on the retrieved context."),
        HumanMessage(content=fallback_prompt)
    ])
    
    return {"messages": messages + [response]}

# 6. Compile Manager Graph
def compile_manager_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("manager_agent", call_manager_agent)
    workflow.add_node("summarize_history", summarize_history)
    workflow.add_node("call_librarian_subgraph", call_librarian_subgraph)
    workflow.add_node("rollover", manager_rollover)
    workflow.add_node("loop_guard_fallback", loop_guard_fallback)
    
    def route_start(state: AgentState):
        if state.get("turn_count", 0) >= 3:
            return "summarize_history"
        return "manager_agent"
        
    def check_tokens_and_continue_manager(state: AgentState):
        messages = state.get("messages") or []
        if count_tokens(messages) > MAX_LIMIT:
            return "rollover"
            
        last_message = messages[-1]
        
        if last_message.tool_calls:
            # Loop guard: Max 3 consult_librarian tool executions
            tool_call_msgs = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
            if len(tool_call_msgs) >= 3:
                logger.info("Manager: Loop guard triggered. Forcing end of task.")
                return "loop_guard_fallback"
            return "call_librarian_subgraph"
        return END
        
    workflow.add_conditional_edges(START, route_start, {
        "summarize_history": "summarize_history",
        "manager_agent": "manager_agent"
    })
    
    workflow.add_edge("summarize_history", "manager_agent")
    
    workflow.add_conditional_edges("manager_agent", check_tokens_and_continue_manager, {
        "rollover": "rollover",
        "call_librarian_subgraph": "call_librarian_subgraph",
        "loop_guard_fallback": "loop_guard_fallback",
        END: END
    })
    
    workflow.add_edge("rollover", "manager_agent")
    workflow.add_edge("call_librarian_subgraph", "manager_agent")
    workflow.add_edge("loop_guard_fallback", END)
    
    return workflow.compile(checkpointer=checkpointer)
