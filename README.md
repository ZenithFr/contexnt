# Contex'nt 2.0 🧠

Contex'nt is an advanced **6-Layer Cognitive Architecture** and FastMCP server designed to provide deep, conversational agent capabilities with autonomous memory management, multi-turn history preservation, and cognitive token constraints. 

By leveraging LangGraph and a local Obsidian vault backend, Contex'nt isolates subgraphs to prevent state collision, dynamically indexes file-backed "skills", and gracefully passes off context using a Super Subs handoff system before hitting token limitations.

## Features
- 🚀 **FastMCP Tool Support**: Out of the box MCP Server deployment for Claude Desktop, Cursor, or your local agent.
- 💾 **Obsidian Vault Backend**: Uses `agentic-zen/contexnt/` in your local Documents directory for permanent, human-readable storage.
- ♻️ **Two-Way Skill Sync**: Dynamically syncs `.md` skills directly into an SQLite WAL-enabled `vault.db`.
- 🔐 **Connection Pooling**: 0ms latency database accesses with a centralized async SQLite pool.
- 🧠 **Manager & Librarian Subgraphs**: Specialized node isolation to safely parse goals and query shelves.
- 🛡️ **Auto-Memory Snapshots**: A background Overseer automatically triggers a hidden Git repository backup on cold memory when sessions go idle.

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```bash
git clone https://github.com/ZenithFr/contexnt.git
cd contexnt

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the `.env.example` file to create your own configuration:
```bash
cp .env.example .env
```
Ensure your `OPENAI_API_KEY` is correctly set. The default points to a local/custom OpenAI-compatible API base (e.g. vLLM or local server).

### 4. Running the Server

#### Desktop Integration (Claude Desktop / Cursor)
Check the `docs/claude_desktop_config.json` snippet to add the FastMCP server into your desktop IDE.

#### Daemon / Docker Deployment
```bash
docker build -t contexnt-server .
docker run -d --name contexnt contexnt-server
```

## Creating Custom Skills
Contex'nt acts as a "Librarian" scanning bookshelves (namespaces). To add a skill to your system, see `examples/example_skill.md`. Place these in your specified vault directory, and the Overseer daemon will automatically sync them into the SQLite database for O(1) retrieval!
