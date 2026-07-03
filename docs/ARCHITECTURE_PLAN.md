# Context'nt 2.0 - The 6-Layer Cognitive Architecture

This document defines the blueprint for an industry-grade, autonomous context management system built as an MCP server. It acts as an external brain for the primary agent (e.g., Hermes), completely insulating it from context exhaustion.

## The 6 Layers

### Layer 1: The Vault (The Categorized Bookshelf)
**Function:** The raw storage layer. 
**Contents:** All `SKILL.md` files, MCP tool schemas, and local memory from the Obsidian Vault. 
**Mechanism:** Highly structured into specific namespaces (e.g., `coding/python`, `writing/emails`). This categorization is critical for latency: the Librarian only searches the relevant "shelf" instead of the entire library.

### Layer 2: The Librarian (Retrieval Agent)
**Function:** The internal search specialist.
**Contents:** A static-context LLM living *inside* the MCP server. 
**Mechanism:** Translates complex requests into precise queries against Layer 1. Because its toolset is static and it only scans specific shelves, it uses very few tokens and returns answers lightning fast.

### Layer 3: The Manager (MCP Orchestrator)
**Function:** The bridge between Hermes and the Librarian.
**Contents:** The primary internal LLM of the MCP server.
**Mechanism:** Receives Hermes' request, routes the Librarian to the correct shelf, receives the raw data, formats it, and returns the contextual payload to Hermes.

### Layer 4: Hot Memory (The 3-Turn Pocket)
**Function:** The immediate working context.
**Contents:** A rolling summary of the current session.
**Mechanism:** Hot Memory is kept alive for exactly **3 turns**. On the 4th turn, it is pushed entirely into Cold Memory, and a fresh pocket begins. This enforces a strict ceiling on active context bloat.

### Layer 5: Cold Memory (Obsidian Archive)
**Function:** Long-term session storage.
**Contents:** Archival logs, exhausted agent contexts, and popped Hot Memories.
**Mechanism:** Written directly to the default public Obsidian vault inside a dedicated `agentic-zen/contextnt/` folder. We will use a hidden Git repo on this folder to protect against accidental human deletion or modification.

### Layer 6: The Overseer (Post-Processing Async Agent)
**Function:** Continuous Improvement and Maintenance.
**Mechanism:** A background daemon that wakes up after a session. It analyzes Cold Memory to update `SKILL.md` files and global primary memory.

---

## The Overflow Protocol (Super Subs)
To preserve 100% data fidelity, we **never** compact or summarize an internal agent's active context if it fills up.
If an internal agent reaches its token limit, we execute a "Super Sub" handoff:
1. Dump the exhausted agent's entire context to Layer 5 (e.g., `librarian-1-context.md`).
2. Spawn a fresh agent (`librarian-2`) to take over.
**Limits:**
- Manager Super Subs: Max 5 per session (`manager-1` to `manager-5`).
- Librarian Super Subs: Max 3 per session (`librarian-1` to `librarian-3`).

---

## Advanced Feature: The Bundling Strategy
*(To be built after the core engine)*
To further reduce Librarian latency to near zero, the Overseer will implement predictive caching:
1. **Bundling:** Group frequently co-used skills into bundles (minimum 5 skills per bundle).
2. **Eligibility:** A bundle must be observed for 5 days and across at least 10 sessions to be eligible.
3. **The Hotbar:** The top 5 ranked eligible bundles are placed on a "Hotbar".
4. **Retrieval:** The Librarian checks the Hotbar first for instant retrieval before searching shelves. 
5. **Continuous Ranking:** The Overseer constantly compares new potential bundles against the Hotbar, swapping them out to approach ideal efficiency over time.
