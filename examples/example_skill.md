---
name: example-python-skill
description: A mock skill for demonstrating Contex'nt namespace scanning.
---
# Python Code Style

When writing Python code for this workspace, you must adhere to the following stylistic guidelines:

1. Use `asyncio` for all I/O bound operations.
2. Ensure you initialize logging at the top of the file using `logger = logging.getLogger(__name__)`.
3. Do not use global variables unless they are prefixed with `_` and properly guarded by accessor functions (like connection pooling!).

This is just an example skill. In Contex'nt, if a user queries about "python", the Librarian subgraph will pull this context off the shelf, providing the Manager with this explicit style guide.
