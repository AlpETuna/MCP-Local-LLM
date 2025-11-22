# MCP-Local-LLM

## Quick start

1. Make sure Docker (with Compose v2) and Node.js are installed.
2. Copy `.env` and adjust any variables you need (defaults work for a local Ollama at `http://localhost:11434`).
3. Run the initializer to install frontend deps, build the backend image (installing Python requirements inside), and start the stack:
   ```bash
   ./intialisation.sh
   ```
4. Visit the UI at `http://localhost:3000`. The backend API is available at `http://localhost:8000`.
