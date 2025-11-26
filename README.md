# MCP-Local-LLM

## Quick start

1. Make sure Docker (with Compose v2) and Node.js are installed.
2. Copy `.env` and adjust any variables you need (defaults point to Ollama on the host via `http://host.docker.internal:11434`; ensure Ollama is running there and the model is pulled).
3. Run the initializer to install frontend deps, build the backend image (installing Python requirements inside), and start the stack:
   ```bash
   ./intialisation.sh
   ```
4. Visit the UI at `http://localhost:3000`. The backend API is available at `http://localhost:8000`.

If you use a different model or Ollama host, update `MODEL` and `OLLAMA_BASE_URL` in `.env`. Make sure the model is pulled in Ollama (e.g., `ollama pull gpt-oss:20b`) before starting the stack.
