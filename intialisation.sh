#!/usr/bin/env bash
set -euo pipefail

# Initializes the MCP-Local-LLM stack:
# 1) Installs frontend Node dependencies (so the UI server can start)
# 2) Builds the backend container (downloads Python requirements inside the image)
# 3) Starts the stack via docker compose

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not installed. Install Docker first." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose v2 is required (try: docker --version and docker compose version)." >&2
  exit 1
fi

echo "==> Installing frontend dependencies (npm install)..."
if [ -d "$ROOT_DIR/frontend" ] && [ -f "$ROOT_DIR/frontend/package.json" ]; then
  (cd "$ROOT_DIR/frontend" && npm install)
else
  echo "Frontend package.json not found; skipping npm install."
fi

echo "==> Building backend container (installs Python requirements into the image)..."
docker compose build

echo "==> Starting containers..."
docker compose up -d

echo "==> Current container status:"
docker compose ps

echo ""
echo "Stack is starting. Backend: http://localhost:8000, Frontend: http://localhost:3000"
