#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to install the edge node runtime." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose v2 is required to install the edge node runtime." >&2
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

echo "Starting the local model runtime..."
docker compose up -d vllm

echo "Launching the interactive node claim flow..."
docker compose run --rm node-agent-bootstrap

echo "Starting the long-running runtime services..."
docker compose up -d node-agent vector

echo "Installation complete. Follow runtime logs with: docker compose logs -f node-agent"
