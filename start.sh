#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
docker compose up -d node-agent vllm vector
echo "Runtime started. Follow logs with: docker compose logs -f node-agent"
