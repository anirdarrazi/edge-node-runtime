#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python 3.11 or newer is required to stop the local node service." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  echo "Python 3.11 or newer is required to stop the local node service." >&2
  exit 1
fi

VENV_PATH=".installer-venv"
if [ ! -d "$VENV_PATH" ]; then
  echo "The local node service virtual environment does not exist yet." >&2
  exit 1
fi

VENV_PYTHON="$VENV_PATH/bin/python"
echo "Stopping the local node runtime service..."
"$VENV_PYTHON" -m node_agent.service stop
