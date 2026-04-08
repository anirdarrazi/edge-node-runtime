#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python 3.11 or newer is required to launch the local node service." >&2
  exit 1
fi

VENV_PATH=".installer-venv"
if [ ! -d "$VENV_PATH" ]; then
  echo "Creating service virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

VENV_PYTHON="$VENV_PATH/bin/python"
echo "Ensuring local node service dependencies are installed..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e .

echo "Starting the local node runtime service..."
"$VENV_PYTHON" -m node_agent.service start --open
