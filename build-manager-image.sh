#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

image_name="${1:-anirdarrazi/autonomousc-ai-edge-runtime:latest}"

python ./scripts/generate_model_artifacts_manifest.py --check

echo "Building Linux runtime manager image ${image_name} ..."
docker build -f Dockerfile.service -t "${image_name}" .
echo "Built ${image_name}"
