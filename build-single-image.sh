#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

image_name="${1:-anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest}"

python ./scripts/generate_model_artifacts_manifest.py --check

echo "Building Linux NVIDIA single-container image ${image_name} ..."
docker build --platform linux/amd64 -f Dockerfile.single -t "${image_name}" .
echo "Built ${image_name}"
