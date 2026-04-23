#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

image_name="${1:-anirdarrazi/autonomousc-ai-edge-runtime:latest}"
build_args=()

if [[ "${PRELOAD_HF_MODELS+x}" == x ]]; then
  build_args+=(--build-arg "PRELOAD_HF_MODELS=${PRELOAD_HF_MODELS}")
fi

python ./scripts/generate_model_artifacts_manifest.py --check

echo "Building unified Linux runtime image ${image_name} ..."
docker build "${build_args[@]}" -f Dockerfile.service -t "${image_name}" .
echo "Built ${image_name}"
