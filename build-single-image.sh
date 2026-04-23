#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

image_name="${1:-anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest}"
build_args=()

if [[ "${PRELOAD_HF_MODELS+x}" == x ]]; then
  build_args+=(--build-arg "PRELOAD_HF_MODELS=${PRELOAD_HF_MODELS}")
fi

python ./scripts/generate_model_artifacts_manifest.py --check

echo "Building Linux NVIDIA single-container image ${image_name} ..."
docker build --platform linux/amd64 "${build_args[@]}" -f Dockerfile.single -t "${image_name}" .
echo "Built ${image_name}"
