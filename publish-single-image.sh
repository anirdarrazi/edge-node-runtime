#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

image_name="${1:-anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest}"

./build-single-image.sh "${image_name}"

echo "Pushing ${image_name} ..."
docker push "${image_name}"
echo "Pushed ${image_name}"
docker buildx imagetools inspect "${image_name}"
