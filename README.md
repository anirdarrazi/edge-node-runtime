# Edge Node Runtime

Python runtime for AUTONOMOUSc edge nodes.

## Open-source security model

This runtime is designed to be publishable as open-source code. The control plane assumes the node owner can inspect and modify the runtime, so trust is enforced server-side instead of by hiding client behavior.

- Community and self-hosted nodes enroll as `untrusted` by default.
- Only first-party nodes and admin-approved partner nodes can be promoted to the `trusted` execution tier.
- Premium `trusted_only` workloads are routed by the control plane and require server-owned trust state plus canary verification for partner nodes.
- The runtime emits a `runtime_receipt` with assignment nonce, declared model, runtime image digest, model manifest digest, tokenizer digest, and aggregated usage. This receipt is audit evidence only and does not grant trust by itself.
- Sensitive files such as `.env`, runtime data, credentials, and diagnostics should stay local and are excluded from the standalone repo with `.gitignore`.

## Supported setup

The runtime has one supported setup path: Docker. There is no installer wizard, browser Quick Start, repo-local app launcher, or generated `./data/service/runtime.env` path.

Prerequisites:

- Docker with NVIDIA GPU support.
- An NVIDIA GPU and driver that can run the selected inference image.
- A configured `.env` based on `.env.example`.

Create the environment file:

```bash
cp .env.example .env
```

Set `NODE_ID` and `NODE_KEY` in `.env` when you already have approved node credentials. For a one-time interactive claim, run the bootstrap command in Docker and keep the same `autonomousc-edge` volume mounted:

```bash
docker run --rm -it \
  --env-file .env \
  -v autonomousc-edge:/var/lib/autonomousc \
  --entrypoint node-agent-bootstrap \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

Run the runtime:

```bash
docker run --rm \
  --gpus all \
  --env-file .env \
  -p 8000:8000 \
  -p 8011:8011 \
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

The container starts the inference server and node agent directly. Startup status is exposed at `http://127.0.0.1:8011/startup-status`; the inference API is exposed on `http://127.0.0.1:8000`.

## Docker Compose

For local development with separate services, copy `.env.example` to `.env`, fill the required identity and model values, then run:

```bash
docker compose up
```

To claim credentials through the terminal bootstrap flow before starting the full stack:

```bash
docker compose run --rm node-agent-bootstrap
```

Compose is a development convenience. The public runtime image remains the normal deployment artifact.

## Configuration

Important `.env` values:

- `EDGE_CONTROL_URL`: control plane URL, defaulting to production.
- `NODE_ID` and `NODE_KEY`: preferred node identity values.
- `OPERATOR_TOKEN`: legacy headless enrollment fallback only.
- `RUNTIME_PROFILE`, `DEPLOYMENT_TARGET`, and `INFERENCE_ENGINE`: runtime profile selection.
- `VLLM_MODEL`, `SUPPORTED_MODELS`, `MAX_CONTEXT_TOKENS`, and `MAX_CONCURRENT_ASSIGNMENTS`: model and capacity settings.
- `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN`: optional token for gated model access.

The container stores credentials, scratch state, and model cache under `/var/lib/autonomousc`, so keep that path on a persistent Docker volume.

## RTX 5060 Ti Gemma Profile

The Vast.ai profile for a 16 GB RTX 5060 Ti node uses `google/gemma-4-E4B-it` through vLLM:

```env
RUNTIME_PROFILE=rtx_5060_ti_16gb_gemma4_e4b
DEPLOYMENT_TARGET=vast_ai
INFERENCE_ENGINE=vllm
RUNTIME_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest
CAPACITY_CLASS=elastic_burst
TEMPORARY_NODE=false
BURST_PROVIDER=vast_ai
GPU_NAME=RTX 5060 Ti
GPU_MEMORY_GB=16
MAX_CONTEXT_TOKENS=8192
MAX_BATCH_TOKENS=8192
MAX_CONCURRENT_ASSIGNMENTS=2
MAX_CONCURRENT_ASSIGNMENTS_CAP=2
VLLM_STARTUP_TIMEOUT_SECONDS=1800
VLLM_MODEL=google/gemma-4-E4B-it
SUPPORTED_MODELS=google/gemma-4-E4B-it
```

## Build and Publish

Build the public runtime image:

```bash
bash build-manager-image.sh
```

Windows PowerShell:

```powershell
.\build-manager-image.ps1
```

Publish the public `latest` image:

```bash
bash publish-latest-image.sh
```

Windows PowerShell:

```powershell
.\publish-latest-image.ps1
```

The image preloads the public bootstrap model cache by default. Override or disable that during build with `PRELOAD_HF_MODELS`, for example:

```bash
PRELOAD_HF_MODELS=BAAI/bge-large-en-v1.5 bash build-manager-image.sh
PRELOAD_HF_MODELS= bash build-manager-image.sh
```

## Repository Layout

- `src/node_agent`: node enrollment, polling, assignment execution, and reporting.
- `Dockerfile.service`: public Docker runtime image.
- `Dockerfile.single`: compatibility single-container runtime build.
- `docker-compose.yml`: local development stack.
- `.env.example`: Docker runtime environment template.
- `build-manager-image.*`: build the public image.
- `publish-latest-image.*`: build and push `anirdarrazi/autonomousc-ai-edge-runtime:latest`.

Run tests with:

```bash
python -m pytest
```
