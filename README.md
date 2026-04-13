# Edge Node Runtime

Python runtime for AUTONOMOUSc edge nodes.

## Open-source security model

This runtime is designed to be publishable as open-source code. The control plane assumes the node owner can inspect and modify the runtime, so trust is enforced server-side instead of by hiding client behavior.

- Community and self-hosted nodes enroll as `untrusted` by default.
- Only first-party nodes and admin-approved partner nodes can be promoted to the `trusted` execution tier.
- Premium `trusted_only` workloads are routed by the control plane and require server-owned trust state plus canary verification for partner nodes.
- The runtime now emits a `runtime_receipt` with assignment nonce, declared model, runtime image digest, model manifest digest, tokenizer digest, and aggregated usage. This receipt is audit evidence only and does not grant trust by itself.
- Sensitive files such as `.env`, runtime data, credentials, and diagnostics should stay local and are excluded from the standalone repo with `.gitignore`.

Contents:

- `src/node_agent`: node enrollment, polling, assignment execution, and reporting
- `Dockerfile`: container image build for the node agent
- `Dockerfile.service`: owner-facing runtime manager image
- `Dockerfile.single`: single-container runtime for Vast.ai Docker-style hosts
- `docker-compose.yml`: local appliance runtime with `vllm` and `vector`
- `.env.example`: advanced-mode environment override template
- `build-manager-image.ps1` / `build-manager-image.sh`: build the runtime manager image

## Single-container runtime

Use `Dockerfile.single` when the node host gives you one Docker container, such as a Vast.ai Docker instance. This image runs `vllm` and `node-agent` in the same container, stores credentials and model cache in mounted volumes, and does not require Docker Compose or `/var/run/docker.sock`.

Build locally:

```bash
docker build -f Dockerfile.single -t autonomousc-edge-node-single:dev .
```

Build the published NVIDIA single-container image tag for Docker Hub:

```bash
bash build-single-image.sh
```

Windows PowerShell:

```powershell
.\build-single-image.ps1
```

Publish the NVIDIA single-container image to Docker Hub without changing the existing manager `latest` tag:

```bash
bash publish-single-image.sh
```

Windows PowerShell:

```powershell
.\publish-single-image.ps1
```

The default published tag is `anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest`.

Run with an operator token for first-time enrollment:

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -e EDGE_CONTROL_URL=https://edge.autonomousc.com \
  -e OPERATOR_TOKEN=<operator-token> \
  -e NODE_LABEL="Vast test node" \
  -e NODE_REGION=us-vast-1 \
  -e VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  -e SUPPORTED_MODELS=meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5 \
  -e HUGGING_FACE_HUB_TOKEN=<hf-token-if-needed> \
  -v autonomousc-edge-credentials:/var/lib/autonomousc/credentials \
  -v autonomousc-edge-scratch:/var/lib/autonomousc/scratch \
  -v autonomousc-hf-cache:/root/.cache/huggingface \
  autonomousc-edge-node-single:dev
```

Or pull the published NVIDIA single-container image directly:

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -e EDGE_CONTROL_URL=https://edge.autonomousc.com \
  -e OPERATOR_TOKEN=<operator-token> \
  -e NODE_LABEL="Vast test node" \
  -e NODE_REGION=us-vast-1 \
  -e VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  -e SUPPORTED_MODELS=meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5 \
  -e HUGGING_FACE_HUB_TOKEN=<hf-token-if-needed> \
  -v autonomousc-edge-credentials:/var/lib/autonomousc/credentials \
  -v autonomousc-edge-scratch:/var/lib/autonomousc/scratch \
  -v autonomousc-hf-cache:/root/.cache/huggingface \
  anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest
```

Run with pre-issued node credentials instead of an operator token:

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  -e EDGE_CONTROL_URL=https://edge.autonomousc.com \
  -e NODE_ID=<node-id> \
  -e NODE_KEY=<node-key> \
  -e VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  -v autonomousc-edge-credentials:/var/lib/autonomousc/credentials \
  -v autonomousc-edge-scratch:/var/lib/autonomousc/scratch \
  -v autonomousc-hf-cache:/root/.cache/huggingface \
  autonomousc-edge-node-single:dev
```

Useful single-container overrides:

- `VLLM_EXTRA_ARGS` adds vLLM server flags, for example `--gpu-memory-utilization 0.85 --max-model-len 8192`.
- `START_VLLM=false` skips the bundled vLLM process and points `node-agent` at `VLLM_BASE_URL`.
- `NODE_AGENT_COMMAND` defaults to `node-agent start`; set it only for advanced debugging.
- `VLLM_STARTUP_TIMEOUT_SECONDS` defaults to `600` because first model warm-up can be slow.

The owner-manager UI remains the best local desktop path. `Dockerfile.single` is the best one-container path for rented GPU containers.

## Owner Install Target

The owner-facing distribution path is the runtime manager Docker container. It orchestrates the runtime stack through the host Docker socket while the repo-local scripts remain available for development.

### Runtime manager container

Build and publish the runtime manager image:

```bash
bash build-manager-image.sh
```

Or on Windows while preparing a release:

```powershell
.\build-manager-image.ps1
```

Once the image is published, a node owner can launch it with:

```bash
docker run -d \
  --name autonomousc-node-runtime-manager \
  -p 8765:8765 \
  --add-host=host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v autonomousc-edge-manager:/var/lib/autonomousc/manager \
  anirdarrazi/autonomousc-ai-edge-runtime@sha256:4662922dd7912bbd928f0703e27472829cacc0a858732a2d48caa167a96561db
```

Then open `http://127.0.0.1:8765` and complete the guided setup.

Notes:

- The manager container bundles the runtime UI and owner compose assets.
- The same published image is also used for the bundled `node-agent` service inside the runtime bundle.
- The actual runtime services still run as sibling containers on the host Docker engine.
- The manager container talks to host-published services through `host.docker.internal`, which is why the `--add-host` flag is required on Linux.

## Repo-local install

Bootstrap and run:

```bash
bash install.sh
```

Windows PowerShell:

```powershell
.\install.ps1
```

Windows double-click launcher:

`Install AUTONOMOUSc Edge Node.cmd`

The install scripts now launch the local node service in your browser. The service handles first-time setup and ongoing operations from one place. It:

- checks Docker and GPU prerequisites
- saves structured local settings in `./data/service/runtime-settings.json`
- generates `./data/service/runtime.env` for the runtime automatically
- starts `vllm`
- creates the node claim and waits for browser approval
- stores credentials locally in `./data/credentials`
- starts `node-agent` and `vector` after the node is approved
- gives you local start, stop, restart, update, and diagnostics controls without needing direct Docker commands

Already installed and just want to start the runtime again:

```bash
bash start.sh
```

Windows PowerShell:

```powershell
.\start.ps1
```

Windows double-click launcher:

`Open AUTONOMOUSc Edge Node.cmd`

Stop the background service:

```bash
bash stop.sh
```

Windows PowerShell:

```powershell
.\stop.ps1
```

Windows double-click launcher:

`Stop AUTONOMOUSc Edge Node.cmd`

Repair the local app and reopen the owner UI:

```bash
bash repair.sh
```

Windows PowerShell:

```powershell
.\repair.ps1
```

Windows double-click launcher:

`Repair AUTONOMOUSc Edge Node.cmd`

Notes:

- `install.sh`, `install.ps1`, `start.sh`, and `start.ps1` create a local service virtual environment and open the background-service UI.
- `repair.sh` and `repair.ps1` restore the owner conveniences, recreate the structured local settings and generated runtime config when possible, and restart the runtime when the node is already claimed.
- On Windows, the friendly `.cmd` launchers bypass PowerShell execution-policy friction so owners can just double-click into setup or reopen the app later.
- The repo-local launch scripts now require Python 3.11 or newer explicitly and guide owners toward reinstalling cleanly if the local environment is incomplete.
- The local UI runs at `http://127.0.0.1:8765` by default and stays available while the background service is running.
- Automatic updates now follow the signed runtime release manifest and only pull digest-pinned images.
- Repo-local installs still use the checked-out `docker-compose.yml`, which builds `node-agent` from source for development.
- Manager-container installs use the bundled runtime assets and the published digest-pinned `anirdarrazi/autonomousc-ai-edge-runtime` image instead of rebuilding from source.
- Diagnostics bundles are written to `./data/diagnostics`.
- `node-agent-bootstrap` is still available as a legacy fallback for direct terminal claim flows.
- `node-agent` runs headless after credentials have been stored in `./data/credentials`.
- `OPERATOR_TOKEN` is now only a legacy fallback for development or controlled migrations.
- `ATTESTATION_PROVIDER=simulated` is fine for local bring-up, but restricted work now requires hardware-backed attestation metadata before the control plane will schedule it.
- Open-source/community nodes remain eligible for community-best-effort workloads, but exact-model audited workloads require the control plane to classify the node as `trusted`.
- `.env.example` is now an advanced-mode template. Normal setup should stay in the browser flow and let the runtime generate `./data/service/runtime.env` for you.
- `.env.example` defaults to the production control plane at `https://edge.autonomousc.com`. Override it only for local Worker development or intentional advanced overrides.
- `runtime_bundle/model-artifacts.json` is generated by the release pipeline from upstream snapshot metadata and is the source of truth for expected model-manifest and tokenizer digests.
- Refresh that manifest with `python ./scripts/generate_model_artifacts_manifest.py` before cutting a new runtime release when model snapshots change.

Run tests with:

```bash
python -m pytest
```
