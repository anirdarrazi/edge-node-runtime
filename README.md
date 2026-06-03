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
- `Dockerfile.service`: unified public install image used for `anirdarrazi/autonomousc-ai-edge-runtime:latest`
- `Dockerfile.single`: legacy single-container runtime build used for compatibility testing
- `docker-compose.yml`: local appliance runtime with `vllm` and `vector`
- `.env.example`: advanced-mode environment override template
- `build-manager-image.ps1` / `build-manager-image.sh`: build the unified public image
- `publish-latest-image.ps1` / `publish-latest-image.sh`: build and push `anirdarrazi/autonomousc-ai-edge-runtime:latest`

## Unified public image

`anirdarrazi/autonomousc-ai-edge-runtime:latest` is now the only public install image. It auto-detects how it should run:

- `manager` mode when `/var/run/docker.sock` is mounted
- `single_container` mode when Docker socket access is unavailable and the image is running as one NVIDIA container

## Owner contract

- install + Quick Start + browser approval from same machine, only `127.0.0.1:8765` needs exposure.

The setup UI is local. Run locally. Open 127.0.0.1:8765 only.
Owner launchers run the owner flow in strict loopback-only mode (--strict-owner-mode) by default.
Manager mode has root-equivalent access to the host through Docker. It should only be enabled explicitly on machines dedicated to this runtime.

### Public setup

Normal setup is one command and one browser screen:

```bash
docker run --gpus all \
-p 127.0.0.1:8765:8765 \
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

Then open `http://127.0.0.1:8765` and use Quick Start. The default owner flow will:

- infer the NVIDIA machine profile automatically
- ask only for a node name, then open the sign-in and approval page automatically
- create the claim session in the UI
- wait for browser approval automatically
- choose the best accessible startup model automatically from NVIDIA hardware, VRAM, region, and saved local model access
- save credentials locally under the mounted `/var/lib/autonomousc` volume
- pull the runtime image, fill the model cache, warm the startup model, and verify setup automatically

Normal owners do not need `NODE_ID`, `NODE_KEY`, `OPERATOR_TOKEN`, model overrides, or trust settings.

Build the unified image locally:

```bash
bash build-manager-image.sh
```

Windows PowerShell:

```powershell
.\build-manager-image.ps1
```

Runtime images do not preseed model weights by default, keeping production Vast images small and avoiding unrelated cache layers. Opt in per profile with `PRELOAD_HF_MODELS` when a specific starter model should be baked into the image, for example `PRELOAD_HF_MODELS=BAAI/bge-large-en-v1.5 bash build-single-image.sh`. Leave it empty, such as `PRELOAD_HF_MODELS= bash build-manager-image.sh`, for the lean default.

### Vast.ai RTX 5060 Ti Gemma profile

The provider profile for the market-data fallback node is `rtx_5060_ti_16gb_gemma4_e4b_it`. It is a Vast.ai `vllm` profile for one RTX 5060 Ti 16GB node serving `google/gemma-4-E4B-it` through the `/v1/responses` API.

Advanced-mode environment values:

```env
RUNTIME_PROFILE=rtx_5060_ti_16gb_gemma4_e4b_it
DEPLOYMENT_TARGET=vast_ai
INFERENCE_ENGINE=vllm
RUNTIME_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest
CAPACITY_CLASS=elastic_burst
TEMPORARY_NODE=false
BURST_PROVIDER=vast_ai
GPU_NAME=RTX 5060 Ti
GPU_MEMORY_GB=16
MAX_CONTEXT_TOKENS=32768
MAX_BATCH_TOKENS=32768
TARGET_BATCH_ITEMS=100
MAX_BATCH_ITEMS=250
TARGET_BATCH_TOKENS=12000
MAX_CONCURRENT_CHUNKS=4
MAX_CONCURRENT_ASSIGNMENTS=8
MAX_CONCURRENT_ASSIGNMENTS_CAP=8
MAX_LOCAL_QUEUE_ASSIGNMENTS=4
PULL_BUNDLE_SIZE=4
VLLM_STARTUP_TIMEOUT_SECONDS=1800
VLLM_MODEL=google/gemma-4-E4B-it
SUPPORTED_MODELS=google/gemma-4-E4B-it
VLLM_EXTRA_ARGS=--quantization fp8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.913 --max-num-seqs 8 --generation-config vllm --skip-mm-profiling
```

The control plane catalogs this profile as exact-model, audited-safetensors, `restricted`-eligible, and `elastic_exact_vast`. The durable Vast launcher uses a 32k context on 16 GB RTX 5060 Ti nodes, advertises 250-item hard BatchRouter chunks with four concurrent chunk lanes, and keeps the local pull reservoir capped at four assignments so slower hosts do not hoard queued chunks from faster fleet peers. It runs vLLM with FP8 weight quantization and FP8 KV cache over the BF16 safetensors source repo, allows a long first-load Gemma warmup, and keeps the contract alive after smoke success. Offer selection is intentionally narrow for this profile: one RTX 5060 Ti 16GB GPU, `cuda_max_good >= 12.9`, at least 80 GB disk, a basic reliability floor, and direct runtime/status endpoint wiring. Failed smoke candidates are destroyed before the launcher tries the next cheapest viable host.

Durable Vast launch helper:

```bash
python -m node_agent.vast_smoke \
  --durable-node \
  --model google/gemma-4-E4B-it \
  --api responses \
  --max-context-tokens 32768 \
  --max-batch-tokens 32768 \
  --max-price 0.20 \
  --min-vram-gb 16 \
  --min-cuda-max-good 12.9 \
  --disk-gb 80 \
  --max-concurrent-assignments 8 \
  --max-local-queue-assignments 4 \
  --pull-bundle-size 4 \
  --vllm-startup-timeout-seconds 1800 \
  --node-region eu-se-1
```

Fleet planning helper:

```bash
python -m node_agent.vast_fleet_plan \
  --nodes 2 \
  --model google/gemma-4-E4B-it \
  --runtime-profile rtx_5060_ti_16gb_gemma4_e4b_it \
  --max-price 0.25 \
  --min-vram-gb 15 \
  --min-cuda-max-good 12.9 \
  --min-reliability 0.98 \
  --min-inet-down-mbps 600 \
  --disk-gb 80
```

The planner now uses the same host-quality constraints as the launcher and inspects 200 offers by default. Keep the planned `--preferred-offer-id` arguments with each launch, and use the generated exclusion flags or `--allow-same-machine` intentionally when building multi-node fleets. Vast offers can disappear between planning and launch; if a preferred offer is stale, the launcher fails before renting anything and tells the operator to re-plan. Vast may report `gpu_frac < 1` for a rented slot on a larger multi-GPU host; the launcher accepts it only when the offer still assigns at least one GPU and reports full per-GPU memory through `gpu_ram` and `gpu_total_ram`. If the Vast market only exposes true partial-memory RTX 5060 Ti slices, the planner rejects them for this 32K Gemma profile and reports a runtime-policy rejection summary instead of launching a node that is likely to fail or contend for memory. Partial fleet plans include `market_diagnostics` so operators can tell whether they are blocked by price, quality floors, runtime policy, or lack of distinct machine identities. If Vast reports a realized instance hourly cost above `--max-price` after creation, the launcher destroys that candidate before model warmup and tries the next eligible host.

Required runtime secrets are read from environment variables: `VAST_API_KEY`, `HUGGING_FACE_HUB_TOKEN`, `EDGE_CONTROL_URL`, `NODE_ID`, and `NODE_KEY`. Do not commit these values.

### Advanced/Support paths

The following flows are for owner support, maintenance, and controlled migration work only. They are explicitly **not** the normal owner Quick Start path.

#### Advanced/Support: remote workflows

These helper commands are advanced/support operator/maintenance flows:

- `python -m node_agent.vast_smoke ...`
- `python -m node_agent.vast_fleet_plan ...`

Build and push the public `latest` image:

```bash
bash publish-latest-image.sh
```

Windows PowerShell:

```powershell
.\publish-latest-image.ps1
```

#### Advanced/Support: local manager mode

Use this when the runtime should manage sibling containers through the host Docker engine:

```bash
docker run --rm \
  --gpus all \
-p 127.0.0.1:8765:8765 \
  -e AUTONOMOUSC_RUNTIME_BACKEND=manager \
  --add-host=host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

Then open `http://127.0.0.1:8765` and complete Quick Start.

Notes:

- The owner setup UI is local. Run locally. Open 127.0.0.1:8765 only.
- In one-container mode, the service starts `vllm` and `node-agent` inside the same container.
- In manager mode, the service orchestrates sibling `vllm`, `node-agent`, and `vector` containers through Docker Compose.
- `Dockerfile.single` remains in the repo for compatibility testing, but `:latest` is the supported install path.
- If you intentionally need advanced operator/admin controls in the UI, open `http://127.0.0.1:8765/?operator=1`.
- If you intentionally need legacy env-based enrollment, `OPERATOR_TOKEN`, `NODE_ID`, and `NODE_KEY` still work, but they are advanced-mode fallbacks rather than the normal install path.

### Advanced/Support: repo-local launchers and legacy scripts

If you are running from a checked-out repo for support or development, use the repo-local owner app instead of deciding between install vs open.

Linux/macOS:

```bash
bash app.sh
```

Windows PowerShell:

```powershell
.\app.ps1
```

Windows double-click launcher:

`AUTONOMOUSc Edge Node App.cmd`

That owner app bootstraps the local service environment on first run, reuses it after that, and opens the same local UI in your browser. The service handles first-time setup and ongoing operations from one place. It:

- checks Docker and GPU prerequisites
- saves structured local settings in `./data/service/runtime-settings.json`
- generates `./data/service/runtime.env` for the runtime automatically
- starts `vllm`
- creates the node claim and waits for browser approval
- stores credentials locally in `./data/credentials`
- starts `node-agent` and `vector` after the node is approved
- gives you local start, stop, restart, update, and diagnostics controls without needing direct Docker commands

The older repo-local launch scripts still exist for support and development:

- `install.sh` / `install.ps1` refresh dependencies and open the app
- `start.sh` / `start.ps1` refresh dependencies and open the app
- `repair.sh` / `repair.ps1` repair the local app and reopen the owner UI
- `stop.sh` / `stop.ps1` stop the background service

Already installed and just want to reopen the owner app:

```bash
bash app.sh
```

Windows PowerShell:

```powershell
.\app.ps1
```

Windows double-click launcher:

`AUTONOMOUSc Edge Node App.cmd`

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

- `app.sh`, `app.ps1`, and `AUTONOMOUSc Edge Node App.cmd` are the single repo-local owner path. They bootstrap the local service environment when needed, then just reopen the app.
- `install.sh`, `install.ps1`, `start.sh`, and `start.ps1` remain available for support or development when you intentionally want a dependency refresh from the repo checkout.
- `repair.sh` and `repair.ps1` restore the owner conveniences, recreate the structured local settings and generated runtime config when possible, and restart the runtime when the node is already claimed.
- On Windows, the friendly `.cmd` launchers bypass PowerShell execution-policy friction so owners can just double-click into setup or reopen the app later.
- The repo-local launch scripts now require Python 3.11 or newer explicitly and guide owners toward reinstalling cleanly if the local environment is incomplete.
- The local UI runs at `http://127.0.0.1:8765` by default and stays available while the background service is running.
- Automatic updates now follow the signed runtime release manifest and only pull digest-pinned images.
- Failed update attempts keep the previous runtime state, auto-roll back when possible, and surface explicit repair guidance in the UI (`Run Fix it` / `/api/repair`) when rollback cannot complete.
- Repo-local installs still use the checked-out `docker-compose.yml`, which builds `node-agent` from source for development.
- Manager-container installs use the bundled runtime assets and the published digest-pinned `anirdarrazi/autonomousc-ai-edge-runtime` image instead of rebuilding from source.
- Diagnostics bundles are written to `./data/diagnostics`.
- `node-agent-bootstrap` is explicitly an advanced/support debug path; normal owners should stay in the setup UI and let Quick Start run the local browser approval flow.
- `docker-compose.yml` usage here is advanced/support-only and intended only for repo-local development or support workflows.
- `node-agent` runs headless after credentials have been stored in `./data/credentials`.
- `OPERATOR_TOKEN`, `NODE_ID`, and `NODE_KEY` are now legacy fallbacks for development, support, or controlled migrations rather than the normal install flow.
- `ATTESTATION_PROVIDER=simulated` is fine for local bring-up, but restricted work now requires hardware-backed attestation metadata before the control plane will schedule it.
- Open-source/community nodes remain eligible for community-best-effort workloads, but exact-model audited workloads require the control plane to classify the node as `trusted`.
- `.env.example` is now an advanced-mode template. Normal setup should stay in the browser flow, let the runtime generate `./data/service/runtime.env` for you, and save a Hugging Face token locally there when the chosen startup model requires one.
- `.env.example` defaults to the production control plane at `https://edge.autonomousc.com`. Override it only for local Worker development or intentional advanced overrides.
- `runtime_bundle/model-artifacts.json` is generated by the release pipeline from upstream snapshot metadata and is the source of truth for expected model-manifest and tokenizer digests.
- Refresh that manifest with `python ./scripts/generate_model_artifacts_manifest.py` before cutting a new runtime release when model snapshots change.

Run tests with:

```bash
python -m pytest
```
