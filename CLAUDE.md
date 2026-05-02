# Edge Node Runtime - Comprehensive Documentation

> **IMPORTANT**: Keep this file in sync with `AGENTS.md`. Both should contain identical comprehensive documentation. When updating technical details, update both files.

## System Architecture Overview

**Radiance** is a distributed AI computing platform with four interconnected projects:

```
┌─────────────────────────────────────────────────────────┐
│         Marketplace Console (React Dashboard)            │
│      Network Operations & Node Management UI              │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│       Edge Control (Cloudflare Workers + D1)             │
│      Central Network Control Plane & Coordination        │
└──────┬────────────────┬────────────────┬────────────────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ OpenBatch    │  │ Edge Nodes   │  │ External AI  │
│ (AI Router)  │  │ (This Repo)  │  │ Providers    │
│              │  │ (Inference)  │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Related Projects**:
- [`OpenBatch/`](../OpenBatch/) - Batch AI job routing
- [`edge-control/`](../edge-control/) - Central control plane
- [`marketplace-console/`](../marketplace-console/) - Operator dashboard frontend

---

## Project Overview

**Edge Node Runtime** is a Python-based runtime for AUTONOMOUSc edge nodes. It provides a secure, containerized environment for running AI inference workloads on distributed edge hardware with GPU support. The runtime handles node enrollment, assignment execution, container orchestration, and audit receipt generation.

### What It Does
- Enrollment and management of edge nodes in the AUTONOMOUSc network
- Handles AI inference assignment execution and reporting
- Manages containerized inference engines (vllm for LLM serving)
- Provides node-agent polling and assignment processing
- Supports NVIDIA GPU acceleration
- Emits runtime receipts for audit and trust verification
- Auto-detects node hardware capabilities and profiles
- Guides owners through setup with browser-based UI

---

## Technology Stack

- **Language**: Python 3.11+
- **Containerization**: Docker with multi-image strategy
- **Inference Engine**: vllm (LLM serving with vLLM OpenAI-compatible API)
- **Vector Store**: Vector database support for semantic search
- **Testing**: pytest
- **GPU Support**: NVIDIA CUDA via Docker runtime
- **Setup UI**: Browser-based (http://127.0.0.1:8765)
- **Orchestration**: Docker Compose for local development
- **Distribution**: Public Docker images on Docker Hub (`anirdarrazi/autonomousc-ai-edge-runtime`)

---

## Architecture

### Operating Modes

1. **Manager Mode** - Runtime orchestrates sibling containers (vllm, node-agent, vector) via Docker Compose
2. **Single Container Mode** - All components run in one NVIDIA container
3. **Auto-Detection** - System automatically chooses based on Docker socket availability

### Key Components

- **`src/node_agent/`** - Core agent logic
  - Node enrollment with control plane
  - Assignment polling loop
  - Execution dispatch and monitoring
  - Status reporting and health checks

- **Inference Engine**
  - vllm for LLM serving
  - NVIDIA GPU acceleration
  - Model caching and warmup
  - Request queuing and batching

- **Setup UI** - Browser-based owner application
  - Hardware auto-detection
  - Guided enrollment flow
  - Model selection
  - Credential storage and management
  - Service lifecycle controls (start, stop, restart, update)
  - Diagnostics collection and export

- **Docker Images**
  - `Dockerfile` - Main unified public image
  - `Dockerfile.single` - Legacy single-container variant
  - `docker-compose.yml` - Local development stack

---

## Important Files & Directories

```
edge-node-runtime/
├── src/
│   └── node_agent/           # Core agent logic
│       ├── __main__.py       # Entry point
│       ├── enrollment.py     # Node enrollment
│       ├── polling.py        # Assignment polling
│       ├── execution.py      # Job execution
│       └── reporting.py      # Status reporting
├── Dockerfile                # Main unified public image
├── Dockerfile.single         # Legacy single-container variant
├── Dockerfile.service        # Service wrapper
├── docker-compose.yml        # Local development setup
├── .env.example              # Advanced-mode environment template
├── app.sh / app.ps1          # Repo-local owner app launcher
├── build-manager-image.sh/ps1    # Build unified image
├── publish-latest-image.sh/ps1   # Publish to Docker Hub
├── install.sh/ps1, start.sh/ps1  # Legacy setup scripts
├── repair.sh/ps1, stop.sh/ps1    # Service management scripts
├── scripts/
│   ├── generate_model_artifacts_manifest.py
│   └── other utilities
├── runtime_bundle/           # Runtime artifact management
│   └── model-artifacts.json  # Model manifest and checksums
├── data/                     # Local runtime data (gitignored)
│   ├── service/
│   │   ├── runtime-settings.json
│   │   └── runtime.env
│   ├── credentials/          # Node credentials
│   └── diagnostics/          # Support diagnostic bundles
└── pytest.ini, setup.cfg     # Test configuration
```

---

## Local Development

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with NVIDIA Container Runtime
- Python 3.11+ (for development)
- Bash or PowerShell (for scripts)
- Modern web browser (for setup UI)

### Quick Start (Recommended)

```bash
cd edge-node-runtime

# Repo-local owner app (single command)
bash app.sh                    # Linux/macOS
.\app.ps1                      # Windows PowerShell
AUTONOMOUSc\ Edge\ Node\ App.cmd  # Windows double-click

# OR manual Docker setup
docker run --gpus all \
  -p 8765:8765 \
  -p 8000:8000 \
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest

# Then open http://127.0.0.1:8765 in browser
```

### Commands Reference

**Repo-Local Launcher**:
- `bash app.sh` / `.\app.ps1` - Open owner app (start service if needed)
- `bash stop.sh` / `.\stop.ps1` - Stop background service
- `bash repair.sh` / `.\repair.ps1` - Repair local app and restart

**Building Images**:
```bash
# Unified public image (single/manager auto-detect)
bash build-manager-image.sh         # Linux/macOS
.\build-manager-image.ps1           # Windows

# Publish to Docker Hub
bash publish-latest-image.sh        # Linux/macOS
.\publish-latest-image.ps1          # Windows
```

**Testing**:
```bash
python -m pytest                    # Run all tests
python -m pytest -v                 # Verbose output
python -m pytest tests/enrollment_test.py  # Specific test file
```

### Environment Variables

**Normal Setup** (via UI):
- No manual environment variables needed
- Setup UI auto-detects hardware
- Generates `./data/service/runtime.env` automatically

**Advanced Mode** (`.env.example` template):
```env
RUNTIME_PROFILE=rtx_5060_ti_16gb_gemma4_e4b
DEPLOYMENT_TARGET=vast_ai
INFERENCE_ENGINE=vllm
RUNTIME_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest
CAPACITY_CLASS=elastic_burst
TEMPORARY_NODE=true
```

---

## Public Installation

Single command with automatic setup (no manual configuration needed):

```bash
docker run --gpus all \
  -p 8765:8765 \
  -p 8000:8000 \
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

Then:
1. Open `http://127.0.0.1:8765` in browser
2. Click "Quick Start"
3. Enter node name (auto-detects hardware)
4. Complete browser approval flow
5. Service auto-starts inference engine
6. Ready for workload assignment

---

## Key Features

### Hardware Management
- **Auto-Detection**: GPU type, VRAM, compute capabilities, region
- **Profile Selection**: Matches startup model to available hardware
- **Resource Monitoring**: Real-time GPU utilization and memory tracking
- **Graceful Degradation**: Reduces workload if resources constrained

### Model Management
- **Auto-Selection**: Picks best startup model for detected hardware
- **Model Caching**: Pre-loads bootstrap models into image layer
- **Hugging Face Integration**: Supports HuggingFace model repositories
- **Token Management**: Stores HF tokens locally for authenticated models
- **Model Warmup**: Pre-loads and verifies models before accepting work

### Trust & Security
- **Runtime Receipts**: Audit evidence with assignment nonce, digests
- **Hardware-Backed Attestation**: Optional for restricted workloads
- **Community Nodes**: Untrusted by default, community-eligible workloads
- **Trusted Nodes**: First-party or admin-approved, all workload types
- **Open-Source Design**: Node owners can inspect and modify code

### Service Management
- **One-Click Installation**: No complex configuration needed
- **Browser-Based UI**: Setup and controls all in browser
- **Automatic Updates**: Updates via signed release manifest
- **Health Checks**: Startup model verification and warmup
- **Diagnostics**: Bundles collected to `./data/diagnostics/`

---

## Specialized Profiles

### RTX 5060 Ti Gemma Profile
- **Name**: `rtx_5060_ti_16gb_gemma4_e4b`
- **Hardware**: NVIDIA RTX 5060 Ti 16GB
- **Engine**: vllm
- **Model**: google/gemma-4-E4B-it
- **Context**: 32k tokens
- **Deployment**: Vast.ai marketplace
- **Capacity**: Elastic burst with single concurrent assignment

### Building Custom Profiles
1. Create Docker image with desired specs
2. Register profile in control plane
3. Advertise capabilities in node registration
4. Let control plane route appropriate workloads

---

## Testing

```bash
python -m pytest                    # Run all tests
python -m pytest -v                 # Verbose output
python -m pytest tests/test_file.py # Specific test file
python -m pytest -k "test_name"     # Specific test
```

---

## Build & Publishing

### Build Unified Public Image Locally
```bash
bash build-manager-image.sh         # Linux/macOS
.\build-manager-image.ps1           # Windows
```

### Build and Push to Docker Hub
```bash
bash publish-latest-image.sh        # Linux/macOS
.\publish-latest-image.ps1          # Windows
```

### Image Variants
- `anirdarrazi/autonomousc-ai-edge-runtime:latest` - Current public release
- `anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest` - Single container variant

### Model Preloading
```bash
# Control which models are pre-loaded (default: bootstrap model)
PRELOAD_HF_MODELS=BAAI/bge-large-en-v1.5 bash build-manager-image.sh
PRELOAD_HF_MODELS= bash build-manager-image.sh  # No preloading
```

---

## Configuration & Advanced Setup

### Normal Setup (Recommended)
- Use browser UI at `http://127.0.0.1:8765`
- Hardware auto-detection
- Guided enrollment flow
- Model auto-selection
- Service auto-management

### Advanced Mode (`.env.example` Template)
- Manual environment variables
- Legacy enrollment via `NODE_ID`, `NODE_KEY`
- `OPERATOR_TOKEN` for auth
- Custom model overrides
- Deployment target selection

### Control Plane Configuration
- Default: Production at `https://edge.autonomousc.com`
- Local development: Override with `CONTROL_PLANE_URL`
- Staging environments: Custom URLs via environment

---

## Security Model

### Trust Enforcement
- **Open-Source Code**: Assume owners can inspect and modify
- **Server-Side Validation**: Control plane enforces trust, not runtime
- **Community Nodes**: Untrusted by default, best-effort workloads
- **Trusted Nodes**: First-party or admin-approved partners
- **Restricted Workloads**: Require hardware-backed attestation

### Data Protection
- **Local Credentials**: Stored in `./data/credentials/` (mounted volume)
- **Sensitive Files**: `.env`, credentials, diagnostics excluded from repo
- **Audit Logging**: All operations logged for compliance
- **.gitignore**: Protects sensitive data from version control

---

## Notes & Constraints

- **GPU Required**: NVIDIA GPU with CUDA support recommended
- **Preloaded Models**: Included in image layer for faster startup
- **Setup UI Only**: Normal setup should use browser, not environment variables
- **Operator Token**: Legacy, use browser flow instead
- **Vast.ai Support**: Market-data fallback with specific profile
- **Docker Socket**: Manager mode auto-detects and uses if available
- **Public Image**: Digest-pinned for reproducibility and security

---

## Integration with Control Plane

### Node Enrollment
1. Manager container starts and opens setup UI
2. Owner enters node name and signs in
3. Creates claim session
4. Browser approval creates credentials
5. Agent enrolls with control plane
6. Receives trust tier and approval status

### Assignment Execution
1. Agent polls control plane for assignments
2. Receives execution request with model, parameters
3. Dispatches to vllm if not already loaded
4. Returns results to control plane
5. Emits runtime receipt with audit data

### Status Reporting
- Continuous health heartbeats
- Metrics reporting (GPU utilization, model load)
- Error reporting for failed assignments
- Graceful shutdown notification

---

## Related Documentation

- **AGENTS.md** - Agent-focused view (identical to this file)
- **OpenBatch/** - Batch job routing
- **edge-control/** - Central control plane
- **marketplace-console/** - Operator dashboards
- **README.md** - Original project README

---

## Qdrant Semantic Memory (Optional)

A `.mcp.json` in this repo root configures a local Qdrant vector store (collection: `edge_node_runtime`). This is **optional developer tooling** — the project builds and runs without it.

**Check availability before using:**
```bash
curl http://localhost:6333/healthz
```

- **Available** (`healthz check passed`): use `qdrant-find` to surface prior context before starting work, and `qdrant-store` to persist architectural decisions, non-obvious patterns, and bug root causes.
- **Not available** (timeout / connection refused): proceed normally without it. Do not prompt the developer to set it up unless they ask — it is a local-only tool, not shared infrastructure.

To set it up: [`../QDRANT.md`](../QDRANT.md) · Full agent usage guide: [`../QDRANT-AGENTS.md`](../QDRANT-AGENTS.md)

---

**Last Updated**: 2026-05-03
