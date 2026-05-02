# Edge Node Runtime - Comprehensive Documentation

> **IMPORTANT**: Keep this file in sync with `CLAUDE.md`. Both should contain identical comprehensive documentation. When updating technical details, update both files.

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

## Main Capabilities

### Node Management
- Automatic enrollment in AUTONOMOUSc network
- Hardware auto-detection (GPU type, VRAM, compute capabilities)
- Guided setup UI (no manual configuration needed for normal setup)
- Local credential storage (mounted volume)
- Operator and admin control endpoints
- Multi-tier trust classification
- Service lifecycle management (start, stop, restart, update)

### Inference Execution
- LLM inference via vllm serving
- Multi-model support with dynamic loading
- GPU-accelerated execution (NVIDIA CUDA)
- Assignment polling from control plane
- Async execution with timeout handling
- Retry mechanisms for transient failures
- Per-item error reporting
- Model warmup and cache management

### Container Orchestration
- **Manager Mode** - orchestrates multiple containers via Docker Compose
  - Separate vllm, node-agent, and vector containers
  - Graceful startup and shutdown
  - Service health checks and recovery
  - Docker socket access required

- **Single Container Mode** - unified container for all services
  - No Docker socket required
  - All components in one NVIDIA container
  - Simplified deployment for single machines
  - No sibling container management

- **Auto-Detection** - automatically chooses mode
  - Detects Docker socket availability
  - Falls back to single-container if needed
  - Single setup UI for both modes

### Hardware Support
- NVIDIA GPU detection and configuration
- Vast.ai integration for temporary/burst capacity
- RTX series GPU optimization (5060 Ti, 4090, A100, etc.)
- VRAM-based model selection
- Context length management per hardware
- GPU utilization monitoring
- Memory pressure handling

### Trust & Security
- Runtime receipt generation with audit evidence
  - Assignment nonce
  - Runtime image digest
  - Declared model
  - Model manifest digest
  - Tokenizer digest
  - Aggregated usage metrics
- Hardware-backed attestation support (optional)
- Community vs trusted tier classification
- Server-side trust enforcement (not in runtime)
- Open-source code reviewability
- Audit logging for compliance

---

## Deployment Targets
- **NVIDIA GPU Hardware** - RTX series, Tesla, A100, etc.
- **Vast.ai Marketplace** - temporary burst capacity with specific profiles
- **Docker Swarm** - orchestrated edge clusters
- **Kubernetes** - containerized deployment (via Docker images)
- **On-Premise** - single node appliances
- **Cloud GPU Instances** - AWS, GCP, Azure GPU VMs

---

## Integration Points

### Upstream (Control Plane)
- **Edge Control** ([`edge-control/`](../edge-control/)) - node enrollment, assignment polling, status reporting, trust management

### Downstream (Inference Consumers)
- **OpenBatch** ([`OpenBatch/`](../OpenBatch/)) - batch job routing
- **External Clients** - direct inference requests via vllm API

### Data Sources
- **Model Registries** - Hugging Face for model pulls and metadata
- **Hardware Metrics** - GPU monitoring and reporting
- **Vector Database** - optional semantic search capability

---

## Primary Development Workflows

### 1. Add Hardware Profile
- Define in advanced environment or UI
- Register capabilities with control plane
- Set appropriate capacity limits
- Test with representative workloads

### 2. Modify Inference Engine
- Change vllm configuration
- Implement alternative inference engine
- Test with multiple models
- Validate performance

### 3. Update Model Support
- Add to model manifest (`model-artifacts.json`)
- Update bootstrap preloading
- Verify tokenizer digests
- Test on target hardware

### 4. Improve Startup Flow
- Enhance setup UI logic
- Refine hardware detection
- Improve model selection algorithm
- Add user guidance

### 5. Add Health Checks
- Extend runtime verification
- Add metrics collection
- Implement diagnostics
- Create alert conditions

---

## Performance & Scale Characteristics

- **Multi-Model Serving**: Concurrent model loading with memory management
- **GPU Queue Management**: Efficient batching and request queuing
- **Async Polling**: Configurable polling intervals for control plane
- **Containerized Isolation**: Resource isolation between workloads
- **Graceful Degradation**: Reduces load when resources constrained
- **Auto Scaling**: Manager mode can add/remove containers dynamically

---

## Technology Stack

- **Language**: Python 3.11+
- **Containerization**: Docker with multi-image strategy
- **Inference Engine**: vllm (LLM serving with OpenAI-compatible API)
- **Vector Store**: Vector database support
- **Testing**: pytest
- **GPU Support**: NVIDIA CUDA via Docker runtime
- **Setup UI**: Browser-based (http://127.0.0.1:8765)
- **Orchestration**: Docker Compose for local development
- **Distribution**: Public Docker images (`anirdarrazi/autonomousc-ai-edge-runtime`)

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
├── docker-compose.yml        # Local development setup
├── .env.example              # Advanced-mode environment template
├── app.sh / app.ps1          # Repo-local owner app launcher
├── build-manager-image.sh/ps1    # Build unified image
├── publish-latest-image.sh/ps1   # Publish to Docker Hub
├── scripts/
│   ├── generate_model_artifacts_manifest.py
│   └── other utilities
├── runtime_bundle/           # Runtime artifact management
│   └── model-artifacts.json  # Model manifest and checksums
└── data/                     # Local runtime data (gitignored)
    ├── service/
    ├── credentials/
    └── diagnostics/
```

---

## Local Development Guide

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
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

### Commands Reference

**Repo-Local Launcher**:
- `bash app.sh` / `.\app.ps1` - Open owner app
- `bash stop.sh` / `.\stop.ps1` - Stop service
- `bash repair.sh` / `.\repair.ps1` - Repair and restart

**Building Images**:
```bash
bash build-manager-image.sh         # Linux/macOS
.\build-manager-image.ps1           # Windows
bash publish-latest-image.sh        # Push to Docker Hub
```

**Testing**:
```bash
python -m pytest
python -m pytest -v
python -m pytest tests/test_file.py
```

---

## Public Installation

Single command with fully automatic setup:

```bash
docker run --gpus all \
  -p 8765:8765 \
  -p 8000:8000 \
  -v autonomousc-edge:/var/lib/autonomousc \
  anirdarrazi/autonomousc-ai-edge-runtime:latest
```

Then open `http://127.0.0.1:8765` and click "Quick Start".

---

## Specialized Profiles

### RTX 5060 Ti Gemma Profile
- **Name**: `rtx_5060_ti_16gb_gemma4_e4b`
- **Hardware**: RTX 5060 Ti 16GB
- **Engine**: vllm
- **Model**: google/gemma-4-E4B-it
- **Context**: 32k tokens
- **Deployment**: Vast.ai marketplace
- **Capacity**: Elastic burst, single concurrent assignment

---

## Configuration

### Normal Setup (Recommended)
- Use browser UI at `http://127.0.0.1:8765`
- Hardware auto-detection
- Guided enrollment
- Model auto-selection
- Service auto-management

### Advanced Mode (`.env.example` Template)
- Manual environment variables
- Legacy enrollment tokens
- Custom model overrides
- Deployment target selection
- For development/support only

---

## Security Model

### Trust Enforcement
- **Open-Source Design**: Owners can inspect and modify
- **Server-Side Validation**: Control plane enforces trust
- **Community Nodes**: Untrusted by default, community workloads
- **Trusted Nodes**: First-party or admin-approved
- **Restricted Workloads**: Hardware-backed attestation required

### Data Protection
- **Local Credentials**: In mounted volume, not in container
- **Audit Logging**: All operations logged
- **Sensitive Files**: Excluded from version control (`.gitignore`)

---

## Notes & Constraints

- **GPU Required**: NVIDIA GPU with CUDA support
- **Preloaded Models**: Included in image layer for fast startup
- **Vast.ai Support**: Market-data fallback with specific profile
- **Public Image**: Digest-pinned for reproducibility
- **Normal Setup**: Use browser UI, not environment variables
- **Auto-Detect**: System automatically chooses manager vs single-container mode

---

## Related Documentation

- **CLAUDE.md** - Complete technical documentation (identical to this file)
- **OpenBatch/** - Batch job routing
- **edge-control/** - Central control plane
- **marketplace-console/** - Operator dashboards
- **README.md** - Original project README

---

**Last Updated**: 2026-05-03
