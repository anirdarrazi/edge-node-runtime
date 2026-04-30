from __future__ import annotations

import argparse
import base64
import io
import json
import locale
import os
import re
import socket
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

import httpx

from .appliance_manifest import (
    inspect_package_signature,
    inspect_runtime_bundle_signature,
    release_channel_label,
)
from .autostart import AutoStartManager
from .config import NodeAgentSettings, NodeClaimSession
from .control_plane import EdgeControlClient
from .desktop_launcher import DesktopLauncherManager
from .fault_injection import DEFAULT_FAULT_INJECTION_STATE_NAME, FaultInjectionController
from .gguf_artifacts import find_gguf_artifact
from .inference_engine import (
    AUTO_RUNTIME_PROFILE,
    DEFAULT_GEMMA_4_E4B_MODEL,
    LLAMA_CPP_INFERENCE_ENGINE,
    VAST_VLLM_SAFETENSORS_PROFILE,
    default_runtime_profile,
    deployment_target_label,
    inference_engine_label,
    llama_cpp_model_source,
    resolve_runtime_profile,
)
from .local_api_security import (
    ADMIN_TOKEN_HEADER,
    LOCAL_SESSION_COOKIE,
    LocalSessionStore,
    browser_access_host,
    cookie_value,
    generate_admin_token,
    origin_matches_host,
    request_query_param,
    require_secure_bind_host,
    serialize_cookie,
    tighten_private_path,
    token_matches,
)
from .model_artifacts import find_model_artifact
from .release_manifest import load_release_manifest, ReleaseManifestError
from .runtime_backend import (
    AUTO_RUNTIME_BACKEND,
    RUNTIME_BACKEND_ENV,
    SINGLE_CONTAINER_RUNTIME_BACKEND,
    detect_runtime_backend,
    normalize_runtime_backend,
    runtime_backend_label,
    runtime_backend_supports_compose,
)
from .runtime_layout import bundled_runtime_dir, ensure_runtime_bundle, resolve_runtime_dir, service_access_host
from .single_container import (
    DEFAULT_STARTUP_STATUS_ENDPOINT_PATH,
    DEFAULT_STARTUP_STATUS_FILENAME,
    DEFAULT_STARTUP_STATUS_HOST,
    DEFAULT_STARTUP_STATUS_PORT,
    EmbeddedRuntimeSupervisor,
    known_safe_max_model_len,
)


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
ControlClientFactory = Callable[[NodeAgentSettings], EdgeControlClient]
SleepFn = Callable[[float], None]
RuntimeStatusProvider = Callable[[], dict[str, Any]]


ENV_ORDER = [
    "SETUP_PROFILE",
    "EDGE_CONTROL_URL",
    "OPERATOR_TOKEN",
    "NODE_ID",
    "NODE_KEY",
    "NODE_LABEL",
    "NODE_REGION",
    "TRUST_TIER",
    "RESTRICTED_CAPABLE",
    "CREDENTIALS_PATH",
    "AUTOPILOT_STATE_PATH",
    "HEAT_GOVERNOR_STATE_PATH",
    "FAULT_INJECTION_STATE_PATH",
    "STARTUP_STATUS_PATH",
    "STARTUP_STATUS_HOST",
    "STARTUP_STATUS_PORT",
    "STARTUP_STATUS_ENDPOINT_PATH",
    "RUNTIME_PROFILE",
    "DEPLOYMENT_TARGET",
    "INFERENCE_ENGINE",
    "RUNTIME_IMAGE",
    "INFERENCE_BASE_URL",
    "CAPACITY_CLASS",
    "TEMPORARY_NODE",
    "BURST_PROVIDER",
    "BURST_LEASE_ID",
    "BURST_LEASE_PHASE",
    "BURST_COST_CEILING_USD",
    "VLLM_BASE_URL",
    "VLLM_STARTUP_TIMEOUT_SECONDS",
    "VLLM_EXTRA_ARGS",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    "GPU_NAME",
    "GPU_MEMORY_GB",
    "MAX_CONTEXT_TOKENS",
    "MAX_BATCH_TOKENS",
    "MAX_CONCURRENT_ASSIGNMENTS",
    "MODEL_CACHE_BUDGET_GB",
    "MODEL_CACHE_RESERVE_FREE_GB",
    "OFFLINE_INSTALL_BUNDLE_DIR",
    "STARTUP_WARM_SOURCE",
    "STARTUP_WARM_SOURCE_LABEL",
    "STARTUP_WARM_SOURCE_DETAIL",
    "STARTUP_WARM_SOURCE_SCOPE",
    "STARTUP_WARM_SOURCE_ORDER",
    "STARTUP_WARM_SOURCE_SELECTED_AT",
    "HEAT_GOVERNOR_MODE",
    "OWNER_OBJECTIVE",
    "TARGET_GPU_UTILIZATION_PCT",
    "MIN_GPU_MEMORY_HEADROOM_PCT",
    "THERMAL_HEADROOM",
    "HEAT_DEMAND",
    "ROOM_TEMP_C",
    "TARGET_TEMP_C",
    "OUTSIDE_TEMP_C",
    "QUIET_HOURS_START_LOCAL",
    "QUIET_HOURS_END_LOCAL",
    "GPU_TEMP_C",
    "GPU_TEMP_LIMIT_C",
    "POWER_WATTS",
    "ESTIMATED_HEAT_OUTPUT_WATTS",
    "GPU_POWER_LIMIT_ENABLED",
    "MAX_POWER_CAP_WATTS",
    "ENERGY_PRICE_KWH",
    "SUPPORTED_MODELS",
    "POLL_INTERVAL_SECONDS",
    "ATTESTATION_PROVIDER",
    "VLLM_MODEL",
    "OWNER_TARGET_MODEL",
    "OWNER_TARGET_SUPPORTED_MODELS",
    "LLAMA_CPP_HF_REPO",
    "LLAMA_CPP_HF_FILE",
    "LLAMA_CPP_ALIAS",
    "LLAMA_CPP_EMBEDDING",
    "LLAMA_CPP_POOLING",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_TOKEN",
    "DOCKER_IMAGE",
    "VLLM_IMAGE",
]

COMPOSE_RUNTIME_CREDENTIALS_PATH = "/var/lib/autonomousc/credentials/node-credentials.json"
COMPOSE_RUNTIME_AUTOPILOT_STATE_PATH = "/var/lib/autonomousc/scratch/autopilot-state.json"
COMPOSE_RUNTIME_HEAT_GOVERNOR_STATE_PATH = "/var/lib/autonomousc/scratch/heat-governor-state.json"
COMPOSE_RUNTIME_FAULT_INJECTION_STATE_PATH = "/var/lib/autonomousc/scratch/fault-injection-state.json"
COMPOSE_RUNTIME_STARTUP_STATUS_PATH = f"/var/lib/autonomousc/scratch/{DEFAULT_STARTUP_STATUS_FILENAME}"

PROFILE_LABELS = {
    "quiet": "Quiet",
    "balanced": "Balanced",
    "performance": "Performance",
}
RECOMMENDED_FREE_DISK_GB = 30.0
MODEL_CACHE_DEFAULT_BUDGET_FRACTION = 0.6
MODEL_CACHE_DEFAULT_RESERVE_FRACTION = 0.1
MODEL_CACHE_MIN_BUDGET_GB = 24.0
MODEL_CACHE_MAX_BUDGET_GB = 768.0
DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RTX_5060_TI_16GB_VLLM_EXTRA_ARGS = (
    "--quantization fp8 "
    "--kv-cache-dtype fp8 "
    "--gpu-memory-utilization 0.913 "
    "--max-num-seqs 12 "
    "--generation-config vllm "
    "--skip-mm-profiling"
)
RUNTIME_SETTINGS_VERSION = 1
INSTALLER_STATE_VERSION = 1
INSTALLER_FLOW = [
    ("checking_docker", "Checking Docker"),
    ("checking_nvidia_runtime", "Checking NVIDIA runtime"),
    ("validating_hf_access", "Validating HF access"),
    ("pulling_image", "Pulling image"),
    ("downloading_model", "Downloading model"),
    ("warming_model", "Warming model"),
    ("claiming_node", "Claiming node"),
    ("node_live", "Node live"),
]
INSTALLER_FLOW_INDEX = {key: index for index, (key, _label) in enumerate(INSTALLER_FLOW)}
INSTALLER_FLOW_LABELS = {key: label for key, label in INSTALLER_FLOW}
INSTALLER_TERMINAL_STAGES = {"idle", "running", "error"}
INSTALLER_FLOW_REMAINING_SECONDS = {
    "checking_docker": 480,
    "checking_nvidia_runtime": 450,
    "validating_hf_access": 420,
    "pulling_image": 360,
    "downloading_model": 240,
    "warming_model": 120,
    "claiming_node": 60,
    "node_live": 30,
}
EU_COUNTRY_CODES = {
    "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU",
    "IE", "IS", "IT", "LT", "LU", "LV", "MT", "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK",
}
APAC_COUNTRY_CODES = {
    "AU", "BD", "CN", "HK", "ID", "IN", "JP", "KR", "MY", "NZ", "PH", "SG", "TH", "TW", "VN",
}
AMERICAS_COUNTRY_CODES = {"AR", "BR", "CA", "CL", "CO", "MX", "PE", "US"}
HF_GATED_REPOSITORY_PREFIXES = ("meta-llama/",)
CLAIM_AUTO_REFRESH_WINDOW_SECONDS = 90
PRECHECK_NETWORK_CACHE_TTL_SECONDS = 10
KNOWN_NODE_REGIONS = {"eu-se-1", "us-east-1", "us-west-1", "ap-sg-1"}
WARM_DOWNLOAD_SLOW_SECONDS = 20.0
WARM_DOWNLOAD_STALL_SECONDS = 45.0
WARM_DOWNLOAD_SLOW_RATE_BYTES_PER_SECOND = 2 * 1024 * 1024
WARM_RUNTIME_LOG_TAIL_LINES = 80
WARM_RUNTIME_LOG_EXCERPT_LINES = 6
WARMUP_HF_AUTH_MARKERS = (
    "gatedrepoerror",
    "hugging face denied access",
    "401 client error",
    "403 client error",
    "unauthorized",
    "forbidden",
    "access to model",
)
WARMUP_HF_NOT_FOUND_MARKERS = (
    "repository not found",
    "revision not found",
    "404 client error",
)
WARMUP_UNSUPPORTED_MODEL_MARKERS = (
    "unsupported model",
    "unsupported architecture",
    "unsupported config",
    "no supported config format",
    "unrecognized configuration class",
    "unknown model type",
)
WARMUP_OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "insufficient memory",
    "cublas_status_alloc_failed",
    "allocation on device",
)
WINDOWS_FIREWALL_RULE_PREFIX = "AUTONOMOUSc Edge Node"
LOCAL_SERVICE_PORT = 8765
LOCAL_INFERENCE_PORT = 8000


@dataclass(frozen=True)
class NvidiaSupportPreset:
    key: str
    label: str
    min_vram_gb: float
    max_vram_gb: float | None
    capacity_label: str
    startup_model: str
    supported_models: tuple[str, ...]
    recommended_profile: str
    quiet_concurrency: str
    balanced_concurrency: str
    performance_concurrency: str
    summary: str
    community_detail: str
    premium_detail: str
    gpu_name_substrings: tuple[str, ...] = ()
    max_context_tokens: int | None = None
    vllm_startup_timeout_seconds: int | None = None
    vllm_extra_args: str = ""
    runtime_env: tuple[tuple[str, str], ...] = ()

    def matches(self, memory_gb: float, gpu_name: str | None = None) -> bool:
        if memory_gb < self.min_vram_gb:
            return False
        if self.max_vram_gb is not None and memory_gb > self.max_vram_gb:
            return False
        if self.gpu_name_substrings:
            normalized_gpu_name = normalize_gpu_name(gpu_name)
            if not normalized_gpu_name:
                return False
            for token in self.gpu_name_substrings:
                if token not in normalized_gpu_name:
                    return False
        return True

    def concurrency_for_profile(self, profile: str) -> str:
        if profile == "quiet":
            return self.quiet_concurrency
        if profile == "performance":
            return self.performance_concurrency
        return self.balanced_concurrency

    def supported_models_csv(self) -> str:
        return ",".join(self.supported_models)

    def runtime_env_defaults(self) -> dict[str, str]:
        values: dict[str, str] = {}
        if self.max_context_tokens is not None:
            values["MAX_CONTEXT_TOKENS"] = str(self.max_context_tokens)
        if self.vllm_startup_timeout_seconds is not None:
            values["VLLM_STARTUP_TIMEOUT_SECONDS"] = str(self.vllm_startup_timeout_seconds)
        if self.vllm_extra_args.strip():
            values["VLLM_EXTRA_ARGS"] = self.vllm_extra_args.strip()
        for key, value in self.runtime_env:
            if key and str(value).strip():
                values[key] = str(value).strip()
        return values


def normalize_gpu_name(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


NVIDIA_SUPPORT_PRESETS = (
    NvidiaSupportPreset(
        key="starter_embeddings",
        label="Under 12 GB NVIDIA",
        min_vram_gb=0.0,
        max_vram_gb=11.9,
        capacity_label="Embeddings/community",
        startup_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=(DEFAULT_EMBEDDING_MODEL,),
        recommended_profile="quiet",
        quiet_concurrency="1",
        balanced_concurrency="1",
        performance_concurrency="1",
        summary=(
            "Under 12 GB NVIDIA stays on the smallest community preset with one active workload at a time."
        ),
        community_detail=(
            "Community capacity stays enabled on the smallest embeddings/community preset today. Premium jobs stay unavailable on this machine."
        ),
        premium_detail=(
            "This NVIDIA class stays on the smallest embeddings/community preset today. Premium jobs stay unavailable on this machine."
        ),
    ),
    NvidiaSupportPreset(
        key="rtx_5060_ti_16gb_gemma4_e4b",
        label="RTX 5060 Ti 16 GB",
        min_vram_gb=15.0,
        max_vram_gb=17.9,
        capacity_label="Gemma 4 E4B FP8 32k",
        startup_model=DEFAULT_GEMMA_4_E4B_MODEL,
        supported_models=(DEFAULT_GEMMA_4_E4B_MODEL,),
        recommended_profile="performance",
        quiet_concurrency="12",
        balanced_concurrency="24",
        performance_concurrency="40",
        summary=(
            "RTX 5060 Ti 16 GB uses the tuned Gemma 4 E4B FP8 32k preset with graph-memory profiling enabled."
        ),
        community_detail=(
            "Community capacity stays enabled on the Gemma 4 E4B FP8 32k preset today. Premium routing stays unavailable until larger or approved hardware is selected."
        ),
        premium_detail=(
            "This RTX 5060 Ti preset is tuned for community Gemma 4 E4B FP8 32k serving today. Premium routing stays unavailable on this machine."
        ),
        gpu_name_substrings=("5060", "ti"),
        max_context_tokens=32768,
        vllm_startup_timeout_seconds=900,
        vllm_extra_args=RTX_5060_TI_16GB_VLLM_EXTRA_ARGS,
        runtime_env=(("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1"),),
    ),
    NvidiaSupportPreset(
        key="community_embeddings",
        label="12-23 GB NVIDIA",
        min_vram_gb=12.0,
        max_vram_gb=23.9,
        capacity_label="Embeddings/community",
        startup_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=(DEFAULT_EMBEDDING_MODEL,),
        recommended_profile="balanced",
        quiet_concurrency="1",
        balanced_concurrency="4",
        performance_concurrency="6",
        summary=(
            "12-23 GB NVIDIA stays on the embeddings/community preset so setup stays predictable on smaller cards."
        ),
        community_detail=(
            "Community capacity stays enabled on the embeddings/community preset today, with premium jobs unavailable on this machine."
        ),
        premium_detail=(
            "This NVIDIA class stays on the embeddings/community preset today, with premium jobs unavailable on this machine."
        ),
    ),
    NvidiaSupportPreset(
        key="llama8b_standard",
        label="24-47 GB NVIDIA",
        min_vram_gb=24.0,
        max_vram_gb=47.9,
        capacity_label="Llama 8B + embeddings",
        startup_model=DEFAULT_VLLM_MODEL,
        supported_models=(DEFAULT_VLLM_MODEL, DEFAULT_EMBEDDING_MODEL),
        recommended_profile="balanced",
        quiet_concurrency="1",
        balanced_concurrency="3",
        performance_concurrency="5",
        summary=(
            "24-47 GB NVIDIA runs the Llama 8B + embeddings preset with balanced tuning by default."
        ),
        community_detail=(
            "Community capacity stays enabled on the Llama 8B + embeddings preset now. Premium routing stays unavailable until approved hardware attestation is available."
        ),
        premium_detail=(
            "This NVIDIA class can serve the Llama 8B + embeddings preset and is eligible for premium routing after approval."
        ),
    ),
    NvidiaSupportPreset(
        key="large_premium_ready",
        label="48+ GB NVIDIA",
        min_vram_gb=48.0,
        max_vram_gb=None,
        capacity_label="Llama 8B + embeddings today",
        startup_model=DEFAULT_VLLM_MODEL,
        supported_models=(DEFAULT_VLLM_MODEL, DEFAULT_EMBEDDING_MODEL),
        recommended_profile="performance",
        quiet_concurrency="2",
        balanced_concurrency="4",
        performance_concurrency="6",
        summary=(
            "48+ GB NVIDIA uses the large preset today and keeps room for larger premium profiles later."
        ),
        community_detail=(
            "Community capacity stays enabled on the Llama 8B + embeddings preset today. Premium routing stays unavailable until approved hardware attestation is available."
        ),
        premium_detail=(
            "This large NVIDIA class is eligible for premium routing now and reserved for larger premium profiles later."
        ),
    ),
)


def format_bytes(value: int | None) -> str:
    if value is None or value <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


def format_rate_bytes_per_second(value: float | None) -> str | None:
    if value is None or value <= 0:
        return None
    return f"{format_bytes(int(round(value)))}/s"


def format_watts(value: float | None) -> str:
    if value is None or value <= 0:
        return "Unknown"
    rounded = round(float(value), 1)
    return f"{int(rounded)} W" if rounded.is_integer() else f"{rounded:.1f} W"


def format_usd_range(low: float | None, high: float | None) -> str:
    if low is None or high is None or low < 0 or high < 0 or high < low:
        return "Demand-based"
    if abs(low - high) < 0.05:
        return f"${low:.2f}/day"
    return f"${low:.2f}-${high:.2f}/day"


def estimate_install_heat_output_watts(
    *,
    gpu_memory_gb: Any,
    estimated_heat_output_watts: Any = None,
    power_watts: Any = None,
) -> float | None:
    configured_heat = coerce_float_or_none(estimated_heat_output_watts)
    if configured_heat is not None and configured_heat > 0:
        return configured_heat
    configured_power = coerce_float_or_none(power_watts)
    if configured_power is not None and configured_power > 0:
        return configured_power
    memory_gb = _safe_float(gpu_memory_gb, None)
    if memory_gb is None or memory_gb <= 0:
        return None
    if memory_gb >= 48:
        return 340.0
    if memory_gb >= 24:
        return 275.0
    if memory_gb >= 12:
        return 185.0
    return 120.0


def estimate_install_gross_earnings_range_per_day(
    *,
    gpu_memory_gb: Any,
    runtime_backend: str,
    setup_profile: str,
) -> tuple[float | None, float | None]:
    memory_gb = _safe_float(gpu_memory_gb, None)
    if memory_gb is None or memory_gb <= 0:
        return (None, None)
    if memory_gb >= 48:
        low, high = 4.0, 14.0
    elif memory_gb >= 24:
        low, high = 1.5, 6.0
    elif memory_gb >= 12:
        low, high = 0.5, 2.5
    else:
        low, high = 0.2, 1.2
    if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
        low *= 0.9
        high *= 0.9
    profile = (setup_profile or "balanced").strip().lower()
    if profile == "quiet":
        low *= 0.7
        high *= 0.7
    elif profile == "performance":
        low *= 1.15
        high *= 1.15
    return (round(low, 2), round(high, 2))


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def resolve_model_cache_budget(
    *,
    total_bytes: int | None,
    free_bytes: int | None,
    cache_bytes: int = 0,
    configured_budget_gb: Any = None,
    configured_reserve_free_gb: Any = None,
) -> dict[str, Any]:
    total_gb = (float(total_bytes) / (1024**3)) if total_bytes is not None and total_bytes > 0 else None
    free_gb = (float(free_bytes) / (1024**3)) if free_bytes is not None and free_bytes > 0 else None
    configured_budget = _safe_float(configured_budget_gb, None)
    configured_reserve = _safe_float(configured_reserve_free_gb, None)

    reserve_free_gb = configured_reserve if configured_reserve and configured_reserve > 0 else None
    if reserve_free_gb is None:
        if total_gb is None:
            reserve_free_gb = RECOMMENDED_FREE_DISK_GB
        else:
            reserve_target_gb = total_gb * MODEL_CACHE_DEFAULT_RESERVE_FRACTION
            if free_gb is not None and free_gb > 0:
                reserve_target_gb = min(reserve_target_gb, free_gb * 0.5)
            reserve_free_gb = max(RECOMMENDED_FREE_DISK_GB, round(reserve_target_gb, 1))

    max_budget_gb = max(0.0, (total_gb or 0.0) - reserve_free_gb) if total_gb is not None else None
    default_budget_gb = None
    if total_gb is not None:
        default_budget_gb = min(
            max_budget_gb or 0.0,
            max(
                MODEL_CACHE_MIN_BUDGET_GB,
                min(MODEL_CACHE_MAX_BUDGET_GB, total_gb * MODEL_CACHE_DEFAULT_BUDGET_FRACTION),
            ),
        )

    budget_source = "configured" if configured_budget and configured_budget > 0 else "derived"
    budget_gb = configured_budget if configured_budget and configured_budget > 0 else default_budget_gb
    if budget_gb is None:
        budget_gb = MODEL_CACHE_MIN_BUDGET_GB
    if max_budget_gb is not None:
        budget_gb = min(max_budget_gb, max(0.0, budget_gb))

    budget_bytes = int(round(budget_gb * (1024**3)))
    reserve_free_bytes = int(round(reserve_free_gb * (1024**3)))
    free_after_reserve_bytes = (
        max(0, int(free_bytes) - reserve_free_bytes) if free_bytes is not None else None
    )
    available_budget_bytes = max(0, budget_bytes - max(0, int(cache_bytes)))
    available_growth_bytes = (
        min(available_budget_bytes, free_after_reserve_bytes)
        if free_after_reserve_bytes is not None
        else available_budget_bytes
    )
    utilization_pct = round((max(0, int(cache_bytes)) / budget_bytes) * 100, 1) if budget_bytes > 0 else None
    if budget_gb >= 400:
        tier = "expanded"
    elif budget_gb >= 120:
        tier = "standard"
    else:
        tier = "compact"
    return {
        "total_bytes": total_bytes,
        "free_bytes": free_bytes,
        "budget_gb": round(budget_gb, 1),
        "budget_bytes": budget_bytes,
        "budget_label": format_bytes(budget_bytes),
        "budget_source": budget_source,
        "reserve_free_gb": round(reserve_free_gb, 1),
        "reserve_free_bytes": reserve_free_bytes,
        "reserve_free_label": format_bytes(reserve_free_bytes),
        "available_budget_bytes": available_budget_bytes,
        "available_growth_bytes": max(0, available_growth_bytes),
        "available_growth_label": format_bytes(max(0, available_growth_bytes)),
        "free_after_reserve_bytes": free_after_reserve_bytes,
        "utilization_pct": utilization_pct,
        "over_budget_bytes": max(0, int(cache_bytes) - budget_bytes),
        "tier": tier,
        "max_budget_gb": round(max_budget_gb, 1) if max_budget_gb is not None else None,
    }


def startup_model_artifact(model: str, runtime_engine: str | None = None) -> Any | None:
    for operation in ("responses", "embeddings"):
        artifact = find_model_artifact(model, operation, runtime_engine=runtime_engine)
        if artifact is not None:
            return artifact
    return None


def artifact_total_size_bytes(artifact: Any | None) -> int | None:
    if artifact is None:
        return None
    return sum(file.size_bytes for file in artifact.model_manifest.files) + sum(
        file.size_bytes for file in artifact.tokenizer_manifest.files
    )


def warmup_log_excerpt(text: str | None, *, max_lines: int = WARM_RUNTIME_LOG_EXCERPT_LINES) -> str | None:
    if not isinstance(text, str):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return " | ".join(lines[-max_lines:])


def installer_html() -> str:
    return Path(__file__).with_name("installer_ui.html").read_text(encoding="utf-8")


def session_bootstrap_html() -> bytes:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='referrer' content='no-referrer'>"
        "<meta http-equiv='cache-control' content='no-store'>"
        "<title>Opening local session...</title></head><body>"
        "<script>history.replaceState(null,'','/');window.location.replace('/');</script>"
        "</body></html>"
    ).encode("utf-8")


def run_command(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{args[0]} is not installed or is not available on PATH.") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"Command failed: {' '.join(args)}"
        raise RuntimeError(detail)
    return completed


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def serialize_env_values(env_values: dict[str, str]) -> str:
    lines = [f"{key}={env_values[key]}" for key in ENV_ORDER if key in env_values]
    return "\n".join(lines) + ("\n" if lines else "")


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def coerce_float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def csv_items(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    values = value.replace("\r", "\n").replace(";", "\n").replace(",", "\n").splitlines()
    items: list[str] = []
    seen: set[str] = set()
    for raw in values:
        candidate = raw.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        items.append(candidate)
    return items


def eta_label(seconds: int | None) -> str:
    if seconds is None:
        return "Waiting for your approval"
    if seconds <= 0:
        return "Live now"
    if seconds < 60:
        return "Less than a minute remaining"
    minutes = max(1, round(seconds / 60))
    if minutes == 1:
        return "About 1 minute remaining"
    return f"About {minutes} minutes remaining"


def locale_country_code() -> str | None:
    candidates = [
        locale.getlocale()[0],
        os.environ.get("LC_ALL"),
        os.environ.get("LANG"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        normalized = str(candidate).split(".", 1)[0].replace("-", "_")
        if "_" not in normalized:
            continue
        country = normalized.rsplit("_", 1)[-1].upper()
        if len(country) == 2 and country.isalpha():
            return country
    return None


def infer_node_region() -> tuple[str, str]:
    country = locale_country_code()
    offset = datetime.now().astimezone().utcoffset()
    offset_hours = int(offset.total_seconds() // 3600) if offset is not None else 0

    if country in EU_COUNTRY_CODES:
        return "eu-se-1", f"Recommended from the local locale ({country})."
    if country in APAC_COUNTRY_CODES:
        return "ap-sg-1", f"Recommended from the local locale ({country})."
    if country in AMERICAS_COUNTRY_CODES:
        region = "us-west-1" if offset_hours <= -7 else "us-east-1"
        return region, f"Recommended from the local locale ({country}) and timezone."
    if offset_hours >= 4:
        return "ap-sg-1", "Recommended from the local timezone."
    if offset_hours <= -7:
        return "us-west-1", "Recommended from the local timezone."
    if offset_hours <= -3:
        return "us-east-1", "Recommended from the local timezone."
    return "eu-se-1", "Recommended from the local timezone."


def nvidia_support_preset(memory_gb: float | None, gpu_name: str | None = None) -> NvidiaSupportPreset | None:
    if memory_gb is None:
        return None
    for preset in NVIDIA_SUPPORT_PRESETS:
        if preset.matches(memory_gb, gpu_name):
            return preset
    return NVIDIA_SUPPORT_PRESETS[0]


def recommended_startup_model(memory_gb: float | None, gpu_name: str | None = None) -> str:
    preset = nvidia_support_preset(memory_gb, gpu_name)
    if preset is not None:
        return preset.startup_model
    return DEFAULT_VLLM_MODEL


def recommended_supported_models(memory_gb: float | None, gpu_name: str | None = None) -> str:
    preset = nvidia_support_preset(memory_gb, gpu_name)
    if preset is not None:
        return preset.supported_models_csv()
    if recommended_startup_model(memory_gb, gpu_name) == DEFAULT_EMBEDDING_MODEL:
        return DEFAULT_EMBEDDING_MODEL
    return f"{DEFAULT_VLLM_MODEL},{DEFAULT_EMBEDDING_MODEL}"


def default_deployment_target(runtime_backend: str) -> str:
    return default_runtime_profile(runtime_backend).deployment_target


def default_inference_engine(runtime_backend: str) -> str:
    return default_runtime_profile(runtime_backend).inference_engine


def llama_cpp_env_for_model(model: str | None) -> dict[str, str]:
    source = llama_cpp_model_source(model)
    if source is None:
        return {
            "LLAMA_CPP_HF_REPO": "",
            "LLAMA_CPP_HF_FILE": "",
            "LLAMA_CPP_ALIAS": (model or "").strip(),
            "LLAMA_CPP_EMBEDDING": "false",
            "LLAMA_CPP_POOLING": "",
        }
    return {
        "LLAMA_CPP_HF_REPO": source.hf_repo,
        "LLAMA_CPP_HF_FILE": source.hf_file,
        "LLAMA_CPP_ALIAS": source.alias,
        "LLAMA_CPP_EMBEDDING": stringify_bool(source.embedding_enabled),
        "LLAMA_CPP_POOLING": source.pooling or "",
    }


def hugging_face_validation_target(
    model: str | None,
    *,
    inference_engine: str,
) -> tuple[str | None, bool]:
    candidate = (model or "").strip()
    if not candidate:
        return None, False
    if inference_engine == LLAMA_CPP_INFERENCE_ENGINE:
        source = llama_cpp_model_source(candidate)
        if source is not None:
            return source.hf_repo, False
    artifact = startup_model_artifact(candidate, runtime_engine=inference_engine)
    if artifact is None or artifact.source != "huggingface":
        return None, False
    repository = artifact.repository
    token_required = repository.startswith(HF_GATED_REPOSITORY_PREFIXES)
    return repository, token_required


def split_supported_models(models: str | None) -> list[str]:
    return [model.strip() for model in str(models or "").split(",") if model.strip()]


def filter_accessible_supported_models(
    models: str | None,
    *,
    token_configured: bool,
    inference_engine: str,
) -> str:
    accessible: list[str] = []
    for model in split_supported_models(models):
        _repository, token_required = hugging_face_validation_target(model, inference_engine=inference_engine)
        if token_required and not token_configured:
            continue
        if model not in accessible:
            accessible.append(model)
    if not accessible:
        accessible.append(DEFAULT_EMBEDDING_MODEL)
    return ",".join(accessible)


def constrain_supported_models_for_runtime_profile(
    models: str | None,
    *,
    runtime_profile: Any,
    preferred_model: str | None = None,
) -> str:
    allowed_models = list(getattr(runtime_profile, "supported_models", ()) or ())
    constrained = [model for model in split_supported_models(models) if model in allowed_models]
    preferred = (preferred_model or "").strip()
    if preferred and preferred in allowed_models and preferred not in constrained:
        constrained.insert(0, preferred)
    if not constrained:
        if preferred and preferred in allowed_models:
            constrained = [preferred]
        elif allowed_models:
            constrained = [allowed_models[0]]
        else:
            constrained = [DEFAULT_EMBEDDING_MODEL]
    return ",".join(dict.fromkeys(constrained))


def resolve_accessible_startup_selection(
    preferred_model: str | None,
    supported_models: str | None,
    *,
    token_configured: bool,
    inference_engine: str,
) -> tuple[str, str, bool]:
    preferred = (preferred_model or "").strip()
    accessible_supported_models = filter_accessible_supported_models(
        supported_models or preferred,
        token_configured=token_configured,
        inference_engine=inference_engine,
    )
    accessible_models = split_supported_models(accessible_supported_models)

    if preferred and preferred in accessible_models:
        return preferred, accessible_supported_models, False

    if preferred:
        repository, token_required = hugging_face_validation_target(preferred, inference_engine=inference_engine)
        if repository is None or not token_required or token_configured:
            return preferred, accessible_supported_models, False

    return accessible_models[0], accessible_supported_models, bool(preferred)


def resolve_startup_model_plan(
    target_model: str | None,
    target_supported_models: str | None,
    *,
    token_configured: bool,
    inference_engine: str,
    active_model: str | None = None,
    active_supported_models: str | None = None,
    bootstrap_first_run: bool = False,
) -> dict[str, Any]:
    resolved_target_model, resolved_target_supported_models, target_fallback = resolve_accessible_startup_selection(
        target_model,
        target_supported_models,
        token_configured=token_configured,
        inference_engine=inference_engine,
    )
    bootstrap_model, bootstrap_supported_models, _bootstrap_fallback = resolve_accessible_startup_selection(
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_EMBEDDING_MODEL,
        token_configured=token_configured,
        inference_engine=inference_engine,
    )
    if active_model or active_supported_models:
        resolved_active_model, resolved_active_supported_models, _active_fallback = resolve_accessible_startup_selection(
            active_model or resolved_target_model,
            active_supported_models or active_model or resolved_target_supported_models,
            token_configured=token_configured,
            inference_engine=inference_engine,
        )
    elif bootstrap_first_run and resolved_target_model != bootstrap_model:
        resolved_active_model = bootstrap_model
        resolved_active_supported_models = bootstrap_supported_models
    else:
        resolved_active_model = resolved_target_model
        resolved_active_supported_models = resolved_target_supported_models
    return {
        "active_model": resolved_active_model,
        "active_supported_models": resolved_active_supported_models,
        "target_model": resolved_target_model,
        "target_supported_models": resolved_target_supported_models,
        "target_fallback": target_fallback,
        "bootstrap_active": resolved_active_model != resolved_target_model,
    }


def detect_attestation_provider(command_runner: CommandRunner, cwd: Path) -> tuple[str, str]:
    if os.name == "nt" and shutil.which("powershell") is not None:
        try:
            completed = command_runner(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "$tpm = Get-Tpm; if ($tpm -and $tpm.TpmPresent -and $tpm.TpmReady) { 'hardware' } else { 'simulated' }",
                ],
                cwd,
            )
            if "hardware" in completed.stdout.lower():
                return "hardware", "A ready TPM was detected on this Windows machine."
        except RuntimeError:
            pass
    if Path("/sys/class/tpm/tpm0").exists() or Path("/dev/tpm0").exists():
        return "hardware", "A local TPM device was detected on this machine."
    return "simulated", "This machine does not expose a ready hardware attestation device yet."


def recommended_trust_tier(attestation_provider: str) -> str:
    return "restricted" if attestation_provider == "hardware" else "standard"


def recommended_restricted_capable(attestation_provider: str) -> bool:
    return attestation_provider == "hardware"


def premium_eligibility(
    memory_gb: float | None,
    attestation_provider: str,
    gpu_name: str | None = None,
) -> dict[str, str | bool]:
    preset = nvidia_support_preset(memory_gb, gpu_name)
    if preset is not None:
        if attestation_provider != "hardware":
            return {
                "premium_eligible": False,
                "premium_eligibility_status": "community_enabled",
                "premium_eligibility_label": "Community capacity enabled",
                "premium_eligibility_detail": preset.community_detail,
            }
        if preset.key in {"llama8b_standard", "large_premium_ready"}:
            return {
                "premium_eligible": True,
                "premium_eligibility_status": "premium_eligible",
                "premium_eligibility_label": "Premium capacity eligible",
                "premium_eligibility_detail": preset.premium_detail,
            }
        return {
            "premium_eligible": False,
            "premium_eligibility_status": "premium_unavailable",
            "premium_eligibility_label": "Premium capacity unavailable on this machine",
            "premium_eligibility_detail": preset.premium_detail,
        }

    if attestation_provider != "hardware":
        return {
            "premium_eligible": False,
            "premium_eligibility_status": "community_enabled",
            "premium_eligibility_label": "Community capacity enabled",
            "premium_eligibility_detail": (
                "This machine is ready for community capacity now. Premium capacity is unavailable on this machine right now."
            ),
        }
    if memory_gb is None or recommended_startup_model(memory_gb, gpu_name) != DEFAULT_VLLM_MODEL:
        return {
            "premium_eligible": False,
            "premium_eligibility_status": "premium_unavailable",
            "premium_eligibility_label": "Premium capacity unavailable on this machine",
            "premium_eligibility_detail": (
                "Community capacity stays enabled, but this machine is currently better suited to lighter workloads."
            ),
        }
    return {
        "premium_eligible": True,
        "premium_eligibility_status": "premium_eligible",
        "premium_eligibility_label": "Premium capacity eligible",
        "premium_eligibility_detail": (
            "Community capacity is enabled now, and this machine can move into premium capacity after approval."
        ),
    }


def stringify_bool(value: bool) -> str:
    return "true" if value else "false"


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return ""


def optional_env_value(value: Any) -> str:
    return "" if value is None else str(value)


def suggest_concurrency(memory_gb: float | None, gpu_name: str | None = None) -> str:
    return profile_concurrency(recommended_setup_profile(memory_gb, gpu_name), memory_gb, gpu_name)


def recommended_setup_profile(memory_gb: float | None, gpu_name: str | None = None) -> str:
    preset = nvidia_support_preset(memory_gb, gpu_name)
    if preset is not None:
        return preset.recommended_profile
    if memory_gb is None or memory_gb < 16:
        return "quiet"
    if memory_gb >= 40:
        return "performance"
    return "balanced"


def normalize_setup_profile(profile: str | None, memory_gb: float | None, gpu_name: str | None = None) -> str:
    normalized = (profile or "").strip().lower()
    if normalized in PROFILE_LABELS:
        return normalized
    return recommended_setup_profile(memory_gb, gpu_name)


def profile_concurrency(profile: str, memory_gb: float | None, gpu_name: str | None = None) -> str:
    preset = nvidia_support_preset(memory_gb, gpu_name)
    if preset is not None:
        return preset.concurrency_for_profile(profile)
    if profile == "quiet":
        if memory_gb is not None and memory_gb >= 48:
            return "2"
        return "1"
    if memory_gb is None:
        return "2" if profile == "balanced" else "3"
    if memory_gb < 24:
        return "2" if profile == "performance" else "1"
    if memory_gb < 48:
        if profile == "performance":
            return "3"
        return "2"
    if profile == "performance":
        return "4"
    return "3"


def profile_thermal_headroom(profile: str, fallback: float) -> str:
    if profile == "quiet":
        return "0.65"
    if profile == "performance":
        return "0.92"
    return f"{fallback:.2f}"


def profile_batch_tokens(profile: str, fallback: int) -> str:
    if profile == "quiet":
        return str(min(fallback, 32000))
    if profile == "performance":
        return str(max(fallback, 65000))
    return str(fallback)


def suggested_node_label(existing: str | None, detected_gpu: str | None, fallback: str) -> str:
    if existing and existing.strip():
        candidate = existing.strip()
        normalized = candidate.lower()
        if normalized not in {
            fallback.lower(),
            "autonomousc edge node",
            "autonomousc nordic node 01",
        } and not (normalized.startswith("autonomousc ") and normalized.endswith(" node 01")):
            return candidate
    if detected_gpu and detected_gpu.strip():
        cleaned = detected_gpu
        for token in ("NVIDIA", "GeForce", "AMD", "Radeon"):
            cleaned = cleaned.replace(token, "")
        compact = " ".join(part for part in cleaned.split() if part)
        if compact:
            return f"AUTONOMOUSc {compact} Node"
    return fallback


def profile_summary(profile: str, gpu_name: str, concurrency: str) -> str:
    label = PROFILE_LABELS.get(profile, PROFILE_LABELS["balanced"])
    if profile == "quiet":
        return f"{label} keeps this {gpu_name or 'machine'} cool and predictable with {concurrency} active workload at a time."
    if profile == "performance":
        return f"{label} pushes this {gpu_name or 'machine'} harder for throughput with up to {concurrency} concurrent workloads."
    return f"{label} is the recommended everyday setting for this {gpu_name or 'machine'}, with up to {concurrency} concurrent workloads."


def override_detail(selected: str, recommended: str, inferred_detail: str, *, override_label: str) -> str:
    if selected.strip() and selected.strip() != recommended.strip():
        return override_label
    return inferred_detail


def startup_mode_recommendation(autostart_status: dict[str, Any]) -> dict[str, str]:
    supported = bool(autostart_status.get("supported"))
    enabled = bool(autostart_status.get("enabled"))
    detail = str(autostart_status.get("detail") or "").strip()
    if supported:
        return {
            "startup_mode": "launch_on_sign_in",
            "startup_mode_label": "Launch on sign-in",
            "startup_mode_detail": (
                detail
                if enabled
                else "Quick Start can enable launch on sign-in automatically on this machine."
            ),
        }
    return {
        "startup_mode": "manual_reopen",
        "startup_mode_label": "Manual reopen",
        "startup_mode_detail": detail or "Automatic launch is unavailable here, so you would reopen the local node app manually after sign-in.",
    }


def detect_disk(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_gb = round(usage.free / (1024**3), 1)
    total_gb = round(usage.total / (1024**3), 1)
    return {
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "free_gb": free_gb,
        "total_gb": total_gb,
        "recommended_free_gb": RECOMMENDED_FREE_DISK_GB,
        "ok": free_gb >= RECOMMENDED_FREE_DISK_GB,
    }


def detect_gpu(command_runner: CommandRunner, cwd: Path) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"detected": False, "name": "", "memory_gb": None}

    try:
        completed = command_runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            cwd,
        )
    except RuntimeError:
        return {"detected": False, "name": "", "memory_gb": None}

    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line or "," not in first_line:
        return {"detected": False, "name": "", "memory_gb": None}

    name, memory = [part.strip() for part in first_line.split(",", 1)]
    try:
        memory_gb = round(float(memory) / 1024, 1)
    except ValueError:
        memory_gb = None
    return {"detected": True, "name": name, "memory_gb": memory_gb}


def detect_nvidia_container_runtime(command_runner: CommandRunner, cwd: Path) -> dict[str, Any]:
    try:
        completed = command_runner(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            cwd,
        )
    except RuntimeError as error:
        return {
            "checked": True,
            "visible": False,
            "error": str(error) or "Docker could not report configured container runtimes.",
        }

    runtime_payload = completed.stdout.strip().lower()
    if not runtime_payload:
        return {
            "checked": True,
            "visible": None,
            "error": None,
        }
    visible = "nvidia" in runtime_payload
    return {
        "checked": True,
        "visible": visible,
        "error": None if visible else "Docker is running, but the NVIDIA container runtime is not visible yet.",
    }


def detect_nvidia_driver_stack(command_runner: CommandRunner, cwd: Path) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {
            "checked": True,
            "present": False,
            "driver_version": None,
            "cuda_version": None,
            "error": "nvidia-smi is not installed or is not available on PATH.",
        }

    driver_version: str | None = None
    cuda_version: str | None = None
    error: str | None = None
    present = False

    try:
        driver_completed = command_runner(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            cwd,
        )
        first_driver_line = next((line.strip() for line in driver_completed.stdout.splitlines() if line.strip()), "")
        if first_driver_line:
            driver_version = first_driver_line.split(",", 1)[0].strip()
            present = True
    except RuntimeError as exc:
        error = str(exc) or "nvidia-smi could not read the local NVIDIA driver."

    try:
        summary_completed = command_runner(["nvidia-smi"], cwd)
        match = re.search(r"CUDA Version:\s*([0-9.]+)", summary_completed.stdout)
        if match:
            cuda_version = match.group(1)
            present = True
        elif driver_version:
            present = True
    except RuntimeError as exc:
        if error is None:
            error = str(exc) or "nvidia-smi could not report CUDA support."

    return {
        "checked": True,
        "present": present,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "error": error,
    }


def build_setup_check(
    key: str,
    label: str,
    status: str,
    summary: str,
    detail: str,
    fix: str,
    *,
    blocking: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "summary": summary,
        "detail": detail,
        "fix": fix,
        "blocking": blocking,
        "metadata": metadata or {},
    }


def summarize_setup_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for item in checks if item.get("status") == "pass")
    warnings = sum(1 for item in checks if item.get("status") == "warn")
    failed = sum(1 for item in checks if item.get("status") == "fail")
    blocking = [
        item for item in checks if item.get("status") == "fail" and bool(item.get("blocking"))
    ]
    ready_for_claim = not blocking
    return {
        "passed": passed,
        "warnings": warnings,
        "failed": failed,
        "total": len(checks),
        "blocking": len(blocking),
        "ready_for_claim": ready_for_claim,
        "summary": (
            f"{passed} of {len(checks)} setup checks are clear. Approval can start now."
            if ready_for_claim
            else f"{failed} setup checks still need attention before approval can start."
        ),
        "blocking_checks": blocking,
    }


def resolve_gpu_name(existing: str | None, detected: str | None, fallback: str) -> str:
    if existing and existing.strip() and existing.strip() not in {fallback, "Generic GPU"}:
        return existing.strip()
    if detected and detected.strip():
        return detected.strip()
    return fallback


def resolve_gpu_memory(existing: str | None, detected: float | None, fallback: float) -> str:
    if existing and existing.strip():
        try:
            parsed = float(existing.strip())
        except ValueError:
            parsed = None
        if parsed is not None and parsed != fallback:
            return existing.strip()
    if detected is not None:
        return str(detected)
    return str(fallback)


def seconds_until_timestamp(expires_at: str) -> int | None:
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    remaining = expiry - datetime.now(timezone.utc)
    return max(0, int(remaining.total_seconds()))


def approval_qr_svg_data_url(value: str) -> str | None:
    approval_url = value.strip()
    if not approval_url:
        return None
    try:
        import qrcode
        from qrcode.image.svg import SvgPathImage
    except ImportError:  # pragma: no cover - packaged installs include qrcode.
        return None

    qr = qrcode.QRCode(
        border=2,
        box_size=6,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
    )
    qr.add_data(approval_url)
    qr.make(fit=True)
    image = qr.make_image(image_factory=SvgPathImage)
    payload = io.BytesIO()
    image.save(payload)
    return "data:image/svg+xml;base64," + base64.b64encode(payload.getvalue()).decode("ascii")


@dataclass
class InstallerClaimState:
    claim_id: str
    claim_code: str
    approval_url: str
    expires_at: str
    poll_interval_seconds: int
    poll_token: str = ""
    status: str = "pending"
    renewal_count: int = 0
    auto_refreshes: bool = True
    approval_qr_svg_data_url: str | None = None


@dataclass
class InstallerState:
    stage: str = "idle"
    busy: bool = False
    message: str = "Review the defaults, then start the guided installer."
    error: str | None = None
    error_step: str | None = None
    logs: list[str] = field(default_factory=list)
    stage_context: dict[str, Any] = field(default_factory=dict)
    claim: InstallerClaimState | None = None
    started_at: float | None = None
    stage_started_at: float | None = None
    resume_config: dict[str, Any] = field(default_factory=dict)
    resume_requested: bool = False
    last_warm_source: dict[str, Any] = field(default_factory=dict)


class GuidedInstaller:
    def __init__(
        self,
        runtime_dir: Path | None = None,
        *,
        command_runner: CommandRunner = run_command,
        control_client_factory: ControlClientFactory = EdgeControlClient,
        autostart_manager: AutoStartManager | None = None,
        desktop_launcher_manager: DesktopLauncherManager | None = None,
        runtime_status_provider: RuntimeStatusProvider | None = None,
        runtime_controller: EmbeddedRuntimeSupervisor | None = None,
        sleep: SleepFn = time.sleep,
    ) -> None:
        self.runtime_dir = ensure_runtime_bundle(runtime_dir or resolve_runtime_dir())
        self.data_dir = self.runtime_dir / "data"
        self.service_dir = self.data_dir / "service"
        self.scratch_dir = self.data_dir / "scratch"
        self.credentials_dir = self.data_dir / "credentials"
        self.credentials_path = self.credentials_dir / "node-credentials.json"
        self.runtime_settings_path = self.service_dir / "runtime-settings.json"
        self.installer_state_path = self.service_dir / "installer-state.json"
        self.runtime_env_path = self.service_dir / "runtime.env"
        self.release_env_path = self.service_dir / "release.env"
        self.env_path = self.runtime_dir / ".env"
        self.example_env_path = self.runtime_dir / ".env.example"
        self.command_runner = command_runner
        self.control_client_factory = control_client_factory
        self.autostart_manager = autostart_manager or AutoStartManager(
            self.runtime_dir,
            command_runner=command_runner,
        )
        self.desktop_launcher_manager = desktop_launcher_manager or DesktopLauncherManager(
            self.runtime_dir,
            command_runner=command_runner,
        )
        self.runtime_backend = detect_runtime_backend()
        self.runtime_status_provider = runtime_status_provider or (lambda: {})
        self.runtime_controller = runtime_controller
        self.sleep = sleep
        self.faults = FaultInjectionController(self.scratch_dir / DEFAULT_FAULT_INJECTION_STATE_NAME)
        self.state = InstallerState()
        self.lock = threading.RLock()
        self.install_thread: threading.Thread | None = None
        self.preflight_probe_cache: dict[str, Any] = {}
        self.load_state()

    def current_runtime_backend(self) -> str:
        configured_backend = normalize_runtime_backend(os.getenv(RUNTIME_BACKEND_ENV))
        if configured_backend != AUTO_RUNTIME_BACKEND:
            return configured_backend
        persisted_backend = normalize_runtime_backend(self.load_persisted_env().get(RUNTIME_BACKEND_ENV))
        if persisted_backend != AUTO_RUNTIME_BACKEND:
            return persisted_backend
        return detect_runtime_backend()

    def offline_install_bundle_candidates(
        self,
        *,
        config: dict[str, Any] | None = None,
        env_values: dict[str, str] | None = None,
    ) -> list[tuple[str, Path]]:
        configured = first_nonempty(
            str((config or {}).get("offline_install_bundle_dir", "")),
            None if env_values is None else env_values.get("OFFLINE_INSTALL_BUNDLE_DIR"),
            self.load_persisted_env().get("OFFLINE_INSTALL_BUNDLE_DIR"),
        )
        if configured:
            return [("configured", Path(configured).expanduser())]
        return [
            ("bundled", self.service_dir / "offline-bundle"),
            ("bundled", self.data_dir / "offline-bundle"),
            ("bundled", self.runtime_dir / "offline-bundle"),
            ("bundled", bundled_runtime_dir() / "data" / "service" / "offline-bundle"),
            ("bundled", bundled_runtime_dir() / "offline-bundle"),
        ]

    def offline_install_bundle_source(
        self,
        *,
        config: dict[str, Any] | None = None,
        env_values: dict[str, str] | None = None,
    ) -> str:
        candidates = self.offline_install_bundle_candidates(config=config, env_values=env_values)
        if not candidates:
            return "none"
        source, first_path = candidates[0]
        if source == "configured":
            return "configured"
        if first_path.exists():
            return source
        for candidate_source, candidate_path in candidates[1:]:
            if candidate_path.exists():
                return candidate_source
        return "none"

    def resolved_runtime_profile(self, source: dict[str, str] | None = None):
        return resolve_runtime_profile(
            None if source is None else source.get("RUNTIME_PROFILE"),
            configured_engine=None if source is None else source.get("INFERENCE_ENGINE"),
            configured_deployment_target=None if source is None else source.get("DEPLOYMENT_TARGET"),
            runtime_backend=self.current_runtime_backend(),
            model=None if source is None else source.get("VLLM_MODEL"),
        )

    def resolved_deployment_target(self, source: dict[str, str] | None = None) -> str:
        return self.resolved_runtime_profile(source).deployment_target

    def resolved_inference_engine(self, source: dict[str, str] | None = None) -> str:
        return self.resolved_runtime_profile(source).inference_engine

    def inference_runtime_label(self, source: dict[str, str] | None = None) -> str:
        return inference_engine_label(self.resolved_inference_engine(source))

    def inference_readiness_path(self, source: dict[str, str] | None = None) -> str:
        return self.resolved_runtime_profile(source).readiness_path

    def runtime_backend_status(self) -> dict[str, Any]:
        status = dict(self.runtime_status_provider())
        if not status and self.runtime_controller is not None:
            status = dict(self.runtime_controller.snapshot())
        return status

    def docker_desktop_launcher(self) -> Path | None:
        candidates: list[Path] = []
        if os.name == "nt":
            for env_key in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
                root = os.environ.get(env_key)
                if root:
                    candidates.append(Path(root) / "Docker" / "Docker" / "Docker Desktop.exe")
            candidates.append(Path.home() / "AppData" / "Local" / "Docker" / "Docker" / "Docker Desktop.exe")
        elif sys.platform == "darwin":
            candidates.append(Path("/Applications/Docker.app/Contents/MacOS/Docker"))

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def start_docker_desktop_for_setup(self) -> dict[str, Any]:
        launcher = self.docker_desktop_launcher()
        if launcher is None:
            return {
                "key": "docker",
                "label": "Docker Desktop",
                "status": "warn",
                "changed": False,
                "resolved": False,
                "detail": "Docker Desktop is still missing locally, so Quick Start cannot start it automatically yet.",
            }
        try:
            subprocess.Popen([str(launcher)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as error:
            return {
                "key": "docker",
                "label": "Docker Desktop",
                "status": "warn",
                "changed": False,
                "resolved": False,
                "detail": f"Docker Desktop is installed, but Quick Start could not open it automatically: {error}",
            }
        return {
            "key": "docker",
            "label": "Docker Desktop",
            "status": "warn",
            "changed": True,
            "resolved": False,
            "detail": "Docker Desktop is opening now. Quick Start will keep checking until the engine is ready.",
        }

    def configured_offline_install_bundle_dir(
        self,
        *,
        config: dict[str, Any] | None = None,
        env_values: dict[str, str] | None = None,
    ) -> Path | None:
        candidates = self.offline_install_bundle_candidates(config=config, env_values=env_values)
        if not candidates:
            return None
        first_source, first_path = candidates[0]
        if first_source == "configured":
            return first_path
        for _source, candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def inspect_offline_install_bundle(
        self,
        *,
        config: dict[str, Any] | None = None,
        env_values: dict[str, str] | None = None,
        startup_model: str,
        inference_engine: str,
        runtime_backend: str,
    ) -> dict[str, Any]:
        bundle_dir = self.configured_offline_install_bundle_dir(config=config, env_values=env_values)
        bundle_source = self.offline_install_bundle_source(config=config, env_values=env_values)
        configured = bundle_dir is not None
        exists = bool(bundle_dir and bundle_dir.exists())
        model_cache_dir = None
        runtime_images_dir = None
        runtime_archives: list[Path] = []
        if bundle_dir and exists:
            for candidate in (bundle_dir / "model-cache", bundle_dir / "starter-model-cache"):
                if candidate.exists():
                    model_cache_dir = candidate
                    break
            for candidate in (bundle_dir / "runtime-images", bundle_dir / "docker-images"):
                if candidate.exists():
                    runtime_images_dir = candidate
                    break
            if runtime_images_dir is not None:
                runtime_archives = sorted(
                    [
                        entry
                        for entry in runtime_images_dir.iterdir()
                        if entry.is_file() and entry.suffix.lower() in {".tar", ".oci"}
                    ]
                )
        expected_bytes = artifact_total_size_bytes(
            startup_model_artifact(startup_model, runtime_engine=inference_engine)
        )
        starter_cache_bytes = directory_size_bytes(model_cache_dir) if model_cache_dir is not None else 0
        starter_cache_ready = starter_cache_bytes > 0
        runtime_images_ready = (
            not runtime_backend_supports_compose(runtime_backend)
            or bool(runtime_archives)
        )
        ready = exists and (starter_cache_ready or runtime_images_ready)
        source_label = "Bundled appliance starter" if bundle_source == "bundled" else "Local offline bundle"
        if not configured:
            summary = "Quick Start will use the connected install path."
            detail = (
                "No local appliance starter bundle is available yet. Quick Start will pull the supported runtime assets and fill the tiny starter-model cache from the network."
            )
            fix = "Optional: add an appliance starter bundle so Quick Start can preseed runtime assets and the tiny starter-model cache locally."
            status = "warn"
        elif bundle_source == "configured" and (not exists or bundle_dir is None):
            summary = "The configured offline install bundle is missing."
            detail = "Quick Start could not find the local bundle directory yet."
            fix = "Create the offline bundle directory or clear OFFLINE_INSTALL_BUNDLE_DIR so Quick Start falls back to network download cleanly."
            status = "warn"
        elif ready:
            summary = f"{source_label} is ready for first startup."
            detail_parts = []
            if runtime_archives:
                detail_parts.append(
                    f"{len(runtime_archives)} runtime image archive{'s' if len(runtime_archives) != 1 else ''} are ready"
                )
            if starter_cache_ready:
                detail_parts.append(
                    f"about {format_bytes(starter_cache_bytes)} of starter-model cache is ready for {startup_model}"
                )
            detail = (
                ". ".join(detail_parts) + "."
                if detail_parts
                else f"{source_label} can seed the first startup locally."
            )
            fix = "No fix needed."
            status = "pass"
        else:
            summary = f"{source_label} exists, but it is still sparse."
            detail = (
                "Quick Start can see the bundle directory, but it does not contain runtime image archives or starter-model cache yet."
            )
            fix = "Add runtime image archives or a starter-model cache snapshot to the bundle, or continue with the normal network path."
            status = "warn"
        return {
            "configured": configured,
            "path": str(bundle_dir) if bundle_dir is not None else "",
            "source": bundle_source,
            "source_label": source_label,
            "exists": exists,
            "ready": ready,
            "status": status,
            "summary": summary,
            "detail": detail,
            "fix": fix,
            "model_cache_dir": str(model_cache_dir) if model_cache_dir is not None else "",
            "runtime_images_dir": str(runtime_images_dir) if runtime_images_dir is not None else "",
            "runtime_archives": [str(path) for path in runtime_archives],
            "starter_cache_bytes": starter_cache_bytes,
            "expected_bytes": expected_bytes,
            "runtime_images_ready": runtime_images_ready,
            "starter_cache_ready": starter_cache_ready,
        }

    def ensure_windows_firewall_exceptions(self) -> dict[str, Any]:
        if os.name != "nt":
            return {
                "supported": False,
                "configured": False,
                "changed": False,
                "detail": "Quick Start only manages Windows Defender Firewall rules automatically on Windows.",
            }
        netsh_path = shutil.which("netsh")
        if not netsh_path or "netsh" not in Path(netsh_path).name.lower():
            return {
                "supported": False,
                "configured": False,
                "changed": False,
                "detail": "netsh is unavailable, so Quick Start cannot verify Windows firewall rules automatically.",
            }

        rules = [
            (f"{WINDOWS_FIREWALL_RULE_PREFIX} Local Service", LOCAL_SERVICE_PORT),
            (f"{WINDOWS_FIREWALL_RULE_PREFIX} Local Inference", LOCAL_INFERENCE_PORT),
        ]
        changed = False
        missing: list[str] = []
        for rule_name, port in rules:
            try:
                self.command_runner(
                    ["netsh", "advfirewall", "firewall", "show", "rule", f"name={rule_name}"],
                    self.runtime_dir,
                )
                continue
            except RuntimeError:
                missing.append(rule_name)
            try:
                self.command_runner(
                    [
                        "netsh",
                        "advfirewall",
                        "firewall",
                        "add",
                        "rule",
                        f"name={rule_name}",
                        "dir=in",
                        "action=allow",
                        "protocol=TCP",
                        f"localport={port}",
                        "profile=private,domain",
                        "remoteip=localsubnet",
                    ],
                    self.runtime_dir,
                )
                changed = True
            except RuntimeError as error:
                return {
                    "supported": True,
                    "configured": False,
                    "changed": changed,
                    "detail": (
                        "Quick Start could not add the local firewall exception automatically. "
                        f"{error}"
                    ),
                }
        if missing:
            return {
                "supported": True,
                "configured": True,
                "changed": changed,
                "detail": (
                    "Quick Start added local Windows firewall rules for the setup UI and local inference runtime."
                ),
            }
        return {
            "supported": True,
            "configured": True,
            "changed": False,
            "detail": "Windows firewall rules for the setup UI and local inference runtime are already present.",
        }

    def attempt_safe_preflight_repairs(
        self,
        *,
        config: dict[str, Any],
        env_values: dict[str, str],
        preflight: dict[str, Any],
    ) -> dict[str, Any]:
        actions: list[dict[str, Any]] = []

        for key, label, manager in (
            ("autostart", "Launch on sign-in", self.autostart_manager),
            ("desktop_launcher", "Desktop launcher", self.desktop_launcher_manager),
        ):
            ensure_enabled = getattr(manager, "ensure_enabled", None)
            current_status = getattr(manager, "status", None)
            try:
                current = current_status() if callable(current_status) else {}
            except Exception:
                current = {}
            if bool(current.get("enabled")) or not callable(ensure_enabled):
                continue
            try:
                status = ensure_enabled()
                enabled = bool(status.get("enabled"))
                actions.append(
                    {
                        "key": key,
                        "label": label,
                        "status": "pass" if enabled else "warn",
                        "changed": enabled,
                        "resolved": enabled,
                        "detail": str(
                            status.get("detail")
                            or (f"{label} is ready." if enabled else f"{label} still needs attention.")
                        ),
                    }
                )
            except Exception as error:
                actions.append(
                    {
                        "key": key,
                        "label": label,
                        "status": "warn",
                        "changed": False,
                        "resolved": False,
                        "detail": f"Quick Start could not repair {label.lower()} automatically: {error}",
                    }
                )

        firewall = self.ensure_windows_firewall_exceptions()
        if firewall.get("supported"):
            actions.append(
                {
                    "key": "firewall",
                    "label": "Firewall",
                    "status": "pass" if firewall.get("configured") else "warn",
                    "changed": bool(firewall.get("changed")),
                    "resolved": bool(firewall.get("configured")),
                    "detail": str(firewall.get("detail") or "Firewall status is unavailable."),
                }
            )

        runtime_backend = str(preflight.get("runtime_backend") or self.current_runtime_backend())
        if (
            runtime_backend_supports_compose(runtime_backend)
            and bool(preflight.get("docker_cli"))
            and bool(preflight.get("docker_compose"))
            and not bool(preflight.get("docker_daemon"))
        ):
            actions.append(self.start_docker_desktop_for_setup())

        changed = sum(1 for action in actions if bool(action.get("changed")))
        resolved = sum(1 for action in actions if bool(action.get("resolved")))
        pending = sum(1 for action in actions if action.get("status") == "warn")
        if not actions:
            summary = "No safe automatic fixes were needed before claim."
        elif pending and resolved:
            summary = (
                f"Quick Start repaired {resolved} local prerequisite{'s' if resolved != 1 else ''} and left "
                f"{pending} action{'s' if pending != 1 else ''} waiting on the machine."
            )
        elif resolved:
            summary = f"Quick Start repaired {resolved} local prerequisite{'s' if resolved != 1 else ''} automatically."
        else:
            summary = "Quick Start attempted the safe local repairs that do not need a browser or manual driver install."
        return {
            "attempted": bool(actions),
            "changed_count": changed,
            "resolved_count": resolved,
            "pending_count": pending,
            "summary": summary,
            "actions": actions,
        }

    def copy_bundle_tree(self, source_dir: Path, target_dir: Path) -> tuple[int, int]:
        copied_files = 0
        copied_bytes = 0
        if not source_dir.exists():
            return (copied_files, copied_bytes)
        for source_path in source_dir.rglob("*"):
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(source_dir)
            target_path = target_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_files += 1
            try:
                copied_bytes += source_path.stat().st_size
            except OSError:
                continue
        return (copied_files, copied_bytes)

    def apply_offline_install_bundle(
        self,
        *,
        env_values: dict[str, str],
        preflight: dict[str, Any],
    ) -> dict[str, Any]:
        bundle = (
            preflight.get("offline_install_bundle")
            if isinstance(preflight.get("offline_install_bundle"), dict)
            else self.inspect_offline_install_bundle(
                env_values=env_values,
                startup_model=str(env_values.get("VLLM_MODEL") or DEFAULT_VLLM_MODEL),
                inference_engine=self.resolved_inference_engine(env_values),
                runtime_backend=str(preflight.get("runtime_backend") or self.current_runtime_backend()),
            )
        )
        bundle_path = Path(str(bundle.get("path") or "")).expanduser() if bundle.get("path") else None
        if bundle_path is None or not bundle_path.exists():
            return {
                "used": False,
                "runtime_images_ready": False,
                "starter_cache_seeded": False,
                "message": "No local appliance starter bundle was available, so Quick Start will use the connected install path.",
            }

        actions: list[str] = []
        starter_cache_seeded = False
        model_cache_dir = Path(str(bundle.get("model_cache_dir") or "")) if bundle.get("model_cache_dir") else None
        if model_cache_dir is not None and model_cache_dir.exists():
            copied_files, copied_bytes = self.copy_bundle_tree(model_cache_dir, self.data_dir / "model-cache")
            if copied_files > 0:
                starter_cache_seeded = True
                actions.append(
                    f"Seeded {copied_files} starter-model cache file{'s' if copied_files != 1 else ''} "
                    f"({format_bytes(copied_bytes)}) from the local bundle."
                )

        runtime_images_ready = False
        runtime_archives = [
            Path(path)
            for path in bundle.get("runtime_archives", [])
            if isinstance(path, str) and path.strip()
        ]
        if runtime_backend_supports_compose(str(preflight.get("runtime_backend") or self.current_runtime_backend())):
            for archive_path in runtime_archives:
                self.log(f"Loading bundled runtime image from {archive_path.name} before Quick Start pulls anything.")
                self.command_runner(["docker", "load", "--input", str(archive_path)], self.runtime_dir)
            if runtime_archives:
                runtime_images_ready = True
                actions.append(
                    f"Loaded {len(runtime_archives)} runtime image archive{'s' if len(runtime_archives) != 1 else ''} from the local bundle."
                )
        else:
            runtime_images_ready = True

        if not actions:
            actions.append("The bundle exists, but Quick Start did not find reusable runtime images or starter-model cache inside it.")
        return {
            "used": bool(actions),
            "runtime_images_ready": runtime_images_ready,
            "starter_cache_seeded": starter_cache_seeded,
            "message": " ".join(actions),
        }

    def startup_model_warmup_diagnostics(
        self,
        env_values: dict[str, str] | None = None,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        runtime_env = dict(env_values or self.effective_runtime_env())
        runtime_profile = self.resolved_runtime_profile(runtime_env)
        inference_engine = runtime_profile.inference_engine
        runtime_label = inference_engine_label(inference_engine)
        startup_model = (model or "").strip() or runtime_env.get("VLLM_MODEL") or DEFAULT_VLLM_MODEL
        supported_operations = tuple(
            operation.strip() for operation in getattr(runtime_profile, "supported_apis", ()) if operation.strip()
        )
        supported_models = tuple(
            candidate.strip()
            for candidate in getattr(runtime_profile, "supported_models", ())
            if isinstance(candidate, str) and candidate.strip()
        )
        supports_model = any(
            find_model_artifact(startup_model, operation, runtime_engine=inference_engine) is not None
            or find_gguf_artifact(startup_model, operation) is not None
            for operation in supported_operations
        )
        configured_max_context_tokens = _safe_int(
            runtime_env.get("MAX_CONTEXT_TOKENS"),
            NodeAgentSettings().max_context_tokens,
        )
        safe_context_limit = known_safe_max_model_len(startup_model)
        gguf_context_limit: int | None = None
        for operation in supported_operations:
            gguf_artifact = find_gguf_artifact(startup_model, operation)
            if gguf_artifact is not None:
                gguf_context_limit = gguf_artifact.expected_context_tokens
                break
        limit_candidates = [value for value in (safe_context_limit, gguf_context_limit) if isinstance(value, int) and value > 0]
        context_limit_tokens = min(limit_candidates) if limit_candidates else None
        if safe_context_limit is not None and safe_context_limit == context_limit_tokens:
            context_limit_source = "safe_limit"
        elif gguf_context_limit is not None and gguf_context_limit == context_limit_tokens:
            context_limit_source = "artifact"
        else:
            context_limit_source = None
        expected_effective_context_tokens = (
            min(configured_max_context_tokens, context_limit_tokens)
            if context_limit_tokens is not None
            else configured_max_context_tokens
        )
        startup_artifact = startup_model_artifact(startup_model, runtime_engine=inference_engine)
        expected_bytes = artifact_total_size_bytes(startup_artifact)
        error: str | None = None
        error_kind: str | None = None
        if not supports_model:
            if supported_models:
                error = (
                    f"This node cannot run {startup_model} on the local {runtime_label} runtime. "
                    f"The selected runtime profile only supports {', '.join(supported_models)}. "
                    "Pick one of those models or switch to a matching runtime profile."
                )
            else:
                error = (
                    f"This node cannot run {startup_model} on the local {runtime_label} runtime because "
                    "no compatible startup artifact is bundled for its supported operations."
                )
            error_kind = "unsupported_model"
        elif context_limit_tokens is not None and configured_max_context_tokens > context_limit_tokens:
            if context_limit_source == "safe_limit":
                error = (
                    f"This node cannot run {startup_model} with MAX_CONTEXT_TOKENS={configured_max_context_tokens}. "
                    f"The safe limit for this model is {context_limit_tokens} tokens. "
                    "Lower MAX_CONTEXT_TOKENS and retry Quick Start."
                )
            else:
                error = (
                    f"This node cannot run {startup_model} with MAX_CONTEXT_TOKENS={configured_max_context_tokens}. "
                    f"The local {runtime_label} artifact for this model only supports {context_limit_tokens} tokens. "
                    "Lower MAX_CONTEXT_TOKENS and retry Quick Start."
                )
            error_kind = "max_context"
        return {
            "startup_model": startup_model,
            "runtime_profile": runtime_profile,
            "runtime_label": runtime_label,
            "inference_engine": inference_engine,
            "supported_models": supported_models,
            "supported_operations": supported_operations,
            "supports_model": supports_model,
            "configured_max_context_tokens": configured_max_context_tokens,
            "context_limit_tokens": context_limit_tokens,
            "context_limit_source": context_limit_source,
            "expected_effective_context_tokens": expected_effective_context_tokens,
            "expected_bytes": expected_bytes,
            "error": error,
            "error_kind": error_kind,
        }

    def warm_source_payload(
        self,
        *,
        env_values: dict[str, str],
        startup_model: str,
        inference_engine: str,
        observed_cache_bytes: int | None = None,
        offline_bundle: dict[str, Any] | None = None,
        offline_bundle_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_bytes = max(0, int(observed_cache_bytes or 0))
        bundle = (
            dict(offline_bundle)
            if isinstance(offline_bundle, dict)
            else self.inspect_offline_install_bundle(
                env_values=env_values,
                startup_model=startup_model,
                inference_engine=inference_engine,
                runtime_backend=self.current_runtime_backend(),
            )
        )
        bundle_report = dict(offline_bundle_report) if isinstance(offline_bundle_report, dict) else {}
        bundle_seeded = bool(bundle_report.get("starter_cache_seeded"))
        bundle_ready = bool(bundle.get("ready"))
        bundle_label = str(bundle.get("source_label") or "Offline appliance bundle")
        mirror_urls = csv_items(env_values.get("ARTIFACT_MIRROR_BASE_URLS"))
        hf_repository, _hf_token_required = hugging_face_validation_target(
            startup_model,
            inference_engine=inference_engine,
        )
        startup_artifact = startup_model_artifact(startup_model, runtime_engine=inference_engine)
        direct_hf_available = bool(
            hf_repository
            or (
                startup_artifact is not None
                and isinstance(getattr(startup_artifact, "source", None), str)
                and getattr(startup_artifact, "source") == "huggingface"
            )
        )
        order = [
            {
                "key": "local_cache",
                "label": "Local cache",
                "available": cache_bytes > 0,
                "detail": (
                    f"About {format_bytes(cache_bytes)} is already present in the local model cache."
                    if cache_bytes > 0
                    else "No reusable local model-cache bytes were found yet."
                ),
            },
            {
                "key": "offline_appliance_bundle",
                "label": "Offline appliance bundle",
                "available": bundle_seeded or bundle_ready,
                "detail": (
                    f"{bundle_label} already seeded the starter-model cache before warm-up began."
                    if bundle_seeded
                    else (
                        str(bundle.get("detail") or f"{bundle_label} is available for starter-model reuse.")
                        if bundle_ready
                        else f"{bundle_label} is not available for this warm-up."
                    )
                ),
            },
            {
                "key": "relay_cache_mirror",
                "label": "Relay/cache mirror",
                "available": bool(mirror_urls),
                "detail": (
                    f"{len(mirror_urls)} relay/cache mirror URL{'s' if len(mirror_urls) != 1 else ''} are configured."
                    if mirror_urls
                    else "No relay/cache mirror URLs are configured for remote warm-up."
                ),
            },
            {
                "key": "hugging_face",
                "label": "Hugging Face",
                "available": direct_hf_available,
                "detail": (
                    f"Direct Hugging Face access is available for {hf_repository}."
                    if hf_repository
                    else "Direct upstream model access remains available as the final remote fallback."
                ),
            },
        ]

        winner_key = "hugging_face"
        scope = "planned_remote"
        if bundle_seeded:
            winner_key = "offline_appliance_bundle"
            scope = "actual"
        elif cache_bytes > 0:
            winner_key = "local_cache"
            scope = "actual"
        elif mirror_urls:
            winner_key = "relay_cache_mirror"
        elif direct_hf_available:
            winner_key = "hugging_face"
        elif bundle_ready:
            winner_key = "offline_appliance_bundle"

        winner_lookup = {item["key"]: item for item in order}
        winner = winner_lookup.get(winner_key, order[-1])
        order_copy = "Warm path order: local cache -> offline appliance bundle -> relay/cache mirror -> Hugging Face."
        if winner_key == "local_cache":
            detail = (
                f"Warm-up is reusing about {format_bytes(cache_bytes)} already present in the local model cache. "
                f"{order_copy}"
            )
        elif winner_key == "offline_appliance_bundle":
            if bundle_seeded:
                detail = (
                    f"The first starter-model bytes came from {bundle_label.lower()}, which seeded the local cache before warm-up. "
                    f"{order_copy}"
                )
            else:
                detail = f"{bundle_label} is the next local warm-path source. {order_copy}"
        elif winner_key == "relay_cache_mirror":
            detail = (
                "No local starter bytes are ready, so the configured relay/cache mirror is the preferred remote warm path "
                "ahead of direct Hugging Face access. "
                f"{order_copy}"
            )
        else:
            repository_label = hf_repository or startup_model
            detail = (
                f"No local cache, offline bundle seed, or relay/cache mirror is ready, so warm-up is relying on direct "
                f"Hugging Face access for {repository_label}. {order_copy}"
            )

        return {
            "winner": winner_key,
            "winner_label": str(winner.get("label") or "Warm source"),
            "detail": detail,
            "scope": scope,
            "startup_model": startup_model,
            "mirror_urls": mirror_urls,
            "order": order,
            "selected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    def persist_last_warm_source(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.state.last_warm_source = json.loads(json.dumps(payload, default=str))
            self.persist_state_unlocked()

    def recent_runtime_logs(self, service: str = "vllm", *, tail_lines: int = WARM_RUNTIME_LOG_TAIL_LINES) -> str:
        if self.current_runtime_backend() == SINGLE_CONTAINER_RUNTIME_BACKEND:
            relay = getattr(self.runtime_controller, "vllm_output_relay", None)
            if relay is None:
                return ""
            tail_text = getattr(relay, "tail_text", None)
            if callable(tail_text):
                try:
                    return str(tail_text() or "").strip()
                except Exception:
                    return ""
            return ""
        if not runtime_backend_supports_compose(self.current_runtime_backend()):
            return ""
        try:
            result = self.command_runner(
                self.compose_command(["logs", "--tail", str(max(1, int(tail_lines))), service]),
                self.runtime_dir,
            )
        except Exception:
            return ""
        return str(result.stdout or "").strip()

    def fail_warmup(self, message: str, *, kind: str, **context: Any) -> None:
        self.update_stage_progress(
            message,
            warm_failure_kind=kind,
            warm_failure_detail=message,
            **context,
        )
        self.log(message)
        raise RuntimeError(message)

    def diagnose_warmup_logs(
        self,
        *,
        startup_model: str,
        runtime_label: str,
        logs: str,
    ) -> dict[str, Any] | None:
        excerpt = warmup_log_excerpt(logs)
        normalized = logs.lower()
        if not normalized.strip():
            return None
        if any(marker in normalized for marker in WARMUP_HF_AUTH_MARKERS):
            return {
                "kind": "hugging_face_auth",
                "message": (
                    f"This node cannot run {startup_model} because Hugging Face denied the model download during "
                    "warm-up. Make sure HUGGING_FACE_HUB_TOKEN or HF_TOKEN is valid and approved for this model, "
                    "then retry Quick Start."
                ),
                "warm_runtime_log_excerpt": excerpt,
            }
        if any(marker in normalized for marker in WARMUP_HF_NOT_FOUND_MARKERS):
            return {
                "kind": "model_unavailable",
                "message": (
                    f"This node cannot run {startup_model} because the model files could not be found during warm-up. "
                    "Check that the startup model name and revision are valid, then retry Quick Start."
                ),
                "warm_runtime_log_excerpt": excerpt,
            }
        if "no space left on device" in normalized or "not enough space" in normalized:
            return {
                "kind": "insufficient_disk",
                "message": (
                    f"This node cannot finish warming {startup_model} because the local runtime ran out of disk space "
                    "while writing model files. Free more disk space and retry Quick Start."
                ),
                "warm_runtime_log_excerpt": excerpt,
            }
        if any(marker in normalized for marker in WARMUP_OOM_MARKERS):
            return {
                "kind": "gpu_memory",
                "message": (
                    f"This node cannot run {startup_model} on its current GPU because warm-up ran out of VRAM. "
                    "Choose a smaller startup model or lower MAX_CONTEXT_TOKENS, then retry Quick Start."
                ),
                "warm_runtime_log_excerpt": excerpt,
            }
        if any(marker in normalized for marker in WARMUP_UNSUPPORTED_MODEL_MARKERS):
            return {
                "kind": "unsupported_model",
                "message": (
                    f"This node cannot run {startup_model} on the local {runtime_label} runtime because the model "
                    "format or architecture is not supported here. Switch to a supported startup model and retry."
                ),
                "warm_runtime_log_excerpt": excerpt,
            }
        return None

    def cached_preflight_network_payload(
        self,
        *,
        signature: dict[str, Any],
        force_refresh: bool,
        resolver: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if not force_refresh:
            with self.lock:
                cached_signature = self.preflight_probe_cache.get("signature")
                cached_checked_at = float(self.preflight_probe_cache.get("checked_at") or 0.0)
                cached_payload = self.preflight_probe_cache.get("payload")
                if (
                    cached_signature == signature
                    and isinstance(cached_payload, dict)
                    and (time.time() - cached_checked_at) < PRECHECK_NETWORK_CACHE_TTL_SECONDS
                ):
                    return dict(cached_payload)

        payload = resolver()
        with self.lock:
            self.preflight_probe_cache = {
                "signature": signature,
                "checked_at": time.time(),
                "payload": dict(payload),
            }
        return payload

    def resolve_setup_dns(self, hosts: list[str]) -> dict[str, Any]:
        targets: list[dict[str, Any]] = []
        resolved_all = True
        for host in hosts:
            try:
                entries = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
                addresses = sorted(
                    {
                        record[-1][0]
                        for record in entries
                        if isinstance(record, tuple)
                        and len(record) >= 5
                        and isinstance(record[-1], tuple)
                        and record[-1]
                    }
                )
                targets.append(
                    {
                        "host": host,
                        "ok": True,
                        "addresses": addresses[:4],
                    }
                )
            except OSError as exc:
                resolved_all = False
                targets.append({"host": host, "ok": False, "error": str(exc)})
        return {"ok": resolved_all, "targets": targets}

    def probe_control_plane(self, edge_control_url: str | None) -> dict[str, Any]:
        if not edge_control_url:
            return {
                "ok": False,
                "status_code": None,
                "probe_url": None,
                "error": "EDGE_CONTROL_URL is blank.",
            }

        probe_url = edge_control_url.rstrip("/") or edge_control_url
        try:
            response = httpx.get(probe_url, timeout=2.0, follow_redirects=True)
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "status_code": None,
                "probe_url": probe_url,
                "error": str(exc),
            }

        payload: dict[str, Any] | None = None
        try:
            candidate = response.json()
            if isinstance(candidate, dict):
                payload = candidate
        except ValueError:
            payload = None

        ok = response.status_code < 500
        return {
            "ok": ok,
            "status_code": response.status_code,
            "probe_url": probe_url,
            "service": payload.get("service") if isinstance(payload, dict) else None,
            "status": payload.get("status") if isinstance(payload, dict) else None,
            "error": None if ok else f"{probe_url} returned HTTP {response.status_code}.",
        }

    def runtime_env_overrides(self) -> dict[str, str]:
        keys = (
            "EDGE_CONTROL_URL",
            "EDGE_CONTROL_FALLBACK_URLS",
            "OPERATOR_TOKEN",
            "NODE_ID",
            "NODE_KEY",
            "NODE_LABEL",
            "NODE_REGION",
            "TRUST_TIER",
            "RESTRICTED_CAPABLE",
            "CREDENTIALS_PATH",
            "AUTOPILOT_STATE_PATH",
            "HEAT_GOVERNOR_STATE_PATH",
            "CONTROL_PLANE_STATE_PATH",
            "RUNTIME_PROFILE",
            "DEPLOYMENT_TARGET",
            "INFERENCE_ENGINE",
            "RUNTIME_IMAGE",
            "INFERENCE_BASE_URL",
            "CAPACITY_CLASS",
            "TEMPORARY_NODE",
            "BURST_PROVIDER",
            "BURST_LEASE_ID",
            "BURST_LEASE_PHASE",
            "BURST_COST_CEILING_USD",
            "VLLM_BASE_URL",
            "VLLM_HOST",
            "VLLM_PORT",
            "VLLM_STARTUP_TIMEOUT_SECONDS",
            "VLLM_EXTRA_ARGS",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
            "GPU_NAME",
            "GPU_MEMORY_GB",
            "MAX_CONTEXT_TOKENS",
            "MAX_BATCH_TOKENS",
            "MAX_CONCURRENT_ASSIGNMENTS",
            "CONTROL_PLANE_GRACE_SECONDS",
            "CONTROL_PLANE_RETRY_FLOOR_SECONDS",
            "CONTROL_PLANE_RETRY_CAP_SECONDS",
            "ARTIFACT_MIRROR_BASE_URLS",
            "MODEL_CACHE_BUDGET_GB",
            "MODEL_CACHE_RESERVE_FREE_GB",
            "OFFLINE_INSTALL_BUNDLE_DIR",
            "HEAT_GOVERNOR_MODE",
            "TARGET_GPU_UTILIZATION_PCT",
            "MIN_GPU_MEMORY_HEADROOM_PCT",
            "THERMAL_HEADROOM",
            "HEAT_DEMAND",
            "ROOM_TEMP_C",
            "TARGET_TEMP_C",
            "OUTSIDE_TEMP_C",
            "GPU_TEMP_C",
            "GPU_TEMP_LIMIT_C",
            "POWER_WATTS",
            "ESTIMATED_HEAT_OUTPUT_WATTS",
            "GPU_POWER_LIMIT_ENABLED",
            "ENERGY_PRICE_KWH",
            "SUPPORTED_MODELS",
            "POLL_INTERVAL_SECONDS",
            "ATTESTATION_PROVIDER",
            "VLLM_MODEL",
            "OWNER_TARGET_MODEL",
            "OWNER_TARGET_SUPPORTED_MODELS",
            "LLAMA_CPP_HF_REPO",
            "LLAMA_CPP_HF_FILE",
            "LLAMA_CPP_ALIAS",
            "LLAMA_CPP_EMBEDDING",
            "LLAMA_CPP_POOLING",
            "DOCKER_IMAGE",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_TOKEN",
            "START_VLLM",
            "VLLM_IMAGE",
        )
        overrides: dict[str, str] = {}
        for key in keys:
            value = os.getenv(key)
            if value is None:
                continue
            stripped = value.strip()
            if stripped:
                overrides[key] = stripped
        return overrides

    def load_runtime_settings(self) -> dict[str, Any]:
        if not self.runtime_settings_path.exists():
            return {}
        try:
            payload = json.loads(self.runtime_settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def runtime_settings_to_env(self, payload: dict[str, Any]) -> dict[str, str]:
        config = payload.get("config")
        if not isinstance(config, dict):
            return {}

        defaults = NodeAgentSettings()
        hugging_face_token = first_nonempty(
            str(config.get("hugging_face_hub_token") or ""),
            str(config.get("hf_token") or ""),
        )
        return {
            RUNTIME_BACKEND_ENV: normalize_runtime_backend(str(config.get("runtime_backend") or "")),
            "SETUP_PROFILE": str(config.get("setup_profile") or ""),
            "EDGE_CONTROL_URL": first_nonempty(str(config.get("edge_control_url", "")), defaults.edge_control_url),
            "EDGE_CONTROL_FALLBACK_URLS": first_nonempty(str(config.get("edge_control_fallback_urls", ""))),
            "OPERATOR_TOKEN": str(config.get("operator_token") or ""),
            "NODE_ID": str(config.get("node_id") or ""),
            "NODE_KEY": str(config.get("node_key") or ""),
            "NODE_LABEL": first_nonempty(str(config.get("node_label", "")), defaults.node_label),
            "NODE_REGION": first_nonempty(str(config.get("node_region", "")), defaults.node_region),
            "TRUST_TIER": first_nonempty(str(config.get("trust_tier", "")), defaults.trust_tier),
            "RESTRICTED_CAPABLE": stringify_bool(bool(config.get("restricted_capable", defaults.restricted_capable))),
            "CREDENTIALS_PATH": first_nonempty(str(config.get("credentials_path", "")), defaults.credentials_path),
            "AUTOPILOT_STATE_PATH": first_nonempty(
                str(config.get("autopilot_state_path", "")),
                defaults.autopilot_state_path,
            ),
            "HEAT_GOVERNOR_STATE_PATH": first_nonempty(
                str(config.get("heat_governor_state_path", "")),
                defaults.heat_governor_state_path,
            ),
            "CONTROL_PLANE_STATE_PATH": first_nonempty(
                str(config.get("control_plane_state_path", "")),
                defaults.control_plane_state_path,
            ),
            "FAULT_INJECTION_STATE_PATH": first_nonempty(
                str(config.get("fault_injection_state_path", "")),
                defaults.fault_injection_state_path,
            ),
            "RUNTIME_PROFILE": first_nonempty(
                str(config.get("runtime_profile", "")),
                defaults.resolved_runtime_profile_id,
            ),
            "DEPLOYMENT_TARGET": first_nonempty(
                str(config.get("deployment_target", "")),
                defaults.resolved_deployment_target,
            ),
            "INFERENCE_ENGINE": first_nonempty(
                str(config.get("inference_engine", "")),
                defaults.resolved_inference_engine,
            ),
            "RUNTIME_IMAGE": first_nonempty(
                str(config.get("runtime_image", "")),
                str(config.get("vllm_image", "")),
                defaults.resolved_runtime_image,
            ),
            "INFERENCE_BASE_URL": first_nonempty(
                str(config.get("inference_base_url", "")),
                str(config.get("vllm_base_url", "")),
                defaults.resolved_inference_base_url,
            ),
            "CAPACITY_CLASS": first_nonempty(str(config.get("capacity_class", "")), defaults.resolved_capacity_class),
            "TEMPORARY_NODE": stringify_bool(coerce_bool(config.get("temporary_node"), defaults.temporary_node)),
            "BURST_PROVIDER": first_nonempty(str(config.get("burst_provider", "")), defaults.burst_provider or ""),
            "BURST_LEASE_ID": first_nonempty(str(config.get("burst_lease_id", ""))),
            "BURST_LEASE_PHASE": first_nonempty(str(config.get("burst_lease_phase", "")), defaults.burst_lease_phase or ""),
            "BURST_COST_CEILING_USD": optional_env_value(config.get("burst_cost_ceiling_usd")),
            "VLLM_BASE_URL": first_nonempty(
                str(config.get("vllm_base_url", "")),
                str(config.get("inference_base_url", "")),
                defaults.resolved_inference_base_url,
            ),
            "VLLM_STARTUP_TIMEOUT_SECONDS": str(
                _safe_int(config.get("vllm_startup_timeout_seconds"), defaults.vllm_startup_timeout_seconds)
            ),
            "VLLM_EXTRA_ARGS": first_nonempty(str(config.get("vllm_extra_args", "")), defaults.vllm_extra_args),
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": stringify_bool(
                coerce_bool(
                    config.get(
                        "vllm_memory_profiler_estimate_cudagraphs",
                        defaults.vllm_memory_profiler_estimate_cudagraphs,
                    ),
                    defaults.vllm_memory_profiler_estimate_cudagraphs,
                )
            ),
            "GPU_NAME": first_nonempty(str(config.get("gpu_name", "")), defaults.gpu_name),
            "GPU_MEMORY_GB": str(_safe_float(config.get("gpu_memory_gb"), defaults.gpu_memory_gb)),
            "MAX_CONTEXT_TOKENS": str(_safe_int(config.get("max_context_tokens"), defaults.max_context_tokens)),
            "MAX_BATCH_TOKENS": str(_safe_int(config.get("max_batch_tokens"), defaults.max_batch_tokens)),
            "MAX_CONCURRENT_ASSIGNMENTS": str(
                _safe_int(config.get("max_concurrent_assignments"), defaults.max_concurrent_assignments)
            ),
            "CONTROL_PLANE_GRACE_SECONDS": str(
                _safe_int(config.get("control_plane_grace_seconds"), defaults.control_plane_grace_seconds)
            ),
            "CONTROL_PLANE_RETRY_FLOOR_SECONDS": str(
                _safe_int(config.get("control_plane_retry_floor_seconds"), defaults.control_plane_retry_floor_seconds)
            ),
            "CONTROL_PLANE_RETRY_CAP_SECONDS": str(
                _safe_int(config.get("control_plane_retry_cap_seconds"), defaults.control_plane_retry_cap_seconds)
            ),
            "ARTIFACT_MIRROR_BASE_URLS": first_nonempty(str(config.get("artifact_mirror_base_urls", ""))),
            "MODEL_CACHE_BUDGET_GB": optional_env_value(config.get("model_cache_budget_gb")),
            "MODEL_CACHE_RESERVE_FREE_GB": optional_env_value(config.get("model_cache_reserve_free_gb")),
            "OFFLINE_INSTALL_BUNDLE_DIR": first_nonempty(str(config.get("offline_install_bundle_dir", ""))),
            "HEAT_GOVERNOR_MODE": first_nonempty(
                str(config.get("heat_governor_mode", "")),
                defaults.heat_governor_mode,
            ),
            "TARGET_GPU_UTILIZATION_PCT": str(
                _safe_int(config.get("target_gpu_utilization_pct"), defaults.target_gpu_utilization_pct)
            ),
            "MIN_GPU_MEMORY_HEADROOM_PCT": str(
                _safe_float(config.get("min_gpu_memory_headroom_pct"), defaults.min_gpu_memory_headroom_pct)
            ),
            "THERMAL_HEADROOM": str(_safe_float(config.get("thermal_headroom"), defaults.thermal_headroom)),
            "HEAT_DEMAND": first_nonempty(str(config.get("heat_demand", "")), defaults.heat_demand),
            "ROOM_TEMP_C": optional_env_value(config.get("room_temp_c")),
            "TARGET_TEMP_C": optional_env_value(config.get("target_temp_c")),
            "OUTSIDE_TEMP_C": optional_env_value(config.get("outside_temp_c")),
            "GPU_TEMP_C": optional_env_value(config.get("gpu_temp_c")),
            "GPU_TEMP_LIMIT_C": str(_safe_float(config.get("gpu_temp_limit_c"), defaults.gpu_temp_limit_c)),
            "POWER_WATTS": optional_env_value(config.get("power_watts")),
            "ESTIMATED_HEAT_OUTPUT_WATTS": optional_env_value(config.get("estimated_heat_output_watts")),
            "GPU_POWER_LIMIT_ENABLED": stringify_bool(
                coerce_bool(config.get("gpu_power_limit_enabled"), defaults.gpu_power_limit_enabled)
            ),
            "ENERGY_PRICE_KWH": optional_env_value(config.get("energy_price_kwh")),
            "SUPPORTED_MODELS": first_nonempty(str(config.get("supported_models", "")), defaults.supported_models),
            "POLL_INTERVAL_SECONDS": str(_safe_int(config.get("poll_interval_seconds"), defaults.poll_interval_seconds)),
            "ATTESTATION_PROVIDER": first_nonempty(
                str(config.get("attestation_provider", "")),
                defaults.attestation_provider,
            ),
            "VLLM_MODEL": first_nonempty(str(config.get("vllm_model", "")), DEFAULT_VLLM_MODEL),
            "OWNER_TARGET_MODEL": first_nonempty(
                str(config.get("owner_target_model", "")),
                str(config.get("vllm_model", "")),
                DEFAULT_VLLM_MODEL,
            ),
            "OWNER_TARGET_SUPPORTED_MODELS": first_nonempty(
                str(config.get("owner_target_supported_models", "")),
                str(config.get("supported_models", "")),
                defaults.supported_models,
            ),
            "LLAMA_CPP_HF_REPO": str(config.get("llama_cpp_hf_repo") or defaults.llama_cpp_hf_repo),
            "LLAMA_CPP_HF_FILE": str(config.get("llama_cpp_hf_file") or defaults.llama_cpp_hf_file),
            "LLAMA_CPP_ALIAS": str(config.get("llama_cpp_alias") or defaults.llama_cpp_alias),
            "LLAMA_CPP_EMBEDDING": stringify_bool(bool(config.get("llama_cpp_embedding", defaults.llama_cpp_embedding))),
            "LLAMA_CPP_POOLING": str(config.get("llama_cpp_pooling") or (defaults.llama_cpp_pooling or "")),
            "HUGGING_FACE_HUB_TOKEN": hugging_face_token,
            "HF_TOKEN": hugging_face_token,
            "DOCKER_IMAGE": first_nonempty(str(config.get("docker_image", "")), defaults.docker_image),
            "VLLM_IMAGE": first_nonempty(
                str(config.get("vllm_image", "")),
                str(config.get("runtime_image", "")),
                defaults.resolved_runtime_image,
            ),
        }

    def runtime_settings_payload(self, env_values: dict[str, str]) -> dict[str, Any]:
        defaults = NodeAgentSettings()
        hugging_face_token = first_nonempty(
            env_values.get("HUGGING_FACE_HUB_TOKEN"),
            env_values.get("HF_TOKEN"),
        )
        runtime_backend = normalize_runtime_backend(
            first_nonempty(env_values.get(RUNTIME_BACKEND_ENV), self.current_runtime_backend())
        )
        gpu_memory_gb = _safe_float(env_values.get("GPU_MEMORY_GB"), defaults.gpu_memory_gb)
        gpu_name = env_values.get("GPU_NAME", defaults.gpu_name)
        return {
            "version": RUNTIME_SETTINGS_VERSION,
            "config": {
                "runtime_backend": runtime_backend,
                "setup_profile": env_values.get("SETUP_PROFILE")
                or recommended_setup_profile(gpu_memory_gb, gpu_name),
                "edge_control_url": env_values.get("EDGE_CONTROL_URL", defaults.edge_control_url),
                "edge_control_fallback_urls": env_values.get("EDGE_CONTROL_FALLBACK_URLS", ""),
                "operator_token": env_values.get("OPERATOR_TOKEN", ""),
                "node_id": env_values.get("NODE_ID", ""),
                "node_key": env_values.get("NODE_KEY", ""),
                "node_label": env_values.get("NODE_LABEL", defaults.node_label),
                "node_region": env_values.get("NODE_REGION", defaults.node_region),
                "trust_tier": env_values.get("TRUST_TIER", defaults.trust_tier),
                "restricted_capable": env_values.get("RESTRICTED_CAPABLE", "").lower() == "true",
                "credentials_path": env_values.get("CREDENTIALS_PATH", defaults.credentials_path),
                "autopilot_state_path": env_values.get("AUTOPILOT_STATE_PATH", defaults.autopilot_state_path),
                "heat_governor_state_path": env_values.get(
                    "HEAT_GOVERNOR_STATE_PATH",
                    defaults.heat_governor_state_path,
                ),
                "control_plane_state_path": env_values.get(
                    "CONTROL_PLANE_STATE_PATH",
                    defaults.control_plane_state_path,
                ),
                "fault_injection_state_path": env_values.get(
                    "FAULT_INJECTION_STATE_PATH",
                    defaults.fault_injection_state_path,
                ),
                "runtime_profile": env_values.get("RUNTIME_PROFILE", defaults.resolved_runtime_profile_id),
                "deployment_target": env_values.get("DEPLOYMENT_TARGET", defaults.resolved_deployment_target),
                "inference_engine": env_values.get("INFERENCE_ENGINE", defaults.resolved_inference_engine),
                "runtime_image": env_values.get("RUNTIME_IMAGE", defaults.resolved_runtime_image),
                "inference_base_url": env_values.get(
                    "INFERENCE_BASE_URL",
                    env_values.get("VLLM_BASE_URL", defaults.resolved_inference_base_url),
                ),
                "capacity_class": env_values.get("CAPACITY_CLASS", defaults.resolved_capacity_class),
                "temporary_node": coerce_bool(env_values.get("TEMPORARY_NODE"), defaults.temporary_node),
                "burst_provider": env_values.get("BURST_PROVIDER", defaults.burst_provider or ""),
                "burst_lease_id": env_values.get("BURST_LEASE_ID", ""),
                "burst_lease_phase": env_values.get("BURST_LEASE_PHASE", defaults.burst_lease_phase or ""),
                "burst_cost_ceiling_usd": coerce_float_or_none(env_values.get("BURST_COST_CEILING_USD")),
                "vllm_base_url": env_values.get(
                    "VLLM_BASE_URL",
                    env_values.get("INFERENCE_BASE_URL", defaults.resolved_inference_base_url),
                ),
                "vllm_startup_timeout_seconds": _safe_int(
                    env_values.get("VLLM_STARTUP_TIMEOUT_SECONDS"),
                    defaults.vllm_startup_timeout_seconds,
                ),
                "vllm_extra_args": env_values.get("VLLM_EXTRA_ARGS", defaults.vllm_extra_args),
                "vllm_memory_profiler_estimate_cudagraphs": coerce_bool(
                    env_values.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
                    defaults.vllm_memory_profiler_estimate_cudagraphs,
                ),
                "gpu_name": env_values.get("GPU_NAME", defaults.gpu_name),
                "gpu_memory_gb": gpu_memory_gb,
                "max_context_tokens": _safe_int(env_values.get("MAX_CONTEXT_TOKENS"), defaults.max_context_tokens),
                "max_batch_tokens": _safe_int(env_values.get("MAX_BATCH_TOKENS"), defaults.max_batch_tokens),
                "max_concurrent_assignments": _safe_int(
                    env_values.get("MAX_CONCURRENT_ASSIGNMENTS"),
                    defaults.max_concurrent_assignments,
                ),
                "control_plane_grace_seconds": _safe_int(
                    env_values.get("CONTROL_PLANE_GRACE_SECONDS"),
                    defaults.control_plane_grace_seconds,
                ),
                "control_plane_retry_floor_seconds": _safe_int(
                    env_values.get("CONTROL_PLANE_RETRY_FLOOR_SECONDS"),
                    defaults.control_plane_retry_floor_seconds,
                ),
                "control_plane_retry_cap_seconds": _safe_int(
                    env_values.get("CONTROL_PLANE_RETRY_CAP_SECONDS"),
                    defaults.control_plane_retry_cap_seconds,
                ),
                "artifact_mirror_base_urls": env_values.get("ARTIFACT_MIRROR_BASE_URLS", ""),
                "model_cache_budget_gb": coerce_float_or_none(env_values.get("MODEL_CACHE_BUDGET_GB")),
                "model_cache_reserve_free_gb": coerce_float_or_none(env_values.get("MODEL_CACHE_RESERVE_FREE_GB")),
                "offline_install_bundle_dir": env_values.get("OFFLINE_INSTALL_BUNDLE_DIR", ""),
                "heat_governor_mode": env_values.get("HEAT_GOVERNOR_MODE", defaults.heat_governor_mode),
                "target_gpu_utilization_pct": _safe_int(
                    env_values.get("TARGET_GPU_UTILIZATION_PCT"),
                    defaults.target_gpu_utilization_pct,
                ),
                "min_gpu_memory_headroom_pct": _safe_float(
                    env_values.get("MIN_GPU_MEMORY_HEADROOM_PCT"),
                    defaults.min_gpu_memory_headroom_pct,
                ),
                "thermal_headroom": _safe_float(env_values.get("THERMAL_HEADROOM"), defaults.thermal_headroom),
                "heat_demand": env_values.get("HEAT_DEMAND", defaults.heat_demand),
                "room_temp_c": coerce_float_or_none(env_values.get("ROOM_TEMP_C")),
                "target_temp_c": coerce_float_or_none(env_values.get("TARGET_TEMP_C")),
                "outside_temp_c": coerce_float_or_none(env_values.get("OUTSIDE_TEMP_C")),
                "gpu_temp_c": coerce_float_or_none(env_values.get("GPU_TEMP_C")),
                "gpu_temp_limit_c": _safe_float(env_values.get("GPU_TEMP_LIMIT_C"), defaults.gpu_temp_limit_c),
                "power_watts": coerce_float_or_none(env_values.get("POWER_WATTS")),
                "estimated_heat_output_watts": coerce_float_or_none(env_values.get("ESTIMATED_HEAT_OUTPUT_WATTS")),
                "gpu_power_limit_enabled": coerce_bool(
                    env_values.get("GPU_POWER_LIMIT_ENABLED"),
                    defaults.gpu_power_limit_enabled,
                ),
                "energy_price_kwh": coerce_float_or_none(env_values.get("ENERGY_PRICE_KWH")),
                "supported_models": env_values.get("SUPPORTED_MODELS", defaults.supported_models),
                "poll_interval_seconds": _safe_int(
                    env_values.get("POLL_INTERVAL_SECONDS"),
                    defaults.poll_interval_seconds,
                ),
                "attestation_provider": env_values.get("ATTESTATION_PROVIDER", defaults.attestation_provider),
                "vllm_model": env_values.get("VLLM_MODEL", DEFAULT_VLLM_MODEL),
                "owner_target_model": env_values.get(
                    "OWNER_TARGET_MODEL",
                    env_values.get("VLLM_MODEL", DEFAULT_VLLM_MODEL),
                ),
                "owner_target_supported_models": env_values.get(
                    "OWNER_TARGET_SUPPORTED_MODELS",
                    env_values.get("SUPPORTED_MODELS", defaults.supported_models),
                ),
                "llama_cpp_hf_repo": env_values.get("LLAMA_CPP_HF_REPO", defaults.llama_cpp_hf_repo),
                "llama_cpp_hf_file": env_values.get("LLAMA_CPP_HF_FILE", defaults.llama_cpp_hf_file),
                "llama_cpp_alias": env_values.get("LLAMA_CPP_ALIAS", defaults.llama_cpp_alias),
                "llama_cpp_embedding": env_values.get("LLAMA_CPP_EMBEDDING", "").lower() == "true",
                "llama_cpp_pooling": env_values.get("LLAMA_CPP_POOLING", defaults.llama_cpp_pooling or ""),
                "hugging_face_hub_token": hugging_face_token,
                "docker_image": env_values.get("DOCKER_IMAGE", defaults.docker_image),
                "vllm_image": env_values.get("VLLM_IMAGE", ""),
            },
        }

    def load_persisted_env(self) -> dict[str, str]:
        values: dict[str, str] = {}
        settings_payload = self.load_runtime_settings()
        settings_env = self.runtime_settings_to_env(settings_payload)
        if settings_env:
            values.update(settings_env)
        if self.env_path.exists():
            values.update(parse_env_file(self.env_path))
        return values

    def load_effective_env(self) -> dict[str, str]:
        values = parse_env_file(self.example_env_path)
        values.update(self.load_persisted_env())
        values.update(self.runtime_env_overrides())
        return values

    def effective_runtime_env(self) -> dict[str, str]:
        values = self.load_effective_env()
        if self.env_path.exists():
            values.update(parse_env_file(self.env_path))
        values.update(self.runtime_env_overrides())
        return values

    def config_present(self) -> bool:
        return self.runtime_settings_path.exists() or self.env_path.exists()

    def write_runtime_settings(self, env_values: dict[str, str]) -> None:
        self.service_dir.mkdir(parents=True, exist_ok=True)
        tighten_private_path(self.service_dir, directory=True)
        self.runtime_settings_path.write_text(
            json.dumps(self.runtime_settings_payload(env_values), indent=2),
            encoding="utf-8",
        )
        tighten_private_path(self.runtime_settings_path)

    def write_runtime_env(self, env_values: dict[str, str]) -> None:
        self.service_dir.mkdir(parents=True, exist_ok=True)
        tighten_private_path(self.service_dir, directory=True)
        self.runtime_env_path.write_text(serialize_env_values(env_values), encoding="utf-8")
        tighten_private_path(self.runtime_env_path)

    def sync_runtime_env(self) -> dict[str, str]:
        env_values = self.effective_runtime_env()
        if not env_values:
            env_values = parse_env_file(self.example_env_path)
        self.write_runtime_env(env_values)
        return env_values

    def current_config(
        self,
        *,
        source: dict[str, str] | None = None,
        gpu: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = source or self.effective_runtime_env()
        persisted = self.load_persisted_env()
        gpu = gpu if gpu is not None else detect_gpu(self.command_runner, self.runtime_dir)
        default_settings = NodeAgentSettings()
        region_value, region_reason = infer_node_region()
        attestation_provider, attestation_reason = detect_attestation_provider(self.command_runner, self.runtime_dir)

        configured_gpu_name = first_nonempty(source.get("GPU_NAME"), default_settings.gpu_name)
        configured_gpu_memory = _safe_float(source.get("GPU_MEMORY_GB"), default_settings.gpu_memory_gb)
        gpu_name = resolve_gpu_name(None, gpu.get("name"), configured_gpu_name)
        gpu_memory_gb = resolve_gpu_memory(None, gpu.get("memory_gb"), configured_gpu_memory)
        numeric_gpu_memory = _safe_float(gpu_memory_gb, default_settings.gpu_memory_gb)
        support_preset = nvidia_support_preset(numeric_gpu_memory, gpu_name)
        inferred_startup_model = recommended_startup_model(numeric_gpu_memory, gpu_name)
        inferred_supported_models = recommended_supported_models(numeric_gpu_memory, gpu_name)
        inferred_trust_tier = recommended_trust_tier(attestation_provider)
        inferred_restricted_capable = recommended_restricted_capable(attestation_provider)
        runtime_backend = self.current_runtime_backend()
        offline_bundle_dir = self.configured_offline_install_bundle_dir(config=None, env_values=source)
        configured_runtime_profile = first_nonempty(persisted.get("RUNTIME_PROFILE"), source.get("RUNTIME_PROFILE"))
        profile_source_model = first_nonempty(
            persisted.get("VLLM_MODEL"),
            source.get("VLLM_MODEL"),
            inferred_startup_model,
        )
        runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=source.get("INFERENCE_ENGINE"),
            configured_deployment_target=source.get("DEPLOYMENT_TARGET"),
            runtime_backend=runtime_backend,
            model=profile_source_model,
        )
        inferred_deployment_target = runtime_profile.deployment_target
        inferred_inference_engine = runtime_profile.inference_engine
        saved_token_configured = bool(
            first_nonempty(persisted.get("HUGGING_FACE_HUB_TOKEN"), persisted.get("HF_TOKEN"))
        )
        preview_token_supplied = bool(
            first_nonempty(source.get("HUGGING_FACE_HUB_TOKEN"), source.get("HF_TOKEN"))
        ) and not saved_token_configured
        token_configured = saved_token_configured or preview_token_supplied
        credentials_present = self.credentials_path.exists() or bool(
            first_nonempty(source.get("NODE_ID"), persisted.get("NODE_ID"))
            and first_nonempty(source.get("NODE_KEY"), persisted.get("NODE_KEY"))
        )
        has_saved_runtime = self.config_present()
        configured_current_supported_models = first_nonempty(
            persisted.get("SUPPORTED_MODELS"),
            source.get("SUPPORTED_MODELS"),
            inferred_supported_models,
        )
        configured_current_model = first_nonempty(
            persisted.get("VLLM_MODEL"),
            source.get("VLLM_MODEL"),
            inferred_startup_model,
        )
        configured_target_supported_models = first_nonempty(
            persisted.get("OWNER_TARGET_SUPPORTED_MODELS"),
            source.get("OWNER_TARGET_SUPPORTED_MODELS"),
            persisted.get("SUPPORTED_MODELS"),
            source.get("SUPPORTED_MODELS"),
            inferred_supported_models,
        )
        configured_target_model = first_nonempty(
            persisted.get("OWNER_TARGET_MODEL"),
            source.get("OWNER_TARGET_MODEL"),
            persisted.get("VLLM_MODEL"),
            source.get("VLLM_MODEL"),
            inferred_startup_model,
        )
        startup_plan = resolve_startup_model_plan(
            configured_target_model,
            configured_target_supported_models,
            token_configured=token_configured,
            inference_engine=inferred_inference_engine,
            active_model=configured_current_model if has_saved_runtime else None,
            active_supported_models=configured_current_supported_models if has_saved_runtime else None,
            bootstrap_first_run=not has_saved_runtime and not credentials_present,
        )
        current_startup_model = str(startup_plan["active_model"])
        current_supported_models = str(startup_plan["active_supported_models"])
        owner_target_model = str(startup_plan["target_model"])
        owner_target_supported_models = str(startup_plan["target_supported_models"])
        startup_model_fallback = bool(startup_plan["target_fallback"])
        bootstrap_pending_upgrade = bool(startup_plan["bootstrap_active"])
        current_runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=inferred_inference_engine,
            configured_deployment_target=inferred_deployment_target,
            runtime_backend=runtime_backend,
            model=current_startup_model,
        )
        current_supported_models = constrain_supported_models_for_runtime_profile(
            current_supported_models,
            runtime_profile=current_runtime_profile,
            preferred_model=current_startup_model,
        )
        owner_target_runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=inferred_inference_engine,
            configured_deployment_target=inferred_deployment_target,
            runtime_backend=runtime_backend,
            model=owner_target_model,
        )
        owner_target_supported_models = constrain_supported_models_for_runtime_profile(
            owner_target_supported_models,
            runtime_profile=owner_target_runtime_profile,
            preferred_model=owner_target_model,
        )
        runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=inferred_inference_engine,
            configured_deployment_target=inferred_deployment_target,
            runtime_backend=runtime_backend,
            model=current_startup_model,
        )
        inferred_deployment_target = runtime_profile.deployment_target
        inferred_inference_engine = runtime_profile.inference_engine
        runtime_image = first_nonempty(
            persisted.get("RUNTIME_IMAGE"),
            source.get("RUNTIME_IMAGE"),
            persisted.get("VLLM_IMAGE"),
            source.get("VLLM_IMAGE"),
            runtime_profile.image,
        )
        hf_repository, hf_token_required = hugging_face_validation_target(
            current_startup_model,
            inference_engine=inferred_inference_engine,
        )
        llama_cpp_settings = llama_cpp_env_for_model(current_startup_model)
        profile = normalize_setup_profile(
            persisted.get("SETUP_PROFILE") or source.get("SETUP_PROFILE"),
            numeric_gpu_memory,
            gpu_name,
        )
        recommended_profile = recommended_setup_profile(numeric_gpu_memory, gpu_name)
        concurrency = first_nonempty(
            persisted.get("MAX_CONCURRENT_ASSIGNMENTS"),
            profile_concurrency(profile, numeric_gpu_memory, gpu_name),
        )
        target_gpu_utilization_pct = first_nonempty(
            persisted.get("TARGET_GPU_UTILIZATION_PCT"),
            str(default_settings.target_gpu_utilization_pct),
        )
        min_gpu_memory_headroom_pct = first_nonempty(
            persisted.get("MIN_GPU_MEMORY_HEADROOM_PCT"),
            str(default_settings.min_gpu_memory_headroom_pct),
        )
        thermal_headroom = first_nonempty(
            persisted.get("THERMAL_HEADROOM"),
            profile_thermal_headroom(profile, default_settings.thermal_headroom),
        )
        heat_governor_mode = first_nonempty(
            persisted.get("HEAT_GOVERNOR_MODE"),
            source.get("HEAT_GOVERNOR_MODE"),
            default_settings.heat_governor_mode,
        )
        disk = detect_disk(self.runtime_dir)
        cache_dir = self.data_dir / "model-cache"
        cache_budget = resolve_model_cache_budget(
            total_bytes=disk.get("total_bytes"),
            free_bytes=disk.get("free_bytes"),
            cache_bytes=directory_size_bytes(cache_dir),
            configured_budget_gb=first_nonempty(
                source.get("MODEL_CACHE_BUDGET_GB"),
                persisted.get("MODEL_CACHE_BUDGET_GB"),
            ),
            configured_reserve_free_gb=first_nonempty(
                source.get("MODEL_CACHE_RESERVE_FREE_GB"),
                persisted.get("MODEL_CACHE_RESERVE_FREE_GB"),
            ),
        )
        premium = premium_eligibility(numeric_gpu_memory, attestation_provider, gpu_name)

        return {
            "edge_control_url": first_nonempty(source.get("EDGE_CONTROL_URL"), default_settings.edge_control_url),
            "edge_control_fallback_urls": first_nonempty(
                source.get("EDGE_CONTROL_FALLBACK_URLS"),
                persisted.get("EDGE_CONTROL_FALLBACK_URLS"),
                "",
            ),
            "operator_token_present": bool(first_nonempty(source.get("OPERATOR_TOKEN"))),
            "node_id_present": bool(first_nonempty(source.get("NODE_ID"))),
            "node_label": suggested_node_label(source.get("NODE_LABEL"), gpu.get("name"), default_settings.node_label),
            "node_region": first_nonempty(source.get("NODE_REGION"), persisted.get("NODE_REGION"), region_value),
            "trust_tier": first_nonempty(
                source.get("TRUST_TIER"),
                persisted.get("TRUST_TIER"),
                inferred_trust_tier,
            ),
            "restricted_capable": (
                first_nonempty(
                    source.get("RESTRICTED_CAPABLE"),
                    persisted.get("RESTRICTED_CAPABLE"),
                    stringify_bool(inferred_restricted_capable),
                ).lower()
                == "true"
            ),
            "deployment_target": inferred_deployment_target,
            "deployment_target_label": deployment_target_label(inferred_deployment_target),
            "inference_engine": inferred_inference_engine,
            "inference_engine_label": inference_engine_label(inferred_inference_engine),
            "runtime_profile": runtime_profile.id,
            "runtime_profile_label": runtime_profile.label,
            "model_format": runtime_profile.model_format,
            "runtime_image": runtime_image,
            "inference_base_url": first_nonempty(
                persisted.get("INFERENCE_BASE_URL"),
                persisted.get("VLLM_BASE_URL"),
                default_settings.resolved_inference_base_url,
            ),
            "readiness_path": runtime_profile.readiness_path,
            "supported_apis": list(runtime_profile.supported_apis),
            "trust_policy": runtime_profile.trust_policy,
            "pricing_tier": runtime_profile.pricing_tier,
            "artifact_manifest_type": runtime_profile.artifact_manifest_type,
            "capacity_class": first_nonempty(
                source.get("CAPACITY_CLASS"),
                persisted.get("CAPACITY_CLASS"),
                runtime_profile.capacity_class,
            ),
            "routing_lane": runtime_profile.routing_lane,
            "routing_lane_label": runtime_profile.routing_lane_label,
            "routing_lane_detail": runtime_profile.routing_lane_detail,
            "routing_lane_policy_summary": runtime_profile.routing_lane_policy_summary,
            "routing_lane_allowed_privacy_tiers": list(runtime_profile.routing_lane_allowed_privacy_tiers),
            "routing_lane_allowed_result_guarantees": list(runtime_profile.routing_lane_allowed_result_guarantees),
            "routing_lane_allowed_trust_requirements": list(runtime_profile.routing_lane_allowed_trust_requirements),
            "max_privacy_tier": runtime_profile.max_privacy_tier,
            "exact_model_guarantee": runtime_profile.exact_model_guarantee,
            "quantized_output_disclosure_required": runtime_profile.quantized_output_disclosure_required,
            "trusted_eligibility": runtime_profile.trusted_eligibility,
            "burst_lifecycle": list(runtime_profile.burst_lifecycle),
            "burst_cost_ceiling_usd": first_nonempty(
                persisted.get("BURST_COST_CEILING_USD"),
                "" if runtime_profile.burst_cost_ceiling_usd is None else str(runtime_profile.burst_cost_ceiling_usd),
            ),
            "burst_provider": first_nonempty(persisted.get("BURST_PROVIDER"), "vast_ai" if runtime_profile.capacity_class == "elastic_burst" else ""),
            "burst_lease_id": persisted.get("BURST_LEASE_ID") or "",
            "burst_lease_phase": persisted.get("BURST_LEASE_PHASE") or "",
            "temporary_node": coerce_bool(persisted.get("TEMPORARY_NODE"), False),
            "vllm_base_url": first_nonempty(
                persisted.get("VLLM_BASE_URL"),
                persisted.get("INFERENCE_BASE_URL"),
                default_settings.resolved_inference_base_url,
            ),
            "vllm_startup_timeout_seconds": _safe_int(
                first_nonempty(
                    source.get("VLLM_STARTUP_TIMEOUT_SECONDS"),
                    persisted.get("VLLM_STARTUP_TIMEOUT_SECONDS"),
                ),
                default_settings.vllm_startup_timeout_seconds,
            ),
            "vllm_extra_args": first_nonempty(
                source.get("VLLM_EXTRA_ARGS"),
                persisted.get("VLLM_EXTRA_ARGS"),
                default_settings.vllm_extra_args,
            ),
            "vllm_memory_profiler_estimate_cudagraphs": coerce_bool(
                first_nonempty(
                    source.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
                    persisted.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
                ),
                default_settings.vllm_memory_profiler_estimate_cudagraphs,
            ),
            "vllm_model": current_startup_model,
            "supported_models": current_supported_models,
            "owner_target_model": owner_target_model,
            "owner_target_supported_models": owner_target_supported_models,
            "bootstrap_pending_upgrade": bootstrap_pending_upgrade,
            "startup_model_fallback": startup_model_fallback,
            "max_context_tokens": _safe_int(
                first_nonempty(source.get("MAX_CONTEXT_TOKENS"), persisted.get("MAX_CONTEXT_TOKENS")),
                default_settings.max_context_tokens,
            ),
            "max_concurrent_assignments": first_nonempty(
                source.get("MAX_CONCURRENT_ASSIGNMENTS"),
                concurrency,
            ),
            "control_plane_grace_seconds": first_nonempty(
                source.get("CONTROL_PLANE_GRACE_SECONDS"),
                persisted.get("CONTROL_PLANE_GRACE_SECONDS"),
                str(default_settings.control_plane_grace_seconds),
            ),
            "control_plane_retry_floor_seconds": first_nonempty(
                source.get("CONTROL_PLANE_RETRY_FLOOR_SECONDS"),
                persisted.get("CONTROL_PLANE_RETRY_FLOOR_SECONDS"),
                str(default_settings.control_plane_retry_floor_seconds),
            ),
            "control_plane_retry_cap_seconds": first_nonempty(
                source.get("CONTROL_PLANE_RETRY_CAP_SECONDS"),
                persisted.get("CONTROL_PLANE_RETRY_CAP_SECONDS"),
                str(default_settings.control_plane_retry_cap_seconds),
            ),
            "artifact_mirror_base_urls": first_nonempty(
                source.get("ARTIFACT_MIRROR_BASE_URLS"),
                persisted.get("ARTIFACT_MIRROR_BASE_URLS"),
                "",
            ),
            "model_cache_budget_gb": cache_budget["budget_gb"],
            "model_cache_budget_bytes": cache_budget["budget_bytes"],
            "model_cache_budget_label": cache_budget["budget_label"],
            "model_cache_budget_source": cache_budget["budget_source"],
            "model_cache_reserve_free_gb": cache_budget["reserve_free_gb"],
            "model_cache_reserve_free_bytes": cache_budget["reserve_free_bytes"],
            "model_cache_available_growth_bytes": cache_budget["available_growth_bytes"],
            "model_cache_available_growth_label": cache_budget["available_growth_label"],
            "model_cache_tier": cache_budget["tier"],
            "target_gpu_utilization_pct": first_nonempty(
                source.get("TARGET_GPU_UTILIZATION_PCT"),
                target_gpu_utilization_pct,
            ),
            "min_gpu_memory_headroom_pct": first_nonempty(
                source.get("MIN_GPU_MEMORY_HEADROOM_PCT"),
                min_gpu_memory_headroom_pct,
            ),
            "setup_profile": profile,
            "recommended_setup_profile": recommended_profile,
            "profile_summary": profile_summary(profile, gpu_name, concurrency),
            "runtime_backend": runtime_backend,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "gpu_support_key": support_preset.key if support_preset is not None else "unknown",
            "gpu_support_label": support_preset.label if support_preset is not None else "Automatic NVIDIA defaults",
            "gpu_support_track": support_preset.capacity_label if support_preset is not None else "Safe defaults",
            "gpu_support_summary": (
                support_preset.summary
                if support_preset is not None
                else "Quick Start keeps using safe defaults until a supported NVIDIA GPU is visible."
            ),
            "offline_install_bundle_dir": first_nonempty(
                source.get("OFFLINE_INSTALL_BUNDLE_DIR"),
                persisted.get("OFFLINE_INSTALL_BUNDLE_DIR"),
                str(offline_bundle_dir) if offline_bundle_dir is not None else "",
                "",
            ),
            "thermal_headroom": first_nonempty(source.get("THERMAL_HEADROOM"), thermal_headroom),
            "heat_governor_mode": first_nonempty(source.get("HEAT_GOVERNOR_MODE"), heat_governor_mode),
            "owner_objective": first_nonempty(
                source.get("OWNER_OBJECTIVE"),
                persisted.get("OWNER_OBJECTIVE"),
                default_settings.owner_objective,
            ),
            "heat_demand": first_nonempty(
                source.get("HEAT_DEMAND"),
                persisted.get("HEAT_DEMAND"),
                default_settings.heat_demand,
            ),
            "room_temp_c": first_nonempty(source.get("ROOM_TEMP_C"), persisted.get("ROOM_TEMP_C"), ""),
            "target_temp_c": first_nonempty(source.get("TARGET_TEMP_C"), persisted.get("TARGET_TEMP_C"), ""),
            "outside_temp_c": first_nonempty(source.get("OUTSIDE_TEMP_C"), persisted.get("OUTSIDE_TEMP_C"), ""),
            "quiet_hours_start_local": first_nonempty(
                source.get("QUIET_HOURS_START_LOCAL"),
                persisted.get("QUIET_HOURS_START_LOCAL"),
                "",
            ),
            "quiet_hours_end_local": first_nonempty(
                source.get("QUIET_HOURS_END_LOCAL"),
                persisted.get("QUIET_HOURS_END_LOCAL"),
                "",
            ),
            "gpu_temp_c": first_nonempty(source.get("GPU_TEMP_C"), persisted.get("GPU_TEMP_C"), ""),
            "gpu_temp_limit_c": first_nonempty(
                source.get("GPU_TEMP_LIMIT_C"),
                persisted.get("GPU_TEMP_LIMIT_C"),
                str(default_settings.gpu_temp_limit_c),
            ),
            "power_watts": first_nonempty(source.get("POWER_WATTS"), persisted.get("POWER_WATTS"), ""),
            "estimated_heat_output_watts": first_nonempty(
                source.get("ESTIMATED_HEAT_OUTPUT_WATTS"),
                persisted.get("ESTIMATED_HEAT_OUTPUT_WATTS"),
                "",
            ),
            "gpu_power_limit_enabled": coerce_bool(
                first_nonempty(source.get("GPU_POWER_LIMIT_ENABLED"), persisted.get("GPU_POWER_LIMIT_ENABLED")),
                default_settings.gpu_power_limit_enabled,
            ),
            "max_power_cap_watts": first_nonempty(
                source.get("MAX_POWER_CAP_WATTS"),
                persisted.get("MAX_POWER_CAP_WATTS"),
                "",
            ),
            "energy_price_kwh": first_nonempty(
                source.get("ENERGY_PRICE_KWH"),
                persisted.get("ENERGY_PRICE_KWH"),
                "",
            ),
            "attestation_provider": first_nonempty(
                source.get("ATTESTATION_PROVIDER"),
                persisted.get("ATTESTATION_PROVIDER"),
                attestation_provider,
            ),
            "hugging_face_repository": hf_repository,
            "hugging_face_token_required": hf_token_required,
            "hugging_face_token_configured": token_configured,
            "hugging_face_token_saved": saved_token_configured,
            "hugging_face_token_preview_supplied": preview_token_supplied,
            "llama_cpp_hf_repo": llama_cpp_settings["LLAMA_CPP_HF_REPO"],
            "llama_cpp_hf_file": llama_cpp_settings["LLAMA_CPP_HF_FILE"],
            "llama_cpp_alias": llama_cpp_settings["LLAMA_CPP_ALIAS"],
            "llama_cpp_embedding": llama_cpp_settings["LLAMA_CPP_EMBEDDING"].lower() == "true",
            "llama_cpp_pooling": llama_cpp_settings["LLAMA_CPP_POOLING"],
            "recommended_node_region": region_value,
            "region_reason": region_reason,
            "recommended_model": inferred_startup_model,
            "recommended_supported_models": inferred_supported_models,
            "recommended_max_context_tokens": (
                support_preset.max_context_tokens
                if support_preset is not None and support_preset.max_context_tokens is not None
                else default_settings.max_context_tokens
            ),
            "recommended_vllm_startup_timeout_seconds": (
                support_preset.vllm_startup_timeout_seconds
                if support_preset is not None and support_preset.vllm_startup_timeout_seconds is not None
                else default_settings.vllm_startup_timeout_seconds
            ),
            "recommended_vllm_extra_args": (
                support_preset.vllm_extra_args
                if support_preset is not None
                else default_settings.vllm_extra_args
            ),
            "recommended_vllm_memory_profiler_estimate_cudagraphs": (
                support_preset.runtime_env_defaults().get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "").lower()
                == "true"
                if support_preset is not None
                else default_settings.vllm_memory_profiler_estimate_cudagraphs
            ),
            "recommended_max_concurrent_assignments": profile_concurrency(
                recommended_profile,
                numeric_gpu_memory,
                gpu_name,
            ),
            "recommended_target_gpu_utilization_pct": default_settings.target_gpu_utilization_pct,
            "recommended_min_gpu_memory_headroom_pct": default_settings.min_gpu_memory_headroom_pct,
            "recommended_thermal_headroom": profile_thermal_headroom(
                recommended_profile,
                default_settings.thermal_headroom,
            ),
            "recommended_trust_tier": inferred_trust_tier,
            "recommended_restricted_capable": inferred_restricted_capable,
            "attestation_reason": attestation_reason,
            "runtime_backend": runtime_backend,
            "runtime_backend_label": runtime_backend_label(runtime_backend),
            **premium,
        }

    def collect_preflight(
        self,
        *,
        gpu: dict[str, Any] | None = None,
        env_values: dict[str, str] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        self.ensure_data_dirs()
        if env_values is None:
            self.sync_runtime_env()
        runtime_backend = self.current_runtime_backend()
        docker_cli = shutil.which("docker") is not None
        compose_ok = False
        daemon_ok = False
        docker_error: str | None = None
        running_services: list[str] = []

        if runtime_backend_supports_compose(runtime_backend) and docker_cli:
            try:
                self.command_runner(["docker", "compose", "version"], self.runtime_dir)
                compose_ok = True
                self.command_runner(["docker", "info"], self.runtime_dir)
                daemon_ok = True
            except RuntimeError as error:
                docker_error = str(error)
        elif runtime_backend_supports_compose(runtime_backend):
            docker_error = "Docker is not installed or is not available on PATH."
        else:
            docker_cli = True
            compose_ok = True
            daemon_ok = True

        if runtime_backend_supports_compose(runtime_backend) and docker_cli and compose_ok and daemon_ok:
            try:
                completed = self.command_runner(
                    ["docker", "compose", "ps", "--services", "--status", "running"],
                    self.runtime_dir,
                )
                running_services = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            except RuntimeError:
                running_services = []
        elif runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
            runtime_status = self.runtime_backend_status()
            running_services = [
                str(service)
                for service in runtime_status.get("running_services", [])
                if isinstance(service, str) and service.strip()
            ]

        gpu_payload = gpu if gpu is not None else detect_gpu(self.command_runner, self.runtime_dir)
        nvidia_driver = detect_nvidia_driver_stack(self.command_runner, self.runtime_dir)
        nvidia_container_runtime = {
            "checked": False,
            "visible": None,
            "error": None,
        }
        if (
            runtime_backend_supports_compose(runtime_backend)
            and docker_cli
            and compose_ok
            and daemon_ok
            and bool(gpu_payload.get("detected"))
        ):
            nvidia_container_runtime = detect_nvidia_container_runtime(self.command_runner, self.runtime_dir)

        disk = detect_disk(self.runtime_dir)
        runtime_env = dict(env_values or self.effective_runtime_env())
        config_preview = self.current_config(source=runtime_env, gpu=gpu_payload)
        credentials_present = self.credentials_path.exists() or bool(
            first_nonempty(runtime_env.get("NODE_ID")) and first_nonempty(runtime_env.get("NODE_KEY"))
        )
        edge_control_url = first_nonempty(
            runtime_env.get("EDGE_CONTROL_URL"),
            str(config_preview.get("edge_control_url") or ""),
        )
        parsed_edge_control = urlparse(edge_control_url) if edge_control_url else None
        dns_hosts: list[str] = []
        if parsed_edge_control.hostname:
            dns_hosts.append(parsed_edge_control.hostname)
        for fallback_url in csv_items(runtime_env.get("EDGE_CONTROL_FALLBACK_URLS")):
            parsed_fallback = urlparse(fallback_url)
            if parsed_fallback.hostname:
                dns_hosts.append(parsed_fallback.hostname)
        for mirror_url in csv_items(runtime_env.get("ARTIFACT_MIRROR_BASE_URLS")):
            parsed_mirror = urlparse(mirror_url)
            if parsed_mirror.hostname:
                dns_hosts.append(parsed_mirror.hostname)
        if runtime_backend_supports_compose(runtime_backend):
            dns_hosts.extend(["registry-1.docker.io", "auth.docker.io"])
        if config_preview.get("hugging_face_repository"):
            dns_hosts.append("huggingface.co")
        deduped_dns_hosts = list(dict.fromkeys(host for host in dns_hosts if host))
        network_signature = {
            "edge_control_url": edge_control_url,
            "dns_hosts": deduped_dns_hosts,
            "runtime_backend": runtime_backend,
        }
        network_payload = self.cached_preflight_network_payload(
            signature=network_signature,
            force_refresh=force_refresh,
            resolver=lambda: {
                "dns": self.resolve_setup_dns(deduped_dns_hosts),
                "control_plane": self.probe_control_plane(edge_control_url),
            },
        )
        dns_payload = (
            network_payload.get("dns")
            if isinstance(network_payload.get("dns"), dict)
            else {"ok": False, "targets": []}
        )
        control_plane = (
            network_payload.get("control_plane")
            if isinstance(network_payload.get("control_plane"), dict)
            else {"ok": False, "status_code": None, "probe_url": edge_control_url, "error": "Control plane probe is unavailable."}
        )

        startup_model = str(config_preview.get("vllm_model") or runtime_env.get("VLLM_MODEL") or DEFAULT_VLLM_MODEL).strip()
        inference_engine = str(
            config_preview.get("inference_engine") or self.resolved_inference_engine(runtime_env)
        ).strip()
        startup_artifact = startup_model_artifact(startup_model, runtime_engine=inference_engine)
        expected_cache_bytes = artifact_total_size_bytes(startup_artifact)
        cache_dir = self.data_dir / "model-cache"
        cache_bytes = directory_size_bytes(cache_dir)
        missing_cache_bytes = (
            max(0, expected_cache_bytes - min(cache_bytes, expected_cache_bytes))
            if expected_cache_bytes is not None
            else None
        )
        free_bytes = int(round(float(disk.get("free_gb") or 0.0) * (1024**3)))
        cache_budget_bytes = max(0, int(config_preview.get("model_cache_budget_bytes") or 0))
        cache_reserve_bytes = max(0, int(config_preview.get("model_cache_reserve_free_bytes") or 0))
        cache_available_growth_bytes = max(0, int(config_preview.get("model_cache_available_growth_bytes") or 0))
        cache_budget_label = str(config_preview.get("model_cache_budget_label") or format_bytes(cache_budget_bytes))
        cache_reserve_label = format_bytes(cache_reserve_bytes)

        runtime_flag_errors: list[str] = []
        runtime_flag_warnings: list[str] = []
        if not edge_control_url:
            runtime_flag_errors.append("Set EDGE_CONTROL_URL to the local or hosted control-plane origin.")
        elif parsed_edge_control.scheme not in {"http", "https"} or not parsed_edge_control.netloc:
            runtime_flag_errors.append("Set EDGE_CONTROL_URL to a full http:// or https:// URL.")

        raw_concurrency = str(runtime_env.get("MAX_CONCURRENT_ASSIGNMENTS") or "").strip()
        try:
            parsed_concurrency = int(float(raw_concurrency))
            if parsed_concurrency < 1:
                raise ValueError("concurrency must be positive")
        except ValueError:
            runtime_flag_errors.append("Set MAX_CONCURRENT_ASSIGNMENTS to a whole number of 1 or more.")

        raw_batch_tokens = str(runtime_env.get("MAX_BATCH_TOKENS") or "").strip()
        try:
            parsed_batch_tokens = int(float(raw_batch_tokens))
            if parsed_batch_tokens < 1:
                raise ValueError("batch tokens must be positive")
        except ValueError:
            runtime_flag_errors.append("Set MAX_BATCH_TOKENS to a whole number of 1 or more.")

        raw_target_util = str(runtime_env.get("TARGET_GPU_UTILIZATION_PCT") or "").strip()
        if raw_target_util:
            try:
                parsed_target_util = int(float(raw_target_util))
                if not 30 <= parsed_target_util <= 100:
                    runtime_flag_warnings.append("TARGET_GPU_UTILIZATION_PCT will be clamped into the 30-100 range.")
            except ValueError:
                runtime_flag_warnings.append("TARGET_GPU_UTILIZATION_PCT is not numeric, so the default of 100% will be used.")

        raw_headroom = str(runtime_env.get("MIN_GPU_MEMORY_HEADROOM_PCT") or "").strip()
        if raw_headroom:
            try:
                parsed_headroom = float(raw_headroom)
                if not 5.0 <= parsed_headroom <= 60.0:
                    runtime_flag_warnings.append("MIN_GPU_MEMORY_HEADROOM_PCT will be clamped into the 5-60% range.")
            except ValueError:
                runtime_flag_warnings.append(
                    "MIN_GPU_MEMORY_HEADROOM_PCT is not numeric, so the default 20% floor will be used."
                )

        raw_gpu_temp_limit = str(runtime_env.get("GPU_TEMP_LIMIT_C") or "").strip()
        if raw_gpu_temp_limit:
            try:
                parsed_gpu_temp_limit = float(raw_gpu_temp_limit)
                if not 65.0 <= parsed_gpu_temp_limit <= 95.0:
                    runtime_flag_warnings.append("GPU_TEMP_LIMIT_C will be clamped into the 65-95 C range.")
            except ValueError:
                runtime_flag_warnings.append("GPU_TEMP_LIMIT_C is not numeric, so the default limit will be used.")

        supported_models = [
            model.strip()
            for model in str(runtime_env.get("SUPPORTED_MODELS") or config_preview.get("supported_models") or "").split(",")
            if model.strip()
        ]
        if startup_model and supported_models and startup_model not in supported_models:
            runtime_flag_errors.append(
                f"Add {startup_model} to SUPPORTED_MODELS so the startup runtime matches the advertised model list."
            )

        region_value = first_nonempty(
            runtime_env.get("NODE_REGION"),
            str(config_preview.get("node_region") or ""),
        )
        recommended_region = str(config_preview.get("recommended_node_region") or infer_node_region()[0])
        region_reason = str(
            config_preview.get("region_reason") or "Recommended from the local timezone and locale."
        )

        hf_repository = str(config_preview.get("hugging_face_repository") or "").strip()
        hf_token_required = bool(config_preview.get("hugging_face_token_required"))
        hf_token_configured = bool(config_preview.get("hugging_face_token_configured"))

        docker_check = build_setup_check(
            "docker",
            "Docker",
            "pass" if (docker_cli and compose_ok and daemon_ok) else "fail",
            (
                "Docker Desktop is installed and the engine is ready."
                if (docker_cli and compose_ok and daemon_ok)
                else "Docker Desktop still needs attention."
            ),
            (
                f"Compose is available for the {runtime_backend_label(runtime_backend)} runtime."
                if (docker_cli and compose_ok and daemon_ok)
                else str(docker_error or "Docker is not ready yet.")
            ),
            (
                "No fix needed."
                if (docker_cli and compose_ok and daemon_ok)
                else (
                    "Install Docker Desktop, open it, and wait until the Docker engine is running."
                    if not docker_cli
                    else "Open Docker Desktop and wait for the engine to finish starting."
                )
            ),
            blocking=not (docker_cli and compose_ok and daemon_ok),
        )

        driver_status = "pass" if (gpu_payload.get("detected") and nvidia_driver.get("present")) else "fail"
        driver_summary = (
            f"NVIDIA driver is responding for {gpu_payload.get('name') or 'the local GPU'}."
            if driver_status == "pass"
            else "The NVIDIA driver is not ready yet."
        )
        driver_detail = (
            f"Driver version {nvidia_driver.get('driver_version')} is visible through nvidia-smi."
            if nvidia_driver.get("driver_version")
            else (
                "nvidia-smi is present, but the local GPU is not visible yet."
                if nvidia_driver.get("present")
                else str(nvidia_driver.get("error") or "The NVIDIA driver is not reporting a usable GPU.")
            )
        )
        driver_fix = (
            "No fix needed."
            if driver_status == "pass"
            else "Install or update the NVIDIA driver until nvidia-smi lists the GPU from this machine."
        )
        driver_check = build_setup_check(
            "nvidia_driver",
            "NVIDIA driver",
            driver_status,
            driver_summary,
            driver_detail,
            driver_fix,
            blocking=driver_status == "fail",
            metadata={
                "driver_version": nvidia_driver.get("driver_version"),
                "gpu_name": gpu_payload.get("name"),
            },
        )

        if driver_status == "fail":
            cuda_check = build_setup_check(
                "cuda",
                "CUDA",
                "fail",
                "CUDA support cannot be verified yet.",
                "Quick Start checks CUDA through nvidia-smi after the NVIDIA driver is visible.",
                "Install or repair the NVIDIA driver first, then re-run the local setup check.",
                blocking=True,
            )
        elif nvidia_driver.get("cuda_version"):
            cuda_check = build_setup_check(
                "cuda",
                "CUDA",
                "pass",
                "CUDA support is visible to the host.",
                f"nvidia-smi reports CUDA {nvidia_driver.get('cuda_version')}.",
                "No fix needed.",
                metadata={"cuda_version": nvidia_driver.get("cuda_version")},
            )
        else:
            cuda_check = build_setup_check(
                "cuda",
                "CUDA",
                "warn",
                "CUDA support could not be read cleanly from nvidia-smi.",
                "The driver is responding, but nvidia-smi did not report a CUDA version in the local probe output.",
                "Update the NVIDIA driver if containers still fail to start with GPU access.",
            )

        if runtime_backend_supports_compose(runtime_backend):
            if gpu_payload.get("detected") and nvidia_container_runtime.get("visible") is False:
                gpu_runtime_check = build_setup_check(
                    "container_gpu",
                    "Container GPU access",
                    "fail",
                    "Docker cannot see the NVIDIA runtime yet.",
                    str(
                        nvidia_container_runtime.get("error")
                        or "Docker is running, but the NVIDIA container runtime is not visible."
                    ),
                    "Enable NVIDIA GPU support in Docker Desktop, then restart Docker Desktop.",
                    blocking=True,
                )
            elif gpu_payload.get("detected") and nvidia_container_runtime.get("visible") is True:
                gpu_runtime_check = build_setup_check(
                    "container_gpu",
                    "Container GPU access",
                    "pass",
                    "Docker can expose the NVIDIA GPU to runtime containers.",
                    "The NVIDIA container runtime is visible in docker info.",
                    "No fix needed.",
                )
            else:
                gpu_runtime_check = build_setup_check(
                    "container_gpu",
                    "Container GPU access",
                    "fail",
                    "The local NVIDIA GPU is not visible yet.",
                    "Docker GPU support can only be checked after the host driver exposes the GPU.",
                    "Install or repair the NVIDIA driver until the GPU appears in nvidia-smi.",
                    blocking=True,
                )
        else:
            gpu_runtime_check = build_setup_check(
                "container_gpu",
                "Container GPU access",
                "pass",
                "The unified NVIDIA runtime does not need Docker Desktop GPU passthrough.",
                "This machine is using the in-container runtime path.",
                "No fix needed.",
            )

        disk_status = "pass" if disk["ok"] else "fail"
        disk_check = build_setup_check(
            "disk",
            "Disk",
            disk_status,
            (
                f"{disk.get('free_gb', 0)} GB is free for the local runtime and model cache."
                if disk_status == "pass"
                else "The local disk is below the recommended free-space floor."
            ),
            (
                f"{disk.get('free_gb', 0)} GB free of {disk.get('total_gb', 0)} GB total. "
                f"Quick Start targets at least {int(disk.get('recommended_free_gb', RECOMMENDED_FREE_DISK_GB))} GB free."
            ),
            (
                "No fix needed."
                if disk_status == "pass"
                else f"Free at least {int(disk.get('recommended_free_gb', RECOMMENDED_FREE_DISK_GB))} GB before continuing."
            ),
            blocking=disk_status == "fail",
        )

        if deduped_dns_hosts and dns_payload.get("ok"):
            dns_check = build_setup_check(
                "dns",
                "DNS",
                "pass",
                "The setup hosts resolve from this network.",
                "Resolved: " + ", ".join(
                    f"{entry.get('host')}" for entry in dns_payload.get("targets", []) if entry.get("ok")
                ),
                "No fix needed.",
                metadata={"targets": dns_payload.get("targets", [])},
            )
        elif deduped_dns_hosts:
            failed_hosts = [
                str(entry.get("host"))
                for entry in dns_payload.get("targets", [])
                if not entry.get("ok")
            ]
            dns_check = build_setup_check(
                "dns",
                "DNS",
                "fail",
                "At least one setup hostname is not resolving from this network.",
                "Could not resolve: " + ", ".join(failed_hosts or deduped_dns_hosts),
                "Check router DNS, VPN, proxy, or firewall settings until the setup hostnames resolve cleanly.",
                blocking=True,
                metadata={"targets": dns_payload.get("targets", [])},
            )
        else:
            dns_check = build_setup_check(
                "dns",
                "DNS",
                "warn",
                "No setup hostnames were available for a DNS probe.",
                "EDGE_CONTROL_URL is blank, so DNS checks could not determine the control-plane host.",
                "Set EDGE_CONTROL_URL to the control-plane origin and re-run the setup check.",
            )

        control_plane_status = "pass" if control_plane.get("ok") else "fail"
        control_plane_check = build_setup_check(
            "control_plane",
            "Claim service",
            control_plane_status,
            (
                "The control plane is reachable from this machine."
                if control_plane_status == "pass"
                else "The control plane could not be reached from this machine."
            ),
            (
                f"{control_plane.get('probe_url')} returned HTTP {control_plane.get('status_code')}."
                if control_plane_status == "pass" and control_plane.get("status_code") is not None
                else str(control_plane.get("error") or "The control-plane probe failed.")
            ),
            (
                "No fix needed."
                if control_plane_status == "pass"
                else "Check EDGE_CONTROL_URL, internet access, proxy rules, and firewall settings, then re-run the setup check."
            ),
            blocking=control_plane_status == "fail",
            metadata={
                "probe_url": control_plane.get("probe_url"),
                "status_code": control_plane.get("status_code"),
                "service": control_plane.get("service"),
            },
        )

        artifact_status = "pass" if (dns_payload.get("ok") and control_plane.get("ok")) else "fail"
        artifact_check = build_setup_check(
            "artifact_store",
            "R2 / artifact path",
            artifact_status,
            (
                "The signed artifact path should be reachable from this machine."
                if artifact_status == "pass"
                else "Signed artifact downloads will likely fail from this network."
            ),
            (
                "The control plane responded and the setup hostnames resolved, so the same HTTPS path used for signed R2 artifact URLs is likely healthy."
                if artifact_status == "pass"
                else "The control plane or DNS probe failed, so signed R2 artifact URLs are unlikely to work reliably yet."
            ),
            (
                "No fix needed."
                if artifact_status == "pass"
                else "Fix DNS or control-plane reachability first, then retry so signed artifact URLs can be fetched cleanly."
            ),
            blocking=artifact_status == "fail",
        )

        if expected_cache_bytes is None:
            model_cache_check = build_setup_check(
                "model_cache",
                "Model cache",
                "warn",
                "Quick Start could not estimate the startup-model cache size yet.",
                f"The bundled manifest does not have a startup-cache estimate for {startup_model}.",
                "No immediate fix is needed. Quick Start will still warm the model if the runtime image can start.",
                metadata={"cache_bytes": cache_bytes},
            )
        elif missing_cache_bytes == 0:
            model_cache_check = build_setup_check(
                "model_cache",
                "Model cache",
                "pass",
                "The startup model is already cached locally.",
                (
                    f"Reusing about {format_bytes(cache_bytes)} from {cache_dir} for {startup_model}. "
                    f"This node's cache budget is {cache_budget_label} with {cache_reserve_label} kept free."
                    if cache_bytes
                    else (
                        f"The local cache already has the files Quick Start needs for {startup_model}. "
                        f"This node's cache budget is {cache_budget_label}."
                    )
                ),
                "No fix needed.",
                metadata={
                    "cache_bytes": cache_bytes,
                    "expected_cache_bytes": expected_cache_bytes,
                    "missing_cache_bytes": missing_cache_bytes,
                    "cache_budget_bytes": cache_budget_bytes,
                    "cache_reserve_free_bytes": cache_reserve_bytes,
                },
            )
        elif free_bytes >= missing_cache_bytes and cache_available_growth_bytes >= missing_cache_bytes:
            model_cache_check = build_setup_check(
                "model_cache",
                "Model cache",
                "warn",
                "The startup model cache is still cold or partial.",
                (
                    f"{startup_model} still needs about {format_bytes(missing_cache_bytes)} in the local model cache. "
                    f"Quick Start will fill that cache automatically on the first warm-up inside a {cache_budget_label} cache budget."
                ),
                f"No immediate fix is needed unless you want more than {cache_budget_label} of local model cache or a smaller free-space reserve than {cache_reserve_label}.",
                metadata={
                    "cache_bytes": cache_bytes,
                    "expected_cache_bytes": expected_cache_bytes,
                    "missing_cache_bytes": missing_cache_bytes,
                    "cache_budget_bytes": cache_budget_bytes,
                    "cache_reserve_free_bytes": cache_reserve_bytes,
                },
            )
        else:
            shortfall_label = format_bytes(
                max(
                    0,
                    missing_cache_bytes - min(free_bytes, cache_available_growth_bytes),
                )
                if missing_cache_bytes is not None
                else 0
            )
            model_cache_check = build_setup_check(
                "model_cache",
                "Model cache",
                "fail",
                "There is not enough free disk space to finish caching the startup model.",
                (
                    f"{startup_model} still needs about {format_bytes(missing_cache_bytes)} of cache space, "
                    f"but this node only has about {format_bytes(free_bytes)} free and {cache_budget_label} of cache budget "
                    f"with {cache_reserve_label} held back. Short by about {shortfall_label}."
                ),
                "Free more disk space, increase MODEL_CACHE_BUDGET_GB, reduce MODEL_CACHE_RESERVE_FREE_GB, or choose a smaller startup model before continuing.",
                blocking=True,
                metadata={
                    "cache_bytes": cache_bytes,
                    "expected_cache_bytes": expected_cache_bytes,
                    "missing_cache_bytes": missing_cache_bytes,
                    "cache_budget_bytes": cache_budget_bytes,
                    "cache_reserve_free_bytes": cache_reserve_bytes,
                    "cache_available_growth_bytes": cache_available_growth_bytes,
                },
            )

        offline_bundle = self.inspect_offline_install_bundle(
            env_values=runtime_env,
            startup_model=startup_model,
            inference_engine=inference_engine,
            runtime_backend=runtime_backend,
        )
        offline_bundle_check = build_setup_check(
            "offline_bundle",
            "Offline bundle",
            str(offline_bundle.get("status") or "warn"),
            str(offline_bundle.get("summary") or "Quick Start will use the network install path."),
            str(offline_bundle.get("detail") or ""),
            str(offline_bundle.get("fix") or "No fix needed."),
            blocking=False,
            metadata=offline_bundle,
        )

        if not hf_repository:
            hf_check = build_setup_check(
                "hugging_face",
                "Hugging Face access",
                "pass",
                "This startup model does not need a Hugging Face token check.",
                "Quick Start can proceed without a gated-model access probe for the current startup model.",
                "No fix needed.",
            )
        elif hf_token_required and not hf_token_configured:
            hf_check = build_setup_check(
                "hugging_face",
                "Hugging Face access",
                "fail",
                "The startup model needs a saved Hugging Face token.",
                f"{hf_repository} is gated and Quick Start cannot validate access without a token.",
                f"Paste a Hugging Face token approved for {hf_repository}, then run the check again.",
                blocking=True,
            )
        elif hf_token_required:
            hf_check = build_setup_check(
                "hugging_face",
                "Hugging Face access",
                "pass",
                "A Hugging Face token is configured for the startup model.",
                f"The local setup flow will validate access to {hf_repository} before the first model download starts.",
                "No fix needed.",
            )
        else:
            hf_check = build_setup_check(
                "hugging_face",
                "Hugging Face access",
                "pass",
                "The startup model is on a public Hugging Face path.",
                f"Quick Start can validate public access to {hf_repository} without a token.",
                "No fix needed.",
            )

        if not region_value:
            region_check = build_setup_check(
                "region",
                "Region",
                "fail",
                "The node region is blank.",
                "Quick Start needs a region so the scheduler can keep this node near the owner.",
                f"Set NODE_REGION to {recommended_region} for this machine.",
                blocking=True,
            )
        elif region_value not in KNOWN_NODE_REGIONS:
            region_check = build_setup_check(
                "region",
                "Region",
                "fail",
                "The node region is not one of the supported scheduler regions.",
                f"{region_value} is not in the supported set: {', '.join(sorted(KNOWN_NODE_REGIONS))}.",
                f"Use {recommended_region} unless you intentionally need a different supported region.",
                blocking=True,
            )
        elif region_value != recommended_region:
            region_check = build_setup_check(
                "region",
                "Region",
                "warn",
                "The selected region differs from the local recommendation.",
                f"{region_value} is configured, but {recommended_region} is recommended here. {region_reason}",
                f"Switch NODE_REGION to {recommended_region} if you want lower-latency local scheduling.",
            )
        else:
            region_check = build_setup_check(
                "region",
                "Region",
                "pass",
                "The node region matches the local recommendation.",
                f"{recommended_region} is recommended for this machine. {region_reason}",
                "No fix needed.",
            )

        if runtime_flag_errors:
            runtime_flag_check = build_setup_check(
                "runtime_flags",
                "Runtime flags",
                "fail",
                "One or more runtime flags still need attention.",
                " ".join(runtime_flag_errors),
                runtime_flag_errors[0],
                blocking=True,
                metadata={"errors": runtime_flag_errors, "warnings": runtime_flag_warnings},
            )
        elif runtime_flag_warnings:
            runtime_flag_check = build_setup_check(
                "runtime_flags",
                "Runtime flags",
                "warn",
                "The runtime flags are usable, but some values will be normalized.",
                " ".join(runtime_flag_warnings),
                runtime_flag_warnings[0],
                metadata={"errors": runtime_flag_errors, "warnings": runtime_flag_warnings},
            )
        else:
            runtime_flag_check = build_setup_check(
                "runtime_flags",
                "Runtime flags",
                "pass",
                "The saved runtime flags are internally consistent.",
                "Quick Start can use the current routing, model, and concurrency settings without rewriting them.",
                "No fix needed.",
            )

        if credentials_present:
            claim_check = build_setup_check(
                "claim_token",
                "Claim token",
                "pass",
                "This machine already has stored node approval.",
                "Quick Start can skip claim-token creation and bring the node online directly.",
                "No fix needed.",
            )
        elif control_plane.get("ok") and not runtime_flag_errors:
            claim_check = build_setup_check(
                "claim_token",
                "Claim token",
                "pass",
                "Quick Start can request a fresh approval token when you continue.",
                "The claim service is reachable and the local runtime flags are ready for approval.",
                "No fix needed.",
            )
        else:
            claim_check = build_setup_check(
                "claim_token",
                "Claim token",
                "fail",
                "Quick Start cannot request a claim token yet.",
                (
                    "The control plane must be reachable and the runtime flags must be valid before Quick Start can open the approval page."
                ),
                (
                    runtime_flag_errors[0]
                    if runtime_flag_errors
                    else "Fix the control-plane reachability checks first, then retry the setup check."
                ),
                blocking=True,
            )

        checks = [
            docker_check,
            driver_check,
            cuda_check,
            gpu_runtime_check,
            disk_check,
            dns_check,
            control_plane_check,
            artifact_check,
            model_cache_check,
            offline_bundle_check,
            hf_check,
            claim_check,
            region_check,
            runtime_flag_check,
        ]
        audit = summarize_setup_checks(checks)
        blockers = [
            f"{check['label']}: {check['fix']}"
            for check in audit["blocking_checks"]
        ]

        return {
            "docker_cli": docker_cli,
            "docker_compose": compose_ok,
            "docker_daemon": daemon_ok,
            "docker_error": docker_error,
            "gpu": gpu_payload,
            "nvidia_driver": nvidia_driver,
            "nvidia_container_runtime": nvidia_container_runtime,
            "disk": disk,
            "running_services": running_services,
            "credentials_present": credentials_present,
            "runtime_backend": runtime_backend,
            "runtime_backend_label": runtime_backend_label(runtime_backend),
            "dns": dns_payload,
            "control_plane": control_plane,
            "edge_control_url": edge_control_url,
            "offline_install_bundle": offline_bundle,
            "setup_checks": checks,
            "setup_audit": audit,
            "claim_gate_blockers": blockers,
            "ready_for_claim": bool(audit["ready_for_claim"]),
            "blockers": blockers,
        }

    def json_safe_config(self, config: dict[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(json.dumps(config, default=str))
        except (TypeError, ValueError):
            return {str(key): str(value) for key, value in config.items()}

    def state_payload_unlocked(self, *, include_private: bool = False) -> dict[str, Any]:
        payload = asdict(self.state)
        claim = payload.get("claim")
        if isinstance(claim, dict) and not include_private:
            claim.pop("poll_token", None)
        resume_config = payload.get("resume_config")
        if isinstance(resume_config, dict) and not include_private:
            payload["resume_config"] = {
                key: ("***" if any(marker in str(key).lower() for marker in ("token", "secret", "password", "key")) else value)
                for key, value in resume_config.items()
            }
        return payload

    def state_payload(self, *, include_private: bool = False) -> dict[str, Any]:
        with self.lock:
            return self.state_payload_unlocked(include_private=include_private)

    def persist_state_unlocked(self) -> None:
        payload = {
            "version": INSTALLER_STATE_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "state": self.state_payload_unlocked(include_private=True),
        }
        try:
            self.service_dir.mkdir(parents=True, exist_ok=True)
            tighten_private_path(self.service_dir, directory=True)
            tmp_path = self.installer_state_path.with_name(f"{self.installer_state_path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.installer_state_path)
            tighten_private_path(self.installer_state_path)
        except (OSError, TypeError, ValueError):
            return

    def persist_state(self) -> None:
        with self.lock:
            self.persist_state_unlocked()

    def claim_state_from_payload(self, payload: Any) -> InstallerClaimState | None:
        if not isinstance(payload, dict):
            return None
        try:
            claim = InstallerClaimState(
                claim_id=str(payload.get("claim_id") or ""),
                claim_code=str(payload.get("claim_code") or ""),
                approval_url=str(payload.get("approval_url") or ""),
                expires_at=str(payload.get("expires_at") or ""),
                poll_interval_seconds=int(payload.get("poll_interval_seconds") or 0),
                poll_token=str(payload.get("poll_token") or ""),
                status=str(payload.get("status") or "pending"),
                renewal_count=max(0, int(payload.get("renewal_count") or 0)),
                auto_refreshes=bool(payload.get("auto_refreshes", True)),
                approval_qr_svg_data_url=(
                    str(payload.get("approval_qr_svg_data_url"))
                    if payload.get("approval_qr_svg_data_url")
                    else None
                ),
            )
        except (TypeError, ValueError):
            return None
        if not (claim.claim_id and claim.claim_code and claim.approval_url and claim.expires_at):
            return None
        if claim.approval_qr_svg_data_url is None:
            claim.approval_qr_svg_data_url = approval_qr_svg_data_url(claim.approval_url)
        remaining_seconds = seconds_until_timestamp(claim.expires_at)
        if remaining_seconds == 0 and claim.status not in {"approved", "consumed"}:
            claim.status = "expired"
        return claim

    def installer_state_from_payload(self, payload: Any) -> InstallerState | None:
        if not isinstance(payload, dict):
            return None

        state = InstallerState()
        state.stage = str(payload.get("stage") or state.stage)
        state.busy = bool(payload.get("busy"))
        state.message = str(payload.get("message") or state.message)
        state.error = str(payload.get("error")) if payload.get("error") is not None else None
        state.error_step = str(payload.get("error_step")) if payload.get("error_step") is not None else None
        logs = payload.get("logs")
        if isinstance(logs, list):
            state.logs = [str(item) for item in logs][-80:]
        stage_context = payload.get("stage_context")
        if isinstance(stage_context, dict):
            state.stage_context = dict(stage_context)
        state.claim = self.claim_state_from_payload(payload.get("claim"))
        for field_name in ("started_at", "stage_started_at"):
            raw_value = payload.get(field_name)
            if raw_value is None:
                continue
            try:
                setattr(state, field_name, float(raw_value))
            except (TypeError, ValueError):
                continue
        resume_config = payload.get("resume_config")
        if isinstance(resume_config, dict):
            state.resume_config = self.json_safe_config(resume_config)
        state.resume_requested = bool(payload.get("resume_requested"))
        if state.busy and state.resume_config:
            state.resume_requested = True
        last_warm_source = payload.get("last_warm_source")
        if isinstance(last_warm_source, dict):
            state.last_warm_source = self.json_safe_config(last_warm_source)
        return state

    def load_state(self) -> None:
        if not self.installer_state_path.exists():
            return
        try:
            payload = json.loads(self.installer_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        state_payload = payload.get("state") if isinstance(payload, dict) and "state" in payload else payload
        state = self.installer_state_from_payload(state_payload)
        if state is None:
            return

        changed = False
        if state.busy:
            state.busy = False
            state.resume_requested = bool(state.resume_config)
            label = INSTALLER_FLOW_LABELS.get(state.stage, "Quick Start")
            state.message = f"Quick Start paused during {label}. Reopening the app will resume from this step automatically."
            note = f"Quick Start was interrupted during {label}; the next app start will resume it automatically."
            if not state.logs or state.logs[-1] != note:
                state.logs.append(note)
                state.logs = state.logs[-80:]
            changed = True

        if state.claim:
            before = state.claim.status
            remaining_seconds = seconds_until_timestamp(state.claim.expires_at)
            if remaining_seconds == 0 and state.claim.status not in {"approved", "consumed"}:
                state.claim.status = "expired"
            if state.claim.approval_qr_svg_data_url is None:
                state.claim.approval_qr_svg_data_url = approval_qr_svg_data_url(state.claim.approval_url)
            changed = changed or before != state.claim.status

        with self.lock:
            self.state = state
            if changed:
                self.persist_state_unlocked()

    def reusable_claim_session(self) -> tuple[NodeClaimSession, int] | None:
        with self.lock:
            claim = self.state.claim
            if claim is None or not claim.poll_token:
                return None
            remaining_seconds = seconds_until_timestamp(claim.expires_at)
            if remaining_seconds == 0:
                return None
            return (
                NodeClaimSession(
                    claim_id=claim.claim_id,
                    claim_code=claim.claim_code,
                    approval_url=claim.approval_url,
                    poll_token=claim.poll_token,
                    expires_at=claim.expires_at,
                    poll_interval_seconds=claim.poll_interval_seconds,
                ),
                max(0, claim.renewal_count),
            )

    def should_defer_stage_for_resume(self, stage: str, resume_from_stage: str | None) -> bool:
        if not resume_from_stage:
            return False
        if stage not in INSTALLER_FLOW_INDEX or resume_from_stage not in INSTALLER_FLOW_INDEX:
            return False
        return INSTALLER_FLOW_INDEX[stage] < INSTALLER_FLOW_INDEX[resume_from_stage]

    def set_install_stage(self, stage: str, message: str, *, resume_from_stage: str | None = None) -> None:
        if not self.should_defer_stage_for_resume(stage, resume_from_stage):
            self.set_stage(stage, message)
            return
        with self.lock:
            self.state.busy = True
            self.state.error = None
            self.state.error_step = None
            self.state.stage_context = {
                **self.state.stage_context,
                "resume_recheck_stage": stage,
                "resume_recheck_label": INSTALLER_FLOW_LABELS.get(stage, stage),
            }
            self.persist_state_unlocked()

    def build_claim_state(
        self,
        claim: NodeClaimSession,
        *,
        status: str = "pending",
        renewal_count: int = 0,
    ) -> InstallerClaimState:
        return InstallerClaimState(
            claim_id=claim.claim_id,
            claim_code=claim.claim_code,
            approval_url=claim.approval_url,
            expires_at=claim.expires_at,
            poll_interval_seconds=claim.poll_interval_seconds,
            poll_token=claim.poll_token,
            status=status,
            renewal_count=max(0, renewal_count),
            approval_qr_svg_data_url=approval_qr_svg_data_url(claim.approval_url),
        )

    def update_claim_status(self, claim_id: str, *, status: str, expires_at: str) -> None:
        with self.lock:
            if self.state.claim and self.state.claim.claim_id == claim_id:
                self.state.claim.status = status
                self.state.claim.expires_at = expires_at
                self.persist_state_unlocked()

    def status_payload(self) -> dict[str, Any]:
        source = self.effective_runtime_env()
        gpu = detect_gpu(self.command_runner, self.runtime_dir)
        config = self.current_config(source=source, gpu=gpu)
        preflight = self.collect_preflight(gpu=gpu, env_values=source)
        state = self.state_payload()
        autostart = self.autostart_manager.status()
        desktop_launcher = self.desktop_launcher_manager.status()
        appliance_release = self.appliance_release_payload()
        return {
            "config": config,
            "preflight": preflight,
            "state": state,
            "owner_setup": self.owner_setup_payload(
                config=config,
                preflight=preflight,
                state=state,
                autostart=autostart,
                appliance_release=appliance_release,
            ),
            "autostart": autostart,
            "desktop_launcher": desktop_launcher,
            "appliance_release": appliance_release,
        }

    def preview_setup_payload(self, config: dict[str, Any]) -> dict[str, Any]:
        preview_config = self.json_safe_config(config)
        gpu = detect_gpu(self.command_runner, self.runtime_dir)
        env_values = self.build_env(preview_config)
        preflight = self.collect_preflight(gpu=gpu, env_values=env_values, force_refresh=True)
        automatic_fixes = self.attempt_safe_preflight_repairs(
            config=preview_config,
            env_values=env_values,
            preflight=preflight,
        )
        if automatic_fixes.get("attempted"):
            preflight = self.collect_preflight(gpu=gpu, env_values=env_values, force_refresh=True)
        preflight["automatic_fixes"] = automatic_fixes
        resolved_config = self.current_config(source=env_values, gpu=gpu)
        state = self.state_payload()
        autostart = self.autostart_manager.status()
        desktop_launcher = self.desktop_launcher_manager.status()
        appliance_release = self.appliance_release_payload()
        return {
            "config": resolved_config,
            "preflight": preflight,
            "state": state,
            "owner_setup": self.owner_setup_payload(
                config=resolved_config,
                preflight=preflight,
                state=state,
                autostart=autostart,
                appliance_release=appliance_release,
            ),
            "autostart": autostart,
            "desktop_launcher": desktop_launcher,
            "appliance_release": appliance_release,
        }

    def appliance_release_payload(self) -> dict[str, Any]:
        package_signature = inspect_package_signature()
        runtime_signature_root = self.runtime_dir if (self.runtime_dir / "appliance-runtime-manifest.json").exists() else None
        runtime_bundle_signature = inspect_runtime_bundle_signature(runtime_signature_root)
        try:
            manifest = load_release_manifest()
            version = manifest.version
            channel = manifest.channel
        except ReleaseManifestError:
            version = package_signature.get("version")
            channel = package_signature.get("channel")
        verified = bool(package_signature.get("verified")) and bool(runtime_bundle_signature.get("verified"))
        detail = (
            f"Signed appliance release {version or 'pending'} on the {release_channel_label(channel).lower()} track "
            "is ready for this machine."
            if verified
            else str(package_signature.get("detail") or runtime_bundle_signature.get("detail") or "Signed appliance verification needs attention.")
        )
        return {
            "verified": verified,
            "detail": detail,
            "version": version,
            "channel": channel,
            "channel_label": release_channel_label(channel),
            "package_signature": package_signature,
            "runtime_bundle_signature": runtime_bundle_signature,
        }

    def owner_setup_payload(
        self,
        *,
        config: dict[str, Any],
        preflight: dict[str, Any],
        state: dict[str, Any],
        autostart: dict[str, Any],
        appliance_release: dict[str, Any],
    ) -> dict[str, Any]:
        claim = state.get("claim") if isinstance(state.get("claim"), dict) else None
        stage_context = state.get("stage_context") if isinstance(state.get("stage_context"), dict) else {}
        last_warm_source = state.get("last_warm_source") if isinstance(state.get("last_warm_source"), dict) else {}
        running_services = preflight.get("running_services", [])
        docker_ready = bool(preflight.get("docker_cli") and preflight.get("docker_compose") and preflight.get("docker_daemon"))
        runtime_backend = str(preflight.get("runtime_backend") or self.current_runtime_backend())
        runtime_backend_name = str(preflight.get("runtime_backend_label") or runtime_backend_label(runtime_backend))
        setup_audit = preflight.get("setup_audit") if isinstance(preflight.get("setup_audit"), dict) else {}
        automatic_fixes = (
            preflight.get("automatic_fixes")
            if isinstance(preflight.get("automatic_fixes"), dict)
            else {"attempted": False, "summary": "", "actions": []}
        )
        offline_bundle = (
            preflight.get("offline_install_bundle")
            if isinstance(preflight.get("offline_install_bundle"), dict)
            else {}
        )
        blocking_checks = setup_audit.get("blocking_checks") if isinstance(setup_audit.get("blocking_checks"), list) else []
        primary_blocker = blocking_checks[0] if blocking_checks else None
        primary_blocker_key = str(primary_blocker.get("key") or "") if isinstance(primary_blocker, dict) else ""
        setup_blocker_headlines = {
            "docker": "Start Docker Desktop",
            "nvidia_driver": "Install NVIDIA driver",
            "cuda": "Repair CUDA support",
            "container_gpu": "Enable Docker GPU access",
            "disk": "Free up disk space",
            "dns": "Fix DNS",
            "control_plane": "Reach claim service",
            "artifact_store": "Repair artifact path",
            "model_cache": "Make room for model cache",
            "hugging_face": "Add Hugging Face token",
            "claim_token": "Ready claim service",
            "region": "Set node region",
            "runtime_flags": "Fix runtime flags",
        }
        disk = preflight.get("disk", {})
        disk_ready = bool(isinstance(disk, dict) and disk.get("ok"))
        gpu = preflight.get("gpu", {}) if isinstance(preflight.get("gpu"), dict) else {}
        gpu_detected = bool(gpu.get("detected"))
        credentials_present = bool(preflight.get("credentials_present"))
        runtime_running = "node-agent" in running_services
        busy = bool(state.get("busy"))
        stage = str(state.get("stage") or "idle")
        error = str(state.get("error") or "").strip() or None
        error_step = str(state.get("error_step") or "").strip() or None
        startup_mode = startup_mode_recommendation(autostart)
        setup_profile = str(config.get("setup_profile") or config.get("recommended_setup_profile") or "balanced")
        setup_profile_label = PROFILE_LABELS.get(setup_profile, PROFILE_LABELS["balanced"])
        startup_model = str(config.get("vllm_model") or config.get("recommended_model") or DEFAULT_VLLM_MODEL)
        owner_target_model = str(config.get("owner_target_model") or config.get("recommended_model") or startup_model)
        bootstrap_pending_upgrade = bool(
            config.get("bootstrap_pending_upgrade")
            and owner_target_model
            and owner_target_model != startup_model
        )
        runtime_delivery_label = (
            "Built-in appliance install"
            if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND
            else "Signed online install"
        )
        runtime_delivery_detail = (
            "This appliance release starts the built-in runtime directly, so first startup does not depend on repo files, backend picks, or local Compose setup."
            if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND
            else "Quick Start will prepare the supported appliance runtime from the signed release manifest. A local appliance starter bundle can still preseed runtime assets and the tiny starter-model cache before any network download begins."
        )
        inference_engine = str(config.get("inference_engine") or self.resolved_inference_engine())
        hf_repository = str(config.get("hugging_face_repository") or "").strip() or None
        hf_token_required = bool(config.get("hugging_face_token_required"))
        if hf_repository is None:
            hf_repository, hf_token_required = hugging_face_validation_target(
                startup_model,
                inference_engine=inference_engine,
            )
        hf_validation_needed = hf_repository is not None
        hf_token_configured = bool(config.get("hugging_face_token_configured"))
        concurrency = str(
            config.get("max_concurrent_assignments")
            or config.get("recommended_max_concurrent_assignments")
            or suggest_concurrency(None)
        )
        region = str(config.get("node_region") or config.get("recommended_node_region") or "eu-se-1")
        gpu_name = str(config.get("gpu_name") or gpu.get("name") or "GPU pending")
        gpu_vram = str(config.get("gpu_memory_gb") or "")
        gpu_support_label = str(config.get("gpu_support_label") or "").strip()
        gpu_support_track = str(config.get("gpu_support_track") or "").strip()
        gpu_support_summary = str(config.get("gpu_support_summary") or "").strip()
        support_preset_ready = bool(gpu_detected and gpu_vram and gpu_support_label and gpu_support_track)
        startup_model_bytes = artifact_total_size_bytes(
            startup_model_artifact(startup_model, runtime_engine=inference_engine)
        )
        owner_target_model_bytes = artifact_total_size_bytes(
            startup_model_artifact(owner_target_model, runtime_engine=inference_engine)
        )
        heat_mode = str(config.get("heat_governor_mode") or "100").strip() or "100"
        estimated_heat_output_watts = estimate_install_heat_output_watts(
            gpu_memory_gb=config.get("gpu_memory_gb"),
            estimated_heat_output_watts=config.get("estimated_heat_output_watts"),
            power_watts=config.get("power_watts"),
        )
        earnings_low, earnings_high = estimate_install_gross_earnings_range_per_day(
            gpu_memory_gb=config.get("gpu_memory_gb"),
            runtime_backend=runtime_backend,
            setup_profile=setup_profile,
        )
        earnings_estimate_label = format_usd_range(earnings_low, earnings_high)
        offline_bundle_value = (
            "Bundled starter ready"
            if offline_bundle.get("ready") and offline_bundle.get("source") == "bundled"
            else (
                "Local bundle ready"
                if offline_bundle.get("ready")
                else (
                    "Bundle path set"
                    if offline_bundle.get("configured")
                    else "Connected download"
                )
            )
        )
        warm_path = (
            {
                "winner": stage_context.get("warm_source_winner"),
                "winner_label": stage_context.get("warm_source_label"),
                "detail": stage_context.get("warm_source_detail"),
                "scope": stage_context.get("warm_source_scope"),
                "order": stage_context.get("warm_source_order"),
            }
            if stage_context.get("warm_source_winner")
            else last_warm_source
        )
        if not isinstance(warm_path, dict) or not warm_path:
            warm_path = self.warm_source_payload(
                env_values=self.build_env(config),
                startup_model=startup_model,
                inference_engine=inference_engine,
                observed_cache_bytes=directory_size_bytes(self.data_dir / "model-cache"),
                offline_bundle=offline_bundle,
            )
        warm_path_value = str(warm_path.get("winner_label") or "Warm path pending")
        if str(warm_path.get("scope") or "") == "planned_remote":
            warm_path_value = f"{warm_path_value} (planned remote path)"
        if support_preset_ready:
            if bootstrap_pending_upgrade:
                machine_plan_summary = (
                    f"Quick Start will use the {gpu_support_label} preset for {gpu_support_track}, bring this node online "
                    f"quickly on {startup_model}, then warm {owner_target_model} in the background, with up to {concurrency} active "
                    f"{'workload' if concurrency == '1' else 'workloads'} on {setup_profile_label.lower()} tuning."
                )
            else:
                machine_plan_summary = (
                    f"Quick Start will use the {gpu_support_label} preset for {gpu_support_track}, start with {startup_model}, "
                    f"and allow up to {concurrency} active {'workload' if concurrency == '1' else 'workloads'} on "
                    f"{setup_profile_label.lower()} tuning."
                )
        else:
            if bootstrap_pending_upgrade:
                machine_plan_summary = (
                    f"Quick Start will bring this node online quickly on {startup_model}, then warm {owner_target_model} in the background, "
                    f"with up to {concurrency} active {'workload' if concurrency == '1' else 'workloads'} on {setup_profile_label.lower()} tuning."
                )
            else:
                machine_plan_summary = (
                    f"Quick Start will use {startup_model} with up to {concurrency} active "
                    f"{'workload' if concurrency == '1' else 'workloads'} on {setup_profile_label.lower()} tuning."
                )
        recommendations = [
            {
                "key": "gpu",
                "label": "GPU",
                "value": gpu_name if gpu_detected else "GPU not detected yet",
                "detail": (
                    f"Detected locally from {gpu_name}."
                    if gpu_detected
                    else "Quick Start will keep using safe defaults until a compatible GPU is visible."
                ),
            },
            {
                "key": "vram",
                "label": "VRAM",
                "value": f"{gpu_vram} GB" if gpu_vram else "Unknown",
                "detail": (
                    (
                        f"Available VRAM maps this machine into the {gpu_support_label} preset for {gpu_support_track}."
                        if support_preset_ready
                        else "Available VRAM is used to choose the startup model and concurrency."
                    )
                    if gpu_detected and gpu_vram
                    else "VRAM is unavailable until GPU detection succeeds."
                ),
            },
            {
                "key": "nvidia_preset",
                "label": "NVIDIA preset",
                "value": (
                    f"{gpu_support_label}: {gpu_support_track}"
                    if support_preset_ready
                    else "Waiting for NVIDIA detection"
                ),
                "detail": (
                    gpu_support_summary
                    if support_preset_ready
                    else "Quick Start maps NVIDIA VRAM into a deterministic runtime preset once detection succeeds."
                ),
            },
            {
                "key": "signed_release",
                "label": "Signed appliance release",
                "value": (
                    f"{appliance_release.get('version') or 'Pending'} · {appliance_release.get('channel_label') or 'Stable'}"
                    if appliance_release.get("verified")
                    else "Needs attention"
                ),
                "detail": str(
                    appliance_release.get("detail")
                    or "The local app verifies the signed installer path and bundled runtime before setup continues."
                ),
            },
            {
                "key": "startup_model",
                "label": "Bootstrap model" if bootstrap_pending_upgrade else "Startup model",
                "value": startup_model,
                "detail": (
                    f"Quick Start uses the tiny public bootstrap model first so this node can claim, register, and show online quickly. "
                    f"After that, it warms {owner_target_model} in the background."
                    if bootstrap_pending_upgrade
                    else override_detail(
                        startup_model,
                        str(config.get("recommended_model") or startup_model),
                        (
                            "Quick Start is staying on the best public startup model for this machine right now. "
                            "Saved gated-model access can unlock the larger preset later."
                            if bool(config.get("startup_model_fallback"))
                            else (
                                f"Chosen automatically from the {gpu_support_label} preset and bundled runtime models."
                                if support_preset_ready
                                else "Chosen automatically from the detected VRAM and bundled runtime models."
                            )
                        ),
                        override_label="Advanced override is active for the startup model.",
                    )
                ),
            },
            {
                "key": "background_target",
                "label": "Background target",
                "value": owner_target_model if bootstrap_pending_upgrade else "Not needed",
                "detail": (
                    (
                        f"This larger model warms after the node is online so responses can move up from the bootstrap path. "
                        f"The local cache will reuse about {format_bytes(owner_target_model_bytes)} once it is ready."
                    )
                    if bootstrap_pending_upgrade and owner_target_model_bytes
                    else (
                        "This larger model warms after the node is online so responses can move up from the bootstrap path."
                        if bootstrap_pending_upgrade
                        else "This machine already starts on its normal owner model."
                    )
                ),
            },
            {
                "key": "hf_access",
                "label": "HF access",
                "value": (
                    f"Configured for {hf_repository}"
                    if hf_validation_needed and hf_token_required and hf_token_configured
                    else (
                        f"Token needed for {hf_repository}"
                        if hf_validation_needed and hf_token_required
                        else (
                            f"Public access for {hf_repository}"
                            if hf_validation_needed
                            else "Not required"
                        )
                    )
                ),
                "detail": (
                    (
                        "Quick Start is staying on a public startup model because no saved gated-model access is configured locally yet."
                        if bool(config.get("startup_model_fallback"))
                        else (
                            f"A Hugging Face token is already saved locally for {hf_repository}."
                            if hf_token_required and hf_token_configured
                            else (
                            f"{hf_repository} is gated on Hugging Face. Add a token in advanced settings if you intentionally want to unlock that startup model."
                            if hf_token_required
                            else "Quick Start validates public Hugging Face access before pulling images and downloading model files."
                        )
                        )
                    )
                    if hf_validation_needed
                    else "This startup model does not need a Hugging Face validation check."
                ),
            },
            {
                "key": "concurrency",
                "label": "Concurrency",
                "value": f"{concurrency} active {'workload' if concurrency == '1' else 'workloads'}",
                "detail": override_detail(
                    concurrency,
                    str(config.get("recommended_max_concurrent_assignments") or concurrency),
                    (
                        f"Chosen automatically from the {gpu_support_label} preset and thermal profile."
                        if support_preset_ready
                        else "Chosen automatically from the detected GPU and thermal profile."
                    ),
                    override_label="Advanced override is active for concurrency.",
                ),
            },
            {
                "key": "thermal_profile",
                "label": "Thermal profile",
                "value": setup_profile_label,
                "detail": config.get("profile_summary") or "Quick Start chooses a conservative thermal profile automatically.",
            },
            {
                "key": "runtime_delivery",
                "label": "Install path",
                "value": runtime_delivery_label,
                "detail": runtime_delivery_detail,
            },
            {
                "key": "offline_bundle",
                "label": "Offline install",
                "value": offline_bundle_value,
                "detail": str(
                    offline_bundle.get("detail")
                    or "Quick Start can reuse a local bootstrap bundle when one is present."
                ),
            },
            {
                "key": "warm_source",
                "label": "Warm path",
                "value": warm_path_value,
                "detail": str(
                    warm_path.get("detail")
                    or "Warm-up chooses local cache first, then the offline appliance bundle, then relay/cache mirror, then Hugging Face."
                ),
            },
            {
                "key": "region",
                "label": "Region",
                "value": region,
                "detail": override_detail(
                    region,
                    str(config.get("recommended_node_region") or region),
                    str(config.get("region_reason") or "Recommended from this machine's locale and timezone."),
                    override_label="Advanced override is active for the node region.",
                ),
            },
            {
                "key": "startup_mode",
                "label": "Startup mode",
                "value": str(startup_mode["startup_mode_label"]),
                "detail": str(startup_mode["startup_mode_detail"]),
            },
            {
                "key": "routing_lane",
                "label": "Routing lane",
                "value": str(config.get("routing_lane_label") or "Automatic"),
                "detail": str(config.get("routing_lane_detail") or ""),
            },
            {
                "key": "privacy_ceiling",
                "label": "Privacy ceiling",
                "value": str(config.get("max_privacy_tier") or "standard").replace("_", " ").title(),
                "detail": (
                    "This runtime can only accept standard community workloads and will not advertise confidential or restricted eligibility."
                    if str(config.get("max_privacy_tier") or "standard") == "standard"
                    else "This runtime can advertise privacy support up to restricted workloads when the selected offer and trust lane allow it."
                ),
            },
            {
                "key": "exactness_ceiling",
                "label": "Exact model guarantee",
                "value": "Available" if bool(config.get("exact_model_guarantee")) else "Not available",
                "detail": (
                    "This runtime stays on exact audited model artifacts and can satisfy exact-model guarantees when the scheduler requests them."
                    if bool(config.get("exact_model_guarantee"))
                    else "This runtime is best-effort only and should not be marketed as an exact-model guarantee lane."
                ),
            },
            {
                "key": "quantized_disclosure",
                "label": "Quantized disclosure",
                "value": "Required" if bool(config.get("quantized_output_disclosure_required")) else "Not required",
                "detail": (
                    "This runtime serves quantized GGUF output, so pricing and UI must disclose that responses come from a quantized model path."
                    if bool(config.get("quantized_output_disclosure_required"))
                    else "This runtime stays on audited exact artifacts, so no extra quantized-output disclosure is required."
                ),
            },
            {
                "key": "premium_eligibility",
                "label": "Capacity status",
                "value": str(config.get("premium_eligibility_label") or "Community capacity enabled"),
                "detail": str(config.get("premium_eligibility_detail") or ""),
            },
        ]

        current_step = "checking_docker"
        if runtime_running:
            current_step = "node_live"
        elif error_step and error_step in INSTALLER_FLOW_INDEX:
            current_step = error_step
        elif busy and stage in INSTALLER_FLOW_INDEX:
            current_step = stage
        elif claim and not credentials_present:
            current_step = "claiming_node"
        elif credentials_present:
            current_step = "node_live"
        elif primary_blocker_key in {"nvidia_driver", "cuda", "container_gpu"}:
            current_step = "checking_nvidia_runtime"
        elif primary_blocker_key == "hugging_face":
            current_step = "validating_hf_access"
        elif primary_blocker_key == "model_cache":
            current_step = "downloading_model"
        elif primary_blocker_key:
            current_step = "checking_docker"
        elif not docker_ready or not disk_ready:
            current_step = "checking_docker"
        elif not gpu_detected:
            current_step = "checking_nvidia_runtime"
        elif hf_validation_needed:
            current_step = "validating_hf_access"
        else:
            current_step = "pulling_image"

        def stage_progress_units(step_key: str) -> float:
            if step_key == "pulling_image":
                total = _safe_int(stage_context.get("download_total_items"), 0)
                completed = _safe_int(stage_context.get("download_completed_items"), 0)
                if total <= 0:
                    return 0.55
                return max(0.2, min(0.95, completed / total))
            if step_key == "downloading_model":
                if bool(stage_context.get("warm_reusing_cache")):
                    return 1.0
                percent = stage_context.get("warm_progress_percent")
                if percent is None:
                    return 0.55
                return max(0.2, min(0.95, int(percent) / 100))
            if step_key == "warming_model":
                if bool(stage_context.get("warm_reusing_cache")):
                    return 0.75
                percent = stage_context.get("warm_progress_percent")
                if percent is not None and int(percent) >= 100:
                    return 0.8
                return 0.55
            return 0.55

        def step_status(step_key: str) -> str:
            if runtime_running:
                return "complete"
            if error and step_key == current_step:
                return "error"
            if step_key == "checking_docker":
                if docker_ready and disk_ready:
                    return "complete"
                if current_step == step_key:
                    return "active"
                return "pending"
            if step_key == "checking_nvidia_runtime":
                if current_step == step_key:
                    return "active"
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key]:
                    return "complete" if gpu_detected else "warning"
                if not docker_ready or not disk_ready:
                    return "pending"
                return "complete" if gpu_detected else "warning"
            if step_key == "validating_hf_access":
                if not hf_validation_needed:
                    return "complete"
                if current_step == step_key:
                    return "active"
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key]:
                    return "complete"
                if not docker_ready or not disk_ready or not gpu_detected:
                    return "pending"
                return "pending"
            if step_key == "claiming_node" and credentials_present:
                return "complete"
            if step_key == "node_live" and credentials_present and not runtime_running and not busy:
                return "active"
            if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key]:
                return "complete"
            if step_key == current_step:
                return "active"
            return "pending"

        def step_detail(step_key: str) -> str:
            if step_key == "checking_docker":
                if docker_ready and disk_ready:
                    if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
                        return "The unified NVIDIA runtime is running in this container and disk space looks healthy."
                    return "Docker Desktop is running, Compose is ready, and disk space looks healthy."
                if not docker_ready:
                    if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
                        return "This container is checking its local runtime prerequisites before setup continues."
                    return str(preflight.get("docker_error") or "Docker needs attention before setup can continue.")
                return (
                    f"This machine has {disk.get('free_gb', 0)} GB free. "
                    f"At least {int(disk.get('recommended_free_gb', RECOMMENDED_FREE_DISK_GB))} GB is recommended."
                )
            if step_key == "checking_nvidia_runtime":
                if gpu_detected:
                    if gpu_support_label and gpu_support_track:
                        return (
                            f"{gpu.get('name') or 'GPU'} detected with {gpu.get('memory_gb')} GB VRAM. "
                            f"Using the {gpu_support_label} preset for {gpu_support_track}."
                        )
                    return f"{gpu.get('name') or 'GPU'} detected with {gpu.get('memory_gb')} GB VRAM."
                return "No NVIDIA GPU runtime was detected yet. Quick Start currently requires a supported NVIDIA GPU before setup can continue."
            if step_key == "validating_hf_access":
                if not hf_validation_needed:
                    return "This startup model does not need a Hugging Face validation check."
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key] or credentials_present:
                    return (
                        f"Validated gated Hugging Face access for {hf_repository} before the first download started."
                        if hf_token_required
                        else f"Validated public Hugging Face access for {hf_repository} before the first download started."
                    )
                if current_step == step_key:
                    if hf_token_required and not hf_token_configured and not busy:
                        return (
                            f"{hf_repository} is gated on Hugging Face. Add a token below so Quick Start can validate access before the first download starts."
                        )
                    if hf_token_required:
                        return f"Checking the configured Hugging Face token and model approval for {hf_repository}."
                    return f"Checking public Hugging Face access for {hf_repository} before the first download starts."
                return "Quick Start validates Hugging Face access before pulling images and downloading model files."
            if step_key == "pulling_image":
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key] or credentials_present:
                    if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
                        return "The unified NVIDIA runtime image is already present in this container."
                    return "The runtime images are cached locally and ready for startup."
                if current_step == step_key:
                    if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
                        return (
                            "The unified NVIDIA runtime image is already running here. "
                            "Quick Start is preparing the in-container services automatically."
                        )
                    total = _safe_int(stage_context.get("download_total_items"), 0)
                    completed = _safe_int(stage_context.get("download_completed_items"), 0)
                    current_service = str(stage_context.get("download_current_item") or "").strip()
                    if total > 0 and current_service:
                        return (
                            f"Pulling the {current_service} runtime image "
                            f"({min(total, completed + 1)}/{total}) so later restarts are faster."
                        )
                    return (
                        "Pulling the runtime containers and saving them locally. "
                        "This can take several minutes the first time."
                    )
                return "Quick Start pre-pulls the runtime image before the local model download starts."
            if step_key == "downloading_model":
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key] or credentials_present:
                    if bool(stage_context.get("warm_resuming_download")):
                        resumed_bytes = _safe_int(stage_context.get("warm_resume_from_bytes"), 0)
                        expected_bytes = stage_context.get("warm_expected_bytes")
                        if isinstance(expected_bytes, int) and expected_bytes > 0:
                            return (
                                f"Resumed {startup_model} from about {format_bytes(resumed_bytes)} already cached locally "
                                f"and finished caching about {format_bytes(expected_bytes)}."
                            )
                    if bool(stage_context.get("warm_reusing_cache")):
                        reused_bytes = _safe_int(stage_context.get("warm_observed_cache_bytes"), 0)
                        return f"Skipped a fresh model download by reusing {format_bytes(reused_bytes)} already cached locally."
                    expected_bytes = stage_context.get("warm_expected_bytes")
                    if isinstance(expected_bytes, int) and expected_bytes > 0:
                        return f"Downloaded about {format_bytes(expected_bytes)} for {startup_model} into the local model cache."
                    return f"The local model files for {startup_model} are ready."
                if current_step == step_key:
                    expected_bytes = stage_context.get("warm_expected_bytes")
                    downloaded_bytes = _safe_int(stage_context.get("warm_downloaded_bytes"), 0)
                    if bool(stage_context.get("warm_resuming_download")):
                        resumed_bytes = _safe_int(stage_context.get("warm_resume_from_bytes"), 0)
                        if bool(stage_context.get("warm_download_stalled")):
                            return (
                                f"Resuming {startup_model} from about {format_bytes(resumed_bytes)} already cached locally, "
                                "but the download now looks stalled."
                            )
                        rate_label = format_rate_bytes_per_second(coerce_float_or_none(stage_context.get("warm_download_rate_bps")))
                        if isinstance(expected_bytes, int) and expected_bytes > 0:
                            if bool(stage_context.get("warm_download_slow")) and rate_label:
                                return (
                                    f"Resuming {startup_model}: {format_bytes(downloaded_bytes)} of about "
                                    f"{format_bytes(expected_bytes)} is cached locally at about {rate_label}."
                                )
                            return (
                                f"Resuming {startup_model}: {format_bytes(downloaded_bytes)} of about "
                                f"{format_bytes(expected_bytes)} is cached locally."
                            )
                        return f"Resuming {startup_model} from about {format_bytes(resumed_bytes)} already cached locally."
                    if bool(stage_context.get("warm_download_stalled")):
                        seconds_without_progress = _safe_int(stage_context.get("warm_seconds_without_progress"), 0)
                        return (
                            f"Downloading {startup_model} looks stalled after about {seconds_without_progress} seconds "
                            f"without cache growth. {warm_path.get('detail') or ''}".strip()
                        )
                    rate_label = format_rate_bytes_per_second(coerce_float_or_none(stage_context.get("warm_download_rate_bps")))
                    if isinstance(expected_bytes, int) and expected_bytes > 0:
                        if bool(stage_context.get("warm_download_slow")) and rate_label:
                            return (
                                f"Downloading {startup_model}: {format_bytes(downloaded_bytes)} of about "
                                f"{format_bytes(expected_bytes)} is cached locally at about {rate_label}. "
                                f"{warm_path.get('detail') or ''}".strip()
                            )
                        return (
                            f"Downloading {startup_model}: {format_bytes(downloaded_bytes)} of about "
                            f"{format_bytes(expected_bytes)} is ready in the local model cache. "
                            f"{warm_path.get('detail') or ''}".strip()
                        )
                    return f"Downloading {startup_model} into the local model cache. {warm_path.get('detail') or ''}".strip()
                return "Quick Start downloads the startup model into the local cache before warming it."
            if step_key == "warming_model":
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key] or credentials_present:
                    return "The startup model responded locally and the runtime is warm."
                if current_step == step_key:
                    excerpt = str(stage_context.get("warm_runtime_log_excerpt") or "").strip()
                    if excerpt and stage_context.get("warm_failure_kind"):
                        return f"Warm-up hit a clear runtime error: {excerpt}"
                    if bool(stage_context.get("warm_reusing_cache")):
                        reused_bytes = _safe_int(stage_context.get("warm_observed_cache_bytes"), 0)
                        return (
                            f"Reusing {format_bytes(reused_bytes)} already in the local model cache while {startup_model} warms up. "
                            f"{warm_path.get('detail') or ''}".strip()
                        )
                    return f"Finishing the local warm-up for {startup_model} now that the model files are cached. {warm_path.get('detail') or ''}".strip()
                return f"The runtime warms the startup model after the local download is complete. {warm_path.get('detail') or ''}".strip()
            if step_key == "claiming_node":
                if credentials_present:
                    return "Local node credentials are stored for this machine."
                if claim:
                    refresh_count = _safe_int(claim.get("renewal_count"), 0)
                    if refresh_count > 0:
                        suffix = "" if refresh_count == 1 else "s"
                        return (
                            f"Claim code {claim.get('claim_code', '')} is waiting for approval after "
                            f"{refresh_count} automatic refresh{suffix}."
                        )
                    return (
                        f"Claim code {claim.get('claim_code', '')} is waiting for approval. "
                        "The app will refresh the approval link automatically if needed."
                    )
                if current_step == step_key:
                    return "Creating a secure in-app approval link and QR code for this machine."
                return (
                    "Quick Start opens the browser approval page automatically here, shows a QR code for other devices, "
                    "and keeps waiting for approval."
                )
            if runtime_running:
                if bootstrap_pending_upgrade:
                    return (
                        f"The runtime is online on {startup_model}. "
                        f"{owner_target_model} keeps warming in the background until this machine is idle enough to switch over."
                    )
                return "The runtime is online and ready to accept work."
            if current_step == step_key:
                return "Starting the local services and bringing this machine online."
            if credentials_present:
                return "This machine is approved and ready to go live."
            return "The runtime goes live automatically after approval."

        steps = [
            {
                "key": step_key,
                "label": step_label,
                "status": step_status(step_key),
                "detail": step_detail(step_key),
            }
            for step_key, step_label in INSTALLER_FLOW
        ]

        if runtime_running:
            headline = "Node live"
            detail = (
                f"The runtime is online on {startup_model} while {owner_target_model} warms in the background."
                if bootstrap_pending_upgrade
                else "The runtime is online and ready to accept work."
            )
            eta_seconds: int | None = 0
        elif error:
            headline = "Setup needs attention"
            detail = error
            eta_seconds = None
        elif claim and not credentials_present:
            headline = "Approve this node"
            detail = (
                "Approve this machine in your browser or by scanning the QR code. "
                "Quick Start keeps polling and refreshes the approval link automatically if sign-in takes a while."
            )
            eta_seconds = INSTALLER_FLOW_REMAINING_SECONDS["claiming_node"]
        elif busy:
            headline = INSTALLER_FLOW_LABELS.get(current_step, "Setting up this machine")
            detail = str(state.get("message") or "Keep this window open while Quick Start finishes the local setup flow.")
            eta_seconds = INSTALLER_FLOW_REMAINING_SECONDS.get(current_step)
        elif credentials_present:
            headline = "Bring this node online"
            detail = "This machine is already approved. Quick Start can bring it online in one click."
            eta_seconds = INSTALLER_FLOW_REMAINING_SECONDS["node_live"]
        elif primary_blocker is not None:
            headline = setup_blocker_headlines.get(
                primary_blocker_key,
                str(primary_blocker.get("label") or "Setup blocker"),
            )
            detail = str(
                primary_blocker.get("fix")
                or primary_blocker.get("detail")
                or primary_blocker.get("summary")
                or "Fix this blocker, then retry Quick Start."
            )
            eta_seconds = None
        elif not docker_ready:
            headline = "Start Docker Desktop"
            detail = str(preflight.get("docker_error") or "Docker needs attention before setup can continue.")
            eta_seconds = None
        elif not disk_ready:
            headline = "Free up disk space"
            detail = step_detail("checking_docker")
            eta_seconds = None
        elif hf_validation_needed and hf_token_required and not hf_token_configured:
            headline = "Add Hugging Face token"
            detail = (
                f"{hf_repository} is gated on Hugging Face. Add a token below and Quick Start will validate it before the first download starts."
            )
            eta_seconds = None
        else:
            headline = "Ready to bring this node online"
            if bootstrap_pending_upgrade and startup_model_bytes:
                target_copy = (
                    f" then warm about {format_bytes(owner_target_model_bytes)} for {owner_target_model} in the background"
                    if owner_target_model_bytes
                    else f" then warm {owner_target_model} in the background"
                )
                detail = (
                    "Quick Start will check the NVIDIA runtime, validate Hugging Face access, pull the runtime image, "
                    f"download about {format_bytes(startup_model_bytes)} for the bootstrap model {startup_model}, warm it locally, "
                    "surface the approval link and QR code in this app for this node, and"
                    f"{target_copy} after the node is online."
                )
            elif startup_model_bytes:
                detail = (
                    "Quick Start will check the NVIDIA runtime, validate Hugging Face access, pull the runtime image, "
                    f"download about {format_bytes(startup_model_bytes)} for {startup_model}, warm it locally, and then "
                    "surface the approval link and QR code in this app for this node."
                )
            else:
                detail = (
                    "Quick Start will check the NVIDIA runtime, validate Hugging Face access, pull the runtime image, "
                    "warm the model, and then surface the approval link and QR code in this app automatically."
                )
            eta_seconds = INSTALLER_FLOW_REMAINING_SECONDS[current_step]

        progress_units = 0.0
        for step in steps:
            status = step["status"]
            if status == "complete" or status == "warning":
                progress_units += 1.0
            elif status == "active" or status == "error":
                progress_units += stage_progress_units(str(step["key"]))
                break
            else:
                break

        eta_copy = (
            "Fix the highlighted issue, then retry."
            if error
            else (
                "About 1 minute after you approve this machine. The approval link refreshes automatically if needed."
                if claim and not credentials_present
                else (
                    "First startup can take several minutes while runtime images are cached locally."
                    if current_step == "pulling_image" and busy
                    else (
                        "Quick Start is validating Hugging Face access before the first download starts."
                        if current_step == "validating_hf_access" and busy
                        else (
                            (
                                "First startup is only warming the tiny bootstrap model. The larger owner target keeps warming in the background after the node is online."
                                if bootstrap_pending_upgrade
                                else "First startup can take several minutes while the local model cache fills."
                            )
                            if current_step in {"downloading_model", "warming_model"} and busy
                            else (
                                "Continue once this machine is ready."
                                if eta_seconds is None and current_step != "claiming_node"
                                else eta_label(eta_seconds)
                            )
                        )
                    )
                )
            )
        )

        heat_mode_label = {
            "0": "0% heat",
            "20": "20% heat",
            "50": "50% heat",
            "100": "100% heat",
            "auto": "Auto heat",
        }.get(heat_mode, f"{heat_mode}% heat")
        inference_test_complete = (
            runtime_running
            or credentials_present
            or current_step in {"claiming_node", "node_live"}
        )
        claim_tone = (
            "success"
            if credentials_present
            else ("danger" if primary_blocker is not None or error else "warning")
        )
        claim_value = (
            "Approved"
            if credentials_present
            else ("Approval open" if claim else ("Waiting on local checks" if primary_blocker is not None else "Ready to claim"))
        )
        claim_detail = (
            "This machine already has local approval saved, so Quick Start can go live directly."
            if credentials_present
            else (
                "The approval link is already open and Quick Start will keep refreshing it automatically until sign-in finishes."
                if claim
                else (
                    str(primary_blocker.get("fix") or detail)
                    if primary_blocker is not None
                    else "Quick Start will open the approval page automatically after the local warm-up check finishes."
                )
            )
        )
        first_run_wizard = {
            "headline": "First-run appliance plan",
            "detail": (
                "Quick Start moves through local hardware checks, heat tuning, one starter-model warm-up, and claim approval without asking you to manage Docker, runtime backends, or repo files."
                if not automatic_fixes.get("attempted")
                else f"{automatic_fixes.get('summary') or 'Quick Start already handled the safe local repairs it could.'}"
            ),
            "steps": [
                {
                    "key": "detect_gpu",
                    "label": "Detect GPU",
                    "tone": "success" if gpu_detected else "danger",
                    "value": (
                        f"{gpu_name} · {gpu_vram} GB"
                        if gpu_detected and gpu_vram
                        else (gpu_name if gpu_detected else "GPU not detected")
                    ),
                    "detail": (
                        f"This machine maps into the {gpu_support_label} preset for {gpu_support_track}."
                        if support_preset_ready
                        else "Quick Start is waiting for a supported NVIDIA GPU before it can continue."
                    ),
                },
                {
                    "key": "heat_profile",
                    "label": "Choose heat profile",
                    "tone": "success",
                    "value": f"{setup_profile_label} · {heat_mode_label}",
                    "detail": (
                        "Balanced first-run tuning keeps the node boring to install, then the heat governor can hold 0%, 20%, 50%, 100%, or auto heat after claim."
                        if heat_mode == "100"
                        else "Quick Start will keep the owner heat target and thermal profile together so this node behaves like a smart radiator."
                    ),
                },
                {
                    "key": "estimate_output",
                    "label": "Estimate heat and earnings",
                    "tone": "success" if estimated_heat_output_watts is not None else "warning",
                    "value": f"{format_watts(estimated_heat_output_watts)} · {earnings_estimate_label}",
                    "detail": (
                        "This is a rough local first-run envelope, not a payout promise. Heat output tracks GPU draw closely once the node is busy."
                    ),
                },
                {
                    "key": "install_path",
                    "label": "Choose install path",
                    "tone": "success" if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND or offline_bundle.get("ready") else "warning",
                    "value": runtime_delivery_label,
                    "detail": (
                        f"{runtime_delivery_detail} Offline path: {offline_bundle_value.lower()}."
                    ),
                },
                {
                    "key": "bootstrap_model",
                    "label": "Pick bootstrap model",
                    "tone": "success",
                    "value": startup_model,
                    "detail": (
                        f"Quick Start will warm about {format_bytes(startup_model_bytes)} locally, then switch toward {owner_target_model} in the background."
                        if bootstrap_pending_upgrade and startup_model_bytes
                        else (
                            f"Quick Start will warm about {format_bytes(startup_model_bytes)} locally before it opens claim."
                            if startup_model_bytes
                            else "Quick Start will warm the starter model locally before it opens claim."
                        )
                    ),
                },
                {
                    "key": "test_inference",
                    "label": "Test one local inference",
                    "tone": (
                        "success"
                        if inference_test_complete
                        else ("danger" if error_step in {"downloading_model", "warming_model"} else "warning")
                    ),
                    "value": "Validated locally" if inference_test_complete else "Pending warm-up",
                    "detail": (
                        "The starter model responded locally, so this node is ready to move into claim and then live serving."
                        if inference_test_complete
                        else step_detail("warming_model")
                    ),
                },
                {
                    "key": "claim_node",
                    "label": "Claim node",
                    "tone": claim_tone,
                    "value": claim_value,
                    "detail": claim_detail,
                },
            ],
        }

        return {
            "headline": headline,
            "detail": detail,
            "setup_profile": config.get("setup_profile"),
            "recommended_setup_profile": config.get("recommended_setup_profile"),
            "profile_summary": config.get("profile_summary"),
            "machine_plan_summary": machine_plan_summary,
            "runtime_backend": runtime_backend,
            "runtime_backend_label": runtime_backend_name,
            "recommendations": recommendations,
            "steps": steps,
            "current_step": current_step,
            "current_step_label": INSTALLER_FLOW_LABELS.get(current_step, "Quick Start"),
            "eta_seconds": eta_seconds,
            "eta_label": eta_copy,
            "progress_percent": int(round((progress_units / len(INSTALLER_FLOW)) * 100)),
            "setup_checks": preflight.get("setup_checks", []),
            "setup_audit": setup_audit,
            "automatic_fixes": automatic_fixes,
            "warm_path": warm_path,
            "first_run_wizard": first_run_wizard,
            "primary_action_label": (
                "Node live"
                if runtime_running
                else (
                    "Retry Quick Start"
                    if error
                    else (
                        "Quick Start is running..."
                        if busy
                        else (
                            "Bring this node online"
                            if credentials_present and not runtime_running
                            else (
                                "Add token and start Quick Start"
                                if (
                                    hf_validation_needed
                                    and hf_token_required
                                    and not hf_token_configured
                                    and current_step == "validating_hf_access"
                                )
                                else "Bring this node online"
                            )
                        )
                    )
                )
            ),
        }

    def start_install(self, config: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
        already_busy = False
        resume_from_stage: str | None = None
        resume_config = self.json_safe_config(config)
        with self.lock:
            if self.state.busy or (self.install_thread is not None and self.install_thread.is_alive()):
                already_busy = True
            elif resume:
                resume_from_stage = self.state.stage if self.state.stage in INSTALLER_FLOW_INDEX else None
                if self.state.started_at is None:
                    self.state.started_at = time.time()
                if self.state.stage not in INSTALLER_FLOW_INDEX:
                    self.state.stage = "checking_docker"
                    self.state.stage_started_at = time.time()
                label = INSTALLER_FLOW_LABELS.get(self.state.stage, "Quick Start")
                self.state.busy = True
                self.state.error = None
                self.state.error_step = None
                self.state.resume_config = resume_config
                self.state.resume_requested = True
                self.state.message = f"Resuming Quick Start from {label}."
                note = f"Quick Start resumed automatically from {label}."
                if not self.state.logs or self.state.logs[-1] != note:
                    self.state.logs.append(note)
                    self.state.logs = self.state.logs[-80:]
                self.persist_state_unlocked()
            else:
                self.state = InstallerState(
                    stage="checking_docker",
                    busy=True,
                    message="Checking Docker and local prerequisites.",
                    logs=[],
                    started_at=time.time(),
                    stage_started_at=time.time(),
                    resume_config=resume_config,
                    resume_requested=True,
                )
                self.persist_state_unlocked()

        if already_busy:
            return self.status_payload()

        self.install_thread = threading.Thread(
            target=self.run_install,
            args=(resume_config,),
            kwargs={"resume_from_stage": resume_from_stage},
            daemon=True,
        )
        self.install_thread.start()
        return self.status_payload()

    def resume_if_needed(self) -> dict[str, Any]:
        with self.lock:
            already_busy = self.state.busy or (self.install_thread is not None and self.install_thread.is_alive())
            resume_requested = self.state.resume_requested
            resume_config = dict(self.state.resume_config)
            stage = self.state.stage
            has_error = bool(self.state.error)
        if already_busy:
            return self.status_payload()
        if not resume_requested or not resume_config or has_error or stage in INSTALLER_TERMINAL_STAGES:
            return self.status_payload()
        return self.start_install(resume_config, resume=True)

    def log(self, message: str) -> None:
        with self.lock:
            self.state.logs.append(message)
            self.state.message = message
            self.state.logs = self.state.logs[-80:]
            self.persist_state_unlocked()

    def set_stage(self, stage: str, message: str) -> None:
        with self.lock:
            now = time.time()
            if self.state.started_at is None:
                self.state.started_at = now
            if self.state.stage != stage:
                self.state.stage_started_at = now
            self.state.stage = stage
            self.state.busy = True
            self.state.error = None
            self.state.error_step = None
            self.state.message = message
            self.state.stage_context = {}
            self.persist_state_unlocked()

    def update_stage_progress(self, message: str | None = None, **context: Any) -> None:
        with self.lock:
            if message is not None:
                self.state.message = message
            merged = dict(self.state.stage_context)
            merged.update(context)
            self.state.stage_context = merged
            self.persist_state_unlocked()

    def set_error(self, message: str, *, step: str | None = None) -> None:
        with self.lock:
            current_stage = step if step in INSTALLER_FLOW_INDEX else None
            if current_stage is None:
                current_stage = self.state.stage if self.state.stage in INSTALLER_FLOW_INDEX else "checking_docker"
            self.state.stage = "error"
            self.state.busy = False
            self.state.error = message
            self.state.error_step = current_stage
            self.state.message = message
            self.state.stage_context = {}
            self.state.resume_requested = False
            self.persist_state_unlocked()

    def set_claim(self, claim: InstallerClaimState) -> None:
        with self.lock:
            self.state.claim = claim
            if self.state.started_at is None:
                self.state.started_at = time.time()
            self.state.stage_started_at = time.time()
            self.state.stage = "claiming_node"
            self.state.busy = True
            self.state.error = None
            self.state.error_step = None
            self.state.message = (
                "Waiting for sign-in and approval so this machine can be registered. "
                "The approval link in this app refreshes automatically if needed."
            )
            self.state.stage_context = {}
            self.state.resume_requested = True
            self.persist_state_unlocked()

    def complete(self, message: str) -> None:
        with self.lock:
            self.state.stage = "running"
            self.state.busy = False
            self.state.error = None
            self.state.error_step = None
            self.state.message = message
            self.state.claim = None
            self.state.stage_started_at = time.time()
            self.state.stage_context = {}
            self.state.resume_config = {}
            self.state.resume_requested = False
            self.persist_state_unlocked()

    def ensure_data_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.service_dir,
            self.data_dir / "model-cache",
            self.data_dir / "scratch",
            self.credentials_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def runtime_credentials_env_path(self, runtime_backend: str) -> str:
        if runtime_backend_supports_compose(runtime_backend):
            return COMPOSE_RUNTIME_CREDENTIALS_PATH
        return str(self.credentials_path)

    def runtime_autopilot_state_env_path(self, runtime_backend: str) -> str:
        if runtime_backend_supports_compose(runtime_backend):
            return COMPOSE_RUNTIME_AUTOPILOT_STATE_PATH
        return str(self.data_dir / "scratch" / "autopilot-state.json")

    def runtime_heat_governor_state_env_path(self, runtime_backend: str) -> str:
        if runtime_backend_supports_compose(runtime_backend):
            return COMPOSE_RUNTIME_HEAT_GOVERNOR_STATE_PATH
        return str(self.scratch_dir / "heat-governor-state.json")

    def runtime_fault_injection_state_env_path(self, runtime_backend: str) -> str:
        if runtime_backend_supports_compose(runtime_backend):
            return COMPOSE_RUNTIME_FAULT_INJECTION_STATE_PATH
        return str(self.scratch_dir / DEFAULT_FAULT_INJECTION_STATE_NAME)

    def runtime_startup_status_env_path(self, runtime_backend: str) -> str:
        if runtime_backend_supports_compose(runtime_backend):
            return COMPOSE_RUNTIME_STARTUP_STATUS_PATH
        return str(self.scratch_dir / DEFAULT_STARTUP_STATUS_FILENAME)

    def startup_warm_source_env_values(
        self,
        env_values: dict[str, str],
        startup_model: str,
        *,
        inference_engine: str,
        runtime_backend: str,
    ) -> dict[str, str]:
        payload = self.state.last_warm_source or {}
        if not payload:
            offline_bundle = self.inspect_offline_install_bundle(
                env_values=env_values,
                startup_model=startup_model,
                inference_engine=inference_engine,
                runtime_backend=runtime_backend,
            )
            payload = self.warm_source_payload(
                env_values=env_values,
                startup_model=startup_model,
                inference_engine=inference_engine,
                offline_bundle=offline_bundle,
            )
        order = payload.get("order")
        order_text = ",".join(str(item).strip() for item in order if str(item).strip()) if isinstance(order, list) else ""
        return {
            "STARTUP_WARM_SOURCE": str(payload.get("winner") or "").strip(),
            "STARTUP_WARM_SOURCE_LABEL": str(payload.get("winner_label") or "").strip(),
            "STARTUP_WARM_SOURCE_DETAIL": str(payload.get("detail") or "").strip(),
            "STARTUP_WARM_SOURCE_SCOPE": str(payload.get("scope") or "").strip(),
            "STARTUP_WARM_SOURCE_ORDER": order_text,
            "STARTUP_WARM_SOURCE_SELECTED_AT": str(payload.get("selected_at") or "").strip(),
        }

    def build_env(self, config: dict[str, Any]) -> dict[str, str]:
        current = self.load_effective_env()
        persisted = self.load_persisted_env()
        defaults = NodeAgentSettings()
        runtime_backend = self.current_runtime_backend()
        persisted_backend = normalize_runtime_backend(persisted.get(RUNTIME_BACKEND_ENV))
        selected_runtime_backend = (
            persisted_backend if persisted_backend != AUTO_RUNTIME_BACKEND else runtime_backend
        )
        credentials_path = self.runtime_credentials_env_path(selected_runtime_backend)
        autopilot_state_path = self.runtime_autopilot_state_env_path(selected_runtime_backend)
        heat_governor_state_path = self.runtime_heat_governor_state_env_path(selected_runtime_backend)
        fault_injection_state_path = self.runtime_fault_injection_state_env_path(selected_runtime_backend)
        startup_status_path = self.runtime_startup_status_env_path(selected_runtime_backend)
        quickstart_mode = str(config.get("setup_mode", "")).strip().lower() == "quickstart"
        operator_mode = bool(config.get("operator_mode"))
        detected_gpu = detect_gpu(self.command_runner, self.runtime_dir)
        inferred_region, _region_reason = infer_node_region()
        inferred_attestation_provider, _attestation_reason = detect_attestation_provider(
            self.command_runner,
            self.runtime_dir,
        )
        config_gpu_name = str(config.get("gpu_name", "")).strip() or None
        config_gpu_memory = str(config.get("gpu_memory_gb", "")).strip() or None
        requested_setup_profile = str(
            config.get("setup_profile", "") or (persisted.get("SETUP_PROFILE", "") if operator_mode else "")
        )
        configured_gpu_name = first_nonempty(current.get("GPU_NAME"), defaults.gpu_name)
        configured_gpu_memory = _safe_float(
            first_nonempty(persisted.get("GPU_MEMORY_GB"), current.get("GPU_MEMORY_GB")),
            defaults.gpu_memory_gb,
        )
        gpu_name = resolve_gpu_name(config_gpu_name, detected_gpu.get("name"), configured_gpu_name)
        gpu_memory = resolve_gpu_memory(
            config_gpu_memory,
            detected_gpu.get("memory_gb"),
            configured_gpu_memory,
        )
        numeric_gpu_memory = _safe_float(gpu_memory, defaults.gpu_memory_gb)
        profile = normalize_setup_profile(requested_setup_profile, numeric_gpu_memory, gpu_name)
        advanced_defaults_only = quickstart_mode and not operator_mode
        use_profile_defaults = quickstart_mode or bool(str(config.get("setup_profile", "")).strip())
        support_preset = nvidia_support_preset(numeric_gpu_memory, gpu_name)
        preset_runtime_env = support_preset.runtime_env_defaults() if support_preset is not None else {}
        default_concurrency = profile_concurrency(profile, numeric_gpu_memory, gpu_name)
        default_batch_tokens = profile_batch_tokens(profile, defaults.max_batch_tokens)
        default_thermal_headroom = profile_thermal_headroom(profile, defaults.thermal_headroom)
        default_max_context_tokens = str(
            support_preset.max_context_tokens
            if support_preset is not None and support_preset.max_context_tokens is not None
            else defaults.max_context_tokens
        )
        default_vllm_startup_timeout_seconds = str(
            support_preset.vllm_startup_timeout_seconds
            if support_preset is not None and support_preset.vllm_startup_timeout_seconds is not None
            else defaults.vllm_startup_timeout_seconds
        )
        default_vllm_extra_args = (
            support_preset.vllm_extra_args
            if support_preset is not None and support_preset.vllm_extra_args
            else defaults.vllm_extra_args
        )
        default_vllm_memory_profiler_estimate_cudagraphs = preset_runtime_env.get(
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
            stringify_bool(defaults.vllm_memory_profiler_estimate_cudagraphs),
        )
        configured_deployment_target = (
            first_nonempty(str(config.get("deployment_target", "")), persisted.get("DEPLOYMENT_TARGET"))
            if operator_mode
            else first_nonempty(persisted.get("DEPLOYMENT_TARGET"))
        )
        configured_inference_engine = (
            first_nonempty(str(config.get("inference_engine", "")), persisted.get("INFERENCE_ENGINE"))
            if operator_mode
            else first_nonempty(persisted.get("INFERENCE_ENGINE"))
        )
        configured_runtime_profile = (
            first_nonempty(str(config.get("runtime_profile", "")), persisted.get("RUNTIME_PROFILE"))
            if operator_mode
            else (None if advanced_defaults_only else first_nonempty(persisted.get("RUNTIME_PROFILE")))
        )
        credentials_present = self.credentials_path.exists() or bool(
            first_nonempty(current.get("NODE_ID"), persisted.get("NODE_ID"))
            and first_nonempty(current.get("NODE_KEY"), persisted.get("NODE_KEY"))
        )
        configured_startup_model = first_nonempty(str(config.get("vllm_model", "")), persisted.get("VLLM_MODEL"))
        runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=configured_inference_engine,
            configured_deployment_target=configured_deployment_target,
            runtime_backend=selected_runtime_backend,
            model=configured_startup_model or recommended_startup_model(numeric_gpu_memory, gpu_name),
        )
        deployment_target = runtime_profile.deployment_target
        inference_engine = runtime_profile.inference_engine
        token_configured = bool(
            first_nonempty(
                str(config.get("hugging_face_hub_token", "")),
                current.get("HUGGING_FACE_HUB_TOKEN"),
                current.get("HF_TOKEN"),
            )
        )
        base_startup_model = runtime_profile.default_model or recommended_startup_model(numeric_gpu_memory, gpu_name)
        base_supported_models = (
            ",".join(runtime_profile.supported_models)
            if configured_runtime_profile
            else recommended_supported_models(numeric_gpu_memory, gpu_name)
        )
        startup_plan = resolve_startup_model_plan(
            base_startup_model,
            base_supported_models,
            token_configured=token_configured,
            inference_engine=inference_engine,
            bootstrap_first_run=advanced_defaults_only and not credentials_present,
        )
        owner_startup_model = str(startup_plan["target_model"])
        owner_supported_models = str(startup_plan["target_supported_models"])
        bootstrap_startup_model = str(startup_plan["active_model"])
        bootstrap_supported_models = str(startup_plan["active_supported_models"])
        owner_runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=inference_engine,
            configured_deployment_target=deployment_target,
            runtime_backend=selected_runtime_backend,
            model=owner_startup_model,
        )
        owner_supported_models = constrain_supported_models_for_runtime_profile(
            owner_supported_models,
            runtime_profile=owner_runtime_profile,
            preferred_model=owner_startup_model,
        )
        bootstrap_runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=inference_engine,
            configured_deployment_target=deployment_target,
            runtime_backend=selected_runtime_backend,
            model=bootstrap_startup_model,
        )
        bootstrap_supported_models = constrain_supported_models_for_runtime_profile(
            bootstrap_supported_models,
            runtime_profile=bootstrap_runtime_profile,
            preferred_model=bootstrap_startup_model,
        )
        selected_startup_model = (
            bootstrap_startup_model
            if advanced_defaults_only
            else first_nonempty(
                str(config.get("vllm_model", "")),
                persisted.get("VLLM_MODEL"),
                base_startup_model,
            )
        )
        selected_supported_models = (
            bootstrap_supported_models
            if advanced_defaults_only
            else first_nonempty(
                str(config.get("supported_models", "")),
                persisted.get("SUPPORTED_MODELS"),
                base_supported_models,
            )
        )
        runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=inference_engine,
            configured_deployment_target=deployment_target,
            runtime_backend=selected_runtime_backend,
            model=selected_startup_model,
        )
        selected_supported_models = constrain_supported_models_for_runtime_profile(
            selected_supported_models,
            runtime_profile=runtime_profile,
            preferred_model=selected_startup_model,
        )
        deployment_target = runtime_profile.deployment_target
        inference_engine = runtime_profile.inference_engine
        llama_cpp_settings = llama_cpp_env_for_model(selected_startup_model)
        runtime_profile_setting = (
            AUTO_RUNTIME_PROFILE
            if advanced_defaults_only
            else runtime_profile.id
        )
        default_trust_tier = recommended_trust_tier(inferred_attestation_provider)
        default_restricted_capable = recommended_restricted_capable(inferred_attestation_provider)
        default_inference_image = runtime_profile.image if runtime_backend_supports_compose(selected_runtime_backend) else ""
        is_burst_profile = runtime_profile.capacity_class == "elastic_burst"
        default_burst_phase = "accept_burst_work" if is_burst_profile else ""
        hugging_face_token = first_nonempty(
            str(config.get("hugging_face_hub_token", "")),
            current.get("HUGGING_FACE_HUB_TOKEN"),
            current.get("HF_TOKEN"),
        )
        resolved_offline_bundle_dir = self.configured_offline_install_bundle_dir(
            config=config,
            env_values=current,
        )
        startup_warm_source_env = self.startup_warm_source_env_values(
            current,
            selected_startup_model,
            inference_engine=inference_engine,
            runtime_backend=selected_runtime_backend,
        )

        return {
            RUNTIME_BACKEND_ENV: selected_runtime_backend,
            "SETUP_PROFILE": profile,
            "EDGE_CONTROL_URL": (
                first_nonempty(str(config.get("edge_control_url", "")), current.get("EDGE_CONTROL_URL"), defaults.edge_control_url)
                if operator_mode
                else first_nonempty(current.get("EDGE_CONTROL_URL"), defaults.edge_control_url)
            ),
            "EDGE_CONTROL_FALLBACK_URLS": first_nonempty(
                str(config.get("edge_control_fallback_urls", "")),
                current.get("EDGE_CONTROL_FALLBACK_URLS"),
            ),
            "OPERATOR_TOKEN": first_nonempty(str(config.get("operator_token", "")), current.get("OPERATOR_TOKEN")),
            "NODE_ID": first_nonempty(str(config.get("node_id", "")), current.get("NODE_ID")),
            "NODE_KEY": first_nonempty(str(config.get("node_key", "")), current.get("NODE_KEY")),
            "NODE_LABEL": suggested_node_label(
                str(config.get("node_label", "")).strip() or current.get("NODE_LABEL"),
                detected_gpu.get("name"),
                defaults.node_label,
            ),
            "NODE_REGION": (
                first_nonempty(str(config.get("node_region", "")), persisted.get("NODE_REGION"), inferred_region)
                if operator_mode
                else first_nonempty(persisted.get("NODE_REGION"), inferred_region)
            ),
            "TRUST_TIER": (
                first_nonempty(str(config.get("trust_tier", "")), persisted.get("TRUST_TIER"), default_trust_tier)
                if not advanced_defaults_only
                else default_trust_tier
            ),
            "RESTRICTED_CAPABLE": (
                stringify_bool(
                    bool(config.get("restricted_capable"))
                    if "restricted_capable" in config
                    else (persisted.get("RESTRICTED_CAPABLE", stringify_bool(default_restricted_capable)).lower() == "true")
                )
                if not advanced_defaults_only
                else stringify_bool(default_restricted_capable)
            ),
            "CREDENTIALS_PATH": credentials_path,
            "AUTOPILOT_STATE_PATH": autopilot_state_path,
            "HEAT_GOVERNOR_STATE_PATH": heat_governor_state_path,
            "FAULT_INJECTION_STATE_PATH": fault_injection_state_path,
            "STARTUP_STATUS_PATH": startup_status_path,
            "STARTUP_STATUS_HOST": first_nonempty(
                str(config.get("startup_status_host", "")),
                persisted.get("STARTUP_STATUS_HOST"),
                DEFAULT_STARTUP_STATUS_HOST,
            ),
            "STARTUP_STATUS_PORT": first_nonempty(
                optional_env_value(config.get("startup_status_port")),
                persisted.get("STARTUP_STATUS_PORT"),
                str(DEFAULT_STARTUP_STATUS_PORT),
            ),
            "STARTUP_STATUS_ENDPOINT_PATH": first_nonempty(
                str(config.get("startup_status_endpoint_path", "")),
                persisted.get("STARTUP_STATUS_ENDPOINT_PATH"),
                DEFAULT_STARTUP_STATUS_ENDPOINT_PATH,
            ),
            "CONTROL_PLANE_STATE_PATH": first_nonempty(
                current.get("CONTROL_PLANE_STATE_PATH"),
                defaults.control_plane_state_path,
            ),
            "RUNTIME_PROFILE": runtime_profile_setting,
            "DEPLOYMENT_TARGET": deployment_target,
            "INFERENCE_ENGINE": inference_engine,
            "RUNTIME_IMAGE": first_nonempty(
                default_inference_image if advanced_defaults_only and default_inference_image else None,
                str(config.get("runtime_image", "")),
                None if advanced_defaults_only else persisted.get("RUNTIME_IMAGE"),
                persisted.get("RUNTIME_IMAGE"),
                str(config.get("vllm_image", "")),
                None if advanced_defaults_only else persisted.get("VLLM_IMAGE"),
                persisted.get("VLLM_IMAGE"),
                default_inference_image,
            ),
            "INFERENCE_BASE_URL": first_nonempty(
                str(config.get("inference_base_url", "")),
                str(config.get("vllm_base_url", "")),
                None if advanced_defaults_only else persisted.get("INFERENCE_BASE_URL"),
                None if advanced_defaults_only else persisted.get("VLLM_BASE_URL"),
                persisted.get("INFERENCE_BASE_URL"),
                persisted.get("VLLM_BASE_URL"),
                defaults.resolved_inference_base_url,
            ),
            "CAPACITY_CLASS": first_nonempty(
                str(config.get("capacity_class", "")),
                persisted.get("CAPACITY_CLASS"),
                runtime_profile.capacity_class,
            ),
            "TEMPORARY_NODE": first_nonempty(
                stringify_bool(coerce_bool(config.get("temporary_node")))
                if "temporary_node" in config
                else "",
                persisted.get("TEMPORARY_NODE"),
                stringify_bool(is_burst_profile),
            ),
            "BURST_PROVIDER": first_nonempty(
                str(config.get("burst_provider", "")),
                persisted.get("BURST_PROVIDER"),
                "vast_ai" if is_burst_profile else "",
            ),
            "BURST_LEASE_ID": first_nonempty(str(config.get("burst_lease_id", "")), persisted.get("BURST_LEASE_ID")),
            "BURST_LEASE_PHASE": first_nonempty(
                str(config.get("burst_lease_phase", "")),
                persisted.get("BURST_LEASE_PHASE"),
                default_burst_phase,
            ),
            "BURST_COST_CEILING_USD": first_nonempty(
                optional_env_value(config.get("burst_cost_ceiling_usd")),
                persisted.get("BURST_COST_CEILING_USD"),
                "" if runtime_profile.burst_cost_ceiling_usd is None else str(runtime_profile.burst_cost_ceiling_usd),
            ),
            "VLLM_BASE_URL": first_nonempty(
                current.get("VLLM_BASE_URL"),
                current.get("INFERENCE_BASE_URL"),
                defaults.resolved_inference_base_url,
            ),
            "GPU_NAME": gpu_name,
            "GPU_MEMORY_GB": gpu_memory,
            "MAX_CONTEXT_TOKENS": (
                default_max_context_tokens
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("max_context_tokens", "")),
                    None if use_profile_defaults else persisted.get("MAX_CONTEXT_TOKENS"),
                    default_max_context_tokens,
                )
            ),
            "VLLM_STARTUP_TIMEOUT_SECONDS": (
                default_vllm_startup_timeout_seconds
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("vllm_startup_timeout_seconds", "")),
                    None if use_profile_defaults else persisted.get("VLLM_STARTUP_TIMEOUT_SECONDS"),
                    default_vllm_startup_timeout_seconds,
                )
            ),
            "VLLM_EXTRA_ARGS": (
                default_vllm_extra_args
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("vllm_extra_args", "")),
                    None if use_profile_defaults else persisted.get("VLLM_EXTRA_ARGS"),
                    default_vllm_extra_args,
                )
            ),
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": (
                default_vllm_memory_profiler_estimate_cudagraphs
                if advanced_defaults_only
                else first_nonempty(
                    stringify_bool(coerce_bool(config.get("vllm_memory_profiler_estimate_cudagraphs")))
                    if "vllm_memory_profiler_estimate_cudagraphs" in config
                    else "",
                    None if use_profile_defaults else persisted.get("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
                    default_vllm_memory_profiler_estimate_cudagraphs,
                )
            ),
            "MAX_BATCH_TOKENS": (
                default_batch_tokens
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("max_batch_tokens", "")),
                    None if use_profile_defaults else persisted.get("MAX_BATCH_TOKENS"),
                    default_batch_tokens,
                )
            ),
            "MAX_CONCURRENT_ASSIGNMENTS": (
                default_concurrency
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("max_concurrent_assignments", "")),
                    None if use_profile_defaults else persisted.get("MAX_CONCURRENT_ASSIGNMENTS"),
                    default_concurrency,
                )
            ),
            "CONTROL_PLANE_GRACE_SECONDS": first_nonempty(
                str(config.get("control_plane_grace_seconds", "")),
                current.get("CONTROL_PLANE_GRACE_SECONDS"),
                str(defaults.control_plane_grace_seconds),
            ),
            "CONTROL_PLANE_RETRY_FLOOR_SECONDS": first_nonempty(
                str(config.get("control_plane_retry_floor_seconds", "")),
                current.get("CONTROL_PLANE_RETRY_FLOOR_SECONDS"),
                str(defaults.control_plane_retry_floor_seconds),
            ),
            "CONTROL_PLANE_RETRY_CAP_SECONDS": first_nonempty(
                str(config.get("control_plane_retry_cap_seconds", "")),
                current.get("CONTROL_PLANE_RETRY_CAP_SECONDS"),
                str(defaults.control_plane_retry_cap_seconds),
            ),
            "ARTIFACT_MIRROR_BASE_URLS": first_nonempty(
                str(config.get("artifact_mirror_base_urls", "")),
                current.get("ARTIFACT_MIRROR_BASE_URLS"),
            ),
            "MODEL_CACHE_BUDGET_GB": (
                ""
                if advanced_defaults_only
                else first_nonempty(
                    optional_env_value(config.get("model_cache_budget_gb")),
                    persisted.get("MODEL_CACHE_BUDGET_GB"),
                )
            ),
            "MODEL_CACHE_RESERVE_FREE_GB": (
                ""
                if advanced_defaults_only
                else first_nonempty(
                    optional_env_value(config.get("model_cache_reserve_free_gb")),
                    persisted.get("MODEL_CACHE_RESERVE_FREE_GB"),
                )
            ),
            "OFFLINE_INSTALL_BUNDLE_DIR": first_nonempty(
                str(config.get("offline_install_bundle_dir", "")),
                persisted.get("OFFLINE_INSTALL_BUNDLE_DIR"),
                str(resolved_offline_bundle_dir) if resolved_offline_bundle_dir is not None else "",
            ),
            **startup_warm_source_env,
            "HEAT_GOVERNOR_MODE": first_nonempty(
                str(config.get("heat_governor_mode", "")),
                persisted.get("HEAT_GOVERNOR_MODE"),
                defaults.heat_governor_mode,
            ),
            "OWNER_OBJECTIVE": first_nonempty(
                str(config.get("owner_objective", "")),
                persisted.get("OWNER_OBJECTIVE"),
                defaults.owner_objective,
            ),
            "TARGET_GPU_UTILIZATION_PCT": (
                str(defaults.target_gpu_utilization_pct)
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("target_gpu_utilization_pct", "")),
                    persisted.get("TARGET_GPU_UTILIZATION_PCT"),
                    str(defaults.target_gpu_utilization_pct),
                )
            ),
            "MIN_GPU_MEMORY_HEADROOM_PCT": (
                str(defaults.min_gpu_memory_headroom_pct)
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("min_gpu_memory_headroom_pct", "")),
                    persisted.get("MIN_GPU_MEMORY_HEADROOM_PCT"),
                    str(defaults.min_gpu_memory_headroom_pct),
                )
            ),
            "THERMAL_HEADROOM": (
                default_thermal_headroom
                if advanced_defaults_only
                else first_nonempty(
                    str(config.get("thermal_headroom", "")),
                    None if use_profile_defaults else persisted.get("THERMAL_HEADROOM"),
                    default_thermal_headroom,
                )
            ),
            "HEAT_DEMAND": first_nonempty(
                str(config.get("heat_demand", "")),
                persisted.get("HEAT_DEMAND"),
                defaults.heat_demand,
            ),
            "ROOM_TEMP_C": first_nonempty(
                optional_env_value(config.get("room_temp_c")),
                persisted.get("ROOM_TEMP_C"),
            ),
            "TARGET_TEMP_C": first_nonempty(
                optional_env_value(config.get("target_temp_c")),
                persisted.get("TARGET_TEMP_C"),
            ),
            "OUTSIDE_TEMP_C": first_nonempty(
                optional_env_value(config.get("outside_temp_c")),
                persisted.get("OUTSIDE_TEMP_C"),
            ),
            "QUIET_HOURS_START_LOCAL": first_nonempty(
                str(config.get("quiet_hours_start_local", "")),
                persisted.get("QUIET_HOURS_START_LOCAL"),
            ),
            "QUIET_HOURS_END_LOCAL": first_nonempty(
                str(config.get("quiet_hours_end_local", "")),
                persisted.get("QUIET_HOURS_END_LOCAL"),
            ),
            "GPU_TEMP_C": first_nonempty(
                optional_env_value(config.get("gpu_temp_c")),
                persisted.get("GPU_TEMP_C"),
            ),
            "GPU_TEMP_LIMIT_C": first_nonempty(
                optional_env_value(config.get("gpu_temp_limit_c")),
                persisted.get("GPU_TEMP_LIMIT_C"),
                str(defaults.gpu_temp_limit_c),
            ),
            "POWER_WATTS": first_nonempty(
                optional_env_value(config.get("power_watts")),
                persisted.get("POWER_WATTS"),
            ),
            "ESTIMATED_HEAT_OUTPUT_WATTS": first_nonempty(
                optional_env_value(config.get("estimated_heat_output_watts")),
                persisted.get("ESTIMATED_HEAT_OUTPUT_WATTS"),
            ),
            "GPU_POWER_LIMIT_ENABLED": first_nonempty(
                stringify_bool(coerce_bool(config.get("gpu_power_limit_enabled")))
                if "gpu_power_limit_enabled" in config
                else "",
                persisted.get("GPU_POWER_LIMIT_ENABLED"),
                stringify_bool(defaults.gpu_power_limit_enabled),
            ),
            "MAX_POWER_CAP_WATTS": first_nonempty(
                optional_env_value(config.get("max_power_cap_watts")),
                persisted.get("MAX_POWER_CAP_WATTS"),
            ),
            "ENERGY_PRICE_KWH": first_nonempty(
                optional_env_value(config.get("energy_price_kwh")),
                persisted.get("ENERGY_PRICE_KWH"),
            ),
            "SUPPORTED_MODELS": (
                selected_supported_models
            ),
            "POLL_INTERVAL_SECONDS": current.get("POLL_INTERVAL_SECONDS", str(defaults.poll_interval_seconds)),
            "ATTESTATION_PROVIDER": first_nonempty(
                str(config.get("attestation_provider", "")),
                persisted.get("ATTESTATION_PROVIDER"),
                inferred_attestation_provider,
            ),
            "VLLM_MODEL": (
                selected_startup_model
                if advanced_defaults_only
                else selected_startup_model
            ),
            "OWNER_TARGET_MODEL": owner_startup_model,
            "OWNER_TARGET_SUPPORTED_MODELS": owner_supported_models,
            **llama_cpp_settings,
            "HUGGING_FACE_HUB_TOKEN": hugging_face_token,
            "HF_TOKEN": hugging_face_token,
            "DOCKER_IMAGE": first_nonempty(str(config.get("docker_image", "")), current.get("DOCKER_IMAGE"), defaults.docker_image),
            "VLLM_IMAGE": first_nonempty(
                default_inference_image if advanced_defaults_only and default_inference_image else None,
                str(config.get("vllm_image", "")),
                str(config.get("runtime_image", "")),
                None if advanced_defaults_only else persisted.get("VLLM_IMAGE"),
                None if advanced_defaults_only else persisted.get("RUNTIME_IMAGE"),
                persisted.get("VLLM_IMAGE"),
                persisted.get("RUNTIME_IMAGE"),
                default_inference_image,
            ),
        }

    def compose_command(self, args: list[str]) -> list[str]:
        self.sync_runtime_env()
        command = ["docker", "compose"]
        if self.release_env_path.exists():
            command.extend(["--env-file", str(self.release_env_path)])
        if self.runtime_env_path.exists():
            command.extend(["--env-file", str(self.runtime_env_path)])
        command.extend(args)
        return command

    def compose_service_definitions(self) -> dict[str, dict[str, bool]]:
        compose_path = self.runtime_dir / "docker-compose.yml"
        try:
            lines = compose_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return {}

        services: dict[str, dict[str, bool]] = {}
        in_services = False
        current_service: str | None = None

        for raw_line in lines:
            without_comment = raw_line.split("#", 1)[0].rstrip("\n")
            if not without_comment.strip():
                continue
            indent = len(without_comment) - len(without_comment.lstrip(" "))
            token = without_comment.strip()

            if token == "services:":
                in_services = True
                current_service = None
                continue
            if not in_services:
                continue
            if indent == 0:
                break
            if indent == 2 and token.endswith(":"):
                current_service = token[:-1]
                services[current_service] = {"has_image": False, "has_build": False}
                continue
            if current_service is None or indent < 4:
                continue
            if token.startswith("image:"):
                services[current_service]["has_image"] = True
            elif token.startswith("build:"):
                services[current_service]["has_build"] = True
        return services

    def validate_nvidia_runtime(self, env_values: dict[str, str], *, preflight: dict[str, Any] | None = None) -> None:
        detected = preflight.get("gpu") if isinstance(preflight, dict) else None
        if not isinstance(detected, dict) or not detected.get("detected"):
            detected = detect_gpu(self.command_runner, self.runtime_dir)
        if not isinstance(detected, dict) or not detected.get("detected"):
            raise RuntimeError(
                "No NVIDIA GPU runtime was detected. Quick Start currently requires a supported NVIDIA GPU with the NVIDIA runtime available."
            )

        gpu_name = str(detected.get("name") or env_values.get("GPU_NAME") or "NVIDIA GPU")
        gpu_memory_gb = _safe_float(detected.get("memory_gb") or env_values.get("GPU_MEMORY_GB"), 0.0)
        preset = nvidia_support_preset(gpu_memory_gb, gpu_name)
        preset_label = preset.label if preset is not None else "automatic NVIDIA defaults"
        preset_track = preset.capacity_label if preset is not None else "safe defaults"
        self.update_stage_progress(
            f"Detected {gpu_name} with {gpu_memory_gb:.1f} GB VRAM. Using the {preset_label} preset for {preset_track}.",
            nvidia_gpu_name=gpu_name,
            nvidia_gpu_memory_gb=gpu_memory_gb,
            nvidia_preset_label=preset_label,
            nvidia_preset_track=preset_track,
        )
        self.log(f"Detected {gpu_name} with {gpu_memory_gb:.1f} GB VRAM. Using the {preset_label} preset for {preset_track}.")

    def validate_hugging_face_access(self, model: str | None) -> None:
        env_values = self.effective_runtime_env()
        repository, token_required = hugging_face_validation_target(
            model,
            inference_engine=self.resolved_inference_engine(env_values),
        )
        if repository is None:
            self.update_stage_progress(
                "No Hugging Face access check is needed for this startup model.",
                hf_repository=None,
                hf_token_required=False,
                hf_validated=True,
            )
            self.log("No Hugging Face access validation is needed for this startup model.")
            return

        token = first_nonempty(
            env_values.get("HUGGING_FACE_HUB_TOKEN"),
            env_values.get("HF_TOKEN"),
            self.runtime_env_overrides().get("HUGGING_FACE_HUB_TOKEN"),
            self.runtime_env_overrides().get("HF_TOKEN"),
            os.getenv("HUGGING_FACE_HUB_TOKEN"),
            os.getenv("HF_TOKEN"),
        )
        if token_required and not token:
            self.update_stage_progress(
                hf_repository=repository,
                hf_token_required=token_required,
                hf_validated=False,
                hf_failure_kind="missing_token",
            )
            raise RuntimeError(
                f"Hugging Face access is required for {repository}. Add HUGGING_FACE_HUB_TOKEN or HF_TOKEN and retry Quick Start."
            )

        self.update_stage_progress(
            f"Validating Hugging Face access for {repository} before the first download starts.",
            hf_repository=repository,
            hf_token_required=token_required,
            hf_validated=False,
        )
        self.log(f"Validating Hugging Face access for {repository} before the first download starts.")
        request_kwargs: dict[str, Any] = {"timeout": 10.0}
        if token:
            request_kwargs["headers"] = {"Authorization": f"Bearer {token}"}
        try:
            response = httpx.get(f"https://huggingface.co/api/models/{repository}", **request_kwargs)
        except httpx.HTTPError as error:
            self.update_stage_progress(
                hf_repository=repository,
                hf_token_required=token_required,
                hf_validated=False,
                hf_failure_kind="network",
            )
            raise RuntimeError(
                f"Could not validate Hugging Face access for {repository}. Check your network connection or Hugging Face token and retry."
            ) from error

        if response.status_code == HTTPStatus.OK:
            detail = (
                f"Validated gated Hugging Face access for {repository}."
                if token_required
                else f"Validated public Hugging Face access for {repository}."
            )
            self.update_stage_progress(
                detail,
                hf_repository=repository,
                hf_token_required=token_required,
                hf_validated=True,
                hf_failure_kind=None,
            )
            self.log(detail)
            return
        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            self.update_stage_progress(
                hf_repository=repository,
                hf_token_required=token_required,
                hf_validated=False,
                hf_failure_kind="access_denied",
                hf_http_status=response.status_code,
            )
            raise RuntimeError(
                f"Hugging Face denied access to {repository}. Make sure HUGGING_FACE_HUB_TOKEN is valid and approved for this model, then retry."
            )
        if response.status_code == HTTPStatus.NOT_FOUND:
            self.update_stage_progress(
                hf_repository=repository,
                hf_token_required=token_required,
                hf_validated=False,
                hf_failure_kind="not_found",
                hf_http_status=response.status_code,
            )
            raise RuntimeError(
                f"The startup model {repository} is unavailable on Hugging Face right now. Retry later or switch to another startup preset in advanced settings."
            )
        self.update_stage_progress(
            hf_repository=repository,
            hf_token_required=token_required,
            hf_validated=False,
            hf_failure_kind="http_error",
            hf_http_status=response.status_code,
        )
        raise RuntimeError(
            f"Could not validate Hugging Face access for {repository}. Hugging Face returned HTTP {response.status_code}. Retry Quick Start once access is restored."
        )

    def pull_runtime_images(self, services: list[str]) -> None:
        if self.current_runtime_backend() == SINGLE_CONTAINER_RUNTIME_BACKEND:
            self.update_stage_progress(
                "The unified NVIDIA runtime image is already running. Quick Start is preparing the in-container services now.",
                download_total_items=0,
                download_completed_items=0,
            )
            self.log("Using the already running unified runtime image. No extra container pulls are needed in this mode.")
            return

        unique_services = list(dict.fromkeys(service for service in services if service))
        service_definitions = self.compose_service_definitions()
        pullable = [service for service in unique_services if service_definitions.get(service, {}).get("has_image")]
        skipped = [service for service in unique_services if service not in pullable]

        if skipped:
            skipped_labels = ", ".join(skipped)
            self.log(
                f"Preparing {skipped_labels} from the local checkout. Pullable runtime images will still be cached first."
            )

        total = len(pullable)
        if total == 0:
            self.update_stage_progress(
                "Preparing the local runtime from this checkout. This first step can still take a minute.",
                download_total_items=0,
                download_completed_items=0,
            )
            return

        for index, service in enumerate(pullable, start=1):
            progress_message = (
                f"Downloading runtime containers ({index}/{total}). "
                "This first startup can take several minutes while everything is cached locally."
            )
            self.update_stage_progress(
                progress_message,
                download_total_items=total,
                download_completed_items=index - 1,
                download_current_item=service,
            )
            self.log(f"Pulling the {service} runtime image ({index}/{total}) so the first startup is less disruptive.")
            self.command_runner(self.compose_command(["pull", service]), self.runtime_dir)
            self.update_stage_progress(
                progress_message,
                download_total_items=total,
                download_completed_items=index,
                download_current_item=service,
            )
            self.log(f"The {service} runtime image is now cached locally.")

    def compose_up(self, services: list[str]) -> None:
        if self.current_runtime_backend() == SINGLE_CONTAINER_RUNTIME_BACKEND:
            if self.runtime_controller is None:
                raise RuntimeError("The in-container runtime controller is unavailable.")
            requested = {service for service in services if service}
            self.runtime_controller.start(
                recreate=False,
                start_vllm="vllm" in requested,
                start_node="node-agent" in requested,
            )
            return
        self.command_runner(self.compose_command(["up", "-d", *services]), self.runtime_dir)

    def maybe_enable_autostart(self) -> None:
        try:
            status = self.autostart_manager.ensure_enabled()
        except Exception as error:
            self.log(f"Automatic start could not be enabled automatically: {error}")
            return
        if status.get("enabled"):
            self.log("Automatic start is enabled. The local node service will launch when you sign in.")
            return
        self.log(str(status.get("detail") or "Automatic start is unavailable on this machine."))

    def maybe_install_desktop_launcher(self) -> None:
        try:
            status = self.desktop_launcher_manager.ensure_enabled()
        except Exception as error:
            self.log(f"The desktop launcher could not be installed automatically: {error}")
            return
        if status.get("enabled"):
            self.log("A desktop launcher is installed so this node app can be reopened with one click.")
            return
        self.log(str(status.get("detail") or "Desktop launcher installation is unavailable on this machine."))

    def wait_for_vllm(
        self,
        timeout_seconds: float = 240.0,
        *,
        model: str | None = None,
        resume_from_stage: str | None = None,
        offline_bundle_report: dict[str, Any] | None = None,
    ) -> None:
        env_values = self.effective_runtime_env()
        diagnostics = self.startup_model_warmup_diagnostics(env_values, model=model)
        startup_model = str(diagnostics["startup_model"])
        runtime_profile = diagnostics["runtime_profile"]
        runtime_label = str(diagnostics["runtime_label"])
        readiness_url = f"http://{service_access_host()}:8000{runtime_profile.readiness_path}"
        expected_bytes = diagnostics["expected_bytes"]
        configured_max_context_tokens = diagnostics["configured_max_context_tokens"]
        context_limit_tokens = diagnostics["context_limit_tokens"]
        context_limit_source = diagnostics["context_limit_source"]
        expected_effective_context_tokens = diagnostics["expected_effective_context_tokens"]
        if diagnostics.get("error"):
            self.fail_warmup(
                str(diagnostics["error"]),
                kind=str(diagnostics.get("error_kind") or "warmup_preflight"),
                warm_model=startup_model,
                warm_runtime_label=runtime_label,
                warm_configured_context_tokens=configured_max_context_tokens,
                warm_context_limit_tokens=context_limit_tokens,
                warm_context_limit_source=context_limit_source,
                warm_expected_effective_context_tokens=expected_effective_context_tokens,
            )
        cache_dir = self.data_dir / "model-cache"
        baseline_cache_bytes = directory_size_bytes(cache_dir)
        expected_missing_bytes = (
            max(0, expected_bytes - min(baseline_cache_bytes, expected_bytes))
            if isinstance(expected_bytes, int) and expected_bytes > 0
            else None
        )
        warm_source = self.warm_source_payload(
            env_values=env_values,
            startup_model=startup_model,
            inference_engine=str(diagnostics.get("inference_engine") or self.resolved_inference_engine(env_values)),
            observed_cache_bytes=baseline_cache_bytes,
            offline_bundle_report=offline_bundle_report,
        )
        self.persist_last_warm_source(warm_source)
        disk_usage = shutil.disk_usage(self.runtime_dir)
        free_disk_bytes = int(getattr(disk_usage, "free", 0))
        if isinstance(expected_missing_bytes, int) and expected_missing_bytes > free_disk_bytes:
            self.fail_warmup(
                f"This node cannot finish downloading {startup_model} because it still needs about "
                f"{format_bytes(expected_missing_bytes)} of local model-cache space, but only about "
                f"{format_bytes(free_disk_bytes)} is free. Free disk space or choose a smaller startup model, "
                "then retry Quick Start.",
                kind="insufficient_disk",
                warm_model=startup_model,
                warm_expected_bytes=expected_bytes,
                warm_free_disk_bytes=free_disk_bytes,
                warm_missing_disk_bytes=expected_missing_bytes,
                warm_resume_from_bytes=baseline_cache_bytes,
                warm_resuming_download=False,
                warm_configured_context_tokens=configured_max_context_tokens,
                warm_context_limit_tokens=context_limit_tokens,
                warm_context_limit_source=context_limit_source,
                warm_expected_effective_context_tokens=expected_effective_context_tokens,
                warm_source_winner=warm_source["winner"],
                warm_source_label=warm_source["winner_label"],
                warm_source_detail=warm_source["detail"],
                warm_source_scope=warm_source["scope"],
                warm_source_order=warm_source["order"],
            )
        resuming_download = bool(
            isinstance(expected_bytes, int)
            and expected_bytes > 0
            and 0 < baseline_cache_bytes < expected_bytes
        )
        cache_complete = bool(
            isinstance(expected_bytes, int)
            and expected_bytes > 0
            and baseline_cache_bytes >= expected_bytes
        )
        reuse_existing_cache = baseline_cache_bytes > 0 and (not expected_bytes or cache_complete)
        if resuming_download or (expected_bytes and not reuse_existing_cache):
            self.set_install_stage(
                "downloading_model",
                f"Downloading {startup_model} into the local model cache.",
                resume_from_stage=resume_from_stage,
            )
        else:
            self.set_install_stage(
                "warming_model",
                f"Warming {startup_model} and checking the local {runtime_label} runtime.",
                resume_from_stage=resume_from_stage,
            )
        if expected_bytes:
            if resuming_download:
                self.log(
                    f"Resuming the local model download for {startup_model} from about "
                    f"{format_bytes(baseline_cache_bytes)} already present in the model cache."
                )
            elif reuse_existing_cache:
                self.log(
                    f"Reusing {format_bytes(baseline_cache_bytes)} already present in the local model cache for {startup_model}."
                )
            else:
                self.log(
                    f"First startup is downloading about {format_bytes(expected_bytes)} into the local model cache for {startup_model}."
                )
        else:
            self.log(
                f"Warming {startup_model}. The first startup can take several minutes while the local model cache fills."
            )

        deadline = time.time() + timeout_seconds
        last_progress_bucket = -1
        last_observed_cache_bytes = baseline_cache_bytes
        last_observed_at = time.time()
        last_progress_at = last_observed_at
        download_rate_bps: float | None = None
        downloaded_cache_bytes = 0
        cached_model_bytes = min(expected_bytes, baseline_cache_bytes) if expected_bytes else baseline_cache_bytes
        observed_cache_bytes = baseline_cache_bytes
        progress_percent: int | None = None
        seconds_without_progress = 0
        download_stalled = False
        download_slow = False
        while time.time() < deadline:
            now = time.time()
            if self.faults.consume("warm_gpu_oom"):
                self.fail_warmup(
                    f"This node cannot run {startup_model} on its current GPU because warm-up ran out of VRAM. "
                    "The live fault drill forced a GPU OOM before the runtime became ready.",
                    kind="gpu_oom",
                    warm_model=startup_model,
                    warm_expected_bytes=expected_bytes,
                    warm_downloaded_bytes=cached_model_bytes,
                    warm_progress_percent=progress_percent,
                    warm_reusing_cache=reuse_existing_cache,
                    warm_observed_cache_bytes=observed_cache_bytes,
                    warm_resuming_download=resuming_download,
                    warm_resume_from_bytes=baseline_cache_bytes,
                    warm_download_rate_bps=download_rate_bps,
                    warm_download_slow=download_slow,
                    warm_download_stalled=download_stalled,
                    warm_seconds_without_progress=seconds_without_progress,
                    warm_free_disk_bytes=free_disk_bytes,
                    warm_missing_disk_bytes=expected_missing_bytes,
                    warm_source_winner=warm_source["winner"],
                    warm_source_label=warm_source["winner_label"],
                    warm_source_detail=warm_source["detail"],
                    warm_source_scope=warm_source["scope"],
                    warm_source_order=warm_source["order"],
                )
            observed_cache_bytes = directory_size_bytes(cache_dir)
            downloaded_cache_bytes = max(0, observed_cache_bytes - baseline_cache_bytes)
            cached_model_bytes = (
                min(expected_bytes, baseline_cache_bytes + downloaded_cache_bytes)
                if isinstance(expected_bytes, int) and expected_bytes > 0
                else observed_cache_bytes
            )
            if observed_cache_bytes > last_observed_cache_bytes:
                elapsed_seconds = max(0.001, now - last_observed_at)
                download_rate_bps = (observed_cache_bytes - last_observed_cache_bytes) / elapsed_seconds
                last_progress_at = now
                last_observed_cache_bytes = observed_cache_bytes
                last_observed_at = now
            progress_percent: int | None = None
            seconds_without_progress = int(max(0, round(now - last_progress_at)))
            download_stalled = bool(
                expected_bytes
                and cached_model_bytes < expected_bytes
                and (now - last_progress_at) >= WARM_DOWNLOAD_STALL_SECONDS
            )
            download_slow = bool(
                expected_bytes
                and cached_model_bytes < expected_bytes
                and (now - last_progress_at) >= WARM_DOWNLOAD_SLOW_SECONDS
                and download_rate_bps is not None
                and download_rate_bps < WARM_DOWNLOAD_SLOW_RATE_BYTES_PER_SECOND
            )
            if expected_bytes and expected_bytes > 0:
                progress_percent = max(0, min(99, int((cached_model_bytes / expected_bytes) * 100)))
                if cached_model_bytes >= expected_bytes:
                    with self.lock:
                        current_stage = self.state.stage
                    if current_stage != "warming_model":
                        self.set_install_stage(
                            "warming_model",
                            f"Finishing the local warm-up now that {startup_model} is cached.",
                            resume_from_stage=resume_from_stage,
                        )
            rate_label = format_rate_bytes_per_second(download_rate_bps)
            if reuse_existing_cache:
                warm_message = (
                    f"Warming {startup_model}. Reusing the existing local model cache, so this step should finish faster once the {runtime_label} runtime is ready."
                )
            elif resuming_download and expected_bytes:
                warm_message = (
                    f"Resuming {startup_model}. About {format_bytes(cached_model_bytes)} of about "
                    f"{format_bytes(expected_bytes)} is cached locally already."
                )
            elif expected_bytes:
                warm_message = (
                    f"Warming {startup_model}. "
                    f"{format_bytes(cached_model_bytes)} of about {format_bytes(expected_bytes)} is ready in the local model cache."
                )
            else:
                warm_message = (
                    f"Warming {startup_model}. The first startup can take several minutes while the local model cache fills."
                )
            if download_stalled:
                warm_message += (
                    f" Download progress has been idle for about {seconds_without_progress} seconds."
                )
            elif download_slow and rate_label:
                warm_message += f" Download is moving slowly at about {rate_label}."
            self.update_stage_progress(
                warm_message,
                warm_model=startup_model,
                warm_expected_bytes=expected_bytes,
                warm_downloaded_bytes=cached_model_bytes,
                warm_progress_percent=progress_percent,
                warm_reusing_cache=reuse_existing_cache,
                warm_observed_cache_bytes=observed_cache_bytes,
                warm_resuming_download=resuming_download,
                warm_resume_from_bytes=baseline_cache_bytes,
                warm_download_rate_bps=download_rate_bps,
                warm_download_slow=download_slow,
                warm_download_stalled=download_stalled,
                warm_seconds_without_progress=seconds_without_progress,
                warm_free_disk_bytes=free_disk_bytes,
                warm_missing_disk_bytes=expected_missing_bytes,
                warm_configured_context_tokens=configured_max_context_tokens,
                warm_context_limit_tokens=context_limit_tokens,
                warm_context_limit_source=context_limit_source,
                warm_expected_effective_context_tokens=expected_effective_context_tokens,
                warm_failure_kind=None,
                warm_failure_detail=None,
                warm_runtime_log_excerpt=None,
                warm_source_winner=warm_source["winner"],
                warm_source_label=warm_source["winner_label"],
                warm_source_detail=warm_source["detail"],
                warm_source_scope=warm_source["scope"],
                warm_source_order=warm_source["order"],
            )
            if progress_percent is not None:
                bucket = progress_percent // 10
                if bucket > last_progress_bucket:
                    last_progress_bucket = bucket
                    self.log(
                        f"Model cache progress for {startup_model}: {format_bytes(cached_model_bytes)} of about {format_bytes(expected_bytes)}."
                    )
            try:
                response = httpx.get(readiness_url, timeout=5.0)
                if response.status_code < 500:
                    self.update_stage_progress(
                        f"{startup_model} is warm and ready locally.",
                        warm_model=startup_model,
                        warm_expected_bytes=expected_bytes,
                        warm_downloaded_bytes=expected_bytes or cached_model_bytes,
                        warm_progress_percent=100 if expected_bytes else None,
                        warm_reusing_cache=reuse_existing_cache,
                        warm_observed_cache_bytes=observed_cache_bytes,
                        warm_resuming_download=resuming_download,
                        warm_resume_from_bytes=baseline_cache_bytes,
                        warm_download_rate_bps=download_rate_bps,
                        warm_download_slow=False,
                        warm_download_stalled=False,
                        warm_seconds_without_progress=0,
                        warm_free_disk_bytes=free_disk_bytes,
                        warm_missing_disk_bytes=expected_missing_bytes,
                        warm_configured_context_tokens=configured_max_context_tokens,
                        warm_context_limit_tokens=context_limit_tokens,
                        warm_context_limit_source=context_limit_source,
                        warm_expected_effective_context_tokens=expected_effective_context_tokens,
                        warm_failure_kind=None,
                        warm_failure_detail=None,
                        warm_runtime_log_excerpt=None,
                        warm_source_winner=warm_source["winner"],
                        warm_source_label=warm_source["winner_label"],
                        warm_source_detail=warm_source["detail"],
                        warm_source_scope=warm_source["scope"],
                        warm_source_order=warm_source["order"],
                    )
                    return
            except httpx.HTTPError:
                pass
            if download_stalled:
                log_diagnosis = self.diagnose_warmup_logs(
                    startup_model=startup_model,
                    runtime_label=runtime_label,
                    logs=self.recent_runtime_logs("vllm"),
                )
                if log_diagnosis is not None:
                    self.fail_warmup(
                        str(log_diagnosis["message"]),
                        kind=str(log_diagnosis["kind"]),
                        warm_model=startup_model,
                        warm_expected_bytes=expected_bytes,
                        warm_downloaded_bytes=cached_model_bytes,
                        warm_progress_percent=progress_percent,
                        warm_reusing_cache=reuse_existing_cache,
                        warm_observed_cache_bytes=observed_cache_bytes,
                        warm_resuming_download=resuming_download,
                        warm_resume_from_bytes=baseline_cache_bytes,
                        warm_download_rate_bps=download_rate_bps,
                        warm_download_slow=download_slow,
                        warm_download_stalled=download_stalled,
                        warm_seconds_without_progress=seconds_without_progress,
                        warm_free_disk_bytes=free_disk_bytes,
                        warm_missing_disk_bytes=expected_missing_bytes,
                        warm_runtime_log_excerpt=log_diagnosis.get("warm_runtime_log_excerpt"),
                    )
            self.sleep(2)
        log_diagnosis = self.diagnose_warmup_logs(
            startup_model=startup_model,
            runtime_label=runtime_label,
            logs=self.recent_runtime_logs("vllm"),
        )
        if log_diagnosis is not None:
            self.fail_warmup(
                str(log_diagnosis["message"]),
                kind=str(log_diagnosis["kind"]),
                warm_model=startup_model,
                warm_expected_bytes=expected_bytes,
                warm_downloaded_bytes=cached_model_bytes,
                warm_progress_percent=progress_percent,
                warm_reusing_cache=reuse_existing_cache,
                warm_observed_cache_bytes=observed_cache_bytes,
                warm_resuming_download=resuming_download,
                warm_resume_from_bytes=baseline_cache_bytes,
                warm_download_rate_bps=download_rate_bps,
                warm_download_slow=download_slow,
                warm_download_stalled=download_stalled,
                warm_seconds_without_progress=seconds_without_progress,
                warm_free_disk_bytes=free_disk_bytes,
                warm_missing_disk_bytes=expected_missing_bytes,
                warm_runtime_log_excerpt=log_diagnosis.get("warm_runtime_log_excerpt"),
            )
        if expected_bytes and not reuse_existing_cache:
            if download_stalled:
                self.fail_warmup(
                    f"Downloading {startup_model} appears stalled at {format_bytes(cached_model_bytes)} of about "
                    f"{format_bytes(expected_bytes)}. The local cache has not grown for about "
                    f"{seconds_without_progress} seconds. Check DNS, Cloudflare/R2 reachability, or Hugging Face "
                    "access, then retry Quick Start. Cached bytes will be reused on the next attempt.",
                    kind="stalled_download",
                    warm_model=startup_model,
                    warm_expected_bytes=expected_bytes,
                    warm_downloaded_bytes=cached_model_bytes,
                    warm_progress_percent=progress_percent,
                    warm_reusing_cache=reuse_existing_cache,
                    warm_observed_cache_bytes=observed_cache_bytes,
                    warm_resuming_download=resuming_download,
                    warm_resume_from_bytes=baseline_cache_bytes,
                    warm_download_rate_bps=download_rate_bps,
                    warm_download_slow=download_slow,
                    warm_download_stalled=download_stalled,
                    warm_seconds_without_progress=seconds_without_progress,
                    warm_free_disk_bytes=free_disk_bytes,
                    warm_missing_disk_bytes=expected_missing_bytes,
                )
            rate_label = format_rate_bytes_per_second(download_rate_bps)
            if download_slow and rate_label:
                self.fail_warmup(
                    f"Downloading {startup_model} is still active but very slow at about {rate_label}. "
                    f"{format_bytes(cached_model_bytes)} of about {format_bytes(expected_bytes)} is cached locally. "
                    "Check the local network path to Hugging Face or signed artifact URLs, then retry Quick Start if "
                    "you want to resume from the cached bytes.",
                    kind="slow_download",
                    warm_model=startup_model,
                    warm_expected_bytes=expected_bytes,
                    warm_downloaded_bytes=cached_model_bytes,
                    warm_progress_percent=progress_percent,
                    warm_reusing_cache=reuse_existing_cache,
                    warm_observed_cache_bytes=observed_cache_bytes,
                    warm_resuming_download=resuming_download,
                    warm_resume_from_bytes=baseline_cache_bytes,
                    warm_download_rate_bps=download_rate_bps,
                    warm_download_slow=download_slow,
                    warm_download_stalled=download_stalled,
                    warm_seconds_without_progress=seconds_without_progress,
                    warm_free_disk_bytes=free_disk_bytes,
                    warm_missing_disk_bytes=expected_missing_bytes,
                )
            self.fail_warmup(
                f"{startup_model} is taking longer than expected to download or warm. "
                f"{format_bytes(cached_model_bytes)} of about {format_bytes(expected_bytes)} is ready in the local "
                "cache. Retry Quick Start if the runtime is no longer making progress; cached bytes will be reused.",
                kind="warm_timeout",
                warm_model=startup_model,
                warm_expected_bytes=expected_bytes,
                warm_downloaded_bytes=cached_model_bytes,
                warm_progress_percent=progress_percent,
                warm_reusing_cache=reuse_existing_cache,
                warm_observed_cache_bytes=observed_cache_bytes,
                warm_resuming_download=resuming_download,
                warm_resume_from_bytes=baseline_cache_bytes,
                warm_download_rate_bps=download_rate_bps,
                warm_download_slow=download_slow,
                warm_download_stalled=download_stalled,
                warm_seconds_without_progress=seconds_without_progress,
                warm_free_disk_bytes=free_disk_bytes,
                warm_missing_disk_bytes=expected_missing_bytes,
            )
        self.fail_warmup(
            f"{startup_model} did not finish warming in time. Retry Quick Start after checking the local runtime logs.",
            kind="warm_timeout",
            warm_model=startup_model,
            warm_expected_bytes=expected_bytes,
            warm_downloaded_bytes=cached_model_bytes,
            warm_progress_percent=progress_percent,
            warm_reusing_cache=reuse_existing_cache,
            warm_observed_cache_bytes=observed_cache_bytes,
            warm_resuming_download=resuming_download,
            warm_resume_from_bytes=baseline_cache_bytes,
            warm_download_rate_bps=download_rate_bps,
            warm_download_slow=download_slow,
            warm_download_stalled=download_stalled,
            warm_seconds_without_progress=seconds_without_progress,
            warm_free_disk_bytes=free_disk_bytes,
            warm_missing_disk_bytes=expected_missing_bytes,
            warm_runtime_log_excerpt=warmup_log_excerpt(self.recent_runtime_logs("vllm")),
        )

    def build_installer_settings(self, env_values: dict[str, str]) -> NodeAgentSettings:
        defaults = NodeAgentSettings()
        return NodeAgentSettings(
            edge_control_url=env_values["EDGE_CONTROL_URL"],
            edge_control_fallback_urls=env_values.get("EDGE_CONTROL_FALLBACK_URLS") or None,
            artifact_mirror_base_urls=env_values.get("ARTIFACT_MIRROR_BASE_URLS") or None,
            operator_token=env_values.get("OPERATOR_TOKEN") or None,
            node_label=env_values["NODE_LABEL"],
            node_region=env_values["NODE_REGION"],
            trust_tier=env_values["TRUST_TIER"],
            restricted_capable=env_values["RESTRICTED_CAPABLE"].lower() == "true",
            node_id=env_values.get("NODE_ID") or None,
            node_key=env_values.get("NODE_KEY") or None,
            credentials_path=str(self.credentials_path),
            autopilot_state_path=str(self.scratch_dir / "autopilot-state.json"),
            heat_governor_state_path=str(self.scratch_dir / "heat-governor-state.json"),
            control_plane_state_path=str(self.scratch_dir / "control-plane-state.json"),
            fault_injection_state_path=str(self.scratch_dir / DEFAULT_FAULT_INJECTION_STATE_NAME),
            runtime_profile=env_values.get("RUNTIME_PROFILE", defaults.runtime_profile),
            deployment_target=env_values.get("DEPLOYMENT_TARGET", defaults.deployment_target),
            inference_engine=env_values.get("INFERENCE_ENGINE", defaults.inference_engine),
            inference_base_url=f"http://{service_access_host()}:8000",
            runtime_image=env_values.get("RUNTIME_IMAGE") or env_values.get("VLLM_IMAGE") or defaults.runtime_image,
            vllm_image=env_values.get("VLLM_IMAGE") or env_values.get("RUNTIME_IMAGE") or defaults.vllm_image,
            vllm_base_url=f"http://{service_access_host()}:8000",
            vllm_model=env_values["VLLM_MODEL"],
            owner_target_model=env_values.get("OWNER_TARGET_MODEL") or env_values["VLLM_MODEL"],
            owner_target_supported_models=env_values.get("OWNER_TARGET_SUPPORTED_MODELS") or env_values["SUPPORTED_MODELS"],
            capacity_class=env_values.get("CAPACITY_CLASS") or None,
            temporary_node=coerce_bool(env_values.get("TEMPORARY_NODE"), defaults.temporary_node),
            burst_provider=env_values.get("BURST_PROVIDER") or None,
            burst_lease_id=env_values.get("BURST_LEASE_ID") or None,
            burst_lease_phase=env_values.get("BURST_LEASE_PHASE") or None,
            burst_cost_ceiling_usd=coerce_float_or_none(env_values.get("BURST_COST_CEILING_USD")),
            llama_cpp_hf_repo=env_values.get("LLAMA_CPP_HF_REPO", defaults.llama_cpp_hf_repo),
            llama_cpp_hf_file=env_values.get("LLAMA_CPP_HF_FILE", defaults.llama_cpp_hf_file),
            llama_cpp_alias=env_values.get("LLAMA_CPP_ALIAS", defaults.llama_cpp_alias),
            llama_cpp_embedding=env_values.get("LLAMA_CPP_EMBEDDING", "").lower() == "true",
            llama_cpp_pooling=env_values.get("LLAMA_CPP_POOLING") or defaults.llama_cpp_pooling,
            gpu_name=env_values["GPU_NAME"],
            gpu_memory_gb=float(env_values["GPU_MEMORY_GB"]),
            max_context_tokens=int(env_values["MAX_CONTEXT_TOKENS"]),
            max_batch_tokens=int(env_values["MAX_BATCH_TOKENS"]),
            max_concurrent_assignments=int(env_values["MAX_CONCURRENT_ASSIGNMENTS"]),
            control_plane_grace_seconds=int(
                env_values.get("CONTROL_PLANE_GRACE_SECONDS", defaults.control_plane_grace_seconds)
            ),
            control_plane_retry_floor_seconds=int(
                env_values.get("CONTROL_PLANE_RETRY_FLOOR_SECONDS", defaults.control_plane_retry_floor_seconds)
            ),
            control_plane_retry_cap_seconds=int(
                env_values.get("CONTROL_PLANE_RETRY_CAP_SECONDS", defaults.control_plane_retry_cap_seconds)
            ),
            heat_governor_mode=env_values.get("HEAT_GOVERNOR_MODE", defaults.heat_governor_mode),
            owner_objective=env_values.get("OWNER_OBJECTIVE", defaults.owner_objective),
            target_gpu_utilization_pct=int(
                env_values.get("TARGET_GPU_UTILIZATION_PCT", defaults.target_gpu_utilization_pct)
            ),
            min_gpu_memory_headroom_pct=float(
                env_values.get("MIN_GPU_MEMORY_HEADROOM_PCT", defaults.min_gpu_memory_headroom_pct)
            ),
            thermal_headroom=float(env_values["THERMAL_HEADROOM"]),
            heat_demand=env_values.get("HEAT_DEMAND", defaults.heat_demand),
            room_temp_c=coerce_float_or_none(env_values.get("ROOM_TEMP_C")),
            target_temp_c=coerce_float_or_none(env_values.get("TARGET_TEMP_C")),
            outside_temp_c=coerce_float_or_none(env_values.get("OUTSIDE_TEMP_C")),
            quiet_hours_start_local=env_values.get("QUIET_HOURS_START_LOCAL") or None,
            quiet_hours_end_local=env_values.get("QUIET_HOURS_END_LOCAL") or None,
            gpu_temp_c=coerce_float_or_none(env_values.get("GPU_TEMP_C")),
            gpu_temp_limit_c=_safe_float(env_values.get("GPU_TEMP_LIMIT_C"), defaults.gpu_temp_limit_c),
            power_watts=coerce_float_or_none(env_values.get("POWER_WATTS")),
            estimated_heat_output_watts=coerce_float_or_none(env_values.get("ESTIMATED_HEAT_OUTPUT_WATTS")),
            gpu_power_limit_enabled=coerce_bool(
                env_values.get("GPU_POWER_LIMIT_ENABLED"),
                defaults.gpu_power_limit_enabled,
            ),
            max_power_cap_watts=(
                None
                if not env_values.get("MAX_POWER_CAP_WATTS")
                else _safe_int(env_values.get("MAX_POWER_CAP_WATTS"), 0) or None
            ),
            energy_price_kwh=coerce_float_or_none(env_values.get("ENERGY_PRICE_KWH")),
            supported_models=env_values["SUPPORTED_MODELS"],
            poll_interval_seconds=int(env_values["POLL_INTERVAL_SECONDS"]),
            agent_version=defaults.agent_version,
            docker_image=env_values.get("DOCKER_IMAGE", defaults.docker_image),
            attestation_provider=env_values["ATTESTATION_PROVIDER"],
        )

    def run_install(self, config: dict[str, Any], *, resume_from_stage: str | None = None) -> None:
        active_stage = "checking_docker"

        def set_stage(stage: str, message: str) -> None:
            nonlocal active_stage
            active_stage = stage
            self.set_install_stage(stage, message, resume_from_stage=resume_from_stage)

        def blocker_stage(preflight: dict[str, Any]) -> str:
            audit = preflight.get("setup_audit") if isinstance(preflight.get("setup_audit"), dict) else {}
            blocking_checks = audit.get("blocking_checks") if isinstance(audit.get("blocking_checks"), list) else []
            first_blocker = blocking_checks[0] if blocking_checks else None
            blocker_key = str(first_blocker.get("key") or "") if isinstance(first_blocker, dict) else ""
            if blocker_key in {"nvidia_driver", "cuda", "container_gpu"}:
                return "checking_nvidia_runtime"
            if blocker_key == "hugging_face":
                return "validating_hf_access"
            if blocker_key == "model_cache":
                return "downloading_model"
            return "checking_docker"

        try:
            set_stage("checking_docker", "Checking Docker and local prerequisites.")
            self.ensure_data_dirs()
            env_values = self.build_env(config)
            preflight = self.collect_preflight(env_values=env_values, force_refresh=True)
            automatic_fixes = self.attempt_safe_preflight_repairs(
                config=self.json_safe_config(config),
                env_values=env_values,
                preflight=preflight,
            )
            if automatic_fixes.get("attempted"):
                self.log(str(automatic_fixes.get("summary") or "Quick Start tried the safe local repairs first."))
                preflight = self.collect_preflight(env_values=env_values, force_refresh=True)
            preflight["automatic_fixes"] = automatic_fixes
            blockers = [
                str(blocker)
                for blocker in preflight.get("claim_gate_blockers", [])
                if isinstance(blocker, str) and blocker.strip()
            ]
            if blockers:
                failing_stage = blocker_stage(preflight)
                set_stage(failing_stage, "Setup still needs attention before Quick Start can continue.")
                raise RuntimeError(blockers[0])
            runtime_backend = str(preflight.get("runtime_backend") or self.current_runtime_backend())

            set_stage("checking_nvidia_runtime", "Checking the NVIDIA runtime and choosing a preset for this machine.")
            self.write_runtime_settings(env_values)
            self.write_runtime_env(env_values)
            self.log("Saved local node settings and generated the runtime config automatically.")
            self.validate_nvidia_runtime(env_values, preflight=preflight)

            if preflight.get("credentials_present"):
                set_stage("node_live", "Starting the local services with the existing node approval.")
                self.log("Existing node credentials or node keys were found. Starting the runtime directly.")
                services = ["vllm", "node-agent", "vector"] if runtime_backend_supports_compose(runtime_backend) else ["vllm", "node-agent"]
                self.compose_up(services)
                self.maybe_enable_autostart()
                self.maybe_install_desktop_launcher()
                self.complete("Runtime started with existing node credentials.")
                return

            set_stage("validating_hf_access", "Validating Hugging Face access for the startup model.")
            self.validate_hugging_face_access(env_values.get("VLLM_MODEL"))

            set_stage("pulling_image", "Pulling the runtime image and preparing the startup services.")
            services = ["vllm", "node-agent", "vector"] if runtime_backend_supports_compose(runtime_backend) else ["vllm", "node-agent"]
            offline_bundle_report = self.apply_offline_install_bundle(
                env_values=env_values,
                preflight=preflight,
            )
            self.log(str(offline_bundle_report.get("message") or ""))
            if not bool(offline_bundle_report.get("runtime_images_ready")):
                self.pull_runtime_images(services)
            else:
                self.update_stage_progress(
                    "Reusing bundled runtime images and starter-model cache before the first warm-up.",
                    download_total_items=0,
                    download_completed_items=0,
                )
            if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
                self.log("Starting the in-container NVIDIA runtime...")
            else:
                self.log(f"Starting the local {self.inference_runtime_label(env_values)} runtime...")
            self.compose_up(["vllm"])
            active_stage = "warming_model"
            self.wait_for_vllm(
                model=env_values.get("VLLM_MODEL"),
                resume_from_stage=resume_from_stage,
                offline_bundle_report=offline_bundle_report,
            )
            self.log(f"{self.inference_runtime_label(env_values)} is ready. Creating a browser-assisted node claim...")

            set_stage("claiming_node", "Creating a secure approval link and QR code for this machine.")
            settings = self.build_installer_settings(env_values)
            control = self.control_client_factory(settings)
            resumed_claim = self.reusable_claim_session()
            if resumed_claim is None:
                claim = control.create_node_claim_session()
                renewal_count = 0
                active_claims = [claim]
                self.set_claim(self.build_claim_state(claim, renewal_count=renewal_count))
                self.log(
                    f"Node claim created for this machine. Claim code: {claim.claim_code}. "
                    f"Approval URL: {claim.approval_url}"
                )
            else:
                claim, renewal_count = resumed_claim
                active_claims = [claim]
                self.set_claim(self.build_claim_state(claim, renewal_count=renewal_count))
                self.log(
                    f"Resuming the existing approval claim for this machine. Claim code: {claim.claim_code}. "
                    f"Approval URL: {claim.approval_url}"
                )
            refreshed_claim_ids: set[str] = set()
            self.log(
                "Open the approval page, sign in or create your operator account, and approve this machine. "
                "You can also scan the QR code shown in this app."
            )

            while True:
                next_active_claims: list[NodeClaimSession] = []
                claim_completed = False
                for active_claim in active_claims:
                    result = control.poll_node_claim_session(active_claim.claim_id, active_claim.poll_token)
                    active_claim.expires_at = result.expires_at
                    self.update_claim_status(
                        active_claim.claim_id,
                        status=result.status,
                        expires_at=result.expires_at,
                    )
                    if result.node_id and result.node_key:
                        self.log("Claim approved. Persisting node credentials...")
                        control.persist_credentials(result.node_id, result.node_key)
                        claim_completed = True
                        break
                    if result.status == "consumed":
                        raise RuntimeError(
                            "Node claim was consumed but did not return credentials. "
                            "Quick Start cannot recover this approval automatically. "
                            "Start Quick Start again from the setup UI."
                        )
                    if result.status not in {"expired", "consumed"}:
                        next_active_claims.append(active_claim)
                if claim_completed:
                    break

                active_claims = next_active_claims
                if not active_claims:
                    renewal_count += 1
                    refreshed_claim = control.create_node_claim_session()
                    active_claims = [refreshed_claim]
                    self.set_claim(self.build_claim_state(refreshed_claim, renewal_count=renewal_count))
                    self.log(
                        "The previous approval link expired, so Quick Start refreshed it automatically. "
                        f"New claim code: {refreshed_claim.claim_code}. Approval URL: {refreshed_claim.approval_url}"
                    )
                    continue

                displayed_claim = active_claims[-1]
                remaining_seconds = seconds_until_timestamp(displayed_claim.expires_at)
                if (
                    remaining_seconds is not None
                    and remaining_seconds <= CLAIM_AUTO_REFRESH_WINDOW_SECONDS
                    and displayed_claim.claim_id not in refreshed_claim_ids
                ):
                    renewal_count += 1
                    refreshed_claim_ids.add(displayed_claim.claim_id)
                    refreshed_claim = control.create_node_claim_session()
                    active_claims.append(refreshed_claim)
                    self.set_claim(self.build_claim_state(refreshed_claim, renewal_count=renewal_count))
                    self.log(
                        "Quick Start refreshed the approval link automatically before it expired. "
                        f"New claim code: {refreshed_claim.claim_code}. Approval URL: {refreshed_claim.approval_url}"
                    )

                self.sleep(max(1, min(claim_session.poll_interval_seconds for claim_session in active_claims)))

            set_stage("node_live", "Starting the node agent and bringing this machine online.")
            if runtime_backend_supports_compose(runtime_backend):
                self.log("Starting node-agent and vector services...")
                self.compose_up(["node-agent", "vector"])
            else:
                self.log("Starting the node agent inside the unified NVIDIA runtime...")
                self.compose_up(["node-agent"])
            if env_values.get("OWNER_TARGET_MODEL") and env_values.get("OWNER_TARGET_MODEL") != env_values.get("VLLM_MODEL"):
                self.log(
                    f"This node is live on the bootstrap model {env_values.get('VLLM_MODEL')}. "
                    f"The larger owner target {env_values.get('OWNER_TARGET_MODEL')} will warm in the background after startup."
                )
            self.maybe_enable_autostart()
            self.maybe_install_desktop_launcher()
            self.complete("Runtime started. The node agent will attest and begin polling the control plane automatically.")
        except Exception as error:  # pragma: no cover - exercised through tests
            self.log(f"Installer failed: {error}")
            self.set_error(str(error), step=active_stage)


def make_handler(installer: GuidedInstaller, admin_token: str) -> type[BaseHTTPRequestHandler]:
    sessions = LocalSessionStore()

    class InstallerHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:  # pragma: no cover
            return

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: bytes) -> None:
            self.send_response(HTTPStatus.OK.value)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("referrer-policy", "no-referrer")
            self.send_header("x-frame-options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _send_session_bootstrap(self) -> None:
            session_token = sessions.issue()
            body = session_bootstrap_html()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("referrer-policy", "no-referrer")
            self.send_header("x-frame-options", "DENY")
            self.send_header("set-cookie", serialize_cookie(LOCAL_SESSION_COOKIE, session_token, max_age=sessions.ttl_seconds))
            self.end_headers()
            self.wfile.write(body)

        def _authorized_via_admin_token(self, *, allow_query: bool) -> bool:
            provided = self.headers.get(ADMIN_TOKEN_HEADER)
            if not provided and allow_query:
                provided = request_query_param(self.path, "token")
            return token_matches(provided.strip() if isinstance(provided, str) else None, admin_token)

        def _authorized_via_session(self) -> bool:
            return sessions.contains(cookie_value(self.headers, LOCAL_SESSION_COOKIE))

        def _authorized(self, *, allow_query: bool) -> bool:
            return self._authorized_via_session() or self._authorized_via_admin_token(allow_query=allow_query)

        def _origin_allowed(self) -> bool:
            return origin_matches_host(self.headers)

        def do_GET(self) -> None:  # pragma: no cover
            path = urlparse(self.path).path
            if path == "/":
                if self._authorized_via_session():
                    body = installer_html().encode("utf-8")
                    self._send_html(body)
                    return
                if self._authorized_via_admin_token(allow_query=True):
                    self._send_session_bootstrap()
                    return
                if not self._authorized(allow_query=False):
                    self._send_json({"error": {"code": "unauthorized", "message": "A local admin token is required."}}, HTTPStatus.UNAUTHORIZED)
                    return
                body = installer_html().encode("utf-8")
                self._send_html(body)
                return
            if path == "/api/status":
                if not self._authorized(allow_query=False):
                    self._send_json({"error": {"code": "unauthorized", "message": "A local admin token is required."}}, HTTPStatus.UNAUTHORIZED)
                    return
                self._send_json(installer.status_payload())
                return
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # pragma: no cover
            path = urlparse(self.path).path
            if not self._authorized(allow_query=False):
                self._send_json({"error": {"code": "unauthorized", "message": "A local admin token is required."}}, HTTPStatus.UNAUTHORIZED)
                return
            if not self._origin_allowed():
                self._send_json({"error": {"code": "forbidden", "message": "Cross-origin requests are not allowed."}}, HTTPStatus.FORBIDDEN)
                return
            if path != "/api/install":
                self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                return

            content_length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
                return

            self._send_json(installer.start_install(payload))

    return InstallerHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the AUTONOMOUSc edge node guided installer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    require_secure_bind_host(args.host)
    installer = GuidedInstaller()
    installer.resume_if_needed()
    admin_token = generate_admin_token()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(installer, admin_token))
    url = f"http://{browser_access_host(args.host)}:{args.port}/?token={quote(admin_token)}"

    print(f"AUTONOMOUSc guided installer is available at {url}")
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down guided installer...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
