from __future__ import annotations

import argparse
import base64
import io
import json
import locale
import os
import shutil
import subprocess
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

from .autostart import AutoStartManager
from .config import NodeAgentSettings, NodeClaimSession
from .control_plane import EdgeControlClient
from .desktop_launcher import DesktopLauncherManager
from .inference_engine import (
    AUTO_RUNTIME_PROFILE,
    LLAMA_CPP_INFERENCE_ENGINE,
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
from .runtime_backend import (
    SINGLE_CONTAINER_RUNTIME_BACKEND,
    detect_runtime_backend,
    runtime_backend_label,
    runtime_backend_supports_compose,
)
from .runtime_layout import ensure_runtime_bundle, resolve_runtime_dir, service_access_host
from .single_container import EmbeddedRuntimeSupervisor


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
    "GPU_NAME",
    "GPU_MEMORY_GB",
    "MAX_CONTEXT_TOKENS",
    "MAX_BATCH_TOKENS",
    "MAX_CONCURRENT_ASSIGNMENTS",
    "THERMAL_HEADROOM",
    "HEAT_DEMAND",
    "ROOM_TEMP_C",
    "TARGET_TEMP_C",
    "GPU_TEMP_C",
    "POWER_WATTS",
    "ESTIMATED_HEAT_OUTPUT_WATTS",
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

PROFILE_LABELS = {
    "quiet": "Quiet",
    "balanced": "Balanced",
    "performance": "Performance",
}
RECOMMENDED_FREE_DISK_GB = 30.0
DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
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

    def matches(self, memory_gb: float) -> bool:
        if memory_gb < self.min_vram_gb:
            return False
        if self.max_vram_gb is None:
            return True
        return memory_gb <= self.max_vram_gb

    def concurrency_for_profile(self, profile: str) -> str:
        if profile == "quiet":
            return self.quiet_concurrency
        if profile == "performance":
            return self.performance_concurrency
        return self.balanced_concurrency

    def supported_models_csv(self) -> str:
        return ",".join(self.supported_models)


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
        key="community_embeddings",
        label="12-23 GB NVIDIA",
        min_vram_gb=12.0,
        max_vram_gb=23.9,
        capacity_label="Embeddings/community",
        startup_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=(DEFAULT_EMBEDDING_MODEL,),
        recommended_profile="balanced",
        quiet_concurrency="1",
        balanced_concurrency="1",
        performance_concurrency="2",
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
        balanced_concurrency="2",
        performance_concurrency="3",
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
        quiet_concurrency="1",
        balanced_concurrency="3",
        performance_concurrency="4",
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


def nvidia_support_preset(memory_gb: float | None) -> NvidiaSupportPreset | None:
    if memory_gb is None:
        return None
    for preset in NVIDIA_SUPPORT_PRESETS:
        if preset.matches(memory_gb):
            return preset
    return NVIDIA_SUPPORT_PRESETS[0]


def recommended_startup_model(memory_gb: float | None) -> str:
    preset = nvidia_support_preset(memory_gb)
    if preset is not None:
        return preset.startup_model
    return DEFAULT_VLLM_MODEL


def recommended_supported_models(memory_gb: float | None) -> str:
    preset = nvidia_support_preset(memory_gb)
    if preset is not None:
        return preset.supported_models_csv()
    if recommended_startup_model(memory_gb) == DEFAULT_EMBEDDING_MODEL:
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


def premium_eligibility(memory_gb: float | None, attestation_provider: str) -> dict[str, str | bool]:
    preset = nvidia_support_preset(memory_gb)
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
    if memory_gb is None or recommended_startup_model(memory_gb) != DEFAULT_VLLM_MODEL:
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


def suggest_concurrency(memory_gb: float | None) -> str:
    preset = nvidia_support_preset(memory_gb)
    if preset is not None:
        return preset.concurrency_for_profile(preset.recommended_profile)
    if memory_gb is None:
        return "2"
    if memory_gb < 16:
        return "1"
    if memory_gb >= 40:
        return "3"
    return "2"


def recommended_setup_profile(memory_gb: float | None) -> str:
    preset = nvidia_support_preset(memory_gb)
    if preset is not None:
        return preset.recommended_profile
    if memory_gb is None or memory_gb < 16:
        return "quiet"
    if memory_gb >= 40:
        return "performance"
    return "balanced"


def normalize_setup_profile(profile: str | None, memory_gb: float | None) -> str:
    normalized = (profile or "").strip().lower()
    if normalized in PROFILE_LABELS:
        return normalized
    return recommended_setup_profile(memory_gb)


def profile_concurrency(profile: str, memory_gb: float | None) -> str:
    preset = nvidia_support_preset(memory_gb)
    if preset is not None:
        return preset.concurrency_for_profile(profile)
    base = max(1, int(suggest_concurrency(memory_gb)))
    if profile == "quiet":
        return "1"
    if profile == "performance":
        if memory_gb is not None and memory_gb >= 24:
            return str(min(4, base + 1))
        return str(base)
    return str(base)


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
        self.state = InstallerState()
        self.lock = threading.RLock()
        self.install_thread: threading.Thread | None = None
        self.load_state()

    def current_runtime_backend(self) -> str:
        return detect_runtime_backend()

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

    def runtime_env_overrides(self) -> dict[str, str]:
        keys = (
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
            "GPU_NAME",
            "GPU_MEMORY_GB",
            "MAX_CONTEXT_TOKENS",
            "MAX_BATCH_TOKENS",
            "MAX_CONCURRENT_ASSIGNMENTS",
            "THERMAL_HEADROOM",
            "HEAT_DEMAND",
            "ROOM_TEMP_C",
            "TARGET_TEMP_C",
            "GPU_TEMP_C",
            "POWER_WATTS",
            "ESTIMATED_HEAT_OUTPUT_WATTS",
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
            "SETUP_PROFILE": str(config.get("setup_profile") or ""),
            "EDGE_CONTROL_URL": first_nonempty(str(config.get("edge_control_url", "")), defaults.edge_control_url),
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
            "GPU_NAME": first_nonempty(str(config.get("gpu_name", "")), defaults.gpu_name),
            "GPU_MEMORY_GB": str(_safe_float(config.get("gpu_memory_gb"), defaults.gpu_memory_gb)),
            "MAX_CONTEXT_TOKENS": str(_safe_int(config.get("max_context_tokens"), defaults.max_context_tokens)),
            "MAX_BATCH_TOKENS": str(_safe_int(config.get("max_batch_tokens"), defaults.max_batch_tokens)),
            "MAX_CONCURRENT_ASSIGNMENTS": str(
                _safe_int(config.get("max_concurrent_assignments"), defaults.max_concurrent_assignments)
            ),
            "THERMAL_HEADROOM": str(_safe_float(config.get("thermal_headroom"), defaults.thermal_headroom)),
            "HEAT_DEMAND": first_nonempty(str(config.get("heat_demand", "")), defaults.heat_demand),
            "ROOM_TEMP_C": optional_env_value(config.get("room_temp_c")),
            "TARGET_TEMP_C": optional_env_value(config.get("target_temp_c")),
            "GPU_TEMP_C": optional_env_value(config.get("gpu_temp_c")),
            "POWER_WATTS": optional_env_value(config.get("power_watts")),
            "ESTIMATED_HEAT_OUTPUT_WATTS": optional_env_value(config.get("estimated_heat_output_watts")),
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
        return {
            "version": RUNTIME_SETTINGS_VERSION,
            "config": {
                "setup_profile": env_values.get("SETUP_PROFILE") or recommended_setup_profile(defaults.gpu_memory_gb),
                "edge_control_url": env_values.get("EDGE_CONTROL_URL", defaults.edge_control_url),
                "operator_token": env_values.get("OPERATOR_TOKEN", ""),
                "node_id": env_values.get("NODE_ID", ""),
                "node_key": env_values.get("NODE_KEY", ""),
                "node_label": env_values.get("NODE_LABEL", defaults.node_label),
                "node_region": env_values.get("NODE_REGION", defaults.node_region),
                "trust_tier": env_values.get("TRUST_TIER", defaults.trust_tier),
                "restricted_capable": env_values.get("RESTRICTED_CAPABLE", "").lower() == "true",
                "credentials_path": env_values.get("CREDENTIALS_PATH", defaults.credentials_path),
                "autopilot_state_path": env_values.get("AUTOPILOT_STATE_PATH", defaults.autopilot_state_path),
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
                "gpu_name": env_values.get("GPU_NAME", defaults.gpu_name),
                "gpu_memory_gb": _safe_float(env_values.get("GPU_MEMORY_GB"), defaults.gpu_memory_gb),
                "max_context_tokens": _safe_int(env_values.get("MAX_CONTEXT_TOKENS"), defaults.max_context_tokens),
                "max_batch_tokens": _safe_int(env_values.get("MAX_BATCH_TOKENS"), defaults.max_batch_tokens),
                "max_concurrent_assignments": _safe_int(
                    env_values.get("MAX_CONCURRENT_ASSIGNMENTS"),
                    defaults.max_concurrent_assignments,
                ),
                "thermal_headroom": _safe_float(env_values.get("THERMAL_HEADROOM"), defaults.thermal_headroom),
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
        support_preset = nvidia_support_preset(numeric_gpu_memory)
        inferred_startup_model = recommended_startup_model(numeric_gpu_memory)
        inferred_supported_models = recommended_supported_models(numeric_gpu_memory)
        inferred_trust_tier = recommended_trust_tier(attestation_provider)
        inferred_restricted_capable = recommended_restricted_capable(attestation_provider)
        runtime_backend = self.current_runtime_backend()
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
        token_configured = bool(first_nonempty(source.get("HUGGING_FACE_HUB_TOKEN"), source.get("HF_TOKEN")))
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
        )
        recommended_profile = recommended_setup_profile(numeric_gpu_memory)
        concurrency = first_nonempty(
            persisted.get("MAX_CONCURRENT_ASSIGNMENTS"),
            profile_concurrency(profile, numeric_gpu_memory),
        )
        thermal_headroom = first_nonempty(
            persisted.get("THERMAL_HEADROOM"),
            profile_thermal_headroom(profile, default_settings.thermal_headroom),
        )
        premium = premium_eligibility(numeric_gpu_memory, attestation_provider)

        return {
            "edge_control_url": first_nonempty(source.get("EDGE_CONTROL_URL"), default_settings.edge_control_url),
            "operator_token_present": bool(first_nonempty(source.get("OPERATOR_TOKEN"))),
            "node_id_present": bool(first_nonempty(source.get("NODE_ID"))),
            "node_label": suggested_node_label(source.get("NODE_LABEL"), gpu.get("name"), default_settings.node_label),
            "node_region": first_nonempty(persisted.get("NODE_REGION"), region_value),
            "trust_tier": first_nonempty(persisted.get("TRUST_TIER"), inferred_trust_tier),
            "restricted_capable": (
                persisted.get("RESTRICTED_CAPABLE", stringify_bool(inferred_restricted_capable)).lower() == "true"
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
            "capacity_class": first_nonempty(persisted.get("CAPACITY_CLASS"), runtime_profile.capacity_class),
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
            "vllm_model": current_startup_model,
            "supported_models": current_supported_models,
            "owner_target_model": owner_target_model,
            "owner_target_supported_models": owner_target_supported_models,
            "bootstrap_pending_upgrade": bootstrap_pending_upgrade,
            "startup_model_fallback": startup_model_fallback,
            "max_concurrent_assignments": concurrency,
            "setup_profile": profile,
            "recommended_setup_profile": recommended_profile,
            "profile_summary": profile_summary(profile, gpu_name, concurrency),
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
            "thermal_headroom": thermal_headroom,
            "heat_demand": first_nonempty(persisted.get("HEAT_DEMAND"), default_settings.heat_demand),
            "room_temp_c": persisted.get("ROOM_TEMP_C") or "",
            "target_temp_c": persisted.get("TARGET_TEMP_C") or "",
            "gpu_temp_c": persisted.get("GPU_TEMP_C") or "",
            "power_watts": persisted.get("POWER_WATTS") or "",
            "estimated_heat_output_watts": persisted.get("ESTIMATED_HEAT_OUTPUT_WATTS") or "",
            "energy_price_kwh": persisted.get("ENERGY_PRICE_KWH") or "",
            "attestation_provider": first_nonempty(persisted.get("ATTESTATION_PROVIDER"), attestation_provider),
            "hugging_face_repository": hf_repository,
            "hugging_face_token_required": hf_token_required,
            "hugging_face_token_configured": token_configured,
            "llama_cpp_hf_repo": llama_cpp_settings["LLAMA_CPP_HF_REPO"],
            "llama_cpp_hf_file": llama_cpp_settings["LLAMA_CPP_HF_FILE"],
            "llama_cpp_alias": llama_cpp_settings["LLAMA_CPP_ALIAS"],
            "llama_cpp_embedding": llama_cpp_settings["LLAMA_CPP_EMBEDDING"].lower() == "true",
            "llama_cpp_pooling": llama_cpp_settings["LLAMA_CPP_POOLING"],
            "recommended_node_region": region_value,
            "region_reason": region_reason,
            "recommended_model": inferred_startup_model,
            "recommended_supported_models": inferred_supported_models,
            "recommended_max_concurrent_assignments": profile_concurrency(recommended_profile, numeric_gpu_memory),
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

    def collect_preflight(self, *, gpu: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ensure_data_dirs()
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
        blockers: list[str] = []
        if runtime_backend_supports_compose(runtime_backend) and not docker_cli:
            blockers.append("Install Docker Desktop to continue.")
        elif runtime_backend_supports_compose(runtime_backend) and not daemon_ok:
            blockers.append("Start Docker Desktop so the runtime can launch.")
        if nvidia_container_runtime.get("visible") is False:
            blockers.append("Enable NVIDIA GPU support in Docker Desktop before the runtime starts.")
        if not disk["ok"]:
            blockers.append(
                f"Free up disk space. The runtime works best with at least {int(RECOMMENDED_FREE_DISK_GB)} GB free."
            )

        runtime_env = self.effective_runtime_env()
        credentials_present = self.credentials_path.exists() or bool(
            first_nonempty(runtime_env.get("NODE_ID")) and first_nonempty(runtime_env.get("NODE_KEY"))
        )

        return {
            "docker_cli": docker_cli,
            "docker_compose": compose_ok,
            "docker_daemon": daemon_ok,
            "docker_error": docker_error,
            "gpu": gpu_payload,
            "nvidia_container_runtime": nvidia_container_runtime,
            "disk": disk,
            "running_services": running_services,
            "credentials_present": credentials_present,
            "runtime_backend": runtime_backend,
            "runtime_backend_label": runtime_backend_label(runtime_backend),
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
        preflight = self.collect_preflight(gpu=gpu)
        state = self.state_payload()
        autostart = self.autostart_manager.status()
        desktop_launcher = self.desktop_launcher_manager.status()
        return {
            "config": config,
            "preflight": preflight,
            "state": state,
            "owner_setup": self.owner_setup_payload(
                config=config,
                preflight=preflight,
                state=state,
                autostart=autostart,
            ),
            "autostart": autostart,
            "desktop_launcher": desktop_launcher,
        }

    def owner_setup_payload(
        self,
        *,
        config: dict[str, Any],
        preflight: dict[str, Any],
        state: dict[str, Any],
        autostart: dict[str, Any],
    ) -> dict[str, Any]:
        claim = state.get("claim") if isinstance(state.get("claim"), dict) else None
        stage_context = state.get("stage_context") if isinstance(state.get("stage_context"), dict) else {}
        running_services = preflight.get("running_services", [])
        docker_ready = bool(preflight.get("docker_cli") and preflight.get("docker_compose") and preflight.get("docker_daemon"))
        runtime_backend = str(preflight.get("runtime_backend") or self.current_runtime_backend())
        runtime_backend_name = str(preflight.get("runtime_backend_label") or runtime_backend_label(runtime_backend))
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
                                f"{hf_repository} is gated on Hugging Face. Add a token in operator/admin mode if you intentionally want to unlock that startup model."
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
                    if isinstance(expected_bytes, int) and expected_bytes > 0:
                        return (
                            f"Downloading {startup_model}: {format_bytes(downloaded_bytes)} of about "
                            f"{format_bytes(expected_bytes)} is ready in the local model cache."
                        )
                    return f"Downloading {startup_model} into the local model cache."
                return "Quick Start downloads the startup model into the local cache before warming it."
            if step_key == "warming_model":
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key] or credentials_present:
                    return "The startup model responded locally and the runtime is warm."
                if current_step == step_key:
                    if bool(stage_context.get("warm_reusing_cache")):
                        reused_bytes = _safe_int(stage_context.get("warm_observed_cache_bytes"), 0)
                        return (
                            f"Reusing {format_bytes(reused_bytes)} already in the local model cache while {startup_model} warms up."
                        )
                    return f"Finishing the local warm-up for {startup_model} now that the model files are cached."
                return "The runtime warms the startup model after the local download is complete."
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
            headline = "Ready for Quick Start"
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
                                else "Start Quick Start"
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

    def build_env(self, config: dict[str, Any]) -> dict[str, str]:
        current = self.load_effective_env()
        persisted = self.load_persisted_env()
        defaults = NodeAgentSettings()
        runtime_backend = self.current_runtime_backend()
        credentials_path = self.runtime_credentials_env_path(runtime_backend)
        autopilot_state_path = self.runtime_autopilot_state_env_path(runtime_backend)
        quickstart_mode = str(config.get("setup_mode", "")).strip().lower() == "quickstart"
        operator_mode = bool(config.get("operator_mode"))
        detected_gpu = detect_gpu(self.command_runner, self.runtime_dir)
        inferred_region, _region_reason = infer_node_region()
        inferred_attestation_provider, _attestation_reason = detect_attestation_provider(
            self.command_runner,
            self.runtime_dir,
        )
        profile = normalize_setup_profile(
            str(config.get("setup_profile", "") or (persisted.get("SETUP_PROFILE", "") if operator_mode else "")),
            detected_gpu.get("memory_gb"),
        )
        advanced_defaults_only = quickstart_mode and not operator_mode
        use_profile_defaults = quickstart_mode or bool(
            str(config.get("setup_profile", "")).strip()
        )

        config_gpu_name = str(config.get("gpu_name", "")).strip() or None
        config_gpu_memory = str(config.get("gpu_memory_gb", "")).strip() or None
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

        default_concurrency = profile_concurrency(profile, numeric_gpu_memory)
        default_batch_tokens = profile_batch_tokens(profile, defaults.max_batch_tokens)
        default_thermal_headroom = profile_thermal_headroom(profile, defaults.thermal_headroom)
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
            else first_nonempty(persisted.get("RUNTIME_PROFILE"))
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
            runtime_backend=runtime_backend,
            model=configured_startup_model or recommended_startup_model(numeric_gpu_memory),
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
        base_startup_model = runtime_profile.default_model or recommended_startup_model(numeric_gpu_memory)
        base_supported_models = (
            ",".join(runtime_profile.supported_models)
            if configured_runtime_profile
            else recommended_supported_models(numeric_gpu_memory)
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
            runtime_backend=runtime_backend,
            model=selected_startup_model,
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
        default_inference_image = runtime_profile.image if runtime_backend_supports_compose(runtime_backend) else ""
        is_burst_profile = runtime_profile.capacity_class == "elastic_burst"
        default_burst_phase = "accept_burst_work" if is_burst_profile else ""
        hugging_face_token = first_nonempty(
            str(config.get("hugging_face_hub_token", "")),
            current.get("HUGGING_FACE_HUB_TOKEN"),
            current.get("HF_TOKEN"),
        )

        return {
            "SETUP_PROFILE": profile,
            "EDGE_CONTROL_URL": (
                first_nonempty(str(config.get("edge_control_url", "")), current.get("EDGE_CONTROL_URL"), defaults.edge_control_url)
                if operator_mode
                else first_nonempty(current.get("EDGE_CONTROL_URL"), defaults.edge_control_url)
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
            "MAX_CONTEXT_TOKENS": current.get("MAX_CONTEXT_TOKENS", str(defaults.max_context_tokens)),
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
            "GPU_TEMP_C": first_nonempty(
                optional_env_value(config.get("gpu_temp_c")),
                persisted.get("GPU_TEMP_C"),
            ),
            "POWER_WATTS": first_nonempty(
                optional_env_value(config.get("power_watts")),
                persisted.get("POWER_WATTS"),
            ),
            "ESTIMATED_HEAT_OUTPUT_WATTS": first_nonempty(
                optional_env_value(config.get("estimated_heat_output_watts")),
                persisted.get("ESTIMATED_HEAT_OUTPUT_WATTS"),
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
        preset = nvidia_support_preset(gpu_memory_gb)
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
            )
            self.log(detail)
            return
        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            raise RuntimeError(
                f"Hugging Face denied access to {repository}. Make sure HUGGING_FACE_HUB_TOKEN is valid and approved for this model, then retry."
            )
        if response.status_code == HTTPStatus.NOT_FOUND:
            raise RuntimeError(
                f"The startup model {repository} is unavailable on Hugging Face right now. Retry later or switch to another startup preset in operator/admin mode."
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
    ) -> None:
        env_values = self.effective_runtime_env()
        runtime_profile = self.resolved_runtime_profile(env_values)
        inference_engine = runtime_profile.inference_engine
        runtime_label = inference_engine_label(inference_engine)
        readiness_url = f"http://{service_access_host()}:8000{runtime_profile.readiness_path}"
        startup_model = (model or "").strip() or env_values.get("VLLM_MODEL") or DEFAULT_VLLM_MODEL
        startup_artifact = startup_model_artifact(startup_model, runtime_engine=inference_engine)
        expected_bytes = artifact_total_size_bytes(startup_artifact)
        cache_dir = self.data_dir / "model-cache"
        baseline_cache_bytes = directory_size_bytes(cache_dir)
        reuse_existing_cache = baseline_cache_bytes > 0
        if reuse_existing_cache or not expected_bytes:
            self.set_install_stage(
                "warming_model",
                f"Warming {startup_model} and checking the local {runtime_label} runtime.",
                resume_from_stage=resume_from_stage,
            )
        else:
            self.set_install_stage(
                "downloading_model",
                f"Downloading {startup_model} into the local model cache.",
                resume_from_stage=resume_from_stage,
            )
        if expected_bytes:
            if reuse_existing_cache:
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
        while time.time() < deadline:
            observed_cache_bytes = directory_size_bytes(cache_dir)
            downloaded_cache_bytes = max(0, observed_cache_bytes - baseline_cache_bytes)
            progress_percent: int | None = None
            if expected_bytes and not reuse_existing_cache and expected_bytes > 0:
                progress_percent = max(0, min(99, int((downloaded_cache_bytes / expected_bytes) * 100)))
                if downloaded_cache_bytes >= expected_bytes:
                    with self.lock:
                        current_stage = self.state.stage
                    if current_stage != "warming_model":
                        self.set_install_stage(
                            "warming_model",
                            f"Finishing the local warm-up now that {startup_model} is cached.",
                            resume_from_stage=resume_from_stage,
                        )
            if reuse_existing_cache:
                warm_message = (
                    f"Warming {startup_model}. Reusing the existing local model cache, so this step should finish faster once the {runtime_label} runtime is ready."
                )
            elif expected_bytes:
                warm_message = (
                    f"Warming {startup_model}. "
                    f"{format_bytes(downloaded_cache_bytes)} of about {format_bytes(expected_bytes)} is ready in the local model cache."
                )
            else:
                warm_message = (
                    f"Warming {startup_model}. The first startup can take several minutes while the local model cache fills."
                )
            self.update_stage_progress(
                warm_message,
                warm_model=startup_model,
                warm_expected_bytes=expected_bytes,
                warm_downloaded_bytes=downloaded_cache_bytes,
                warm_progress_percent=progress_percent,
                warm_reusing_cache=reuse_existing_cache,
                warm_observed_cache_bytes=observed_cache_bytes,
            )
            if progress_percent is not None:
                bucket = progress_percent // 10
                if bucket > last_progress_bucket:
                    last_progress_bucket = bucket
                    self.log(
                        f"Model cache progress for {startup_model}: {format_bytes(downloaded_cache_bytes)} of about {format_bytes(expected_bytes)}."
                    )
            try:
                response = httpx.get(readiness_url, timeout=5.0)
                if response.status_code < 500:
                    self.update_stage_progress(
                        f"{startup_model} is warm and ready locally.",
                        warm_model=startup_model,
                        warm_expected_bytes=expected_bytes,
                        warm_downloaded_bytes=expected_bytes or downloaded_cache_bytes,
                        warm_progress_percent=100 if expected_bytes else None,
                        warm_reusing_cache=reuse_existing_cache,
                        warm_observed_cache_bytes=observed_cache_bytes,
                    )
                    return
            except httpx.HTTPError:
                pass
            self.sleep(2)
        if expected_bytes and not reuse_existing_cache:
            raise RuntimeError(
                f"{startup_model} is taking longer than expected to download or warm. "
                f"{format_bytes(downloaded_cache_bytes)} of about {format_bytes(expected_bytes)} is ready in the local cache. Retry Quick Start if progress has stalled."
            )
        raise RuntimeError(
            f"{startup_model} did not finish warming in time. Retry Quick Start after checking the local runtime logs."
        )

    def build_installer_settings(self, env_values: dict[str, str]) -> NodeAgentSettings:
        defaults = NodeAgentSettings()
        return NodeAgentSettings(
            edge_control_url=env_values["EDGE_CONTROL_URL"],
            operator_token=env_values.get("OPERATOR_TOKEN") or None,
            node_label=env_values["NODE_LABEL"],
            node_region=env_values["NODE_REGION"],
            trust_tier=env_values["TRUST_TIER"],
            restricted_capable=env_values["RESTRICTED_CAPABLE"].lower() == "true",
            node_id=env_values.get("NODE_ID") or None,
            node_key=env_values.get("NODE_KEY") or None,
            credentials_path=str(self.credentials_path),
            autopilot_state_path=str(self.data_dir / "scratch" / "autopilot-state.json"),
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
            thermal_headroom=float(env_values["THERMAL_HEADROOM"]),
            heat_demand=env_values.get("HEAT_DEMAND", defaults.heat_demand),
            room_temp_c=coerce_float_or_none(env_values.get("ROOM_TEMP_C")),
            target_temp_c=coerce_float_or_none(env_values.get("TARGET_TEMP_C")),
            gpu_temp_c=coerce_float_or_none(env_values.get("GPU_TEMP_C")),
            power_watts=coerce_float_or_none(env_values.get("POWER_WATTS")),
            estimated_heat_output_watts=coerce_float_or_none(env_values.get("ESTIMATED_HEAT_OUTPUT_WATTS")),
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

        try:
            set_stage("checking_docker", "Checking Docker and local prerequisites.")
            preflight = self.collect_preflight()
            if not (preflight["docker_cli"] and preflight["docker_compose"] and preflight["docker_daemon"]):
                self.set_stage("checking_docker", "Docker needs attention before Quick Start can resume.")
                raise RuntimeError(preflight["docker_error"] or "Docker preflight failed.")
            runtime_backend = str(preflight.get("runtime_backend") or self.current_runtime_backend())

            set_stage("checking_nvidia_runtime", "Checking the NVIDIA runtime and choosing a preset for this machine.")
            self.ensure_data_dirs()
            env_values = self.build_env(config)
            self.write_runtime_settings(env_values)
            self.sync_runtime_env()
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
            self.pull_runtime_images(services)
            if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
                self.log("Starting the in-container NVIDIA runtime...")
            else:
                self.log(f"Starting the local {self.inference_runtime_label(env_values)} runtime...")
            self.compose_up(["vllm"])
            active_stage = "warming_model"
            self.wait_for_vllm(model=env_values.get("VLLM_MODEL"), resume_from_stage=resume_from_stage)
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
