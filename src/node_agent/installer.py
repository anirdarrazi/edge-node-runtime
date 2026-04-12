from __future__ import annotations

import argparse
import json
import locale
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

import httpx

from .autostart import AutoStartManager
from .config import NodeAgentSettings
from .control_plane import EdgeControlClient
from .desktop_launcher import DesktopLauncherManager
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
from .runtime_layout import ensure_runtime_bundle, resolve_runtime_dir, service_access_host


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
ControlClientFactory = Callable[[NodeAgentSettings], EdgeControlClient]
SleepFn = Callable[[float], None]


ENV_ORDER = [
    "SETUP_PROFILE",
    "EDGE_CONTROL_URL",
    "OPERATOR_TOKEN",
    "NODE_LABEL",
    "NODE_REGION",
    "TRUST_TIER",
    "RESTRICTED_CAPABLE",
    "CREDENTIALS_PATH",
    "VLLM_BASE_URL",
    "GPU_NAME",
    "GPU_MEMORY_GB",
    "MAX_CONTEXT_TOKENS",
    "MAX_BATCH_TOKENS",
    "MAX_CONCURRENT_ASSIGNMENTS",
    "THERMAL_HEADROOM",
    "SUPPORTED_MODELS",
    "POLL_INTERVAL_SECONDS",
    "ATTESTATION_PROVIDER",
    "VLLM_MODEL",
]

PROFILE_LABELS = {
    "quiet": "Quiet",
    "balanced": "Balanced",
    "performance": "Performance",
}
RECOMMENDED_FREE_DISK_GB = 30.0
DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RUNTIME_SETTINGS_VERSION = 1
INSTALLER_FLOW = [
    ("checking_docker", "Checking Docker"),
    ("checking_gpu", "Checking GPU"),
    ("downloading_runtime", "Downloading runtime"),
    ("warming_model", "Warming model"),
    ("claiming_node", "Claiming node"),
    ("node_live", "Node live"),
]
INSTALLER_FLOW_INDEX = {key: index for index, (key, _label) in enumerate(INSTALLER_FLOW)}
INSTALLER_FLOW_LABELS = {key: label for key, label in INSTALLER_FLOW}
INSTALLER_FLOW_REMAINING_SECONDS = {
    "checking_docker": 330,
    "checking_gpu": 300,
    "downloading_runtime": 270,
    "warming_model": 180,
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


def startup_model_artifact(model: str) -> Any | None:
    for operation in ("responses", "embeddings"):
        artifact = find_model_artifact(model, operation)
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


def recommended_startup_model(memory_gb: float | None) -> str:
    if memory_gb is None:
        return DEFAULT_VLLM_MODEL
    response_artifact = find_model_artifact(DEFAULT_VLLM_MODEL, "responses")
    if response_artifact is None:
        return DEFAULT_VLLM_MODEL
    artifact_size_gb = sum(file.size_bytes for file in response_artifact.model_manifest.files) / (1024**3)
    safe_floor_gb = max(16.0, round(artifact_size_gb + 4.0, 1))
    if memory_gb >= safe_floor_gb:
        return DEFAULT_VLLM_MODEL
    return DEFAULT_EMBEDDING_MODEL


def recommended_supported_models(memory_gb: float | None) -> str:
    if recommended_startup_model(memory_gb) == DEFAULT_EMBEDDING_MODEL:
        return DEFAULT_EMBEDDING_MODEL
    return f"{DEFAULT_VLLM_MODEL},{DEFAULT_EMBEDDING_MODEL}"


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


def first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return ""


def suggest_concurrency(memory_gb: float | None) -> str:
    if memory_gb is None:
        return "2"
    if memory_gb < 16:
        return "1"
    if memory_gb >= 40:
        return "3"
    return "2"


def recommended_setup_profile(memory_gb: float | None) -> str:
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


@dataclass
class InstallerClaimState:
    claim_id: str
    claim_code: str
    approval_url: str
    expires_at: str
    poll_interval_seconds: int
    status: str = "pending"


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


class GuidedInstaller:
    def __init__(
        self,
        runtime_dir: Path | None = None,
        *,
        command_runner: CommandRunner = run_command,
        control_client_factory: ControlClientFactory = EdgeControlClient,
        autostart_manager: AutoStartManager | None = None,
        desktop_launcher_manager: DesktopLauncherManager | None = None,
        sleep: SleepFn = time.sleep,
    ) -> None:
        self.runtime_dir = ensure_runtime_bundle(runtime_dir or resolve_runtime_dir())
        self.data_dir = self.runtime_dir / "data"
        self.service_dir = self.data_dir / "service"
        self.credentials_dir = self.data_dir / "credentials"
        self.credentials_path = self.credentials_dir / "node-credentials.json"
        self.runtime_settings_path = self.service_dir / "runtime-settings.json"
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
        self.sleep = sleep
        self.state = InstallerState()
        self.lock = threading.Lock()
        self.install_thread: threading.Thread | None = None

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
        return {
            "SETUP_PROFILE": str(config.get("setup_profile") or ""),
            "EDGE_CONTROL_URL": first_nonempty(str(config.get("edge_control_url", "")), defaults.edge_control_url),
            "OPERATOR_TOKEN": str(config.get("operator_token") or ""),
            "NODE_LABEL": first_nonempty(str(config.get("node_label", "")), defaults.node_label),
            "NODE_REGION": first_nonempty(str(config.get("node_region", "")), defaults.node_region),
            "TRUST_TIER": first_nonempty(str(config.get("trust_tier", "")), defaults.trust_tier),
            "RESTRICTED_CAPABLE": stringify_bool(bool(config.get("restricted_capable", defaults.restricted_capable))),
            "CREDENTIALS_PATH": first_nonempty(str(config.get("credentials_path", "")), defaults.credentials_path),
            "VLLM_BASE_URL": first_nonempty(str(config.get("vllm_base_url", "")), defaults.vllm_base_url),
            "GPU_NAME": first_nonempty(str(config.get("gpu_name", "")), defaults.gpu_name),
            "GPU_MEMORY_GB": str(_safe_float(config.get("gpu_memory_gb"), defaults.gpu_memory_gb)),
            "MAX_CONTEXT_TOKENS": str(_safe_int(config.get("max_context_tokens"), defaults.max_context_tokens)),
            "MAX_BATCH_TOKENS": str(_safe_int(config.get("max_batch_tokens"), defaults.max_batch_tokens)),
            "MAX_CONCURRENT_ASSIGNMENTS": str(
                _safe_int(config.get("max_concurrent_assignments"), defaults.max_concurrent_assignments)
            ),
            "THERMAL_HEADROOM": str(_safe_float(config.get("thermal_headroom"), defaults.thermal_headroom)),
            "SUPPORTED_MODELS": first_nonempty(str(config.get("supported_models", "")), defaults.supported_models),
            "POLL_INTERVAL_SECONDS": str(_safe_int(config.get("poll_interval_seconds"), defaults.poll_interval_seconds)),
            "ATTESTATION_PROVIDER": first_nonempty(
                str(config.get("attestation_provider", "")),
                defaults.attestation_provider,
            ),
            "VLLM_MODEL": first_nonempty(str(config.get("vllm_model", "")), DEFAULT_VLLM_MODEL),
        }

    def runtime_settings_payload(self, env_values: dict[str, str]) -> dict[str, Any]:
        defaults = NodeAgentSettings()
        return {
            "version": RUNTIME_SETTINGS_VERSION,
            "config": {
                "setup_profile": env_values.get("SETUP_PROFILE") or recommended_setup_profile(defaults.gpu_memory_gb),
                "edge_control_url": env_values.get("EDGE_CONTROL_URL", defaults.edge_control_url),
                "operator_token": env_values.get("OPERATOR_TOKEN", ""),
                "node_label": env_values.get("NODE_LABEL", defaults.node_label),
                "node_region": env_values.get("NODE_REGION", defaults.node_region),
                "trust_tier": env_values.get("TRUST_TIER", defaults.trust_tier),
                "restricted_capable": env_values.get("RESTRICTED_CAPABLE", "").lower() == "true",
                "credentials_path": env_values.get("CREDENTIALS_PATH", defaults.credentials_path),
                "vllm_base_url": env_values.get("VLLM_BASE_URL", defaults.vllm_base_url),
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
        return values

    def effective_runtime_env(self) -> dict[str, str]:
        values = self.load_effective_env()
        if self.env_path.exists():
            values.update(parse_env_file(self.env_path))
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

        gpu_name = resolve_gpu_name(source.get("GPU_NAME"), gpu.get("name"), default_settings.gpu_name)
        gpu_memory_gb = resolve_gpu_memory(source.get("GPU_MEMORY_GB"), gpu.get("memory_gb"), default_settings.gpu_memory_gb)
        numeric_gpu_memory = _safe_float(gpu_memory_gb, default_settings.gpu_memory_gb)
        inferred_startup_model = recommended_startup_model(numeric_gpu_memory)
        inferred_supported_models = recommended_supported_models(numeric_gpu_memory)
        inferred_trust_tier = recommended_trust_tier(attestation_provider)
        inferred_restricted_capable = recommended_restricted_capable(attestation_provider)
        profile = normalize_setup_profile(
            persisted.get("SETUP_PROFILE") or source.get("SETUP_PROFILE"),
            gpu.get("memory_gb"),
        )
        recommended_profile = recommended_setup_profile(gpu.get("memory_gb"))
        concurrency = first_nonempty(
            persisted.get("MAX_CONCURRENT_ASSIGNMENTS"),
            profile_concurrency(profile, gpu.get("memory_gb")),
        )
        thermal_headroom = first_nonempty(
            persisted.get("THERMAL_HEADROOM"),
            profile_thermal_headroom(profile, default_settings.thermal_headroom),
        )
        premium = premium_eligibility(numeric_gpu_memory, attestation_provider)

        return {
            "edge_control_url": first_nonempty(source.get("EDGE_CONTROL_URL"), default_settings.edge_control_url),
            "node_label": suggested_node_label(source.get("NODE_LABEL"), gpu.get("name"), default_settings.node_label),
            "node_region": first_nonempty(persisted.get("NODE_REGION"), region_value),
            "trust_tier": first_nonempty(persisted.get("TRUST_TIER"), inferred_trust_tier),
            "restricted_capable": (
                persisted.get("RESTRICTED_CAPABLE", stringify_bool(inferred_restricted_capable)).lower() == "true"
            ),
            "vllm_model": first_nonempty(persisted.get("VLLM_MODEL"), inferred_startup_model),
            "supported_models": first_nonempty(persisted.get("SUPPORTED_MODELS"), inferred_supported_models),
            "max_concurrent_assignments": concurrency,
            "setup_profile": profile,
            "recommended_setup_profile": recommended_profile,
            "profile_summary": profile_summary(profile, gpu_name, concurrency),
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "thermal_headroom": thermal_headroom,
            "attestation_provider": first_nonempty(persisted.get("ATTESTATION_PROVIDER"), attestation_provider),
            "recommended_node_region": region_value,
            "region_reason": region_reason,
            "recommended_model": inferred_startup_model,
            "recommended_supported_models": inferred_supported_models,
            "recommended_max_concurrent_assignments": profile_concurrency(recommended_profile, gpu.get("memory_gb")),
            "recommended_thermal_headroom": profile_thermal_headroom(
                recommended_profile,
                default_settings.thermal_headroom,
            ),
            "recommended_trust_tier": inferred_trust_tier,
            "recommended_restricted_capable": inferred_restricted_capable,
            "attestation_reason": attestation_reason,
            **premium,
        }

    def collect_preflight(self, *, gpu: dict[str, Any] | None = None) -> dict[str, Any]:
        self.ensure_data_dirs()
        self.sync_runtime_env()
        docker_cli = shutil.which("docker") is not None
        compose_ok = False
        daemon_ok = False
        docker_error: str | None = None

        if docker_cli:
            try:
                self.command_runner(["docker", "compose", "version"], self.runtime_dir)
                compose_ok = True
                self.command_runner(["docker", "info"], self.runtime_dir)
                daemon_ok = True
            except RuntimeError as error:
                docker_error = str(error)
        else:
            docker_error = "Docker is not installed or is not available on PATH."

        running_services: list[str] = []
        if docker_cli and compose_ok and daemon_ok:
            try:
                completed = self.command_runner(
                    ["docker", "compose", "ps", "--services", "--status", "running"],
                    self.runtime_dir,
                )
                running_services = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            except RuntimeError:
                running_services = []

        disk = detect_disk(self.runtime_dir)
        blockers: list[str] = []
        if not docker_cli:
            blockers.append("Install Docker Desktop to continue.")
        elif not daemon_ok:
            blockers.append("Start Docker Desktop so the runtime can launch.")
        if not disk["ok"]:
            blockers.append(
                f"Free up disk space. The runtime works best with at least {int(RECOMMENDED_FREE_DISK_GB)} GB free."
            )

        return {
            "docker_cli": docker_cli,
            "docker_compose": compose_ok,
            "docker_daemon": daemon_ok,
            "docker_error": docker_error,
            "gpu": gpu if gpu is not None else detect_gpu(self.command_runner, self.runtime_dir),
            "disk": disk,
            "running_services": running_services,
            "credentials_present": self.credentials_path.exists(),
            "blockers": blockers,
        }

    def state_payload(self) -> dict[str, Any]:
        with self.lock:
            return asdict(self.state)

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
        concurrency = str(
            config.get("max_concurrent_assignments")
            or config.get("recommended_max_concurrent_assignments")
            or suggest_concurrency(None)
        )
        region = str(config.get("node_region") or config.get("recommended_node_region") or "eu-se-1")
        gpu_name = str(config.get("gpu_name") or gpu.get("name") or "GPU pending")
        gpu_vram = str(config.get("gpu_memory_gb") or "")
        startup_model_bytes = artifact_total_size_bytes(startup_model_artifact(startup_model))
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
                    "Available VRAM is used to choose the startup model and concurrency."
                    if gpu_detected and gpu_vram
                    else "VRAM is unavailable until GPU detection succeeds."
                ),
            },
            {
                "key": "startup_model",
                "label": "Startup model",
                "value": startup_model,
                "detail": override_detail(
                    startup_model,
                    str(config.get("recommended_model") or startup_model),
                    "Chosen automatically from the detected VRAM and bundled runtime models.",
                    override_label="Advanced override is active for the startup model.",
                ),
            },
            {
                "key": "concurrency",
                "label": "Concurrency",
                "value": f"{concurrency} active {'workload' if concurrency == '1' else 'workloads'}",
                "detail": override_detail(
                    concurrency,
                    str(config.get("recommended_max_concurrent_assignments") or concurrency),
                    "Chosen automatically from the detected GPU and thermal profile.",
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
        else:
            current_step = "downloading_runtime"

        def stage_progress_units(step_key: str) -> float:
            if step_key == "downloading_runtime":
                total = _safe_int(stage_context.get("download_total_items"), 0)
                completed = _safe_int(stage_context.get("download_completed_items"), 0)
                if total <= 0:
                    return 0.55
                return max(0.2, min(0.95, completed / total))
            if step_key == "warming_model":
                if bool(stage_context.get("warm_reusing_cache")):
                    return 0.7
                percent = stage_context.get("warm_progress_percent")
                if percent is None:
                    return 0.55
                return max(0.2, min(0.95, int(percent) / 100))
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
            if step_key == "checking_gpu":
                if current_step == step_key:
                    return "active"
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key]:
                    return "complete" if gpu_detected else "warning"
                if not docker_ready or not disk_ready:
                    return "pending"
                return "complete" if gpu_detected else "warning"
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
                    return "Docker Desktop is running, Compose is ready, and disk space looks healthy."
                if not docker_ready:
                    return str(preflight.get("docker_error") or "Docker needs attention before setup can continue.")
                return (
                    f"This machine has {disk.get('free_gb', 0)} GB free. "
                    f"At least {int(disk.get('recommended_free_gb', RECOMMENDED_FREE_DISK_GB))} GB is recommended."
                )
            if step_key == "checking_gpu":
                if gpu_detected:
                    return f"{gpu.get('name') or 'GPU'} detected with {gpu.get('memory_gb')} GB VRAM."
                return "No GPU was detected yet. Quick Start will keep using safe defaults until one is available."
            if step_key == "downloading_runtime":
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key] or credentials_present:
                    return "Local settings are saved and the runtime containers are cached locally."
                if current_step == step_key:
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
                return "Quick Start pre-pulls the runtime containers after the machine checks pass."
            if step_key == "warming_model":
                if INSTALLER_FLOW_INDEX[current_step] > INSTALLER_FLOW_INDEX[step_key] or credentials_present:
                    return "The startup model responded locally and the runtime is warm."
                if current_step == step_key:
                    if bool(stage_context.get("warm_reusing_cache")):
                        reused_bytes = _safe_int(stage_context.get("warm_observed_cache_bytes"), 0)
                        return (
                            f"Reusing {format_bytes(reused_bytes)} already in the local model cache while {startup_model} warms up."
                        )
                    expected_bytes = stage_context.get("warm_expected_bytes")
                    downloaded_bytes = _safe_int(stage_context.get("warm_downloaded_bytes"), 0)
                    if isinstance(expected_bytes, int) and expected_bytes > 0:
                        return (
                            f"Caching {startup_model} locally: {format_bytes(downloaded_bytes)} of about "
                            f"{format_bytes(expected_bytes)} is ready. The first startup can take several minutes."
                        )
                    return (
                        "Starting vLLM and filling the local model cache. "
                        "The first startup can take several minutes."
                    )
                return "The runtime warms the startup model before registration continues."
            if step_key == "claiming_node":
                if credentials_present:
                    return "Local node credentials are stored for this machine."
                if claim:
                    return f"Claim code {claim.get('claim_code', '')} is waiting for approval in your browser."
                if current_step == step_key:
                    return "Creating a secure browser approval flow for this machine."
                return "Quick Start opens a browser approval step here."
            if runtime_running:
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
            detail = "The runtime is online and ready to accept work."
            eta_seconds: int | None = 0
        elif error:
            headline = "Setup needs attention"
            detail = error
            eta_seconds = None
        elif claim and not credentials_present:
            headline = "Approve this node"
            detail = "Approve this machine in your browser and Quick Start will finish automatically."
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
        else:
            headline = "Ready for Quick Start"
            if startup_model_bytes:
                detail = (
                    "Quick Start will move through the checks, pre-pull the runtime, and cache about "
                    f"{format_bytes(startup_model_bytes)} for {startup_model} before the node goes live."
                )
            else:
                detail = (
                    "Quick Start will move through the checks, pre-pull the runtime, warm the model, "
                    "and claim this node automatically."
                )
            eta_seconds = INSTALLER_FLOW_REMAINING_SECONDS["downloading_runtime"]

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
                "About 1 minute after you approve this machine."
                if claim and not credentials_present
                else (
                    "First startup can take several minutes while runtime images are cached locally."
                    if current_step == "downloading_runtime" and busy
                    else (
                        "First startup can take several minutes while the local model cache fills."
                        if current_step == "warming_model" and busy
                        else (
                            "Continue once this machine is ready."
                            if eta_seconds is None and current_step != "claiming_node"
                            else eta_label(eta_seconds)
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
                    else ("Quick Start is running..." if busy else ("Bring this node online" if credentials_present and not runtime_running else "Start Quick Start"))
                )
            ),
        }

    def start_install(self, config: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.state.busy:
                return self.status_payload()

            self.state = InstallerState(
                stage="checking_docker",
                busy=True,
                message="Checking Docker and local prerequisites.",
                logs=[],
                started_at=time.time(),
                stage_started_at=time.time(),
            )

        self.install_thread = threading.Thread(target=self.run_install, args=(config,), daemon=True)
        self.install_thread.start()
        return self.status_payload()

    def log(self, message: str) -> None:
        with self.lock:
            self.state.logs.append(message)
            self.state.message = message
            self.state.logs = self.state.logs[-80:]

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

    def update_stage_progress(self, message: str | None = None, **context: Any) -> None:
        with self.lock:
            if message is not None:
                self.state.message = message
            merged = dict(self.state.stage_context)
            merged.update(context)
            self.state.stage_context = merged

    def set_error(self, message: str) -> None:
        with self.lock:
            current_stage = self.state.stage if self.state.stage in INSTALLER_FLOW_INDEX else "checking_docker"
            self.state.stage = "error"
            self.state.busy = False
            self.state.error = message
            self.state.error_step = current_stage
            self.state.message = message
            self.state.stage_context = {}

    def set_claim(self, claim: InstallerClaimState) -> None:
        with self.lock:
            self.state.claim = claim
            if self.state.started_at is None:
                self.state.started_at = time.time()
            self.state.stage_started_at = time.time()
            self.state.stage = "claiming_node"
            self.state.error = None
            self.state.error_step = None
            self.state.message = "Waiting for browser approval so this machine can be registered."
            self.state.stage_context = {}

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

    def ensure_data_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.service_dir,
            self.data_dir / "model-cache",
            self.data_dir / "scratch",
            self.credentials_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def build_env(self, config: dict[str, Any]) -> dict[str, str]:
        current = self.load_effective_env()
        persisted = self.load_persisted_env()
        defaults = NodeAgentSettings()
        detected_gpu = detect_gpu(self.command_runner, self.runtime_dir)
        inferred_region, _region_reason = infer_node_region()
        inferred_attestation_provider, _attestation_reason = detect_attestation_provider(
            self.command_runner,
            self.runtime_dir,
        )
        profile = normalize_setup_profile(
            str(config.get("setup_profile", "") or persisted.get("SETUP_PROFILE", "")),
            detected_gpu.get("memory_gb"),
        )
        use_profile_defaults = str(config.get("setup_mode", "")).strip().lower() == "quickstart" or bool(
            str(config.get("setup_profile", "")).strip()
        )

        config_gpu_name = str(config.get("gpu_name", "")).strip() or None
        config_gpu_memory = str(config.get("gpu_memory_gb", "")).strip() or None
        gpu_name = resolve_gpu_name(config_gpu_name or current.get("GPU_NAME"), detected_gpu.get("name"), defaults.gpu_name)
        gpu_memory = resolve_gpu_memory(
            config_gpu_memory or persisted.get("GPU_MEMORY_GB") or current.get("GPU_MEMORY_GB"),
            detected_gpu.get("memory_gb"),
            defaults.gpu_memory_gb,
        )
        numeric_gpu_memory = _safe_float(gpu_memory, defaults.gpu_memory_gb)

        default_concurrency = profile_concurrency(profile, numeric_gpu_memory)
        default_batch_tokens = profile_batch_tokens(profile, defaults.max_batch_tokens)
        default_thermal_headroom = profile_thermal_headroom(profile, defaults.thermal_headroom)
        default_startup_model = recommended_startup_model(numeric_gpu_memory)
        default_supported_models = recommended_supported_models(numeric_gpu_memory)
        default_trust_tier = recommended_trust_tier(inferred_attestation_provider)
        default_restricted_capable = recommended_restricted_capable(inferred_attestation_provider)

        return {
            "SETUP_PROFILE": profile,
            "EDGE_CONTROL_URL": first_nonempty(str(config.get("edge_control_url", "")), current.get("EDGE_CONTROL_URL"), defaults.edge_control_url),
            "OPERATOR_TOKEN": current.get("OPERATOR_TOKEN", ""),
            "NODE_LABEL": suggested_node_label(
                str(config.get("node_label", "")).strip() or current.get("NODE_LABEL"),
                detected_gpu.get("name"),
                defaults.node_label,
            ),
            "NODE_REGION": first_nonempty(str(config.get("node_region", "")), persisted.get("NODE_REGION"), inferred_region),
            "TRUST_TIER": first_nonempty(str(config.get("trust_tier", "")), persisted.get("TRUST_TIER"), default_trust_tier),
            "RESTRICTED_CAPABLE": stringify_bool(
                bool(config.get("restricted_capable"))
                if "restricted_capable" in config
                else (persisted.get("RESTRICTED_CAPABLE", stringify_bool(default_restricted_capable)).lower() == "true")
            ),
            "CREDENTIALS_PATH": "/var/lib/autonomousc/credentials/node-credentials.json",
            "VLLM_BASE_URL": current.get("VLLM_BASE_URL", defaults.vllm_base_url),
            "GPU_NAME": gpu_name,
            "GPU_MEMORY_GB": gpu_memory,
            "MAX_CONTEXT_TOKENS": current.get("MAX_CONTEXT_TOKENS", str(defaults.max_context_tokens)),
            "MAX_BATCH_TOKENS": first_nonempty(
                str(config.get("max_batch_tokens", "")),
                None if use_profile_defaults else persisted.get("MAX_BATCH_TOKENS"),
                default_batch_tokens,
            ),
            "MAX_CONCURRENT_ASSIGNMENTS": first_nonempty(
                str(config.get("max_concurrent_assignments", "")),
                None if use_profile_defaults else persisted.get("MAX_CONCURRENT_ASSIGNMENTS"),
                default_concurrency,
            ),
            "THERMAL_HEADROOM": first_nonempty(
                str(config.get("thermal_headroom", "")),
                None if use_profile_defaults else persisted.get("THERMAL_HEADROOM"),
                default_thermal_headroom,
            ),
            "SUPPORTED_MODELS": first_nonempty(
                str(config.get("supported_models", "")),
                persisted.get("SUPPORTED_MODELS"),
                default_supported_models,
            ),
            "POLL_INTERVAL_SECONDS": current.get("POLL_INTERVAL_SECONDS", str(defaults.poll_interval_seconds)),
            "ATTESTATION_PROVIDER": first_nonempty(
                str(config.get("attestation_provider", "")),
                persisted.get("ATTESTATION_PROVIDER"),
                inferred_attestation_provider,
            ),
            "VLLM_MODEL": first_nonempty(
                str(config.get("vllm_model", "")),
                persisted.get("VLLM_MODEL"),
                default_startup_model,
            ),
        }

    def compose_command(self, args: list[str]) -> list[str]:
        self.sync_runtime_env()
        command = ["docker", "compose"]
        if self.runtime_env_path.exists():
            command.extend(["--env-file", str(self.runtime_env_path)])
        if self.release_env_path.exists():
            command.extend(["--env-file", str(self.release_env_path)])
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

    def pull_runtime_images(self, services: list[str]) -> None:
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

    def wait_for_vllm(self, timeout_seconds: float = 240.0, *, model: str | None = None) -> None:
        vllm_url = f"http://{service_access_host()}:8000/v1/models"
        startup_model = (model or "").strip() or self.effective_runtime_env().get("VLLM_MODEL") or DEFAULT_VLLM_MODEL
        startup_artifact = startup_model_artifact(startup_model)
        expected_bytes = artifact_total_size_bytes(startup_artifact)
        cache_dir = self.data_dir / "model-cache"
        baseline_cache_bytes = directory_size_bytes(cache_dir)
        reuse_existing_cache = baseline_cache_bytes > 0
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
            if reuse_existing_cache:
                warm_message = (
                    f"Warming {startup_model}. Reusing the existing local model cache, so this step should finish faster once vLLM is ready."
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
                response = httpx.get(vllm_url, timeout=5.0)
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
        raise RuntimeError("vLLM did not become ready in time.")

    def build_installer_settings(self, env_values: dict[str, str]) -> NodeAgentSettings:
        defaults = NodeAgentSettings()
        return NodeAgentSettings(
            edge_control_url=env_values["EDGE_CONTROL_URL"],
            node_label=env_values["NODE_LABEL"],
            node_region=env_values["NODE_REGION"],
            trust_tier=env_values["TRUST_TIER"],
            restricted_capable=env_values["RESTRICTED_CAPABLE"].lower() == "true",
            credentials_path=str(self.credentials_path),
            vllm_base_url=f"http://{service_access_host()}:8000",
            vllm_model=env_values["VLLM_MODEL"],
            gpu_name=env_values["GPU_NAME"],
            gpu_memory_gb=float(env_values["GPU_MEMORY_GB"]),
            max_context_tokens=int(env_values["MAX_CONTEXT_TOKENS"]),
            max_batch_tokens=int(env_values["MAX_BATCH_TOKENS"]),
            max_concurrent_assignments=int(env_values["MAX_CONCURRENT_ASSIGNMENTS"]),
            thermal_headroom=float(env_values["THERMAL_HEADROOM"]),
            supported_models=env_values["SUPPORTED_MODELS"],
            poll_interval_seconds=int(env_values["POLL_INTERVAL_SECONDS"]),
            agent_version=defaults.agent_version,
            docker_image=defaults.docker_image,
            attestation_provider=env_values["ATTESTATION_PROVIDER"],
        )

    def run_install(self, config: dict[str, Any]) -> None:
        try:
            self.set_stage("checking_docker", "Checking Docker and local prerequisites.")
            preflight = self.collect_preflight()
            if not (preflight["docker_cli"] and preflight["docker_compose"] and preflight["docker_daemon"]):
                raise RuntimeError(preflight["docker_error"] or "Docker preflight failed.")

            self.set_stage("checking_gpu", "Checking GPU availability and choosing a safe runtime profile.")
            self.ensure_data_dirs()
            env_values = self.build_env(config)
            self.write_runtime_settings(env_values)
            self.sync_runtime_env()
            self.log("Saved local node settings and generated the runtime config automatically.")

            if self.credentials_path.exists():
                self.set_stage("node_live", "Starting the local services with the existing node approval.")
                self.log("Existing node credentials were found. Starting the runtime directly.")
                self.compose_up(["vllm", "node-agent", "vector"])
                self.maybe_enable_autostart()
                self.maybe_install_desktop_launcher()
                self.complete("Runtime started with existing node credentials.")
                return

            self.set_stage("downloading_runtime", "Downloading the local runtime and preparing the startup containers.")
            self.pull_runtime_images(["vllm", "node-agent", "vector"])
            self.log("Starting the local vLLM runtime...")
            self.compose_up(["vllm"])
            self.set_stage("warming_model", "Warming the startup model so this machine can answer its first request quickly.")
            self.wait_for_vllm(model=env_values.get("VLLM_MODEL"))
            self.log("vLLM is ready. Creating a browser-assisted node claim...")

            self.set_stage("claiming_node", "Creating a secure browser approval flow for this machine.")
            settings = self.build_installer_settings(env_values)
            control = self.control_client_factory(settings)
            claim = control.create_node_claim_session()
            claim_state = InstallerClaimState(
                claim_id=claim.claim_id,
                claim_code=claim.claim_code,
                approval_url=claim.approval_url,
                expires_at=claim.expires_at,
                poll_interval_seconds=claim.poll_interval_seconds,
            )
            self.set_claim(claim_state)
            self.log("Node claim created. Open the approval page, sign in, and approve this machine.")

            while True:
                result = control.poll_node_claim_session(claim.claim_id, claim.poll_token)
                with self.lock:
                    if self.state.claim:
                        self.state.claim.status = result.status
                        self.state.claim.expires_at = result.expires_at
                if result.node_id and result.node_key:
                    self.log("Claim approved. Persisting node credentials...")
                    control.persist_credentials(result.node_id, result.node_key)
                    break
                if result.status == "expired":
                    raise RuntimeError("The node claim expired before approval completed. Start the installer again to request a fresh claim.")
                self.sleep(max(1, claim.poll_interval_seconds))

            self.set_stage("node_live", "Starting the node agent and bringing this machine online.")
            self.log("Starting node-agent and vector services...")
            self.compose_up(["node-agent", "vector"])
            self.maybe_enable_autostart()
            self.maybe_install_desktop_launcher()
            self.complete("Runtime started. The node agent will attest and begin polling the control plane automatically.")
        except Exception as error:  # pragma: no cover - exercised through tests
            self.log(f"Installer failed: {error}")
            self.set_error(str(error))


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
