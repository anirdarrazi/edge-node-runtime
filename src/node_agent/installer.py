from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass, field
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
    token_matches,
)
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
    logs: list[str] = field(default_factory=list)
    claim: InstallerClaimState | None = None


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
        self.credentials_dir = self.data_dir / "credentials"
        self.credentials_path = self.credentials_dir / "node-credentials.json"
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

    def current_config(
        self,
        *,
        source: dict[str, str] | None = None,
        gpu: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = source or parse_env_file(self.env_path if self.env_path.exists() else self.example_env_path)
        gpu = gpu if gpu is not None else detect_gpu(self.command_runner, self.runtime_dir)
        default_settings = NodeAgentSettings()

        gpu_name = resolve_gpu_name(source.get("GPU_NAME"), gpu.get("name"), default_settings.gpu_name)
        gpu_memory_gb = resolve_gpu_memory(source.get("GPU_MEMORY_GB"), gpu.get("memory_gb"), default_settings.gpu_memory_gb)
        profile = normalize_setup_profile(source.get("SETUP_PROFILE"), gpu.get("memory_gb"))
        recommended_profile = recommended_setup_profile(gpu.get("memory_gb"))
        concurrency = first_nonempty(
            source.get("MAX_CONCURRENT_ASSIGNMENTS"),
            profile_concurrency(profile, gpu.get("memory_gb")),
        )

        return {
            "edge_control_url": first_nonempty(source.get("EDGE_CONTROL_URL"), default_settings.edge_control_url),
            "node_label": suggested_node_label(source.get("NODE_LABEL"), gpu.get("name"), default_settings.node_label),
            "node_region": first_nonempty(source.get("NODE_REGION"), default_settings.node_region),
            "trust_tier": first_nonempty(source.get("TRUST_TIER"), default_settings.trust_tier),
            "restricted_capable": source.get("RESTRICTED_CAPABLE", stringify_bool(default_settings.restricted_capable)).lower() == "true",
            "vllm_model": first_nonempty(source.get("VLLM_MODEL"), DEFAULT_VLLM_MODEL),
            "supported_models": first_nonempty(source.get("SUPPORTED_MODELS"), default_settings.supported_models),
            "max_concurrent_assignments": concurrency,
            "setup_profile": profile,
            "recommended_setup_profile": recommended_profile,
            "profile_summary": profile_summary(profile, gpu_name, concurrency),
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
        }

    def collect_preflight(self, *, gpu: dict[str, Any] | None = None) -> dict[str, Any]:
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
        source = parse_env_file(self.env_path if self.env_path.exists() else self.example_env_path)
        gpu = detect_gpu(self.command_runner, self.runtime_dir)
        config = self.current_config(source=source, gpu=gpu)
        preflight = self.collect_preflight(gpu=gpu)
        state = self.state_payload()
        return {
            "config": config,
            "preflight": preflight,
            "state": state,
            "owner_setup": self.owner_setup_payload(config=config, preflight=preflight, state=state),
            "autostart": self.autostart_manager.status(),
            "desktop_launcher": self.desktop_launcher_manager.status(),
        }

    def owner_setup_payload(
        self,
        *,
        config: dict[str, Any],
        preflight: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        claim = state.get("claim") if isinstance(state.get("claim"), dict) else None
        running_services = preflight.get("running_services", [])
        docker_ready = bool(preflight.get("docker_cli") and preflight.get("docker_compose") and preflight.get("docker_daemon"))
        disk = preflight.get("disk", {})
        disk_ready = bool(isinstance(disk, dict) and disk.get("ok"))
        credentials_present = bool(preflight.get("credentials_present"))
        runtime_running = "node-agent" in running_services

        if state.get("busy"):
            headline = "Finishing setup"
            detail = str(state.get("message") or "Keep this window open while the local runtime finishes setup.")
        elif claim:
            headline = "Approve this node"
            detail = "Open the approval page, sign in, and approve this machine. The installer will finish automatically."
        elif not docker_ready:
            headline = "Start Docker Desktop"
            detail = str(preflight.get("docker_error") or "Docker needs attention before setup can continue.")
        elif not disk_ready:
            headline = "Free up disk space"
            detail = (
                f"This machine has {disk.get('free_gb', 0)} GB free. "
                f"At least {int(disk.get('recommended_free_gb', RECOMMENDED_FREE_DISK_GB))} GB is recommended."
            )
        elif runtime_running:
            headline = "Node is live"
            detail = "The local runtime is running and ready to accept work."
        elif credentials_present:
            headline = "Start the runtime"
            detail = "This node is already approved. Start the runtime to bring it online."
        else:
            headline = "Start Quick Start"
            detail = "Use the recommended profile, approve the node in your browser, and let the runtime finish automatically."

        steps = [
            {
                "key": "machine",
                "label": "Check this machine",
                "status": "complete" if docker_ready and disk_ready else "active",
                "detail": "Docker and disk space are ready." if docker_ready and disk_ready else (preflight.get("blockers") or ["Docker, GPU, or disk still need attention."])[0],
            },
            {
                "key": "approval",
                "label": "Approve this node",
                "status": "complete" if credentials_present else ("active" if claim else "pending"),
                "detail": "Credentials are stored locally." if credentials_present else ("Approval is waiting in the browser." if claim else "Quick Start will open a claim for this machine."),
            },
            {
                "key": "runtime",
                "label": "Start the runtime",
                "status": "complete" if runtime_running else ("active" if credentials_present else "pending"),
                "detail": "The runtime is online." if runtime_running else ("Ready to start now." if credentials_present else "This happens automatically after approval."),
            },
        ]

        return {
            "headline": headline,
            "detail": detail,
            "setup_profile": config.get("setup_profile"),
            "recommended_setup_profile": config.get("recommended_setup_profile"),
            "profile_summary": config.get("profile_summary"),
            "steps": steps,
        }

    def start_install(self, config: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.state.busy:
                return self.status_payload()

            self.state = InstallerState(
                stage="starting",
                busy=True,
                message="Preparing Docker services and local configuration.",
                logs=[],
            )

        self.install_thread = threading.Thread(target=self.run_install, args=(config,), daemon=True)
        self.install_thread.start()
        return self.status_payload()

    def log(self, message: str) -> None:
        with self.lock:
            self.state.logs.append(message)
            self.state.message = message
            self.state.logs = self.state.logs[-80:]

    def set_error(self, message: str) -> None:
        with self.lock:
            self.state.stage = "error"
            self.state.busy = False
            self.state.error = message
            self.state.message = message

    def set_claim(self, claim: InstallerClaimState) -> None:
        with self.lock:
            self.state.claim = claim
            self.state.stage = "waiting_for_claim"
            self.state.error = None

    def complete(self, message: str) -> None:
        with self.lock:
            self.state.stage = "running"
            self.state.busy = False
            self.state.error = None
            self.state.message = message

    def ensure_data_dirs(self) -> None:
        for path in (self.data_dir, self.data_dir / "model-cache", self.data_dir / "scratch", self.credentials_dir):
            path.mkdir(parents=True, exist_ok=True)

    def build_env(self, config: dict[str, Any]) -> dict[str, str]:
        current = parse_env_file(self.env_path if self.env_path.exists() else self.example_env_path)
        defaults = NodeAgentSettings()
        detected_gpu = detect_gpu(self.command_runner, self.runtime_dir)
        profile = normalize_setup_profile(
            str(config.get("setup_profile", "") or current.get("SETUP_PROFILE", "")),
            detected_gpu.get("memory_gb"),
        )
        use_profile_defaults = str(config.get("setup_mode", "")).strip().lower() == "quickstart" or bool(
            str(config.get("setup_profile", "")).strip()
        )

        config_gpu_name = str(config.get("gpu_name", "")).strip() or None
        config_gpu_memory = str(config.get("gpu_memory_gb", "")).strip() or None
        gpu_name = resolve_gpu_name(config_gpu_name or current.get("GPU_NAME"), detected_gpu.get("name"), defaults.gpu_name)
        gpu_memory = resolve_gpu_memory(
            config_gpu_memory or current.get("GPU_MEMORY_GB"),
            detected_gpu.get("memory_gb"),
            defaults.gpu_memory_gb,
        )

        default_concurrency = profile_concurrency(profile, float(gpu_memory) if gpu_memory else None)
        default_batch_tokens = profile_batch_tokens(profile, defaults.max_batch_tokens)
        default_thermal_headroom = profile_thermal_headroom(profile, defaults.thermal_headroom)

        return {
            "SETUP_PROFILE": profile,
            "EDGE_CONTROL_URL": first_nonempty(str(config.get("edge_control_url", "")), current.get("EDGE_CONTROL_URL"), defaults.edge_control_url),
            "OPERATOR_TOKEN": current.get("OPERATOR_TOKEN", ""),
            "NODE_LABEL": suggested_node_label(
                str(config.get("node_label", "")).strip() or current.get("NODE_LABEL"),
                detected_gpu.get("name"),
                defaults.node_label,
            ),
            "NODE_REGION": first_nonempty(str(config.get("node_region", "")), current.get("NODE_REGION"), defaults.node_region),
            "TRUST_TIER": first_nonempty(str(config.get("trust_tier", "")), current.get("TRUST_TIER"), defaults.trust_tier),
            "RESTRICTED_CAPABLE": stringify_bool(bool(config.get("restricted_capable", True))),
            "CREDENTIALS_PATH": "/var/lib/autonomousc/credentials/node-credentials.json",
            "VLLM_BASE_URL": current.get("VLLM_BASE_URL", defaults.vllm_base_url),
            "GPU_NAME": gpu_name,
            "GPU_MEMORY_GB": gpu_memory,
            "MAX_CONTEXT_TOKENS": current.get("MAX_CONTEXT_TOKENS", str(defaults.max_context_tokens)),
            "MAX_BATCH_TOKENS": first_nonempty(
                str(config.get("max_batch_tokens", "")),
                None if use_profile_defaults else current.get("MAX_BATCH_TOKENS"),
                default_batch_tokens,
            ),
            "MAX_CONCURRENT_ASSIGNMENTS": first_nonempty(
                str(config.get("max_concurrent_assignments", "")),
                None if use_profile_defaults else current.get("MAX_CONCURRENT_ASSIGNMENTS"),
                default_concurrency,
            ),
            "THERMAL_HEADROOM": first_nonempty(
                str(config.get("thermal_headroom", "")),
                None if use_profile_defaults else current.get("THERMAL_HEADROOM"),
                default_thermal_headroom,
            ),
            "SUPPORTED_MODELS": first_nonempty(str(config.get("supported_models", "")), current.get("SUPPORTED_MODELS"), defaults.supported_models),
            "POLL_INTERVAL_SECONDS": current.get("POLL_INTERVAL_SECONDS", str(defaults.poll_interval_seconds)),
            "ATTESTATION_PROVIDER": current.get("ATTESTATION_PROVIDER", defaults.attestation_provider),
            "VLLM_MODEL": first_nonempty(str(config.get("vllm_model", "")), current.get("VLLM_MODEL"), DEFAULT_VLLM_MODEL),
        }

    def write_env_file(self, env_values: dict[str, str]) -> None:
        lines = [f"{key}={env_values[key]}" for key in ENV_ORDER if key in env_values]
        self.env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def compose_up(self, services: list[str]) -> None:
        self.command_runner(["docker", "compose", "up", "-d", *services], self.runtime_dir)

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

    def wait_for_vllm(self, timeout_seconds: float = 240.0) -> None:
        vllm_url = f"http://{service_access_host()}:8000/v1/models"
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                response = httpx.get(vllm_url, timeout=5.0)
                if response.status_code < 500:
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
            preflight = self.collect_preflight()
            if not (preflight["docker_cli"] and preflight["docker_compose"] and preflight["docker_daemon"]):
                raise RuntimeError(preflight["docker_error"] or "Docker preflight failed.")

            self.ensure_data_dirs()
            env_values = self.build_env(config)
            self.write_env_file(env_values)
            self.log("Saved node configuration to .env.")

            if self.credentials_path.exists():
                self.log("Existing node credentials were found. Starting the runtime directly.")
                self.compose_up(["vllm", "node-agent", "vector"])
                self.maybe_enable_autostart()
                self.maybe_install_desktop_launcher()
                self.complete("Runtime started with existing node credentials.")
                return

            self.log("Starting the local vLLM runtime...")
            self.compose_up(["vllm"])
            self.wait_for_vllm()
            self.log("vLLM is ready. Creating a browser-assisted node claim...")

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
