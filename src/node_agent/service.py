from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote, unquote, urlparse

import httpx

from .appliance_manifest import (
    inspect_package_signature,
    inspect_runtime_bundle_signature,
    normalize_release_channel,
    release_channel_label,
    release_channel_matches_preference,
)
from .autostart import AutoStartManager
from .config import NodeAgentSettings
from .control_plane import EdgeControlClient, encrypt_artifact
from .control_plane_transport import EdgeControlTransport
from .desktop_launcher import DesktopLauncherManager
from .fault_injection import DEFAULT_FAULT_INJECTION_STATE_NAME, FaultInjectionController
from .heat_governor import (
    build_heat_governor_plan,
    load_heat_governor_state,
    normalize_heat_governor_mode,
    normalize_local_clock,
    normalize_owner_objective,
    write_heat_governor_state,
)
from .installer import (
    CommandRunner,
    GuidedInstaller,
    artifact_total_size_bytes,
    coerce_bool,
    constrain_supported_models_for_runtime_profile,
    detect_disk,
    directory_size_bytes,
    format_bytes,
    llama_cpp_env_for_model,
    nvidia_support_preset,
    optional_env_value,
    parse_env_file,
    profile_concurrency,
    profile_thermal_headroom,
    recommended_setup_profile,
    recommended_supported_models,
    resolve_model_cache_budget,
    resolve_accessible_startup_selection,
    run_command,
    stringify_bool,
    startup_model_artifact,
)
from .inference_engine import AUTO_RUNTIME_PROFILE, resolve_runtime_profile
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
from .release_manifest import (
    RELEASE_ENV_CHANNEL_KEY,
    RELEASE_ENV_VAR_BY_SERVICE,
    RELEASE_ENV_VERSION_KEY,
    ReleaseManifest,
    ReleaseManifestError,
    load_release_manifest,
)
from .runtime_backend import (
    SINGLE_CONTAINER_RUNTIME_BACKEND,
    detect_runtime_backend,
    runtime_backend_label,
    runtime_backend_required_services,
    runtime_backend_supports_compose,
)
from .runtime_layout import ensure_runtime_bundle, resolve_runtime_dir, service_access_host
from .single_container import EmbeddedRuntimeSupervisor

RUNTIME_SERVICES = ("vllm", "node-agent", "vector")
SELF_HEAL_INTERVAL_SECONDS = 45
REMOTE_DASHBOARD_CACHE_TTL_SECONDS = 30
MODEL_CACHE_COLD_SECONDS = 7 * 24 * 60 * 60
MODEL_CACHE_RECENT_SECONDS = 6 * 60 * 60
MODEL_CACHE_HOT_MODEL_LIMIT = 2
VLLM_WARM_RETRY_ATTEMPTS = 2
IDLE_QUEUE_WATCHDOG_SECONDS = 120
IDLE_QUEUE_GPU_UTILIZATION_THRESHOLD_PCT = 10.0
RUNTIME_WEDGE_WATCHDOG_SECONDS = 180
WATCHDOG_RESTART_COOLDOWN_SECONDS = 180
MODEL_WARM_FAILURE_THRESHOLD = 3
MODEL_WARM_RETRY_COOLDOWN_SECONDS = 30 * 60
LOCAL_DOCTOR_INTERVAL_SECONDS = 4 * 60 * 60
LOCAL_DOCTOR_LOOP_SECONDS = 5 * 60
OWNER_TIMELINE_LIMIT = 20
OWNER_TIMELINE_DISPLAY_LIMIT = 6
RUNTIME_TUPLE_ENV_KEYS = (
    "SETUP_PROFILE",
    "RUNTIME_PROFILE",
    "DEPLOYMENT_TARGET",
    "INFERENCE_ENGINE",
    "RUNTIME_IMAGE",
    "INFERENCE_BASE_URL",
    "VLLM_BASE_URL",
    "VLLM_STARTUP_TIMEOUT_SECONDS",
    "VLLM_EXTRA_ARGS",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
    "MAX_CONTEXT_TOKENS",
    "MAX_BATCH_TOKENS",
    "MAX_CONCURRENT_ASSIGNMENTS",
    "THERMAL_HEADROOM",
    "TARGET_GPU_UTILIZATION_PCT",
    "MIN_GPU_MEMORY_HEADROOM_PCT",
    "SUPPORTED_MODELS",
    "VLLM_MODEL",
    "OWNER_TARGET_MODEL",
    "OWNER_TARGET_SUPPORTED_MODELS",
    "LLAMA_CPP_HF_REPO",
    "LLAMA_CPP_HF_FILE",
    "LLAMA_CPP_ALIAS",
    "LLAMA_CPP_EMBEDDING",
    "LLAMA_CPP_POOLING",
)

SENSITIVE_ENV_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "COOKIE", "CERT")
DOCKER_DESKTOP_INSTALL_URL = "https://www.docker.com/products/docker-desktop/"
DOCKER_GPU_SUPPORT_URL = "https://docs.docker.com/desktop/features/gpu/"
NVIDIA_DRIVER_DOWNLOAD_URL = "https://www.nvidia.com/Download/index.aspx"
SUPPORTED_FAULT_DRILLS = (
    "docker_restart_mid_run",
    "dns_flap",
    "disk_almost_full",
    "partial_download_resume",
    "power_loss_cache_write",
    "warm_gpu_oom",
)
DEFAULT_UPDATE_CHANNEL = "stable"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_id() -> str:
    return os.urandom(8).hex()


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and len(parts) < 2:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{secs}s")
    return " ".join(parts[:2])


def format_relative_time(value: str | None) -> str:
    timestamp = parse_iso_timestamp(value)
    if timestamp is None:
        return "Waiting"
    delta_seconds = max(0, int((datetime.now(timezone.utc) - timestamp).total_seconds()))
    if delta_seconds < 10:
        return "Just now"
    if delta_seconds < 60:
        return f"{delta_seconds}s ago"
    if delta_seconds < 3600:
        return f"{delta_seconds // 60}m ago"
    if delta_seconds < 86400:
        return f"{delta_seconds // 3600}h ago"
    return f"{delta_seconds // 86400}d ago"


def format_usd(amount: Any) -> str:
    try:
        numeric = float(amount)
    except (TypeError, ValueError):
        return "$0.00"
    if numeric == 0:
        return "$0.00"
    if abs(numeric) >= 1:
        return f"${numeric:,.2f}"
    return f"${numeric:,.4f}"


def format_watts(value: Any) -> str:
    numeric = coerce_float(value)
    if numeric is None:
        return "unknown"
    if abs(numeric - round(numeric)) < 0.05:
        return f"{int(round(numeric))} W"
    return f"{numeric:.1f} W"


def coerce_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(float(str(value).strip())))
    except (TypeError, ValueError):
        return 0


def coerce_float(value: Any) -> float | None:
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


def update_channel_copy(value: str | None) -> str:
    return release_channel_label(normalize_release_channel(value))


def service_html() -> str:
    return Path(__file__).with_name("service_ui.html").read_text(encoding="utf-8")


def session_bootstrap_html() -> bytes:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='referrer' content='no-referrer'>"
        "<meta http-equiv='cache-control' content='no-store'>"
        "<title>Opening local session...</title></head><body>"
        "<script>history.replaceState(null,'','/');window.location.replace('/');</script>"
        "</body></html>"
    ).encode("utf-8")


class _DrillHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        body: bytes,
        content_type: str,
        status_code: int = 200,
        enable_range: bool = False,
    ) -> None:
        super().__init__(server_address, _DrillHTTPHandler)
        self.body = body
        self.content_type = content_type
        self.status_code = status_code
        self.enable_range = enable_range
        self.requests: list[dict[str, Any]] = []


class _DrillHTTPHandler(BaseHTTPRequestHandler):
    server: _DrillHTTPServer

    def _send_body(self) -> None:
        body = self.server.body
        range_header = self.headers.get("Range")
        start = 0
        end = len(body) - 1
        status_code = self.server.status_code
        content_range: str | None = None
        if self.server.enable_range and range_header:
            raw = range_header.removeprefix("bytes=")
            start_raw, _sep, end_raw = raw.partition("-")
            start = max(0, int(start_raw or 0))
            end = len(body) - 1 if not end_raw else min(len(body) - 1, int(end_raw))
            if start > len(body) - 1:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE.value)
                self.send_header("Content-Range", f"bytes */{len(body)}")
                self.end_headers()
                return
            status_code = HTTPStatus.PARTIAL_CONTENT.value
            content_range = f"bytes {start}-{end}/{len(body)}"
        payload = body[start : end + 1]
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "range": range_header,
                "status_code": status_code,
                "content_length": len(payload),
            }
        )
        self.send_response(status_code)
        self.send_header("content-type", self.server.content_type)
        self.send_header("content-length", str(len(payload)))
        self.send_header("accept-ranges", "bytes")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.end_headers()
        chunk_size = 64 * 1024
        for offset in range(0, len(payload), chunk_size):
            self.wfile.write(payload[offset : offset + chunk_size])
            self.wfile.flush()

    def do_GET(self) -> None:  # pragma: no cover - exercised through live drill helpers
        self._send_body()

    def do_POST(self) -> None:  # pragma: no cover - exercised through live drill helpers
        _ = self.rfile.read(int(self.headers.get("content-length", "0") or 0))
        self._send_body()

    def log_message(self, _format: str, *_args: object) -> None:
        return


@dataclass
class UpdateState:
    auto_update_enabled: bool = False
    interval_hours: int = 24
    preferred_channel: str = DEFAULT_UPDATE_CHANNEL
    last_checked_at: str | None = None
    last_result: str = "No update checks have run yet."
    last_error: str | None = None
    pending_restart: bool = False
    updated_images: list[str] = field(default_factory=list)
    release_version: str | None = None
    release_channel: str | None = None


@dataclass
class DiagnosticsState:
    last_bundle_name: str | None = None
    last_bundle_created_at: str | None = None
    last_bundle_sent_at: str | None = None
    last_case_id: str | None = None
    last_result: str = "No support bundle has been sent yet."
    last_error: str | None = None


@dataclass
class LocalDoctorState:
    last_checked_at: str | None = None
    status: str = "standing_by"
    headline: str = "Local Doctor is standing by"
    detail: str = (
        "Run Local Doctor to re-check Docker, GPU access, network reachability, model cache, warm readiness, "
        "and one tiny local inference."
    )
    last_error: str | None = None
    recommended_fix: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    warm_readiness: dict[str, Any] = field(default_factory=dict)
    inference_probe: dict[str, Any] = field(default_factory=dict)
    attached_bundle_name: str | None = None
    attached_bundle_created_at: str | None = None
    last_trigger: str | None = None
    last_check_mode: str = "manual"
    last_background_check_at: str | None = None
    last_background_status: str | None = None
    last_transition_alert: dict[str, Any] = field(default_factory=dict)
    last_fix_attempt: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfHealState:
    enabled: bool = True
    interval_seconds: int = SELF_HEAL_INTERVAL_SECONDS
    status: str = "standing_by"
    last_checked_at: str | None = None
    last_result: str = "Self-healing is standing by."
    last_error: str | None = None
    last_issue: str | None = None
    last_action: str | None = None
    last_repaired_at: str | None = None
    last_healthy_at: str | None = None
    fix_available: bool = True
    last_known_good_release_env: dict[str, str] | None = None
    last_known_good_runtime_env: dict[str, str] | None = None
    last_known_good_bootstrap_runtime_env: dict[str, str] | None = None
    model_warm_failures: dict[str, int] = field(default_factory=dict)
    model_warm_retry_after: dict[str, str] = field(default_factory=dict)
    model_warm_last_error: dict[str, str] = field(default_factory=dict)
    idle_queue_detected_at: str | None = None
    runtime_wedge_detected_at: str | None = None
    last_targeted_inference_restart_at: str | None = None
    last_targeted_node_restart_at: str | None = None


@dataclass
class RemoteDashboardCacheState:
    fetched_at: str | None = None
    payload: dict[str, Any] | None = None
    last_error: str | None = None


@dataclass
class ModelCacheState:
    enabled: bool = True
    last_checked_at: str | None = None
    last_result: str = "Model cache management is standing by."
    last_error: str | None = None
    likely_model: str | None = None
    likely_model_source: str | None = None
    last_warmed_model: str | None = None
    last_warmed_at: str | None = None
    last_evicted_model: str | None = None
    last_evicted_bytes: int = 0
    last_evicted_at: str | None = None
    cache_bytes: int = 0
    cache_budget_bytes: int = 0
    reserve_free_bytes: int = 0
    cache_budget_source: str = "derived"
    hot_models: list[str] = field(default_factory=list)
    pinned_models: list[str] = field(default_factory=list)
    last_used_by_model: dict[str, str] = field(default_factory=dict)


class NodeRuntimeService:
    def __init__(
        self,
        runtime_dir: Path | None = None,
        *,
        command_runner: CommandRunner = run_command,
        autostart_manager: AutoStartManager | None = None,
        desktop_launcher_manager: DesktopLauncherManager | None = None,
    ) -> None:
        self.runtime_dir = ensure_runtime_bundle(runtime_dir or resolve_runtime_dir())
        self.data_dir = self.runtime_dir / "data"
        self.service_dir = self.data_dir / "service"
        self.diagnostics_dir = self.data_dir / "diagnostics"
        self.scratch_dir = self.data_dir / "scratch"
        self.autopilot_state_path = self.scratch_dir / "autopilot-state.json"
        self.heat_governor_state_path = self.scratch_dir / "heat-governor-state.json"
        self.control_plane_state_path = self.scratch_dir / "control-plane-state.json"
        self.fault_injection_state_path = self.scratch_dir / DEFAULT_FAULT_INJECTION_STATE_NAME
        self.pid_path = self.service_dir / "service.pid"
        self.meta_path = self.service_dir / "service-meta.json"
        self.state_path = self.service_dir / "service-state.json"
        self.release_env_path = self.service_dir / "release.env"
        self.command_runner = command_runner
        self.runtime_backend = detect_runtime_backend()
        self.required_runtime_services = runtime_backend_required_services(self.runtime_backend)
        self.autostart_manager = autostart_manager or AutoStartManager(
            self.runtime_dir,
            command_runner=command_runner,
        )
        self.desktop_launcher_manager = desktop_launcher_manager or DesktopLauncherManager(
            self.runtime_dir,
            command_runner=command_runner,
        )
        self.runtime_controller = None
        self.guided_installer = GuidedInstaller(
            runtime_dir=self.runtime_dir,
            command_runner=command_runner,
            autostart_manager=self.autostart_manager,
            desktop_launcher_manager=self.desktop_launcher_manager,
            runtime_status_provider=self.runtime_backend_status,
            runtime_controller=None,
        )
        self.runtime_backend = self.guided_installer.current_runtime_backend()
        self.required_runtime_services = runtime_backend_required_services(self.runtime_backend)
        if self.runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
            self.runtime_controller = EmbeddedRuntimeSupervisor(
                self.runtime_env_values,
                cache_dir=self.data_dir / "model-cache",
                credentials_dir=self.data_dir / "credentials",
                scratch_dir=self.scratch_dir,
                log=self.log,
            )
        self.guided_installer.runtime_controller = self.runtime_controller
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.repair_lock = threading.Lock()
        self.local_doctor_lock = threading.Lock()
        self.local_doctor_request_lock = threading.Lock()
        self.local_doctor_wakeup = threading.Event()
        self.local_doctor_pending_triggers: list[str] = []
        self.logs: list[str] = []
        self.update_state = UpdateState()
        self.diagnostics_state = DiagnosticsState()
        self.local_doctor_state = LocalDoctorState()
        self.self_heal_state = SelfHealState()
        self.owner_timeline: list[dict[str, Any]] = []
        self.remote_dashboard_state = RemoteDashboardCacheState()
        self.model_cache_state = ModelCacheState()
        self.started_at = now_iso()
        self.started_at_epoch = time.time()
        self.admin_token = generate_admin_token()
        self.sessions = LocalSessionStore()
        self.host = "127.0.0.1"
        self.port = 8765
        self.ensure_dirs()
        self.load_state()

    def resume_setup_if_needed(self) -> None:
        with self.guided_installer.lock:
            should_resume = (
                self.guided_installer.state.resume_requested
                and bool(self.guided_installer.state.resume_config)
                and not self.guided_installer.state.busy
                and self.guided_installer.state.stage not in {"idle", "running", "error"}
            )
        if not should_resume:
            return
        self.log("Resuming interrupted Quick Start setup from the saved local checkpoint.")
        self.guided_installer.resume_if_needed()

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.service_dir, self.diagnostics_dir, self.scratch_dir):
            path.mkdir(parents=True, exist_ok=True)
            tighten_private_path(path, directory=True)

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.update_state = UpdateState(**payload.get("updates", {}))
        self.update_state.preferred_channel = normalize_release_channel(self.update_state.preferred_channel)
        self.diagnostics_state = DiagnosticsState(**payload.get("diagnostics", {}))
        self.local_doctor_state = LocalDoctorState(**payload.get("local_doctor", {}))
        self.self_heal_state = SelfHealState(**payload.get("self_healing", {}))
        raw_owner_timeline = payload.get("owner_timeline")
        if isinstance(raw_owner_timeline, list):
            self.owner_timeline = [dict(item) for item in raw_owner_timeline if isinstance(item, dict)]
        self.model_cache_state = ModelCacheState(**payload.get("model_cache", {}))

    def save_state(self) -> None:
        payload = {
            "updates": asdict(self.update_state),
            "diagnostics": asdict(self.diagnostics_state),
            "local_doctor": asdict(self.local_doctor_state),
            "self_healing": asdict(self.self_heal_state),
            "owner_timeline": self.owner_timeline,
            "model_cache": asdict(self.model_cache_state),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tighten_private_path(self.state_path)

    def log(self, message: str) -> None:
        with self.lock:
            stamped = f"[{now_iso()}] {message}"
            self.logs.append(stamped)
            self.logs = self.logs[-120:]

    def update_self_heal_state(self, **updates: Any) -> None:
        previous_action = self.self_heal_state.last_action
        previous_repaired_at = self.self_heal_state.last_repaired_at
        changed = False
        for key, value in updates.items():
            if getattr(self.self_heal_state, key) != value:
                setattr(self.self_heal_state, key, value)
                changed = True
        if changed:
            self.save_state()
            current_action = self.self_heal_state.last_action
            current_repaired_at = self.self_heal_state.last_repaired_at or self.self_heal_state.last_checked_at
            if current_action != previous_action or current_repaired_at != previous_repaired_at:
                event = self.owner_timeline_event_from_self_heal(
                    action=current_action,
                    issue=self.self_heal_state.last_issue,
                    result=self.self_heal_state.last_result,
                    observed_at=current_repaired_at,
                )
                if event is not None:
                    self.record_owner_timeline_event(
                        code=str(event.get("code") or "self_heal"),
                        title=str(event.get("title") or "Owner update"),
                        detail=str(event.get("detail") or ""),
                        tone=str(event.get("tone") or "warning"),
                        source=str(event.get("source") or "self_healing"),
                        observed_at=str(event.get("observed_at") or current_repaired_at or now_iso()),
                    )

    def update_local_doctor_state(self, **updates: Any) -> None:
        previous_bundle_name = self.local_doctor_state.attached_bundle_name
        previous_bundle_at = self.local_doctor_state.attached_bundle_created_at
        changed = False
        for key, value in updates.items():
            if getattr(self.local_doctor_state, key) != value:
                setattr(self.local_doctor_state, key, value)
                changed = True
        if changed:
            self.save_state()
            if (
                self.local_doctor_state.attached_bundle_name
                and (
                    self.local_doctor_state.attached_bundle_name != previous_bundle_name
                    or self.local_doctor_state.attached_bundle_created_at != previous_bundle_at
                )
            ):
                self.record_owner_timeline_event(
                    code="local_doctor_attached_bundle",
                    title="Local Doctor attached bundle",
                    detail=(
                        f"Attached diagnostics bundle {self.local_doctor_state.attached_bundle_name} so support evidence is ready."
                    ),
                    tone="warning",
                    source="local_doctor",
                    observed_at=self.local_doctor_state.attached_bundle_created_at or self.local_doctor_state.last_checked_at,
                )

    @staticmethod
    def owner_timeline_item(
        *,
        code: str,
        title: str,
        detail: str,
        tone: str = "warning",
        source: str = "service",
        observed_at: str | None = None,
    ) -> dict[str, Any]:
        normalized_tone = tone if tone in {"success", "warning", "danger"} else "warning"
        item: dict[str, Any] = {
            "code": str(code or "event").strip(),
            "title": str(title or "Owner update").strip(),
            "detail": str(detail or "").strip(),
            "tone": normalized_tone,
            "source": str(source or "service").strip(),
            "observed_at": str(observed_at or now_iso()),
        }
        relative = format_relative_time(item["observed_at"])
        if relative != "Waiting":
            item["observed_label"] = relative
        return item

    @staticmethod
    def owner_timeline_sort_key(item: dict[str, Any]) -> tuple[float, str]:
        timestamp = parse_iso_timestamp(str(item.get("observed_at") or ""))
        return (
            timestamp.timestamp() if timestamp is not None else 0.0,
            str(item.get("code") or ""),
        )

    def record_owner_timeline_event(
        self,
        *,
        code: str,
        title: str,
        detail: str,
        tone: str = "warning",
        source: str = "service",
        observed_at: str | None = None,
    ) -> None:
        item = self.owner_timeline_item(
            code=code,
            title=title,
            detail=detail,
            tone=tone,
            source=source,
            observed_at=observed_at,
        )
        dedupe_key = (
            str(item.get("code") or ""),
            str(item.get("observed_at") or ""),
            str(item.get("detail") or ""),
        )
        retained = [
            existing
            for existing in self.owner_timeline
            if (
                str(existing.get("code") or ""),
                str(existing.get("observed_at") or ""),
                str(existing.get("detail") or ""),
            )
            != dedupe_key
        ]
        retained.append(item)
        retained.sort(key=self.owner_timeline_sort_key, reverse=True)
        self.owner_timeline = retained[:OWNER_TIMELINE_LIMIT]
        self.save_state()

    def owner_timeline_event_from_self_heal(
        self,
        *,
        action: str | None,
        issue: str | None,
        result: str | None,
        observed_at: str | None,
    ) -> dict[str, Any] | None:
        normalized_action = str(action or "").strip()
        if not normalized_action or normalized_action in {"monitor", "waiting_for_quick_start"}:
            return None
        issue_text = str(issue or "").strip()
        result_text = str(result or "").strip() or "The node handled a recovery action locally."
        combined = f"{issue_text} {result_text}".lower()
        if normalized_action == "rollback_owner_target_model":
            return self.owner_timeline_item(
                code="target_model_bootstrap_rollback",
                title="Target model rolled back to bootstrap",
                detail=result_text,
                tone="warning",
                source="self_healing",
                observed_at=observed_at,
            )
        if normalized_action == "rollback_bad_update":
            return self.owner_timeline_item(
                code="update_rolled_back",
                title="Update rolled back automatically",
                detail=result_text,
                tone="warning",
                source="self_healing",
                observed_at=observed_at,
            )
        if normalized_action in {"restart_runtime", "restart_vllm", "restart_node_agent"}:
            if "docker" in combined or "container" in combined:
                return self.owner_timeline_item(
                    code="docker_restart_recovered",
                    title="Node recovered automatically after Docker restart",
                    detail=(f"{issue_text} {result_text}").strip(),
                    tone="success",
                    source="self_healing",
                    observed_at=observed_at,
                )
            return self.owner_timeline_item(
                code="node_recovered",
                title="Node recovered automatically",
                detail=result_text,
                tone="success",
                source="self_healing",
                observed_at=observed_at,
            )
        return None

    def owner_timeline_payload(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [dict(item) for item in self.owner_timeline if isinstance(item, dict)]
        synthesized = [
            self.owner_timeline_event_from_self_heal(
                action=self.self_heal_state.last_action,
                issue=self.self_heal_state.last_issue,
                result=self.self_heal_state.last_result,
                observed_at=self.self_heal_state.last_repaired_at or self.self_heal_state.last_checked_at,
            )
        ]
        if self.local_doctor_state.attached_bundle_name:
            synthesized.append(
                self.owner_timeline_item(
                    code="local_doctor_attached_bundle",
                    title="Local Doctor attached bundle",
                    detail=(
                        f"Attached diagnostics bundle {self.local_doctor_state.attached_bundle_name} so support evidence is ready."
                    ),
                    tone="warning",
                    source="local_doctor",
                    observed_at=self.local_doctor_state.attached_bundle_created_at or self.local_doctor_state.last_checked_at,
                )
            )
        update_result = str(self.update_state.last_result or "").strip()
        if update_result and "applied successfully" in update_result.lower():
            synthesized.append(
                self.owner_timeline_item(
                    code="update_applied",
                    title="Update applied successfully",
                    detail=update_result,
                    tone="success",
                    source="updates",
                    observed_at=self.update_state.last_checked_at,
                )
            )
        deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in [*events, *[entry for entry in synthesized if isinstance(entry, dict)]]:
            normalized = self.owner_timeline_item(
                code=str(item.get("code") or "event"),
                title=str(item.get("title") or "Owner update"),
                detail=str(item.get("detail") or ""),
                tone=str(item.get("tone") or "warning"),
                source=str(item.get("source") or "service"),
                observed_at=str(item.get("observed_at") or now_iso()),
            )
            dedupe_key = (
                str(normalized.get("code") or ""),
                str(normalized.get("observed_at") or ""),
                str(normalized.get("detail") or ""),
            )
            deduped[dedupe_key] = normalized
        merged = list(deduped.values())
        merged.sort(key=self.owner_timeline_sort_key, reverse=True)
        return merged[:OWNER_TIMELINE_DISPLAY_LIMIT]

    def runtime_env_values(self) -> dict[str, str]:
        return self.guided_installer.effective_runtime_env()

    def local_doctor_payload(self) -> dict[str, Any]:
        payload = asdict(self.local_doctor_state)
        if not payload.get("headline"):
            payload["headline"] = "Local Doctor is standing by"
        if not payload.get("detail"):
            payload["detail"] = (
                "Run Local Doctor to re-check Docker, GPU access, network reachability, model cache, warm readiness, "
                "and one tiny local inference."
            )
        return payload

    @staticmethod
    def local_doctor_status_rank(status: str | None) -> int:
        normalized = str(status or "").strip().lower()
        if normalized in {"fail", "error", "attention", "blocked"}:
            return 2
        if normalized in {"warn", "warning", "waiting", "active", "unavailable"}:
            return 1
        return 0

    @staticmethod
    def local_doctor_status_tone(status: str | None) -> str:
        normalized = str(status or "").strip().lower()
        if normalized in {"pass", "healthy", "ok"}:
            return "success"
        if normalized in {"warn", "warning", "waiting", "active", "unavailable"}:
            return "warning"
        return "danger"

    @staticmethod
    def local_doctor_status_bucket(status: str | None) -> str:
        normalized = str(status or "").strip().lower()
        if normalized in {"healthy", "pass", "ok"}:
            return "healthy"
        if normalized in {"warning", "warn", "attention", "fail", "error", "blocked"}:
            return "issue"
        return "neutral"

    @staticmethod
    def local_doctor_trigger_label(trigger: str | None) -> str:
        normalized = str(trigger or "").strip().lower()
        if normalized in {"manual", "manual_recheck", "manual_precheck"}:
            return "a manual check"
        if normalized == "auto_fix_verify":
            return "the automatic fix"
        if normalized == "service_start":
            return "service start"
        if normalized == "idle_interval":
            return "an idle-time re-check"
        if normalized == "update":
            return "an update"
        if normalized.startswith("self_heal:"):
            action = normalized.partition(":")[2]
            return {
                "autopilot_tuning": "self-healing retuning the node",
                "restart_runtime": "self-healing restarting the runtime",
                "restart_vllm": "self-healing restarting the inference runtime",
                "restart_node_agent": "self-healing restarting the node agent",
                "rollback_bad_update": "self-healing rolling back an update",
                "rollback_owner_target_model": "self-healing restoring the bootstrap model",
                "repair_startup": "self-healing repairing startup",
                "repair_local_state": "self-healing restoring local state",
            }.get(action, "self-healing")
        return normalized.replace("_", " ") if normalized else "the latest check"

    def local_doctor_background_due(self) -> bool:
        last_checked = parse_iso_timestamp(self.local_doctor_state.last_background_check_at)
        if last_checked is None:
            return True
        return (datetime.now(timezone.utc) - last_checked).total_seconds() >= LOCAL_DOCTOR_INTERVAL_SECONDS

    def local_doctor_can_run_in_background(self) -> bool:
        if self.repair_lock.locked():
            return False
        installer_snapshot = self.guided_installer.status_payload()
        installer_state = installer_snapshot.get("state") if isinstance(installer_snapshot.get("state"), dict) else {}
        if bool(installer_state.get("busy")):
            return False
        autopilot = self.load_autopilot_payload()
        signals = autopilot.get("signals") if isinstance(autopilot.get("signals"), dict) else {}
        if coerce_nonnegative_int(signals.get("queue_depth")) > 0:
            return False
        if coerce_nonnegative_int(signals.get("active_assignments")) > 0:
            return False
        return True

    def queue_background_doctor(self, trigger: str) -> None:
        normalized = str(trigger or "").strip().lower().replace(" ", "_")
        if not normalized or self.shutdown_event.is_set():
            return
        with self.local_doctor_request_lock:
            if normalized not in self.local_doctor_pending_triggers:
                self.local_doctor_pending_triggers.append(normalized)
        self.local_doctor_wakeup.set()

    def pop_background_doctor_trigger(self) -> str | None:
        with self.local_doctor_request_lock:
            if not self.local_doctor_pending_triggers:
                return None
            return self.local_doctor_pending_triggers.pop(0)

    def local_doctor_background_trigger_for_self_heal(self, previous_action: str | None) -> str | None:
        current_action = str(self.self_heal_state.last_action or "").strip()
        if not current_action or current_action == str(previous_action or "").strip():
            return None
        if current_action in {
            "monitor",
            "waiting_for_quick_start",
            "waiting_for_owner",
            "waiting_for_approval",
            "waiting_for_docker",
            "waiting_for_disk_space",
            "waiting_for_gpu",
            "waiting_for_startup_support",
            "self_heal_loop",
            "repair_runtime",
        }:
            return None
        return f"self_heal:{current_action}"

    def local_doctor_transition_alert(
        self,
        *,
        previous_bucket: str | None,
        current_bucket: str,
        status: str,
        headline: str,
        detail: str,
        trigger: str,
        observed_at: str,
    ) -> dict[str, Any]:
        if previous_bucket == current_bucket:
            return {}
        if previous_bucket == "healthy" and current_bucket == "issue":
            return {
                "code": "local_doctor_attention",
                "title": "Local Doctor found something new to fix",
                "detail": f"{headline}. {detail}",
                "tone": "warning" if status == "warning" else "danger",
                "source": "local_doctor",
                "observed_at": observed_at,
            }
        if previous_bucket == "issue" and current_bucket == "healthy":
            trigger_label = self.local_doctor_trigger_label(trigger)
            return {
                "code": "local_doctor_recovered",
                "title": "Local Doctor verified the node recovered",
                "detail": f"{headline}. The latest automatic re-check passed after {trigger_label}.",
                "tone": "success",
                "source": "local_doctor",
                "observed_at": observed_at,
            }
        return {}

    @staticmethod
    def local_doctor_fix_is_actionable(label: str | None, detail: str | None) -> bool:
        normalized = f"{label or ''} {detail or ''}".strip().lower()
        return not (
            "no fix needed" in normalized
            or "no immediate fix needed" in normalized
            or "no immediate fix is needed" in normalized
            or "keep serving on the bootstrap model for now" in normalized
        )

    @staticmethod
    def local_doctor_issue_fix(
        *,
        code: str,
        label: str,
        detail: str,
        source: str,
        automated: bool = False,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "label": label,
            "detail": detail,
            "source": source,
            "automated": automated,
        }

    @staticmethod
    def local_doctor_setup_check_map(preflight: dict[str, Any]) -> dict[str, dict[str, Any]]:
        checks = preflight.get("setup_checks") if isinstance(preflight.get("setup_checks"), list) else []
        result: dict[str, dict[str, Any]] = {}
        for check in checks:
            if not isinstance(check, dict):
                continue
            key = str(check.get("key") or "").strip()
            if key:
                result[key] = check
        return result

    def local_doctor_group_check(
        self,
        *,
        setup_checks: dict[str, dict[str, Any]],
        key: str,
        label: str,
        member_keys: tuple[str, ...],
        success_summary: str,
        success_detail: str,
    ) -> dict[str, Any]:
        selected = [setup_checks[member_key] for member_key in member_keys if member_key in setup_checks]
        if not selected:
            return {
                "key": key,
                "label": label,
                "status": "warn",
                "summary": f"{label} has not been checked yet.",
                "detail": "Run Local Doctor again after local setup data finishes loading.",
                "fix": "Wait for local setup data to load, then run Local Doctor again.",
                "tone": "warning",
                "source_checks": [],
            }

        non_pass = [
            check
            for check in selected
            if self.local_doctor_status_rank(check.get("status")) > 0
        ]
        chosen = non_pass[0] if non_pass else selected[0]
        status = str(chosen.get("status") or "pass")
        summary = str(chosen.get("summary") or success_summary)
        detail = str(chosen.get("detail") or success_detail)
        fix = str(chosen.get("fix") or "No fix needed.")
        if not non_pass:
            status = "pass"
            summary = success_summary
            detail = success_detail
            fix = "No fix needed."
        return {
            "key": key,
            "label": label,
            "status": status,
            "summary": summary,
            "detail": detail,
            "fix": fix,
            "blocking": any(bool(check.get("blocking")) for check in selected),
            "tone": self.local_doctor_status_tone(status),
            "source_checks": [str(check.get("key") or "") for check in selected],
        }

    def fault_controller(self) -> FaultInjectionController:
        return FaultInjectionController(self.fault_injection_state_path)

    def build_fault_drill_settings(self, *, edge_control_url: str | None = None) -> NodeAgentSettings:
        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})
        next_env = dict(env_values)
        next_env["CONTROL_PLANE_STATE_PATH"] = str(self.control_plane_state_path)
        next_env["FAULT_INJECTION_STATE_PATH"] = str(self.fault_injection_state_path)
        if edge_control_url:
            next_env["EDGE_CONTROL_URL"] = edge_control_url
        settings = self.guided_installer.build_installer_settings(next_env)
        settings.control_plane_state_path = str(self.control_plane_state_path)
        settings.fault_injection_state_path = str(self.fault_injection_state_path)
        if edge_control_url:
            settings.edge_control_url = edge_control_url
        return settings

    @contextmanager
    def local_drill_server(
        self,
        *,
        body: bytes,
        content_type: str,
        status_code: int = 200,
        enable_range: bool = False,
    ):
        server = _DrillHTTPServer(
            ("127.0.0.1", 0),
            body=body,
            content_type=content_type,
            status_code=status_code,
            enable_range=enable_range,
        )
        thread = threading.Thread(target=server.serve_forever, name="fault-drill-http", daemon=True)
        thread.start()
        host, port = server.server_address[:2]
        try:
            yield server, f"http://{host}:{port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def fault_drill_payload(self) -> dict[str, Any]:
        snapshot = self.fault_controller().snapshot()
        return {
            "state_path": str(self.fault_injection_state_path),
            "supported_scenarios": list(SUPPORTED_FAULT_DRILLS),
            "active_faults": snapshot.get("active_faults", []),
            "last_drill": snapshot.get("last_drill", {}),
        }

    def apply_fault_overrides_to_installer_snapshot(self, installer_snapshot: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(installer_snapshot, dict):
            return installer_snapshot
        disk_pressure = self.fault_controller().peek("disk_pressure")
        if not disk_pressure:
            return installer_snapshot
        snapshot = json.loads(json.dumps(installer_snapshot))
        preflight = snapshot.get("preflight") if isinstance(snapshot.get("preflight"), dict) else {}
        disk = dict(preflight.get("disk") or {})
        free_bytes = max(
            0,
            int(
                (disk_pressure.get("metadata") or {}).get("free_bytes")
                or 1 * (1024**3)
            ),
        )
        disk.update(
            {
                "free_bytes": free_bytes,
                "free_gb": round(free_bytes / float(1024**3), 1),
                "ok": False,
                "injected_fault": "disk_pressure",
            }
        )
        preflight["disk"] = disk
        snapshot["preflight"] = preflight
        return snapshot

    def runtime_backend_status(self) -> dict[str, Any]:
        if self.runtime_controller is None:
            return {}
        return self.runtime_controller.snapshot()

    def compose_command(self, args: list[str]) -> list[str]:
        if not runtime_backend_supports_compose(self.runtime_backend):
            raise RuntimeError("Docker Compose is unavailable in the unified in-container runtime mode.")
        self.guided_installer.sync_runtime_env()
        command = ["docker", "compose"]
        if self.release_env_path.exists():
            command.extend(["--env-file", str(self.release_env_path)])
        if self.guided_installer.runtime_env_path.exists():
            command.extend(["--env-file", str(self.guided_installer.runtime_env_path)])
        command.extend(args)
        return command

    def compose(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return self.command_runner(self.compose_command(args), self.runtime_dir)

    def signed_release_manifest(self) -> ReleaseManifest:
        manifest = load_release_manifest()
        self.update_state.release_version = manifest.version
        self.update_state.release_channel = normalize_release_channel(manifest.channel)
        return manifest

    def appliance_release_payload(self) -> dict[str, Any]:
        package_signature = inspect_package_signature()
        runtime_signature_root = self.runtime_dir if (self.runtime_dir / "appliance-runtime-manifest.json").exists() else None
        runtime_bundle_signature = inspect_runtime_bundle_signature(runtime_signature_root)
        release_version = self.update_state.release_version
        release_channel = self.update_state.release_channel
        if not release_version:
            try:
                manifest = load_release_manifest()
                release_version = manifest.version
                release_channel = manifest.channel
            except ReleaseManifestError:
                release_version = package_signature.get("version")
                release_channel = package_signature.get("channel")
        preferred_channel = normalize_release_channel(self.update_state.preferred_channel)
        release_verified = bool(package_signature.get("verified")) and bool(runtime_bundle_signature.get("verified"))
        if release_verified:
            detail = (
                f"Signed appliance release {release_version or 'pending'} is verified locally. "
                f"Update track: {update_channel_copy(preferred_channel)}."
            )
        else:
            detail = str(package_signature.get("detail") or runtime_bundle_signature.get("detail") or "Signed appliance verification needs attention.")
        return {
            "verified": release_verified,
            "detail": detail,
            "version": release_version,
            "channel": normalize_release_channel(release_channel),
            "channel_label": update_channel_copy(release_channel),
            "preferred_channel": preferred_channel,
            "preferred_channel_label": update_channel_copy(preferred_channel),
            "package_signature": package_signature,
            "runtime_bundle_signature": runtime_bundle_signature,
        }

    def current_release_env(self, manifest: ReleaseManifest) -> dict[str, str]:
        values = parse_env_file(self.release_env_path)
        return {
            RELEASE_ENV_VERSION_KEY: values.get(RELEASE_ENV_VERSION_KEY, manifest.version),
            RELEASE_ENV_CHANNEL_KEY: values.get(RELEASE_ENV_CHANNEL_KEY, manifest.channel),
            **{
                env_key: values.get(env_key, manifest.images[service].ref)
                for service, env_key in RELEASE_ENV_VAR_BY_SERVICE.items()
            },
        }

    def write_release_env(self, values: dict[str, str]) -> None:
        lines = [
            f"{RELEASE_ENV_VERSION_KEY}={values[RELEASE_ENV_VERSION_KEY]}",
            f"{RELEASE_ENV_CHANNEL_KEY}={values[RELEASE_ENV_CHANNEL_KEY]}",
        ]
        for service in ("node-agent", "vllm", "vector"):
            env_key = RELEASE_ENV_VAR_BY_SERVICE[service]
            lines.append(f"{env_key}={values[env_key]}")
        self.release_env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def restore_release_env(self, previous_content: str | None) -> None:
        if previous_content is None:
            if self.release_env_path.exists():
                self.release_env_path.unlink()
            return
        self.release_env_path.write_text(previous_content, encoding="utf-8")

    def current_release_snapshot(self) -> dict[str, str] | None:
        try:
            manifest = self.signed_release_manifest()
            return self.current_release_env(manifest)
        except ReleaseManifestError:
            values = parse_env_file(self.release_env_path)
            return values or None

    def runtime_tuple_snapshot(self, env_values: dict[str, str] | None = None) -> dict[str, str] | None:
        values = env_values or self.runtime_env_values()
        if not values:
            return None
        snapshot: dict[str, str] = {}
        for key in RUNTIME_TUPLE_ENV_KEYS:
            raw_value = values.get(key)
            if raw_value is None:
                continue
            value = str(raw_value).strip()
            if value:
                snapshot[key] = value
        model = snapshot.get("VLLM_MODEL")
        if model:
            for key, value in llama_cpp_env_for_model(model).items():
                normalized = str(value).strip()
                if normalized:
                    snapshot[key] = normalized
        return snapshot or None

    def remember_known_good_release(self) -> None:
        snapshot = self.current_release_snapshot()
        if not snapshot:
            return
        self.update_self_heal_state(
            last_known_good_release_env=snapshot,
            last_healthy_at=now_iso(),
        )

    def remember_known_good_runtime_tuple(self) -> None:
        snapshot = self.runtime_tuple_snapshot()
        if not snapshot:
            return
        updates: dict[str, Any] = {
            "last_known_good_runtime_env": snapshot,
            "last_healthy_at": now_iso(),
        }
        current_model = str(snapshot.get("VLLM_MODEL") or "").strip()
        owner_target_model = str(snapshot.get("OWNER_TARGET_MODEL") or "").strip()
        if current_model and (not owner_target_model or current_model != owner_target_model):
            updates["last_known_good_bootstrap_runtime_env"] = snapshot
        self.update_self_heal_state(**updates)

    def remember_known_good_runtime_state(self) -> None:
        self.remember_known_good_release()
        self.remember_known_good_runtime_tuple()

    def apply_runtime_tuple_snapshot(self, snapshot: dict[str, str]) -> bool:
        if not snapshot:
            return False
        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})
        next_env = dict(env_values)
        changed = False
        for key in RUNTIME_TUPLE_ENV_KEYS:
            if key not in snapshot:
                continue
            value = str(snapshot[key]).strip()
            if next_env.get(key) != value:
                next_env[key] = value
                changed = True
        model = str(snapshot.get("VLLM_MODEL") or next_env.get("VLLM_MODEL") or "").strip()
        if model:
            for key, value in llama_cpp_env_for_model(model).items():
                normalized = str(value).strip()
                if next_env.get(key) != normalized:
                    next_env[key] = normalized
                    changed = True
        if not changed:
            return False
        self.guided_installer.write_runtime_settings(next_env)
        self.guided_installer.write_runtime_env(next_env)
        return True

    def restore_known_good_runtime_tuple(self, failure_reason: str, *, prefer_bootstrap: bool = False) -> bool:
        target = (
            self.self_heal_state.last_known_good_bootstrap_runtime_env
            if prefer_bootstrap
            else self.self_heal_state.last_known_good_runtime_env
        )
        if prefer_bootstrap and not target:
            target = self.self_heal_state.last_known_good_runtime_env
        if not target:
            return False
        current = self.runtime_tuple_snapshot()
        if current == target:
            return False
        rollback_id = request_id()
        target_model = str(target.get("VLLM_MODEL") or "the last known healthy startup model")
        self.log(
            f"[request {rollback_id}] Restoring the last known healthy runtime tuple "
            f"({target_model}) after: {failure_reason}"
        )
        return self.apply_runtime_tuple_snapshot(target)

    def clear_model_warm_failure(self, model: str | None) -> None:
        normalized_model = (model or "").strip()
        if not normalized_model:
            return
        failures = dict(self.self_heal_state.model_warm_failures)
        retry_after = dict(self.self_heal_state.model_warm_retry_after)
        last_error = dict(self.self_heal_state.model_warm_last_error)
        changed = False
        for mapping in (failures, retry_after, last_error):
            if normalized_model in mapping:
                mapping.pop(normalized_model, None)
                changed = True
        if changed:
            self.update_self_heal_state(
                model_warm_failures=failures,
                model_warm_retry_after=retry_after,
                model_warm_last_error=last_error,
            )

    def model_warm_retry_hold(self, model: str | None) -> str | None:
        normalized_model = (model or "").strip()
        if not normalized_model:
            return None
        retry_after = parse_iso_timestamp(self.self_heal_state.model_warm_retry_after.get(normalized_model))
        if retry_after is None:
            return None
        if retry_after <= datetime.now(timezone.utc):
            self.clear_model_warm_failure(normalized_model)
            return None
        return (
            f"{normalized_model} recently failed to warm repeatedly, so self-healing is keeping the last known-good "
            f"bootstrap model active until {retry_after.isoformat()}."
        )

    def record_model_warm_failure(self, model: str | None, error: Exception | str) -> tuple[int, str | None]:
        normalized_model = (model or "").strip()
        if not normalized_model:
            return 0, None
        failures = dict(self.self_heal_state.model_warm_failures)
        retry_after = dict(self.self_heal_state.model_warm_retry_after)
        last_error = dict(self.self_heal_state.model_warm_last_error)
        count = max(0, int(failures.get(normalized_model, 0) or 0)) + 1
        failures[normalized_model] = count
        last_error[normalized_model] = str(error) or "Model warm-up failed."
        cooldown_until: str | None = None
        if count >= MODEL_WARM_FAILURE_THRESHOLD:
            cooldown_until = (datetime.now(timezone.utc) + timedelta(seconds=MODEL_WARM_RETRY_COOLDOWN_SECONDS)).isoformat()
            retry_after[normalized_model] = cooldown_until
        self.update_self_heal_state(
            model_warm_failures=failures,
            model_warm_retry_after=retry_after,
            model_warm_last_error=last_error,
        )
        return count, cooldown_until

    def watchdog_timer_elapsed(self, field_name: str, *, active: bool, now: datetime | None = None) -> float | None:
        current_time = now or datetime.now(timezone.utc)
        started_raw = getattr(self.self_heal_state, field_name)
        started_at = parse_iso_timestamp(started_raw if isinstance(started_raw, str) else None)
        if active:
            if started_at is None:
                self.update_self_heal_state(**{field_name: current_time.isoformat()})
                return 0.0
            return max(0.0, (current_time - started_at).total_seconds())
        if started_raw is not None:
            self.update_self_heal_state(**{field_name: None})
        return None

    def watchdog_restart_ready(self, field_name: str, *, now: datetime | None = None) -> bool:
        current_time = now or datetime.now(timezone.utc)
        last_restart_raw = getattr(self.self_heal_state, field_name)
        last_restart = parse_iso_timestamp(last_restart_raw if isinstance(last_restart_raw, str) else None)
        if last_restart is None:
            return True
        return (current_time - last_restart).total_seconds() >= WATCHDOG_RESTART_COOLDOWN_SECONDS

    def rollback_to_known_good_release(self, failure_reason: str) -> bool:
        target = self.self_heal_state.last_known_good_release_env
        if not target:
            return False
        current = self.current_release_snapshot()
        if current == target:
            return False

        previous_content = self.release_env_path.read_text(encoding="utf-8") if self.release_env_path.exists() else None
        rollback_id = request_id()
        self.log(
            f"[request {rollback_id}] Rolling back to the last known healthy signed release after: {failure_reason}"
        )
        try:
            self.write_release_env(target)
            self.start_runtime_services(recreate=True)
            self.wait_for_runtime_health()
        except Exception as error:
            self.restore_release_env(previous_content)
            self.log(
                f"[request {rollback_id}] Rollback to the last known healthy signed release failed: {error}"
            )
            return False

        self.update_state.pending_restart = False
        self.update_state.last_error = None
        self.update_state.last_result = "Self-healing rolled back to the last known healthy signed release."
        self.update_self_heal_state(
            status="healthy",
            last_result="Self-healing rolled back to the last known healthy signed release.",
            last_error=None,
            last_issue=failure_reason,
            last_action="rollback_bad_update",
            last_repaired_at=now_iso(),
            fix_available=False,
        )
        self.remember_known_good_runtime_state()
        self.save_state()
        return True

    def wait_for_runtime_health(self, timeout_seconds: float = 90.0) -> None:
        if self.runtime_controller is not None:
            self.runtime_controller.wait_for_runtime_health(timeout_seconds)
            return
        env_values = self.runtime_env_values()
        runtime_label = self.guided_installer.inference_runtime_label(env_values)
        readiness_path = self.guided_installer.inference_readiness_path(env_values)
        readiness_url = f"http://{service_access_host()}:8000{readiness_path}"
        deadline = time.time() + timeout_seconds
        required_services = set(self.required_runtime_services)
        last_failure = "Runtime services are still starting."
        while time.time() < deadline:
            try:
                completed = self.compose(["ps", "--services", "--status", "running"])
                running = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
                if not required_services.issubset(running):
                    missing = ", ".join(sorted(required_services - running))
                    last_failure = f"Runtime services are not healthy yet. Missing: {missing or 'unknown'}."
                    time.sleep(2)
                    continue
                response = httpx.get(readiness_url, timeout=4.0)
                if response.status_code < 500:
                    return
                last_failure = f"{runtime_label} health check returned HTTP {response.status_code}."
            except (RuntimeError, httpx.HTTPError) as error:
                last_failure = str(error) or "Runtime health check failed."
            time.sleep(2)
        raise RuntimeError(last_failure)

    def runtime_status(
        self,
        *,
        preflight: dict[str, Any] | None = None,
        installer_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preflight = preflight or self.guided_installer.collect_preflight()
        env_values = self.guided_installer.effective_runtime_env()
        config_present = self.guided_installer.config_present()
        running_services = preflight.get("running_services", [])
        inference_service_running = "vllm" in running_services or "inference-runtime" in running_services
        runtime_profile = self.guided_installer.resolved_runtime_profile(env_values)
        inference_engine = self.guided_installer.resolved_inference_engine(env_values)
        inference_engine_label = self.guided_installer.inference_runtime_label(env_values)
        deployment_target = self.guided_installer.resolved_deployment_target(env_values)
        readiness_path = self.guided_installer.inference_readiness_path(env_values)
        readiness_url = f"http://{service_access_host()}:8000{readiness_path}"
        inference_ready = False
        current_model = str(env_values.get("VLLM_MODEL") or "").strip() or None
        if inference_service_running:
            try:
                response = httpx.get(readiness_url, timeout=4.0)
                inference_ready = response.status_code < 500
                if response.status_code < 500:
                    model_response = response
                    if readiness_path != "/v1/models":
                        try:
                            candidate = httpx.get(f"http://{service_access_host()}:8000/v1/models", timeout=4.0)
                            if candidate.status_code < 500:
                                model_response = candidate
                        except httpx.HTTPError:
                            pass
                    try:
                        payload = model_response.json()
                    except ValueError:
                        payload = None
                    if isinstance(payload, dict):
                        data = payload.get("data")
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
                                    current_model = item["id"].strip()
                                    break
            except httpx.HTTPError:
                inference_ready = False

        installer_state = installer_state or self.guided_installer.state_payload()
        if installer_state.get("busy"):
            stage = installer_state.get("stage", "installing")
            message = installer_state.get("message", "Installer is running.")
        elif "node-agent" in running_services:
            stage = "running"
            message = "The runtime service is online."
        elif preflight.get("credentials_present"):
            stage = "ready"
            message = "Credentials are present. Start the runtime when ready."
        elif config_present and env_values:
            stage = "registration_required"
            message = "Node configuration is present, but this machine still needs approval."
        else:
            stage = "setup_required"
            message = "Run the guided setup from the local UI."

        return {
            "stage": stage,
            "message": message,
            "preflight": preflight,
            "running_services": running_services,
            "inference_runtime_ready": inference_ready,
            "vllm_ready": inference_ready,
            "inference_ready": inference_ready,
            "inference_engine": inference_engine,
            "inference_engine_label": inference_engine_label,
            "deployment_target": deployment_target,
            "runtime_profile": runtime_profile.id,
            "runtime_profile_label": runtime_profile.label,
            "model_format": runtime_profile.model_format,
            "runtime_image": env_values.get("RUNTIME_IMAGE") or env_values.get("VLLM_IMAGE") or runtime_profile.image,
            "inference_base_url": env_values.get("INFERENCE_BASE_URL")
            or env_values.get("VLLM_BASE_URL")
            or f"http://{service_access_host()}:8000",
            "readiness_path": runtime_profile.readiness_path,
            "supported_apis": list(runtime_profile.supported_apis),
            "trust_policy": runtime_profile.trust_policy,
            "pricing_tier": runtime_profile.pricing_tier,
            "artifact_manifest_type": runtime_profile.artifact_manifest_type,
            "capacity_class": runtime_profile.capacity_class,
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
            "burst_cost_ceiling_usd": runtime_profile.burst_cost_ceiling_usd,
            "current_model": current_model,
            "config_present": config_present,
            "credentials_present": preflight.get("credentials_present", False),
            "runtime_backend": preflight.get("runtime_backend", self.runtime_backend),
            "runtime_backend_label": preflight.get("runtime_backend_label", runtime_backend_label(self.runtime_backend)),
        }

    def load_autopilot_payload(self) -> dict[str, Any]:
        if not self.autopilot_state_path.exists():
            return {}
        try:
            payload = json.loads(self.autopilot_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def heat_governor_payload(self, autopilot_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})
        env_values["HEAT_GOVERNOR_STATE_PATH"] = self.guided_installer.runtime_heat_governor_state_env_path(
            self.runtime_backend
        )
        settings = self.guided_installer.build_installer_settings(env_values)
        settings.heat_governor_state_path = str(self.heat_governor_state_path)
        state = load_heat_governor_state(self.heat_governor_state_path)
        autopilot_payload = autopilot_payload or self.load_autopilot_payload()
        recommendation = autopilot_payload.get("recommendation")
        signals = autopilot_payload.get("signals") if isinstance(autopilot_payload.get("signals"), dict) else {}
        live_plan = (
            recommendation.get("heat_governor")
            if isinstance(recommendation, dict) and isinstance(recommendation.get("heat_governor"), dict)
            else None
        )
        fallback_plan = build_heat_governor_plan(
            settings,
            state=state,
            gpu_temp_c=coerce_float(signals.get("gpu_temp_c")),
            gpu_utilization_pct=coerce_float(signals.get("gpu_utilization_pct")),
            queue_depth=coerce_nonnegative_int(signals.get("queue_depth")),
            active_assignments=coerce_nonnegative_int(signals.get("active_assignments")),
        ).payload()
        plan = dict(fallback_plan)
        if live_plan:
            plan.update(live_plan)
        policy_parts = [str(plan.get("owner_objective_label") or "Balanced")]
        quiet_start = str(plan.get("quiet_hours_start_local") or "").strip()
        quiet_end = str(plan.get("quiet_hours_end_local") or "").strip()
        if quiet_start and quiet_end:
            policy_parts.append(f"quiet hours {quiet_start}-{quiet_end}")
        power_cap = coerce_nonnegative_int(plan.get("max_power_cap_watts"))
        if power_cap > 0:
            policy_parts.append(f"power cap {power_cap} W")
        energy_price = coerce_float(state.get("energy_price_kwh", env_values.get("ENERGY_PRICE_KWH")))
        if energy_price is not None:
            policy_parts.append(f"${energy_price:.2f}/kWh power")
        return {
            "mode": normalize_heat_governor_mode(state.get("mode", env_values.get("HEAT_GOVERNOR_MODE"))),
            "state_path": str(self.heat_governor_state_path),
            "plan": plan,
            "alerts": plan.get("owner_alerts", []) if isinstance(plan.get("owner_alerts"), list) else [],
            "room_temp_c": state.get("room_temp_c", env_values.get("ROOM_TEMP_C") or None),
            "target_temp_c": state.get("target_temp_c", env_values.get("TARGET_TEMP_C") or None),
            "outside_temp_c": state.get("outside_temp_c", env_values.get("OUTSIDE_TEMP_C") or None),
            "owner_objective": normalize_owner_objective(
                state.get("owner_objective", env_values.get("OWNER_OBJECTIVE") or getattr(settings, "owner_objective", "balanced"))
            ),
            "quiet_hours_start_local": state.get(
                "quiet_hours_start_local",
                env_values.get("QUIET_HOURS_START_LOCAL") or None,
            ),
            "quiet_hours_end_local": state.get(
                "quiet_hours_end_local",
                env_values.get("QUIET_HOURS_END_LOCAL") or None,
            ),
            "gpu_temp_limit_c": state.get("gpu_temp_limit_c", env_values.get("GPU_TEMP_LIMIT_C") or settings.gpu_temp_limit_c),
            "gpu_power_limit_enabled": coerce_bool(
                state.get("gpu_power_limit_enabled", env_values.get("GPU_POWER_LIMIT_ENABLED")),
                settings.gpu_power_limit_enabled,
            ),
            "max_power_cap_watts": state.get(
                "max_power_cap_watts",
                env_values.get("MAX_POWER_CAP_WATTS") or None,
            ),
            "energy_price_kwh": state.get("energy_price_kwh", env_values.get("ENERGY_PRICE_KWH") or None),
            "policy_summary": ", ".join(policy_parts),
            "updated_at": state.get("updated_at"),
            "last_observed_at": signals.get("last_observed_at") if isinstance(signals, dict) else None,
        }

    def configure_heat_governor(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_dirs()
        self.guided_installer.ensure_data_dirs()
        mode = normalize_heat_governor_mode(payload.get("mode"))
        updates: dict[str, Any] = {"mode": mode}
        for key in ("room_temp_c", "target_temp_c", "outside_temp_c", "gpu_temp_limit_c"):
            if key in payload:
                updates[key] = coerce_float(payload.get(key))
        if "owner_objective" in payload:
            updates["owner_objective"] = normalize_owner_objective(payload.get("owner_objective"))
        for key in ("quiet_hours_start_local", "quiet_hours_end_local"):
            if key in payload:
                updates[key] = normalize_local_clock(payload.get(key))
        if "gpu_power_limit_enabled" in payload:
            updates["gpu_power_limit_enabled"] = bool(payload.get("gpu_power_limit_enabled"))
        if "max_power_cap_watts" in payload:
            max_power_cap = coerce_float(payload.get("max_power_cap_watts"))
            updates["max_power_cap_watts"] = None if max_power_cap is None or max_power_cap <= 0 else int(round(max_power_cap))
        if "energy_price_kwh" in payload:
            updates["energy_price_kwh"] = coerce_float(payload.get("energy_price_kwh"))
        state = write_heat_governor_state(self.heat_governor_state_path, updates)

        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})
        env_values["HEAT_GOVERNOR_STATE_PATH"] = self.guided_installer.runtime_heat_governor_state_env_path(
            self.runtime_backend
        )
        env_values["HEAT_GOVERNOR_MODE"] = mode
        env_values["ROOM_TEMP_C"] = optional_env_value(state.get("room_temp_c"))
        env_values["TARGET_TEMP_C"] = optional_env_value(state.get("target_temp_c"))
        env_values["OUTSIDE_TEMP_C"] = optional_env_value(state.get("outside_temp_c"))
        env_values["OWNER_OBJECTIVE"] = str(state.get("owner_objective") or "balanced")
        env_values["QUIET_HOURS_START_LOCAL"] = str(state.get("quiet_hours_start_local") or "")
        env_values["QUIET_HOURS_END_LOCAL"] = str(state.get("quiet_hours_end_local") or "")
        env_values["GPU_TEMP_LIMIT_C"] = optional_env_value(state.get("gpu_temp_limit_c"))
        env_values["GPU_POWER_LIMIT_ENABLED"] = stringify_bool(
            bool(state.get("gpu_power_limit_enabled", True))
        )
        env_values["MAX_POWER_CAP_WATTS"] = optional_env_value(state.get("max_power_cap_watts"))
        env_values["ENERGY_PRICE_KWH"] = optional_env_value(state.get("energy_price_kwh"))
        self.guided_installer.write_runtime_settings(env_values)
        self.guided_installer.write_runtime_env(env_values)
        self.log(f"Heat governor set to {mode}.")
        return self.status_payload()

    def startup_health_payload(
        self,
        *,
        autostart: dict[str, Any],
        desktop_launcher: dict[str, Any],
        config_present: bool,
        credentials_present: bool,
    ) -> dict[str, Any] | None:
        platform_name = str(
            getattr(
                self.autostart_manager,
                "system_name",
                getattr(self.autostart_manager, "platform_name", os.name),
            )
        )
        if not (credentials_present or config_present):
            return None

        autostart_supported = bool(autostart.get("supported"))
        autostart_enabled = bool(autostart.get("enabled"))
        launcher_supported = bool(desktop_launcher.get("supported"))
        launcher_enabled = bool(desktop_launcher.get("enabled"))
        startup_label = {
            "nt": "Windows sign-in launch",
            "win32": "Windows sign-in launch",
            "darwin": "macOS login launch",
            "linux": "Linux user service",
            "posix": "Launch-on-sign-in",
        }.get(platform_name, "Launch-on-sign-in")

        if autostart_supported and autostart_enabled:
            return None

        if not autostart_supported:
            detail = str(
                autostart.get("detail")
                or f"{startup_label} is unavailable on this machine right now."
            )
            return {
                "issue_code": "startup_unavailable",
                "issue_detail": detail,
                "automatic_fix_available": False,
                "manual_fix_available": True,
                "issue_action_label": "Check startup",
            }

        detail = (
            f"{startup_label} is disabled or missing for this node. "
            "Self-healing can reinstall it automatically so the node comes back after you sign in."
        )
        if launcher_supported and not launcher_enabled:
            detail += " The desktop launcher can be repaired at the same time."
        return {
            "issue_code": "startup_not_configured",
            "issue_detail": detail,
            "automatic_fix_available": True,
            "manual_fix_available": True,
            "issue_action_label": "Fix startup",
        }

    def prerequisite_action_payload(self, issue_code: str | None, issue_detail: str | None) -> dict[str, Any] | None:
        if not issue_code:
            return None
        detail = issue_detail or "This machine has a prerequisite that needs attention before setup can continue."
        if issue_code == "docker_not_running":
            return {
                "code": "start_docker_desktop",
                "label": "Start Docker",
                "detail": "Open Docker Desktop and re-check the engine before continuing setup.",
                "automatic": False,
            }
        if issue_code == "docker_unavailable":
            return {
                "code": "open_docker_install",
                "label": "Install Docker",
                "detail": "Open the Docker Desktop installer page, then return here after Docker is installed and running.",
                "automatic": False,
            }
        if issue_code == "nvidia_runtime_missing":
            return {
                "code": "open_gpu_runtime_help",
                "label": "Fix GPU runtime",
                "detail": "Open Docker Desktop GPU support guidance so NVIDIA containers can see the local GPU.",
                "automatic": False,
            }
        if issue_code == "gpu_missing":
            return {
                "code": "open_nvidia_driver_help",
                "label": "Install GPU driver",
                "detail": "Open NVIDIA driver setup, then re-run the local check after the GPU is visible.",
                "automatic": False,
            }
        if issue_code == "disk_low":
            return {
                "code": "free_disk_space",
                "label": "Free space",
                "detail": "Prune unused Docker data and cold model cache, then re-check available disk space.",
                "automatic": False,
            }
        if issue_code == "startup_not_configured":
            return {
                "code": "repair_startup",
                "label": "Fix startup",
                "detail": "Repair launch-on-sign-in and the desktop launcher so the node comes back after sign-in.",
                "automatic": True,
            }
        if issue_code == "startup_unavailable":
            return {
                "code": "open_startup_help",
                "label": "Check startup",
                "detail": detail,
                "automatic": False,
            }
        return None

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

    def open_support_url(self, url: str, fallback_message: str) -> bool:
        try:
            return bool(webbrowser.open(url))
        except Exception as error:
            self.log(f"{fallback_message}: {error}")
            return False

    def start_docker_desktop(self) -> dict[str, Any]:
        launcher = self.docker_desktop_launcher()
        if launcher is not None:
            subprocess.Popen([str(launcher)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {
                "handled": True,
                "resolved": False,
                "action": "start_docker_desktop",
                "message": "Docker Desktop is opening. Keep this app open while Docker finishes starting, then Quick Start will re-check it automatically.",
            }

        opened = self.open_support_url(DOCKER_DESKTOP_INSTALL_URL, "Docker Desktop install page could not be opened")
        return {
            "handled": True,
            "resolved": False,
            "action": "open_docker_install",
            "message": (
                "Docker Desktop was not found locally, so the install page was opened."
                if opened
                else "Docker Desktop was not found locally. Install Docker Desktop, start it, then return here."
            ),
        }

    def free_prerequisite_disk_space(self, health: dict[str, Any]) -> dict[str, Any]:
        messages: list[str] = []
        installer_snapshot = (
            health.get("installer_snapshot")
            if isinstance(health.get("installer_snapshot"), dict)
            else self.guided_installer.status_payload()
        )
        cache_changed, cache_result, _cache_action = self.manage_model_cache(
            health=health,
            installer_snapshot=installer_snapshot,
        )
        if cache_changed:
            messages.append(cache_result)

        preflight = health.get("preflight") if isinstance(health.get("preflight"), dict) else {}
        docker_ready = bool(
            preflight.get("docker_cli")
            and preflight.get("docker_compose")
            and preflight.get("docker_daemon")
        )
        if docker_ready and runtime_backend_supports_compose(str(health.get("runtime_backend") or self.runtime_backend)):
            try:
                completed = self.command_runner(["docker", "system", "prune", "-f"], self.runtime_dir)
                prune_detail = completed.stdout.strip() or "Docker pruned unused containers, networks, images, and build cache."
                messages.append(prune_detail)
            except Exception as error:
                messages.append(f"Docker prune could not finish automatically: {error}")

        if not messages:
            messages.append(
                "No safe unused Docker data or cold model cache was available to prune automatically. Free space manually, then re-check setup."
            )

        return {
            "handled": True,
            "resolved": False,
            "action": "free_disk_space",
            "message": " ".join(messages),
        }

    def repair_prerequisite_blocker(self, health: dict[str, Any]) -> dict[str, Any] | None:
        action = health.get("prerequisite_action") if isinstance(health.get("prerequisite_action"), dict) else None
        if not action:
            return None
        code = str(action.get("code") or "")
        if code == "start_docker_desktop":
            return self.start_docker_desktop()
        if code == "open_docker_install":
            opened = self.open_support_url(DOCKER_DESKTOP_INSTALL_URL, "Docker Desktop install page could not be opened")
            return {
                "handled": True,
                "resolved": False,
                "action": code,
                "message": (
                    "Opened the Docker Desktop install page. Install Docker, start it, then return here."
                    if opened
                    else "Install Docker Desktop, start it, then return here."
                ),
            }
        if code == "open_gpu_runtime_help":
            opened = self.open_support_url(DOCKER_GPU_SUPPORT_URL, "Docker GPU support guidance could not be opened")
            return {
                "handled": True,
                "resolved": False,
                "action": code,
                "message": (
                    "Opened Docker Desktop GPU support guidance. Enable NVIDIA GPU support, restart Docker, then re-check setup."
                    if opened
                    else "Enable NVIDIA GPU support in Docker Desktop, restart Docker, then re-check setup."
                ),
            }
        if code == "open_nvidia_driver_help":
            opened = self.open_support_url(NVIDIA_DRIVER_DOWNLOAD_URL, "NVIDIA driver page could not be opened")
            return {
                "handled": True,
                "resolved": False,
                "action": code,
                "message": (
                    "Opened the NVIDIA driver page. Install or update the driver, reboot if prompted, then re-check setup."
                    if opened
                    else "Install or update the NVIDIA driver, reboot if prompted, then re-check setup."
                ),
            }
        if code == "free_disk_space":
            return self.free_prerequisite_disk_space(health)
        if code == "repair_startup":
            autostart = self.autostart_manager.ensure_enabled()
            launcher = self.desktop_launcher_manager.ensure_enabled()
            return {
                "handled": True,
                "resolved": bool(autostart.get("enabled")) or bool(launcher.get("enabled")),
                "action": code,
                "message": str(
                    autostart.get("detail")
                    or launcher.get("detail")
                    or "Launch-on-sign-in and the desktop launcher were re-checked."
                ),
            }
        if code == "open_startup_help":
            return {
                "handled": True,
                "resolved": False,
                "action": code,
                "message": str(action.get("detail") or "Automatic startup needs manual attention on this machine."),
            }
        return None

    def nvidia_recovery_plan(
        self,
        *,
        config: dict[str, Any],
        gpu: dict[str, Any],
        runtime: dict[str, Any],
    ) -> dict[str, Any] | None:
        memory_gb = coerce_float(gpu.get("memory_gb"))
        if memory_gb is None:
            memory_gb = coerce_float(config.get("gpu_memory_gb"))
        gpu_name = str(gpu.get("name") or config.get("gpu_name") or "this NVIDIA GPU").strip()
        preset = nvidia_support_preset(memory_gb, gpu_name)
        if preset is None:
            return None

        current_model = str(
            runtime.get("current_model")
            or config.get("vllm_model")
            or ""
        ).strip()
        if not current_model or current_model in set(preset.supported_models):
            return None

        desired_profile = recommended_setup_profile(memory_gb, gpu_name)
        desired_concurrency = profile_concurrency(desired_profile, memory_gb, gpu_name)
        desired_supported_models = recommended_supported_models(memory_gb, gpu_name)
        desired_thermal_headroom = str(
            profile_thermal_headroom(desired_profile, NodeAgentSettings().thermal_headroom)
        )
        return {
            "issue_code": "startup_model_too_large",
            "issue_detail": (
                f"{current_model} is outside the supported {preset.label} preset for {gpu_name}. "
                f"Self-healing can switch this machine to {preset.startup_model}, reset the model list, "
                "and retry the NVIDIA warm-up automatically."
            ),
            "target_model": preset.startup_model,
            "target_supported_models": desired_supported_models,
            "target_profile": desired_profile,
            "target_concurrency": desired_concurrency,
            "target_thermal_headroom": desired_thermal_headroom,
            "target_max_context_tokens": (
                str(preset.max_context_tokens)
                if preset.max_context_tokens is not None
                else str(NodeAgentSettings().max_context_tokens)
            ),
            "target_vllm_startup_timeout_seconds": (
                str(preset.vllm_startup_timeout_seconds)
                if preset.vllm_startup_timeout_seconds is not None
                else str(NodeAgentSettings().vllm_startup_timeout_seconds)
            ),
            "target_vllm_extra_args": preset.vllm_extra_args or NodeAgentSettings().vllm_extra_args,
            "target_vllm_memory_profiler_estimate_cudagraphs": preset.runtime_env_defaults().get(
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
                "false",
            ),
            "target_preset_label": preset.label,
            "target_capacity_label": preset.capacity_label,
        }

    def runtime_health_snapshot(self, *, installer_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        installer_snapshot = installer_snapshot or self.guided_installer.status_payload()
        installer_snapshot = self.apply_fault_overrides_to_installer_snapshot(installer_snapshot)
        preflight = installer_snapshot["preflight"]
        installer_state = installer_snapshot["state"]
        runtime = self.runtime_status(preflight=preflight, installer_state=installer_state)
        running_services = set(preflight.get("running_services", []))
        docker_ready = bool(preflight.get("docker_cli") and preflight.get("docker_compose") and preflight.get("docker_daemon"))
        runtime_backend = str(preflight.get("runtime_backend") or self.runtime_backend)
        config_present = bool(runtime.get("config_present"))
        credentials_present = bool(runtime.get("credentials_present"))
        credentials_file_present = self.guided_installer.credentials_path.exists()
        disk = preflight.get("disk") if isinstance(preflight.get("disk"), dict) else {}
        gpu = preflight.get("gpu") if isinstance(preflight.get("gpu"), dict) else {}
        nvidia_container_runtime = (
            preflight.get("nvidia_container_runtime")
            if isinstance(preflight.get("nvidia_container_runtime"), dict)
            else {}
        )
        autostart = (
            installer_snapshot.get("autostart")
            if isinstance(installer_snapshot.get("autostart"), dict)
            else self.autostart_manager.status()
        )
        desktop_launcher = (
            installer_snapshot.get("desktop_launcher")
            if isinstance(installer_snapshot.get("desktop_launcher"), dict)
            else self.desktop_launcher_manager.status()
        )
        required_running = set(self.required_runtime_services).issubset(running_services)
        inference_ready = bool(runtime.get("inference_ready", runtime.get("vllm_ready")))
        runtime_healthy = bool(
            credentials_present
            and runtime.get("stage") == "running"
            and required_running
            and inference_ready
        )

        issue_code: str | None = None
        issue_detail: str | None = None
        automatic_fix_available = False
        manual_fix_available = False
        issue_action_label = "Fix it"
        startup_issue = self.startup_health_payload(
            autostart=autostart if isinstance(autostart, dict) else {},
            desktop_launcher=desktop_launcher if isinstance(desktop_launcher, dict) else {},
            config_present=config_present,
            credentials_present=credentials_present,
        )
        nvidia_recovery = self.nvidia_recovery_plan(
            config=installer_snapshot.get("config", {}) if isinstance(installer_snapshot.get("config"), dict) else {},
            gpu=gpu,
            runtime=runtime,
        )

        if installer_state.get("busy"):
            issue_code = "installer_busy"
            issue_detail = "Quick Start is already running, so self-healing is waiting for it to finish first."
        elif (
            not config_present
            or not self.guided_installer.runtime_settings_path.exists()
            or not self.guided_installer.runtime_env_path.exists()
        ):
            issue_code = "missing_config"
            issue_detail = "Local runtime settings are missing or incomplete on this machine."
            automatic_fix_available = True
            manual_fix_available = True
        elif disk and not bool(disk.get("ok", True)):
            free_gb = coerce_nonnegative_int(disk.get("free_gb"))
            recommended_free_gb = coerce_nonnegative_int(disk.get("recommended_free_gb"))
            issue_code = "disk_low"
            issue_detail = (
                f"Only {free_gb} GB is free on this machine. Free up at least {recommended_free_gb} GB so "
                "runtime images, model cache, and repairs can finish safely."
            )
            manual_fix_available = True
            issue_action_label = "Check storage"
        elif runtime_backend_supports_compose(runtime_backend) and not docker_ready:
            if bool(preflight.get("docker_cli")) and bool(preflight.get("docker_compose")) and not bool(preflight.get("docker_daemon")):
                issue_code = "docker_not_running"
                issue_detail = str(
                    preflight.get("docker_error")
                    or "Docker Desktop is installed, but the Docker engine is not running yet. "
                    "Start Docker Desktop and wait for the engine to be ready."
                )
            else:
                issue_code = "docker_unavailable"
                issue_detail = str(
                    preflight.get("docker_error")
                    or "Docker needs attention before the runtime can be repaired."
            )
            manual_fix_available = True
            issue_action_label = "Check Docker"
        elif docker_ready and bool(gpu.get("detected")) and nvidia_container_runtime.get("visible") is False:
            issue_code = "nvidia_runtime_missing"
            issue_detail = str(
                nvidia_container_runtime.get("error")
                or "The NVIDIA GPU is visible on the host, but Docker cannot expose it to containers yet."
            )
            manual_fix_available = True
            issue_action_label = "Fix GPU runtime"
        elif not bool(gpu.get("detected")):
            issue_code = "gpu_missing"
            issue_detail = (
                "No compatible NVIDIA GPU was detected. Check that the GPU is visible to Windows, "
                "NVIDIA drivers are installed, and Docker Desktop GPU support is enabled."
            )
            manual_fix_available = True
            issue_action_label = "Check GPU"
        elif nvidia_recovery is not None:
            issue_code = str(nvidia_recovery["issue_code"])
            issue_detail = str(nvidia_recovery["issue_detail"])
            automatic_fix_available = True
            manual_fix_available = True
            issue_action_label = "Use safer NVIDIA preset"
        elif credentials_present and not running_services:
            issue_code = "runtime_stopped"
            issue_detail = "The runtime is stopped even though this machine already has stored node credentials."
            automatic_fix_available = True
            manual_fix_available = True
        elif credentials_present and not required_running:
            missing = ", ".join(sorted(set(self.required_runtime_services) - running_services)) or "unknown services"
            issue_code = "stuck_containers"
            if runtime_backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
                issue_detail = f"Some in-container runtime services look stuck or missing: {missing}."
            else:
                issue_detail = f"Some runtime containers look stuck or missing: {missing}."
            automatic_fix_available = True
            manual_fix_available = True
        elif credentials_present and not inference_ready:
            issue_code = "runtime_unhealthy"
            issue_detail = (
                f"The runtime containers are up, but {runtime.get('inference_engine_label') or 'the local inference runtime'} is not responding yet."
            )
            automatic_fix_available = True
            manual_fix_available = True
        elif startup_issue is not None:
            issue_code = str(startup_issue["issue_code"])
            issue_detail = str(startup_issue["issue_detail"])
            automatic_fix_available = bool(startup_issue["automatic_fix_available"])
            manual_fix_available = bool(startup_issue["manual_fix_available"])
            issue_action_label = str(startup_issue.get("issue_action_label") or "Fix it")
        elif not credentials_present and config_present:
            issue_code = "approval_required"
            issue_detail = "This machine still needs approval before the runtime can come online."
            manual_fix_available = True
            issue_action_label = "Resume setup"

        prerequisite_action = self.prerequisite_action_payload(issue_code, issue_detail)
        if prerequisite_action is not None:
            issue_action_label = str(prerequisite_action.get("label") or issue_action_label)

        return {
            "installer_snapshot": installer_snapshot,
            "preflight": preflight,
            "installer_state": installer_state,
            "runtime": runtime,
            "issue_code": issue_code,
            "issue_detail": issue_detail,
            "issue_action_label": issue_action_label,
            "runtime_healthy": runtime_healthy,
            "docker_ready": docker_ready,
            "runtime_backend": runtime_backend,
            "runtime_backend_label": runtime_backend_label(runtime_backend),
            "credentials_present": credentials_present,
            "credentials_file_present": credentials_file_present,
            "config_present": config_present,
            "autostart": autostart,
            "desktop_launcher": desktop_launcher,
            "running_services": sorted(running_services),
            "automatic_fix_available": automatic_fix_available,
            "manual_fix_available": manual_fix_available,
            "nvidia_recovery_plan": nvidia_recovery,
            "prerequisite_action": prerequisite_action,
        }

    def self_heal_payload(self, health: dict[str, Any]) -> dict[str, Any]:
        repairing = self.repair_lock.locked()
        if repairing:
            status = "repairing"
            headline = "Self-healing is fixing this machine"
            detail = self.self_heal_state.last_result or "Self-healing is applying a repair now."
        elif health.get("issue_code"):
            if health.get("manual_fix_available"):
                status = "attention"
                headline = "Prerequisite-healing found the next fix"
            else:
                status = "waiting"
                headline = "Self-healing is standing by"
            detail = str(
                health.get("issue_detail")
                or self.self_heal_state.last_result
                or "Self-healing is monitoring this machine."
            )
        elif health.get("runtime_healthy"):
            status = "healthy"
            headline = "Self-healing is active"
            detail = self.self_heal_state.last_result or "The runtime looks healthy and self-healing is monitoring it."
        else:
            status = "standing_by"
            headline = "Self-healing is standing by"
            detail = self.self_heal_state.last_result or "Self-healing is monitoring this machine."

        return {
            **asdict(self.self_heal_state),
            "status": status,
            "headline": headline,
            "detail": detail,
            "current_issue": health.get("issue_detail"),
            "issue_code": health.get("issue_code"),
            "manual_fix_available": bool(health.get("manual_fix_available")),
            "automatic_fix_available": bool(health.get("automatic_fix_available")),
            "action_label": str(health.get("issue_action_label") or "Fix it"),
            "prerequisite_action": (
                health.get("prerequisite_action")
                if isinstance(health.get("prerequisite_action"), dict)
                else None
            ),
        }

    def local_doctor_warm_readiness(
        self,
        *,
        installer_snapshot: dict[str, Any],
        health: dict[str, Any],
    ) -> dict[str, Any]:
        env_values = self.runtime_env_values()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})
        runtime = health.get("runtime") if isinstance(health.get("runtime"), dict) else {}
        installer_state = (
            installer_snapshot.get("state")
            if isinstance(installer_snapshot.get("state"), dict)
            else {}
        )
        stage_context = (
            installer_state.get("stage_context")
            if isinstance(installer_state.get("stage_context"), dict)
            else {}
        )
        current_model = str(
            runtime.get("current_model")
            or env_values.get("VLLM_MODEL")
            or ""
        ).strip()
        owner_target_model = str(env_values.get("OWNER_TARGET_MODEL") or "").strip()
        diagnostics = self.guided_installer.startup_model_warmup_diagnostics(
            env_values,
            model=current_model or None,
        )
        runtime_label = str(diagnostics.get("runtime_label") or runtime.get("inference_engine_label") or "local inference runtime")
        current_stage = str(installer_state.get("stage") or "").strip()
        warm_failure_detail = str(stage_context.get("warm_failure_detail") or "").strip()
        warm_failure_kind = str(stage_context.get("warm_failure_kind") or "").strip()
        warm_excerpt = str(stage_context.get("warm_runtime_log_excerpt") or "").strip()
        current_failure = self.self_heal_state.model_warm_last_error.get(owner_target_model or current_model or "")
        hold_message = self.model_warm_retry_hold(owner_target_model or current_model or None)

        if diagnostics.get("error"):
            detail = str(diagnostics.get("error"))
            return {
                "status": "fail",
                "summary": "This machine cannot warm the configured startup model with the current local runtime tuple.",
                "detail": detail,
                "tone": "danger",
                "recommended_fix": self.local_doctor_issue_fix(
                    code=str(diagnostics.get("error_kind") or "warmup_preflight"),
                    label="Adjust the startup model",
                    detail=detail,
                    source="warmup_preflight",
                ),
            }
        if warm_failure_kind and warm_failure_detail:
            detail = warm_failure_detail
            if warm_excerpt:
                detail = f"{detail} Recent runtime logs: {warm_excerpt}"
            return {
                "status": "fail",
                "summary": "The most recent model warm-up failed on this machine.",
                "detail": detail,
                "tone": "danger",
                "recommended_fix": self.local_doctor_issue_fix(
                    code=warm_failure_kind,
                    label="Retry the warm-up on a safer tuple",
                    detail=detail,
                    source="warmup_runtime_logs",
                ),
            }
        if current_stage in {"downloading_model", "warming_model"} and bool(installer_state.get("busy")):
            return {
                "status": "warn",
                "summary": "Quick Start is still preparing the startup model.",
                "detail": str(installer_state.get("message") or f"{current_model or 'The startup model'} is still warming."),
                "tone": "warning",
                "recommended_fix": self.local_doctor_issue_fix(
                    code="wait_for_warmup",
                    label="Wait for the current warm-up",
                    detail="Quick Start is already filling cache or warming the model. Let it finish, then run Local Doctor again if the runtime still looks stuck.",
                    source="installer_stage",
                ),
            }
        if hold_message:
            detail = hold_message
            if current_failure:
                detail = f"{detail} Latest error: {current_failure}"
            return {
                "status": "warn",
                "summary": "The larger owner target model is paused while the bootstrap model keeps serving.",
                "detail": detail,
                "tone": "warning",
                "recommended_fix": self.local_doctor_issue_fix(
                    code="owner_target_retry_hold",
                    label="Keep serving on the bootstrap model for now",
                    detail=detail,
                    source="owner_target_warm_hold",
                ),
            }
        if owner_target_model and current_model and owner_target_model != current_model:
            return {
                "status": "warn",
                "summary": "The bootstrap model is serving while the owner target model warms in the background.",
                "detail": (
                    f"{current_model} is ready now. {owner_target_model} will switch in after its background warm-up finishes."
                ),
                "tone": "warning",
                "recommended_fix": self.local_doctor_issue_fix(
                    code="bootstrap_target_warming",
                    label="No immediate fix needed",
                    detail="The node is already serving safely on the bootstrap model while it prepares the larger owner target in the background.",
                    source="background_warmup",
                ),
            }
        if bool(runtime.get("inference_ready")):
            return {
                "status": "pass",
                "summary": f"{current_model or 'The startup model'} is warm and ready for local inference.",
                "detail": f"{runtime_label} answered readiness checks successfully on this machine.",
                "tone": "success",
                "recommended_fix": self.local_doctor_issue_fix(
                    code="none",
                    label="No fix needed",
                    detail="The startup model is warm and ready locally.",
                    source="warm_readiness",
                ),
            }
        return {
            "status": "fail",
            "summary": "The local runtime is not warm enough to serve inference yet.",
            "detail": str(
                health.get("issue_detail")
                or runtime.get("message")
                or "The local inference runtime is not responding yet."
            ),
            "tone": "danger",
            "recommended_fix": self.local_doctor_issue_fix(
                code="repair_runtime",
                label="Run prerequisite-healing",
                detail="The runtime did not pass the warm-readiness check. Use Fix machine once, then run Local Doctor again.",
                source="warm_readiness",
                automated=True,
            ),
        }

    def probe_local_inference(self, runtime: dict[str, Any]) -> dict[str, Any]:
        current_model = str(runtime.get("current_model") or self.runtime_env_values().get("VLLM_MODEL") or "").strip()
        supported_apis = [
            str(api).strip()
            for api in runtime.get("supported_apis", [])
            if str(api).strip()
        ]
        if runtime.get("stage") != "running" or not bool(runtime.get("inference_ready")):
            return {
                "status": "fail",
                "summary": "A tiny local inference could not run because the runtime is not ready yet.",
                "detail": str(runtime.get("message") or "Start the runtime and wait for the model to finish warming."),
                "tone": "danger",
                "attempted": False,
                "recommended_fix": self.local_doctor_issue_fix(
                    code="repair_runtime",
                    label="Run prerequisite-healing",
                    detail="The local runtime is not ready enough to accept a tiny test request yet.",
                    source="inference_probe",
                    automated=True,
                ),
            }

        base_url = f"http://{service_access_host()}:8000"
        timeout = 10.0
        try:
            if "responses" in supported_apis:
                response = httpx.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "model": current_model,
                        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                output_text = ""
                choices = payload.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(message, dict):
                        output_text = str(message.get("content") or "").strip()
                detail = (
                    f"The local responses path answered on {current_model or 'the startup model'}."
                    + (f" Sample reply: {output_text}" if output_text else "")
                )
                return {
                    "status": "pass",
                    "summary": "A tiny local responses probe completed successfully.",
                    "detail": detail,
                    "tone": "success",
                    "attempted": True,
                    "api": "responses",
                    "path": "/v1/chat/completions",
                }
            if "embeddings" in supported_apis:
                response = httpx.post(
                    f"{base_url}/v1/embeddings",
                    json={
                        "model": current_model,
                        "input": "local doctor probe",
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                embedding_count = len(payload.get("data", [])) if isinstance(payload.get("data"), list) else 0
                return {
                    "status": "pass",
                    "summary": "A tiny local embeddings probe completed successfully.",
                    "detail": (
                        f"The local embeddings path answered on {current_model or 'the startup model'} "
                        f"with {embedding_count} embedding result{'s' if embedding_count != 1 else ''}."
                    ),
                    "tone": "success",
                    "attempted": True,
                    "api": "embeddings",
                    "path": "/v1/embeddings",
                }
        except (ValueError, httpx.HTTPError) as error:
            return {
                "status": "fail",
                "summary": "The tiny local inference probe failed.",
                "detail": f"The runtime passed readiness, but the test request did not complete cleanly: {error}",
                "tone": "danger",
                "attempted": True,
                "recommended_fix": self.local_doctor_issue_fix(
                    code="repair_runtime",
                    label="Run prerequisite-healing",
                    detail="The runtime answered readiness but did not pass a tiny local inference. Restart the local runtime once, then run Local Doctor again.",
                    source="inference_probe",
                    automated=True,
                ),
            }

        return {
            "status": "fail",
            "summary": "The local runtime does not advertise a supported inference API for Local Doctor yet.",
            "detail": "This machine did not expose either the responses or embeddings API required for the tiny local probe.",
            "tone": "danger",
            "attempted": False,
            "recommended_fix": self.local_doctor_issue_fix(
                code="unsupported_local_probe",
                label="Check the runtime profile",
                detail="Switch this machine to a supported runtime profile that exposes either responses or embeddings locally.",
                source="inference_probe",
            ),
        }

    def verify_runtime_canary(self, *, context: str) -> dict[str, Any]:
        runtime = self.runtime_status()
        probe = self.probe_local_inference(runtime)
        if str(probe.get("status") or "").strip().lower() == "pass":
            return probe
        summary = str(probe.get("summary") or "The local inference canary failed.")
        detail = str(probe.get("detail") or "").strip()
        if detail:
            raise RuntimeError(f"{context}: {summary} {detail}".strip())
        raise RuntimeError(f"{context}: {summary}".strip())

    def rollback_update_to_known_good_runtime_state(
        self,
        *,
        failure_reason: str,
        previous_release_content: str | None,
        previous_runtime_tuple: dict[str, str] | None,
    ) -> tuple[bool, str]:
        target_release = self.self_heal_state.last_known_good_release_env
        target_runtime_tuple = self.self_heal_state.last_known_good_runtime_env or previous_runtime_tuple
        rollback_id = request_id()
        release_detail = (
            f"signed release {target_release.get(RELEASE_ENV_VERSION_KEY) or 'unknown'}"
            if isinstance(target_release, dict) and target_release
            else "the previous signed release"
        )
        tuple_detail = (
            str(target_runtime_tuple.get("VLLM_MODEL") or "the last known healthy startup model")
            if isinstance(target_runtime_tuple, dict) and target_runtime_tuple
            else "the current local runtime plan"
        )
        self.log(
            f"[request {rollback_id}] Rolling back update canary failure to {release_detail} "
            f"and {tuple_detail} after: {failure_reason}"
        )
        try:
            if target_runtime_tuple:
                self.apply_runtime_tuple_snapshot(target_runtime_tuple)
            if target_release:
                self.write_release_env(target_release)
            else:
                self.restore_release_env(previous_release_content)
            self.start_runtime_services(recreate=True)
            self.wait_for_runtime_health()
            self.verify_runtime_canary(context="Rollback canary")
        except Exception as error:
            self.log(
                f"[request {rollback_id}] Automatic rollback after the failed update canary did not recover cleanly: {error}"
            )
            return False, str(error) or "Automatic rollback failed."

        message = (
            "The update canary failed, so the node rolled back automatically to the last known healthy "
            "release/runtime tuple."
        )
        self.update_self_heal_state(
            status="healthy",
            last_result=message,
            last_error=None,
            last_issue=failure_reason,
            last_action="rollback_bad_update",
            last_repaired_at=now_iso(),
            fix_available=False,
        )
        self.remember_known_good_runtime_state()
        return True, message

    def choose_local_doctor_fix(
        self,
        *,
        health: dict[str, Any],
        grouped_checks: list[dict[str, Any]],
        warm_readiness: dict[str, Any],
        inference_probe: dict[str, Any],
    ) -> dict[str, Any]:
        prerequisite_action = (
            health.get("prerequisite_action")
            if isinstance(health.get("prerequisite_action"), dict)
            else None
        )
        if prerequisite_action:
            return self.local_doctor_issue_fix(
                code=str(prerequisite_action.get("code") or "repair_runtime"),
                label=str(prerequisite_action.get("label") or health.get("issue_action_label") or "Fix machine"),
                detail=str(prerequisite_action.get("detail") or health.get("issue_detail") or "Use Fix machine once."),
                source="prerequisite_action",
                automated=True,
            )

        if health.get("issue_code"):
            label = str(health.get("issue_action_label") or "Run prerequisite-healing")
            if label == "Fix it" and bool(health.get("automatic_fix_available")):
                label = "Run prerequisite-healing"
            return self.local_doctor_issue_fix(
                code=str(health.get("issue_code") or "repair_runtime"),
                label=label,
                detail=str(health.get("issue_detail") or "The local runtime needs attention before it can serve reliably."),
                source="runtime_health",
                automated=bool(health.get("automatic_fix_available")),
            )

        for check in grouped_checks:
            if self.local_doctor_status_rank(check.get("status")) > 0 and self.local_doctor_fix_is_actionable(
                None,
                str(check.get("fix") or check.get("detail") or ""),
            ):
                label = {
                    "docker": "Fix Docker",
                    "gpu": "Fix GPU runtime",
                    "network": "Fix network reachability",
                    "cache": "Make room for cache",
                }.get(str(check.get("key") or ""), "Follow the recommended fix")
                return self.local_doctor_issue_fix(
                    code=str(check.get("key") or "setup_check"),
                    label=label,
                    detail=str(check.get("fix") or check.get("detail") or "Follow the setup guidance and run Local Doctor again."),
                    source="setup_check",
                )

        for payload in (warm_readiness, inference_probe):
            recommendation = (
                payload.get("recommended_fix")
                if isinstance(payload.get("recommended_fix"), dict)
                else None
            )
            if recommendation and self.local_doctor_fix_is_actionable(
                str(recommendation.get("label") or ""),
                str(recommendation.get("detail") or ""),
            ):
                return recommendation

        return self.local_doctor_issue_fix(
            code="none",
            label="No fix needed",
            detail="Docker, GPU access, network reachability, warm readiness, and a tiny local inference all look healthy.",
            source="local_doctor",
        )

    def _run_local_doctor_locked(
        self,
        *,
        background: bool,
        trigger: str,
        attach_bundle_on_failure: bool,
    ) -> dict[str, Any]:
        env_values = self.runtime_env_values()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})
        installer_snapshot = self.guided_installer.status_payload()
        fresh_preflight = self.guided_installer.collect_preflight(
            env_values=env_values,
            force_refresh=True,
        )
        installer_snapshot["preflight"] = fresh_preflight
        health = self.runtime_health_snapshot(installer_snapshot=installer_snapshot)
        preflight = health.get("preflight") if isinstance(health.get("preflight"), dict) else fresh_preflight
        setup_checks = self.local_doctor_setup_check_map(preflight)
        grouped_checks = [
            self.local_doctor_group_check(
                setup_checks=setup_checks,
                key="docker",
                label="Docker",
                member_keys=("docker",),
                success_summary="Docker is ready for the local runtime.",
                success_detail="Docker Desktop and its engine look healthy for this machine.",
            ),
            self.local_doctor_group_check(
                setup_checks=setup_checks,
                key="gpu",
                label="GPU",
                member_keys=("nvidia_driver", "cuda", "container_gpu"),
                success_summary="The GPU stack is ready for local inference.",
                success_detail="The NVIDIA driver, CUDA support, and container GPU access all look healthy.",
            ),
            self.local_doctor_group_check(
                setup_checks=setup_checks,
                key="network",
                label="Network",
                member_keys=("dns", "control_plane", "artifact_store"),
                success_summary="DNS and control-plane reachability look healthy.",
                success_detail="The machine resolved setup hostnames, reached the claim service, and can likely fetch signed artifacts.",
            ),
            self.local_doctor_group_check(
                setup_checks=setup_checks,
                key="cache",
                label="Cache",
                member_keys=("model_cache", "disk"),
                success_summary="The local cache budget looks healthy for warm-ups.",
                success_detail="Disk space and the local model cache budget look good for this machine.",
            ),
        ]
        warm_readiness = self.local_doctor_warm_readiness(
            installer_snapshot=installer_snapshot,
            health=health,
        )
        inference_probe = self.probe_local_inference(
            health.get("runtime") if isinstance(health.get("runtime"), dict) else {}
        )

        overall_rank = 0
        for payload in grouped_checks + [warm_readiness, inference_probe]:
            rank = self.local_doctor_status_rank(payload.get("status"))
            recommendation = (
                payload.get("recommended_fix")
                if isinstance(payload.get("recommended_fix"), dict)
                else {}
            )
            fix_label = str(recommendation.get("label") or payload.get("fix") or "")
            fix_detail = str(recommendation.get("detail") or payload.get("detail") or "")
            if rank == 1 and not self.local_doctor_fix_is_actionable(fix_label, fix_detail):
                rank = 0
            overall_rank = max(overall_rank, rank)

        recommended_fix = self.choose_local_doctor_fix(
            health=health,
            grouped_checks=grouped_checks,
            warm_readiness=warm_readiness,
            inference_probe=inference_probe,
        )

        checked_at = now_iso()
        headline = "Local Doctor found one thing to fix next"
        detail = (
            str(recommended_fix.get("detail") or "Follow the recommended fix, then run Local Doctor again.")
            if overall_rank > 0
            else "Docker, GPU access, network reachability, warm readiness, and a tiny local inference all look healthy."
        )
        status = "attention"
        if overall_rank == 0:
            status = "healthy"
            headline = "Local Doctor passed"
        elif overall_rank == 1:
            status = "warning"
            headline = "Local Doctor found a non-blocking warning"

        state_updates: dict[str, Any] = {
            "last_checked_at": checked_at,
            "status": status,
            "headline": headline,
            "detail": detail,
            "last_error": None if overall_rank < 2 else detail,
            "checks": grouped_checks,
            "warm_readiness": warm_readiness,
            "inference_probe": inference_probe,
            "recommended_fix": recommended_fix,
            "attached_bundle_name": None,
            "attached_bundle_created_at": None,
            "last_check_mode": "background" if background else "manual",
            "last_trigger": trigger,
        }
        if background:
            current_bucket = self.local_doctor_status_bucket(status)
            state_updates.update(
                last_background_check_at=checked_at,
                last_background_status=current_bucket,
                last_transition_alert=self.local_doctor_transition_alert(
                    previous_bucket=self.local_doctor_state.last_background_status,
                    current_bucket=current_bucket,
                    status=status,
                    headline=headline,
                    detail=detail,
                    trigger=trigger,
                    observed_at=checked_at,
                ),
            )
        self.update_local_doctor_state(**state_updates)

        if overall_rank >= 2 and attach_bundle_on_failure:
            _bundle_path, bundle_name, generated_at = self.write_diagnostics_bundle()
            self.diagnostics_state.last_result = (
                f"Local Doctor attached {bundle_name} so support evidence stays ready on this machine."
            )
            self.save_state()
            self.log(
                "Local Doctor attached a diagnostics bundle automatically because the machine still needs attention."
            )
            self.update_local_doctor_state(
                attached_bundle_name=bundle_name,
                attached_bundle_created_at=generated_at,
                detail=f"{detail} Attached diagnostics bundle: {bundle_name}.",
                last_error=f"{detail} Attached diagnostics bundle: {bundle_name}.",
            )

        self.log(
            f"Local Doctor completed with status: {status} "
            f"({self.local_doctor_trigger_label(trigger)}, {'background' if background else 'manual'})."
        )
        return self.status_payload()

    def run_local_doctor(
        self,
        *,
        background: bool = False,
        trigger: str = "manual",
        attach_bundle_on_failure: bool = True,
    ) -> dict[str, Any]:
        if not self.local_doctor_lock.acquire(blocking=False):
            if not background:
                self.log("Local Doctor is already running for this machine.")
            return self.status_payload()
        try:
            return self._run_local_doctor_locked(
                background=background,
                trigger=trigger,
                attach_bundle_on_failure=attach_bundle_on_failure,
            )
        finally:
            self.local_doctor_lock.release()

    def apply_local_doctor_fix(self) -> dict[str, Any]:
        if not self.local_doctor_lock.acquire(blocking=False):
            self.log("Local Doctor fix is already running for this machine.")
            return self.status_payload()
        try:
            if not self.local_doctor_state.last_checked_at:
                self._run_local_doctor_locked(
                    background=False,
                    trigger="manual_precheck",
                    attach_bundle_on_failure=True,
                )

            before_state = self.local_doctor_payload()
            recommended_fix = (
                before_state.get("recommended_fix")
                if isinstance(before_state.get("recommended_fix"), dict)
                else {}
            )
            fix_label = str(recommended_fix.get("label") or "Local Doctor")
            before_status = str(before_state.get("status") or "standing_by")
            before_headline = str(before_state.get("headline") or "Local Doctor is standing by")
            before_detail = str(before_state.get("detail") or "")
            before_bucket = self.local_doctor_status_bucket(before_status)

            if not self.local_doctor_fix_is_actionable(
                str(recommended_fix.get("label") or ""),
                str(recommended_fix.get("detail") or ""),
            ):
                self.update_local_doctor_state(
                    last_fix_attempt={
                        "started_at": now_iso(),
                        "completed_at": now_iso(),
                        "applied_fix_code": str(recommended_fix.get("code") or "none"),
                        "applied_fix_label": fix_label,
                        "automated": False,
                        "recovered": before_bucket == "healthy",
                        "changed": False,
                        "before_status": before_status,
                        "before_headline": before_headline,
                        "before_detail": before_detail,
                        "after_status": before_status,
                        "after_headline": before_headline,
                        "after_detail": before_detail,
                        "summary": "Local Doctor did not find anything to fix automatically.",
                        "before_after": f"Before: {before_headline}. After: {before_headline}.",
                    }
                )
                return self.status_payload()

            if not bool(recommended_fix.get("automated")):
                self.update_local_doctor_state(
                    last_fix_attempt={
                        "started_at": now_iso(),
                        "completed_at": now_iso(),
                        "applied_fix_code": str(recommended_fix.get("code") or "manual_fix"),
                        "applied_fix_label": fix_label,
                        "automated": False,
                        "recovered": False,
                        "changed": False,
                        "before_status": before_status,
                        "before_headline": before_headline,
                        "before_detail": before_detail,
                        "after_status": before_status,
                        "after_headline": before_headline,
                        "after_detail": before_detail,
                        "summary": f"{fix_label} still needs owner action before the node can recover fully.",
                        "before_after": f"Before: {before_headline}. After: {before_headline}.",
                    }
                )
                return self.status_payload()

            started_at = now_iso()
            self.log(f"Local Doctor is applying the recommended fix: {fix_label}.")
            self.repair_runtime(allow_quickstart_resume=False)
            payload = self._run_local_doctor_locked(
                background=False,
                trigger="auto_fix_verify",
                attach_bundle_on_failure=True,
            )
            after_state = payload.get("local_doctor") if isinstance(payload.get("local_doctor"), dict) else {}
            after_status = str(after_state.get("status") or "standing_by")
            after_headline = str(after_state.get("headline") or "Local Doctor finished the verification check")
            after_detail = str(after_state.get("detail") or "")
            after_bucket = self.local_doctor_status_bucket(after_status)
            recovered = before_bucket == "issue" and after_bucket == "healthy"
            changed = (
                before_status != after_status
                or before_headline != after_headline
                or before_detail != after_detail
            )
            summary = (
                f"Applied {fix_label} and Local Doctor passed on the automatic re-check."
                if recovered
                else f"Applied {fix_label} and re-checked the machine."
            )
            if not changed:
                summary = f"Applied {fix_label}, but Local Doctor still sees the same issue."

            self.update_local_doctor_state(
                last_fix_attempt={
                    "started_at": started_at,
                    "completed_at": now_iso(),
                    "applied_fix_code": str(recommended_fix.get("code") or "repair_runtime"),
                    "applied_fix_label": fix_label,
                    "automated": True,
                    "recovered": recovered,
                    "changed": changed,
                    "before_status": before_status,
                    "before_headline": before_headline,
                    "before_detail": before_detail,
                    "after_status": after_status,
                    "after_headline": after_headline,
                    "after_detail": after_detail,
                    "summary": summary,
                    "before_after": f"Before: {before_headline}. After: {after_headline}.",
                }
            )
            return self.status_payload()
        finally:
            self.local_doctor_lock.release()

    def build_control_client(self) -> EdgeControlClient | Any | None:
        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            return None
        settings = self.guided_installer.build_installer_settings(env_values)
        client = self.guided_installer.control_client_factory(settings)
        raw_client = getattr(client, "client", None)
        if raw_client is not None and hasattr(raw_client, "timeout"):
            try:
                raw_client.timeout = httpx.Timeout(5.0)
            except Exception:
                pass
        return client

    def remote_dashboard_snapshot(self, *, force: bool = False) -> dict[str, Any]:
        with self.lock:
            cached_state = RemoteDashboardCacheState(
                fetched_at=self.remote_dashboard_state.fetched_at,
                payload=self.remote_dashboard_state.payload,
                last_error=self.remote_dashboard_state.last_error,
            )
        if not force and cached_state.fetched_at:
            fetched_at = parse_iso_timestamp(cached_state.fetched_at)
            if fetched_at and (datetime.now(timezone.utc) - fetched_at).total_seconds() < REMOTE_DASHBOARD_CACHE_TTL_SECONDS:
                return {
                    "summary": cached_state.payload,
                    "synced_at": cached_state.fetched_at,
                    "last_error": cached_state.last_error,
                    "stale": False,
                }

        client = self.build_control_client()
        if client is None:
            return {
                "summary": cached_state.payload,
                "synced_at": cached_state.fetched_at,
                "last_error": None,
                "stale": bool(cached_state.payload),
            }

        try:
            summary = client.fetch_node_dashboard_summary()
        except RuntimeError as error:
            message = str(error) or "Node dashboard sync is unavailable."
            if "No stored node credentials were found" in message:
                return {
                    "summary": cached_state.payload,
                    "synced_at": cached_state.fetched_at,
                    "last_error": None,
                    "stale": bool(cached_state.payload),
                }
            with self.lock:
                self.remote_dashboard_state.last_error = message
            return {
                "summary": cached_state.payload,
                "synced_at": cached_state.fetched_at,
                "last_error": message,
                "stale": bool(cached_state.payload),
            }
        except Exception as error:
            message = str(error) or "Node dashboard sync is unavailable."
            with self.lock:
                self.remote_dashboard_state.last_error = message
            return {
                "summary": cached_state.payload,
                "synced_at": cached_state.fetched_at,
                "last_error": message,
                "stale": bool(cached_state.payload),
            }

        fetched_at = now_iso()
        with self.lock:
            self.remote_dashboard_state = RemoteDashboardCacheState(
                fetched_at=fetched_at,
                payload=summary,
                last_error=None,
            )
        return {
            "summary": summary,
            "synced_at": fetched_at,
            "last_error": None,
            "stale": False,
        }

    def load_control_plane_connectivity_state(self) -> dict[str, Any]:
        if not self.control_plane_state_path.exists():
            return {}
        try:
            payload = json.loads(self.control_plane_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def control_plane_connectivity_payload(self, *, remote: dict[str, Any] | None = None) -> dict[str, Any]:
        remote = remote or self.remote_dashboard_snapshot()
        remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
        remote_error = remote.get("last_error") if isinstance(remote.get("last_error"), str) else None
        runtime_env = self.guided_installer.effective_runtime_env()
        persisted_state = self.load_control_plane_connectivity_state()
        status = str(persisted_state.get("status") or "").strip().lower()
        if status not in {"healthy", "degraded", "offline"}:
            if remote_error and remote_summary:
                status = "degraded"
            elif remote_error:
                status = "offline"
            else:
                status = "healthy"
        grace_deadline_at = (
            str(persisted_state.get("grace_deadline_at")) if persisted_state.get("grace_deadline_at") else None
        )
        grace_deadline = parse_iso_timestamp(grace_deadline_at)
        grace_active = bool(grace_deadline and datetime.now(timezone.utc) <= grace_deadline)
        fallback_active = bool(
            persisted_state.get("active_base_url")
            and persisted_state.get("primary_base_url")
            and persisted_state.get("active_base_url") != persisted_state.get("primary_base_url")
        )
        mirror_urls = csv_items(runtime_env.get("ARTIFACT_MIRROR_BASE_URLS"))
        fallback_urls = csv_items(runtime_env.get("EDGE_CONTROL_FALLBACK_URLS"))
        if status == "healthy":
            value = "Healthy"
            tone = "success"
            detail = "The primary control-plane path is healthy and this node is syncing normally."
        elif status == "degraded":
            value = "Degraded mode"
            tone = "warning"
            detail = (
                str(persisted_state.get("last_error") or remote_error or "").strip()
                or "Control-plane reachability is intermittent, so the node is serving from its local reservoir and stretching retries."
            )
        elif grace_active:
            value = "Offline grace"
            tone = "warning"
            detail = (
                str(persisted_state.get("last_error") or remote_error or "").strip()
                or "The control plane is temporarily unreachable, and this node is using its local grace window to keep serving staged work."
            )
        else:
            value = "Offline"
            tone = "danger"
            detail = (
                str(persisted_state.get("last_error") or remote_error or "").strip()
                or "The control plane is unreachable and the local grace window has expired."
            )
        if fallback_active:
            detail = (
                f"{detail} Using {persisted_state.get('active_base_url')} as the temporary control-plane path."
            )
        elif fallback_urls and status != "healthy":
            detail = f"{detail} Fallback control-plane hosts are configured for recovery."
        if mirror_urls:
            detail = f"{detail} Artifact mirrors are configured for warm paths."
        return {
            **persisted_state,
            "status": status,
            "value": value,
            "detail": detail,
            "tone": tone,
            "grace_active": grace_active,
            "grace_deadline_at": grace_deadline_at,
            "fallback_active": fallback_active,
            "fallback_urls": fallback_urls,
            "artifact_mirror_urls": mirror_urls,
            "remote_sync_error": remote_error,
            "remote_sync_stale": bool(remote.get("stale")),
        }

    def load_autopilot_state(self) -> dict[str, Any] | None:
        if not self.autopilot_state_path.exists():
            return None
        try:
            payload = json.loads(self.autopilot_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def save_autopilot_state(self, payload: dict[str, Any]) -> None:
        self.autopilot_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.autopilot_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tighten_private_path(self.autopilot_state_path)

    def autopilot_payload(self) -> dict[str, Any]:
        state = self.load_autopilot_state()
        if not state:
            return {
                "enabled": True,
                "status": "learning",
                "value": "Learning",
                "tone": "warning",
                "detail": "Autopilot starts tuning after this node completes real work.",
                "signals": {},
                "recommendation": None,
                "recent_model_mix": {},
            }
        recommendation = state.get("recommendation") if isinstance(state.get("recommendation"), dict) else {}
        signals = state.get("signals") if isinstance(state.get("signals"), dict) else {}
        pending_restart = bool(recommendation.get("pending_restart"))
        startup_model = str(recommendation.get("startup_model") or state.get("current_model") or "current model")
        profile = str(recommendation.get("setup_profile") or "balanced").replace("_", " ")
        concurrency = str(recommendation.get("max_concurrent_assignments") or "1")
        reason = str(recommendation.get("reason") or "Autopilot is monitoring this machine.")
        pressure = signals.get("gpu_memory_pressure")
        failure_rate = signals.get("failure_rate")
        latency = signals.get("ewma_latency_seconds")
        signal_parts: list[str] = []
        if isinstance(pressure, (int, float)):
            signal_parts.append(f"GPU memory {round(float(pressure) * 100)}%")
        if isinstance(failure_rate, (int, float)):
            signal_parts.append(f"failure rate {round(float(failure_rate) * 100)}%")
        if isinstance(latency, (int, float)):
            signal_parts.append(f"latency {round(float(latency), 1)}s")
        signal_copy = ", ".join(signal_parts)
        detail = f"{reason} Target: {startup_model}, {profile} profile, {concurrency} concurrent."
        if signal_copy:
            detail = f"{detail} Signals: {signal_copy}."
        if pending_restart:
            detail = f"{detail} A safe restart is needed to switch the startup model."
        return {
            "enabled": bool(state.get("enabled", True)),
            "status": "restart_pending" if pending_restart else "active",
            "value": "Restart pending" if pending_restart else "Active",
            "tone": "warning" if pending_restart else "success",
            "detail": detail,
            "signals": signals,
            "recommendation": recommendation,
            "recent_model_mix": (
                state.get("recent_model_mix") if isinstance(state.get("recent_model_mix"), dict) else {}
            ),
            "updated_at": state.get("updated_at"),
            "history": state.get("history") if isinstance(state.get("history"), list) else [],
        }

    def supported_models_from_config(self, config: dict[str, Any]) -> set[str]:
        supported = str(config.get("supported_models") or "").split(",")
        models = {model.strip() for model in supported if model.strip()}
        env_values = self.guided_installer.effective_runtime_env()
        if env_values:
            models.update(
                model.strip()
                for model in str(env_values.get("SUPPORTED_MODELS") or "").split(",")
                if model.strip()
            )
            for key in ("VLLM_MODEL", "OWNER_TARGET_MODEL"):
                value = str(env_values.get(key) or "").strip()
                if value:
                    models.add(value)
            models.update(
                model.strip()
                for model in str(env_values.get("OWNER_TARGET_SUPPORTED_MODELS") or "").split(",")
                if model.strip()
            )
        for key in ("vllm_model", "owner_target_model", "recommended_model"):
            value = str(config.get(key) or "").strip()
            if value:
                models.add(value)
        models.update(
            model.strip()
            for model in str(config.get("owner_target_supported_models") or "").split(",")
            if model.strip()
        )
        return models

    def ranked_demand_models(self, demand: Any) -> list[tuple[str, float]]:
        entries: list[tuple[str, float]] = []

        def add_model(model: Any, score: float) -> None:
            if isinstance(model, str) and model.strip():
                entries.append((model.strip(), score))

        def score_from_payload(payload: dict[str, Any], fallback: float) -> float:
            for key in ("score", "queue_depth", "pending_count", "ready_count", "item_count"):
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
                try:
                    return float(str(value))
                except (TypeError, ValueError):
                    continue
            return fallback

        if isinstance(demand, dict):
            for key in ("recommended_model", "likely_model", "hot_model", "model"):
                add_model(demand.get(key), 100.0)
            ranked = demand.get("models") or demand.get("ranked_models") or demand.get("queues")
            if isinstance(ranked, list):
                for index, item in enumerate(ranked):
                    if isinstance(item, str):
                        add_model(item, 50.0 - index)
                    elif isinstance(item, dict):
                        add_model(item.get("model"), score_from_payload(item, 50.0 - index))
        elif isinstance(demand, list):
            for index, item in enumerate(demand):
                if isinstance(item, str):
                    add_model(item, 50.0 - index)
                elif isinstance(item, dict):
                    add_model(item.get("model"), score_from_payload(item, 50.0 - index))

        return sorted(entries, key=lambda entry: entry[1], reverse=True)

    def recent_assignment_mix(self, autopilot: dict[str, Any]) -> list[tuple[str, float]]:
        mix_payload = autopilot.get("recent_model_mix") if isinstance(autopilot.get("recent_model_mix"), dict) else {}
        entries: list[tuple[str, float]] = []
        for model, raw_score in mix_payload.items():
            if not isinstance(model, str) or not model.strip():
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if score <= 0:
                continue
            entries.append((model.strip(), score))
        return sorted(entries, key=lambda entry: entry[1], reverse=True)

    def cache_budget_payload(
        self,
        *,
        cache_bytes: int | None = None,
        env_values: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        runtime_env = env_values or self.guided_installer.effective_runtime_env()
        cache_total_bytes = (
            max(0, int(cache_bytes))
            if cache_bytes is not None
            else directory_size_bytes(self.data_dir / "model-cache")
        )
        disk = detect_disk(self.runtime_dir)
        return resolve_model_cache_budget(
            total_bytes=disk.get("total_bytes"),
            free_bytes=disk.get("free_bytes"),
            cache_bytes=cache_total_bytes,
            configured_budget_gb=(runtime_env or {}).get("MODEL_CACHE_BUDGET_GB"),
            configured_reserve_free_gb=(runtime_env or {}).get("MODEL_CACHE_RESERVE_FREE_GB"),
        )

    def should_keep_hot_models_warm(
        self,
        *,
        config: dict[str, Any],
        remote_summary: dict[str, Any],
        remote_node: dict[str, Any],
        autopilot: dict[str, Any],
    ) -> bool:
        heat_mode = normalize_heat_governor_mode(config.get("heat_governor_mode"))
        if heat_mode in {"0", "0%", "off"}:
            return False

        target_pct = coerce_nonnegative_int(config.get("target_gpu_utilization_pct"))
        heat_demand = str(config.get("heat_demand") or "").strip().lower()
        outside_temp_c = coerce_float(config.get("outside_temp_c"))
        room_temp_c = coerce_float(config.get("room_temp_c"))
        target_temp_c = coerce_float(config.get("target_temp_c"))
        queue_depth = coerce_nonnegative_int(remote_node.get("queue_depth"))
        active_assignments = coerce_nonnegative_int(remote_node.get("active_assignments"))
        earnings = remote_summary.get("earnings") if isinstance(remote_summary.get("earnings"), dict) else {}

        earnings_signal = False
        for key in ("today_usd", "last_24h_usd", "current_hour_usd"):
            value = earnings.get(key)
            if isinstance(value, (int, float)) and float(value) > 0:
                earnings_signal = True
                break

        recent_mix_active = bool(self.recent_assignment_mix(autopilot))
        weather_supports_heat = (
            (outside_temp_c is not None and outside_temp_c <= 15.0)
            or (
                room_temp_c is not None
                and target_temp_c is not None
                and room_temp_c < (target_temp_c - 0.3)
            )
        )
        owner_wants_heat = target_pct >= 50 or heat_demand in {"medium", "high"}
        return owner_wants_heat and (
            weather_supports_heat
            or earnings_signal
            or queue_depth > 0
            or active_assignments > 0
            or recent_mix_active
        )

    def ranked_hot_models(
        self,
        *,
        config: dict[str, Any],
        remote_summary: dict[str, Any],
        remote_node: dict[str, Any],
        autopilot: dict[str, Any],
    ) -> list[str]:
        supported_models = self.supported_models_from_config(config)
        scores: dict[str, float] = {}

        def add(model: str | None, score: float) -> None:
            if not isinstance(model, str):
                return
            candidate = model.strip()
            if not candidate:
                return
            if supported_models and candidate not in supported_models:
                return
            scores[candidate] = scores.get(candidate, 0.0) + max(0.0, float(score))

        for demand in (
            remote_summary.get("model_demand"),
            remote_node.get("model_demand"),
            (remote_node.get("runtime") or {}).get("model_demand")
            if isinstance(remote_node.get("runtime"), dict)
            else None,
        ):
            for model, score in self.ranked_demand_models(demand):
                add(model, score)

        for model, score in self.recent_assignment_mix(autopilot):
            add(model, score * 25.0)

        owner_target_model = str(config.get("owner_target_model") or "").strip()
        if owner_target_model and self.should_keep_hot_models_warm(
            config=config,
            remote_summary=remote_summary,
            remote_node=remote_node,
            autopilot=autopilot,
        ):
            add(owner_target_model, 60.0)

        return [
            model
            for model, _score in sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:MODEL_CACHE_HOT_MODEL_LIMIT]
        ]

    def protected_model_set(
        self,
        *,
        current_model: str | None,
        likely_model: str | None,
        config: dict[str, Any],
        hot_models: list[str],
        keep_hot_models_warm: bool,
    ) -> set[str]:
        protected_models = {model for model in (current_model, likely_model) if model}
        for runtime_env in (
            self.self_heal_state.last_known_good_bootstrap_runtime_env or {},
            self.self_heal_state.last_known_good_runtime_env or {},
        ):
            bootstrap_model = str(runtime_env.get("VLLM_MODEL") or "").strip()
            if bootstrap_model:
                protected_models.add(bootstrap_model)
        if keep_hot_models_warm:
            protected_models.update(model for model in hot_models if model)
            owner_target_model = str(config.get("owner_target_model") or "").strip()
            if owner_target_model:
                protected_models.add(owner_target_model)
        return protected_models

    def select_likely_model(
        self,
        *,
        runtime: dict[str, Any],
        config: dict[str, Any],
        remote_summary: dict[str, Any],
        remote_node: dict[str, Any],
        autopilot: dict[str, Any],
    ) -> tuple[str | None, str]:
        supported_models = self.supported_models_from_config(config)

        def supported(model: str | None) -> str | None:
            if not model:
                return None
            model = model.strip()
            if not model:
                return None
            if supported_models and model not in supported_models:
                return None
            return model

        demand_sources = (
            remote_summary.get("model_demand"),
            remote_node.get("model_demand"),
            (remote_node.get("runtime") or {}).get("model_demand")
            if isinstance(remote_node.get("runtime"), dict)
            else None,
        )
        for demand in demand_sources:
            for model, _score in self.ranked_demand_models(demand):
                candidate = supported(model)
                if candidate:
                    return candidate, "control_plane_demand"

        owner_target_model = supported(str(config.get("owner_target_model") or "").strip())
        current_runtime_model = str(runtime.get("current_model") or "").strip()
        current_config_model = str(config.get("vllm_model") or "").strip()
        keep_hot_models_warm = self.should_keep_hot_models_warm(
            config=config,
            remote_summary=remote_summary,
            remote_node=remote_node,
            autopilot=autopilot,
        )
        if owner_target_model and keep_hot_models_warm and owner_target_model not in {current_runtime_model, current_config_model}:
            hot_models = self.ranked_hot_models(
                config=config,
                remote_summary=remote_summary,
                remote_node=remote_node,
                autopilot=autopilot,
            )
            if owner_target_model in hot_models:
                return owner_target_model, "owner_target_heat"

        for model, _score in self.recent_assignment_mix(autopilot):
            candidate = supported(model)
            if candidate and candidate not in {current_runtime_model, current_config_model}:
                return candidate, "recent_assignment_mix"

        if owner_target_model and owner_target_model not in {current_runtime_model, current_config_model}:
            return owner_target_model, "bootstrap_target"

        recommendation = autopilot.get("recommendation") if isinstance(autopilot.get("recommendation"), dict) else {}
        for model, source in (
            (str(recommendation.get("startup_model") or "").strip(), "autopilot"),
            (str(runtime.get("current_model") or "").strip(), "current_runtime"),
            (str(config.get("vllm_model") or "").strip(), "local_settings"),
            (str(config.get("recommended_model") or "").strip(), "installer_recommendation"),
        ):
            candidate = supported(model)
            if candidate:
                return candidate, source
        return None, "unknown"

    def model_cache_paths(self, model: str) -> list[Path]:
        cache_key = f"models--{model.replace('/', '--')}"
        cache_dir = self.data_dir / "model-cache"
        return [
            cache_dir / "hub" / cache_key,
            cache_dir / cache_key,
        ]

    def model_cache_bytes_for_model(self, model: str | None) -> int:
        if not model:
            return 0
        return sum(directory_size_bytes(path) for path in self.model_cache_paths(model))

    def cached_model_names(self) -> set[str]:
        cache_dir = self.data_dir / "model-cache"
        models: set[str] = set()
        for root in (cache_dir / "hub", cache_dir):
            if not root.exists():
                continue
            for path in root.iterdir():
                if path.is_dir() and path.name.startswith("models--"):
                    model = path.name.removeprefix("models--").replace("--", "/")
                    if model:
                        models.add(model)
        return models

    def touch_model_usage(self, *models: str | None) -> None:
        timestamp = now_iso()
        for model in models:
            if isinstance(model, str) and model.strip():
                self.model_cache_state.last_used_by_model[model.strip()] = timestamp

    def model_cache_payload(
        self,
        *,
        health: dict[str, Any],
        runtime: dict[str, Any],
        config: dict[str, Any],
        remote_summary: dict[str, Any],
        remote_node: dict[str, Any],
        autopilot: dict[str, Any],
    ) -> dict[str, Any]:
        cache_dir = self.data_dir / "model-cache"
        cache_bytes = directory_size_bytes(cache_dir)
        runtime_env = self.guided_installer.effective_runtime_env()
        budget = self.cache_budget_payload(cache_bytes=cache_bytes, env_values=runtime_env)
        likely_model, likely_source = self.select_likely_model(
            runtime=runtime,
            config=config,
            remote_summary=remote_summary,
            remote_node=remote_node,
            autopilot=autopilot,
        )
        current_model = str(runtime.get("current_model") or config.get("vllm_model") or "").strip() or None
        keep_hot_models_warm = self.should_keep_hot_models_warm(
            config=config,
            remote_summary=remote_summary,
            remote_node=remote_node,
            autopilot=autopilot,
        )
        hot_models = self.ranked_hot_models(
            config=config,
            remote_summary=remote_summary,
            remote_node=remote_node,
            autopilot=autopilot,
        )
        pinned_models = sorted(
            self.protected_model_set(
                current_model=current_model,
                likely_model=likely_model,
                config=config,
                hot_models=hot_models,
                keep_hot_models_warm=keep_hot_models_warm,
            )
        )
        likely_cache_bytes = self.model_cache_bytes_for_model(likely_model)
        runtime_engine = str(
            runtime.get("inference_engine")
            or config.get("inference_engine")
            or self.guided_installer.resolved_inference_engine()
        )
        expected_bytes = (
            artifact_total_size_bytes(startup_model_artifact(likely_model, runtime_engine=runtime_engine))
            if likely_model
            else None
        )
        background_upgrade = bool(
            likely_source == "bootstrap_target"
            and current_model
            and likely_model
            and current_model != likely_model
        )
        reuse_percent: int | None = None
        if expected_bytes and expected_bytes > 0:
            reuse_percent = max(0, min(100, int((likely_cache_bytes / expected_bytes) * 100)))

        self.model_cache_state.cache_bytes = cache_bytes
        self.model_cache_state.cache_budget_bytes = int(budget.get("budget_bytes") or 0)
        self.model_cache_state.reserve_free_bytes = int(budget.get("reserve_free_bytes") or 0)
        self.model_cache_state.cache_budget_source = str(budget.get("budget_source") or "derived")
        self.model_cache_state.likely_model = likely_model
        self.model_cache_state.likely_model_source = likely_source
        self.model_cache_state.hot_models = hot_models
        self.model_cache_state.pinned_models = pinned_models

        if not likely_model:
            value = "Checking cache"
            detail = "The runtime is still deciding which model is most likely to be needed next."
            tone = "warning"
        elif current_model == likely_model and runtime.get("stage") == "running" and health.get("runtime_healthy"):
            value = "Warm"
            detail = (
                f"{likely_model} is already loaded. The local cache has {format_bytes(likely_cache_bytes or cache_bytes)} "
                "ready to reuse across restarts, so future warm-ups should be faster."
            )
            tone = "success"
        elif likely_cache_bytes > 0:
            cache_label = format_bytes(likely_cache_bytes)
            if reuse_percent is not None and reuse_percent < 90:
                value = "Background upgrade" if background_upgrade else "Partly cached"
                detail = (
                    f"The node is already online on {current_model}. Reusing {cache_label} of cached data for the larger owner target "
                    f"{likely_model} ({reuse_percent}% of the expected snapshot) before the background switch finishes."
                    if background_upgrade
                    else (
                        f"The next likely model is {likely_model}. Reusing {cache_label} already in the local model cache "
                        f"({reuse_percent}% of the expected snapshot), then filling the rest when it warms."
                    )
                )
                tone = "warning"
            else:
                value = "Background upgrade" if background_upgrade else "Cache ready"
                detail = (
                    f"The node is already online on {current_model}. Reusing {cache_label} from the local cache so the larger owner target "
                    f"{likely_model} can switch in with a shorter background warm-up."
                    if background_upgrade
                    else (
                        f"The next likely model is {likely_model}. Reusing {cache_label} from the local model cache, "
                        "so the next warm-up should avoid most of the first-run download."
                    )
                )
                tone = "success"
        else:
            value = "Background upgrade" if background_upgrade else "Will warm on first use"
            expected_label = f" about {format_bytes(expected_bytes)}" if expected_bytes else ""
            detail = (
                f"The node is already online on {current_model}. No reusable cache was found yet for the larger owner target "
                f"{likely_model}, so the background warm-up may still download{expected_label} before switching over."
                if background_upgrade
                else (
                    f"The next likely model is {likely_model}. No reusable local cache was found yet, so the first warm-up "
                    f"may download{expected_label}; after that, this machine will reuse the local cache automatically."
                )
            )
            tone = "warning"

        budget_copy = (
            f" Cache budget: {budget.get('budget_label')} with {budget.get('reserve_free_label')} kept free."
        )
        if budget.get("over_budget_bytes"):
            budget_copy += (
                f" Cached models are over budget by {format_bytes(int(budget['over_budget_bytes']))}, "
                "so cold cache eviction will stay active."
            )
        elif budget.get("available_growth_bytes") is not None:
            budget_copy += f" About {budget.get('available_growth_label')} is free for the next warm-up."
        detail = f"{detail}{budget_copy}"

        return {
            "enabled": self.model_cache_state.enabled,
            "value": value,
            "detail": detail,
            "tone": tone,
            "cache_dir": str(cache_dir),
            "cache_bytes": cache_bytes,
            "cache_label": format_bytes(cache_bytes),
            "likely_model": likely_model,
            "likely_model_source": likely_source,
            "current_model": current_model,
            "likely_model_cache_bytes": likely_cache_bytes,
            "likely_model_cache_label": format_bytes(likely_cache_bytes),
            "expected_bytes": expected_bytes,
            "expected_label": format_bytes(expected_bytes),
            "reuse_percent": reuse_percent,
            "cache_budget_bytes": budget.get("budget_bytes"),
            "cache_budget_label": budget.get("budget_label"),
            "cache_budget_source": budget.get("budget_source"),
            "cache_reserve_free_bytes": budget.get("reserve_free_bytes"),
            "cache_reserve_free_label": budget.get("reserve_free_label"),
            "cache_available_growth_bytes": budget.get("available_growth_bytes"),
            "cache_available_growth_label": budget.get("available_growth_label"),
            "cache_budget_tier": budget.get("tier"),
            "keep_hot_models_warm": keep_hot_models_warm,
            "hot_models": hot_models,
            "pinned_models": pinned_models,
            "last_checked_at": self.model_cache_state.last_checked_at,
            "last_warmed_model": self.model_cache_state.last_warmed_model,
            "last_warmed_at": self.model_cache_state.last_warmed_at,
            "last_evicted_model": self.model_cache_state.last_evicted_model,
            "last_evicted_bytes": self.model_cache_state.last_evicted_bytes,
            "last_evicted_at": self.model_cache_state.last_evicted_at,
            "cached_models": sorted(self.cached_model_names()),
        }

    def safe_model_cache_path(self, path: Path) -> bool:
        try:
            cache_root = (self.data_dir / "model-cache").resolve()
            target = path.resolve()
            return os.path.commonpath([str(cache_root), str(target)]) == str(cache_root)
        except (OSError, ValueError):
            return False

    def ensure_model_cache_capacity(
        self,
        *,
        target_model: str | None,
        required_growth_bytes: int,
        protected_models: set[str],
        hot_models: list[str],
        budget: dict[str, Any],
    ) -> tuple[bool, list[tuple[str, int]], str | None]:
        if not self.model_cache_state.enabled:
            return False, [], "Model cache management is disabled."

        current_cache_bytes = directory_size_bytes(self.data_dir / "model-cache")
        current_free_bytes = max(0, int(budget.get("free_bytes") or 0))
        budget_bytes = max(0, int(budget.get("budget_bytes") or 0))
        reserve_free_bytes = max(0, int(budget.get("reserve_free_bytes") or 0))
        required_relief_bytes = max(
            0,
            int(required_growth_bytes),
            current_cache_bytes - budget_bytes,
            reserve_free_bytes - current_free_bytes,
        )
        if required_relief_bytes <= 0:
            return True, [], None

        evictions: list[tuple[str, int]] = []
        current_time = datetime.now(timezone.utc)
        while required_relief_bytes > 0:
            evicted_model, evicted_bytes = self.evict_cold_model_cache(
                protected_models=protected_models,
                cache_pressure=True,
                now=current_time,
                hot_models=set(hot_models),
            )
            if not evicted_model or evicted_bytes <= 0:
                break
            evictions.append((evicted_model, evicted_bytes))
            current_cache_bytes = max(0, current_cache_bytes - evicted_bytes)
            current_free_bytes += evicted_bytes
            required_relief_bytes = max(
                0,
                int(required_growth_bytes) - min(
                    max(0, budget_bytes - current_cache_bytes),
                    max(0, current_free_bytes - reserve_free_bytes),
                ),
                current_cache_bytes - budget_bytes,
                reserve_free_bytes - current_free_bytes,
            )

        if required_relief_bytes <= 0:
            return True, evictions, None

        target_label = target_model or "the next model"
        message = (
            f"{target_label} needs about {format_bytes(required_growth_bytes)} of additional cache space, but this node keeps "
            f"{budget.get('reserve_free_label')} free and caps the model cache at {budget.get('budget_label')}."
        )
        if evictions:
            message += " Cold cache was already evicted, but there still is not enough budget left for another warm-up."
        return False, evictions, message

    def evict_cold_model_cache(
        self,
        *,
        protected_models: set[str],
        cache_pressure: bool,
        now: datetime | None = None,
        hot_models: set[str] | None = None,
    ) -> tuple[str | None, int]:
        if not cache_pressure or not self.model_cache_state.enabled:
            return None, 0
        current_time = now or datetime.now(timezone.utc)
        candidates: list[tuple[int, int, int, str]] = []
        hot = hot_models or set()
        for model in sorted(self.cached_model_names()):
            if model in protected_models or model in hot:
                continue
            last_used = parse_iso_timestamp(self.model_cache_state.last_used_by_model.get(model))
            age_seconds = int((current_time - last_used).total_seconds()) if last_used else MODEL_CACHE_COLD_SECONDS * 4
            if age_seconds >= MODEL_CACHE_COLD_SECONDS:
                age_rank = 0
            elif age_seconds >= 24 * 60 * 60:
                age_rank = 1
            elif age_seconds >= MODEL_CACHE_RECENT_SECONDS:
                age_rank = 2
            else:
                age_rank = 3
            size_bytes = self.model_cache_bytes_for_model(model)
            candidates.append((age_rank, -size_bytes, age_seconds, model))

        for _age_rank, _negative_size, _age_seconds, model in sorted(candidates):
            evicted_bytes = 0
            for path in self.model_cache_paths(model):
                if not path.exists() or not self.safe_model_cache_path(path):
                    continue
                evicted_bytes += directory_size_bytes(path)
                shutil.rmtree(path, ignore_errors=True)
            if evicted_bytes > 0:
                return model, evicted_bytes
        return None, 0

    def prewarm_likely_model(
        self,
        likely_model: str,
        current_model: str | None,
        *,
        source: str = "control_plane_demand",
    ) -> tuple[bool, str]:
        retry_hold = self.model_warm_retry_hold(likely_model)
        if retry_hold:
            return False, retry_hold
        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            return False, "Model cache management is waiting for local runtime settings."
        supported_models = {model.strip() for model in env_values.get("SUPPORTED_MODELS", "").split(",") if model.strip()}
        owner_target_model = str(env_values.get("OWNER_TARGET_MODEL") or "").strip()
        owner_target_supported_models = str(env_values.get("OWNER_TARGET_SUPPORTED_MODELS") or "").strip()
        if supported_models and likely_model not in supported_models and likely_model != owner_target_model:
            return False, f"{likely_model} is not advertised by this machine, so pre-warm was skipped."
        if current_model == likely_model and env_values.get("VLLM_MODEL") == likely_model:
            return False, f"{likely_model} is already the warm startup model."

        previous_env = dict(env_values)
        next_env = dict(env_values)
        configured_runtime_profile = str(env_values.get("RUNTIME_PROFILE") or "").strip()
        configured_engine = str(env_values.get("INFERENCE_ENGINE") or "").strip()
        configured_deployment_target = str(env_values.get("DEPLOYMENT_TARGET") or "").strip()
        runtime_profile = resolve_runtime_profile(
            configured_runtime_profile,
            configured_engine=configured_engine,
            configured_deployment_target=configured_deployment_target,
            runtime_backend=self.guided_installer.current_runtime_backend(),
            model=likely_model,
        )
        desired_supported_models = (
            owner_target_supported_models
            if owner_target_model and owner_target_model == likely_model and owner_target_supported_models
            else ",".join(runtime_profile.supported_models)
        )
        desired_supported_models = constrain_supported_models_for_runtime_profile(
            desired_supported_models,
            runtime_profile=runtime_profile,
            preferred_model=likely_model,
        )
        selected_model, selected_supported_models, _fallback = resolve_accessible_startup_selection(
            likely_model,
            desired_supported_models,
            token_configured=bool(str(env_values.get("HUGGING_FACE_HUB_TOKEN") or "").strip()),
            inference_engine=runtime_profile.inference_engine,
        )
        if selected_model != likely_model:
            return False, f"{likely_model} is not accessible on this machine yet, so pre-warm was skipped."
        next_env["VLLM_MODEL"] = selected_model
        next_env["SUPPORTED_MODELS"] = selected_supported_models
        next_env.update(llama_cpp_env_for_model(selected_model))
        if owner_target_model and owner_target_model == selected_model:
            next_env["RUNTIME_PROFILE"] = AUTO_RUNTIME_PROFILE
        try:
            self.guided_installer.write_runtime_settings(next_env)
            self.guided_installer.write_runtime_env(next_env)
            if source == "bootstrap_target":
                self.log(
                    f"Quick Start brought this node online on {current_model or previous_env.get('VLLM_MODEL') or 'the bootstrap model'}. "
                    f"Now pre-warming the larger owner target {likely_model} in the background."
                )
            else:
                self.log(
                    f"Demand-aware model management is pre-warming {likely_model} while the node is idle. "
                    "Existing local cache will be reused where possible."
                )
            self.start_runtime_services(recreate=True)
            self.wait_for_runtime_health(timeout_seconds=240.0)
        except Exception as error:
            self.guided_installer.write_runtime_settings(previous_env)
            self.guided_installer.write_runtime_env(previous_env)
            rollback_prefix = "Background owner-model warm-up" if source == "bootstrap_target" else "Demand-aware pre-warm"
            self.log(
                f"{rollback_prefix} for {likely_model} failed. Rolling back to "
                f"{previous_env.get('VLLM_MODEL') or 'the previous startup model'}."
            )
            try:
                self.start_runtime_services(recreate=True)
                self.wait_for_runtime_health(timeout_seconds=90.0)
                self.remember_known_good_runtime_state()
            except Exception as rollback_error:
                self.log(f"Rollback after automatic model pre-warm also needs attention: {rollback_error}")
            failure_count, cooldown_until = self.record_model_warm_failure(likely_model, error)
            if source == "bootstrap_target" and cooldown_until is not None:
                raise RuntimeError(
                    f"{likely_model} failed to warm {failure_count} times in a row, so self-healing rolled back to the "
                    f"last known-good bootstrap model until {cooldown_until}. Latest error: {error}"
                ) from error
            raise
        warmed_at = now_iso()
        self.model_cache_state.last_warmed_model = likely_model
        self.model_cache_state.last_warmed_at = warmed_at
        self.model_cache_state.last_error = None
        self.touch_model_usage(likely_model)
        self.clear_model_warm_failure(likely_model)
        self.remember_known_good_runtime_tuple()
        if source == "bootstrap_target":
            return (
                True,
                f"Pre-warmed the larger owner target {likely_model} after this node came online on the bootstrap model.",
            )
        return True, f"Pre-warmed {likely_model} because it is the most likely next model for this node."

    def manage_model_cache(self, *, health: dict[str, Any], installer_snapshot: dict[str, Any]) -> tuple[bool, str, str]:
        if not self.model_cache_state.enabled:
            return False, "Model cache management is disabled.", "model_cache_disabled"

        runtime = health.get("runtime", {})
        config = installer_snapshot.get("config", {})
        autopilot = self.load_autopilot_payload()
        remote = self.remote_dashboard_snapshot()
        remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
        remote_node = remote_summary.get("node") if isinstance(remote_summary.get("node"), dict) else {}
        model_cache = self.model_cache_payload(
            health=health,
            runtime=runtime,
            config=config,
            remote_summary=remote_summary,
            remote_node=remote_node,
            autopilot=autopilot,
        )
        likely_model = model_cache.get("likely_model") if isinstance(model_cache.get("likely_model"), str) else None
        current_model = model_cache.get("current_model") if isinstance(model_cache.get("current_model"), str) else None
        hot_models = [
            model
            for model in model_cache.get("hot_models", [])
            if isinstance(model, str) and model.strip()
        ]
        protected_models = {
            model
            for model in model_cache.get("pinned_models", [])
            if isinstance(model, str) and model.strip()
        }
        self.touch_model_usage(current_model, likely_model, *hot_models)

        disk = health.get("preflight", {}).get("disk") if isinstance(health.get("preflight"), dict) else {}
        cache_pressure = bool(
            isinstance(disk, dict)
            and (
                not bool(disk.get("ok", True))
                or int(model_cache.get("cache_bytes") or 0) > int(model_cache.get("cache_budget_bytes") or 0)
                or int(model_cache.get("cache_available_growth_bytes") or 0) <= 0
            )
        )
        _capacity_ok, background_evictions, _capacity_message = self.ensure_model_cache_capacity(
            target_model=likely_model,
            required_growth_bytes=0,
            protected_models=protected_models,
            hot_models=hot_models,
            budget=self.cache_budget_payload(cache_bytes=int(model_cache.get("cache_bytes") or 0)),
        )
        if background_evictions:
            last_evicted_model, last_evicted_bytes = background_evictions[-1]
            total_evicted_bytes = sum(evicted_bytes for _model, evicted_bytes in background_evictions)
            self.model_cache_state.last_checked_at = now_iso()
            self.model_cache_state.last_evicted_model = last_evicted_model
            self.model_cache_state.last_evicted_bytes = total_evicted_bytes
            self.model_cache_state.last_evicted_at = self.model_cache_state.last_checked_at
            self.model_cache_state.last_result = (
                f"Evicted {format_bytes(total_evicted_bytes)} of colder model cache to stay within this node's local cache budget."
            )
            self.model_cache_state.last_error = None
            self.save_state()
            self.log(self.model_cache_state.last_result)
            return True, self.model_cache_state.last_result, "model_cache_eviction"

        active_assignments = coerce_nonnegative_int(remote_node.get("active_assignments"))
        can_prewarm = (
            health.get("runtime_healthy")
            and likely_model
            and current_model != likely_model
            and model_cache.get("likely_model_source") in {
                "control_plane_demand",
                "bootstrap_target",
                "recent_assignment_mix",
                "owner_target_heat",
            }
            and active_assignments == 0
            and not self.update_state.pending_restart
            and not bool(health.get("installer_state", {}).get("busy"))
        )
        if can_prewarm:
            required_growth_bytes = max(
                0,
                int(model_cache.get("expected_bytes") or 0) - int(model_cache.get("likely_model_cache_bytes") or 0),
            )
            capacity_ok, prewarm_evictions, capacity_message = self.ensure_model_cache_capacity(
                target_model=likely_model,
                required_growth_bytes=required_growth_bytes,
                protected_models=protected_models,
                hot_models=hot_models,
                budget=self.cache_budget_payload(cache_bytes=int(model_cache.get("cache_bytes") or 0)),
            )
            if prewarm_evictions:
                last_evicted_model, _last_evicted_bytes = prewarm_evictions[-1]
                self.model_cache_state.last_evicted_model = last_evicted_model
                self.model_cache_state.last_evicted_bytes = sum(
                    evicted_bytes for _model, evicted_bytes in prewarm_evictions
                )
                self.model_cache_state.last_evicted_at = now_iso()
            if not capacity_ok and capacity_message:
                self.model_cache_state.last_error = None
                self.model_cache_state.last_result = capacity_message
                self.model_cache_state.last_checked_at = now_iso()
                self.save_state()
                self.log(capacity_message)
                return True, capacity_message, "model_cache_budget_hold"
            try:
                applied, message = self.prewarm_likely_model(
                    likely_model,
                    current_model,
                    source=str(model_cache.get("likely_model_source") or "control_plane_demand"),
                )
            except Exception as error:
                message = f"Model pre-warm could not finish automatically: {error}"
                self.model_cache_state.last_error = message
                self.model_cache_state.last_result = message
                self.model_cache_state.last_checked_at = now_iso()
                self.save_state()
                self.log(message)
                return False, message, "model_cache_prewarm_failed"
            self.model_cache_state.last_checked_at = now_iso()
            self.model_cache_state.last_result = message
            self.model_cache_state.last_error = None
            self.save_state()
            return applied, message, "model_cache_prewarm" if applied else "model_cache_monitor"

        self.model_cache_state.last_checked_at = now_iso()
        self.model_cache_state.last_result = str(model_cache.get("detail") or "Model cache is being monitored.")
        self.model_cache_state.last_error = None
        self.save_state()
        return False, self.model_cache_state.last_result, "model_cache_monitor"

    def idle_reason_payload(
        self,
        *,
        health: dict[str, Any],
        runtime: dict[str, Any],
        config: dict[str, Any],
        updates: dict[str, Any],
        autopilot: dict[str, Any],
        heat_governor: dict[str, Any],
        remote: dict[str, Any],
        remote_summary: dict[str, Any],
        remote_node: dict[str, Any],
        remote_observability: dict[str, Any],
        blocked_reason: str | None,
    ) -> dict[str, str]:
        issue_code = str(health.get("issue_code") or "")
        remote_status = str(remote_node.get("status") or "")
        approval_status = str(remote_node.get("approval_status") or "")
        queue_depth = coerce_nonnegative_int(remote_node.get("queue_depth"))
        active_assignments = coerce_nonnegative_int(remote_node.get("active_assignments"))
        schedulable_raw = remote_summary.get("schedulable")
        if isinstance(schedulable_raw, bool):
            schedulable = schedulable_raw
        else:
            schedulable = bool(remote_observability.get("schedulable"))
        premium_status = str(config.get("premium_eligibility_status") or "")
        premium_detail = str(
            config.get("premium_eligibility_detail")
            or "Community capacity stays enabled, but premium jobs are unavailable on this machine right now."
        )
        premium_unavailable = premium_status in {"community_enabled", "premium_unavailable"}
        routing_lane_label = str(
            config.get("routing_lane_label")
            or runtime.get("routing_lane_label")
            or "current routing"
        ).strip()
        remote_error = remote.get("last_error") if isinstance(remote.get("last_error"), str) else None
        heat_plan = heat_governor.get("plan") if isinstance(heat_governor.get("plan"), dict) else {}
        autopilot_pending_restart = bool(
            autopilot.get("status") == "restart_pending"
            or (
                isinstance(autopilot.get("recommendation"), dict)
                and autopilot["recommendation"].get("pending_restart") is True
            )
        )

        if updates.get("pending_restart") or autopilot_pending_restart:
            detail = str(
                updates.get("last_result")
                or "A staged runtime change is waiting for a restart before this machine can use it."
            )
            return {
                "value": "Update staged, restart pending",
                "detail": detail,
                "tone": "warning",
            }

        waiting_for_approval = (
            issue_code == "approval_required"
            or approval_status == "pending"
            or remote_status == "pending_attestation"
            or runtime.get("stage") in {"ready", "registration_required"}
        )
        if waiting_for_approval or (blocked_reason and "approval" in blocked_reason.lower()):
            detail = str(
                blocked_reason
                or runtime.get("message")
                or health.get("issue_detail")
                or "Open the approval page to finish bringing this machine online."
            )
            return {
                "value": "Waiting for approval",
                "detail": detail,
                "tone": "warning",
            }

        if active_assignments <= 0 and bool(heat_plan.get("paused")):
            pause_reason = str(heat_plan.get("pause_reason") or "")
            reason = str(heat_plan.get("reason") or "The owner heat governor paused new assignments.")
            if pause_reason == "gpu_temperature_limit":
                return {
                    "value": "Thermal protection paused new work",
                    "detail": reason,
                    "tone": "danger",
                }
            return {
                "value": "Owner heat target paused new work",
                "detail": reason,
                "tone": "warning",
            }

        if (
            active_assignments <= 0
            and queue_depth > 0
            and bool(heat_plan.get("quiet_hours_active"))
            and coerce_nonnegative_int(heat_plan.get("effective_target_pct")) <= 20
        ):
            return {
                "value": "Quiet hours are holding a low-noise trickle",
                "detail": str(
                    heat_plan.get("reason")
                    or "Quiet hours are active, so the node is staying in a low-noise trickle mode."
                ),
                "tone": "warning",
            }

        if remote_status in {"paused", "revoked"} and blocked_reason:
            return {
                "value": "Action needed",
                "detail": blocked_reason,
                "tone": "danger",
            }

        if active_assignments > 0:
            detail = f"This machine is serving {active_assignments} assignment"
            if active_assignments != 1:
                detail += "s"
            detail += " right now."
            if queue_depth > active_assignments:
                detail += f" {queue_depth} matching jobs are still waiting in the network queue."
            return {
                "value": "Serving jobs right now",
                "detail": detail,
                "tone": "success",
            }

        if premium_unavailable and queue_depth > 0 and schedulable and health.get("runtime_healthy"):
            return {
                "value": "Premium jobs unavailable on this machine",
                "detail": premium_detail,
                "tone": "warning",
            }

        if health.get("runtime_healthy") and schedulable:
            detail = "This machine is healthy and online. Work will start automatically when a matching job arrives."
            if queue_depth > 0:
                detail = f"The network has work, but nothing currently matches the {routing_lane_label.lower()} lane on this machine."
            elif premium_unavailable:
                detail = (
                    f"{premium_detail} Community jobs will still start automatically when they match this machine."
                )
            return {
                "value": "No matching jobs right now",
                "detail": detail,
                "tone": "success",
            }

        if blocked_reason:
            return {
                "value": "Action needed",
                "detail": blocked_reason,
                "tone": "warning",
            }

        if remote_error and not remote_summary:
            return {
                "value": "Waiting for control plane sync",
                "detail": "The local service is online, but the control-plane summary is temporarily unavailable.",
                "tone": "warning",
            }

        if health.get("issue_detail"):
            return {
                "value": "Still starting up",
                "detail": str(health.get("issue_detail")),
                "tone": "warning",
            }

        return {
            "value": "Checking node activity",
            "detail": "The local service is still collecting enough state to explain this machine's idle time.",
            "tone": "warning",
        }

    def change_summary_payload(self, *, updates: dict[str, Any], autopilot: dict[str, Any]) -> dict[str, str]:
        def describe_with_time(detail: str, timestamp: str | None) -> str:
            if not timestamp:
                return detail
            relative = format_relative_time(timestamp)
            if relative == "Waiting":
                return detail
            return f"{detail} Last change {relative}."

        if updates.get("pending_restart"):
            detail = str(
                updates.get("last_result") or "A signed runtime update was downloaded and is waiting for a restart."
            )
            return {
                "value": "Signed update staged",
                "detail": describe_with_time(detail, updates.get("last_checked_at")),
                "tone": "warning",
            }

        if updates.get("last_error"):
            return {
                "value": "Update check needs attention",
                "detail": describe_with_time(str(updates["last_error"]), updates.get("last_checked_at")),
                "tone": "warning",
            }

        self_heal_action = str(self.self_heal_state.last_action or "")
        if self_heal_action and self_heal_action not in {"monitor", "waiting_for_quick_start"}:
            change_value = {
                "autopilot_tuning": "Autopilot retuned this machine",
                "start_runtime": "Runtime started",
                "restart_runtime": "Runtime restarted",
                "restart_vllm": "Inference runtime restarted",
                "restart_node_agent": "Node agent restarted",
                "rollback_bad_update": "Rolled back to the last healthy release",
                "rollback_owner_target_model": "Bootstrap model restored",
                "resume_quick_start": "Quick Start resumed",
                "waiting_for_approval": "Waiting for approval",
            }.get(self_heal_action, "Local runtime changed")
            change_tone = "warning" if self_heal_action in {"waiting_for_approval", "resume_quick_start"} else "success"
            return {
                "value": change_value,
                "detail": describe_with_time(
                    str(self.self_heal_state.last_result or "The local runtime changed."),
                    self.self_heal_state.last_repaired_at or self.self_heal_state.last_checked_at,
                ),
                "tone": change_tone,
            }

        history = autopilot.get("history") if isinstance(autopilot.get("history"), list) else []
        last_history = history[-1] if history and isinstance(history[-1], dict) else {}
        if last_history:
            target_model = str(last_history.get("startup_model") or "the current startup model")
            reason = str(last_history.get("reason") or "Autopilot updated the machine plan.")
            return {
                "value": "Autopilot updated the machine plan",
                "detail": describe_with_time(f"{reason} Target model: {target_model}.", last_history.get("at")),
                "tone": "success",
            }

        update_result = str(updates.get("last_result") or "")
        if update_result and update_result not in {
            "No update checks have run yet.",
            "Runtime already matches signed release None.",
        }:
            lowered = update_result.lower()
            if "applied successfully" in lowered:
                change_value = "Signed runtime updated"
            elif "rolled back" in lowered or "restored" in lowered:
                change_value = "Signed runtime restored"
            else:
                change_value = "No recent changes"
            if change_value != "No recent changes":
                return {
                    "value": change_value,
                    "detail": describe_with_time(update_result, updates.get("last_checked_at")),
                    "tone": "success",
                }

        return {
            "value": "No recent changes",
            "detail": "This machine is still running with its current local plan.",
            "tone": "success",
        }

    def dashboard_alerts_payload(
        self,
        *,
        config: dict[str, Any],
        connectivity: dict[str, Any],
        heat_governor: dict[str, Any],
        current_model: str | None,
    ) -> list[dict[str, Any]]:
        def describe_with_time(detail: str, timestamp: str | None) -> str:
            if not timestamp:
                return detail
            relative = format_relative_time(timestamp)
            if relative == "Waiting":
                return detail
            return f"{detail} Last change {relative}."

        alerts: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_alert(
            *,
            code: str,
            title: str,
            detail: str,
            tone: str = "warning",
            source: str,
            observed_at: str | None = None,
        ) -> None:
            normalized_code = str(code or "").strip() or title.strip().lower().replace(" ", "_")
            dedupe_key = normalized_code or title.strip()
            if not title.strip() or not detail.strip() or dedupe_key in seen:
                return
            alert: dict[str, Any] = {
                "code": normalized_code,
                "title": title.strip(),
                "detail": detail.strip(),
                "tone": tone if tone in {"success", "warning", "danger"} else "warning",
                "source": source.strip() or "service",
            }
            if observed_at:
                alert["observed_at"] = observed_at
                relative = format_relative_time(observed_at)
                if relative != "Waiting":
                    alert["observed_label"] = relative
            seen.add(dedupe_key)
            alerts.append(alert)

        heat_plan = heat_governor.get("plan") if isinstance(heat_governor.get("plan"), dict) else {}
        for alert in heat_plan.get("owner_alerts", []) if isinstance(heat_plan.get("owner_alerts"), list) else []:
            if not isinstance(alert, dict):
                continue
            add_alert(
                code=str(alert.get("code") or "heat_governor"),
                title=str(alert.get("title") or "Heat governor alert"),
                detail=str(alert.get("detail") or heat_plan.get("reason") or "The heat governor has an update."),
                tone=str(alert.get("tone") or "warning"),
                source=str(alert.get("source") or "heat_governor"),
                observed_at=str(heat_governor.get("last_observed_at") or heat_governor.get("updated_at") or ""),
            )

        local_doctor_alert = (
            self.local_doctor_state.last_transition_alert
            if isinstance(self.local_doctor_state.last_transition_alert, dict)
            else {}
        )
        if local_doctor_alert:
            add_alert(
                code=str(local_doctor_alert.get("code") or "local_doctor"),
                title=str(local_doctor_alert.get("title") or "Local Doctor update"),
                detail=str(local_doctor_alert.get("detail") or "Local Doctor has a new update."),
                tone=str(local_doctor_alert.get("tone") or "warning"),
                source=str(local_doctor_alert.get("source") or "local_doctor"),
                observed_at=str(local_doctor_alert.get("observed_at") or ""),
            )

        connectivity_status = str(connectivity.get("status") or "")
        if connectivity_status in {"degraded", "offline"} and bool(connectivity.get("fallback_active")):
            add_alert(
                code="fallback_connectivity",
                title="Running in degraded mode on fallback connectivity",
                detail=str(
                    connectivity.get("detail")
                    or "The primary control-plane path is unstable, so the node is staying online on a fallback path."
                ),
                tone="danger" if connectivity_status == "offline" and not connectivity.get("grace_active") else "warning",
                source="connectivity",
            )

        owner_target_model = str(config.get("owner_target_model") or "").strip()
        bootstrap_pending_upgrade = bool(
            coerce_bool(config.get("bootstrap_pending_upgrade"), False)
            and owner_target_model
            and current_model
            and owner_target_model != current_model
        )
        if bootstrap_pending_upgrade:
            add_alert(
                code="bootstrap_target_warming",
                title="Bootstrap model is serving while target model warms",
                detail=(
                    f"{current_model} is serving so this node stays online quickly while {owner_target_model} finishes "
                    "its background warm-up."
                ),
                tone="warning",
                source="model_lifecycle",
            )
        elif str(self.self_heal_state.last_action or "") == "rollback_owner_target_model":
            target_model = owner_target_model or "the owner target model"
            add_alert(
                code="bootstrap_target_warming",
                title="Bootstrap model is serving while target model warms",
                detail=describe_with_time(
                    str(
                        self.self_heal_state.last_result
                        or f"The node restored its bootstrap model while {target_model} cools down and retries later."
                    ),
                    self.self_heal_state.last_repaired_at or self.self_heal_state.last_checked_at,
                ),
                tone="warning",
                source="self_healing",
                observed_at=self.self_heal_state.last_repaired_at or self.self_heal_state.last_checked_at,
            )

        self_heal_action = str(self.self_heal_state.last_action or "")
        self_heal_issue = str(self.self_heal_state.last_issue or "")
        self_heal_result = str(self.self_heal_state.last_result or "")
        recovery_text = f"{self_heal_issue} {self_heal_result}".lower()
        if self_heal_action in {"restart_runtime", "restart_vllm", "restart_node_agent"} and (
            "docker" in recovery_text or "container" in recovery_text
        ):
            add_alert(
                code="docker_restart_recovered",
                title="Node recovered automatically after Docker restart",
                detail=describe_with_time(
                    self_heal_result or "The node restarted the affected runtime layer and returned to its last known healthy local plan.",
                    self.self_heal_state.last_repaired_at or self.self_heal_state.last_checked_at,
                ),
                tone="success",
                source="self_healing",
                observed_at=self.self_heal_state.last_repaired_at or self.self_heal_state.last_checked_at,
            )

        priority = {"danger": 0, "warning": 1, "success": 2}
        alerts.sort(key=lambda alert: (priority.get(str(alert.get("tone") or "warning"), 3), str(alert.get("title") or "")))
        return alerts

    def heat_usefulness_ratio(self, *, heat_plan: dict[str, Any]) -> tuple[float, str]:
        if bool(heat_plan.get("paused")):
            return 0.0, "Heat output is paused right now."

        objective = normalize_owner_objective(heat_plan.get("owner_objective"))
        effective_target_pct = coerce_nonnegative_int(heat_plan.get("effective_target_pct"))
        if effective_target_pct <= 20:
            ratio = 0.35
        elif effective_target_pct <= 50:
            ratio = 0.70
        else:
            ratio = 1.0

        room_temp_c = coerce_float(heat_plan.get("room_temp_c"))
        target_temp_c = coerce_float(heat_plan.get("target_temp_c"))
        outside_temp_c = coerce_float(heat_plan.get("outside_temp_c"))

        reason = "No room or weather hint is configured yet, so this uses a balanced default."
        if room_temp_c is not None and target_temp_c is not None:
            gap_c = target_temp_c - room_temp_c
            if gap_c <= -0.3:
                ratio = 0.0
                reason = "The room is already above the owner target, so extra heat is unlikely to be useful."
            elif gap_c <= 0.2:
                ratio = min(ratio, 0.40)
                reason = "The room is already near the target, so only part of the current heat output is likely to be useful."
            elif gap_c <= 1.0:
                ratio = max(ratio, 0.75)
                reason = f"The room is about {round(gap_c, 1)} C below target, so most of the heat output is still useful."
            else:
                ratio = 1.0
                reason = f"The room is about {round(gap_c, 1)} C below target, so the heat output is fully useful."
        elif outside_temp_c is not None:
            if outside_temp_c >= 18:
                ratio = min(ratio, 0.20)
                reason = f"The outside-temperature hint is {outside_temp_c:.0f} C, so only a small part of the heat is likely to be useful."
            elif outside_temp_c >= 12:
                ratio = min(ratio, 0.50)
                reason = f"The outside-temperature hint is {outside_temp_c:.0f} C, so heat value is moderate."
            elif outside_temp_c <= 5:
                ratio = max(ratio, 0.85)
                reason = f"The outside-temperature hint is {outside_temp_c:.0f} C, so most of the heat is likely to be useful."

        if objective == "heat_first":
            ratio = min(1.0, ratio + 0.15)
            reason = f"{reason} Heat-first mode nudges the estimate upward."
        elif objective == "earnings_only":
            ratio = max(0.0, ratio - 0.25)
            reason = f"{reason} Earnings-only mode treats less of the heat as owner-value."

        return round(ratio, 3), reason

    def economics_payload(
        self,
        *,
        heat_governor: dict[str, Any],
        autopilot: dict[str, Any],
        remote_summary: dict[str, Any],
    ) -> dict[str, Any]:
        plan = heat_governor.get("plan") if isinstance(heat_governor.get("plan"), dict) else {}
        signals = autopilot.get("signals") if isinstance(autopilot.get("signals"), dict) else {}
        earnings = remote_summary.get("earnings") if isinstance(remote_summary.get("earnings"), dict) else {}

        today_earnings_usd: float | None = None
        today_source = ""
        today_source_key = ""
        today_source_confidence = "low"
        today_source_detail = "No control-plane earnings source is available yet."
        for key, label, confidence, detail in (
            (
                "today_usd",
                "today",
                "high",
                "Using control-plane today_usd directly for today's earnings.",
            ),
            (
                "last_24h_usd",
                "last 24h",
                "medium",
                "Using rolling last_24h_usd because today_usd is not available yet.",
            ),
            (
                "current_hour_usd",
                "current hour",
                "low",
                "Using current_hour_usd because longer-window earnings data is not available yet.",
            ),
        ):
            value = coerce_float(earnings.get(key))
            if value is not None:
                today_earnings_usd = value
                today_source = label
                today_source_key = key
                today_source_confidence = confidence
                today_source_detail = detail
                break
        if today_earnings_usd is None:
            today_earnings_usd = 0.0

        power_watts = None
        power_source = ""
        power_source_key = ""
        power_source_confidence = "low"
        power_source_detail = "No power source is available yet."
        for raw_value, key, label, confidence, detail in (
            (
                signals.get("power_watts"),
                "power_watts",
                "live GPU watts",
                "high",
                "Measured from local GPU telemetry.",
            ),
            (
                signals.get("estimated_heat_output_watts"),
                "estimated_heat_output_watts",
                "estimated heat telemetry",
                "medium",
                "Derived from runtime heat telemetry because live GPU watts are unavailable.",
            ),
            (
                plan.get("desired_power_limit_watts"),
                "desired_power_limit_watts",
                "capped estimate",
                "low",
                (
                    f"Using the current {format_watts(plan.get('desired_power_limit_watts'))} power cap because "
                    "live GPU watts are unavailable."
                ),
            ),
        ):
            numeric = coerce_float(raw_value)
            if numeric is not None and numeric > 0:
                power_watts = numeric
                power_source = label
                power_source_key = key
                power_source_confidence = confidence
                power_source_detail = detail
                break

        energy_price_kwh = coerce_float(
            heat_governor.get("energy_price_kwh")
            if heat_governor.get("energy_price_kwh") is not None
            else signals.get("energy_price_kwh")
        )

        local_now = datetime.now().astimezone()
        service_started = parse_iso_timestamp(self.started_at) or datetime.fromtimestamp(
            self.started_at_epoch,
            tz=timezone.utc,
        )
        if service_started.tzinfo is None:
            service_started = service_started.replace(tzinfo=timezone.utc)
        service_started_local = service_started.astimezone(local_now.tzinfo)
        local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        basis_start = max(local_midnight, service_started_local)
        basis_hours = max(0.0, (local_now - basis_start).total_seconds() / 3600.0)
        basis_label = "local midnight" if basis_start == local_midnight else "since service came online"
        basis_confidence = "high"
        basis_detail = (
            "Using a local midnight basis because this runtime was already online when the local day started."
            if basis_start == local_midnight
            else "Using a since-service-came-online basis because this runtime started after local midnight."
        )

        electricity_cost_usd: float | None = None
        if power_watts is not None and energy_price_kwh is not None:
            electricity_cost_usd = (power_watts / 1000.0) * basis_hours * energy_price_kwh

        heat_ratio, heat_reason = self.heat_usefulness_ratio(heat_plan=plan)
        room_temp_c = coerce_float(plan.get("room_temp_c"))
        target_temp_c = coerce_float(plan.get("target_temp_c"))
        outside_temp_c = coerce_float(plan.get("outside_temp_c"))
        if room_temp_c is not None and target_temp_c is not None and outside_temp_c is not None:
            heat_assumption_source = "room target plus outside-temperature hint"
            heat_assumption_confidence = "high"
        elif room_temp_c is not None and target_temp_c is not None:
            heat_assumption_source = "room target"
            heat_assumption_confidence = "high"
        elif outside_temp_c is not None:
            heat_assumption_source = "outside-temperature hint"
            heat_assumption_confidence = "medium"
        else:
            heat_assumption_source = "balanced default"
            heat_assumption_confidence = "low"
        heat_usefulness_pct = round(heat_ratio * 100.0, 1)
        heat_offset_usd = None if electricity_cost_usd is None else electricity_cost_usd * heat_ratio
        net_value_usd = (
            None
            if electricity_cost_usd is None or heat_offset_usd is None
            else today_earnings_usd + heat_offset_usd - electricity_cost_usd
        )

        if today_source_key == "today_usd":
            today_detail = "Using control-plane today_usd directly for today's earnings. Confidence: high."
        elif today_source_key == "last_24h_usd":
            today_detail = (
                "Using rolling last_24h_usd because today_usd is not available yet, so this is an estimate instead of "
                "a local-day total. Confidence: medium."
            )
        elif today_source_key == "current_hour_usd":
            today_detail = (
                "Using current_hour_usd because longer-window earnings are not available yet, so this is a short-window "
                "estimate. Confidence: low."
            )
        else:
            today_detail = "No control-plane earnings have been reported yet."

        if energy_price_kwh is None:
            electricity_card = {
                "value": "Add power price",
                "detail": "Enter electricity price in USD/kWh so this node can estimate cost, heat offset, and net value.",
                "tone": "warning",
            }
            heat_offset_card = {
                "value": "Add power price",
                "detail": "Heat offset needs the local electricity price to value the recovered heat.",
                "tone": "warning",
            }
            net_value_card = {
                "value": "Add power price",
                "detail": "Net value appears after the local electricity price is saved.",
                "tone": "warning",
            }
        elif power_watts is None:
            electricity_card = {
                "value": "Waiting for power telemetry",
                "detail": "The node has not reported live GPU power yet, so electricity and heat-value estimates are waiting.",
                "tone": "warning",
            }
            heat_offset_card = {
                "value": "Waiting for power telemetry",
                "detail": "Heat offset appears after the node reports live or capped GPU power.",
                "tone": "warning",
            }
            net_value_card = {
                "value": "Waiting for power telemetry",
                "detail": "Net value appears after the node reports GPU power telemetry.",
                "tone": "warning",
            }
        else:
            electricity_card = {
                "value": format_usd(electricity_cost_usd),
                "detail": (
                    f"Estimated from {format_watts(power_watts)} at ${energy_price_kwh:.2f}/kWh. "
                    f"Source: {power_source} ({power_source_key}, {power_source_confidence} confidence). "
                    f"Basis: {basis_label}. {power_source_detail}"
                ),
                "tone": "warning" if electricity_cost_usd and electricity_cost_usd > 0 else "success",
            }
            heat_offset_card = {
                "value": format_usd(heat_offset_usd),
                "detail": (
                    f"Assumes {heat_usefulness_pct:.0f}% of current heat output offsets household heating. "
                    f"Source: {heat_assumption_source} ({heat_assumption_confidence} confidence). {heat_reason}"
                ),
                "tone": "success" if heat_offset_usd and heat_offset_usd > 0 else "warning",
            }
            net_value_card = {
                "value": format_usd(net_value_usd),
                "detail": (
                    f"Combines {today_source_key or 'no control-plane earnings source'} earnings with electricity on a "
                    f"{basis_label} basis. Power source confidence: {power_source_confidence}. "
                    f"Heat assumptions: {heat_assumption_source} ({heat_assumption_confidence} confidence)."
                ),
                "tone": "success" if net_value_usd is not None and net_value_usd >= 0 else "warning",
            }

        return {
            "summary": {
                "today_earnings_usd": today_earnings_usd,
                "today_source": today_source or None,
                "today_source_key": today_source_key or None,
                "today_source_confidence": today_source_confidence if today_source_key else None,
                "today_source_detail": today_source_detail,
                "power_watts": power_watts,
                "power_source": power_source or None,
                "power_source_key": power_source_key or None,
                "power_source_confidence": power_source_confidence if power_source_key else None,
                "power_source_detail": power_source_detail,
                "basis_label": basis_label,
                "basis_hours": round(basis_hours, 3),
                "basis_detail": basis_detail,
                "basis_confidence": basis_confidence,
                "basis_started_at": basis_start.isoformat(),
                "energy_price_kwh": energy_price_kwh,
                "heat_usefulness_ratio": heat_ratio,
                "heat_usefulness_pct": heat_usefulness_pct,
                "heat_usefulness_detail": heat_reason,
                "heat_assumption_source": heat_assumption_source,
                "heat_assumption_confidence": heat_assumption_confidence,
                "electricity_cost_usd": electricity_cost_usd,
                "heat_offset_usd": heat_offset_usd,
                "net_value_usd": net_value_usd,
            },
            "cards": {
                "today_earnings": {
                    "value": format_usd(today_earnings_usd),
                    "detail": today_detail,
                    "tone": "success" if today_earnings_usd > 0 else "warning",
                },
                "electricity": electricity_card,
                "heat_offset": heat_offset_card,
                "net_value": net_value_card,
            },
        }

    def apply_autopilot_tuning(self) -> tuple[bool, str]:
        state = self.load_autopilot_state()
        if not state:
            return False, "Autopilot has not produced a tuning recommendation yet."
        recommendation = state.get("recommendation") if isinstance(state.get("recommendation"), dict) else {}
        env_updates = recommendation.get("env_updates") if isinstance(recommendation.get("env_updates"), dict) else {}
        if not env_updates:
            return False, "Autopilot does not have env changes to apply yet."
        if recommendation.get("safe_to_apply") is False:
            return False, "Autopilot is waiting until the current assignment is finished before applying changes."

        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            return False, "Autopilot is waiting for local runtime settings to exist."
        next_env = dict(env_values)
        changed = False
        for key, value in env_updates.items():
            if key not in {"SETUP_PROFILE", "MAX_CONCURRENT_ASSIGNMENTS", "THERMAL_HEADROOM", "SUPPORTED_MODELS", "VLLM_MODEL", "RUNTIME_PROFILE"}:
                continue
            next_value = str(value)
            if next_env.get(key) != next_value:
                next_env[key] = next_value
                changed = True
        if not changed:
            recommendation["pending_restart"] = False
            recommendation["applied_at"] = now_iso()
            state["recommendation"] = recommendation
            self.save_autopilot_state(state)
            return False, "Autopilot tuning already matches the local runtime config."

        model_changed = env_values.get("VLLM_MODEL") != next_env.get("VLLM_MODEL")
        self.guided_installer.write_runtime_settings(next_env)
        self.guided_installer.write_runtime_env(next_env)
        recommendation["applied_at"] = now_iso()
        recommendation["pending_restart"] = False
        state["recommendation"] = recommendation
        self.save_autopilot_state(state)

        if model_changed:
            self.log("Autopilot is restarting the runtime to apply a safer startup model.")
            self.restart_runtime()
            return True, f"Autopilot switched the startup model to {next_env.get('VLLM_MODEL')} and restarted the runtime."

        return True, "Autopilot applied updated concurrency and thermal tuning for the next runtime cycle."

    def dashboard_payload(
        self,
        health: dict[str, Any],
        installer_snapshot: dict[str, Any],
        *,
        heat_governor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = health.get("runtime", {})
        installer_state = health.get("installer_state", {})
        config = installer_snapshot.get("config", {})
        updates = asdict(self.update_state)
        autopilot = self.autopilot_payload()
        autopilot_state = self.load_autopilot_payload()
        heat_governor = heat_governor or self.heat_governor_payload(autopilot_payload=autopilot_state)
        remote = self.remote_dashboard_snapshot()
        remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
        remote_node = remote_summary.get("node") if isinstance(remote_summary.get("node"), dict) else {}
        remote_observability = (
            remote_node.get("observability") if isinstance(remote_node.get("observability"), dict) else {}
        )
        remote_runtime = remote_node.get("runtime") if isinstance(remote_node.get("runtime"), dict) else {}
        earnings = remote_summary.get("earnings") if isinstance(remote_summary.get("earnings"), dict) else {}
        setup_verification = (
            remote_summary.get("setup_verification")
            if isinstance(remote_summary.get("setup_verification"), dict)
            else {}
        )
        connectivity = self.control_plane_connectivity_payload(remote=remote)
        model_cache = self.model_cache_payload(
            health=health,
            runtime=runtime,
            config=config,
            remote_summary=remote_summary,
            remote_node=remote_node,
            autopilot=autopilot_state,
        )

        blocked_reason = (
            remote_summary.get("blocked_reason")
            if isinstance(remote_summary.get("blocked_reason"), str)
            else remote_observability.get("schedulability_reason")
        )
        if not isinstance(blocked_reason, str) or not blocked_reason.strip():
            blocked_reason = None
        idle_reason = self.idle_reason_payload(
            health=health,
            runtime=runtime,
            config=config,
            updates=updates,
            autopilot=autopilot,
            heat_governor=heat_governor,
            remote=remote,
            remote_summary=remote_summary,
            remote_node=remote_node,
            remote_observability=remote_observability,
            blocked_reason=blocked_reason,
        )
        change_summary = self.change_summary_payload(updates=updates, autopilot=autopilot)
        economics = self.economics_payload(
            heat_governor=heat_governor,
            autopilot=autopilot,
            remote_summary=remote_summary,
        )

        heartbeat_at = remote_node.get("last_heartbeat_at") if isinstance(remote_node.get("last_heartbeat_at"), str) else None
        heartbeat_age_seconds: float | None = None
        heartbeat_timestamp = parse_iso_timestamp(heartbeat_at)
        if heartbeat_timestamp is not None:
            heartbeat_age_seconds = max(0.0, (datetime.now(timezone.utc) - heartbeat_timestamp).total_seconds())

        current_model = (
            runtime.get("current_model")
            or remote_runtime.get("current_model")
            or config.get("vllm_model")
            or config.get("recommended_model")
        )
        if current_model:
            current_model = str(current_model)

        setup_verification_status = (
            str(setup_verification.get("status") or "").strip()
            if isinstance(setup_verification.get("status"), str)
            else ""
        )
        setup_verification_detail = (
            str(setup_verification.get("detail") or "").strip()
            if isinstance(setup_verification.get("detail"), str)
            else ""
        )
        setup_verification_notification = None
        if setup_verification_status == "passed":
            setup_verification_notification = {
                "show": True,
                "title": "Setup verified",
                "message": (
                    setup_verification_detail
                    or "A tiny end-to-end canary completed through edge.autonomousc.com, so this setup is verified."
                ),
                "tone": "success",
            }

        accrued_value = format_usd(earnings.get("accrued_usd"))
        transferred_value = format_usd(earnings.get("transferred_usd"))
        last_payout = earnings.get("last_payout") if isinstance(earnings.get("last_payout"), dict) else None
        last_payout_detail = ""
        if last_payout is not None:
            payout_amount = (
                format_usd(last_payout.get("amount", {}).get("amount"))
                if isinstance(last_payout.get("amount"), dict)
                else "$0.00"
            )
            payout_status = str(last_payout.get("status") or "pending").replace("_", " ")
            last_payout_detail = f" Last payout {payout_amount} is {payout_status}."

        if health.get("runtime_healthy") and not blocked_reason:
            if connectivity.get("status") in {"degraded", "offline"}:
                health_value = str(connectivity.get("value") or "Degraded mode")
                health_tone = str(connectivity.get("tone") or "warning")
                health_detail = str(connectivity.get("detail") or "The node is serving through a degraded network path.")
            elif setup_verification_status == "passed":
                health_value = "Setup verified"
                health_tone = "success"
                health_detail = (
                    setup_verification_detail
                    or "A tiny end-to-end canary completed through edge.autonomousc.com, so this setup is verified."
                )
            elif setup_verification_status in {"pending", "not_started"}:
                health_value = "Verifying setup"
                health_tone = "warning"
                health_detail = (
                    setup_verification_detail
                    or "A tiny end-to-end canary is still running through edge.autonomousc.com."
                )
            elif setup_verification_status in {"failed", "mismatch"}:
                health_value = "Needs attention"
                health_tone = "danger"
                health_detail = (
                    setup_verification_detail
                    or "Setup verification did not finish successfully. Use Fix it, then retry setup."
                )
            else:
                health_value = "Healthy"
                health_tone = "success"
                health_detail = "This machine is online, healthy, and ready for work."
        elif installer_state.get("busy"):
            health_value = "Setting up"
            health_tone = "warning"
            health_detail = str(installer_state.get("message") or "Quick Start is still running.")
        elif blocked_reason:
            remote_status = str(remote_node.get("status") or "")
            health_value = "Blocked"
            health_tone = "danger" if remote_status in {"paused", "revoked"} else "warning"
            health_detail = blocked_reason
        elif health.get("issue_detail"):
            health_value = "Needs attention"
            health_tone = "danger"
            health_detail = str(health["issue_detail"])
        elif runtime.get("stage") in {"registration_required", "ready"}:
            health_value = "Waiting"
            health_tone = "warning"
            health_detail = str(runtime.get("message") or "This machine still needs approval before it can serve work.")
        else:
            health_value = "Starting"
            health_tone = "warning"
            health_detail = str(runtime.get("message") or "The runtime is still coming online.")

        if remote:
            remote_issue = remote.get("last_error")
            if isinstance(remote_issue, str) and remote_issue and not remote.get("summary"):
                health_detail = f"{health_detail} Control plane sync is unavailable right now."

        if earnings:
            earnings_value = f"{accrued_value} pending"
            earnings_detail = f"{transferred_value} transferred lifetime.{last_payout_detail}".strip()
            earnings_tone = "success"
        elif isinstance(remote.get("last_error"), str) and remote["last_error"]:
            earnings_value = "Sync unavailable"
            earnings_detail = str(remote["last_error"])
            earnings_tone = "warning"
        else:
            earnings_value = "$0.00 pending"
            earnings_detail = "Claim this node and keep it online to start accruing earnings."
            earnings_tone = "warning"

        if heartbeat_at:
            heartbeat_value = format_relative_time(heartbeat_at)
            heartbeat_detail = (
                f"Last heartbeat at {heartbeat_at}. Queue depth {remote_node.get('queue_depth', 0)}. "
                f"Active assignments {remote_node.get('active_assignments', 0)}."
            )
            heartbeat_tone = (
                "success"
                if heartbeat_age_seconds is not None and heartbeat_age_seconds <= 120
                else "warning" if heartbeat_age_seconds is not None and heartbeat_age_seconds <= 300 else "danger"
            )
        elif isinstance(remote.get("last_error"), str) and remote["last_error"]:
            heartbeat_value = "Sync unavailable"
            heartbeat_detail = str(remote["last_error"])
            heartbeat_tone = "warning"
        else:
            heartbeat_value = "Waiting"
            heartbeat_detail = "The control plane has not recorded a heartbeat from this node yet."
            heartbeat_tone = "warning"

        if current_model:
            owner_target_model = str(config.get("owner_target_model") or "").strip()
            bootstrap_pending_upgrade = bool(
                config.get("bootstrap_pending_upgrade")
                and owner_target_model
                and owner_target_model != current_model
            )
            model_value = current_model
            if bootstrap_pending_upgrade:
                model_detail = (
                    f"{current_model} is loaded so this node could come online quickly. "
                    f"{owner_target_model} is the larger owner target and will switch in after the background warm-up finishes."
                )
            else:
                model_detail = (
                    "This model is loaded in the local runtime."
                    if runtime.get("stage") == "running"
                    else "Quick Start will use this startup model on this machine."
                )
            model_tone = "success" if runtime.get("stage") == "running" else "warning"
        else:
            model_value = "Pending"
            model_detail = "The local runtime has not selected a startup model yet."
            model_tone = "warning"

        uptime_value = format_elapsed(time.time() - self.started_at_epoch)
        uptime_detail = f"Local service online since {self.started_at}."
        uptime_tone = "success"
        alerts = self.dashboard_alerts_payload(
            config=config,
            connectivity=connectivity,
            heat_governor=heat_governor,
            current_model=current_model if isinstance(current_model, str) and current_model else None,
        )
        timeline = self.owner_timeline_payload()

        if health_value == "Setup verified":
            headline = "Setup verified"
            detail = idle_reason["detail"] or health_detail
        elif health_value == "Verifying setup":
            headline = "Verifying setup"
            detail = health_detail
        elif health_value == "Degraded mode":
            headline = "Degraded mode"
            detail = health_detail
        elif health_value == "Offline grace":
            headline = "Offline grace"
            detail = health_detail
        elif health_value == "Healthy":
            headline = "Node live"
            detail = idle_reason["detail"]
        elif health_value == "Setting up":
            headline = "Quick Start is in progress"
            detail = health_detail
        elif health_value == "Blocked":
            headline = "Node needs attention"
            detail = idle_reason["detail"]
        else:
            headline = "Owner dashboard"
            detail = idle_reason["detail"] if idle_reason["detail"] else health_detail

        return {
            "headline": headline,
            "detail": detail,
            "sync": {
                "synced_at": remote.get("synced_at"),
                "last_error": remote.get("last_error"),
                "stale": bool(remote.get("stale")),
            },
            "connectivity": connectivity,
            "cards": {
                "health": {"value": health_value, "detail": health_detail, "tone": health_tone},
                "earnings": {"value": earnings_value, "detail": earnings_detail, "tone": earnings_tone},
                "today_earnings": economics["cards"]["today_earnings"],
                "heartbeat": {"value": heartbeat_value, "detail": heartbeat_detail, "tone": heartbeat_tone},
                "model": {"value": model_value, "detail": model_detail, "tone": model_tone},
                "uptime": {"value": uptime_value, "detail": uptime_detail, "tone": uptime_tone},
                "idle": {"value": idle_reason["value"], "detail": idle_reason["detail"], "tone": idle_reason["tone"]},
                "electricity": economics["cards"]["electricity"],
                "heat_offset": economics["cards"]["heat_offset"],
                "net_value": economics["cards"]["net_value"],
                "changes": {
                    "value": change_summary["value"],
                    "detail": change_summary["detail"],
                    "tone": change_summary["tone"],
                },
                "autopilot": {
                    "value": autopilot["value"],
                    "detail": autopilot["detail"],
                    "tone": autopilot["tone"],
                },
                "model_cache": {
                    "value": model_cache["value"],
                    "detail": model_cache["detail"],
                    "tone": model_cache["tone"],
                },
            },
            "setup_verification": {
                "status": setup_verification_status or None,
                "detail": setup_verification_detail,
                "notification": setup_verification_notification,
            },
            "alerts": alerts,
            "timeline": timeline,
            "connectivity": connectivity,
            "autopilot": autopilot,
            "economics": economics["summary"],
            "model_cache": model_cache,
        }

    def status_payload(self) -> dict[str, Any]:
        installer_snapshot = self.guided_installer.status_payload()
        preflight = installer_snapshot["preflight"]
        installer_state = installer_snapshot["state"]
        health = self.runtime_health_snapshot(installer_snapshot=installer_snapshot)
        try:
            self.signed_release_manifest()
        except ReleaseManifestError:
            pass
        with self.lock:
            service_logs = list(self.logs)
        heat_governor = self.heat_governor_payload()
        dashboard = self.dashboard_payload(health, installer_snapshot, heat_governor=heat_governor)
        appliance_release = self.appliance_release_payload()
        return {
            "service": {
                "host": self.host,
                "port": self.port,
                "url": f"http://{self.host}:{self.port}",
                "logs": service_logs,
            },
            "runtime": health["runtime"],
            "installer": installer_snapshot,
            "owner_setup": installer_snapshot.get("owner_setup", {}),
            "dashboard": dashboard,
            "alerts": dashboard.get("alerts", []),
            "connectivity": dashboard.get("connectivity", {}),
            "heat_governor": heat_governor,
            "model_cache": dashboard.get("model_cache", {}),
            "autostart": self.autostart_manager.status(),
            "desktop_launcher": self.desktop_launcher_manager.status(),
            "updates": asdict(self.update_state),
            "appliance_release": appliance_release,
            "diagnostics": asdict(self.diagnostics_state),
            "local_doctor": self.local_doctor_payload(),
            "self_healing": self.self_heal_payload(health),
            "fault_drills": self.fault_drill_payload(),
        }

    def _run_docker_restart_mid_run_drill(self) -> tuple[str, str, dict[str, Any]]:
        health = self.runtime_health_snapshot()
        if not health.get("runtime_healthy"):
            return (
                "unavailable",
                "The runtime needs to be healthy before the Docker restart drill can run.",
                {"issue_code": health.get("issue_code"), "issue_detail": health.get("issue_detail")},
            )
        if self.runtime_controller is not None:
            self.runtime_controller.crash_vllm()
            injected = "crash_vllm"
        else:
            self.log("Injecting a live fault by killing the vllm container.")
            self.compose(["kill", "vllm"])
            injected = "docker_compose_kill_vllm"
        payload = self.self_heal_check()
        post_health = self.runtime_health_snapshot()
        action = self.self_heal_state.last_action or ""
        if not post_health.get("runtime_healthy"):
            raise RuntimeError(
                "The runtime did not return to a healthy state after the restart drill. "
                f"Last self-heal action: {action or 'unknown'}."
            )
        return (
            "passed",
            "Self-healing recovered after the local inference service was killed mid-run.",
            {
                "injected": injected,
                "self_heal_action": action,
                "self_heal_result": self.self_heal_state.last_result,
                "payload_status": payload.get("self_healing", {}).get("status") if isinstance(payload, dict) else None,
            },
        )

    def _run_dns_flap_drill(self) -> tuple[str, str, dict[str, Any]]:
        controller = self.fault_controller()
        body = json.dumps({"ok": True, "service": "fault-drill"}).encode("utf-8")
        with self.local_drill_server(body=body, content_type="application/json") as (server, base_url):
            controller.activate(
                "dns_flap",
                remaining_triggers=2,
                note="Live fault drill: simulate a temporary DNS resolution failure.",
            )
            settings = self.build_fault_drill_settings(edge_control_url=base_url)
            transport = EdgeControlTransport(settings)
            response = transport._request_with_retry("GET", "/drill")
            snapshot = transport.snapshot()
            payload = response.json()
        if payload.get("ok") is not True:
            raise RuntimeError("The DNS flap drill did not receive a healthy response after retries.")
        if int(snapshot.get("dns_failures") or 0) < 1:
            raise RuntimeError("The DNS flap drill did not record any DNS failures.")
        return (
            "passed",
            "The control-plane transport recovered from an injected DNS flap and recorded degraded mode correctly.",
            {
                "transport_status": snapshot.get("status"),
                "degraded_reason": snapshot.get("degraded_reason"),
                "dns_failures": snapshot.get("dns_failures"),
                "requests_served": len(server.requests),
            },
        )

    def _run_resumable_download_drill(self, fault_name: str, summary: str) -> tuple[str, str, dict[str, Any]]:
        controller = self.fault_controller()
        payload = {
            "items": [
                {
                    "batch_item_id": "drill-item-1",
                    "input": {"text": "live fault drill " * 12000},
                }
            ]
        }
        encrypted = encrypt_artifact(payload)
        with self.local_drill_server(
            body=encrypted["ciphertext"],
            content_type="application/octet-stream",
            enable_range=True,
        ) as (server, base_url):
            controller.activate(
                fault_name,
                remaining_triggers=1,
                note=f"Live fault drill: {fault_name}.",
            )
            settings = self.build_fault_drill_settings(edge_control_url=base_url)
            client = EdgeControlClient(settings)
            assignment = SimpleNamespace(
                assignment_id=f"drill-{fault_name}",
                input_artifact_url=f"{base_url}/artifact.bin",
                input_artifact_sha256=encrypted["ciphertext_sha256"],
                input_artifact_encryption=encrypted["encryption"],
                input_artifact_expires_at=None,
                input_artifact_mirror_urls=[],
            )
            decrypted = client.fetch_artifact(assignment)
            resume_dir = Path(settings.autopilot_state_path).parent / "artifact-resume"
            leftover_parts = sorted(path.name for path in resume_dir.glob("*.part")) if resume_dir.exists() else []
        range_requests = [request.get("range") for request in server.requests if request.get("range")]
        if decrypted != payload:
            raise RuntimeError("The resumable download drill did not round-trip the decrypted payload.")
        if not range_requests:
            raise RuntimeError("The resumable download drill never resumed with an HTTP range request.")
        if leftover_parts:
            raise RuntimeError(
                "The resumable download drill left partial files behind: " + ", ".join(leftover_parts)
            )
        return (
            "passed",
            summary,
            {
                "requests_served": len(server.requests),
                "range_requests": range_requests,
                "resume_dir": str(resume_dir),
            },
        )

    def _run_disk_pressure_drill(self) -> tuple[str, str, dict[str, Any]]:
        controller = self.fault_controller()
        controller.activate(
            "disk_pressure",
            remaining_triggers=8,
            metadata={"free_bytes": 1 * (1024**3)},
            note="Live fault drill: simulate a nearly full local disk.",
        )
        health = self.runtime_health_snapshot()
        if health.get("issue_code") != "disk_low":
            raise RuntimeError("The disk-pressure drill did not push the node into the low-disk path.")
        payload = self.self_heal_check()
        return (
            "passed",
            "The self-heal loop recognized injected low disk space and ran the storage recovery path.",
            {
                "issue_code": health.get("issue_code"),
                "self_heal_action": self.self_heal_state.last_action,
                "self_heal_result": self.self_heal_state.last_result,
                "payload_status": payload.get("self_healing", {}).get("status") if isinstance(payload, dict) else None,
            },
        )

    def _run_warm_gpu_oom_drill(self) -> tuple[str, str, dict[str, Any]]:
        env_values = self.runtime_env_values()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})
        owner_target_model = str(env_values.get("OWNER_TARGET_MODEL") or "").strip()
        if not owner_target_model:
            return (
                "unavailable",
                "This node does not have an owner target model configured, so the warm-up OOM rollback drill is unavailable.",
                {},
            )
        if not self.runtime_health_snapshot().get("runtime_healthy"):
            return (
                "unavailable",
                "The runtime needs to be healthy before the warm-up OOM drill can run.",
                {},
            )
        if not self.self_heal_state.last_known_good_bootstrap_runtime_env:
            self.remember_known_good_runtime_state()
        if not self.self_heal_state.last_known_good_bootstrap_runtime_env:
            return (
                "unavailable",
                "The node has not recorded a last known-good bootstrap runtime tuple yet, so rollback cannot be rehearsed.",
                {},
            )
        controller = self.fault_controller()
        controller.activate(
            "warm_gpu_oom",
            remaining_triggers=1,
            note="Live fault drill: simulate GPU OOM during warm-up.",
        )
        success, message, action = self.attempt_vllm_recovery(model=owner_target_model, attempts=1)
        if not success or action != "rollback_owner_target_model":
            raise RuntimeError(message)
        return (
            "passed",
            "The node rolled back to the last known-good bootstrap model after an injected warm-up GPU OOM.",
            {
                "owner_target_model": owner_target_model,
                "self_heal_action": action,
                "self_heal_result": message,
            },
        )

    def run_fault_drill(self, scenario: str) -> dict[str, Any]:
        normalized = str(scenario or "").strip().lower().replace("-", "_")
        if normalized not in SUPPORTED_FAULT_DRILLS:
            raise ValueError(
                f"Unsupported fault drill {scenario!r}. Choose one of: {', '.join(SUPPORTED_FAULT_DRILLS)}."
            )
        controller = self.fault_controller()
        controller.clear_all()
        started_at = now_iso()
        status = "failed"
        summary = f"The {normalized} drill did not complete."
        details: dict[str, Any] = {}
        try:
            if normalized == "docker_restart_mid_run":
                status, summary, details = self._run_docker_restart_mid_run_drill()
            elif normalized == "dns_flap":
                status, summary, details = self._run_dns_flap_drill()
            elif normalized == "disk_almost_full":
                status, summary, details = self._run_disk_pressure_drill()
            elif normalized == "partial_download_resume":
                status, summary, details = self._run_resumable_download_drill(
                    "partial_artifact_download",
                    "The download path resumed cleanly after an injected partial download interruption.",
                )
            elif normalized == "power_loss_cache_write":
                status, summary, details = self._run_resumable_download_drill(
                    "cache_write_interrupt",
                    "The download path resumed cleanly after an injected cache-write interruption.",
                )
            elif normalized == "warm_gpu_oom":
                status, summary, details = self._run_warm_gpu_oom_drill()
        except Exception as error:
            details = {"error": str(error) or error.__class__.__name__}
            summary = str(error) or summary
            status = "failed"
        details["scenario"] = normalized
        details["fault_snapshot"] = self.fault_controller().snapshot()
        controller.record_drill(
            scenario=normalized,
            status=status,
            summary=summary,
            details=details,
            started_at=started_at,
            completed_at=now_iso(),
        )
        controller.clear_all()
        self.log(f"Fault drill {normalized} finished with status={status}: {summary}")
        return self.status_payload()

    def setup_preflight_payload(self, config: dict[str, Any]) -> dict[str, Any]:
        preview_snapshot = self.guided_installer.preview_setup_payload(config)
        payload = self.status_payload()
        installer_payload = payload.get("installer") if isinstance(payload.get("installer"), dict) else {}
        installer_payload.update(preview_snapshot)
        payload["installer"] = installer_payload
        payload["owner_setup"] = preview_snapshot.get("owner_setup", {})
        payload["setup_preview"] = preview_snapshot
        return payload

    def ensure_local_config(self) -> bool:
        self.ensure_dirs()
        self.guided_installer.ensure_data_dirs()

        repaired = False
        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            env_values = self.guided_installer.build_env({"setup_mode": "quickstart"})

        if not self.guided_installer.runtime_settings_path.exists():
            self.guided_installer.write_runtime_settings(env_values)
            repaired = True
        if not self.guided_installer.runtime_env_path.exists():
            self.guided_installer.write_runtime_env(env_values)
            repaired = True
        elif repaired:
            self.guided_installer.sync_runtime_env()

        if repaired:
            self.log("Self-healing recreated the local node settings and generated a fresh runtime config.")
        return repaired

    def ensure_local_credentials(self) -> bool:
        env_values = self.guided_installer.effective_runtime_env()
        node_id = str(env_values.get("NODE_ID") or "").strip()
        node_key = str(env_values.get("NODE_KEY") or "").strip()
        if not node_id or not node_key:
            return False

        existing_matches = False
        if self.guided_installer.credentials_path.exists():
            try:
                payload = json.loads(self.guided_installer.credentials_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            existing_matches = (
                isinstance(payload, dict)
                and str(payload.get("node_id") or "").strip() == node_id
                and str(payload.get("node_key") or "").strip() == node_key
            )
            if existing_matches:
                return False

        self.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
        tighten_private_path(self.guided_installer.credentials_path.parent, directory=True)
        self.guided_installer.credentials_path.write_text(
            json.dumps({"node_id": node_id, "node_key": node_key}, indent=2),
            encoding="utf-8",
        )
        tighten_private_path(self.guided_installer.credentials_path)
        self.log("Self-healing restored the local node credentials from the saved node approval.")
        return True

    def start_runtime_services(self, *, recreate: bool) -> None:
        if self.runtime_controller is not None:
            if recreate:
                self.log("Restarting the in-container runtime services...")
                self.runtime_controller.start(recreate=True, start_vllm=True, start_node=True)
                self.update_state.pending_restart = False
                self.save_state()
                self.log("In-container runtime services restarted.")
                return

            self.log("Starting the in-container runtime services...")
            self.runtime_controller.start(recreate=False, start_vllm=True, start_node=True)
            self.log("In-container runtime services started.")
            return

        if recreate:
            self.log("Restarting runtime services...")
            self.compose(["up", "-d", "--force-recreate", "vllm", "node-agent", "vector"])
            self.update_state.pending_restart = False
            self.save_state()
            self.log("Runtime services restarted.")
            return

        self.log("Starting runtime services...")
        self.compose(["up", "-d", "vllm", "node-agent", "vector"])
        self.log("Runtime services started.")

    def restart_inference_runtime_service(self) -> None:
        self.restart_vllm_service()

    def restart_node_agent_service(self) -> None:
        if self.runtime_controller is not None:
            self.log("Restarting the in-container node agent...")
            self.runtime_controller.restart_node_agent()
            return
        self.log("Restarting the node agent container...")
        self.compose(["up", "-d", "--force-recreate", "node-agent"])

    def restart_vllm_service(self) -> None:
        runtime_label = self.guided_installer.inference_runtime_label(self.runtime_env_values())
        if self.runtime_controller is not None:
            self.log(f"Restarting the in-container {runtime_label} model server...")
            self.runtime_controller.restart_vllm()
            return
        self.log(f"Restarting the {runtime_label} model server container...")
        self.compose(["up", "-d", "--force-recreate", "vllm"])

    def wait_for_inference_runtime_readiness(self, *, model: str | None = None, timeout_seconds: float = 180.0) -> None:
        self.wait_for_vllm_readiness(model=model, timeout_seconds=timeout_seconds)

    def wait_for_vllm_readiness(self, *, model: str | None = None, timeout_seconds: float = 180.0) -> None:
        env_values = self.runtime_env_values()
        runtime_label = self.guided_installer.inference_runtime_label(env_values)
        readiness_path = self.guided_installer.inference_readiness_path(env_values)
        readiness_url = f"http://{service_access_host()}:8000{readiness_path}"
        deadline = time.time() + max(1.0, timeout_seconds)
        current_model = (model or "").strip() or env_values.get("VLLM_MODEL") or "the startup model"
        last_failure = f"{current_model} is still warming."
        faults = self.fault_controller()
        while time.time() < deadline:
            if faults.consume("warm_gpu_oom"):
                raise RuntimeError(
                    f"CUDA out of memory while warming {current_model}. "
                    "The live fault drill forced a GPU OOM before readiness completed."
                )
            try:
                response = httpx.get(readiness_url, timeout=5.0)
                if response.status_code < 500:
                    return
                last_failure = f"{runtime_label} health check returned HTTP {response.status_code}."
            except httpx.HTTPError as error:
                last_failure = str(error) or last_failure
            time.sleep(2)
        raise RuntimeError(f"{current_model} did not finish warming in time. {last_failure}")

    def apply_nvidia_recovery_plan(self, plan: dict[str, Any]) -> tuple[bool, str, str]:
        env_values = self.guided_installer.effective_runtime_env()
        if not env_values:
            return False, "The local runtime config is still missing, so the NVIDIA preset could not be refreshed yet.", "repair_local_config"

        next_env = dict(env_values)
        target_model = str(plan.get("target_model") or "").strip()
        target_supported_models = str(plan.get("target_supported_models") or "").strip()
        target_profile = str(plan.get("target_profile") or "").strip()
        target_concurrency = str(plan.get("target_concurrency") or "").strip()
        target_thermal_headroom = str(plan.get("target_thermal_headroom") or "").strip()
        target_max_context_tokens = str(plan.get("target_max_context_tokens") or "").strip()
        target_vllm_startup_timeout_seconds = str(plan.get("target_vllm_startup_timeout_seconds") or "").strip()
        target_vllm_extra_args = str(plan.get("target_vllm_extra_args") or "").strip()
        target_vllm_memory_profiler_estimate_cudagraphs = str(
            plan.get("target_vllm_memory_profiler_estimate_cudagraphs") or ""
        ).strip()

        changed = False
        for key, value in (
            ("VLLM_MODEL", target_model),
            ("SUPPORTED_MODELS", target_supported_models),
            ("SETUP_PROFILE", target_profile),
            ("MAX_CONTEXT_TOKENS", target_max_context_tokens),
            ("MAX_CONCURRENT_ASSIGNMENTS", target_concurrency),
            ("THERMAL_HEADROOM", target_thermal_headroom),
            ("VLLM_STARTUP_TIMEOUT_SECONDS", target_vllm_startup_timeout_seconds),
            ("VLLM_EXTRA_ARGS", target_vllm_extra_args),
            (
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
                target_vllm_memory_profiler_estimate_cudagraphs,
            ),
        ):
            if value and next_env.get(key) != value:
                next_env[key] = value
                changed = True

        if not changed:
            return (
                False,
                f"The safer NVIDIA preset is already selected with {target_model}.",
                "downgrade_startup_model",
            )

        self.guided_installer.write_runtime_settings(next_env)
        self.guided_installer.write_runtime_env(next_env)
        self.log(
            "Self-healing switched this machine to the "
            f"{plan.get('target_preset_label') or 'recommended NVIDIA'} preset and will retry the warm-up with {target_model}."
        )
        return (
            True,
            f"Switched this machine to {target_model} on the {plan.get('target_capacity_label') or 'recommended NVIDIA'} preset.",
            "downgrade_startup_model",
        )

    def attempt_runtime_recovery(self, *, recreate: bool, failure_reason: str) -> tuple[bool, str, str]:
        restored_tuple = self.restore_known_good_runtime_tuple(failure_reason)
        recreate_runtime = recreate or restored_tuple
        self.start_runtime_services(recreate=recreate_runtime)
        try:
            self.wait_for_runtime_health()
        except Exception as error:
            detail = str(error) or failure_reason
            if self.rollback_to_known_good_release(detail):
                return True, "Self-healing rolled back to the last known healthy signed release.", "rollback_bad_update"
            return False, detail, "restart_runtime" if recreate_runtime else "start_runtime"

        self.remember_known_good_runtime_state()
        if restored_tuple:
            return True, "Restarted the runtime from the last known healthy local plan.", "restart_runtime"
        return True, "The runtime is healthy again.", "restart_runtime" if recreate_runtime else "start_runtime"

    def attempt_inference_runtime_recovery(
        self,
        *,
        model: str | None,
        attempts: int = VLLM_WARM_RETRY_ATTEMPTS,
    ) -> tuple[bool, str, str]:
        return self.attempt_vllm_recovery(model=model, attempts=attempts)

    def attempt_vllm_recovery(self, *, model: str | None, attempts: int = VLLM_WARM_RETRY_ATTEMPTS) -> tuple[bool, str, str]:
        env_values = self.runtime_env_values()
        runtime_label = self.guided_installer.inference_runtime_label(env_values)
        target_model = (model or "").strip() or env_values.get("VLLM_MODEL") or "the startup model"
        retry_hold = self.model_warm_retry_hold(target_model)
        if retry_hold:
            return False, retry_hold, "rollback_owner_target_model"
        last_error = f"{runtime_label} still needs attention."
        for attempt in range(1, max(1, attempts) + 1):
            try:
                self.log(
                    f"Self-healing is restarting {runtime_label} and retrying the warm-up for {target_model} "
                    f"({attempt}/{max(1, attempts)})."
                )
                self.restart_inference_runtime_service()
                self.wait_for_inference_runtime_readiness(model=target_model)
                self.clear_model_warm_failure(target_model)
                self.remember_known_good_runtime_state()
                return True, f"Restarted {runtime_label} and re-warmed {target_model} successfully.", "restart_vllm"
            except Exception as error:
                last_error = str(error) or last_error
                if attempt < max(1, attempts):
                    self.log(f"{runtime_label} warm-up retry {attempt} did not stick: {last_error}")
        failure_count, cooldown_until = self.record_model_warm_failure(target_model, last_error)
        owner_target_model = str(env_values.get("OWNER_TARGET_MODEL") or "").strip()
        if target_model == owner_target_model and cooldown_until is not None:
            restore_error = None
            try:
                restored = self.restore_known_good_runtime_tuple(last_error, prefer_bootstrap=True)
                self.start_runtime_services(recreate=restored)
                self.wait_for_runtime_health()
                self.remember_known_good_runtime_state()
                bootstrap_tuple = (
                    self.self_heal_state.last_known_good_bootstrap_runtime_env
                    or self.self_heal_state.last_known_good_runtime_env
                    or {}
                )
                bootstrap_model = str(
                    bootstrap_tuple.get("VLLM_MODEL")
                    or env_values.get("VLLM_MODEL")
                    or "the bootstrap model"
                )
                return (
                    True,
                    f"{target_model} failed to warm {failure_count} times in a row, so self-healing rolled back to "
                    f"{bootstrap_model} until {cooldown_until}.",
                    "rollback_owner_target_model",
                )
            except Exception as error:
                restore_error = str(error) or "Bootstrap rollback failed."
            return (
                False,
                f"{last_error} Self-healing also could not restore the bootstrap model automatically: {restore_error}",
                "rollback_owner_target_model",
            )
        return False, last_error, "restart_vllm"

    def attempt_node_agent_recovery(self) -> tuple[bool, str, str]:
        try:
            self.log("Self-healing is restarting the local node agent process.")
            self.restart_node_agent_service()
            return True, "The local node agent was restarted.", "restart_node_agent"
        except Exception as error:
            return False, str(error) or "The local node agent still needs attention.", "restart_node_agent"

    def watchdog_recovery(self, *, health: dict[str, Any]) -> tuple[bool, str, str] | None:
        if not health.get("runtime_healthy") or health.get("issue_code"):
            self.update_self_heal_state(
                idle_queue_detected_at=None,
                runtime_wedge_detected_at=None,
            )
            return None

        autopilot_state = self.load_autopilot_state() or {}
        signals = autopilot_state.get("signals") if isinstance(autopilot_state.get("signals"), dict) else {}
        remote = self.remote_dashboard_snapshot()
        remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
        remote_node = remote_summary.get("node") if isinstance(remote_summary.get("node"), dict) else {}
        current_time = datetime.now(timezone.utc)

        observed_at = parse_iso_timestamp(
            str(signals.get("last_observed_at")) if signals.get("last_observed_at") else None
        )
        observed_age_seconds = (
            max(0.0, (current_time - observed_at).total_seconds())
            if observed_at is not None
            else None
        )
        heartbeat_at = parse_iso_timestamp(
            str(remote_node.get("last_heartbeat_at")) if remote_node.get("last_heartbeat_at") else None
        )
        heartbeat_age_seconds = (
            max(0.0, (current_time - heartbeat_at).total_seconds())
            if heartbeat_at is not None
            else None
        )
        queue_depth = max(
            coerce_nonnegative_int(signals.get("queue_depth")),
            coerce_nonnegative_int(remote_node.get("queue_depth")),
        )
        active_assignments = max(
            coerce_nonnegative_int(signals.get("active_assignments")),
            coerce_nonnegative_int(remote_node.get("active_assignments")),
        )
        gpu_utilization_pct = coerce_float(signals.get("gpu_utilization_pct"))

        idle_with_queue = bool(
            observed_age_seconds is not None
            and observed_age_seconds <= max(SELF_HEAL_INTERVAL_SECONDS * 2, IDLE_QUEUE_WATCHDOG_SECONDS)
            and queue_depth > 0
            and active_assignments == 0
            and (
                gpu_utilization_pct is None
                or gpu_utilization_pct <= IDLE_QUEUE_GPU_UTILIZATION_THRESHOLD_PCT
            )
        )
        idle_with_queue_elapsed = self.watchdog_timer_elapsed(
            "idle_queue_detected_at",
            active=idle_with_queue,
            now=current_time,
        )

        remote_sync_error = bool(str(remote.get("last_error") or "").strip())
        local_signals_stalled = bool(
            observed_age_seconds is not None and observed_age_seconds >= RUNTIME_WEDGE_WATCHDOG_SECONDS
        )
        heartbeat_stalled = bool(
            heartbeat_age_seconds is not None
            and heartbeat_age_seconds >= RUNTIME_WEDGE_WATCHDOG_SECONDS
            and not remote_sync_error
        )
        runtime_wedged = local_signals_stalled or (heartbeat_stalled and queue_depth > 0)
        runtime_wedged_elapsed = self.watchdog_timer_elapsed(
            "runtime_wedge_detected_at",
            active=runtime_wedged,
            now=current_time,
        )

        runtime = health.get("runtime") if isinstance(health.get("runtime"), dict) else {}
        current_model = str(
            runtime.get("current_model")
            or self.runtime_env_values().get("VLLM_MODEL")
            or ""
        ).strip()

        if (
            idle_with_queue_elapsed is not None
            and idle_with_queue_elapsed >= IDLE_QUEUE_WATCHDOG_SECONDS
            and self.watchdog_restart_ready("last_targeted_inference_restart_at", now=current_time)
        ):
            success, message, action = self.attempt_inference_runtime_recovery(
                model=current_model,
                attempts=1,
            )
            self.update_self_heal_state(
                idle_queue_detected_at=None,
                last_targeted_inference_restart_at=now_iso(),
            )
            return (
                success,
                f"The local watchdog saw queued work with an idle GPU, so {message}",
                action,
            )

        if (
            runtime_wedged_elapsed is not None
            and runtime_wedged_elapsed >= RUNTIME_WEDGE_WATCHDOG_SECONDS
            and self.watchdog_restart_ready("last_targeted_node_restart_at", now=current_time)
        ):
            success, message, action = self.attempt_node_agent_recovery()
            self.update_self_heal_state(
                runtime_wedge_detected_at=None,
                last_targeted_node_restart_at=now_iso(),
            )
            reason = (
                "local runtime signals stopped updating"
                if local_signals_stalled
                else "the control-plane heartbeat stalled while work was still queued"
            )
            return success, f"The local watchdog restarted the node agent because {reason}. {message}", action

        return None

    def attempt_nvidia_runtime_recovery(self, health: dict[str, Any]) -> tuple[bool, str, str] | None:
        issue_code = str(health.get("issue_code") or "")
        if not issue_code:
            return None

        runtime = health.get("runtime") if isinstance(health.get("runtime"), dict) else {}
        running_services = {
            str(service)
            for service in health.get("running_services", [])
            if isinstance(service, str) and service.strip()
        }
        missing_services = set(self.required_runtime_services) - running_services
        plan = health.get("nvidia_recovery_plan") if isinstance(health.get("nvidia_recovery_plan"), dict) else None
        failure_reason = str(health.get("issue_detail") or issue_code)

        if issue_code == "startup_model_too_large" and plan is not None:
            applied, plan_message, plan_action = self.apply_nvidia_recovery_plan(plan)
            if not applied:
                return False, plan_message, plan_action
            success, recovery_message, _recovery_action = self.attempt_runtime_recovery(
                recreate=True,
                failure_reason=failure_reason,
            )
            if success:
                return True, f"{plan_message} The NVIDIA runtime restarted successfully.", plan_action
            return False, f"{plan_message} {recovery_message}", plan_action

        if not health.get("credentials_present"):
            return None

        if issue_code == "runtime_unhealthy" or (issue_code == "stuck_containers" and missing_services == {"vllm"}):
            model = str(runtime.get("current_model") or self.runtime_env_values().get("VLLM_MODEL") or "").strip()
            success, message, action = self.attempt_inference_runtime_recovery(model=model)
            if success:
                return success, message, action
            if plan is not None:
                applied, plan_message, plan_action = self.apply_nvidia_recovery_plan(plan)
                if applied:
                    success, recovery_message, _recovery_action = self.attempt_runtime_recovery(
                        recreate=True,
                        failure_reason=message,
                    )
                    if success:
                        return True, f"{plan_message} The NVIDIA runtime restarted successfully.", plan_action
                    return False, f"{plan_message} {recovery_message}", plan_action
            success, recovery_message, recovery_action = self.attempt_runtime_recovery(
                recreate=True,
                failure_reason=message,
            )
            if success:
                return True, "A full NVIDIA runtime restart brought the node back online.", recovery_action
            return False, recovery_message, recovery_action

        return None

    def repair_runtime(self, *, allow_quickstart_resume: bool = True) -> dict[str, Any]:
        if not self.repair_lock.acquire(blocking=False):
            self.log("A repair is already in progress for this machine.")
            return self.status_payload()

        payload: dict[str, Any] | None = None
        use_status_payload = True
        try:
            self.log("Starting local repair for this machine...")
            self.update_self_heal_state(
                status="repairing",
                last_checked_at=now_iso(),
                last_result="Self-healing is fixing this machine now.",
                last_error=None,
                last_action="repair_runtime",
                fix_available=True,
            )
            pre_repair_health = self.runtime_health_snapshot()
            autostart_before = (
                pre_repair_health.get("autostart")
                if isinstance(pre_repair_health.get("autostart"), dict)
                else {}
            )
            launcher_before = (
                pre_repair_health.get("desktop_launcher")
                if isinstance(pre_repair_health.get("desktop_launcher"), dict)
                else {}
            )
            startup_repaired = False
            launcher_repaired = False
            config_repaired = self.ensure_local_config()
            credentials_repaired = self.ensure_local_credentials()

            try:
                autostart = self.autostart_manager.ensure_enabled()
                self.log(str(autostart.get("detail") or "Automatic start status is unavailable."))
                startup_repaired = bool(autostart_before.get("supported")) and not bool(autostart_before.get("enabled")) and bool(autostart.get("enabled"))
            except Exception as error:
                self.log(f"Automatic start could not be repaired automatically: {error}")

            try:
                launcher = self.desktop_launcher_manager.ensure_enabled()
                self.log(str(launcher.get("detail") or "Desktop launcher status is unavailable."))
                launcher_repaired = bool(launcher_before.get("supported")) and not bool(launcher_before.get("enabled")) and bool(launcher.get("enabled"))
            except Exception as error:
                self.log(f"The desktop launcher could not be repaired automatically: {error}")

            health = self.runtime_health_snapshot()
            installer_state = health["installer_state"]
            preflight = health["preflight"]
            prerequisite_result = self.repair_prerequisite_blocker(health)

            if prerequisite_result is not None:
                message = str(prerequisite_result.get("message") or "Prerequisite-healing ran the next setup fix.")
                action = str(prerequisite_result.get("action") or "prerequisite_heal")
                resolved = bool(prerequisite_result.get("resolved"))
                self.log(message)
                self.update_self_heal_state(
                    status="healthy" if resolved else "attention",
                    last_result=message,
                    last_issue=str(health.get("issue_detail") or ""),
                    last_error=None,
                    last_action=action,
                    last_repaired_at=now_iso(),
                    fix_available=not resolved,
                )
                use_status_payload = True
            elif installer_state.get("busy"):
                self.update_self_heal_state(
                    status="standing_by",
                    last_result="Quick Start is already running, so repair is waiting for that setup flow to finish.",
                    last_issue=str(health.get("issue_detail") or ""),
                    last_error=None,
                    last_action="waiting_for_quick_start",
                    fix_available=False,
                )
                use_status_payload = True
            elif not health["docker_ready"]:
                message = str(health.get("issue_detail") or "Docker needs attention before repair can continue.")
                self.log(message)
                self.update_self_heal_state(
                    status="attention",
                    last_result=message,
                    last_issue=message,
                    last_error=None,
                    last_action="waiting_for_docker",
                    fix_available=False,
                )
                use_status_payload = True
            elif health.get("issue_code") in {"disk_low"}:
                message = str(health.get("issue_detail") or "Free up disk space before this machine can recover.")
                self.log(message)
                self.update_self_heal_state(
                    status="attention",
                    last_result=message,
                    last_issue=message,
                    last_error=None,
                    last_action="waiting_for_disk_space",
                    fix_available=False,
                )
                use_status_payload = True
            elif health.get("issue_code") in {"docker_unavailable", "docker_not_running"}:
                message = str(health.get("issue_detail") or "Docker needs attention before repair can continue.")
                self.log(message)
                self.update_self_heal_state(
                    status="attention",
                    last_result=message,
                    last_issue=message,
                    last_error=None,
                    last_action="waiting_for_docker",
                    fix_available=False,
                )
                use_status_payload = True
            elif health.get("issue_code") == "gpu_missing":
                message = str(
                    health.get("issue_detail")
                    or "A compatible GPU was not detected, so this machine cannot recover into serving mode yet."
                )
                self.log(message)
                self.update_self_heal_state(
                    status="attention",
                    last_result=message,
                    last_issue=message,
                    last_error=None,
                    last_action="waiting_for_gpu",
                    fix_available=False,
                )
                use_status_payload = True
            elif health.get("issue_code") == "startup_unavailable":
                message = str(
                    health.get("issue_detail")
                    or "Windows sign-in launch is unavailable and needs manual attention."
                )
                self.log(message)
                self.update_self_heal_state(
                    status="attention",
                    last_result=message,
                    last_issue=message,
                    last_error=None,
                    last_action="waiting_for_startup_support",
                    fix_available=True,
                )
                use_status_payload = True
            elif health.get("issue_code") == "startup_not_configured":
                message = str(
                    health.get("issue_detail")
                    or "Windows sign-in launch still needs to be repaired on this machine."
                )
                self.log(message)
                self.update_self_heal_state(
                    status="attention",
                    last_result=message,
                    last_issue=message,
                    last_error=None,
                    last_action="repair_startup",
                    fix_available=True,
                )
                use_status_payload = True
            elif (nvidia_recovery := self.attempt_nvidia_runtime_recovery(health)) is not None:
                success, message, action = nvidia_recovery
                self.update_self_heal_state(
                    status="healthy" if success else "error",
                    last_result=(
                        message
                        if success
                        else f"Self-healing could not recover the NVIDIA runtime automatically: {message}"
                    ),
                    last_issue=str(health.get("issue_detail") or ""),
                    last_error=None if success else message,
                    last_action=action,
                    last_repaired_at=now_iso() if success else self.self_heal_state.last_repaired_at,
                    fix_available=not success,
                )
                use_status_payload = True
            elif health["runtime_healthy"]:
                self.remember_known_good_runtime_state()
                if startup_repaired or launcher_repaired or config_repaired or credentials_repaired:
                    startup_result = "Self-healing repaired local setup"
                    if startup_repaired and launcher_repaired:
                        startup_result += ", refreshed automatic startup, and refreshed the desktop launcher."
                    elif startup_repaired:
                        startup_result += " and repaired automatic startup so this node can relaunch after sign-in."
                    elif launcher_repaired:
                        startup_result += " and refreshed the desktop launcher."
                    if config_repaired and credentials_repaired:
                        startup_result += " Saved config and node approval were restored locally."
                    elif config_repaired:
                        startup_result += " Saved config was restored locally."
                    elif credentials_repaired:
                        startup_result += " Saved node approval was restored locally."
                    self.update_self_heal_state(
                        status="healthy",
                        last_result=startup_result,
                        last_issue=None,
                        last_error=None,
                        last_action=(
                            "repair_startup"
                            if startup_repaired or launcher_repaired
                            else "repair_local_state"
                        ),
                        last_repaired_at=now_iso(),
                        fix_available=False,
                    )
                else:
                    self.update_self_heal_state(
                        status="healthy",
                        last_result="The runtime already looks healthy. Self-healing is standing by.",
                        last_issue=None,
                        last_error=None,
                        last_action="monitor",
                        last_repaired_at=now_iso(),
                        fix_available=False,
                    )
                use_status_payload = True
            elif health["credentials_present"]:
                recreate = bool(preflight.get("running_services")) or self.update_state.pending_restart
                if recreate:
                    self.log("Repair is recreating the runtime services using stored node credentials.")
                else:
                    self.log("Repair is starting the runtime with stored node credentials.")
                success, message, action = self.attempt_runtime_recovery(
                    recreate=recreate,
                    failure_reason=str(health.get("issue_detail") or "The runtime did not recover."),
                )
                self.update_self_heal_state(
                    status="healthy" if success else "error",
                    last_result=(
                        "Self-healing fixed the runtime and it should be healthy again."
                        if success
                        else f"Self-healing could not recover the runtime automatically: {message}"
                    ),
                    last_issue=str(health.get("issue_detail") or ""),
                    last_error=None if success else message,
                    last_action=action,
                    last_repaired_at=now_iso() if success else self.self_heal_state.last_repaired_at,
                    fix_available=not success,
                )
                use_status_payload = True
            elif not allow_quickstart_resume:
                self.update_self_heal_state(
                    status="attention",
                    last_result="Local setup files were repaired, but this machine still needs approval before it can come online.",
                    last_issue=str(health.get("issue_detail") or ""),
                    last_error=None,
                    last_action="waiting_for_approval",
                    fix_available=True,
                )
                use_status_payload = True
            else:
                repair_config = self.guided_installer.current_config()
                repair_config["setup_mode"] = "quickstart"
                repair_config["setup_profile"] = str(
                    repair_config.get("setup_profile")
                    or repair_config.get("recommended_setup_profile")
                    or "balanced"
                )
                self.log("Repair is resuming Quick Start so this machine can be approved again.")
                self.update_self_heal_state(
                    status="repairing",
                    last_result="Quick Start is resuming so this machine can be approved again.",
                    last_issue=str(health.get("issue_detail") or ""),
                    last_error=None,
                    last_action="resume_quick_start",
                    fix_available=True,
                )
                payload = self.guided_installer.start_install(repair_config)
                use_status_payload = False
        finally:
            self.repair_lock.release()

        if use_status_payload:
            return self.status_payload()
        return payload if payload is not None else self.status_payload()

    def start_runtime(self) -> dict[str, Any]:
        self.start_runtime_services(recreate=False)
        return self.status_payload()

    def stop_runtime(self) -> dict[str, Any]:
        self.log("Stopping runtime services...")
        if self.runtime_controller is not None:
            self.runtime_controller.stop()
            self.log("In-container runtime services stopped.")
        else:
            self.compose(["stop", "node-agent", "vllm", "vector"])
            self.log("Runtime services stopped.")
        return self.status_payload()

    def restart_runtime(self) -> dict[str, Any]:
        self.start_runtime_services(recreate=True)
        return self.status_payload()

    def docker_image_id(self, image: str) -> str | None:
        try:
            completed = self.command_runner(
                ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                self.runtime_dir,
            )
        except RuntimeError:
            return None
        value = completed.stdout.strip()
        return value or None

    def check_for_updates(self, apply: bool = False, *, automatic: bool = False) -> dict[str, Any]:
        if not runtime_backend_supports_compose(self.runtime_backend):
            self.update_state.last_checked_at = now_iso()
            self.update_state.last_error = None
            self.update_state.pending_restart = False
            self.update_state.last_result = (
                "The unified runtime image updates by pulling a new container image and recreating the container."
            )
            self.save_state()
            if apply:
                self.queue_background_doctor("update")
            return self.status_payload()

        rollout_id = request_id()
        try:
            manifest = self.signed_release_manifest()
            preferred_channel = normalize_release_channel(self.update_state.preferred_channel)
            release_channel = normalize_release_channel(manifest.channel)
            self.update_state.last_checked_at = now_iso()
            self.update_state.release_version = manifest.version
            self.update_state.release_channel = release_channel
            self.update_state.last_error = None
            if not release_channel_matches_preference(
                release_channel=release_channel,
                preferred_channel=preferred_channel,
            ):
                self.update_state.updated_images = []
                self.update_state.pending_restart = False
                self.update_state.last_result = (
                    f"Signed release {manifest.version} is on the {update_channel_copy(release_channel).lower()} track. "
                    f"This node stays on the {update_channel_copy(preferred_channel).lower()} track until a matching "
                    "appliance release is available."
                )
                self.log(
                    f"[request {rollout_id}] Skipping signed runtime release {manifest.version} on the "
                    f"{release_channel} track because this node prefers {preferred_channel}."
                )
                self.save_state()
                return self.status_payload()
            current_release = self.current_release_env(manifest)
            target_release = manifest.release_env()

            self.log(
                f"[request {rollout_id}] Checking signed runtime release "
                f"{manifest.version} ({manifest.channel})..."
            )

            updated: list[str] = []
            changed_services = [
                service
                for service, env_key in RELEASE_ENV_VAR_BY_SERVICE.items()
                if current_release.get(env_key) != target_release[env_key]
            ]

            for service in ("node-agent", "vllm", "vector"):
                target_ref = target_release[RELEASE_ENV_VAR_BY_SERVICE[service]]
                before = self.docker_image_id(target_ref)
                if service in changed_services or before is None:
                    self.command_runner(["docker", "pull", target_ref], self.runtime_dir)
                    after = self.docker_image_id(target_ref)
                    if after and before != after:
                        updated.append(service)

            self.update_state.updated_images = updated
            self.update_state.release_version = manifest.version
            self.update_state.release_channel = release_channel
            self.update_state.last_error = None

            if not changed_services:
                self.update_state.last_result = f"Runtime already matches signed release {manifest.version}."
                self.update_state.pending_restart = False
                self.log(f"[request {rollout_id}] Signed runtime release {manifest.version} is already applied.")
                self.save_state()
                return self.status_payload()

            if not apply:
                self.update_state.last_result = (
                    f"Signed release {manifest.version} was downloaded and is ready to apply."
                )
                self.update_state.pending_restart = True
                self.log(
                    f"[request {rollout_id}] Downloaded signed release {manifest.version} "
                    f"for: {', '.join(changed_services)}."
                )
                self.save_state()
                return self.status_payload()

            previous_content = self.release_env_path.read_text(encoding="utf-8") if self.release_env_path.exists() else None
            previous_runtime_tuple = self.runtime_tuple_snapshot()
            self.write_release_env(target_release)
            try:
                self.start_runtime_services(recreate=True)
                self.wait_for_runtime_health()
                canary = self.verify_runtime_canary(
                    context=f"Signed release {manifest.version} local inference canary"
                )
            except Exception as error:
                self.log(
                    f"[request {rollout_id}] Signed release {manifest.version} failed readiness/canary checks: {error}"
                )
                rollback_ok, rollback_message = self.rollback_update_to_known_good_runtime_state(
                    failure_reason=str(error) or f"Signed release {manifest.version} failed the update canary.",
                    previous_release_content=previous_content,
                    previous_runtime_tuple=previous_runtime_tuple,
                )
                if rollback_ok:
                    self.update_state.last_error = None
                    self.update_state.last_result = (
                        f"Signed release {manifest.version} failed the local canary and rolled back automatically. "
                        f"{rollback_message}"
                    )
                    self.record_owner_timeline_event(
                        code="update_rolled_back",
                        title="Update rolled back automatically",
                        detail=self.update_state.last_result,
                        tone="warning",
                        source="updates",
                        observed_at=self.update_state.last_checked_at,
                    )
                else:
                    self.update_state.last_error = (
                        f"Signed release {manifest.version} could not be applied automatically, and rollback needs attention. "
                        f"Check the service log with request id {rollout_id}."
                    )
                    self.update_state.last_result = "Automatic rollback after the failed signed release update needs attention."
                self.update_state.pending_restart = False
                self.save_state()
                return self.status_payload()

            self.update_state.last_result = (
                f"Signed release {manifest.version} was applied successfully after readiness "
                f"and a tiny local {str(canary.get('api') or 'inference')} canary passed."
            )
            self.update_state.pending_restart = False
            self.record_owner_timeline_event(
                code="update_applied",
                title="Update applied successfully",
                detail=self.update_state.last_result,
                tone="success",
                source="updates",
                observed_at=self.update_state.last_checked_at,
            )
            self.remember_known_good_runtime_state()
            self.log(f"[request {rollout_id}] Signed runtime release {manifest.version} is now active.")
            self.save_state()
            return self.status_payload()
        except ReleaseManifestError as error:
            self.update_state.last_checked_at = now_iso()
            self.update_state.updated_images = []
            self.update_state.pending_restart = False
            self.update_state.last_error = (
                f"Signed runtime release verification failed. Check the service log with request id {rollout_id}."
            )
            self.log(f"[request {rollout_id}] Signed release verification failed: {error}")
            self.save_state()
            return self.status_payload()
        except Exception as error:
            self.update_state.last_checked_at = now_iso()
            self.update_state.updated_images = []
            self.update_state.pending_restart = False
            self.update_state.last_error = (
                f"Signed runtime release check failed. Check the service log with request id {rollout_id}."
            )
            self.log(f"[request {rollout_id}] Signed release check failed: {error}")
            self.save_state()
            return self.status_payload()
        finally:
            if apply:
                self.queue_background_doctor("update")

    def configure_updates(self, enabled: bool, interval_hours: int, preferred_channel: str | None = None) -> dict[str, Any]:
        self.update_state.auto_update_enabled = enabled
        self.update_state.interval_hours = max(1, int(interval_hours))
        self.update_state.preferred_channel = normalize_release_channel(preferred_channel or self.update_state.preferred_channel)
        self.update_state.last_error = None
        self.save_state()
        self.log(
            "Auto-update settings changed: "
            f"{'enabled' if enabled else 'disabled'} every {self.update_state.interval_hours} hour(s) on the "
            f"{update_channel_copy(self.update_state.preferred_channel).lower()} track."
        )
        return self.status_payload()

    def redacted_env(self) -> str:
        env_values = self.guided_installer.effective_runtime_env()
        lines: list[str] = []
        for key, value in env_values.items():
            if any(marker in key for marker in SENSITIVE_ENV_MARKERS):
                redacted = "***REDACTED***" if value else ""
                lines.append(f"{key}={redacted}")
            else:
                lines.append(f"{key}={value}")
        return "\n".join(lines) + ("\n" if lines else "")

    def redacted_runtime_settings(self) -> str:
        payload = self.guided_installer.load_runtime_settings()
        if not payload:
            return ""

        config = payload.get("config")
        if isinstance(config, dict):
            redacted = dict(config)
            for key in list(redacted):
                if any(marker.lower() in key.lower() for marker in SENSITIVE_ENV_MARKERS):
                    redacted[key] = "***REDACTED***" if redacted[key] else ""
            payload = dict(payload)
            payload["config"] = redacted
        return json.dumps(payload, indent=2) + "\n"

    def command_output(self, args: list[str]) -> str:
        try:
            command = self.compose_command(args[2:]) if args[:2] == ["docker", "compose"] else args
            completed = self.command_runner(command, self.runtime_dir)
            output = completed.stdout.strip() or completed.stderr.strip()
            return output or "<no output>"
        except RuntimeError as error:
            return f"<command failed>\n{error}"

    def write_diagnostics_bundle(self) -> tuple[Path, str, str]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        bundle_name = f"diagnostics-{timestamp}.zip"
        bundle_path = self.diagnostics_dir / bundle_name
        generated_at = now_iso()
        self.log("Creating diagnostics bundle...")

        payload = self.status_payload()
        payload["generated_at"] = generated_at
        dashboard = payload.get("dashboard") if isinstance(payload.get("dashboard"), dict) else {}
        alerts = dashboard.get("alerts") if isinstance(dashboard.get("alerts"), list) else []
        diagnostics_summary = {
            "generated_at": generated_at,
            "dashboard_headline": dashboard.get("headline"),
            "dashboard_detail": dashboard.get("detail"),
            "connectivity": payload.get("connectivity"),
            "alerts": alerts,
            "timeline": dashboard.get("timeline"),
            "heat_governor": payload.get("heat_governor"),
            "economics": dashboard.get("economics"),
            "local_doctor": payload.get("local_doctor"),
            "self_healing": payload.get("self_healing"),
            "fault_drills": payload.get("fault_drills"),
        }
        support_summary_lines = [
            f"Generated at: {generated_at}",
            f"Dashboard: {dashboard.get('headline') or 'Owner dashboard'}",
            f"Detail: {dashboard.get('detail') or 'No dashboard detail available.'}",
        ]
        if alerts:
            support_summary_lines.append("Owner alerts:")
            for alert in alerts[:5]:
                if not isinstance(alert, dict):
                    continue
                title = str(alert.get("title") or "Alert").strip()
                detail = str(alert.get("detail") or "").strip()
                if detail:
                    support_summary_lines.append(f"- {title}: {detail}")
                else:
                    support_summary_lines.append(f"- {title}")
        timeline = dashboard.get("timeline") if isinstance(dashboard.get("timeline"), list) else []
        if timeline:
            support_summary_lines.append("Recent timeline:")
            for event in timeline[:5]:
                if not isinstance(event, dict):
                    continue
                title = str(event.get("title") or "Owner update").strip()
                detail = str(event.get("detail") or "").strip()
                observed = str(event.get("observed_label") or event.get("observed_at") or "").strip()
                prefix = f"- {title}"
                if observed:
                    prefix = f"{prefix} ({observed})"
                support_summary_lines.append(f"{prefix}: {detail}" if detail else prefix)
        heat_governor = payload.get("heat_governor") if isinstance(payload.get("heat_governor"), dict) else {}
        heat_plan = heat_governor.get("plan") if isinstance(heat_governor.get("plan"), dict) else {}
        if heat_plan:
            support_summary_lines.append(
                "Heat governor: "
                f"{heat_plan.get('effective_target_pct', heat_plan.get('requested_target_pct', '?'))}% target. "
                f"{heat_plan.get('reason') or ''}".strip()
            )
        local_doctor = payload.get("local_doctor") if isinstance(payload.get("local_doctor"), dict) else {}
        if local_doctor:
            support_summary_lines.append(
                "Local Doctor: "
                f"{local_doctor.get('status') or 'unknown'}. "
                f"{local_doctor.get('headline') or ''}".strip()
            )
            if local_doctor.get("detail"):
                support_summary_lines.append(f"Local Doctor detail: {local_doctor.get('detail')}")
            recommended_fix = (
                local_doctor.get("recommended_fix")
                if isinstance(local_doctor.get("recommended_fix"), dict)
                else {}
            )
            if recommended_fix.get("label") or recommended_fix.get("detail"):
                support_summary_lines.append(
                    "Recommended fix: "
                    f"{recommended_fix.get('label') or 'Fix'}"
                    + (f" - {recommended_fix.get('detail')}" if recommended_fix.get("detail") else "")
                )
            last_fix_attempt = (
                local_doctor.get("last_fix_attempt")
                if isinstance(local_doctor.get("last_fix_attempt"), dict)
                else {}
            )
            if last_fix_attempt.get("summary"):
                support_summary_lines.append(f"Local Doctor fix loop: {last_fix_attempt.get('summary')}")
            if last_fix_attempt.get("before_after"):
                support_summary_lines.append(str(last_fix_attempt.get("before_after")))
        economics = dashboard.get("economics") if isinstance(dashboard.get("economics"), dict) else {}
        if economics:
            economics_sources: list[str] = []
            if economics.get("today_source_key"):
                economics_sources.append(
                    f"earnings {economics.get('today_source_key')} ({economics.get('today_source_confidence') or 'unknown'} confidence)"
                )
            if economics.get("power_source_key"):
                economics_sources.append(
                    f"power {economics.get('power_source_key')} ({economics.get('power_source_confidence') or 'unknown'} confidence)"
                )
            if economics.get("basis_label"):
                economics_sources.append(f"basis {economics.get('basis_label')}")
            if economics.get("heat_assumption_source"):
                economics_sources.append(
                    f"heat assumptions {economics.get('heat_assumption_source')} "
                    f"({economics.get('heat_assumption_confidence') or 'unknown'} confidence)"
                )
            support_summary_lines.append(
                "Economics: "
                f"today {format_usd(economics.get('today_earnings_usd'))}, "
                f"electricity {format_usd(economics.get('electricity_cost_usd'))}, "
                f"heat offset {format_usd(economics.get('heat_offset_usd'))}, "
                f"net {format_usd(economics.get('net_value_usd'))}. "
                + (f"Sources: {'; '.join(economics_sources)}." if economics_sources else "")
            )
        fault_drills = payload.get("fault_drills") if isinstance(payload.get("fault_drills"), dict) else {}
        last_drill = fault_drills.get("last_drill") if isinstance(fault_drills.get("last_drill"), dict) else {}
        if last_drill.get("scenario"):
            support_summary_lines.append(
                "Fault drill: "
                f"{last_drill.get('scenario')} -> {last_drill.get('status')}. "
                f"{last_drill.get('summary') or ''}".strip()
            )

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("status.json", json.dumps(payload, indent=2))
            archive.writestr("diagnostics-summary.json", json.dumps(diagnostics_summary, indent=2))
            archive.writestr("support-summary.txt", "\n".join(support_summary_lines) + "\n")
            archive.writestr("env.redacted", self.redacted_env())
            if self.guided_installer.runtime_settings_path.exists():
                archive.writestr("runtime-settings.redacted.json", self.redacted_runtime_settings())
            archive.writestr("service.log", "\n".join(payload["service"]["logs"]) + "\n")
            archive.writestr("installer.log", "\n".join(payload["installer"]["state"]["logs"]) + "\n")
            if runtime_backend_supports_compose(self.runtime_backend):
                archive.writestr(
                    "docker-compose-ps.txt",
                    self.command_output(["docker", "compose", "ps"]),
                )
                archive.writestr(
                    "docker-compose.logs.txt",
                    self.command_output(["docker", "compose", "logs", "--tail", "200", "node-agent", "vllm", "vector"]),
                )
            else:
                runtime_snapshot = self.runtime_backend_status()
                archive.writestr("runtime-processes.json", json.dumps(runtime_snapshot, indent=2))
            archive.writestr(
                "nvidia-smi.txt",
                self.command_output(["nvidia-smi"]),
            )
            recovery_note = self.guided_installer.credentials_dir / "recovery-note.txt"
            if recovery_note.exists():
                archive.writestr("recovery-note.txt", recovery_note.read_text(encoding="utf-8"))
            if self.fault_injection_state_path.exists():
                archive.writestr(
                    "fault-injection-state.json",
                    self.fault_injection_state_path.read_text(encoding="utf-8"),
                )

        self.diagnostics_state.last_bundle_name = bundle_name
        self.diagnostics_state.last_bundle_created_at = generated_at
        self.diagnostics_state.last_error = None
        self.save_state()
        self.log(f"Diagnostics bundle created: {bundle_name}")
        return bundle_path, bundle_name, generated_at

    def create_diagnostics_bundle(self) -> dict[str, Any]:
        _bundle_path, bundle_name, _generated_at = self.write_diagnostics_bundle()
        self.diagnostics_state.last_result = f"{bundle_name} is ready to download from this machine."
        self.save_state()
        return self.status_payload()

    def send_support_bundle(self) -> dict[str, Any]:
        bundle_path, bundle_name, generated_at = self.write_diagnostics_bundle()
        client = self.build_control_client()
        if client is None:
            self.diagnostics_state.last_error = (
                "Support upload needs a saved control-plane connection before this machine can send bundles automatically."
            )
            self.diagnostics_state.last_result = f"{bundle_name} was created locally and is ready to download."
            self.save_state()
            self.log("Support upload is unavailable because the control plane is not configured for this machine yet.")
            return self.status_payload()

        try:
            payload = client.submit_support_bundle(
                bundle_name,
                bundle_path.read_bytes(),
                generated_at=generated_at,
            )
        except Exception as error:
            self.diagnostics_state.last_error = (
                "Support upload could not be completed automatically. Download the local bundle if manual sharing is needed."
            )
            self.diagnostics_state.last_result = f"{bundle_name} was created locally and is ready to download."
            self.save_state()
            self.log(f"Support bundle upload failed: {error}")
            return self.status_payload()

        case_id = str(payload.get("case_id") or "").strip() or f"local-{hashlib.sha256(bundle_name.encode('utf-8')).hexdigest()[:12]}"
        received_at = str(payload.get("received_at") or generated_at)
        self.diagnostics_state.last_case_id = case_id
        self.diagnostics_state.last_bundle_sent_at = received_at
        self.diagnostics_state.last_result = f"Support bundle sent successfully. Case ID: {case_id}."
        self.diagnostics_state.last_error = None
        self.save_state()
        self.log(f"Support bundle sent successfully. Case ID: {case_id}.")
        return self.status_payload()

    def open_ui(self) -> None:
        open_browser(self.host, self.port, self.admin_token)

    def enable_autostart(self) -> dict[str, Any]:
        status = self.autostart_manager.enable()
        self.log(str(status.get("detail") or "Automatic start is enabled."))
        return self.status_payload()

    def disable_autostart(self) -> dict[str, Any]:
        status = self.autostart_manager.disable()
        self.log(str(status.get("detail") or "Automatic start is disabled."))
        return self.status_payload()

    def install_desktop_launcher(self) -> dict[str, Any]:
        status = self.desktop_launcher_manager.enable()
        self.log(str(status.get("detail") or "The desktop launcher is installed."))
        return self.status_payload()

    def remove_desktop_launcher(self) -> dict[str, Any]:
        status = self.desktop_launcher_manager.disable()
        self.log(str(status.get("detail") or "The desktop launcher was removed."))
        return self.status_payload()

    def auto_update_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                if self.update_state.auto_update_enabled and not self.repair_lock.locked():
                    due = False
                    if not self.update_state.last_checked_at:
                        due = True
                    else:
                        last_checked = datetime.fromisoformat(self.update_state.last_checked_at)
                        due = (datetime.now(timezone.utc) - last_checked).total_seconds() >= self.update_state.interval_hours * 3600
                    if due:
                        self.check_for_updates(apply=True, automatic=True)
            except Exception as error:  # pragma: no cover
                loop_request_id = request_id()
                self.update_state.last_error = (
                    f"Automatic signed runtime release check failed. "
                    f"Check the service log with request id {loop_request_id}."
                )
                self.save_state()
                self.log(f"[request {loop_request_id}] Automatic signed runtime release check failed: {error}")
            self.shutdown_event.wait(60)

    def local_doctor_loop(self) -> None:
        self.queue_background_doctor("service_start")
        while not self.shutdown_event.is_set():
            try:
                trigger = self.pop_background_doctor_trigger()
                if trigger:
                    self.run_local_doctor(
                        background=True,
                        trigger=trigger,
                        attach_bundle_on_failure=False,
                    )
                    continue
                if self.local_doctor_background_due() and self.local_doctor_can_run_in_background():
                    self.run_local_doctor(
                        background=True,
                        trigger="idle_interval",
                        attach_bundle_on_failure=False,
                    )
                    continue
            except Exception as error:  # pragma: no cover
                loop_request_id = request_id()
                self.update_local_doctor_state(
                    last_error=(
                        "Automatic Local Doctor hit an unexpected error. "
                        f"Check the local service log with request id {loop_request_id}."
                    ),
                    last_check_mode="background",
                )
                self.log(f"[request {loop_request_id}] Automatic Local Doctor failed: {error}")
            self.local_doctor_wakeup.wait(LOCAL_DOCTOR_LOOP_SECONDS)
            self.local_doctor_wakeup.clear()

    def self_heal_check(self) -> dict[str, Any]:
        health = self.runtime_health_snapshot()
        self.update_self_heal_state(
            last_checked_at=now_iso(),
            fix_available=bool(health.get("manual_fix_available")),
        )

        if health.get("issue_code") == "disk_low":
            installer_snapshot = (
                health.get("installer_snapshot") if isinstance(health.get("installer_snapshot"), dict) else self.guided_installer.status_payload()
            )
            cache_changed, cache_result, cache_action = self.manage_model_cache(
                health=health,
                installer_snapshot=installer_snapshot,
            )
            if cache_changed:
                self.update_self_heal_state(
                    status="healthy" if not health.get("issue_code") else "attention",
                    last_result=cache_result,
                    last_issue=str(health.get("issue_detail") or ""),
                    last_error=None,
                    last_action=cache_action,
                    last_repaired_at=now_iso(),
                    fix_available=True,
                )
                return self.status_payload()

        if health["runtime_healthy"] and not health.get("issue_code"):
            config_repaired = self.ensure_local_config()
            credentials_repaired = self.ensure_local_credentials()
            if config_repaired or credentials_repaired:
                repair_bits = []
                if config_repaired:
                    repair_bits.append("recreated the saved runtime config")
                if credentials_repaired:
                    repair_bits.append("restored the saved node approval")
                self.update_self_heal_state(
                    status="healthy",
                    last_result="Self-healing " + " and ".join(repair_bits) + ".",
                    last_issue=None,
                    last_error=None,
                    last_action="repair_local_state",
                    last_repaired_at=now_iso(),
                    fix_available=False,
                )
                return self.status_payload()
            self.remember_known_good_runtime_state()
            watchdog_recovery = self.watchdog_recovery(health=health)
            if watchdog_recovery is not None:
                success, message, action = watchdog_recovery
                self.update_self_heal_state(
                    status="healthy" if success else "error",
                    last_result=(
                        message
                        if success
                        else f"Self-healing could not complete the watchdog recovery automatically: {message}"
                    ),
                    last_issue=None,
                    last_error=None if success else message,
                    last_action=action,
                    last_repaired_at=now_iso() if success else self.self_heal_state.last_repaired_at,
                    fix_available=not success,
                )
                return self.status_payload()
            applied, autopilot_result = self.apply_autopilot_tuning()
            if applied:
                self.update_self_heal_state(
                    status="healthy",
                    last_result=autopilot_result,
                    last_issue=None,
                    last_error=None,
                    last_action="autopilot_tuning",
                    fix_available=False,
                )
                return self.status_payload()
            installer_snapshot = (
                health.get("installer_snapshot") if isinstance(health.get("installer_snapshot"), dict) else self.guided_installer.status_payload()
            )
            cache_changed, cache_result, cache_action = self.manage_model_cache(
                health=health,
                installer_snapshot=installer_snapshot,
            )
            if cache_changed:
                self.update_self_heal_state(
                    status="healthy",
                    last_result=cache_result,
                    last_issue=None,
                    last_error=None,
                    last_action=cache_action,
                    last_repaired_at=now_iso(),
                    fix_available=False,
                )
                return self.status_payload()
            self.update_self_heal_state(
                status="healthy",
                last_result="The runtime looks healthy and self-healing is monitoring it.",
                last_issue=None,
                last_error=None,
                last_action="monitor",
                fix_available=False,
            )
            return self.status_payload()

        if health["issue_code"] == "installer_busy":
            self.update_self_heal_state(
                status="standing_by",
                last_result=str(health.get("issue_detail") or "Quick Start is still running."),
                last_issue=str(health.get("issue_detail") or ""),
                last_error=None,
                last_action="waiting_for_quick_start",
                fix_available=False,
            )
            return self.status_payload()

        nvidia_recovery = self.attempt_nvidia_runtime_recovery(health)
        if nvidia_recovery is not None:
            success, message, action = nvidia_recovery
            self.update_self_heal_state(
                status="healthy" if success else "error",
                last_result=(
                    message
                    if success
                    else f"Self-healing could not recover the NVIDIA runtime automatically: {message}"
                ),
                last_issue=str(health.get("issue_detail") or ""),
                last_error=None if success else message,
                last_action=action,
                last_repaired_at=now_iso() if success else self.self_heal_state.last_repaired_at,
                fix_available=not success,
            )
            return self.status_payload()

        if not health["automatic_fix_available"]:
            self.update_self_heal_state(
                status="attention",
                last_result=str(
                    health.get("issue_detail")
                    or "Self-healing found something it cannot repair automatically yet."
                ),
                last_issue=str(health.get("issue_detail") or ""),
                last_error=None,
                last_action="waiting_for_owner",
                fix_available=bool(health.get("manual_fix_available")),
            )
            return self.status_payload()

        self.log(
            "Self-healing detected a runtime issue and is attempting an automatic repair: "
            f"{health.get('issue_detail') or health.get('issue_code') or 'runtime issue'}"
        )
        return self.repair_runtime(allow_quickstart_resume=False)

    def self_heal_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                if self.self_heal_state.enabled and not self.repair_lock.locked():
                    previous_action = self.self_heal_state.last_action
                    self.self_heal_check()
                    trigger = self.local_doctor_background_trigger_for_self_heal(previous_action)
                    if trigger:
                        self.queue_background_doctor(trigger)
            except Exception as error:  # pragma: no cover
                loop_request_id = request_id()
                self.update_self_heal_state(
                    status="error",
                    last_checked_at=now_iso(),
                    last_result=(
                        "Automatic self-healing hit an unexpected error. "
                        f"Check the local service log with request id {loop_request_id}."
                    ),
                    last_error=str(error),
                    last_action="self_heal_loop",
                    fix_available=True,
                )
                self.log(f"[request {loop_request_id}] Automatic self-healing failed: {error}")
            self.shutdown_event.wait(self.self_heal_state.interval_seconds)

    def request_shutdown(self) -> dict[str, Any]:
        self.log("Shutdown requested for node runtime service.")
        self.shutdown_event.set()
        self.local_doctor_wakeup.set()
        return {"ok": True}

    def write_meta(self) -> None:
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self.meta_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": self.host,
                    "port": self.port,
                    "started_at": self.started_at,
                    "admin_token": self.admin_token,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tighten_private_path(self.pid_path)
        tighten_private_path(self.meta_path)

    def clear_meta(self) -> None:
        for path in (self.pid_path, self.meta_path):
            if path.exists():
                path.unlink()


def make_handler(service: NodeRuntimeService, server_ref: dict[str, ThreadingHTTPServer]) -> type[BaseHTTPRequestHandler]:
    class ServiceHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:  # pragma: no cover
            return

        def _send_json(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
            *,
            correlation_id: str | None = None,
        ) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            if correlation_id:
                self.send_header("x-correlation-id", correlation_id)
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: bytes, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status.value)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("referrer-policy", "no-referrer")
            self.send_header("x-frame-options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def _send_session_bootstrap(self) -> None:
            session_token = service.sessions.issue()
            body = session_bootstrap_html()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("referrer-policy", "no-referrer")
            self.send_header("x-frame-options", "DENY")
            self.send_header(
                "set-cookie",
                serialize_cookie(LOCAL_SESSION_COOKIE, session_token, max_age=service.sessions.ttl_seconds),
            )
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def _request_token(self, *, allow_query: bool) -> str | None:
            token = self.headers.get(ADMIN_TOKEN_HEADER)
            if token:
                return token.strip()
            if not allow_query:
                return None
            return request_query_param(self.path, "token")

        def _authorized_via_admin_token(self, *, allow_query: bool) -> bool:
            return token_matches(self._request_token(allow_query=allow_query), service.admin_token)

        def _authorized_via_session(self) -> bool:
            return service.sessions.contains(cookie_value(self.headers, LOCAL_SESSION_COOKIE))

        def _authorized(self, *, allow_query: bool) -> bool:
            return self._authorized_via_session() or self._authorized_via_admin_token(allow_query=allow_query)

        def _origin_allowed(self) -> bool:
            return origin_matches_host(self.headers)

        def _reject(self, status: HTTPStatus, code: str, message: str) -> None:
            self._send_json(
                {
                    "error": {
                        "code": code,
                        "message": message,
                    }
                },
                status,
            )

        def do_GET(self) -> None:  # pragma: no cover
            path = urlparse(self.path).path
            if path == "/":
                if self._authorized_via_session():
                    body = service_html().encode("utf-8")
                    self._send_html(body)
                    return
                if self._authorized_via_admin_token(allow_query=True):
                    self._send_session_bootstrap()
                    return
                if not self._authorized(allow_query=False):
                    self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized", "A local admin token is required.")
                    return
                body = service_html().encode("utf-8")
                self._send_html(body)
                return
            if path == "/api/healthz":
                self._send_json({"ok": True, "service": "node-runtime"})
                return
            if path == "/api/status":
                if not self._authorized(allow_query=False):
                    self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized", "A local admin token is required.")
                    return
                self._send_json(service.status_payload())
                return
            if path.startswith("/downloads/"):
                if not self._authorized(allow_query=False):
                    self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized", "A local admin token is required.")
                    return
                requested = unquote(path.removeprefix("/downloads/"))
                target = (service.diagnostics_dir / requested).resolve()
                if not str(target).startswith(str(service.diagnostics_dir.resolve())) or not target.exists():
                    self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                    return
                body = target.read_bytes()
                self.send_response(HTTPStatus.OK.value)
                self.send_header("content-type", "application/zip")
                self.send_header("content-length", str(len(body)))
                self.send_header("content-disposition", f'attachment; filename="{target.name}"')
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # pragma: no cover
            path = urlparse(self.path).path
            if not self._authorized(allow_query=False):
                self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized", "A local admin token is required.")
                return
            if not self._origin_allowed():
                self._reject(HTTPStatus.FORBIDDEN, "forbidden", "Cross-origin requests are not allowed.")
                return
            try:
                if path == "/api/install":
                    payload = self._read_json()
                    self._send_json(service.guided_installer.start_install(payload))
                    return
                if path == "/api/setup-preflight":
                    payload = self._read_json()
                    self._send_json(service.setup_preflight_payload(payload))
                    return
                if path == "/api/runtime/start":
                    self._send_json(service.start_runtime())
                    return
                if path == "/api/runtime/stop":
                    self._send_json(service.stop_runtime())
                    return
                if path == "/api/runtime/restart":
                    self._send_json(service.restart_runtime())
                    return
                if path == "/api/heat-governor":
                    payload = self._read_json()
                    self._send_json(service.configure_heat_governor(payload))
                    return
                if path == "/api/updates/check":
                    payload = self._read_json()
                    self._send_json(service.check_for_updates(apply=bool(payload.get("apply"))))
                    return
                if path == "/api/updates/configure":
                    payload = self._read_json()
                    self._send_json(
                        service.configure_updates(
                            enabled=bool(payload.get("enabled")),
                            interval_hours=int(payload.get("interval_hours", 24)),
                            preferred_channel=str(payload.get("preferred_channel", DEFAULT_UPDATE_CHANNEL)),
                        )
                    )
                    return
                if path == "/api/autostart/enable":
                    self._send_json(service.enable_autostart())
                    return
                if path == "/api/autostart/disable":
                    self._send_json(service.disable_autostart())
                    return
                if path == "/api/launcher/install":
                    self._send_json(service.install_desktop_launcher())
                    return
                if path == "/api/launcher/remove":
                    self._send_json(service.remove_desktop_launcher())
                    return
                if path == "/api/repair":
                    self._send_json(service.repair_runtime())
                    return
                if path == "/api/local-doctor":
                    self._send_json(service.run_local_doctor())
                    return
                if path == "/api/local-doctor/fix":
                    self._send_json(service.apply_local_doctor_fix())
                    return
                if path == "/api/diagnostics":
                    self._send_json(service.create_diagnostics_bundle())
                    return
                if path == "/api/support/send":
                    self._send_json(service.send_support_bundle())
                    return
                if path == "/api/fault-drills/run":
                    payload = self._read_json()
                    self._send_json(service.run_fault_drill(str(payload.get("scenario") or "")))
                    return
                if path == "/api/shutdown":
                    self._send_json(service.request_shutdown())
                    threading.Thread(target=server_ref["server"].shutdown, daemon=True).start()
                    return
            except Exception as error:
                correlation_id = request_id()
                service.log(f"[request {correlation_id}] {path} failed: {error}")
                self._send_json(
                    {
                        "error": {
                            "code": "request_failed",
                            "message": (
                                "The request could not be completed. "
                                f"Check the local service log with request id {correlation_id}."
                            ),
                            "request_id": correlation_id,
                        }
                    },
                    HTTPStatus.BAD_REQUEST,
                    correlation_id=correlation_id,
                )
                return
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    return ServiceHandler


def spawn_background(runtime_dir: Path, host: str, port: int) -> None:
    args = [sys.executable, "-m", "node_agent.service", "run", "--host", host, "--port", str(port)]
    kwargs: dict[str, Any] = {
        "cwd": runtime_dir,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(args, **kwargs)


def load_meta(runtime_dir: Path) -> dict[str, Any] | None:
    meta_path = runtime_dir / "data" / "service" / "service-meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def wait_for_service(host: str, port: int, timeout_seconds: float = 20.0) -> None:
    deadline = time.time() + timeout_seconds
    request_host = browser_access_host(host)
    while time.time() < deadline:
        try:
            response = httpx.get(f"http://{request_host}:{port}/api/healthz", timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("The node runtime service did not become ready in time.")


def open_browser(host: str, port: int, admin_token: str | None = None) -> None:
    url = f"http://{browser_access_host(host)}:{port}"
    if admin_token:
        url = f"{url}/?token={quote(admin_token)}"
    webbrowser.open(url, new=2)


def clear_service_meta(runtime_dir: Path) -> None:
    service_dir = runtime_dir / "data" / "service"
    for filename in ("service.pid", "service-meta.json"):
        path = service_dir / filename
        if path.exists():
            path.unlink()


def command_run(host: str, port: int) -> int:
    require_secure_bind_host(host)
    service = NodeRuntimeService()
    service.host = host
    service.port = port
    service.write_meta()
    service.log(f"Node runtime service started at http://{host}:{port}")
    service.resume_setup_if_needed()
    access_url = f"http://{browser_access_host(host)}:{port}/?token={quote(service.admin_token)}"
    print(f"AUTONOMOUSc node setup UI is available at {access_url}", flush=True)
    print(
        f"Runtime backend: {runtime_backend_label(service.runtime_backend)} on port 8765. "
        "Open that URL to run Quick Start.",
        flush=True,
    )

    server_ref: dict[str, ThreadingHTTPServer] = {}
    handler = make_handler(service, server_ref)
    server = ThreadingHTTPServer((host, port), handler)
    server_ref["server"] = server
    auto_thread = threading.Thread(target=service.auto_update_loop, daemon=True)
    auto_thread.start()
    local_doctor_thread = threading.Thread(target=service.local_doctor_loop, daemon=True)
    local_doctor_thread.start()
    self_heal_thread = threading.Thread(target=service.self_heal_loop, daemon=True)
    self_heal_thread.start()

    def request_stop(signum: int, _frame: Any) -> None:
        service.log(f"Node runtime service received signal {signum}.")
        service.shutdown_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, request_stop)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        service.log("Node runtime service interrupted from the terminal.")
    finally:
        service.shutdown_event.set()
        server.server_close()
        if service.runtime_controller is not None:
            service.runtime_controller.stop()
        service.clear_meta()
    return 0


def command_start(runtime_dir: Path, host: str, port: int, open_ui_flag: bool) -> int:
    require_secure_bind_host(host)
    meta = load_meta(runtime_dir)
    if meta:
        existing_host = str(meta.get("host", host))
        existing_port = int(meta.get("port", port))
        existing_token = meta.get("admin_token")
        try:
            wait_for_service(existing_host, existing_port, timeout_seconds=2.0)
            if open_ui_flag:
                open_browser(existing_host, existing_port, str(existing_token) if existing_token else None)
            return 0
        except RuntimeError:
            pass

    spawn_background(runtime_dir, host, port)
    wait_for_service(host, port)
    if open_ui_flag:
        meta = load_meta(runtime_dir)
        open_browser(host, port, str(meta.get("admin_token")) if meta and meta.get("admin_token") else None)
    return 0


def command_stop(runtime_dir: Path) -> int:
    meta = load_meta(runtime_dir)
    if meta:
        host = str(meta.get("host", "127.0.0.1"))
        request_host = browser_access_host(host)
        port = int(meta.get("port", 8765))
        admin_token = str(meta.get("admin_token", "") or "")
        try:
            httpx.post(
                f"http://{request_host}:{port}/api/shutdown",
                timeout=3.0,
                headers={ADMIN_TOKEN_HEADER: admin_token} if admin_token else None,
            )
            return 0
        except httpx.HTTPError:
            pass

        pid = int(meta.get("pid", 0) or 0)
        if pid:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        clear_service_meta(runtime_dir)
    return 0


def command_status(runtime_dir: Path) -> int:
    meta = load_meta(runtime_dir)
    if not meta:
        print("Node runtime service is not running.")
        return 0
    host = str(meta.get("host", "127.0.0.1"))
    request_host = browser_access_host(host)
    port = int(meta.get("port", 8765))
    admin_token = str(meta.get("admin_token", "") or "")
    headers = {ADMIN_TOKEN_HEADER: admin_token} if admin_token else None
    try:
        response = httpx.get(f"http://{request_host}:{port}/api/status", timeout=15.0, headers=headers)
        payload = response.json()
        print(json.dumps(payload["runtime"], indent=2))
    except httpx.HTTPError:
        try:
            health = httpx.get(f"http://{request_host}:{port}/api/healthz", timeout=2.0)
            if health.status_code == 200:
                print(
                    f"Node runtime service is online at http://{request_host}:{port}, "
                    "but full runtime status is still loading. Open the local UI or try again in a few seconds."
                )
                return 0
        except httpx.HTTPError:
            pass
        print("Node runtime service metadata exists, but the service did not respond.")
    return 0


def command_repair(runtime_dir: Path, host: str, port: int, open_ui_flag: bool) -> int:
    command_start(runtime_dir, host, port, False)
    meta = load_meta(runtime_dir)
    if not meta:
        raise RuntimeError("The local node runtime service could not be reached for repair.")

    target_host = str(meta.get("host", host))
    request_host = browser_access_host(target_host)
    target_port = int(meta.get("port", port))
    admin_token = str(meta.get("admin_token", "") or "")
    headers = {ADMIN_TOKEN_HEADER: admin_token} if admin_token else None
    response = httpx.post(f"http://{request_host}:{target_port}/api/repair", timeout=75.0, headers=headers)
    payload = response.json()
    if response.status_code >= 400 or payload.get("error"):
        detail = payload.get("error", {})
        message = (
            detail.get("message")
            if isinstance(detail, dict)
            else None
        ) or "The local repair request could not be completed."
        raise RuntimeError(str(message))
    if open_ui_flag:
        open_browser(target_host, target_port, admin_token or None)
    print(
        payload.get("self_healing", {}).get("detail")
        or payload.get("runtime", {}).get("message")
        or "Local repair started."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AUTONOMOUSc local node runtime service.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="Run the node runtime service in the foreground.")
    run_parser.add_argument("--host", default="127.0.0.1")
    run_parser.add_argument("--port", type=int, default=8765)

    start_parser = subparsers.add_parser("start", help="Start the node runtime service in the background.")
    start_parser.add_argument("--host", default="127.0.0.1")
    start_parser.add_argument("--port", type=int, default=8765)
    start_parser.add_argument("--open", action="store_true")

    subparsers.add_parser("stop", help="Stop the node runtime service.")
    subparsers.add_parser("status", help="Print the current runtime status.")

    open_parser = subparsers.add_parser("open", help="Open the local runtime UI in your browser.")
    open_parser.add_argument("--host", default="127.0.0.1")
    open_parser.add_argument("--port", type=int, default=8765)

    repair_parser = subparsers.add_parser("repair", help="Repair local owner setup and restart the runtime when safe.")
    repair_parser.add_argument("--host", default="127.0.0.1")
    repair_parser.add_argument("--port", type=int, default=8765)
    repair_parser.add_argument("--open", action="store_true")

    args = parser.parse_args(argv)
    command = args.command or "run"
    runtime_dir = ensure_runtime_bundle(resolve_runtime_dir())

    if command == "run":
        return command_run(args.host, args.port)
    if command == "start":
        return command_start(runtime_dir, args.host, args.port, args.open)
    if command == "stop":
        return command_stop(runtime_dir)
    if command == "status":
        return command_status(runtime_dir)
    if command == "open":
        meta = load_meta(runtime_dir)
        if meta:
            open_browser(
                str(meta.get("host", args.host)),
                int(meta.get("port", args.port)),
                str(meta.get("admin_token")) if meta.get("admin_token") else None,
            )
            return 0
        open_browser(args.host, args.port)
        return 0
    if command == "repair":
        return command_repair(runtime_dir, args.host, args.port, args.open)
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
