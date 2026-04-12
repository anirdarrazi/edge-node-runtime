from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import httpx

from .autostart import AutoStartManager
from .control_plane import EdgeControlClient
from .desktop_launcher import DesktopLauncherManager
from .installer import CommandRunner, GuidedInstaller, parse_env_file, run_command
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
from .runtime_layout import ensure_runtime_bundle, resolve_runtime_dir, service_access_host

RUNTIME_SERVICES = ("vllm", "node-agent", "vector")
SELF_HEAL_INTERVAL_SECONDS = 45
REMOTE_DASHBOARD_CACHE_TTL_SECONDS = 30

SENSITIVE_ENV_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "COOKIE", "CERT")


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


@dataclass
class UpdateState:
    auto_update_enabled: bool = False
    interval_hours: int = 24
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
    last_error: str | None = None


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


@dataclass
class RemoteDashboardCacheState:
    fetched_at: str | None = None
    payload: dict[str, Any] | None = None
    last_error: str | None = None


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
        self.pid_path = self.service_dir / "service.pid"
        self.meta_path = self.service_dir / "service-meta.json"
        self.state_path = self.service_dir / "service-state.json"
        self.release_env_path = self.service_dir / "release.env"
        self.command_runner = command_runner
        self.autostart_manager = autostart_manager or AutoStartManager(
            self.runtime_dir,
            command_runner=command_runner,
        )
        self.desktop_launcher_manager = desktop_launcher_manager or DesktopLauncherManager(
            self.runtime_dir,
            command_runner=command_runner,
        )
        self.guided_installer = GuidedInstaller(
            runtime_dir=self.runtime_dir,
            command_runner=command_runner,
            autostart_manager=self.autostart_manager,
            desktop_launcher_manager=self.desktop_launcher_manager,
        )
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.repair_lock = threading.Lock()
        self.logs: list[str] = []
        self.update_state = UpdateState()
        self.diagnostics_state = DiagnosticsState()
        self.self_heal_state = SelfHealState()
        self.remote_dashboard_state = RemoteDashboardCacheState()
        self.started_at = now_iso()
        self.started_at_epoch = time.time()
        self.admin_token = generate_admin_token()
        self.sessions = LocalSessionStore()
        self.host = "127.0.0.1"
        self.port = 8765
        self.ensure_dirs()
        self.load_state()

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.service_dir, self.diagnostics_dir):
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
        self.diagnostics_state = DiagnosticsState(**payload.get("diagnostics", {}))
        self.self_heal_state = SelfHealState(**payload.get("self_healing", {}))

    def save_state(self) -> None:
        payload = {
            "updates": asdict(self.update_state),
            "diagnostics": asdict(self.diagnostics_state),
            "self_healing": asdict(self.self_heal_state),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tighten_private_path(self.state_path)

    def log(self, message: str) -> None:
        with self.lock:
            stamped = f"[{now_iso()}] {message}"
            self.logs.append(stamped)
            self.logs = self.logs[-120:]

    def update_self_heal_state(self, **updates: Any) -> None:
        changed = False
        for key, value in updates.items():
            if getattr(self.self_heal_state, key) != value:
                setattr(self.self_heal_state, key, value)
                changed = True
        if changed:
            self.save_state()

    def compose_command(self, args: list[str]) -> list[str]:
        self.guided_installer.sync_runtime_env()
        command = ["docker", "compose"]
        if self.guided_installer.runtime_env_path.exists():
            command.extend(["--env-file", str(self.guided_installer.runtime_env_path)])
        if self.release_env_path.exists():
            command.extend(["--env-file", str(self.release_env_path)])
        command.extend(args)
        return command

    def compose(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return self.command_runner(self.compose_command(args), self.runtime_dir)

    def signed_release_manifest(self) -> ReleaseManifest:
        manifest = load_release_manifest()
        self.update_state.release_version = manifest.version
        self.update_state.release_channel = manifest.channel
        return manifest

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

    def remember_known_good_release(self) -> None:
        snapshot = self.current_release_snapshot()
        if not snapshot:
            return
        self.update_self_heal_state(
            last_known_good_release_env=snapshot,
            last_healthy_at=now_iso(),
        )

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
        self.remember_known_good_release()
        self.save_state()
        return True

    def wait_for_runtime_health(self, timeout_seconds: float = 90.0) -> None:
        deadline = time.time() + timeout_seconds
        required_services = set(RUNTIME_SERVICES)
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
                response = httpx.get(f"http://{service_access_host()}:8000/v1/models", timeout=4.0)
                if response.status_code < 500:
                    return
                last_failure = f"vLLM health check returned HTTP {response.status_code}."
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
        vllm_ready = False
        current_model = str(env_values.get("VLLM_MODEL") or "").strip() or None
        if "vllm" in running_services:
            try:
                response = httpx.get(f"http://{service_access_host()}:8000/v1/models", timeout=4.0)
                vllm_ready = response.status_code < 500
                if response.status_code < 500:
                    payload = response.json()
                    if isinstance(payload, dict):
                        data = payload.get("data")
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
                                    current_model = item["id"].strip()
                                    break
            except (ValueError, httpx.HTTPError):
                vllm_ready = False

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
            "vllm_ready": vllm_ready,
            "current_model": current_model,
            "config_present": config_present,
            "credentials_present": preflight.get("credentials_present", False),
        }

    def runtime_health_snapshot(self, *, installer_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        installer_snapshot = installer_snapshot or self.guided_installer.status_payload()
        preflight = installer_snapshot["preflight"]
        installer_state = installer_snapshot["state"]
        runtime = self.runtime_status(preflight=preflight, installer_state=installer_state)
        running_services = set(preflight.get("running_services", []))
        docker_ready = bool(preflight.get("docker_cli") and preflight.get("docker_compose") and preflight.get("docker_daemon"))
        config_present = bool(runtime.get("config_present"))
        credentials_present = bool(runtime.get("credentials_present"))
        required_running = set(RUNTIME_SERVICES).issubset(running_services)
        runtime_healthy = bool(
            credentials_present
            and runtime.get("stage") == "running"
            and required_running
            and runtime.get("vllm_ready")
        )

        issue_code: str | None = None
        issue_detail: str | None = None
        automatic_fix_available = False
        manual_fix_available = False

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
        elif not docker_ready:
            issue_code = "docker_unavailable"
            issue_detail = str(
                preflight.get("docker_error")
                or "Docker needs attention before the runtime can be repaired."
            )
        elif credentials_present and not running_services:
            issue_code = "runtime_stopped"
            issue_detail = "The runtime is stopped even though this machine already has stored node credentials."
            automatic_fix_available = True
            manual_fix_available = True
        elif credentials_present and not required_running:
            missing = ", ".join(sorted(set(RUNTIME_SERVICES) - running_services)) or "unknown services"
            issue_code = "stuck_containers"
            issue_detail = f"Some runtime containers look stuck or missing: {missing}."
            automatic_fix_available = True
            manual_fix_available = True
        elif credentials_present and not runtime.get("vllm_ready"):
            issue_code = "runtime_unhealthy"
            issue_detail = "The runtime containers are up, but vLLM is not responding yet."
            automatic_fix_available = True
            manual_fix_available = True
        elif not credentials_present and config_present:
            issue_code = "approval_required"
            issue_detail = "This machine still needs approval before the runtime can come online."
            manual_fix_available = True

        return {
            "installer_snapshot": installer_snapshot,
            "preflight": preflight,
            "installer_state": installer_state,
            "runtime": runtime,
            "issue_code": issue_code,
            "issue_detail": issue_detail,
            "runtime_healthy": runtime_healthy,
            "docker_ready": docker_ready,
            "credentials_present": credentials_present,
            "config_present": config_present,
            "running_services": sorted(running_services),
            "automatic_fix_available": automatic_fix_available,
            "manual_fix_available": manual_fix_available,
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
                headline = "Self-healing found something to fix"
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
            "action_label": "Fix it",
        }

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

    def dashboard_payload(self, health: dict[str, Any], installer_snapshot: dict[str, Any]) -> dict[str, Any]:
        runtime = health.get("runtime", {})
        installer_state = health.get("installer_state", {})
        config = installer_snapshot.get("config", {})
        updates = asdict(self.update_state)
        remote = self.remote_dashboard_snapshot()
        remote_summary = remote.get("summary") if isinstance(remote.get("summary"), dict) else {}
        remote_node = remote_summary.get("node") if isinstance(remote_summary.get("node"), dict) else {}
        remote_observability = (
            remote_node.get("observability") if isinstance(remote_node.get("observability"), dict) else {}
        )
        remote_runtime = remote_node.get("runtime") if isinstance(remote_node.get("runtime"), dict) else {}
        earnings = remote_summary.get("earnings") if isinstance(remote_summary.get("earnings"), dict) else {}

        blocked_reason = (
            remote_summary.get("blocked_reason")
            if isinstance(remote_summary.get("blocked_reason"), str)
            else remote_observability.get("schedulability_reason")
        )
        if not isinstance(blocked_reason, str) or not blocked_reason.strip():
            blocked_reason = None

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
            model_value = current_model
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

        if blocked_reason:
            blocked_value = "Action needed"
            blocked_detail = blocked_reason
            blocked_tone = "danger" if str(remote_node.get("status") or "") in {"paused", "revoked"} else "warning"
        else:
            blocked_value = "None"
            blocked_detail = "The control plane is not currently blocking this node."
            blocked_tone = "success"

        if updates.get("last_error"):
            update_value = "Needs attention"
            update_detail = str(updates["last_error"])
            update_tone = "danger"
        elif updates.get("pending_restart"):
            update_value = "Restart needed"
            update_detail = str(updates.get("last_result") or "A signed runtime update is ready to take over after a restart.")
            update_tone = "warning"
        elif updates.get("release_version"):
            update_value = "Up to date"
            update_detail = str(
                updates.get("last_result")
                or f"Signed release {updates['release_version']} is active on this machine."
            )
            update_tone = "success"
        else:
            update_value = "Manual"
            update_detail = str(updates.get("last_result") or "No signed update checks have run yet.")
            update_tone = "warning"

        if health_value == "Healthy":
            headline = "Node live"
            detail = "Health, earnings, heartbeat, and update state are all visible here."
        elif health_value == "Setting up":
            headline = "Quick Start is in progress"
            detail = health_detail
        elif health_value == "Blocked":
            headline = "Node needs attention"
            detail = health_detail
        else:
            headline = "Owner dashboard"
            detail = health_detail

        return {
            "headline": headline,
            "detail": detail,
            "sync": {
                "synced_at": remote.get("synced_at"),
                "last_error": remote.get("last_error"),
                "stale": bool(remote.get("stale")),
            },
            "cards": {
                "health": {"value": health_value, "detail": health_detail, "tone": health_tone},
                "earnings": {"value": earnings_value, "detail": earnings_detail, "tone": earnings_tone},
                "heartbeat": {"value": heartbeat_value, "detail": heartbeat_detail, "tone": heartbeat_tone},
                "model": {"value": model_value, "detail": model_detail, "tone": model_tone},
                "uptime": {"value": uptime_value, "detail": uptime_detail, "tone": uptime_tone},
                "blocked": {"value": blocked_value, "detail": blocked_detail, "tone": blocked_tone},
                "updates": {"value": update_value, "detail": update_detail, "tone": update_tone},
            },
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
            "dashboard": self.dashboard_payload(health, installer_snapshot),
            "autostart": self.autostart_manager.status(),
            "desktop_launcher": self.desktop_launcher_manager.status(),
            "updates": asdict(self.update_state),
            "diagnostics": asdict(self.diagnostics_state),
            "self_healing": self.self_heal_payload(health),
        }

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

    def start_runtime_services(self, *, recreate: bool) -> None:
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

    def attempt_runtime_recovery(self, *, recreate: bool, failure_reason: str) -> tuple[bool, str, str]:
        self.start_runtime_services(recreate=recreate)
        try:
            self.wait_for_runtime_health()
        except Exception as error:
            detail = str(error) or failure_reason
            if self.rollback_to_known_good_release(detail):
                return True, "Self-healing rolled back to the last known healthy signed release.", "rollback_bad_update"
            return False, detail, "restart_runtime" if recreate else "start_runtime"

        self.remember_known_good_release()
        return True, "The runtime is healthy again.", "restart_runtime" if recreate else "start_runtime"

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
            self.ensure_local_config()

            try:
                autostart = self.autostart_manager.ensure_enabled()
                self.log(str(autostart.get("detail") or "Automatic start status is unavailable."))
            except Exception as error:
                self.log(f"Automatic start could not be repaired automatically: {error}")

            try:
                launcher = self.desktop_launcher_manager.ensure_enabled()
                self.log(str(launcher.get("detail") or "Desktop launcher status is unavailable."))
            except Exception as error:
                self.log(f"The desktop launcher could not be repaired automatically: {error}")

            health = self.runtime_health_snapshot()
            installer_state = health["installer_state"]
            preflight = health["preflight"]

            if installer_state.get("busy"):
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
            elif health["runtime_healthy"]:
                self.remember_known_good_release()
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

    def check_for_updates(self, apply: bool = False) -> dict[str, Any]:
        rollout_id = request_id()
        try:
            manifest = self.signed_release_manifest()
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

            self.update_state.last_checked_at = now_iso()
            self.update_state.updated_images = updated
            self.update_state.release_version = manifest.version
            self.update_state.release_channel = manifest.channel
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
            self.write_release_env(target_release)
            try:
                self.restart_runtime()
                self.wait_for_runtime_health()
            except Exception as error:
                self.restore_release_env(previous_content)
                rollback_id = request_id()
                self.log(
                    f"[request {rollout_id}] Signed release {manifest.version} failed health checks: {error}"
                )
                try:
                    self.log(f"[request {rollback_id}] Rolling back to the previous signed release.")
                    self.restart_runtime()
                    self.wait_for_runtime_health()
                except Exception as rollback_error:  # pragma: no cover - defensive fallback
                    self.log(
                        f"[request {rollback_id}] Rollback failed after signed release "
                        f"{manifest.version}: {rollback_error}"
                    )
                self.update_state.last_error = (
                    f"Signed release {manifest.version} could not be applied automatically. "
                    f"Check the service log with request id {rollout_id}."
                )
                self.update_state.last_result = "The previous signed runtime release was restored."
                self.update_state.pending_restart = False
                self.save_state()
                return self.status_payload()

            self.update_state.last_result = f"Signed release {manifest.version} was applied successfully."
            self.update_state.pending_restart = False
            self.remember_known_good_release()
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

    def configure_updates(self, enabled: bool, interval_hours: int) -> dict[str, Any]:
        self.update_state.auto_update_enabled = enabled
        self.update_state.interval_hours = max(1, int(interval_hours))
        self.update_state.last_error = None
        self.save_state()
        self.log(
            "Auto-update settings changed: "
            f"{'enabled' if enabled else 'disabled'} every {self.update_state.interval_hours} hour(s)."
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

    def create_diagnostics_bundle(self) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        bundle_name = f"diagnostics-{timestamp}.zip"
        bundle_path = self.diagnostics_dir / bundle_name
        self.log("Creating diagnostics bundle...")

        payload = self.status_payload()
        payload["generated_at"] = now_iso()

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("status.json", json.dumps(payload, indent=2))
            archive.writestr("env.redacted", self.redacted_env())
            if self.guided_installer.runtime_settings_path.exists():
                archive.writestr("runtime-settings.redacted.json", self.redacted_runtime_settings())
            archive.writestr("service.log", "\n".join(payload["service"]["logs"]) + "\n")
            archive.writestr("installer.log", "\n".join(payload["installer"]["state"]["logs"]) + "\n")
            archive.writestr(
                "docker-compose-ps.txt",
                self.command_output(["docker", "compose", "ps"]),
            )
            archive.writestr(
                "docker-compose.logs.txt",
                self.command_output(["docker", "compose", "logs", "--tail", "200", "node-agent", "vllm", "vector"]),
            )
            archive.writestr(
                "nvidia-smi.txt",
                self.command_output(["nvidia-smi"]),
            )
            recovery_note = self.guided_installer.credentials_dir / "recovery-note.txt"
            if recovery_note.exists():
                archive.writestr("recovery-note.txt", recovery_note.read_text(encoding="utf-8"))

        self.diagnostics_state.last_bundle_name = bundle_name
        self.diagnostics_state.last_bundle_created_at = now_iso()
        self.diagnostics_state.last_error = None
        self.save_state()
        self.log(f"Diagnostics bundle created: {bundle_name}")
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
                        self.check_for_updates(apply=True)
            except Exception as error:  # pragma: no cover
                loop_request_id = request_id()
                self.update_state.last_error = (
                    f"Automatic signed runtime release check failed. "
                    f"Check the service log with request id {loop_request_id}."
                )
                self.save_state()
                self.log(f"[request {loop_request_id}] Automatic signed runtime release check failed: {error}")
            self.shutdown_event.wait(60)

    def self_heal_check(self) -> dict[str, Any]:
        health = self.runtime_health_snapshot()
        self.update_self_heal_state(
            last_checked_at=now_iso(),
            fix_available=bool(health.get("manual_fix_available")),
        )

        if health["runtime_healthy"]:
            self.remember_known_good_release()
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
                    self.self_heal_check()
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
                if path == "/api/runtime/start":
                    self._send_json(service.start_runtime())
                    return
                if path == "/api/runtime/stop":
                    self._send_json(service.stop_runtime())
                    return
                if path == "/api/runtime/restart":
                    self._send_json(service.restart_runtime())
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
                if path == "/api/diagnostics":
                    self._send_json(service.create_diagnostics_bundle())
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
    while time.time() < deadline:
        try:
            response = httpx.get(f"http://{host}:{port}/api/healthz", timeout=2.0)
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

    server_ref: dict[str, ThreadingHTTPServer] = {}
    handler = make_handler(service, server_ref)
    server = ThreadingHTTPServer((host, port), handler)
    server_ref["server"] = server
    auto_thread = threading.Thread(target=service.auto_update_loop, daemon=True)
    auto_thread.start()
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
        port = int(meta.get("port", 8765))
        admin_token = str(meta.get("admin_token", "") or "")
        try:
            httpx.post(
                f"http://{host}:{port}/api/shutdown",
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
    port = int(meta.get("port", 8765))
    admin_token = str(meta.get("admin_token", "") or "")
    headers = {ADMIN_TOKEN_HEADER: admin_token} if admin_token else None
    try:
        response = httpx.get(f"http://{host}:{port}/api/status", timeout=15.0, headers=headers)
        payload = response.json()
        print(json.dumps(payload["runtime"], indent=2))
    except httpx.HTTPError:
        try:
            health = httpx.get(f"http://{host}:{port}/api/healthz", timeout=2.0)
            if health.status_code == 200:
                print(
                    f"Node runtime service is online at http://{host}:{port}, "
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
    target_port = int(meta.get("port", port))
    admin_token = str(meta.get("admin_token", "") or "")
    headers = {ADMIN_TOKEN_HEADER: admin_token} if admin_token else None
    response = httpx.post(f"http://{target_host}:{target_port}/api/repair", timeout=75.0, headers=headers)
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
