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
from urllib.parse import unquote, urlparse

import httpx

from .installer import CommandRunner, GuidedInstaller, parse_env_file, run_command
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

SENSITIVE_ENV_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "COOKIE", "CERT")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_id() -> str:
    return os.urandom(8).hex()


def service_html() -> str:
    return Path(__file__).with_name("service_ui.html").read_text(encoding="utf-8")


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


class NodeRuntimeService:
    def __init__(
        self,
        runtime_dir: Path | None = None,
        *,
        command_runner: CommandRunner = run_command,
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
        self.guided_installer = GuidedInstaller(runtime_dir=self.runtime_dir, command_runner=command_runner)
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.logs: list[str] = []
        self.update_state = UpdateState()
        self.diagnostics_state = DiagnosticsState()
        self.host = "127.0.0.1"
        self.port = 8765
        self.ensure_dirs()
        self.load_state()

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.service_dir, self.diagnostics_dir):
            path.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.update_state = UpdateState(**payload.get("updates", {}))
        self.diagnostics_state = DiagnosticsState(**payload.get("diagnostics", {}))

    def save_state(self) -> None:
        payload = {
            "updates": asdict(self.update_state),
            "diagnostics": asdict(self.diagnostics_state),
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log(self, message: str) -> None:
        with self.lock:
            stamped = f"[{now_iso()}] {message}"
            self.logs.append(stamped)
            self.logs = self.logs[-120:]

    def compose_command(self, args: list[str]) -> list[str]:
        command = ["docker", "compose"]
        if self.guided_installer.env_path.exists():
            command.extend(["--env-file", str(self.guided_installer.env_path)])
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
        env_values = parse_env_file(self.guided_installer.env_path)
        running_services = preflight.get("running_services", [])
        vllm_ready = False
        if "vllm" in running_services:
            try:
                response = httpx.get(f"http://{service_access_host()}:8000/v1/models", timeout=4.0)
                vllm_ready = response.status_code < 500
            except httpx.HTTPError:
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
        elif env_values:
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
            "config_present": self.guided_installer.env_path.exists(),
            "credentials_present": preflight.get("credentials_present", False),
        }

    def status_payload(self) -> dict[str, Any]:
        installer_snapshot = self.guided_installer.status_payload()
        preflight = installer_snapshot["preflight"]
        installer_state = installer_snapshot["state"]
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
            "runtime": self.runtime_status(preflight=preflight, installer_state=installer_state),
            "installer": installer_snapshot,
            "updates": asdict(self.update_state),
            "diagnostics": asdict(self.diagnostics_state),
        }

    def start_runtime(self) -> dict[str, Any]:
        self.log("Starting runtime services...")
        self.compose(["up", "-d", "vllm", "node-agent", "vector"])
        self.log("Runtime services started.")
        return self.status_payload()

    def stop_runtime(self) -> dict[str, Any]:
        self.log("Stopping runtime services...")
        self.compose(["stop", "node-agent", "vllm", "vector"])
        self.log("Runtime services stopped.")
        return self.status_payload()

    def restart_runtime(self) -> dict[str, Any]:
        self.log("Restarting runtime services...")
        self.compose(["up", "-d", "--force-recreate", "vllm", "node-agent", "vector"])
        self.update_state.pending_restart = False
        self.save_state()
        self.log("Runtime services restarted.")
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
        env_values = parse_env_file(
            self.guided_installer.env_path
            if self.guided_installer.env_path.exists()
            else self.guided_installer.example_env_path
        )
        lines: list[str] = []
        for key, value in env_values.items():
            if any(marker in key for marker in SENSITIVE_ENV_MARKERS):
                redacted = "***REDACTED***" if value else ""
                lines.append(f"{key}={redacted}")
            else:
                lines.append(f"{key}={value}")
        return "\n".join(lines) + ("\n" if lines else "")

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
        webbrowser.open(f"http://{self.host}:{self.port}", new=2)

    def auto_update_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                if self.update_state.auto_update_enabled:
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

    def request_shutdown(self) -> dict[str, Any]:
        self.log("Shutdown requested for node runtime service.")
        self.shutdown_event.set()
        return {"ok": True}

    def write_meta(self) -> None:
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self.meta_path.write_text(
            json.dumps({"pid": os.getpid(), "host": self.host, "port": self.port}, indent=2),
            encoding="utf-8",
        )

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

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(content_length) if content_length else b"{}"
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # pragma: no cover
            path = urlparse(self.path).path
            if path == "/":
                body = service_html().encode("utf-8")
                self.send_response(HTTPStatus.OK.value)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/healthz":
                self._send_json({"ok": True, "service": "node-runtime"})
                return
            if path == "/api/status":
                self._send_json(service.status_payload())
                return
            if path.startswith("/downloads/"):
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


def open_browser(host: str, port: int) -> None:
    webbrowser.open(f"http://{host}:{port}", new=2)


def clear_service_meta(runtime_dir: Path) -> None:
    service_dir = runtime_dir / "data" / "service"
    for filename in ("service.pid", "service-meta.json"):
        path = service_dir / filename
        if path.exists():
            path.unlink()


def command_run(host: str, port: int) -> int:
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
    meta = load_meta(runtime_dir)
    if meta:
        existing_host = str(meta.get("host", host))
        existing_port = int(meta.get("port", port))
        try:
            wait_for_service(existing_host, existing_port, timeout_seconds=2.0)
            if open_ui_flag:
                open_browser(existing_host, existing_port)
            return 0
        except RuntimeError:
            pass

    spawn_background(runtime_dir, host, port)
    wait_for_service(host, port)
    if open_ui_flag:
        open_browser(host, port)
    return 0


def command_stop(runtime_dir: Path) -> int:
    meta = load_meta(runtime_dir)
    if meta:
        host = str(meta.get("host", "127.0.0.1"))
        port = int(meta.get("port", 8765))
        try:
            httpx.post(f"http://{host}:{port}/api/shutdown", timeout=3.0)
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
    try:
        response = httpx.get(f"http://{host}:{port}/api/status", timeout=15.0)
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
        open_browser(args.host, args.port)
        return 0
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
