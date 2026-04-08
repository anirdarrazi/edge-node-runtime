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
from urllib.parse import urlparse

import httpx

from .config import NodeAgentSettings
from .control_plane import EdgeControlClient


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
ControlClientFactory = Callable[[NodeAgentSettings], EdgeControlClient]
SleepFn = Callable[[float], None]


ENV_ORDER = [
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


def installer_html() -> str:
    return Path(__file__).with_name("installer_ui.html").read_text(encoding="utf-8")


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
        sleep: SleepFn = time.sleep,
    ) -> None:
        self.runtime_dir = runtime_dir or Path(__file__).resolve().parents[2]
        self.data_dir = self.runtime_dir / "data"
        self.credentials_dir = self.data_dir / "credentials"
        self.credentials_path = self.credentials_dir / "node-credentials.json"
        self.env_path = self.runtime_dir / ".env"
        self.example_env_path = self.runtime_dir / ".env.example"
        self.command_runner = command_runner
        self.control_client_factory = control_client_factory
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

        return {
            "edge_control_url": first_nonempty(source.get("EDGE_CONTROL_URL"), default_settings.edge_control_url),
            "node_label": first_nonempty(source.get("NODE_LABEL"), default_settings.node_label),
            "node_region": first_nonempty(source.get("NODE_REGION"), default_settings.node_region),
            "trust_tier": first_nonempty(source.get("TRUST_TIER"), default_settings.trust_tier),
            "restricted_capable": source.get("RESTRICTED_CAPABLE", stringify_bool(default_settings.restricted_capable)).lower() == "true",
            "vllm_model": first_nonempty(source.get("VLLM_MODEL"), "meta-llama/Llama-3.1-8B-Instruct"),
            "supported_models": first_nonempty(source.get("SUPPORTED_MODELS"), default_settings.supported_models),
            "max_concurrent_assignments": first_nonempty(
                source.get("MAX_CONCURRENT_ASSIGNMENTS"),
                suggest_concurrency(gpu.get("memory_gb")),
            ),
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

        return {
            "docker_cli": docker_cli,
            "docker_compose": compose_ok,
            "docker_daemon": daemon_ok,
            "docker_error": docker_error,
            "gpu": gpu if gpu is not None else detect_gpu(self.command_runner, self.runtime_dir),
            "running_services": running_services,
            "credentials_present": self.credentials_path.exists(),
        }

    def state_payload(self) -> dict[str, Any]:
        with self.lock:
            return asdict(self.state)

    def status_payload(self) -> dict[str, Any]:
        source = parse_env_file(self.env_path if self.env_path.exists() else self.example_env_path)
        gpu = detect_gpu(self.command_runner, self.runtime_dir)
        return {
            "config": self.current_config(source=source, gpu=gpu),
            "preflight": self.collect_preflight(gpu=gpu),
            "state": self.state_payload(),
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

        config_gpu_name = str(config.get("gpu_name", "")).strip() or None
        config_gpu_memory = str(config.get("gpu_memory_gb", "")).strip() or None
        gpu_name = resolve_gpu_name(config_gpu_name or current.get("GPU_NAME"), detected_gpu.get("name"), defaults.gpu_name)
        gpu_memory = resolve_gpu_memory(
            config_gpu_memory or current.get("GPU_MEMORY_GB"),
            detected_gpu.get("memory_gb"),
            defaults.gpu_memory_gb,
        )

        return {
            "EDGE_CONTROL_URL": first_nonempty(str(config.get("edge_control_url", "")), current.get("EDGE_CONTROL_URL"), defaults.edge_control_url),
            "OPERATOR_TOKEN": current.get("OPERATOR_TOKEN", ""),
            "NODE_LABEL": first_nonempty(str(config.get("node_label", "")), current.get("NODE_LABEL"), defaults.node_label),
            "NODE_REGION": first_nonempty(str(config.get("node_region", "")), current.get("NODE_REGION"), defaults.node_region),
            "TRUST_TIER": first_nonempty(str(config.get("trust_tier", "")), current.get("TRUST_TIER"), defaults.trust_tier),
            "RESTRICTED_CAPABLE": stringify_bool(bool(config.get("restricted_capable", True))),
            "CREDENTIALS_PATH": "/var/lib/autonomousc/credentials/node-credentials.json",
            "VLLM_BASE_URL": current.get("VLLM_BASE_URL", defaults.vllm_base_url),
            "GPU_NAME": gpu_name,
            "GPU_MEMORY_GB": gpu_memory,
            "MAX_CONTEXT_TOKENS": current.get("MAX_CONTEXT_TOKENS", str(defaults.max_context_tokens)),
            "MAX_BATCH_TOKENS": current.get("MAX_BATCH_TOKENS", str(defaults.max_batch_tokens)),
            "MAX_CONCURRENT_ASSIGNMENTS": first_nonempty(
                str(config.get("max_concurrent_assignments", "")),
                current.get("MAX_CONCURRENT_ASSIGNMENTS"),
                suggest_concurrency(float(gpu_memory) if gpu_memory else None),
            ),
            "THERMAL_HEADROOM": current.get("THERMAL_HEADROOM", str(defaults.thermal_headroom)),
            "SUPPORTED_MODELS": first_nonempty(str(config.get("supported_models", "")), current.get("SUPPORTED_MODELS"), defaults.supported_models),
            "POLL_INTERVAL_SECONDS": current.get("POLL_INTERVAL_SECONDS", str(defaults.poll_interval_seconds)),
            "ATTESTATION_PROVIDER": current.get("ATTESTATION_PROVIDER", defaults.attestation_provider),
            "VLLM_MODEL": first_nonempty(str(config.get("vllm_model", "")), current.get("VLLM_MODEL"), "meta-llama/Llama-3.1-8B-Instruct"),
        }

    def write_env_file(self, env_values: dict[str, str]) -> None:
        lines = [f"{key}={env_values[key]}" for key in ENV_ORDER if key in env_values]
        self.env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def compose_up(self, services: list[str]) -> None:
        self.command_runner(["docker", "compose", "up", "-d", *services], self.runtime_dir)

    def wait_for_vllm(self, timeout_seconds: float = 240.0) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                response = httpx.get("http://127.0.0.1:8000/v1/models", timeout=5.0)
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
            vllm_base_url="http://127.0.0.1:8000",
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
            self.complete("Runtime started. The node agent will attest and begin polling the control plane automatically.")
        except Exception as error:  # pragma: no cover - exercised through tests
            self.log(f"Installer failed: {error}")
            self.set_error(str(error))


def make_handler(installer: GuidedInstaller) -> type[BaseHTTPRequestHandler]:
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

        def do_GET(self) -> None:  # pragma: no cover
            path = urlparse(self.path).path
            if path == "/":
                body = installer_html().encode("utf-8")
                self.send_response(HTTPStatus.OK.value)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/status":
                self._send_json(installer.status_payload())
                return
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # pragma: no cover
            path = urlparse(self.path).path
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

    installer = GuidedInstaller()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(installer))
    local_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{local_host}:{args.port}"

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
