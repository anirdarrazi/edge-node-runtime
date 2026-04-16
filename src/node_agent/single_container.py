from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return default


def split_command(value: str | None) -> list[str]:
    return shlex.split(value or "")


@dataclass(frozen=True)
class SingleContainerConfig:
    vllm_model: str
    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8000
    vllm_startup_timeout_seconds: int = 600
    vllm_server_command: tuple[str, ...] = (sys.executable, "-m", "vllm.entrypoints.openai.api_server")
    vllm_extra_args: tuple[str, ...] = ()
    node_agent_command: tuple[str, ...] = ("node-agent", "start")
    start_vllm: bool = True

    @classmethod
    def from_mapping(cls, env: Mapping[str, str] | None = None) -> "SingleContainerConfig":
        values = env or os.environ

        def read(name: str, default: str | None = None) -> str | None:
            return values.get(name, default)

        return cls(
            vllm_model=str(read("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")).strip(),
            vllm_host=str(read("VLLM_HOST", "0.0.0.0")).strip() or "0.0.0.0",
            vllm_port=env_int_from_mapping(values, "VLLM_PORT", 8000),
            vllm_startup_timeout_seconds=env_int_from_mapping(values, "VLLM_STARTUP_TIMEOUT_SECONDS", 600),
            vllm_server_command=tuple(
                split_command(read("VLLM_SERVER_COMMAND"))
                or [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
            ),
            vllm_extra_args=tuple(split_command(read("VLLM_EXTRA_ARGS"))),
            node_agent_command=tuple(split_command(read("NODE_AGENT_COMMAND")) or ["node-agent", "start"]),
            start_vllm=env_bool_from_mapping(values, "START_VLLM", True),
        )

    @classmethod
    def from_env(cls) -> "SingleContainerConfig":
        return cls.from_mapping(os.environ)

    @property
    def local_inference_url(self) -> str:
        return f"http://127.0.0.1:{self.vllm_port}"

    @property
    def local_vllm_url(self) -> str:
        return self.local_inference_url


def build_vllm_command(config: SingleContainerConfig) -> list[str]:
    command = list(config.vllm_server_command)
    command.extend(
        [
            "--model",
            config.vllm_model,
            "--host",
            config.vllm_host,
            "--port",
            str(config.vllm_port),
        ]
    )
    command.extend(config.vllm_extra_args)
    return command


def env_bool_from_mapping(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int_from_mapping(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(str(env.get(name, str(default))).strip())
    except ValueError:
        return default


def wait_for_inference_runtime_ready(config: SingleContainerConfig, process: subprocess.Popen[str] | None) -> None:
    deadline = time.monotonic() + max(1, config.vllm_startup_timeout_seconds)
    url = f"{config.local_inference_url}/v1/models"
    last_error = "The local inference runtime is still starting."
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"The local inference runtime exited before it became ready with status {process.returncode}."
            )
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code < 500:
                print(f"The local inference runtime is ready at {url}.", flush=True)
                return
            last_error = f"The local inference runtime returned HTTP {response.status_code}."
        except httpx.HTTPError as error:
            last_error = str(error) or last_error
        print(f"Waiting for the local inference runtime model {config.vllm_model} to warm: {last_error}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"The local inference runtime did not become ready in time: {last_error}")


def wait_for_vllm_ready(config: SingleContainerConfig, process: subprocess.Popen[str] | None) -> None:
    wait_for_inference_runtime_ready(config, process)


def terminate_process(process: subprocess.Popen[str] | None, *, grace_seconds: float = 15.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_any_process(processes: Sequence[subprocess.Popen[str]]) -> subprocess.Popen[str]:
    while True:
        for process in processes:
            if process.poll() is not None:
                return process
        time.sleep(1)


class EmbeddedRuntimeSupervisor:
    def __init__(
        self,
        env_provider: Callable[[], dict[str, str]],
        *,
        cache_dir: Path,
        credentials_dir: Path,
        scratch_dir: Path,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.env_provider = env_provider
        self.cache_dir = cache_dir
        self.credentials_dir = credentials_dir
        self.scratch_dir = scratch_dir
        self.log = log or (lambda _message: None)
        self.lock = threading.Lock()
        self.vllm_process: subprocess.Popen[str] | None = None
        self.node_process: subprocess.Popen[str] | None = None
        self.last_exit_codes: dict[str, int] = {}

    def env_values(self) -> dict[str, str]:
        values = dict(self.env_provider())
        values.setdefault("VLLM_MODEL", values.get("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
        values.setdefault("RUNTIME_PROFILE", values.get("RUNTIME_PROFILE", "vast_vllm_safetensors"))
        values.setdefault("DEPLOYMENT_TARGET", values.get("DEPLOYMENT_TARGET", "vast_ai"))
        values.setdefault("INFERENCE_ENGINE", values.get("INFERENCE_ENGINE", "vllm"))
        values.setdefault("CAPACITY_CLASS", values.get("CAPACITY_CLASS", "elastic_burst"))
        values.setdefault("TEMPORARY_NODE", values.get("TEMPORARY_NODE", "true"))
        values.setdefault("BURST_PROVIDER", values.get("BURST_PROVIDER", "vast_ai"))
        values.setdefault("BURST_LEASE_PHASE", values.get("BURST_LEASE_PHASE", "accept_burst_work"))
        values.setdefault("BURST_COST_CEILING_USD", values.get("BURST_COST_CEILING_USD", "0.25"))
        values.setdefault("VLLM_HOST", values.get("VLLM_HOST", "0.0.0.0"))
        values.setdefault("VLLM_PORT", values.get("VLLM_PORT", "8000"))
        inference_base_url = values.get("INFERENCE_BASE_URL") or values.get("VLLM_BASE_URL")
        if not inference_base_url:
            inference_base_url = f"http://127.0.0.1:{values['VLLM_PORT']}"
        values.setdefault("INFERENCE_BASE_URL", inference_base_url)
        values.setdefault("VLLM_BASE_URL", values["INFERENCE_BASE_URL"])
        values.setdefault("CREDENTIALS_PATH", str(self.credentials_dir / "node-credentials.json"))
        values.setdefault("ATTESTATION_STATE_PATH", str(self.credentials_dir / "attestation-state.json"))
        values.setdefault("RECOVERY_NOTE_PATH", str(self.credentials_dir / "recovery-note.txt"))
        values.setdefault("AUTOPILOT_STATE_PATH", str(self.scratch_dir / "autopilot-state.json"))
        values.setdefault("HF_HOME", str(self.cache_dir))
        return values

    def config(self, env_values: Mapping[str, str] | None = None) -> SingleContainerConfig:
        return SingleContainerConfig.from_mapping(env_values or self.env_values())

    def process_env(self, env_values: Mapping[str, str] | None = None) -> dict[str, str]:
        merged = dict(os.environ)
        merged.update(self.env_values() if env_values is None else dict(env_values))
        return merged

    @staticmethod
    def process_running(process: subprocess.Popen[str] | None) -> bool:
        return process is not None and process.poll() is None

    def sync_processes(self) -> None:
        for name, process in (("vllm", self.vllm_process), ("node-agent", self.node_process)):
            if process is None:
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            self.last_exit_codes[name] = int(returncode)
            if name == "vllm":
                self.vllm_process = None
            else:
                self.node_process = None

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            self.sync_processes()
            running_services = []
            if self.process_running(self.vllm_process):
                running_services.append("vllm")
            if self.process_running(self.node_process):
                running_services.append("node-agent")
            return {
                "running_services": running_services,
                "service_statuses": {
                    "vllm": {
                        "running": self.process_running(self.vllm_process),
                        "pid": self.vllm_process.pid if self.process_running(self.vllm_process) else None,
                        "last_exit_code": self.last_exit_codes.get("vllm"),
                    },
                    "node-agent": {
                        "running": self.process_running(self.node_process),
                        "pid": self.node_process.pid if self.process_running(self.node_process) else None,
                        "last_exit_code": self.last_exit_codes.get("node-agent"),
                    },
                },
            }

    def ensure_dirs(self) -> None:
        for path in (self.cache_dir, self.credentials_dir, self.scratch_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _start_vllm_locked(
        self,
        *,
        config: SingleContainerConfig,
        process_env: Mapping[str, str],
    ) -> subprocess.Popen[str]:
        command = build_vllm_command(config)
        self.log(f"Starting the local inference runtime in this container: {' '.join(shlex.quote(part) for part in command)}")
        self.vllm_process = subprocess.Popen(command, text=True, env=dict(process_env))
        return self.vllm_process

    def start(
        self,
        *,
        recreate: bool,
        start_vllm: bool,
        start_node: bool,
    ) -> None:
        env_values = self.env_values()
        config = self.config(env_values)
        process_env = self.process_env(env_values)
        self.ensure_dirs()

        with self.lock:
            if recreate:
                self.stop_locked()
            self.sync_processes()
            if start_vllm and not self.process_running(self.vllm_process):
                self._start_vllm_locked(config=config, process_env=process_env)
            vllm_process = self.vllm_process

        if start_vllm:
            wait_for_inference_runtime_ready(config, vllm_process)

        with self.lock:
            self.sync_processes()
            if start_node and not self.process_running(self.node_process):
                command = list(config.node_agent_command)
                self.log(f"Starting node agent in this container: {' '.join(shlex.quote(part) for part in command)}")
                self.node_process = subprocess.Popen(command, text=True, env=process_env)

    def restart_vllm(self) -> None:
        env_values = self.env_values()
        config = self.config(env_values)
        process_env = self.process_env(env_values)
        self.ensure_dirs()

        with self.lock:
            terminate_process(self.vllm_process)
            self.sync_processes()
            vllm_process = self._start_vllm_locked(config=config, process_env=process_env)

        wait_for_inference_runtime_ready(config, vllm_process)

    def stop_locked(self) -> None:
        terminate_process(self.node_process)
        terminate_process(self.vllm_process)
        self.sync_processes()
        self.node_process = None
        self.vllm_process = None

    def stop(self) -> None:
        with self.lock:
            self.stop_locked()

    def wait_for_runtime_health(self, timeout_seconds: float = 90.0) -> None:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        config = self.config()
        last_failure = "Runtime services are still starting."
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            running_services = set(snapshot.get("running_services", []))
            if not {"vllm", "node-agent"}.issubset(running_services):
                missing = ", ".join(sorted({"vllm", "node-agent"} - running_services))
                last_failure = f"Runtime services are not healthy yet. Missing: {missing or 'unknown'}."
                time.sleep(2)
                continue
            try:
                response = httpx.get(f"{config.local_inference_url}/v1/models", timeout=4.0)
                if response.status_code < 500:
                    return
                last_failure = f"Local inference runtime health check returned HTTP {response.status_code}."
            except httpx.HTTPError as error:
                last_failure = str(error) or last_failure
            time.sleep(2)
        raise RuntimeError(last_failure)


def main() -> int:
    config = SingleContainerConfig.from_env()
    os.environ.setdefault("RUNTIME_PROFILE", "vast_vllm_safetensors")
    os.environ.setdefault("DEPLOYMENT_TARGET", "vast_ai")
    os.environ.setdefault("INFERENCE_ENGINE", "vllm")
    os.environ.setdefault("CAPACITY_CLASS", "elastic_burst")
    os.environ.setdefault("TEMPORARY_NODE", "true")
    os.environ.setdefault("BURST_PROVIDER", "vast_ai")
    os.environ.setdefault("BURST_LEASE_PHASE", "accept_burst_work")
    os.environ.setdefault("BURST_COST_CEILING_USD", "0.25")
    os.environ.setdefault("INFERENCE_BASE_URL", config.local_inference_url)
    os.environ.setdefault("VLLM_BASE_URL", os.environ["INFERENCE_BASE_URL"])
    os.environ.setdefault("CREDENTIALS_PATH", "/var/lib/autonomousc/credentials/node-credentials.json")
    os.environ.setdefault("ATTESTATION_STATE_PATH", "/var/lib/autonomousc/credentials/attestation-state.json")
    os.environ.setdefault("RECOVERY_NOTE_PATH", "/var/lib/autonomousc/credentials/recovery-note.txt")
    os.environ.setdefault("AUTOPILOT_STATE_PATH", "/var/lib/autonomousc/scratch/autopilot-state.json")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    vllm_process: subprocess.Popen[str] | None = None
    node_process: subprocess.Popen[str] | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        print(f"Single-container runtime received signal {signum}; shutting down.", flush=True)
        terminate_process(node_process)
        terminate_process(vllm_process)
        raise SystemExit(128 + signum)

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, handle_signal)

    try:
        if config.start_vllm:
            vllm_command = build_vllm_command(config)
            print(
                f"Starting the local inference runtime in this container: {' '.join(shlex.quote(part) for part in vllm_command)}",
                flush=True,
            )
            vllm_process = subprocess.Popen(vllm_command, text=True)
            wait_for_inference_runtime_ready(config, vllm_process)
        else:
            print(
                "START_VLLM=false; expecting INFERENCE_BASE_URL (or deprecated VLLM_BASE_URL) to point at an already running inference runtime.",
                flush=True,
            )

        node_command = list(config.node_agent_command)
        print(f"Starting node agent in this container: {' '.join(shlex.quote(part) for part in node_command)}", flush=True)
        node_process = subprocess.Popen(node_command, text=True)

        running = [process for process in (vllm_process, node_process) if process is not None]
        exited = wait_for_any_process(running)
        print(f"Process exited with status {exited.returncode}; stopping the single-container runtime.", flush=True)
        return int(exited.returncode or 0)
    finally:
        terminate_process(node_process)
        terminate_process(vllm_process)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
