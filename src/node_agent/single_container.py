from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
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
    def from_env(cls) -> "SingleContainerConfig":
        return cls(
            vllm_model=os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct").strip(),
            vllm_host=os.getenv("VLLM_HOST", "0.0.0.0").strip() or "0.0.0.0",
            vllm_port=env_int("VLLM_PORT", 8000),
            vllm_startup_timeout_seconds=env_int("VLLM_STARTUP_TIMEOUT_SECONDS", 600),
            vllm_server_command=tuple(
                split_command(os.getenv("VLLM_SERVER_COMMAND"))
                or [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
            ),
            vllm_extra_args=tuple(split_command(os.getenv("VLLM_EXTRA_ARGS"))),
            node_agent_command=tuple(split_command(os.getenv("NODE_AGENT_COMMAND")) or ["node-agent", "start"]),
            start_vllm=env_bool("START_VLLM", True),
        )

    @property
    def local_vllm_url(self) -> str:
        return f"http://127.0.0.1:{self.vllm_port}"


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


def wait_for_vllm_ready(config: SingleContainerConfig, process: subprocess.Popen[str] | None) -> None:
    deadline = time.monotonic() + max(1, config.vllm_startup_timeout_seconds)
    url = f"{config.local_vllm_url}/v1/models"
    last_error = "vLLM is still starting."
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited before it became ready with status {process.returncode}.")
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code < 500:
                print(f"vLLM is ready at {url}.", flush=True)
                return
            last_error = f"vLLM returned HTTP {response.status_code}."
        except httpx.HTTPError as error:
            last_error = str(error) or last_error
        print(f"Waiting for vLLM model {config.vllm_model} to warm: {last_error}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become ready in time: {last_error}")


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


def main() -> int:
    config = SingleContainerConfig.from_env()
    os.environ.setdefault("VLLM_BASE_URL", config.local_vllm_url)
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
            print(f"Starting vLLM in this container: {' '.join(shlex.quote(part) for part in vllm_command)}", flush=True)
            vllm_process = subprocess.Popen(vllm_command, text=True)
            wait_for_vllm_ready(config, vllm_process)
        else:
            print("START_VLLM=false; expecting VLLM_BASE_URL to point at an already running server.", flush=True)

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
