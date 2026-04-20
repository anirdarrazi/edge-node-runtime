from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Sequence

import httpx

DEFAULT_GATED_STARTUP_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PUBLIC_BOOTSTRAP_MODEL = "BAAI/bge-large-en-v1.5"
MIN_VRAM_FOR_GATED_STARTUP_GB = 24.0
GATED_HF_REPOSITORY_PREFIXES = ("meta-llama/",)
KNOWN_SAFE_MAX_MODEL_LEN: dict[str, int] = {
    DEFAULT_PUBLIC_BOOTSTRAP_MODEL: 512,
}


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    return stripped.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def env_float_from_mapping(env: Mapping[str, str], name: str, default: float | None = None) -> float | None:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def split_command(value: str | None) -> list[str]:
    return shlex.split(value or "")


def known_safe_max_model_len(model: str | None) -> int | None:
    normalized = str(model or "").strip()
    value = KNOWN_SAFE_MAX_MODEL_LEN.get(normalized)
    return value if isinstance(value, int) and value > 0 else None


def normalized_max_context_tokens(model: str | None, configured_max_context_tokens: int) -> int:
    if configured_max_context_tokens <= 0:
        return configured_max_context_tokens
    safe_limit = known_safe_max_model_len(model)
    if safe_limit is None:
        return configured_max_context_tokens
    return min(configured_max_context_tokens, safe_limit)


@dataclass(frozen=True)
class SingleContainerConfig:
    vllm_model: str
    vllm_host: str = "0.0.0.0"
    vllm_port: int = 8000
    max_context_tokens: int = 32768
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

        vllm_model = nonempty_value(values, "VLLM_MODEL", DEFAULT_GATED_STARTUP_MODEL)

        return cls(
            vllm_model=vllm_model,
            vllm_host=nonempty_value(values, "VLLM_HOST", "0.0.0.0"),
            vllm_port=env_int_from_mapping(values, "VLLM_PORT", 8000),
            max_context_tokens=normalized_max_context_tokens(
                vllm_model,
                env_int_from_mapping(values, "MAX_CONTEXT_TOKENS", 32768),
            ),
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
    effective_max_context_tokens = normalized_max_context_tokens(config.vllm_model, config.max_context_tokens)
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
    if effective_max_context_tokens > 0 and not has_cli_flag(command, "--max-model-len"):
        command.extend(["--max-model-len", str(effective_max_context_tokens)])
    return command


def has_cli_flag(command: Sequence[str], flag: str) -> bool:
    return any(part == flag or part.startswith(f"{flag}=") for part in command)


def env_bool_from_mapping(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    return stripped.lower() in {"1", "true", "yes", "on"}


def env_int_from_mapping(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def nonempty_value(values: Mapping[str, str], name: str, default: str) -> str:
    value = values.get(name)
    if value is None:
        return default
    stripped = str(value).strip()
    return stripped or default


def value_is_blank(values: Mapping[str, str], name: str) -> bool:
    value = values.get(name)
    return value is None or not str(value).strip()


def force_when_blank_or(values: MutableMapping[str, str], name: str, default: str, legacy_values: set[str]) -> None:
    current = str(values.get(name, "")).strip()
    normalized = current.lower().replace("-", "_")
    if not current or normalized in legacy_values:
        values[name] = default


def configured_hugging_face_token(values: Mapping[str, str]) -> str | None:
    for key in ("HUGGING_FACE_HUB_TOKEN", "HF_TOKEN"):
        token = str(values.get(key, "")).strip()
        if token:
            return token
    return None


def hugging_face_token_configured(values: Mapping[str, str]) -> bool:
    return configured_hugging_face_token(values) is not None


def requires_gated_hugging_face_access(model: str | None) -> bool:
    normalized = str(model or "").strip().lower()
    return any(normalized.startswith(prefix) for prefix in GATED_HF_REPOSITORY_PREFIXES)


def should_use_public_bootstrap(values: Mapping[str, str], current_model: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if requires_gated_hugging_face_access(current_model) and not hugging_face_token_configured(values):
        reasons.append(
            f"no Hugging Face token is configured for the gated startup model {current_model}"
        )
    gpu_memory_gb = env_float_from_mapping(values, "GPU_MEMORY_GB")
    if gpu_memory_gb is not None and gpu_memory_gb < MIN_VRAM_FOR_GATED_STARTUP_GB:
        reasons.append(
            f"this machine reports only {gpu_memory_gb:.1f} GB VRAM, so the public embeddings bootstrap is safer"
        )
    return bool(reasons), reasons


def apply_public_bootstrap_fallback(values: MutableMapping[str, str]) -> str | None:
    current_model = nonempty_value(values, "VLLM_MODEL", DEFAULT_GATED_STARTUP_MODEL)
    if current_model == DEFAULT_PUBLIC_BOOTSTRAP_MODEL:
        return None

    should_fallback, reasons = should_use_public_bootstrap(values, current_model)
    if not should_fallback:
        return None

    current_supported_models = str(values.get("SUPPORTED_MODELS", "")).strip()
    if not str(values.get("OWNER_TARGET_MODEL", "")).strip():
        values["OWNER_TARGET_MODEL"] = current_model
    if not str(values.get("OWNER_TARGET_SUPPORTED_MODELS", "")).strip():
        values["OWNER_TARGET_SUPPORTED_MODELS"] = (
            current_supported_models or f"{current_model},{DEFAULT_PUBLIC_BOOTSTRAP_MODEL}"
        )

    values["VLLM_MODEL"] = DEFAULT_PUBLIC_BOOTSTRAP_MODEL
    values["SUPPORTED_MODELS"] = DEFAULT_PUBLIC_BOOTSTRAP_MODEL
    reason_text = " and ".join(reasons)
    owner_target = str(values.get("OWNER_TARGET_MODEL") or current_model).strip() or current_model
    return (
        f"Using the public bootstrap model {DEFAULT_PUBLIC_BOOTSTRAP_MODEL} because {reason_text}. "
        f"{owner_target} stays saved as the larger owner target."
    )


def apply_single_container_runtime_defaults(
    values: MutableMapping[str, str],
    *,
    local_inference_url: str | None = None,
) -> str | None:
    values["VLLM_MODEL"] = nonempty_value(values, "VLLM_MODEL", DEFAULT_GATED_STARTUP_MODEL)
    values["VLLM_HOST"] = nonempty_value(values, "VLLM_HOST", "0.0.0.0")
    values["VLLM_PORT"] = str(env_int_from_mapping(values, "VLLM_PORT", 8000))

    force_when_blank_or(
        values,
        "RUNTIME_PROFILE",
        "vast_vllm_safetensors",
        {"auto", "home_llama_cpp_gguf", "home_embeddings_llama_cpp"},
    )
    force_when_blank_or(values, "DEPLOYMENT_TARGET", "vast_ai", {"auto", "home_edge"})
    force_when_blank_or(values, "INFERENCE_ENGINE", "vllm", {"auto", "llama_cpp"})
    force_when_blank_or(values, "CAPACITY_CLASS", "elastic_burst", {"home_heat", "generic"})
    force_when_blank_or(values, "TEMPORARY_NODE", "true", {"false", "0", "no", "off"})
    force_when_blank_or(values, "BURST_PROVIDER", "vast_ai", set())
    force_when_blank_or(values, "BURST_LEASE_PHASE", "accept_burst_work", set())
    force_when_blank_or(values, "BURST_COST_CEILING_USD", "0.25", set())
    fallback_note = apply_public_bootstrap_fallback(values)
    values["MAX_CONTEXT_TOKENS"] = str(
        normalized_max_context_tokens(
            values["VLLM_MODEL"],
            env_int_from_mapping(values, "MAX_CONTEXT_TOKENS", 32768),
        )
    )

    start_vllm = env_bool_from_mapping(values, "START_VLLM", True)
    inference_url = local_inference_url or f"http://127.0.0.1:{values['VLLM_PORT']}"
    compose_service_urls = {"http://inference-runtime:8000", "http://vllm:8000"}
    for key in ("INFERENCE_BASE_URL", "VLLM_BASE_URL"):
        current = str(values.get(key, "")).strip()
        if value_is_blank(values, key) or (start_vllm and current in compose_service_urls):
            values[key] = inference_url
    return fallback_note


def validate_gated_model_access(values: Mapping[str, str], model: str) -> None:
    if not requires_gated_hugging_face_access(model):
        return

    token = configured_hugging_face_token(values)
    if not token:
        return

    try:
        response = httpx.get(
            f"https://huggingface.co/api/models/{model}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"Could not validate Hugging Face access for {model}. Check the network connection and token, then retry."
        ) from error

    if response.status_code == HTTPStatus.OK:
        return
    if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
        raise RuntimeError(
            f"Hugging Face denied access to {model}. Make sure HUGGING_FACE_HUB_TOKEN or HF_TOKEN is valid and approved for this Meta model."
        )
    if response.status_code == HTTPStatus.NOT_FOUND:
        raise RuntimeError(f"The startup model {model} is unavailable on Hugging Face right now.")
    raise RuntimeError(
        f"Could not validate Hugging Face access for {model}. Hugging Face returned HTTP {response.status_code}."
    )


class ProcessOutputRelay:
    def __init__(self, *, max_lines: int = 40) -> None:
        self.max_lines = max_lines
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None

    def attach(self, process: subprocess.Popen[str]) -> None:
        stream = getattr(process, "stdout", None)
        if stream is None:
            return

        def _pump() -> None:
            try:
                for raw_line in stream:
                    line = raw_line.rstrip("\r\n")
                    print(line, flush=True)
                    with self.lock:
                        self.lines.append(line)
            finally:
                stream.close()

        self.thread = threading.Thread(target=_pump, name="vllm-output-relay", daemon=True)
        self.thread.start()

    def tail_text(self) -> str:
        with self.lock:
            return "\n".join(self.lines).strip()


def wait_for_inference_runtime_ready(
    config: SingleContainerConfig,
    process: subprocess.Popen[str] | None,
    *,
    output_tail: Callable[[], str] | None = None,
) -> None:
    deadline = time.monotonic() + max(1, config.vllm_startup_timeout_seconds)
    url = f"{config.local_inference_url}/v1/models"
    last_error = "The local inference runtime is still starting."
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            recent_output = output_tail().strip() if output_tail is not None else ""
            if recent_output:
                raise RuntimeError(
                    "The local inference runtime exited before it became ready "
                    f"with status {process.returncode}. Recent vLLM output:\n{recent_output}"
                )
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
        self.vllm_output_relay: ProcessOutputRelay | None = None

    def prepared_env_values(self) -> tuple[dict[str, str], str | None]:
        values = dict(self.env_provider())
        fallback_note = apply_single_container_runtime_defaults(values)
        values.setdefault("CREDENTIALS_PATH", str(self.credentials_dir / "node-credentials.json"))
        values.setdefault("ATTESTATION_STATE_PATH", str(self.credentials_dir / "attestation-state.json"))
        values.setdefault("RECOVERY_NOTE_PATH", str(self.credentials_dir / "recovery-note.txt"))
        values.setdefault("AUTOPILOT_STATE_PATH", str(self.scratch_dir / "autopilot-state.json"))
        values.setdefault("HF_HOME", str(self.cache_dir))
        return values, fallback_note

    def env_values(self) -> dict[str, str]:
        values, _fallback_note = self.prepared_env_values()
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
        self.vllm_output_relay = ProcessOutputRelay()
        self.vllm_process = subprocess.Popen(
            command,
            text=True,
            env=dict(process_env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self.vllm_output_relay.attach(self.vllm_process)
        return self.vllm_process

    def start(
        self,
        *,
        recreate: bool,
        start_vllm: bool,
        start_node: bool,
    ) -> None:
        env_values, fallback_note = self.prepared_env_values()
        config = self.config(env_values)
        process_env = self.process_env(env_values)
        self.ensure_dirs()
        if fallback_note:
            self.log(fallback_note)
        validate_gated_model_access(process_env, config.vllm_model)

        with self.lock:
            if recreate:
                self.stop_locked()
            self.sync_processes()
            if start_vllm and not self.process_running(self.vllm_process):
                self._start_vllm_locked(config=config, process_env=process_env)
            vllm_process = self.vllm_process

        if start_vllm:
            wait_for_inference_runtime_ready(
                config,
                vllm_process,
                output_tail=self.vllm_output_relay.tail_text if self.vllm_output_relay is not None else None,
            )

        with self.lock:
            self.sync_processes()
            if start_node and not self.process_running(self.node_process):
                command = list(config.node_agent_command)
                self.log(f"Starting node agent in this container: {' '.join(shlex.quote(part) for part in command)}")
                self.node_process = subprocess.Popen(command, text=True, env=process_env)

    def restart_vllm(self) -> None:
        env_values, fallback_note = self.prepared_env_values()
        config = self.config(env_values)
        process_env = self.process_env(env_values)
        self.ensure_dirs()
        if fallback_note:
            self.log(fallback_note)
        validate_gated_model_access(process_env, config.vllm_model)

        with self.lock:
            terminate_process(self.vllm_process)
            self.sync_processes()
            vllm_process = self._start_vllm_locked(config=config, process_env=process_env)

        wait_for_inference_runtime_ready(
            config,
            vllm_process,
            output_tail=self.vllm_output_relay.tail_text if self.vllm_output_relay is not None else None,
        )

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
    initial_config = SingleContainerConfig.from_env()
    fallback_note = apply_single_container_runtime_defaults(
        os.environ,
        local_inference_url=initial_config.local_inference_url,
    )
    config = SingleContainerConfig.from_env()
    os.environ.setdefault("CREDENTIALS_PATH", "/var/lib/autonomousc/credentials/node-credentials.json")
    os.environ.setdefault("ATTESTATION_STATE_PATH", "/var/lib/autonomousc/credentials/attestation-state.json")
    os.environ.setdefault("RECOVERY_NOTE_PATH", "/var/lib/autonomousc/credentials/recovery-note.txt")
    os.environ.setdefault("AUTOPILOT_STATE_PATH", "/var/lib/autonomousc/scratch/autopilot-state.json")
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    if fallback_note:
        print(fallback_note, flush=True)
    validate_gated_model_access(os.environ, config.vllm_model)

    vllm_process: subprocess.Popen[str] | None = None
    node_process: subprocess.Popen[str] | None = None
    vllm_output_relay: ProcessOutputRelay | None = None

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
            vllm_output_relay = ProcessOutputRelay()
            vllm_process = subprocess.Popen(
                vllm_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            vllm_output_relay.attach(vllm_process)
            wait_for_inference_runtime_ready(
                config,
                vllm_process,
                output_tail=vllm_output_relay.tail_text if vllm_output_relay is not None else None,
            )
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
