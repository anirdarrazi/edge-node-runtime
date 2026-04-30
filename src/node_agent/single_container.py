from __future__ import annotations

import json
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
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

import httpx

from .fault_injection import DEFAULT_FAULT_INJECTION_STATE_NAME, FaultInjectionController

DEFAULT_GATED_STARTUP_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PUBLIC_BOOTSTRAP_MODEL = "BAAI/bge-large-en-v1.5"
MIN_VRAM_FOR_GATED_STARTUP_GB = 24.0
GATED_HF_REPOSITORY_PREFIXES = ("meta-llama/",)
DEFAULT_RUN_MODE = "full"
SERVE_ONLY_RUN_MODE = "serve_only"
DEFAULT_STARTUP_STATUS_FILENAME = "startup-status.json"
DEFAULT_STARTUP_STATUS_HOST = "0.0.0.0"
DEFAULT_STARTUP_STATUS_PORT = 8011
DEFAULT_STARTUP_STATUS_ENDPOINT_PATH = "/startup-status"
DEFAULT_STARTUP_EMBEDDINGS_PATH = "/v1/embeddings"
DEFAULT_STARTUP_RESPONSES_PATH = "/v1/responses"
DEFAULT_STARTUP_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_STARTUP_EMBEDDINGS_INPUT = ["startup readiness probe"]
DEFAULT_STARTUP_RESPONSES_INPUT = "Reply with the single word ready."
KNOWN_SAFE_MAX_MODEL_LEN: dict[str, int] = {
    DEFAULT_PUBLIC_BOOTSTRAP_MODEL.lower(): 512,
    "google/gemma-4-e4b-it": 32768,
    "google/gemma-4-26b-a4b-it": 8192,
    "aeyeops/gemma-4-26b-a4b-it-fp8": 8192,
}
KNOWN_SAFE_STARTUP_TIMEOUT_SECONDS: dict[str, int] = {
    "google/gemma-4-e4b-it": 900,
    "google/gemma-4-26b-a4b-it": 900,
    "google/gemma-4-31b-it": 2400,
    "aeyeops/gemma-4-26b-a4b-it-fp8": 900,
}
VLLM_PROCESS_ENV_EXCLUDE = (
    "VLLM_MODEL",
    "VLLM_HOST",
    "VLLM_PORT",
    "VLLM_BASE_URL",
    "INFERENCE_BASE_URL",
    "VLLM_SERVER_COMMAND",
    "VLLM_EXTRA_ARGS",
    "NODE_AGENT_COMMAND",
    "START_NODE_AGENT",
    "START_VLLM",
    "RUN_MODE",
    "SUPPORTED_MODELS",
    "OWNER_TARGET_MODEL",
    "OWNER_TARGET_SUPPORTED_MODELS",
)


def trimmed_output(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


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


def normalize_run_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized == SERVE_ONLY_RUN_MODE:
        return SERVE_ONLY_RUN_MODE
    return DEFAULT_RUN_MODE


def fault_injection_state_path_from_mapping(env: Mapping[str, str] | None = None) -> Path:
    values = env or os.environ
    configured = str(values.get("FAULT_INJECTION_STATE_PATH") or "").strip()
    if configured:
        return Path(configured)
    autopilot_path = str(values.get("AUTOPILOT_STATE_PATH") or "").strip()
    if autopilot_path:
        return Path(autopilot_path).with_name(DEFAULT_FAULT_INJECTION_STATE_NAME)
    credentials_path = str(values.get("CREDENTIALS_PATH") or "").strip()
    if credentials_path:
        return Path(credentials_path).parent / DEFAULT_FAULT_INJECTION_STATE_NAME
    return Path(".") / DEFAULT_FAULT_INJECTION_STATE_NAME


def startup_status_path_from_mapping(env: Mapping[str, str] | None = None) -> Path:
    values = env or os.environ
    configured = str(values.get("STARTUP_STATUS_PATH") or "").strip()
    if configured:
        return Path(configured)
    autopilot_path = str(values.get("AUTOPILOT_STATE_PATH") or "").strip()
    if autopilot_path:
        return Path(autopilot_path).with_name(DEFAULT_STARTUP_STATUS_FILENAME)
    credentials_path = str(values.get("CREDENTIALS_PATH") or "").strip()
    if credentials_path:
        return Path(credentials_path).parent / DEFAULT_STARTUP_STATUS_FILENAME
    return Path(".") / DEFAULT_STARTUP_STATUS_FILENAME


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_endpoint_path(value: str | None) -> str:
    stripped = str(value or "").strip()
    if not stripped:
        return DEFAULT_STARTUP_STATUS_ENDPOINT_PATH
    return stripped if stripped.startswith("/") else f"/{stripped}"


def nonnegative_int_from_mapping(env: Mapping[str, str], name: str, default: int) -> int:
    value = env_int_from_mapping(env, name, default)
    return value if value >= 0 else default


def normalized_model_lookup_key(model: str | None) -> str:
    normalized = str(model or "").strip().lower()
    if "gemma-4-26b-a4b-it" in normalized:
        return "google/gemma-4-26b-a4b-it"
    if "gemma-4-e4b-it" in normalized:
        return "google/gemma-4-e4b-it"
    return normalized


def known_safe_max_model_len(model: str | None) -> int | None:
    value = KNOWN_SAFE_MAX_MODEL_LEN.get(normalized_model_lookup_key(model))
    return value if isinstance(value, int) and value > 0 else None


def known_safe_startup_timeout_seconds(model: str | None) -> int | None:
    value = KNOWN_SAFE_STARTUP_TIMEOUT_SECONDS.get(normalized_model_lookup_key(model))
    return value if isinstance(value, int) and value > 0 else None


def normalized_max_context_tokens(model: str | None, configured_max_context_tokens: int) -> int:
    if configured_max_context_tokens <= 0:
        return configured_max_context_tokens
    safe_limit = known_safe_max_model_len(model)
    if safe_limit is None:
        return configured_max_context_tokens
    return min(configured_max_context_tokens, safe_limit)


def sanitized_vllm_process_env(env_values: Mapping[str, str]) -> dict[str, str]:
    merged = dict(os.environ)
    merged.update(dict(env_values))
    for key in VLLM_PROCESS_ENV_EXCLUDE:
        merged.pop(key, None)
    return merged


def startup_failure_kind(message: str | None, recent_output: str | None = None) -> str | None:
    haystack = f"{message or ''}\n{recent_output or ''}".lower()
    if any(token in haystack for token in ("cuda out of memory", "not enough gpu memory", "outofmemoryerror", "out of memory")):
        return "gpu_memory_oom"
    if (
        "keyerror:" in haystack
        and ("experts.0.down_proj.weight" in haystack or "params_dict[name]" in haystack)
    ):
        return "checkpoint_incompatible"
    if "unsupported display driver / cuda driver combination" in haystack:
        return "cuda_driver_mismatch"
    if "engine core initialization failed" in haystack:
        return "engine_core_start_failed"
    return None


def startup_failure_detail(
    config: "SingleContainerConfig",
    failure_kind: str | None,
    recent_output: str | None = None,
) -> str | None:
    detail: str | None = None
    if failure_kind == "gpu_memory_oom":
        detail = (
            f"Not enough GPU memory to warm {config.vllm_model} on this host. "
            "The model-load path ran out of memory before readiness. "
            "Lower MAX_CONTEXT_TOKENS, use a larger-VRAM host, or add tensor parallelism."
        )
    elif failure_kind == "checkpoint_incompatible":
        detail = (
            f"The checkpoint for {config.vllm_model} does not match the Gemma4 weight layout "
            "expected by the current vLLM loader. "
            "This is a checkpoint/runtime compatibility problem, not just a slow warmup. "
            "Try a different checkpoint build for this model family or use a runtime with confirmed support for it."
        )
    if detail and recent_output:
        detail = f"{detail} Recent vLLM output:\n{recent_output}"
    return detail


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
    run_mode: str = DEFAULT_RUN_MODE
    start_vllm: bool = True
    start_node_agent: bool = True
    startup_status_path: Path = Path(DEFAULT_STARTUP_STATUS_FILENAME)
    startup_status_host: str = DEFAULT_STARTUP_STATUS_HOST
    startup_status_port: int = DEFAULT_STARTUP_STATUS_PORT
    startup_status_endpoint_path: str = DEFAULT_STARTUP_STATUS_ENDPOINT_PATH

    @classmethod
    def from_mapping(cls, env: Mapping[str, str] | None = None) -> "SingleContainerConfig":
        values = env or os.environ

        def read(name: str, default: str | None = None) -> str | None:
            return values.get(name, default)

        vllm_model = nonempty_value(values, "VLLM_MODEL", DEFAULT_GATED_STARTUP_MODEL)
        run_mode = normalize_run_mode(values.get("RUN_MODE"))
        start_node_agent = env_bool_from_mapping(values, "START_NODE_AGENT", True)
        if run_mode == SERVE_ONLY_RUN_MODE:
            start_node_agent = False

        return cls(
            vllm_model=vllm_model,
            vllm_host=nonempty_value(values, "VLLM_HOST", "0.0.0.0"),
            vllm_port=env_int_from_mapping(values, "VLLM_PORT", 8000),
            max_context_tokens=normalized_max_context_tokens(
                vllm_model,
                env_int_from_mapping(values, "MAX_CONTEXT_TOKENS", 32768),
            ),
            vllm_startup_timeout_seconds=(
                env_int_from_mapping(values, "VLLM_STARTUP_TIMEOUT_SECONDS", 600)
                if str(values.get("VLLM_STARTUP_TIMEOUT_SECONDS") or "").strip()
                else (known_safe_startup_timeout_seconds(vllm_model) or 600)
            ),
            vllm_server_command=tuple(
                split_command(read("VLLM_SERVER_COMMAND"))
                or [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
            ),
            vllm_extra_args=tuple(split_command(read("VLLM_EXTRA_ARGS"))),
            node_agent_command=tuple(split_command(read("NODE_AGENT_COMMAND")) or ["node-agent", "start"]),
            run_mode=run_mode,
            start_vllm=env_bool_from_mapping(values, "START_VLLM", True),
            start_node_agent=start_node_agent,
            startup_status_path=startup_status_path_from_mapping(values),
            startup_status_host=nonempty_value(values, "STARTUP_STATUS_HOST", DEFAULT_STARTUP_STATUS_HOST),
            startup_status_port=nonnegative_int_from_mapping(values, "STARTUP_STATUS_PORT", DEFAULT_STARTUP_STATUS_PORT),
            startup_status_endpoint_path=normalized_endpoint_path(
                values.get("STARTUP_STATUS_ENDPOINT_PATH", DEFAULT_STARTUP_STATUS_ENDPOINT_PATH)
            ),
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

    @property
    def local_startup_status_url(self) -> str | None:
        if self.startup_status_port <= 0:
            return None
        return f"http://127.0.0.1:{self.startup_status_port}{self.startup_status_endpoint_path}"


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


def startup_runtime_detail(
    config: SingleContainerConfig,
    *,
    startup_stage: str,
    last_ready_error: str | None = None,
    recent_vllm_output: str | None = None,
    vllm_process_exit_code: int | None = None,
    failed_service: str | None = None,
    failure_kind: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "startup_stage": startup_stage,
        "local_inference_url": config.local_inference_url,
        "effective_max_context_tokens": config.max_context_tokens,
        "vllm_command": build_vllm_command(config),
    }
    recent_output = trimmed_output(recent_vllm_output)
    if last_ready_error:
        payload["last_ready_error"] = last_ready_error
    if recent_output:
        payload["recent_vllm_output"] = recent_output
    if vllm_process_exit_code is not None:
        payload["vllm_process_exit_code"] = int(vllm_process_exit_code)
    if failed_service:
        payload["failed_service"] = failed_service
    if failure_kind:
        payload["failure_kind"] = failure_kind
    return payload


def startup_probe_kind_for_model(model: str | None) -> str:
    normalized = str(model or "").strip()
    if normalized == DEFAULT_PUBLIC_BOOTSTRAP_MODEL:
        return "embeddings"
    return "responses"


def run_local_inference_probe(config: SingleContainerConfig, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
    base_url = config.local_inference_url
    probe_kind = startup_probe_kind_for_model(config.vllm_model)
    if probe_kind == "embeddings":
        response = httpx.post(
            f"{base_url}{DEFAULT_STARTUP_EMBEDDINGS_PATH}",
            json={"model": config.vllm_model, "input": DEFAULT_STARTUP_EMBEDDINGS_INPUT},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        embedding_length = 0
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            embedding = data[0].get("embedding")
            if isinstance(embedding, list):
                embedding_length = len(embedding)
        if embedding_length <= 0:
            raise RuntimeError(
                f"The startup embeddings probe succeeded with HTTP {response.status_code}, but no embedding vector was returned."
            )
        return {
            "probe_api": "embeddings",
            "probe_path": DEFAULT_STARTUP_EMBEDDINGS_PATH,
            "probe_status": int(response.status_code),
            "embedding_length": embedding_length,
        }

    response = httpx.post(
        f"{base_url}{DEFAULT_STARTUP_RESPONSES_PATH}",
        json={"model": config.vllm_model, "input": DEFAULT_STARTUP_RESPONSES_INPUT},
        timeout=timeout_seconds,
    )
    try:
        response.raise_for_status()
        payload = response.json()
        return {
            "probe_api": "responses",
            "probe_path": DEFAULT_STARTUP_RESPONSES_PATH,
            "probe_status": int(response.status_code),
            "output_present": bool(trimmed_output(json.dumps(payload) if isinstance(payload, dict) else "")),
        }
    except httpx.HTTPStatusError as error:
        if error.response.status_code not in {400, 404, 405, 422}:
            raise

    fallback = httpx.post(
        f"{base_url}{DEFAULT_STARTUP_CHAT_COMPLETIONS_PATH}",
        json={
            "model": config.vllm_model,
            "messages": [{"role": "user", "content": DEFAULT_STARTUP_RESPONSES_INPUT}],
            "max_tokens": 8,
        },
        timeout=timeout_seconds,
    )
    fallback.raise_for_status()
    payload = fallback.json()
    choices = payload.get("choices")
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
        raise RuntimeError(
            f"The startup chat-completions probe succeeded with HTTP {fallback.status_code}, but no choices were returned."
        )
    return {
        "probe_api": "chat_completions_fallback",
        "probe_path": DEFAULT_STARTUP_CHAT_COMPLETIONS_PATH,
        "probe_status": int(fallback.status_code),
    }


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
    run_mode = normalize_run_mode(values.get("RUN_MODE"))
    values["RUN_MODE"] = run_mode
    if run_mode == SERVE_ONLY_RUN_MODE:
        values["START_NODE_AGENT"] = "false"
    elif value_is_blank(values, "START_NODE_AGENT"):
        values["START_NODE_AGENT"] = "true"
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


def startup_warm_source_payload(values: Mapping[str, str], *, current_model: str) -> dict[str, Any]:
    warm_source = str(values.get("STARTUP_WARM_SOURCE") or "").strip()
    warm_source_label = str(values.get("STARTUP_WARM_SOURCE_LABEL") or "").strip()
    warm_source_detail = str(values.get("STARTUP_WARM_SOURCE_DETAIL") or "").strip()
    warm_source_scope = str(values.get("STARTUP_WARM_SOURCE_SCOPE") or "").strip()
    warm_source_order = str(values.get("STARTUP_WARM_SOURCE_ORDER") or "").strip()
    warm_source_selected_at = str(values.get("STARTUP_WARM_SOURCE_SELECTED_AT") or "").strip()

    if not warm_source:
        if not env_bool_from_mapping(values, "START_VLLM", True):
            warm_source = "external_runtime"
            warm_source_label = warm_source_label or "External runtime"
            warm_source_detail = warm_source_detail or "This container expects an already running inference runtime."
            warm_source_scope = warm_source_scope or "external"
        elif current_model == DEFAULT_PUBLIC_BOOTSTRAP_MODEL:
            warm_source = "bootstrap_fallback"
            warm_source_label = warm_source_label or "Bootstrap fallback"
            warm_source_detail = warm_source_detail or (
                "The container fell back to the public bootstrap model because it was the safest startup choice."
            )
            warm_source_scope = warm_source_scope or "planned"
        else:
            warm_source = "unknown"
            warm_source_label = warm_source_label or "Unknown"
            warm_source_scope = warm_source_scope or "unknown"

    payload: dict[str, Any] = {
        "warm_source": warm_source,
        "warm_source_label": warm_source_label or None,
        "warm_source_detail": warm_source_detail or None,
        "warm_source_scope": warm_source_scope or None,
        "warm_source_selected_at": warm_source_selected_at or None,
    }
    if warm_source_order:
        payload["warm_source_order"] = [item.strip() for item in warm_source_order.split(",") if item.strip()]
    return payload


class StartupStatusPublisher:
    def __init__(
        self,
        path: Path,
        *,
        host: str = DEFAULT_STARTUP_STATUS_HOST,
        port: int = DEFAULT_STARTUP_STATUS_PORT,
        endpoint_path: str = DEFAULT_STARTUP_STATUS_ENDPOINT_PATH,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path
        self.host = host
        self.port = max(0, int(port))
        self.endpoint_path = normalized_endpoint_path(endpoint_path)
        self.log = log or (lambda _message: None)
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {}
        self.server: ThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.server_error: str | None = None
        self.self_check_error: str | None = None
        self.self_check_ok: bool | None = None

    @property
    def local_url(self) -> str | None:
        if self.port <= 0:
            return None
        return f"http://127.0.0.1:{self.port}{self.endpoint_path}"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload = dict(self.payload)
            payload["status_path"] = str(self.path)
            payload["status_url"] = self.local_url
            if self.server_error:
                payload["status_url_error"] = self.server_error
            if self.self_check_ok is not None:
                payload["status_url_self_check_ok"] = self.self_check_ok
            if self.self_check_error:
                payload["status_url_self_check_error"] = self.self_check_error
            return payload

    def _write_payload_locked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f"{self.path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(self.path)
        self.payload = payload

    def publish(
        self,
        *,
        status: str,
        current_model: str,
        failure_reason: str | None = None,
        warm_source: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "current_model": current_model,
            "failure_reason": failure_reason or None,
            "updated_at": utc_now_iso(),
        }
        if warm_source:
            payload.update({key: value for key, value in dict(warm_source).items() if value not in (None, "", [])})
        if extra:
            payload.update({key: value for key, value in dict(extra).items() if value is not None})
        payload["status_path"] = str(self.path)
        payload["status_url"] = self.local_url
        if self.server_error:
            payload["status_url_error"] = self.server_error
        with self.lock:
            self._write_payload_locked(payload)
        self.ensure_server()
        self.refresh_self_check()
        return self.snapshot()

    def ensure_server(self) -> None:
        if self.port <= 0:
            return
        with self.lock:
            if self.server is not None:
                return
            publisher = self

            class _StartupStatusHandler(BaseHTTPRequestHandler):
                def do_GET(self) -> None:  # noqa: N802
                    if self.path != publisher.endpoint_path:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    body = json.dumps(publisher.snapshot(), sort_keys=True).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                    return

            class _ThreadedStartupStatusServer(ThreadingHTTPServer):
                daemon_threads = True

            try:
                self.server = _ThreadedStartupStatusServer((self.host, self.port), _StartupStatusHandler)
            except OSError as error:
                self.server_error = f"Could not bind startup-status endpoint on {self.host}:{self.port}: {error}"
                self.log(self.server_error)
                return

            def _serve() -> None:
                assert self.server is not None
                try:
                    self.server.serve_forever(poll_interval=0.5)
                except Exception as error:  # pragma: no cover - defensive
                    self.server_error = f"Startup-status endpoint stopped unexpectedly: {error}"
                    self.log(self.server_error)

            self.server_thread = threading.Thread(target=_serve, name="startup-status-server", daemon=True)
            self.server_thread.start()

    def refresh_self_check(self) -> None:
        url = self.local_url
        if self.port <= 0 or not url or self.server is None:
            return
        last_error = None
        for _ in range(5):
            try:
                response = httpx.get(url, timeout=1.0)
                if response.status_code == HTTPStatus.OK:
                    with self.lock:
                        self.self_check_ok = True
                        self.self_check_error = None
                        if self.payload:
                            payload = dict(self.payload)
                            payload["status_url_self_check_ok"] = True
                            payload.pop("status_url_self_check_error", None)
                            self._write_payload_locked(payload)
                    return
                last_error = f"Startup-status self-check returned HTTP {response.status_code}."
            except httpx.HTTPError as error:
                last_error = str(error) or "Startup-status self-check failed."
            time.sleep(0.1)
        with self.lock:
            self.self_check_ok = False
            self.self_check_error = last_error
            if self.payload:
                payload = dict(self.payload)
                payload["status_url_self_check_ok"] = False
                if last_error:
                    payload["status_url_self_check_error"] = last_error
                self._write_payload_locked(payload)

    def close(self) -> None:
        with self.lock:
            server = self.server
            thread = self.server_thread
            self.server = None
            self.server_thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)


class ProcessOutputRelay:
    def __init__(self, *, max_lines: int = 120) -> None:
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
    fault_state_path: Path | None = None,
    progress_callback: Callable[[str, str | None, int | None], None] | None = None,
) -> None:
    faults = FaultInjectionController(fault_state_path or fault_injection_state_path_from_mapping())
    deadline = time.monotonic() + max(1, config.vllm_startup_timeout_seconds)
    url = f"{config.local_inference_url}/v1/models"
    last_error = "The local inference runtime is still starting."
    while time.monotonic() < deadline:
        recent_output = trimmed_output(output_tail() if output_tail is not None else None)
        if faults.consume("warm_gpu_oom"):
            if progress_callback is not None:
                progress_callback(last_error, recent_output, None)
            raise RuntimeError(
                f"CUDA out of memory while warming {config.vllm_model}. "
                "The live fault drill forced a GPU OOM before the runtime became ready."
            )
        if process is not None and process.poll() is not None:
            if progress_callback is not None:
                progress_callback(last_error, recent_output, process.returncode)
            failure_kind = startup_failure_kind(last_error, recent_output)
            detail = startup_failure_detail(config, failure_kind, recent_output)
            if detail:
                raise RuntimeError(detail)
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
                payload = response.json()
                if isinstance(payload, dict):
                    try:
                        probe_result = run_local_inference_probe(config, timeout_seconds=5.0)
                        print(
                            "The local inference runtime is ready at "
                            f"{url} after {probe_result['probe_api']} succeeded on {probe_result['probe_path']}.",
                            flush=True,
                        )
                        return
                    except (httpx.HTTPError, RuntimeError, ValueError) as error:
                        last_error = str(error) or "The startup inference probe is still waiting for the runtime."
                else:
                    last_error = f"{url} returned non-JSON content."
            else:
                last_error = f"The local inference runtime returned HTTP {response.status_code}."
        except httpx.HTTPError as error:
            last_error = str(error) or last_error
        if progress_callback is not None:
            progress_callback(last_error, recent_output, None)
        print(f"Waiting for the local inference runtime model {config.vllm_model} to warm: {last_error}", flush=True)
        time.sleep(5)
    recent_output = trimmed_output(output_tail() if output_tail is not None else None)
    if progress_callback is not None:
        progress_callback(last_error, recent_output, process.returncode if process is not None else None)
    failure_kind = startup_failure_kind(last_error, recent_output)
    detail = startup_failure_detail(config, failure_kind, recent_output)
    if detail:
        raise RuntimeError(detail)
    if recent_output:
        raise RuntimeError(
            "The local inference runtime did not become ready in time: "
            f"{last_error}. Recent vLLM output:\n{recent_output}"
        )
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


def expected_runtime_services(config: SingleContainerConfig) -> tuple[str, ...]:
    services: list[str] = []
    if config.start_vllm:
        services.append("vllm")
    if config.start_node_agent:
        services.append("node-agent")
    return tuple(services)


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
        self.startup_status_publisher: StartupStatusPublisher | None = None
        self.startup_status_signature: tuple[str, str, int, str] | None = None

    def prepared_env_values(self) -> tuple[dict[str, str], str | None]:
        values = dict(self.env_provider())
        fallback_note = apply_single_container_runtime_defaults(values)
        values.setdefault("CREDENTIALS_PATH", str(self.credentials_dir / "node-credentials.json"))
        values.setdefault("ATTESTATION_STATE_PATH", str(self.credentials_dir / "attestation-state.json"))
        values.setdefault("RECOVERY_NOTE_PATH", str(self.credentials_dir / "recovery-note.txt"))
        values.setdefault("AUTOPILOT_STATE_PATH", str(self.scratch_dir / "autopilot-state.json"))
        values.setdefault("HEAT_GOVERNOR_STATE_PATH", str(self.scratch_dir / "heat-governor-state.json"))
        values.setdefault("FAULT_INJECTION_STATE_PATH", str(self.scratch_dir / DEFAULT_FAULT_INJECTION_STATE_NAME))
        values.setdefault("STARTUP_STATUS_PATH", str(self.scratch_dir / DEFAULT_STARTUP_STATUS_FILENAME))
        values.setdefault("STARTUP_STATUS_HOST", DEFAULT_STARTUP_STATUS_HOST)
        values.setdefault("STARTUP_STATUS_PORT", str(DEFAULT_STARTUP_STATUS_PORT))
        values.setdefault("STARTUP_STATUS_ENDPOINT_PATH", DEFAULT_STARTUP_STATUS_ENDPOINT_PATH)
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

    def startup_status_publisher_for_env(self, env_values: Mapping[str, str]) -> StartupStatusPublisher:
        config = self.config(env_values)
        signature = (
            str(config.startup_status_path),
            config.startup_status_host,
            int(config.startup_status_port),
            config.startup_status_endpoint_path,
        )
        if self.startup_status_publisher is None or self.startup_status_signature != signature:
            if self.startup_status_publisher is not None:
                self.startup_status_publisher.close()
            self.startup_status_publisher = StartupStatusPublisher(
                config.startup_status_path,
                host=config.startup_status_host,
                port=config.startup_status_port,
                endpoint_path=config.startup_status_endpoint_path,
                log=self.log,
            )
            self.startup_status_signature = signature
        return self.startup_status_publisher

    def publish_startup_status(
        self,
        env_values: Mapping[str, str],
        *,
        status: str,
        config: SingleContainerConfig | None = None,
        failure_reason: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_config = config or self.config(env_values)
        publisher = self.startup_status_publisher_for_env(env_values)
        return publisher.publish(
            status=status,
            current_model=resolved_config.vllm_model,
            failure_reason=failure_reason,
            warm_source=startup_warm_source_payload(env_values, current_model=resolved_config.vllm_model),
            extra={
                "run_mode": resolved_config.run_mode,
                "start_vllm": resolved_config.start_vllm,
                "start_node_agent": resolved_config.start_node_agent,
                **(dict(extra) if extra is not None else {}),
            },
        )

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
        env_values, _fallback_note = self.prepared_env_values()
        config = self.config(env_values)
        startup_status = self.startup_status_publisher_for_env(env_values).snapshot()
        with self.lock:
            self.sync_processes()
            running_services = []
            if self.process_running(self.vllm_process):
                running_services.append("vllm")
            if self.process_running(self.node_process):
                running_services.append("node-agent")
            return {
                "running_services": running_services,
                "expected_services": list(expected_runtime_services(config)),
                "run_mode": config.run_mode,
                "service_statuses": {
                    "vllm": {
                        "enabled": config.start_vllm,
                        "running": self.process_running(self.vllm_process),
                        "pid": self.vllm_process.pid if self.process_running(self.vllm_process) else None,
                        "last_exit_code": self.last_exit_codes.get("vllm"),
                    },
                    "node-agent": {
                        "enabled": config.start_node_agent,
                        "running": self.process_running(self.node_process),
                        "pid": self.node_process.pid if self.process_running(self.node_process) else None,
                        "last_exit_code": self.last_exit_codes.get("node-agent"),
                    },
                },
                "startup_status": startup_status,
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
            env=sanitized_vllm_process_env(process_env),
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
        effective_start_vllm = start_vllm and config.start_vllm
        effective_start_node = start_node and config.start_node_agent
        self.publish_startup_status(
            env_values,
            status="warming",
            config=config,
            extra=startup_runtime_detail(config, startup_stage="launching_vllm"),
        )

        try:
            with self.lock:
                if recreate:
                    self.stop_locked()
                self.sync_processes()
                if effective_start_vllm and not self.process_running(self.vllm_process):
                    self._start_vllm_locked(config=config, process_env=process_env)
                vllm_process = self.vllm_process

            if effective_start_vllm:
                def _publish_progress(last_ready_error: str, recent_vllm_output: str | None, exit_code: int | None) -> None:
                    self.publish_startup_status(
                        env_values,
                        status="warming",
                        config=config,
                        extra=startup_runtime_detail(
                            config,
                            startup_stage="warming_model",
                            last_ready_error=last_ready_error,
                            recent_vllm_output=recent_vllm_output,
                            vllm_process_exit_code=exit_code,
                        ),
                    )

                wait_for_inference_runtime_ready(
                    config,
                    vllm_process,
                    output_tail=self.vllm_output_relay.tail_text if self.vllm_output_relay is not None else None,
                    fault_state_path=fault_injection_state_path_from_mapping(process_env),
                    progress_callback=_publish_progress,
                )

            with self.lock:
                self.sync_processes()
                if effective_start_node and not self.process_running(self.node_process):
                    self.publish_startup_status(
                        env_values,
                        status="warming",
                        config=config,
                        extra=startup_runtime_detail(config, startup_stage="starting_node_agent"),
                    )
                    command = list(config.node_agent_command)
                    self.log(f"Starting node agent in this container: {' '.join(shlex.quote(part) for part in command)}")
                    self.node_process = subprocess.Popen(command, text=True, env=process_env)
            self.publish_startup_status(
                env_values,
                status="ready",
                config=config,
                extra=startup_runtime_detail(config, startup_stage="ready"),
            )
        except Exception as error:
            recent_vllm_output = self.vllm_output_relay.tail_text() if self.vllm_output_relay is not None else None
            failure_kind = startup_failure_kind(str(error), recent_vllm_output)
            self.publish_startup_status(
                env_values,
                status="failed",
                config=config,
                failure_reason=str(error),
                extra=startup_runtime_detail(
                    config,
                    startup_stage="failed",
                    recent_vllm_output=recent_vllm_output,
                    vllm_process_exit_code=self.vllm_process.returncode if self.vllm_process is not None and self.vllm_process.poll() is not None else None,
                    failure_kind=failure_kind,
                ),
            )
            raise

    def restart_vllm(self) -> None:
        env_values, fallback_note = self.prepared_env_values()
        config = self.config(env_values)
        process_env = self.process_env(env_values)
        self.ensure_dirs()
        if fallback_note:
            self.log(fallback_note)
        validate_gated_model_access(process_env, config.vllm_model)
        self.publish_startup_status(
            env_values,
            status="warming",
            config=config,
            extra=startup_runtime_detail(config, startup_stage="launching_vllm"),
        )

        try:
            with self.lock:
                terminate_process(self.vllm_process)
                self.sync_processes()
                vllm_process = self._start_vllm_locked(config=config, process_env=process_env)

            def _publish_progress(last_ready_error: str, recent_vllm_output: str | None, exit_code: int | None) -> None:
                self.publish_startup_status(
                    env_values,
                    status="warming",
                    config=config,
                    extra=startup_runtime_detail(
                        config,
                        startup_stage="warming_model",
                        last_ready_error=last_ready_error,
                        recent_vllm_output=recent_vllm_output,
                        vllm_process_exit_code=exit_code,
                    ),
                )

            wait_for_inference_runtime_ready(
                config,
                vllm_process,
                output_tail=self.vllm_output_relay.tail_text if self.vllm_output_relay is not None else None,
                fault_state_path=fault_injection_state_path_from_mapping(process_env),
                progress_callback=_publish_progress,
            )
            self.publish_startup_status(
                env_values,
                status="ready",
                config=config,
                extra=startup_runtime_detail(config, startup_stage="ready"),
            )
        except Exception as error:
            recent_vllm_output = self.vllm_output_relay.tail_text() if self.vllm_output_relay is not None else None
            failure_kind = startup_failure_kind(str(error), recent_vllm_output)
            self.publish_startup_status(
                env_values,
                status="failed",
                config=config,
                failure_reason=str(error),
                extra=startup_runtime_detail(
                    config,
                    startup_stage="failed",
                    recent_vllm_output=recent_vllm_output,
                    vllm_process_exit_code=self.vllm_process.returncode if self.vllm_process is not None and self.vllm_process.poll() is not None else None,
                    failure_kind=failure_kind,
                ),
            )
            raise

    def restart_node_agent(self) -> None:
        env_values, fallback_note = self.prepared_env_values()
        config = self.config(env_values)
        process_env = self.process_env(env_values)
        self.ensure_dirs()
        if fallback_note:
            self.log(fallback_note)
        if not config.start_node_agent:
            raise RuntimeError("Serve-only mode disabled the in-container node agent, so it cannot be restarted.")

        with self.lock:
            terminate_process(self.node_process)
            self.sync_processes()
            command = list(config.node_agent_command)
            self.log(f"Starting node agent in this container: {' '.join(shlex.quote(part) for part in command)}")
            self.node_process = subprocess.Popen(command, text=True, env=process_env)
        self.publish_startup_status(
            env_values,
            status="ready",
            config=config,
            extra=startup_runtime_detail(config, startup_stage="ready"),
        )

    def crash_vllm(self) -> None:
        with self.lock:
            self.sync_processes()
            if self.process_running(self.vllm_process):
                self.log("Injecting a live fault by terminating the in-container model server.")
                self.vllm_process.kill()
                self.sync_processes()

    def crash_node_agent(self) -> None:
        with self.lock:
            self.sync_processes()
            if self.process_running(self.node_process):
                self.log("Injecting a live fault by terminating the in-container node agent.")
                self.node_process.kill()
                self.sync_processes()

    def stop_locked(self) -> None:
        terminate_process(self.node_process)
        terminate_process(self.vllm_process)
        self.sync_processes()
        self.node_process = None
        self.vllm_process = None

    def stop(self) -> None:
        with self.lock:
            self.stop_locked()
        if self.startup_status_publisher is not None:
            self.startup_status_publisher.close()
            self.startup_status_publisher = None
            self.startup_status_signature = None

    def wait_for_runtime_health(self, timeout_seconds: float = 90.0) -> None:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        config = self.config()
        required_services = set(expected_runtime_services(config))
        if not required_services:
            raise RuntimeError("Single-container runtime health check has no enabled services to monitor.")
        last_failure = "Runtime services are still starting."
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            running_services = set(snapshot.get("running_services", []))
            if not required_services.issubset(running_services):
                missing = ", ".join(sorted(required_services - running_services))
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
    os.environ.setdefault("CREDENTIALS_PATH", "/var/lib/autonomousc/credentials/node-credentials.json")
    os.environ.setdefault("ATTESTATION_STATE_PATH", "/var/lib/autonomousc/credentials/attestation-state.json")
    os.environ.setdefault("RECOVERY_NOTE_PATH", "/var/lib/autonomousc/credentials/recovery-note.txt")
    os.environ.setdefault("AUTOPILOT_STATE_PATH", "/var/lib/autonomousc/scratch/autopilot-state.json")
    os.environ.setdefault("HEAT_GOVERNOR_STATE_PATH", "/var/lib/autonomousc/scratch/heat-governor-state.json")
    os.environ.setdefault("FAULT_INJECTION_STATE_PATH", "/var/lib/autonomousc/scratch/fault-injection-state.json")
    os.environ.setdefault("STARTUP_STATUS_PATH", "/var/lib/autonomousc/scratch/startup-status.json")
    os.environ.setdefault("STARTUP_STATUS_HOST", DEFAULT_STARTUP_STATUS_HOST)
    os.environ.setdefault("STARTUP_STATUS_PORT", str(DEFAULT_STARTUP_STATUS_PORT))
    os.environ.setdefault("STARTUP_STATUS_ENDPOINT_PATH", DEFAULT_STARTUP_STATUS_ENDPOINT_PATH)
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    initial_config = SingleContainerConfig.from_env()
    fallback_note = apply_single_container_runtime_defaults(
        os.environ,
        local_inference_url=initial_config.local_inference_url,
    )
    config = SingleContainerConfig.from_env()
    if fallback_note:
        print(fallback_note, flush=True)
    validate_gated_model_access(os.environ, config.vllm_model)
    if not config.start_vllm and not config.start_node_agent:
        raise RuntimeError(
            "Single-container runtime has nothing to run. Enable START_VLLM or START_NODE_AGENT."
        )

    vllm_process: subprocess.Popen[str] | None = None
    node_process: subprocess.Popen[str] | None = None
    vllm_output_relay: ProcessOutputRelay | None = None
    vllm_process_env = sanitized_vllm_process_env(os.environ)
    startup_status = StartupStatusPublisher(
        config.startup_status_path,
        host=config.startup_status_host,
        port=config.startup_status_port,
        endpoint_path=config.startup_status_endpoint_path,
        log=lambda message: print(message, flush=True),
    )

    def publish_startup_status(status: str, *, failure_reason: str | None = None, extra: Mapping[str, Any] | None = None) -> None:
        startup_status.publish(
            status=status,
            current_model=config.vllm_model,
            failure_reason=failure_reason,
            warm_source=startup_warm_source_payload(os.environ, current_model=config.vllm_model),
            extra={
                "run_mode": config.run_mode,
                "start_vllm": config.start_vllm,
                "start_node_agent": config.start_node_agent,
                **(dict(extra) if extra is not None else {}),
            },
        )

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
        publish_startup_status("warming", extra=startup_runtime_detail(config, startup_stage="launching_vllm"))
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
                env=vllm_process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            vllm_output_relay.attach(vllm_process)

            def publish_warming_progress(last_ready_error: str, recent_vllm_output: str | None, exit_code: int | None) -> None:
                publish_startup_status(
                    "warming",
                    extra=startup_runtime_detail(
                        config,
                        startup_stage="warming_model",
                        last_ready_error=last_ready_error,
                        recent_vllm_output=recent_vllm_output,
                        vllm_process_exit_code=exit_code,
                    ),
                )

            wait_for_inference_runtime_ready(
                config,
                vllm_process,
                output_tail=vllm_output_relay.tail_text if vllm_output_relay is not None else None,
                progress_callback=publish_warming_progress,
            )
        else:
            print(
                "START_VLLM=false; expecting INFERENCE_BASE_URL (or deprecated VLLM_BASE_URL) to point at an already running inference runtime.",
                flush=True,
            )

        if config.start_node_agent:
            publish_startup_status("warming", extra=startup_runtime_detail(config, startup_stage="starting_node_agent"))
            node_command = list(config.node_agent_command)
            print(f"Starting node agent in this container: {' '.join(shlex.quote(part) for part in node_command)}", flush=True)
            node_process = subprocess.Popen(node_command, text=True)
        else:
            print(
                "Serve-only mode is enabled; skipping in-container node agent startup.",
                flush=True,
            )

        publish_startup_status("ready", extra=startup_runtime_detail(config, startup_stage="ready"))

        running = [process for process in (vllm_process, node_process) if process is not None]
        if not running:
            error = RuntimeError("Single-container runtime did not start any processes.")
            publish_startup_status("failed", failure_reason=str(error))
            raise error
        exited = wait_for_any_process(running)
        exited_service = "vllm" if exited is vllm_process else "node-agent" if exited is node_process else "runtime"
        publish_startup_status(
            "failed",
            failure_reason=f"{exited_service} exited with status {exited.returncode}.",
            extra=startup_runtime_detail(
                config,
                startup_stage="failed",
                recent_vllm_output=vllm_output_relay.tail_text() if vllm_output_relay is not None else None,
                vllm_process_exit_code=vllm_process.returncode if vllm_process is not None and vllm_process.poll() is not None else None,
                failed_service=exited_service,
                failure_kind=startup_failure_kind(
                    f"{exited_service} exited with status {exited.returncode}.",
                    vllm_output_relay.tail_text() if vllm_output_relay is not None else None,
                ),
            ),
        )
        print(f"Process exited with status {exited.returncode}; stopping the single-container runtime.", flush=True)
        return int(exited.returncode or 0)
    except Exception as error:
        recent_vllm_output = vllm_output_relay.tail_text() if vllm_output_relay is not None else None
        publish_startup_status(
            "failed",
            failure_reason=str(error),
            extra=startup_runtime_detail(
                config,
                startup_stage="failed",
                recent_vllm_output=recent_vllm_output,
                vllm_process_exit_code=vllm_process.returncode if vllm_process is not None and vllm_process.poll() is not None else None,
                failure_kind=startup_failure_kind(str(error), recent_vllm_output),
            ),
        )
        raise
    finally:
        terminate_process(node_process)
        terminate_process(vllm_process)
        startup_status.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
