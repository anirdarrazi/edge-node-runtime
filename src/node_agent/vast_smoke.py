from __future__ import annotations

import argparse
import concurrent.futures
import math
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from .runtime_profiles import (
    VAST_VLLM_SAFETENSORS_PROFILE,
    default_vast_launch_profile,
    runtime_profile_by_id,
)
from .runtime_layout import appliance_runtime_root
from .single_container import (
    DEFAULT_STARTUP_STATUS_ENDPOINT_PATH,
    DEFAULT_STARTUP_STATUS_HOST,
    DEFAULT_STARTUP_STATUS_PORT,
    known_safe_max_model_len,
    normalized_model_lookup_key,
    normalized_max_context_tokens,
)


DEFAULT_VAST_API_BASE_URL = "https://console.vast.ai/api/v0"
DEFAULT_VAST_SMOKE_IMAGE = "anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest"
DEFAULT_VAST_SMOKE_LABEL = "autonomousc-runtime-smoke-test"
DEFAULT_VAST_RUNTIME_PROFILE = runtime_profile_by_id(VAST_VLLM_SAFETENSORS_PROFILE)
if DEFAULT_VAST_RUNTIME_PROFILE is None:  # pragma: no cover - guarded by static profile definitions
    raise RuntimeError("Vast.ai runtime profile is not available.")
DEFAULT_VAST_LAUNCH_PROFILE = default_vast_launch_profile()
DEFAULT_VAST_SMOKE_MODEL = DEFAULT_VAST_RUNTIME_PROFILE.smoke_test_model
DEFAULT_VAST_SMOKE_API_PATH = DEFAULT_VAST_RUNTIME_PROFILE.smoke_test_api_path
DEFAULT_OFFER_LIMIT = 20
DEFAULT_MIN_VRAM_GB = 16.0
DEFAULT_DISK_GB = DEFAULT_VAST_LAUNCH_PROFILE.min_disk_gb
DEFAULT_MIN_CUDA_MAX_GOOD = 12.9
DEFAULT_MIN_RELIABILITY = 0.98
DEFAULT_MIN_INET_DOWN_MBPS = 250.0
DEFAULT_LAUNCH_TIMEOUT_SECONDS = 240.0
DEFAULT_LAUNCH_PROGRESS_GRACE_SECONDS = 240.0
DEFAULT_READINESS_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_PROBE_RETRY_ATTEMPTS = 3
DEFAULT_PROBE_RETRY_DELAY_SECONDS = 2.0
DEFAULT_POST_PROBE_STATUS_GRACE_SECONDS = 20.0
DEFAULT_POST_PROBE_STATUS_POLL_SECONDS = 2.0
DEFAULT_VAST_API_RETRY_ATTEMPTS = 5
DEFAULT_VAST_API_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_PRICING_PLATFORM_OVERHEAD_PCT = 0.10
DEFAULT_PRICING_IDLE_RESERVE_PCT = 0.15
DEFAULT_PRICING_TARGET_MARGIN_PCT = 0.35
DEFAULT_PRICING_WARMUP_REFERENCE_WINDOW_SECONDS = 3600.0
DEFAULT_MODELS_PATH = DEFAULT_VAST_RUNTIME_PROFILE.readiness_path
DEFAULT_EMBEDDINGS_PATH = "/v1/embeddings"
DEFAULT_RESPONSES_PATH = "/v1/responses"
DEFAULT_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_EMBEDDINGS_INPUT = ["hello from autonomousc smoke test"]
DEFAULT_RESPONSES_INPUT = "Reply with the single word ready."
DEFAULT_BENCHMARK_REQUESTS = 0
DEFAULT_BENCHMARK_CONCURRENCY = 1
DEFAULT_BENCHMARK_PROFILE = "balanced"
DEFAULT_BENCHMARK_RESPONSES_INPUT = (
    "Write exactly 180 plain-text words about keeping GPU inference cheap on rented nodes. "
    "Do not use bullet points or markdown. Do not include a preamble or conclusion."
)
DEFAULT_BENCHMARK_MAX_OUTPUT_TOKENS = 256
DEFAULT_VAST_SMOKE_MODEL_MAX_CONTEXT_TOKENS = {
    "qwen/qwen2.5-1.5b-instruct": 16384,
    "meta-llama/llama-3.1-8b-instruct": 8192,
    "google/gemma-4-e4b-it": 32768,
    "google/gemma-4-26b-a4b-it": 8192,
}
DEFAULT_VAST_SMOKE_MODEL_MIN_VRAM_GB = {
    "meta-llama/llama-3.1-8b-instruct": 24.0,
    "google/gemma-4-e4b-it": 16.0,
    "google/gemma-4-26b-a4b-it": 70.0,
}
DEFAULT_VAST_SMOKE_MODEL_MIN_INET_DOWN_MBPS = {
    "qwen/qwen2.5-1.5b-instruct": 500.0,
    "meta-llama/llama-3.1-8b-instruct": 600.0,
    "google/gemma-4-e4b-it": 600.0,
    "google/gemma-4-26b-a4b-it": 700.0,
}
DEFAULT_VAST_SMOKE_MODEL_PREFERRED_VRAM_GB = {
    "baai/bge-large-en-v1.5": 24.0,
    "qwen/qwen2.5-1.5b-instruct": 24.0,
    "meta-llama/llama-3.1-8b-instruct": 24.0,
    "google/gemma-4-e4b-it": 16.0,
    "google/gemma-4-26b-a4b-it": 70.0,
}
VAST_SMOKE_CONFIG_ENV = "NODE_AGENT_VAST_SMOKE_CONFIG"
DEFAULT_VAST_SMOKE_CONFIG_NAME = "vast-smoke.json"


class VastSmokeError(RuntimeError):
    pass


class VastInstanceLaunchError(VastSmokeError):
    def __init__(self, message: str, *, retryable: bool = False, offer_id: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.offer_id = offer_id


@dataclass(frozen=True)
class VastSmokeConfig:
    api_key: str
    model: str = DEFAULT_VAST_SMOKE_MODEL
    max_price: float = DEFAULT_VAST_LAUNCH_PROFILE.safe_price_ceiling_usd
    image: str = DEFAULT_VAST_SMOKE_IMAGE
    label: str = DEFAULT_VAST_SMOKE_LABEL
    runtype: str = DEFAULT_VAST_LAUNCH_PROFILE.runtype
    disk_gb: int = DEFAULT_DISK_GB
    min_vram_gb: float = DEFAULT_MIN_VRAM_GB
    min_cuda_max_good: float | None = DEFAULT_MIN_CUDA_MAX_GOOD
    min_reliability: float = DEFAULT_MIN_RELIABILITY
    min_inet_down_mbps: float = DEFAULT_MIN_INET_DOWN_MBPS
    offer_limit: int = DEFAULT_OFFER_LIMIT
    launch_timeout_seconds: float = DEFAULT_LAUNCH_TIMEOUT_SECONDS
    readiness_timeout_seconds: float = DEFAULT_READINESS_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    api_kind: str = "auto"
    smoke_test_api_path: str = DEFAULT_VAST_SMOKE_API_PATH
    max_context_tokens: int = 32768
    hf_token: str | None = None
    benchmark_requests: int = DEFAULT_BENCHMARK_REQUESTS
    benchmark_concurrency: int = DEFAULT_BENCHMARK_CONCURRENCY
    benchmark_profile: str = DEFAULT_BENCHMARK_PROFILE
    vllm_extra_args: str = ""

    def __post_init__(self) -> None:
        normalized_api_kind = str(self.api_kind or "auto").strip().lower() or "auto"
        object.__setattr__(self, "api_kind", normalized_api_kind)
        expected_path = expected_api_path_for(normalized_api_kind, self.model)
        current_path = str(self.smoke_test_api_path or "").strip()
        if not current_path or (
            current_path == DEFAULT_VAST_SMOKE_API_PATH and expected_path != DEFAULT_VAST_SMOKE_API_PATH
        ):
            object.__setattr__(self, "smoke_test_api_path", expected_path)
        if int(self.max_context_tokens) == 32768:
            recommended = recommended_vast_smoke_max_context_tokens(self.model)
            if recommended != 32768:
                object.__setattr__(self, "max_context_tokens", recommended)
        if float(self.min_vram_gb) == DEFAULT_MIN_VRAM_GB:
            recommended_vram = recommended_vast_smoke_min_vram_gb(self.model)
            if recommended_vram != DEFAULT_MIN_VRAM_GB:
                object.__setattr__(self, "min_vram_gb", recommended_vram)
        if float(self.min_inet_down_mbps) == DEFAULT_MIN_INET_DOWN_MBPS:
            recommended_inet = recommended_vast_smoke_min_inet_down_mbps(self.model)
            if recommended_inet != DEFAULT_MIN_INET_DOWN_MBPS:
                object.__setattr__(self, "min_inet_down_mbps", recommended_inet)
        normalized_benchmark_profile = str(self.benchmark_profile or DEFAULT_BENCHMARK_PROFILE).strip().lower()
        if normalized_benchmark_profile not in {"balanced", "input_heavy", "output_heavy"}:
            normalized_benchmark_profile = DEFAULT_BENCHMARK_PROFILE
        object.__setattr__(self, "benchmark_profile", normalized_benchmark_profile)

    @property
    def effective_max_context_tokens(self) -> int:
        return normalized_max_context_tokens(self.model, self.max_context_tokens)


def default_vast_smoke_config_path() -> Path:
    return appliance_runtime_root() / "local-secrets" / DEFAULT_VAST_SMOKE_CONFIG_NAME


def _config_value(values: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = str(values.get(key) or "").strip()
        if value:
            return value
    return None


def first_nonempty(*values: str | None) -> str | None:
    for value in values:
        candidate = str(value or "").strip()
        if candidate:
            return candidate
    return None


def recommended_vast_smoke_max_context_tokens(model: str | None) -> int:
    safe_limit = known_safe_max_model_len(model)
    if safe_limit is not None:
        return safe_limit
    normalized_model = normalized_model_lookup_key(model)
    configured = DEFAULT_VAST_SMOKE_MODEL_MAX_CONTEXT_TOKENS.get(normalized_model)
    return int(configured) if isinstance(configured, int) and configured > 0 else 32768


def recommended_vast_smoke_min_vram_gb(model: str | None) -> float:
    normalized_model = normalized_model_lookup_key(model)
    configured = DEFAULT_VAST_SMOKE_MODEL_MIN_VRAM_GB.get(normalized_model)
    return float(configured) if isinstance(configured, (int, float)) and configured > 0 else DEFAULT_MIN_VRAM_GB


def recommended_vast_smoke_min_inet_down_mbps(model: str | None) -> float:
    normalized_model = normalized_model_lookup_key(model)
    configured = DEFAULT_VAST_SMOKE_MODEL_MIN_INET_DOWN_MBPS.get(normalized_model)
    return float(configured) if isinstance(configured, (int, float)) and configured > 0 else DEFAULT_MIN_INET_DOWN_MBPS


def preferred_vast_smoke_vram_gb(model: str | None) -> float:
    normalized_model = normalized_model_lookup_key(model)
    configured = DEFAULT_VAST_SMOKE_MODEL_PREFERRED_VRAM_GB.get(normalized_model)
    if isinstance(configured, (int, float)) and configured > 0:
        return float(configured)
    return recommended_vast_smoke_min_vram_gb(model)


def benchmark_embeddings_input_for_model(model: str | None) -> list[str]:
    safe_context_tokens = max(48, recommended_vast_smoke_max_context_tokens(model))
    target_words = max(48, min(256, safe_context_tokens // 6))
    return [
        " ".join(
            "edge heat compute cache queue token"
            for _ in range(target_words // 6)
        )
    ]


def benchmark_profile_name(value: str | None) -> str:
    normalized = str(value or DEFAULT_BENCHMARK_PROFILE).strip().lower()
    if normalized in {"input_heavy", "output_heavy"}:
        return normalized
    return DEFAULT_BENCHMARK_PROFILE


def benchmark_responses_input(profile: str, *, model: str | None = None) -> str:
    normalized = benchmark_profile_name(profile)
    if normalized == "input_heavy":
        safe_context_tokens = max(2048, recommended_vast_smoke_max_context_tokens(model))
        target_groups = max(96, min(320, safe_context_tokens // 24))
        repeated_context = " ".join(
            "prefix cache warm token economics queue latency throughput"
            for _ in range(target_groups)
        )
        return (
            "Read this context and answer with the single word ready. "
            f"{repeated_context}"
        )
    if normalized == "output_heavy":
        return (
            "Write exactly 320 plain-text words about keeping AI inference economically competitive on rented GPUs. "
            "Do not use bullet points or markdown. Do not include a preamble or conclusion."
        )
    return DEFAULT_BENCHMARK_RESPONSES_INPUT


def benchmark_max_output_tokens(profile: str) -> int:
    normalized = benchmark_profile_name(profile)
    if normalized == "input_heavy":
        return 16
    if normalized == "output_heavy":
        return 384
    return DEFAULT_BENCHMARK_MAX_OUTPUT_TOKENS


def expected_api_path_for(api_kind: str, model: str) -> str:
    normalized_kind = str(api_kind or "auto").strip().lower()
    if normalized_kind == "embeddings":
        return DEFAULT_EMBEDDINGS_PATH
    if normalized_kind == "responses":
        return DEFAULT_RESPONSES_PATH
    if normalized_kind == "chat_completions":
        return DEFAULT_CHAT_COMPLETIONS_PATH
    preferred = preferred_api_for_model(model)
    if preferred == "embeddings":
        return DEFAULT_EMBEDDINGS_PATH
    return DEFAULT_RESPONSES_PATH


def resolve_vast_smoke_config_path(explicit: str | None = None) -> tuple[Path | None, bool]:
    candidate = str(explicit or "").strip()
    if candidate:
        return Path(candidate).expanduser(), True
    configured = str(os.getenv(VAST_SMOKE_CONFIG_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser(), True
    default_path = default_vast_smoke_config_path()
    if default_path.exists():
        return default_path, False
    return None, False


def load_vast_smoke_config(explicit: str | None = None) -> tuple[dict[str, Any], Path | None]:
    config_path, required = resolve_vast_smoke_config_path(explicit)
    if config_path is None:
        return {}, None
    if not config_path.exists():
        if required:
            raise VastSmokeError(
                f"Vast smoke config file {config_path} does not exist."
            )
        return {}, config_path
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise VastSmokeError(f"Could not read Vast smoke config file {config_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise VastSmokeError(f"Vast smoke config file {config_path} is not valid JSON.") from error
    if not isinstance(payload, dict):
        raise VastSmokeError(f"Vast smoke config file {config_path} must contain a JSON object.")
    return payload, config_path


class VastAPI:
    def __init__(self, api_key: str, *, base_url: str = DEFAULT_VAST_API_BASE_URL, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request_with_retries(
        self,
        method: str,
        path: str,
        *,
        attempts: int = DEFAULT_VAST_API_RETRY_ATTEMPTS,
        retry_status_codes: frozenset[int] = DEFAULT_VAST_API_RETRY_STATUS_CODES,
        **kwargs: Any,
    ) -> Any:
        caller = getattr(self._client, method.lower(), None)
        if caller is None:
            raise VastSmokeError(f"Configured Vast client does not support HTTP {method.upper()} calls.")
        last_response: Any = None
        for attempt in range(1, max(1, int(attempts)) + 1):
            response = caller(path, **kwargs)
            last_response = response
            status_code = getattr(response, "status_code", None)
            if status_code not in retry_status_codes or attempt >= max(1, int(attempts)):
                return response
            time.sleep(float(attempt))
        return last_response

    def search_offers(self, config: VastSmokeConfig) -> list[dict[str, Any]]:
        payload = {
            "limit": max(1, int(config.offer_limit)),
            "type": "ondemand",
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "gpu_frac": {"eq": 1},
            "gpu_ram": {"gte": int(config.min_vram_gb * 1024)},
            "disk_space": {"gte": int(config.disk_gb)},
            "reliability": {"gte": float(config.min_reliability)},
            "inet_down": {"gte": float(config.min_inet_down_mbps)},
        }
        response = self._request_with_retries("post", "/bundles/", json=payload)
        response.raise_for_status()
        body = response.json()
        offers = body.get("offers")
        if not isinstance(offers, list):
            raise VastSmokeError("Vast offer search returned an unexpected payload.")
        return [offer for offer in offers if isinstance(offer, dict)]

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        env: dict[str, str],
        disk_gb: int,
        label: str,
        runtype: str,
    ) -> int:
        payload = {
            "image": image,
            "label": label,
            "disk": int(disk_gb),
            "runtype": runtype,
            "args": [],
            "target_state": "running",
            "cancel_unavail": True,
            "env": env,
        }
        response = self._request_with_retries("put", f"/asks/{offer_id}/", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            body = response_json_object(error.response)
            message = format_vast_launch_error(
                offer_id,
                body=body,
                status_code=getattr(error.response, "status_code", None),
                fallback_text=getattr(error.response, "text", None),
            )
            raise VastInstanceLaunchError(
                message,
                retryable=is_retryable_offer_error(body),
                offer_id=offer_id,
            ) from error
        body = response_json_object(response)
        if body is None:
            raise VastInstanceLaunchError(
                f"Vast launch response for ask {offer_id} was not a JSON object.",
                offer_id=offer_id,
            )
        if not body.get("success"):
            raise VastInstanceLaunchError(
                format_vast_launch_error(offer_id, body=body),
                retryable=is_retryable_offer_error(body),
                offer_id=offer_id,
            )
        instance_id = body.get("new_contract")
        if not isinstance(instance_id, int):
            raise VastSmokeError("Vast instance launch did not return a new contract id.")
        return instance_id

    def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        response = self._request_with_retries("get", f"/instances/{instance_id}/")
        response.raise_for_status()
        body = response.json()
        instance = body.get("instances")
        return instance if isinstance(instance, dict) else None

    def destroy_instance(self, instance_id: int) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            response = self._client.delete(f"/instances/{instance_id}/")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                last_error = error
                status_code = getattr(error.response, "status_code", None)
                if status_code in {429, 500, 502, 503, 504} and attempt < 5:
                    time.sleep(float(attempt))
                    continue
                raise
            body = response.json()
            if body.get("success"):
                return
            message = str(body.get("msg") or body.get("error") or "Vast instance destroy failed.")
            last_error = VastSmokeError(message)
            break
        if isinstance(last_error, Exception):
            raise last_error
        raise VastSmokeError("Vast instance destroy failed.")


class RuntimeProbeClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get(self, url: str) -> httpx.Response:
        return self._client.get(url)

    def post(self, url: str, *, json_body: dict[str, Any]) -> httpx.Response:
        return self._client.post(url, json=json_body)


def _float_value(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key) or default)
    except (TypeError, ValueError):
        return default


def _int_value(payload: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(payload.get(key) or default))
    except (TypeError, ValueError):
        return default


def response_json_object(response: Any) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except (AttributeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def is_retryable_offer_error(body: dict[str, Any] | None) -> bool:
    if not isinstance(body, dict):
        return False
    normalized_error = str(body.get("error") or "").strip().lower()
    normalized_msg = str(body.get("msg") or "").strip().lower()
    return any(
        token in normalized_error or token in normalized_msg
        for token in (
            "no_such_ask",
            "ask is not available",
            "already rented",
            "already taken",
            "unavailable",
        )
    )


def format_vast_launch_error(
    offer_id: int,
    *,
    body: dict[str, Any] | None = None,
    status_code: int | None = None,
    fallback_text: str | None = None,
) -> str:
    detail = first_nonempty(
        str(body.get("msg") or "") if isinstance(body, dict) else "",
        str(body.get("error") or "") if isinstance(body, dict) else "",
        str(fallback_text or ""),
    )
    detail = detail or "Vast instance launch failed."
    if isinstance(body, dict) and body.get("ask_id") not in {None, ""}:
        detail = f"{detail} (ask_id={body.get('ask_id')})"
    if status_code is not None:
        return f"Vast rejected ask {offer_id} with HTTP {status_code}: {detail}"
    return f"Vast rejected ask {offer_id}: {detail}"


def summarize_offer(offer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": offer.get("id"),
        "gpu_name": offer.get("gpu_name"),
        "gpu_ram_gb": round(_float_value(offer, "gpu_ram") / 1024.0, 1),
        "dph_total": round(_float_value(offer, "dph_total"), 6),
        "reliability": round(_float_value(offer, "reliability") or _float_value(offer, "reliability2"), 6),
        "inet_down": round(_float_value(offer, "inet_down"), 1),
        "disk_space": round(_float_value(offer, "disk_space"), 1),
        "cuda_max_good": offer.get("cuda_max_good"),
        "geolocation": offer.get("geolocation"),
        "verification": offer.get("verification") or ("verified" if offer.get("verified") else None),
    }


def offer_fit_tier(offer: dict[str, Any], *, model: str | None = None) -> int:
    gpu_ram_gb = _float_value(offer, "gpu_ram") / 1024.0
    preferred_vram_gb = preferred_vast_smoke_vram_gb(model)
    preferred_inet_down_mbps = recommended_vast_smoke_min_inet_down_mbps(model)
    meets_preferred_vram = gpu_ram_gb >= preferred_vram_gb
    meets_preferred_inet = _float_value(offer, "inet_down") >= preferred_inet_down_mbps
    if meets_preferred_vram and meets_preferred_inet:
        return 0
    if meets_preferred_vram:
        return 1
    if meets_preferred_inet:
        return 2
    return 3


def offer_readiness_sort_key(offer: dict[str, Any], *, model: str | None = None) -> tuple[int, float, float, float, float, float, int]:
    gpu_ram_gb = _float_value(offer, "gpu_ram") / 1024.0
    preferred_vram_gb = preferred_vast_smoke_vram_gb(model)
    preferred_inet_down_mbps = recommended_vast_smoke_min_inet_down_mbps(model)
    return (
        offer_fit_tier(offer, model=model),
        _float_value(offer, "dph_total", default=10**9),
        -(_float_value(offer, "reliability") or _float_value(offer, "reliability2")),
        -min(_float_value(offer, "inet_down"), preferred_inet_down_mbps),
        -min(gpu_ram_gb, preferred_vram_gb),
        -_float_value(offer, "disk_space"),
        _int_value(offer, "id", default=10**9),
    )


def offer_supports_minimum_cuda(offer: dict[str, Any], minimum_cuda_max_good: float | None) -> bool:
    if minimum_cuda_max_good is None or minimum_cuda_max_good <= 0:
        return True
    return _float_value(offer, "cuda_max_good") >= float(minimum_cuda_max_good)


def choose_cheapest_offer(offers: list[dict[str, Any]], *, max_price: float, model: str | None = None) -> dict[str, Any]:
    return affordable_offers(offers, max_price=max_price, min_cuda_max_good=None, model=model)[0]


def affordable_offers(
    offers: list[dict[str, Any]],
    *,
    max_price: float,
    min_cuda_max_good: float | None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    affordable = [
        offer for offer in offers if _float_value(offer, "dph_total", default=10**9) <= float(max_price)
    ]
    if not affordable:
        raise VastSmokeError(
            f"No suitable Vast offers were available at or below ${max_price:.2f}/hr."
        )
    supported = [
        offer for offer in affordable if offer_supports_minimum_cuda(offer, min_cuda_max_good)
    ]
    if not supported:
        raise VastSmokeError(
            f"No suitable Vast offers were available at or below ${max_price:.2f}/hr "
            f"after requiring cuda_max_good >= {float(min_cuda_max_good):.1f}."
        )
    return sorted(
        supported,
        key=lambda offer: offer_readiness_sort_key(offer, model=model),
    )


def build_launch_env(config: VastSmokeConfig) -> dict[str, str]:
    env = {
        "-p 8000:8000": "1",
        f"-p {DEFAULT_STARTUP_STATUS_PORT}:{DEFAULT_STARTUP_STATUS_PORT}": "1",
        "RUN_MODE": "serve_only",
        "START_NODE_AGENT": "false",
        "VLLM_MODEL": config.model,
        "SUPPORTED_MODELS": config.model,
        "MAX_CONTEXT_TOKENS": str(config.effective_max_context_tokens),
        "STARTUP_STATUS_HOST": DEFAULT_STARTUP_STATUS_HOST,
        "STARTUP_STATUS_PORT": str(DEFAULT_STARTUP_STATUS_PORT),
        "STARTUP_STATUS_ENDPOINT_PATH": DEFAULT_STARTUP_STATUS_ENDPOINT_PATH,
    }
    if config.hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = config.hf_token
    if str(config.vllm_extra_args or "").strip():
        env["VLLM_EXTRA_ARGS"] = str(config.vllm_extra_args).strip()
    return env


def extract_host_port(instance: dict[str, Any], *, container_port: int = 8000) -> int:
    ports = instance.get("ports")
    if not isinstance(ports, dict):
        raise VastSmokeError("Vast instance did not report port mappings yet.")
    bindings = ports.get(f"{container_port}/tcp")
    if not isinstance(bindings, list) or not bindings:
        raise VastSmokeError(f"Vast instance did not expose port {container_port}/tcp yet.")
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        host_port = binding.get("HostPort")
        try:
            return int(str(host_port))
        except (TypeError, ValueError):
            continue
    raise VastSmokeError(f"Vast instance reported port {container_port}/tcp without a usable host port.")


def first_served_model(models_payload: dict[str, Any]) -> str | None:
    data = models_payload.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    return None


def startup_status_summary(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("status") or "").strip().lower()
    if not status:
        return None
    model = str(payload.get("current_model") or "").strip()
    failure_reason = str(payload.get("failure_reason") or "").strip()
    summary = f"{DEFAULT_STARTUP_STATUS_ENDPOINT_PATH} reports {status}"
    if model:
        summary = f"{summary} for {model}"
    if failure_reason:
        summary = f"{summary}: {failure_reason}"
    return summary


def startup_status_state(runtime_report: Mapping[str, Any] | None) -> str:
    payload = runtime_report.get("startup_status") if isinstance(runtime_report, Mapping) else None
    if not isinstance(payload, Mapping):
        return ""
    return str(payload.get("status") or "").strip().lower()


def should_allow_launch_grace(status_text: str | None) -> bool:
    normalized = str(status_text or "").strip().lower()
    if not normalized:
        return False
    return any(
        token in normalized
        for token in (
            "pull",
            "pulling",
            "download",
            "extract",
            "unpack",
            "layer",
            "fetch",
            "image",
            "verifying",
            "checksum",
            "loaded",
            "loading",
            "starting",
            "running",
            "success",
            "public ip",
            "port",
        )
    )


def should_retry_candidate_after_error(error: Exception, runtime_report: Mapping[str, Any] | None) -> bool:
    if startup_status_state(runtime_report) == "failed":
        payload = runtime_report.get("startup_status") if isinstance(runtime_report, Mapping) else None
        failure_reason = str(payload.get("failure_reason") or "").strip().lower() if isinstance(payload, Mapping) else ""
        if any(
            token in failure_reason
            for token in (
                "engine core initialization failed",
                "failed core proc",
                "cuda out of memory",
                "out of memory",
                "insufficient memory",
                "not enough memory",
                "engine process failed to start",
            )
        ):
            return True
        return False
    if isinstance(error, VastInstanceLaunchError):
        return bool(error.retryable)
    if isinstance(error, httpx.HTTPError):
        return True
    message = str(error or "").strip().lower()
    return any(
        token in message
        for token in (
            "did not become ready in time",
            "connection refused",
            "timed out",
            "temporarily unavailable",
            "transport error",
            "server disconnected",
            "public ip",
            "port ",
            "still starting",
            "runtime is unavailable",
            "openai-compatible runtime did not become ready",
        )
    )


def preferred_api_for_model(model: str | None) -> str:
    normalized = str(model or "").strip()
    if normalized == DEFAULT_VAST_SMOKE_MODEL:
        return "embeddings"
    return "responses"


def preferred_api_for_path(path: str | None) -> str | None:
    normalized = str(path or "").strip()
    if normalized == DEFAULT_EMBEDDINGS_PATH:
        return "embeddings"
    if normalized == DEFAULT_RESPONSES_PATH:
        return "responses"
    if normalized == DEFAULT_CHAT_COMPLETIONS_PATH:
        return "chat_completions"
    return None


def api_kind_for_probe_result(probe: Mapping[str, Any] | None) -> str:
    if not isinstance(probe, Mapping):
        return "auto"
    probe_api = str(probe.get("api") or "").strip().lower()
    if probe_api == "embeddings":
        return "embeddings"
    if probe_api == "responses":
        return "responses"
    if probe_api == "chat_completions_fallback":
        return "chat_completions"
    return "auto"


def normalized_usage_from_probe(probe: Mapping[str, Any] | None) -> dict[str, int]:
    usage = probe.get("usage") if isinstance(probe, Mapping) else None
    if not isinstance(usage, Mapping):
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    input_token_details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), Mapping) else {}
    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or 0
    )
    cached_input_tokens = int(input_token_details.get("cached_tokens") or 0)
    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return {
        "input_tokens": max(0, input_tokens),
        "cached_input_tokens": max(0, min(input_tokens, cached_input_tokens)),
        "uncached_input_tokens": max(0, input_tokens - max(0, min(input_tokens, cached_input_tokens))),
        "output_tokens": max(0, output_tokens),
        "total_tokens": max(0, total_tokens),
    }


def round_metric(value: float, *, digits: int = 6) -> float:
    return round(float(value), digits)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, pct / 100.0)) * (len(ordered) - 1)
    low_index = int(math.floor(rank))
    high_index = int(math.ceil(rank))
    if low_index == high_index:
        return ordered[low_index]
    low_value = ordered[low_index]
    high_value = ordered[high_index]
    fraction = rank - low_index
    return low_value + (high_value - low_value) * fraction


def economics_for_tokens(
    *,
    hourly_cost_usd: float,
    elapsed_seconds: float,
    input_tokens: int,
    cached_input_tokens: int,
    uncached_input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> dict[str, Any]:
    safe_elapsed_seconds = max(float(elapsed_seconds), 1e-9)
    input_tokens_per_second = float(input_tokens) / safe_elapsed_seconds
    cached_input_tokens_per_second = float(cached_input_tokens) / safe_elapsed_seconds
    uncached_input_tokens_per_second = float(uncached_input_tokens) / safe_elapsed_seconds
    output_tokens_per_second = float(output_tokens) / safe_elapsed_seconds
    total_tokens_per_second = float(total_tokens) / safe_elapsed_seconds

    def usd_per_million(tokens_per_second: float) -> float | None:
        if tokens_per_second <= 0:
            return None
        return round_metric((float(hourly_cost_usd) * 1_000_000.0) / (tokens_per_second * 3600.0))

    return {
        "hourly_cost_usd": round_metric(hourly_cost_usd),
        "input_tokens_per_second": round_metric(input_tokens_per_second),
        "cached_input_tokens_per_second": round_metric(cached_input_tokens_per_second),
        "uncached_input_tokens_per_second": round_metric(uncached_input_tokens_per_second),
        "output_tokens_per_second": round_metric(output_tokens_per_second),
        "total_tokens_per_second": round_metric(total_tokens_per_second),
        "usd_per_million_input_tokens": usd_per_million(input_tokens_per_second),
        "usd_per_million_cached_input_tokens": usd_per_million(cached_input_tokens_per_second),
        "usd_per_million_uncached_input_tokens": usd_per_million(uncached_input_tokens_per_second),
        "usd_per_million_output_tokens": usd_per_million(output_tokens_per_second),
        "usd_per_million_total_tokens": usd_per_million(total_tokens_per_second),
    }


def pricing_guidance_for_economics(
    economics: Mapping[str, Any],
    *,
    warmup_seconds: float = 0.0,
    platform_overhead_pct: float = DEFAULT_PRICING_PLATFORM_OVERHEAD_PCT,
    idle_reserve_pct: float = DEFAULT_PRICING_IDLE_RESERVE_PCT,
    target_margin_pct: float = DEFAULT_PRICING_TARGET_MARGIN_PCT,
    warmup_reference_window_seconds: float = DEFAULT_PRICING_WARMUP_REFERENCE_WINDOW_SECONDS,
) -> dict[str, Any]:
    normalized_warmup_seconds = max(0.0, float(warmup_seconds or 0.0))
    normalized_reference_window = max(1.0, float(warmup_reference_window_seconds))
    warmup_overhead_pct = round_metric(min(1.0, normalized_warmup_seconds / normalized_reference_window))
    conversion_denominator = max(
        0.05,
        1.0 - float(platform_overhead_pct) - float(idle_reserve_pct) - float(target_margin_pct),
    )

    def effective_floor(key: str) -> float | None:
        base_value = economics.get(key)
        if base_value in {None, ""}:
            return None
        return round_metric(float(base_value) * (1.0 + warmup_overhead_pct))

    def recommended_price(key: str) -> float | None:
        adjusted_floor = effective_floor(key)
        if adjusted_floor is None:
            return None
        return round_metric(adjusted_floor / conversion_denominator)

    return {
        "platform_overhead_pct": round_metric(platform_overhead_pct),
        "idle_reserve_pct": round_metric(idle_reserve_pct),
        "target_margin_pct": round_metric(target_margin_pct),
        "warmup_seconds": round_metric(normalized_warmup_seconds),
        "warmup_reference_window_seconds": round_metric(normalized_reference_window),
        "warmup_overhead_pct": warmup_overhead_pct,
        "effective_floor_usd_per_million_input_tokens": effective_floor("usd_per_million_input_tokens"),
        "effective_floor_usd_per_million_output_tokens": effective_floor("usd_per_million_output_tokens"),
        "effective_floor_usd_per_million_total_tokens": effective_floor("usd_per_million_total_tokens"),
        "recommended_price_usd_per_million_input_tokens": recommended_price("usd_per_million_input_tokens"),
        "recommended_price_usd_per_million_output_tokens": recommended_price("usd_per_million_output_tokens"),
        "recommended_price_usd_per_million_total_tokens": recommended_price("usd_per_million_total_tokens"),
    }


def extract_response_text(payload: dict[str, Any]) -> str | None:
    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
        if parts:
            return " ".join(parts)
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    return None


class VastSmokeRunner:
    def __init__(
        self,
        api: VastAPI,
        runtime: RuntimeProbeClient,
        *,
        monotonic: callable = time.monotonic,
        sleep: callable = time.sleep,
    ) -> None:
        self.api = api
        self.runtime = runtime
        self.monotonic = monotonic
        self.sleep = sleep

    def wait_for_instance(
        self,
        instance_id: int,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
        required_container_ports: tuple[int, ...] = (8000, DEFAULT_STARTUP_STATUS_PORT),
    ) -> dict[str, Any]:
        start_time = self.monotonic()
        deadline = start_time + max(1.0, timeout_seconds)
        hard_deadline = deadline + DEFAULT_LAUNCH_PROGRESS_GRACE_SECONDS
        last_status = "Vast instance is still starting."
        while self.monotonic() < hard_deadline:
            instance = self.api.get_instance(instance_id)
            if not isinstance(instance, dict):
                last_status = "Vast instance details are not available yet."
                if self.monotonic() >= deadline and not should_allow_launch_grace(last_status):
                    break
                self.sleep(poll_interval_seconds)
                continue
            actual_status = str(instance.get("actual_status") or "").strip().lower()
            cur_state = str(instance.get("cur_state") or "").strip().lower()
            status_msg = str(instance.get("status_msg") or "").strip()
            if actual_status == "running":
                try:
                    for container_port in required_container_ports:
                        extract_host_port(instance, container_port=container_port)
                    public_ip = str(instance.get("public_ipaddr") or "").strip()
                    if public_ip:
                        return instance
                    last_status = "Vast instance is running but does not have a public IP address yet."
                except VastSmokeError as error:
                    last_status = str(error)
            elif actual_status in {"dead", "exited", "offline"} or cur_state in {"stopped", "deleted"}:
                raise VastSmokeError(status_msg or f"Vast instance entered terminal state {actual_status or cur_state}.")
            elif status_msg:
                last_status = status_msg
            if self.monotonic() >= deadline and not should_allow_launch_grace(last_status):
                break
            self.sleep(poll_interval_seconds)
        raise VastSmokeError(f"Vast instance did not become ready in time: {last_status}")

    def wait_for_runtime_ready(
        self,
        runtime_report: dict[str, Any],
        base_url: str,
        *,
        startup_status_url: str | None,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> tuple[dict[str, Any], int]:
        deadline = self.monotonic() + max(1.0, timeout_seconds)
        last_error = "The OpenAI-compatible runtime is still starting."
        models_url = f"{base_url}{DEFAULT_MODELS_PATH}"
        runtime_report["models_path"] = DEFAULT_MODELS_PATH
        runtime_report["startup_status_path"] = DEFAULT_STARTUP_STATUS_ENDPOINT_PATH
        runtime_report["startup_status_url"] = startup_status_url
        while self.monotonic() < deadline:
            startup_summary = None
            if startup_status_url:
                try:
                    startup_response = self.runtime.get(startup_status_url)
                    runtime_report["startup_status_code"] = int(startup_response.status_code)
                    startup_payload = response_json_object(startup_response)
                    if startup_payload is not None:
                        runtime_report["startup_status"] = startup_payload
                        startup_summary = startup_status_summary(startup_payload)
                        if startup_summary:
                            last_error = startup_summary
                        if str(startup_payload.get("status") or "").strip().lower() == "failed":
                            failure_reason = str(startup_payload.get("failure_reason") or "").strip()
                            reason_text = failure_reason or "no failure_reason was provided."
                            raise VastSmokeError(
                                f"Runtime startup failed before {DEFAULT_MODELS_PATH} became ready: {reason_text}"
                            )
                    elif startup_response.status_code >= 500:
                        last_error = f"{DEFAULT_STARTUP_STATUS_ENDPOINT_PATH} returned HTTP {startup_response.status_code}."
                    else:
                        last_error = f"{DEFAULT_STARTUP_STATUS_ENDPOINT_PATH} returned non-JSON content."
                except VastSmokeError:
                    raise
                except (httpx.HTTPError, ValueError) as error:
                    last_error = str(error) or f"{DEFAULT_STARTUP_STATUS_ENDPOINT_PATH} is unavailable."
                    runtime_report["startup_status_poll_error"] = last_error
            try:
                response = self.runtime.get(models_url)
                runtime_report["models_status"] = int(response.status_code)
                if response.status_code < 500:
                    payload = response.json()
                    if isinstance(payload, dict):
                        return payload, int(response.status_code)
                    last_error = f"{DEFAULT_MODELS_PATH} returned non-JSON content."
                else:
                    last_error = f"{DEFAULT_MODELS_PATH} returned HTTP {response.status_code}."
            except (httpx.HTTPError, ValueError) as error:
                models_error = str(error) or last_error
                last_error = f"{startup_summary} / {models_error}" if startup_summary else models_error
            self.sleep(poll_interval_seconds)
        raise VastSmokeError(f"The OpenAI-compatible runtime did not become ready in time: {last_error}")

    def refresh_startup_status(
        self,
        runtime_report: dict[str, Any],
        *,
        startup_status_url: str | None,
    ) -> dict[str, Any] | None:
        if not startup_status_url:
            return None
        try:
            response = self.runtime.get(startup_status_url)
            runtime_report["startup_status_code"] = int(response.status_code)
            payload = response_json_object(response)
            if payload is not None:
                runtime_report["startup_status"] = payload
                runtime_report.pop("startup_status_post_ready_error", None)
                return payload
            runtime_report["startup_status_post_ready_error"] = (
                f"{DEFAULT_STARTUP_STATUS_ENDPOINT_PATH} returned non-JSON content after readiness."
            )
        except (httpx.HTTPError, ValueError) as error:
            runtime_report["startup_status_post_ready_error"] = (
                str(error) or f"{DEFAULT_STARTUP_STATUS_ENDPOINT_PATH} is unavailable after readiness."
            )
        return None

    def wait_for_startup_status_ready(
        self,
        runtime_report: dict[str, Any],
        *,
        startup_status_url: str | None,
        timeout_seconds: float = DEFAULT_POST_PROBE_STATUS_GRACE_SECONDS,
        poll_interval_seconds: float = DEFAULT_POST_PROBE_STATUS_POLL_SECONDS,
    ) -> dict[str, Any] | None:
        if not startup_status_url:
            return None
        deadline = self.monotonic() + max(0.0, float(timeout_seconds))
        payload = self.refresh_startup_status(
            runtime_report,
            startup_status_url=startup_status_url,
        )
        while self.monotonic() < deadline and startup_status_state(runtime_report) not in {"ready", "failed"}:
            self.sleep(poll_interval_seconds)
            payload = self.refresh_startup_status(
                runtime_report,
                startup_status_url=startup_status_url,
            )
        return payload

    def _run_embeddings_probe(
        self,
        base_url: str,
        model: str,
        *,
        benchmark: bool = False,
        benchmark_profile: str = DEFAULT_BENCHMARK_PROFILE,
    ) -> dict[str, Any]:
        response = self.runtime.post(
            f"{base_url}{DEFAULT_EMBEDDINGS_PATH}",
            json_body={
                "model": model,
                "input": benchmark_embeddings_input_for_model(model) if benchmark else DEFAULT_EMBEDDINGS_INPUT,
            },
        )
        response.raise_for_status()
        payload = response.json()
        embedding_len = 0
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            embedding = data[0].get("embedding")
            if isinstance(embedding, list):
                embedding_len = len(embedding)
        return {
            "api": "embeddings",
            "path": DEFAULT_EMBEDDINGS_PATH,
            "status": int(response.status_code),
            "embedding_length": embedding_len,
            "usage": payload.get("usage"),
        }

    def _run_responses_probe(
        self,
        base_url: str,
        model: str,
        *,
        benchmark: bool = False,
        benchmark_profile: str = DEFAULT_BENCHMARK_PROFILE,
    ) -> dict[str, Any]:
        json_body: dict[str, Any] = {
            "model": model,
            "input": DEFAULT_RESPONSES_INPUT,
            "max_output_tokens": 8,
            "temperature": 0,
        }
        if benchmark:
            json_body["input"] = benchmark_responses_input(benchmark_profile, model=model)
            json_body["max_output_tokens"] = benchmark_max_output_tokens(benchmark_profile)
            json_body["temperature"] = 0
        response = self.runtime.post(
            f"{base_url}{DEFAULT_RESPONSES_PATH}",
            json_body=json_body,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "api": "responses",
            "path": DEFAULT_RESPONSES_PATH,
            "status": int(response.status_code),
            "output_text": extract_response_text(payload),
            "usage": payload.get("usage"),
        }

    def _run_chat_completions_probe(
        self,
        base_url: str,
        model: str,
        *,
        benchmark: bool = False,
        benchmark_profile: str = DEFAULT_BENCHMARK_PROFILE,
    ) -> dict[str, Any]:
        message_content = benchmark_responses_input(benchmark_profile, model=model) if benchmark else DEFAULT_RESPONSES_INPUT
        json_body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": benchmark_max_output_tokens(benchmark_profile) if benchmark else 8,
            "temperature": 0,
        }
        response = self.runtime.post(
            f"{base_url}{DEFAULT_CHAT_COMPLETIONS_PATH}",
            json_body=json_body,
        )
        response.raise_for_status()
        payload = response.json()
        output_text: str | None = None
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    output_text = content.strip()
        return {
            "api": "chat_completions_fallback",
            "path": DEFAULT_CHAT_COMPLETIONS_PATH,
            "status": int(response.status_code),
            "output_text": output_text,
            "usage": payload.get("usage"),
        }

    def run_probe(
        self,
        base_url: str,
        *,
        model: str,
        api_kind: str,
        smoke_test_api_path: str | None = None,
        benchmark: bool = False,
        benchmark_profile: str = DEFAULT_BENCHMARK_PROFILE,
    ) -> dict[str, Any]:
        normalized = str(api_kind or "auto").strip().lower()
        if normalized == "embeddings":
            return self._run_embeddings_probe(base_url, model, benchmark=benchmark, benchmark_profile=benchmark_profile)
        if normalized == "responses":
            return self._run_responses_probe(base_url, model, benchmark=benchmark, benchmark_profile=benchmark_profile)
        if normalized == "chat_completions":
            return self._run_chat_completions_probe(
                base_url,
                model,
                benchmark=benchmark,
                benchmark_profile=benchmark_profile,
            )
        path_preference = preferred_api_for_path(smoke_test_api_path)
        if path_preference == "embeddings":
            return self._run_embeddings_probe(base_url, model, benchmark=benchmark, benchmark_profile=benchmark_profile)
        if path_preference == "responses":
            try:
                return self._run_responses_probe(
                    base_url,
                    model,
                    benchmark=benchmark,
                    benchmark_profile=benchmark_profile,
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in {400, 404, 405, 422}:
                    raise
            except httpx.HTTPError:
                pass
            return self._run_chat_completions_probe(
                base_url,
                model,
                benchmark=benchmark,
                benchmark_profile=benchmark_profile,
            )
        if path_preference == "chat_completions":
            return self._run_chat_completions_probe(
                base_url,
                model,
                benchmark=benchmark,
                benchmark_profile=benchmark_profile,
            )
        preferred = preferred_api_for_model(model)
        if preferred == "embeddings":
            return self._run_embeddings_probe(base_url, model, benchmark=benchmark, benchmark_profile=benchmark_profile)
        try:
            return self._run_responses_probe(
                base_url,
                model,
                benchmark=benchmark,
                benchmark_profile=benchmark_profile,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {400, 404, 405, 422}:
                raise
        except httpx.HTTPError:
            pass
        return self._run_chat_completions_probe(
            base_url,
            model,
            benchmark=benchmark,
            benchmark_profile=benchmark_profile,
        )

    def run_probe_with_retries(
        self,
        base_url: str,
        *,
        model: str,
        api_kind: str,
        smoke_test_api_path: str | None = None,
        attempts: int = DEFAULT_PROBE_RETRY_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_PROBE_RETRY_DELAY_SECONDS,
        benchmark: bool = False,
        benchmark_profile: str = DEFAULT_BENCHMARK_PROFILE,
    ) -> tuple[dict[str, Any], list[str]]:
        normalized_attempts = max(1, int(attempts))
        retry_notes: list[str] = []
        last_error: Exception | None = None
        for attempt in range(1, normalized_attempts + 1):
            try:
                probe = self.run_probe(
                    base_url,
                    model=model,
                    api_kind=api_kind,
                    smoke_test_api_path=smoke_test_api_path,
                    benchmark=benchmark,
                    benchmark_profile=benchmark_profile,
                )
                return probe, retry_notes
            except httpx.HTTPStatusError as error:
                last_error = error
                status_code = getattr(error.response, "status_code", None)
                if status_code not in {500, 502, 503, 504} or attempt >= normalized_attempts:
                    raise
                retry_notes.append(
                    f"Probe attempt {attempt} failed with HTTP {status_code}; retrying."
                )
            except httpx.HTTPError as error:
                last_error = error
                if attempt >= normalized_attempts:
                    raise
                retry_notes.append(
                    f"Probe attempt {attempt} failed with transient runtime error: {error}. Retrying."
                )
            self.sleep(retry_delay_seconds)
        if last_error is not None:
            raise last_error
        raise VastSmokeError("Smoke probe failed before any attempt could complete.")

    def run_benchmark(
        self,
        base_url: str,
        *,
        model: str,
        api_kind: str,
        smoke_test_api_path: str | None,
        request_count: int,
        concurrency: int,
        benchmark_profile: str,
        hourly_cost_usd: float,
        warmup_seconds: float = 0.0,
    ) -> tuple[dict[str, Any], list[str]]:
        normalized_request_count = max(1, int(request_count))
        normalized_concurrency = max(1, min(int(concurrency), normalized_request_count))
        latencies_seconds: list[float] = []
        api_counts: dict[str, int] = {}
        notes: list[str] = []
        input_tokens = 0
        cached_input_tokens = 0
        uncached_input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        started_at = self.monotonic()
        last_probe: dict[str, Any] | None = None

        def run_one_benchmark_probe() -> tuple[dict[str, Any], list[str], float]:
            request_started_at = self.monotonic()
            probe, retry_notes = self.run_probe_with_retries(
                base_url,
                model=model,
                api_kind=api_kind,
                smoke_test_api_path=smoke_test_api_path,
                benchmark=True,
                benchmark_profile=benchmark_profile,
            )
            latency_seconds = max(0.0, self.monotonic() - request_started_at)
            return probe, retry_notes, latency_seconds

        if normalized_concurrency == 1:
            benchmark_results = [run_one_benchmark_probe() for _ in range(normalized_request_count)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=normalized_concurrency) as executor:
                futures = [executor.submit(run_one_benchmark_probe) for _ in range(normalized_request_count)]
                benchmark_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        for probe, retry_notes, latency_seconds in benchmark_results:
            latencies_seconds.append(latency_seconds)
            notes.extend(retry_notes)
            last_probe = probe
            probe_api = str(probe.get("api") or "unknown").strip() or "unknown"
            api_counts[probe_api] = api_counts.get(probe_api, 0) + 1
            usage = normalized_usage_from_probe(probe)
            input_tokens += usage["input_tokens"]
            cached_input_tokens += usage["cached_input_tokens"]
            uncached_input_tokens += usage["uncached_input_tokens"]
            output_tokens += usage["output_tokens"]
            total_tokens += usage["total_tokens"]

        elapsed_seconds = max(0.0, self.monotonic() - started_at)
        economics = economics_for_tokens(
            hourly_cost_usd=hourly_cost_usd,
            elapsed_seconds=elapsed_seconds,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            uncached_input_tokens=uncached_input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        benchmark_report: dict[str, Any] = {
            "request_count": normalized_request_count,
            "concurrency": normalized_concurrency,
            "benchmark_profile": benchmark_profile_name(benchmark_profile),
            "elapsed_seconds": round_metric(elapsed_seconds),
            "requests_per_second": round_metric(normalized_request_count / max(elapsed_seconds, 1e-9)),
            "latency_seconds": {
                "avg": round_metric(sum(latencies_seconds) / len(latencies_seconds)) if latencies_seconds else 0.0,
                "min": round_metric(min(latencies_seconds)) if latencies_seconds else 0.0,
                "p50": round_metric(percentile(latencies_seconds, 50.0)) if latencies_seconds else 0.0,
                "p95": round_metric(percentile(latencies_seconds, 95.0)) if latencies_seconds else 0.0,
                "max": round_metric(max(latencies_seconds)) if latencies_seconds else 0.0,
            },
            "api_counts": api_counts,
            "usage": {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "uncached_input_tokens": uncached_input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "avg_input_tokens_per_request": round_metric(input_tokens / normalized_request_count),
                "avg_cached_input_tokens_per_request": round_metric(cached_input_tokens / normalized_request_count),
                "avg_uncached_input_tokens_per_request": round_metric(uncached_input_tokens / normalized_request_count),
                "avg_output_tokens_per_request": round_metric(output_tokens / normalized_request_count),
                "avg_total_tokens_per_request": round_metric(total_tokens / normalized_request_count),
            },
            "economics": economics,
            "pricing": pricing_guidance_for_economics(
                economics,
                warmup_seconds=warmup_seconds,
            ),
        }
        if last_probe is not None:
            benchmark_report["last_probe"] = {
                "api": last_probe.get("api"),
                "path": last_probe.get("path"),
                "status": last_probe.get("status"),
            }
        return benchmark_report, notes

    def run(self, config: VastSmokeConfig) -> dict[str, Any]:
        started_at = self.monotonic()
        report: dict[str, Any] = {
            "status": "error",
            "requested": {
                "model": config.model,
                "max_price": round(float(config.max_price), 6),
                "image": config.image,
                "api": config.api_kind,
                "runtype": config.runtype,
                "run_mode": "serve_only",
                "max_context_tokens": config.effective_max_context_tokens,
                "min_cuda_max_good": config.min_cuda_max_good,
                "expected_api_path": config.smoke_test_api_path,
                "readiness_path": DEFAULT_MODELS_PATH,
                "startup_status_path": DEFAULT_STARTUP_STATUS_ENDPOINT_PATH,
                "benchmark_requests": int(config.benchmark_requests),
                "benchmark_concurrency": int(config.benchmark_concurrency),
                "benchmark_profile": config.benchmark_profile,
                "vllm_extra_args": str(config.vllm_extra_args or "").strip() or None,
            },
            "selected_offer": None,
            "instance": None,
            "runtime": None,
            "probe": None,
            "benchmark": None,
            "candidate_failures": [],
            "cleanup": {"destroyed": False},
            "notes": [],
        }
        instance_id: int | None = None
        successful_launch_started_at: float | None = None
        ready_at: float | None = None
        try:
            offers = self.api.search_offers(config)
            candidate_offers = affordable_offers(
                offers,
                max_price=config.max_price,
                min_cuda_max_good=config.min_cuda_max_good,
                model=config.model,
            )
            launch_errors: list[str] = []
            for selected_offer in candidate_offers:
                candidate_summary = summarize_offer(selected_offer)
                report["selected_offer"] = candidate_summary
                report["instance"] = None
                report["runtime"] = None
                report["probe"] = None
                report["benchmark"] = None
                candidate_launch_started_at = self.monotonic()
                try:
                    instance_id = self.api.create_instance(
                        _int_value(selected_offer, "id"),
                        image=config.image,
                        env=build_launch_env(config),
                        disk_gb=config.disk_gb,
                        label=config.label,
                        runtype=config.runtype,
                    )
                    report["instance"] = {"id": instance_id, "runtype": config.runtype}
                    instance = self.wait_for_instance(
                        instance_id,
                        timeout_seconds=config.launch_timeout_seconds,
                        poll_interval_seconds=config.poll_interval_seconds,
                    )
                    host_port = extract_host_port(instance)
                    startup_status_host_port = extract_host_port(instance, container_port=DEFAULT_STARTUP_STATUS_PORT)
                    public_ip = str(instance.get("public_ipaddr") or "").strip()
                    if not public_ip:
                        raise VastSmokeError("Vast instance became ready without a public IP address.")
                    base_url = f"http://{public_ip}:{host_port}"
                    startup_status_url = f"http://{public_ip}:{startup_status_host_port}{DEFAULT_STARTUP_STATUS_ENDPOINT_PATH}"
                    report["instance"].update(
                        {
                            "public_ip": public_ip,
                            "host_port": host_port,
                            "startup_status_host_port": startup_status_host_port,
                            "base_url": base_url,
                            "gpu_name": instance.get("gpu_name"),
                            "gpu_ram_gb": round(_float_value(instance, "gpu_ram") / 1024.0, 1),
                            "dph_total": round(_float_value(instance, "dph_total"), 6),
                            "geolocation": instance.get("geolocation"),
                        }
                    )
                    report["runtime"] = {
                        "base_url": base_url,
                        "startup_status_path": DEFAULT_STARTUP_STATUS_ENDPOINT_PATH,
                        "startup_status_url": startup_status_url,
                    }
                    models_payload, models_status = self.wait_for_runtime_ready(
                        report["runtime"],
                        base_url,
                        startup_status_url=startup_status_url,
                        timeout_seconds=config.readiness_timeout_seconds,
                        poll_interval_seconds=config.poll_interval_seconds,
                    )
                    successful_launch_started_at = candidate_launch_started_at
                    ready_at = self.monotonic()
                    served_model = first_served_model(models_payload)
                    report["runtime"].update(
                        {
                            "models_path": DEFAULT_MODELS_PATH,
                            "models_status": models_status,
                            "served_model": served_model,
                        }
                    )
                    if startup_status_state(report["runtime"]) != "ready":
                        self.refresh_startup_status(
                            report["runtime"],
                            startup_status_url=startup_status_url,
                        )
                    probe, probe_retry_notes = self.run_probe_with_retries(
                        base_url,
                        model=served_model or config.model,
                        api_kind=config.api_kind,
                        smoke_test_api_path=config.smoke_test_api_path,
                    )
                    report["probe"] = probe
                    if probe_retry_notes:
                        report["notes"].extend(probe_retry_notes)
                    if startup_status_state(report["runtime"]) != "ready":
                        self.wait_for_startup_status_ready(
                            report["runtime"],
                            startup_status_url=startup_status_url,
                        )
                    if int(config.benchmark_requests) > 0:
                        benchmark, benchmark_notes = self.run_benchmark(
                            base_url,
                            model=served_model or config.model,
                            api_kind=api_kind_for_probe_result(probe),
                            smoke_test_api_path=str(probe.get("path") or config.smoke_test_api_path or "").strip() or None,
                            request_count=int(config.benchmark_requests),
                            concurrency=int(config.benchmark_concurrency),
                            benchmark_profile=config.benchmark_profile,
                            hourly_cost_usd=_float_value(instance, "dph_total"),
                            warmup_seconds=max(0.0, (ready_at or self.monotonic()) - candidate_launch_started_at),
                        )
                        report["benchmark"] = benchmark
                        if benchmark_notes:
                            report["notes"].extend(benchmark_notes)
                        if startup_status_state(report["runtime"]) != "ready":
                            self.wait_for_startup_status_ready(
                                report["runtime"],
                                startup_status_url=startup_status_url,
                            )
                    if served_model and served_model != config.model:
                        report["notes"].append(
                            f"Requested model {config.model} resolved to served model {served_model} during warmup."
                        )
                    report["status"] = "ok"
                    break
                except VastInstanceLaunchError as error:
                    launch_errors.append(str(error))
                    if error.retryable:
                        report["notes"].append(f"{error} Trying the next cheapest suitable offer.")
                        continue
                    raise
                except Exception as error:
                    report["candidate_failures"].append(
                        {
                            "offer": candidate_summary,
                            "error": str(error) or error.__class__.__name__,
                        }
                    )
                    retry_candidate = should_retry_candidate_after_error(error, report.get("runtime"))
                    current_instance_id = instance_id
                    instance_id = None
                    if current_instance_id is not None:
                        try:
                            self.api.destroy_instance(current_instance_id)
                            report["cleanup"] = {"destroyed": True, "instance_id": current_instance_id}
                        except Exception as destroy_error:  # pragma: no cover - exercised in live failures
                            report["cleanup"] = {
                                "destroyed": False,
                                "instance_id": current_instance_id,
                                "error": str(destroy_error) or destroy_error.__class__.__name__,
                            }
                            report["notes"].append(
                                f"Cleanup failed for Vast instance {current_instance_id}: {destroy_error}"
                            )
                    if retry_candidate and len(report["candidate_failures"]) < len(candidate_offers):
                        report["notes"].append(
                            f"{error} Trying the next suitable Vast host."
                        )
                        continue
                    raise
            if report["status"] != "ok":
                if launch_errors:
                    raise VastSmokeError(
                        f"No launchable Vast offers were left after retrying affordable candidates. Last error: {launch_errors[-1]}"
                    )
                if report["candidate_failures"]:
                    raise VastSmokeError(str(report["candidate_failures"][-1]["error"]))
                raise VastSmokeError("Vast offer search returned affordable offers, but none were launchable.")
        except Exception as error:
            report["error"] = str(error) or error.__class__.__name__
        finally:
            cleanup_error: str | None = None
            if instance_id is not None:
                try:
                    self.api.destroy_instance(instance_id)
                    report["cleanup"] = {"destroyed": True, "instance_id": instance_id}
                except Exception as error:  # pragma: no cover - exercised when Vast cleanup fails live
                    cleanup_error = str(error) or error.__class__.__name__
                    report["cleanup"] = {
                        "destroyed": False,
                        "instance_id": instance_id,
                        "error": cleanup_error,
                    }
            total_seconds = round(self.monotonic() - started_at, 2)
            timings = {"total_seconds": total_seconds}
            if successful_launch_started_at is not None and report.get("instance"):
                timings["launch_seconds"] = round((ready_at or self.monotonic()) - successful_launch_started_at, 2)
            if successful_launch_started_at is not None and ready_at is not None:
                timings["warm_seconds"] = round(ready_at - successful_launch_started_at, 2)
            report["timings"] = timings
            if cleanup_error:
                report.setdefault("notes", []).append(
                    f"Cleanup failed for Vast instance {instance_id}: {cleanup_error}"
                )
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a cheap Vast.ai serve-only smoke test for the AUTONOMOUSc runtime.")
    parser.add_argument(
        "--api-key",
        default="",
        help="Temporary Vast.ai API key. Prefer VAST_API_KEY or a local config file over typing it directly.",
    )
    parser.add_argument(
        "--config",
        default="",
        help=(
            "Optional local JSON config file for secrets such as api_key and hf_token. "
            f"Defaults to {default_vast_smoke_config_path()} when that file already exists."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_VAST_SMOKE_MODEL, help="Model to warm for the smoke test.")
    parser.add_argument("--image", default=DEFAULT_VAST_SMOKE_IMAGE, help="Container image to launch on Vast.ai.")
    parser.add_argument("--max-price", type=float, default=DEFAULT_VAST_LAUNCH_PROFILE.safe_price_ceiling_usd, help="Maximum hourly price in USD.")
    parser.add_argument("--disk-gb", type=int, default=DEFAULT_DISK_GB, help="Requested disk size in GB.")
    parser.add_argument("--min-vram-gb", type=float, default=DEFAULT_MIN_VRAM_GB, help="Minimum GPU VRAM in GB.")
    parser.add_argument(
        "--min-cuda-max-good",
        type=float,
        default=DEFAULT_MIN_CUDA_MAX_GOOD,
        help="Minimum cuda_max_good host capability for the selected runtime image.",
    )
    parser.add_argument("--min-reliability", type=float, default=DEFAULT_MIN_RELIABILITY, help="Minimum host reliability score.")
    parser.add_argument("--min-inet-down-mbps", type=float, default=DEFAULT_MIN_INET_DOWN_MBPS, help="Minimum internet download speed in Mbps.")
    parser.add_argument("--offer-limit", type=int, default=DEFAULT_OFFER_LIMIT, help="Maximum number of Vast offers to inspect.")
    parser.add_argument("--launch-timeout-seconds", type=float, default=DEFAULT_LAUNCH_TIMEOUT_SECONDS, help="How long to wait for the Vast instance to boot.")
    parser.add_argument("--readiness-timeout-seconds", type=float, default=DEFAULT_READINESS_TIMEOUT_SECONDS, help="How long to wait for /v1/models to come up.")
    parser.add_argument("--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS, help="Polling interval for Vast and runtime readiness.")
    parser.add_argument("--api", choices=("auto", "embeddings", "responses", "chat_completions"), default="auto", help="Smoke probe type to run after /v1/models succeeds.")
    parser.add_argument("--runtype", default=DEFAULT_VAST_LAUNCH_PROFILE.runtype, help="Vast runtype to use. Defaults to the runtime profile launch metadata.")
    parser.add_argument("--max-context-tokens", type=int, default=32768, help="Requested max context tokens before model-safe clamping.")
    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=DEFAULT_BENCHMARK_REQUESTS,
        help="Optional number of post-readiness benchmark requests to run for throughput and cost-per-million-token metrics.",
    )
    parser.add_argument(
        "--benchmark-concurrency",
        type=int,
        default=DEFAULT_BENCHMARK_CONCURRENCY,
        help="How many benchmark requests to keep in flight at once for burst throughput testing.",
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=("balanced", "input_heavy", "output_heavy"),
        default=DEFAULT_BENCHMARK_PROFILE,
        help="Benchmark workload shape for separating prompt-heavy and decode-heavy economics.",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help="Optional extra vLLM server arguments passed through VLLM_EXTRA_ARGS during the Vast launch.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Optional Hugging Face token for gated models. Prefer HUGGING_FACE_HUB_TOKEN or the local config file.",
    )
    parser.add_argument("--json-indent", type=int, default=2, help="JSON indentation level for the report.")
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> VastSmokeConfig:
    config_values, _config_path = load_vast_smoke_config(str(args.config or ""))
    api_key = first_nonempty(
        str(args.api_key or ""),
        os.getenv("VAST_API_KEY"),
        _config_value(config_values, "api_key", "vast_api_key"),
    )
    if not api_key:
        default_path = default_vast_smoke_config_path()
        raise VastSmokeError(
            "A Vast.ai API key is required. Pass --api-key, set VAST_API_KEY, "
            f"or add api_key to a local config file such as {default_path}."
        )
    hf_token = first_nonempty(
        str(args.hf_token or ""),
        os.getenv("HUGGING_FACE_HUB_TOKEN"),
        os.getenv("HF_TOKEN"),
        _config_value(config_values, "hf_token", "hugging_face_hub_token"),
    )
    return VastSmokeConfig(
        api_key=api_key,
        model=str(args.model).strip() or DEFAULT_VAST_SMOKE_MODEL,
        runtype=str(args.runtype).strip() or DEFAULT_VAST_LAUNCH_PROFILE.runtype,
        max_price=float(args.max_price),
        image=str(args.image).strip() or DEFAULT_VAST_SMOKE_IMAGE,
        disk_gb=max(1, int(args.disk_gb)),
        min_vram_gb=max(1.0, float(args.min_vram_gb)),
        min_cuda_max_good=float(args.min_cuda_max_good) if args.min_cuda_max_good is not None else None,
        min_reliability=max(0.0, min(1.0, float(args.min_reliability))),
        min_inet_down_mbps=max(0.0, float(args.min_inet_down_mbps)),
        offer_limit=max(1, int(args.offer_limit)),
        launch_timeout_seconds=max(30.0, float(args.launch_timeout_seconds)),
        readiness_timeout_seconds=max(30.0, float(args.readiness_timeout_seconds)),
        poll_interval_seconds=max(1.0, float(args.poll_interval_seconds)),
        api_kind=str(args.api).strip().lower(),
        smoke_test_api_path=expected_api_path_for(
            str(args.api).strip().lower(),
            str(args.model).strip() or DEFAULT_VAST_SMOKE_MODEL,
        ),
        max_context_tokens=max(1, int(args.max_context_tokens)),
        hf_token=hf_token or None,
        benchmark_requests=max(0, int(args.benchmark_requests)),
        benchmark_concurrency=max(1, int(args.benchmark_concurrency)),
        benchmark_profile=benchmark_profile_name(str(args.benchmark_profile or DEFAULT_BENCHMARK_PROFILE)),
        vllm_extra_args=str(args.vllm_extra_args or "").strip(),
    )


def emit_json_report(payload: Mapping[str, Any], *, indent: int) -> None:
    text = json.dumps(payload, indent=indent, ensure_ascii=False)
    try:
        print(text)
        return
    except UnicodeEncodeError:
        pass
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:  # pragma: no cover - highly unusual in practice
        raise
    buffer.write(text.encode("utf-8", errors="replace"))
    buffer.write(b"\n")
    buffer.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    json_indent = max(0, int(args.json_indent))
    try:
        config = build_config_from_args(args)
    except VastSmokeError as error:
        emit_json_report({"status": "error", "error": str(error)}, indent=json_indent)
        return 1

    api = VastAPI(config.api_key)
    runtime = RuntimeProbeClient()
    try:
        report = VastSmokeRunner(api, runtime).run(config)
    finally:
        runtime.close()
        api.close()

    emit_json_report(report, indent=json_indent)
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
