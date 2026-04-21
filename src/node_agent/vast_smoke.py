from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .runtime_profiles import (
    VAST_VLLM_SAFETENSORS_PROFILE,
    default_vast_launch_profile,
    runtime_profile_by_id,
)
from .runtime_layout import appliance_runtime_root
from .single_container import normalized_max_context_tokens


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
DEFAULT_MIN_RELIABILITY = 0.98
DEFAULT_MIN_INET_DOWN_MBPS = 50.0
DEFAULT_LAUNCH_TIMEOUT_SECONDS = 240.0
DEFAULT_READINESS_TIMEOUT_SECONDS = 900.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_MODELS_PATH = DEFAULT_VAST_RUNTIME_PROFILE.readiness_path
DEFAULT_EMBEDDINGS_PATH = "/v1/embeddings"
DEFAULT_RESPONSES_PATH = "/v1/responses"
DEFAULT_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
DEFAULT_EMBEDDINGS_INPUT = ["hello from autonomousc smoke test"]
DEFAULT_RESPONSES_INPUT = "Reply with the single word ready."
VAST_SMOKE_CONFIG_ENV = "NODE_AGENT_VAST_SMOKE_CONFIG"
DEFAULT_VAST_SMOKE_CONFIG_NAME = "vast-smoke.json"


class VastSmokeError(RuntimeError):
    pass


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
        response = self._client.post("/bundles/", json=payload)
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
        response = self._client.put(f"/asks/{offer_id}/", json=payload)
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise VastSmokeError(str(body.get("msg") or body.get("error") or "Vast instance launch failed."))
        instance_id = body.get("new_contract")
        if not isinstance(instance_id, int):
            raise VastSmokeError("Vast instance launch did not return a new contract id.")
        return instance_id

    def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        response = self._client.get(f"/instances/{instance_id}/")
        response.raise_for_status()
        body = response.json()
        instance = body.get("instances")
        return instance if isinstance(instance, dict) else None

    def destroy_instance(self, instance_id: int) -> None:
        response = self._client.delete(f"/instances/{instance_id}/")
        response.raise_for_status()
        body = response.json()
        if not body.get("success"):
            raise VastSmokeError(str(body.get("msg") or body.get("error") or "Vast instance destroy failed."))


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


def choose_cheapest_offer(offers: list[dict[str, Any]], *, max_price: float) -> dict[str, Any]:
    affordable = [
        offer for offer in offers if _float_value(offer, "dph_total", default=10**9) <= float(max_price)
    ]
    if not affordable:
        raise VastSmokeError(
            f"No suitable Vast offers were available at or below ${max_price:.2f}/hr."
        )
    return sorted(
        affordable,
        key=lambda offer: (
            _float_value(offer, "dph_total", default=10**9),
            -_float_value(offer, "reliability") or -_float_value(offer, "reliability2"),
            -_float_value(offer, "inet_down"),
            _int_value(offer, "id", default=10**9),
        ),
    )[0]


def build_launch_env(config: VastSmokeConfig) -> dict[str, str]:
    env = {
        "-p 8000:8000": "1",
        "RUN_MODE": "serve_only",
        "START_NODE_AGENT": "false",
        "VLLM_MODEL": config.model,
        "SUPPORTED_MODELS": config.model,
        "MAX_CONTEXT_TOKENS": str(config.effective_max_context_tokens),
    }
    if config.hf_token:
        env["HUGGING_FACE_HUB_TOKEN"] = config.hf_token
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


def preferred_api_for_model(model: str | None) -> str:
    normalized = str(model or "").strip()
    if normalized == DEFAULT_VAST_SMOKE_MODEL:
        return "embeddings"
    return "responses"


def preferred_api_for_path(path: str | None) -> str | None:
    normalized = str(path or "").strip()
    if normalized == DEFAULT_EMBEDDINGS_PATH:
        return "embeddings"
    if normalized in {DEFAULT_RESPONSES_PATH, DEFAULT_CHAT_COMPLETIONS_PATH}:
        return "responses"
    return None


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

    def wait_for_instance(self, instance_id: int, *, timeout_seconds: float, poll_interval_seconds: float) -> dict[str, Any]:
        deadline = self.monotonic() + max(1.0, timeout_seconds)
        last_status = "Vast instance is still starting."
        while self.monotonic() < deadline:
            instance = self.api.get_instance(instance_id)
            if not isinstance(instance, dict):
                last_status = "Vast instance details are not available yet."
                self.sleep(poll_interval_seconds)
                continue
            actual_status = str(instance.get("actual_status") or "").strip().lower()
            cur_state = str(instance.get("cur_state") or "").strip().lower()
            status_msg = str(instance.get("status_msg") or "").strip()
            if actual_status == "running":
                try:
                    extract_host_port(instance)
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
            self.sleep(poll_interval_seconds)
        raise VastSmokeError(f"Vast instance did not become ready in time: {last_status}")

    def wait_for_models(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> tuple[dict[str, Any], int]:
        deadline = self.monotonic() + max(1.0, timeout_seconds)
        last_error = "The OpenAI-compatible runtime is still starting."
        url = f"{base_url}{DEFAULT_MODELS_PATH}"
        while self.monotonic() < deadline:
            try:
                response = self.runtime.get(url)
                if response.status_code < 500:
                    payload = response.json()
                    if isinstance(payload, dict):
                        return payload, int(response.status_code)
                    last_error = f"{DEFAULT_MODELS_PATH} returned non-JSON content."
                else:
                    last_error = f"{DEFAULT_MODELS_PATH} returned HTTP {response.status_code}."
            except (httpx.HTTPError, ValueError) as error:
                last_error = str(error) or last_error
            self.sleep(poll_interval_seconds)
        raise VastSmokeError(f"The OpenAI-compatible runtime did not become ready in time: {last_error}")

    def _run_embeddings_probe(self, base_url: str, model: str) -> dict[str, Any]:
        response = self.runtime.post(
            f"{base_url}{DEFAULT_EMBEDDINGS_PATH}",
            json_body={"model": model, "input": DEFAULT_EMBEDDINGS_INPUT},
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

    def _run_responses_probe(self, base_url: str, model: str) -> dict[str, Any]:
        response = self.runtime.post(
            f"{base_url}{DEFAULT_RESPONSES_PATH}",
            json_body={"model": model, "input": DEFAULT_RESPONSES_INPUT},
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

    def _run_chat_completions_probe(self, base_url: str, model: str) -> dict[str, Any]:
        response = self.runtime.post(
            f"{base_url}{DEFAULT_CHAT_COMPLETIONS_PATH}",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": DEFAULT_RESPONSES_INPUT}],
                "max_tokens": 8,
            },
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
    ) -> dict[str, Any]:
        normalized = str(api_kind or "auto").strip().lower()
        if normalized == "embeddings":
            return self._run_embeddings_probe(base_url, model)
        if normalized == "responses":
            return self._run_responses_probe(base_url, model)
        path_preference = preferred_api_for_path(smoke_test_api_path)
        if path_preference == "embeddings":
            return self._run_embeddings_probe(base_url, model)
        if path_preference == "responses":
            try:
                return self._run_responses_probe(base_url, model)
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in {400, 404, 405, 422}:
                    raise
            return self._run_chat_completions_probe(base_url, model)
        preferred = preferred_api_for_model(model)
        if preferred == "embeddings":
            return self._run_embeddings_probe(base_url, model)
        try:
            return self._run_responses_probe(base_url, model)
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {400, 404, 405, 422}:
                raise
        return self._run_chat_completions_probe(base_url, model)

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
                "expected_api_path": config.smoke_test_api_path,
                "readiness_path": DEFAULT_MODELS_PATH,
            },
            "selected_offer": None,
            "instance": None,
            "runtime": None,
            "probe": None,
            "cleanup": {"destroyed": False},
            "notes": [],
        }
        instance_id: int | None = None
        launch_started_at: float | None = None
        ready_at: float | None = None
        try:
            offers = self.api.search_offers(config)
            selected_offer = choose_cheapest_offer(offers, max_price=config.max_price)
            report["selected_offer"] = summarize_offer(selected_offer)
            launch_started_at = self.monotonic()
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
            public_ip = str(instance.get("public_ipaddr") or "").strip()
            if not public_ip:
                raise VastSmokeError("Vast instance became ready without a public IP address.")
            base_url = f"http://{public_ip}:{host_port}"
            report["instance"].update(
                {
                    "public_ip": public_ip,
                    "host_port": host_port,
                    "base_url": base_url,
                    "gpu_name": instance.get("gpu_name"),
                    "gpu_ram_gb": round(_float_value(instance, "gpu_ram") / 1024.0, 1),
                    "dph_total": round(_float_value(instance, "dph_total"), 6),
                    "geolocation": instance.get("geolocation"),
                }
            )
            models_payload, models_status = self.wait_for_models(
                base_url,
                timeout_seconds=config.readiness_timeout_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
            )
            ready_at = self.monotonic()
            served_model = first_served_model(models_payload)
            report["runtime"] = {
                "models_path": DEFAULT_MODELS_PATH,
                "models_status": models_status,
                "served_model": served_model,
                "base_url": base_url,
            }
            probe = self.run_probe(
                base_url,
                model=served_model or config.model,
                api_kind=config.api_kind,
                smoke_test_api_path=config.smoke_test_api_path,
            )
            report["probe"] = probe
            if served_model and served_model != config.model:
                report["notes"].append(
                    f"Requested model {config.model} resolved to served model {served_model} during warmup."
                )
            report["status"] = "ok"
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
            if launch_started_at is not None and report.get("instance"):
                timings["launch_seconds"] = round((ready_at or self.monotonic()) - launch_started_at, 2)
            if launch_started_at is not None and ready_at is not None:
                timings["warm_seconds"] = round(ready_at - launch_started_at, 2)
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
    parser.add_argument("--min-reliability", type=float, default=DEFAULT_MIN_RELIABILITY, help="Minimum host reliability score.")
    parser.add_argument("--min-inet-down-mbps", type=float, default=DEFAULT_MIN_INET_DOWN_MBPS, help="Minimum internet download speed in Mbps.")
    parser.add_argument("--offer-limit", type=int, default=DEFAULT_OFFER_LIMIT, help="Maximum number of Vast offers to inspect.")
    parser.add_argument("--launch-timeout-seconds", type=float, default=DEFAULT_LAUNCH_TIMEOUT_SECONDS, help="How long to wait for the Vast instance to boot.")
    parser.add_argument("--readiness-timeout-seconds", type=float, default=DEFAULT_READINESS_TIMEOUT_SECONDS, help="How long to wait for /v1/models to come up.")
    parser.add_argument("--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS, help="Polling interval for Vast and runtime readiness.")
    parser.add_argument("--api", choices=("auto", "embeddings", "responses"), default="auto", help="Smoke probe type to run after /v1/models succeeds.")
    parser.add_argument("--runtype", default=DEFAULT_VAST_LAUNCH_PROFILE.runtype, help="Vast runtype to use. Defaults to the runtime profile launch metadata.")
    parser.add_argument("--max-context-tokens", type=int, default=32768, help="Requested max context tokens before model-safe clamping.")
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
        min_reliability=max(0.0, min(1.0, float(args.min_reliability))),
        min_inet_down_mbps=max(0.0, float(args.min_inet_down_mbps)),
        offer_limit=max(1, int(args.offer_limit)),
        launch_timeout_seconds=max(30.0, float(args.launch_timeout_seconds)),
        readiness_timeout_seconds=max(30.0, float(args.readiness_timeout_seconds)),
        poll_interval_seconds=max(1.0, float(args.poll_interval_seconds)),
        api_kind=str(args.api).strip().lower(),
        smoke_test_api_path=DEFAULT_VAST_SMOKE_API_PATH,
        max_context_tokens=max(1, int(args.max_context_tokens)),
        hf_token=hf_token or None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    json_indent = max(0, int(args.json_indent))
    try:
        config = build_config_from_args(args)
    except VastSmokeError as error:
        print(json.dumps({"status": "error", "error": str(error)}, indent=json_indent))
        return 1

    api = VastAPI(config.api_key)
    runtime = RuntimeProbeClient()
    try:
        report = VastSmokeRunner(api, runtime).run(config)
    finally:
        runtime.close()
        api.close()

    print(json.dumps(report, indent=json_indent, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
