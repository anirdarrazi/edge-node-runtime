from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .config import NodeAgentSettings
from .fault_injection import FaultInjectionController

TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
TRANSIENT_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0, 4.0)
DNS_RETRY_DELAYS_SECONDS = (0.25, 0.75, 1.5, 3.0)
CONTROL_PLANE_STATE_VERSION = 1
DNS_ERROR_MARKERS = (
    "temporary failure in name resolution",
    "name or service not known",
    "nodename nor servname provided",
    "no address associated with hostname",
    "getaddrinfo failed",
    "hostname nor servname provided",
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _csv_urls(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    candidates = value.replace("\r", "\n").replace(";", "\n").replace(",", "\n").splitlines()
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        candidate = raw.strip().rstrip("/")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _state_path_from_settings(settings: NodeAgentSettings) -> Path:
    configured = str(getattr(settings, "control_plane_state_path", "") or "").strip()
    if configured:
        return Path(configured)
    autopilot_path = str(getattr(settings, "autopilot_state_path", "") or "").strip()
    if autopilot_path:
        return Path(autopilot_path).with_name("control-plane-state.json")
    credentials_path = str(getattr(settings, "credentials_path", "") or "").strip()
    if credentials_path:
        return Path(credentials_path).parent / "control-plane-state.json"
    return Path(".") / "control-plane-state.json"


@dataclass
class ControlPlaneReachabilityState:
    version: int = CONTROL_PLANE_STATE_VERSION
    primary_base_url: str = ""
    active_base_url: str = ""
    fallback_base_urls: list[str] | None = None
    status: str = "healthy"
    degraded_reason: str | None = None
    last_success_at: str | None = None
    last_failure_at: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    grace_deadline_at: str | None = None
    last_retry_after_seconds: float | None = None
    dns_failures: int = 0
    last_dns_error_at: str | None = None
    last_request_url: str | None = None


class EdgeControlTransport:
    """HTTP transport for control-plane calls with fallback hosts and persisted reachability state."""

    def __init__(self, settings: NodeAgentSettings):
        self.settings = settings
        self.primary_base_url = str(settings.edge_control_url).rstrip("/")
        self.fallback_base_urls = _csv_urls(getattr(settings, "edge_control_fallback_urls", None))
        self.base_urls = self._resolved_base_urls()
        self.base_url = self.primary_base_url
        self._timeout = httpx.Timeout(300.0, connect=30.0)
        self.client = httpx.Client(base_url=self.primary_base_url, timeout=self._timeout)
        self.state_path = _state_path_from_settings(settings)
        self.faults = FaultInjectionController.from_settings(settings)
        self._state = self._load_state()
        self._coerce_recovered_state()

    def _resolved_base_urls(self) -> list[str]:
        ordered = [self.primary_base_url, *self.fallback_base_urls]
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in ordered:
            normalized = str(candidate or "").strip().rstrip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped or [self.primary_base_url]

    def _load_state(self) -> ControlPlaneReachabilityState:
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                active_base_url = str(payload.get("active_base_url") or self.primary_base_url).rstrip("/")
                if active_base_url not in self.base_urls:
                    active_base_url = self.primary_base_url
                fallback_base_urls = [
                    url for url in payload.get("fallback_base_urls", []) if isinstance(url, str) and url.rstrip("/")
                ]
                return ControlPlaneReachabilityState(
                    version=int(payload.get("version") or CONTROL_PLANE_STATE_VERSION),
                    primary_base_url=self.primary_base_url,
                    active_base_url=active_base_url,
                    fallback_base_urls=fallback_base_urls or list(self.fallback_base_urls),
                    status=str(payload.get("status") or "healthy"),
                    degraded_reason=(
                        str(payload.get("degraded_reason")) if payload.get("degraded_reason") else None
                    ),
                    last_success_at=str(payload.get("last_success_at")) if payload.get("last_success_at") else None,
                    last_failure_at=str(payload.get("last_failure_at")) if payload.get("last_failure_at") else None,
                    last_error=str(payload.get("last_error")) if payload.get("last_error") else None,
                    consecutive_failures=max(0, int(payload.get("consecutive_failures") or 0)),
                    grace_deadline_at=(
                        str(payload.get("grace_deadline_at")) if payload.get("grace_deadline_at") else None
                    ),
                    last_retry_after_seconds=(
                        float(payload.get("last_retry_after_seconds"))
                        if payload.get("last_retry_after_seconds") is not None
                        else None
                    ),
                    dns_failures=max(0, int(payload.get("dns_failures") or 0)),
                    last_dns_error_at=(
                        str(payload.get("last_dns_error_at")) if payload.get("last_dns_error_at") else None
                    ),
                    last_request_url=(
                        str(payload.get("last_request_url")) if payload.get("last_request_url") else None
                    ),
                )
        return ControlPlaneReachabilityState(
            primary_base_url=self.primary_base_url,
            active_base_url=self.primary_base_url,
            fallback_base_urls=list(self.fallback_base_urls),
        )

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self._state), indent=2), encoding="utf-8")

    def _coerce_recovered_state(self) -> None:
        if self._state.status != "degraded":
            return
        grace_deadline = _parse_timestamp(self._state.grace_deadline_at)
        last_success = _parse_timestamp(self._state.last_success_at)
        last_failure = _parse_timestamp(self._state.last_failure_at)
        if grace_deadline is None or _now_utc() <= grace_deadline:
            return
        if last_success is None:
            return
        if last_failure is not None and last_success < last_failure:
            return
        self._state.status = "healthy"
        self._state.degraded_reason = None
        self._state.last_error = None
        self._state.last_retry_after_seconds = None
        self._state.consecutive_failures = 0
        self._state.active_base_url = self.primary_base_url
        self._save_state()

    def snapshot(self) -> dict[str, Any]:
        self._coerce_recovered_state()
        grace_deadline = _parse_timestamp(self._state.grace_deadline_at)
        return {
            **asdict(self._state),
            "fallback_active": self._state.active_base_url not in {"", self.primary_base_url},
            "grace_active": bool(grace_deadline and _now_utc() <= grace_deadline),
            "base_urls": list(self.base_urls),
        }

    def recommended_retry_delay_seconds(self) -> float:
        floor = max(1.0, float(getattr(self.settings, "control_plane_retry_floor_seconds", 3) or 3))
        cap = max(floor, float(getattr(self.settings, "control_plane_retry_cap_seconds", 30) or 30))
        exponent = max(0, self._state.consecutive_failures - 1)
        base_delay = min(cap, floor * (2**exponent))
        jitter = random.uniform(0.0, max(0.5, base_delay * 0.25))
        return min(cap, base_delay + jitter)

    def is_auth_error(self, error: Exception) -> bool:
        if not isinstance(error, httpx.HTTPStatusError) or error.response.status_code not in {401, 403}:
            return False
        request_url = str(error.request.url)
        return any(request_url.startswith(base_url) for base_url in self.base_urls)

    @staticmethod
    def is_dns_error(error: Exception) -> bool:
        if not isinstance(error, httpx.TransportError):
            return False
        lowered = str(error).strip().lower()
        return any(marker in lowered for marker in DNS_ERROR_MARKERS)

    @staticmethod
    def is_transient_network_error(error: Exception) -> bool:
        if isinstance(error, httpx.TransportError):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in TRANSIENT_STATUS_CODES
        return False

    @staticmethod
    def _is_relative_target(path_or_url: str) -> bool:
        parsed = urlparse(str(path_or_url))
        return not parsed.scheme and not parsed.netloc

    def _resolve_target(self, base_url: str, path_or_url: str) -> str:
        if not self._is_relative_target(path_or_url):
            return str(path_or_url)
        return urljoin(f"{base_url.rstrip('/')}/", str(path_or_url).lstrip("/"))

    def _ordered_base_urls(self) -> list[str]:
        active = self._state.active_base_url if self._state.active_base_url in self.base_urls else self.primary_base_url
        ordered = [active, self.primary_base_url, *self.base_urls]
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in ordered:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped or [self.primary_base_url]

    def _request_once(self, method: str, path_or_url: str, *, base_url: str | None = None, **kwargs: Any) -> httpx.Response:
        target = self._resolve_target(base_url or self.primary_base_url, path_or_url)
        request = getattr(self.client, "request", None)
        if callable(request):
            if base_url is None or base_url == self.primary_base_url:
                return request(method, path_or_url, **kwargs)
            return request(method, target, **kwargs)

        method_handler = getattr(self.client, method.lower(), None)
        if not callable(method_handler):
            raise AttributeError(f"client does not support {method.lower()}() or request()")

        supported_kwargs = {key: value for key, value in kwargs.items() if key in {"json", "content", "headers"}}
        if base_url is None or base_url == self.primary_base_url:
            return method_handler(path_or_url, **supported_kwargs)
        return method_handler(target, **supported_kwargs)

    def _record_success(self, *, base_url: str, request_url: str, had_retryable_failure: bool) -> None:
        now = _now_utc()
        self._state.primary_base_url = self.primary_base_url
        self._state.active_base_url = base_url
        self._state.fallback_base_urls = list(self.fallback_base_urls)
        self._state.last_request_url = request_url
        self._state.last_success_at = now.isoformat()
        self._state.grace_deadline_at = (
            now + timedelta(seconds=max(30, int(getattr(self.settings, "control_plane_grace_seconds", 180) or 180)))
        ).isoformat()
        self._state.last_retry_after_seconds = None
        self._state.consecutive_failures = 0
        if had_retryable_failure or base_url != self.primary_base_url:
            self._state.status = "degraded"
            self._state.degraded_reason = "fallback_host" if base_url != self.primary_base_url else "intermittent_connectivity"
        else:
            self._state.status = "healthy"
            self._state.degraded_reason = None
            self._state.last_error = None
        self._save_state()

    def _record_failure(self, *, error: Exception, request_url: str, base_url: str) -> None:
        now = _now_utc()
        self._state.primary_base_url = self.primary_base_url
        self._state.fallback_base_urls = list(self.fallback_base_urls)
        self._state.last_request_url = request_url
        self._state.last_failure_at = now.isoformat()
        self._state.last_error = str(error) or error.__class__.__name__
        self._state.consecutive_failures += 1
        self._state.last_retry_after_seconds = self.recommended_retry_delay_seconds()
        if self.is_dns_error(error):
            self._state.dns_failures += 1
            self._state.last_dns_error_at = now.isoformat()
            self._state.degraded_reason = "dns_resolution"
        else:
            self._state.degraded_reason = "control_plane_unreachable"
        grace_deadline = _parse_timestamp(self._state.grace_deadline_at)
        if grace_deadline is not None and now <= grace_deadline:
            self._state.status = "degraded"
        else:
            self._state.status = "offline"
        self._state.active_base_url = base_url
        self._save_state()

    def _retry_plan(self, *, dns_error: bool) -> tuple[float, ...]:
        return DNS_RETRY_DELAYS_SECONDS if dns_error else TRANSIENT_RETRY_DELAYS_SECONDS

    def _request_with_retry(self, method: str, path_or_url: str, **kwargs: Any) -> httpx.Response:
        is_relative_target = self._is_relative_target(path_or_url)
        had_retryable_failure = False
        last_error: Exception | None = None
        retry_plan: tuple[float, ...] = TRANSIENT_RETRY_DELAYS_SECONDS
        attempt_index = 0
        while True:
            delay_seconds = retry_plan[attempt_index] if attempt_index < len(retry_plan) else None
            for base_url in (self._ordered_base_urls() if is_relative_target else [self.primary_base_url]):
                request_url = self._resolve_target(base_url, path_or_url) if is_relative_target else str(path_or_url)
                try:
                    if is_relative_target and self.faults.consume("dns_flap"):
                        raise httpx.ConnectError("temporary failure in name resolution")
                    response = self._request_once(method, path_or_url, base_url=base_url if is_relative_target else None, **kwargs)
                    self._raise_for_status_with_body(response)
                    self._record_success(
                        base_url=base_url,
                        request_url=request_url,
                        had_retryable_failure=had_retryable_failure,
                    )
                    return response
                except Exception as error:
                    last_error = error
                    if not self.is_transient_network_error(error):
                        raise
                    had_retryable_failure = True
                    self._record_failure(error=error, request_url=request_url, base_url=base_url)
                    retry_plan = self._retry_plan(dns_error=self.is_dns_error(error))
            if delay_seconds is None:
                break
            time.sleep(delay_seconds + random.uniform(0.0, min(0.35, delay_seconds * 0.25)))
            attempt_index += 1
        if last_error is not None:
            raise last_error
        raise RuntimeError("control plane retry loop exhausted")

    @staticmethod
    def _raise_for_status_with_body(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            body = str(getattr(response, "text", "") or "").strip()
            if not body:
                raise
            if len(body) > 1200:
                body = f"{body[:1200]}..."
            raise httpx.HTTPStatusError(
                f"{error} Response body: {body}",
                request=error.request,
                response=error.response,
            ) from error

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        response = self._request_with_retry("POST", path, json=payload)
        return response.json()

    def post_content(self, path: str, payload: bytes, headers: dict[str, str]) -> Any:
        response = self._request_with_retry("POST", path, content=payload, headers=headers)
        return response.json() if response.content else {}

    def put_content(self, url: str, payload: bytes, headers: dict[str, str]) -> Any:
        response = self._request_with_retry("PUT", url, content=payload, headers=headers)
        return response.json() if response.content else {}

    def get_content(self, url: str) -> bytes:
        response = self._request_with_retry("GET", url)
        return response.content
