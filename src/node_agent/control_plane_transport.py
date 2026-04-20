from __future__ import annotations

import time
from typing import Any

import httpx

from .config import NodeAgentSettings

TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
TRANSIENT_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)


class EdgeControlTransport:
    """Thin HTTP transport for control-plane calls."""

    def __init__(self, settings: NodeAgentSettings):
        self.settings = settings
        self.base_url = settings.edge_control_url
        self.client = httpx.Client(base_url=settings.edge_control_url, timeout=httpx.Timeout(300.0, connect=30.0))

    def is_auth_error(self, error: Exception) -> bool:
        if not isinstance(error, httpx.HTTPStatusError) or error.response.status_code not in {401, 403}:
            return False
        return str(error.request.url).startswith(self.base_url)

    @staticmethod
    def is_transient_network_error(error: Exception) -> bool:
        if isinstance(error, httpx.TransportError):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in TRANSIENT_STATUS_CODES
        return False

    def _request_once(self, method: str, path_or_url: str, **kwargs: Any) -> httpx.Response:
        request = getattr(self.client, "request", None)
        if callable(request):
            return request(method, path_or_url, **kwargs)

        method_handler = getattr(self.client, method.lower(), None)
        if not callable(method_handler):
            raise AttributeError(f"client does not support {method.lower()}() or request()")

        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"json", "content", "headers"}
        }
        return method_handler(path_or_url, **supported_kwargs)

    def _request_with_retry(self, method: str, path_or_url: str, **kwargs: Any) -> httpx.Response:
        for delay_seconds in (*TRANSIENT_RETRY_DELAYS_SECONDS, None):
            try:
                response = self._request_once(method, path_or_url, **kwargs)
                response.raise_for_status()
                return response
            except Exception as error:
                if delay_seconds is None or not self.is_transient_network_error(error):
                    raise
            time.sleep(delay_seconds)
        raise RuntimeError("control plane retry loop exhausted")

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
