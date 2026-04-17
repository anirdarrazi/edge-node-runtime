from __future__ import annotations

from typing import Any

import httpx

from .config import NodeAgentSettings


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

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        response = self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def post_content(self, path: str, payload: bytes, headers: dict[str, str]) -> Any:
        response = self.client.post(path, content=payload, headers=headers)
        response.raise_for_status()
        return response.json() if response.content else {}

    def put_content(self, url: str, payload: bytes, headers: dict[str, str]) -> Any:
        response = self.client.put(url, content=payload, headers=headers)
        response.raise_for_status()
        return response.json() if response.content else {}

    def get_content(self, url: str) -> bytes:
        response = self.client.get(url)
        response.raise_for_status()
        return response.content
