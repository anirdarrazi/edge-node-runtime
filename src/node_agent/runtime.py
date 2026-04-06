from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx


class VLLMRuntime:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url, timeout=120.0)

    def execute(self, operation: str, model: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if operation == "embeddings":
            return [self._embedding_result(model, item) for item in items]
        return [self._response_result(model, item) for item in items]

    def _embedding_result(self, model: str, item: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(
            "/v1/embeddings",
            json={"model": model, "input": item["input"]},
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        return {
            "batch_item_id": item["batch_item_id"],
            "customer_item_id": item["customer_item_id"],
            "provider": "autonomousc_edge",
            "provider_model": model,
            "status": "completed",
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "input_texts": 1,
                "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0)),
            },
            "cost": {
                "provider_cost": {"currency": "usd", "amount": "0.0001"},
                "customer_charge": {"currency": "usd", "amount": "0.0002"},
                "platform_margin": {"currency": "usd", "amount": "0.0001"},
                "pricing_source": "provider_usage_rate_card",
            },
            "output": payload,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _response_result(self, model: str, item: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": item["input"].get("messages", [{"role": "user", "content": str(item["input"])}]),
            },
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        return {
            "batch_item_id": item["batch_item_id"],
            "customer_item_id": item["customer_item_id"],
            "provider": "autonomousc_edge",
            "provider_model": model,
            "status": "completed",
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            "cost": {
                "provider_cost": {"currency": "usd", "amount": "0.0010"},
                "customer_charge": {"currency": "usd", "amount": "0.0014"},
                "platform_margin": {"currency": "usd", "amount": "0.0004"},
                "pricing_source": "provider_usage_rate_card",
            },
            "output": payload,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
