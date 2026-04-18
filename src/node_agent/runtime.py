from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx


class VLLMRuntime:
    def __init__(self, base_url: str, *, engine: str = "vllm"):
        self.client = httpx.Client(base_url=base_url, timeout=120.0)
        self.engine = engine

    def execute(self, operation: str, model: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if operation == "embeddings":
            return self._embedding_results(model, items)
        return [self._response_result(model, item) for item in items]

    def execute_microbatch(
        self,
        operation: str,
        model: str,
        assignment_items: list[tuple[str, list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        if operation == "embeddings":
            return self._embedding_microbatch_results(model, assignment_items)
        return {
            assignment_id: self.execute(operation, model, items)
            for assignment_id, items in assignment_items
        }

    def _normalize_embedding_texts(self, item_input: Any) -> list[str]:
        if isinstance(item_input, str):
            return [item_input]
        if isinstance(item_input, list) and all(isinstance(entry, str) for entry in item_input):
            return item_input
        if isinstance(item_input, dict):
            openai_input = item_input.get("input")
            if isinstance(openai_input, str):
                return [openai_input]
            if isinstance(openai_input, list) and all(isinstance(entry, str) for entry in openai_input):
                return openai_input
            texts = item_input.get("texts")
            if isinstance(texts, list) and all(isinstance(entry, str) for entry in texts):
                return texts
            text = item_input.get("text")
            if isinstance(text, str):
                return [text]
        raise ValueError("Embeddings input must be a string, a list of strings, or an object with input/text/texts.")

    def _embedding_results(self, model: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._embedding_microbatch_results(model, [("__single_assignment__", items)])["__single_assignment__"]

    @staticmethod
    def _usage_share(usage: dict[str, Any], text_count: int, total_texts: int) -> dict[str, int]:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens) or 0)
        if total_texts <= 0:
            return {"input_tokens": prompt_tokens, "total_tokens": total_tokens}
        return {
            "input_tokens": round(prompt_tokens * text_count / total_texts),
            "total_tokens": round(total_tokens * text_count / total_texts),
        }

    def _embedding_microbatch_results(
        self,
        model: str,
        assignment_items: list[tuple[str, list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        flattened_texts: list[str] = []
        item_slices: list[tuple[str, dict[str, Any], int, int]] = []
        for assignment_id, items in assignment_items:
            for item in items:
                item_texts = self._normalize_embedding_texts(item["input"])
                start_index = len(flattened_texts)
                flattened_texts.extend(item_texts)
                item_slices.append((assignment_id, item, start_index, len(item_texts)))
        if not flattened_texts:
            raise ValueError("Embeddings microbatch must include at least one text.")
        response = self.client.post(
            "/v1/embeddings",
            json={"model": model, "input": flattened_texts},
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        data = payload.get("data")
        output_by_assignment: dict[str, list[dict[str, Any]]] = {
            assignment_id: [] for assignment_id, _items in assignment_items
        }
        for assignment_id, item, start_index, text_count in item_slices:
            usage_share = self._usage_share(usage, text_count, len(flattened_texts))
            item_payload = dict(payload)
            if isinstance(data, list):
                item_payload["data"] = data[start_index : start_index + text_count]
            item_payload["usage"] = {
                "prompt_tokens": usage_share["input_tokens"],
                "total_tokens": usage_share["total_tokens"],
            }
            output_by_assignment.setdefault(assignment_id, []).append(
                {
                    "batch_item_id": item["batch_item_id"],
                    "customer_item_id": item["customer_item_id"],
                    "provider": "autonomousc_edge",
                    "provider_model": model,
                    "status": "completed",
                    "usage": {
                        "input_tokens": usage_share["input_tokens"],
                        "input_texts": text_count,
                        "total_tokens": usage_share["total_tokens"],
                    },
                    "cost": {
                        "provider_cost": {"currency": "usd", "amount": "0.0001"},
                        "customer_charge": {"currency": "usd", "amount": "0.0002"},
                        "platform_margin": {"currency": "usd", "amount": "0.0001"},
                        "pricing_source": "provider_usage_rate_card",
                    },
                    "output": item_payload,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return output_by_assignment

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


class OpenAICompatibleRuntime(VLLMRuntime):
    pass
