from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import httpx

from .vast_smoke import (
    DEFAULT_DURABLE_MAX_BATCH_ITEMS,
    DEFAULT_DURABLE_MAX_BATCH_TOKENS,
    DEFAULT_DURABLE_MAX_CONCURRENT_ASSIGNMENTS,
    DEFAULT_DURABLE_MAX_CONCURRENT_CHUNKS,
    DEFAULT_DURABLE_MAX_LOCAL_QUEUE_ASSIGNMENTS,
    DEFAULT_DURABLE_AVAILABLE_QUEUE_ITEMS,
    DEFAULT_DURABLE_AVAILABLE_QUEUE_TOKENS,
    DEFAULT_DURABLE_MAX_QUEUED_ITEMS,
    DEFAULT_DURABLE_NODE_REGION,
    DEFAULT_DURABLE_PULL_BUNDLE_SIZE,
    DEFAULT_DURABLE_RUNTIME_PROFILE,
    DEFAULT_DURABLE_TARGET_BATCH_ITEMS,
    DEFAULT_DURABLE_TARGET_BATCH_TOKENS,
    DEFAULT_GEMMA_E4B_VLLM_EXTRA_ARGS,
    DEFAULT_MIN_CUDA_MAX_GOOD,
    DEFAULT_MIN_INET_DOWN_MBPS,
    DEFAULT_VAST_SMOKE_IMAGE,
    RuntimeProbeClient,
    VastAPI,
    VastSmokeConfig,
    VastSmokeError,
    VastSmokeRunner,
    first_nonempty,
    offer_machine_id_values,
    parse_identity_list,
    redact_sensitive_payload,
    vast_account_launch_blocker_for_api,
)


DEFAULT_BATCHROUTER_BASE_URL = "https://api.batchrouter.com"
DEFAULT_EDGE_CONTROL_URL = "https://edge.autonomousc.com"
DEFAULT_BATCHROUTER_PROVIDER = "autonomousc"
DEFAULT_BATCHROUTER_MODEL = "gemma-4-e4b-it"
DEFAULT_VAST_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_OUTPUT_TOKENS = 64
DEFAULT_MAX_QUOTE_USD = 0.05
DEFAULT_MANIFEST_UPLOAD_THRESHOLD_ITEMS = 5_000
DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS = 900.0
DEFAULT_COMPLETION_TIMEOUT_SECONDS = 2700.0
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_VAST_NODES = 2
DEFAULT_VAST_MAX_PRICE = 0.25
DEFAULT_NODE_LAUNCH_ATTEMPTS = 3
DEFAULT_LAUNCH_TIMEOUT_SECONDS = 240.0
DEFAULT_LAUNCH_PROGRESS_GRACE_SECONDS = 120.0
DEFAULT_STARTUP_PROGRESS_STALE_SECONDS = 600.0
DEFAULT_STARTUP_MAX_VLLM_RESTARTS = 0
DEFAULT_EXECUTIONS_D1 = "autonomousc-edge-network-executions-db"
TERMINAL_BATCH_STATES = {"completed", "failed", "canceled", "expired"}


class LiveVastBatchRouterDrillError(RuntimeError):
    pass


@dataclass(frozen=True)
class DrillConfig:
    vast_api_key: str
    batchrouter_api_key: str
    operator_token: str
    edge_control_url: str = DEFAULT_EDGE_CONTROL_URL
    batchrouter_base_url: str = DEFAULT_BATCHROUTER_BASE_URL
    edge_control_cwd: Path = Path(".")
    executions_d1_database: str = DEFAULT_EXECUTIONS_D1
    artifact_dir: Path = Path("test artifacts/live-vast-batchrouter-drill")
    launch_nodes: int = DEFAULT_VAST_NODES
    batch_size: int = DEFAULT_BATCH_SIZE
    provider: str = DEFAULT_BATCHROUTER_PROVIDER
    batchrouter_model: str = DEFAULT_BATCHROUTER_MODEL
    vast_model: str = DEFAULT_VAST_MODEL
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_quote_usd: float = DEFAULT_MAX_QUOTE_USD
    manifest_upload_threshold_items: int = DEFAULT_MANIFEST_UPLOAD_THRESHOLD_ITEMS
    max_price: float = DEFAULT_VAST_MAX_PRICE
    poll_seconds: float = DEFAULT_POLL_SECONDS
    assignment_timeout_seconds: float = DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS
    completion_timeout_seconds: float = DEFAULT_COMPLETION_TIMEOUT_SECONDS
    require_accepted_assignment: bool = True
    destroy_node_after_assignment: bool = True
    node_region: str = DEFAULT_DURABLE_NODE_REGION
    runtime_profile: str = DEFAULT_DURABLE_RUNTIME_PROFILE
    image: str = DEFAULT_VAST_SMOKE_IMAGE
    hf_token: str | None = None
    min_cuda_max_good: float | None = DEFAULT_MIN_CUDA_MAX_GOOD
    min_reliability: float = 0.95
    min_inet_down_mbps: float = DEFAULT_MIN_INET_DOWN_MBPS
    preferred_offer_ids: tuple[int, ...] = ()
    exclude_offer_ids: tuple[int, ...] = ()
    exclude_machine_ids: tuple[str, ...] = ()
    launch_attempts_per_node: int = DEFAULT_NODE_LAUNCH_ATTEMPTS
    launch_timeout_seconds: float = DEFAULT_LAUNCH_TIMEOUT_SECONDS
    launch_progress_grace_seconds: float = DEFAULT_LAUNCH_PROGRESS_GRACE_SECONDS
    disk_gb: int = 80
    max_context_tokens: int = 32768
    max_batch_tokens: int = DEFAULT_DURABLE_MAX_BATCH_TOKENS
    target_batch_items: int = DEFAULT_DURABLE_TARGET_BATCH_ITEMS
    max_batch_items: int = DEFAULT_DURABLE_MAX_BATCH_ITEMS
    target_batch_tokens: int = DEFAULT_DURABLE_TARGET_BATCH_TOKENS
    max_concurrent_chunks: int = DEFAULT_DURABLE_MAX_CONCURRENT_CHUNKS
    max_concurrent_assignments: int = DEFAULT_DURABLE_MAX_CONCURRENT_ASSIGNMENTS
    max_local_queue_assignments: int = DEFAULT_DURABLE_MAX_LOCAL_QUEUE_ASSIGNMENTS
    available_queue_items: int = DEFAULT_DURABLE_AVAILABLE_QUEUE_ITEMS
    available_queue_tokens: int = DEFAULT_DURABLE_AVAILABLE_QUEUE_TOKENS
    max_queued_items: int = DEFAULT_DURABLE_MAX_QUEUED_ITEMS
    pull_bundle_size: int = DEFAULT_DURABLE_PULL_BUNDLE_SIZE
    vllm_startup_timeout_seconds: int = 1800
    startup_progress_stale_seconds: float = DEFAULT_STARTUP_PROGRESS_STALE_SECONDS
    startup_max_vllm_restarts: int = DEFAULT_STARTUP_MAX_VLLM_RESTARTS
    wrangler_timeout_seconds: float = 60.0
    request_timeout_seconds: float = 90.0

    def __post_init__(self) -> None:
        if self.destroy_node_after_assignment and self.launch_nodes < 2:
            raise LiveVastBatchRouterDrillError("At least two Vast nodes are required for the failure drill.")
        if self.launch_nodes < 1:
            raise LiveVastBatchRouterDrillError("At least one Vast node is required.")
        if self.batch_size < 1:
            raise LiveVastBatchRouterDrillError("batch_size must be positive.")
        if self.max_output_tokens < 1:
            raise LiveVastBatchRouterDrillError("max_output_tokens must be positive.")
        if self.manifest_upload_threshold_items < 0:
            raise LiveVastBatchRouterDrillError("manifest_upload_threshold_items cannot be negative.")
        if self.poll_seconds <= 0:
            raise LiveVastBatchRouterDrillError("poll_seconds must be positive.")
        if not str(self.vast_api_key or "").strip():
            raise LiveVastBatchRouterDrillError("VAST_API_KEY is required.")
        if not str(self.batchrouter_api_key or "").strip():
            raise LiveVastBatchRouterDrillError(
                "A BatchRouter API key is required. Set BATCHROUTER_SMOKE_API_KEY or BATCHROUTER_API_KEY."
            )
        if not str(self.operator_token or "").strip():
            raise LiveVastBatchRouterDrillError("AUTONOMOUSC_OPERATOR_API_KEY is required.")
        if self.launch_attempts_per_node < 1:
            raise LiveVastBatchRouterDrillError("launch_attempts_per_node must be positive.")


@dataclass
class DrillNode:
    node_id: str
    node_key: str
    launch_label: str | None = None
    instance_id: int | None = None
    machine_ids: tuple[str, ...] = ()
    launch_report: dict[str, Any] | None = None
    destroyed: bool = False
    revoked: bool = False
    destroy_error: str | None = None
    revoke_error: str | None = None


@dataclass(frozen=True)
class D1QueryResult:
    rows: list[dict[str, Any]]
    rows_read: int = 0
    raw: Any = None


@dataclass
class DrillState:
    run_id: str
    started_at: str
    nodes: list[DrillNode] = field(default_factory=list)
    batch_id: str | None = None
    quote_id: str | None = None
    input_file_id: str | None = None
    input_manifest_path: str | None = None
    input_manifest_bytes: int | None = None
    victim_node_id: str | None = None
    victim_instance_id: int | None = None
    assignment_distribution_before_failure: list[dict[str, Any]] = field(default_factory=list)
    assignment_distribution_after_failure: list[dict[str, Any]] = field(default_factory=list)
    execution_distribution: list[dict[str, Any]] = field(default_factory=list)
    d1_rows_read: int = 0
    batch: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    billing_receipt: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


class BatchRouterClient:
    def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 90.0) -> None:
        self._client = httpx.Client(
            base_url=str(base_url).strip().rstrip("/"),
            timeout=timeout_seconds,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        response = self._client.request(method, path, json=json_body, headers=dict(headers or {}))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = response.text[:1200]
            raise LiveVastBatchRouterDrillError(
                f"BatchRouter {method} {path} returned HTTP {response.status_code}: {detail}"
            ) from error
        payload = response.json()
        if not isinstance(payload, dict):
            raise LiveVastBatchRouterDrillError(f"BatchRouter {method} {path} returned a non-object payload.")
        return payload

    def quote_batch(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", "/v1/batches/quote", json_body=manifest)

    def preflight(self) -> dict[str, Any]:
        return self._request_json("GET", "/v1/route-policies")

    def create_batch(self, manifest: Mapping[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            "/v1/batches",
            json_body=manifest,
            headers={"idempotency-key": idempotency_key},
        )

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/v1/batches/{batch_id}?include_billing_receipt=true")

    def get_results(self, batch_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/v1/batches/{batch_id}/results")

    def get_billing_receipt(self, batch_id: str) -> dict[str, Any]:
        return self._request_json("GET", f"/v1/batches/{batch_id}/billing-receipt")

    def upload_input_manifest(
        self,
        manifest_path: Path,
        *,
        item_count: int,
        sha256: str,
        size_bytes: int,
    ) -> dict[str, Any]:
        headers = {
            "content-type": "application/x-ndjson",
            "content-length": str(size_bytes),
            "x-batchrouter-item-count": str(item_count),
            "x-content-sha256": sha256,
        }
        with manifest_path.open("rb") as handle:
            response = self._client.post(
                "/v1/batches/input-manifests",
                content=handle,
                headers=headers,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = response.text[:1200]
            raise LiveVastBatchRouterDrillError(
                f"BatchRouter POST /v1/batches/input-manifests returned HTTP "
                f"{response.status_code}: {detail}"
            ) from error
        payload = response.json()
        if not isinstance(payload, dict):
            raise LiveVastBatchRouterDrillError(
                "BatchRouter POST /v1/batches/input-manifests returned a non-object payload."
            )
        return payload


class EdgeControlAPI:
    def __init__(self, base_url: str, operator_token: str, *, timeout_seconds: float = 60.0) -> None:
        self._operator_token = operator_token
        self._client = httpx.Client(
            base_url=str(base_url).strip().rstrip("/"),
            timeout=timeout_seconds,
            headers={"content-type": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        bearer: bool = True,
    ) -> dict[str, Any]:
        headers = {"authorization": f"Bearer {self._operator_token}"} if bearer else {}
        response = self._client.request(method, path, json=json_body, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = response.text[:1200]
            raise LiveVastBatchRouterDrillError(
                f"edge-control {method} {path} returned HTTP {response.status_code}: {detail}"
            ) from error
        payload = response.json()
        if not isinstance(payload, dict):
            raise LiveVastBatchRouterDrillError(f"edge-control {method} {path} returned a non-object payload.")
        return payload

    def enroll_node(self, config: DrillConfig, *, label: str) -> DrillNode:
        payload = build_node_enroll_payload(config, label=label)
        try:
            result = self._request_json("POST", "/nodes/enroll", json_body=payload, bearer=True)
        except LiveVastBatchRouterDrillError:
            body_token_payload = {"operator_token": self._operator_token, **payload}
            result = self._request_json("POST", "/nodes/enroll", json_body=body_token_payload, bearer=False)
        node_id = str(result.get("node_id") or "").strip()
        node_key = str(result.get("node_key") or "").strip()
        if not node_id or not node_key:
            raise LiveVastBatchRouterDrillError("/nodes/enroll did not return node credentials.")
        if result.get("approved") is False:
            raise LiveVastBatchRouterDrillError(
                f"Enrolled node {node_id} is not approved. Use an AUTONOMOUSc operator API key for this drill."
            )
        return DrillNode(node_id=node_id, node_key=node_key, launch_label=label)

    def revoke_node(self, node_id: str) -> dict[str, Any]:
        return self._request_json("POST", f"/operator/nodes/{node_id}/revoke", json_body={})


class WranglerD1Client:
    def __init__(
        self,
        *,
        cwd: Path,
        database: str,
        timeout_seconds: float = 60.0,
        platform_name: str | None = None,
    ) -> None:
        self.cwd = Path(cwd)
        self.database = database
        self.timeout_seconds = timeout_seconds
        self.platform_name = platform_name or os.name

    def query(self, sql: str) -> D1QueryResult:
        normalized_sql = " ".join(str(sql or "").strip().split())
        if normalized_sql and not normalized_sql.endswith(";"):
            normalized_sql = f"{normalized_sql};"
        args = [
            "npx",
            "wrangler",
            "d1",
            "execute",
            self.database,
            "--remote",
            "--json",
            "--command",
            normalized_sql,
        ]
        command = command_for_platform(args, platform_name=self.platform_name)
        completed = subprocess.run(
            command,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise LiveVastBatchRouterDrillError(
                f"wrangler d1 query failed with exit code {completed.returncode}: {stderr[:1200]}"
            )
        raw = parse_json_output(completed.stdout)
        rows, rows_read = extract_d1_rows(raw)
        return D1QueryResult(rows=rows, rows_read=rows_read, raw=raw)

    def assignment_summary(self, batch_id: str) -> D1QueryResult:
        return self.query(
            """
            SELECT
              aa.node_id AS node_id,
              aa.status AS status,
              COUNT(*) AS attempts,
              COALESCE(SUM(pe.item_count), 0) AS items,
              MIN(aa.created_at) AS first_created_at,
              MAX(aa.updated_at) AS last_updated_at
            FROM assignment_attempts aa
            INNER JOIN provider_executions pe ON pe.id = aa.execution_id
            WHERE pe.source_batch_id = {batch_id}
            GROUP BY aa.node_id, aa.status
            ORDER BY aa.node_id ASC, aa.status ASC
            """.format(batch_id=sql_quote(batch_id))
        )

    def execution_summary(self, batch_id: str) -> D1QueryResult:
        return self.query(
            """
            SELECT
              selected_node_id AS node_id,
              status,
              COUNT(*) AS executions,
              COALESCE(SUM(item_count), 0) AS items,
              COALESCE(SUM(retry_count), 0) AS retries
            FROM provider_executions
            WHERE source_batch_id = {batch_id}
            GROUP BY selected_node_id, status
            ORDER BY selected_node_id ASC, status ASC
            """.format(batch_id=sql_quote(batch_id))
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def command_for_platform(args: list[str], *, platform_name: str | None = None) -> list[str]:
    normalized_platform = platform_name or os.name
    if normalized_platform == "nt":
        if args and args[0].lower() == "npx":
            return ["npx.cmd", *args[1:]]
    return args


def parse_json_output(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise LiveVastBatchRouterDrillError("wrangler returned empty JSON output.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    starts = [index for index, char in enumerate(raw) if char in "[{"]
    for start in starts:
        try:
            return json.loads(raw[start:])
        except json.JSONDecodeError:
            continue
    raise LiveVastBatchRouterDrillError(f"Could not parse wrangler JSON output: {raw[:500]}")


def extract_d1_rows(raw: Any) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    rows_read = 0
    payloads = raw if isinstance(raw, list) else [raw]
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        result_rows = payload.get("results")
        if isinstance(result_rows, list):
            rows.extend(row for row in result_rows if isinstance(row, dict))
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            try:
                rows_read += int(meta.get("rows_read") or 0)
            except (TypeError, ValueError):
                pass
    return rows, rows_read


def numeric_amount(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def build_node_enroll_payload(config: DrillConfig, *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "region": config.node_region,
        "trust_tier": "standard",
        "restricted_capable": False,
        "capabilities": {
            "supported_models": [config.vast_model],
            "operations": ["responses"],
            "gpu_name": "RTX 5060 Ti",
            "gpu_memory_gb": 16,
            "max_context_tokens": config.max_context_tokens,
            "max_batch_tokens": config.max_batch_tokens,
            "target_batch_tokens": config.target_batch_tokens,
            "target_batch_items": config.target_batch_items,
            "max_batch_items": config.max_batch_items,
            "recommended_batch_items": config.target_batch_items,
            "max_concurrent_assignments": config.max_concurrent_assignments,
            "max_concurrent_chunks": config.max_concurrent_chunks,
            "max_local_queue_assignments": config.max_local_queue_assignments,
            "max_pull_bundle_assignments": max(config.max_local_queue_assignments, config.pull_bundle_size),
            "available_queue_items": config.available_queue_items,
            "available_queue_tokens": config.available_queue_tokens,
            "max_queued_items": config.max_queued_items,
            "capacity_status": "active",
            "heartbeat_ttl_seconds": 120,
            "batchrouter_capacity_tier": "edge",
            "target_gpu_utilization_pct": 100,
            "min_gpu_memory_headroom_pct": 5,
            "thermal_headroom": 0.95,
            "heat_demand": "none",
            "heat_governor_mode": "100",
        },
        "runtime": {
            "agent_version": "live-vast-batchrouter-drill",
            "runtime_profile": config.runtime_profile,
            "inference_engine": "vllm",
            "deployment_target": "vast_ai",
            "model_format": "safetensors",
            "runtime_image": config.image,
            "readiness_path": "/v1/models",
            "supported_apis": ["responses"],
            "capacity_class": "elastic_burst",
            "temporary_node": False,
            "burst_provider": "vast_ai",
            "inference_base_url": "http://127.0.0.1:8000",
            "vllm_base_url": "http://127.0.0.1:8000",
            "docker_image": config.image,
            "current_model": config.vast_model,
            "effective_context_tokens": config.max_context_tokens,
        },
    }


def build_batch_item(config: DrillConfig, *, run_id: str, index: int) -> dict[str, Any]:
    return {
        "customer_item_id": f"{run_id}-{index:06d}",
        "operation": "responses",
        "model": config.batchrouter_model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Return one short sentence that includes the item number "
                        f"{index} and the word ready."
                    ),
                }
            ],
            "max_output_tokens": config.max_output_tokens,
        },
        "metadata": {
            "live_vast_batchrouter_drill": True,
            "run_id": run_id,
            "item_index": index,
        },
    }


def build_batch_base_payload(config: DrillConfig, *, run_id: str) -> dict[str, Any]:
    return {
        "sla_tier": "standard",
        "routing_mode": "cheapest",
        "privacy_tier": "standard",
        "allowed_regions": ["global"],
        "provider_preferences": {
            "only": [config.provider],
            "allow_fallbacks": False,
            "data_collection": "deny",
            "zdr": True,
        },
        "metadata": {
            "live_vast_batchrouter_drill": True,
            "run_id": run_id,
            "failure_mode": (
                "destroy_node_after_assignment"
                if config.destroy_node_after_assignment
                else "none"
            ),
            "provider": config.provider,
        },
        "max_price": f"{config.max_quote_usd:.4f}",
    }


def build_batch_manifest(config: DrillConfig, *, run_id: str) -> dict[str, Any]:
    return {
        **build_batch_base_payload(config, run_id=run_id),
        "items": [build_batch_item(config, run_id=run_id, index=index) for index in range(config.batch_size)],
    }


def build_manifest_file_batch_payload(
    config: DrillConfig,
    *,
    run_id: str,
    input_file_id: str,
    item_count: int,
) -> dict[str, Any]:
    return {
        **build_batch_base_payload(config, run_id=run_id),
        "input_file_id": input_file_id,
        "input_item_count": item_count,
    }


def write_batch_manifest_jsonl(config: DrillConfig, *, run_id: str) -> dict[str, Any]:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = config.artifact_dir / "batchrouter-input-manifest.jsonl"
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("wb") as handle:
        for index in range(config.batch_size):
            line = json.dumps(
                build_batch_item(config, run_id=run_id, index=index),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            handle.write(line)
            digest.update(line)
            size_bytes += len(line)
    return {
        "path": path,
        "item_count": config.batch_size,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def machine_ids_from_offer_report(offer: Mapping[str, Any]) -> tuple[str, ...]:
    machine_ids = set(offer_machine_id_values(offer))
    reported = offer.get("machine_ids")
    if isinstance(reported, list):
        for value in reported:
            normalized = str(value or "").strip()
            if normalized:
                machine_ids.add(normalized)
    return tuple(sorted(machine_ids))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(redact_sensitive_payload(payload), indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def resource_checkpoint_payload(state: DrillState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "updated_at": now_iso(),
        "batch_id": state.batch_id,
        "quote_id": state.quote_id,
        "input_file_id": state.input_file_id,
        "input_manifest_path": state.input_manifest_path,
        "input_manifest_bytes": state.input_manifest_bytes,
        "nodes": [
            {
                "node_id": node.node_id,
                "launch_label": node.launch_label,
                "instance_id": node.instance_id,
                "machine_ids": list(node.machine_ids),
                "destroyed": node.destroyed,
                "revoked": node.revoked,
                "destroy_error": node.destroy_error,
                "revoke_error": node.revoke_error,
            }
            for node in state.nodes
        ],
    }


def write_resource_checkpoint(config: DrillConfig, state: DrillState) -> None:
    write_json(config.artifact_dir / "resource-checkpoint.json", resource_checkpoint_payload(state))


def append_event(config: DrillConfig, event: str, **fields: Any) -> None:
    record = {
        "at": now_iso(),
        "event": event,
        **dict(redact_sensitive_payload(fields)),
    }
    line = json.dumps(record, sort_keys=True, default=json_default)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    with (config.artifact_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")
    print(line, flush=True)


def config_report(config: DrillConfig) -> dict[str, Any]:
    secret_keys = {"vast_api_key", "batchrouter_api_key", "operator_token", "hf_token"}
    return {
        key: ("***REDACTED***" if key in secret_keys and value else value)
        for key, value in config.__dict__.items()
    }


def choose_victim_node(
    rows: Iterable[Mapping[str, Any]],
    nodes: Iterable[DrillNode],
    *,
    require_accepted_assignment: bool,
) -> DrillNode | None:
    node_map = {node.node_id: node for node in nodes}
    preferred_statuses = ("accepted",) if require_accepted_assignment else ("accepted", "assigned")
    for status in preferred_statuses:
        for row in rows:
            if str(row.get("status") or "").strip() != status:
                continue
            node_id = str(row.get("node_id") or "").strip()
            try:
                attempts = int(row.get("attempts") or 0)
            except (TypeError, ValueError):
                attempts = 0
            if attempts > 0 and node_id in node_map:
                return node_map[node_id]
    return None


def extract_quote_id(quote_payload: Mapping[str, Any]) -> str:
    quote_id = str(quote_payload.get("quote_id") or "").strip()
    if quote_id:
        return quote_id
    quote_lock = quote_payload.get("quote_lock")
    if isinstance(quote_lock, Mapping):
        quote_id = str(quote_lock.get("quote_id") or "").strip()
    if quote_id:
        return quote_id
    raise LiveVastBatchRouterDrillError("BatchRouter quote response did not include quote_id.")


def ensure_quote_under_cap(quote_payload: Mapping[str, Any], *, max_quote_usd: float) -> None:
    pricing = quote_payload.get("pricing_estimate")
    total = numeric_amount(pricing.get("total") if isinstance(pricing, Mapping) else None)
    if total is None:
        return
    if total > max_quote_usd + 1e-12:
        raise LiveVastBatchRouterDrillError(
            f"BatchRouter quote ${total:.8f} exceeds safety cap ${max_quote_usd:.8f}."
        )


def extract_batch_id(create_payload: Mapping[str, Any]) -> str:
    batch = create_payload.get("batch")
    if isinstance(batch, Mapping):
        batch_id = str(batch.get("id") or "").strip()
        if batch_id:
            return batch_id
    raise LiveVastBatchRouterDrillError("BatchRouter create response did not include batch.id.")


def batch_counts(batch_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    batch = batch_payload.get("batch")
    if not isinstance(batch, Mapping):
        return {}
    counts = batch.get("counts")
    return counts if isinstance(counts, Mapping) else {}


def batch_state(batch_payload: Mapping[str, Any]) -> str:
    batch = batch_payload.get("batch")
    if not isinstance(batch, Mapping):
        return ""
    return str(batch.get("state") or "").strip()


def launch_vast_nodes(config: DrillConfig, state: DrillState, edge: EdgeControlAPI) -> None:
    machine_exclusions: set[str] = set()
    api = VastAPI(config.vast_api_key)
    runtime = RuntimeProbeClient()
    try:
        runner = VastSmokeRunner(api, runtime)
        for index in range(config.launch_nodes):
            account_blocker = vast_account_launch_blocker_for_api(api)
            if account_blocker:
                raise LiveVastBatchRouterDrillError(account_blocker)
            ready_node: DrillNode | None = None
            last_error = "Vast node launch did not run."
            for attempt in range(config.launch_attempts_per_node):
                label_suffix = f"node-{index + 1}" if attempt == 0 else f"node-{index + 1}-attempt-{attempt + 1}"
                label = f"batchrouter-drill-{state.run_id}-{label_suffix}"
                append_event(config, "node.enroll.start", node_index=index + 1, attempt=attempt + 1, label=label)
                node = edge.enroll_node(config, label=label)
                state.nodes.append(node)
                write_resource_checkpoint(config, state)
                append_event(
                    config,
                    "node.enrolled",
                    node_index=index + 1,
                    attempt=attempt + 1,
                    node_id=node.node_id,
                    label=label,
                )
                smoke_config = VastSmokeConfig(
                    api_key=config.vast_api_key,
                    model=config.vast_model,
                    label=label,
                    max_price=config.max_price,
                    image=config.image,
                    disk_gb=config.disk_gb,
                    min_vram_gb=15,
                    min_cuda_max_good=config.min_cuda_max_good,
                    min_reliability=config.min_reliability,
                    min_inet_down_mbps=config.min_inet_down_mbps,
                    preferred_offer_id=(
                        int(config.preferred_offer_ids[index])
                        if index < len(config.preferred_offer_ids) and attempt == 0
                        else None
                    ),
                    exclude_offer_ids=tuple(config.exclude_offer_ids),
                    exclude_machine_ids=tuple(sorted(set(config.exclude_machine_ids) | machine_exclusions)),
                    launch_timeout_seconds=config.launch_timeout_seconds,
                    launch_progress_grace_seconds=config.launch_progress_grace_seconds,
                    readiness_timeout_seconds=1200,
                    startup_status_connect_timeout_seconds=240,
                    startup_progress_stale_seconds=config.startup_progress_stale_seconds,
                    startup_max_vllm_restarts=config.startup_max_vllm_restarts,
                    poll_interval_seconds=10,
                    api_kind="responses",
                    max_context_tokens=config.max_context_tokens,
                    hf_token=config.hf_token,
                    benchmark_requests=0,
                    vllm_extra_args=DEFAULT_GEMMA_E4B_VLLM_EXTRA_ARGS,
                    durable_node=True,
                    edge_control_url=config.edge_control_url,
                    node_id=node.node_id,
                    node_key=node.node_key,
                    operator_token=config.operator_token,
                    node_region=config.node_region,
                    runtime_profile=config.runtime_profile,
                    max_batch_tokens=config.max_batch_tokens,
                    target_batch_items=config.target_batch_items,
                    max_batch_items=config.max_batch_items,
                    target_batch_tokens=config.target_batch_tokens,
                    max_concurrent_chunks=config.max_concurrent_chunks,
                    max_concurrent_assignments=config.max_concurrent_assignments,
                    max_local_queue_assignments=config.max_local_queue_assignments,
                    pull_bundle_size=config.pull_bundle_size,
                    vllm_startup_timeout_seconds=config.vllm_startup_timeout_seconds,
                )
                append_event(
                    config,
                    "vast.launch.start",
                    node_index=index + 1,
                    attempt=attempt + 1,
                    node_id=node.node_id,
                    excluded_machine_ids=sorted(set(config.exclude_machine_ids) | machine_exclusions),
                )
                try:
                    report = runner.run(smoke_config)
                except Exception as error:
                    last_error = str(error) or error.__class__.__name__
                    append_event(
                        config,
                        "vast.launch.error",
                        node_index=index + 1,
                        attempt=attempt + 1,
                        node_id=node.node_id,
                        error=last_error,
                    )
                    try:
                        edge.revoke_node(node.node_id)
                        node.revoked = True
                        append_event(config, "node.revoke.after_launch_error", node_id=node.node_id)
                    except Exception as revoke_error:  # pragma: no cover - live cleanup protection
                        node.revoke_error = str(revoke_error) or revoke_error.__class__.__name__
                        append_event(config, "node.revoke.after_launch_error.failed", node_id=node.node_id, error=node.revoke_error)
                    write_resource_checkpoint(config, state)
                    continue
                node.launch_report = report
                write_json(config.artifact_dir / f"vast-node-{index + 1}-attempt-{attempt + 1}.json", report)
                if attempt == 0 or report.get("status") == "ok":
                    write_json(config.artifact_dir / f"vast-node-{index + 1}.json", report)
                instance = report.get("instance")
                if isinstance(instance, Mapping):
                    node.instance_id = int(instance.get("id") or 0) or None
                cleanup = report.get("cleanup")
                if isinstance(cleanup, Mapping) and cleanup.get("destroyed"):
                    node.destroyed = True
                selected_offer = report.get("selected_offer")
                if isinstance(selected_offer, Mapping):
                    node.machine_ids = machine_ids_from_offer_report(selected_offer)
                    machine_exclusions.update(node.machine_ids)
                write_resource_checkpoint(config, state)
                if report.get("status") != "ok":
                    last_error = str(report.get("error") or report.get("status") or "unknown launch error")
                    append_event(
                        config,
                        "vast.launch.unhealthy",
                        node_index=index + 1,
                        attempt=attempt + 1,
                        node_id=node.node_id,
                        instance_id=node.instance_id,
                        machine_ids=list(node.machine_ids),
                        status=report.get("status"),
                        error=report.get("error"),
                    )
                    try:
                        edge.revoke_node(node.node_id)
                        node.revoked = True
                        append_event(config, "node.revoke.after_unhealthy_launch", node_id=node.node_id)
                    except Exception as revoke_error:  # pragma: no cover - live cleanup protection
                        node.revoke_error = str(revoke_error) or revoke_error.__class__.__name__
                        append_event(config, "node.revoke.after_unhealthy_launch.failed", node_id=node.node_id, error=node.revoke_error)
                    write_resource_checkpoint(config, state)
                    continue
                append_event(
                    config,
                    "vast.launch.ready",
                    node_index=index + 1,
                    attempt=attempt + 1,
                    node_id=node.node_id,
                    instance_id=node.instance_id,
                    machine_ids=list(node.machine_ids),
                    timings=report.get("timings") if isinstance(report.get("timings"), Mapping) else None,
                )
                ready_node = node
                break
            if ready_node is None:
                raise LiveVastBatchRouterDrillError(
                    f"Vast node {index + 1} failed to launch after "
                    f"{config.launch_attempts_per_node} attempts: {last_error}"
                )
    finally:
        runtime.close()
        api.close()


def wait_for_assignment(
    *,
    config: DrillConfig,
    d1: WranglerD1Client,
    state: DrillState,
    batch_id: str,
) -> DrillNode:
    deadline = time.monotonic() + config.assignment_timeout_seconds
    while time.monotonic() < deadline:
        summary = d1.assignment_summary(batch_id)
        state.d1_rows_read += summary.rows_read
        state.assignment_distribution_before_failure = summary.rows
        write_json(config.artifact_dir / "assignment-summary-before-failure.json", summary.raw)
        victim = choose_victim_node(
            summary.rows,
            state.nodes,
            require_accepted_assignment=config.require_accepted_assignment,
        )
        if victim:
            return victim
        time.sleep(config.poll_seconds)
    expected = "accepted" if config.require_accepted_assignment else "assigned or accepted"
    raise LiveVastBatchRouterDrillError(
        f"Timed out waiting for a {expected} assignment for batch {batch_id}."
    )


def wait_for_batch_terminal(
    *,
    config: DrillConfig,
    batchrouter: BatchRouterClient,
    state: DrillState,
    batch_id: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.completion_timeout_seconds
    polls: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            payload = batchrouter.get_batch(batch_id)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            poll_error = str(error) or error.__class__.__name__
            polls.append({"at": now_iso(), "error": poll_error})
            write_json(config.artifact_dir / "batch-polls.json", polls)
            append_event(config, "batchrouter.batch.poll.retry", batch_id=batch_id, error=poll_error)
            time.sleep(config.poll_seconds)
            continue
        state.batch = payload
        current_state = batch_state(payload)
        counts = dict(batch_counts(payload))
        polls.append({"at": now_iso(), "state": current_state, "counts": counts})
        if current_state in TERMINAL_BATCH_STATES:
            write_json(config.artifact_dir / "batch-polls.json", polls)
            return payload
        time.sleep(config.poll_seconds)
    write_json(config.artifact_dir / "batch-polls.json", polls)
    raise LiveVastBatchRouterDrillError(f"Timed out waiting for BatchRouter batch {batch_id} to finish.")


def destroy_node_instance(config: DrillConfig, node: DrillNode) -> None:
    if node.instance_id is None:
        raise LiveVastBatchRouterDrillError(f"Node {node.node_id} has no Vast instance id to destroy.")
    api = VastAPI(config.vast_api_key)
    try:
        api.destroy_instance(node.instance_id)
        node.destroyed = True
        append_event(config, "vast.instance.destroyed", node_id=node.node_id, instance_id=node.instance_id)
    finally:
        api.close()


def active_vast_instance_ids_for_label(api: VastAPI, label: str | None) -> list[int]:
    normalized_label = str(label or "").strip()
    if not normalized_label:
        return []
    response = api._request_with_retries("get", "/instances/")
    response.raise_for_status()
    body = response.json()
    instances = body.get("instances") if isinstance(body, Mapping) else body
    matched: list[int] = []
    for instance in instances or []:
        if not isinstance(instance, Mapping):
            continue
        instance_label = str(instance.get("label") or instance.get("name") or "").strip()
        if instance_label != normalized_label:
            continue
        try:
            instance_id = int(instance.get("id") or 0)
        except (TypeError, ValueError):
            instance_id = 0
        if instance_id > 0:
            matched.append(instance_id)
    return matched


def cleanup_resources(config: DrillConfig, state: DrillState, edge: EdgeControlAPI) -> dict[str, Any]:
    cleanup: dict[str, Any] = {"destroyed_instances": [], "revoked_nodes": [], "errors": []}
    api = VastAPI(config.vast_api_key)
    try:
        for node in state.nodes:
            instance_ids = [node.instance_id] if node.instance_id is not None else []
            if node.instance_id is None and not node.destroyed:
                try:
                    instance_ids.extend(active_vast_instance_ids_for_label(api, node.launch_label))
                except Exception as error:  # pragma: no cover - live cleanup protection
                    cleanup["errors"].append(
                        {
                            "node_id": node.node_id,
                            "launch_label": node.launch_label,
                            "error": str(error) or error.__class__.__name__,
                        }
                    )
            for instance_id in dict.fromkeys(instance_ids):
                if instance_id is None or node.destroyed:
                    continue
                try:
                    append_event(config, "cleanup.destroy.start", node_id=node.node_id, instance_id=instance_id)
                    api.destroy_instance(instance_id)
                    node.instance_id = int(instance_id)
                    node.destroyed = True
                    cleanup["destroyed_instances"].append(instance_id)
                    append_event(config, "cleanup.destroy.ok", node_id=node.node_id, instance_id=instance_id)
                except Exception as error:  # pragma: no cover - live cleanup protection
                    node.destroy_error = str(error) or error.__class__.__name__
                    cleanup["errors"].append({"instance_id": instance_id, "error": node.destroy_error})
                    append_event(
                        config,
                        "cleanup.destroy.error",
                        node_id=node.node_id,
                        instance_id=instance_id,
                        error=node.destroy_error,
                    )
            if not node.revoked:
                try:
                    append_event(config, "cleanup.revoke.start", node_id=node.node_id)
                    edge.revoke_node(node.node_id)
                    node.revoked = True
                    cleanup["revoked_nodes"].append(node.node_id)
                    append_event(config, "cleanup.revoke.ok", node_id=node.node_id)
                except Exception as error:  # pragma: no cover - live cleanup protection
                    node.revoke_error = str(error) or error.__class__.__name__
                    cleanup["errors"].append({"node_id": node.node_id, "error": node.revoke_error})
                    append_event(config, "cleanup.revoke.error", node_id=node.node_id, error=node.revoke_error)
            write_resource_checkpoint(config, state)
    finally:
        api.close()
    return cleanup


def run_drill(config: DrillConfig) -> dict[str, Any]:
    run_id = f"vast_drill_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    state = DrillState(run_id=run_id, started_at=now_iso())
    edge = EdgeControlAPI(config.edge_control_url, config.operator_token, timeout_seconds=config.request_timeout_seconds)
    batchrouter = BatchRouterClient(
        config.batchrouter_base_url,
        config.batchrouter_api_key,
        timeout_seconds=config.request_timeout_seconds,
    )
    d1 = WranglerD1Client(
        cwd=config.edge_control_cwd,
        database=config.executions_d1_database,
        timeout_seconds=config.wrangler_timeout_seconds,
    )
    status = "error"
    cleanup: dict[str, Any] = {}
    error_message: str | None = None
    try:
        write_json(config.artifact_dir / "drill-config.redacted.json", {"run_id": run_id, "config": config_report(config)})
        append_event(config, "drill.started", run_id=run_id, batch_size=config.batch_size, launch_nodes=config.launch_nodes)
        append_event(config, "batchrouter.auth.preflight.start", base_url=config.batchrouter_base_url)
        batchrouter.preflight()
        append_event(config, "batchrouter.auth.preflight.ok")
        launch_vast_nodes(config, state, edge)
        ready_nodes = [
            node
            for node in state.nodes
            if isinstance(node.launch_report, Mapping) and node.launch_report.get("status") == "ok"
        ]
        append_event(config, "vast.nodes.ready", node_count=len(ready_nodes), launch_attempts=len(state.nodes))
        use_manifest_file = config.batch_size > config.manifest_upload_threshold_items
        if use_manifest_file:
            append_event(
                config,
                "batchrouter.input_manifest.write.start",
                item_count=config.batch_size,
            )
            input_manifest = write_batch_manifest_jsonl(config, run_id=run_id)
            state.input_manifest_path = str(input_manifest["path"])
            state.input_manifest_bytes = int(input_manifest["size_bytes"])
            append_event(
                config,
                "batchrouter.input_manifest.write.ok",
                item_count=input_manifest["item_count"],
                size_bytes=input_manifest["size_bytes"],
            )
            append_event(
                config,
                "batchrouter.input_manifest.upload.start",
                item_count=input_manifest["item_count"],
                size_bytes=input_manifest["size_bytes"],
            )
            upload = batchrouter.upload_input_manifest(
                input_manifest["path"],
                item_count=int(input_manifest["item_count"]),
                sha256=str(input_manifest["sha256"]),
                size_bytes=int(input_manifest["size_bytes"]),
            )
            write_json(config.artifact_dir / "batchrouter-input-file.json", upload)
            state.input_file_id = str(upload.get("input_file_id") or "").strip()
            if not state.input_file_id:
                raise LiveVastBatchRouterDrillError("BatchRouter input manifest upload did not return input_file_id.")
            write_resource_checkpoint(config, state)
            append_event(
                config,
                "batchrouter.input_manifest.upload.ok",
                input_file_id=state.input_file_id,
            )
            manifest = build_manifest_file_batch_payload(
                config,
                run_id=run_id,
                input_file_id=state.input_file_id,
                item_count=config.batch_size,
            )
        else:
            manifest = build_batch_manifest(config, run_id=run_id)
        append_event(
            config,
            "batchrouter.quote.start",
            item_count=config.batch_size,
            request_source="input_file" if use_manifest_file else "inline",
        )
        quote = batchrouter.quote_batch(manifest)
        ensure_quote_under_cap(quote, max_quote_usd=config.max_quote_usd)
        state.quote_id = extract_quote_id(quote)
        write_json(config.artifact_dir / "batchrouter-quote.json", quote)
        write_resource_checkpoint(config, state)
        append_event(config, "batchrouter.quote.ok", quote_id=state.quote_id)
        create_manifest = {**manifest, "quote_id": state.quote_id}
        append_event(config, "batchrouter.create.start", quote_id=state.quote_id)
        created = batchrouter.create_batch(create_manifest, idempotency_key=f"{run_id}-create")
        state.batch_id = extract_batch_id(created)
        write_json(config.artifact_dir / "batchrouter-created.json", created)
        write_resource_checkpoint(config, state)
        append_event(config, "batchrouter.create.ok", batch_id=state.batch_id)

        append_event(config, "assignment.wait.start", batch_id=state.batch_id)
        victim = wait_for_assignment(config=config, d1=d1, state=state, batch_id=state.batch_id)
        state.victim_node_id = victim.node_id
        state.victim_instance_id = victim.instance_id
        write_resource_checkpoint(config, state)
        append_event(
            config,
            "assignment.victim.selected",
            batch_id=state.batch_id,
            node_id=victim.node_id,
            instance_id=victim.instance_id,
        )
        if config.destroy_node_after_assignment:
            destroy_node_instance(config, victim)
            write_resource_checkpoint(config, state)
        else:
            append_event(
                config,
                "assignment.failure_drill.skipped",
                batch_id=state.batch_id,
                node_id=victim.node_id,
                instance_id=victim.instance_id,
            )

        final_batch = wait_for_batch_terminal(
            config=config,
            batchrouter=batchrouter,
            state=state,
            batch_id=state.batch_id,
        )
        state.batch = final_batch
        state.results = batchrouter.get_results(state.batch_id)
        state.billing_receipt = batchrouter.get_billing_receipt(state.batch_id)
        assignment_after = d1.assignment_summary(state.batch_id)
        execution_summary = d1.execution_summary(state.batch_id)
        state.d1_rows_read += assignment_after.rows_read + execution_summary.rows_read
        state.assignment_distribution_after_failure = assignment_after.rows
        state.execution_distribution = execution_summary.rows
        write_json(config.artifact_dir / "assignment-summary-after-failure.json", assignment_after.raw)
        write_json(config.artifact_dir / "execution-summary.json", execution_summary.raw)
        write_json(config.artifact_dir / "batchrouter-final-batch.json", final_batch)
        write_json(config.artifact_dir / "batchrouter-results.json", state.results)
        write_json(config.artifact_dir / "batchrouter-billing-receipt.json", state.billing_receipt)
        write_resource_checkpoint(config, state)
        append_event(
            config,
            "batchrouter.batch.terminal",
            batch_id=state.batch_id,
            state=batch_state(final_batch),
            counts=dict(batch_counts(final_batch)),
        )

        counts = batch_counts(final_batch)
        completed = int(counts.get("completed") or 0)
        failed = int(counts.get("failed") or 0)
        canceled = int(counts.get("canceled") or 0)
        total = int(counts.get("total") or 0)
        if batch_state(final_batch) != "completed" or completed != total or failed or canceled:
            raise LiveVastBatchRouterDrillError(
                f"Batch finished unhealthy: state={batch_state(final_batch)} counts={dict(counts)}"
            )
        status = "ok"
    except Exception as error:
        error_message = str(error) or error.__class__.__name__
        append_event(config, "drill.error", error=error_message)
    finally:
        cleanup = cleanup_resources(config, state, edge)
        append_event(config, "cleanup.finished", cleanup=cleanup)
        batchrouter.close()
        edge.close()

    summary = {
        "status": status,
        "error": error_message,
        "run_id": state.run_id,
        "started_at": state.started_at,
        "finished_at": now_iso(),
        "artifact_dir": str(config.artifact_dir),
        "batch_id": state.batch_id,
        "quote_id": state.quote_id,
        "input_file_id": state.input_file_id,
        "input_manifest_path": state.input_manifest_path,
        "input_manifest_bytes": state.input_manifest_bytes,
        "victim_node_id": state.victim_node_id,
        "victim_instance_id": state.victim_instance_id,
        "batch_state": batch_state(state.batch or {}),
        "batch_counts": dict(batch_counts(state.batch or {})),
        "assignment_distribution_before_failure": state.assignment_distribution_before_failure,
        "assignment_distribution_after_failure": state.assignment_distribution_after_failure,
        "execution_distribution": state.execution_distribution,
        "d1_rows_read": state.d1_rows_read,
        "nodes": [
            {
                "node_id": node.node_id,
                "launch_label": node.launch_label,
                "instance_id": node.instance_id,
                "machine_ids": list(node.machine_ids),
                "destroyed": node.destroyed,
                "revoked": node.revoked,
                "destroy_error": node.destroy_error,
                "revoke_error": node.revoke_error,
            }
            for node in state.nodes
        ],
        "cleanup": cleanup,
        "notes": state.notes,
    }
    write_json(config.artifact_dir / "summary.json", summary)
    append_event(config, "drill.finished", status=status, error=error_message, batch_id=state.batch_id)
    return redact_sensitive_payload(summary)


def env_first(*names: str) -> str:
    return first_nonempty(*(os.getenv(name) for name in names)) or ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch live Vast AUTONOMOUSc nodes, submit a pinned BatchRouter batch, "
            "destroy one node after assignment, and verify recovery."
        )
    )
    parser.add_argument("--vast-api-key", default="", help="Defaults to VAST_API_KEY.")
    parser.add_argument(
        "--batchrouter-api-key",
        default="",
        help="Defaults to BATCHROUTER_SMOKE_API_KEY, BATCHROUTER_API_KEY, then BATCHROUTER_ADMIN_API_KEY.",
    )
    parser.add_argument(
        "--operator-token",
        default="",
        help="Defaults to AUTONOMOUSC_OPERATOR_API_KEY, OPERATOR_TOKEN, then SMOKE_OPERATOR_TOKEN.",
    )
    parser.add_argument("--edge-control-url", default="", help=f"Defaults to {DEFAULT_EDGE_CONTROL_URL}.")
    parser.add_argument("--batchrouter-base-url", default="", help=f"Defaults to {DEFAULT_BATCHROUTER_BASE_URL}.")
    parser.add_argument("--edge-control-cwd", default="", help="Path to the edge-control repo for wrangler D1 queries.")
    parser.add_argument("--executions-d1-database", default=DEFAULT_EXECUTIONS_D1)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--launch-nodes", type=int, default=DEFAULT_VAST_NODES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--provider", default=DEFAULT_BATCHROUTER_PROVIDER)
    parser.add_argument("--batchrouter-model", default=DEFAULT_BATCHROUTER_MODEL)
    parser.add_argument("--vast-model", default=DEFAULT_VAST_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-quote-usd", type=float, default=DEFAULT_MAX_QUOTE_USD)
    parser.add_argument(
        "--manifest-upload-threshold-items",
        type=int,
        default=DEFAULT_MANIFEST_UPLOAD_THRESHOLD_ITEMS,
        help=(
            "Use BatchRouter's input_file_id manifest flow once batch-size exceeds this item count. "
            "Use 0 to always upload a manifest."
        ),
    )
    parser.add_argument("--max-price", type=float, default=DEFAULT_VAST_MAX_PRICE)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--assignment-timeout-seconds", type=float, default=DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS)
    parser.add_argument("--completion-timeout-seconds", type=float, default=DEFAULT_COMPLETION_TIMEOUT_SECONDS)
    parser.add_argument(
        "--allow-assigned-victim",
        action="store_true",
        help="Allow destroying a node after an assigned row even if it has not reached accepted yet.",
    )
    parser.add_argument(
        "--skip-failure-drill",
        action="store_true",
        help="Do not destroy a node after assignment; use the full launched fleet for throughput runs.",
    )
    parser.add_argument("--node-region", default=DEFAULT_DURABLE_NODE_REGION)
    parser.add_argument("--runtime-profile", default=DEFAULT_DURABLE_RUNTIME_PROFILE)
    parser.add_argument("--max-batch-tokens", type=int, default=DEFAULT_DURABLE_MAX_BATCH_TOKENS)
    parser.add_argument("--target-batch-tokens", type=int, default=DEFAULT_DURABLE_TARGET_BATCH_TOKENS)
    parser.add_argument("--available-queue-items", type=int, default=DEFAULT_DURABLE_AVAILABLE_QUEUE_ITEMS)
    parser.add_argument("--available-queue-tokens", type=int, default=DEFAULT_DURABLE_AVAILABLE_QUEUE_TOKENS)
    parser.add_argument("--max-queued-items", type=int, default=DEFAULT_DURABLE_MAX_QUEUED_ITEMS)
    parser.add_argument("--image", default=DEFAULT_VAST_SMOKE_IMAGE)
    parser.add_argument("--hf-token", default="", help="Defaults to HUGGING_FACE_HUB_TOKEN, then HF_TOKEN.")
    parser.add_argument("--min-cuda-max-good", type=float, default=DEFAULT_MIN_CUDA_MAX_GOOD)
    parser.add_argument("--min-reliability", type=float, default=0.95)
    parser.add_argument("--min-inet-down-mbps", type=float, default=DEFAULT_MIN_INET_DOWN_MBPS)
    parser.add_argument(
        "--preferred-offer-id",
        action="append",
        type=int,
        default=[],
        help="Preferred Vast offer id for each launched node, in order. Repeat for multiple nodes.",
    )
    parser.add_argument(
        "--exclude-offer-id",
        action="append",
        type=int,
        default=[],
        help="Vast offer id to exclude. Repeat as needed.",
    )
    parser.add_argument(
        "--exclude-machine-id",
        action="append",
        default=[],
        help="Vast machine/host id to exclude. Repeat as needed.",
    )
    parser.add_argument("--exclude-offer-ids", default="", help="Comma-separated Vast offer ids to exclude.")
    parser.add_argument("--exclude-machine-ids", default="", help="Comma-separated Vast machine/host ids to exclude.")
    parser.add_argument(
        "--launch-attempts-per-node",
        type=int,
        default=DEFAULT_NODE_LAUNCH_ATTEMPTS,
        help="Fresh Vast search/enrollment attempts allowed for each required ready node.",
    )
    parser.add_argument(
        "--launch-timeout-seconds",
        type=float,
        default=DEFAULT_LAUNCH_TIMEOUT_SECONDS,
        help="How long to wait for a Vast instance to expose ports before launch progress grace starts.",
    )
    parser.add_argument(
        "--launch-progress-grace-seconds",
        type=float,
        default=DEFAULT_LAUNCH_PROGRESS_GRACE_SECONDS,
        help="Extra launch time allowed while Vast reports active image pull or boot progress.",
    )
    parser.add_argument(
        "--startup-progress-stale-seconds",
        type=float,
        default=DEFAULT_STARTUP_PROGRESS_STALE_SECONDS,
        help=(
            "Fail and retry a Vast host when startup status, stage, and vLLM output stop changing "
            "for this many seconds before /v1/models is ready."
        ),
    )
    parser.add_argument(
        "--startup-max-vllm-restarts",
        type=int,
        default=DEFAULT_STARTUP_MAX_VLLM_RESTARTS,
        help="Maximum vLLM restarts tolerated while a Vast host warms. Use -1 to disable.",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> DrillConfig:
    artifact_dir = Path(str(args.artifact_dir or "").strip() or "test artifacts/live-vast-batchrouter-drill")
    edge_control_cwd = Path(
        str(args.edge_control_cwd or "").strip()
        or env_first("EDGE_CONTROL_CWD")
        or Path.cwd()
    )
    return DrillConfig(
        vast_api_key=first_nonempty(str(args.vast_api_key or ""), os.getenv("VAST_API_KEY")) or "",
        batchrouter_api_key=first_nonempty(
            str(args.batchrouter_api_key or ""),
            os.getenv("BATCHROUTER_SMOKE_API_KEY"),
            os.getenv("BATCHROUTER_API_KEY"),
            os.getenv("BATCHROUTER_ADMIN_API_KEY"),
        )
        or "",
        operator_token=first_nonempty(
            str(args.operator_token or ""),
            os.getenv("AUTONOMOUSC_OPERATOR_API_KEY"),
            os.getenv("OPERATOR_TOKEN"),
            os.getenv("SMOKE_OPERATOR_TOKEN"),
        )
        or "",
        edge_control_url=first_nonempty(
            str(args.edge_control_url or ""),
            os.getenv("EDGE_CONTROL_URL"),
            os.getenv("AUTONOMOUSC_BASE_URL"),
            DEFAULT_EDGE_CONTROL_URL,
        )
        or DEFAULT_EDGE_CONTROL_URL,
        batchrouter_base_url=first_nonempty(
            str(args.batchrouter_base_url or ""),
            os.getenv("BATCHROUTER_BASE_URL"),
            DEFAULT_BATCHROUTER_BASE_URL,
        )
        or DEFAULT_BATCHROUTER_BASE_URL,
        edge_control_cwd=edge_control_cwd,
        executions_d1_database=str(args.executions_d1_database or DEFAULT_EXECUTIONS_D1).strip(),
        artifact_dir=artifact_dir,
        launch_nodes=max(1 if bool(args.skip_failure_drill) else 2, int(args.launch_nodes)),
        batch_size=max(1, int(args.batch_size)),
        provider=str(args.provider or DEFAULT_BATCHROUTER_PROVIDER).strip(),
        batchrouter_model=str(args.batchrouter_model or DEFAULT_BATCHROUTER_MODEL).strip(),
        vast_model=str(args.vast_model or DEFAULT_VAST_MODEL).strip(),
        max_output_tokens=max(1, int(args.max_output_tokens)),
        max_quote_usd=max(0.0, float(args.max_quote_usd)),
        manifest_upload_threshold_items=max(0, int(args.manifest_upload_threshold_items)),
        max_price=max(0.0, float(args.max_price)),
        poll_seconds=max(1.0, float(args.poll_seconds)),
        assignment_timeout_seconds=max(1.0, float(args.assignment_timeout_seconds)),
        completion_timeout_seconds=max(1.0, float(args.completion_timeout_seconds)),
        require_accepted_assignment=not bool(args.allow_assigned_victim),
        destroy_node_after_assignment=not bool(args.skip_failure_drill),
        node_region=str(args.node_region or DEFAULT_DURABLE_NODE_REGION).strip(),
        runtime_profile=str(args.runtime_profile or DEFAULT_DURABLE_RUNTIME_PROFILE).strip(),
        max_batch_tokens=max(1, int(args.max_batch_tokens)),
        target_batch_tokens=max(1, int(args.target_batch_tokens)),
        available_queue_items=max(0, int(args.available_queue_items)),
        available_queue_tokens=max(0, int(args.available_queue_tokens)),
        max_queued_items=max(0, int(args.max_queued_items)),
        image=str(args.image or DEFAULT_VAST_SMOKE_IMAGE).strip(),
        hf_token=first_nonempty(str(args.hf_token or ""), os.getenv("HUGGING_FACE_HUB_TOKEN"), os.getenv("HF_TOKEN")),
        min_cuda_max_good=float(args.min_cuda_max_good) if args.min_cuda_max_good is not None else None,
        min_reliability=max(0.0, min(1.0, float(args.min_reliability))),
        min_inet_down_mbps=max(0.0, float(args.min_inet_down_mbps)),
        preferred_offer_ids=tuple(
            int(value)
            for value in (getattr(args, "preferred_offer_id", []) or [])
            if int(value) > 0
        ),
        exclude_offer_ids=tuple(
            sorted(
                {
                    int(value)
                    for value in [
                        *(getattr(args, "exclude_offer_id", []) or []),
                        *[
                            part.strip()
                            for part in str(getattr(args, "exclude_offer_ids", "") or "").split(",")
                            if part.strip()
                        ],
                    ]
                    if int(value) > 0
                }
            )
        ),
        exclude_machine_ids=tuple(
            parse_identity_list(
                list(getattr(args, "exclude_machine_id", []) or ()),
                csv_value=str(getattr(args, "exclude_machine_ids", "") or ""),
            )
        ),
        launch_attempts_per_node=max(
            1,
            int(getattr(args, "launch_attempts_per_node", DEFAULT_NODE_LAUNCH_ATTEMPTS)),
        ),
        launch_timeout_seconds=max(30.0, float(args.launch_timeout_seconds)),
        launch_progress_grace_seconds=max(0.0, float(args.launch_progress_grace_seconds)),
        startup_progress_stale_seconds=max(0.0, float(args.startup_progress_stale_seconds)),
        startup_max_vllm_restarts=int(args.startup_max_vllm_restarts),
    )


def emit_report(payload: Mapping[str, Any], *, indent: int) -> None:
    text = json.dumps(redact_sensitive_payload(payload), indent=max(0, int(indent)), ensure_ascii=False)
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = build_config_from_args(args)
        report = run_drill(config)
    except (LiveVastBatchRouterDrillError, VastSmokeError) as error:
        report = {"status": "error", "error": str(error)}
    emit_report(report, indent=args.json_indent)
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
