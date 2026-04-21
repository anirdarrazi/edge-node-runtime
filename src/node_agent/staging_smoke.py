from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, parse_qs

import httpx

from .config import AssignmentEnvelope, NodeAgentSettings
from .control_plane import EdgeControlClient
from .runtime_profiles import (
    BURST_PHASE_ACCEPT_BURST_WORK,
    CAPACITY_CLASS_ELASTIC_BURST,
    DEFAULT_RESPONSE_MODEL,
    PARTNER_VLLM_TRUSTED_PROFILE,
    default_public_smoke_test_model,
)

DEFAULT_EDGE_CONTROL_URL = "http://127.0.0.1:8787"
DEFAULT_STAGING_ORG_NAME = "AUTONOMOUSc Disposable Staging Org"
DEFAULT_STAGING_API_KEY_LABEL = "Disposable staging lane key"
DEFAULT_NODE_LABEL = "AUTONOMOUSc Disposable Staging Node"
DEFAULT_REGION = "eu-se-1"
DEFAULT_STAGING_SMOKE_MODEL = default_public_smoke_test_model()
DEFAULT_SEED_CREDIT_UNITS = 100_000_000
DEFAULT_ADMISSION_TIMEOUT_SECONDS = 20.0
DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS = 20.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.25


class StagingSmokeError(RuntimeError):
    pass


@dataclass(frozen=True)
class StagingSmokeConfig:
    edge_control_url: str
    model: str = DEFAULT_STAGING_SMOKE_MODEL
    operation: str = "auto"
    region: str = DEFAULT_REGION
    email: str | None = None
    organization_name: str = DEFAULT_STAGING_ORG_NAME
    api_key_label: str = DEFAULT_STAGING_API_KEY_LABEL
    node_label: str = DEFAULT_NODE_LABEL
    seed_credit_units: int = DEFAULT_SEED_CREDIT_UNITS
    admission_timeout_seconds: float = DEFAULT_ADMISSION_TIMEOUT_SECONDS
    assignment_timeout_seconds: float = DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS


class StagingControlPlaneAPI:
    def __init__(self, base_url: str, *, client: httpx.Client | None = None) -> None:
        normalized = str(base_url).strip().rstrip("/")
        self._client = client or httpx.Client(base_url=normalized, timeout=60.0)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        operator_token: str | None = None,
        api_key: str | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {"content-type": "application/json"}
        if operator_token:
            headers["authorization"] = f"Bearer {operator_token}"
        elif api_key:
            headers["authorization"] = f"Bearer {api_key}"
        response = self._client.request(method, path, headers=headers, json=json_body)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise StagingSmokeError(f"{path} returned a non-object payload.")
        return payload

    def bootstrap_disposable_lane(self, config: StagingSmokeConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "organization_name": config.organization_name,
            "api_key_label": config.api_key_label,
            "credit_units": config.seed_credit_units,
        }
        if config.email:
            payload["email"] = config.email
        return self._request_json("POST", "/staging/disposable-lane/bootstrap", json_body=payload)

    def approve_node_claim(self, claim_id: str, claim_token: str, operator_token: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"/operator/node-claims/{claim_id}/approve",
            operator_token=operator_token,
            json_body={"token": claim_token},
        )

    def approve_temporary_node(self, node_id: str, operator_token: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"/staging/disposable-lane/nodes/{node_id}/approve",
            operator_token=operator_token,
            json_body={},
        )

    def create_execution(self, api_key: str, *, model: str, operation: str, region: str) -> dict[str, Any]:
        item_input: Any
        token_budget: dict[str, int]
        if operation == "embeddings":
            item_input = "hello from the disposable staging lane"
            token_budget = {
                "estimated_input_tokens": 8,
                "estimated_output_tokens": 0,
                "total_tokens": 8,
            }
        else:
            item_input = {"messages": [{"role": "user", "content": "Reply with the single word ready."}]}
            token_budget = {
                "estimated_input_tokens": 12,
                "estimated_output_tokens": 8,
                "total_tokens": 20,
            }
        return self._request_json(
            "POST",
            "/provider/executions",
            api_key=api_key,
            json_body={
                "idempotency_key": f"staging-smoke-{int(time.time() * 1000)}",
                "source_batch_id": "staging-smoke-batch",
                "source_work_unit_id": "staging-smoke-work-unit",
                "operation": operation,
                "model": model,
                "item_count": 1,
                "token_budget": token_budget,
                "sla_tier": "standard",
                "privacy_tier": "standard",
                "allowed_regions": [region],
                "items": [
                    {
                        "batch_item_id": "item-1",
                        "customer_item_id": "customer-item-1",
                        "operation": operation,
                        "model": model,
                        "input": item_input,
                    }
                ],
            },
        )

    def get_execution_admission(self, admission_id: str, api_key: str) -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"/provider/execution-admissions/{admission_id}",
            api_key=api_key,
        )

    def get_execution(self, execution_id: str, api_key: str) -> dict[str, Any]:
        return self._request_json("GET", f"/provider/executions/{execution_id}", api_key=api_key)

    def revoke_node(self, node_id: str, operator_token: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"/operator/nodes/{node_id}/revoke",
            operator_token=operator_token,
            json_body={},
        )

    def revoke_api_key(self, api_key_id: str, operator_token: str) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"/operator/api-keys/{api_key_id}/revoke",
            operator_token=operator_token,
            json_body={},
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_operation(model: str, requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized in {"responses", "embeddings"}:
        return normalized
    return "embeddings" if model == DEFAULT_STAGING_SMOKE_MODEL else "responses"


def claim_token_from_approval_url(approval_url: str) -> str:
    parsed = urlparse(str(approval_url))
    token = parse_qs(parsed.query).get("claim_token", [None])[0]
    if not token:
        raise StagingSmokeError("Node claim approval URL did not include a claim_token.")
    return str(token)


def first_item_ids(payload: dict[str, Any]) -> tuple[str, str]:
    items = payload.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        batch_item_id = str(items[0].get("batch_item_id") or "item-1")
        customer_item_id = str(items[0].get("customer_item_id") or "customer-item-1")
        return batch_item_id, customer_item_id
    return "item-1", "customer-item-1"


def build_item_results(
    *,
    operation: str,
    model: str,
    assignment: AssignmentEnvelope,
    decrypted_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    batch_item_id, customer_item_id = first_item_ids(decrypted_payload)
    usage: dict[str, Any]
    output: dict[str, Any]
    if operation == "embeddings":
        usage = {
            "input_texts": 1,
            "total_tokens": 8,
        }
        output = {"embedding": [0.0, 1.0, 2.0]}
    else:
        usage = {
            "input_tokens": 12,
            "output_tokens": 1,
            "total_tokens": 13,
        }
        output = {"text": "ready"}
    return [
        {
            "batch_item_id": batch_item_id,
            "customer_item_id": customer_item_id,
            "provider": "autonomousc_edge",
            "provider_model": model,
            "status": "completed",
            "usage": usage,
            "cost": {
                "provider_cost": {"amount": "0.0001", "currency": "usd"},
                "customer_charge": {"amount": "0.0002", "currency": "usd"},
                "platform_margin": {"amount": "0.0001", "currency": "usd"},
                "pricing_source": "catalog",
            },
            "output": output,
            "completed_at": now_iso(),
        }
    ]


def unauthorized_http_error(url: str, status_code: int = 401) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)


class StagingSmokeRunner:
    def __init__(
        self,
        api: StagingControlPlaneAPI,
        *,
        control_client_factory: Any = EdgeControlClient,
        monotonic: Any = time.monotonic,
        sleep: Any = time.sleep,
    ) -> None:
        self.api = api
        self.control_client_factory = control_client_factory
        self.monotonic = monotonic
        self.sleep = sleep

    def _wait_for_execution(
        self,
        api_key: str,
        admission_id: str,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = self.monotonic() + max(1.0, timeout_seconds)
        last_status = "queued"
        while self.monotonic() < deadline:
            payload = self.api.get_execution_admission(admission_id, api_key)
            admission = payload.get("admission")
            execution = payload.get("execution")
            if isinstance(admission, dict):
                last_status = str(admission.get("status") or last_status)
                if last_status == "failed":
                    raise StagingSmokeError(f"Execution admission failed: {json.dumps(admission.get('error') or {})}")
            if isinstance(execution, dict):
                return admission if isinstance(admission, dict) else {}, execution
            self.sleep(poll_interval_seconds)
        raise StagingSmokeError(
            f"Execution admission {admission_id} did not yield an execution in time (last status: {last_status})."
        )

    def _wait_for_assignment(
        self,
        control: Any,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> AssignmentEnvelope:
        deadline = self.monotonic() + max(1.0, timeout_seconds)
        while self.monotonic() < deadline:
            assignments = control.pull_assignments(1)
            if assignments:
                return assignments[0]
            control.heartbeat(queue_depth=0, active_assignments=0, status="active")
            self.sleep(poll_interval_seconds)
        raise StagingSmokeError("No leased assignment arrived in time for the disposable staging node.")

    @staticmethod
    def _node_settings(config: StagingSmokeConfig, scratch_dir: str) -> NodeAgentSettings:
        operation = resolve_operation(config.model, config.operation)
        return NodeAgentSettings(
            edge_control_url=config.edge_control_url,
            node_label=config.node_label,
            node_region=config.region,
            trust_tier="restricted",
            restricted_capable=True,
            credentials_path=os.path.join(scratch_dir, "node-credentials.json"),
            attestation_state_path=os.path.join(scratch_dir, "attestation-state.json"),
            recovery_note_path=os.path.join(scratch_dir, "recovery-note.txt"),
            control_plane_state_path=os.path.join(scratch_dir, "control-plane-state.json"),
            runtime_profile=PARTNER_VLLM_TRUSTED_PROFILE,
            vllm_model=config.model,
            supported_models=config.model,
            gpu_name="Disposable staging GPU",
            gpu_memory_gb=24.0,
            max_context_tokens=8192 if operation == "embeddings" else 32768,
            max_batch_tokens=4096,
            max_concurrent_assignments=1,
            temporary_node=True,
            capacity_class=CAPACITY_CLASS_ELASTIC_BURST,
            burst_provider="vast_ai",
            burst_lease_id=f"staging-{int(time.time())}",
            burst_lease_phase=BURST_PHASE_ACCEPT_BURST_WORK,
        )

    def run(self, config: StagingSmokeConfig) -> dict[str, Any]:
        started_at = self.monotonic()
        operation = resolve_operation(config.model, config.operation)
        report: dict[str, Any] = {
            "status": "error",
            "requested": {
                "edge_control_url": config.edge_control_url,
                "model": config.model,
                "operation": operation,
                "region": config.region,
                "temporary_node": True,
            },
            "bootstrap": None,
            "claim": None,
            "node": None,
            "execution": None,
            "assignment": None,
            "cleanup": {
                "node_revoked": False,
                "api_key_revoked": False,
                "node_credentials_rejected": False,
                "api_key_rejected": False,
            },
            "notes": [],
        }
        operator_token: str | None = None
        api_key: str | None = None
        api_key_id: str | None = None
        node_id: str | None = None
        execution_id: str | None = None

        with tempfile.TemporaryDirectory(prefix="autonomousc-staging-lane-") as scratch_dir:
            settings = self._node_settings(config, scratch_dir)
            control = self.control_client_factory(settings)
            try:
                bootstrap = self.api.bootstrap_disposable_lane(config)
                operator_token = str(bootstrap["operator_token"])
                api_key = str(bootstrap["api_key"])
                api_key_id = str(bootstrap["api_key_id"])
                report["bootstrap"] = {
                    "email": bootstrap.get("email"),
                    "organization_id": bootstrap.get("organization_id"),
                    "approval_status": bootstrap.get("approval_status"),
                    "api_key_id": api_key_id,
                    "seeded_credit_units": bootstrap.get("seeded_credit_units"),
                }

                claim = control.create_node_claim_session()
                claim_token = claim_token_from_approval_url(claim.approval_url)
                self.api.approve_node_claim(claim.claim_id, claim_token, operator_token)
                polled = control.poll_node_claim_session(claim.claim_id, claim.poll_token)
                if not polled.node_id or not polled.node_key:
                    raise StagingSmokeError("Disposable staging claim was approved but did not return node credentials.")
                settings.node_id = polled.node_id
                settings.node_key = polled.node_key
                node_id = polled.node_id
                report["claim"] = {
                    "claim_id": claim.claim_id,
                    "status": polled.status,
                    "claim_code": claim.claim_code,
                    "node_id": polled.node_id,
                }

                control.attest()
                approved_node = self.api.approve_temporary_node(polled.node_id, operator_token)
                control.heartbeat(queue_depth=0, active_assignments=0, status="active")
                report["node"] = {
                    "node_id": polled.node_id,
                    "approval_status": (
                        approved_node.get("node", {}).get("approval_status")
                        if isinstance(approved_node.get("node"), dict)
                        else None
                    ),
                    "temporary_node": True,
                    "attested": True,
                    "heartbeats_sent": 1,
                }

                created = self.api.create_execution(api_key, model=config.model, operation=operation, region=config.region)
                admission = created.get("admission")
                if not isinstance(admission, dict) or not admission.get("id"):
                    raise StagingSmokeError("Disposable staging execution creation did not return an admission id.")
                admission_payload, execution = self._wait_for_execution(
                    api_key,
                    str(admission["id"]),
                    timeout_seconds=config.admission_timeout_seconds,
                    poll_interval_seconds=config.poll_interval_seconds,
                )
                execution_id = str(execution.get("id") or "")
                if not execution_id:
                    raise StagingSmokeError("Disposable staging execution admission never produced an execution id.")
                report["execution"] = {
                    "admission_id": admission.get("id"),
                    "admission_status": admission_payload.get("status"),
                    "execution_id": execution_id,
                    "status": execution.get("status"),
                }

                assignment = self._wait_for_assignment(
                    control,
                    timeout_seconds=config.assignment_timeout_seconds,
                    poll_interval_seconds=config.poll_interval_seconds,
                )
                control.accept_assignment(assignment.assignment_id)
                control.report_progress(assignment.assignment_id, {"stage": "staging_lane", "state": "running"})
                decrypted_payload = control.fetch_artifact(assignment)
                item_results = build_item_results(
                    operation=operation,
                    model=config.model,
                    assignment=assignment,
                    decrypted_payload=decrypted_payload,
                )
                control.complete_assignment(assignment.assignment_id, item_results)

                finalized = self.api.get_execution(execution_id, api_key)
                execution_payload = finalized.get("execution")
                report["assignment"] = {
                    "assignment_id": assignment.assignment_id,
                    "execution_id": assignment.execution_id,
                    "artifact_items": len(decrypted_payload.get("items", []))
                    if isinstance(decrypted_payload.get("items"), list)
                    else None,
                    "accepted": True,
                    "completed": True,
                }
                if isinstance(execution_payload, dict):
                    report["execution"] = {
                        **(report["execution"] or {}),
                        "status": execution_payload.get("status"),
                        "selected_node_id": execution_payload.get("selected_node_id"),
                        "verification_status": execution_payload.get("verification_status"),
                    }
                report["status"] = "ok"
            except Exception as error:
                report["error"] = str(error) or error.__class__.__name__
            finally:
                cleanup = dict(report.get("cleanup") or {})
                if node_id and operator_token:
                    try:
                        self.api.revoke_node(node_id, operator_token)
                        cleanup["node_revoked"] = True
                        try:
                            control.heartbeat(queue_depth=0, active_assignments=0, status="active")
                        except httpx.HTTPStatusError as error:
                            cleanup["node_credentials_rejected"] = error.response.status_code == 401
                    except Exception as error:
                        cleanup["node_revoke_error"] = str(error) or error.__class__.__name__
                if api_key_id and operator_token:
                    try:
                        self.api.revoke_api_key(api_key_id, operator_token)
                        cleanup["api_key_revoked"] = True
                        if execution_id and api_key:
                            try:
                                self.api.get_execution(execution_id, api_key)
                            except httpx.HTTPStatusError as error:
                                cleanup["api_key_rejected"] = error.response.status_code == 401
                    except Exception as error:
                        cleanup["api_key_revoke_error"] = str(error) or error.__class__.__name__
                report["cleanup"] = cleanup
                report["timings"] = {
                    "total_seconds": round(self.monotonic() - started_at, 2),
                }
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a disposable full-path control-plane staging smoke test for node claim, heartbeat, assignment, completion, and credential revocation."
    )
    parser.add_argument(
        "--edge-control-url",
        default=os.getenv("EDGE_CONTROL_URL", DEFAULT_EDGE_CONTROL_URL),
        help="Staging edge-control base URL. Defaults to EDGE_CONTROL_URL.",
    )
    parser.add_argument("--model", default=DEFAULT_STAGING_SMOKE_MODEL, help="Model to request for the leased assignment.")
    parser.add_argument(
        "--operation",
        choices=("auto", "responses", "embeddings"),
        default="auto",
        help="Operation to request. Defaults to auto based on the model.",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help="Node region to advertise and request.")
    parser.add_argument("--email", default=os.getenv("NODE_AGENT_STAGING_EMAIL") or "", help="Optional disposable operator email override.")
    parser.add_argument("--organization-name", default=DEFAULT_STAGING_ORG_NAME, help="Disposable operator organization name.")
    parser.add_argument("--api-key-label", default=DEFAULT_STAGING_API_KEY_LABEL, help="Disposable provider API key label.")
    parser.add_argument("--node-label", default=DEFAULT_NODE_LABEL, help="Disposable node label.")
    parser.add_argument("--seed-credit-units", type=int, default=DEFAULT_SEED_CREDIT_UNITS, help="Staging credit units to seed into the disposable org.")
    parser.add_argument("--admission-timeout-seconds", type=float, default=DEFAULT_ADMISSION_TIMEOUT_SECONDS, help="How long to wait for execution admission to produce an execution.")
    parser.add_argument("--assignment-timeout-seconds", type=float, default=DEFAULT_ASSIGNMENT_TIMEOUT_SECONDS, help="How long to wait for an assignment to lease to the disposable node.")
    parser.add_argument("--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS, help="Polling interval for admission and assignment waits.")
    parser.add_argument("--json-indent", type=int, default=2, help="JSON indentation level for the report.")
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> StagingSmokeConfig:
    edge_control_url = str(args.edge_control_url or "").strip().rstrip("/")
    if not edge_control_url:
        raise StagingSmokeError("An edge-control base URL is required. Pass --edge-control-url or set EDGE_CONTROL_URL.")
    return StagingSmokeConfig(
        edge_control_url=edge_control_url,
        model=str(args.model or DEFAULT_STAGING_SMOKE_MODEL).strip() or DEFAULT_STAGING_SMOKE_MODEL,
        operation=str(args.operation or "auto").strip().lower(),
        region=str(args.region or DEFAULT_REGION).strip() or DEFAULT_REGION,
        email=str(args.email or "").strip() or None,
        organization_name=str(args.organization_name or DEFAULT_STAGING_ORG_NAME).strip() or DEFAULT_STAGING_ORG_NAME,
        api_key_label=str(args.api_key_label or DEFAULT_STAGING_API_KEY_LABEL).strip() or DEFAULT_STAGING_API_KEY_LABEL,
        node_label=str(args.node_label or DEFAULT_NODE_LABEL).strip() or DEFAULT_NODE_LABEL,
        seed_credit_units=max(1, int(args.seed_credit_units)),
        admission_timeout_seconds=max(1.0, float(args.admission_timeout_seconds)),
        assignment_timeout_seconds=max(1.0, float(args.assignment_timeout_seconds)),
        poll_interval_seconds=max(0.05, float(args.poll_interval_seconds)),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    json_indent = max(0, int(args.json_indent))
    try:
        config = build_config_from_args(args)
    except StagingSmokeError as error:
        print(json.dumps({"status": "error", "error": str(error)}, indent=json_indent))
        return 1

    api = StagingControlPlaneAPI(config.edge_control_url)
    try:
        report = StagingSmokeRunner(api).run(config)
    finally:
        api.close()

    print(json.dumps(report, indent=json_indent, ensure_ascii=False))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
