from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from .config import AssignmentEnvelope, NodeAgentSettings, NodeClaimPollResult, NodeClaimSession
from .concurrency import (
    resolved_embeddings_concurrency_limit,
    resolved_embeddings_microbatch_assignment_limit,
    resolved_local_queue_assignment_limit,
)
from .control_plane_bootstrap import NodeBootstrapOrchestrator
from .control_plane_store import NodeCredentialStore
from .control_plane_transport import EdgeControlTransport
from .gguf_artifacts import resolved_default_gguf_artifact_contract
from .runtime_quality import resolved_runtime_quality
from .runtime_evidence import resolved_signed_runtime_evidence
from .runtime_tuple import resolved_default_runtime_tuple


ARTIFACT_FETCH_RETRY_STATUSES = {404, 408, 409, 425, 429, 500, 502, 503, 504}
ARTIFACT_FETCH_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)
ARTIFACT_URL_REFRESH_THRESHOLD_SECONDS = 120
ARTIFACT_UPLOAD_MAX_ATTEMPTS = 3


class ArtifactFlowError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def _parse_iso_datetime(value: str | None) -> datetime | None:
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
    urls: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        candidate = raw.strip().rstrip("/")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)
    return urls


class EdgeControlClient:
    def __init__(self, settings: NodeAgentSettings):
        self.settings = settings
        self.transport = EdgeControlTransport(settings)
        self.credentials = NodeCredentialStore(settings)
        self.bootstrapper = NodeBootstrapOrchestrator(
            settings,
            self.credentials,
            legacy_enroll=self.enroll_if_needed,
            create_claim_session=self.create_node_claim_session,
            poll_claim_session=self.poll_node_claim_session,
            persist_from_response=self._persist_from_response,
            terminal_available=self._interactive_terminal_available,
            sleep=lambda seconds: time.sleep(seconds),
        )
        self.last_control_plane_queue_depth = 0

    @property
    def client(self) -> Any:
        return self.transport.client

    @client.setter
    def client(self, value: Any) -> None:
        self.transport.client = value

    def _load_persisted_credentials(self) -> tuple[str, str] | None:
        return self.credentials.load_credentials()

    def _persist_credentials(self, node_id: str, node_key: str) -> None:
        self.credentials.persist_credentials_file(node_id, node_key)

    def load_attestation_state(self) -> dict[str, Any] | None:
        return self.credentials.load_attestation_state()

    def _persist_attestation_state(self, *, attestation_provider: str) -> None:
        self.credentials.persist_attestation_state(attestation_provider=attestation_provider)

    def clear_attestation_state(self) -> None:
        self.credentials.clear_attestation_state()

    @staticmethod
    def _tighten_directory_permissions(path: Path) -> None:
        NodeCredentialStore.tighten_directory_permissions(path)

    @classmethod
    def _tighten_credentials_permissions(cls, path: Path) -> None:
        NodeCredentialStore.tighten_credentials_permissions(path)

    def _interactive_terminal_available(self) -> bool:
        return sys.stdin.isatty() and sys.stdout.isatty()

    def _node_capabilities_payload(self) -> dict[str, Any]:
        runtime_profile = self.settings.resolved_runtime_profile
        runtime_tuple = resolved_default_runtime_tuple(self.settings)
        embedding_concurrency_limit = resolved_embeddings_concurrency_limit(
            supported_models=self.settings.supported_models,
            operations=list(runtime_profile.supported_apis),
            gpu_memory_gb=self.settings.gpu_memory_gb,
            max_concurrent_assignments=self.settings.max_concurrent_assignments,
            override=getattr(self.settings, "max_concurrent_assignments_embeddings", None),
        )
        embedding_microbatch_limit = resolved_embeddings_microbatch_assignment_limit(
            supported_models=self.settings.supported_models,
            operations=list(runtime_profile.supported_apis),
            gpu_memory_gb=self.settings.gpu_memory_gb,
            max_concurrent_assignments=self.settings.max_concurrent_assignments,
            pull_bundle_size=self.settings.pull_bundle_size,
            override=getattr(self.settings, "max_microbatch_assignments_embeddings", None),
        )
        local_queue_limit = resolved_local_queue_assignment_limit(
            supported_models=self.settings.supported_models,
            operations=list(runtime_profile.supported_apis),
            gpu_memory_gb=self.settings.gpu_memory_gb,
            max_concurrent_assignments=max(
                self.settings.max_concurrent_assignments,
                embedding_concurrency_limit or self.settings.max_concurrent_assignments,
            ),
            pull_bundle_size=self.settings.pull_bundle_size,
            max_microbatch_assignments=embedding_microbatch_limit,
            override=getattr(self.settings, "max_local_queue_assignments", None),
        )
        pull_bundle_limit = max(local_queue_limit, self.settings.pull_bundle_size)
        payload = {
            "supported_models": [model.strip() for model in self.settings.supported_models.split(",") if model.strip()],
            "operations": list(runtime_profile.supported_apis),
            "gpu_name": self.settings.gpu_name,
            "gpu_memory_gb": self.settings.gpu_memory_gb,
            "max_context_tokens": runtime_tuple.effective_context_tokens or self.settings.max_context_tokens,
            "max_batch_tokens": self.settings.max_batch_tokens,
            "max_concurrent_assignments": self.settings.max_concurrent_assignments,
            "max_local_queue_assignments": local_queue_limit,
            "max_pull_bundle_assignments": pull_bundle_limit,
            "heat_governor_mode": self.settings.heat_governor_mode,
            "thermal_headroom": self.settings.thermal_headroom,
            "heat_demand": self.settings.heat_demand,
            "target_gpu_utilization_pct": self.settings.target_gpu_utilization_pct,
            "min_gpu_memory_headroom_pct": self.settings.min_gpu_memory_headroom_pct,
        }
        if embedding_concurrency_limit is not None:
            payload["max_concurrent_assignments_embeddings"] = embedding_concurrency_limit
        if embedding_microbatch_limit is not None:
            payload["max_microbatch_assignments_embeddings"] = embedding_microbatch_limit
        for key in (
            "room_temp_c",
            "target_temp_c",
            "outside_temp_c",
            "gpu_temp_c",
            "gpu_temp_limit_c",
            "power_watts",
            "estimated_heat_output_watts",
            "energy_price_kwh",
        ):
            value = getattr(self.settings, key)
            if value is not None:
                payload[key] = value
        return payload

    def _node_runtime_payload(self, *, autopilot: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime_tuple = resolved_default_runtime_tuple(self.settings)
        gguf_artifact = resolved_default_gguf_artifact_contract(self.settings)
        runtime_quality = resolved_runtime_quality(self.settings, gguf_artifact)
        runtime_profile = self.settings.resolved_runtime_profile
        runtime = {
            "agent_version": self.settings.agent_version,
            "runtime_profile": runtime_profile.id,
            "runtime_profile_label": runtime_profile.label,
            "inference_engine": self.settings.resolved_inference_engine,
            "deployment_target": self.settings.resolved_deployment_target,
            "model_format": runtime_profile.model_format,
            "runtime_image": self.settings.resolved_runtime_image,
            "readiness_path": runtime_profile.readiness_path,
            "supported_apis": list(runtime_profile.supported_apis),
            "trust_policy": runtime_profile.trust_policy,
            "pricing_tier": runtime_profile.pricing_tier,
            "artifact_manifest_type": runtime_profile.artifact_manifest_type,
            "capacity_class": self.settings.resolved_capacity_class,
            "routing_lane": runtime_profile.routing_lane,
            "routing_lane_label": runtime_profile.routing_lane_label,
            "routing_lane_detail": runtime_profile.routing_lane_detail,
            "routing_lane_policy_summary": runtime_profile.routing_lane_policy_summary,
            "routing_lane_allowed_privacy_tiers": list(runtime_profile.routing_lane_allowed_privacy_tiers),
            "routing_lane_allowed_result_guarantees": list(runtime_profile.routing_lane_allowed_result_guarantees),
            "routing_lane_allowed_trust_requirements": list(runtime_profile.routing_lane_allowed_trust_requirements),
            "max_privacy_tier": runtime_profile.max_privacy_tier,
            "exact_model_guarantee": runtime_profile.exact_model_guarantee,
            "quantized_output_disclosure_required": runtime_profile.quantized_output_disclosure_required,
            "quality_class": runtime_quality["quality_class"],
            "exactness_class": runtime_quality["exactness_class"],
            "trusted_eligibility": runtime_profile.trusted_eligibility,
            "inference_base_url": self.settings.resolved_inference_base_url,
            "vllm_base_url": self.settings.resolved_inference_base_url,
            "docker_image": self.settings.docker_image,
            "current_model": self.settings.current_model,
            "effective_context_tokens": runtime_tuple.effective_context_tokens,
            "runtime_tuple_digest": runtime_tuple.runtime_tuple_digest,
        }
        if runtime_profile.burst_lifecycle:
            runtime["burst_lifecycle"] = list(runtime_profile.burst_lifecycle)
        if runtime_profile.burst_cost_ceiling_usd is not None or self.settings.burst_cost_ceiling_usd is not None:
            runtime["burst_cost_ceiling_usd"] = (
                self.settings.burst_cost_ceiling_usd
                if self.settings.burst_cost_ceiling_usd is not None
                else runtime_profile.burst_cost_ceiling_usd
            )
        if self.settings.resolved_capacity_class == "elastic_burst":
            runtime["temporary_node"] = self.settings.temporary_node
            runtime["burst_provider"] = self.settings.burst_provider or "vast_ai"
            if self.settings.burst_lease_id:
                runtime["burst_lease_id"] = self.settings.burst_lease_id
            if self.settings.burst_lease_phase:
                runtime["burst_lease_phase"] = self.settings.burst_lease_phase
        if runtime_tuple.model_manifest_digest is not None:
            runtime["model_manifest_digest"] = runtime_tuple.model_manifest_digest
        if runtime_tuple.tokenizer_digest is not None:
            runtime["tokenizer_digest"] = runtime_tuple.tokenizer_digest
        if runtime_tuple.chat_template_digest is not None:
            runtime["chat_template_digest"] = runtime_tuple.chat_template_digest
        if gguf_artifact is not None:
            runtime["gguf_artifact"] = gguf_artifact.payload()
        runtime_evidence = resolved_signed_runtime_evidence(self.settings, runtime_tuple, gguf_artifact)
        if runtime_evidence is not None:
            runtime["runtime_evidence"] = runtime_evidence
        if autopilot is not None:
            runtime["autopilot"] = autopilot
        return runtime

    @staticmethod
    def _nonnegative_int(value: Any, fallback: int = 0) -> int:
        try:
            return max(0, int(float(str(value))))
        except (TypeError, ValueError):
            return fallback

    def _node_request_payload(self) -> dict[str, Any]:
        return {
            "label": self.settings.node_label,
            "region": self.settings.node_region,
            "trust_tier": self.settings.trust_tier,
            "restricted_capable": self.settings.restricted_capable,
            "capabilities": self._node_capabilities_payload(),
            "runtime": self._node_runtime_payload(),
        }

    def node_capabilities_payload(self) -> dict[str, Any]:
        return self._node_capabilities_payload()

    def node_runtime_payload(self, *, autopilot: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._node_runtime_payload(autopilot=autopilot)

    def control_plane_snapshot(self) -> dict[str, Any]:
        snapshot = getattr(self.transport, "snapshot", None)
        if callable(snapshot):
            payload = snapshot()
            return payload if isinstance(payload, dict) else {}
        return {}

    def recommended_control_plane_retry_delay_seconds(self) -> float:
        recommender = getattr(self.transport, "recommended_retry_delay_seconds", None)
        if callable(recommender):
            try:
                return max(0.5, float(recommender()))
            except (TypeError, ValueError):
                return max(0.5, float(self.settings.poll_interval_seconds))
        return max(0.5, float(self.settings.poll_interval_seconds))

    @staticmethod
    def _format_remaining_time(expires_at: str) -> str:
        return NodeBootstrapOrchestrator.format_remaining_time(expires_at)

    def _print_claim_instructions(self, claim: NodeClaimSession) -> None:
        self.bootstrapper.print_claim_instructions(claim)

    @staticmethod
    def _setup_ui_claim_message() -> str:
        return NodeBootstrapOrchestrator.setup_ui_claim_message()

    def has_credentials(self) -> bool:
        return self.bootstrapper.has_credentials()

    def require_credentials(self) -> tuple[str, str]:
        return self.bootstrapper.require_credentials()

    def _node_identity_payload(self) -> dict[str, Any]:
        node_id, node_key = self.require_credentials()
        return {
            "node_id": node_id,
            "node_key": node_key,
            "node_session_id": self.settings.resolved_node_session_id,
            "boot_id": self.settings.boot_id,
        }

    def _node_identity_headers(self) -> dict[str, str]:
        identity = self._node_identity_payload()
        return {
            "x-node-id": str(identity["node_id"]),
            "x-node-key": str(identity["node_key"]),
            "x-node-session-id": str(identity["node_session_id"]),
            "x-boot-id": str(identity["boot_id"]),
        }

    def clear_credentials(self) -> None:
        self.credentials.clear_credentials()

    def write_recovery_note(self, message: str) -> None:
        self.credentials.write_recovery_note(message)

    def clear_recovery_note(self) -> None:
        self.credentials.clear_recovery_note()

    def persist_credentials(self, node_id: str, node_key: str) -> tuple[str, str]:
        return self.credentials.persist_credentials(node_id, node_key)

    def is_auth_error(self, error: Exception) -> bool:
        return self.transport.is_auth_error(error)

    def is_transient_network_error(self, error: Exception) -> bool:
        if isinstance(error, ArtifactFlowError):
            return error.retryable
        return self.transport.is_transient_network_error(error)

    def _persist_from_response(self, payload: dict[str, Any]) -> tuple[str, str]:
        return self.persist_credentials(str(payload["node_id"]), str(payload["node_key"]))

    def enroll_if_needed(self) -> tuple[str, str]:
        if self.settings.node_id and self.settings.node_key:
            return self.settings.node_id, self.settings.node_key
        persisted = self.credentials.load_credentials()
        if persisted:
            return persisted
        if not self.settings.operator_token:
            raise RuntimeError(
                "OPERATOR_TOKEN is only needed for legacy headless enrollment. "
                "Normal setup should use the setup UI claim flow instead."
            )

        payload = self.transport.post_json(
            "/nodes/enroll",
            {
                "operator_token": self.settings.operator_token,
                **self._node_request_payload(),
            },
        )
        return self._persist_from_response(payload)

    def create_node_claim_session(self) -> NodeClaimSession:
        payload = self.transport.post_json("/node-claims", self._node_request_payload())
        return NodeClaimSession.model_validate(payload)

    def poll_node_claim_session(self, claim_id: str, poll_token: str) -> NodeClaimPollResult:
        payload = self.transport.post_json(f"/node-claims/{claim_id}/poll", {"poll_token": poll_token})
        return NodeClaimPollResult.model_validate(payload)

    def bootstrap(self, interactive: bool = True) -> tuple[str, str]:
        return self.bootstrapper.bootstrap(interactive=interactive)

    def attest(self) -> None:
        if not self.settings.node_id or not self.settings.node_key:
            raise RuntimeError("node must be enrolled before attestation")
        attestation_provider = self.settings.attestation_provider
        self.transport.post_json(
            "/nodes/attest",
            {
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "tpm_quote": f"{attestation_provider}-{self.settings.node_id}",
                "measurements": {
                    "pcr0": "simulated" if attestation_provider == "simulated" else "hardware",
                    "pcr7": "simulated" if attestation_provider == "simulated" else "hardware",
                    "attestation_provider": attestation_provider,
                },
                "inventory": {
                    "gpu_name": self.settings.gpu_name,
                    "driver": "simulated" if attestation_provider == "simulated" else "verified",
                    "attestation_provider": attestation_provider,
                },
                "restricted_capable": self.settings.restricted_capable,
            },
        )
        self._persist_attestation_state(attestation_provider=attestation_provider)

    def heartbeat(
        self,
        queue_depth: int = 0,
        active_assignments: int = 0,
        *,
        status: str = "active",
        capabilities: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        autopilot: dict[str, Any] | None = None,
        include_capabilities: bool = True,
        include_runtime: bool = True,
    ) -> None:
        payload: dict[str, Any] = {
            **self._node_identity_payload(),
            "status": status,
            "queue_depth": queue_depth,
            "active_assignments": active_assignments,
        }
        if include_capabilities:
            payload["capabilities"] = capabilities or self._node_capabilities_payload()
        if include_runtime:
            payload["runtime"] = runtime or self._node_runtime_payload(autopilot=autopilot)
        self.transport.post_json("/nodes/heartbeat", payload)
        self.clear_recovery_note()

    def fetch_node_dashboard_summary(self) -> dict[str, Any]:
        node_id, node_key = self.require_credentials()
        payload = self.transport.post_json(
            "/nodes/dashboard",
            {
                "node_id": node_id,
                "node_key": node_key,
            },
        )
        return payload if isinstance(payload, dict) else {}

    def submit_support_bundle(
        self,
        bundle_name: str,
        bundle_bytes: bytes,
        *,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        node_id, node_key = self.require_credentials()
        payload = self.transport.post_json(
            "/nodes/support-bundles",
            {
                "node_id": node_id,
                "node_key": node_key,
                "bundle_name": bundle_name,
                "bundle_content_base64": base64.b64encode(bundle_bytes).decode("ascii"),
                "bundle_size_bytes": len(bundle_bytes),
                "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
                "generated_at": generated_at,
            },
        )
        return payload if isinstance(payload, dict) else {}

    def pull_assignments(
        self,
        limit: int,
        active_assignment_ids: list[str] | None = None,
    ) -> list[AssignmentEnvelope]:
        if limit <= 0:
            return []
        payload = self.transport.post_json(
            "/nodes/assignments/pull",
            {
                **self._node_identity_payload(),
                "limit": limit,
                "active_assignment_ids": active_assignment_ids or [],
            },
        )
        self.last_control_plane_queue_depth = self._nonnegative_int(payload.get("queue_depth"))
        assignments = payload.get("assignments")
        if isinstance(assignments, list):
            return [AssignmentEnvelope.model_validate(assignment) for assignment in assignments if assignment]
        assignment = payload.get("assignment")
        return [AssignmentEnvelope.model_validate(assignment)] if assignment else []

    def pull_assignment(self) -> AssignmentEnvelope | None:
        assignments = self.pull_assignments(1)
        return assignments[0] if assignments else None

    def accept_assignment(self, assignment_id: str) -> None:
        self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/accept",
            self._node_identity_payload(),
        )

    def report_progress(self, assignment_id: str, progress: dict[str, Any]) -> None:
        self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/progress",
            {
                **self._node_identity_payload(),
                "progress": progress,
            },
        )

    def touch_assignments(self, assignment_ids: list[str]) -> None:
        normalized_assignment_ids = [assignment_id for assignment_id in dict.fromkeys(assignment_ids) if assignment_id]
        if not normalized_assignment_ids:
            return
        self.transport.post_json(
            "/nodes/assignments/touch",
            {
                **self._node_identity_payload(),
                "assignment_ids": normalized_assignment_ids,
            },
        )

    @staticmethod
    def _upload_headers(upload_headers: Any) -> dict[str, str]:
        if not isinstance(upload_headers, dict):
            raise ValueError("artifact upload plan is missing upload headers")
        return {str(key): str(value) for key, value in upload_headers.items()}

    def _artifact_url_expires_soon(self, expires_at: str | None) -> bool:
        parsed = _parse_iso_datetime(expires_at)
        if parsed is None:
            return False
        remaining_seconds = (parsed - datetime.now(timezone.utc)).total_seconds()
        return remaining_seconds <= ARTIFACT_URL_REFRESH_THRESHOLD_SECONDS

    def refresh_input_artifact_url(self, assignment_id: str) -> dict[str, Any]:
        return self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/input-artifact-url",
            self._node_identity_payload(),
        )

    def _request_result_artifact_upload_plan(
        self,
        assignment_id: str,
        encrypted_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        return self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/result-artifact/upload-url",
            {
                **self._node_identity_payload(),
                "ciphertext_sha256": encrypted_artifact["ciphertext_sha256"],
                "plaintext_sha256": encrypted_artifact["plaintext_sha256"],
                "content_length_bytes": len(encrypted_artifact["ciphertext"]),
            },
        )

    @staticmethod
    def _is_retryable_artifact_upload_error(error: Exception) -> bool:
        if isinstance(error, ArtifactFlowError):
            return error.retryable
        if isinstance(error, httpx.TransportError):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {401, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504}
        return False

    def _upload_single_result_artifact(self, artifact: dict[str, Any], encrypted_artifact: dict[str, Any]) -> dict[str, Any]:
        upload_url = artifact.get("upload_url")
        upload_headers = artifact.get("upload_headers")
        if isinstance(upload_url, str) and isinstance(upload_headers, dict):
            self.transport.put_content(
                upload_url,
                encrypted_artifact["ciphertext"],
                {str(key): str(value) for key, value in upload_headers.items()},
            )
        return {
            "result_artifact_key": str(artifact["result_artifact_key"]),
            "result_artifact_encryption": artifact.get(
                "result_artifact_encryption", encrypted_artifact["encryption"]
            ),
        }

    def _upload_sharded_result_artifact(
        self,
        artifact: dict[str, Any],
        encrypted_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        manifest_upload = artifact.get("manifest_upload")
        shards = artifact.get("shards")
        if not isinstance(manifest_upload, dict) or not isinstance(shards, list) or not shards:
            raise ValueError("sharded result artifact upload plan is incomplete")

        ciphertext = encrypted_artifact["ciphertext"]
        manifest_shards: list[dict[str, Any]] = []
        for raw_shard in shards:
            if not isinstance(raw_shard, dict):
                raise ValueError("sharded result artifact upload plan contains an invalid shard")
            index = int(raw_shard["index"])
            offset = int(raw_shard["offset_bytes"])
            size = int(raw_shard["size_bytes"])
            upload_url = raw_shard.get("upload_url")
            if not isinstance(upload_url, str):
                raise ValueError("sharded result artifact shard is missing an upload URL")
            shard_bytes = ciphertext[offset : offset + size]
            if len(shard_bytes) != size:
                raise ValueError("sharded result artifact upload plan does not match encrypted payload size")
            self.transport.put_content(upload_url, shard_bytes, self._upload_headers(raw_shard.get("upload_headers")))
            manifest_shards.append(
                {
                    "index": index,
                    "result_artifact_key": str(raw_shard["result_artifact_key"]),
                    "offset_bytes": offset,
                    "size_bytes": size,
                    "ciphertext_sha256": hashlib.sha256(shard_bytes).hexdigest(),
                }
            )

        manifest_payload = {
            "artifact_kind": "provider_execution_result",
            "artifact_format": "sharded_aes_256_gcm_json",
            "version": 1,
            "payload_encryption": encrypted_artifact["encryption"],
            "payload_plaintext_sha256": encrypted_artifact["plaintext_sha256"],
            "payload_ciphertext_sha256": encrypted_artifact["ciphertext_sha256"],
            "payload_ciphertext_bytes": len(ciphertext),
            "shard_size_bytes": int(artifact.get("shard_size_bytes") or 0),
            "total_shards": len(manifest_shards),
            "shards": manifest_shards,
        }
        encrypted_manifest = encrypt_artifact(manifest_payload)
        manifest_upload_url = manifest_upload.get("upload_url")
        if not isinstance(manifest_upload_url, str):
            raise ValueError("sharded result artifact manifest is missing an upload URL")
        self.transport.put_content(
            manifest_upload_url,
            encrypted_manifest["ciphertext"],
            self._upload_headers(manifest_upload.get("upload_headers")),
        )

        return {
            "result_artifact_key": str(manifest_upload["result_artifact_key"]),
            "result_artifact_encryption": encrypted_manifest["encryption"],
            "result_artifact_format": "sharded",
            "result_artifact_size_bytes": len(ciphertext),
            "result_artifact_ciphertext_sha256": encrypted_artifact["ciphertext_sha256"],
            "result_artifact_plaintext_sha256": encrypted_artifact["plaintext_sha256"],
            "result_artifact_shards": manifest_shards,
        }

    def _upload_result_artifact_with_retry(
        self,
        assignment_id: str,
        encrypted_artifact: dict[str, Any],
        artifact: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        current_plan = artifact
        for attempt in range(1, ARTIFACT_UPLOAD_MAX_ATTEMPTS + 1):
            try:
                if isinstance(current_plan.get("manifest_upload"), dict) or isinstance(current_plan.get("shards"), list):
                    return self._upload_sharded_result_artifact(current_plan, encrypted_artifact)
                return self._upload_single_result_artifact(current_plan, encrypted_artifact)
            except Exception as error:
                last_error = error
                if attempt >= ARTIFACT_UPLOAD_MAX_ATTEMPTS or not self._is_retryable_artifact_upload_error(error):
                    break
                current_plan = self._request_result_artifact_upload_plan(assignment_id, encrypted_artifact)
        if isinstance(last_error, ArtifactFlowError):
            raise last_error
        if isinstance(last_error, httpx.HTTPStatusError):
            status = last_error.response.status_code
            raise ArtifactFlowError(
                "result_artifact_upload_failed",
                f"Result artifact upload failed with HTTP {status}.",
                retryable=status in {401, 403, 404, 408, 409, 425, 429, 500, 502, 503, 504},
            ) from last_error
        if isinstance(last_error, httpx.TransportError):
            raise ArtifactFlowError(
                "result_artifact_upload_network_error",
                str(last_error),
                retryable=True,
            ) from last_error
        raise ArtifactFlowError(
            "result_artifact_upload_failed",
            str(last_error) if last_error else "Result artifact upload failed.",
            retryable=False,
        ) from last_error

    def complete_assignment(
        self,
        assignment_id: str,
        item_results: list[dict[str, Any]],
        runtime_receipt: dict[str, Any] | None = None,
    ) -> None:
        result_payload = {"item_results": item_results}
        encrypted_artifact = encrypt_artifact(result_payload)
        try:
            artifact = self._request_result_artifact_upload_plan(assignment_id, encrypted_artifact)
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {404, 405, 501}:
                raise
            artifact = self.transport.post_content(
                f"/nodes/assignments/{assignment_id}/result-artifact",
                encrypted_artifact["ciphertext"],
                headers={
                    "content-type": "application/octet-stream",
                    **self._node_identity_headers(),
                    "x-artifact-ciphertext-sha256": encrypted_artifact["ciphertext_sha256"],
                    "x-artifact-plaintext-sha256": encrypted_artifact["plaintext_sha256"],
                    "x-artifact-key-b64": encrypted_artifact["encryption"]["key_b64"],
                    "x-artifact-iv-b64": encrypted_artifact["encryption"]["iv_b64"],
                },
            )
        result_artifact = self._upload_result_artifact_with_retry(assignment_id, encrypted_artifact, artifact)

        self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/complete",
            {
                **self._node_identity_payload(),
                "runtime_receipt": runtime_receipt,
                "result_artifact": result_artifact,
            },
        )

    def fail_assignment(self, assignment_id: str, code: str, message: str, retryable: bool = True) -> None:
        self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/fail",
            {
                **self._node_identity_payload(),
                "error": {"code": code, "message": message, "retryable": retryable},
            },
        )

    def fetch_artifact(self, assignment: AssignmentEnvelope) -> dict[str, Any]:
        content = self._fetch_artifact_content(assignment)
        ciphertext_sha256 = hashlib.sha256(content).hexdigest()
        if ciphertext_sha256 != assignment.input_artifact_sha256:
            self._clear_artifact_resume_state(assignment)
            raise ArtifactFlowError(
                "input_artifact_integrity_failed",
                "input artifact integrity check failed: "
                f"expected {assignment.input_artifact_sha256}, got {ciphertext_sha256}",
                retryable=False,
            )
        self._clear_artifact_resume_state(assignment)
        return decrypt_artifact(content, assignment.input_artifact_encryption)

    def _refresh_assignment_input_artifact(self, assignment: AssignmentEnvelope) -> None:
        payload = self.refresh_input_artifact_url(assignment.assignment_id)
        url = payload.get("input_artifact_url")
        if not isinstance(url, str) or not url:
            raise ArtifactFlowError(
                "input_artifact_refresh_failed",
                "Control plane did not return a refreshed input artifact URL.",
                retryable=True,
            )
        assignment.input_artifact_url = url
        expires_at = payload.get("input_artifact_expires_at")
        assignment.input_artifact_expires_at = expires_at if isinstance(expires_at, str) else None

    def _artifact_resume_path(self, assignment: AssignmentEnvelope) -> Path:
        base_path = Path(
            getattr(self.settings, "autopilot_state_path", "")
            or getattr(self.settings, "credentials_path", "")
            or "."
        )
        resume_root = (base_path.parent if base_path.suffix else base_path) / "artifact-resume"
        resume_root.mkdir(parents=True, exist_ok=True)
        assignment_id = str(getattr(assignment, "assignment_id", "") or "").strip()
        if not assignment_id:
            assignment_id = hashlib.sha256(str(getattr(assignment, "input_artifact_url", "")).encode("utf-8")).hexdigest()[:16]
        artifact_hash = str(getattr(assignment, "input_artifact_sha256", "") or "").strip()[:16] or "artifact"
        return resume_root / f"{assignment_id}-{artifact_hash}.part"

    def _clear_artifact_resume_state(self, assignment: AssignmentEnvelope) -> None:
        resume_path = self._artifact_resume_path(assignment)
        try:
            if resume_path.exists():
                resume_path.unlink()
        except OSError:
            return

    def _transport_supports_streaming(self) -> bool:
        return callable(getattr(self.transport.client, "stream", None))

    def _transport_get_content_is_default(self) -> bool:
        getter = getattr(self.transport, "get_content", None)
        return getattr(getter, "__func__", None) is EdgeControlTransport.get_content

    def _download_artifact_with_resume(self, assignment: AssignmentEnvelope) -> bytes:
        return self._download_artifact_from_url_with_resume(assignment, assignment.input_artifact_url)

    def _download_artifact_from_url_with_resume(self, assignment: AssignmentEnvelope, url: str) -> bytes:
        resume_path = self._artifact_resume_path(assignment)
        resume_from = resume_path.stat().st_size if resume_path.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else None
        stream = getattr(self.transport.client, "stream", None)
        if not callable(stream):
            if resume_from > 0 and resume_path.exists():
                try:
                    resume_path.unlink()
                except OSError:
                    pass
            return self.transport.get_content(url)

        with stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and resume_from > 0:
                try:
                    resume_path.unlink()
                except OSError:
                    pass
                raise ArtifactFlowError(
                    "input_artifact_resume_reset",
                    "The partial input artifact no longer matches the remote object, so the download will restart.",
                    retryable=True,
                )
            response.raise_for_status()
            append_mode = response.status_code == 206 and resume_from > 0
            file_mode = "ab" if append_mode else "wb"
            with resume_path.open(file_mode) as handle:
                wrote_any_chunk = False
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
                        wrote_any_chunk = True
                    if wrote_any_chunk and self.transport.faults.consume("partial_artifact_download"):
                        raise httpx.ReadError("Simulated network reset during partial artifact download.")
                    if wrote_any_chunk and self.transport.faults.consume("cache_write_interrupt"):
                        raise httpx.ReadError("Simulated power loss while writing the local cache.")
            return resume_path.read_bytes()

    def _artifact_candidate_urls(self, assignment: AssignmentEnvelope) -> list[str]:
        candidates: list[str] = []
        primary_url = str(getattr(assignment, "input_artifact_url", "") or "").strip()
        if primary_url:
            candidates.append(primary_url)
        mirror_urls = getattr(assignment, "input_artifact_mirror_urls", None)
        if isinstance(mirror_urls, list):
            for raw_url in mirror_urls:
                candidate = str(raw_url or "").strip()
                if candidate:
                    candidates.append(candidate)
        relay_bases = _csv_urls(getattr(self.settings, "artifact_mirror_base_urls", None))
        parsed_primary = urlparse(primary_url) if primary_url else None
        if parsed_primary and parsed_primary.scheme and parsed_primary.netloc:
            relative_path = parsed_primary.path.lstrip("/")
            for relay_base in relay_bases:
                relay_url = urljoin(f"{relay_base.rstrip('/')}/", relative_path)
                if parsed_primary.query:
                    relay_url = f"{relay_url}?{parsed_primary.query}"
                candidates.append(relay_url)
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    def _fetch_artifact_content(self, assignment: AssignmentEnvelope) -> bytes:
        if self._artifact_url_expires_soon(getattr(assignment, "input_artifact_expires_at", None)):
            self._refresh_assignment_input_artifact(assignment)
        refreshed_after_auth_error = False
        candidate_urls = self._artifact_candidate_urls(assignment)
        last_status_error: httpx.HTTPStatusError | None = None
        last_network_error: httpx.TransportError | None = None
        for delay_seconds in (*ARTIFACT_FETCH_RETRY_DELAYS_SECONDS, None):
            retry_round = False
            for candidate_url in candidate_urls:
                try:
                    if self._artifact_resume_path(assignment).exists() or (
                        self._transport_supports_streaming() and self._transport_get_content_is_default()
                    ):
                        return self._download_artifact_from_url_with_resume(assignment, candidate_url)
                    return self.transport.get_content(candidate_url)
                except ArtifactFlowError as error:
                    if error.code == "input_artifact_resume_reset":
                        retry_round = True
                        continue
                    raise
                except httpx.HTTPStatusError as exc:
                    last_status_error = exc
                    status = exc.response.status_code
                    if candidate_url == assignment.input_artifact_url and status in {401, 403} and not refreshed_after_auth_error:
                        refreshed_after_auth_error = True
                        self._refresh_assignment_input_artifact(assignment)
                        candidate_urls = self._artifact_candidate_urls(assignment)
                        retry_round = True
                        break
                    retryable_status = status in ARTIFACT_FETCH_RETRY_STATUSES
                    if status in {404, 410, 422} and candidate_url != candidate_urls[-1]:
                        continue
                    if retryable_status and candidate_url != candidate_urls[-1]:
                        retry_round = True
                        continue
                    if delay_seconds is not None and retryable_status:
                        retry_round = True
                        break
                    if status == 404:
                        raise ArtifactFlowError(
                            "input_artifact_missing",
                            f"{candidate_url} returned HTTP 404 after retries.",
                            retryable=False,
                        ) from exc
                    if status in {410, 422}:
                        raise ArtifactFlowError(
                            "input_artifact_unavailable",
                            f"{candidate_url} returned HTTP {status}.",
                            retryable=False,
                        ) from exc
                    raise ArtifactFlowError(
                        "input_artifact_fetch_failed",
                        f"{candidate_url} returned HTTP {status}.",
                        retryable=retryable_status,
                    ) from exc
                except httpx.TransportError as exc:
                    last_network_error = exc
                    if candidate_url != candidate_urls[-1]:
                        retry_round = True
                        continue
                    if delay_seconds is not None:
                        retry_round = True
                        break
            if retry_round and delay_seconds is not None:
                time.sleep(delay_seconds)
                continue
            if retry_round:
                break
        if last_status_error is not None:
            status = last_status_error.response.status_code
            raise ArtifactFlowError(
                "input_artifact_fetch_failed",
                f"{assignment.input_artifact_url} returned HTTP {status}.",
                retryable=status in ARTIFACT_FETCH_RETRY_STATUSES,
            ) from last_status_error
        if last_network_error is not None:
            raise ArtifactFlowError(
                "input_artifact_network_error",
                f"Network error fetching {assignment.input_artifact_url}.",
                retryable=True,
            ) from last_network_error
        raise ArtifactFlowError("input_artifact_fetch_failed", "input artifact fetch retry loop exhausted", retryable=True)


def decrypt_artifact(payload: bytes, encryption: dict[str, str]) -> dict[str, Any]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - packaged in container when needed
        raise RuntimeError("cryptography is required for artifact decryption") from exc

    key = base64.b64decode(encryption["key_b64"])
    iv = base64.b64decode(encryption["iv_b64"])
    aes = AESGCM(key)
    decrypted = aes.decrypt(iv, payload, None)
    return json.loads(decrypted.decode("utf-8"))


def encrypt_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - packaged in container when needed
        raise RuntimeError("cryptography is required for artifact encryption") from exc

    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    key = AESGCM.generate_key(bit_length=256)
    iv = os.urandom(12)
    aes = AESGCM(key)
    ciphertext = aes.encrypt(iv, plaintext, None)
    return {
        "ciphertext": ciphertext,
        "encryption": {
            "key_b64": base64.b64encode(key).decode("ascii"),
            "iv_b64": base64.b64encode(iv).decode("ascii"),
        },
        "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
    }
