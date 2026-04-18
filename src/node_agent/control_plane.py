from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from .config import AssignmentEnvelope, NodeAgentSettings, NodeClaimPollResult, NodeClaimSession
from .concurrency import resolved_embeddings_concurrency_limit, resolved_embeddings_microbatch_assignment_limit
from .control_plane_bootstrap import NodeBootstrapOrchestrator
from .control_plane_store import NodeCredentialStore
from .control_plane_transport import EdgeControlTransport
from .gguf_artifacts import resolved_default_gguf_artifact_contract
from .runtime_quality import resolved_runtime_quality
from .runtime_evidence import resolved_signed_runtime_evidence
from .runtime_tuple import resolved_default_runtime_tuple


ARTIFACT_FETCH_RETRY_STATUSES = {404, 408, 409, 425, 429, 500, 502, 503, 504}
ARTIFACT_FETCH_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)


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
        payload = {
            "supported_models": [model.strip() for model in self.settings.supported_models.split(",") if model.strip()],
            "operations": list(runtime_profile.supported_apis),
            "gpu_name": self.settings.gpu_name,
            "gpu_memory_gb": self.settings.gpu_memory_gb,
            "max_context_tokens": self.settings.max_context_tokens,
            "max_batch_tokens": self.settings.max_batch_tokens,
            "max_concurrent_assignments": self.settings.max_concurrent_assignments,
            "max_pull_bundle_assignments": max(
                self.settings.max_concurrent_assignments,
                embedding_microbatch_limit or self.settings.max_concurrent_assignments,
            ),
            "thermal_headroom": self.settings.thermal_headroom,
            "heat_demand": self.settings.heat_demand,
        }
        if embedding_concurrency_limit is not None:
            payload["max_concurrent_assignments_embeddings"] = embedding_concurrency_limit
        if embedding_microbatch_limit is not None:
            payload["max_microbatch_assignments_embeddings"] = embedding_microbatch_limit
        for key in (
            "room_temp_c",
            "target_temp_c",
            "gpu_temp_c",
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
        capabilities: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        autopilot: dict[str, Any] | None = None,
        include_capabilities: bool = True,
        include_runtime: bool = True,
    ) -> None:
        payload: dict[str, Any] = {
            "node_id": self.settings.node_id,
            "node_key": self.settings.node_key,
            "status": "active",
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
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
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
            {"node_id": self.settings.node_id, "node_key": self.settings.node_key},
        )

    def report_progress(self, assignment_id: str, progress: dict[str, Any]) -> None:
        self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/progress",
            {
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "progress": progress,
            },
        )

    @staticmethod
    def _upload_headers(upload_headers: Any) -> dict[str, str]:
        if not isinstance(upload_headers, dict):
            raise ValueError("artifact upload plan is missing upload headers")
        return {str(key): str(value) for key, value in upload_headers.items()}

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

    def complete_assignment(
        self,
        assignment_id: str,
        item_results: list[dict[str, Any]],
        runtime_receipt: dict[str, Any] | None = None,
    ) -> None:
        node_id, node_key = self.require_credentials()
        result_payload = {"item_results": item_results}
        encrypted_artifact = encrypt_artifact(result_payload)
        try:
            artifact = self.transport.post_json(
                f"/nodes/assignments/{assignment_id}/result-artifact/upload-url",
                {
                    "node_id": node_id,
                    "node_key": node_key,
                    "ciphertext_sha256": encrypted_artifact["ciphertext_sha256"],
                    "plaintext_sha256": encrypted_artifact["plaintext_sha256"],
                    "content_length_bytes": len(encrypted_artifact["ciphertext"]),
                },
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {404, 405, 501}:
                raise
            artifact = self.transport.post_content(
                f"/nodes/assignments/{assignment_id}/result-artifact",
                encrypted_artifact["ciphertext"],
                headers={
                    "content-type": "application/octet-stream",
                    "x-node-id": node_id,
                    "x-node-key": node_key,
                    "x-artifact-ciphertext-sha256": encrypted_artifact["ciphertext_sha256"],
                    "x-artifact-plaintext-sha256": encrypted_artifact["plaintext_sha256"],
                    "x-artifact-key-b64": encrypted_artifact["encryption"]["key_b64"],
                    "x-artifact-iv-b64": encrypted_artifact["encryption"]["iv_b64"],
                },
            )

        upload_url = artifact.get("upload_url")
        upload_headers = artifact.get("upload_headers")
        if isinstance(artifact.get("manifest_upload"), dict) or isinstance(artifact.get("shards"), list):
            result_artifact = self._upload_sharded_result_artifact(artifact, encrypted_artifact)
        elif isinstance(upload_url, str) and isinstance(upload_headers, dict):
            self.transport.put_content(
                upload_url,
                encrypted_artifact["ciphertext"],
                {str(key): str(value) for key, value in upload_headers.items()},
            )
            result_artifact = {
                "result_artifact_key": str(artifact["result_artifact_key"]),
                "result_artifact_encryption": artifact.get(
                    "result_artifact_encryption", encrypted_artifact["encryption"]
                ),
            }
        else:
            result_artifact = {
                "result_artifact_key": str(artifact["result_artifact_key"]),
                "result_artifact_encryption": artifact.get(
                    "result_artifact_encryption", encrypted_artifact["encryption"]
                ),
            }

        self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/complete",
            {
                "node_id": node_id,
                "node_key": node_key,
                "runtime_receipt": runtime_receipt,
                "result_artifact": result_artifact,
            },
        )

    def fail_assignment(self, assignment_id: str, code: str, message: str, retryable: bool = True) -> None:
        self.transport.post_json(
            f"/nodes/assignments/{assignment_id}/fail",
            {
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "error": {"code": code, "message": message, "retryable": retryable},
            },
        )

    def fetch_artifact(self, assignment: AssignmentEnvelope) -> dict[str, Any]:
        content = self._fetch_artifact_content(assignment.input_artifact_url)
        ciphertext_sha256 = hashlib.sha256(content).hexdigest()
        if ciphertext_sha256 != assignment.input_artifact_sha256:
            raise ValueError(
                "input artifact integrity check failed: "
                f"expected {assignment.input_artifact_sha256}, got {ciphertext_sha256}"
            )
        return decrypt_artifact(content, assignment.input_artifact_encryption)

    def _fetch_artifact_content(self, url: str) -> bytes:
        for attempt, delay_seconds in enumerate((*ARTIFACT_FETCH_RETRY_DELAYS_SECONDS, None)):
            try:
                return self.transport.get_content(url)
            except httpx.HTTPStatusError as exc:
                retryable_status = exc.response.status_code in ARTIFACT_FETCH_RETRY_STATUSES
                if delay_seconds is None or not retryable_status:
                    raise
            except httpx.TransportError:
                if delay_seconds is None:
                    raise
            time.sleep(delay_seconds)
        raise RuntimeError("input artifact fetch retry loop exhausted")


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
