from __future__ import annotations

import base64
import json
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .config import AssignmentEnvelope, NodeAgentSettings, NodeClaimPollResult, NodeClaimSession


class EdgeControlClient:
    def __init__(self, settings: NodeAgentSettings):
        self.settings = settings
        self.client = httpx.Client(base_url=settings.edge_control_url, timeout=30.0)

    def _load_persisted_credentials(self) -> tuple[str, str] | None:
        credentials_path = Path(self.settings.credentials_path)
        if not credentials_path.exists():
            return None

        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        node_id = payload.get("node_id")
        node_key = payload.get("node_key")
        if not isinstance(node_id, str) or not isinstance(node_key, str):
            raise RuntimeError(f"invalid credentials file at {credentials_path}")

        self.settings.node_id = node_id
        self.settings.node_key = node_key
        return node_id, node_key

    def _persist_credentials(self, node_id: str, node_key: str) -> None:
        credentials_path = Path(self.settings.credentials_path)
        credentials_path.parent.mkdir(parents=True, exist_ok=True)
        credentials_path.write_text(
            json.dumps({"node_id": node_id, "node_key": node_key}, indent=2),
            encoding="utf-8",
        )
        self.clear_recovery_note()

    def _interactive_terminal_available(self) -> bool:
        return sys.stdin.isatty() and sys.stdout.isatty()

    def _node_request_payload(self) -> dict[str, Any]:
        return {
            "label": self.settings.node_label,
            "region": self.settings.node_region,
            "trust_tier": self.settings.trust_tier,
            "restricted_capable": self.settings.restricted_capable,
            "capabilities": {
                "supported_models": [model.strip() for model in self.settings.supported_models.split(",") if model.strip()],
                "operations": ["responses", "embeddings"],
                "gpu_name": self.settings.gpu_name,
                "gpu_memory_gb": self.settings.gpu_memory_gb,
                "max_context_tokens": self.settings.max_context_tokens,
                "max_batch_tokens": self.settings.max_batch_tokens,
                "max_concurrent_assignments": self.settings.max_concurrent_assignments,
                "thermal_headroom": self.settings.thermal_headroom,
            },
            "runtime": {
                "agent_version": self.settings.agent_version,
                "vllm_base_url": self.settings.vllm_base_url,
                "docker_image": self.settings.docker_image,
            },
        }

    @staticmethod
    def _format_remaining_time(expires_at: str) -> str:
        remaining_seconds = max(
            0,
            int(datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp() - datetime.now(timezone.utc).timestamp()),
        )
        minutes, seconds = divmod(remaining_seconds, 60)
        if minutes >= 10:
            return f"{minutes} min remaining"
        if minutes > 0:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def _print_claim_instructions(self, claim: NodeClaimSession) -> None:
        print()
        print("AUTONOMOUSc Edge Node Claim")
        print("---------------------------")
        print("Open the approval URL in your browser, sign in as an existing operator, and claim this node.")
        print(f"Claim code: {claim.claim_code}")
        print(f"Approval URL: {claim.approval_url}")
        print(f"Claim expires at: {claim.expires_at}")
        try:
            opened = webbrowser.open(claim.approval_url, new=2)
        except Exception:
            opened = False
        if opened:
            print("A browser tab was opened for you. If it did not appear, copy the approval URL above.")
        else:
            print("If your browser does not open automatically, copy the approval URL above into a browser on this device.")
        print("Waiting for browser approval...")
        print()

    def has_credentials(self) -> bool:
        if self.settings.node_id and self.settings.node_key:
            return True
        return self._load_persisted_credentials() is not None

    def require_credentials(self) -> tuple[str, str]:
        if self.settings.node_id and self.settings.node_key:
            return self.settings.node_id, self.settings.node_key
        persisted = self._load_persisted_credentials()
        if persisted:
            return persisted
        raise RuntimeError(
            "No stored node credentials were found. Run `node-agent bootstrap` from an interactive terminal to claim this node."
        )

    def clear_credentials(self) -> None:
        credentials_path = Path(self.settings.credentials_path)
        self.settings.node_id = None
        self.settings.node_key = None
        if credentials_path.exists():
            credentials_path.unlink()

    def write_recovery_note(self, message: str) -> None:
        note_path = Path(self.settings.recovery_note_path)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(message + "\n", encoding="utf-8")

    def clear_recovery_note(self) -> None:
        note_path = Path(self.settings.recovery_note_path)
        if note_path.exists():
            note_path.unlink()

    @staticmethod
    def is_auth_error(error: Exception) -> bool:
        return isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {401, 403}

    def _persist_from_response(self, payload: dict[str, Any]) -> tuple[str, str]:
        node_id = str(payload["node_id"])
        node_key = str(payload["node_key"])
        self.settings.node_id = node_id
        self.settings.node_key = node_key
        self._persist_credentials(node_id, node_key)
        return node_id, node_key

    def enroll_if_needed(self) -> tuple[str, str]:
        if self.settings.node_id and self.settings.node_key:
            return self.settings.node_id, self.settings.node_key
        persisted = self._load_persisted_credentials()
        if persisted:
            return persisted
        if not self.settings.operator_token:
            raise RuntimeError("operator_token is required for legacy node enrollment")

        response = self.client.post(
            "/nodes/enroll",
            json={
                "operator_token": self.settings.operator_token,
                **self._node_request_payload(),
            },
        )
        response.raise_for_status()
        return self._persist_from_response(response.json())

    def create_node_claim_session(self) -> NodeClaimSession:
        response = self.client.post("/node-claims", json=self._node_request_payload())
        response.raise_for_status()
        return NodeClaimSession.model_validate(response.json())

    def poll_node_claim_session(self, claim_id: str, poll_token: str) -> NodeClaimPollResult:
        response = self.client.post(f"/node-claims/{claim_id}/poll", json={"poll_token": poll_token})
        response.raise_for_status()
        return NodeClaimPollResult.model_validate(response.json())

    def bootstrap(self, interactive: bool = True) -> tuple[str, str]:
        if self.settings.node_id and self.settings.node_key:
            return self.settings.node_id, self.settings.node_key
        persisted = self._load_persisted_credentials()
        if persisted:
            return persisted
        if self.settings.operator_token:
            return self.enroll_if_needed()
        if not interactive or not self._interactive_terminal_available():
            raise RuntimeError(
                "No stored node credentials were found. Run `node-agent bootstrap` from an interactive terminal to claim this node."
            )

        claim = self.create_node_claim_session()
        self._print_claim_instructions(claim)
        last_status: str | None = None
        last_remaining: str | None = None

        while True:
            result = self.poll_node_claim_session(claim.claim_id, claim.poll_token)
            if result.node_id and result.node_key:
                print("Claim approved. Storing node credentials and finishing bootstrap...")
                return self._persist_from_response(result.model_dump())
            if result.status == "expired":
                raise RuntimeError("Node claim expired before approval completed. Run `node-agent bootstrap` again.")
            remaining = self._format_remaining_time(result.expires_at)
            if result.status != last_status or remaining != last_remaining:
                print(f"Status: waiting for operator login and claim approval. Time remaining: {remaining}.")
                last_status = result.status
                last_remaining = remaining
            time.sleep(claim.poll_interval_seconds)

    def attest(self) -> None:
        if not self.settings.node_id or not self.settings.node_key:
            raise RuntimeError("node must be enrolled before attestation")
        response = self.client.post(
            "/nodes/attest",
            json={
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "tpm_quote": f"simulated-{self.settings.node_id}",
                "measurements": {"pcr0": "simulated", "pcr7": "simulated"},
                "inventory": {"gpu_name": self.settings.gpu_name, "driver": "simulated"},
                "restricted_capable": self.settings.restricted_capable,
            },
        )
        response.raise_for_status()

    def heartbeat(self, queue_depth: int = 0, active_assignments: int = 0) -> None:
        response = self.client.post(
            "/nodes/heartbeat",
            json={
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "status": "active",
                "queue_depth": queue_depth,
                "active_assignments": active_assignments,
                "runtime": {"agent_version": self.settings.agent_version},
            },
        )
        response.raise_for_status()
        self.clear_recovery_note()

    def pull_assignment(self) -> AssignmentEnvelope | None:
        response = self.client.post(
            "/nodes/assignments/pull",
            json={
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
            },
        )
        response.raise_for_status()
        assignment = response.json().get("assignment")
        return AssignmentEnvelope.model_validate(assignment) if assignment else None

    def accept_assignment(self, assignment_id: str) -> None:
        response = self.client.post(
            f"/nodes/assignments/{assignment_id}/accept",
            json={"node_id": self.settings.node_id, "node_key": self.settings.node_key},
        )
        response.raise_for_status()

    def report_progress(self, assignment_id: str, progress: dict[str, Any]) -> None:
        response = self.client.post(
            f"/nodes/assignments/{assignment_id}/progress",
            json={
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "progress": progress,
            },
        )
        response.raise_for_status()

    def complete_assignment(self, assignment_id: str, item_results: list[dict[str, Any]]) -> None:
        response = self.client.post(
            f"/nodes/assignments/{assignment_id}/complete",
            json={
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "item_results": item_results,
            },
        )
        response.raise_for_status()

    def fail_assignment(self, assignment_id: str, code: str, message: str, retryable: bool = True) -> None:
        response = self.client.post(
            f"/nodes/assignments/{assignment_id}/fail",
            json={
                "node_id": self.settings.node_id,
                "node_key": self.settings.node_key,
                "error": {"code": code, "message": message, "retryable": retryable},
            },
        )
        response.raise_for_status()

    def fetch_artifact(self, assignment: AssignmentEnvelope) -> dict[str, Any]:
        response = self.client.get(assignment.input_artifact_url)
        response.raise_for_status()
        return decrypt_artifact(response.content, assignment.input_artifact_encryption)


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
