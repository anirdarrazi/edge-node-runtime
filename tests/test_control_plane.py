import base64
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from node_agent.config import NodeAgentSettings
from node_agent.control_plane import EdgeControlClient
from node_agent.gguf_artifacts import find_gguf_artifact
from node_agent.model_artifacts import find_model_artifact


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class RecordingClient:
    def __init__(self):
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        return DummyResponse({"node_id": "node_123", "node_key": "key_123456789012345678901234"})


class FailIfCalledClient:
    def post(self, path, json):
        raise AssertionError(f"unexpected network call: {path} {json}")


class PullClient:
    def __init__(self):
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        return DummyResponse({"assignment": None, "queue_depth": 3})


def build_settings(credentials_path: Path, operator_token: str | None):
    return NodeAgentSettings(
        edge_control_url="http://localhost:8787",
        vllm_base_url="http://localhost:8000",
        credentials_path=str(credentials_path),
        operator_token=operator_token,
    )


def test_enroll_persists_and_restores_credentials(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"

    settings = build_settings(credentials_path, operator_token="operator_token")
    client = EdgeControlClient(settings)
    recording_client = RecordingClient()
    client.client = recording_client

    node_id, node_key = client.enroll_if_needed()

    assert node_id == "node_123"
    assert node_key == "key_123456789012345678901234"
    assert credentials_path.exists()
    assert len(recording_client.calls) == 1

    restored_settings = build_settings(credentials_path, operator_token=None)
    restored_client = EdgeControlClient(restored_settings)
    restored_client.client = FailIfCalledClient()

    restored_node_id, restored_node_key = restored_client.enroll_if_needed()

    assert restored_node_id == "node_123"
    assert restored_node_key == "key_123456789012345678901234"


class ClaimClient:
    def __init__(self):
        self.poll_count = 0
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        if path == "/node-claims":
            return DummyResponse(
                {
                    "claim_id": "claim_123",
                    "claim_code": "ABC123",
                    "approval_url": "https://ai.autonomousc.com/?claim_id=claim_123&claim_token=approval-token",
                    "poll_token": "poll-token",
                    "expires_at": "2099-01-01T00:00:00Z",
                    "poll_interval_seconds": 0,
                }
            )

        self.poll_count += 1
        if self.poll_count == 1:
            return DummyResponse({"status": "pending", "expires_at": "2099-01-01T00:00:00Z"})

        return DummyResponse(
            {
                "status": "consumed",
                "expires_at": "2099-01-01T00:00:00Z",
                "node_id": "node_claimed",
                "node_key": "key_claimed_12345678901234567890",
            }
        )


class ConsumedWithoutCredentialsClaimClient:
    def __init__(self):
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        if path == "/node-claims":
            return DummyResponse(
                {
                    "claim_id": "claim_123",
                    "claim_code": "ABC123",
                    "approval_url": "https://ai.autonomousc.com/?claim_id=claim_123&claim_token=approval-token",
                    "poll_token": "poll-token",
                    "expires_at": "2099-01-01T00:00:00Z",
                    "poll_interval_seconds": 0,
                }
            )
        return DummyResponse({"status": "consumed", "expires_at": "2099-01-01T00:00:00Z"})


def test_bootstrap_claims_node_via_browser_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token=None)
    client = EdgeControlClient(settings)
    client.client = ClaimClient()

    monkeypatch.setattr("node_agent.control_plane.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("node_agent.control_plane.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("node_agent.control_plane.time.sleep", lambda _seconds: None)

    node_id, node_key = client.bootstrap(interactive=True)

    assert node_id == "node_claimed"
    assert node_key == "key_claimed_12345678901234567890"
    assert credentials_path.exists()


def test_bootstrap_fails_fast_when_claim_is_consumed_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token=None)
    client = EdgeControlClient(settings)
    client.client = ConsumedWithoutCredentialsClaimClient()

    monkeypatch.setattr("node_agent.control_plane.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("node_agent.control_plane.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("node_agent.control_plane.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="consumed but did not return credentials"):
        client.bootstrap(interactive=True)


def test_bootstrap_requires_interactive_terminal_without_credentials(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token=None)
    client = EdgeControlClient(settings)

    with pytest.raises(RuntimeError, match="Open the setup UI and run Quick Start"):
        client.bootstrap(interactive=False)


def test_require_credentials_points_missing_nodes_back_to_setup_ui(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token=None)
    client = EdgeControlClient(settings)

    with pytest.raises(RuntimeError, match="Open the setup UI and run Quick Start"):
        client.require_credentials()


def test_attest_declares_attestation_provider(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.node_id = "node_123"
    settings.node_key = "key_123456789012345678901234"
    settings.attestation_provider = "simulated"
    settings.attestation_state_path = str(tmp_path / "credentials" / "attestation-state.json")
    client = EdgeControlClient(settings)
    recording_client = RecordingClient()
    client.client = recording_client

    client.attest()

    path, payload = recording_client.calls[0]
    assert path == "/nodes/attest"
    assert payload["measurements"]["attestation_provider"] == "simulated"
    assert payload["inventory"]["attestation_provider"] == "simulated"
    assert Path(settings.attestation_state_path).exists()
    persisted = client.load_attestation_state()
    assert persisted is not None
    assert persisted["node_id"] == "node_123"
    assert persisted["attestation_provider"] == "simulated"
    assert persisted["status"] == "verified"


def test_heartbeat_reports_current_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.node_id = "node_123"
    settings.node_key = "key_123456789012345678901234"
    settings.runtime_profile = "partner_vllm_trusted"
    settings.vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
    settings.docker_image = "sha256:" + ("a" * 64)
    settings.gpu_name = "RTX 4090"
    settings.gpu_memory_gb = 24.0
    monkeypatch.setenv("CUDA_VERSION", "12.4")
    monkeypatch.setenv("NVIDIA_DRIVER_VERSION", "555.42.02")
    client = EdgeControlClient(settings)
    recording_client = RecordingClient()
    client.client = recording_client

    client.heartbeat(queue_depth=2, active_assignments=1)

    path, payload = recording_client.calls[0]
    assert path == "/nodes/heartbeat"
    assert payload["runtime"]["current_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert payload["capabilities"]["max_concurrent_assignments"] == settings.max_concurrent_assignments
    evidence = payload["runtime"]["runtime_evidence"]
    assert evidence["digest"].startswith("sha256:")
    assert evidence["signature"].startswith("sha256:")
    assert evidence["signature_algorithm"] == "hmac-sha256"
    assert evidence["bundle"]["runtime_profile"] == payload["runtime"]["runtime_profile"]
    assert evidence["bundle"]["quality_class"] == "exact_audited"
    assert evidence["bundle"]["exactness_class"] == "exact_audited"
    assert evidence["bundle"]["runtime_image_digest"] == settings.docker_image
    assert evidence["bundle"]["runtime_tuple_digest"] == payload["runtime"]["runtime_tuple_digest"]
    assert evidence["bundle"]["startup_args"]["served_model"] == settings.vllm_model
    assert evidence["bundle"]["driver_metadata"]["gpu_name"] == "RTX 4090"
    assert evidence["bundle"]["driver_metadata"]["cuda_version"] == "12.4"
    assert evidence["bundle"]["driver_metadata"]["driver_version"] == "555.42.02"


def test_heartbeat_reports_home_heating_telemetry(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.node_id = "node_123"
    settings.node_key = "key_123456789012345678901234"
    settings.heat_demand = "medium"
    settings.room_temp_c = 19.0
    settings.target_temp_c = 21.0
    settings.gpu_temp_c = 62.0
    settings.power_watts = 280.0
    settings.estimated_heat_output_watts = 275.0
    settings.energy_price_kwh = 0.14
    client = EdgeControlClient(settings)
    recording_client = RecordingClient()
    client.client = recording_client

    client.heartbeat(queue_depth=0, active_assignments=0)

    _path, payload = recording_client.calls[0]
    assert payload["capabilities"]["heat_demand"] == "medium"
    assert payload["capabilities"]["room_temp_c"] == 19.0
    assert payload["capabilities"]["target_temp_c"] == 21.0
    assert payload["capabilities"]["gpu_temp_c"] == 62.0
    assert payload["capabilities"]["power_watts"] == 280.0
    assert payload["capabilities"]["estimated_heat_output_watts"] == 275.0
    assert payload["capabilities"]["energy_price_kwh"] == 0.14


def test_pull_assignment_tracks_control_plane_queue_depth(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.node_id = "node_123"
    settings.node_key = "key_123456789012345678901234"
    client = EdgeControlClient(settings)
    client.client = PullClient()

    assignment = client.pull_assignment()

    assert assignment is None
    assert client.last_control_plane_queue_depth == 3


def test_node_request_payload_uses_bundled_release_metadata_for_the_primary_model(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.runtime_profile = "partner_vllm_trusted"
    settings.vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
    client = EdgeControlClient(settings)

    payload = client._node_request_payload()
    artifact = find_model_artifact("meta-llama/Llama-3.1-8B-Instruct", "responses")

    assert artifact is not None
    assert payload["runtime"]["routing_lane"] == "trusted_exact_partner"
    assert payload["runtime"]["routing_lane_label"] == "Trusted exact partner"
    assert payload["runtime"]["routing_lane_allowed_privacy_tiers"] == ["standard", "confidential", "restricted"]
    assert payload["runtime"]["routing_lane_allowed_result_guarantees"] == [
        "community_best_effort",
        "exact_model_audited",
    ]
    assert payload["runtime"]["routing_lane_allowed_trust_requirements"] == [
        "untrusted_allowed",
        "trusted_only",
    ]
    assert payload["runtime"]["max_privacy_tier"] == "restricted"
    assert payload["runtime"]["exact_model_guarantee"] is True
    assert payload["runtime"]["quantized_output_disclosure_required"] is False
    assert payload["runtime"]["quality_class"] == "exact_audited"
    assert payload["runtime"]["exactness_class"] == "exact_audited"
    assert payload["runtime"]["model_manifest_digest"] == artifact.model_manifest_digest
    assert payload["runtime"]["tokenizer_digest"] == artifact.tokenizer_digest


def test_node_request_payload_marks_vast_as_temporary_burst_capacity(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.runtime_profile = "vast_vllm_safetensors"
    settings.capacity_class = "elastic_burst"
    settings.temporary_node = True
    settings.burst_provider = "vast_ai"
    settings.burst_lease_id = "lease_123"
    settings.burst_lease_phase = "accept_burst_work"
    settings.burst_cost_ceiling_usd = 0.18
    client = EdgeControlClient(settings)

    payload = client._node_request_payload()

    assert payload["runtime"]["capacity_class"] == "elastic_burst"
    assert payload["runtime"]["routing_lane"] == "elastic_exact_vast"
    assert payload["runtime"]["trusted_eligibility"] == "runtime_and_model_digest_match"
    assert payload["runtime"]["quality_class"] == "exact_audited"
    assert payload["runtime"]["exactness_class"] == "exact_audited"
    assert payload["runtime"]["temporary_node"] is True
    assert payload["runtime"]["burst_provider"] == "vast_ai"
    assert payload["runtime"]["burst_lease_id"] == "lease_123"
    assert payload["runtime"]["burst_lease_phase"] == "accept_burst_work"
    assert payload["runtime"]["burst_cost_ceiling_usd"] == 0.18
    assert payload["runtime"]["burst_lifecycle"] == [
        "provision",
        "warm_model",
        "register_temporary_node",
        "accept_burst_work",
        "drain",
        "terminate",
    ]


def test_node_request_payload_uses_gguf_contract_for_home_llama_cpp(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.runtime_profile = "home_llama_cpp_gguf"
    client = EdgeControlClient(settings)

    payload = client._node_request_payload()
    artifact = find_gguf_artifact("meta-llama/Llama-3.1-8B-Instruct", "responses")

    assert artifact is not None
    assert "model_manifest_digest" not in payload["runtime"]
    assert "tokenizer_digest" not in payload["runtime"]
    assert payload["runtime"]["routing_lane"] == "community_quantized_home"
    assert payload["runtime"]["routing_lane_policy_summary"].startswith(
        "Routes home llama.cpp GGUF capacity into the community lane only."
    )
    assert payload["runtime"]["max_privacy_tier"] == "standard"
    assert payload["runtime"]["exact_model_guarantee"] is False
    assert payload["runtime"]["quantized_output_disclosure_required"] is True
    assert payload["runtime"]["quality_class"] == "quantized_economy"
    assert payload["runtime"]["exactness_class"] == "quantized_best_effort"
    assert payload["runtime"]["artifact_manifest_type"] == "gguf_hf_file"
    assert payload["runtime"]["gguf_artifact"]["file_digest"] == artifact.file_digest
    assert payload["runtime"]["gguf_artifact"]["quality_class"] == "quantized_economy"
    assert payload["runtime"]["gguf_artifact"]["exactness_class"] == "quantized_best_effort"
    assert payload["runtime"]["gguf_artifact"]["quantization_type"] == "Q4_K_M"


class DashboardClient:
    def __init__(self):
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        return DummyResponse(
            {
                "node": {
                    "id": "node_123",
                    "label": "Nordic node",
                    "last_heartbeat_at": "2026-04-12T10:00:00Z",
                },
                "earnings": {
                    "accrued_usd": "1.2500",
                    "transferred_usd": "0.5000",
                    "last_payout": None,
                },
                "schedulable": True,
                "blocked_reason": None,
            }
        )


class SupportClient:
    def __init__(self):
        self.calls = []

    def post(self, path, json):
        self.calls.append((path, json))
        return DummyResponse(
            {
                "case_id": "support_123",
                "status": "received",
                "bundle_name": json["bundle_name"],
                "received_at": "2026-04-12T12:00:00Z",
            }
        )


def test_fetch_node_dashboard_summary_uses_stored_credentials(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token=None)
    client = EdgeControlClient(settings)
    client.persist_credentials("node_123", "key_123456789012345678901234")
    dashboard_client = DashboardClient()
    client.client = dashboard_client

    payload = client.fetch_node_dashboard_summary()

    assert payload["node"]["id"] == "node_123"
    assert dashboard_client.calls == [
        (
            "/nodes/dashboard",
            {
                "node_id": "node_123",
                "node_key": "key_123456789012345678901234",
            },
        )
    ]


def test_submit_support_bundle_uses_stored_credentials(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token=None)
    client = EdgeControlClient(settings)
    client.persist_credentials("node_123", "key_123456789012345678901234")
    support_client = SupportClient()
    client.client = support_client

    payload = client.submit_support_bundle(
        "diagnostics-20260412-120000.zip",
        b"zip-bytes",
        generated_at="2026-04-12T11:59:00Z",
    )

    assert payload["case_id"] == "support_123"
    path, body = support_client.calls[0]
    assert path == "/nodes/support-bundles"
    assert body["node_id"] == "node_123"
    assert body["node_key"] == "key_123456789012345678901234"
    assert body["bundle_name"] == "diagnostics-20260412-120000.zip"
    assert body["bundle_size_bytes"] == len(b"zip-bytes")
    assert body["bundle_sha256"] == hashlib.sha256(b"zip-bytes").hexdigest()
    assert body["generated_at"] == "2026-04-12T11:59:00Z"
    assert base64.b64decode(body["bundle_content_base64"]) == b"zip-bytes"


class ArtifactResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class ArtifactClient:
    def __init__(self, content: bytes):
        self.content = content

    def get(self, path: str):
        return ArtifactResponse(self.content)


def test_fetch_artifact_verifies_ciphertext_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    client = EdgeControlClient(build_settings(tmp_path / "credentials" / "node.json", operator_token=None))
    ciphertext = b"ciphertext"
    client.client = ArtifactClient(ciphertext)
    decrypt_calls = []

    def fake_decrypt(payload: bytes, encryption: dict[str, str]):
        decrypt_calls.append((payload, encryption))
        return {"items": []}

    monkeypatch.setattr("node_agent.control_plane.decrypt_artifact", fake_decrypt)

    payload = client.fetch_artifact(
        SimpleNamespace(
            input_artifact_url="https://edge.autonomousc.test/artifacts/pexec_123/input",
            input_artifact_sha256=hashlib.sha256(ciphertext).hexdigest(),
            input_artifact_encryption={"key_b64": "a2V5", "iv_b64": "aXY="},
        )
    )

    assert payload == {"items": []}
    assert decrypt_calls == [(ciphertext, {"key_b64": "a2V5", "iv_b64": "aXY="})]


def test_fetch_artifact_rejects_hash_mismatch_before_decrypt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    client = EdgeControlClient(build_settings(tmp_path / "credentials" / "node.json", operator_token=None))
    client.client = ArtifactClient(b"ciphertext")
    decrypt_called = False

    def fake_decrypt(_payload: bytes, _encryption: dict[str, str]):
        nonlocal decrypt_called
        decrypt_called = True
        return {"items": []}

    monkeypatch.setattr("node_agent.control_plane.decrypt_artifact", fake_decrypt)

    with pytest.raises(ValueError, match="integrity check failed"):
        client.fetch_artifact(
            SimpleNamespace(
                input_artifact_url="https://edge.autonomousc.test/artifacts/pexec_123/input",
                input_artifact_sha256="0" * 64,
                input_artifact_encryption={"key_b64": "a2V5", "iv_b64": "aXY="},
            )
        )

    assert decrypt_called is False


def test_enroll_persists_credentials_with_owner_only_permissions(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.attestation_state_path = str(tmp_path / "credentials" / "attestation-state.json")
    client = EdgeControlClient(settings)
    client.client = RecordingClient()

    client.enroll_if_needed()

    if os.name != "nt":
        assert credentials_path.stat().st_mode & 0o777 == 0o600
        assert credentials_path.parent.stat().st_mode & 0o777 == 0o700


def test_persist_credentials_clears_stale_attestation_state(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.attestation_state_path = str(tmp_path / "credentials" / "attestation-state.json")
    client = EdgeControlClient(settings)
    Path(settings.attestation_state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.attestation_state_path).write_text(
        '{"node_id":"node_old","attestation_provider":"hardware","status":"verified","attested_at":"2026-04-10T00:00:00+00:00"}',
        encoding="utf-8",
    )

    client.persist_credentials("node_123", "key_123456789012345678901234")

    assert Path(settings.attestation_state_path).exists() is False


def test_is_auth_error_only_matches_control_plane_requests(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    client = EdgeControlClient(build_settings(credentials_path, operator_token="operator_token"))

    control_request = httpx.Request("POST", "http://localhost:8787/nodes/heartbeat")
    runtime_request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    control_error = httpx.HTTPStatusError("unauthorized", request=control_request, response=httpx.Response(401, request=control_request))
    runtime_error = httpx.HTTPStatusError("unauthorized", request=runtime_request, response=httpx.Response(401, request=runtime_request))

    assert client.is_auth_error(control_error) is True
    assert client.is_auth_error(runtime_error) is False
