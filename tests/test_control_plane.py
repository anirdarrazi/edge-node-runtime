from pathlib import Path

import httpx
import pytest

from node_agent.config import NodeAgentSettings
from node_agent.control_plane import EdgeControlClient


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


def test_bootstrap_requires_interactive_terminal_without_credentials(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token=None)
    client = EdgeControlClient(settings)

    with pytest.raises(RuntimeError):
        client.bootstrap(interactive=False)


def test_attest_declares_attestation_provider(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path, operator_token="operator_token")
    settings.node_id = "node_123"
    settings.node_key = "key_123456789012345678901234"
    settings.attestation_provider = "simulated"
    client = EdgeControlClient(settings)
    recording_client = RecordingClient()
    client.client = recording_client

    client.attest()

    path, payload = recording_client.calls[0]
    assert path == "/nodes/attest"
    assert payload["measurements"]["attestation_provider"] == "simulated"
    assert payload["inventory"]["attestation_provider"] == "simulated"


def test_is_auth_error_only_matches_control_plane_requests(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    client = EdgeControlClient(build_settings(credentials_path, operator_token="operator_token"))

    control_request = httpx.Request("POST", "http://localhost:8787/nodes/heartbeat")
    runtime_request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    control_error = httpx.HTTPStatusError("unauthorized", request=control_request, response=httpx.Response(401, request=control_request))
    runtime_error = httpx.HTTPStatusError("unauthorized", request=runtime_request, response=httpx.Response(401, request=runtime_request))

    assert client.is_auth_error(control_error) is True
    assert client.is_auth_error(runtime_error) is False
