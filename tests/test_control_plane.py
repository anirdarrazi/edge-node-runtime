from pathlib import Path

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
