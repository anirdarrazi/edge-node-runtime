import json
from contextlib import redirect_stdout
from io import StringIO

import node_agent.staging_smoke as staging_smoke
from node_agent.config import AssignmentEnvelope, NodeClaimPollResult, NodeClaimSession


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeStagingAPI:
    def __init__(self, client_holder: dict[str, "FakeNodeClient"] | None = None) -> None:
        self.client_holder = client_holder if client_holder is not None else {}
        self.bootstrap_calls = 0
        self.approved_claims: list[tuple[str, str, str]] = []
        self.approved_nodes: list[tuple[str, str]] = []
        self.revoked_nodes: list[tuple[str, str]] = []
        self.revoked_api_keys: list[tuple[str, str]] = []
        self.execution_fetch_revoked = False
        self.admission_calls = 0

    def bootstrap_disposable_lane(self, config: staging_smoke.StagingSmokeConfig) -> dict[str, object]:
        self.bootstrap_calls += 1
        return {
            "email": "staging@example.com",
            "organization_id": "org_stage",
            "approval_status": "approved",
            "operator_token": "operator-token",
            "api_key": "provider-api-key",
            "api_key_id": "key_stage",
            "seeded_credit_units": config.seed_credit_units,
        }

    def approve_node_claim(self, claim_id: str, claim_token: str, operator_token: str) -> dict[str, object]:
        self.approved_claims.append((claim_id, claim_token, operator_token))
        return {"status": "approved"}

    def approve_temporary_node(self, node_id: str, operator_token: str) -> dict[str, object]:
        self.approved_nodes.append((node_id, operator_token))
        return {"node": {"id": node_id, "approval_status": "approved"}}

    def create_execution(self, api_key: str, *, model: str, operation: str, region: str) -> dict[str, object]:
        assert api_key == "provider-api-key"
        assert model
        assert operation in {"embeddings", "responses"}
        assert region
        return {
            "admission": {"id": "adm_123", "status": "queued"},
            "execution": None,
        }

    def get_execution_admission(self, admission_id: str, api_key: str) -> dict[str, object]:
        assert admission_id == "adm_123"
        assert api_key == "provider-api-key"
        self.admission_calls += 1
        if self.admission_calls == 1:
            return {"admission": {"id": admission_id, "status": "processing"}, "execution": None}
        return {
            "admission": {"id": admission_id, "status": "accepted"},
            "execution": {"id": "exec_123", "status": "queued", "selected_node_id": None},
        }

    def get_execution(self, execution_id: str, api_key: str) -> dict[str, object]:
        assert execution_id == "exec_123"
        if self.execution_fetch_revoked:
            raise staging_smoke.unauthorized_http_error("http://edge-control/provider/executions/exec_123")
        assert api_key == "provider-api-key"
        return {
            "execution": {
                "id": execution_id,
                "status": "completed",
                "selected_node_id": "node_stage",
                "verification_status": "not_required",
            }
        }

    def revoke_node(self, node_id: str, operator_token: str) -> dict[str, object]:
        self.revoked_nodes.append((node_id, operator_token))
        client = self.client_holder.get("client")
        if client is not None:
            client.revoked = True
        return {"node": {"id": node_id, "status": "revoked"}}

    def revoke_api_key(self, api_key_id: str, operator_token: str) -> dict[str, object]:
        self.revoked_api_keys.append((api_key_id, operator_token))
        self.execution_fetch_revoked = True
        return {"key": {"id": api_key_id, "revoked_at": "2026-04-21T12:00:00Z"}}


class FakeNodeClient:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.revoked = False
        self.accepted: list[str] = []
        self.progress_updates: list[tuple[str, dict[str, object]]] = []
        self.completed: list[tuple[str, list[dict[str, object]]]] = []
        self.heartbeats = 0
        self.assignment = AssignmentEnvelope(
            assignment_id="assign_123",
            execution_id="exec_123",
            assignment_nonce="nonce_123",
            operation="embeddings",
            model="BAAI/bge-large-en-v1.5",
            privacy_tier="standard",
            node_trust_requirement="untrusted_allowed",
            result_guarantee="community_best_effort",
            allowed_regions=["eu-se-1"],
            required_vram_gb=8.0,
            required_context_tokens=512,
            token_budget={"estimated_input_tokens": 8, "estimated_output_tokens": 0, "total_tokens": 8},
            item_count=1,
            input_artifact_url="https://example.com/input",
            input_artifact_sha256="a" * 64,
            input_artifact_encryption={"key_b64": "key", "iv_b64": "iv"},
            input_artifact_mirror_urls=[],
        )

    def create_node_claim_session(self) -> NodeClaimSession:
        return NodeClaimSession(
            claim_id="claim_123",
            claim_code="ABC123",
            approval_url="https://ai.autonomousc.com/claim?claim_token=claim-token-123",
            poll_token="poll-token-123",
            expires_at="2026-04-21T12:30:00Z",
            poll_interval_seconds=1,
        )

    def poll_node_claim_session(self, claim_id: str, poll_token: str) -> NodeClaimPollResult:
        assert claim_id == "claim_123"
        assert poll_token == "poll-token-123"
        return NodeClaimPollResult(
            status="consumed",
            expires_at="2026-04-21T12:30:00Z",
            node_id="node_stage",
            node_key="node-key-stage",
        )

    def attest(self) -> None:
        return None

    def heartbeat(self, queue_depth: int = 0, active_assignments: int = 0, *, status: str = "active", **_: object) -> None:
        if self.revoked:
            raise staging_smoke.unauthorized_http_error("http://edge-control/nodes/heartbeat")
        assert queue_depth >= 0
        assert active_assignments >= 0
        assert status
        self.heartbeats += 1

    def pull_assignments(self, limit: int) -> list[AssignmentEnvelope]:
        assert limit == 1
        return [self.assignment]

    def accept_assignment(self, assignment_id: str) -> None:
        self.accepted.append(assignment_id)

    def report_progress(self, assignment_id: str, progress: dict[str, object]) -> None:
        self.progress_updates.append((assignment_id, progress))

    def fetch_artifact(self, assignment: AssignmentEnvelope) -> dict[str, object]:
        assert assignment.assignment_id == "assign_123"
        return {
            "items": [
                {
                    "batch_item_id": "item-1",
                    "customer_item_id": "customer-item-1",
                }
            ]
        }

    def complete_assignment(
        self,
        assignment_id: str,
        item_results: list[dict[str, object]],
        runtime_receipt: dict[str, object] | None = None,
    ) -> None:
        assert runtime_receipt is None
        self.completed.append((assignment_id, item_results))


def test_runner_executes_full_disposable_lane_and_cleans_up() -> None:
    clock = FakeClock()
    holder: dict[str, FakeNodeClient] = {}
    api = FakeStagingAPI(holder)

    def build_client(settings) -> FakeNodeClient:
        client = FakeNodeClient(settings)
        holder["client"] = client
        return client

    config = staging_smoke.StagingSmokeConfig(edge_control_url="http://edge-control.test")
    report = staging_smoke.StagingSmokeRunner(
        api,
        control_client_factory=build_client,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)
    client = holder["client"]

    assert report["status"] == "ok"
    assert report["bootstrap"]["organization_id"] == "org_stage"
    assert report["claim"]["node_id"] == "node_stage"
    assert report["execution"]["execution_id"] == "exec_123"
    assert report["assignment"]["assignment_id"] == "assign_123"
    assert report["cleanup"]["node_revoked"] is True
    assert report["cleanup"]["api_key_revoked"] is True
    assert report["cleanup"]["node_credentials_rejected"] is True
    assert report["cleanup"]["api_key_rejected"] is True
    assert api.approved_claims == [("claim_123", "claim-token-123", "operator-token")]
    assert api.approved_nodes == [("node_stage", "operator-token")]
    assert api.revoked_nodes == [("node_stage", "operator-token")]
    assert api.revoked_api_keys == [("key_stage", "operator-token")]
    assert client.accepted == ["assign_123"]
    assert client.progress_updates == [("assign_123", {"stage": "staging_lane", "state": "running"})]
    assert client.completed[0][0] == "assign_123"
    assert client.completed[0][1][0]["output"] == {"embedding": [0.0, 1.0, 2.0]}


def test_build_config_requires_edge_control_url() -> None:
    try:
        staging_smoke.build_config_from_args(
            staging_smoke.parse_args(["--edge-control-url", ""])
        )
    except staging_smoke.StagingSmokeError as error:
        assert "edge-control base URL" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected StagingSmokeError for blank edge-control URL.")


def test_staging_smoke_defaults_to_public_bootstrap_model() -> None:
    config = staging_smoke.build_config_from_args(
        staging_smoke.parse_args(["--edge-control-url", "http://edge-control.test"])
    )

    assert config.model == staging_smoke.DEFAULT_STAGING_SMOKE_MODEL
    assert config.operation == "auto"
    assert staging_smoke.resolve_operation(config.model, config.operation) == "embeddings"


def test_main_prints_json_report(monkeypatch) -> None:
    fake_report = {"status": "ok", "cleanup": {"node_revoked": True}}

    class FakeAPI:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def close(self) -> None:
            return None

    class FakeRunner:
        def __init__(self, api, **_: object) -> None:
            self.api = api

        def run(self, config: staging_smoke.StagingSmokeConfig) -> dict[str, object]:
            assert config.edge_control_url == "http://edge-control.test"
            return dict(fake_report)

    monkeypatch.setattr(staging_smoke, "StagingControlPlaneAPI", FakeAPI)
    monkeypatch.setattr(staging_smoke, "StagingSmokeRunner", FakeRunner)

    buffer = StringIO()
    with redirect_stdout(buffer):
        exit_code = staging_smoke.main(["--edge-control-url", "http://edge-control.test"])

    assert exit_code == 0
    assert json.loads(buffer.getvalue()) == fake_report
