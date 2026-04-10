from types import SimpleNamespace
import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import node_agent.main as main_module


class FakeControl:
    def __init__(self, has_credentials: bool = True) -> None:
        self.settings = SimpleNamespace(
            poll_interval_seconds=0,
            node_region="eu-se-1",
            trust_tier="restricted",
            restricted_capable=True,
            node_id="node_123",
            supported_models="meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            gpu_memory_gb=24.0,
            max_context_tokens=32768,
            max_batch_tokens=50000,
            attestation_provider="hardware",
            restricted_attestation_max_age_seconds=3600,
        )
        self._has_credentials = has_credentials
        self.bootstrap_calls = 0
        self.require_calls = 0
        self.attest_calls = 0
        self.clear_calls = 0
        self.recovery_notes = []
        self.auth_fail_on_heartbeat = False
        self.progress_updates = []
        self.failures = []
        self.completions = []
        self.attestation_state = {
            "node_id": "node_123",
            "attestation_provider": "hardware",
            "status": "verified",
            "attested_at": datetime.now(timezone.utc).isoformat(),
        }

    def has_credentials(self) -> bool:
        return self._has_credentials

    def bootstrap(self, interactive: bool = True):
        self.bootstrap_calls += 1
        self._has_credentials = True
        self.settings.node_id = "node_123"
        return "node_123", "key_123"

    def require_credentials(self):
        self.require_calls += 1
        if not self._has_credentials:
            raise RuntimeError("missing credentials")
        self.settings.node_id = "node_123"
        return "node_123", "key_123"

    def attest(self):
        self.attest_calls += 1
        self.attestation_state = {
            "node_id": "node_123",
            "attestation_provider": self.settings.attestation_provider,
            "status": "verified",
            "attested_at": datetime.now(timezone.utc).isoformat(),
        }

    def heartbeat(self):
        if self.auth_fail_on_heartbeat:
            request = httpx.Request("POST", "http://edge.test/nodes/heartbeat")
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)
        raise KeyboardInterrupt()

    def pull_assignment(self):
        return None

    def accept_assignment(self, _assignment_id: str):
        return None

    def report_progress(self, assignment_id: str, progress):
        self.progress_updates.append((assignment_id, progress))

    def complete_assignment(self, assignment_id: str, results):
        self.completions.append((assignment_id, results))

    def fail_assignment(self, assignment_id: str, code: str, message: str, retryable: bool = True):
        self.failures.append((assignment_id, code, message, retryable))

    def clear_credentials(self):
        self.clear_calls += 1
        self._has_credentials = False

    def write_recovery_note(self, message: str):
        self.recovery_notes.append(message)

    def is_auth_error(self, error: Exception) -> bool:
        return isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 401

    def load_attestation_state(self):
        return self.attestation_state


def test_command_bootstrap_runs_claim_bootstrap(monkeypatch: pytest.MonkeyPatch):
    control = FakeControl(has_credentials=False)
    monkeypatch.setattr(main_module, "NodeAgentSettings", lambda: SimpleNamespace(vllm_base_url="http://localhost:8000"))
    monkeypatch.setattr(main_module, "EdgeControlClient", lambda _settings: control)

    result = main_module.main(["bootstrap"])

    assert result == 0
    assert control.bootstrap_calls == 1
    assert control.attest_calls == 1


def test_command_default_bootstraps_when_credentials_are_missing(monkeypatch: pytest.MonkeyPatch):
    settings = SimpleNamespace(vllm_base_url="http://localhost:8000")
    control = FakeControl(has_credentials=False)

    monkeypatch.setattr(main_module, "NodeAgentSettings", lambda: settings)
    monkeypatch.setattr(main_module, "EdgeControlClient", lambda _settings: control)
    monkeypatch.setattr(main_module, "VLLMRuntime", lambda _base_url: object())

    with pytest.raises(KeyboardInterrupt):
        main_module.main([])

    assert control.bootstrap_calls == 1
    assert control.attest_calls == 1


def test_run_worker_loop_clears_credentials_after_auth_failure():
    control = FakeControl(has_credentials=True)
    control.auth_fail_on_heartbeat = True

    with pytest.raises(RuntimeError):
        main_module.run_worker_loop(control, object(), attest_on_start=False)

    assert control.clear_calls == 1
    assert control.recovery_notes


def test_run_worker_loop_reports_assignment_failure():
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pulled = False

        def heartbeat(self):
            return None

        def pull_assignment(self):
            if self.pulled:
                raise KeyboardInterrupt()
            self.pulled = True
            return SimpleNamespace(
                assignment_id="assign_123",
                execution_id="pexec_123",
                item_count=1,
                operation="responses",
                model="meta-llama/Llama-3.1-8B-Instruct",
                privacy_tier="restricted",
                allowed_regions=["eu-se-1"],
                required_vram_gb=16.0,
                required_context_tokens=8192,
                token_budget={"total_tokens": 2048},
            )

        def fetch_artifact(self, _assignment):
            raise ValueError("invalid payload")

    class RuntimeStub:
        def execute(self, _operation, _model, _items):
            raise AssertionError("runtime should not run when payload fetch fails")

    control = AssignmentControl()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, RuntimeStub(), attest_on_start=False)

    assert control.failures == [("assign_123", "invalid_assignment_payload", "invalid payload", False)]
    assert control.progress_updates[-1][1]["state"] == "failed"


def test_run_worker_loop_keeps_assignments_fresh_while_runtime_is_busy(monkeypatch: pytest.MonkeyPatch):
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pulled = False

        def heartbeat(self):
            return None

        def pull_assignment(self):
            if self.pulled:
                raise KeyboardInterrupt()
            self.pulled = True
            return SimpleNamespace(
                assignment_id="assign_keepalive",
                execution_id="pexec_keepalive",
                item_count=1,
                operation="responses",
                model="meta-llama/Llama-3.1-8B-Instruct",
                privacy_tier="restricted",
                allowed_regions=["eu-se-1"],
                required_vram_gb=16.0,
                required_context_tokens=8192,
                token_budget={"total_tokens": 2048},
            )

        def fetch_artifact(self, _assignment):
            return {
                "items": [
                    {
                        "batch_item_id": "item_1",
                        "customer_item_id": "cust_1",
                        "operation": "responses",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "input": {"messages": [{"role": "user", "content": "hello"}]},
                    }
                ]
            }

    class SlowRuntime:
        def execute(self, _operation, _model, _items):
            time.sleep(0.05)
            return [{"status": "completed"}]

    control = AssignmentControl()
    monkeypatch.setattr(main_module, "assignment_progress_keepalive_seconds", 0.01)

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, SlowRuntime(), attest_on_start=False)

    running_updates = [progress for _assignment_id, progress in control.progress_updates if progress["state"] == "running"]
    assert len(running_updates) >= 2
    assert any(progress.get("keepalive") is True for progress in running_updates[1:])
    assert control.progress_updates[-1][1]["state"] == "completed"


def test_validate_assignment_rejects_restricted_work_without_hardware_attestation():
    control = FakeControl(has_credentials=True)
    control.settings.attestation_provider = "simulated"
    control.attestation_state["attestation_provider"] = "simulated"
    assignment = SimpleNamespace(
        assignment_id="assign_restricted",
        execution_id="pexec_restricted",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="restricted",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
    )

    with pytest.raises(ValueError, match="not hardware-backed"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )


def test_validate_assignment_rejects_restricted_work_with_stale_local_attestation():
    control = FakeControl(has_credentials=True)
    control.attestation_state["attested_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    assignment = SimpleNamespace(
        assignment_id="assign_restricted_stale",
        execution_id="pexec_restricted_stale",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="restricted",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
    )

    with pytest.raises(ValueError, match="fresh local hardware attestation record"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )


def test_run_worker_loop_refreshes_stale_restricted_attestation_before_polling():
    control = FakeControl(has_credentials=True)
    control.attestation_state["attested_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, object(), attest_on_start=False)

    assert control.attest_calls == 1


def test_validate_assignment_rejects_item_model_mismatch():
    control = FakeControl(has_credentials=True)
    assignment = SimpleNamespace(
        assignment_id="assign_model_mismatch",
        execution_id="pexec_model_mismatch",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="restricted",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
    )

    with pytest.raises(ValueError, match="does not match envelope"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "BAAI/bge-large-en-v1.5",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )
