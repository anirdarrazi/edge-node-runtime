from types import SimpleNamespace

import httpx
import pytest

import node_agent.main as main_module


class FakeControl:
    def __init__(self, has_credentials: bool = True) -> None:
        self.settings = SimpleNamespace(poll_interval_seconds=0)
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

    def has_credentials(self) -> bool:
        return self._has_credentials

    def bootstrap(self, interactive: bool = True):
        self.bootstrap_calls += 1
        self._has_credentials = True
        return "node_123", "key_123"

    def require_credentials(self):
        self.require_calls += 1
        if not self._has_credentials:
            raise RuntimeError("missing credentials")
        return "node_123", "key_123"

    def attest(self):
        self.attest_calls += 1

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
                operation="responses",
                model="meta-llama/Llama-3.1-8B-Instruct",
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
