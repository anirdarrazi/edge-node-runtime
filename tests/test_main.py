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
        self.auth_fail_on_heartbeat = False

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

    def clear_credentials(self):
        self.clear_calls += 1
        self._has_credentials = False

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
