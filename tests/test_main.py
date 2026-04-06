from types import SimpleNamespace

import httpx
import pytest

import node_agent.main as main_module


class FakeControl:
    def __init__(self) -> None:
        self.enroll_calls = 0
        self.attest_calls = 0
        self.clear_calls = 0

    def enroll_if_needed(self):
        self.enroll_calls += 1
        return "node_123", "key_123"

    def attest(self):
        self.attest_calls += 1
        if self.attest_calls == 1:
            request = httpx.Request("POST", "http://edge.test/nodes/attest")
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    def clear_credentials(self):
        self.clear_calls += 1

    def is_auth_error(self, error: Exception) -> bool:
        return isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 401

    def heartbeat(self):
        raise KeyboardInterrupt()

    def pull_assignment(self):
        return None


def test_run_retries_bootstrap_after_auth_failure(monkeypatch: pytest.MonkeyPatch):
    settings = SimpleNamespace(vllm_base_url="http://localhost:8000", poll_interval_seconds=0)
    control = FakeControl()

    monkeypatch.setattr(main_module, "NodeAgentSettings", lambda: settings)
    monkeypatch.setattr(main_module, "EdgeControlClient", lambda _settings: control)
    monkeypatch.setattr(main_module, "VLLMRuntime", lambda _base_url: object())
    monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(KeyboardInterrupt):
        main_module.run()

    assert control.clear_calls == 1
    assert control.enroll_calls == 2
    assert control.attest_calls == 2
