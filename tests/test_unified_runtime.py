from pathlib import Path

import pytest

import node_agent.launcher as launcher_module
import node_agent.runtime_backend as runtime_backend_module


RUNTIME_ROOT = Path(__file__).resolve().parents[1]


def test_detect_runtime_backend_defaults_to_manager_outside_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime_backend_module.RUNTIME_BACKEND_ENV, raising=False)
    monkeypatch.setattr(runtime_backend_module, "docker_socket_present", lambda: False)
    monkeypatch.setattr(runtime_backend_module, "running_inside_container", lambda: False)

    assert runtime_backend_module.detect_runtime_backend() == runtime_backend_module.MANAGER_RUNTIME_BACKEND


def test_detect_runtime_backend_uses_single_container_inside_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime_backend_module.RUNTIME_BACKEND_ENV, raising=False)
    monkeypatch.setattr(runtime_backend_module, "docker_socket_present", lambda: False)
    monkeypatch.setattr(runtime_backend_module, "running_inside_container", lambda: True)

    assert runtime_backend_module.detect_runtime_backend() == runtime_backend_module.SINGLE_CONTAINER_RUNTIME_BACKEND


def test_launcher_defaults_to_single_container_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    monkeypatch.setattr(launcher_module, "single_container_main", lambda: called.append("single-container") or 0)

    assert launcher_module.main([]) == 0
    assert called == ["single-container"]


def test_public_runtime_image_starts_headless_single_container_runtime() -> None:
    dockerfile = (RUNTIME_ROOT / "Dockerfile.service").read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["node-agent-single-container"]' in dockerfile
    assert "EXPOSE 8000 8011" in dockerfile
    assert "node-agent-launcher" not in dockerfile
    assert "EXPOSE 8765" not in dockerfile
