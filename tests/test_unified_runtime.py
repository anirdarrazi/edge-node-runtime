import os
from pathlib import Path

import pytest

import node_agent.installer as installer_module
import node_agent.launcher as launcher_module
import node_agent.runtime_backend as runtime_backend_module


def write_example_env(runtime_dir: Path) -> None:
    (runtime_dir / ".env.example").write_text(
        "\n".join(
            [
                "EDGE_CONTROL_URL=https://edge.autonomousc.com",
                "OPERATOR_TOKEN=",
                "NODE_LABEL=AUTONOMOUSc Edge Node",
                "NODE_REGION=eu-se-1",
                "TRUST_TIER=standard",
                "RESTRICTED_CAPABLE=false",
                "CREDENTIALS_PATH=/var/lib/autonomousc/data/credentials/node-credentials.json",
                "AUTOPILOT_STATE_PATH=/var/lib/autonomousc/data/scratch/autopilot-state.json",
                "VLLM_BASE_URL=http://127.0.0.1:8000",
                "GPU_NAME=Generic GPU",
                "GPU_MEMORY_GB=24",
                "MAX_CONTEXT_TOKENS=32768",
                "MAX_BATCH_TOKENS=50000",
                "MAX_CONCURRENT_ASSIGNMENTS=2",
                "THERMAL_HEADROOM=0.8",
                "SUPPORTED_MODELS=meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
                "POLL_INTERVAL_SECONDS=10",
                "ATTESTATION_PROVIDER=simulated",
                "VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct",
                "DOCKER_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime:latest",
                "",
            ]
        ),
        encoding="utf-8",
    )


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


def test_launcher_defaults_to_container_service_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[list[str]] = []

    monkeypatch.delenv(runtime_backend_module.RUNTIME_BACKEND_ENV, raising=False)
    monkeypatch.setattr(launcher_module, "detect_runtime_backend", lambda: runtime_backend_module.SINGLE_CONTAINER_RUNTIME_BACKEND)
    monkeypatch.setattr(launcher_module, "service_main", lambda args: recorded.append(list(args)) or 0)

    assert launcher_module.main([]) == 0
    assert recorded == [["run", "--host", "0.0.0.0", "--port", "8765"]]
    assert os.environ[runtime_backend_module.RUNTIME_BACKEND_ENV] == runtime_backend_module.SINGLE_CONTAINER_RUNTIME_BACKEND


def test_guided_installer_uses_embedded_runtime_in_single_container_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRuntimeController:
        def __init__(self) -> None:
            self.calls: list[dict[str, bool]] = []

        def start(self, *, recreate: bool, start_vllm: bool, start_node: bool) -> None:
            self.calls.append(
                {
                    "recreate": recreate,
                    "start_vllm": start_vllm,
                    "start_node": start_node,
                }
            )

    write_example_env(tmp_path)
    controller = FakeRuntimeController()
    monkeypatch.setenv(runtime_backend_module.RUNTIME_BACKEND_ENV, runtime_backend_module.SINGLE_CONTAINER_RUNTIME_BACKEND)
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: None)

    installer = installer_module.GuidedInstaller(
        runtime_dir=tmp_path,
        runtime_status_provider=lambda: {"running_services": ["vllm", "node-agent"]},
        runtime_controller=controller,  # type: ignore[arg-type]
    )

    preflight = installer.collect_preflight()

    assert preflight["docker_cli"] is True
    assert preflight["docker_compose"] is True
    assert preflight["docker_daemon"] is True
    assert preflight["running_services"] == ["vllm", "node-agent"]

    installer.compose_up(["vllm"])
    installer.compose_up(["node-agent"])

    assert controller.calls == [
        {"recreate": False, "start_vllm": True, "start_node": False},
        {"recreate": False, "start_vllm": False, "start_node": True},
    ]
