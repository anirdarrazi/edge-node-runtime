import subprocess

import node_agent.single_container as single_container


def test_build_vllm_command_includes_model_host_port_and_extra_args() -> None:
    config = single_container.SingleContainerConfig(
        vllm_model="BAAI/bge-large-en-v1.5",
        vllm_host="0.0.0.0",
        vllm_port=9000,
        vllm_server_command=("python", "-m", "vllm.entrypoints.openai.api_server"),
        vllm_extra_args=("--gpu-memory-utilization", "0.85"),
    )

    command = single_container.build_vllm_command(config)

    assert command == [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "BAAI/bge-large-en-v1.5",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--gpu-memory-utilization",
        "0.85",
    ]


def test_config_from_env_allows_custom_commands(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MODEL", "test/model")
    monkeypatch.setenv("VLLM_PORT", "8123")
    monkeypatch.setenv("VLLM_SERVER_COMMAND", "vllm serve")
    monkeypatch.setenv("VLLM_EXTRA_ARGS", "--dtype auto --max-model-len 4096")
    monkeypatch.setenv("NODE_AGENT_COMMAND", "node-agent run")
    monkeypatch.setenv("START_VLLM", "false")

    config = single_container.SingleContainerConfig.from_env()

    assert config.vllm_model == "test/model"
    assert config.vllm_port == 8123
    assert config.vllm_server_command == ("vllm", "serve")
    assert config.vllm_extra_args == ("--dtype", "auto", "--max-model-len", "4096")
    assert config.node_agent_command == ("node-agent", "run")
    assert config.start_vllm is False
    assert config.local_vllm_url == "http://127.0.0.1:8123"


def test_main_starts_node_agent_without_nested_docker_when_vllm_is_external(monkeypatch) -> None:
    started: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, text=False) -> None:
            started.append(list(command))

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None) -> int:
            return self.returncode

    monkeypatch.setenv("START_VLLM", "false")
    monkeypatch.setenv("NODE_AGENT_COMMAND", "node-agent run")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setattr(single_container.subprocess, "Popen", FakeProcess)

    exit_code = single_container.main()

    assert exit_code == 0
    assert started == [["node-agent", "run"]]


def test_wait_for_vllm_fails_fast_when_process_exits() -> None:
    class FailedProcess:
        returncode = 2

        def poll(self):
            return self.returncode

    config = single_container.SingleContainerConfig(
        vllm_model="test/model",
        vllm_startup_timeout_seconds=1,
    )

    try:
        single_container.wait_for_vllm_ready(config, FailedProcess())  # type: ignore[arg-type]
    except RuntimeError as error:
        assert "vLLM exited" in str(error)
    else:  # pragma: no cover
        raise AssertionError("wait_for_vllm_ready should fail when vLLM exits early")
