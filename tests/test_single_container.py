import subprocess
import sys

import node_agent.single_container as single_container


def test_build_vllm_command_includes_model_host_port_and_extra_args() -> None:
    config = single_container.SingleContainerConfig(
        vllm_model="BAAI/bge-large-en-v1.5",
        vllm_host="0.0.0.0",
        vllm_port=9000,
        max_context_tokens=8192,
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
        "--max-model-len",
        "8192",
    ]


def test_config_from_env_allows_custom_commands(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MODEL", "test/model")
    monkeypatch.setenv("VLLM_PORT", "8123")
    monkeypatch.setenv("MAX_CONTEXT_TOKENS", "16384")
    monkeypatch.setenv("VLLM_SERVER_COMMAND", "vllm serve")
    monkeypatch.setenv("VLLM_EXTRA_ARGS", "--dtype auto --max-model-len 4096")
    monkeypatch.setenv("NODE_AGENT_COMMAND", "node-agent run")
    monkeypatch.setenv("START_VLLM", "false")

    config = single_container.SingleContainerConfig.from_env()

    assert config.vllm_model == "test/model"
    assert config.vllm_port == 8123
    assert config.max_context_tokens == 16384
    assert config.vllm_server_command == ("vllm", "serve")
    assert config.vllm_extra_args == ("--dtype", "auto", "--max-model-len", "4096")
    assert config.node_agent_command == ("node-agent", "run")
    assert config.start_vllm is False
    assert config.local_inference_url == "http://127.0.0.1:8123"
    assert config.local_vllm_url == "http://127.0.0.1:8123"


def test_build_vllm_command_does_not_duplicate_explicit_max_model_len() -> None:
    config = single_container.SingleContainerConfig(
        vllm_model="meta-llama/Llama-3.1-8B-Instruct",
        max_context_tokens=32768,
        vllm_server_command=("vllm", "serve"),
        vllm_extra_args=("--dtype", "auto", "--max-model-len", "4096"),
    )

    command = single_container.build_vllm_command(config)

    assert command.count("--max-model-len") == 1
    assert command[-2:] == ["--max-model-len", "4096"]


def test_embedded_runtime_defaults_to_vast_burst_capacity(tmp_path) -> None:
    supervisor = single_container.EmbeddedRuntimeSupervisor(
        lambda: {},
        cache_dir=tmp_path / "cache",
        credentials_dir=tmp_path / "credentials",
        scratch_dir=tmp_path / "scratch",
    )

    values = supervisor.env_values()

    assert values["RUNTIME_PROFILE"] == "vast_vllm_safetensors"
    assert values["DEPLOYMENT_TARGET"] == "vast_ai"
    assert values["INFERENCE_ENGINE"] == "vllm"
    assert values["CAPACITY_CLASS"] == "elastic_burst"
    assert values["TEMPORARY_NODE"] == "true"
    assert values["BURST_PROVIDER"] == "vast_ai"
    assert values["BURST_LEASE_PHASE"] == "accept_burst_work"
    assert values["BURST_COST_CEILING_USD"] == "0.25"
    assert values["VLLM_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert values["SUPPORTED_MODELS"] == "BAAI/bge-large-en-v1.5"
    assert values["OWNER_TARGET_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert values["OWNER_TARGET_SUPPORTED_MODELS"] == "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    assert values["INFERENCE_BASE_URL"] == "http://127.0.0.1:8000"
    assert values["VLLM_BASE_URL"] == "http://127.0.0.1:8000"


def test_embedded_runtime_rewrites_home_defaults_for_single_container(tmp_path) -> None:
    supervisor = single_container.EmbeddedRuntimeSupervisor(
        lambda: {
            "RUNTIME_PROFILE": "auto",
            "DEPLOYMENT_TARGET": "home_edge",
            "INFERENCE_ENGINE": "llama_cpp",
            "CAPACITY_CLASS": "home_heat",
            "TEMPORARY_NODE": "false",
            "BURST_PROVIDER": "",
            "BURST_LEASE_PHASE": "",
            "BURST_COST_CEILING_USD": "",
            "INFERENCE_BASE_URL": "http://inference-runtime:8000",
            "VLLM_BASE_URL": "http://inference-runtime:8000",
            "VLLM_PORT": "",
        },
        cache_dir=tmp_path / "cache",
        credentials_dir=tmp_path / "credentials",
        scratch_dir=tmp_path / "scratch",
    )

    values = supervisor.env_values()

    assert values["RUNTIME_PROFILE"] == "vast_vllm_safetensors"
    assert values["DEPLOYMENT_TARGET"] == "vast_ai"
    assert values["INFERENCE_ENGINE"] == "vllm"
    assert values["CAPACITY_CLASS"] == "elastic_burst"
    assert values["TEMPORARY_NODE"] == "true"
    assert values["BURST_PROVIDER"] == "vast_ai"
    assert values["BURST_LEASE_PHASE"] == "accept_burst_work"
    assert values["BURST_COST_CEILING_USD"] == "0.25"
    assert values["VLLM_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert values["SUPPORTED_MODELS"] == "BAAI/bge-large-en-v1.5"
    assert values["OWNER_TARGET_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert values["OWNER_TARGET_SUPPORTED_MODELS"] == "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    assert values["INFERENCE_BASE_URL"] == "http://127.0.0.1:8000"
    assert values["VLLM_BASE_URL"] == "http://127.0.0.1:8000"


def test_embedded_runtime_keeps_gated_startup_when_token_is_configured(tmp_path) -> None:
    supervisor = single_container.EmbeddedRuntimeSupervisor(
        lambda: {
            "VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
            "SUPPORTED_MODELS": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            "GPU_MEMORY_GB": "24",
            "HF_TOKEN": "hf_secret_token",
        },
        cache_dir=tmp_path / "cache",
        credentials_dir=tmp_path / "credentials",
        scratch_dir=tmp_path / "scratch",
    )

    values = supervisor.env_values()

    assert values["VLLM_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert values["SUPPORTED_MODELS"] == "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    assert "OWNER_TARGET_MODEL" not in values


def test_embedded_runtime_uses_public_bootstrap_on_low_vram_even_with_token(tmp_path) -> None:
    supervisor = single_container.EmbeddedRuntimeSupervisor(
        lambda: {
            "VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
            "SUPPORTED_MODELS": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            "GPU_MEMORY_GB": "16",
            "HF_TOKEN": "hf_secret_token",
        },
        cache_dir=tmp_path / "cache",
        credentials_dir=tmp_path / "credentials",
        scratch_dir=tmp_path / "scratch",
    )

    values = supervisor.env_values()

    assert values["VLLM_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert values["SUPPORTED_MODELS"] == "BAAI/bge-large-en-v1.5"
    assert values["OWNER_TARGET_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert values["OWNER_TARGET_SUPPORTED_MODELS"] == "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"


def test_main_starts_node_agent_without_nested_docker_when_vllm_is_external(monkeypatch) -> None:
    started: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, text=False, **_kwargs) -> None:
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


def test_main_recomputes_startup_model_after_defaults(monkeypatch) -> None:
    started: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, text=False, **_kwargs) -> None:
            started.append(list(command))

        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None) -> int:
            return self.returncode

    monkeypatch.delenv("VLLM_MODEL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("NODE_AGENT_COMMAND", "node-agent run")
    monkeypatch.setattr(single_container.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(single_container, "wait_for_inference_runtime_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(single_container, "wait_for_any_process", lambda processes: processes[1])

    exit_code = single_container.main()

    assert exit_code == 0
    assert started[0][:5] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "BAAI/bge-large-en-v1.5",
    ]


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
        assert "local inference runtime exited" in str(error)
    else:  # pragma: no cover
        raise AssertionError("wait_for_vllm_ready should fail when vLLM exits early")


def test_wait_for_vllm_includes_recent_output_when_process_exits() -> None:
    class FailedProcess:
        returncode = 1

        def poll(self):
            return self.returncode

    config = single_container.SingleContainerConfig(
        vllm_model="meta-llama/Llama-3.1-8B-Instruct",
        vllm_startup_timeout_seconds=1,
    )

    try:
        single_container.wait_for_inference_runtime_ready(
            config,
            FailedProcess(),  # type: ignore[arg-type]
            output_tail=lambda: "CUDA out of memory\nEngine process failed to start",
        )
    except RuntimeError as error:
        detail = str(error)
        assert "Recent vLLM output" in detail
        assert "CUDA out of memory" in detail
    else:  # pragma: no cover
        raise AssertionError("wait_for_inference_runtime_ready should include recent process output")


def test_validate_gated_model_access_rejects_unauthorized_token(monkeypatch) -> None:
    class FakeResponse:
        status_code = 403

    monkeypatch.setattr(single_container.httpx, "get", lambda *args, **kwargs: FakeResponse())

    try:
        single_container.validate_gated_model_access(
            {"HF_TOKEN": "hf_invalid"},
            "meta-llama/Llama-3.1-8B-Instruct",
        )
    except RuntimeError as error:
        assert "Hugging Face denied access" in str(error)
    else:  # pragma: no cover
        raise AssertionError("validate_gated_model_access should reject unauthorized tokens")
