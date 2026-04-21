import json
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import node_agent.installer as installer_module


def completed(args: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


def normalize_compose_args(args: list[str]) -> list[str]:
    if args[:2] != ["docker", "compose"]:
        return args
    normalized = ["docker", "compose"]
    index = 2
    while index + 1 < len(args) and args[index] == "--env-file":
        index += 2
    normalized.extend(args[index:])
    return normalized


def write_example_env(runtime_dir: Path) -> None:
    (runtime_dir / ".env.example").write_text(
        "\n".join(
            [
                "EDGE_CONTROL_URL=https://edge.autonomousc.com",
                "OPERATOR_TOKEN=",
                "NODE_LABEL=AUTONOMOUSc Nordic Node 01",
                "NODE_REGION=eu-se-1",
                "TRUST_TIER=restricted",
                "RESTRICTED_CAPABLE=true",
                "CREDENTIALS_PATH=/var/lib/autonomousc/credentials/node-credentials.json",
                "RUNTIME_PROFILE=auto",
                "DEPLOYMENT_TARGET=home_edge",
                "INFERENCE_ENGINE=llama_cpp",
                "RUNTIME_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda",
                "INFERENCE_BASE_URL=http://inference-runtime:8000",
                "VLLM_BASE_URL=http://inference-runtime:8000",
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
                "LLAMA_CPP_HF_REPO=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                "LLAMA_CPP_HF_FILE=Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "LLAMA_CPP_ALIAS=meta-llama/Llama-3.1-8B-Instruct",
                "LLAMA_CPP_EMBEDDING=false",
                "LLAMA_CPP_POOLING=",
                "VLLM_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda",
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def stable_machine_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        installer_module,
        "infer_node_region",
        lambda: ("eu-se-1", "Recommended from the local locale (SE)."),
    )
    monkeypatch.setattr(
        installer_module,
        "detect_attestation_provider",
        lambda *_args, **_kwargs: (
            "simulated",
            "This machine does not expose a ready hardware attestation device yet.",
        ),
    )


@pytest.fixture(autouse=True)
def stable_setup_connectivity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        installer_module.GuidedInstaller,
        "resolve_setup_dns",
        lambda _self, hosts: {
            "ok": True,
            "targets": [
                {
                    "host": host,
                    "ok": True,
                    "addresses": ["127.0.0.1"],
                }
                for host in hosts
            ],
        },
    )
    monkeypatch.setattr(
        installer_module.GuidedInstaller,
        "probe_control_plane",
        lambda _self, url: {
            "ok": True,
            "status_code": 200,
            "probe_url": url,
            "service": "autonomousc-edge-network",
            "status": "ok",
        },
    )


def installer_for_detected_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gpu_line: str,
    *,
    attestation_provider: str = "simulated",
) -> installer_module.GuidedInstaller:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(
        installer_module,
        "detect_attestation_provider",
        lambda *_args, **_kwargs: (
            attestation_provider,
            f"{attestation_provider} attestation for matrix test.",
        ),
    )

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout=f"{gpu_line}\n")
        raise AssertionError(f"Unexpected command: {args}")

    return installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)


def test_current_config_prefills_detected_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=runtime_dir, command_runner=runner)
    config = installer.current_config()

    assert config["gpu_name"] == "RTX 4090"
    assert config["gpu_memory_gb"] == "24.0"
    assert config["max_concurrent_assignments"] == "2"
    assert config["setup_profile"] == "balanced"
    assert config["deployment_target"] == "home_edge"
    assert config["inference_engine"] == "llama_cpp"
    assert config["vllm_model"] == "BAAI/bge-large-en-v1.5"
    assert config["supported_models"] == "BAAI/bge-large-en-v1.5"
    assert config["owner_target_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert config["owner_target_supported_models"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert config["bootstrap_pending_upgrade"] is True
    assert config["startup_model_fallback"] is False
    assert config["recommended_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert config["gpu_support_label"] == "24-47 GB NVIDIA"
    assert config["gpu_support_track"] == "Llama 8B + embeddings"
    assert config["recommended_node_region"] == "eu-se-1"
    assert config["routing_lane"] == "community_quantized_home"
    assert config["routing_lane_label"] == "Community quantized home"
    assert config["routing_lane_allowed_privacy_tiers"] == ["standard"]
    assert config["routing_lane_allowed_result_guarantees"] == ["community_best_effort"]
    assert config["routing_lane_allowed_trust_requirements"] == ["untrusted_allowed"]
    assert config["premium_eligibility_label"] == "Community capacity enabled"
    assert config["hugging_face_repository"] == "CompendiumLabs/bge-large-en-v1.5-gguf"
    assert config["hugging_face_token_required"] is False
    assert config["hugging_face_token_configured"] is False
    assert "everyday setting" in config["profile_summary"]


def test_current_config_uses_embeddings_preset_for_16gb_nvidia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = installer_for_detected_gpu(tmp_path, monkeypatch, "RTX 4080 Laptop GPU, 16384")

    config = installer.current_config()

    assert config["recommended_model"] == "BAAI/bge-large-en-v1.5"
    assert config["recommended_supported_models"] == "BAAI/bge-large-en-v1.5"
    assert config["recommended_setup_profile"] == "balanced"
    assert config["recommended_max_concurrent_assignments"] == "1"
    assert config["gpu_support_label"] == "12-23 GB NVIDIA"
    assert config["gpu_support_track"] == "Embeddings/community"
    assert config["premium_eligibility_label"] == "Community capacity enabled"
    assert "embeddings/community preset" in config["premium_eligibility_detail"]


def test_current_config_uses_llama8b_preset_for_24gb_hardware_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = installer_for_detected_gpu(
        tmp_path,
        monkeypatch,
        "RTX 4090, 24564",
        attestation_provider="hardware",
    )

    config = installer.current_config()

    assert config["recommended_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert config["recommended_supported_models"] == "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    assert config["recommended_setup_profile"] == "balanced"
    assert config["recommended_max_concurrent_assignments"] == "2"
    assert config["gpu_support_label"] == "24-47 GB NVIDIA"
    assert config["routing_lane"] == "community_quantized_home"
    assert config["premium_eligibility_label"] == "Premium capacity eligible"
    assert "Llama 8B + embeddings preset" in config["premium_eligibility_detail"]


def test_current_config_uses_large_nvidia_preset_for_48gb_hardware_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = installer_for_detected_gpu(
        tmp_path,
        monkeypatch,
        "RTX 6000 Ada Generation, 49152",
        attestation_provider="hardware",
    )

    config = installer.current_config()

    assert config["recommended_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert config["recommended_supported_models"] == "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    assert config["recommended_setup_profile"] == "performance"
    assert config["recommended_max_concurrent_assignments"] == "4"
    assert config["gpu_support_label"] == "48+ GB NVIDIA"
    assert config["gpu_support_track"] == "Llama 8B + embeddings today"
    assert config["premium_eligibility_label"] == "Premium capacity eligible"
    assert "larger premium profiles later" in config["premium_eligibility_detail"]


def test_build_env_uses_quickstart_profile_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)
    env_values = installer.build_env(
        {
            "setup_mode": "quickstart",
            "setup_profile": "performance",
            "node_label": "",
            "max_concurrent_assignments": "",
        }
    )

    assert env_values["SETUP_PROFILE"] == "performance"
    assert env_values["MAX_CONCURRENT_ASSIGNMENTS"] == "3"
    assert env_values["THERMAL_HEADROOM"] == "0.92"
    assert env_values["MAX_BATCH_TOKENS"] == "65000"
    assert env_values["NODE_LABEL"] == "AUTONOMOUSc RTX 4090 Node"
    assert env_values["NODE_REGION"] == "eu-se-1"
    assert env_values["TRUST_TIER"] == "standard"
    assert env_values["RESTRICTED_CAPABLE"] == "false"
    assert env_values["DEPLOYMENT_TARGET"] == "home_edge"
    assert env_values["INFERENCE_ENGINE"] == "llama_cpp"
    assert env_values["RUNTIME_PROFILE"] == "auto"
    assert env_values["VLLM_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert env_values["SUPPORTED_MODELS"] == "BAAI/bge-large-en-v1.5"
    assert env_values["OWNER_TARGET_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert env_values["OWNER_TARGET_SUPPORTED_MODELS"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert env_values["LLAMA_CPP_HF_REPO"] == "CompendiumLabs/bge-large-en-v1.5-gguf"
    assert env_values["LLAMA_CPP_EMBEDDING"] == "true"
    assert env_values["VLLM_IMAGE"] == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    assert env_values["CREDENTIALS_PATH"] == installer_module.COMPOSE_RUNTIME_CREDENTIALS_PATH
    assert env_values["AUTOPILOT_STATE_PATH"] == installer_module.COMPOSE_RUNTIME_AUTOPILOT_STATE_PATH
    assert env_values["STARTUP_STATUS_PATH"] == installer_module.COMPOSE_RUNTIME_STARTUP_STATUS_PATH
    assert env_values["STARTUP_STATUS_PORT"] == str(installer_module.DEFAULT_STARTUP_STATUS_PORT)
    assert env_values["STARTUP_STATUS_ENDPOINT_PATH"] == installer_module.DEFAULT_STARTUP_STATUS_ENDPOINT_PATH
    assert env_values["STARTUP_WARM_SOURCE"] != ""


def test_build_env_prefers_detected_gpu_over_stale_saved_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="NVIDIA GeForce RTX 5060 Ti, 16276\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)
    installer.write_runtime_settings(
        {
            "GPU_NAME": "RTX 4090",
            "GPU_MEMORY_GB": "24.0",
        }
    )

    env_values = installer.build_env({"setup_mode": "quickstart", "node_label": "Owner Node"})

    assert env_values["GPU_NAME"] == "NVIDIA GeForce RTX 5060 Ti"
    assert env_values["GPU_MEMORY_GB"] == "15.9"
    assert env_values["NODE_LABEL"] == "Owner Node"


def test_build_env_uses_host_paths_for_single_container_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)
    monkeypatch.setattr(installer, "current_runtime_backend", lambda: installer_module.SINGLE_CONTAINER_RUNTIME_BACKEND)

    env_values = installer.build_env({"setup_mode": "quickstart", "node_label": "Owner Node"})

    assert env_values["CREDENTIALS_PATH"] == str(installer.credentials_path)
    assert env_values["AUTOPILOT_STATE_PATH"] == str(tmp_path / "data" / "scratch" / "autopilot-state.json")
    assert env_values["STARTUP_STATUS_PATH"] == str(tmp_path / "data" / "scratch" / "startup-status.json")
    assert env_values[installer_module.RUNTIME_BACKEND_ENV] == installer_module.SINGLE_CONTAINER_RUNTIME_BACKEND


def test_build_env_auto_discovers_offline_appliance_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        raise AssertionError(f"Unexpected command: {args}")

    bundle_dir = tmp_path / "offline-bundle" / "model-cache"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "starter.gguf").write_text("starter", encoding="utf-8")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)

    env_values = installer.build_env({"setup_mode": "quickstart", "node_label": "Owner Node"})

    assert env_values["OFFLINE_INSTALL_BUNDLE_DIR"] == str(tmp_path / "offline-bundle")


def test_current_runtime_backend_prefers_persisted_backend_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.delenv(installer_module.RUNTIME_BACKEND_ENV, raising=False)
    monkeypatch.setattr(installer_module, "detect_runtime_backend", lambda: "manager")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    installer.write_runtime_settings(
        {
            installer_module.RUNTIME_BACKEND_ENV: installer_module.SINGLE_CONTAINER_RUNTIME_BACKEND,
            "NODE_LABEL": "Owner Node",
        }
    )

    assert installer.current_runtime_backend() == installer_module.SINGLE_CONTAINER_RUNTIME_BACKEND


def test_build_env_persists_hugging_face_token_for_quickstart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)
    env_values = installer.build_env(
        {
            "setup_mode": "quickstart",
            "node_label": "Nordic Heat Compute",
            "hugging_face_hub_token": "hf_secret_token",
        }
    )
    installer.write_runtime_settings(env_values)

    persisted = installer.load_persisted_env()
    config = installer.current_config()

    assert env_values["HUGGING_FACE_HUB_TOKEN"] == "hf_secret_token"
    assert env_values["HF_TOKEN"] == "hf_secret_token"
    assert persisted["HUGGING_FACE_HUB_TOKEN"] == "hf_secret_token"
    assert persisted["HF_TOKEN"] == "hf_secret_token"
    assert config["hugging_face_token_configured"] is True
    assert env_values["VLLM_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert env_values["SUPPORTED_MODELS"] == "BAAI/bge-large-en-v1.5"
    assert env_values["OWNER_TARGET_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert env_values["OWNER_TARGET_SUPPORTED_MODELS"] == "meta-llama/Llama-3.1-8B-Instruct"


def test_build_env_quickstart_resets_advanced_overrides_without_operator_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)
    installer.write_runtime_settings(
        {
            "EDGE_CONTROL_URL": "https://custom-control.example",
            "NODE_LABEL": "Operator Override",
            "NODE_REGION": "us-east-1",
            "TRUST_TIER": "restricted",
            "RESTRICTED_CAPABLE": "true",
            "CREDENTIALS_PATH": str(installer.credentials_path),
            "AUTOPILOT_STATE_PATH": str(tmp_path / "data" / "scratch" / "autopilot-state.json"),
            "INFERENCE_BASE_URL": "http://127.0.0.1:8000",
            "VLLM_BASE_URL": "http://127.0.0.1:8000",
            "GPU_NAME": "RTX 4090",
            "GPU_MEMORY_GB": "24.0",
            "MAX_CONTEXT_TOKENS": "32768",
            "MAX_BATCH_TOKENS": "99999",
            "MAX_CONCURRENT_ASSIGNMENTS": "7",
            "THERMAL_HEADROOM": "0.99",
            "SUPPORTED_MODELS": "custom/model",
            "POLL_INTERVAL_SECONDS": "10",
            "ATTESTATION_PROVIDER": "hardware",
            "VLLM_MODEL": "custom/model",
            "DOCKER_IMAGE": "anirdarrazi/autonomousc-ai-edge-runtime:latest",
        }
    )

    env_values = installer.build_env({"setup_mode": "quickstart", "node_label": "Owner Node"})

    assert env_values["NODE_LABEL"] == "Owner Node"
    assert env_values["EDGE_CONTROL_URL"] == "https://custom-control.example"
    assert env_values["TRUST_TIER"] == "standard"
    assert env_values["RESTRICTED_CAPABLE"] == "false"
    assert env_values["DEPLOYMENT_TARGET"] == "home_edge"
    assert env_values["INFERENCE_ENGINE"] == "llama_cpp"
    assert env_values["VLLM_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert env_values["SUPPORTED_MODELS"] == "BAAI/bge-large-en-v1.5"
    assert env_values["OWNER_TARGET_MODEL"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert env_values["OWNER_TARGET_SUPPORTED_MODELS"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert env_values["MAX_CONCURRENT_ASSIGNMENTS"] == "2"
    assert env_values["THERMAL_HEADROOM"] == "0.80"
    assert env_values["VLLM_IMAGE"] == "ghcr.io/ggml-org/llama.cpp:server-cuda"


def test_collect_preflight_reports_missing_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: None)

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    preflight = installer.collect_preflight()

    assert preflight["docker_cli"] is False
    assert preflight["docker_daemon"] is False
    assert "Docker is not installed" in preflight["docker_error"]
    assert preflight["disk"]["total_gb"] > 0
    assert preflight["blockers"]
    assert preflight["ready_for_claim"] is False
    assert preflight["claim_gate_blockers"][0].startswith("Docker:")
    docker_check = next(check for check in preflight["setup_checks"] if check["key"] == "docker")
    assert docker_check["status"] == "fail"
    assert "Install Docker Desktop" in docker_check["fix"]


def test_collect_preflight_reports_missing_nvidia_container_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"runc":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, command_runner=runner)
    preflight = installer.collect_preflight()

    assert preflight["docker_daemon"] is True
    assert preflight["gpu"]["detected"] is True
    assert preflight["nvidia_container_runtime"]["visible"] is False
    assert "NVIDIA GPU support" in " ".join(preflight["blockers"])


def test_preview_setup_payload_uses_unsaved_operator_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    class FakeManager:
        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": False,
                "label": "Available",
                "detail": "Available.",
            }

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(
        runtime_dir=tmp_path,
        command_runner=runner,
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
    )

    preview = installer.preview_setup_payload(
        {
            "setup_mode": "quickstart",
            "operator_mode": True,
            "node_label": "Preview Node",
            "node_region": "moon-test-1",
            "trust_tier": "standard",
            "restricted_capable": False,
            "max_concurrent_assignments": "0",
            "target_gpu_utilization_pct": "25",
            "min_gpu_memory_headroom_pct": "70",
            "hugging_face_hub_token": "hf_preview_token",
        }
    )

    assert preview["config"]["node_label"] == "Preview Node"
    assert preview["config"]["node_region"] == "moon-test-1"
    assert preview["config"]["trust_tier"] == "standard"
    assert preview["config"]["restricted_capable"] is False
    assert preview["config"]["max_concurrent_assignments"] == "0"
    assert preview["config"]["target_gpu_utilization_pct"] == "25"
    assert preview["config"]["min_gpu_memory_headroom_pct"] == "70"
    assert preview["config"]["hugging_face_token_preview_supplied"] is True
    assert preview["config"]["hugging_face_token_saved"] is False
    assert preview["preflight"]["ready_for_claim"] is False
    assert any(blocker.startswith("Region:") for blocker in preview["preflight"]["claim_gate_blockers"])
    assert any(blocker.startswith("Runtime flags:") for blocker in preview["preflight"]["claim_gate_blockers"])


def test_preview_setup_payload_reports_appliance_wizard_and_safe_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(
        installer_module.shutil,
        "which",
        lambda name: "docker" if name == "docker" else ("nvidia-smi" if name == "nvidia-smi" else None),
    )

    class FixableManager:
        def __init__(self, label: str) -> None:
            self.label = label
            self.ensure_calls = 0

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": False,
                "label": self.label,
                "detail": f"{self.label} is available.",
            }

        def ensure_enabled(self) -> dict[str, object]:
            self.ensure_calls += 1
            return {
                "supported": True,
                "enabled": True,
                "label": self.label,
                "detail": f"{self.label} is ready.",
            }

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        raise AssertionError(f"Unexpected command: {args}")

    autostart = FixableManager("Enabled")
    launcher = FixableManager("Installed")
    installer = installer_module.GuidedInstaller(
        runtime_dir=tmp_path,
        command_runner=runner,
        autostart_manager=autostart,
        desktop_launcher_manager=launcher,
    )

    preview = installer.preview_setup_payload({"setup_mode": "quickstart", "node_label": "Warm House"})

    assert preview["preflight"]["automatic_fixes"]["attempted"] is True
    assert autostart.ensure_calls == 1
    assert launcher.ensure_calls == 1
    assert any(
        action["key"] == "autostart" and action["resolved"]
        for action in preview["preflight"]["automatic_fixes"]["actions"]
    )
    wizard = preview["owner_setup"]["first_run_wizard"]
    assert wizard["headline"] == "First-run appliance plan"
    assert [step["key"] for step in wizard["steps"]] == [
        "detect_gpu",
        "heat_profile",
        "estimate_output",
        "install_path",
        "bootstrap_model",
        "test_inference",
        "claim_node",
    ]
    assert wizard["steps"][3]["value"] == "Signed online install"


def test_run_install_stops_before_claim_when_setup_checks_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        raise AssertionError(f"Unexpected command: {args}")

    claim_attempted = False

    class FakeControlClient:
        def __init__(self, _settings) -> None:
            return

        def create_node_claim_session(self):
            nonlocal claim_attempted
            claim_attempted = True
            raise AssertionError("Claim creation should be blocked by setup preflight.")

    installer = installer_module.GuidedInstaller(
        runtime_dir=tmp_path,
        command_runner=runner,
        control_client_factory=FakeControlClient,
        sleep=lambda _seconds: None,
    )

    installer.run_install(
        {
            "setup_mode": "quickstart",
            "operator_mode": True,
            "node_label": "Blocked Node",
            "node_region": "moon-test-1",
            "vllm_model": "BAAI/bge-large-en-v1.5",
            "supported_models": "BAAI/bge-large-en-v1.5",
        }
    )

    assert claim_attempted is False
    assert installer.state.stage == "error"
    assert installer.state.error is not None
    assert installer.state.error.startswith("Region:")
    assert installer.state.error_step == "checking_docker"


def test_run_install_creates_claim_and_starts_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if normalized[:3] == ["docker", "compose", "pull"]:
            return completed(args)
        if normalized[:3] == ["docker", "compose", "up"]:
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    class FakeControlClient:
        def __init__(self, settings):
            self.settings = settings
            self.poll_count = 0

        def create_node_claim_session(self):
            return SimpleNamespace(
                claim_id="claim_123",
                claim_code="ABC123",
                approval_url="https://edge.autonomousc.com/?claim=123",
                poll_token="poll-token",
                expires_at="2099-01-01T00:00:00Z",
                poll_interval_seconds=0,
            )

        def poll_node_claim_session(self, _claim_id: str, _poll_token: str):
            self.poll_count += 1
            if self.poll_count == 1:
                return SimpleNamespace(status="pending", expires_at="2099-01-01T00:00:00Z", node_id=None, node_key=None)
            return SimpleNamespace(
                status="consumed",
                expires_at="2099-01-01T00:00:00Z",
                node_id="node_123",
                node_key="key_123456789012345678901234",
            )

        def persist_credentials(self, node_id: str, node_key: str):
            path = Path(self.settings.credentials_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"node_id": node_id, "node_key": node_key}), encoding="utf-8")

    class FakeAutoStartManager:
        def __init__(self) -> None:
            self.ensure_calls = 0

        def ensure_enabled(self):
            self.ensure_calls += 1
            return {
                "supported": True,
                "enabled": True,
                "label": "Enabled",
                "detail": "Automatic start is enabled.",
            }

        def status(self):
            return {
                "supported": True,
                "enabled": True,
                "label": "Enabled",
                "detail": "Automatic start is enabled.",
            }

    class FakeDesktopLauncherManager:
        def __init__(self) -> None:
            self.ensure_calls = 0

        def ensure_enabled(self):
            self.ensure_calls += 1
            return {
                "supported": True,
                "enabled": True,
                "label": "Installed",
                "detail": "The desktop launcher is installed.",
            }

        def status(self):
            return {
                "supported": True,
                "enabled": True,
                "label": "Installed",
                "detail": "The desktop launcher is installed.",
            }

    autostart = FakeAutoStartManager()
    launcher = FakeDesktopLauncherManager()
    validated_models: list[str | None] = []
    warmed_models: list[str | None] = []

    installer = installer_module.GuidedInstaller(
        runtime_dir=runtime_dir,
        command_runner=runner,
        control_client_factory=FakeControlClient,
        autostart_manager=autostart,
        desktop_launcher_manager=launcher,
        sleep=lambda _seconds: None,
    )
    installer.validate_hugging_face_access = lambda model=None: validated_models.append(model)  # type: ignore[method-assign]
    installer.wait_for_vllm = lambda timeout_seconds=240.0, model=None, **_kwargs: warmed_models.append(model)  # type: ignore[method-assign]

    installer.run_install(
        {
            "setup_mode": "quickstart",
            "edge_control_url": "https://edge.autonomousc.com",
            "node_label": "Nordic Heat Compute",
            "node_region": "eu-se-1",
            "trust_tier": "restricted",
            "restricted_capable": True,
            "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "supported_models": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            "max_concurrent_assignments": "2",
        }
    )

    assert installer.state.stage == "running"
    assert installer.credentials_path.exists()
    assert any(normalize_compose_args(command) == ["docker", "compose", "pull", "vllm"] for command in commands)
    assert any(normalize_compose_args(command) == ["docker", "compose", "pull", "node-agent"] for command in commands)
    assert any(normalize_compose_args(command) == ["docker", "compose", "pull", "vector"] for command in commands)
    assert any(normalize_compose_args(command) == ["docker", "compose", "up", "-d", "vllm"] for command in commands)
    assert any(
        normalize_compose_args(command) == ["docker", "compose", "up", "-d", "node-agent", "vector"]
        for command in commands
    )
    assert not (runtime_dir / ".env").exists()
    assert "Nordic Heat Compute" in installer.runtime_env_path.read_text(encoding="utf-8")
    assert "SETUP_PROFILE=balanced" in installer.runtime_env_path.read_text(encoding="utf-8")
    assert "VLLM_MODEL=BAAI/bge-large-en-v1.5" in installer.runtime_env_path.read_text(encoding="utf-8")
    assert "OWNER_TARGET_MODEL=meta-llama/Llama-3.1-8B-Instruct" in installer.runtime_env_path.read_text(encoding="utf-8")
    settings_payload = json.loads(installer.runtime_settings_path.read_text(encoding="utf-8"))
    assert settings_payload["config"]["node_label"] == "Nordic Heat Compute"
    assert settings_payload["config"]["setup_profile"] == "balanced"
    assert settings_payload["config"]["owner_target_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert validated_models == ["BAAI/bge-large-en-v1.5"]
    assert warmed_models == ["BAAI/bge-large-en-v1.5"]
    assert any("Claim code: ABC123" in message for message in installer.state.logs)
    assert any("https://edge.autonomousc.com/?claim=123" in message for message in installer.state.logs)
    assert autostart.ensure_calls == 1
    assert launcher.ensure_calls == 1


def test_run_install_uses_offline_bundle_to_seed_cache_and_skip_runtime_pulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []
    bundle_dir = runtime_dir / "offline-bundle"
    model_cache_dir = bundle_dir / "model-cache" / "models"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    (model_cache_dir / "starter.gguf").write_text("starter-cache", encoding="utf-8")
    runtime_images_dir = bundle_dir / "runtime-images"
    runtime_images_dir.mkdir(parents=True, exist_ok=True)
    (runtime_images_dir / "runtime.tar").write_text("fake-image", encoding="utf-8")

    monkeypatch.setattr(
        installer_module.shutil,
        "which",
        lambda name: "docker" if name == "docker" else ("nvidia-smi" if name == "nvidia-smi" else None),
    )

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if args[:3] == ["docker", "load", "--input"]:
            return completed(args, stdout="Loaded image: runtime\n")
        if normalized[:3] == ["docker", "compose", "up"]:
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    class FakeControlClient:
        def __init__(self, settings):
            self.settings = settings
            self.poll_count = 0

        def create_node_claim_session(self):
            return SimpleNamespace(
                claim_id="claim_bundle",
                claim_code="BUNDLE1",
                approval_url="https://edge.autonomousc.com/?claim=bundle",
                poll_token="poll-token",
                expires_at="2099-01-01T00:00:00Z",
                poll_interval_seconds=0,
            )

        def poll_node_claim_session(self, _claim_id: str, _poll_token: str):
            self.poll_count += 1
            if self.poll_count == 1:
                return SimpleNamespace(status="pending", expires_at="2099-01-01T00:00:00Z", node_id=None, node_key=None)
            return SimpleNamespace(
                status="consumed",
                expires_at="2099-01-01T00:00:00Z",
                node_id="node_bundle",
                node_key="key_bundle_12345678901234567890",
            )

        def persist_credentials(self, node_id: str, node_key: str):
            path = Path(self.settings.credentials_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"node_id": node_id, "node_key": node_key}), encoding="utf-8")

    class ReadyManager:
        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

    installer = installer_module.GuidedInstaller(
        runtime_dir=runtime_dir,
        command_runner=runner,
        control_client_factory=FakeControlClient,
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
        sleep=lambda _seconds: None,
    )
    installer.validate_hugging_face_access = lambda model=None: None  # type: ignore[method-assign]
    installer.wait_for_vllm = lambda timeout_seconds=240.0, model=None, **_kwargs: None  # type: ignore[method-assign]

    installer.run_install(
        {
            "setup_mode": "quickstart",
            "edge_control_url": "https://edge.autonomousc.com",
            "node_label": "Offline Bundle Node",
            "node_region": "eu-se-1",
            "trust_tier": "restricted",
            "restricted_capable": True,
            "offline_install_bundle_dir": str(bundle_dir),
            "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "supported_models": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            "max_concurrent_assignments": "2",
        }
    )

    assert installer.state.stage == "running"
    assert any(command[:3] == ["docker", "load", "--input"] for command in commands)
    assert not any(normalize_compose_args(command)[:3] == ["docker", "compose", "pull"] for command in commands)
    assert (runtime_dir / "data" / "model-cache" / "models" / "starter.gguf").exists()


def test_build_claim_state_includes_qr_data_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: None)

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    claim = installer_module.NodeClaimSession(
        claim_id="claim_123",
        claim_code="ABC123",
        approval_url="https://edge.autonomousc.com/?claim=123",
        poll_token="poll-token",
        expires_at="2099-01-01T00:00:00Z",
        poll_interval_seconds=10,
    )

    claim_state = installer.build_claim_state(claim, renewal_count=2)

    assert claim_state.renewal_count == 2
    assert claim_state.auto_refreshes is True
    assert claim_state.approval_qr_svg_data_url is not None
    assert claim_state.approval_qr_svg_data_url.startswith("data:image/svg+xml;base64,")


def test_installer_state_survives_restart_with_progress_and_private_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: None)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    claim = installer_module.NodeClaimSession(
        claim_id="claim_resume",
        claim_code="RESUME1",
        approval_url="https://edge.autonomousc.com/?claim=resume",
        poll_token="private-poll-token",
        expires_at=expires_at,
        poll_interval_seconds=10,
    )
    with installer.lock:
        installer.state = installer_module.InstallerState(
            stage="downloading_model",
            busy=True,
            message="Downloading bootstrap model.",
            logs=["First chunk finished."],
            stage_context={
                "warm_model": "BAAI/bge-large-en-v1.5",
                "warm_expected_bytes": 1000,
                "warm_downloaded_bytes": 420,
                "warm_progress_percent": 42,
            },
            claim=installer.build_claim_state(claim),
            resume_config={
                "setup_mode": "quickstart",
                "node_label": "Resume Node",
                "hugging_face_hub_token": "hf_private",
            },
            resume_requested=True,
        )
        installer.persist_state_unlocked()

    reloaded = installer_module.GuidedInstaller(runtime_dir=tmp_path)

    assert reloaded.state.stage == "downloading_model"
    assert reloaded.state.busy is False
    assert reloaded.state.resume_requested is True
    assert reloaded.state.stage_context["warm_progress_percent"] == 42
    assert reloaded.state.claim is not None
    assert reloaded.state.claim.poll_token == "private-poll-token"
    public_state = reloaded.state_payload()
    assert "poll_token" not in public_state["claim"]
    assert public_state["resume_config"]["hugging_face_hub_token"] == "***"


def test_run_install_reuses_valid_saved_claim_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if normalized[:3] == ["docker", "compose", "pull"]:
            return completed(args)
        if normalized[:3] == ["docker", "compose", "up"]:
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    class FakeControlClient:
        created_claims = 0
        polled: list[tuple[str, str]] = []

        def __init__(self, settings):
            self.settings = settings

        def create_node_claim_session(self):
            self.__class__.created_claims += 1
            raise AssertionError("Quick Start should reuse the valid saved claim instead of creating a new one.")

        def poll_node_claim_session(self, claim_id: str, poll_token: str):
            self.__class__.polled.append((claim_id, poll_token))
            return SimpleNamespace(
                status="consumed",
                expires_at=expires_at,
                node_id="node_from_resume",
                node_key="key_from_resume_123456789012345",
            )

        def persist_credentials(self, node_id: str, node_key: str):
            path = Path(self.settings.credentials_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"node_id": node_id, "node_key": node_key}), encoding="utf-8")

    class FakeManager:
        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

    installer = installer_module.GuidedInstaller(
        runtime_dir=runtime_dir,
        command_runner=runner,
        control_client_factory=FakeControlClient,
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
        sleep=lambda _seconds: None,
    )
    installer.set_claim(
        installer_module.InstallerClaimState(
            claim_id="claim_saved",
            claim_code="SAVED1",
            approval_url="https://edge.autonomousc.com/?claim=saved",
            expires_at=expires_at,
            poll_interval_seconds=0,
            poll_token="saved-poll-token",
        )
    )
    installer.validate_hugging_face_access = lambda model=None: None  # type: ignore[method-assign]
    installer.wait_for_vllm = lambda timeout_seconds=240.0, model=None, **_kwargs: None  # type: ignore[method-assign]

    installer.run_install(
        {
            "setup_mode": "quickstart",
            "edge_control_url": "https://edge.autonomousc.com",
            "node_label": "Nordic Heat Compute",
            "node_region": "eu-se-1",
            "trust_tier": "restricted",
            "restricted_capable": True,
            "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "supported_models": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            "max_concurrent_assignments": "2",
        },
        resume_from_stage="claiming_node",
    )

    assert installer.state.stage == "running"
    assert installer.credentials_path.exists()
    assert FakeControlClient.created_claims == 0
    assert FakeControlClient.polled == [("claim_saved", "saved-poll-token")]


def test_run_install_refreshes_claim_before_expiry_and_accepts_older_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if normalized[:3] == ["docker", "compose", "pull"]:
            return completed(args)
        if normalized[:3] == ["docker", "compose", "up"]:
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
    later = (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")

    class FakeControlClient:
        def __init__(self, settings):
            self.settings = settings
            self.created_claim_ids: list[str] = []
            self.poll_counts: dict[str, int] = {}

        def create_node_claim_session(self):
            claim_number = len(self.created_claim_ids) + 1
            claim_id = f"claim_{claim_number}"
            self.created_claim_ids.append(claim_id)
            return SimpleNamespace(
                claim_id=claim_id,
                claim_code=f"CODE{claim_number}",
                approval_url=f"https://edge.autonomousc.com/?claim={claim_number}",
                poll_token=f"poll-{claim_number}",
                expires_at=soon if claim_number == 1 else later,
                poll_interval_seconds=0,
            )

        def poll_node_claim_session(self, claim_id: str, _poll_token: str):
            self.poll_counts[claim_id] = self.poll_counts.get(claim_id, 0) + 1
            if claim_id == "claim_1" and self.poll_counts[claim_id] == 1:
                return SimpleNamespace(status="pending", expires_at=soon, node_id=None, node_key=None)
            if claim_id == "claim_1":
                return SimpleNamespace(
                    status="consumed",
                    expires_at=soon,
                    node_id="node_approved_from_first_claim",
                    node_key="key_approved_from_first_claim_123456",
                )
            return SimpleNamespace(status="pending", expires_at=later, node_id=None, node_key=None)

        def persist_credentials(self, node_id: str, node_key: str):
            path = Path(self.settings.credentials_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"node_id": node_id, "node_key": node_key}), encoding="utf-8")

    class FakeManager:
        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

    def control_factory(settings):
        client = FakeControlClient(settings)
        captured["client"] = client
        return client

    installer = installer_module.GuidedInstaller(
        runtime_dir=runtime_dir,
        command_runner=runner,
        control_client_factory=control_factory,  # type: ignore[arg-type]
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
        sleep=lambda _seconds: None,
    )
    validated_models: list[str | None] = []
    warmed_models: list[str | None] = []
    installer.validate_hugging_face_access = lambda model=None: validated_models.append(model)  # type: ignore[method-assign]
    installer.wait_for_vllm = lambda timeout_seconds=240.0, model=None, **_kwargs: warmed_models.append(model)  # type: ignore[method-assign]

    installer.run_install(
        {
            "setup_mode": "quickstart",
            "edge_control_url": "https://edge.autonomousc.com",
            "node_label": "Nordic Heat Compute",
            "node_region": "eu-se-1",
            "trust_tier": "restricted",
            "restricted_capable": True,
            "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "supported_models": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            "max_concurrent_assignments": "2",
        }
    )

    fake_client = captured["client"]
    assert isinstance(fake_client, FakeControlClient)
    assert installer.state.stage == "running"
    assert installer.credentials_path.exists()
    assert fake_client.created_claim_ids == ["claim_1", "claim_2"]
    assert fake_client.poll_counts.get("claim_1") == 2
    assert fake_client.poll_counts.get("claim_2", 0) == 0
    assert validated_models == ["BAAI/bge-large-en-v1.5"]
    assert warmed_models == ["BAAI/bge-large-en-v1.5"]
    assert any("refreshed the approval link automatically before it expired" in message for message in installer.state.logs)


def test_run_install_stops_when_claim_is_consumed_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if normalized[:3] == ["docker", "compose", "pull"]:
            return completed(args)
        if normalized[:3] == ["docker", "compose", "up"]:
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    class FakeControlClient:
        created_claims = 0

        def __init__(self, settings):
            self.settings = settings

        def create_node_claim_session(self):
            self.__class__.created_claims += 1
            return SimpleNamespace(
                claim_id="claim_123",
                claim_code="ABC123",
                approval_url="https://edge.autonomousc.com/?claim=123",
                poll_token="poll-token",
                expires_at="2099-01-01T00:00:00Z",
                poll_interval_seconds=0,
            )

        def poll_node_claim_session(self, _claim_id: str, _poll_token: str):
            return SimpleNamespace(
                status="consumed",
                expires_at="2099-01-01T00:00:00Z",
                node_id=None,
                node_key=None,
            )

        def persist_credentials(self, _node_id: str, _node_key: str):
            raise AssertionError("Quick Start should not persist credentials when the claim response is incomplete.")

    class FakeManager:
        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

    installer = installer_module.GuidedInstaller(
        runtime_dir=runtime_dir,
        command_runner=runner,
        control_client_factory=FakeControlClient,
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
        sleep=lambda _seconds: None,
    )
    installer.validate_hugging_face_access = lambda model=None: None  # type: ignore[method-assign]
    installer.wait_for_vllm = lambda timeout_seconds=240.0, model=None, **_kwargs: None  # type: ignore[method-assign]

    installer.run_install(
        {
            "setup_mode": "quickstart",
            "edge_control_url": "https://edge.autonomousc.com",
            "node_label": "Nordic Heat Compute",
            "node_region": "eu-se-1",
            "trust_tier": "restricted",
            "restricted_capable": True,
            "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "supported_models": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            "max_concurrent_assignments": "2",
        }
    )

    assert installer.state.stage == "error"
    assert installer.state.error is not None
    assert "consumed but did not return credentials" in installer.state.error
    assert FakeControlClient.created_claims == 1
    assert not installer.credentials_path.exists()
    assert any(normalize_compose_args(command) == ["docker", "compose", "up", "-d", "vllm"] for command in commands)
    assert not any(
        normalize_compose_args(command) == ["docker", "compose", "up", "-d", "node-agent", "vector"]
        for command in commands
    )


def test_validate_hugging_face_access_requires_token_for_gated_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    env_values = installer.effective_runtime_env()
    env_values["DEPLOYMENT_TARGET"] = "vast_ai"
    env_values["INFERENCE_ENGINE"] = "vllm"
    installer.write_runtime_settings(env_values)

    with pytest.raises(RuntimeError, match="HUGGING_FACE_HUB_TOKEN"):
        installer.validate_hugging_face_access("meta-llama/Llama-3.1-8B-Instruct")


def test_validate_hugging_face_access_accepts_saved_token_for_gated_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    env_values = installer.effective_runtime_env()
    env_values["DEPLOYMENT_TARGET"] = "vast_ai"
    env_values["INFERENCE_ENGINE"] = "vllm"
    env_values["HUGGING_FACE_HUB_TOKEN"] = "hf_saved_token"
    installer.write_runtime_settings(env_values)

    class FakeResponse:
        status_code = 200

    monkeypatch.setattr(installer_module.httpx, "get", lambda _url, **_kwargs: FakeResponse())

    installer.validate_hugging_face_access("meta-llama/Llama-3.1-8B-Instruct")

    assert any(
        "Validated gated Hugging Face access for meta-llama/Llama-3.1-8B-Instruct." in message
        for message in installer.state.logs
    )


def test_runtime_settings_accept_hf_token_alias(tmp_path: Path) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    env_values = installer.effective_runtime_env()
    env_values["HUGGING_FACE_HUB_TOKEN"] = ""
    env_values["HF_TOKEN"] = "hf_alias_token"
    installer.write_runtime_settings(env_values)

    persisted = installer.load_persisted_env()
    config = installer.current_config(gpu={"name": "RTX 4090", "memory_gb": 24.0})

    assert persisted["HUGGING_FACE_HUB_TOKEN"] == "hf_alias_token"
    assert persisted["HF_TOKEN"] == "hf_alias_token"
    assert config["hugging_face_token_configured"] is True


def test_validate_hugging_face_access_accepts_public_model_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)

    class FakeResponse:
        status_code = 200

    monkeypatch.setattr(installer_module.httpx, "get", lambda _url, **_kwargs: FakeResponse())

    installer.validate_hugging_face_access("BAAI/bge-large-en-v1.5")

    assert any(
        "Validated public Hugging Face access for CompendiumLabs/bge-large-en-v1.5-gguf." in message
        for message in installer.state.logs
    )


def test_run_install_skips_claim_when_credentials_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize_compose_args(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if normalized[:3] == ["docker", "compose", "up"]:
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(runtime_dir=runtime_dir, command_runner=runner)
    installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    installer.credentials_path.write_text("{}", encoding="utf-8")

    installer.run_install(
        {
            "edge_control_url": "https://edge.autonomousc.com",
            "node_label": "Existing Node",
            "node_region": "eu-se-1",
            "trust_tier": "restricted",
            "restricted_capable": True,
            "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "supported_models": "meta-llama/Llama-3.1-8B-Instruct",
            "max_concurrent_assignments": "2",
        }
    )

    assert installer.state.stage == "running"
    assert any(
        normalize_compose_args(command) == ["docker", "compose", "up", "-d", "vllm", "node-agent", "vector"]
        for command in commands
    )


def test_installer_rejects_remote_bind_even_if_remote_flag_is_requested() -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        installer_module.require_secure_bind_host("0.0.0.0", True)


def test_installer_cli_rejects_removed_allow_remote_flag() -> None:
    with pytest.raises(SystemExit):
        installer_module.main(["--allow-remote"])


def test_installer_bootstrap_exchanges_query_token_for_cookie_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: None)

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    admin_token = "local-admin-token"
    server = installer_module.ThreadingHTTPServer(
        ("127.0.0.1", 0), installer_module.make_handler(installer, admin_token)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(follow_redirects=False, timeout=5.0) as client:
            bootstrap = client.get(f"{base_url}/?token={admin_token}")
            assert bootstrap.status_code == 200
            assert installer_module.LOCAL_SESSION_COOKIE in bootstrap.headers.get("set-cookie", "")
            assert "window.location.replace('/')" in bootstrap.text

            ui = client.get(f"{base_url}/")
            assert ui.status_code == 200
            assert "AUTONOMOUSc Edge Node Installer" in ui.text

            status = client.get(f"{base_url}/api/status")
            assert status.status_code == 200
            assert status.json()["state"]["stage"] == "idle"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def test_status_payload_includes_owner_setup_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: None)

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    payload = installer.status_payload()

    assert payload["owner_setup"]["headline"] == "Start Docker Desktop"
    assert [step["label"] for step in payload["owner_setup"]["steps"]] == [
        "Checking Docker",
        "Checking NVIDIA runtime",
        "Validating HF access",
        "Pulling image",
        "Downloading model",
        "Warming model",
        "Claiming node",
        "Node live",
    ]
    assert payload["owner_setup"]["current_step"] == "checking_docker"
    assert payload["owner_setup"]["eta_label"] == "Continue once this machine is ready."
    assert payload["owner_setup"]["primary_action_label"] == "Bring this node online"
    assert "desktop_launcher" in payload


def test_status_payload_marks_hf_validation_active_when_machine_is_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    class FakeManager:
        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": False,
                "label": "Available",
                "detail": "Available.",
            }

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(
        runtime_dir=tmp_path,
        command_runner=runner,
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
    )
    payload = installer.status_payload()

    assert payload["owner_setup"]["current_step"] == "validating_hf_access"
    assert payload["owner_setup"]["headline"] == "Ready to bring this node online"
    assert payload["owner_setup"]["steps"][0]["status"] == "complete"
    assert payload["owner_setup"]["steps"][1]["status"] == "complete"
    assert payload["owner_setup"]["steps"][2]["status"] == "active"
    recommendations = {item["key"]: item for item in payload["owner_setup"]["recommendations"]}
    assert recommendations["nvidia_preset"]["value"] == "24-47 GB NVIDIA: Llama 8B + embeddings"
    assert recommendations["hf_access"]["value"] == "Public access for CompendiumLabs/bge-large-en-v1.5-gguf"
    assert "validates public Hugging Face access" in recommendations["hf_access"]["detail"]
    assert recommendations["startup_model"]["label"] == "Bootstrap model"
    assert recommendations["startup_model"]["value"] == "BAAI/bge-large-en-v1.5"
    assert recommendations["background_target"]["value"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert recommendations["warm_source"]["value"] == "Hugging Face (planned remote path)"
    assert "Warm path order" in recommendations["warm_source"]["detail"]
    assert recommendations["concurrency"]["value"] == "2 active workloads"
    assert recommendations["thermal_profile"]["value"] == "Balanced"
    assert recommendations["region"]["value"] == "eu-se-1"
    assert recommendations["startup_mode"]["value"] == "Launch on sign-in"
    assert recommendations["routing_lane"]["value"] == "Community quantized home"
    assert recommendations["privacy_ceiling"]["value"] == "Standard"
    assert recommendations["exactness_ceiling"]["value"] == "Not available"
    assert recommendations["quantized_disclosure"]["value"] == "Required"
    assert recommendations["premium_eligibility"]["value"] == "Community capacity enabled"
    assert payload["owner_setup"]["primary_action_label"] == "Bring this node online"


def test_status_payload_shows_model_cache_progress_during_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    class FakeManager:
        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Enabled",
                "detail": "Enabled.",
            }

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        raise AssertionError(f"Unexpected command: {args}")

    installer = installer_module.GuidedInstaller(
        runtime_dir=tmp_path,
        command_runner=runner,
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
    )
    installer.state = installer_module.InstallerState(
        stage="warming_model",
        busy=True,
        message="Warming the startup model and checking the local cache.",
        stage_context={
            "warm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "warm_expected_bytes": 1024**3,
            "warm_downloaded_bytes": 512 * 1024**2,
            "warm_progress_percent": 50,
            "warm_reusing_cache": False,
            "warm_observed_cache_bytes": 512 * 1024**2,
        },
    )

    payload = installer.status_payload()

    assert payload["owner_setup"]["current_step"] == "warming_model"
    assert payload["owner_setup"]["eta_label"] == (
        "First startup is only warming the tiny bootstrap model. The larger owner target keeps warming in the background after the node is online."
    )
    assert payload["owner_setup"]["steps"][4]["detail"].startswith("Downloaded about 1.0 GB")
    assert payload["owner_setup"]["steps"][5]["detail"].startswith("Finishing the local warm-up")
    assert payload["owner_setup"]["progress_percent"] > 50


def test_wait_for_vllm_tracks_cache_progress_before_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)

    observed_sizes = iter([0, 256 * 1024**2, 1024**3, 1024**3])

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    responses = iter([FakeResponse(503), FakeResponse(200)])

    monkeypatch.setattr(installer_module, "startup_model_artifact", lambda _model, runtime_engine=None: object())
    monkeypatch.setattr(installer_module, "artifact_total_size_bytes", lambda _artifact: 1024**3)
    monkeypatch.setattr(installer_module, "directory_size_bytes", lambda _path: next(observed_sizes))
    monkeypatch.setattr(installer_module.httpx, "get", lambda _url, timeout=5.0: next(responses))

    installer.wait_for_vllm(model="meta-llama/Llama-3.1-8B-Instruct")

    assert installer.state.stage == "warming_model"
    assert installer.state.stage_context["warm_progress_percent"] == 100
    assert any("First startup is downloading about 1.0 GB" in message for message in installer.state.logs)
    assert any("Model cache progress" in message for message in installer.state.logs)


def test_wait_for_vllm_resumes_partial_cache_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)

    monkeypatch.setattr(
        installer,
        "startup_model_warmup_diagnostics",
        lambda _env_values, model=None: {
            "startup_model": model or "meta-llama/Llama-3.1-8B-Instruct",
            "runtime_profile": SimpleNamespace(readiness_path="/v1/models"),
            "runtime_label": "vLLM",
            "expected_bytes": 1024**3,
            "configured_max_context_tokens": 32768,
            "context_limit_tokens": None,
            "context_limit_source": None,
            "expected_effective_context_tokens": 32768,
            "error": None,
            "error_kind": None,
        },
    )
    observed_sizes = iter([256 * 1024**2, 768 * 1024**2, 1024**3, 1024**3])

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    responses = iter([FakeResponse(503), FakeResponse(200)])

    monkeypatch.setattr(installer_module, "directory_size_bytes", lambda _path: next(observed_sizes))
    monkeypatch.setattr(installer_module.httpx, "get", lambda _url, timeout=5.0: next(responses))

    installer.wait_for_vllm(model="meta-llama/Llama-3.1-8B-Instruct")

    assert installer.state.stage == "warming_model"
    assert installer.state.stage_context["warm_resuming_download"] is True
    assert installer.state.stage_context["warm_resume_from_bytes"] == 256 * 1024**2
    assert installer.state.stage_context["warm_progress_percent"] == 100
    assert any("Resuming the local model download" in message for message in installer.state.logs)


def test_warm_source_payload_prefers_relay_mirror_before_hugging_face(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)

    payload = installer.warm_source_payload(
        env_values={
            **installer.effective_runtime_env(),
            "ARTIFACT_MIRROR_BASE_URLS": "https://relay.example/cache",
        },
        startup_model="meta-llama/Llama-3.1-8B-Instruct",
        inference_engine="llama_cpp",
        observed_cache_bytes=0,
        offline_bundle={"ready": False, "source_label": "Offline appliance bundle"},
    )

    assert payload["winner"] == "relay_cache_mirror"
    assert payload["scope"] == "planned_remote"
    assert [entry["key"] for entry in payload["order"]] == [
        "local_cache",
        "offline_appliance_bundle",
        "relay_cache_mirror",
        "hugging_face",
    ]


def test_wait_for_vllm_records_offline_bundle_as_warm_source_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)

    monkeypatch.setattr(
        installer,
        "startup_model_warmup_diagnostics",
        lambda _env_values, model=None: {
            "startup_model": model or "meta-llama/Llama-3.1-8B-Instruct",
            "runtime_profile": SimpleNamespace(readiness_path="/v1/models"),
            "runtime_label": "vLLM",
            "inference_engine": "llama_cpp",
            "expected_bytes": 1024**3,
            "configured_max_context_tokens": 32768,
            "context_limit_tokens": None,
            "context_limit_source": None,
            "expected_effective_context_tokens": 32768,
            "error": None,
            "error_kind": None,
        },
    )
    observed_sizes = iter([0, 768 * 1024**2, 1024**3, 1024**3])

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    responses = iter([FakeResponse(503), FakeResponse(200)])

    monkeypatch.setattr(installer_module, "directory_size_bytes", lambda _path: next(observed_sizes))
    monkeypatch.setattr(installer_module.httpx, "get", lambda _url, timeout=5.0: next(responses))

    installer.wait_for_vllm(
        model="meta-llama/Llama-3.1-8B-Instruct",
        offline_bundle_report={"starter_cache_seeded": True},
    )

    assert installer.state.stage_context["warm_source_winner"] == "offline_appliance_bundle"
    assert installer.state.stage_context["warm_source_scope"] == "actual"
    assert installer.state.last_warm_source["winner"] == "offline_appliance_bundle"


def test_wait_for_vllm_fails_fast_when_disk_cannot_fit_remaining_model_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)

    monkeypatch.setattr(
        installer,
        "startup_model_warmup_diagnostics",
        lambda _env_values, model=None: {
            "startup_model": model or "meta-llama/Llama-3.1-8B-Instruct",
            "runtime_profile": SimpleNamespace(readiness_path="/v1/models"),
            "runtime_label": "vLLM",
            "expected_bytes": 10 * 1024**3,
            "configured_max_context_tokens": 32768,
            "context_limit_tokens": None,
            "context_limit_source": None,
            "expected_effective_context_tokens": 32768,
            "error": None,
            "error_kind": None,
        },
    )
    monkeypatch.setattr(installer_module, "directory_size_bytes", lambda _path: 1024**3)
    monkeypatch.setattr(
        installer_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=99 * 1024**3, free=1024**3),
    )

    with pytest.raises(RuntimeError, match="cannot finish downloading meta-llama/Llama-3.1-8B-Instruct"):
        installer.wait_for_vllm(model="meta-llama/Llama-3.1-8B-Instruct")

    assert installer.state.stage_context["warm_failure_kind"] == "insufficient_disk"
    assert installer.state.stage_context["warm_missing_disk_bytes"] == 9 * 1024**3


def test_wait_for_vllm_reports_slow_download_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)

    monkeypatch.setattr(
        installer,
        "startup_model_warmup_diagnostics",
        lambda _env_values, model=None: {
            "startup_model": model or "meta-llama/Llama-3.1-8B-Instruct",
            "runtime_profile": SimpleNamespace(readiness_path="/v1/models"),
            "runtime_label": "vLLM",
            "expected_bytes": 1024**3,
            "configured_max_context_tokens": 32768,
            "context_limit_tokens": None,
            "context_limit_source": None,
            "expected_effective_context_tokens": 32768,
            "error": None,
            "error_kind": None,
        },
    )
    observed_sizes = iter([0, 10 * 1024**2, 10 * 1024**2])

    class FakeResponse:
        status_code = 503

    times = iter([0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 50.0, 50.0])

    monkeypatch.setattr(installer_module, "directory_size_bytes", lambda _path: next(observed_sizes))
    monkeypatch.setattr(installer_module.httpx, "get", lambda _url, timeout=5.0: FakeResponse())
    monkeypatch.setattr(installer_module.time, "time", lambda: next(times))
    monkeypatch.setattr(installer, "recent_runtime_logs", lambda *_args, **_kwargs: "")

    with pytest.raises(RuntimeError, match="very slow at about"):
        installer.wait_for_vllm(model="meta-llama/Llama-3.1-8B-Instruct", timeout_seconds=40.0)

    assert installer.state.stage_context["warm_failure_kind"] == "slow_download"
    assert installer.state.stage_context["warm_download_slow"] is True


def test_wait_for_vllm_detects_hugging_face_auth_failure_from_runtime_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)

    monkeypatch.setattr(
        installer,
        "startup_model_warmup_diagnostics",
        lambda _env_values, model=None: {
            "startup_model": model or "meta-llama/Llama-3.1-8B-Instruct",
            "runtime_profile": SimpleNamespace(readiness_path="/v1/models"),
            "runtime_label": "vLLM",
            "expected_bytes": 1024**3,
            "configured_max_context_tokens": 32768,
            "context_limit_tokens": None,
            "context_limit_source": None,
            "expected_effective_context_tokens": 32768,
            "error": None,
            "error_kind": None,
        },
    )

    class FakeResponse:
        status_code = 503

    times = iter([0.0, 0.0, 0.0, 0.0, 50.0])

    monkeypatch.setattr(installer_module, "directory_size_bytes", lambda _path: 0)
    monkeypatch.setattr(installer_module.httpx, "get", lambda _url, timeout=5.0: FakeResponse())
    monkeypatch.setattr(installer_module.time, "time", lambda: next(times))
    monkeypatch.setattr(
        installer,
        "recent_runtime_logs",
        lambda *_args, **_kwargs: "403 Client Error: Forbidden for url | GatedRepoError: access to model denied",
    )

    with pytest.raises(RuntimeError, match="Hugging Face denied the model download"):
        installer.wait_for_vllm(model="meta-llama/Llama-3.1-8B-Instruct", timeout_seconds=1.0)

    assert installer.state.stage_context["warm_failure_kind"] == "hugging_face_auth"
    assert "403 Client Error" in installer.state.stage_context["warm_runtime_log_excerpt"]


def test_wait_for_vllm_rejects_max_context_before_warmup(tmp_path: Path) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)

    env_values = installer.effective_runtime_env()
    env_values["RUNTIME_PROFILE"] = "home_embeddings_llama_cpp"
    env_values["DEPLOYMENT_TARGET"] = "home_edge"
    env_values["INFERENCE_ENGINE"] = "llama_cpp"
    env_values["VLLM_MODEL"] = "BAAI/bge-large-en-v1.5"
    env_values["SUPPORTED_MODELS"] = "BAAI/bge-large-en-v1.5"
    env_values["MAX_CONTEXT_TOKENS"] = "4096"
    installer.write_runtime_settings(env_values)
    installer.write_runtime_env(env_values)

    with pytest.raises(RuntimeError, match="safe limit for this model is 512 tokens"):
        installer.wait_for_vllm(model="BAAI/bge-large-en-v1.5")

    assert installer.state.stage_context["warm_failure_kind"] == "max_context"


def test_wait_for_vllm_rejects_startup_model_that_runtime_cannot_run(tmp_path: Path) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path, sleep=lambda _seconds: None)
    installer.state = installer_module.InstallerState(stage="warming_model", busy=True)
    env_values = installer.effective_runtime_env()
    env_values["RUNTIME_PROFILE"] = "vast_vllm_safetensors"
    env_values["DEPLOYMENT_TARGET"] = "vast_ai"
    env_values["INFERENCE_ENGINE"] = "vllm"
    env_values["HUGGING_FACE_HUB_TOKEN"] = "hf_token"
    env_values["VLLM_MODEL"] = "custom/unsupported-model"
    env_values["SUPPORTED_MODELS"] = "custom/unsupported-model"
    installer.write_runtime_settings(env_values)
    installer.write_runtime_env(env_values)

    with pytest.raises(RuntimeError, match="This node cannot run custom/unsupported-model"):
        installer.wait_for_vllm(model="custom/unsupported-model")

    assert installer.state.stage_context["warm_failure_kind"] == "unsupported_model"
