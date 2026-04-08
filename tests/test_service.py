import subprocess
import zipfile
from pathlib import Path

import pytest

import node_agent.installer as installer_module
import node_agent.service as service_module


def completed(args: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


def write_example_env(runtime_dir: Path) -> None:
    (runtime_dir / ".env.example").write_text(
        "\n".join(
            [
                "EDGE_CONTROL_URL=https://edge.autonomousc.com",
                "OPERATOR_TOKEN=super-secret-token",
                "NODE_LABEL=AUTONOMOUSc Nordic Node 01",
                "NODE_REGION=eu-se-1",
                "TRUST_TIER=restricted",
                "RESTRICTED_CAPABLE=true",
                "CREDENTIALS_PATH=/var/lib/autonomousc/credentials/node-credentials.json",
                "VLLM_BASE_URL=http://vllm:8000",
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
                "",
            ]
        ),
        encoding="utf-8",
    )


def base_runner_factory(
    commands: list[list[str]],
    *,
    running_services: str = "",
    image_ids: dict[str, str] | None = None,
    update_on_pull: dict[str, str] | None = None,
):
    image_ids = dict(image_ids or {})
    update_on_pull = dict(update_on_pull or {})

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args[:2] == ["docker", "info"]:
            return completed(args)
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout=running_services)
        if args[:3] == ["docker", "compose", "ps"]:
            return completed(args, stdout="node-agent running\nvllm running\nvector running\n")
        if args[:3] == ["docker", "compose", "logs"]:
            return completed(args, stdout="node-agent ready\nvector healthy\n")
        if args[:3] == ["docker", "compose", "up"]:
            return completed(args)
        if args[:3] == ["docker", "image", "inspect"]:
            image = args[3]
            return completed(args, stdout=f"{image_ids.get(image, '')}\n")
        if args[:2] == ["docker", "pull"]:
            image = args[2]
            if image in update_on_pull:
                image_ids[image] = update_on_pull[image]
            return completed(args, stdout=f"Pulled {image}\n")
        if args[0] == "nvidia-smi":
            if "--query-gpu=name,memory.total" in args:
                return completed(args, stdout="RTX 4090, 24564\n")
            return completed(args, stdout="GPU 0: RTX 4090\n")
        raise AssertionError(f"Unexpected command: {args}")

    return runner


def test_create_diagnostics_bundle_redacts_sensitive_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.log("Service started.")
    service.guided_installer.log("Installer is waiting for approval.")

    payload = service.create_diagnostics_bundle()
    bundle_path = service.diagnostics_dir / payload["diagnostics"]["last_bundle_name"]

    assert bundle_path.exists()
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        assert "status.json" in names
        assert "env.redacted" in names
        assert "service.log" in names
        assert "installer.log" in names
        assert "docker-compose-ps.txt" in names
        assert "docker-compose.logs.txt" in names
        assert "nvidia-smi.txt" in names
        assert "env.raw.snapshot" not in names

        redacted_env = archive.read("env.redacted").decode("utf-8")
        assert "OPERATOR_TOKEN=***REDACTED***" in redacted_env
        assert "super-secret-token" not in redacted_env

        service_log = archive.read("service.log").decode("utf-8")
        installer_log = archive.read("installer.log").decode("utf-8")
        assert "Service started." in service_log
        assert "Installer is waiting for approval." in installer_log


def test_check_for_updates_marks_runtime_for_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    original_id = "sha256:old"
    updated_id = "sha256:new"

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(
            commands,
            image_ids={
                service_module.UPDATE_IMAGES[0]: original_id,
                service_module.UPDATE_IMAGES[1]: original_id,
            },
            update_on_pull={service_module.UPDATE_IMAGES[0]: updated_id},
        ),
    )

    payload = service.check_for_updates(apply=False)

    assert payload["updates"]["pending_restart"] is True
    assert payload["updates"]["updated_images"] == [service_module.UPDATE_IMAGES[0]]
    assert ["docker", "pull", service_module.UPDATE_IMAGES[0]] in commands
    assert ["docker", "pull", service_module.UPDATE_IMAGES[1]] in commands


def test_check_for_updates_can_apply_and_restart_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    original_id = "sha256:old"
    updated_id = "sha256:new"

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(
            commands,
            image_ids={
                service_module.UPDATE_IMAGES[0]: original_id,
                service_module.UPDATE_IMAGES[1]: original_id,
            },
            update_on_pull={service_module.UPDATE_IMAGES[0]: updated_id},
        ),
    )

    payload = service.check_for_updates(apply=True)

    assert payload["updates"]["pending_restart"] is False
    assert payload["updates"]["updated_images"] == [service_module.UPDATE_IMAGES[0]]
    assert ["docker", "compose", "up", "-d", "--force-recreate", "vllm", "node-agent", "vector"] in commands
