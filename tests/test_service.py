import subprocess
import zipfile
from pathlib import Path

import pytest

import node_agent.installer as installer_module
import node_agent.release_manifest as release_manifest_module
import node_agent.runtime_layout as layout_module
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

    def normalize(args: list[str]) -> list[str]:
        if args[:2] != ["docker", "compose"]:
            return args
        normalized = ["docker", "compose"]
        index = 2
        while index + 1 < len(args) and args[index] == "--env-file":
            index += 2
        normalized.extend(args[index:])
        return normalized

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        normalized = normalize(args)
        if normalized[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args[:2] == ["docker", "info"]:
            return completed(args)
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout=running_services)
        if normalized[:3] == ["docker", "compose", "ps"]:
            return completed(args, stdout="node-agent running\nvllm running\nvector running\n")
        if normalized[:3] == ["docker", "compose", "logs"]:
            return completed(args, stdout="node-agent ready\nvector healthy\n")
        if normalized[:3] == ["docker", "compose", "up"]:
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
    manifest = release_manifest_module.load_release_manifest()
    updated_id = "sha256:new-node-agent"

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(
            commands,
            update_on_pull={manifest.images["node-agent"].ref: updated_id},
        ),
    )
    service.release_env_path.write_text(
        "\n".join(
            [
                "AUTONOMOUSC_RELEASE_VERSION=2026.04.01.1",
                "AUTONOMOUSC_RELEASE_CHANNEL=stable",
                "NODE_AGENT_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime@sha256:old",
                f"VLLM_IMAGE={manifest.images['vllm'].ref}",
                f"VECTOR_IMAGE={manifest.images['vector'].ref}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    payload = service.check_for_updates(apply=False)

    assert payload["updates"]["pending_restart"] is True
    assert payload["updates"]["updated_images"] == ["node-agent"]
    assert ["docker", "pull", manifest.images["node-agent"].ref] in commands


def test_check_for_updates_can_apply_and_restart_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    manifest = release_manifest_module.load_release_manifest()
    updated_id = "sha256:new-node-agent"

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(
            commands,
            update_on_pull={manifest.images["node-agent"].ref: updated_id},
        ),
    )
    service.release_env_path.write_text(
        "\n".join(
            [
                "AUTONOMOUSC_RELEASE_VERSION=2026.04.01.1",
                "AUTONOMOUSC_RELEASE_CHANNEL=stable",
                "NODE_AGENT_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime@sha256:old",
                f"VLLM_IMAGE={manifest.images['vllm'].ref}",
                f"VECTOR_IMAGE={manifest.images['vector'].ref}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    payload = service.check_for_updates(apply=True)

    assert payload["updates"]["pending_restart"] is False
    assert payload["updates"]["updated_images"] == ["node-agent"]
    assert manifest.version in service.release_env_path.read_text(encoding="utf-8")
    assert any(
        command[-6:] == ["up", "-d", "--force-recreate", "vllm", "node-agent", "vector"]
        for command in commands
    )


def test_runtime_service_populates_owner_bundle_when_runtime_dir_is_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(layout_module.RUNTIME_DIR_ENV, str(tmp_path))

    service = service_module.NodeRuntimeService(command_runner=base_runner_factory([]))

    assert service.runtime_dir == tmp_path.resolve()
    assert (tmp_path / "docker-compose.yml").exists()
    assert (tmp_path / ".env.example").exists()
    assert (tmp_path / "vector.toml").exists()


def test_spawn_background_uses_module_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(service_module.subprocess, "Popen", fake_popen)

    service_module.spawn_background(tmp_path, "127.0.0.1", 8765)

    assert captured["args"] == [
        service_module.sys.executable,
        "-m",
        "node_agent.service",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]


def test_wait_for_service_uses_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_urls: list[str] = []

    class FakeResponse:
        status_code = 200

    def fake_get(url: str, timeout: float):
        requested_urls.append(url)
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr(service_module.httpx, "get", fake_get)

    service_module.wait_for_service("127.0.0.1", 8765, timeout_seconds=0.5)

    assert requested_urls == ["http://127.0.0.1:8765/api/healthz"]


def test_command_status_reports_online_when_health_endpoint_responds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    service_dir = tmp_path / "data" / "service"
    service_dir.mkdir(parents=True)
    (service_dir / "service-meta.json").write_text(
        '{"host":"127.0.0.1","port":8765,"pid":123}',
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object] | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, timeout: float):
        if url.endswith("/api/status"):
            raise service_module.httpx.ReadTimeout("timed out")
        if url.endswith("/api/healthz"):
            return FakeResponse(200, {"ok": True})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(service_module.httpx, "get", fake_get)

    exit_code = service_module.command_status(tmp_path)

    assert exit_code == 0
    assert "Node runtime service is online" in capsys.readouterr().out
