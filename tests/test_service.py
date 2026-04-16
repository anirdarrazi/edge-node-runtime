import subprocess
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
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
        if args[0] == "powershell":
            return completed(args, stdout="simulated\n")
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
        if args[:3] == ["docker", "system", "prune"]:
            return completed(args, stdout="Total reclaimed space: 1.2GB\n")
        if args[:2] == ["schtasks", "/Query"]:
            raise RuntimeError("ERROR: The system cannot find the file specified.")
        if args[:2] == ["schtasks", "/Create"] or args[:2] == ["schtasks", "/Delete"]:
            return completed(args)
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
        assert "runtime-settings.redacted.json" not in names
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


def test_service_resumes_interrupted_quick_start_from_saved_checkpoint(tmp_path: Path) -> None:
    write_example_env(tmp_path)
    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    with installer.lock:
        installer.state = installer_module.InstallerState(
            stage="warming_model",
            busy=True,
            message="Warming the bootstrap model.",
            resume_config={"setup_mode": "quickstart", "node_label": "Resume Node"},
            resume_requested=True,
        )
        installer.persist_state_unlocked()

    commands: list[list[str]] = []
    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    resume_calls = 0

    def fake_resume() -> dict[str, object]:
        nonlocal resume_calls
        resume_calls += 1
        return {"state": {"busy": True}}

    service.guided_installer.resume_if_needed = fake_resume  # type: ignore[method-assign]

    service.resume_setup_if_needed()

    assert resume_calls == 1
    assert any("Resuming interrupted Quick Start setup" in message for message in service.logs)


def test_send_support_bundle_uploads_bundle_and_records_case_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeControlClient:
        def __init__(self, _settings) -> None:
            self.calls: list[tuple[str, bytes, str | None]] = []

        def submit_support_bundle(
            self,
            bundle_name: str,
            bundle_bytes: bytes,
            *,
            generated_at: str | None = None,
        ) -> dict[str, object]:
            self.calls.append((bundle_name, bundle_bytes, generated_at))
            return {
                "case_id": "support_123",
                "status": "received",
                "bundle_name": bundle_name,
                "received_at": "2026-04-12T12:00:00+00:00",
            }

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text("{}", encoding="utf-8")
    fake_client = FakeControlClient(None)
    service.guided_installer.control_client_factory = lambda _settings: fake_client  # type: ignore[method-assign]

    payload = service.send_support_bundle()

    assert service.diagnostics_state.last_case_id == "support_123"
    assert service.diagnostics_state.last_bundle_name is not None
    assert service.diagnostics_state.last_bundle_sent_at == "2026-04-12T12:00:00+00:00"
    assert "support bundle sent successfully" in service.diagnostics_state.last_result.lower()
    assert len(fake_client.calls) == 1
    bundle_name, bundle_bytes, generated_at = fake_client.calls[0]
    assert bundle_name == service.diagnostics_state.last_bundle_name
    assert bundle_bytes[:2] == b"PK"
    assert generated_at == service.diagnostics_state.last_bundle_created_at
    assert payload["diagnostics"]["last_case_id"] == "support_123"


def test_send_support_bundle_keeps_local_download_when_upload_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )

    payload = service.send_support_bundle()

    assert service.diagnostics_state.last_bundle_name is not None
    assert service.diagnostics_state.last_case_id is None
    assert service.diagnostics_state.last_error is not None
    assert "ready to download" in service.diagnostics_state.last_result.lower()
    assert payload["diagnostics"]["last_bundle_name"] == service.diagnostics_state.last_bundle_name


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


def test_status_payload_exposes_owner_setup_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
    )
    payload = service.status_payload()

    assert payload["owner_setup"]["headline"] in {
        "Add Hugging Face token",
        "Ready for Quick Start",
        "Start Quick Start",
        "Start the runtime",
    }
    assert payload["owner_setup"]["steps"][0]["label"] == "Checking Docker"
    assert payload["owner_setup"]["recommendations"]
    recommendations = {item["key"]: item for item in payload["owner_setup"]["recommendations"]}
    assert recommendations["routing_lane"]["value"] == "Community quantized home"
    assert recommendations["privacy_ceiling"]["value"] == "Standard"
    assert recommendations["exactness_ceiling"]["value"] == "Not available"
    assert recommendations["quantized_disclosure"]["value"] == "Required"
    assert payload["dashboard"]["cards"]["health"]["detail"]
    assert payload["runtime"]["routing_lane_allowed_privacy_tiers"] == ["standard"]
    assert payload["runtime"]["max_privacy_tier"] == "standard"
    assert payload["runtime"]["exact_model_guarantee"] is False
    assert payload["runtime"]["quantized_output_disclosure_required"] is True
    assert payload["self_healing"]["headline"]
    assert payload["dashboard"]["cards"]["health"]["value"]
    assert "autostart" in payload
    assert "desktop_launcher" in payload


def test_status_payload_merges_remote_owner_dashboard_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    class FakeControlClient:
        def __init__(self, _settings) -> None:
            self.calls = 0

        def fetch_node_dashboard_summary(self) -> dict[str, object]:
            self.calls += 1
            return {
                "node": {
                    "id": "node_123",
                    "status": "active",
                    "queue_depth": 1,
                    "active_assignments": 0,
                    "last_heartbeat_at": "2026-04-12T10:00:00+00:00",
                    "runtime": {"current_model": "meta-llama/Llama-3.1-8B-Instruct"},
                    "observability": {
                        "schedulable": True,
                        "schedulability_reason": None,
                    },
                },
                "earnings": {
                    "accrued_usd": "1.2500",
                    "transferred_usd": "0.5000",
                    "last_payout": {
                        "id": "payout_123",
                        "status": "accrued",
                        "amount": {"currency": "usd", "amount": "0.2500"},
                        "settlement_after": "2026-04-13T10:00:00+00:00",
                        "execution_id": "pexec_123",
                        "node_id": "node_123",
                        "created_at": "2026-04-12T09:55:00+00:00",
                    },
                },
                "schedulable": True,
                "blocked_reason": None,
            }

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    class FakeManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
    )
    service.started_at = "2026-04-12T09:00:00+00:00"
    service.started_at_epoch = max(1.0, service.started_at_epoch - 3600)
    service.guided_installer.control_client_factory = FakeControlClient
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )

    payload = service.status_payload()
    dashboard = payload["dashboard"]["cards"]

    assert payload["dashboard"]["headline"] == "Node live"
    assert dashboard["earnings"]["value"] == "$1.25 pending"
    assert "transferred lifetime" in dashboard["earnings"]["detail"]
    assert dashboard["heartbeat"]["value"]
    assert dashboard["model"]["value"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert dashboard["idle"]["value"] == "Premium jobs unavailable on this machine"
    assert "community capacity" in dashboard["idle"]["detail"].lower()
    assert dashboard["changes"]["value"] == "No recent changes"
    assert dashboard["uptime"]["value"]


def test_dashboard_payload_reports_setup_verification_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 0,
                "active_assignments": 0,
                "last_heartbeat_at": "2026-04-12T10:00:00+00:00",
                "runtime": {"current_model": "meta-llama/Llama-3.1-8B-Instruct"},
                "observability": {
                    "schedulable": True,
                    "schedulability_reason": None,
                },
            },
            "earnings": {},
            "setup_verification": {
                "status": "pending",
                "detail": "A tiny end-to-end canary is running through edge.autonomousc.com to verify this setup.",
            },
            "schedulable": True,
            "blocked_reason": None,
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }

    payload = service.status_payload()

    assert payload["dashboard"]["headline"] == "Verifying setup"
    assert payload["dashboard"]["cards"]["health"]["value"] == "Verifying setup"
    assert "canary" in payload["dashboard"]["cards"]["health"]["detail"].lower()


def test_dashboard_payload_reports_setup_verified_after_canary_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 0,
                "active_assignments": 0,
                "last_heartbeat_at": "2026-04-12T10:00:00+00:00",
                "runtime": {"current_model": "meta-llama/Llama-3.1-8B-Instruct"},
                "observability": {
                    "schedulable": True,
                    "schedulability_reason": None,
                },
            },
            "earnings": {},
            "setup_verification": {
                "status": "passed",
                "detail": "A tiny end-to-end canary completed through edge.autonomousc.com, so this setup is verified.",
            },
            "schedulable": True,
            "blocked_reason": None,
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }

    payload = service.status_payload()

    assert payload["dashboard"]["headline"] == "Setup verified"
    assert payload["dashboard"]["cards"]["health"]["value"] == "Setup verified"
    assert "setup is verified" in payload["dashboard"]["cards"]["health"]["detail"].lower()
    notification = payload["dashboard"]["setup_verification"]["notification"]
    assert notification["show"] is True
    assert notification["title"] == "Setup verified"
    assert "setup is verified" in notification["message"].lower()


def test_dashboard_payload_reports_waiting_for_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
    )
    service.ensure_local_config()
    installer_snapshot = service.guided_installer.status_payload()
    health = service.runtime_health_snapshot(installer_snapshot=installer_snapshot)
    dashboard = service.dashboard_payload(health, installer_snapshot)["cards"]

    assert dashboard["idle"]["value"] == "Waiting for approval"
    assert "approval" in dashboard["idle"]["detail"].lower()


def test_dashboard_payload_reports_no_matching_jobs_when_idle_and_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())
    monkeypatch.setattr(
        installer_module,
        "detect_attestation_provider",
        lambda *_args, **_kwargs: (
            "hardware",
            "This machine is ready for premium capacity after approval.",
        ),
    )

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    env_values = service.guided_installer.effective_runtime_env()
    assert env_values is not None
    env_values["ATTESTATION_PROVIDER"] = "hardware"
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 0,
                "active_assignments": 0,
                "last_heartbeat_at": "2026-04-12T10:00:00+00:00",
                "runtime": {"current_model": "meta-llama/Llama-3.1-8B-Instruct"},
                "observability": {
                    "schedulable": True,
                    "schedulability_reason": None,
                },
            },
            "earnings": {},
            "schedulable": True,
            "blocked_reason": None,
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }

    payload = service.status_payload()
    dashboard = payload["dashboard"]["cards"]

    assert dashboard["idle"]["value"] == "No matching jobs right now"
    assert "matching job" in dashboard["idle"]["detail"].lower()


def test_dashboard_payload_explains_model_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    cached_model_dir = (
        tmp_path
        / "data"
        / "model-cache"
        / "hub"
        / "models--meta-llama--Llama-3.1-8B-Instruct"
        / "blobs"
    )
    cached_model_dir.mkdir(parents=True)
    (cached_model_dir / "model.safetensors").write_bytes(b"x" * 1024)

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    class FakeManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {},
        "synced_at": None,
        "last_error": None,
        "stale": False,
    }

    payload = service.status_payload()
    model_cache = payload["dashboard"]["cards"]["model_cache"]

    assert model_cache["value"] == "Warm"
    assert "local cache" in model_cache["detail"].lower()
    assert "future warm-ups should be faster" in model_cache["detail"].lower()
    assert payload["model_cache"]["likely_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert payload["model_cache"]["likely_model_cache_bytes"] == 1024


def test_self_heal_check_prewarms_control_plane_likely_model_when_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    class FakeReadyManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
        autostart_manager=FakeReadyManager(),
        desktop_launcher_manager=FakeReadyManager(),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.ensure_local_config()
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 3,
                "active_assignments": 0,
                "runtime": {"current_model": "meta-llama/Llama-3.1-8B-Instruct"},
                "observability": {
                    "schedulable": True,
                    "schedulability_reason": None,
                },
            },
            "model_demand": {
                "source": "queued_matching_work",
                "recommended_model": "BAAI/bge-large-en-v1.5",
                "models": [
                    {
                        "model": "BAAI/bge-large-en-v1.5",
                        "ready_count": 4,
                        "item_count": 4,
                    }
                ],
            },
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    payload = service.self_heal_check()

    env_text = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert "VLLM_MODEL=BAAI/bge-large-en-v1.5" in env_text
    assert service.model_cache_state.last_warmed_model == "BAAI/bge-large-en-v1.5"
    assert service.self_heal_state.last_action == "model_cache_prewarm"
    assert any(
        command[:2] == ["docker", "compose"] and "--force-recreate" in command
        for command in commands
    )
    assert payload["dashboard"]["cards"]["model_cache"]["value"]


def test_self_heal_check_prewarms_owner_target_after_bootstrap_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "BAAI/bge-large-en-v1.5"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    class FakeReadyManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
        autostart_manager=FakeReadyManager(),
        desktop_launcher_manager=FakeReadyManager(),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.ensure_local_config()
    env_values = service.guided_installer.effective_runtime_env()
    env_values["RUNTIME_PROFILE"] = "auto"
    env_values["VLLM_MODEL"] = "BAAI/bge-large-en-v1.5"
    env_values["SUPPORTED_MODELS"] = "BAAI/bge-large-en-v1.5"
    env_values["OWNER_TARGET_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
    env_values["OWNER_TARGET_SUPPORTED_MODELS"] = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    env_values.update(installer_module.llama_cpp_env_for_model("BAAI/bge-large-en-v1.5"))
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 0,
                "active_assignments": 0,
                "runtime": {"current_model": "BAAI/bge-large-en-v1.5"},
                "observability": {
                    "schedulable": True,
                    "schedulability_reason": None,
                },
            },
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    payload = service.self_heal_check()

    env_text = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert "RUNTIME_PROFILE=auto" in env_text
    assert "VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct" in env_text
    assert "SUPPORTED_MODELS=meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5" in env_text
    assert "LLAMA_CPP_ALIAS=meta-llama/Llama-3.1-8B-Instruct" in env_text
    assert "LLAMA_CPP_EMBEDDING=false" in env_text
    assert service.model_cache_state.last_warmed_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert service.self_heal_state.last_action == "model_cache_prewarm"
    assert any(
        command[:2] == ["docker", "compose"] and "--force-recreate" in command
        for command in commands
    )
    assert payload["dashboard"]["cards"]["model"]["value"] in {
        "BAAI/bge-large-en-v1.5",
        "meta-llama/Llama-3.1-8B-Instruct",
    }


def test_prewarm_rolls_back_bad_model_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
    )
    service.ensure_local_config()
    attempts = {"count": 0}

    def fail_then_recover(timeout_seconds: float = 90.0) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("new model failed to warm")

    monkeypatch.setattr(service, "wait_for_runtime_health", fail_then_recover)

    with pytest.raises(RuntimeError, match="new model failed"):
        service.prewarm_likely_model("BAAI/bge-large-en-v1.5", "meta-llama/Llama-3.1-8B-Instruct")

    env_text = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert "VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct" in env_text
    assert attempts["count"] == 2
    assert sum(1 for command in commands if command[:2] == ["docker", "compose"] and "--force-recreate" in command) == 2


def test_evict_cold_model_cache_preserves_current_and_likely_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    hot_path = (
        tmp_path
        / "data"
        / "model-cache"
        / "hub"
        / "models--meta-llama--Llama-3.1-8B-Instruct"
        / "blobs"
    )
    cold_path = (
        tmp_path
        / "data"
        / "model-cache"
        / "hub"
        / "models--BAAI--bge-large-en-v1.5"
        / "blobs"
    )
    hot_path.mkdir(parents=True)
    cold_path.mkdir(parents=True)
    (hot_path / "model.safetensors").write_bytes(b"hot")
    (cold_path / "model.safetensors").write_bytes(b"cold-cache")

    now = datetime(2026, 4, 12, tzinfo=timezone.utc)
    service.model_cache_state.last_used_by_model = {
        "meta-llama/Llama-3.1-8B-Instruct": now.isoformat(),
        "BAAI/bge-large-en-v1.5": (now - timedelta(days=9)).isoformat(),
    }

    evicted_model, evicted_bytes = service.evict_cold_model_cache(
        protected_models={"meta-llama/Llama-3.1-8B-Instruct"},
        cache_pressure=True,
        now=now,
    )

    assert evicted_model == "BAAI/bge-large-en-v1.5"
    assert evicted_bytes == len(b"cold-cache")
    assert hot_path.exists()
    assert not cold_path.parent.exists()


def test_dashboard_payload_reports_staged_update_as_idle_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
    )
    service.update_state.pending_restart = True
    service.update_state.last_checked_at = "2026-04-12T10:05:00+00:00"
    service.update_state.last_result = "Signed release 2026.04.12 was downloaded and is ready to apply."

    installer_snapshot = service.guided_installer.status_payload()
    health = service.runtime_health_snapshot(installer_snapshot=installer_snapshot)
    dashboard = service.dashboard_payload(health, installer_snapshot)["cards"]

    assert dashboard["idle"]["value"] == "Update staged, restart pending"
    assert dashboard["changes"]["value"] == "Signed update staged"
    assert "ready to apply" in dashboard["changes"]["detail"].lower()


def test_runtime_health_snapshot_reports_low_disk_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.ensure_local_config()

    installer_snapshot = service.guided_installer.status_payload()
    installer_snapshot["preflight"]["disk"] = {
        "free_gb": 12,
        "total_gb": 100,
        "recommended_free_gb": 30,
        "ok": False,
    }

    health = service.runtime_health_snapshot(installer_snapshot=installer_snapshot)
    self_healing = service.self_heal_payload(health)

    assert health["issue_code"] == "disk_low"
    assert "free up at least 30 gb" in str(health["issue_detail"]).lower()
    assert self_healing["action_label"] == "Free space"
    assert self_healing["prerequisite_action"]["code"] == "free_disk_space"


def test_runtime_health_snapshot_reports_docker_not_running_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.ensure_local_config()

    installer_snapshot = service.guided_installer.status_payload()
    installer_snapshot["preflight"]["docker_cli"] = True
    installer_snapshot["preflight"]["docker_compose"] = True
    installer_snapshot["preflight"]["docker_daemon"] = False
    installer_snapshot["preflight"]["docker_error"] = "Docker Desktop is installed, but the engine is not running."

    health = service.runtime_health_snapshot(installer_snapshot=installer_snapshot)
    self_healing = service.self_heal_payload(health)

    assert health["issue_code"] == "docker_not_running"
    assert "docker" in str(health["issue_detail"]).lower()
    assert self_healing["action_label"] == "Start Docker"
    assert self_healing["prerequisite_action"]["code"] == "start_docker_desktop"


def test_runtime_health_snapshot_reports_missing_gpu_guidance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.ensure_local_config()

    installer_snapshot = service.guided_installer.status_payload()
    installer_snapshot["preflight"]["gpu"] = {
        "detected": False,
        "name": "",
        "memory_gb": None,
    }

    health = service.runtime_health_snapshot(installer_snapshot=installer_snapshot)
    self_healing = service.self_heal_payload(health)

    assert health["issue_code"] == "gpu_missing"
    assert "nvidia gpu" in str(health["issue_detail"]).lower()
    assert self_healing["action_label"] == "Install GPU driver"
    assert self_healing["prerequisite_action"]["code"] == "open_nvidia_driver_help"


def test_runtime_health_snapshot_reports_missing_nvidia_container_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.ensure_local_config()

    installer_snapshot = service.guided_installer.status_payload()
    installer_snapshot["preflight"]["nvidia_container_runtime"] = {
        "checked": True,
        "visible": False,
        "error": "Docker is running, but the NVIDIA container runtime is not visible yet.",
    }

    health = service.runtime_health_snapshot(installer_snapshot=installer_snapshot)
    self_healing = service.self_heal_payload(health)

    assert health["issue_code"] == "nvidia_runtime_missing"
    assert "nvidia container runtime" in str(health["issue_detail"]).lower()
    assert self_healing["action_label"] == "Fix GPU runtime"
    assert self_healing["prerequisite_action"]["code"] == "open_gpu_runtime_help"


def test_runtime_health_snapshot_flags_oversized_startup_model_for_small_nvidia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(
        installer_module,
        "detect_gpu",
        lambda *_args, **_kwargs: {"detected": True, "name": "RTX 4080", "memory_gb": 16.0},
    )

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    env_values = service.guided_installer.effective_runtime_env()
    env_values["GPU_MEMORY_GB"] = "16"
    env_values["GPU_NAME"] = "RTX 4080"
    env_values["VLLM_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
    env_values["SUPPORTED_MODELS"] = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)

    installer_snapshot = service.guided_installer.status_payload()
    health = service.runtime_health_snapshot(installer_snapshot=installer_snapshot)

    assert health["issue_code"] == "startup_model_too_large"
    assert "switch this machine to BAAI/bge-large-en-v1.5" in str(health["issue_detail"])


def test_attempt_vllm_recovery_restarts_only_vllm_and_retries_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    attempts = {"count": 0}

    def fail_then_recover(*, model: str | None = None, timeout_seconds: float = 180.0) -> None:
        assert model == "meta-llama/Llama-3.1-8B-Instruct"
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("still warming")

    monkeypatch.setattr(service, "wait_for_vllm_readiness", fail_then_recover)

    success, message, action = service.attempt_vllm_recovery(
        model="meta-llama/Llama-3.1-8B-Instruct",
        attempts=2,
    )

    assert success is True
    assert action == "restart_vllm"
    assert "re-warmed meta-llama/Llama-3.1-8B-Instruct" in message
    assert attempts["count"] == 2
    assert any(
        command[:2] == ["docker", "compose"]
        and "--force-recreate" in command
        and "vllm" in command
        and "node-agent" not in command
        for command in commands
    )


def test_self_heal_check_restores_saved_credentials_when_runtime_is_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    class ReadyManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
    )
    env_values = service.guided_installer.effective_runtime_env()
    env_values["NODE_ID"] = "node_123"
    env_values["NODE_KEY"] = "key_123456789012345678901234"
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)
    if service.guided_installer.credentials_path.exists():
        service.guided_installer.credentials_path.unlink()

    payload = service.self_heal_check()

    assert service.guided_installer.credentials_path.exists()
    assert service.self_heal_state.last_action == "repair_local_state"
    assert "saved node approval" in str(service.self_heal_state.last_result).lower()
    assert payload["self_healing"]["status"] == "healthy"


def test_repair_runtime_recreates_env_and_restarts_claimed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeAutoStartManager:
        def __init__(self) -> None:
            self.ensure_calls = 0

        def ensure_enabled(self) -> dict[str, object]:
            self.ensure_calls += 1
            return {
                "supported": True,
                "enabled": True,
                "label": "Enabled",
                "detail": "Automatic start is enabled.",
            }

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Enabled",
                "detail": "Automatic start is enabled.",
            }

    class FakeDesktopLauncherManager:
        def __init__(self) -> None:
            self.ensure_calls = 0

        def ensure_enabled(self) -> dict[str, object]:
            self.ensure_calls += 1
            return {
                "supported": True,
                "enabled": True,
                "label": "Installed",
                "detail": "The desktop launcher is installed.",
            }

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Installed",
                "detail": "The desktop launcher is installed.",
            }

    autostart = FakeAutoStartManager()
    launcher = FakeDesktopLauncherManager()

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
        autostart_manager=autostart,
        desktop_launcher_manager=launcher,
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    payload = service.repair_runtime()

    assert service.guided_installer.runtime_settings_path.exists()
    assert service.guided_installer.runtime_env_path.exists()
    assert "SETUP_PROFILE=" in service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert any(
        command[:2] == ["docker", "compose"] and command[-5:] == ["up", "-d", "vllm", "node-agent", "vector"]
        for command in commands
    )
    assert autostart.ensure_calls == 1
    assert launcher.ensure_calls == 1
    assert payload["runtime"]["config_present"] is True
    assert service.self_heal_state.last_action in {"start_runtime", "restart_runtime"}


def test_repair_runtime_downgrades_oversized_model_for_small_nvidia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())
    monkeypatch.setattr(
        installer_module,
        "detect_gpu",
        lambda *_args, **_kwargs: {"detected": True, "name": "RTX 4080", "memory_gb": 16.0},
    )

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    env_values = service.guided_installer.effective_runtime_env()
    env_values["GPU_MEMORY_GB"] = "16"
    env_values["GPU_NAME"] = "RTX 4080"
    env_values["NODE_ID"] = "node_123"
    env_values["NODE_KEY"] = "key_123456789012345678901234"
    env_values["VLLM_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
    env_values["SUPPORTED_MODELS"] = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)

    payload = service.repair_runtime()
    repaired_env = service.guided_installer.effective_runtime_env()

    assert repaired_env["VLLM_MODEL"] == "BAAI/bge-large-en-v1.5"
    assert repaired_env["SUPPORTED_MODELS"] == "BAAI/bge-large-en-v1.5"
    assert repaired_env["MAX_CONCURRENT_ASSIGNMENTS"] == "1"
    assert service.self_heal_state.last_action == "downgrade_startup_model"
    assert payload["self_healing"]["headline"]
    assert any(
        command[:2] == ["docker", "compose"]
        and "--force-recreate" in command
        and "node-agent" in command
        and "vllm" in command
        for command in commands
    )


def test_repair_runtime_resumes_quick_start_for_unclaimed_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    captured: dict[str, object] = {}

    class FakeManager:
        def ensure_enabled(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
    )

    def fake_start_install(config: dict[str, object]) -> dict[str, object]:
        captured["config"] = dict(config)
        return {"repair": "started"}

    service.guided_installer.start_install = fake_start_install  # type: ignore[method-assign]

    payload = service.repair_runtime()

    assert payload == {"repair": "started"}
    assert captured["config"]["setup_mode"] == "quickstart"
    assert captured["config"]["setup_profile"] == "balanced"


def test_repair_runtime_starts_docker_desktop_for_docker_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    launched: list[list[str]] = []

    class ReadyManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args[:2] == ["docker", "info"]:
            raise RuntimeError("Docker Desktop is installed, but the engine is not running.")
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[:2] == ["schtasks", "/Query"]:
            return completed(args, stdout="Status: Ready\n")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(
        service_module.subprocess,
        "Popen",
        lambda args, **_kwargs: launched.append(list(args)) or object(),
    )

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=runner,
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
    )
    monkeypatch.setattr(service, "docker_desktop_launcher", lambda: Path("C:/Program Files/Docker/Docker/Docker Desktop.exe"))

    payload = service.repair_runtime()

    assert launched == [["C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"]]
    assert service.self_heal_state.last_action == "start_docker_desktop"
    assert "docker desktop is opening" in str(service.self_heal_state.last_result).lower()
    assert payload["self_healing"]["action_label"] == "Start Docker"


def test_repair_runtime_opens_gpu_runtime_help_for_missing_nvidia_container_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    opened_urls: list[str] = []

    class ReadyManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"runc":{}}\n')
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[:2] == ["schtasks", "/Query"]:
            return completed(args, stdout="Status: Ready\n")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.webbrowser, "open", lambda url: opened_urls.append(url) or True)

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=runner,
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
    )

    payload = service.repair_runtime()

    assert opened_urls == [service_module.DOCKER_GPU_SUPPORT_URL]
    assert service.self_heal_state.last_action == "open_gpu_runtime_help"
    assert payload["self_healing"]["action_label"] == "Fix GPU runtime"


def test_repair_runtime_prunes_safe_disk_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class ReadyManager:
        def ensure_enabled(self) -> dict[str, object]:
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(
        installer_module,
        "detect_disk",
        lambda _path: {"free_gb": 8, "total_gb": 100, "recommended_free_gb": 30, "ok": False},
    )

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
    )

    payload = service.repair_runtime()

    assert any(command[:3] == ["docker", "system", "prune"] for command in commands)
    assert service.self_heal_state.last_action == "free_disk_space"
    assert "total reclaimed space" in str(service.self_heal_state.last_result).lower()
    assert payload["self_healing"]["action_label"] == "Free space"


def test_self_heal_check_starts_stopped_claimed_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    payload = service.self_heal_check()

    assert any(
        command[:2] == ["docker", "compose"] and command[-5:] == ["up", "-d", "vllm", "node-agent", "vector"]
        for command in commands
    )
    assert service.self_heal_state.last_action == "start_runtime"
    assert payload["self_healing"]["action_label"] == "Fix it"


def test_self_heal_check_rolls_back_bad_signed_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    manifest = release_manifest_module.load_release_manifest()

    class FakeModelResponse:
        status_code = 503

        def json(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text("{}", encoding="utf-8")
    service.self_heal_state.last_known_good_release_env = manifest.release_env()
    service.release_env_path.write_text(
        "\n".join(
            [
                f"{service_module.RELEASE_ENV_VERSION_KEY}=2026.04.99.1",
                f"{service_module.RELEASE_ENV_CHANNEL_KEY}=stable",
                "NODE_AGENT_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime@sha256:bad",
                f"VLLM_IMAGE={manifest.images['vllm'].ref}",
                f"VECTOR_IMAGE={manifest.images['vector'].ref}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    attempts = {"count": 0}

    def fake_wait(timeout_seconds: float = 90.0) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("vLLM health check returned HTTP 503.")

    monkeypatch.setattr(service, "wait_for_runtime_health", fake_wait)
    monkeypatch.setattr(
        service,
        "attempt_vllm_recovery",
        lambda *, model=None, attempts=service_module.VLLM_WARM_RETRY_ATTEMPTS: (
            False,
            "vLLM health check returned HTTP 503.",
            "restart_vllm",
        ),
    )

    payload = service.self_heal_check()

    assert attempts["count"] == 2
    assert service.self_heal_state.last_action == "rollback_bad_update"
    assert service.update_state.last_result == "Self-healing rolled back to the last known healthy signed release."
    assert f"{service_module.RELEASE_ENV_VERSION_KEY}={manifest.version}" in service.release_env_path.read_text(encoding="utf-8")
    assert sum(
        1
        for command in commands
        if command[:2] == ["docker", "compose"] and "--force-recreate" in command
    ) >= 2
    assert payload["self_healing"]["headline"]


def test_self_heal_check_repairs_automatic_startup_when_runtime_is_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    class FakeAutoStartManager:
        platform_name = "nt"

        def __init__(self) -> None:
            self.enabled = False
            self.ensure_calls = 0

        def ensure_enabled(self) -> dict[str, object]:
            self.ensure_calls += 1
            self.enabled = True
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": self.enabled,
                "label": "Enabled" if self.enabled else "Disabled",
                "detail": (
                    "Automatic start is enabled."
                    if self.enabled
                    else "Automatic start is disabled."
                ),
            }

    class FakeDesktopLauncherManager:
        def __init__(self) -> None:
            self.enabled = False
            self.ensure_calls = 0

        def ensure_enabled(self) -> dict[str, object]:
            self.ensure_calls += 1
            self.enabled = True
            return self.status()

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": self.enabled,
                "label": "Installed" if self.enabled else "Missing",
                "detail": (
                    "The desktop launcher is installed."
                    if self.enabled
                    else "The desktop launcher is missing."
                ),
            }

    autostart = FakeAutoStartManager()
    launcher = FakeDesktopLauncherManager()

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
        autostart_manager=autostart,
        desktop_launcher_manager=launcher,
    )
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text("{}", encoding="utf-8")

    payload = service.self_heal_check()

    assert autostart.ensure_calls == 1
    assert launcher.ensure_calls == 1
    assert service.self_heal_state.last_action == "repair_startup"
    assert "automatic startup" in str(service.self_heal_state.last_result).lower()
    assert payload["self_healing"]["status"] == "healthy"
    assert not any(
        command[:3] == ["docker", "compose", "up"]
        for command in commands
    )


def test_startup_health_payload_reports_linux_user_service_blocker(tmp_path: Path) -> None:
    class FakeAutoStartManager:
        system_name = "linux"

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        autostart_manager=FakeAutoStartManager(),
    )

    health = service.startup_health_payload(
        autostart={
            "supported": True,
            "enabled": False,
            "detail": "Linux user systemd is available for this node but is not enabled yet.",
        },
        desktop_launcher={"supported": False, "enabled": False},
        config_present=True,
        credentials_present=False,
    )

    assert health is not None
    assert health["issue_code"] == "startup_not_configured"
    assert "linux user service" in str(health["issue_detail"]).lower()
    assert health["issue_action_label"] == "Fix startup"


def test_self_heal_check_repairs_missing_config_without_resuming_quick_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FakeManager:
        def ensure_enabled(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

        def status(self) -> dict[str, object]:
            return {
                "supported": True,
                "enabled": True,
                "label": "Ready",
                "detail": "Ready.",
            }

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
        autostart_manager=FakeManager(),
        desktop_launcher_manager=FakeManager(),
    )
    service.guided_installer.start_install = lambda config: {"repair": config}  # type: ignore[method-assign]

    if service.guided_installer.runtime_settings_path.exists():
        service.guided_installer.runtime_settings_path.unlink()
    if service.guided_installer.runtime_env_path.exists():
        service.guided_installer.runtime_env_path.unlink()

    payload = service.self_heal_check()

    assert service.guided_installer.runtime_settings_path.exists()
    assert service.guided_installer.runtime_env_path.exists()
    assert payload["self_healing"]["status"] in {"attention", "waiting", "standing_by"}
    assert service.self_heal_state.last_action == "waiting_for_approval"


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


def test_command_run_rejects_remote_bind_even_if_remote_flag_is_requested() -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        service_module.require_secure_bind_host("0.0.0.0", True)


def test_service_cli_rejects_removed_allow_remote_flag() -> None:
    with pytest.raises(SystemExit):
        service_module.main(["run", "--allow-remote"])


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


def test_command_repair_calls_local_service_endpoint_and_opens_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_command_start(runtime_dir: Path, host: str, port: int, open_ui_flag: bool) -> int:
        captured["start"] = (runtime_dir, host, port, open_ui_flag)
        return 0

    def fake_load_meta(_runtime_dir: Path) -> dict[str, object]:
        return {
            "host": "127.0.0.1",
            "port": 8765,
            "admin_token": "local-admin-token",
        }

    class FakeResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"runtime": {"message": "Repair started."}}

    def fake_post(url: str, timeout: float, headers=None):
        captured["post"] = (url, timeout, headers)
        return FakeResponse()

    def fake_open_browser(host: str, port: int, admin_token: str | None = None) -> None:
        captured["open"] = (host, port, admin_token)

    monkeypatch.setattr(service_module, "command_start", fake_command_start)
    monkeypatch.setattr(service_module, "load_meta", fake_load_meta)
    monkeypatch.setattr(service_module.httpx, "post", fake_post)
    monkeypatch.setattr(service_module, "open_browser", fake_open_browser)

    exit_code = service_module.command_repair(tmp_path, "127.0.0.1", 8765, True)

    assert exit_code == 0
    assert captured["start"] == (tmp_path, "127.0.0.1", 8765, False)
    assert captured["post"] == (
        "http://127.0.0.1:8765/api/repair",
        75.0,
        {service_module.ADMIN_TOKEN_HEADER: "local-admin-token"},
    )
    assert captured["open"] == ("127.0.0.1", 8765, "local-admin-token")
    assert "Repair started." in capsys.readouterr().out


def test_command_status_reports_online_when_health_endpoint_responds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    service_dir = tmp_path / "data" / "service"
    service_dir.mkdir(parents=True)
    (service_dir / "service-meta.json").write_text(
        '{"host":"127.0.0.1","port":8765,"pid":123,"admin_token":"local-admin-token"}',
        encoding="utf-8",
    )

    observed_headers: list[dict[str, str] | None] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object] | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, timeout: float, headers=None):
        observed_headers.append(headers)
        if url.endswith("/api/status"):
            raise service_module.httpx.ReadTimeout("timed out")
        if url.endswith("/api/healthz"):
            return FakeResponse(200, {"ok": True})
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(service_module.httpx, "get", fake_get)

    exit_code = service_module.command_status(tmp_path)

    assert exit_code == 0
    assert "Node runtime service is online" in capsys.readouterr().out
    assert observed_headers[0] == {service_module.ADMIN_TOKEN_HEADER: "local-admin-token"}


def test_browser_bootstrap_exchanges_query_token_for_cookie_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda _name: None)

    service = service_module.NodeRuntimeService(runtime_dir=tmp_path)
    service.admin_token = "local-admin-token"
    server_ref: dict[str, service_module.ThreadingHTTPServer] = {}
    server = service_module.ThreadingHTTPServer(("127.0.0.1", 0), service_module.make_handler(service, server_ref))
    server_ref["server"] = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(follow_redirects=False, timeout=5.0) as client:
            bootstrap = client.get(f"{base_url}/?token=local-admin-token")
            assert bootstrap.status_code == 200
            assert service_module.LOCAL_SESSION_COOKIE in bootstrap.headers.get("set-cookie", "")
            assert "window.location.replace('/')" in bootstrap.text

            ui = client.get(f"{base_url}/")
            assert ui.status_code == 200
            assert "AUTONOMOUSc Local Node Service" in ui.text

            status = client.get(f"{base_url}/api/status")
            assert status.status_code == 200
            assert status.json()["service"]["url"] == f"http://{service.host}:{service.port}"
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()
