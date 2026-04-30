import json
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
        if args == ["docker", "info"]:
            return completed(args)
        if args[:3] == ["docker", "info", "--format"]:
            return completed(args, stdout='{"nvidia":{}}\n')
        if normalized[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout=running_services)
        if normalized[:3] == ["docker", "compose", "ps"]:
            return completed(args, stdout="node-agent running\nvllm running\nvector running\n")
        if normalized[:3] == ["docker", "compose", "logs"]:
            return completed(args, stdout="node-agent ready\nvector healthy\n")
        if normalized[:3] == ["docker", "compose", "up"]:
            return completed(args)
        if normalized[:3] == ["docker", "compose", "kill"]:
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


class HealthyModelResponse:
    status_code = 200

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
        self.model = model

    def json(self) -> dict[str, object]:
        return {"data": [{"id": self.model}]}


class HealthyChatCompletionResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": "OK",
                    }
                }
            ]
        }


def prepare_local_doctor_runtime(
    service: service_module.NodeRuntimeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_manager = ReadyManager()
    service.autostart_manager = ready_manager
    service.desktop_launcher_manager = ready_manager
    service.guided_installer.autostart_manager = ready_manager
    service.guided_installer.desktop_launcher_manager = ready_manager
    service.ensure_local_config()
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text("{}", encoding="utf-8")
    original_collect_preflight = service.guided_installer.collect_preflight

    def healthy_collect_preflight(*args, **kwargs):
        preflight = original_collect_preflight(*args, **kwargs)
        checks = preflight.get("setup_checks") if isinstance(preflight.get("setup_checks"), list) else []
        for check in checks:
            if not isinstance(check, dict):
                continue
            if check.get("key") == "cuda":
                check.update(
                    status="pass",
                    summary="CUDA support is visible to the host.",
                    detail="nvidia-smi reports a usable CUDA version for this machine.",
                    fix="No fix needed.",
                )
            if check.get("key") == "model_cache":
                check.update(
                    status="pass",
                    summary="The startup model cache is ready.",
                    detail="The startup model has enough local cache for fast reuse on this machine.",
                    fix="No fix needed.",
                )
        preflight["setup_checks"] = checks
        preflight["setup_audit"] = {
            "failed": 0,
            "warnings": 0,
            "total": len(checks),
            "summary": "Everything needed for claim and startup is ready on this machine.",
            "blocking_checks": [],
        }
        return preflight

    monkeypatch.setattr(service.guided_installer, "collect_preflight", healthy_collect_preflight)


def test_run_local_doctor_records_healthy_runtime_without_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda *args, **kwargs: HealthyModelResponse())
    monkeypatch.setattr(service_module.httpx, "post", lambda *args, **kwargs: HealthyChatCompletionResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
    )
    prepare_local_doctor_runtime(service, monkeypatch)

    payload = service.run_local_doctor()
    local_doctor = payload["local_doctor"]

    assert local_doctor["status"] == "healthy"
    assert local_doctor["inference_probe"]["status"] == "pass"
    assert local_doctor["warm_readiness"]["status"] == "pass"
    assert local_doctor["recommended_fix"]["label"] == "No fix needed"
    assert local_doctor["attached_bundle_name"] is None
    assert payload["diagnostics"]["last_bundle_name"] is None


def test_run_local_doctor_attaches_bundle_when_runtime_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    def fail_get(*args, **kwargs):
        raise httpx.ConnectError(
            "connection refused",
            request=httpx.Request("GET", str(args[0]) if args else "http://127.0.0.1:8000/v1/models"),
        )

    monkeypatch.setattr(service_module.httpx, "get", fail_get)

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    prepare_local_doctor_runtime(service, monkeypatch)

    payload = service.run_local_doctor()
    local_doctor = payload["local_doctor"]

    assert local_doctor["status"] == "attention"
    assert local_doctor["recommended_fix"]["label"] == "Run prerequisite-healing"
    assert local_doctor["attached_bundle_name"]
    assert payload["diagnostics"]["last_bundle_name"] == local_doctor["attached_bundle_name"]
    assert "attached diagnostics bundle" in str(local_doctor["detail"]).lower()


def test_apply_local_doctor_fix_repairs_and_records_before_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda *args, **kwargs: HealthyModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    prepare_local_doctor_runtime(service, monkeypatch)

    probes = iter(
        [
            {
                "status": "fail",
                "summary": "The tiny local inference probe failed.",
                "detail": "The runtime passed readiness, but the test request did not complete cleanly.",
                "tone": "danger",
                "attempted": True,
                "recommended_fix": {
                    "code": "repair_runtime",
                    "label": "Run prerequisite-healing",
                    "detail": "Restart the local runtime once, then let Local Doctor re-check the machine.",
                    "source": "inference_probe",
                    "automated": True,
                },
            },
            {
                "status": "pass",
                "summary": "A tiny local responses probe completed successfully.",
                "detail": "The local responses path answered on the startup model.",
                "tone": "success",
                "attempted": True,
            },
        ]
    )
    monkeypatch.setattr(service, "probe_local_inference", lambda _runtime: next(probes))

    repair_calls: list[bool] = []

    def fake_repair_runtime(*, allow_quickstart_resume: bool = True):
        repair_calls.append(allow_quickstart_resume)
        service.update_self_heal_state(
            status="healthy",
            last_result="Self-healing restarted the runtime cleanly.",
            last_action="restart_runtime",
            last_repaired_at=service_module.now_iso(),
            fix_available=False,
        )
        return service.status_payload()

    monkeypatch.setattr(service, "repair_runtime", fake_repair_runtime)

    first_payload = service.run_local_doctor()
    assert first_payload["local_doctor"]["status"] == "attention"

    payload = service.apply_local_doctor_fix()
    local_doctor = payload["local_doctor"]
    last_fix_attempt = local_doctor["last_fix_attempt"]

    assert repair_calls == [False]
    assert local_doctor["status"] == "healthy"
    assert last_fix_attempt["automated"] is True
    assert last_fix_attempt["recovered"] is True
    assert "Applied Run prerequisite-healing" in last_fix_attempt["summary"]
    assert last_fix_attempt["before_after"] == (
        "Before: Local Doctor found one thing to fix next. After: Local Doctor passed."
    )


def test_background_local_doctor_only_alerts_on_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda *args, **kwargs: HealthyModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    prepare_local_doctor_runtime(service, monkeypatch)

    probes = iter(
        [
            {
                "status": "pass",
                "summary": "A tiny local responses probe completed successfully.",
                "detail": "The local responses path answered on the startup model.",
                "tone": "success",
                "attempted": True,
            },
            {
                "status": "fail",
                "summary": "The tiny local inference probe failed.",
                "detail": "The runtime passed readiness, but the test request did not complete cleanly.",
                "tone": "danger",
                "attempted": True,
                "recommended_fix": {
                    "code": "repair_runtime",
                    "label": "Run prerequisite-healing",
                    "detail": "Restart the local runtime once, then let Local Doctor re-check the machine.",
                    "source": "inference_probe",
                    "automated": True,
                },
            },
            {
                "status": "fail",
                "summary": "The tiny local inference probe failed.",
                "detail": "The runtime passed readiness, but the test request did not complete cleanly.",
                "tone": "danger",
                "attempted": True,
                "recommended_fix": {
                    "code": "repair_runtime",
                    "label": "Run prerequisite-healing",
                    "detail": "Restart the local runtime once, then let Local Doctor re-check the machine.",
                    "source": "inference_probe",
                    "automated": True,
                },
            },
            {
                "status": "pass",
                "summary": "A tiny local responses probe completed successfully.",
                "detail": "The local responses path answered on the startup model.",
                "tone": "success",
                "attempted": True,
            },
        ]
    )
    monkeypatch.setattr(service, "probe_local_inference", lambda _runtime: next(probes))

    service.run_local_doctor(background=True, trigger="service_start", attach_bundle_on_failure=False)
    assert service.local_doctor_state.last_background_status == "healthy"
    assert service.local_doctor_state.last_transition_alert == {}
    assert service.local_doctor_state.attached_bundle_name is None

    service.run_local_doctor(background=True, trigger="self_heal:restart_runtime", attach_bundle_on_failure=False)
    assert service.local_doctor_state.last_background_status == "issue"
    assert service.local_doctor_state.last_transition_alert["code"] == "local_doctor_attention"
    assert service.local_doctor_state.attached_bundle_name is None

    service.run_local_doctor(background=True, trigger="idle_interval", attach_bundle_on_failure=False)
    assert service.local_doctor_state.last_transition_alert == {}

    payload = service.run_local_doctor(background=True, trigger="update", attach_bundle_on_failure=False)
    assert service.local_doctor_state.last_background_status == "healthy"
    assert service.local_doctor_state.last_transition_alert["code"] == "local_doctor_recovered"
    assert any(alert["code"] == "local_doctor_recovered" for alert in payload["alerts"])


def test_self_heal_loop_queues_background_doctor_after_recovery_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    queued: list[str] = []
    monkeypatch.setattr(service, "queue_background_doctor", lambda trigger: queued.append(trigger))

    def fake_self_heal_check() -> dict[str, object]:
        service.update_self_heal_state(
            status="healthy",
            last_result="Self-healing restarted the runtime cleanly.",
            last_action="restart_runtime",
            fix_available=False,
        )
        service.shutdown_event.set()
        return {}

    monkeypatch.setattr(service, "self_heal_check", fake_self_heal_check)

    service.self_heal_loop()

    assert queued == ["self_heal:restart_runtime"]


def test_apply_update_check_queues_background_doctor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.runtime_backend = service_module.SINGLE_CONTAINER_RUNTIME_BACKEND
    queued: list[str] = []
    monkeypatch.setattr(service, "queue_background_doctor", lambda trigger: queued.append(trigger))

    service.check_for_updates(apply=True)

    assert queued == ["update"]


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
        assert "diagnostics-summary.json" in names
        assert "support-summary.txt" in names
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
        support_summary = archive.read("support-summary.txt").decode("utf-8")
        assert "Service started." in service_log
        assert "Installer is waiting for approval." in installer_log
        assert "Dashboard:" in support_summary


def test_run_fault_drill_partial_download_resume_records_bundle_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda *args, **kwargs: HealthyModelResponse())
    monkeypatch.setattr("node_agent.control_plane.time.sleep", lambda _seconds: None)

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )

    payload = service.run_fault_drill("partial_download_resume")
    drill = payload["fault_drills"]["last_drill"]

    assert drill["status"] == "passed"
    assert drill["scenario"] == "partial_download_resume"
    assert "resumed cleanly" in drill["summary"].lower()

    bundle_payload = service.create_diagnostics_bundle()
    bundle_path = service.diagnostics_dir / bundle_payload["diagnostics"]["last_bundle_name"]
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        assert "fault-injection-state.json" in names
        support_summary = archive.read("support-summary.txt").decode("utf-8")
        fault_state = json.loads(archive.read("fault-injection-state.json").decode("utf-8"))

    assert "Fault drill:" in support_summary
    assert fault_state["last_drill"]["scenario"] == "partial_download_resume"
    assert fault_state["last_drill"]["status"] == "passed"


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


def test_setup_preflight_payload_overlays_preview_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )

    payload = service.setup_preflight_payload(
        {
            "setup_mode": "quickstart",
            "operator_mode": True,
            "node_label": "Preview Node",
            "node_region": "moon-test-1",
            "vllm_model": "BAAI/bge-large-en-v1.5",
            "supported_models": "BAAI/bge-large-en-v1.5",
        }
    )

    assert payload["installer"]["config"]["node_label"] == "Preview Node"
    assert payload["installer"]["config"]["node_region"] == "moon-test-1"
    assert payload["installer"]["preflight"]["ready_for_claim"] is False
    assert payload["installer"]["preflight"]["claim_gate_blockers"][0].startswith("Region:")
    assert payload["owner_setup"]["headline"] == "Set node region"
    assert payload["setup_preview"]["owner_setup"]["detail"].startswith("Use eu-se-1")
    assert payload["owner_setup"]["first_run_wizard"]["headline"] == "First-run appliance plan"


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
    monkeypatch.setattr(
        service,
        "verify_runtime_canary",
        lambda *, context: {
            "status": "pass",
            "api": "responses",
            "summary": f"{context} passed.",
        },
    )

    payload = service.check_for_updates(apply=True)

    assert payload["updates"]["pending_restart"] is False
    assert payload["updates"]["updated_images"] == ["node-agent"]
    assert "tiny local responses canary passed" in payload["updates"]["last_result"]
    assert manifest.version in service.release_env_path.read_text(encoding="utf-8")
    assert any(
        command[-6:] == ["up", "-d", "--force-recreate", "vllm", "node-agent", "vector"]
        for command in commands
    )


def test_check_for_updates_rolls_back_after_failed_local_canary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    manifest = release_manifest_module.load_release_manifest()
    updated_id = "sha256:new-node-agent"
    old_release_content = "\n".join(
        [
            "AUTONOMOUSC_RELEASE_VERSION=2026.04.01.1",
            "AUTONOMOUSC_RELEASE_CHANNEL=stable",
            "NODE_AGENT_IMAGE=anirdarrazi/autonomousc-ai-edge-runtime@sha256:old",
            f"VLLM_IMAGE={manifest.images['vllm'].ref}",
            f"VECTOR_IMAGE={manifest.images['vector'].ref}",
            "",
        ]
    )

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(
            commands,
            update_on_pull={manifest.images["node-agent"].ref: updated_id},
        ),
    )
    service.release_env_path.write_text(old_release_content, encoding="utf-8")
    service.ensure_local_config()
    previous_runtime_tuple = service.runtime_tuple_snapshot()
    service.update_self_heal_state(
        last_known_good_release_env=service.current_release_snapshot(),
        last_known_good_runtime_env=previous_runtime_tuple,
    )
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    canary_calls: list[str] = []

    def fake_canary(*, context: str):
        canary_calls.append(context)
        if len(canary_calls) == 1:
            raise RuntimeError("The tiny local inference canary failed.")
        return {
            "status": "pass",
            "api": "responses",
            "summary": f"{context} passed.",
        }

    monkeypatch.setattr(service, "verify_runtime_canary", fake_canary)

    payload = service.check_for_updates(apply=True)

    assert len(canary_calls) == 2
    assert payload["updates"]["pending_restart"] is False
    assert payload["updates"]["last_error"] is None
    assert "failed the local canary and rolled back automatically" in payload["updates"]["last_result"]
    assert service.release_env_path.read_text(encoding="utf-8") == old_release_content
    assert payload["self_healing"]["last_action"] == "rollback_bad_update"


def test_check_for_updates_respects_stable_track_preference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    manifest = release_manifest_module.load_release_manifest()

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(
        service_module,
        "load_release_manifest",
        lambda: release_manifest_module.ReleaseManifest(
            version="2026.04.15.1",
            channel="early",
            published_at=manifest.published_at,
            images=manifest.images,
        ),
    )

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands),
    )
    service.update_state.preferred_channel = "stable"

    payload = service.check_for_updates(apply=True)

    assert payload["updates"]["pending_restart"] is False
    assert payload["updates"]["updated_images"] == []
    assert "stable track" in payload["updates"]["last_result"].lower()
    assert not any(command[:2] == ["docker", "pull"] for command in commands)


def test_runtime_service_populates_owner_bundle_when_runtime_dir_is_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(layout_module.RUNTIME_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        service_module,
        "inspect_package_signature",
        lambda: {
            "verified": True,
            "kind": "appliance_package",
            "detail": "Signed package verification is isolated in this test.",
            "version": "2026.04.10.1",
            "channel": "stable",
        },
    )

    service = service_module.NodeRuntimeService(command_runner=base_runner_factory([]))

    assert service.runtime_dir == tmp_path.resolve()
    assert (tmp_path / "docker-compose.yml").exists()
    assert (tmp_path / ".env.example").exists()
    assert (tmp_path / "appliance-runtime-manifest.json").exists()
    assert (tmp_path / "release-manifest.json").exists()
    assert (tmp_path / "vector.toml").exists()
    payload = service.status_payload()
    assert payload["appliance_release"]["verified"] is True


def test_runtime_service_copies_bundled_offline_appliance_bundle_when_runtime_dir_is_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled_root = tmp_path / "bundled-runtime"
    (bundled_root / "data" / "service" / "offline-bundle" / "model-cache").mkdir(parents=True, exist_ok=True)
    (bundled_root / ".env.example").write_text("EDGE_CONTROL_URL=https://edge.autonomousc.com\n", encoding="utf-8")
    (bundled_root / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (bundled_root / "vector.toml").write_text("[sources]\n", encoding="utf-8")
    (
        bundled_root / "data" / "service" / "offline-bundle" / "model-cache" / "starter.gguf"
    ).write_text("starter", encoding="utf-8")

    monkeypatch.setenv(layout_module.RUNTIME_DIR_ENV, str(tmp_path / "owner-runtime"))
    monkeypatch.setattr(layout_module, "bundled_runtime_dir", lambda: bundled_root)
    monkeypatch.setattr(layout_module, "verify_runtime_bundle_dir", lambda _path=None: None)

    service = service_module.NodeRuntimeService(command_runner=base_runner_factory([]))

    assert (
        service.runtime_dir / "data" / "service" / "offline-bundle" / "model-cache" / "starter.gguf"
    ).exists()


def test_runtime_service_uses_persisted_single_container_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.delenv(installer_module.RUNTIME_BACKEND_ENV, raising=False)
    monkeypatch.setattr(service_module, "detect_runtime_backend", lambda: "manager")
    monkeypatch.setattr(installer_module, "detect_runtime_backend", lambda: "manager")

    installer = installer_module.GuidedInstaller(runtime_dir=tmp_path)
    installer.write_runtime_settings(
        {
            installer_module.RUNTIME_BACKEND_ENV: installer_module.SINGLE_CONTAINER_RUNTIME_BACKEND,
            "NODE_LABEL": "Owner Node",
        }
    )

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory([]),
    )

    assert service.runtime_backend == service_module.SINGLE_CONTAINER_RUNTIME_BACKEND
    assert service.runtime_controller is not None


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
        "Ready to bring this node online",
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


def test_configure_heat_governor_persists_live_owner_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    real_build_heat_governor_plan = service_module.build_heat_governor_plan

    def build_heat_governor_plan_at_noon(*args, **kwargs):
        kwargs.setdefault("now_local", datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc))
        return real_build_heat_governor_plan(*args, **kwargs)

    monkeypatch.setattr(service_module, "build_heat_governor_plan", build_heat_governor_plan_at_noon)

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
    )

    payload = service.configure_heat_governor(
        {
            "mode": "auto",
            "owner_objective": "heat_first",
            "room_temp_c": 19.0,
            "target_temp_c": 21.0,
            "outside_temp_c": -2.0,
            "quiet_hours_start_local": "22:30",
            "quiet_hours_end_local": "06:00",
            "gpu_temp_limit_c": 80.0,
            "gpu_power_limit_enabled": False,
            "max_power_cap_watts": 220,
            "energy_price_kwh": 0.18,
        }
    )

    state = service_module.json.loads(service.heat_governor_state_path.read_text(encoding="utf-8"))
    runtime_env = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert state["mode"] == "auto"
    assert state["owner_objective"] == "heat_first"
    assert state["room_temp_c"] == 19.0
    assert "HEAT_GOVERNOR_MODE=auto" in runtime_env
    assert "OWNER_OBJECTIVE=heat_first" in runtime_env
    assert "ROOM_TEMP_C=19.0" in runtime_env
    assert "TARGET_TEMP_C=21.0" in runtime_env
    assert "OUTSIDE_TEMP_C=-2.0" in runtime_env
    assert "QUIET_HOURS_START_LOCAL=22:30" in runtime_env
    assert "QUIET_HOURS_END_LOCAL=06:00" in runtime_env
    assert "GPU_TEMP_LIMIT_C=80.0" in runtime_env
    assert "GPU_POWER_LIMIT_ENABLED=false" in runtime_env
    assert "MAX_POWER_CAP_WATTS=220" in runtime_env
    assert "ENERGY_PRICE_KWH=0.18" in runtime_env
    assert payload["heat_governor"]["mode"] == "auto"
    assert payload["heat_governor"]["owner_objective"] == "heat_first"
    assert payload["heat_governor"]["quiet_hours_start_local"] == "22:30"
    assert payload["heat_governor"]["max_power_cap_watts"] == 220
    assert payload["heat_governor"]["plan"]["effective_target_pct"] == 100


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
                    "today_usd": "0.7500",
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
    assert dashboard["today_earnings"]["value"] == "$0.7500"
    assert "transferred lifetime" in dashboard["earnings"]["detail"]
    assert dashboard["heartbeat"]["value"]
    assert dashboard["model"]["value"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert dashboard["idle"]["value"] == "Premium jobs unavailable on this machine"
    assert "community capacity" in dashboard["idle"]["detail"].lower()
    assert dashboard["changes"]["value"] == "No recent changes"
    assert dashboard["uptime"]["value"]


def test_status_payload_surfaces_owner_heat_pause_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    service.configure_heat_governor(
        {
            "mode": "auto",
            "room_temp_c": 22.0,
            "target_temp_c": 21.0,
            "outside_temp_c": 18.0,
        }
    )
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 2,
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

    assert payload["dashboard"]["cards"]["idle"]["value"] == "Owner heat target paused new work"
    assert "room is already above the owner target" in payload["dashboard"]["cards"]["idle"]["detail"].lower()
    assert any(
        alert.get("title") == "Paused because room target was reached"
        for alert in payload["dashboard"]["alerts"]
        if isinstance(alert, dict)
    )


def test_status_payload_reports_owner_economics_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            current = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
            if tz is None:
                return current
            return current.astimezone(tz)

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())
    monkeypatch.setattr(service_module, "datetime", FixedDateTime)

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
    )
    service.started_at = "2026-04-11T19:00:00+00:00"
    service.started_at_epoch = datetime(2026, 4, 11, 19, 0, 0, tzinfo=timezone.utc).timestamp()
    runtime_env = service.guided_installer.effective_runtime_env()
    runtime_env["OWNER_TARGET_MODEL"] = "meta-llama/Llama-3.1-70B-Instruct"
    runtime_env["OWNER_TARGET_SUPPORTED_MODELS"] = (
        "meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-70B-Instruct"
    )
    service.guided_installer.write_runtime_settings(runtime_env)
    service.guided_installer.write_runtime_env(runtime_env)
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.configure_heat_governor(
        {
            "mode": "auto",
            "owner_objective": "heat_first",
            "room_temp_c": 19.0,
            "target_temp_c": 21.0,
            "outside_temp_c": 4.0,
            "energy_price_kwh": 0.20,
            "max_power_cap_watts": 220,
        }
    )
    service.autopilot_state_path.write_text(
        service_module.json.dumps(
            {
                "signals": {
                    "power_watts": 200,
                    "estimated_heat_output_watts": 200,
                    "queue_depth": 0,
                    "active_assignments": 0,
                    "last_observed_at": "2026-04-12T10:00:00+00:00",
                },
                "recommendation": {
                    "heat_governor": {
                        "mode": "auto",
                        "requested_target_pct": 100,
                        "effective_target_pct": 100,
                        "paused": False,
                        "pause_reason": None,
                        "heat_demand": "high",
                        "setup_profile": "performance",
                        "thermal_headroom": 0.92,
                        "concurrency_scale": 1.0,
                        "microbatch_scale": 1.0,
                        "desired_power_limit_watts": 220,
                        "reason": "Heat-first mode raised output because the room is 2.0 C below target.",
                        "owner_objective": "heat_first",
                        "owner_objective_label": "Heat first",
                        "max_power_cap_watts": 220,
                        "decision_reasons": [
                            "Auto heat is using full compute because the room is well below the owner target.",
                            "Heat-first mode raised output because the room is 2.0 C below target.",
                            "Owner power cap is limiting GPU power to 220 W.",
                        ],
                        "room_temp_c": 19.0,
                        "target_temp_c": 21.0,
                        "outside_temp_c": 4.0,
                        "gpu_temp_limit_c": 82.0,
                    }
                },
            }
        ),
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
            "earnings": {
                "today_usd": "2.5000",
                "accrued_usd": "3.0000",
                "transferred_usd": "1.0000",
            },
            "schedulable": True,
            "blocked_reason": None,
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }

    payload = service.status_payload()
    cards = payload["dashboard"]["cards"]
    economics = payload["dashboard"]["economics"]

    assert payload["heat_governor"]["policy_summary"].startswith("Heat first")
    assert cards["today_earnings"]["value"] == "$2.50"
    assert cards["electricity"]["value"].startswith("$")
    assert cards["heat_offset"]["value"].startswith("$")
    assert cards["net_value"]["value"].startswith("$")
    assert economics["today_source_key"] == "today_usd"
    assert economics["today_source_confidence"] == "high"
    assert economics["power_source_key"] == "power_watts"
    assert economics["power_source_confidence"] == "high"
    assert economics["basis_label"] == "local midnight"
    assert economics["heat_assumption_source"] == "room target plus outside-temperature hint"
    assert economics["heat_assumption_confidence"] == "high"
    assert "today_usd" in cards["today_earnings"]["detail"]
    assert "Source: live GPU watts (power_watts, high confidence)." in cards["electricity"]["detail"]
    assert "Basis: local midnight." in cards["electricity"]["detail"]
    assert "room target plus outside-temperature hint (high confidence)" in cards["heat_offset"]["detail"]
    assert "Combines today_usd earnings" in cards["net_value"]["detail"]
    assert any(
        alert.get("title") == "Power cap is limiting GPU to 220 W"
        for alert in payload["dashboard"]["alerts"]
        if isinstance(alert, dict)
    )
    assert any(
        alert.get("title") == "Bootstrap model is serving while target model warms"
        for alert in payload["dashboard"]["alerts"]
        if isinstance(alert, dict)
    )


def test_status_payload_reports_estimated_economics_sources_when_live_data_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            current = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)
            if tz is None:
                return current
            return current.astimezone(tz)

    class FakeModelResponse:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: FakeModelResponse())
    monkeypatch.setattr(service_module, "datetime", FixedDateTime)

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="vllm\nnode-agent\nvector\n"),
    )
    service.started_at = "2026-04-12T10:30:00+00:00"
    service.started_at_epoch = datetime(2026, 4, 12, 10, 30, 0, tzinfo=timezone.utc).timestamp()
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.configure_heat_governor(
        {
            "mode": "auto",
            "outside_temp_c": 14.0,
            "energy_price_kwh": 0.20,
            "max_power_cap_watts": 180,
        }
    )
    service.autopilot_state_path.write_text(
        service_module.json.dumps(
            {
                "signals": {
                    "queue_depth": 0,
                    "active_assignments": 0,
                    "last_observed_at": "2026-04-12T10:00:00+00:00",
                },
                "recommendation": {
                    "heat_governor": {
                        "mode": "auto",
                        "requested_target_pct": 50,
                        "effective_target_pct": 50,
                        "paused": False,
                        "pause_reason": None,
                        "heat_demand": "medium",
                        "setup_profile": "balanced",
                        "thermal_headroom": 0.8,
                        "concurrency_scale": 0.6,
                        "microbatch_scale": 0.8,
                        "desired_power_limit_watts": 180,
                        "reason": "Mild weather is trimming heat demand.",
                        "owner_objective": "balanced",
                        "owner_objective_label": "Balanced",
                        "max_power_cap_watts": 180,
                        "decision_reasons": [
                            "Auto heat is trimming output because the outside-temperature hint is mild.",
                            "Owner power cap is limiting GPU power to 180 W.",
                        ],
                        "outside_temp_c": 14.0,
                        "gpu_temp_limit_c": 82.0,
                    }
                },
            }
        ),
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
            "earnings": {
                "last_24h_usd": "1.7500",
                "accrued_usd": "2.0000",
            },
            "schedulable": True,
            "blocked_reason": None,
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }

    payload = service.status_payload()
    cards = payload["dashboard"]["cards"]
    economics = payload["dashboard"]["economics"]

    assert economics["today_source_key"] == "last_24h_usd"
    assert economics["today_source_confidence"] == "medium"
    assert economics["power_source_key"] == "desired_power_limit_watts"
    assert economics["power_source_confidence"] == "low"
    assert economics["basis_label"] == "since service came online"
    assert economics["heat_assumption_source"] == "outside-temperature hint"
    assert economics["heat_assumption_confidence"] == "medium"
    assert "last_24h_usd" in cards["today_earnings"]["detail"]
    assert "Source: capped estimate (desired_power_limit_watts, low confidence)." in cards["electricity"]["detail"]
    assert "Basis: since service came online." in cards["electricity"]["detail"]
    assert "current 180 W power cap" in cards["electricity"]["detail"]
    assert "outside-temperature hint (medium confidence)" in cards["heat_offset"]["detail"]
    assert "Combines last_24h_usd earnings" in cards["net_value"]["detail"]


def test_status_payload_surfaces_degraded_connectivity_with_fallback_and_mirrors(
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
    runtime_env = service.guided_installer.effective_runtime_env()
    runtime_env["EDGE_CONTROL_FALLBACK_URLS"] = "https://edge-fallback.autonomousc.com"
    runtime_env["ARTIFACT_MIRROR_BASE_URLS"] = "https://cache-eu.autonomousc.com"
    service.guided_installer.write_runtime_settings(runtime_env)
    service.guided_installer.write_runtime_env(runtime_env)
    service.control_plane_state_path.parent.mkdir(parents=True, exist_ok=True)
    service.control_plane_state_path.write_text(
        service_module.json.dumps(
            {
                "status": "degraded",
                "primary_base_url": "https://edge.autonomousc.com",
                "active_base_url": "https://edge-fallback.autonomousc.com",
                "last_error": "temporary failure in name resolution",
                "grace_deadline_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    payload = service.status_payload()

    assert payload["connectivity"]["status"] == "degraded"
    assert payload["connectivity"]["value"] == "Degraded mode"
    assert payload["dashboard"]["cards"]["health"]["value"] == "Degraded mode"
    assert "edge-fallback.autonomousc.com" in payload["connectivity"]["detail"]
    assert "artifact mirrors are configured" in payload["connectivity"]["detail"].lower()
    assert any(
        alert.get("title") == "Running in degraded mode on fallback connectivity"
        for alert in payload["dashboard"]["alerts"]
        if isinstance(alert, dict)
    )


def test_status_payload_surfaces_docker_restart_recovery_alert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            "schedulable": True,
            "blocked_reason": None,
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }
    service.update_self_heal_state(
        last_action="restart_runtime",
        last_issue="Docker restarted under this node.",
        last_result="Restarted the runtime from the last known healthy local plan.",
        last_repaired_at="2026-04-12T10:05:00+00:00",
    )

    payload = service.status_payload()

    assert payload["dashboard"]["cards"]["changes"]["value"] == "Runtime restarted"
    assert any(
        alert.get("title") == "Node recovered automatically after Docker restart"
        for alert in payload["dashboard"]["alerts"]
        if isinstance(alert, dict)
    )


def test_dashboard_payload_includes_recent_owner_timeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
            "schedulable": True,
            "blocked_reason": None,
        },
        "synced_at": "2026-04-12T10:00:00+00:00",
        "last_error": None,
        "stale": False,
    }
    service.update_self_heal_state(
        last_action="restart_runtime",
        last_issue="Docker restarted under this node.",
        last_result="Restarted the runtime from the last known healthy local plan.",
        last_repaired_at="2026-04-12T10:05:00+00:00",
    )
    service.update_local_doctor_state(
        attached_bundle_name="diagnostics-20260412-100600.zip",
        attached_bundle_created_at="2026-04-12T10:06:00+00:00",
    )
    service.update_state.last_checked_at = "2026-04-12T10:07:00+00:00"
    service.update_state.last_result = (
        "Signed release 2026.04.12 was applied successfully after readiness and a tiny local inference canary passed."
    )

    payload = service.status_payload()

    titles = [
        str(item.get("title") or "")
        for item in payload["dashboard"]["timeline"]
        if isinstance(item, dict)
    ]
    assert "Update applied successfully" in titles
    assert "Local Doctor attached bundle" in titles
    assert "Node recovered automatically after Docker restart" in titles


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


def test_self_heal_check_prewarms_recent_assignment_mix_when_idle(
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
    service.save_autopilot_state(
        {
            "enabled": True,
            "current_model": "meta-llama/Llama-3.1-8B-Instruct",
            "signals": {"queue_depth": 0, "active_assignments": 0},
            "recommendation": {
                "startup_model": "meta-llama/Llama-3.1-8B-Instruct",
                "max_concurrent_assignments": 2,
            },
            "recent_model_mix": {
                "BAAI/bge-large-en-v1.5": 4.0,
                "meta-llama/Llama-3.1-8B-Instruct": 1.0,
            },
        }
    )
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 0,
                "active_assignments": 0,
                "runtime": {"current_model": "meta-llama/Llama-3.1-8B-Instruct"},
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
    assert "VLLM_MODEL=BAAI/bge-large-en-v1.5" in env_text
    assert service.model_cache_state.last_warmed_model == "BAAI/bge-large-en-v1.5"
    assert payload["model_cache"]["cache_budget_label"]
    assert service.self_heal_state.last_action == "model_cache_prewarm"


def test_self_heal_check_holds_prewarm_when_cache_budget_is_too_small(
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
        service_module,
        "detect_disk",
        lambda _path: {
            "free_bytes": 40 * (1024**3),
            "total_bytes": 128 * (1024**3),
            "free_gb": 40.0,
            "total_gb": 128.0,
            "recommended_free_gb": 30.0,
            "ok": True,
        },
    )
    monkeypatch.setattr(service_module, "artifact_total_size_bytes", lambda _artifact: 12 * (1024**3))

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
    env_values["MODEL_CACHE_BUDGET_GB"] = "4"
    env_values["MODEL_CACHE_RESERVE_FREE_GB"] = "30"
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)
    service.save_autopilot_state(
        {
            "enabled": True,
            "current_model": "meta-llama/Llama-3.1-8B-Instruct",
            "signals": {"queue_depth": 0, "active_assignments": 0},
            "recommendation": {
                "startup_model": "meta-llama/Llama-3.1-8B-Instruct",
                "max_concurrent_assignments": 2,
            },
            "recent_model_mix": {
                "BAAI/bge-large-en-v1.5": 5.0,
            },
        }
    )
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "id": "node_123",
                "status": "active",
                "approval_status": "approved",
                "queue_depth": 0,
                "active_assignments": 0,
                "runtime": {"current_model": "meta-llama/Llama-3.1-8B-Instruct"},
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

    payload = service.self_heal_check()

    env_text = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert "VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct" in env_text
    assert service.self_heal_state.last_action == "model_cache_budget_hold"
    assert "caps the model cache" in str(service.self_heal_state.last_result)
    assert payload["model_cache"]["cache_budget_label"] == service_module.format_bytes(4 * (1024**3))


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


def test_self_heal_check_restarts_inference_when_gpu_stays_idle_with_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    recovery_calls: list[tuple[str | None, int]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: HealthyModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
    )
    service.ensure_local_config()
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.save_autopilot_state(
        {
            "signals": {
                "queue_depth": 12,
                "active_assignments": 0,
                "gpu_utilization_pct": 3.0,
                "last_observed_at": datetime.now(timezone.utc).isoformat(),
            },
            "recommendation": {},
        }
    )
    service.self_heal_state.idle_queue_detected_at = (
        datetime.now(timezone.utc) - timedelta(seconds=service_module.IDLE_QUEUE_WATCHDOG_SECONDS + 5)
    ).isoformat()
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "queue_depth": 12,
                "active_assignments": 0,
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
            }
        },
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "last_error": None,
        "stale": False,
    }
    monkeypatch.setattr(
        service,
        "attempt_inference_runtime_recovery",
        lambda *, model=None, attempts=service_module.VLLM_WARM_RETRY_ATTEMPTS: (
            recovery_calls.append((model, attempts)) or True,
            "Restarted the local inference runtime.",
            "restart_vllm",
        ),
    )

    payload = service.self_heal_check()

    assert recovery_calls == [("meta-llama/Llama-3.1-8B-Instruct", 1)]
    assert service.self_heal_state.last_action == "restart_vllm"
    assert "idle gpu" in str(service.self_heal_state.last_result).lower()
    assert payload["self_healing"]["status"] == "healthy"


def test_self_heal_check_restarts_only_node_agent_when_runtime_signals_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []
    recovery_calls = 0

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")
    monkeypatch.setattr(service_module.httpx, "get", lambda url, timeout=4.0: HealthyModelResponse())

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
        autostart_manager=ReadyManager(),
        desktop_launcher_manager=ReadyManager(),
    )
    service.ensure_local_config()
    service.guided_installer.credentials_path.parent.mkdir(parents=True, exist_ok=True)
    service.guided_installer.credentials_path.write_text(
        '{"node_id":"node_123","node_key":"key_123456789012345678901234"}',
        encoding="utf-8",
    )
    service.save_autopilot_state(
        {
            "signals": {
                "queue_depth": 4,
                "active_assignments": 0,
                "gpu_utilization_pct": 0.0,
                "last_observed_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=service_module.RUNTIME_WEDGE_WATCHDOG_SECONDS + 10)
                ).isoformat(),
            },
            "recommendation": {},
        }
    )
    service.self_heal_state.runtime_wedge_detected_at = (
        datetime.now(timezone.utc) - timedelta(seconds=service_module.RUNTIME_WEDGE_WATCHDOG_SECONDS + 5)
    ).isoformat()
    service.remote_dashboard_snapshot = lambda force=False: {  # type: ignore[method-assign]
        "summary": {
            "node": {
                "queue_depth": 4,
                "active_assignments": 0,
                "last_heartbeat_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=service_module.RUNTIME_WEDGE_WATCHDOG_SECONDS + 10)
                ).isoformat(),
            }
        },
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "last_error": None,
        "stale": False,
    }

    def fake_node_agent_recovery() -> tuple[bool, str, str]:
        nonlocal recovery_calls
        recovery_calls += 1
        return True, "The local node agent was restarted.", "restart_node_agent"

    monkeypatch.setattr(service, "attempt_node_agent_recovery", fake_node_agent_recovery)

    payload = service.self_heal_check()

    assert recovery_calls == 1
    assert service.self_heal_state.last_action == "restart_node_agent"
    assert "signals stopped updating" in str(service.self_heal_state.last_result).lower()
    assert payload["self_healing"]["status"] == "healthy"


def test_attempt_runtime_recovery_restores_last_known_good_runtime_tuple_after_docker_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services=""),
    )
    env_values = service.guided_installer.effective_runtime_env()
    env_values["VLLM_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
    env_values["SUPPORTED_MODELS"] = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)
    service.self_heal_state.last_known_good_runtime_env = {
        "VLLM_MODEL": "BAAI/bge-large-en-v1.5",
        "SUPPORTED_MODELS": "BAAI/bge-large-en-v1.5",
        "OWNER_TARGET_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
        "OWNER_TARGET_SUPPORTED_MODELS": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
        "RUNTIME_PROFILE": "auto",
        "INFERENCE_ENGINE": "llama_cpp",
        "DEPLOYMENT_TARGET": "home_edge",
    }
    service.self_heal_state.last_known_good_bootstrap_runtime_env = dict(service.self_heal_state.last_known_good_runtime_env)
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    success, message, action = service.attempt_runtime_recovery(
        recreate=False,
        failure_reason="Docker restarted under this node.",
    )

    env_text = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert success is True
    assert action == "restart_runtime"
    assert "last known healthy local plan" in message
    assert "VLLM_MODEL=BAAI/bge-large-en-v1.5" in env_text
    assert any(
        command[:2] == ["docker", "compose"] and "--force-recreate" in command
        for command in commands
    )


@pytest.mark.parametrize(
    ("scenario", "error_message"),
    [
        ("dns_flaps", "Temporary failure in name resolution"),
        ("hf_r2_slowness", "ReadTimeout while downloading from huggingface.co"),
        ("oom_on_warm", "CUDA out of memory while warming the owner target"),
    ],
    ids=["dns_flaps", "hf_r2_slowness", "oom_on_warm"],
)
def test_attempt_vllm_recovery_rolls_back_owner_target_after_repeated_warm_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, error_message: str
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    env_values = service.guided_installer.effective_runtime_env()
    env_values["VLLM_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
    env_values["SUPPORTED_MODELS"] = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    env_values["OWNER_TARGET_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
    env_values["OWNER_TARGET_SUPPORTED_MODELS"] = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    service.guided_installer.write_runtime_settings(env_values)
    service.guided_installer.write_runtime_env(env_values)
    service.self_heal_state.last_known_good_bootstrap_runtime_env = {
        "VLLM_MODEL": "BAAI/bge-large-en-v1.5",
        "SUPPORTED_MODELS": "BAAI/bge-large-en-v1.5",
        "OWNER_TARGET_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
        "OWNER_TARGET_SUPPORTED_MODELS": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
        "RUNTIME_PROFILE": "auto",
        "INFERENCE_ENGINE": "llama_cpp",
        "DEPLOYMENT_TARGET": "home_edge",
    }
    service.self_heal_state.last_known_good_runtime_env = dict(service.self_heal_state.last_known_good_bootstrap_runtime_env)
    service.self_heal_state.model_warm_failures = {
        "meta-llama/Llama-3.1-8B-Instruct": service_module.MODEL_WARM_FAILURE_THRESHOLD - 1
    }
    monkeypatch.setattr(
        service,
        "wait_for_inference_runtime_readiness",
        lambda *, model=None, timeout_seconds=180.0: (_ for _ in ()).throw(RuntimeError(error_message)),
    )
    monkeypatch.setattr(service, "wait_for_runtime_health", lambda timeout_seconds=90.0: None)

    success, message, action = service.attempt_vllm_recovery(
        model="meta-llama/Llama-3.1-8B-Instruct",
        attempts=1,
    )

    env_text = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert scenario in {"dns_flaps", "hf_r2_slowness", "oom_on_warm"}
    assert success is True
    assert action == "rollback_owner_target_model"
    assert "rolled back" in message.lower()
    assert "VLLM_MODEL=BAAI/bge-large-en-v1.5" in env_text
    assert "OWNER_TARGET_MODEL=meta-llama/Llama-3.1-8B-Instruct" in env_text
    assert service.self_heal_state.model_warm_retry_after["meta-llama/Llama-3.1-8B-Instruct"]


def test_prewarm_likely_model_rolls_back_after_power_loss_mid_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_example_env(tmp_path)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    service = service_module.NodeRuntimeService(
        runtime_dir=tmp_path,
        command_runner=base_runner_factory(commands, running_services="node-agent\nvllm\nvector\n"),
    )
    service.ensure_local_config()
    service.self_heal_state.model_warm_failures = {
        "BAAI/bge-large-en-v1.5": service_module.MODEL_WARM_FAILURE_THRESHOLD - 1
    }

    def fail_on_both_waits(timeout_seconds: float = 90.0) -> None:
        raise RuntimeError("Connection reset while resuming a partial model download after power loss")

    monkeypatch.setattr(service, "wait_for_runtime_health", fail_on_both_waits)

    with pytest.raises(RuntimeError, match="power loss|Connection reset|last known-good bootstrap model"):
        service.prewarm_likely_model(
            "BAAI/bge-large-en-v1.5",
            "meta-llama/Llama-3.1-8B-Instruct",
            source="bootstrap_target",
        )

    env_text = service.guided_installer.runtime_env_path.read_text(encoding="utf-8")
    assert "VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct" in env_text
    assert service.self_heal_state.model_warm_retry_after["BAAI/bge-large-en-v1.5"]


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
    assert repaired_env["MAX_CONCURRENT_ASSIGNMENTS"] == "4"
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
        if args == ["docker", "info"]:
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
