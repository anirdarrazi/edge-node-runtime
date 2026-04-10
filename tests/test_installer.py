import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import node_agent.installer as installer_module


def completed(args: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


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


def test_current_config_prefills_detected_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args[:2] == ["docker", "info"]:
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
    assert "everyday setting" in config["profile_summary"]


def test_build_env_uses_quickstart_profile_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_example_env(tmp_path)
    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
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


def test_run_install_creates_claim_and_starts_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else "nvidia-smi")

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args[0] == "nvidia-smi":
            return completed(args, stdout="RTX 4090, 24564\n")
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args[:2] == ["docker", "info"]:
            return completed(args)
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if args[:3] == ["docker", "compose", "up"]:
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

    autostart = FakeAutoStartManager()

    installer = installer_module.GuidedInstaller(
        runtime_dir=runtime_dir,
        command_runner=runner,
        control_client_factory=FakeControlClient,
        autostart_manager=autostart,
        sleep=lambda _seconds: None,
    )
    installer.wait_for_vllm = lambda timeout_seconds=240.0: None  # type: ignore[method-assign]

    installer.run_install(
        {
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
    assert ["docker", "compose", "up", "-d", "vllm"] in commands
    assert ["docker", "compose", "up", "-d", "node-agent", "vector"] in commands
    assert "Nordic Heat Compute" in (runtime_dir / ".env").read_text(encoding="utf-8")
    assert "SETUP_PROFILE=balanced" in (runtime_dir / ".env").read_text(encoding="utf-8")
    assert autostart.ensure_calls == 1


def test_run_install_skips_claim_when_credentials_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_dir = tmp_path
    write_example_env(runtime_dir)
    commands: list[list[str]] = []

    monkeypatch.setattr(installer_module.shutil, "which", lambda name: "docker" if name == "docker" else None)

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args[:3] == ["docker", "compose", "version"]:
            return completed(args)
        if args[:2] == ["docker", "info"]:
            return completed(args)
        if args[:4] == ["docker", "compose", "ps", "--services"]:
            return completed(args, stdout="")
        if args[:3] == ["docker", "compose", "up"]:
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
    assert ["docker", "compose", "up", "-d", "vllm", "node-agent", "vector"] in commands


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
    assert payload["owner_setup"]["steps"][0]["label"] == "Check this machine"
