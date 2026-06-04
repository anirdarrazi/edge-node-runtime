from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILES = (
    RUNTIME_ROOT / "docker-compose.yml",
    RUNTIME_ROOT / "src" / "node_agent" / "runtime_bundle" / "docker-compose.yml",
)


@pytest.mark.parametrize("compose_path", COMPOSE_FILES)
def test_inference_compose_service_requests_gpus(compose_path: Path) -> None:
    content = compose_path.read_text(encoding="utf-8")

    assert "  vllm:\n" in content
    assert "    gpus: all\n" in content


@pytest.mark.parametrize("compose_path", COMPOSE_FILES)
def test_inference_compose_command_keeps_container_shell_variables(compose_path: Path) -> None:
    content = compose_path.read_text(encoding="utf-8")

    assert "    command:\n      - |\n" in content
    assert 'profile="$${RUNTIME_PROFILE:-auto}";' in content
    assert 'engine="$${INFERENCE_ENGINE:-auto}";' in content
    assert 'if [ "$$engine" = "auto" ]' in content
    assert 'case "$$profile" in' in content
    assert '--hf-repo "$${LLAMA_CPP_HF_REPO:-$$default_hf_repo}"' in content
    assert 'set -- "$$@" --embedding;' in content
    assert 'exec "$$@";' in content
    assert '--model "$${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"' in content

    assert 'profile="${RUNTIME_PROFILE:-auto}";' not in content
    assert 'engine="${INFERENCE_ENGINE:-auto}";' not in content
    assert "    command: |\n" not in content
    assert 'if [ "$engine" = "auto" ]' not in content
    assert 'case "$profile" in' not in content
    assert 'exec "$@";' not in content


def test_manager_mode_readme_requests_gpu_access() -> None:
    readme = (RUNTIME_ROOT / "README.md").read_text(encoding="utf-8")
    manager_section_parts = readme.split("#### Advanced/Support: local manager mode", 1)
    if len(manager_section_parts) == 1:
        manager_section_parts = readme.split("#### Local manager mode", 1)
    manager_section = manager_section_parts[1].split("Then open", 1)[0]

    assert "docker run --rm \\\n  --gpus all \\" in manager_section


def test_single_container_dockerfile_does_not_bake_credential_paths_as_env() -> None:
    content = (RUNTIME_ROOT / "Dockerfile.single").read_text(encoding="utf-8")

    assert "SecretsUsedInArgOrEnv" not in content
    assert "ENV CREDENTIALS_PATH=" not in content
    assert "ENV ATTESTATION_STATE_PATH=" not in content
    assert "ENV RECOVERY_NOTE_PATH=" not in content
    assert "ENV AUTOPILOT_STATE_PATH=" not in content


@pytest.mark.parametrize("dockerfile", ("Dockerfile", "Dockerfile.service", "Dockerfile.single"))
def test_runtime_dockerfiles_pin_external_base_images(dockerfile: str) -> None:
    content = (RUNTIME_ROOT / dockerfile).read_text(encoding="utf-8")

    for line in content.splitlines():
        if line.startswith(("ARG PYTHON_BASE_IMAGE=", "ARG DOCKER_CLI_IMAGE=", "ARG VLLM_BASE_IMAGE=")):
            assert "@sha256:" in line


def test_service_dockerfile_lets_runtime_code_own_credential_paths() -> None:
    content = (RUNTIME_ROOT / "Dockerfile.service").read_text(encoding="utf-8")

    assert "ENV CREDENTIALS_PATH=" not in content
    assert "ENV ATTESTATION_STATE_PATH=" not in content
    assert "ENV RECOVERY_NOTE_PATH=" not in content
    assert "ENV AUTOPILOT_STATE_PATH=" not in content


@pytest.mark.parametrize("dockerfile", ("Dockerfile.service", "Dockerfile.single"))
def test_runtime_cuda_images_do_not_preload_models_by_default(dockerfile: str) -> None:
    content = (RUNTIME_ROOT / dockerfile).read_text(encoding="utf-8")

    assert "ARG PRELOAD_HF_MODELS=\n" in content
    assert "ARG PRELOAD_HF_MODELS=BAAI/bge-large-en-v1.5" not in content


def test_gitignore_excludes_live_drill_artifacts() -> None:
    content = (RUNTIME_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "test artifacts/" in content
    assert "test-artifacts/" in content


def test_dockerignore_excludes_local_secret_and_runtime_artifacts() -> None:
    content = (RUNTIME_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for pattern in (
        ".env",
        ".env.*",
        "local-secrets/",
        "data/",
        "logs/",
        "diagnostics/",
        "test artifacts/",
        "test-artifacts/",
        "live-drill-artifacts/",
        "*.jsonl",
    ):
        assert pattern in content


def test_built_wheel_includes_hidden_runtime_env_example(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
            str(RUNTIME_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    wheels = list(tmp_path.glob("autonomousc_node_agent-*.whl"))
    assert wheels, completed.stdout

    with zipfile.ZipFile(wheels[0]) as wheel:
        assert "node_agent/runtime_bundle/.env.example" in wheel.namelist()
        assert "node_agent/appliance-package-manifest.json" in wheel.namelist()
        assert "node_agent/appliance-package-manifest.pub" in wheel.namelist()
        assert "node_agent/runtime_bundle/appliance-runtime-manifest.json" in wheel.namelist()
        assert "node_agent/runtime_bundle/appliance-runtime-manifest.pub" in wheel.namelist()
        assert "node_agent/runtime_bundle/release-manifest.json" in wheel.namelist()
