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


def test_readme_documents_docker_only_runtime_setup() -> None:
    readme = (RUNTIME_ROOT / "README.md").read_text(encoding="utf-8")

    assert "The runtime has one supported setup path: Docker." in readme
    assert "--gpus all" in readme
    assert "node-agent-bootstrap" in readme
    assert "browser Quick Start" in readme
    assert "open `http://127.0.0.1:8765`" not in readme


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
