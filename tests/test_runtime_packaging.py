from pathlib import Path

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
    manager_section = readme.split("#### Local manager mode", 1)[1].split("Then open", 1)[0]

    assert "docker run --rm \\\n  --gpus all \\" in manager_section
