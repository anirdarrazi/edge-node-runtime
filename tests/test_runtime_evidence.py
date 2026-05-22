from pathlib import Path
import hashlib
import json

from node_agent.config import NodeAgentSettings
from node_agent.runtime_evidence import resolved_signed_runtime_evidence
from node_agent.runtime_tuple import resolved_default_runtime_tuple


def build_settings(credentials_path: Path) -> NodeAgentSettings:
    return NodeAgentSettings(
        edge_control_url="http://localhost:8787",
        inference_base_url="http://localhost:8000",
        credentials_path=str(credentials_path),
        operator_token="operator_token",
        node_id="node_123",
        node_key="key_123456789012345678901234",
    )


def _sorted_json(value):
    if isinstance(value, dict):
        return {key: _sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _has_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_has_none(entry) for entry in value.values())
    if isinstance(value, list):
        return any(_has_none(entry) for entry in value)
    return False


def _digest_bundle(bundle: dict) -> str:
    payload = json.dumps(_sorted_json(bundle), separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def test_runtime_evidence_digest_changes_when_effective_runtime_changes(monkeypatch, tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path)
    settings.runtime_profile = "partner_vllm_trusted"
    settings.docker_image = "sha256:" + ("a" * 64)
    settings.vllm_model = "meta-llama/Llama-3.1-8B-Instruct"
    settings.max_context_tokens = 32768
    monkeypatch.setenv("CUDA_VERSION", "12.4")

    baseline_tuple = resolved_default_runtime_tuple(settings)
    baseline_evidence = resolved_signed_runtime_evidence(settings, baseline_tuple, None)

    settings.max_context_tokens = 16384
    changed_tuple = resolved_default_runtime_tuple(settings)
    changed_evidence = resolved_signed_runtime_evidence(settings, changed_tuple, None)

    assert baseline_evidence is not None
    assert changed_evidence is not None
    assert baseline_evidence["bundle"]["effective_context_tokens"] == 32768
    assert changed_evidence["bundle"]["effective_context_tokens"] == 16384
    assert baseline_evidence["bundle"]["startup_args"]["max_context_tokens"] == 32768
    assert changed_evidence["bundle"]["startup_args"]["max_context_tokens"] == 16384
    assert baseline_evidence["bundle"]["quality_class"] == "exact_audited"
    assert baseline_evidence["bundle"]["exactness_class"] == "exact_audited"
    assert baseline_evidence["bundle"]["driver_metadata"]["cuda_version"] == "12.4"
    assert baseline_evidence["digest"] != changed_evidence["digest"]
    assert baseline_evidence["signature"] != changed_evidence["signature"]


def test_runtime_evidence_signs_the_null_stripped_heartbeat_bundle(tmp_path: Path):
    credentials_path = tmp_path / "credentials" / "node.json"
    settings = build_settings(credentials_path)
    settings.runtime_profile = "vast_vllm_safetensors"
    settings.vllm_model = "google/gemma-4-E4B-it"
    settings.max_context_tokens = 32768
    settings.gpu_name = "RTX 4060 Ti"
    settings.gpu_memory_gb = 16.0

    evidence = resolved_signed_runtime_evidence(settings, resolved_default_runtime_tuple(settings), None)

    assert evidence is not None
    assert _has_none(evidence["bundle"]) is False
    assert evidence["digest"] == _digest_bundle(evidence["bundle"])
