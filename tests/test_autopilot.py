from pathlib import Path

import pytest

from node_agent.autopilot import AutopilotController, DEFAULT_EMBEDDING_MODEL, DEFAULT_RESPONSE_MODEL
from node_agent.config import NodeAgentSettings


def build_settings(tmp_path: Path) -> NodeAgentSettings:
    return NodeAgentSettings(
        vllm_model=DEFAULT_RESPONSE_MODEL,
        supported_models=f"{DEFAULT_RESPONSE_MODEL},{DEFAULT_EMBEDDING_MODEL}",
        max_concurrent_assignments=3,
        thermal_headroom=0.8,
        autopilot_state_path=str(tmp_path / "autopilot-state.json"),
    )


def test_autopilot_reduces_concurrency_under_gpu_pressure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_settings(tmp_path)
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.93)

    autopilot.observe_assignment_success(latency_seconds=220.0)

    recommendation = autopilot.state.recommendation
    assert recommendation.setup_profile == "quiet"
    assert recommendation.max_concurrent_assignments == 1
    assert recommendation.thermal_headroom < 0.7
    assert "reduced concurrency" in recommendation.reason
    assert autopilot.capabilities_payload()["max_concurrent_assignments"] == 1


def test_autopilot_stages_smaller_model_after_repeated_pressure_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = build_settings(tmp_path)
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.97)

    for _ in range(3):
        autopilot.observe_assignment_failure(code="upstream_unavailable", retryable=True)

    recommendation = autopilot.state.recommendation
    assert recommendation.startup_model == DEFAULT_EMBEDDING_MODEL
    assert recommendation.supported_models == DEFAULT_EMBEDDING_MODEL
    assert recommendation.operations == ["embeddings"]
    assert recommendation.pending_restart is True
    assert recommendation.env_updates["VLLM_MODEL"] == DEFAULT_EMBEDDING_MODEL


def test_autopilot_uses_performance_when_demand_is_high_and_pressure_is_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = build_settings(tmp_path)
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.55)

    for _ in range(3):
        autopilot.observe_assignment_success(latency_seconds=20.0, queue_depth=3)

    recommendation = autopilot.state.recommendation
    assert recommendation.setup_profile == "performance"
    assert recommendation.max_concurrent_assignments >= 3
    assert recommendation.startup_model == DEFAULT_RESPONSE_MODEL
    assert recommendation.pending_restart is False


def test_autopilot_reports_heating_telemetry_in_capabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_settings(tmp_path)
    settings.heat_demand = "high"
    settings.room_temp_c = 18.5
    settings.target_temp_c = 21.0
    settings.energy_price_kwh = 0.12
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.55)
    monkeypatch.setattr(
        autopilot,
        "sample_gpu_thermal_metrics",
        lambda: {"gpu_temp_c": 61.0, "power_watts": 285.0, "estimated_heat_output_watts": 285.0},
    )

    autopilot.observe_idle()

    capabilities = autopilot.capabilities_payload()
    assert capabilities["heat_demand"] == "high"
    assert capabilities["room_temp_c"] == 18.5
    assert capabilities["target_temp_c"] == 21.0
    assert capabilities["gpu_temp_c"] == 61.0
    assert capabilities["power_watts"] == 285.0
    assert capabilities["estimated_heat_output_watts"] == 285.0
    assert capabilities["energy_price_kwh"] == 0.12


def test_autopilot_reports_higher_embeddings_concurrency_for_embedding_only_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=1,
        gpu_memory_gb=12.0,
        autopilot_state_path=str(tmp_path / "autopilot-embeddings.json"),
    )
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.4)

    autopilot.observe_idle()

    capabilities = autopilot.capabilities_payload()
    assert capabilities["max_concurrent_assignments"] == 1
    assert capabilities["max_concurrent_assignments_embeddings"] == 3
    assert capabilities["max_microbatch_assignments_embeddings"] == 16
    assert capabilities["max_pull_bundle_assignments"] == 16
