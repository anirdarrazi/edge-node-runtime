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
    assert recommendation.max_concurrent_assignments == 2
    assert recommendation.thermal_headroom < 0.7
    assert "VRAM headroom floor" in recommendation.reason
    capabilities = autopilot.capabilities_payload()
    assert capabilities["max_concurrent_assignments"] == 2
    assert capabilities["dynamic_concurrency"]["last_constraint"] == "vram_headroom_floor"


def test_autopilot_can_allow_high_vram_pressure_for_tuned_vllm_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = build_settings(tmp_path)
    settings.max_concurrent_assignments = 2
    settings.allow_high_gpu_memory_pressure = True
    settings.min_gpu_memory_headroom_pct = 5
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.93)

    autopilot.observe_idle(queue_depth=0, active_assignments=0)

    recommendation = autopilot.state.recommendation
    assert recommendation.max_concurrent_assignments == 2
    assert "VRAM headroom floor" not in recommendation.reason


def test_autopilot_respects_configured_concurrency_cap(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.max_concurrent_assignments = 2
    settings.max_concurrent_assignments_cap = 2
    settings.allow_high_gpu_memory_pressure = True
    settings.min_gpu_memory_headroom_pct = 5
    settings.target_gpu_utilization_pct = 100
    autopilot = AutopilotController(settings)

    for _ in range(4):
        autopilot.observe_idle(
            queue_depth=8,
            active_assignments=autopilot.state.recommendation.max_concurrent_assignments,
            gpu_sample=type(
                "Sample",
                (),
                {
                    "utilization_percent": 20.0,
                    "memory_utilization_percent": 52.0,
                    "power_watts": 90.0,
                    "temperature_c": 50.0,
                },
            )(),
        )

    capabilities = autopilot.capabilities_payload()
    assert autopilot.state.recommendation.max_concurrent_assignments == 2
    assert capabilities["dynamic_concurrency"]["configured_cap"] == 2
    assert capabilities["dynamic_concurrency"]["effective_ceiling"] == 2


def test_autopilot_uses_tuned_gemma_5060_profile_concurrency_targets(tmp_path: Path) -> None:
    model = "google/gemma-4-E4B-it"
    settings = NodeAgentSettings(
        vllm_model=model,
        supported_models=model,
        runtime_profile="rtx_5060_ti_16gb_gemma4_e4b",
        max_concurrent_assignments=12,
        max_concurrent_assignments_cap=12,
        max_local_queue_assignments=24,
        pull_bundle_size=40,
        gpu_memory_gb=16.0,
        allow_high_gpu_memory_pressure=True,
        min_gpu_memory_headroom_pct=5,
        target_gpu_utilization_pct=100,
        autopilot_state_path=str(tmp_path / "autopilot-gemma-5060-tuned.json"),
    )
    autopilot = AutopilotController(settings)

    autopilot.observe_idle(queue_depth=24, active_assignments=12)

    capabilities = autopilot.capabilities_payload()
    assert autopilot.state.recommendation.max_concurrent_assignments == 12
    assert capabilities["max_concurrent_assignments"] == 12
    assert capabilities["max_local_queue_assignments"] == 24
    assert capabilities["max_pull_bundle_assignments"] == 40
    assert capabilities["dynamic_concurrency"]["hardware_ceiling"] == 12
    assert capabilities["dynamic_concurrency"]["effective_ceiling"] == 12


def test_tuned_gemma_5060_profile_advertises_configured_capacity_from_stale_state(tmp_path: Path) -> None:
    model = "google/gemma-4-E4B-it"
    settings = NodeAgentSettings(
        vllm_model=model,
        supported_models=model,
        runtime_profile="rtx_5060_ti_16gb_gemma4_e4b",
        max_concurrent_assignments=12,
        max_concurrent_assignments_cap=12,
        max_local_queue_assignments=24,
        pull_bundle_size=40,
        gpu_memory_gb=16.0,
        allow_high_gpu_memory_pressure=True,
        min_gpu_memory_headroom_pct=5,
        target_gpu_utilization_pct=100,
        autopilot_state_path=str(tmp_path / "autopilot-gemma-5060-stale.json"),
    )
    autopilot = AutopilotController(settings)
    autopilot.state.recommendation.max_concurrent_assignments = 3

    capabilities = autopilot.capabilities_payload()

    assert capabilities["max_concurrent_assignments"] == 12
    assert capabilities["max_local_queue_assignments"] == 24
    assert capabilities["max_pull_bundle_assignments"] == 40


def test_autopilot_does_not_pin_response_concurrency_on_slow_single_worker(
    tmp_path: Path,
) -> None:
    model = "google/gemma-4-E4B-it"
    settings = NodeAgentSettings(
        vllm_model=model,
        supported_models=model,
        max_concurrent_assignments=2,
        max_concurrent_assignments_cap=2,
        gpu_memory_gb=16.0,
        allow_high_gpu_memory_pressure=True,
        min_gpu_memory_headroom_pct=5,
        target_gpu_utilization_pct=100,
        autopilot_state_path=str(tmp_path / "autopilot-gemma-slow-single-worker.json"),
    )
    autopilot = AutopilotController(settings)

    autopilot.observe_assignment_success(
        latency_seconds=240.0,
        queue_depth=8,
        active_assignments=1,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 66.0,
                "memory_utilization_percent": 78.0,
                "power_watts": 118.0,
                "temperature_c": 56.0,
            },
        )(),
    )

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert recommendation.max_concurrent_assignments == 2
    assert capabilities["dynamic_concurrency"]["learned_ceiling"] is None
    assert capabilities["dynamic_concurrency"]["last_constraint"] is None


def test_autopilot_recovers_soft_latency_ceiling_when_underutilized(
    tmp_path: Path,
) -> None:
    model = "google/gemma-4-E4B-it"
    settings = NodeAgentSettings(
        vllm_model=model,
        supported_models=model,
        max_concurrent_assignments=2,
        max_concurrent_assignments_cap=2,
        gpu_memory_gb=16.0,
        allow_high_gpu_memory_pressure=True,
        min_gpu_memory_headroom_pct=5,
        target_gpu_utilization_pct=100,
        autopilot_state_path=str(tmp_path / "autopilot-gemma-latency-recovery.json"),
    )
    autopilot = AutopilotController(settings)
    autopilot.remember_safe_concurrency(
        model=model,
        safe_limit=1,
        constraint="latency_spike",
        last_limit=2,
    )
    autopilot.state.recommendation.max_concurrent_assignments = 1

    autopilot.observe_idle(
        queue_depth=8,
        active_assignments=1,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 62.0,
                "memory_utilization_percent": 74.0,
                "power_watts": 116.0,
                "temperature_c": 54.0,
            },
        )(),
    )

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert recommendation.max_concurrent_assignments == 2
    assert capabilities["dynamic_concurrency"]["learned_ceiling"] is None
    assert capabilities["dynamic_concurrency"]["last_constraint"] is None


def test_autopilot_keeps_gemma_concurrency_when_latency_is_high_but_gpu_is_under_target(
    tmp_path: Path,
) -> None:
    model = "google/gemma-4-E4B-it"
    settings = NodeAgentSettings(
        vllm_model=model,
        supported_models=model,
        max_concurrent_assignments=3,
        max_concurrent_assignments_cap=3,
        gpu_memory_gb=16.0,
        allow_high_gpu_memory_pressure=True,
        min_gpu_memory_headroom_pct=5,
        target_gpu_utilization_pct=100,
        autopilot_state_path=str(tmp_path / "autopilot-gemma-high-latency-under-target.json"),
    )
    autopilot = AutopilotController(settings)
    autopilot.state.recommendation.max_concurrent_assignments = 3

    autopilot.observe_assignment_success(
        latency_seconds=240.0,
        queue_depth=12,
        active_assignments=2,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 66.0,
                "memory_utilization_percent": 78.0,
                "power_watts": 124.0,
                "temperature_c": 62.0,
            },
        )(),
    )

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert recommendation.max_concurrent_assignments == 3
    assert capabilities["dynamic_concurrency"]["learned_ceiling"] is None
    assert capabilities["dynamic_concurrency"]["last_constraint"] is None


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


def test_autopilot_capabilities_remain_on_live_model_until_restart_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = build_settings(tmp_path)
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.97)

    for _ in range(3):
        autopilot.observe_assignment_failure(code="upstream_unavailable", retryable=True)

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert recommendation.startup_model == DEFAULT_EMBEDDING_MODEL
    assert recommendation.pending_restart is True
    assert capabilities["supported_models"] == [DEFAULT_RESPONSE_MODEL]
    assert capabilities["operations"] == ["responses"]
    assert "max_concurrent_assignments_embeddings" not in capabilities
    assert "max_microbatch_assignments_embeddings" not in capabilities
    assert capabilities["max_context_tokens"] == settings.max_context_tokens


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


def test_autopilot_tracks_recent_model_mix_for_cache_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = build_settings(tmp_path)
    autopilot = AutopilotController(settings)
    monkeypatch.setattr(autopilot, "sample_gpu_memory_pressure", lambda: 0.45)

    autopilot.observe_assignment_success(latency_seconds=12.0, queue_depth=2)
    settings.vllm_model = DEFAULT_EMBEDDING_MODEL
    autopilot.observe_assignment_success(latency_seconds=9.0, queue_depth=1)

    assert DEFAULT_EMBEDDING_MODEL in autopilot.state.recent_model_mix
    assert DEFAULT_RESPONSE_MODEL in autopilot.state.recent_model_mix
    assert (
        autopilot.state.recent_model_mix[DEFAULT_EMBEDDING_MODEL]
        > autopilot.state.recent_model_mix[DEFAULT_RESPONSE_MODEL]
    )


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
    assert capabilities["supported_models"] == [DEFAULT_EMBEDDING_MODEL]
    assert capabilities["operations"] == ["embeddings"]
    assert capabilities["max_context_tokens"] == 512
    assert capabilities["max_concurrent_assignments"] == 1
    assert capabilities["max_concurrent_assignments_embeddings"] == 1
    assert capabilities["max_microbatch_assignments_embeddings"] == 16
    assert capabilities["max_local_queue_assignments"] == 20
    assert capabilities["max_pull_bundle_assignments"] == 20


def test_autopilot_scales_up_dynamic_concurrency_when_gpu_is_below_target(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=1,
        gpu_memory_gb=16.0,
        autopilot_state_path=str(tmp_path / "autopilot-dynamic.json"),
    )
    autopilot = AutopilotController(settings)

    autopilot.observe_idle(
        queue_depth=3,
        active_assignments=1,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 22.0,
                "memory_utilization_percent": 41.0,
                "power_watts": 80.0,
                "temperature_c": 52.0,
            },
        )(),
    )

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert recommendation.max_concurrent_assignments >= 2
    assert capabilities["max_concurrent_assignments_embeddings"] == recommendation.max_concurrent_assignments
    assert capabilities["max_local_queue_assignments"] >= 24
    assert capabilities["max_pull_bundle_assignments"] >= 24


def test_autopilot_respects_owner_utilization_target_when_scaling(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=1,
        gpu_memory_gb=24.0,
        target_gpu_utilization_pct=50,
        autopilot_state_path=str(tmp_path / "autopilot-owner-target.json"),
    )
    autopilot = AutopilotController(settings)

    for _ in range(5):
        autopilot.observe_idle(
            queue_depth=4,
            active_assignments=autopilot.state.recommendation.max_concurrent_assignments,
            gpu_sample=type(
                "Sample",
                (),
                {
                    "utilization_percent": 18.0,
                    "memory_utilization_percent": 36.0,
                    "power_watts": 70.0,
                    "temperature_c": 49.0,
                },
            )(),
        )

    recommendation = autopilot.state.recommendation
    assert recommendation.max_concurrent_assignments <= 4
    assert autopilot.capabilities_payload()["target_gpu_utilization_pct"] == 50


def test_autopilot_dynamic_embeddings_capacity_is_not_pinned_by_static_default(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=1,
        max_concurrent_assignments_embeddings=2,
        gpu_memory_gb=24.0,
        target_gpu_utilization_pct=100,
        autopilot_state_path=str(tmp_path / "autopilot-dynamic-embeddings.json"),
    )
    autopilot = AutopilotController(settings)
    initial_capabilities = autopilot.capabilities_payload()

    assert initial_capabilities["max_concurrent_assignments"] == 1
    assert initial_capabilities["max_concurrent_assignments_embeddings"] == 1

    for _ in range(5):
        autopilot.observe_idle(
            queue_depth=16,
            active_assignments=autopilot.state.recommendation.max_concurrent_assignments,
            gpu_sample=type(
                "Sample",
                (),
                {
                    "utilization_percent": 20.0,
                    "memory_utilization_percent": 38.0,
                    "power_watts": 85.0,
                    "temperature_c": 51.0,
                },
            )(),
        )

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert recommendation.max_concurrent_assignments > 2
    assert capabilities["max_concurrent_assignments_embeddings"] == recommendation.max_concurrent_assignments


def test_heat_governor_zero_target_pauses_new_work(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=4,
        heat_governor_mode="0",
        autopilot_state_path=str(tmp_path / "autopilot-heat-pause.json"),
    )
    autopilot = AutopilotController(settings)

    autopilot.observe_idle(
        queue_depth=8,
        active_assignments=0,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 0.0,
                "memory_utilization_percent": 20.0,
                "power_watts": 30.0,
                "temperature_c": 42.0,
            },
        )(),
    )

    capabilities = autopilot.capabilities_payload()
    assert autopilot.state.recommendation.max_concurrent_assignments == 0
    assert capabilities["target_gpu_utilization_pct"] == 0
    assert capabilities["heat_governor_paused"] is True
    assert capabilities["max_local_queue_assignments"] == 0
    assert capabilities["max_pull_bundle_assignments"] == 0


def test_heat_governor_scales_embedding_microbatch_at_twenty_percent(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=4,
        gpu_memory_gb=24.0,
        heat_governor_mode="20",
        autopilot_state_path=str(tmp_path / "autopilot-heat-20.json"),
    )
    autopilot = AutopilotController(settings)
    autopilot.observe_idle(queue_depth=12, active_assignments=1)

    capabilities = autopilot.capabilities_payload()
    assert capabilities["target_gpu_utilization_pct"] == 20
    assert capabilities["heat_governor"]["microbatch_scale"] == 0.2
    assert capabilities["max_concurrent_assignments"] <= 2
    assert capabilities["max_microbatch_assignments_embeddings"] <= 5


def test_auto_heat_mode_pauses_when_room_is_above_target(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_RESPONSE_MODEL,
        supported_models=f"{DEFAULT_RESPONSE_MODEL},{DEFAULT_EMBEDDING_MODEL}",
        max_concurrent_assignments=3,
        heat_governor_mode="auto",
        room_temp_c=22.0,
        target_temp_c=21.0,
        outside_temp_c=4.0,
        autopilot_state_path=str(tmp_path / "autopilot-auto-heat.json"),
    )
    autopilot = AutopilotController(settings)
    autopilot.observe_idle(queue_depth=3, active_assignments=0)

    capabilities = autopilot.capabilities_payload()
    assert capabilities["heat_governor_paused"] is True
    assert capabilities["heat_governor"]["pause_reason"] == "owner_heat_target_zero"
    assert "room is already above" in capabilities["heat_governor"]["reason"]


def test_autopilot_remembers_safe_limit_after_gpu_oom(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=1,
        gpu_memory_gb=24.0,
        autopilot_state_path=str(tmp_path / "autopilot-oom-memory.json"),
    )
    autopilot = AutopilotController(settings)

    for _ in range(4):
        autopilot.observe_idle(
            queue_depth=12,
            active_assignments=autopilot.state.recommendation.max_concurrent_assignments,
            gpu_sample=type(
                "Sample",
                (),
                {
                    "utilization_percent": 18.0,
                    "memory_utilization_percent": 42.0,
                    "power_watts": 90.0,
                    "temperature_c": 54.0,
                },
            )(),
        )

    learned_before_oom = autopilot.state.recommendation.max_concurrent_assignments
    assert learned_before_oom >= 3

    autopilot.observe_assignment_failure(
        code="gpu_oom",
        retryable=True,
        queue_depth=12,
        active_assignments=learned_before_oom,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 74.0,
                "memory_utilization_percent": 97.0,
                "power_watts": 210.0,
                "temperature_c": 70.0,
            },
        )(),
    )

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert recommendation.max_concurrent_assignments == learned_before_oom - 1
    assert capabilities["dynamic_concurrency"]["learned_ceiling"] == learned_before_oom - 1
    assert capabilities["dynamic_concurrency"]["last_constraint"] == "gpu_oom"

    for _ in range(4):
        autopilot.observe_idle(
            queue_depth=12,
            active_assignments=recommendation.max_concurrent_assignments,
            gpu_sample=type(
                "Sample",
                (),
                {
                    "utilization_percent": 16.0,
                    "memory_utilization_percent": 44.0,
                    "power_watts": 88.0,
                    "temperature_c": 53.0,
                },
            )(),
        )

    assert autopilot.state.recommendation.max_concurrent_assignments == learned_before_oom - 1


def test_autopilot_backs_off_after_latency_spike(tmp_path: Path) -> None:
    settings = NodeAgentSettings(
        vllm_model=DEFAULT_EMBEDDING_MODEL,
        supported_models=DEFAULT_EMBEDDING_MODEL,
        max_concurrent_assignments=2,
        gpu_memory_gb=24.0,
        autopilot_state_path=str(tmp_path / "autopilot-latency-spike.json"),
    )
    autopilot = AutopilotController(settings)

    autopilot.observe_assignment_success(
        latency_seconds=18.0,
        queue_depth=8,
        active_assignments=2,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 46.0,
                "memory_utilization_percent": 40.0,
                "power_watts": 110.0,
                "temperature_c": 56.0,
            },
        )(),
    )
    autopilot.observe_assignment_success(
        latency_seconds=22.0,
        queue_depth=8,
        active_assignments=2,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 48.0,
                "memory_utilization_percent": 41.0,
                "power_watts": 112.0,
                "temperature_c": 56.0,
            },
            )(),
        )
    concurrency_before_spike = autopilot.state.recommendation.max_concurrent_assignments
    autopilot.observe_assignment_success(
        latency_seconds=120.0,
        queue_depth=8,
        active_assignments=3,
        gpu_sample=type(
            "Sample",
            (),
            {
                "utilization_percent": 96.0,
                "memory_utilization_percent": 43.0,
                "power_watts": 118.0,
                "temperature_c": 57.0,
            },
        )(),
    )

    recommendation = autopilot.state.recommendation
    capabilities = autopilot.capabilities_payload()
    assert concurrency_before_spike > 2
    assert recommendation.max_concurrent_assignments == concurrency_before_spike - 1
    assert capabilities["dynamic_concurrency"]["last_constraint"] == "latency_spike"
    assert "latency spiked" in recommendation.reason
