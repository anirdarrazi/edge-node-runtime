from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from node_agent.config import NodeAgentSettings
from node_agent.heat_governor import (
    NvidiaPowerLimitController,
    PowerLimitRange,
    build_heat_governor_plan,
)


def build_settings(tmp_path: Path, **overrides) -> NodeAgentSettings:
    base = {
        "vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
        "supported_models": "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
        "autopilot_state_path": str(tmp_path / "autopilot-state.json"),
    }
    base.update(overrides)
    return NodeAgentSettings(**base)


@pytest.mark.parametrize(
    ("mode", "effective_target_pct", "paused", "setup_profile", "heat_demand", "microbatch_scale", "power_limit_watts"),
    [
        ("0", 0, True, "quiet", "none", 0.0, 100),
        ("20", 20, False, "quiet", "low", 0.2, 140),
        ("50", 50, False, "balanced", "medium", 0.5, 200),
        ("100", 100, False, "performance", "high", 1.0, 300),
    ],
)
def test_fixed_heat_profiles_map_to_expected_throttling(
    tmp_path: Path,
    mode: str,
    effective_target_pct: int,
    paused: bool,
    setup_profile: str,
    heat_demand: str,
    microbatch_scale: float,
    power_limit_watts: int,
) -> None:
    settings = build_settings(tmp_path, heat_governor_mode=mode)

    plan = build_heat_governor_plan(
        settings,
        power_limits=PowerLimitRange(minimum_watts=100, default_watts=300, maximum_watts=320),
        now_local=datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert plan.effective_target_pct == effective_target_pct
    assert plan.paused is paused
    assert plan.setup_profile == setup_profile
    assert plan.heat_demand == heat_demand
    assert plan.microbatch_scale == pytest.approx(microbatch_scale)
    assert plan.desired_power_limit_watts == power_limit_watts


def test_auto_heat_earnings_only_keeps_trickle_when_room_is_already_warm(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        heat_governor_mode="auto",
        owner_objective="earnings_only",
        room_temp_c=22.0,
        target_temp_c=21.0,
        outside_temp_c=18.0,
    )

    plan = build_heat_governor_plan(
        settings,
        queue_depth=0,
        active_assignments=0,
        now_local=datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert plan.effective_target_pct == 20
    assert plan.paused is False
    assert any("small trickle" in reason.lower() for reason in plan.decision_reasons)


def test_auto_heat_earnings_only_raises_output_when_gpu_demand_returns(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        heat_governor_mode="auto",
        owner_objective="earnings_only",
        room_temp_c=21.1,
        target_temp_c=21.0,
        outside_temp_c=15.0,
    )

    plan = build_heat_governor_plan(
        settings,
        queue_depth=6,
        active_assignments=2,
        now_local=datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert plan.effective_target_pct == 50
    assert any("gpu demand returned" in reason.lower() for reason in plan.decision_reasons)


def test_auto_heat_heat_first_boosts_when_room_is_below_target(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        heat_governor_mode="auto",
        owner_objective="heat_first",
        room_temp_c=20.4,
        target_temp_c=21.0,
        outside_temp_c=12.0,
    )

    plan = build_heat_governor_plan(
        settings,
        now_local=datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert plan.requested_target_pct == 100
    assert plan.effective_target_pct == 100
    assert any("heat-first mode raised output" in reason.lower() for reason in plan.decision_reasons)


def test_quiet_hours_cap_output_deterministically(tmp_path: Path) -> None:
    stockholm = timezone(timedelta(hours=2))
    settings = build_settings(
        tmp_path,
        heat_governor_mode="100",
        quiet_hours_start_local="22:30",
        quiet_hours_end_local="06:00",
    )

    plan = build_heat_governor_plan(
        settings,
        now_local=datetime(2026, 4, 12, 23, 15, 0, tzinfo=stockholm),
    )

    assert plan.quiet_hours_active is True
    assert plan.effective_target_pct == 20
    assert "quiet hours are active" in plan.reason.lower()


@pytest.mark.parametrize(
    ("gpu_temp_c", "effective_target_pct", "pause_reason"),
    [
        (78.0, 50, None),
        (82.0, 20, None),
        (85.0, 0, "gpu_temperature_limit"),
    ],
)
def test_gpu_temperature_limits_override_owner_target(
    tmp_path: Path,
    gpu_temp_c: float,
    effective_target_pct: int,
    pause_reason: str | None,
) -> None:
    settings = build_settings(
        tmp_path,
        heat_governor_mode="100",
        gpu_temp_limit_c=82.0,
    )

    plan = build_heat_governor_plan(
        settings,
        gpu_temp_c=gpu_temp_c,
        now_local=datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert plan.effective_target_pct == effective_target_pct
    assert plan.pause_reason == pause_reason


def test_power_cap_limits_gpu_power_and_emits_owner_alert(tmp_path: Path) -> None:
    settings = build_settings(
        tmp_path,
        heat_governor_mode="100",
        max_power_cap_watts=220,
    )

    plan = build_heat_governor_plan(
        settings,
        power_limits=PowerLimitRange(minimum_watts=100, default_watts=300, maximum_watts=320),
        now_local=datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert plan.desired_power_limit_watts == 220
    assert any(alert.get("code") == "power_cap_limited" for alert in plan.owner_alerts)
    assert any("220 W" in alert.get("title", "") for alert in plan.owner_alerts)


def test_nvidia_power_limit_controller_retries_apply_interval_and_parses_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit" in args:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="250, 300, 120, 320\n", stderr="")
        if args[:2] == ["nvidia-smi", "-pl"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr("node_agent.heat_governor.shutil.which", lambda _name: "nvidia-smi")

    controller = NvidiaPowerLimitController(command_runner=runner, apply_interval_seconds=30.0)
    power_range = controller.power_limits(now_monotonic=10.0)

    assert power_range == PowerLimitRange(
        minimum_watts=120.0,
        default_watts=300.0,
        maximum_watts=320.0,
        current_watts=250.0,
    )

    applied_first = controller.apply(220, now_monotonic=20.0)
    applied_second = controller.apply(220, now_monotonic=35.0)
    applied_third = controller.apply(220, now_monotonic=55.0)

    assert applied_first is True
    assert applied_second is False
    assert applied_third is True
    assert commands.count(["nvidia-smi", "-pl", "220"]) == 2
