from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_HEAT_GOVERNOR_MODE = "100"
HEAT_GOVERNOR_MODES = {"0", "20", "50", "100", "auto"}
DEFAULT_OWNER_OBJECTIVE = "balanced"
OWNER_OBJECTIVES = {"balanced", "earnings_only", "heat_first"}
DEFAULT_GPU_TEMP_LIMIT_C = 82.0
DEFAULT_POWER_LIMIT_APPLY_INTERVAL_SECONDS = 30.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def safe_float(value: Any, fallback: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return fallback
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def safe_bool(value: Any, fallback: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return fallback
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def normalize_heat_governor_mode(value: Any) -> str:
    normalized = str(value or DEFAULT_HEAT_GOVERNOR_MODE).strip().lower().replace("_", " ")
    normalized = normalized.removesuffix("%").strip()
    if normalized in {"off", "pause", "paused", "stop", "0"}:
        return "0"
    if normalized in {"low", "20"}:
        return "20"
    if normalized in {"half", "medium", "50"}:
        return "50"
    if normalized in {"full", "high", "max", "100"}:
        return "100"
    if normalized in {"auto", "auto heat", "auto heat mode", "automatic"}:
        return "auto"
    return DEFAULT_HEAT_GOVERNOR_MODE


def normalize_owner_objective(value: Any) -> str:
    normalized = str(value or DEFAULT_OWNER_OBJECTIVE).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"earnings", "earnings_only", "income", "income_only", "profit", "profit_first"}:
        return "earnings_only"
    if normalized in {"heat", "heat_first", "comfort", "heating", "warmth"}:
        return "heat_first"
    return DEFAULT_OWNER_OBJECTIVE


def normalize_local_clock(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if ":" not in text:
        return None
    hours_raw, minutes_raw = text.split(":", 1)
    try:
        hours = int(hours_raw)
        minutes = int(minutes_raw)
    except ValueError:
        return None
    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
        return None
    return f"{hours:02d}:{minutes:02d}"


def local_clock_minutes(value: Any) -> int | None:
    normalized = normalize_local_clock(value)
    if normalized is None:
        return None
    hours, minutes = normalized.split(":", 1)
    return (int(hours) * 60) + int(minutes)


def format_celsius(value: float | None) -> str:
    if value is None:
        return "unknown"
    rounded = round(float(value), 1)
    if abs(rounded - round(rounded)) < 0.05:
        return f"{int(round(rounded))} C"
    return f"{rounded:.1f} C"


def objective_label(value: str) -> str:
    return {
        "balanced": "Balanced",
        "earnings_only": "Earnings only",
        "heat_first": "Heat first",
    }.get(normalize_owner_objective(value), "Balanced")


def quiet_hours_state(
    *,
    start_local: str | None,
    end_local: str | None,
    now_local: datetime | None = None,
) -> dict[str, Any]:
    normalized_start = normalize_local_clock(start_local)
    normalized_end = normalize_local_clock(end_local)
    clock_now = now_local or datetime.now().astimezone()
    state = {
        "enabled": False,
        "active": False,
        "start_local": normalized_start,
        "end_local": normalized_end,
        "local_time": clock_now.strftime("%H:%M"),
        "timezone": clock_now.tzname() or "local",
    }
    if normalized_start is None or normalized_end is None or normalized_start == normalized_end:
        return state
    start_minutes = local_clock_minutes(normalized_start)
    end_minutes = local_clock_minutes(normalized_end)
    if start_minutes is None or end_minutes is None:
        return state
    state["enabled"] = True
    now_minutes = (clock_now.hour * 60) + clock_now.minute
    if start_minutes < end_minutes:
        state["active"] = start_minutes <= now_minutes < end_minutes
    else:
        state["active"] = now_minutes >= start_minutes or now_minutes < end_minutes
    return state


def heat_governor_state_path_from_settings(settings: object) -> Path:
    configured = str(getattr(settings, "heat_governor_state_path", "") or "").strip()
    if configured:
        return Path(configured)
    autopilot_path = str(getattr(settings, "autopilot_state_path", "") or "").strip()
    if autopilot_path:
        return Path(autopilot_path).with_name("heat-governor-state.json")
    return Path(os.getenv("TMPDIR") or os.getenv("TEMP") or ".") / "autonomousc-heat-governor-state.json"


def load_heat_governor_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_heat_governor_state(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    current = load_heat_governor_state(path)
    payload = {
        **current,
        **updates,
        "mode": normalize_heat_governor_mode(updates.get("mode", current.get("mode", DEFAULT_HEAT_GOVERNOR_MODE))),
        "updated_at": now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if os.name != "nt":
        os.chmod(path, 0o600)
    return payload


@dataclass(frozen=True)
class PowerLimitRange:
    minimum_watts: float | None = None
    default_watts: float | None = None
    maximum_watts: float | None = None
    current_watts: float | None = None


@dataclass(frozen=True)
class HeatGovernorPlan:
    mode: str
    requested_target_pct: int
    effective_target_pct: int
    paused: bool
    pause_reason: str | None
    heat_demand: str
    setup_profile: str
    thermal_headroom: float
    concurrency_scale: float
    microbatch_scale: float
    desired_power_limit_watts: int | None
    reason: str
    owner_objective: str = DEFAULT_OWNER_OBJECTIVE
    owner_objective_label: str = "Balanced"
    quiet_hours_start_local: str | None = None
    quiet_hours_end_local: str | None = None
    quiet_hours_active: bool = False
    quiet_hours_local_time: str | None = None
    quiet_hours_timezone: str | None = None
    max_power_cap_watts: int | None = None
    decision_reasons: list[str] = field(default_factory=list)
    owner_alerts: list[dict[str, Any]] = field(default_factory=list)
    room_temp_c: float | None = None
    target_temp_c: float | None = None
    outside_temp_c: float | None = None
    gpu_temp_c: float | None = None
    gpu_temp_limit_c: float = DEFAULT_GPU_TEMP_LIMIT_C

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision_reasons"] = list(self.decision_reasons or [])
        return payload


def automatic_heat_target_pct(
    *,
    room_temp_c: float | None,
    target_temp_c: float | None,
    outside_temp_c: float | None,
) -> tuple[int, str]:
    if room_temp_c is not None and target_temp_c is not None:
        gap_c = target_temp_c - room_temp_c
        if gap_c <= -0.5:
            return 0, "Auto heat paused because the room is already above the owner target."
        if gap_c <= 0.2:
            target = 20
            reason = "Auto heat is holding a small trickle because the room is near the owner target."
        elif gap_c <= 1.0:
            target = 50
            reason = "Auto heat is warming moderately because the room is below the owner target."
        else:
            target = 100
            reason = "Auto heat is using full compute because the room is well below the owner target."

        if outside_temp_c is not None and outside_temp_c <= 0 and target in {20, 50} and gap_c > 0.1:
            return min(100, target * 2), (
                "Auto heat raised output because the outside-temperature hint says heat loss is high."
            )
        if outside_temp_c is not None and outside_temp_c >= 16 and target == 100 and gap_c < 1.5:
            return 50, "Auto heat eased back because the outside-temperature hint is mild."
        return target, reason

    if outside_temp_c is not None:
        if outside_temp_c <= 2:
            return 100, "Auto heat is using full compute from a cold outside-temperature hint."
        if outside_temp_c <= 10:
            return 50, "Auto heat is using moderate compute from the outside-temperature hint."
        if outside_temp_c >= 18:
            return 20, "Auto heat is using a low trickle from a mild outside-temperature hint."
    return 50, "Auto heat is using a moderate default until room temperature data is available."


def heat_demand_label(target_pct: int, *, paused: bool) -> str:
    if paused or target_pct <= 0:
        return "none"
    if target_pct <= 20:
        return "low"
    if target_pct <= 50:
        return "medium"
    return "high"


def setup_profile_for_target(target_pct: int) -> str:
    if target_pct <= 20:
        return "quiet"
    if target_pct >= 100:
        return "performance"
    return "balanced"


def thermal_headroom_for_target(target_pct: int, *, configured_headroom: float) -> float:
    if target_pct <= 0:
        return 0.45
    if target_pct <= 20:
        return min(configured_headroom, 0.58)
    if target_pct <= 50:
        return min(configured_headroom, 0.72)
    return clamp(max(configured_headroom, 0.92), 0.2, 1.0)


def power_limit_for_target(target_pct: int, power_limits: PowerLimitRange | None) -> int | None:
    if power_limits is None:
        return None
    upper = power_limits.default_watts or power_limits.maximum_watts
    if upper is None or upper <= 0:
        return None
    lower = power_limits.minimum_watts
    if lower is None or lower <= 0:
        lower = max(1.0, upper * 0.35)
    if target_pct >= 100:
        desired = upper
    elif target_pct <= 0:
        desired = lower
    else:
        desired = lower + ((upper - lower) * (target_pct / 100.0))
    if power_limits.maximum_watts is not None:
        desired = min(power_limits.maximum_watts, desired)
    if power_limits.minimum_watts is not None:
        desired = max(power_limits.minimum_watts, desired)
    return max(1, int(round(desired)))


def owner_alert(*, code: str, title: str, detail: str, tone: str = "warning") -> dict[str, Any]:
    return {
        "code": str(code).strip() or "owner_alert",
        "title": str(title).strip() or "Owner alert",
        "detail": str(detail).strip() or "The heat governor has an update for this node.",
        "tone": str(tone or "warning").strip() or "warning",
        "source": "heat_governor",
    }


def build_heat_governor_plan(
    settings: object,
    *,
    state: dict[str, Any] | None = None,
    gpu_temp_c: float | None = None,
    gpu_utilization_pct: float | None = None,
    queue_depth: int | None = None,
    active_assignments: int | None = None,
    power_limits: PowerLimitRange | None = None,
    now_local: datetime | None = None,
) -> HeatGovernorPlan:
    owner_state = state or {}
    mode = normalize_heat_governor_mode(
        owner_state.get("mode", getattr(settings, "heat_governor_mode", DEFAULT_HEAT_GOVERNOR_MODE))
    )
    owner_objective = normalize_owner_objective(
        owner_state.get("owner_objective", getattr(settings, "owner_objective", DEFAULT_OWNER_OBJECTIVE))
    )
    room_temp_c = safe_float(owner_state.get("room_temp_c"), getattr(settings, "room_temp_c", None))
    target_temp_c = safe_float(owner_state.get("target_temp_c"), getattr(settings, "target_temp_c", None))
    outside_temp_c = safe_float(owner_state.get("outside_temp_c"), getattr(settings, "outside_temp_c", None))
    quiet_hours_start_local = normalize_local_clock(
        owner_state.get("quiet_hours_start_local", getattr(settings, "quiet_hours_start_local", None))
    )
    quiet_hours_end_local = normalize_local_clock(
        owner_state.get("quiet_hours_end_local", getattr(settings, "quiet_hours_end_local", None))
    )
    quiet_hours = quiet_hours_state(
        start_local=quiet_hours_start_local,
        end_local=quiet_hours_end_local,
        now_local=now_local,
    )
    max_power_cap_watts = safe_float(
        owner_state.get("max_power_cap_watts", getattr(settings, "max_power_cap_watts", None))
    )
    max_power_cap_watts = None if max_power_cap_watts is None or max_power_cap_watts <= 0 else int(round(max_power_cap_watts))
    gpu_temp_limit_c = clamp(
        safe_float(owner_state.get("gpu_temp_limit_c"), getattr(settings, "gpu_temp_limit_c", DEFAULT_GPU_TEMP_LIMIT_C))
        or DEFAULT_GPU_TEMP_LIMIT_C,
        65.0,
        95.0,
    )
    configured_headroom = clamp(float(getattr(settings, "thermal_headroom", 0.8) or 0.8), 0.2, 1.0)
    observed_gpu_temp_c = (
        gpu_temp_c
        if gpu_temp_c is not None
        else safe_float(owner_state.get("gpu_temp_c"), getattr(settings, "gpu_temp_c", None))
    )
    observed_queue_depth = max(0, int(queue_depth or 0))
    observed_active_assignments = max(0, int(active_assignments or 0))
    demand_active = observed_queue_depth > 0 or observed_active_assignments > 0
    decision_reasons: list[str] = []
    owner_alerts: list[dict[str, Any]] = []

    if mode == "auto":
        requested_target_pct, reason = automatic_heat_target_pct(
            room_temp_c=room_temp_c,
            target_temp_c=target_temp_c,
            outside_temp_c=outside_temp_c,
        )
    else:
        requested_target_pct = int(mode)
        reason = f"Owner heat target is set to {requested_target_pct}%."
    decision_reasons.append(reason)

    if owner_objective == "earnings_only" and mode == "auto":
        if demand_active and requested_target_pct < 50:
            requested_target_pct = 50
            reason = "Earnings-only mode raised output because matching GPU demand returned."
            decision_reasons.append(reason)
        elif requested_target_pct <= 0:
            requested_target_pct = 20
            reason = "Earnings-only mode kept a small trickle so the node stays ready for paid work."
            decision_reasons.append(reason)
    elif owner_objective == "heat_first" and mode == "auto":
        boosted_target_pct = requested_target_pct
        if room_temp_c is not None and target_temp_c is not None:
            gap_c = target_temp_c - room_temp_c
            if gap_c > 0.4 and requested_target_pct < 100:
                boosted_target_pct = 50 if requested_target_pct <= 20 else 100
                reason = (
                    f"Heat-first mode raised output because the room is {round(gap_c, 1)} C below target."
                )
        elif outside_temp_c is not None and outside_temp_c <= 5 and requested_target_pct < 100:
            boosted_target_pct = 50 if requested_target_pct <= 20 else 100
            reason = (
                f"Heat-first mode raised output because the outside-temperature hint is {format_celsius(outside_temp_c)}."
            )
        if boosted_target_pct != requested_target_pct:
            requested_target_pct = boosted_target_pct
            decision_reasons.append(reason)

    advanced_utilization_cap = safe_float(getattr(settings, "target_gpu_utilization_pct", 100), 100.0) or 100.0
    advanced_utilization_cap = int(clamp(advanced_utilization_cap, 0.0, 100.0))
    if 0 < advanced_utilization_cap < requested_target_pct:
        requested_target_pct = advanced_utilization_cap
        reason = f"{reason} Advanced GPU utilization cap is limiting output to {advanced_utilization_cap}%."
        decision_reasons.append(f"Advanced GPU utilization cap is limiting output to {advanced_utilization_cap}%.")

    if quiet_hours.get("active") and requested_target_pct > 20:
        requested_target_pct = 20
        reason = (
            f"Quiet hours are active from {quiet_hours_start_local} to {quiet_hours_end_local} "
            f"{quiet_hours.get('timezone')}, so output is capped at 20%."
        )
        decision_reasons.append(reason)

    pause_reason: str | None = None
    effective_target_pct = requested_target_pct
    if requested_target_pct <= 0:
        pause_reason = "owner_heat_target_zero"
        reason = f"{reason} The node is pausing new assignments and draining safely."
        decision_reasons.append("The node is pausing new assignments and draining safely.")

    if observed_gpu_temp_c is not None:
        if observed_gpu_temp_c >= gpu_temp_limit_c + 3.0:
            effective_target_pct = 0
            pause_reason = "gpu_temperature_limit"
            reason = "GPU temperature is above the safe limit, so the governor paused new assignments."
            decision_reasons.append(reason)
        elif observed_gpu_temp_c >= gpu_temp_limit_c:
            effective_target_pct = min(effective_target_pct, 20)
            reason = "GPU temperature reached the limit, so the governor reduced heat output."
            decision_reasons.append(reason)
        elif observed_gpu_temp_c >= gpu_temp_limit_c - 4.0:
            effective_target_pct = min(effective_target_pct, 50)
            reason = "GPU temperature is near the limit, so the governor eased back."
            decision_reasons.append(reason)

    effective_target_pct = int(clamp(float(effective_target_pct), 0.0, 100.0))
    paused = effective_target_pct <= 0
    if paused and pause_reason is None:
        pause_reason = "heat_governor_paused"

    concurrency_scale = effective_target_pct / 100.0
    microbatch_scale = 0.0 if paused else max(0.20, concurrency_scale)
    desired_power_limit_watts = power_limit_for_target(effective_target_pct, power_limits)
    power_cap_limited = False
    if max_power_cap_watts is not None:
        if desired_power_limit_watts is None or desired_power_limit_watts > max_power_cap_watts:
            desired_power_limit_watts = max_power_cap_watts
            reason = f"Owner power cap is limiting GPU power to {max_power_cap_watts} W."
            decision_reasons.append(reason)
            power_cap_limited = True
    if mode == "auto" and paused and pause_reason == "owner_heat_target_zero" and room_temp_c is not None and target_temp_c is not None:
        owner_alerts.append(
            owner_alert(
                code="room_target_reached",
                title="Paused because room target was reached",
                detail=(
                    f"Room temperature is {format_celsius(room_temp_c)} and the owner target is {format_celsius(target_temp_c)}, "
                    "so the node paused new work until more heat is useful."
                ),
                tone="warning",
            )
        )
    if pause_reason == "gpu_temperature_limit":
        owner_alerts.append(
            owner_alert(
                code="gpu_temperature_limit",
                title="Thermal protection paused new work",
                detail=reason,
                tone="danger",
            )
        )
    elif power_cap_limited and max_power_cap_watts is not None:
        owner_alerts.append(
            owner_alert(
                code="power_cap_limited",
                title=f"Power cap is limiting GPU to {max_power_cap_watts} W",
                detail=(
                    f"The owner power cap is holding the GPU at {max_power_cap_watts} W while the node keeps serving "
                    "within that limit."
                ),
                tone="warning",
            )
        )
    profile = setup_profile_for_target(effective_target_pct)
    thermal_headroom = thermal_headroom_for_target(effective_target_pct, configured_headroom=configured_headroom)

    return HeatGovernorPlan(
        mode=mode,
        requested_target_pct=requested_target_pct,
        effective_target_pct=effective_target_pct,
        paused=paused,
        pause_reason=pause_reason,
        heat_demand=heat_demand_label(effective_target_pct, paused=paused),
        setup_profile=profile,
        thermal_headroom=thermal_headroom,
        concurrency_scale=concurrency_scale,
        microbatch_scale=microbatch_scale,
        desired_power_limit_watts=desired_power_limit_watts,
        reason=reason,
        owner_objective=owner_objective,
        owner_objective_label=objective_label(owner_objective),
        quiet_hours_start_local=quiet_hours_start_local,
        quiet_hours_end_local=quiet_hours_end_local,
        quiet_hours_active=bool(quiet_hours.get("active")),
        quiet_hours_local_time=str(quiet_hours.get("local_time") or ""),
        quiet_hours_timezone=str(quiet_hours.get("timezone") or ""),
        max_power_cap_watts=max_power_cap_watts,
        decision_reasons=decision_reasons,
        owner_alerts=owner_alerts,
        room_temp_c=room_temp_c,
        target_temp_c=target_temp_c,
        outside_temp_c=outside_temp_c,
        gpu_temp_c=observed_gpu_temp_c,
        gpu_temp_limit_c=gpu_temp_limit_c,
    )


class NvidiaPowerLimitController:
    def __init__(
        self,
        *,
        enabled: bool = True,
        command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
        apply_interval_seconds: float = DEFAULT_POWER_LIMIT_APPLY_INTERVAL_SECONDS,
    ) -> None:
        self.enabled = enabled
        self.command_runner = command_runner or self._run
        self.apply_interval_seconds = max(1.0, float(apply_interval_seconds))
        self._last_limit_watts: int | None = None
        self._last_apply_at: float | None = None
        self._last_range: PowerLimitRange | None = None
        self._range_checked_at: float | None = None
        self.last_error: str | None = None

    @staticmethod
    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=3.0)

    @staticmethod
    def _parse_csv_floats(output: str) -> list[float | None]:
        first_line = output.splitlines()[0].strip() if output.splitlines() else ""
        values: list[float | None] = []
        for part in first_line.split(","):
            values.append(safe_float(part.strip()))
        return values

    def power_limits(self, *, now_monotonic: float | None = None) -> PowerLimitRange | None:
        if not self.enabled or shutil.which("nvidia-smi") is None:
            return None
        current_time = time.monotonic() if now_monotonic is None else now_monotonic
        if self._last_range is not None and self._range_checked_at is not None and current_time - self._range_checked_at < 300:
            return self._last_range
        try:
            completed = self.command_runner(
                [
                    "nvidia-smi",
                    "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit",
                    "--format=csv,noheader,nounits",
                ]
            )
        except Exception as error:
            self.last_error = str(error)
            return None
        values = self._parse_csv_floats(completed.stdout)
        while len(values) < 4:
            values.append(None)
        current, default, minimum, maximum = values[:4]
        power_range = PowerLimitRange(
            minimum_watts=minimum,
            default_watts=default,
            maximum_watts=maximum,
            current_watts=current,
        )
        self._last_range = power_range
        self._range_checked_at = current_time
        self.last_error = None
        return power_range

    def apply(self, target_watts: int | None, *, now_monotonic: float | None = None) -> bool:
        if not self.enabled or target_watts is None or shutil.which("nvidia-smi") is None:
            return False
        current_time = time.monotonic() if now_monotonic is None else now_monotonic
        if (
            self._last_limit_watts == target_watts
            and self._last_apply_at is not None
            and current_time - self._last_apply_at < self.apply_interval_seconds
        ):
            return False
        try:
            self.command_runner(["nvidia-smi", "-pl", str(int(target_watts))])
        except Exception as error:
            self.last_error = str(error)
            return False
        self._last_limit_watts = int(target_watts)
        self._last_apply_at = current_time
        self.last_error = None
        return True
