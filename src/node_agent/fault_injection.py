from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .config import NodeAgentSettings

DEFAULT_FAULT_INJECTION_STATE_NAME = "fault-injection-state.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def state_path_from_settings(settings: NodeAgentSettings) -> Path:
    configured = str(getattr(settings, "fault_injection_state_path", "") or "").strip()
    if configured:
        return Path(configured)
    autopilot_path = str(getattr(settings, "autopilot_state_path", "") or "").strip()
    if autopilot_path:
        return Path(autopilot_path).with_name(DEFAULT_FAULT_INJECTION_STATE_NAME)
    credentials_path = str(getattr(settings, "credentials_path", "") or "").strip()
    if credentials_path:
        return Path(credentials_path).parent / DEFAULT_FAULT_INJECTION_STATE_NAME
    return Path(".") / DEFAULT_FAULT_INJECTION_STATE_NAME


@dataclass
class FaultScenarioState:
    active: bool = False
    remaining_triggers: int = 0
    activated_at: str | None = None
    expires_at: str | None = None
    last_triggered_at: str | None = None
    trigger_count: int = 0
    note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultDrillRecord:
    scenario: str = ""
    status: str = "idle"
    started_at: str | None = None
    completed_at: str | None = None
    summary: str = "No fault drills have run yet."
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultInjectionState:
    scenarios: dict[str, FaultScenarioState] = field(default_factory=dict)
    last_drill: FaultDrillRecord = field(default_factory=FaultDrillRecord)


def _scenario_from_payload(payload: Any) -> FaultScenarioState:
    if not isinstance(payload, dict):
        return FaultScenarioState()
    metadata = payload.get("metadata")
    return FaultScenarioState(
        active=bool(payload.get("active")),
        remaining_triggers=max(0, int(payload.get("remaining_triggers") or 0)),
        activated_at=str(payload.get("activated_at")) if payload.get("activated_at") else None,
        expires_at=str(payload.get("expires_at")) if payload.get("expires_at") else None,
        last_triggered_at=str(payload.get("last_triggered_at")) if payload.get("last_triggered_at") else None,
        trigger_count=max(0, int(payload.get("trigger_count") or 0)),
        note=str(payload.get("note")) if payload.get("note") else None,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _drill_from_payload(payload: Any) -> FaultDrillRecord:
    if not isinstance(payload, dict):
        return FaultDrillRecord()
    details = payload.get("details")
    return FaultDrillRecord(
        scenario=str(payload.get("scenario") or ""),
        status=str(payload.get("status") or "idle"),
        started_at=str(payload.get("started_at")) if payload.get("started_at") else None,
        completed_at=str(payload.get("completed_at")) if payload.get("completed_at") else None,
        summary=str(payload.get("summary") or "No fault drills have run yet."),
        details=dict(details) if isinstance(details, dict) else {},
    )


class FaultInjectionController:
    def __init__(self, state_path: Path | str):
        self.state_path = Path(state_path)
        self._lock = Lock()

    @classmethod
    def from_settings(cls, settings: NodeAgentSettings) -> "FaultInjectionController":
        return cls(state_path_from_settings(settings))

    def _load(self) -> FaultInjectionState:
        if not self.state_path.exists():
            return FaultInjectionState()
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return FaultInjectionState()
        if not isinstance(payload, dict):
            return FaultInjectionState()
        scenarios_payload = payload.get("scenarios")
        scenarios = {
            str(name): _scenario_from_payload(value)
            for name, value in scenarios_payload.items()
            if isinstance(scenarios_payload, dict)
        }
        return FaultInjectionState(
            scenarios=scenarios,
            last_drill=_drill_from_payload(payload.get("last_drill")),
        )

    def _save(self, state: FaultInjectionState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

    def _expire_faults(self, state: FaultInjectionState) -> bool:
        changed = False
        current_time = datetime.now(timezone.utc)
        for scenario in state.scenarios.values():
            if not scenario.active:
                continue
            expires_at = parse_iso_timestamp(scenario.expires_at)
            if expires_at is None or current_time <= expires_at:
                continue
            scenario.active = False
            changed = True
        return changed

    def _scenario_payload(self, name: str, scenario: FaultScenarioState) -> dict[str, Any]:
        payload = asdict(scenario)
        payload["name"] = name
        return payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._load()
            if self._expire_faults(state):
                self._save(state)
            active_faults = [
                self._scenario_payload(name, scenario)
                for name, scenario in sorted(state.scenarios.items())
                if scenario.active
            ]
            return {
                "state_path": str(self.state_path),
                "active_faults": active_faults,
                "scenarios": {
                    name: self._scenario_payload(name, scenario)
                    for name, scenario in sorted(state.scenarios.items())
                },
                "last_drill": asdict(state.last_drill),
            }

    def activate(
        self,
        name: str,
        *,
        remaining_triggers: int = 1,
        expires_in_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._load()
            scenario = state.scenarios.get(name, FaultScenarioState())
            scenario.active = True
            scenario.remaining_triggers = max(0, int(remaining_triggers))
            scenario.activated_at = now_iso()
            scenario.last_triggered_at = None
            scenario.trigger_count = 0
            scenario.note = note
            scenario.metadata = dict(metadata or {})
            scenario.expires_at = (
                (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, float(expires_in_seconds)))).isoformat()
                if expires_in_seconds is not None
                else None
            )
            state.scenarios[name] = scenario
            self._save(state)
            return self._scenario_payload(name, scenario)

    def clear(self, name: str) -> None:
        with self._lock:
            state = self._load()
            if name in state.scenarios:
                state.scenarios.pop(name, None)
                self._save(state)

    def clear_all(self) -> None:
        with self._lock:
            state = self._load()
            if state.scenarios:
                state.scenarios = {}
                self._save(state)

    def peek(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._load()
            changed = self._expire_faults(state)
            scenario = state.scenarios.get(name)
            if changed:
                self._save(state)
            if scenario is None or not scenario.active:
                return None
            return self._scenario_payload(name, scenario)

    def consume(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._load()
            changed = self._expire_faults(state)
            scenario = state.scenarios.get(name)
            if scenario is None or not scenario.active:
                if changed:
                    self._save(state)
                return None
            scenario.last_triggered_at = now_iso()
            scenario.trigger_count += 1
            if scenario.remaining_triggers > 0:
                scenario.remaining_triggers -= 1
                if scenario.remaining_triggers <= 0 and parse_iso_timestamp(scenario.expires_at) is None:
                    scenario.active = False
            self._save(state)
            return self._scenario_payload(name, scenario)

    def record_drill(
        self,
        *,
        scenario: str,
        status: str,
        summary: str,
        details: dict[str, Any] | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._load()
            state.last_drill = FaultDrillRecord(
                scenario=scenario,
                status=status,
                started_at=started_at or now_iso(),
                completed_at=completed_at or now_iso(),
                summary=summary,
                details=dict(details or {}),
            )
            self._save(state)
            return asdict(state.last_drill)
