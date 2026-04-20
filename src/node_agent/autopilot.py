from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import NodeAgentSettings
from .concurrency import (
    resolved_embeddings_concurrency_limit,
    resolved_embeddings_microbatch_assignment_limit,
)
from .runtime_profiles import (
    HOME_EMBEDDINGS_LLAMA_CPP_PROFILE,
    HOME_LLAMA_CPP_GGUF_PROFILE,
    LLAMA_CPP_INFERENCE_ENGINE,
    llama_cpp_model_source,
)
from .runtime_tuple import resolved_runtime_tuple

DEFAULT_RESPONSE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
AUTOPILOT_STATE_VERSION = 1
HEAT_DEMAND_LEVELS = {"none", "low", "medium", "high"}
MIN_TARGET_GPU_UTILIZATION_PCT = 30
MAX_TARGET_GPU_UTILIZATION_PCT = 100
MIN_GPU_MEMORY_HEADROOM_PCT = 5.0
MAX_GPU_MEMORY_HEADROOM_PCT = 60.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, fallback: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def safe_int(value: Any, fallback: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return fallback


def settings_value(settings: object, name: str, fallback: Any) -> Any:
    value = getattr(settings, name, fallback)
    return fallback if value is None else value


def settings_str(settings: object, name: str, fallback: str) -> str:
    return str(settings_value(settings, name, fallback))


def settings_int(settings: object, name: str, fallback: int) -> int:
    return safe_int(settings_value(settings, name, fallback), fallback)


def settings_float(settings: object, name: str, fallback: float) -> float:
    return safe_float(settings_value(settings, name, fallback), fallback)


def settings_optional_float(settings: object, name: str) -> float | None:
    value = getattr(settings, name, None)
    if value is None or str(value).strip() == "":
        return None
    return safe_float(value, 0.0)


def normalize_heat_demand(value: Any) -> str:
    normalized = str(value or "none").strip().lower()
    return normalized if normalized in HEAT_DEMAND_LEVELS else "none"


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def configured_models(settings: object) -> list[str]:
    fallback_model = settings_str(settings, "vllm_model", DEFAULT_RESPONSE_MODEL)
    models = [model.strip() for model in settings_str(settings, "supported_models", fallback_model).split(",") if model.strip()]
    return models or [fallback_model]


def live_runtime_model(settings: object) -> str:
    configured_model = settings_str(settings, "vllm_model", DEFAULT_RESPONSE_MODEL)
    current_model = settings_str(settings, "current_model", configured_model)
    return current_model or configured_model or DEFAULT_RESPONSE_MODEL


def operations_for_model(model: str, settings: object) -> list[str]:
    normalized_model = str(model or "").strip()
    if not normalized_model:
        return ["responses"]

    llama_cpp_source = llama_cpp_model_source(normalized_model)
    if llama_cpp_source is not None:
        return ["embeddings"] if llama_cpp_source.embedding_enabled else ["responses"]

    if normalized_model == DEFAULT_EMBEDDING_MODEL:
        return ["embeddings"]

    runtime_profile = getattr(settings, "resolved_runtime_profile", None)
    supported_apis = getattr(runtime_profile, "supported_apis", None)
    if isinstance(supported_apis, (list, tuple)):
        filtered_supported_apis = [str(api) for api in supported_apis if str(api) in {"responses", "embeddings"}]
        if len(filtered_supported_apis) == 1:
            return filtered_supported_apis

    return ["responses"]


@dataclass
class AutopilotSignals:
    completed_count: int = 0
    failed_count: int = 0
    ewma_latency_seconds: float | None = None
    last_latency_seconds: float | None = None
    failure_rate: float = 0.0
    queue_depth: int = 0
    active_assignments: int = 0
    gpu_memory_pressure: float | None = None
    gpu_utilization_pct: float | None = None
    gpu_memory_utilization_pct: float | None = None
    gpu_temp_c: float | None = None
    power_watts: float | None = None
    estimated_heat_output_watts: float | None = None
    last_failure_code: str | None = None
    last_failure_retryable: bool | None = None
    last_observed_at: str | None = None


@dataclass
class AutopilotRecommendation:
    setup_profile: str = "balanced"
    max_concurrent_assignments: int = 1
    thermal_headroom: float = 0.8
    startup_model: str = DEFAULT_RESPONSE_MODEL
    supported_models: str = DEFAULT_RESPONSE_MODEL
    operations: list[str] = field(default_factory=lambda: ["responses", "embeddings"])
    reason: str = "Autopilot is collecting enough runtime data before changing the plan."
    pending_restart: bool = False
    safe_to_apply: bool = True
    env_updates: dict[str, str] = field(default_factory=dict)


@dataclass
class AutopilotState:
    version: int = AUTOPILOT_STATE_VERSION
    enabled: bool = True
    baseline_concurrency: int = 1
    baseline_model: str = DEFAULT_RESPONSE_MODEL
    current_model: str = DEFAULT_RESPONSE_MODEL
    updated_at: str | None = None
    signals: AutopilotSignals = field(default_factory=AutopilotSignals)
    recommendation: AutopilotRecommendation = field(default_factory=AutopilotRecommendation)
    history: list[dict[str, Any]] = field(default_factory=list)


def state_from_payload(payload: dict[str, Any], settings: NodeAgentSettings) -> AutopilotState:
    configured_concurrency = settings_int(settings, "max_concurrent_assignments", 1)
    configured_model = settings_str(settings, "vllm_model", DEFAULT_RESPONSE_MODEL)
    configured_models_csv = settings_str(settings, "supported_models", configured_model)
    baseline_concurrency = max(1, safe_int(payload.get("baseline_concurrency"), configured_concurrency))
    baseline_model = str(payload.get("baseline_model") or configured_model or DEFAULT_RESPONSE_MODEL)
    recommendation_payload = payload.get("recommendation") if isinstance(payload.get("recommendation"), dict) else {}
    signals_payload = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
    return AutopilotState(
        version=safe_int(payload.get("version"), AUTOPILOT_STATE_VERSION),
        enabled=bool(payload.get("enabled", True)),
        baseline_concurrency=baseline_concurrency,
        baseline_model=baseline_model,
        current_model=str(payload.get("current_model") or configured_model or baseline_model),
        updated_at=payload.get("updated_at") if isinstance(payload.get("updated_at"), str) else None,
        signals=AutopilotSignals(
            completed_count=max(0, safe_int(signals_payload.get("completed_count"), 0)),
            failed_count=max(0, safe_int(signals_payload.get("failed_count"), 0)),
            ewma_latency_seconds=(
                safe_float(signals_payload.get("ewma_latency_seconds"), 0.0)
                if signals_payload.get("ewma_latency_seconds") is not None
                else None
            ),
            last_latency_seconds=(
                safe_float(signals_payload.get("last_latency_seconds"), 0.0)
                if signals_payload.get("last_latency_seconds") is not None
                else None
            ),
            failure_rate=clamp(safe_float(signals_payload.get("failure_rate"), 0.0), 0.0, 1.0),
            queue_depth=max(0, safe_int(signals_payload.get("queue_depth"), 0)),
            active_assignments=max(0, safe_int(signals_payload.get("active_assignments"), 0)),
            gpu_memory_pressure=(
                clamp(safe_float(signals_payload.get("gpu_memory_pressure"), 0.0), 0.0, 1.0)
                if signals_payload.get("gpu_memory_pressure") is not None
                else None
            ),
            gpu_utilization_pct=(
                clamp(safe_float(signals_payload.get("gpu_utilization_pct"), 0.0), 0.0, 100.0)
                if signals_payload.get("gpu_utilization_pct") is not None
                else None
            ),
            gpu_memory_utilization_pct=(
                clamp(safe_float(signals_payload.get("gpu_memory_utilization_pct"), 0.0), 0.0, 100.0)
                if signals_payload.get("gpu_memory_utilization_pct") is not None
                else None
            ),
            gpu_temp_c=(
                safe_float(signals_payload.get("gpu_temp_c"), 0.0)
                if signals_payload.get("gpu_temp_c") is not None
                else None
            ),
            power_watts=(
                safe_float(signals_payload.get("power_watts"), 0.0)
                if signals_payload.get("power_watts") is not None
                else None
            ),
            estimated_heat_output_watts=(
                safe_float(signals_payload.get("estimated_heat_output_watts"), 0.0)
                if signals_payload.get("estimated_heat_output_watts") is not None
                else None
            ),
            last_failure_code=(
                str(signals_payload.get("last_failure_code")) if signals_payload.get("last_failure_code") else None
            ),
            last_failure_retryable=(
                bool(signals_payload.get("last_failure_retryable"))
                if signals_payload.get("last_failure_retryable") is not None
                else None
            ),
            last_observed_at=(
                str(signals_payload.get("last_observed_at")) if signals_payload.get("last_observed_at") else None
            ),
        ),
        recommendation=AutopilotRecommendation(
            setup_profile=str(recommendation_payload.get("setup_profile") or "balanced"),
            max_concurrent_assignments=max(
                1,
                safe_int(recommendation_payload.get("max_concurrent_assignments"), max(1, configured_concurrency)),
            ),
            thermal_headroom=clamp(safe_float(recommendation_payload.get("thermal_headroom"), 0.8), 0.2, 1.0),
            startup_model=str(recommendation_payload.get("startup_model") or configured_model or baseline_model),
            supported_models=str(recommendation_payload.get("supported_models") or configured_models_csv),
            operations=[
                str(operation)
                for operation in recommendation_payload.get("operations", ["responses", "embeddings"])
                if str(operation) in {"responses", "embeddings"}
            ]
            or ["responses", "embeddings"],
            reason=str(recommendation_payload.get("reason") or "Autopilot is monitoring this machine."),
            pending_restart=bool(recommendation_payload.get("pending_restart", False)),
            safe_to_apply=bool(recommendation_payload.get("safe_to_apply", True)),
            env_updates={
                str(key): str(value)
                for key, value in (recommendation_payload.get("env_updates") or {}).items()
                if isinstance(key, str)
            },
        ),
        history=[
            item for item in payload.get("history", []) if isinstance(item, dict)
        ][-20:],
    )


class AutopilotController:
    def __init__(self, settings: NodeAgentSettings, *, state_path: Path | None = None) -> None:
        self.settings = settings
        self.state_path = Path(
            state_path
            or getattr(settings, "autopilot_state_path", "")
            or (Path(tempfile.gettempdir()) / "autonomousc-autopilot-state.json")
        )
        self.state = self.load_state()
        self.evaluate()

    def load_state(self) -> AutopilotState:
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return state_from_payload(payload, self.settings)
            except (OSError, json.JSONDecodeError):
                pass
        configured_concurrency = max(1, settings_int(self.settings, "max_concurrent_assignments", 1))
        configured_model = settings_str(self.settings, "vllm_model", DEFAULT_RESPONSE_MODEL)
        configured_models_csv = settings_str(self.settings, "supported_models", configured_model)
        configured_headroom = clamp(settings_float(self.settings, "thermal_headroom", 0.8), 0.2, 1.0)
        return AutopilotState(
            baseline_concurrency=configured_concurrency,
            baseline_model=configured_model,
            current_model=configured_model,
            recommendation=AutopilotRecommendation(
                setup_profile=self.profile_from_headroom(configured_headroom),
                max_concurrent_assignments=configured_concurrency,
                thermal_headroom=configured_headroom,
                startup_model=configured_model,
                supported_models=configured_models_csv,
                operations=operations_for_model(configured_model, self.settings),
                env_updates={},
            ),
        )

    @staticmethod
    def profile_from_headroom(thermal_headroom: float) -> str:
        if thermal_headroom <= 0.62:
            return "quiet"
        if thermal_headroom >= 0.9:
            return "performance"
        return "balanced"

    def save(self) -> None:
        self.state.updated_at = now_iso()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
        if os.name != "nt":
            os.chmod(self.state_path, 0o600)

    def sample_gpu_memory_pressure(self) -> float | None:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output_lines = completed.stdout.splitlines()
        first_line = output_lines[0].strip() if output_lines else ""
        if "," not in first_line:
            return None
        used_raw, total_raw = [part.strip() for part in first_line.split(",", 1)]
        used = safe_float(used_raw, 0.0)
        total = safe_float(total_raw, 0.0)
        if total <= 0:
            return None
        return clamp(used / total, 0.0, 1.0)

    def sample_gpu_utilization_pct(self) -> float | None:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        first_line = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
        if not first_line:
            return None
        return clamp(safe_float(first_line, 0.0), 0.0, 100.0)

    def sample_gpu_thermal_metrics(self) -> dict[str, float] | None:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        first_line = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
        if "," not in first_line:
            return None
        temp_raw, power_raw = [part.strip() for part in first_line.split(",", 1)]
        gpu_temp_c = safe_float(temp_raw, 0.0)
        power_watts = max(0.0, safe_float(power_raw, 0.0))
        return {
            "gpu_temp_c": gpu_temp_c,
            "power_watts": power_watts,
            "estimated_heat_output_watts": power_watts,
        }

    def refresh_gpu_signals(self, *, gpu_sample: Any | None = None) -> None:
        if gpu_sample is not None:
            sampled_memory_utilization_raw = getattr(gpu_sample, "memory_utilization_percent", None)
            sampled_utilization_raw = getattr(gpu_sample, "utilization_percent", None)
            if sampled_memory_utilization_raw is not None:
                sampled_memory_utilization_pct = safe_float(sampled_memory_utilization_raw, 0.0)
                memory_utilization_pct = clamp(sampled_memory_utilization_pct, 0.0, 100.0)
                self.state.signals.gpu_memory_utilization_pct = memory_utilization_pct
                self.state.signals.gpu_memory_pressure = clamp(memory_utilization_pct / 100.0, 0.0, 1.0)
            if sampled_utilization_raw is not None:
                sampled_utilization_pct = safe_float(sampled_utilization_raw, 0.0)
                self.state.signals.gpu_utilization_pct = clamp(sampled_utilization_pct, 0.0, 100.0)
            sampled_power_watts = getattr(gpu_sample, "power_watts", None)
            if sampled_power_watts is not None:
                self.state.signals.power_watts = max(0.0, safe_float(sampled_power_watts, 0.0))
                self.state.signals.estimated_heat_output_watts = self.state.signals.power_watts
            sampled_gpu_temp_c = getattr(gpu_sample, "temperature_c", None)
            if sampled_gpu_temp_c is not None:
                self.state.signals.gpu_temp_c = max(0.0, safe_float(sampled_gpu_temp_c, 0.0))
            return

        sampled_pressure = self.sample_gpu_memory_pressure()
        if sampled_pressure is not None:
            self.state.signals.gpu_memory_pressure = sampled_pressure
            self.state.signals.gpu_memory_utilization_pct = clamp(sampled_pressure * 100.0, 0.0, 100.0)
        sampled_utilization_pct = self.sample_gpu_utilization_pct()
        if sampled_utilization_pct is not None:
            self.state.signals.gpu_utilization_pct = sampled_utilization_pct
        sampled_thermal = self.sample_gpu_thermal_metrics()
        if sampled_thermal is not None:
            self.state.signals.gpu_temp_c = sampled_thermal["gpu_temp_c"]
            self.state.signals.power_watts = sampled_thermal["power_watts"]
            self.state.signals.estimated_heat_output_watts = sampled_thermal["estimated_heat_output_watts"]

    @staticmethod
    def _target_concurrency_ceiling_for_model(model: str, settings: NodeAgentSettings) -> int:
        gpu_memory_gb = settings_float(settings, "gpu_memory_gb", 24.0)
        operations = operations_for_model(model, settings)
        embedding_only = operations == ["embeddings"]
        if embedding_only:
            if gpu_memory_gb < 12:
                return 1
            if gpu_memory_gb < 16:
                return 3
            if gpu_memory_gb < 24:
                return 5
            if gpu_memory_gb < 48:
                return 7
            return 10
        if gpu_memory_gb < 12:
            return 1
        if gpu_memory_gb < 16:
            return 2
        if gpu_memory_gb < 24:
            return 3
        if gpu_memory_gb < 48:
            return 5
        return 6

    def configured_target_gpu_utilization_pct(self) -> int:
        return int(
            clamp(
                float(settings_int(self.settings, "target_gpu_utilization_pct", MAX_TARGET_GPU_UTILIZATION_PCT)),
                MIN_TARGET_GPU_UTILIZATION_PCT,
                MAX_TARGET_GPU_UTILIZATION_PCT,
            )
        )

    def configured_min_gpu_memory_headroom_pct(self) -> float:
        return clamp(
            settings_float(self.settings, "min_gpu_memory_headroom_pct", 20.0),
            MIN_GPU_MEMORY_HEADROOM_PCT,
            MAX_GPU_MEMORY_HEADROOM_PCT,
        )

    def configured_memory_utilization_ceiling_pct(self) -> float:
        return clamp(100.0 - self.configured_min_gpu_memory_headroom_pct(), 40.0, 95.0)

    def dynamic_concurrency_ceiling(self, *, model: str) -> int:
        safe_ceiling = self._target_concurrency_ceiling_for_model(model, self.settings)
        target_gpu_utilization_pct = self.configured_target_gpu_utilization_pct()
        scaled_ceiling = max(1, round(safe_ceiling * (target_gpu_utilization_pct / 100.0)))
        return max(1, scaled_ceiling)

    def target_thermal_headroom(self, *, configured_headroom: float, target_gpu_utilization_pct: int) -> float:
        scaled_headroom = 0.55 + (
            (
                clamp(float(target_gpu_utilization_pct), MIN_TARGET_GPU_UTILIZATION_PCT, MAX_TARGET_GPU_UTILIZATION_PCT)
                - MIN_TARGET_GPU_UTILIZATION_PCT
            )
            / (MAX_TARGET_GPU_UTILIZATION_PCT - MIN_TARGET_GPU_UTILIZATION_PCT)
        ) * 0.37
        if target_gpu_utilization_pct >= MAX_TARGET_GPU_UTILIZATION_PCT:
            return clamp(max(configured_headroom, scaled_headroom), 0.2, 1.0)
        return clamp(min(configured_headroom, scaled_headroom), 0.2, 1.0)

    def observe_idle(self, *, queue_depth: int = 0, active_assignments: int = 0, gpu_sample: Any | None = None) -> None:
        signals = self.state.signals
        signals.queue_depth = max(0, queue_depth)
        signals.active_assignments = max(0, active_assignments)
        self.refresh_gpu_signals(gpu_sample=gpu_sample)
        signals.last_observed_at = now_iso()
        self.evaluate()
        self.save()

    def observe_assignment_success(
        self,
        *,
        latency_seconds: float,
        queue_depth: int = 0,
        active_assignments: int = 0,
        gpu_sample: Any | None = None,
    ) -> None:
        signals = self.state.signals
        signals.completed_count += 1
        signals.last_latency_seconds = max(0.0, latency_seconds)
        signals.ewma_latency_seconds = (
            signals.last_latency_seconds
            if signals.ewma_latency_seconds is None
            else (signals.ewma_latency_seconds * 0.72) + (signals.last_latency_seconds * 0.28)
        )
        signals.queue_depth = max(0, queue_depth)
        signals.active_assignments = max(0, active_assignments)
        signals.last_failure_code = None
        signals.last_failure_retryable = None
        self._refresh_failure_rate()
        self.refresh_gpu_signals(gpu_sample=gpu_sample)
        signals.last_observed_at = now_iso()
        self.evaluate()
        self.save()

    def observe_assignment_failure(
        self,
        *,
        code: str,
        retryable: bool,
        queue_depth: int = 0,
        active_assignments: int = 0,
        gpu_sample: Any | None = None,
    ) -> None:
        signals = self.state.signals
        signals.failed_count += 1
        signals.last_failure_code = code
        signals.last_failure_retryable = retryable
        signals.queue_depth = max(0, queue_depth)
        signals.active_assignments = max(0, active_assignments)
        self._refresh_failure_rate()
        self.refresh_gpu_signals(gpu_sample=gpu_sample)
        signals.last_observed_at = now_iso()
        self.evaluate()
        self.save()

    def _refresh_failure_rate(self) -> None:
        total = self.state.signals.completed_count + self.state.signals.failed_count
        self.state.signals.failure_rate = 0.0 if total == 0 else clamp(self.state.signals.failed_count / total, 0.0, 1.0)

    def smaller_model_available(self) -> bool:
        return DEFAULT_EMBEDDING_MODEL in configured_models(self.settings)

    def evaluate(self) -> AutopilotRecommendation:
        signals = self.state.signals
        baseline = max(1, self.state.baseline_concurrency)
        previous_recommendation = self.state.recommendation
        pressure = signals.gpu_memory_pressure
        gpu_memory_utilization_pct = (
            signals.gpu_memory_utilization_pct
            if signals.gpu_memory_utilization_pct is not None
            else (pressure * 100.0 if pressure is not None else None)
        )
        gpu_utilization_pct = signals.gpu_utilization_pct
        latency = signals.ewma_latency_seconds
        enough_samples = signals.completed_count + signals.failed_count >= 3
        failure_high = enough_samples and signals.failure_rate >= 0.25
        memory_utilization_ceiling_pct = self.configured_memory_utilization_ceiling_pct()
        pressure_high = (
            gpu_memory_utilization_pct is not None
            and gpu_memory_utilization_pct >= memory_utilization_ceiling_pct
        ) or (pressure is not None and pressure >= 0.92)
        pressure_critical = (
            gpu_memory_utilization_pct is not None
            and gpu_memory_utilization_pct >= min(99.0, memory_utilization_ceiling_pct + 8.0)
        ) or (pressure is not None and pressure >= 0.96)
        latency_high = latency is not None and latency >= 180
        target_gpu_utilization_pct = self.configured_target_gpu_utilization_pct()
        demand_high = signals.queue_depth >= 1 or signals.active_assignments >= max(1, previous_recommendation.max_concurrent_assignments)
        stable = not failure_high and not pressure_high and (latency is None or latency < 120)

        configured_model = settings_str(self.settings, "vllm_model", self.state.baseline_model or DEFAULT_RESPONSE_MODEL)
        configured_models_csv = settings_str(self.settings, "supported_models", configured_model)
        configured_headroom = clamp(settings_float(self.settings, "thermal_headroom", 0.8), 0.2, 1.0)
        current_concurrency = max(1, previous_recommendation.max_concurrent_assignments or baseline)
        target_model = configured_model
        supported_models = configured_models_csv
        concurrency = current_concurrency
        thermal_headroom = self.target_thermal_headroom(
            configured_headroom=configured_headroom,
            target_gpu_utilization_pct=target_gpu_utilization_pct,
        )
        profile = self.profile_from_headroom(thermal_headroom)
        reason = "Autopilot is holding the current plan while it collects enough GPU utilization data to retune safely."
        target_concurrency_ceiling = self.dynamic_concurrency_ceiling(model=configured_model)
        utilization_below_target = (
            gpu_utilization_pct is None
            or gpu_utilization_pct <= max(0.0, target_gpu_utilization_pct - 12.0)
        )
        utilization_far_above_target = (
            gpu_utilization_pct is not None
            and gpu_utilization_pct >= min(100.0, target_gpu_utilization_pct + 18.0)
        )

        if (pressure_critical or failure_high) and self.smaller_model_available():
            target_model = DEFAULT_EMBEDDING_MODEL
            supported_models = DEFAULT_EMBEDDING_MODEL
            profile = "quiet"
            concurrency = 1
            thermal_headroom = 0.55
            reason = (
                "Autopilot is moving to a lighter model because recent failures or GPU memory pressure "
                "suggest the current plan is too aggressive."
            )
        elif pressure_high or failure_high or latency_high:
            profile = "quiet"
            concurrency = max(1, current_concurrency - 1)
            thermal_headroom = min(thermal_headroom, 0.6)
            reason = (
                "Autopilot reduced active concurrency because latency, failures, or the configured GPU memory "
                "headroom guardrail was breached."
            )
        elif demand_high and stable and current_concurrency < target_concurrency_ceiling and utilization_below_target:
            concurrency = min(target_concurrency_ceiling, current_concurrency + 1)
            thermal_headroom = self.target_thermal_headroom(
                configured_headroom=configured_headroom,
                target_gpu_utilization_pct=target_gpu_utilization_pct,
            )
            profile = self.profile_from_headroom(thermal_headroom)
            supported_models = configured_models_csv
            reason = (
                "Autopilot increased active concurrency because work is queued, the GPU is still below the owner "
                "utilization target, and memory headroom remains healthy."
            )
        elif stable and utilization_far_above_target and current_concurrency > 1:
            concurrency = max(1, current_concurrency - 1)
            thermal_headroom = min(thermal_headroom, configured_headroom)
            profile = self.profile_from_headroom(thermal_headroom)
            supported_models = configured_models_csv
            reason = (
                "Autopilot reduced active concurrency because the node is already above the owner utilization target "
                "without needing more parallel work."
            )
        else:
            concurrency = min(current_concurrency, target_concurrency_ceiling)
            thermal_headroom = self.target_thermal_headroom(
                configured_headroom=configured_headroom,
                target_gpu_utilization_pct=target_gpu_utilization_pct,
            )
            profile = self.profile_from_headroom(thermal_headroom)

        operations = operations_for_model(target_model, self.settings)
        pending_restart = target_model != configured_model
        runtime_profile = getattr(self.settings, "resolved_runtime_profile", None)
        runtime_profile_id = getattr(runtime_profile, "id", None)
        if getattr(runtime_profile, "inference_engine", None) == LLAMA_CPP_INFERENCE_ENGINE:
            runtime_profile_id = (
                HOME_EMBEDDINGS_LLAMA_CPP_PROFILE
                if target_model == DEFAULT_EMBEDDING_MODEL
                else HOME_LLAMA_CPP_GGUF_PROFILE
            )
        env_updates = {
            "SETUP_PROFILE": profile,
            "MAX_CONCURRENT_ASSIGNMENTS": str(concurrency),
            "THERMAL_HEADROOM": str(thermal_headroom),
            "SUPPORTED_MODELS": supported_models,
            "VLLM_MODEL": target_model,
        }
        if isinstance(runtime_profile_id, str) and runtime_profile_id:
            env_updates["RUNTIME_PROFILE"] = runtime_profile_id
        recommendation = AutopilotRecommendation(
            setup_profile=profile,
            max_concurrent_assignments=concurrency,
            thermal_headroom=thermal_headroom,
            startup_model=target_model,
            supported_models=supported_models,
            operations=operations,
            reason=reason,
            pending_restart=pending_restart,
            safe_to_apply=signals.active_assignments == 0,
            env_updates=env_updates,
        )
        previous = self.state.recommendation
        if previous.reason != recommendation.reason or previous.startup_model != recommendation.startup_model:
            self.state.history.append(
                {
                    "at": now_iso(),
                    "reason": recommendation.reason,
                    "startup_model": recommendation.startup_model,
                    "max_concurrent_assignments": recommendation.max_concurrent_assignments,
                    "gpu_memory_pressure": signals.gpu_memory_pressure,
                    "gpu_utilization_pct": signals.gpu_utilization_pct,
                    "gpu_memory_utilization_pct": signals.gpu_memory_utilization_pct,
                    "failure_rate": signals.failure_rate,
                    "ewma_latency_seconds": signals.ewma_latency_seconds,
                }
            )
            self.state.history = self.state.history[-20:]
        self.state.recommendation = recommendation
        self.state.current_model = configured_model
        return recommendation

    def capabilities_payload(self) -> dict[str, Any]:
        recommendation = self.state.recommendation
        signals = self.state.signals
        live_model = live_runtime_model(self.settings)
        live_supported_models = live_model
        live_operations = operations_for_model(live_model, self.settings)
        runtime_tuple = resolved_runtime_tuple(
            self.settings,
            live_model,
            live_operations[0] if live_operations else None,
        )
        embedding_concurrency_limit = resolved_embeddings_concurrency_limit(
            supported_models=live_supported_models,
            operations=live_operations,
            gpu_memory_gb=settings_float(self.settings, "gpu_memory_gb", 24.0),
            max_concurrent_assignments=max(
                1,
                min(
                    recommendation.max_concurrent_assignments,
                    settings_int(
                        self.settings,
                        "max_concurrent_assignments_embeddings",
                        recommendation.max_concurrent_assignments,
                    ),
                ),
            ),
            override=max(
                1,
                min(
                    recommendation.max_concurrent_assignments,
                    settings_int(
                        self.settings,
                        "max_concurrent_assignments_embeddings",
                        recommendation.max_concurrent_assignments,
                    ),
                ),
            ),
        )
        prefetch_concurrency_target = max(
            recommendation.max_concurrent_assignments,
            self.dynamic_concurrency_ceiling(model=live_model),
        )
        embedding_microbatch_limit = resolved_embeddings_microbatch_assignment_limit(
            supported_models=live_supported_models,
            operations=live_operations,
            gpu_memory_gb=settings_float(self.settings, "gpu_memory_gb", 24.0),
            max_concurrent_assignments=prefetch_concurrency_target,
            pull_bundle_size=settings_int(self.settings, "pull_bundle_size", 16),
            override=getattr(self.settings, "max_microbatch_assignments_embeddings", None),
        )
        payload: dict[str, Any] = {
            "supported_models": [live_model],
            "operations": live_operations,
            "gpu_name": settings_str(self.settings, "gpu_name", "Generic GPU"),
            "gpu_memory_gb": settings_float(self.settings, "gpu_memory_gb", 24.0),
            "max_context_tokens": runtime_tuple.effective_context_tokens
            or settings_int(self.settings, "max_context_tokens", 32768),
            "max_batch_tokens": settings_int(self.settings, "max_batch_tokens", 50000),
            "max_concurrent_assignments": recommendation.max_concurrent_assignments,
            "target_gpu_utilization_pct": self.configured_target_gpu_utilization_pct(),
            "min_gpu_memory_headroom_pct": self.configured_min_gpu_memory_headroom_pct(),
            "thermal_headroom": recommendation.thermal_headroom,
            "heat_demand": normalize_heat_demand(settings_str(self.settings, "heat_demand", "none")),
        }
        if embedding_concurrency_limit is not None:
            payload["max_concurrent_assignments_embeddings"] = embedding_concurrency_limit
        if embedding_microbatch_limit is not None:
            payload["max_microbatch_assignments_embeddings"] = embedding_microbatch_limit
            payload["max_pull_bundle_assignments"] = max(
                recommendation.max_concurrent_assignments,
                embedding_microbatch_limit,
            )
        configured_heat_output_watts = settings_optional_float(self.settings, "estimated_heat_output_watts")
        optional_values = {
            "room_temp_c": settings_optional_float(self.settings, "room_temp_c"),
            "target_temp_c": settings_optional_float(self.settings, "target_temp_c"),
            "gpu_temp_c": signals.gpu_temp_c if signals.gpu_temp_c is not None else settings_optional_float(self.settings, "gpu_temp_c"),
            "power_watts": signals.power_watts if signals.power_watts is not None else settings_optional_float(self.settings, "power_watts"),
            "estimated_heat_output_watts": configured_heat_output_watts
            if configured_heat_output_watts is not None
            else signals.estimated_heat_output_watts,
            "energy_price_kwh": settings_optional_float(self.settings, "energy_price_kwh"),
        }
        if optional_values["estimated_heat_output_watts"] is None:
            optional_values["estimated_heat_output_watts"] = optional_values["power_watts"]
        for key, value in optional_values.items():
            if value is not None:
                payload[key] = value
        return payload

    def runtime_payload(self) -> dict[str, Any]:
        payload = asdict(self.state)
        payload["state_path"] = str(self.state_path)
        return payload
