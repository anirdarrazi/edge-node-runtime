from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from queue import Empty, Queue
import socket
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

try:  # pragma: no cover - optional dependency
    import pynvml  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

from .autopilot import AutopilotController
from .config import NodeAgentSettings
from .concurrency import (
    max_local_queue_assignments_from_capabilities,
    max_microbatch_assignments_from_capabilities,
    max_worker_assignments_from_capabilities,
)
from .control_plane import ArtifactFlowError, EdgeControlClient
from .gguf_artifacts import find_gguf_artifact, resolved_gguf_artifact_contract
from .heat_governor import NvidiaPowerLimitController, power_limit_for_target
from .inference_engine import VLLM_INFERENCE_ENGINE
from .model_artifacts import find_model_artifact
from .runtime import VLLMRuntime
from .single_container import known_safe_max_model_len
from .runtime_tuple import resolved_runtime_tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("autonomousc-node-agent")
assignment_progress_keepalive_seconds = 30.0
assignment_lease_keepalive_seconds = 30.0
heartbeat_interval_seconds = 15.0
throughput_log_interval_seconds = 30.0
gpu_telemetry_refresh_interval_seconds = 5.0
supported_operations = frozenset({"responses", "embeddings"})
vram_comparison_tolerance_gb = 0.25
low_utilization_alert_threshold_pct = 25.0
queued_idle_alert_threshold_fraction = 0.10
local_reservoir_floor_assignments = 16
degraded_reservoir_floor_assignments = 32
degraded_reservoir_cap_assignments = 96
oom_message_markers = (
    "out of memory",
    "cuda out of memory",
    "cuda error: out of memory",
    "insufficient memory",
    "cublas_status_alloc_failed",
    "allocation on device",
)


def normalize_region_token(value: object) -> str:
    return str(value).strip().lower() if isinstance(value, str) else ""


def infer_country_code_from_region(region: str) -> str | None:
    trimmed = region.strip()
    if not trimmed:
        return None

    parts = [part for part in trimmed.replace("_", "-").split("-") if part]
    last_part = parts[-1] if parts else ""
    if (
        len(parts) >= 3
        and len(parts[0]) == 2
        and parts[0].isalpha()
        and len(parts[1]) == 2
        and parts[1].isalpha()
        and last_part.isdigit()
    ):
        return parts[1].upper()

    if len(parts) >= 2 and len(parts[0]) == 2 and parts[0].isalpha() and len(parts[1]) > 2:
        return parts[0].upper()

    return None


def node_region_scope_tokens(region: str) -> set[str]:
    trimmed = region.strip()
    if not trimmed:
        return set()

    scopes = {normalize_region_token(trimmed)}
    country_code = infer_country_code_from_region(trimmed)
    if country_code:
        scopes.add(normalize_region_token(country_code))
    return scopes


def allowed_regions_include_node_region(allowed_regions: object, node_region: str) -> bool:
    if not isinstance(allowed_regions, list):
        return False
    normalized_allowed_regions = {
        normalize_region_token(value) for value in allowed_regions if normalize_region_token(value)
    }
    if "global" in normalized_allowed_regions:
        return True
    return any(scope in normalized_allowed_regions for scope in node_region_scope_tokens(node_region))


@dataclass
class AssignmentWorkerResult:
    assignment_id: str
    kind: str
    queue_depth: int
    latency_seconds: float | None = None
    operation: str | None = None
    model: str | None = None
    item_count: int = 0
    usage_summary: dict[str, int] | None = None
    microbatch_assignments: int | None = None
    code: str | None = None
    retryable: bool | None = None
    error: Exception | None = None


@dataclass
class PreparedAssignment:
    assignment: object
    items: list[dict[str, object]]
    item_count: int
    prefetched_at_monotonic: float

    def __getattr__(self, name: str) -> object:
        return getattr(self.assignment, name)


@dataclass
class PrefetchWorkerResult:
    assignment: object
    prepared_assignment: PreparedAssignment | None = None
    error: Exception | None = None


@dataclass
class ActiveBundleWorker:
    thread: threading.Thread
    assignment_ids: set[str]


@dataclass
class HeartbeatState:
    last_sent_at: float | None = None
    status: str | None = None
    active_assignments: int | None = None
    capabilities_signature: str | None = None
    runtime_signature: str | None = None
    consecutive_failures: int = 0
    retry_after_at: float | None = None


@dataclass
class LeaseKeepaliveState:
    last_sent_at: float | None = None
    assignment_signature: str | None = None
    consecutive_failures: int = 0
    retry_after_at: float | None = None


@dataclass
class ThroughputWindowBucket:
    operation: str
    model: str
    completed_assignments: int = 0
    completed_items: int = 0
    input_texts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_seconds_sum: float = 0.0
    latency_observations: int = 0
    microbatch_assignments_sum: int = 0
    microbatch_observations: int = 0


@dataclass
class GPUTelemetrySample:
    utilization_percent: float
    memory_utilization_percent: float
    power_watts: float | None = None
    temperature_c: float | None = None
    source: str = "nvidia-smi"


class GPUTelemetrySampler:
    def __init__(self, refresh_interval_seconds: float = gpu_telemetry_refresh_interval_seconds) -> None:
        self.refresh_interval_seconds = max(1.0, float(refresh_interval_seconds))
        self._last_sample_time: float | None = None
        self._last_sample: GPUTelemetrySample | None = None
        self._nvml_handle = None
        self._nvml_disabled = False

    def sample(self, *, now_monotonic: float | None = None) -> GPUTelemetrySample | None:
        current_time = time.monotonic() if now_monotonic is None else now_monotonic
        if (
            self._last_sample_time is not None
            and current_time - self._last_sample_time < self.refresh_interval_seconds
        ):
            return self._last_sample

        sample = self._sample_nvml() or self._sample_nvidia_smi()
        self._last_sample_time = current_time
        self._last_sample = sample
        return sample

    def _sample_nvml(self) -> GPUTelemetrySample | None:
        if pynvml is None or self._nvml_disabled:
            return None
        try:
            if self._nvml_handle is None:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            handle = self._nvml_handle
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            temperature_c = pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU,
            )
            memory_utilization_percent = (
                (float(memory.used) / float(memory.total)) * 100.0 if getattr(memory, "total", 0) else 0.0
            )
            return GPUTelemetrySample(
                utilization_percent=max(0.0, float(utilization.gpu)),
                memory_utilization_percent=max(0.0, memory_utilization_percent),
                power_watts=max(0.0, float(power_mw) / 1000.0),
                temperature_c=max(0.0, float(temperature_c)),
                source="nvml",
            )
        except Exception:
            self._nvml_disabled = True
            return None

    @staticmethod
    def _safe_float(value: str) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _sample_nvidia_smi(self) -> GPUTelemetrySample | None:
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
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
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 5:
            return None
        utilization_percent = self._safe_float(parts[0])
        memory_used_mb = self._safe_float(parts[1])
        memory_total_mb = self._safe_float(parts[2])
        power_watts = self._safe_float(parts[3])
        temperature_c = self._safe_float(parts[4])
        if utilization_percent is None or memory_used_mb is None or memory_total_mb is None or memory_total_mb <= 0:
            return None
        memory_utilization_percent = (memory_used_mb / memory_total_mb) * 100.0
        return GPUTelemetrySample(
            utilization_percent=max(0.0, utilization_percent),
            memory_utilization_percent=max(0.0, memory_utilization_percent),
            power_watts=None if power_watts is None else max(0.0, power_watts),
            temperature_c=None if temperature_c is None else max(0.0, temperature_c),
            source="nvidia-smi",
        )


class NodeThroughputLogger:
    def __init__(self, interval_seconds: float = throughput_log_interval_seconds) -> None:
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._reset_window(time.monotonic())

    def _reset_window(self, now_monotonic: float) -> None:
        self._window_started_at = now_monotonic
        self._last_observed_at = now_monotonic
        self._sample_count = 0
        self._active_assignments_sum = 0
        self._worker_limit_sum = 0
        self._queue_depth_sum = 0
        self._prefetch_queue_depth_sum = 0
        self._active_assignments_peak = 0
        self._last_active_assignments = 0
        self._last_worker_limit = 0
        self._last_queue_depth = 0
        self._last_prefetch_queue_depth = 0
        self._queued_idle_seconds = 0.0
        self._refill_gap_started_at: float | None = None
        self._refill_gap_seconds_sum = 0.0
        self._refill_gap_count = 0
        self._refill_gap_max_seconds = 0.0
        self._gpu_utilization_sum = 0.0
        self._gpu_utilization_count = 0
        self._gpu_utilization_peak = 0.0
        self._gpu_memory_utilization_sum = 0.0
        self._gpu_memory_utilization_count = 0
        self._gpu_memory_utilization_peak = 0.0
        self._gpu_power_sum = 0.0
        self._gpu_power_count = 0
        self._gpu_power_peak = 0.0
        self._gpu_temperature_sum = 0.0
        self._gpu_temperature_count = 0
        self._gpu_temperature_peak = 0.0
        self._gpu_source: str | None = None
        self._microbatch_assignments_sum = 0
        self._microbatch_observations = 0
        self._buckets: dict[tuple[str, str], ThroughputWindowBucket] = {}

    @staticmethod
    def _metric_value(summary: dict[str, int] | None, key: str) -> int:
        if not isinstance(summary, dict):
            return 0
        value = summary.get(key, 0)
        return value if isinstance(value, int) and value > 0 else 0

    @staticmethod
    def _format_metric(value: float | None, *, suffix: str = "", decimals: int = 1) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{decimals}f}{suffix}"

    def observe_loop(
        self,
        *,
        active_assignments: int,
        worker_limit: int,
        queue_depth: int,
        prefetch_queue_depth: int = 0,
        gpu_sample: GPUTelemetrySample | None = None,
        now_monotonic: float | None = None,
    ) -> None:
        current_time = time.monotonic() if now_monotonic is None else now_monotonic
        active_value = max(0, int(active_assignments))
        worker_limit_value = max(0, int(worker_limit))
        queue_depth_value = max(0, int(queue_depth))
        prefetch_queue_depth_value = max(0, int(prefetch_queue_depth))
        if self._sample_count:
            interval_seconds = max(0.0, current_time - self._last_observed_at)
            if self._last_queue_depth > 0 and self._last_active_assignments == 0:
                self._queued_idle_seconds += interval_seconds
        if self._refill_gap_started_at is not None and (active_value > 0 or queue_depth_value <= 0):
            refill_gap_seconds = max(0.0, current_time - self._refill_gap_started_at)
            self._refill_gap_seconds_sum += refill_gap_seconds
            self._refill_gap_count += 1
            self._refill_gap_max_seconds = max(self._refill_gap_max_seconds, refill_gap_seconds)
            self._refill_gap_started_at = None
        if self._refill_gap_started_at is None and queue_depth_value > 0 and active_value == 0:
            self._refill_gap_started_at = current_time

        self._sample_count += 1
        self._active_assignments_sum += active_value
        self._worker_limit_sum += worker_limit_value
        self._queue_depth_sum += queue_depth_value
        self._prefetch_queue_depth_sum += prefetch_queue_depth_value
        self._active_assignments_peak = max(self._active_assignments_peak, active_value)
        self._last_active_assignments = active_value
        self._last_worker_limit = worker_limit_value
        self._last_queue_depth = queue_depth_value
        self._last_prefetch_queue_depth = prefetch_queue_depth_value
        self._last_observed_at = current_time
        if gpu_sample is not None:
            self._gpu_source = gpu_sample.source
            self._gpu_utilization_sum += max(0.0, gpu_sample.utilization_percent)
            self._gpu_utilization_count += 1
            self._gpu_utilization_peak = max(self._gpu_utilization_peak, gpu_sample.utilization_percent)
            self._gpu_memory_utilization_sum += max(0.0, gpu_sample.memory_utilization_percent)
            self._gpu_memory_utilization_count += 1
            self._gpu_memory_utilization_peak = max(
                self._gpu_memory_utilization_peak,
                gpu_sample.memory_utilization_percent,
            )
            if gpu_sample.power_watts is not None:
                self._gpu_power_sum += max(0.0, gpu_sample.power_watts)
                self._gpu_power_count += 1
                self._gpu_power_peak = max(self._gpu_power_peak, gpu_sample.power_watts)
            if gpu_sample.temperature_c is not None:
                self._gpu_temperature_sum += max(0.0, gpu_sample.temperature_c)
                self._gpu_temperature_count += 1
                self._gpu_temperature_peak = max(self._gpu_temperature_peak, gpu_sample.temperature_c)

    def observe_result(self, result: AssignmentWorkerResult) -> None:
        if result.kind != "success":
            return
        operation = result.operation or "unknown"
        model = result.model or "unknown"
        bucket = self._buckets.setdefault(
            (operation, model),
            ThroughputWindowBucket(operation=operation, model=model),
        )
        bucket.completed_assignments += 1
        bucket.completed_items += max(0, int(result.item_count))
        bucket.input_texts += self._metric_value(result.usage_summary, "input_texts")
        bucket.input_tokens += self._metric_value(result.usage_summary, "input_tokens")
        bucket.output_tokens += self._metric_value(result.usage_summary, "output_tokens")
        bucket.total_tokens += self._metric_value(result.usage_summary, "total_tokens")
        if result.latency_seconds is not None:
            bucket.latency_seconds_sum += max(0.0, float(result.latency_seconds))
            bucket.latency_observations += 1
        if isinstance(result.microbatch_assignments, int) and result.microbatch_assignments > 0:
            bucket.microbatch_assignments_sum += result.microbatch_assignments
            bucket.microbatch_observations += 1
            self._microbatch_assignments_sum += result.microbatch_assignments
            self._microbatch_observations += 1

    def maybe_log(self, *, now_monotonic: float | None = None) -> None:
        current_time = time.monotonic() if now_monotonic is None else now_monotonic
        elapsed = current_time - self._window_started_at
        if elapsed < self.interval_seconds:
            return
        effective_elapsed = max(elapsed, 0.001)
        average_active_assignments = (
            self._active_assignments_sum / self._sample_count
            if self._sample_count
            else float(self._last_active_assignments)
        )
        average_worker_limit = (
            self._worker_limit_sum / self._sample_count
            if self._sample_count
            else float(self._last_worker_limit)
        )
        average_queue_depth = (
            self._queue_depth_sum / self._sample_count
            if self._sample_count
            else float(self._last_queue_depth)
        )
        average_prefetch_queue_depth = (
            self._prefetch_queue_depth_sum / self._sample_count
            if self._sample_count
            else float(self._last_prefetch_queue_depth)
        )
        average_slot_utilization = (
            average_active_assignments / average_worker_limit if average_worker_limit > 0 else 0.0
        )
        queued_idle_fraction = self._queued_idle_seconds / effective_elapsed if effective_elapsed > 0 else 0.0
        average_refill_gap_seconds = (
            self._refill_gap_seconds_sum / self._refill_gap_count if self._refill_gap_count else 0.0
        )
        average_gpu_utilization = (
            self._gpu_utilization_sum / self._gpu_utilization_count if self._gpu_utilization_count else None
        )
        average_gpu_memory_utilization = (
            self._gpu_memory_utilization_sum / self._gpu_memory_utilization_count
            if self._gpu_memory_utilization_count
            else None
        )
        average_gpu_power = (
            self._gpu_power_sum / self._gpu_power_count if self._gpu_power_count else None
        )
        average_gpu_temperature = (
            self._gpu_temperature_sum / self._gpu_temperature_count if self._gpu_temperature_count else None
        )
        average_microbatch_assignments = (
            self._microbatch_assignments_sum / self._microbatch_observations
            if self._microbatch_observations
            else None
        )
        completed_assignments = sum(bucket.completed_assignments for bucket in self._buckets.values())
        completed_items = sum(bucket.completed_items for bucket in self._buckets.values())
        if completed_assignments == 0 and self._active_assignments_peak == 0 and average_queue_depth <= 0.0:
            self._reset_window(current_time)
            return

        LOGGER.info(
            "node throughput summary window=%.1fs active_now=%s active_avg=%.2f active_peak=%s "
            "worker_limit_now=%s worker_limit_avg=%.2f slot_utilization_avg=%.2f "
            "queue_depth_now=%s queue_depth_avg=%.2f prefetch_queue_now=%s prefetch_queue_avg=%.2f "
            "queued_idle_s=%.2f queued_idle_pct=%.2f "
            "refill_gap_avg_s=%.2f refill_gap_max_s=%.2f gpu_source=%s gpu_util_avg=%s gpu_util_peak=%s "
            "gpu_mem_avg=%s gpu_mem_peak=%s gpu_power_avg=%s gpu_power_peak=%s gpu_temp_avg=%s gpu_temp_peak=%s "
            "microbatch_avg=%s completed_assignments=%s completed_items=%s",
            effective_elapsed,
            self._last_active_assignments,
            average_active_assignments,
            self._active_assignments_peak,
            self._last_worker_limit,
            average_worker_limit,
            average_slot_utilization,
            self._last_queue_depth,
            average_queue_depth,
            self._last_prefetch_queue_depth,
            average_prefetch_queue_depth,
            self._queued_idle_seconds,
            queued_idle_fraction,
            average_refill_gap_seconds,
            self._refill_gap_max_seconds,
            self._gpu_source or "n/a",
            self._format_metric(average_gpu_utilization, suffix="%", decimals=1),
            self._format_metric(
                self._gpu_utilization_peak if self._gpu_utilization_count else None,
                suffix="%",
                decimals=1,
            ),
            self._format_metric(average_gpu_memory_utilization, suffix="%", decimals=1),
            self._format_metric(
                self._gpu_memory_utilization_peak if self._gpu_memory_utilization_count else None,
                suffix="%",
                decimals=1,
            ),
            self._format_metric(average_gpu_power, suffix="W", decimals=1),
            self._format_metric(
                self._gpu_power_peak if self._gpu_power_count else None,
                suffix="W",
                decimals=1,
            ),
            self._format_metric(average_gpu_temperature, suffix="C", decimals=1),
            self._format_metric(
                self._gpu_temperature_peak if self._gpu_temperature_count else None,
                suffix="C",
                decimals=1,
            ),
            self._format_metric(average_microbatch_assignments, decimals=2),
            completed_assignments,
            completed_items,
        )
        if average_queue_depth >= 1.0 and (
            queued_idle_fraction >= queued_idle_alert_threshold_fraction
            or (
                average_gpu_utilization is not None
                and average_gpu_utilization < low_utilization_alert_threshold_pct
                and average_slot_utilization < 0.75
            )
        ):
            LOGGER.warning(
                "node utilization alert condition=queued_capable_idle_or_low_util window=%.1fs "
                "queue_depth_avg=%.2f prefetch_queue_avg=%.2f active_avg=%.2f worker_limit_avg=%.2f "
                "slot_utilization_avg=%.2f queued_idle_pct=%.2f refill_gap_avg_s=%.2f gpu_util_avg=%s",
                effective_elapsed,
                average_queue_depth,
                average_prefetch_queue_depth,
                average_active_assignments,
                average_worker_limit,
                average_slot_utilization,
                queued_idle_fraction,
                average_refill_gap_seconds,
                self._format_metric(average_gpu_utilization, suffix="%", decimals=1),
            )
        for bucket in sorted(self._buckets.values(), key=lambda value: (value.operation, value.model)):
            average_latency = (
                bucket.latency_seconds_sum / bucket.latency_observations
                if bucket.latency_observations
                else 0.0
            )
            average_microbatch_assignments = (
                bucket.microbatch_assignments_sum / bucket.microbatch_observations
                if bucket.microbatch_observations
                else 0.0
            )
            LOGGER.info(
                "node throughput detail window=%.1fs op=%s model=%s completed_assignments=%s "
                "completed_items=%s assignments_per_s=%.2f items_per_s=%.2f input_texts=%s "
                "texts_per_s=%.2f input_tokens=%s input_tokens_per_s=%.1f output_tokens=%s "
                "output_tokens_per_s=%.1f total_tokens=%s total_tokens_per_s=%.1f "
                "avg_latency_s=%.2f avg_microbatch_assignments=%.2f",
                effective_elapsed,
                bucket.operation,
                bucket.model,
                bucket.completed_assignments,
                bucket.completed_items,
                bucket.completed_assignments / effective_elapsed,
                bucket.completed_items / effective_elapsed,
                bucket.input_texts,
                bucket.input_texts / effective_elapsed,
                bucket.input_tokens,
                bucket.input_tokens / effective_elapsed,
                bucket.output_tokens,
                bucket.output_tokens / effective_elapsed,
                bucket.total_tokens,
                bucket.total_tokens / effective_elapsed,
                average_latency,
                average_microbatch_assignments,
            )
        self._reset_window(current_time)


def heartbeat_payload_signature(payload: dict[str, object] | None) -> str | None:
    if payload is None:
        return None
    signature_payload = dict(payload)
    for volatile_key in ("gpu_temp_c", "power_watts", "estimated_heat_output_watts"):
        signature_payload.pop(volatile_key, None)
    return json.dumps(signature_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def control_plane_connectivity_snapshot(control: object) -> dict[str, Any]:
    snapshot = getattr(control, "control_plane_snapshot", None)
    if callable(snapshot):
        payload = snapshot()
        return payload if isinstance(payload, dict) else {}
    transport = getattr(control, "transport", None)
    transport_snapshot = getattr(transport, "snapshot", None)
    if callable(transport_snapshot):
        payload = transport_snapshot()
        return payload if isinstance(payload, dict) else {}
    return {}


def recommended_control_plane_retry_delay_seconds(control: object) -> float:
    recommender = getattr(control, "recommended_control_plane_retry_delay_seconds", None)
    if callable(recommender):
        try:
            return max(0.5, float(recommender()))
        except (TypeError, ValueError):
            pass
    return max(0.5, float(getattr(getattr(control, "settings", object()), "poll_interval_seconds", 10.0) or 10.0))


def control_plane_dispatch_allowed(connectivity: dict[str, Any]) -> bool:
    status = str(connectivity.get("status") or "").strip().lower()
    if status not in {"degraded", "offline"}:
        return True
    return bool(connectivity.get("grace_active"))


def recommended_local_reservoir_target(
    *,
    worker_limit: int,
    local_queue_limit: int,
    connectivity: dict[str, Any] | None = None,
) -> int:
    queue_cap = max(0, int(local_queue_limit))
    if queue_cap <= 0:
        return 0
    healthy_target = min(
        queue_cap,
        max(local_reservoir_floor_assignments, max(1, worker_limit) * 8),
    )
    connectivity = connectivity or {}
    status = str(connectivity.get("status") or "").strip().lower()
    if status in {"degraded", "offline"}:
        return min(
            queue_cap,
            degraded_reservoir_cap_assignments,
            max(healthy_target, degraded_reservoir_floor_assignments, max(1, worker_limit) * 12),
        )
    return min(queue_cap, degraded_reservoir_cap_assignments, healthy_target)


def maybe_send_heartbeat(
    control: EdgeControlClient,
    heartbeat_state: HeartbeatState,
    *,
    status: str,
    queue_depth: int,
    active_assignments: int,
    capabilities: dict[str, object] | None,
    runtime: dict[str, object] | None,
    now_monotonic: float | None = None,
) -> HeartbeatState:
    current_time = time.monotonic() if now_monotonic is None else now_monotonic
    capabilities_signature = heartbeat_payload_signature(capabilities)
    runtime_signature = heartbeat_payload_signature(runtime)
    status_changed = heartbeat_state.status != status
    active_assignments_changed = heartbeat_state.active_assignments != active_assignments
    capabilities_changed = (
        capabilities_signature is not None and heartbeat_state.capabilities_signature != capabilities_signature
    )
    runtime_changed = runtime_signature is not None and heartbeat_state.runtime_signature != runtime_signature
    interval_elapsed = (
        heartbeat_state.last_sent_at is None or current_time - heartbeat_state.last_sent_at >= heartbeat_interval_seconds
    )
    retry_blocked = heartbeat_state.retry_after_at is not None and current_time < heartbeat_state.retry_after_at

    if not (
        heartbeat_state.last_sent_at is None
        or status_changed
        or active_assignments_changed
        or capabilities_changed
        or runtime_changed
        or interval_elapsed
    ) or retry_blocked:
        return heartbeat_state

    include_capabilities = heartbeat_state.last_sent_at is None or capabilities_changed
    include_runtime = heartbeat_state.last_sent_at is None or runtime_changed
    try:
        control.heartbeat(
            status=status,
            queue_depth=queue_depth,
            active_assignments=active_assignments,
            capabilities=capabilities if include_capabilities else None,
            runtime=runtime if include_runtime else None,
            include_capabilities=include_capabilities,
            include_runtime=include_runtime,
        )
    except Exception as error:
        if hasattr(control, "is_transient_network_error") and control.is_transient_network_error(error):
            retry_delay = recommended_control_plane_retry_delay_seconds(control)
            LOGGER.warning("control plane connectivity degraded temporarily; heartbeat failed: %s", error)
            return HeartbeatState(
                last_sent_at=heartbeat_state.last_sent_at,
                status=status,
                active_assignments=active_assignments,
                capabilities_signature=capabilities_signature or heartbeat_state.capabilities_signature,
                runtime_signature=runtime_signature or heartbeat_state.runtime_signature,
                consecutive_failures=heartbeat_state.consecutive_failures + 1,
                retry_after_at=current_time + retry_delay,
            )
        raise
    return HeartbeatState(
        last_sent_at=current_time,
        status=status,
        active_assignments=active_assignments,
        capabilities_signature=capabilities_signature or heartbeat_state.capabilities_signature,
        runtime_signature=runtime_signature or heartbeat_state.runtime_signature,
        consecutive_failures=0,
        retry_after_at=None,
    )


def maybe_touch_assignment_leases(
    control: EdgeControlClient,
    lease_keepalive_state: LeaseKeepaliveState,
    assignment_ids: list[str],
    *,
    now_monotonic: float | None = None,
) -> LeaseKeepaliveState:
    current_time = time.monotonic() if now_monotonic is None else now_monotonic
    unique_assignment_ids = sorted({assignment_id for assignment_id in assignment_ids if assignment_id})
    if not unique_assignment_ids:
        return LeaseKeepaliveState()

    assignment_signature = ",".join(unique_assignment_ids)
    assignments_changed = lease_keepalive_state.assignment_signature != assignment_signature
    interval_elapsed = (
        lease_keepalive_state.last_sent_at is None
        or current_time - lease_keepalive_state.last_sent_at >= assignment_lease_keepalive_seconds
    )
    retry_blocked = lease_keepalive_state.retry_after_at is not None and current_time < lease_keepalive_state.retry_after_at
    if not assignments_changed and not interval_elapsed:
        return lease_keepalive_state
    if retry_blocked and not assignments_changed:
        return lease_keepalive_state

    try:
        control.touch_assignments(unique_assignment_ids)
    except Exception as error:
        if hasattr(control, "is_transient_network_error") and control.is_transient_network_error(error):
            retry_delay = recommended_control_plane_retry_delay_seconds(control)
            LOGGER.warning(
                "control plane connectivity degraded temporarily; assignment lease keepalive failed: %s",
                error,
            )
            return LeaseKeepaliveState(
                last_sent_at=lease_keepalive_state.last_sent_at,
                assignment_signature=assignment_signature,
                consecutive_failures=lease_keepalive_state.consecutive_failures + 1,
                retry_after_at=current_time + retry_delay,
            )
        raise
    return LeaseKeepaliveState(
        last_sent_at=current_time,
        assignment_signature=assignment_signature,
        consecutive_failures=0,
        retry_after_at=None,
    )


def resolved_inference_engine_for_settings(settings: object) -> str:
    value = getattr(settings, "resolved_inference_engine", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(settings, "inference_engine", None)
    if isinstance(value, str) and value:
        return value
    return VLLM_INFERENCE_ENGINE


def resolved_runtime_profile_for_settings(settings: object) -> str | None:
    value = getattr(settings, "resolved_runtime_profile_id", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(settings, "runtime_profile", None)
    if isinstance(value, str) and value and value != "auto":
        return value
    return None


def settings_support_trusted_assignments(settings: object) -> bool:
    value = getattr(settings, "supports_trusted_assignments", None)
    if isinstance(value, bool):
        return value
    return resolved_inference_engine_for_settings(settings) == VLLM_INFERENCE_ENGINE


def resolved_inference_base_url_for_settings(settings: object) -> str:
    value = getattr(settings, "resolved_inference_base_url", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(settings, "inference_base_url", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(settings, "vllm_base_url", None)
    if isinstance(value, str) and value:
        return value
    return "http://inference-runtime:8000"


def is_transient_http_error(error: Exception) -> bool:
    if isinstance(error, ArtifactFlowError):
        return error.retryable
    if isinstance(error, httpx.RequestError):
        return True
    if not isinstance(error, httpx.HTTPStatusError):
        return False
    return error.response.status_code in {408, 409, 425, 429} or error.response.status_code >= 500


def looks_like_gpu_oom(error: Exception | str) -> bool:
    if isinstance(error, Exception):
        text = str(error)
        if isinstance(error, httpx.HTTPStatusError):
            try:
                response_text = error.response.text
            except Exception:
                response_text = ""
            text = f"{text} {response_text}".strip()
    else:
        text = str(error)
    normalized = text.strip().lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in oom_message_markers):
        return True
    return " oom " in f" {normalized} "


def classify_assignment_failure(error: Exception) -> tuple[str, str, bool]:
    if isinstance(error, ArtifactFlowError):
        return error.code, str(error), error.retryable
    if looks_like_gpu_oom(error):
        return "gpu_oom", str(error) or "GPU ran out of memory.", True
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        request_url = str(error.request.url)
        if status in {400, 404, 409, 413, 422}:
            return "upstream_rejected", f"{request_url} returned HTTP {status}.", False
        if status in {408, 425, 429} or status >= 500:
            return "upstream_unavailable", f"{request_url} returned HTTP {status}.", True
        return "upstream_http_error", f"{request_url} returned HTTP {status}.", False
    if isinstance(error, httpx.RequestError):
        return "upstream_network_error", str(error), True
    if isinstance(error, (KeyError, TypeError, ValueError)):
        return "invalid_assignment_payload", str(error), False
    return "node_runtime_error", str(error) or error.__class__.__name__, True


def assignment_progress(state: str, item_count: int, **details: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "state": state,
        "item_count": item_count,
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(details)
    return payload


def control_plane_queue_depth(control: object) -> int:
    try:
        return max(0, int(getattr(control, "last_control_plane_queue_depth", 0) or 0))
    except (TypeError, ValueError):
        return 0


def configured_supported_models(settings: NodeAgentSettings) -> set[str]:
    return {model.strip() for model in str(settings.supported_models).split(",") if model.strip()}


def configured_supported_operations(settings: NodeAgentSettings) -> set[str]:
    return {operation.strip() for operation in settings.resolved_runtime_profile.supported_apis if operation.strip()}


def model_supports_runtime_operation(model: str, operation: str, inference_engine: str) -> bool:
    return (
        find_model_artifact(model, operation, runtime_engine=inference_engine) is not None
        or find_gguf_artifact(model, operation) is not None
    )


def _looks_like_node_region(region: str) -> bool:
    normalized = normalize_region_token(region).replace("_", "-")
    if normalized == "global":
        return True
    parts = [part for part in normalized.split("-") if part]
    return len(parts) >= 3 and all(part.isalnum() for part in parts)


def validate_startup_settings(settings: NodeAgentSettings) -> None:
    errors: list[str] = []
    if not _looks_like_node_region(settings.node_region):
        errors.append(
            f"NODE_REGION={settings.node_region!r} is invalid. Use 'global' for marketplace capacity "
            "or a concrete region like 'eu-se-1' or 'us-ca-1'."
        )

    supported_models = configured_supported_models(settings)
    if not supported_models:
        errors.append("SUPPORTED_MODELS must advertise at least one model.")

    current_model = settings.current_model
    if supported_models and current_model not in supported_models:
        errors.append(
            f"Current model {current_model!r} is not present in SUPPORTED_MODELS={sorted(supported_models)!r}."
        )

    supported_operations = configured_supported_operations(settings)
    if not supported_operations:
        errors.append(
            f"Runtime profile {settings.resolved_runtime_profile_id!r} does not advertise any supported operations."
        )

    incompatible_models = sorted(
        model
        for model in supported_models
        if not any(
            model_supports_runtime_operation(model, operation, settings.resolved_inference_engine)
            for operation in supported_operations
        )
    )
    if incompatible_models:
        errors.append(
            "SUPPORTED_MODELS advertises models the active runtime cannot actually serve for its configured "
            f"operations: {incompatible_models!r}."
        )

    safe_context_limit = known_safe_max_model_len(current_model)
    if safe_context_limit is not None and settings.max_context_tokens > safe_context_limit:
        errors.append(
            f"MAX_CONTEXT_TOKENS={settings.max_context_tokens} exceeds the safe limit {safe_context_limit} "
            f"for startup model {current_model!r}."
        )

    control_plane_host = urlparse(settings.edge_control_url).hostname
    if not control_plane_host:
        errors.append(f"EDGE_CONTROL_URL={settings.edge_control_url!r} is missing a valid hostname.")
    else:
        try:
            socket.getaddrinfo(control_plane_host, None)
        except OSError as error:
            errors.append(
                f"Could not resolve control-plane host {control_plane_host!r} during startup: {error}."
            )

    if errors:
        raise RuntimeError("Node runtime startup validation failed:\n- " + "\n- ".join(errors))


def restricted_attestation_state(control: EdgeControlClient) -> dict[str, object] | None:
    payload = control.load_attestation_state()
    if payload is None:
        return None
    if payload.get("status") != "verified":
        return None
    if payload.get("attestation_provider") != "hardware":
        return None
    attested_at = payload.get("attested_at")
    if not isinstance(attested_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(attested_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    node_id = payload.get("node_id")
    if not isinstance(node_id, str) or node_id != control.settings.node_id:
        return None
    return {
        "node_id": node_id,
        "attested_at": parsed.astimezone(timezone.utc),
    }


def restricted_attestation_is_fresh(
    control: EdgeControlClient,
    *,
    now: datetime | None = None,
) -> bool:
    state = restricted_attestation_state(control)
    if state is None:
        return False
    attested_at = state["attested_at"]
    if not isinstance(attested_at, datetime):
        return False
    current_time = now or datetime.now(timezone.utc)
    max_age_seconds = max(1, int(control.settings.restricted_attestation_max_age_seconds))
    age_seconds = (current_time - attested_at).total_seconds()
    return age_seconds >= 0 and age_seconds <= max_age_seconds


def validate_assignment_policy(control: EdgeControlClient, assignment: object) -> None:
    settings = control.settings
    assignment_model = getattr(assignment, "model", None)
    assignment_operation = getattr(assignment, "operation", None)
    if getattr(assignment, "operation", None) not in supported_operations:
        raise ValueError(f"assignment operation {getattr(assignment, 'operation', None)!r} is not supported by this node")
    if assignment_model not in configured_supported_models(settings):
        raise ValueError(f"assignment model {getattr(assignment, 'model', None)!r} is not supported by this node")
    allowed_regions = getattr(assignment, "allowed_regions", [])
    if not allowed_regions_include_node_region(allowed_regions, settings.node_region):
        raise ValueError(f"assignment is not allowed to run in node region {settings.node_region!r}")

    token_budget = getattr(assignment, "token_budget", {})
    total_tokens = token_budget.get("total_tokens") if isinstance(token_budget, dict) else None
    if not isinstance(total_tokens, int) or total_tokens < 0:
        raise ValueError("assignment token budget is invalid")
    if total_tokens > settings.max_batch_tokens:
        raise ValueError(
            f"assignment token budget {total_tokens} exceeds local batch limit {settings.max_batch_tokens}"
        )

    required_vram_gb = getattr(assignment, "required_vram_gb", None)
    if not isinstance(required_vram_gb, (int, float)) or required_vram_gb <= 0:
        raise ValueError("assignment required_vram_gb is invalid")
    if float(required_vram_gb) > settings.gpu_memory_gb + vram_comparison_tolerance_gb:
        raise ValueError(
            f"assignment requires {required_vram_gb} GiB of VRAM but this node only reports {settings.gpu_memory_gb}"
        )

    required_context_tokens = getattr(assignment, "required_context_tokens", None)
    if not isinstance(required_context_tokens, int) or required_context_tokens <= 0:
        raise ValueError("assignment required_context_tokens is invalid")
    if required_context_tokens > settings.max_context_tokens:
        raise ValueError(
            "assignment context window requirement "
            f"{required_context_tokens} exceeds local limit {settings.max_context_tokens}"
        )

    privacy_tier = getattr(assignment, "privacy_tier", None)
    if privacy_tier == "restricted":
        if not settings.restricted_capable:
            raise ValueError("restricted assignment rejected because the node is not marked restricted_capable")
        if settings.trust_tier != "restricted":
            raise ValueError(
                f"restricted assignment rejected because node trust tier is {settings.trust_tier!r}"
            )
        if settings.attestation_provider != "hardware":
            raise ValueError(
                "restricted assignment rejected because the current attestation provider is not hardware-backed"
            )
        if not restricted_attestation_is_fresh(control):
            raise ValueError(
                "restricted assignment rejected because no fresh local hardware attestation record is available"
            )
    if privacy_tier == "confidential" and settings.trust_tier == "standard":
        raise ValueError("confidential assignment rejected because the node trust tier is only standard")

    runtime_tuple = resolved_runtime_tuple(settings, assignment_model, assignment_operation)
    node_trust_requirement = getattr(assignment, "node_trust_requirement", None)
    if node_trust_requirement == "trusted_only":
        if not settings_support_trusted_assignments(settings):
            raise ValueError(
                "trusted assignment rejected because the current runtime profile is not trusted-work capable"
            )
        expected_runtime_image_digest = getattr(assignment, "expected_runtime_image_digest", None)
        if isinstance(expected_runtime_image_digest, str) and expected_runtime_image_digest:
            if runtime_tuple.runtime_image_digest != expected_runtime_image_digest:
                raise ValueError(
                    "trusted assignment rejected because the local runtime image digest "
                    f"{runtime_tuple.runtime_image_digest!r} does not match expected {expected_runtime_image_digest!r}"
                )
        expected_model_manifest_digest = getattr(assignment, "expected_model_manifest_digest", None)
        if isinstance(expected_model_manifest_digest, str) and expected_model_manifest_digest:
            if runtime_tuple.model_manifest_digest != expected_model_manifest_digest:
                raise ValueError(
                    "trusted assignment rejected because the local model manifest digest "
                    f"{runtime_tuple.model_manifest_digest!r} does not match expected {expected_model_manifest_digest!r}"
                )
        expected_tokenizer_digest = getattr(assignment, "expected_tokenizer_digest", None)
        if isinstance(expected_tokenizer_digest, str) and expected_tokenizer_digest:
            if runtime_tuple.tokenizer_digest != expected_tokenizer_digest:
                raise ValueError(
                    "trusted assignment rejected because the local tokenizer digest "
                    f"{runtime_tuple.tokenizer_digest!r} does not match expected {expected_tokenizer_digest!r}"
                )
        expected_chat_template_digest = getattr(assignment, "expected_chat_template_digest", None)
        if isinstance(expected_chat_template_digest, str) and expected_chat_template_digest:
            if runtime_tuple.chat_template_digest != expected_chat_template_digest:
                raise ValueError(
                    "trusted assignment rejected because the local chat template digest "
                    f"{runtime_tuple.chat_template_digest!r} does not match expected {expected_chat_template_digest!r}"
                )
        expected_effective_context_tokens = getattr(assignment, "expected_effective_context_tokens", None)
        if isinstance(expected_effective_context_tokens, int) and expected_effective_context_tokens > 0:
            if runtime_tuple.effective_context_tokens != expected_effective_context_tokens:
                raise ValueError(
                    "trusted assignment rejected because the local effective context size "
                    f"{runtime_tuple.effective_context_tokens!r} does not match expected "
                    f"{expected_effective_context_tokens!r}"
                )
        expected_runtime_tuple_digest = getattr(assignment, "expected_runtime_tuple_digest", None)
        if isinstance(expected_runtime_tuple_digest, str) and expected_runtime_tuple_digest:
            if runtime_tuple.runtime_tuple_digest != expected_runtime_tuple_digest:
                raise ValueError(
                    "trusted assignment rejected because the local runtime tuple digest "
                    f"{runtime_tuple.runtime_tuple_digest!r} does not match expected {expected_runtime_tuple_digest!r}"
                )
    expected_gguf_file_digest = getattr(assignment, "expected_gguf_file_digest", None)
    if isinstance(expected_gguf_file_digest, str) and expected_gguf_file_digest:
        if runtime_tuple.gguf_file_digest != expected_gguf_file_digest:
            raise ValueError(
                "GGUF assignment rejected because the local GGUF file digest "
                f"{runtime_tuple.gguf_file_digest!r} does not match expected {expected_gguf_file_digest!r}"
            )
    expected_quantization_type = getattr(assignment, "expected_quantization_type", None)
    if isinstance(expected_quantization_type, str) and expected_quantization_type:
        if runtime_tuple.quantization_type != expected_quantization_type:
            raise ValueError(
                "GGUF assignment rejected because the local quantization type "
                f"{runtime_tuple.quantization_type!r} does not match expected {expected_quantization_type!r}"
            )


def validate_assignment_items(assignment: object, items: object) -> list[dict[str, object]]:
    if not isinstance(items, list) or not items:
        raise ValueError("assignment payload must include a non-empty items list")
    if len(items) != getattr(assignment, "item_count", None):
        raise ValueError(
            f"assignment item count mismatch: expected {getattr(assignment, 'item_count', None)}, got {len(items)}"
        )

    validated_items: list[dict[str, object]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"assignment item {index} is not an object")
        if item.get("operation") != getattr(assignment, "operation", None):
            raise ValueError(
                f"assignment item {index} operation {item.get('operation')!r} does not match envelope"
            )
        if item.get("model") != getattr(assignment, "model", None):
            raise ValueError(f"assignment item {index} model {item.get('model')!r} does not match envelope")
        if not isinstance(item.get("batch_item_id"), str) or not item["batch_item_id"]:
            raise ValueError(f"assignment item {index} is missing batch_item_id")
        if not isinstance(item.get("customer_item_id"), str) or not item["customer_item_id"]:
            raise ValueError(f"assignment item {index} is missing customer_item_id")
        if "input" not in item:
            raise ValueError(f"assignment item {index} is missing input")
        validated_items.append(item)
    return validated_items


def validate_assignment(control: EdgeControlClient, assignment: object, items: object) -> list[dict[str, object]]:
    validate_assignment_policy(control, assignment)
    return validate_assignment_items(assignment, items)


class AssignmentProgressKeepalive:
    def __init__(self, control: EdgeControlClient, assignment_id: str, item_count: int) -> None:
        self.control = control
        self.assignment_id = assignment_id
        self.item_count = item_count
        self._auth_error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"assignment-progress-{assignment_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self._auth_error is not None:
            raise self._auth_error

    def _run(self) -> None:
        while not self._stop.wait(assignment_progress_keepalive_seconds):
            try:
                self.control.report_progress(
                    self.assignment_id,
                    assignment_progress("running", self.item_count, keepalive=True),
                )
            except Exception as error:
                if self.control.is_auth_error(error):
                    self._auth_error = error
                    self._stop.set()
                    return
                LOGGER.warning("progress keepalive for %s failed; will retry: %s", self.assignment_id, error)


def summarize_provider_usage(item_results: list[dict[str, object]]) -> dict[str, object]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_texts": 0,
    }
    for item in item_results:
        usage = item.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in list(totals):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return {
        **totals,
        "item_count": len(item_results),
    }


def build_runtime_receipt(
    control: EdgeControlClient,
    assignment: object,
    item_results: list[dict[str, object]],
) -> dict[str, object]:
    assignment_model = getattr(assignment, "model", None)
    assignment_operation = getattr(assignment, "operation", None)
    runtime_tuple = resolved_runtime_tuple(control.settings, assignment_model, assignment_operation)
    gguf_artifact = resolved_gguf_artifact_contract(control.settings, assignment_model, assignment_operation)
    receipt: dict[str, object] = {
        "assignment_nonce": getattr(assignment, "assignment_nonce", ""),
        "declared_model": assignment_model or "",
        "declared_runtime_profile": resolved_runtime_profile_for_settings(control.settings),
        "declared_runtime_engine": resolved_inference_engine_for_settings(control.settings),
        "declared_runtime_image_digest": runtime_tuple.runtime_image_digest,
        "declared_model_manifest_digest": runtime_tuple.model_manifest_digest,
        "declared_tokenizer_digest": runtime_tuple.tokenizer_digest,
        "declared_chat_template_digest": runtime_tuple.chat_template_digest,
        "declared_effective_context_tokens": runtime_tuple.effective_context_tokens,
        "declared_runtime_tuple_digest": runtime_tuple.runtime_tuple_digest,
        "provider_usage_summary": summarize_provider_usage(item_results),
    }
    if gguf_artifact is not None:
        receipt["declared_gguf_artifact"] = gguf_artifact.payload()
    return receipt


def complete_assignment_with_retry(
    control: EdgeControlClient,
    assignment_id: str,
    item_results: list[dict[str, object]],
    runtime_receipt: dict[str, object] | None,
    max_attempts: int = 3,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            control.complete_assignment(assignment_id, item_results, runtime_receipt=runtime_receipt)
            return
        except Exception as error:
            if control.is_auth_error(error) or attempt >= max_attempts or not is_transient_http_error(error):
                raise
            LOGGER.warning(
                "completion acknowledgement for %s failed on attempt %s/%s; retrying",
                assignment_id,
                attempt,
                max_attempts,
            )
            time.sleep(min(attempt, 3))


def report_assignment_failure(
    control: EdgeControlClient,
    assignment: object,
    assignment_error: Exception,
    *,
    item_count: int,
    queue_depth: int,
    results_ready: bool = False,
    latency_seconds: float | None = None,
) -> AssignmentWorkerResult:
    if control.is_auth_error(assignment_error):
        raise assignment_error

    assignment_id = getattr(assignment, "assignment_id", "unknown")
    code, message, retryable = classify_assignment_failure(assignment_error)
    if results_ready and retryable:
        raise assignment_error

    exc_info = (type(assignment_error), assignment_error, assignment_error.__traceback__)
    if results_ready:
        LOGGER.error(
            "assignment %s completion acknowledgement failed after results were computed",
            assignment_id,
            exc_info=exc_info,
        )
    else:
        LOGGER.error("assignment %s failed", assignment_id, exc_info=exc_info)

    try:
        control.report_progress(
            assignment_id,
            assignment_progress("failed", item_count, code=code, retryable=retryable),
        )
    except Exception as progress_error:
        if control.is_auth_error(progress_error):
            raise
        if control.is_transient_network_error(progress_error):
            LOGGER.warning(
                "terminal progress report for %s deferred because control-plane connectivity is degraded: %s",
                assignment_id,
                progress_error,
            )
        else:
            LOGGER.warning("failed to report terminal progress for %s: %s", assignment_id, progress_error)

    try:
        control.fail_assignment(assignment_id, code, message, retryable=retryable)
    except Exception as fail_error:
        if control.is_auth_error(fail_error):
            raise
        if control.is_transient_network_error(fail_error):
            LOGGER.warning(
                "assignment %s failure report deferred because control-plane connectivity is degraded; lease reclaim will handle it: %s",
                assignment_id,
                fail_error,
            )
        else:
            LOGGER.warning("failed to mark assignment %s as failed; stale reclaim will handle it: %s", assignment_id, fail_error)

    return AssignmentWorkerResult(
        assignment_id=assignment_id,
        kind="success" if results_ready else "failure",
        queue_depth=max(1, queue_depth),
        latency_seconds=latency_seconds,
        operation=getattr(assignment, "operation", None),
        model=getattr(assignment, "model", None),
        item_count=max(0, int(item_count)),
        code=code,
        retryable=retryable,
    )


def return_assignment_for_heat_pause(
    control: EdgeControlClient,
    assignment: object,
    *,
    item_count: int,
    queue_depth: int,
    pause_reason: str | None,
) -> AssignmentWorkerResult:
    assignment_id = str(getattr(assignment, "assignment_id", "unknown"))
    message = (
        "Owner heat target paused this node before the assignment started. "
        "The assignment can be retried after the owner resumes heat output."
    )
    try:
        control.report_progress(
            assignment_id,
            assignment_progress(
                "failed",
                item_count,
                error_code="owner_heat_pause",
                retryable=True,
                pause_reason=pause_reason or "heat_governor_paused",
            ),
        )
        control.fail_assignment(assignment_id, "owner_heat_pause", message, retryable=True)
    except Exception as fail_error:
        if control.is_auth_error(fail_error):
            raise
        if control.is_transient_network_error(fail_error):
            LOGGER.warning(
                "assignment %s pause handoff deferred because control-plane connectivity is degraded; lease reclaim will handle it: %s",
                assignment_id,
                fail_error,
            )
        else:
            LOGGER.warning("assignment %s pause handoff failed; stale reclaim will handle it: %s", assignment_id, fail_error)

    return AssignmentWorkerResult(
        assignment_id=assignment_id,
        kind="failure",
        queue_depth=max(1, queue_depth),
        operation=getattr(assignment, "operation", None),
        model=getattr(assignment, "model", None),
        item_count=max(0, int(item_count)),
        code="owner_heat_pause",
        retryable=True,
    )


def return_local_queue_for_heat_pause(
    control: EdgeControlClient,
    leased_assignments: list[object],
    ready_assignments: list[PreparedAssignment],
    *,
    queue_depth: int,
    pause_reason: str | None,
) -> list[AssignmentWorkerResult]:
    worker_results: list[AssignmentWorkerResult] = []
    while ready_assignments:
        prepared_assignment = ready_assignments.pop(0)
        worker_results.append(
            return_assignment_for_heat_pause(
                control,
                prepared_assignment.assignment,
                item_count=prepared_assignment.item_count,
                queue_depth=queue_depth,
                pause_reason=pause_reason,
            )
        )
    while leased_assignments:
        assignment = leased_assignments.pop(0)
        worker_results.append(
            return_assignment_for_heat_pause(
                control,
                assignment,
                item_count=max(0, int(getattr(assignment, "item_count", 0) or 0)),
                queue_depth=queue_depth,
                pause_reason=pause_reason,
            )
        )
    return worker_results


def process_accepted_assignment(
    control: EdgeControlClient,
    runtime: VLLMRuntime,
    assignment: object,
    *,
    queue_depth: int,
) -> AssignmentWorkerResult:
    results_ready = False
    prepared_assignment: PreparedAssignment | object = assignment
    item_count = max(0, int(getattr(assignment, "item_count", 0) or 0))
    latency_seconds: float | None = None
    try:
        prepared_assignment = ensure_prepared_assignment(control, assignment)
        item_count = max(0, int(prepared_assignment.item_count))
        control.report_progress(
            prepared_assignment.assignment_id,
            assignment_progress("running", item_count),
        )
        progress_keepalive = AssignmentProgressKeepalive(
            control,
            prepared_assignment.assignment_id,
            item_count,
        )
        progress_keepalive.start()
        results = []
        runtime_error: Exception | None = None
        runtime_started_at = time.monotonic()
        try:
            results = runtime.execute(
                prepared_assignment.operation,
                prepared_assignment.model,
                prepared_assignment.items,
            )
        except Exception as error:
            runtime_error = error
        try:
            progress_keepalive.stop()
        except Exception as error:
            if runtime_error is None or control.is_auth_error(error):
                runtime_error = error
            else:
                LOGGER.warning(
                    "progress keepalive for %s stopped with an auth error after runtime failure: %s",
                    prepared_assignment.assignment_id,
                    error,
                )
        if runtime_error is not None:
            raise runtime_error
        latency_seconds = time.monotonic() - runtime_started_at
        results_ready = True
        control.report_progress(
            prepared_assignment.assignment_id,
            assignment_progress("completed", item_count),
        )
        usage_summary = summarize_provider_usage(results)
        runtime_receipt = build_runtime_receipt(control, prepared_assignment, results)
        complete_assignment_with_retry(
            control,
            prepared_assignment.assignment_id,
            results,
            runtime_receipt,
        )
        return AssignmentWorkerResult(
            assignment_id=prepared_assignment.assignment_id,
            kind="success",
            queue_depth=max(1, queue_depth),
            latency_seconds=latency_seconds,
            operation=getattr(prepared_assignment, "operation", None),
            model=getattr(prepared_assignment, "model", None),
            item_count=int(usage_summary.get("item_count", item_count) or item_count),
            usage_summary={
                key: int(value) for key, value in usage_summary.items() if isinstance(value, int)
            },
            microbatch_assignments=1,
        )
    except Exception as assignment_error:
        return report_assignment_failure(
            control,
            prepared_assignment,
            assignment_error,
            item_count=item_count,
            queue_depth=queue_depth,
            results_ready=results_ready,
            latency_seconds=latency_seconds,
        )


def ensure_prepared_assignment(control: EdgeControlClient, assignment: object) -> PreparedAssignment:
    if isinstance(assignment, PreparedAssignment):
        return assignment

    item_count = max(0, int(getattr(assignment, "item_count", 0) or 0))
    payload = control.fetch_artifact(assignment)
    items = validate_assignment(control, assignment, payload.get("items", []))
    resolved_item_count = len(items) if items else item_count
    return PreparedAssignment(
        assignment=assignment,
        items=items,
        item_count=resolved_item_count,
        prefetched_at_monotonic=time.monotonic(),
    )


def assignment_microbatch_key(assignment: object) -> str | None:
    if getattr(assignment, "operation", None) != "embeddings":
        return None
    explicit_key = getattr(assignment, "microbatch_key", None)
    if isinstance(explicit_key, str) and explicit_key:
        return explicit_key
    allowed_regions = getattr(assignment, "allowed_regions", None)
    region_key = ""
    if isinstance(allowed_regions, list):
        region_key = ",".join(
            sorted(
                normalize_region_token(region)
                for region in allowed_regions
                if normalize_region_token(region)
            )
        )
    return "|".join(
        str(getattr(assignment, field, "") or "")
        for field in (
            "operation",
            "model",
            "privacy_tier",
            "node_trust_requirement",
            "result_guarantee",
            "expected_runtime_tuple_digest",
            "expected_runtime_image_digest",
            "expected_model_manifest_digest",
            "expected_tokenizer_digest",
            "expected_chat_template_digest",
            "expected_gguf_file_digest",
            "expected_quantization_type",
            "expected_effective_context_tokens",
        )
    ) + f"|regions={region_key}"


def group_assignments_for_local_execution(assignments: list[object]) -> list[list[object]]:
    grouped: dict[str, list[object]] = {}
    ordered_groups: list[list[object]] = []
    for assignment in assignments:
        key = assignment_microbatch_key(assignment)
        if key is None:
            ordered_groups.append([assignment])
            continue
        group = grouped.get(key)
        if group is None:
            group = []
            grouped[key] = group
            ordered_groups.append(group)
        group.append(assignment)
    return ordered_groups


def select_assignment_bundles_for_dispatch(
    pending_assignments: list[object],
    *,
    available_slots: int,
    max_microbatch_assignments: int = 1,
) -> list[list[object]]:
    if available_slots <= 0 or not pending_assignments:
        return []

    remaining = list(pending_assignments)
    bundles: list[list[object]] = []

    while remaining and available_slots > 0:
        grouped = group_assignments_for_local_execution(remaining)
        ranked_groups = sorted(
            enumerate(grouped),
            key=lambda entry: (
                min(len(entry[1]), max(1, max_microbatch_assignments)),
                1 if assignment_microbatch_key(entry[1][0]) is not None else 0,
                -entry[0],
            ),
            reverse=True,
        )
        _group_index, selected_group = ranked_groups[0]
        selected_key = assignment_microbatch_key(selected_group[0])
        bundle_size = (
            1
            if selected_key is None
            else min(len(selected_group), max(1, max_microbatch_assignments))
        )
        bundle = selected_group[:bundle_size]
        bundle_ids = {
            getattr(assignment, "assignment_id", "")
            for assignment in bundle
            if getattr(assignment, "assignment_id", "")
        }
        remaining = [
            assignment
            for assignment in remaining
            if getattr(assignment, "assignment_id", "") not in bundle_ids
        ]
        available_slots -= 1
        bundles.append(bundle)

    pending_assignments[:] = remaining
    return bundles


def local_work_reservoir_depth(
    leased_assignments: list[object],
    ready_assignments: list[PreparedAssignment],
    active_prefetch_workers: dict[str, threading.Thread],
) -> int:
    return len(leased_assignments) + len(ready_assignments) + len(active_prefetch_workers)


def recommended_prefetch_worker_limit(
    *,
    worker_limit: int,
    max_microbatch_assignments: int,
    local_queue_limit: int,
) -> int:
    reservoir_parallelism = max(1, (max(0, local_queue_limit) + 3) // 4)
    return min(
        8,
        max(
            2,
            max(0, worker_limit) * 2,
            min(8, max(1, max_microbatch_assignments)),
            reservoir_parallelism,
        ),
    )


def run_prefetch_assignment_worker(
    control: EdgeControlClient,
    assignment: object,
    *,
    completion_queue: Queue[PrefetchWorkerResult],
) -> None:
    try:
        completion_queue.put(
            PrefetchWorkerResult(
                assignment=assignment,
                prepared_assignment=ensure_prepared_assignment(control, assignment),
            )
        )
    except Exception as error:
        completion_queue.put(PrefetchWorkerResult(assignment=assignment, error=error))


def prefetch_assignments_for_local_queue(
    leased_assignments: list[object],
    *,
    desired_ready_assignments: int,
    max_prefetch_workers: int,
    ready_assignments: list[PreparedAssignment],
    active_prefetch_workers: dict[str, threading.Thread],
    prefetch_completion_queue: Queue[PrefetchWorkerResult],
    control: EdgeControlClient,
) -> None:
    if desired_ready_assignments <= 0 or max_prefetch_workers <= 0:
        return
    while (
        leased_assignments
        and len(active_prefetch_workers) < max_prefetch_workers
        and (len(ready_assignments) + len(active_prefetch_workers)) < max(0, desired_ready_assignments)
    ):
        assignment = leased_assignments.pop(0)
        assignment_id = str(getattr(assignment, "assignment_id", "") or f"prefetch-{time.monotonic_ns()}")
        worker = threading.Thread(
            target=run_prefetch_assignment_worker,
            args=(control, assignment),
            kwargs={"completion_queue": prefetch_completion_queue},
            name=f"prefetch-assignment-{assignment_id}",
            daemon=True,
        )
        active_prefetch_workers[assignment_id] = worker
        worker.start()


def drain_prefetched_assignments_for_local_queue(
    control: EdgeControlClient,
    ready_assignments: list[PreparedAssignment],
    *,
    active_prefetch_workers: dict[str, threading.Thread],
    prefetch_completion_queue: Queue[PrefetchWorkerResult],
    queue_depth: int,
) -> list[AssignmentWorkerResult]:
    worker_results: list[AssignmentWorkerResult] = []
    while True:
        try:
            prefetch_result = prefetch_completion_queue.get_nowait()
        except Empty:
            break
        assignment_id = str(getattr(prefetch_result.assignment, "assignment_id", "") or "")
        worker = active_prefetch_workers.pop(assignment_id, None)
        if worker is not None:
            worker.join()
        if prefetch_result.prepared_assignment is not None and prefetch_result.error is None:
            ready_assignments.append(prefetch_result.prepared_assignment)
            continue
        assignment_error = prefetch_result.error or RuntimeError(
            "prefetch worker completed without a prepared assignment"
        )
        worker_results.append(
            report_assignment_failure(
                control,
                prefetch_result.assignment,
                assignment_error,
                item_count=max(0, int(getattr(prefetch_result.assignment, "item_count", 0) or 0)),
                queue_depth=max(1, queue_depth),
            )
        )
    return worker_results


def process_microbatch_assignments(
    control: EdgeControlClient,
    runtime: VLLMRuntime,
    assignments: list[object],
    *,
    queue_depth: int,
) -> list[AssignmentWorkerResult]:
    prepared: list[PreparedAssignment] = []
    worker_results: list[AssignmentWorkerResult] = []
    microbatch_key = assignment_microbatch_key(assignments[0]) if assignments else None

    for assignment in assignments:
        item_count = max(0, int(getattr(assignment, "item_count", 0) or 0))
        try:
            prepared_assignment = ensure_prepared_assignment(control, assignment)
            item_count = prepared_assignment.item_count
            prepared.append(prepared_assignment)
        except Exception as assignment_error:
            worker_results.append(
                report_assignment_failure(
                    control,
                    assignment,
                    assignment_error,
                    item_count=item_count,
                    queue_depth=queue_depth,
                )
            )

    if not prepared:
        return worker_results

    keepalives: list[AssignmentProgressKeepalive] = []
    runtime_error: Exception | None = None
    runtime_started_at = time.monotonic()
    results_by_assignment: dict[str, list[dict[str, object]]] = {}
    try:
        for prepared_assignment in prepared:
            control.report_progress(
                prepared_assignment.assignment_id,
                assignment_progress("running", prepared_assignment.item_count, microbatch_key=microbatch_key, microbatch_assignments=len(prepared)),
            )
            keepalive = AssignmentProgressKeepalive(
                control,
                prepared_assignment.assignment_id,
                prepared_assignment.item_count,
            )
            keepalive.start()
            keepalives.append(keepalive)
        results_by_assignment = runtime.execute_microbatch(
            "embeddings",
            getattr(assignments[0], "model"),
            [
                (prepared_assignment.assignment_id, prepared_assignment.items)
                for prepared_assignment in prepared
            ],
        )
    except Exception as error:
        runtime_error = error
    finally:
        for keepalive in keepalives:
            try:
                keepalive.stop()
            except Exception as error:
                if runtime_error is None or control.is_auth_error(error):
                    runtime_error = error
                else:
                    LOGGER.warning("progress keepalive stopped after runtime failure: %s", error)

    latency_seconds = time.monotonic() - runtime_started_at
    if runtime_error is not None:
        for prepared_assignment in prepared:
            worker_results.append(
                report_assignment_failure(
                    control,
                    prepared_assignment,
                    runtime_error,
                    item_count=prepared_assignment.item_count,
                    queue_depth=queue_depth,
                )
            )
        return worker_results

    for prepared_assignment in prepared:
        try:
            results = results_by_assignment.get(prepared_assignment.assignment_id)
            if results is None:
                raise RuntimeError("runtime did not return microbatch results for assignment")
            control.report_progress(
                prepared_assignment.assignment_id,
                assignment_progress("completed", prepared_assignment.item_count),
            )
            usage_summary = summarize_provider_usage(results)
            runtime_receipt = build_runtime_receipt(control, prepared_assignment, results)
            complete_assignment_with_retry(
                control,
                prepared_assignment.assignment_id,
                results,
                runtime_receipt,
            )
            worker_results.append(
                AssignmentWorkerResult(
                    assignment_id=prepared_assignment.assignment_id,
                    kind="success",
                    queue_depth=max(1, queue_depth),
                    latency_seconds=latency_seconds,
                    operation=getattr(prepared_assignment, "operation", None),
                    model=getattr(prepared_assignment, "model", None),
                    item_count=int(
                        usage_summary.get("item_count", prepared_assignment.item_count)
                        or prepared_assignment.item_count
                    ),
                    usage_summary={
                        key: int(value) for key, value in usage_summary.items() if isinstance(value, int)
                    },
                    microbatch_assignments=len(prepared),
                )
            )
        except Exception as assignment_error:
            worker_results.append(
                report_assignment_failure(
                    control,
                    prepared_assignment,
                    assignment_error,
                    item_count=prepared_assignment.item_count,
                    queue_depth=queue_depth,
                    results_ready=True,
                    latency_seconds=latency_seconds,
                )
            )
    return worker_results


def process_assignment_bundle(
    control: EdgeControlClient,
    runtime: VLLMRuntime,
    assignments: list[object],
    *,
    queue_depth: int,
) -> list[AssignmentWorkerResult]:
    if len(assignments) > 1 and assignment_microbatch_key(assignments[0]) is not None:
        return process_microbatch_assignments(control, runtime, assignments, queue_depth=queue_depth)
    return [
        process_accepted_assignment(control, runtime, assignment, queue_depth=queue_depth)
        for assignment in assignments
    ]


def run_assignment_worker(
    control: EdgeControlClient,
    runtime: VLLMRuntime,
    assignment: object,
    *,
    queue_depth: int,
    completion_queue: Queue[AssignmentWorkerResult],
) -> None:
    try:
        completion_queue.put(
            process_accepted_assignment(control, runtime, assignment, queue_depth=queue_depth)
        )
    except Exception as error:
        completion_queue.put(
            AssignmentWorkerResult(
                assignment_id=getattr(assignment, "assignment_id", "unknown"),
                kind="fatal",
                queue_depth=max(1, queue_depth),
                error=error,
            )
        )


def run_assignment_bundle_worker(
    control: EdgeControlClient,
    runtime: VLLMRuntime,
    assignments: list[object],
    *,
    queue_depth: int,
    completion_queue: Queue[AssignmentWorkerResult],
) -> None:
    try:
        for result in process_assignment_bundle(control, runtime, assignments, queue_depth=queue_depth):
            completion_queue.put(result)
    except Exception as error:
        fallback_assignment_ids = [
            getattr(assignment, "assignment_id", "unknown") for assignment in assignments
        ] or ["unknown"]
        for assignment_id in fallback_assignment_ids:
            completion_queue.put(
                AssignmentWorkerResult(
                    assignment_id=assignment_id,
                    kind="fatal",
                    queue_depth=max(1, queue_depth),
                    error=error,
                )
            )


def bootstrap_node(control: EdgeControlClient, interactive: bool) -> str:
    print("Starting node bootstrap...")
    node_id, _node_key = control.bootstrap(interactive=interactive)
    LOGGER.info("node enrolled or restored: %s", node_id)
    print("Claim completed. Running node attestation...")
    control.attest()
    LOGGER.info("node attested: %s", node_id)
    print("Node attested and ready for runtime startup.")
    return node_id


def run_worker_loop(control: EdgeControlClient, runtime: VLLMRuntime, attest_on_start: bool = True) -> None:
    node_id, _node_key = control.require_credentials()
    autopilot = AutopilotController(control.settings)
    heartbeat_state = HeartbeatState()
    lease_keepalive_state = LeaseKeepaliveState()
    throughput_logger = NodeThroughputLogger()
    gpu_telemetry_sampler = GPUTelemetrySampler()
    power_limit_controller = NvidiaPowerLimitController(
        enabled=bool(getattr(control.settings, "gpu_power_limit_enabled", True))
    )
    completion_queue: Queue[AssignmentWorkerResult] = Queue()
    prefetch_completion_queue: Queue[PrefetchWorkerResult] = Queue()
    active_bundle_workers: dict[str, ActiveBundleWorker] = {}
    active_assignment_bundle_ids: dict[str, str] = {}
    active_prefetch_workers: dict[str, threading.Thread] = {}
    leased_assignments: list[object] = []
    ready_assignments: list[PreparedAssignment] = []
    LOGGER.info("node enrolled or restored: %s", node_id)
    if attest_on_start:
        print("Refreshing node attestation before entering the worker loop...")
        control.attest()
        LOGGER.info("node attested: %s", node_id)
    print("Node agent is online. Polling the control plane for assignments...")

    while True:
        try:
            loop_gpu_sample = gpu_telemetry_sampler.sample()
            while True:
                try:
                    result = completion_queue.get_nowait()
                except Empty:
                    break
                bundle_id = active_assignment_bundle_ids.pop(result.assignment_id, None)
                if bundle_id is not None:
                    bundle_worker = active_bundle_workers.get(bundle_id)
                    if bundle_worker is not None:
                        bundle_worker.assignment_ids.discard(result.assignment_id)
                        if not bundle_worker.assignment_ids:
                            bundle_worker.thread.join()
                            active_bundle_workers.pop(bundle_id, None)
                if result.kind == "fatal":
                    raise result.error if result.error is not None else RuntimeError("assignment worker failed")
                if result.kind == "success":
                    throughput_logger.observe_result(result)
                    autopilot.observe_assignment_success(
                        latency_seconds=max(0.0, result.latency_seconds or 0.0),
                        queue_depth=result.queue_depth,
                        active_assignments=len(active_bundle_workers),
                        gpu_sample=loop_gpu_sample,
                    )
                else:
                    autopilot.observe_assignment_failure(
                        code=result.code or "node_runtime_error",
                        retryable=bool(result.retryable),
                        queue_depth=result.queue_depth,
                        active_assignments=len(active_bundle_workers),
                        gpu_sample=loop_gpu_sample,
                    )
            prefetch_results = drain_prefetched_assignments_for_local_queue(
                control,
                ready_assignments,
                active_prefetch_workers=active_prefetch_workers,
                prefetch_completion_queue=prefetch_completion_queue,
                queue_depth=max(
                    1,
                    local_work_reservoir_depth(
                        leased_assignments,
                        ready_assignments,
                        active_prefetch_workers,
                    ),
                ),
            )
            for result in prefetch_results:
                autopilot.observe_assignment_failure(
                    code=result.code or "node_runtime_error",
                    retryable=bool(result.retryable),
                    queue_depth=result.queue_depth,
                    active_assignments=len(active_bundle_workers),
                    gpu_sample=loop_gpu_sample,
                )

            if (
                control.settings.restricted_capable
                and control.settings.trust_tier == "restricted"
                and control.settings.attestation_provider == "hardware"
                and not restricted_attestation_is_fresh(control)
            ):
                LOGGER.info("local restricted attestation is stale or missing; refreshing it before polling")
                control.attest()
            queue_depth = control_plane_queue_depth(control)
            active_assignments = len(active_bundle_workers)
            active_running_assignments = len(active_assignment_bundle_ids)
            loop_time = time.monotonic()
            gpu_sample = gpu_telemetry_sampler.sample(now_monotonic=loop_time)
            local_queue_depth = local_work_reservoir_depth(
                leased_assignments,
                ready_assignments,
                active_prefetch_workers,
            )
            observable_queue_depth = max(queue_depth, local_queue_depth)
            autopilot.observe_idle(
                queue_depth=observable_queue_depth,
                active_assignments=active_assignments,
                gpu_sample=gpu_sample,
            )
            capabilities = autopilot.capabilities_payload()
            heat_governor_payload = (
                dict(capabilities.get("heat_governor"))
                if isinstance(capabilities.get("heat_governor"), dict)
                else {}
            )
            target_power_raw = capabilities.get("target_gpu_utilization_pct")
            target_power_pct = 100 if target_power_raw is None else int(target_power_raw)
            power_limits = power_limit_controller.power_limits(now_monotonic=loop_time)
            desired_power_limit_watts = power_limit_for_target(target_power_pct, power_limits)
            if desired_power_limit_watts is not None:
                heat_governor_payload["desired_power_limit_watts"] = desired_power_limit_watts
                capabilities["heat_governor"] = heat_governor_payload
                capabilities["gpu_power_limit_watts"] = desired_power_limit_watts
                if power_limit_controller.apply(desired_power_limit_watts, now_monotonic=loop_time):
                    LOGGER.info("heat governor set GPU power limit to %sW", desired_power_limit_watts)
            if power_limit_controller.last_error:
                capabilities["gpu_power_limit_error"] = power_limit_controller.last_error
            governor_paused = bool(capabilities.get("heat_governor_paused"))
            governor_pause_reason = (
                str(capabilities.get("heat_governor_pause_reason") or "")
                or str(heat_governor_payload.get("pause_reason") or "")
                or None
            )
            if governor_paused and (leased_assignments or ready_assignments):
                returned_results = return_local_queue_for_heat_pause(
                    control,
                    leased_assignments,
                    ready_assignments,
                    queue_depth=max(queue_depth, local_queue_depth),
                    pause_reason=governor_pause_reason,
                )
                LOGGER.info(
                    "heat governor returned %s staged assignments before pause reason=%s",
                    len(returned_results),
                    governor_pause_reason or "heat_governor_paused",
                )
                local_queue_depth = local_work_reservoir_depth(
                    leased_assignments,
                    ready_assignments,
                    active_prefetch_workers,
                )
                observable_queue_depth = max(queue_depth, local_queue_depth)
            worker_limit = max_worker_assignments_from_capabilities(capabilities)
            microbatch_assignment_limit = max_microbatch_assignments_from_capabilities(capabilities)
            local_queue_limit = max_local_queue_assignments_from_capabilities(capabilities)
            connectivity = control_plane_connectivity_snapshot(control)
            reservoir_target = recommended_local_reservoir_target(
                worker_limit=worker_limit,
                local_queue_limit=local_queue_limit,
                connectivity=connectivity,
            )
            throughput_logger.observe_loop(
                active_assignments=active_assignments,
                worker_limit=worker_limit,
                queue_depth=observable_queue_depth,
                prefetch_queue_depth=len(ready_assignments) + len(active_prefetch_workers),
                gpu_sample=gpu_sample,
                now_monotonic=loop_time,
            )
            throughput_logger.maybe_log(now_monotonic=loop_time)
            heartbeat_state = maybe_send_heartbeat(
                queue_depth=local_queue_depth,
                control=control,
                heartbeat_state=heartbeat_state,
                status="paused" if governor_paused else "active",
                active_assignments=active_running_assignments,
                capabilities=capabilities,
                runtime=control.node_runtime_payload(),
                now_monotonic=loop_time,
            )
            pulled_assignment = False
            local_assignment_ids = [
                assignment_id
                for assignment_id in (
                    list(active_assignment_bundle_ids.keys())
                    + list(active_prefetch_workers.keys())
                    + [
                        str(getattr(assignment, "assignment_id", "") or "")
                        for assignment in leased_assignments
                    ]
                    + [
                        str(getattr(assignment, "assignment_id", "") or "")
                        for assignment in ready_assignments
                    ]
                )
                if assignment_id
            ]
            lease_keepalive_state = maybe_touch_assignment_leases(
                control,
                lease_keepalive_state,
                local_assignment_ids,
                now_monotonic=loop_time,
            )
            if governor_paused:
                LOGGER.debug("heat governor pause active; skipping assignment pull and dispatch")
                time.sleep(
                    0.25 if active_bundle_workers or active_prefetch_workers else control.settings.poll_interval_seconds
                )
                continue
            # Keep the local reservoir independent from active runtime
            # concurrency so network fetch, decrypt, and validation stay ahead
            # of the GPU.
            pull_budget = max(0, reservoir_target - local_queue_depth)
            assignments: list[object] = []
            if pull_budget > 0:
                try:
                    assignments = control.pull_assignments(
                        pull_budget,
                        active_assignment_ids=local_assignment_ids,
                    )
                    queue_depth = control_plane_queue_depth(control)
                    connectivity = control_plane_connectivity_snapshot(control)
                except Exception as error:
                    if control.is_transient_network_error(error):
                        LOGGER.warning("control plane connectivity degraded temporarily; pull failed: %s", error)
                        connectivity = control_plane_connectivity_snapshot(control)
                        assignments = []
                    else:
                        raise
            pending_assignment_ids = {
                str(getattr(assignment, "assignment_id", "") or "")
                for assignment in leased_assignments + ready_assignments
            }
            pending_assignment_ids.update(active_prefetch_workers.keys())
            for assignment in assignments[:pull_budget]:
                if (
                    assignment.assignment_id in active_assignment_bundle_ids
                    or assignment.assignment_id in pending_assignment_ids
                ):
                    LOGGER.debug("assignment %s is already running locally; skipping duplicate pull", assignment.assignment_id)
                    continue
                leased_assignments.append(assignment)
                pending_assignment_ids.add(assignment.assignment_id)
                pulled_assignment = True

            prefetch_assignments_for_local_queue(
                leased_assignments,
                desired_ready_assignments=max(1, reservoir_target),
                max_prefetch_workers=recommended_prefetch_worker_limit(
                    worker_limit=worker_limit,
                    max_microbatch_assignments=microbatch_assignment_limit,
                    local_queue_limit=reservoir_target,
                ),
                ready_assignments=ready_assignments,
                active_prefetch_workers=active_prefetch_workers,
                prefetch_completion_queue=prefetch_completion_queue,
                control=control,
            )

            dispatch_slots = max(0, worker_limit - len(active_bundle_workers))
            dispatch_queue_depth = max(
                queue_depth,
                local_work_reservoir_depth(
                    leased_assignments,
                    ready_assignments,
                    active_prefetch_workers,
                ),
            )
            connectivity = control_plane_connectivity_snapshot(control)
            if control_plane_dispatch_allowed(connectivity):
                for bundle in select_assignment_bundles_for_dispatch(
                    ready_assignments,
                    available_slots=dispatch_slots,
                    max_microbatch_assignments=microbatch_assignment_limit,
                ):
                    if not bundle:
                        continue
                    worker = threading.Thread(
                        target=run_assignment_bundle_worker,
                        args=(control, runtime, bundle),
                        kwargs={
                            "queue_depth": max(1, dispatch_queue_depth),
                            "completion_queue": completion_queue,
                        },
                        name=f"assignment-bundle-{getattr(bundle[0], 'assignment_id', 'unknown')}",
                        daemon=True,
                    )
                    bundle_id = str(getattr(bundle[0], "assignment_id", "") or f"bundle-{time.monotonic_ns()}")
                    active_bundle_workers[bundle_id] = ActiveBundleWorker(
                        thread=worker,
                        assignment_ids={
                            str(getattr(assignment, "assignment_id", "unknown")) for assignment in bundle
                        },
                    )
                    for assignment in bundle:
                        active_assignment_bundle_ids[str(getattr(assignment, "assignment_id", "unknown"))] = bundle_id
                    worker.start()
                    LOGGER.info(
                        "claimed assignment bundle size=%s ids=%s",
                        len(bundle),
                        ",".join(str(getattr(assignment, "assignment_id", "unknown")) for assignment in bundle),
                    )
                    dispatch_slots = max(0, dispatch_slots - 1)
            elif ready_assignments:
                LOGGER.info(
                    "holding %s ready assignments until control-plane grace reconnects",
                    len(ready_assignments),
                )
            if (
                not active_bundle_workers
                and not active_prefetch_workers
                and not leased_assignments
                and not ready_assignments
                and not pulled_assignment
            ):
                if max(
                    queue_depth,
                    local_work_reservoir_depth(
                        leased_assignments,
                        ready_assignments,
                        active_prefetch_workers,
                    ),
                ) > 0:
                    autopilot.observe_idle(
                        queue_depth=max(
                            queue_depth,
                            local_work_reservoir_depth(
                                leased_assignments,
                                ready_assignments,
                                active_prefetch_workers,
                            ),
                        ),
                        active_assignments=0,
                    )
                time.sleep(control.settings.poll_interval_seconds)
                continue
            time.sleep(
                0.25 if active_bundle_workers or active_prefetch_workers else control.settings.poll_interval_seconds
            )
        except Exception as error:  # pragma: no cover - long-running loop
            if control.is_auth_error(error):
                LOGGER.warning(
                    "node credentials were rejected by the control plane; clearing local credentials and requiring a new bootstrap"
                )
                control.write_recovery_note(
                    "This node lost control-plane access because its credentials were rejected or revoked. "
                    "Open the setup UI and run Quick Start to re-approve this machine. "
                    "Use `node-agent bootstrap` only for direct terminal debugging."
                )
                control.clear_credentials()
                raise RuntimeError(
                    "Node credentials were rejected by the control plane. "
                    "Open the setup UI and run Quick Start to reclaim this node."
                ) from error
            if control.is_transient_network_error(error):
                LOGGER.warning("control plane connectivity degraded temporarily: %s", error)
                time.sleep(recommended_control_plane_retry_delay_seconds(control))
                continue
            LOGGER.exception("node agent loop failed")
            time.sleep(control.settings.poll_interval_seconds)


def command_bootstrap() -> int:
    settings = NodeAgentSettings()
    validate_startup_settings(settings)
    control = EdgeControlClient(settings)
    bootstrap_node(control, interactive=True)
    return 0


def command_run() -> int:
    settings = NodeAgentSettings()
    validate_startup_settings(settings)
    control = EdgeControlClient(settings)
    runtime = VLLMRuntime(
        resolved_inference_base_url_for_settings(settings),
        engine=resolved_inference_engine_for_settings(settings),
    )
    run_worker_loop(control, runtime, attest_on_start=True)
    return 0


def command_default() -> int:
    settings = NodeAgentSettings()
    validate_startup_settings(settings)
    control = EdgeControlClient(settings)
    runtime = VLLMRuntime(
        resolved_inference_base_url_for_settings(settings),
        engine=resolved_inference_engine_for_settings(settings),
    )

    bootstrapped_now = False
    if not control.has_credentials():
        bootstrap_node(control, interactive=True)
        bootstrapped_now = True

    run_worker_loop(control, runtime, attest_on_start=not bootstrapped_now)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "default"

    if command == "bootstrap":
        return command_bootstrap()
    if command == "run":
        return command_run()
    if command in {"default", "start"}:
        return command_default()
    if command in {"-h", "--help", "help"}:
        print("Usage: node-agent [bootstrap|run]")
        return 0

    raise SystemExit(f"Unknown command: {command}")


def bootstrap_entrypoint() -> None:
    raise SystemExit(main(["bootstrap"]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
