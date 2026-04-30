import logging
from types import SimpleNamespace
import threading
import time
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

import node_agent.main as main_module
from node_agent.concurrency import (
    max_local_queue_assignments_from_capabilities,
    max_worker_assignments_from_capabilities,
)
from node_agent.gguf_artifacts import find_gguf_artifact
from node_agent.model_artifacts import find_model_artifact


class FakeControl:
    def __init__(self, has_credentials: bool = True) -> None:
        self.settings = SimpleNamespace(
            poll_interval_seconds=0,
            node_region="eu-se-1",
            trust_tier="restricted",
            restricted_capable=True,
            node_id="node_123",
            supported_models="meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
            vllm_model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_memory_gb=24.0,
            max_context_tokens=32768,
            max_batch_tokens=50000,
            max_concurrent_assignments=2,
            thermal_headroom=0.8,
            attestation_provider="hardware",
            restricted_attestation_max_age_seconds=3600,
            docker_image="anirdarrazi/autonomousc-ai-edge-runtime@sha256:4662922dd7912bbd928f0703e27472829cacc0a858732a2d48caa167a96561db",
            model_manifest_digest=None,
            tokenizer_digest=None,
            autopilot_state_path=str(Path(tempfile.gettempdir()) / f"autopilot-{id(self)}.json"),
        )
        self._has_credentials = has_credentials
        self.bootstrap_calls = 0
        self.require_calls = 0
        self.attest_calls = 0
        self.clear_calls = 0
        self.recovery_notes = []
        self.auth_fail_on_heartbeat = False
        self.progress_updates = []
        self.touched_assignments = []
        self.failures = []
        self.completions = []
        self.attestation_state = {
            "node_id": "node_123",
            "attestation_provider": "hardware",
            "status": "verified",
            "attested_at": datetime.now(timezone.utc).isoformat(),
        }

    def has_credentials(self) -> bool:
        return self._has_credentials

    def bootstrap(self, interactive: bool = True):
        self.bootstrap_calls += 1
        self._has_credentials = True
        self.settings.node_id = "node_123"
        return "node_123", "key_123"

    def require_credentials(self):
        self.require_calls += 1
        if not self._has_credentials:
            raise RuntimeError("missing credentials")
        self.settings.node_id = "node_123"
        return "node_123", "key_123"

    def attest(self):
        self.attest_calls += 1
        self.attestation_state = {
            "node_id": "node_123",
            "attestation_provider": self.settings.attestation_provider,
            "status": "verified",
            "attested_at": datetime.now(timezone.utc).isoformat(),
        }

    def heartbeat(self, *args, **kwargs):
        if self.auth_fail_on_heartbeat:
            request = httpx.Request("POST", "http://edge.test/nodes/heartbeat")
            response = httpx.Response(401, request=request)
            raise httpx.HTTPStatusError("unauthorized", request=request, response=response)
        raise KeyboardInterrupt()

    def node_runtime_payload(self):
        return {
            "current_model": self.settings.vllm_model,
            "docker_image": self.settings.docker_image,
        }

    def pull_assignment(self):
        return None

    def pull_assignments(self, _limit: int, active_assignment_ids=None):
        assignment = self.pull_assignment()
        return [assignment] if assignment else []

    def accept_assignment(self, _assignment_id: str):
        return None

    def report_progress(self, assignment_id: str, progress):
        self.progress_updates.append((assignment_id, progress))

    def touch_assignments(self, assignment_ids):
        self.touched_assignments.append(list(assignment_ids))

    def complete_assignment(self, assignment_id: str, results, runtime_receipt=None):
        self.completions.append((assignment_id, results, runtime_receipt))

    def fail_assignment(self, assignment_id: str, code: str, message: str, retryable: bool = True):
        self.failures.append((assignment_id, code, message, retryable))

    def clear_credentials(self):
        self.clear_calls += 1
        self._has_credentials = False

    def write_recovery_note(self, message: str):
        self.recovery_notes.append(message)

    def is_auth_error(self, error: Exception) -> bool:
        return isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 401

    def is_transient_network_error(self, error: Exception) -> bool:
        if isinstance(error, main_module.ArtifactFlowError):
            return error.retryable
        if isinstance(error, httpx.TransportError):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {408, 409, 425, 429} or error.response.status_code >= 500
        return False

    def load_attestation_state(self):
        return self.attestation_state


def test_command_bootstrap_runs_claim_bootstrap(monkeypatch: pytest.MonkeyPatch):
    control = FakeControl(has_credentials=False)
    monkeypatch.setattr(
        main_module,
        "NodeAgentSettings",
        lambda: SimpleNamespace(inference_base_url="http://localhost:8000", vllm_base_url="http://localhost:8000"),
    )
    monkeypatch.setattr(main_module, "validate_startup_settings", lambda _settings: None)
    monkeypatch.setattr(main_module, "EdgeControlClient", lambda _settings: control)

    result = main_module.main(["bootstrap"])

    assert result == 0
    assert control.bootstrap_calls == 1
    assert control.attest_calls == 1


def test_command_default_bootstraps_when_credentials_are_missing(monkeypatch: pytest.MonkeyPatch):
    settings = SimpleNamespace(inference_base_url="http://localhost:8000", vllm_base_url="http://localhost:8000")
    control = FakeControl(has_credentials=False)

    monkeypatch.setattr(main_module, "NodeAgentSettings", lambda: settings)
    monkeypatch.setattr(main_module, "validate_startup_settings", lambda _settings: None)
    monkeypatch.setattr(main_module, "EdgeControlClient", lambda _settings: control)
    monkeypatch.setattr(main_module, "VLLMRuntime", lambda _base_url, **_kwargs: object())

    with pytest.raises(KeyboardInterrupt):
        main_module.main([])

    assert control.bootstrap_calls == 1
    assert control.attest_calls == 1


def test_run_worker_loop_clears_credentials_after_auth_failure():
    control = FakeControl(has_credentials=True)
    control.auth_fail_on_heartbeat = True

    with pytest.raises(RuntimeError, match="Open the setup UI and run Quick Start"):
        main_module.run_worker_loop(control, object(), attest_on_start=False)

    assert control.clear_calls == 1
    assert control.recovery_notes
    assert "Open the setup UI and run Quick Start" in control.recovery_notes[-1]


def test_recommended_local_reservoir_target_expands_during_degraded_mode():
    assert (
        main_module.recommended_local_reservoir_target(
            worker_limit=2,
            local_queue_limit=64,
            connectivity={},
        )
        == 16
    )
    assert (
        main_module.recommended_local_reservoir_target(
            worker_limit=2,
            local_queue_limit=64,
            connectivity={"status": "degraded"},
        )
        == 32
    )
    assert (
        main_module.recommended_local_reservoir_target(
            worker_limit=2,
            local_queue_limit=4,
            connectivity={"status": "degraded"},
        )
        == 4
    )


def test_run_worker_loop_reports_assignment_failure():
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pull_calls = 0

        def heartbeat(self, *args, **kwargs):
            return None

        def pull_assignment(self):
            self.pull_calls += 1
            if self.pull_calls > 1:
                if self.failures:
                    raise KeyboardInterrupt()
                return None
            return SimpleNamespace(
                assignment_id="assign_123",
                execution_id="pexec_123",
                item_count=1,
                operation="responses",
                model="meta-llama/Llama-3.1-8B-Instruct",
                privacy_tier="restricted",
                allowed_regions=["eu-se-1"],
                required_vram_gb=16.0,
                required_context_tokens=8192,
                token_budget={"total_tokens": 2048},
            )

        def fetch_artifact(self, _assignment):
            raise ValueError("invalid payload")

    class RuntimeStub:
        def execute(self, _operation, _model, _items):
            raise AssertionError("runtime should not run when payload fetch fails")

    control = AssignmentControl()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, RuntimeStub(), attest_on_start=False)

    assert control.failures == [("assign_123", "invalid_assignment_payload", "invalid payload", False)]
    assert control.progress_updates[-1][1]["state"] == "failed"


def policy_assignment(**overrides):
    values = {
        "assignment_id": "assign_policy",
        "execution_id": "pexec_policy",
        "operation": "embeddings",
        "model": "BAAI/bge-large-en-v1.5",
        "privacy_tier": "standard",
        "allowed_regions": ["eu-se-1"],
        "required_vram_gb": 1.0,
        "required_context_tokens": 512,
        "token_budget": {"total_tokens": 8},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_assignment_policy_accepts_country_code_region_scope():
    control = FakeControl()

    main_module.validate_assignment_policy(control, policy_assignment(allowed_regions=["SE"]))


def test_validate_assignment_policy_rejects_unmatched_country_code_region_scope():
    control = FakeControl()

    with pytest.raises(ValueError, match="node region 'eu-se-1'"):
        main_module.validate_assignment_policy(control, policy_assignment(allowed_regions=["US"]))


def test_validate_assignment_policy_accepts_small_vram_rounding_gap():
    control = FakeControl()
    control.settings.gpu_memory_gb = 11.9

    main_module.validate_assignment_policy(control, policy_assignment(required_vram_gb=12.0))


def test_validate_assignment_policy_rejects_vram_gap_beyond_tolerance():
    control = FakeControl()
    control.settings.gpu_memory_gb = 11.7

    with pytest.raises(ValueError, match="requires 12.0 GiB of VRAM"):
        main_module.validate_assignment_policy(control, policy_assignment(required_vram_gb=12.0))


def test_classify_assignment_failure_detects_gpu_oom():
    error = RuntimeError("CUDA out of memory while allocating attention cache")

    code, message, retryable = main_module.classify_assignment_failure(error)

    assert code == "gpu_oom"
    assert "out of memory" in message.lower()
    assert retryable is True


def test_worker_limit_uses_embedding_specific_capacity():
    assert (
        max_worker_assignments_from_capabilities(
            {
                "max_concurrent_assignments": 1,
                "max_concurrent_assignments_embeddings": 4,
            }
        )
        == 4
    )


def test_worker_limit_ignores_prefetch_depth():
    assert (
        max_worker_assignments_from_capabilities(
            {
                "max_concurrent_assignments": 1,
                "max_concurrent_assignments_embeddings": 2,
                "max_microbatch_assignments_embeddings": 8,
                "max_pull_bundle_assignments": 16,
            }
        )
        == 2
    )


def test_local_queue_limit_uses_prefetch_depth():
    assert (
        max_local_queue_assignments_from_capabilities(
            {
                "max_concurrent_assignments": 1,
                "max_concurrent_assignments_embeddings": 2,
                "max_microbatch_assignments_embeddings": 8,
                "max_pull_bundle_assignments": 16,
            }
        )
        == 16
    )


def test_local_queue_limit_prefers_explicit_local_queue_depth():
    assert (
        max_local_queue_assignments_from_capabilities(
            {
                "max_concurrent_assignments": 1,
                "max_concurrent_assignments_embeddings": 2,
                "max_microbatch_assignments_embeddings": 8,
                "max_pull_bundle_assignments": 16,
                "max_local_queue_assignments": 24,
            }
        )
        == 24
    )


def test_prefetch_assignments_for_local_queue_starts_parallel_reservoir_workers():
    fetch_release = threading.Event()

    class AssignmentControl(FakeControl):
        def fetch_artifact(self, assignment):
            assert fetch_release.wait(1.0)
            return {
                "items": [
                    {
                        "batch_item_id": f"item_{assignment.assignment_id}",
                        "customer_item_id": f"cust_{assignment.assignment_id}",
                        "operation": "embeddings",
                        "model": "BAAI/bge-large-en-v1.5",
                        "input": {"text": assignment.assignment_id},
                    }
                ]
            }

    def assignment(assignment_id: str):
        return SimpleNamespace(
            assignment_id=assignment_id,
            execution_id=f"pexec_{assignment_id}",
            assignment_nonce=f"nonce_{assignment_id}",
            item_count=1,
            operation="embeddings",
            model="BAAI/bge-large-en-v1.5",
            privacy_tier="standard",
            node_trust_requirement="untrusted_allowed",
            result_guarantee="community_best_effort",
            allowed_regions=["global"],
            required_vram_gb=1.0,
            required_context_tokens=512,
            token_budget={"total_tokens": 8},
            microbatch_key="embeddings|BAAI/bge-large-en-v1.5|standard",
        )

    control = AssignmentControl()
    leased = [assignment(f"assign_prefetch_{index}") for index in range(16)]
    ready = []
    active_prefetch_workers = {}
    prefetch_completion_queue = main_module.Queue()

    main_module.prefetch_assignments_for_local_queue(
        leased,
        desired_ready_assignments=16,
        max_prefetch_workers=6,
        ready_assignments=ready,
        active_prefetch_workers=active_prefetch_workers,
        prefetch_completion_queue=prefetch_completion_queue,
        control=control,
    )

    assert len(leased) == 10
    assert len(active_prefetch_workers) == 6
    assert ready == []

    fetch_release.set()
    deadline = time.time() + 2.0
    while any(worker.is_alive() for worker in active_prefetch_workers.values()) and time.time() < deadline:
        time.sleep(0.01)

    results = main_module.drain_prefetched_assignments_for_local_queue(
        control,
        ready,
        active_prefetch_workers=active_prefetch_workers,
        prefetch_completion_queue=prefetch_completion_queue,
        queue_depth=16,
    )

    assert results == []
    assert active_prefetch_workers == {}
    assert len(ready) == 6
    assert sorted(prepared.assignment_id for prepared in ready) == [
        f"assign_prefetch_{index}" for index in range(6)
    ]


def test_drain_prefetched_assignments_reports_failures():
    class AssignmentControl(FakeControl):
        def fetch_artifact(self, _assignment):
            raise ValueError("invalid payload")

    assignment = SimpleNamespace(
        assignment_id="assign_prefetch_fail",
        execution_id="pexec_prefetch_fail",
        assignment_nonce="nonce_prefetch_fail",
        item_count=1,
        operation="embeddings",
        model="BAAI/bge-large-en-v1.5",
        privacy_tier="standard",
        node_trust_requirement="untrusted_allowed",
        result_guarantee="community_best_effort",
        allowed_regions=["global"],
        required_vram_gb=1.0,
        required_context_tokens=512,
        token_budget={"total_tokens": 8},
        microbatch_key="embeddings|BAAI/bge-large-en-v1.5|standard",
    )
    control = AssignmentControl()
    leased = [assignment]
    ready = []
    active_prefetch_workers = {}
    prefetch_completion_queue = main_module.Queue()

    main_module.prefetch_assignments_for_local_queue(
        leased,
        desired_ready_assignments=1,
        max_prefetch_workers=1,
        ready_assignments=ready,
        active_prefetch_workers=active_prefetch_workers,
        prefetch_completion_queue=prefetch_completion_queue,
        control=control,
    )

    deadline = time.time() + 2.0
    while any(worker.is_alive() for worker in active_prefetch_workers.values()) and time.time() < deadline:
        time.sleep(0.01)

    results = main_module.drain_prefetched_assignments_for_local_queue(
        control,
        ready,
        active_prefetch_workers=active_prefetch_workers,
        prefetch_completion_queue=prefetch_completion_queue,
        queue_depth=1,
    )

    assert ready == []
    assert len(results) == 1
    assert results[0].assignment_id == "assign_prefetch_fail"
    assert results[0].code == "invalid_assignment_payload"
    assert results[0].retryable is False
    assert control.failures == [
        ("assign_prefetch_fail", "invalid_assignment_payload", "invalid payload", False)
    ]


def test_select_assignment_bundles_for_dispatch_prefers_largest_compatible_group():
    assignments = [
        SimpleNamespace(assignment_id="assign_single", operation="responses", microbatch_key=None),
        SimpleNamespace(assignment_id="assign_embed_1", operation="embeddings", microbatch_key="embed|A"),
        SimpleNamespace(assignment_id="assign_embed_2", operation="embeddings", microbatch_key="embed|A"),
        SimpleNamespace(assignment_id="assign_embed_3", operation="embeddings", microbatch_key="embed|B"),
    ]

    bundles = main_module.select_assignment_bundles_for_dispatch(
        assignments,
        available_slots=2,
        max_microbatch_assignments=2,
    )

    assert [[assignment.assignment_id for assignment in bundle] for bundle in bundles] == [
        ["assign_embed_1", "assign_embed_2"],
        ["assign_embed_3"],
    ]
    assert [assignment.assignment_id for assignment in assignments] == ["assign_single"]


def test_select_assignment_bundles_can_fill_one_worker_slot_with_large_microbatch():
    assignments = [
        SimpleNamespace(assignment_id=f"assign_embed_{index}", operation="embeddings", microbatch_key="embed|A")
        for index in range(4)
    ]

    bundles = main_module.select_assignment_bundles_for_dispatch(
        assignments,
        available_slots=1,
        max_microbatch_assignments=8,
    )

    assert [[assignment.assignment_id for assignment in bundle] for bundle in bundles] == [
        ["assign_embed_0", "assign_embed_1", "assign_embed_2", "assign_embed_3"],
    ]
    assert assignments == []


def test_process_assignment_bundle_microbatches_compatible_embeddings():
    class AssignmentControl(FakeControl):
        def fetch_artifact(self, assignment):
            return {
                "items": [
                    {
                        "batch_item_id": f"item_{assignment.assignment_id}",
                        "customer_item_id": f"cust_{assignment.assignment_id}",
                        "operation": "embeddings",
                        "model": "BAAI/bge-large-en-v1.5",
                        "input": {"text": assignment.assignment_id},
                    }
                ]
            }

    class RuntimeStub:
        def __init__(self) -> None:
            self.calls = []

        def execute_microbatch(self, operation, model, assignment_items):
            self.calls.append(
                (
                    operation,
                    model,
                    [(assignment_id, [item["batch_item_id"] for item in items]) for assignment_id, items in assignment_items],
                )
            )
            return {
                assignment_id: [
                    {
                        "batch_item_id": items[0]["batch_item_id"],
                        "customer_item_id": items[0]["customer_item_id"],
                        "provider": "autonomousc_edge",
                        "provider_model": model,
                        "status": "completed",
                        "usage": {"input_texts": 1, "total_tokens": 1},
                        "cost": {
                            "provider_cost": {"amount": "0.0001", "currency": "usd"},
                            "customer_charge": {"amount": "0.0002", "currency": "usd"},
                            "platform_margin": {"amount": "0.0001", "currency": "usd"},
                        },
                        "output": {"data": [{"embedding": [1.0]}]},
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                ]
                for assignment_id, items in assignment_items
            }

    def assignment(assignment_id: str):
        return SimpleNamespace(
            assignment_id=assignment_id,
            execution_id=f"pexec_{assignment_id}",
            assignment_nonce=f"nonce_{assignment_id}",
            item_count=1,
            operation="embeddings",
            model="BAAI/bge-large-en-v1.5",
            privacy_tier="standard",
            node_trust_requirement="untrusted_allowed",
            result_guarantee="community_best_effort",
            allowed_regions=["global"],
            required_vram_gb=1.0,
            required_context_tokens=512,
            token_budget={"total_tokens": 8},
            microbatch_key="embeddings|BAAI/bge-large-en-v1.5|standard",
        )

    control = AssignmentControl()
    runtime = RuntimeStub()
    assignments = [assignment("assign_1"), assignment("assign_2")]

    results = main_module.process_assignment_bundle(control, runtime, assignments, queue_depth=2)

    assert runtime.calls == [
        (
            "embeddings",
            "BAAI/bge-large-en-v1.5",
            [("assign_1", ["item_assign_1"]), ("assign_2", ["item_assign_2"])],
        )
    ]
    assert [result.kind for result in results] == ["success", "success"]
    assert all(result.operation == "embeddings" for result in results)
    assert all(result.model == "BAAI/bge-large-en-v1.5" for result in results)
    assert all(result.usage_summary == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 1, "input_texts": 1, "item_count": 1} for result in results)
    assert all(result.microbatch_assignments == 2 for result in results)
    assert [completion[0] for completion in control.completions] == ["assign_1", "assign_2"]
    running_updates = [progress for _assignment_id, progress in control.progress_updates if progress["state"] == "running"]
    assert running_updates
    assert all(progress["microbatch_assignments"] == 2 for progress in running_updates)


def test_node_throughput_logger_logs_embedding_and_response_metrics(caplog: pytest.LogCaptureFixture):
    throughput_logger = main_module.NodeThroughputLogger(interval_seconds=30.0)
    start_time = time.monotonic()

    with caplog.at_level(logging.INFO, logger="autonomousc-node-agent"):
        throughput_logger.observe_loop(
            active_assignments=3,
            worker_limit=4,
            queue_depth=8,
            prefetch_queue_depth=5,
            gpu_sample=main_module.GPUTelemetrySample(
                utilization_percent=82.0,
                memory_utilization_percent=64.0,
                power_watts=210.0,
                temperature_c=69.0,
                source="nvidia-smi",
            ),
            now_monotonic=start_time,
        )
        throughput_logger.observe_loop(
            active_assignments=4,
            worker_limit=4,
            queue_depth=6,
            prefetch_queue_depth=4,
            gpu_sample=main_module.GPUTelemetrySample(
                utilization_percent=94.0,
                memory_utilization_percent=71.0,
                power_watts=235.0,
                temperature_c=72.0,
                source="nvidia-smi",
            ),
            now_monotonic=start_time + 10.0,
        )
        throughput_logger.observe_result(
            main_module.AssignmentWorkerResult(
                assignment_id="assign_embed",
                kind="success",
                queue_depth=6,
                latency_seconds=3.0,
                operation="embeddings",
                model="BAAI/bge-large-en-v1.5",
                item_count=2,
                usage_summary={
                    "input_texts": 4,
                    "input_tokens": 120,
                    "total_tokens": 120,
                    "item_count": 2,
                },
                microbatch_assignments=2,
            )
        )
        throughput_logger.observe_result(
            main_module.AssignmentWorkerResult(
                assignment_id="assign_text",
                kind="success",
                queue_depth=6,
                latency_seconds=1.5,
                operation="responses",
                model="meta-llama/Llama-3.1-8B-Instruct",
                item_count=1,
                usage_summary={
                    "input_tokens": 30,
                    "output_tokens": 10,
                    "total_tokens": 40,
                    "item_count": 1,
                },
                microbatch_assignments=1,
            )
        )
        throughput_logger.maybe_log(now_monotonic=start_time + 31.0)

    messages = [record.getMessage() for record in caplog.records if "node throughput" in record.getMessage()]

    assert any("node throughput summary" in message for message in messages)
    assert any("slot_utilization_avg=0.88" in message for message in messages)
    assert any(
        "gpu_source=nvidia-smi" in message
        and "gpu_util_avg=88.0%" in message
        and "gpu_mem_avg=67.5%" in message
        and "gpu_power_avg=222.5W" in message
        and "prefetch_queue_avg=4.50" in message
        and "queued_idle_s=0.00" in message
        and "microbatch_avg=1.50" in message
        for message in messages
    )
    assert any(
        "op=embeddings" in message
        and "texts_per_s=0.13" in message
        and "input_tokens_per_s=3.9" in message
        and "avg_microbatch_assignments=2.00" in message
        for message in messages
    )
    assert any(
        "op=responses" in message
        and "output_tokens_per_s=0.3" in message
        and "avg_microbatch_assignments=1.00" in message
        for message in messages
    )


def test_node_throughput_logger_warns_when_queued_work_is_idle(caplog: pytest.LogCaptureFixture):
    throughput_logger = main_module.NodeThroughputLogger(interval_seconds=30.0)
    start_time = time.monotonic()

    with caplog.at_level(logging.WARNING, logger="autonomousc-node-agent"):
        throughput_logger.observe_loop(
            active_assignments=0,
            worker_limit=4,
            queue_depth=6,
            prefetch_queue_depth=0,
            gpu_sample=main_module.GPUTelemetrySample(
                utilization_percent=4.0,
                memory_utilization_percent=21.0,
                power_watts=55.0,
                temperature_c=48.0,
                source="nvml",
            ),
            now_monotonic=start_time,
        )
        throughput_logger.observe_loop(
            active_assignments=0,
            worker_limit=4,
            queue_depth=6,
            prefetch_queue_depth=0,
            gpu_sample=main_module.GPUTelemetrySample(
                utilization_percent=6.0,
                memory_utilization_percent=22.0,
                power_watts=57.0,
                temperature_c=49.0,
                source="nvml",
            ),
            now_monotonic=start_time + 20.0,
        )
        throughput_logger.maybe_log(now_monotonic=start_time + 31.0)

    messages = [record.getMessage() for record in caplog.records]
    assert any("condition=queued_capable_idle_or_low_util" in message for message in messages)


def test_validate_startup_settings_rejects_invalid_region(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main_module.socket, "getaddrinfo", lambda *_args, **_kwargs: [object()])
    settings = main_module.NodeAgentSettings(
        edge_control_url="https://edge.autonomousc.com",
        node_region="stockholm",
        vllm_model="BAAI/bge-large-en-v1.5",
        supported_models="BAAI/bge-large-en-v1.5",
        max_context_tokens=512,
    )

    with pytest.raises(RuntimeError, match="NODE_REGION='stockholm'"):
        main_module.validate_startup_settings(settings)


def test_validate_startup_settings_accepts_global_marketplace_region(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main_module.socket, "getaddrinfo", lambda *_args, **_kwargs: [object()])
    settings = main_module.NodeAgentSettings(
        edge_control_url="https://edge.autonomousc.com",
        node_region="global",
        runtime_profile="rtx_5060_ti_16gb_gemma4_e4b",
        inference_engine="vllm",
        vllm_model="google/gemma-4-E4B-it",
        supported_models="google/gemma-4-E4B-it",
        max_context_tokens=32768,
    )

    main_module.validate_startup_settings(settings)


def test_validate_startup_settings_rejects_unsafe_embedding_context(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main_module.socket, "getaddrinfo", lambda *_args, **_kwargs: [object()])
    settings = main_module.NodeAgentSettings(
        edge_control_url="https://edge.autonomousc.com",
        node_region="eu-se-1",
        vllm_model="BAAI/bge-large-en-v1.5",
        supported_models="BAAI/bge-large-en-v1.5",
        max_context_tokens=32768,
    )

    with pytest.raises(RuntimeError, match="MAX_CONTEXT_TOKENS=32768"):
        main_module.validate_startup_settings(settings)


def test_validate_startup_settings_rejects_capability_ads_runtime_cannot_serve(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main_module.socket, "getaddrinfo", lambda *_args, **_kwargs: [object()])
    settings = main_module.NodeAgentSettings(
        edge_control_url="https://edge.autonomousc.com",
        node_region="eu-se-1",
        runtime_profile="home_llama_cpp_gguf",
        inference_engine="llama_cpp",
        vllm_model="meta-llama/Llama-3.1-8B-Instruct",
        supported_models="meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5",
        max_context_tokens=32768,
    )

    with pytest.raises(RuntimeError, match="cannot actually serve"):
        main_module.validate_startup_settings(settings)


def test_gpu_telemetry_sampler_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch):
    sampler = main_module.GPUTelemetrySampler(refresh_interval_seconds=5.0)

    monkeypatch.setattr(main_module.shutil, "which", lambda name: "nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(
        main_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="87, 10240, 16384, 218.50, 71\n"),
    )
    monkeypatch.setattr(main_module, "pynvml", None)

    sample = sampler.sample(now_monotonic=100.0)

    assert sample is not None
    assert sample.source == "nvidia-smi"
    assert sample.utilization_percent == 87.0
    assert sample.memory_utilization_percent == pytest.approx(62.5)
    assert sample.power_watts == pytest.approx(218.5)
    assert sample.temperature_c == pytest.approx(71.0)


def test_run_worker_loop_prefetches_local_queue_and_drains_it_in_microbatches(
    monkeypatch: pytest.MonkeyPatch,
):
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pull_limits = []
            self.batch_returned = False

        def heartbeat(self, *args, **kwargs):
            return None

        def pull_assignments(self, limit: int, active_assignment_ids=None):
            self.pull_limits.append(limit)
            if len(self.completions) >= 4:
                raise KeyboardInterrupt()
            if self.batch_returned:
                return []
            self.batch_returned = True
            return [
                SimpleNamespace(
                    assignment_id=f"assign_prefetch_{index}",
                    execution_id=f"pexec_prefetch_{index}",
                    assignment_nonce=f"nonce_prefetch_{index}",
                    item_count=1,
                    operation="embeddings",
                    model="BAAI/bge-large-en-v1.5",
                    privacy_tier="standard",
                    node_trust_requirement="untrusted_allowed",
                    result_guarantee="community_best_effort",
                    allowed_regions=["global"],
                    required_vram_gb=1.0,
                    required_context_tokens=512,
                    token_budget={"total_tokens": 8},
                    microbatch_key=None,
                )
                for index in range(4)
            ]

        def fetch_artifact(self, assignment):
            return {
                "items": [
                    {
                        "batch_item_id": f"item_{assignment.assignment_id}",
                        "customer_item_id": f"cust_{assignment.assignment_id}",
                        "operation": "embeddings",
                        "model": "BAAI/bge-large-en-v1.5",
                        "input": {"text": assignment.assignment_id},
                    }
                ]
            }

    class RuntimeStub:
        def __init__(self) -> None:
            self.calls = []

        def execute_microbatch(self, operation, model, assignment_items):
            self.calls.append([assignment_id for assignment_id, _items in assignment_items])
            return {
                assignment_id: [
                    {
                        "batch_item_id": items[0]["batch_item_id"],
                        "customer_item_id": items[0]["customer_item_id"],
                        "provider": "autonomousc_edge",
                        "provider_model": model,
                        "status": "completed",
                        "usage": {"input_texts": 1, "total_tokens": 1},
                        "cost": {
                            "provider_cost": {"amount": "0.0001", "currency": "usd"},
                            "customer_charge": {"amount": "0.0002", "currency": "usd"},
                            "platform_margin": {"amount": "0.0001", "currency": "usd"},
                        },
                        "output": {"data": [{"embedding": [1.0]}]},
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                ]
                for assignment_id, items in assignment_items
            }

    monkeypatch.setattr(main_module, "max_worker_assignments_from_capabilities", lambda _capabilities: 2)
    monkeypatch.setattr(main_module, "max_microbatch_assignments_from_capabilities", lambda _capabilities: 4)
    monkeypatch.setattr(main_module, "max_local_queue_assignments_from_capabilities", lambda _capabilities: 4)
    monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)

    control = AssignmentControl()
    runtime = RuntimeStub()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, runtime, attest_on_start=False)

    assert control.pull_limits[0] == 4
    assert runtime.calls == [["assign_prefetch_0", "assign_prefetch_1", "assign_prefetch_2", "assign_prefetch_3"]]
    assert [completion[0] for completion in control.completions] == [
        "assign_prefetch_0",
        "assign_prefetch_1",
        "assign_prefetch_2",
        "assign_prefetch_3",
    ]


def test_run_worker_loop_prefetches_next_batch_while_microbatch_is_active(
    monkeypatch: pytest.MonkeyPatch,
):
    second_batch_prefetched = threading.Event()

    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pull_limits = []
            self.batch_index = 0
            self.second_prefetch_count = 0

        def heartbeat(self, *args, **kwargs):
            return None

        def pull_assignments(self, limit: int, active_assignment_ids=None):
            self.pull_limits.append((limit, list(active_assignment_ids or [])))
            if len(self.completions) >= 4 and second_batch_prefetched.is_set():
                raise KeyboardInterrupt()
            if limit <= 0:
                return []
            if self.batch_index >= 2:
                return []
            prefix = "assign_active" if self.batch_index == 0 else "assign_next"
            self.batch_index += 1
            return [
                SimpleNamespace(
                    assignment_id=f"{prefix}_{index}",
                    execution_id=f"pexec_{prefix}_{index}",
                    assignment_nonce=f"nonce_{prefix}_{index}",
                    item_count=1,
                    operation="embeddings",
                    model="BAAI/bge-large-en-v1.5",
                    privacy_tier="standard",
                    node_trust_requirement="untrusted_allowed",
                    result_guarantee="community_best_effort",
                    allowed_regions=["global"],
                    required_vram_gb=1.0,
                    required_context_tokens=512,
                    token_budget={"total_tokens": 8},
                    microbatch_key=None,
                )
                for index in range(4)
            ]

        def fetch_artifact(self, assignment):
            if str(assignment.assignment_id).startswith("assign_next_"):
                self.second_prefetch_count += 1
                if self.second_prefetch_count >= 4:
                    second_batch_prefetched.set()
            return {
                "items": [
                    {
                        "batch_item_id": f"item_{assignment.assignment_id}",
                        "customer_item_id": f"cust_{assignment.assignment_id}",
                        "operation": "embeddings",
                        "model": "BAAI/bge-large-en-v1.5",
                        "input": {"text": assignment.assignment_id},
                    }
                ]
            }

    class RuntimeStub:
        def __init__(self) -> None:
            self.calls = []

        def execute_microbatch(self, operation, model, assignment_items):
            assignment_ids = [assignment_id for assignment_id, _items in assignment_items]
            self.calls.append(assignment_ids)
            if assignment_ids and assignment_ids[0].startswith("assign_active_"):
                assert second_batch_prefetched.wait(2.0)
            return {
                assignment_id: [
                    {
                        "batch_item_id": items[0]["batch_item_id"],
                        "customer_item_id": items[0]["customer_item_id"],
                        "provider": "autonomousc_edge",
                        "provider_model": model,
                        "status": "completed",
                        "usage": {"input_texts": 1, "total_tokens": 1},
                        "cost": {
                            "provider_cost": {"amount": "0.0001", "currency": "usd"},
                            "customer_charge": {"amount": "0.0002", "currency": "usd"},
                            "platform_margin": {"amount": "0.0001", "currency": "usd"},
                        },
                        "output": {"data": [{"embedding": [1.0]}]},
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                ]
                for assignment_id, items in assignment_items
            }

    monkeypatch.setattr(main_module, "max_worker_assignments_from_capabilities", lambda _capabilities: 1)
    monkeypatch.setattr(main_module, "max_microbatch_assignments_from_capabilities", lambda _capabilities: 4)
    monkeypatch.setattr(main_module, "max_local_queue_assignments_from_capabilities", lambda _capabilities: 4)
    monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)

    control = AssignmentControl()
    runtime = RuntimeStub()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, runtime, attest_on_start=False)

    assert second_batch_prefetched.is_set()
    assert runtime.calls[0] == [
        "assign_active_0",
        "assign_active_1",
        "assign_active_2",
        "assign_active_3",
    ]
    assert any(limit == 4 for limit, _active_ids in control.pull_limits[1:])
    second_pull = next(
        active_ids
        for limit, active_ids in control.pull_limits[1:]
        if limit == 4
    )
    assert "assign_active_0" in second_pull


def test_run_worker_loop_marks_nonretryable_completion_failures_as_failed():
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pull_calls = 0

        def heartbeat(self, *args, **kwargs):
            return None

        def pull_assignment(self):
            self.pull_calls += 1
            if self.pull_calls > 1:
                if self.failures:
                    raise KeyboardInterrupt()
                return None
            return SimpleNamespace(
                assignment_id="assign_complete_413",
                execution_id="pexec_complete_413",
                item_count=1,
                operation="responses",
                model="meta-llama/Llama-3.1-8B-Instruct",
                privacy_tier="restricted",
                allowed_regions=["eu-se-1"],
                required_vram_gb=16.0,
                required_context_tokens=8192,
                token_budget={"total_tokens": 2048},
            )

        def fetch_artifact(self, _assignment):
            return {
                "items": [
                    {
                        "batch_item_id": "item_1",
                        "customer_item_id": "cust_1",
                        "operation": "responses",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "input": {"messages": [{"role": "user", "content": "hello"}]},
                    }
                ]
            }

        def complete_assignment(self, assignment_id: str, results, runtime_receipt=None):
            request = httpx.Request("POST", f"http://edge.test/nodes/assignments/{assignment_id}/complete")
            response = httpx.Response(413, request=request)
            raise httpx.HTTPStatusError("too large", request=request, response=response)

    class RuntimeStub:
        def execute(self, _operation, _model, _items):
            return [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "provider": "autonomousc_edge",
                    "provider_model": "meta-llama/Llama-3.1-8B-Instruct",
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    "cost": {
                        "provider_cost": {"amount": "0.0001", "currency": "usd"},
                        "customer_charge": {"amount": "0.0002", "currency": "usd"},
                        "platform_margin": {"amount": "0.0001", "currency": "usd"},
                    },
                    "output": {"text": "hello"},
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            ]

    control = AssignmentControl()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, RuntimeStub(), attest_on_start=False)

    assert control.failures == [
        (
            "assign_complete_413",
            "upstream_rejected",
            "http://edge.test/nodes/assignments/assign_complete_413/complete returned HTTP 413.",
            False,
        )
    ]
    assert control.progress_updates[-1][1]["state"] == "failed"


def test_complete_assignment_with_retry_retries_retryable_artifact_flow_error(
    monkeypatch: pytest.MonkeyPatch,
):
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.attempts = 0

        def complete_assignment(self, assignment_id: str, results, runtime_receipt=None):
            self.attempts += 1
            if self.attempts == 1:
                raise main_module.ArtifactFlowError(
                    "result_artifact_upload_network_error",
                    "temporary failure in name resolution",
                    retryable=True,
                )
            super().complete_assignment(assignment_id, results, runtime_receipt=runtime_receipt)

    monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)
    control = AssignmentControl()

    main_module.complete_assignment_with_retry(
        control,
        "assign_retry_artifact",
        [{"batch_item_id": "item_1"}],
        {"assignment_nonce": "nonce_1"},
        max_attempts=2,
    )

    assert control.attempts == 2
    assert control.completions == [
        (
            "assign_retry_artifact",
            [{"batch_item_id": "item_1"}],
            {"assignment_nonce": "nonce_1"},
        )
    ]


def test_report_assignment_failure_classifies_transient_reporting_errors(
    caplog: pytest.LogCaptureFixture,
):
    class AssignmentControl(FakeControl):
        def fail_assignment(self, assignment_id: str, code: str, message: str, retryable: bool = True):
            raise httpx.ConnectError("temporary failure in name resolution")

    control = AssignmentControl()
    assignment = SimpleNamespace(
        assignment_id="assign_report_dns",
        operation="embeddings",
        model="BAAI/bge-large-en-v1.5",
    )

    with caplog.at_level(logging.WARNING):
        result = main_module.report_assignment_failure(
            control,
            assignment,
            ValueError("invalid payload"),
            item_count=1,
            queue_depth=3,
        )

    assert result.kind == "failure"
    assert result.code == "invalid_assignment_payload"
    assert result.retryable is False
    assert control.progress_updates[-1][1]["state"] == "failed"
    assert any("connectivity is degraded" in record.getMessage() for record in caplog.records)


def test_run_worker_loop_treats_heartbeat_dns_failure_as_transient(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.heartbeat_calls = 0

        def heartbeat(self, *args, **kwargs):
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 1:
                raise httpx.ConnectError("temporary failure in name resolution")
            raise KeyboardInterrupt()

        def pull_assignments(self, _limit: int, active_assignment_ids=None):
            return []

    monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)
    control = AssignmentControl()

    with caplog.at_level(logging.WARNING, logger="autonomousc-node-agent"):
        with pytest.raises(KeyboardInterrupt):
            main_module.run_worker_loop(control, object(), attest_on_start=False)

    messages = [record.getMessage() for record in caplog.records]
    assert control.heartbeat_calls == 2
    assert any("control plane connectivity degraded temporarily" in message for message in messages)
    assert not any("node agent loop failed" in message for message in messages)


def test_run_worker_loop_treats_pull_dns_failure_as_transient(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pull_calls = 0

        def heartbeat(self, *args, **kwargs):
            return None

        def pull_assignments(self, _limit: int, active_assignment_ids=None):
            self.pull_calls += 1
            if self.pull_calls == 1:
                raise httpx.ConnectError("temporary failure in name resolution")
            raise KeyboardInterrupt()

    monkeypatch.setattr(main_module.time, "sleep", lambda _seconds: None)
    control = AssignmentControl()

    with caplog.at_level(logging.WARNING, logger="autonomousc-node-agent"):
        with pytest.raises(KeyboardInterrupt):
            main_module.run_worker_loop(control, object(), attest_on_start=False)

    messages = [record.getMessage() for record in caplog.records]
    assert control.pull_calls == 2
    assert any("control plane connectivity degraded temporarily" in message for message in messages)
    assert not any("node agent loop failed" in message for message in messages)


def test_run_worker_loop_does_not_pull_when_heat_governor_is_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class PausedControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.settings.heat_governor_mode = "0"
            self.settings.heat_governor_state_path = str(tmp_path / "heat-governor-state.json")
            self.pull_calls = 0

        def heartbeat(self, *args, **kwargs):
            return None

        def pull_assignments(self, _limit: int, active_assignment_ids=None):
            self.pull_calls += 1
            return []

    def stop_after_pause_sleep(_seconds: float) -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_module.time, "sleep", stop_after_pause_sleep)
    control = PausedControl()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, object(), attest_on_start=False)

    assert control.pull_calls == 0


def test_heat_pause_returns_staged_assignments_retryably():
    control = FakeControl(has_credentials=True)
    leased = [
        SimpleNamespace(
            assignment_id="assign_leased",
            item_count=1,
            operation="responses",
            model="meta-llama/Llama-3.1-8B-Instruct",
        )
    ]
    ready = [
        main_module.PreparedAssignment(
            assignment=SimpleNamespace(
                assignment_id="assign_ready",
                item_count=2,
                operation="embeddings",
                model="BAAI/bge-large-en-v1.5",
            ),
            items=[],
            item_count=2,
            prefetched_at_monotonic=100.0,
        )
    ]

    results = main_module.return_local_queue_for_heat_pause(
        control,
        leased,
        ready,
        queue_depth=2,
        pause_reason="owner_heat_target_zero",
    )

    assert leased == []
    assert ready == []
    assert [result.assignment_id for result in results] == ["assign_ready", "assign_leased"]
    assert all(result.code == "owner_heat_pause" and result.retryable for result in results)
    assert control.failures == [
        (
            "assign_ready",
            "owner_heat_pause",
            "Owner heat target paused this node before the assignment started. The assignment can be retried after the owner resumes heat output.",
            True,
        ),
        (
            "assign_leased",
            "owner_heat_pause",
            "Owner heat target paused this node before the assignment started. The assignment can be retried after the owner resumes heat output.",
            True,
        ),
    ]


def test_run_worker_loop_keeps_assignments_fresh_while_runtime_is_busy(monkeypatch: pytest.MonkeyPatch):
    class AssignmentControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.pull_calls = 0

        def heartbeat(self, *args, **kwargs):
            return None

        def pull_assignment(self):
            self.pull_calls += 1
            if self.pull_calls > 1:
                if any(progress["state"] == "completed" for _assignment_id, progress in self.progress_updates):
                    raise KeyboardInterrupt()
                return None
            return SimpleNamespace(
                assignment_id="assign_keepalive",
                execution_id="pexec_keepalive",
                item_count=1,
                operation="responses",
                model="meta-llama/Llama-3.1-8B-Instruct",
                privacy_tier="restricted",
                allowed_regions=["eu-se-1"],
                required_vram_gb=16.0,
                required_context_tokens=8192,
                token_budget={"total_tokens": 2048},
            )

        def fetch_artifact(self, _assignment):
            return {
                "items": [
                    {
                        "batch_item_id": "item_1",
                        "customer_item_id": "cust_1",
                        "operation": "responses",
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "input": {"messages": [{"role": "user", "content": "hello"}]},
                    }
                ]
            }

    class SlowRuntime:
        def execute(self, _operation, _model, _items):
            time.sleep(0.05)
            return [{"status": "completed"}]

    control = AssignmentControl()
    monkeypatch.setattr(main_module, "assignment_progress_keepalive_seconds", 0.01)

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, SlowRuntime(), attest_on_start=False)

    running_updates = [progress for _assignment_id, progress in control.progress_updates if progress["state"] == "running"]
    assert len(running_updates) >= 2
    assert any(progress.get("keepalive") is True for progress in running_updates[1:])
    assert control.progress_updates[-1][1]["state"] == "completed"


def test_maybe_send_heartbeat_throttles_idle_updates_and_flushes_slot_changes():
    class HeartbeatRecordingControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.heartbeat_calls = []

        def heartbeat(self, *args, **kwargs):
            self.heartbeat_calls.append((args, kwargs))
            return None

    control = HeartbeatRecordingControl()
    state = main_module.HeartbeatState()
    capabilities = {"max_concurrent_assignments": 2}
    runtime = {"current_model": "meta-llama/Llama-3.1-8B-Instruct"}

    state = main_module.maybe_send_heartbeat(
        control,
        state,
        status="active",
        queue_depth=0,
        active_assignments=0,
        capabilities=capabilities,
        runtime=runtime,
        now_monotonic=100.0,
    )
    state = main_module.maybe_send_heartbeat(
        control,
        state,
        status="active",
        queue_depth=0,
        active_assignments=0,
        capabilities=capabilities,
        runtime=runtime,
        now_monotonic=110.0,
    )
    assert len(control.heartbeat_calls) == 1

    state = main_module.maybe_send_heartbeat(
        control,
        state,
        status="active",
        queue_depth=0,
        active_assignments=1,
        capabilities=capabilities,
        runtime=runtime,
        now_monotonic=111.0,
    )

    assert len(control.heartbeat_calls) == 2
    assert control.heartbeat_calls[-1][1]["include_capabilities"] is False
    assert control.heartbeat_calls[-1][1]["include_runtime"] is False


def test_maybe_send_heartbeat_ignores_volatile_gpu_telemetry_for_metadata_flush():
    class HeartbeatRecordingControl(FakeControl):
        def __init__(self) -> None:
            super().__init__(has_credentials=True)
            self.heartbeat_calls = []

        def heartbeat(self, *args, **kwargs):
            self.heartbeat_calls.append((args, kwargs))
            return None

    control = HeartbeatRecordingControl()
    state = main_module.HeartbeatState()
    runtime = {"current_model": "BAAI/bge-large-en-v1.5"}
    capabilities = {
        "supported_models": ["BAAI/bge-large-en-v1.5"],
        "operations": ["embeddings"],
        "gpu_temp_c": 41.0,
        "power_watts": 18.0,
        "estimated_heat_output_watts": 18.0,
    }

    state = main_module.maybe_send_heartbeat(
        control,
        state,
        status="active",
        queue_depth=0,
        active_assignments=0,
        capabilities=capabilities,
        runtime=runtime,
        now_monotonic=100.0,
    )
    state = main_module.maybe_send_heartbeat(
        control,
        state,
        status="active",
        queue_depth=0,
        active_assignments=0,
        capabilities={
            **capabilities,
            "gpu_temp_c": 45.0,
            "power_watts": 27.0,
            "estimated_heat_output_watts": 27.0,
        },
        runtime=runtime,
        now_monotonic=105.0,
    )

    assert len(control.heartbeat_calls) == 1


def test_maybe_touch_assignment_leases_throttles_and_flushes_assignment_set_changes():
    control = FakeControl(has_credentials=True)
    state = main_module.LeaseKeepaliveState()

    state = main_module.maybe_touch_assignment_leases(
        control,
        state,
        ["assign_b", "assign_a", "assign_a"],
        now_monotonic=100.0,
    )
    state = main_module.maybe_touch_assignment_leases(
        control,
        state,
        ["assign_a", "assign_b"],
        now_monotonic=110.0,
    )
    state = main_module.maybe_touch_assignment_leases(
        control,
        state,
        ["assign_a", "assign_b", "assign_c"],
        now_monotonic=111.0,
    )
    state = main_module.maybe_touch_assignment_leases(
        control,
        state,
        ["assign_a", "assign_b", "assign_c"],
        now_monotonic=141.0,
    )

    assert control.touched_assignments == [
        ["assign_a", "assign_b"],
        ["assign_a", "assign_b", "assign_c"],
        ["assign_a", "assign_b", "assign_c"],
    ]


def test_validate_assignment_rejects_restricted_work_without_hardware_attestation():
    control = FakeControl(has_credentials=True)
    control.settings.attestation_provider = "simulated"
    control.attestation_state["attestation_provider"] = "simulated"
    assignment = SimpleNamespace(
        assignment_id="assign_restricted",
        execution_id="pexec_restricted",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="restricted",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
    )

    with pytest.raises(ValueError, match="not hardware-backed"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )


def test_validate_assignment_rejects_restricted_work_with_stale_local_attestation():
    control = FakeControl(has_credentials=True)
    control.attestation_state["attested_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    assignment = SimpleNamespace(
        assignment_id="assign_restricted_stale",
        execution_id="pexec_restricted_stale",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="restricted",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
    )

    with pytest.raises(ValueError, match="fresh local hardware attestation record"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )


def test_run_worker_loop_refreshes_stale_restricted_attestation_before_polling():
    control = FakeControl(has_credentials=True)
    control.attestation_state["attested_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

    with pytest.raises(KeyboardInterrupt):
        main_module.run_worker_loop(control, object(), attest_on_start=False)

    assert control.attest_calls == 1


def test_validate_assignment_rejects_item_model_mismatch():
    control = FakeControl(has_credentials=True)
    assignment = SimpleNamespace(
        assignment_id="assign_model_mismatch",
        execution_id="pexec_model_mismatch",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="restricted",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
    )

    with pytest.raises(ValueError, match="does not match envelope"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "BAAI/bge-large-en-v1.5",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )


def test_build_runtime_receipt_uses_release_metadata_for_the_assignment_model():
    control = FakeControl(has_credentials=True)
    assignment = SimpleNamespace(
        assignment_id="assign_receipt",
        execution_id="pexec_receipt",
        assignment_nonce="nonce_123",
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    artifact = find_model_artifact("meta-llama/Llama-3.1-8B-Instruct", "responses")

    receipt = main_module.build_runtime_receipt(control, assignment, [{"usage": {"total_tokens": 5}}])

    assert artifact is not None
    assert receipt["declared_runtime_image_digest"] == control.settings.docker_image
    assert receipt["declared_model_manifest_digest"] == artifact.model_manifest_digest
    assert receipt["declared_tokenizer_digest"] == artifact.tokenizer_digest
    assert isinstance(receipt["declared_chat_template_digest"], str)
    assert receipt["declared_effective_context_tokens"] == control.settings.max_context_tokens
    assert isinstance(receipt["declared_runtime_tuple_digest"], str)


def test_build_runtime_receipt_uses_embedding_model_context_limit():
    control = FakeControl(has_credentials=True)
    assignment = SimpleNamespace(
        assignment_id="assign_receipt_embedding",
        execution_id="pexec_receipt_embedding",
        assignment_nonce="nonce_embedding",
        operation="embeddings",
        model="BAAI/bge-large-en-v1.5",
    )
    artifact = find_model_artifact("BAAI/bge-large-en-v1.5", "embeddings")

    receipt = main_module.build_runtime_receipt(control, assignment, [{"usage": {"total_tokens": 5}}])

    assert artifact is not None
    assert receipt["declared_model_manifest_digest"] == artifact.model_manifest_digest
    assert receipt["declared_tokenizer_digest"] == artifact.tokenizer_digest
    assert receipt["declared_effective_context_tokens"] == 512
    assert isinstance(receipt["declared_runtime_tuple_digest"], str)


def test_build_runtime_receipt_declares_gguf_artifact_for_llama_cpp():
    control = FakeControl(has_credentials=True)
    control.settings.runtime_profile = "home_llama_cpp_gguf"
    control.settings.resolved_runtime_profile_id = "home_llama_cpp_gguf"
    control.settings.resolved_inference_engine = "llama_cpp"
    control.settings.current_model = "meta-llama/Llama-3.1-8B-Instruct"
    assignment = SimpleNamespace(
        assignment_id="assign_receipt_gguf",
        execution_id="pexec_receipt_gguf",
        assignment_nonce="nonce_gguf",
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
    )
    artifact = find_gguf_artifact("meta-llama/Llama-3.1-8B-Instruct", "responses")

    receipt = main_module.build_runtime_receipt(control, assignment, [{"usage": {"total_tokens": 5}}])

    assert artifact is not None
    assert receipt["declared_model_manifest_digest"] is None
    assert receipt["declared_tokenizer_digest"] is None
    assert receipt["declared_gguf_artifact"]["file_digest"] == artifact.file_digest
    assert receipt["declared_gguf_artifact"]["quantization_type"] == "Q4_K_M"


def test_validate_assignment_rejects_mismatched_gguf_file_digest():
    control = FakeControl(has_credentials=True)
    control.settings.runtime_profile = "home_llama_cpp_gguf"
    control.settings.resolved_runtime_profile_id = "home_llama_cpp_gguf"
    control.settings.resolved_inference_engine = "llama_cpp"
    control.settings.current_model = "meta-llama/Llama-3.1-8B-Instruct"
    assignment = SimpleNamespace(
        assignment_id="assign_gguf_mismatch",
        execution_id="pexec_gguf_mismatch",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="standard",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
        expected_gguf_file_digest="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="local GGUF file digest"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )


def test_validate_assignment_rejects_mismatched_runtime_tuple_digest():
    control = FakeControl(has_credentials=True)
    assignment = SimpleNamespace(
        assignment_id="assign_runtime_tuple_mismatch",
        execution_id="pexec_runtime_tuple_mismatch",
        item_count=1,
        operation="responses",
        model="meta-llama/Llama-3.1-8B-Instruct",
        privacy_tier="restricted",
        allowed_regions=["eu-se-1"],
        required_vram_gb=16.0,
        required_context_tokens=8192,
        token_budget={"total_tokens": 2048},
        node_trust_requirement="trusted_only",
        expected_runtime_tuple_digest="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="runtime tuple digest"):
        main_module.validate_assignment(
            control,
            assignment,
            [
                {
                    "batch_item_id": "item_1",
                    "customer_item_id": "cust_1",
                    "operation": "responses",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )
