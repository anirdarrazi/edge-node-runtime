from types import SimpleNamespace
import time
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

import node_agent.main as main_module
from node_agent.concurrency import max_worker_assignments_from_capabilities
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

    def load_attestation_state(self):
        return self.attestation_state


def test_command_bootstrap_runs_claim_bootstrap(monkeypatch: pytest.MonkeyPatch):
    control = FakeControl(has_credentials=False)
    monkeypatch.setattr(
        main_module,
        "NodeAgentSettings",
        lambda: SimpleNamespace(inference_base_url="http://localhost:8000", vllm_base_url="http://localhost:8000"),
    )
    monkeypatch.setattr(main_module, "EdgeControlClient", lambda _settings: control)

    result = main_module.main(["bootstrap"])

    assert result == 0
    assert control.bootstrap_calls == 1
    assert control.attest_calls == 1


def test_command_default_bootstraps_when_credentials_are_missing(monkeypatch: pytest.MonkeyPatch):
    settings = SimpleNamespace(inference_base_url="http://localhost:8000", vllm_base_url="http://localhost:8000")
    control = FakeControl(has_credentials=False)

    monkeypatch.setattr(main_module, "NodeAgentSettings", lambda: settings)
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
    assert [completion[0] for completion in control.completions] == ["assign_1", "assign_2"]
    running_updates = [progress for _assignment_id, progress in control.progress_updates if progress["state"] == "running"]
    assert running_updates
    assert all(progress["microbatch_assignments"] == 2 for progress in running_updates)


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
