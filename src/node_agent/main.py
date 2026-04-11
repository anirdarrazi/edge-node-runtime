from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime, timezone

import httpx

from .config import NodeAgentSettings
from .control_plane import EdgeControlClient
from .model_artifacts import resolved_model_manifest_digest, resolved_tokenizer_digest
from .runtime import VLLMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("autonomousc-node-agent")
assignment_progress_keepalive_seconds = 30.0
supported_operations = frozenset({"responses", "embeddings"})


def is_transient_http_error(error: Exception) -> bool:
    if isinstance(error, httpx.RequestError):
        return True
    if not isinstance(error, httpx.HTTPStatusError):
        return False
    return error.response.status_code in {408, 409, 425, 429} or error.response.status_code >= 500


def classify_assignment_failure(error: Exception) -> tuple[str, str, bool]:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        request_url = str(error.request.url)
        if status in {400, 404, 409, 422}:
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


def configured_supported_models(settings: NodeAgentSettings) -> set[str]:
    return {model.strip() for model in str(settings.supported_models).split(",") if model.strip()}


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
    if not isinstance(allowed_regions, list) or (
        "global" not in allowed_regions and settings.node_region not in allowed_regions
    ):
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
    if float(required_vram_gb) > settings.gpu_memory_gb:
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

    node_trust_requirement = getattr(assignment, "node_trust_requirement", None)
    if node_trust_requirement == "trusted_only":
        expected_runtime_image_digest = getattr(assignment, "expected_runtime_image_digest", None)
        if isinstance(expected_runtime_image_digest, str) and expected_runtime_image_digest:
            if settings.docker_image != expected_runtime_image_digest:
                raise ValueError(
                    "trusted assignment rejected because the local runtime image digest "
                    f"{settings.docker_image!r} does not match expected {expected_runtime_image_digest!r}"
                )
        expected_model_manifest_digest = getattr(assignment, "expected_model_manifest_digest", None)
        if isinstance(expected_model_manifest_digest, str) and expected_model_manifest_digest:
            declared_model_manifest_digest = resolved_model_manifest_digest(
                settings, assignment_model, assignment_operation
            )
            if declared_model_manifest_digest != expected_model_manifest_digest:
                raise ValueError(
                    "trusted assignment rejected because the local model manifest digest "
                    f"{declared_model_manifest_digest!r} does not match expected {expected_model_manifest_digest!r}"
                )
        expected_tokenizer_digest = getattr(assignment, "expected_tokenizer_digest", None)
        if isinstance(expected_tokenizer_digest, str) and expected_tokenizer_digest:
            declared_tokenizer_digest = resolved_tokenizer_digest(settings, assignment_model, assignment_operation)
            if declared_tokenizer_digest != expected_tokenizer_digest:
                raise ValueError(
                    "trusted assignment rejected because the local tokenizer digest "
                    f"{declared_tokenizer_digest!r} does not match expected {expected_tokenizer_digest!r}"
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
    return {
        "assignment_nonce": getattr(assignment, "assignment_nonce", ""),
        "declared_model": assignment_model or "",
        "declared_runtime_image_digest": control.settings.docker_image,
        "declared_model_manifest_digest": resolved_model_manifest_digest(
            control.settings, assignment_model, assignment_operation
        ),
        "declared_tokenizer_digest": resolved_tokenizer_digest(
            control.settings, assignment_model, assignment_operation
        ),
        "provider_usage_summary": summarize_provider_usage(item_results),
    }


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
    LOGGER.info("node enrolled or restored: %s", node_id)
    if attest_on_start:
        print("Refreshing node attestation before entering the worker loop...")
        control.attest()
        LOGGER.info("node attested: %s", node_id)
    print("Node agent is online. Polling the control plane for assignments...")

    while True:
        try:
            if (
                control.settings.restricted_capable
                and control.settings.trust_tier == "restricted"
                and control.settings.attestation_provider == "hardware"
                and not restricted_attestation_is_fresh(control)
            ):
                LOGGER.info("local restricted attestation is stale or missing; refreshing it before polling")
                control.attest()
            control.heartbeat()
            assignment = control.pull_assignment()
            if not assignment:
                time.sleep(control.settings.poll_interval_seconds)
                continue

            assignment_claimed = False
            results_ready = False
            try:
                LOGGER.info("accepted assignment %s", assignment.assignment_id)
                control.accept_assignment(assignment.assignment_id)
                assignment_claimed = True
                payload = control.fetch_artifact(assignment)
                items = validate_assignment(control, assignment, payload.get("items", []))
                item_count = len(items)
                control.report_progress(assignment.assignment_id, assignment_progress("running", item_count))
                progress_keepalive = AssignmentProgressKeepalive(control, assignment.assignment_id, item_count)
                progress_keepalive.start()
                results = []
                runtime_error: Exception | None = None
                try:
                    results = runtime.execute(assignment.operation, assignment.model, items)
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
                            assignment.assignment_id,
                            error,
                        )
                if runtime_error is not None:
                    raise runtime_error
                results_ready = True
                control.report_progress(assignment.assignment_id, assignment_progress("completed", item_count))
                runtime_receipt = build_runtime_receipt(control, assignment, results)
                complete_assignment_with_retry(control, assignment.assignment_id, results, runtime_receipt)
            except Exception as assignment_error:
                if control.is_auth_error(assignment_error):
                    raise
                if not assignment_claimed:
                    LOGGER.warning("assignment %s was no longer claimable: %s", assignment.assignment_id, assignment_error)
                    continue
                if results_ready:
                    raise

                code, message, retryable = classify_assignment_failure(assignment_error)
                LOGGER.exception("assignment %s failed", assignment.assignment_id)
                try:
                    control.report_progress(
                        assignment.assignment_id,
                        assignment_progress("failed", assignment.item_count, code=code, retryable=retryable),
                    )
                except Exception as progress_error:
                    if control.is_auth_error(progress_error):
                        raise
                    LOGGER.warning(
                        "failed to report terminal progress for %s: %s",
                        assignment.assignment_id,
                        progress_error,
                    )
                try:
                    control.fail_assignment(assignment.assignment_id, code, message, retryable=retryable)
                except Exception as fail_error:
                    if control.is_auth_error(fail_error):
                        raise
                    LOGGER.warning(
                        "failed to mark assignment %s as failed; stale reclaim will handle it: %s",
                        assignment.assignment_id,
                        fail_error,
                    )
        except Exception as error:  # pragma: no cover - long-running loop
            if control.is_auth_error(error):
                LOGGER.warning(
                    "node credentials were rejected by the control plane; clearing local credentials and requiring a new bootstrap"
                )
                control.write_recovery_note(
                    "This node lost control-plane access because its credentials were rejected or revoked. "
                    "Run `node-agent bootstrap` again from an interactive terminal to reclaim it."
                )
                control.clear_credentials()
                raise RuntimeError(
                    "Node credentials were rejected by the control plane. Run `node-agent bootstrap` again to reclaim this node."
                ) from error
            LOGGER.exception("node agent loop failed")
            time.sleep(control.settings.poll_interval_seconds)


def command_bootstrap() -> int:
    settings = NodeAgentSettings()
    control = EdgeControlClient(settings)
    bootstrap_node(control, interactive=True)
    return 0


def command_run() -> int:
    settings = NodeAgentSettings()
    control = EdgeControlClient(settings)
    runtime = VLLMRuntime(settings.vllm_base_url)
    run_worker_loop(control, runtime, attest_on_start=True)
    return 0


def command_default() -> int:
    settings = NodeAgentSettings()
    control = EdgeControlClient(settings)
    runtime = VLLMRuntime(settings.vllm_base_url)

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
