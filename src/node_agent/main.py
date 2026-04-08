from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime, timezone

import httpx

from .config import NodeAgentSettings
from .control_plane import EdgeControlClient
from .runtime import VLLMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("autonomousc-node-agent")
assignment_progress_keepalive_seconds = 30.0


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


def complete_assignment_with_retry(
    control: EdgeControlClient,
    assignment_id: str,
    item_results: list[dict[str, object]],
    max_attempts: int = 3,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            control.complete_assignment(assignment_id, item_results)
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
                items = payload.get("items", [])
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
                complete_assignment_with_retry(control, assignment.assignment_id, results)
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
