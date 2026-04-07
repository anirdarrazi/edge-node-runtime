from __future__ import annotations

import logging
import sys
import time

from .config import NodeAgentSettings
from .control_plane import EdgeControlClient
from .runtime import VLLMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("autonomousc-node-agent")


def bootstrap_node(control: EdgeControlClient, interactive: bool) -> str:
    node_id, _node_key = control.bootstrap(interactive=interactive)
    LOGGER.info("node enrolled or restored: %s", node_id)
    control.attest()
    LOGGER.info("node attested: %s", node_id)
    return node_id


def run_worker_loop(control: EdgeControlClient, runtime: VLLMRuntime, attest_on_start: bool = True) -> None:
    node_id, _node_key = control.require_credentials()
    LOGGER.info("node enrolled or restored: %s", node_id)
    if attest_on_start:
        control.attest()
        LOGGER.info("node attested: %s", node_id)

    while True:
        try:
            control.heartbeat()
            assignment = control.pull_assignment()
            if not assignment:
                time.sleep(control.settings.poll_interval_seconds)
                continue

            LOGGER.info("accepted assignment %s", assignment.assignment_id)
            control.accept_assignment(assignment.assignment_id)
            payload = control.fetch_artifact(assignment)
            items = payload.get("items", [])
            control.report_progress(assignment.assignment_id, {"state": "running", "item_count": len(items)})
            results = runtime.execute(assignment.operation, assignment.model, items)
            control.complete_assignment(assignment.assignment_id, results)
        except Exception as error:  # pragma: no cover - long-running loop
            if control.is_auth_error(error):
                LOGGER.warning(
                    "node credentials were rejected by the control plane; clearing local credentials and requiring a new bootstrap"
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
