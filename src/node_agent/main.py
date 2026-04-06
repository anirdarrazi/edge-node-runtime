from __future__ import annotations

import logging
import time

from .config import NodeAgentSettings
from .control_plane import EdgeControlClient
from .runtime import VLLMRuntime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("autonomousc-node-agent")


def run() -> None:
    settings = NodeAgentSettings()
    control = EdgeControlClient(settings)
    runtime = VLLMRuntime(settings.vllm_base_url)
    bootstrapped = False

    while True:
        try:
            if not bootstrapped:
                node_id, _node_key = control.enroll_if_needed()
                LOGGER.info("node enrolled or restored: %s", node_id)
                control.attest()
                bootstrapped = True

            control.heartbeat()
            assignment = control.pull_assignment()
            if not assignment:
                time.sleep(settings.poll_interval_seconds)
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
                LOGGER.warning("node credentials were rejected by the control plane; clearing local credentials and retrying enrollment")
                control.clear_credentials()
                bootstrapped = False
            LOGGER.exception("node agent loop failed")
            time.sleep(settings.poll_interval_seconds)


if __name__ == "__main__":  # pragma: no cover
    run()
