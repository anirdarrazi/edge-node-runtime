from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _quiet_noisy_http_logs() -> None:
    # httpx logs full presigned artifact URLs at INFO, including short-lived tokens.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def drain_once(limit: int) -> int:
    _ensure_src_on_path()

    from node_agent.config import NodeAgentSettings
    from node_agent.control_plane import EdgeControlClient
    from node_agent.main import (
        process_accepted_assignment,
        resolved_inference_base_url_for_settings,
        resolved_inference_engine_for_settings,
    )
    from node_agent.runtime import VLLMRuntime

    settings = NodeAgentSettings()
    control = EdgeControlClient(settings)
    control.heartbeat(
        queue_depth=0,
        active_assignments=0,
        capabilities=control.node_capabilities_payload(),
        runtime=control.node_runtime_payload(),
    )
    assignments = control.pull_assignments(max(1, min(64, limit)), active_assignment_ids=[])
    print(
        json.dumps(
            {
                "event": "pulled",
                "assignment_count": len(assignments),
                "assignment_ids": [str(getattr(assignment, "assignment_id", "") or "") for assignment in assignments],
            }
        ),
        flush=True,
    )
    runtime = VLLMRuntime(
        resolved_inference_base_url_for_settings(settings),
        engine=resolved_inference_engine_for_settings(settings),
    )
    for index, assignment in enumerate(assignments, start=1):
        result = process_accepted_assignment(
            control,
            runtime,
            assignment,
            queue_depth=max(1, len(assignments) - index + 1),
        )
        print(
            json.dumps(
                {
                    "event": "processed",
                    "assignment_id": result.assignment_id,
                    "kind": result.kind,
                    "code": result.code,
                    "retryable": result.retryable,
                    "latency_seconds": result.latency_seconds,
                    "usage_summary": result.usage_summary,
                }
            ),
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    _quiet_noisy_http_logs()

    parser = argparse.ArgumentParser(description="Pull and process one node-agent assignment bundle.")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--loop", action="store_true", help="Keep draining bundles until interrupted.")
    parser.add_argument("--idle-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--error-sleep-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)

    if not args.loop:
        return drain_once(args.limit)

    while True:
        try:
            drain_once(args.limit)
            time.sleep(max(0.1, args.idle_sleep_seconds))
        except KeyboardInterrupt:
            return 0
        except Exception as error:
            print(
                json.dumps(
                    {
                        "event": "drain_loop_error",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "retry_after_seconds": max(1.0, args.error_sleep_seconds),
                    }
                ),
                flush=True,
            )
            time.sleep(max(1.0, args.error_sleep_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
