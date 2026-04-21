from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import NodeRuntimeService, SUPPORTED_FAULT_DRILLS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a live node fault-recovery drill and record the result."
    )
    parser.add_argument(
        "scenario",
        choices=SUPPORTED_FAULT_DRILLS,
        help="Which live drill to run.",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=None,
        help="Optional runtime bundle directory override.",
    )
    args = parser.parse_args(argv)

    service = NodeRuntimeService(runtime_dir=args.runtime_dir)
    payload = service.run_fault_drill(args.scenario)
    fault_drills = payload.get("fault_drills") if isinstance(payload.get("fault_drills"), dict) else {}
    print(json.dumps(fault_drills, indent=2))
    last_drill = fault_drills.get("last_drill") if isinstance(fault_drills.get("last_drill"), dict) else {}
    status = str(last_drill.get("status") or "").strip().lower()
    return 0 if status in {"passed", "unavailable"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
