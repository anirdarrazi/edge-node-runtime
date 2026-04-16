from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from node_agent.service import main as service_main
    from node_agent.runtime_backend import RUNTIME_BACKEND_ENV, detect_runtime_backend
else:
    from .service import main as service_main
    from .runtime_backend import RUNTIME_BACKEND_ENV, detect_runtime_backend


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        runtime_backend = detect_runtime_backend()
        os.environ.setdefault(RUNTIME_BACKEND_ENV, runtime_backend)
        return service_main(["run", "--host", "0.0.0.0", "--port", "8765"])
    return service_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
