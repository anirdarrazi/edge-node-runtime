from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from node_agent.service import main as service_main
else:
    from .service import main as service_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return service_main(["start", "--open"])
    return service_main(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
