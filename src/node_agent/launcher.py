from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from node_agent.single_container import main as single_container_main
else:
    from .single_container import main as single_container_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        raise SystemExit("node-agent-launcher now starts the Docker runtime directly and does not accept service commands.")
    return single_container_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
