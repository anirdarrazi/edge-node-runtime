from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .appliance_manifest import verify_package_dir, verify_runtime_bundle_dir
from .runtime_backend import default_service_access_host, detect_runtime_backend

RUNTIME_DIR_ENV = "AUTONOMOUSC_RUNTIME_DIR"
RUNTIME_HOST_ENV = "AUTONOMOUSC_RUNTIME_HOST"


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def checkout_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "node_agent").exists():
        return candidate.resolve()
    return None


def appliance_runtime_root() -> Path:
    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "AUTONOMOUSc Edge Node"
    if os.name == "posix" and "darwin" in sys.platform:
        return Path.home() / "Library" / "Application Support" / "AUTONOMOUSc Edge Node"
    return Path.home() / ".local" / "share" / "autonomousc-edge-node"


def bundled_runtime_dir() -> Path:
    return package_dir() / "runtime_bundle"


def resolve_runtime_dir() -> Path:
    override = os.getenv(RUNTIME_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    checkout = checkout_root()
    if checkout is not None:
        return checkout
    return appliance_runtime_root().expanduser().resolve()


def service_access_host() -> str:
    override = os.getenv(RUNTIME_HOST_ENV)
    if override:
        return override.strip() or "127.0.0.1"
    return default_service_access_host(detect_runtime_backend())


def ensure_runtime_bundle(runtime_dir: Path) -> Path:
    target = runtime_dir.resolve()
    source = bundled_runtime_dir()
    checkout = checkout_root()

    verify_runtime_bundle_dir(source)
    if checkout is None:
        verify_package_dir(package_dir())

    if checkout is not None and target == checkout:
        return target

    target.mkdir(parents=True, exist_ok=True)
    for source_path in source.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source)
        target_path = target / relative_path
        if relative_path == Path(".env.example") and target_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)
    verify_runtime_bundle_dir(target)
    return target
