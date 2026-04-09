from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


RUNTIME_DIR_ENV = "AUTONOMOUSC_RUNTIME_DIR"
RUNTIME_HOST_ENV = "AUTONOMOUSC_RUNTIME_HOST"
RUNTIME_BUNDLE_FILES = (".env.example", "docker-compose.yml", "vector.toml")


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bundled_runtime_dir() -> Path:
    return Path(__file__).with_name("runtime_bundle")


def default_runtime_home() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / "AUTONOMOUSc Edge Node Runtime"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AUTONOMOUSc Edge Node Runtime"
    base = Path(os.getenv("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / "autonomousc-edge-node-runtime"


def resolve_runtime_dir() -> Path:
    override = os.getenv(RUNTIME_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return default_runtime_home().resolve()
    return package_root().resolve()


def service_access_host() -> str:
    return os.getenv(RUNTIME_HOST_ENV, "127.0.0.1").strip() or "127.0.0.1"


def ensure_runtime_bundle(runtime_dir: Path) -> Path:
    target = runtime_dir.resolve()
    source = bundled_runtime_dir()

    if target == package_root().resolve():
        return target

    target.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_BUNDLE_FILES:
        source_path = source / name
        target_path = target / name
        if not target_path.exists():
            shutil.copyfile(source_path, target_path)
    return target
