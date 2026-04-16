from __future__ import annotations

import os
from pathlib import Path


AUTO_RUNTIME_BACKEND = "auto"
MANAGER_RUNTIME_BACKEND = "manager"
SINGLE_CONTAINER_RUNTIME_BACKEND = "single_container"
RUNTIME_BACKEND_ENV = "AUTONOMOUSC_RUNTIME_BACKEND"


def normalize_runtime_backend(value: str | None) -> str:
    normalized = (value or AUTO_RUNTIME_BACKEND).strip().lower().replace("-", "_")
    if normalized in {MANAGER_RUNTIME_BACKEND, SINGLE_CONTAINER_RUNTIME_BACKEND}:
        return normalized
    return AUTO_RUNTIME_BACKEND


def docker_socket_present() -> bool:
    return Path("/var/run/docker.sock").exists()


def running_inside_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.getenv("container"))


def detect_runtime_backend() -> str:
    configured = normalize_runtime_backend(os.getenv(RUNTIME_BACKEND_ENV))
    if configured != AUTO_RUNTIME_BACKEND:
        return configured
    if docker_socket_present():
        return MANAGER_RUNTIME_BACKEND
    if not running_inside_container():
        return MANAGER_RUNTIME_BACKEND
    return SINGLE_CONTAINER_RUNTIME_BACKEND


def runtime_backend_label(backend: str) -> str:
    if backend == MANAGER_RUNTIME_BACKEND:
        return "Host Docker"
    if backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
        return "Single container"
    return "Automatic"


def runtime_backend_required_services(backend: str) -> tuple[str, ...]:
    if backend == SINGLE_CONTAINER_RUNTIME_BACKEND:
        return ("vllm", "node-agent")
    return ("vllm", "node-agent", "vector")


def runtime_backend_supports_compose(backend: str) -> bool:
    return backend == MANAGER_RUNTIME_BACKEND


def default_service_access_host(backend: str) -> str:
    if backend == MANAGER_RUNTIME_BACKEND:
        return "host.docker.internal"
    return "127.0.0.1"
