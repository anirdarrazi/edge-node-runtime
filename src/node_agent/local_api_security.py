from __future__ import annotations

import ipaddress
import os
import secrets
import stat
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlsplit

ADMIN_TOKEN_HEADER = "x-local-admin-token"


def generate_admin_token() -> str:
    return secrets.token_urlsafe(32)


def token_matches(provided: str | None, expected: str) -> bool:
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def browser_access_host(host: str) -> str:
    normalized = host.strip().lower()
    if normalized in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return host


def require_secure_bind_host(host: str, allow_remote: bool) -> None:
    if allow_remote:
        return
    if not is_loopback_host(host):
        raise ValueError(
            "Refusing to bind the local control service to a non-loopback host "
            "unless --allow-remote is explicitly enabled."
        )


def request_query_param(path: str, name: str) -> str | None:
    values = parse_qs(urlsplit(path).query).get(name)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def request_origin(headers: Mapping[str, str]) -> str | None:
    origin = headers.get("origin")
    if origin:
        return origin
    referer = headers.get("referer")
    if not referer:
        return None
    parsed = urlsplit(referer)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def origin_matches_host(headers: Mapping[str, str]) -> bool:
    origin = request_origin(headers)
    if not origin:
        return True
    host = headers.get("host")
    if not host:
        return False
    return origin == f"http://{host}"


def tighten_private_path(path: Path, *, directory: bool = False) -> None:
    try:
        mode = stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if directory else 0)
        os.chmod(path, mode)
    except OSError:
        return
