from __future__ import annotations

import ipaddress
import os
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlsplit

ADMIN_TOKEN_HEADER = "x-local-admin-token"
LOCAL_SESSION_COOKIE = "autonomousc_local_session"
LOCAL_SESSION_TTL_SECONDS = 15 * 60


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


def require_secure_bind_host(host: str, allow_remote: bool | None = None) -> None:
    if not is_loopback_host(host):
        raise ValueError(
            "Refusing to bind the local control service to a non-loopback host. "
            "Remote mode has been removed; use an authenticated tunnel if you need remote access."
        )


def request_query_param(path: str, name: str) -> str | None:
    values = parse_qs(urlsplit(path).query).get(name)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def cookie_value(headers: Mapping[str, str], name: str) -> str | None:
    cookie_header = headers.get("cookie")
    if not cookie_header:
        return None
    for cookie in cookie_header.split(";"):
        raw_name, _, raw_value = cookie.strip().partition("=")
        if raw_name == name:
            value = raw_value.strip()
            return value or None
    return None


def serialize_cookie(
    name: str,
    value: str,
    *,
    max_age: int,
    http_only: bool = True,
    same_site: str = "Strict",
    path: str = "/",
) -> str:
    parts = [f"{name}={value}", f"Max-Age={max(0, int(max_age))}", f"Path={path}", f"SameSite={same_site}"]
    if http_only:
        parts.append("HttpOnly")
    return "; ".join(parts)


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


class LocalSessionStore:
    def __init__(self, ttl_seconds: int = LOCAL_SESSION_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    def issue(self) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._prune_locked()
            self._sessions[token] = expires_at
        return token

    def contains(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            self._prune_locked()
            expires_at = self._sessions.get(token)
            return expires_at is not None and expires_at > time.time()

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [token for token, expires_at in self._sessions.items() if expires_at <= now]
        for token in expired:
            self._sessions.pop(token, None)
