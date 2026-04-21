from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


PACKAGE_MANIFEST_NAME = "appliance-package-manifest.json"
PACKAGE_PUBLIC_KEY_NAME = "appliance-package-manifest.pub"
RUNTIME_MANIFEST_NAME = "appliance-runtime-manifest.json"
RUNTIME_PUBLIC_KEY_NAME = "appliance-runtime-manifest.pub"

DEFAULT_RELEASE_CHANNEL = "stable"
EARLY_RELEASE_CHANNELS = {"early", "beta", "preview", "canary", "nightly"}


class ApplianceManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApplianceFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ApplianceManifest:
    kind: str
    version: str
    channel: str
    published_at: str
    files: dict[str, ApplianceFile]


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def bundled_runtime_dir() -> Path:
    return package_dir() / "runtime_bundle"


def normalize_release_channel(value: str | None) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in EARLY_RELEASE_CHANNELS:
        return "early"
    return DEFAULT_RELEASE_CHANNEL


def release_channel_label(value: str | None) -> str:
    channel = normalize_release_channel(value)
    return "Early access" if channel == "early" else "Stable"


def release_channel_matches_preference(*, release_channel: str | None, preferred_channel: str | None) -> bool:
    preferred = normalize_release_channel(preferred_channel)
    release = normalize_release_channel(release_channel)
    if preferred == "early":
        return release in {"stable", "early"}
    return release == "stable"


def package_manifest_path(package_root: Path | None = None) -> Path:
    root = package_root.resolve() if package_root is not None else package_dir()
    return root / PACKAGE_MANIFEST_NAME


def package_public_key_path(package_root: Path | None = None) -> Path:
    root = package_root.resolve() if package_root is not None else package_dir()
    return root / PACKAGE_PUBLIC_KEY_NAME


def runtime_manifest_path(bundle_root: Path | None = None) -> Path:
    root = bundle_root.resolve() if bundle_root is not None else bundled_runtime_dir()
    return root / RUNTIME_MANIFEST_NAME


def runtime_public_key_path(bundle_root: Path | None = None) -> Path:
    root = bundle_root.resolve() if bundle_root is not None else bundled_runtime_dir()
    return root / RUNTIME_PUBLIC_KEY_NAME


def _decode_base64(field: str, value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ApplianceManifestError(f"Signed appliance manifest field {field} is missing.")
    try:
        return base64.b64decode(value)
    except Exception as exc:  # pragma: no cover - validated through caller behavior
        raise ApplianceManifestError(f"Signed appliance manifest field {field} is invalid.") from exc


def _load_public_key(path: Path) -> Ed25519PublicKey:
    try:
        public_key = serialization.load_pem_public_key(path.read_bytes())
    except FileNotFoundError as exc:
        raise ApplianceManifestError(f"Signed appliance public key is missing at {path}.") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ApplianceManifestError("Signed appliance manifest public key is not an Ed25519 key.")
    return public_key


def _parse_manifest_payload(payload: dict[str, Any], *, expected_kind: str) -> ApplianceManifest:
    kind = payload.get("kind")
    version = payload.get("version")
    channel = payload.get("channel")
    published_at = payload.get("published_at")
    files_payload = payload.get("files")
    if kind != expected_kind:
        raise ApplianceManifestError(f"Signed appliance manifest kind {kind!r} is invalid for {expected_kind}.")
    if not isinstance(version, str) or not version:
        raise ApplianceManifestError("Signed appliance manifest version is missing.")
    if not isinstance(channel, str) or not channel:
        raise ApplianceManifestError("Signed appliance manifest channel is missing.")
    if not isinstance(published_at, str) or not published_at:
        raise ApplianceManifestError("Signed appliance manifest published_at value is missing.")
    if not isinstance(files_payload, dict) or not files_payload:
        raise ApplianceManifestError("Signed appliance manifest file list is missing.")

    files: dict[str, ApplianceFile] = {}
    for relative_path, file_payload in files_payload.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise ApplianceManifestError("Signed appliance manifest contains an invalid file path.")
        if not isinstance(file_payload, dict):
            raise ApplianceManifestError(f"Signed appliance manifest entry for {relative_path} is invalid.")
        sha256 = file_payload.get("sha256")
        size = file_payload.get("size")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ApplianceManifestError(f"Signed appliance manifest sha256 for {relative_path} is invalid.")
        if not isinstance(size, int) or size < 0:
            raise ApplianceManifestError(f"Signed appliance manifest size for {relative_path} is invalid.")
        files[relative_path] = ApplianceFile(path=relative_path, sha256=sha256, size=size)

    return ApplianceManifest(
        kind=kind,
        version=version,
        channel=normalize_release_channel(channel),
        published_at=published_at,
        files=files,
    )


def verify_appliance_manifest_envelope(
    envelope: dict[str, Any],
    *,
    public_key_path: Path,
    expected_kind: str,
) -> ApplianceManifest:
    if envelope.get("algorithm") != "ed25519":
        raise ApplianceManifestError("Signed appliance manifest uses an unsupported signature algorithm.")

    payload_bytes = _decode_base64("payload", envelope.get("payload"))
    signature = _decode_base64("signature", envelope.get("signature"))
    try:
        _load_public_key(public_key_path).verify(signature, payload_bytes)
    except InvalidSignature as exc:
        raise ApplianceManifestError("Signed appliance manifest signature verification failed.") from exc

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ApplianceManifestError("Signed appliance manifest payload is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ApplianceManifestError("Signed appliance manifest payload must be a JSON object.")
    return _parse_manifest_payload(payload, expected_kind=expected_kind)


def load_package_manifest(package_root: Path | None = None) -> ApplianceManifest:
    manifest_path = package_manifest_path(package_root)
    try:
        envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ApplianceManifestError("Signed appliance package manifest is missing.") from exc
    except json.JSONDecodeError as exc:
        raise ApplianceManifestError("Signed appliance package manifest is not valid JSON.") from exc
    return verify_appliance_manifest_envelope(
        envelope,
        public_key_path=package_public_key_path(package_root),
        expected_kind="appliance_package",
    )


def load_runtime_bundle_manifest(bundle_root: Path | None = None) -> ApplianceManifest:
    manifest_path = runtime_manifest_path(bundle_root)
    try:
        envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ApplianceManifestError("Signed appliance runtime bundle manifest is missing.") from exc
    except json.JSONDecodeError as exc:
        raise ApplianceManifestError("Signed appliance runtime bundle manifest is not valid JSON.") from exc
    return verify_appliance_manifest_envelope(
        envelope,
        public_key_path=runtime_public_key_path(bundle_root),
        expected_kind="appliance_runtime_bundle",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_files(root: Path, manifest: ApplianceManifest) -> None:
    for relative_path, entry in manifest.files.items():
        target = root / Path(relative_path)
        if not target.exists():
            raise ApplianceManifestError(f"Signed appliance file {relative_path} is missing.")
        if not target.is_file():
            raise ApplianceManifestError(f"Signed appliance file {relative_path} is not a file.")
        actual_size = target.stat().st_size
        if actual_size != entry.size:
            raise ApplianceManifestError(
                f"Signed appliance file {relative_path} size mismatch: expected {entry.size}, got {actual_size}."
            )
        actual_sha256 = _sha256_file(target)
        if actual_sha256 != entry.sha256:
            raise ApplianceManifestError(
                f"Signed appliance file {relative_path} digest mismatch: expected {entry.sha256}, got {actual_sha256}."
            )


def verify_package_dir(package_root: Path | None = None) -> ApplianceManifest:
    root = package_root.resolve() if package_root is not None else package_dir()
    manifest = load_package_manifest(root)
    _verify_files(root, manifest)
    return manifest


def verify_runtime_bundle_dir(bundle_root: Path | None = None) -> ApplianceManifest:
    root = bundle_root.resolve() if bundle_root is not None else bundled_runtime_dir()
    manifest = load_runtime_bundle_manifest(root)
    _verify_files(root, manifest)
    return manifest


def inspect_package_signature(package_root: Path | None = None) -> dict[str, Any]:
    root = package_root.resolve() if package_root is not None else package_dir()
    try:
        manifest = verify_package_dir(root)
    except ApplianceManifestError as error:
        return {
            "verified": False,
            "kind": "appliance_package",
            "detail": str(error),
            "version": None,
            "channel": None,
            "channel_label": release_channel_label(None),
        }
    return {
        "verified": True,
        "kind": manifest.kind,
        "detail": (
            f"Signed appliance package {manifest.version} on the {release_channel_label(manifest.channel).lower()} track "
            "was verified locally."
        ),
        "version": manifest.version,
        "channel": manifest.channel,
        "channel_label": release_channel_label(manifest.channel),
        "published_at": manifest.published_at,
    }


def inspect_runtime_bundle_signature(bundle_root: Path | None = None) -> dict[str, Any]:
    root = bundle_root.resolve() if bundle_root is not None else bundled_runtime_dir()
    try:
        manifest = verify_runtime_bundle_dir(root)
    except ApplianceManifestError as error:
        return {
            "verified": False,
            "kind": "appliance_runtime_bundle",
            "detail": str(error),
            "version": None,
            "channel": None,
            "channel_label": release_channel_label(None),
        }
    return {
        "verified": True,
        "kind": manifest.kind,
        "detail": (
            f"Signed appliance runtime bundle {manifest.version} on the "
            f"{release_channel_label(manifest.channel).lower()} track was verified locally."
        ),
        "version": manifest.version,
        "channel": manifest.channel,
        "channel_label": release_channel_label(manifest.channel),
        "published_at": manifest.published_at,
    }
