from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


RELEASE_ENV_VAR_BY_SERVICE = {
    "node-agent": "NODE_AGENT_IMAGE",
    "vllm": "VLLM_IMAGE",
    "vector": "VECTOR_IMAGE",
}
REQUIRED_RELEASE_SERVICES = tuple(RELEASE_ENV_VAR_BY_SERVICE.keys())
RELEASE_ENV_VERSION_KEY = "AUTONOMOUSC_RELEASE_VERSION"
RELEASE_ENV_CHANNEL_KEY = "AUTONOMOUSC_RELEASE_CHANNEL"
DEFAULT_NODE_AGENT_IMAGE = (
    "anirdarrazi/autonomousc-ai-edge-runtime@"
    "sha256:4662922dd7912bbd928f0703e27472829cacc0a858732a2d48caa167a96561db"
)
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504"
DEFAULT_VECTOR_IMAGE = "timberio/vector@sha256:f5704c730ea10e0d7272491f4293a596f5ebc695fec64e29d29f5364895ef997"


class ReleaseManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseImage:
    service: str
    ref: str


@dataclass(frozen=True)
class ReleaseManifest:
    version: str
    channel: str
    published_at: str
    images: dict[str, ReleaseImage]

    def release_env(self) -> dict[str, str]:
        return {
            RELEASE_ENV_VERSION_KEY: self.version,
            RELEASE_ENV_CHANNEL_KEY: self.channel,
            RELEASE_ENV_VAR_BY_SERVICE["node-agent"]: self.images["node-agent"].ref,
            RELEASE_ENV_VAR_BY_SERVICE["vllm"]: self.images["vllm"].ref,
            RELEASE_ENV_VAR_BY_SERVICE["vector"]: self.images["vector"].ref,
        }


def bundled_release_dir() -> Path:
    return Path(__file__).with_name("runtime_bundle")


def _manifest_public_key_path() -> Path:
    return bundled_release_dir() / "release-manifest.pub"


def _manifest_envelope_path() -> Path:
    return bundled_release_dir() / "release-manifest.json"


def _decode_base64(field: str, value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ReleaseManifestError(f"Signed release manifest field {field} is missing.")
    try:
        return base64.b64decode(value)
    except Exception as exc:  # pragma: no cover - exercised through validation
        raise ReleaseManifestError(f"Signed release manifest field {field} is invalid.") from exc


def _load_public_key() -> Ed25519PublicKey:
    public_key = serialization.load_pem_public_key(_manifest_public_key_path().read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise ReleaseManifestError("Signed release manifest public key is not an Ed25519 key.")
    return public_key


def verify_release_manifest_envelope(envelope: dict[str, Any]) -> ReleaseManifest:
    if envelope.get("algorithm") != "ed25519":
        raise ReleaseManifestError("Signed release manifest uses an unsupported signature algorithm.")

    payload_bytes = _decode_base64("payload", envelope.get("payload"))
    signature = _decode_base64("signature", envelope.get("signature"))

    try:
        _load_public_key().verify(signature, payload_bytes)
    except InvalidSignature as exc:
        raise ReleaseManifestError("Signed release manifest signature verification failed.") from exc

    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseManifestError("Signed release manifest payload is not valid JSON.") from exc

    version = payload.get("version")
    channel = payload.get("channel")
    published_at = payload.get("published_at")
    images_payload = payload.get("images")
    if not isinstance(version, str) or not version:
        raise ReleaseManifestError("Signed release manifest version is missing.")
    if not isinstance(channel, str) or not channel:
        raise ReleaseManifestError("Signed release manifest channel is missing.")
    if not isinstance(published_at, str) or not published_at:
        raise ReleaseManifestError("Signed release manifest published_at value is missing.")
    if not isinstance(images_payload, dict):
        raise ReleaseManifestError("Signed release manifest images payload is missing.")

    images: dict[str, ReleaseImage] = {}
    for service in REQUIRED_RELEASE_SERVICES:
        image_payload = images_payload.get(service)
        if not isinstance(image_payload, dict):
            raise ReleaseManifestError(f"Signed release manifest image entry for {service} is missing.")
        ref = image_payload.get("ref")
        manifest_service = image_payload.get("service")
        if manifest_service != service:
            raise ReleaseManifestError(f"Signed release manifest service entry for {service} is invalid.")
        if not isinstance(ref, str) or "@sha256:" not in ref:
            raise ReleaseManifestError(f"Signed release manifest reference for {service} must be digest-pinned.")
        images[service] = ReleaseImage(service=service, ref=ref)

    return ReleaseManifest(version=version, channel=channel, published_at=published_at, images=images)


def load_release_manifest() -> ReleaseManifest:
    try:
        envelope = json.loads(_manifest_envelope_path().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseManifestError("Signed release manifest is missing from the runtime bundle.") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseManifestError("Signed release manifest envelope is not valid JSON.") from exc
    return verify_release_manifest_envelope(envelope)
