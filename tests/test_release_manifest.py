import json

import pytest

from node_agent.release_manifest import (
    ReleaseManifestError,
    bundled_release_dir,
    load_release_manifest,
    verify_release_manifest_envelope,
)


def test_embedded_release_manifest_is_signed_and_digest_pinned() -> None:
    manifest = load_release_manifest()

    assert manifest.version == "2026.04.10.1"
    assert manifest.images["node-agent"].ref.startswith("anirdarrazi/autonomousc-ai-edge-runtime@sha256:")
    assert manifest.images["vllm"].ref.startswith("vllm/vllm-openai@sha256:")
    assert manifest.images["vector"].ref.startswith("timberio/vector@sha256:")


def test_release_manifest_rejects_tampering() -> None:
    manifest_path = bundled_release_dir() / "release-manifest.json"
    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    envelope["payload"] = payload[:-4] + "AAAA"

    with pytest.raises(ReleaseManifestError):
        verify_release_manifest_envelope(envelope)
