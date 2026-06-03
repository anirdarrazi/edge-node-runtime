import json
import shutil

import pytest

from node_agent.appliance_manifest import (
    ApplianceManifestError,
    bundled_runtime_dir,
    load_package_manifest,
    load_runtime_bundle_manifest,
    package_dir,
    package_manifest_path,
    runtime_manifest_path,
    verify_appliance_manifest_envelope,
    verify_runtime_bundle_dir,
)


def test_embedded_appliance_package_manifest_is_signed() -> None:
    manifest = load_package_manifest()

    assert manifest.kind == "appliance_package"
    assert manifest.version == "2026.04.10.1"
    assert "service.py" in manifest.files
    assert "service_ui.html" in manifest.files


def test_embedded_appliance_runtime_bundle_manifest_is_signed() -> None:
    manifest = load_runtime_bundle_manifest()

    assert manifest.kind == "appliance_runtime_bundle"
    assert manifest.version == "2026.04.10.1"
    assert "docker-compose.yml" in manifest.files
    assert "release-manifest.json" in manifest.files


def test_embedded_appliance_runtime_bundle_files_are_verified() -> None:
    manifest = verify_runtime_bundle_dir()

    assert manifest.kind == "appliance_runtime_bundle"
    assert "model-artifacts.json" in manifest.files


def test_runtime_bundle_verifies_text_line_ending_equivalent_files(tmp_path) -> None:
    bundle = tmp_path / "runtime_bundle"
    shutil.copytree(bundled_runtime_dir(), bundle)

    compose_path = bundle / "docker-compose.yml"
    compose_path.write_bytes(compose_path.read_bytes().replace(b"\n", b"\r\n"))

    model_artifacts_path = bundle / "model-artifacts.json"
    model_artifacts_path.write_bytes(model_artifacts_path.read_bytes().replace(b"\r\n", b"\n"))

    assert verify_runtime_bundle_dir(bundle).kind == "appliance_runtime_bundle"


def test_runtime_bundle_rejects_non_line_ending_text_tampering(tmp_path) -> None:
    bundle = tmp_path / "runtime_bundle"
    shutil.copytree(bundled_runtime_dir(), bundle)
    (bundle / "docker-compose.yml").write_bytes(
        (bundle / "docker-compose.yml").read_bytes() + b"\n# unsigned change\n"
    )

    with pytest.raises(ApplianceManifestError):
        verify_runtime_bundle_dir(bundle)


def test_package_manifest_rejects_tampering() -> None:
    envelope = json.loads(package_manifest_path(package_dir()).read_text(encoding="utf-8"))
    envelope["payload"] = envelope["payload"][:-4] + "AAAA"

    with pytest.raises(ApplianceManifestError):
        verify_appliance_manifest_envelope(
            envelope,
            public_key_path=package_dir() / "appliance-package-manifest.pub",
            expected_kind="appliance_package",
        )


def test_runtime_bundle_manifest_rejects_tampering() -> None:
    manifest_path = runtime_manifest_path()
    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope["payload"] = envelope["payload"][:-4] + "AAAA"

    with pytest.raises(ApplianceManifestError):
        verify_appliance_manifest_envelope(
            envelope,
            public_key_path=manifest_path.with_name("appliance-runtime-manifest.pub"),
            expected_kind="appliance_runtime_bundle",
        )
