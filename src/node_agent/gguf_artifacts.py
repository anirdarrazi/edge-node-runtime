from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .release_manifest import load_release_manifest
from .runtime_quality import (
    default_exactness_class_for_quality_class,
    quality_class_for_model_format,
)
from .runtime_profiles import (
    ARTIFACT_MANIFEST_GGUF,
    LLAMA_CPP_INFERENCE_ENGINE,
    MODEL_FORMAT_GGUF,
    SUPPORTED_API_EMBEDDINGS,
    SUPPORTED_API_RESPONSES,
    runtime_profile_by_id,
)

if TYPE_CHECKING:
    from .config import NodeAgentSettings


class GgufArtifactsError(RuntimeError):
    pass


_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_QUANTIZATION_RE = re.compile(r"^[A-Z0-9_]+$")


@dataclass(frozen=True)
class GgufCapabilities:
    chat: bool
    embeddings: bool

    def payload(self) -> dict[str, bool]:
        return {
            "chat": self.chat,
            "embeddings": self.embeddings,
        }


@dataclass(frozen=True)
class GgufArtifactContract:
    artifact_kind: str
    artifact_manifest_type: str
    model: str
    operation: str
    source: str
    repository: str
    filename: str
    revision: str
    runtime_engine: str
    model_format: str
    quality_class: str
    exactness_class: str
    quantization_type: str
    file_digest: str
    expected_context_tokens: int
    capabilities: GgufCapabilities
    tokenizer_compatibility_notes: str

    def payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_manifest_type": self.artifact_manifest_type,
            "model": self.model,
            "operation": self.operation,
            "source": self.source,
            "repository": self.repository,
            "filename": self.filename,
            "revision": self.revision,
            "runtime_engine": self.runtime_engine,
            "model_format": self.model_format,
            "quality_class": self.quality_class,
            "exactness_class": self.exactness_class,
            "quantization_type": self.quantization_type,
            "file_digest": self.file_digest,
            "expected_context_tokens": self.expected_context_tokens,
            "capabilities": self.capabilities.payload(),
            "tokenizer_compatibility_notes": self.tokenizer_compatibility_notes,
        }


@dataclass(frozen=True)
class GgufArtifactsManifest:
    version: str
    generated_at: str
    artifacts: tuple[GgufArtifactContract, ...]


def _manifest_path() -> Path:
    return Path(__file__).with_name("runtime_bundle") / "gguf-artifacts.json"


def _required_string(raw: dict[str, Any], key: str, *, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise GgufArtifactsError(f"{label} {key} is missing.")
    return value


def _parse_capabilities(raw: Any, *, label: str, operation: str) -> GgufCapabilities:
    if not isinstance(raw, dict):
        raise GgufArtifactsError(f"{label} capabilities are missing.")
    chat = raw.get("chat")
    embeddings = raw.get("embeddings")
    if not isinstance(chat, bool) or not isinstance(embeddings, bool):
        raise GgufArtifactsError(f"{label} capabilities must declare chat and embeddings booleans.")
    if operation == SUPPORTED_API_RESPONSES and not chat:
        raise GgufArtifactsError(f"{label} responses artifacts must declare chat capability.")
    if operation == SUPPORTED_API_EMBEDDINGS and not embeddings:
        raise GgufArtifactsError(f"{label} embeddings artifacts must declare embedding capability.")
    return GgufCapabilities(chat=chat, embeddings=embeddings)


def _parse_contract(raw: Any) -> GgufArtifactContract:
    if not isinstance(raw, dict):
        raise GgufArtifactsError("GGUF artifact entry is invalid.")

    model = _required_string(raw, "model", label="GGUF artifact")
    operation = _required_string(raw, "operation", label=f"{model} GGUF artifact")
    label = f"{model} {operation} GGUF artifact"
    artifact_kind = _required_string(raw, "artifact_kind", label=label)
    artifact_manifest_type = _required_string(raw, "artifact_manifest_type", label=label)
    source = _required_string(raw, "source", label=label)
    repository = _required_string(raw, "repository", label=label)
    filename = _required_string(raw, "filename", label=label)
    revision = _required_string(raw, "revision", label=label)
    runtime_engine = _required_string(raw, "runtime_engine", label=label)
    model_format = _required_string(raw, "model_format", label=label)
    quality_class = _required_string(raw, "quality_class", label=label)
    exactness_class = _required_string(raw, "exactness_class", label=label)
    quantization_type = _required_string(raw, "quantization_type", label=label)
    file_digest = _required_string(raw, "file_digest", label=label)
    tokenizer_notes = _required_string(raw, "tokenizer_compatibility_notes", label=label)
    expected_context_tokens = raw.get("expected_context_tokens")

    if artifact_kind != "gguf_model_file":
        raise GgufArtifactsError(f"{label} artifact_kind must be gguf_model_file.")
    if artifact_manifest_type != ARTIFACT_MANIFEST_GGUF:
        raise GgufArtifactsError(f"{label} artifact_manifest_type must be {ARTIFACT_MANIFEST_GGUF}.")
    if runtime_engine != LLAMA_CPP_INFERENCE_ENGINE:
        raise GgufArtifactsError(f"{label} runtime_engine must be llama_cpp.")
    if model_format != MODEL_FORMAT_GGUF:
        raise GgufArtifactsError(f"{label} model_format must be gguf.")
    if operation not in {SUPPORTED_API_RESPONSES, SUPPORTED_API_EMBEDDINGS}:
        raise GgufArtifactsError(f"{label} operation is not supported.")
    if not _QUANTIZATION_RE.match(quantization_type):
        raise GgufArtifactsError(f"{label} quantization_type is invalid.")
    expected_quality_class = quality_class_for_model_format(model_format, quantization_type)
    if quality_class != expected_quality_class:
        raise GgufArtifactsError(
            f"{label} quality_class must match the quantization_type-derived tier {expected_quality_class}."
        )
    expected_exactness_class = default_exactness_class_for_quality_class(quality_class)
    if exactness_class != expected_exactness_class:
        raise GgufArtifactsError(
            f"{label} exactness_class must match the quality_class-derived default {expected_exactness_class}."
        )
    if not _SHA256_DIGEST_RE.match(file_digest):
        raise GgufArtifactsError(f"{label} file_digest must be a sha256 digest.")
    if not isinstance(expected_context_tokens, int) or expected_context_tokens <= 0:
        raise GgufArtifactsError(f"{label} expected_context_tokens is invalid.")

    return GgufArtifactContract(
        artifact_kind=artifact_kind,
        artifact_manifest_type=artifact_manifest_type,
        model=model,
        operation=operation,
        source=source,
        repository=repository,
        filename=filename,
        revision=revision,
        runtime_engine=runtime_engine,
        model_format=model_format,
        quality_class=quality_class,
        exactness_class=exactness_class,
        quantization_type=quantization_type,
        file_digest=file_digest,
        expected_context_tokens=expected_context_tokens,
        capabilities=_parse_capabilities(raw.get("capabilities"), label=label, operation=operation),
        tokenizer_compatibility_notes=tokenizer_notes,
    )


@lru_cache(maxsize=1)
def load_gguf_artifacts_manifest() -> GgufArtifactsManifest:
    try:
        raw = json.loads(_manifest_path().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GgufArtifactsError("Bundled GGUF artifacts manifest is missing.") from exc
    except json.JSONDecodeError as exc:
        raise GgufArtifactsError("Bundled GGUF artifacts manifest is not valid JSON.") from exc

    version = raw.get("version")
    generated_at = raw.get("generated_at")
    artifacts_raw = raw.get("artifacts")
    if not isinstance(version, str) or not version:
        raise GgufArtifactsError("Bundled GGUF artifacts manifest version is missing.")
    if not isinstance(generated_at, str) or not generated_at:
        raise GgufArtifactsError("Bundled GGUF artifacts manifest generated_at is missing.")
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise GgufArtifactsError("Bundled GGUF artifacts manifest artifacts are missing.")

    release_manifest = load_release_manifest()
    if version != release_manifest.version:
        raise GgufArtifactsError("Bundled GGUF artifacts manifest version does not match the signed runtime release manifest.")

    return GgufArtifactsManifest(
        version=version,
        generated_at=generated_at,
        artifacts=tuple(_parse_contract(record_raw) for record_raw in artifacts_raw),
    )


def find_gguf_artifact(model: str, operation: str) -> GgufArtifactContract | None:
    manifest = load_gguf_artifacts_manifest()
    for artifact in manifest.artifacts:
        if artifact.model == model and artifact.operation == operation:
            return artifact
    return None


def _settings_uses_gguf_contract(settings: object) -> bool:
    profile_id = getattr(settings, "resolved_runtime_profile_id", None)
    if not isinstance(profile_id, str) or not profile_id:
        profile_id = getattr(settings, "runtime_profile", None)
    profile = runtime_profile_by_id(profile_id)
    if profile is not None:
        return profile.artifact_manifest_type == ARTIFACT_MANIFEST_GGUF
    return getattr(settings, "resolved_inference_engine", None) == LLAMA_CPP_INFERENCE_ENGINE


def _settings_current_model(settings: object) -> str | None:
    value = getattr(settings, "current_model", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(settings, "vllm_model", None)
    if isinstance(value, str) and value:
        return value
    return None


def _default_operation_for_settings(settings: object) -> str:
    profile_id = getattr(settings, "resolved_runtime_profile_id", None)
    if not isinstance(profile_id, str) or not profile_id:
        profile_id = getattr(settings, "runtime_profile", None)
    profile = runtime_profile_by_id(profile_id)
    if profile is not None and profile.supported_apis == (SUPPORTED_API_EMBEDDINGS,):
        return SUPPORTED_API_EMBEDDINGS
    return SUPPORTED_API_RESPONSES


def resolved_gguf_artifact_contract(
    settings: NodeAgentSettings,
    model: str | None,
    operation: str | None,
) -> GgufArtifactContract | None:
    if not _settings_uses_gguf_contract(settings):
        return None
    if not isinstance(model, str) or not model or not isinstance(operation, str) or not operation:
        return None
    return find_gguf_artifact(model, operation)


def resolved_default_gguf_artifact_contract(settings: NodeAgentSettings) -> GgufArtifactContract | None:
    return resolved_gguf_artifact_contract(
        settings,
        _settings_current_model(settings),
        _default_operation_for_settings(settings),
    )
