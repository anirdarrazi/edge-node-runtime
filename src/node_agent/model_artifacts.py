from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .release_manifest import load_release_manifest
from .runtime_profiles import runtime_profile_by_id

if TYPE_CHECKING:
    from .config import NodeAgentSettings


class ModelArtifactsError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    digest: str
    size_bytes: int


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_kind: str
    operation: str
    repository: str
    revision: str
    runtime_engine: str
    files: tuple[ArtifactFile, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "operation": self.operation,
            "repository": self.repository,
            "revision": self.revision,
            "runtime_engine": self.runtime_engine,
            "files": [
                {
                    "path": file.path,
                    "digest": file.digest,
                    "size_bytes": file.size_bytes,
                }
                for file in self.files
            ],
        }


@dataclass(frozen=True)
class ModelArtifactRecord:
    model: str
    operation: str
    source: str
    repository: str
    revision: str
    model_manifest_digest: str
    tokenizer_digest: str
    model_manifest: ArtifactManifest
    tokenizer_manifest: ArtifactManifest


@dataclass(frozen=True)
class ModelArtifactsManifest:
    version: str
    generated_at: str
    artifacts: tuple[ModelArtifactRecord, ...]


CHAT_TEMPLATE_RELEVANT_PATHS = {
    "chat_template.jinja",
    "chat_template.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
}


def _manifest_path() -> Path:
    return Path(__file__).with_name("runtime_bundle") / "model-artifacts.json"


def _sorted_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(_sorted_json(value), separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _chat_template_digest(manifest: ArtifactManifest, operation: str) -> str | None:
    if operation != "responses":
        return None
    relevant_files = tuple(file for file in manifest.files if file.path in CHAT_TEMPLATE_RELEVANT_PATHS)
    if not relevant_files:
        return _sha256_json(manifest.payload())
    return _sha256_json(
        {
            **manifest.payload(),
            "files": [
                {
                    "path": file.path,
                    "digest": file.digest,
                    "size_bytes": file.size_bytes,
                }
                for file in relevant_files
            ],
        }
    )


def _parse_artifact_file(raw: Any, *, model: str, operation: str, label: str) -> ArtifactFile:
    if not isinstance(raw, dict):
        raise ModelArtifactsError(f"{model} {operation} {label} entry is invalid.")
    path = raw.get("path")
    digest = raw.get("digest")
    size_bytes = raw.get("size_bytes")
    if not isinstance(path, str) or not path:
        raise ModelArtifactsError(f"{model} {operation} {label} path is missing.")
    if not isinstance(digest, str) or not (
        digest.startswith("sha256:") or digest.startswith("gitsha1:")
    ):
        raise ModelArtifactsError(f"{model} {operation} {label} digest is missing.")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ModelArtifactsError(f"{model} {operation} {label} size is invalid.")
    return ArtifactFile(path=path, digest=digest, size_bytes=size_bytes)


def _parse_artifact_manifest(raw: Any, *, model: str, operation: str, label: str) -> ArtifactManifest:
    if not isinstance(raw, dict):
        raise ModelArtifactsError(f"{model} {operation} {label} manifest is invalid.")
    files_raw = raw.get("files")
    if not isinstance(files_raw, list) or not files_raw:
        raise ModelArtifactsError(f"{model} {operation} {label} files are missing.")
    artifact_kind = raw.get("artifact_kind")
    repository = raw.get("repository")
    revision = raw.get("revision")
    runtime_engine = raw.get("runtime_engine")
    if not isinstance(artifact_kind, str) or not artifact_kind:
        raise ModelArtifactsError(f"{model} {operation} {label} artifact_kind is missing.")
    if not isinstance(repository, str) or not repository:
        raise ModelArtifactsError(f"{model} {operation} {label} repository is missing.")
    if not isinstance(revision, str) or not revision:
        raise ModelArtifactsError(f"{model} {operation} {label} revision is missing.")
    if not isinstance(runtime_engine, str) or not runtime_engine:
        raise ModelArtifactsError(f"{model} {operation} {label} runtime_engine is missing.")
    files = tuple(
        _parse_artifact_file(file_raw, model=model, operation=operation, label=f"{label} file")
        for file_raw in files_raw
    )
    return ArtifactManifest(
        artifact_kind=artifact_kind,
        operation=operation,
        repository=repository,
        revision=revision,
        runtime_engine=runtime_engine,
        files=files,
    )


def _parse_record(raw: Any) -> ModelArtifactRecord:
    if not isinstance(raw, dict):
        raise ModelArtifactsError("Model artifact record is invalid.")
    model = raw.get("model")
    operation = raw.get("operation")
    source = raw.get("source")
    repository = raw.get("repository")
    revision = raw.get("revision")
    model_manifest_digest = raw.get("model_manifest_digest")
    tokenizer_digest = raw.get("tokenizer_digest")
    if not isinstance(model, str) or not model:
        raise ModelArtifactsError("Model artifact model is missing.")
    if not isinstance(operation, str) or not operation:
        raise ModelArtifactsError(f"{model} operation is missing.")
    if not isinstance(source, str) or not source:
        raise ModelArtifactsError(f"{model} {operation} source is missing.")
    if not isinstance(repository, str) or not repository:
        raise ModelArtifactsError(f"{model} {operation} repository is missing.")
    if not isinstance(revision, str) or not revision:
        raise ModelArtifactsError(f"{model} {operation} revision is missing.")
    if not isinstance(model_manifest_digest, str) or not model_manifest_digest.startswith("sha256:"):
        raise ModelArtifactsError(f"{model} {operation} model manifest digest is missing.")
    if not isinstance(tokenizer_digest, str) or not tokenizer_digest.startswith("sha256:"):
        raise ModelArtifactsError(f"{model} {operation} tokenizer digest is missing.")

    model_manifest = _parse_artifact_manifest(raw.get("model_manifest"), model=model, operation=operation, label="model")
    tokenizer_manifest = _parse_artifact_manifest(
        raw.get("tokenizer_manifest"), model=model, operation=operation, label="tokenizer"
    )
    if _sha256_json(model_manifest.payload()) != model_manifest_digest:
        raise ModelArtifactsError(f"{model} {operation} model manifest digest does not match the bundled descriptor.")
    if _sha256_json(tokenizer_manifest.payload()) != tokenizer_digest:
        raise ModelArtifactsError(f"{model} {operation} tokenizer digest does not match the bundled descriptor.")

    return ModelArtifactRecord(
        model=model,
        operation=operation,
        source=source,
        repository=repository,
        revision=revision,
        model_manifest_digest=model_manifest_digest,
        tokenizer_digest=tokenizer_digest,
        model_manifest=model_manifest,
        tokenizer_manifest=tokenizer_manifest,
    )


@lru_cache(maxsize=1)
def load_model_artifacts_manifest() -> ModelArtifactsManifest:
    try:
        raw = json.loads(_manifest_path().read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelArtifactsError("Bundled model artifacts manifest is missing.") from exc
    except json.JSONDecodeError as exc:
        raise ModelArtifactsError("Bundled model artifacts manifest is not valid JSON.") from exc

    version = raw.get("version")
    generated_at = raw.get("generated_at")
    artifacts_raw = raw.get("artifacts")
    if not isinstance(version, str) or not version:
        raise ModelArtifactsError("Bundled model artifacts manifest version is missing.")
    if not isinstance(generated_at, str) or not generated_at:
        raise ModelArtifactsError("Bundled model artifacts manifest generated_at is missing.")
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise ModelArtifactsError("Bundled model artifacts manifest artifacts are missing.")

    release_manifest = load_release_manifest()
    if version != release_manifest.version:
        raise ModelArtifactsError(
            "Bundled model artifacts manifest version does not match the signed runtime release manifest."
        )

    artifacts = tuple(_parse_record(record_raw) for record_raw in artifacts_raw)
    return ModelArtifactsManifest(version=version, generated_at=generated_at, artifacts=artifacts)


def find_model_artifact(model: str, operation: str, runtime_engine: str | None = None) -> ModelArtifactRecord | None:
    manifest = load_model_artifacts_manifest()
    for artifact in manifest.artifacts:
        if artifact.model != model or artifact.operation != operation:
            continue
        if runtime_engine and artifact.model_manifest.runtime_engine != runtime_engine:
            continue
        return artifact
    return None


def _settings_runtime_engine(settings: object) -> str:
    profile_id = getattr(settings, "resolved_runtime_profile_id", None)
    if not isinstance(profile_id, str) or not profile_id:
        profile_id = getattr(settings, "runtime_profile", None)
    profile = runtime_profile_by_id(profile_id)
    if profile is not None:
        return profile.inference_engine
    value = getattr(settings, "resolved_inference_engine", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(settings, "inference_engine", None)
    if isinstance(value, str) and value:
        return value
    return "vllm"


def _settings_reports_audited_manifests(settings: object) -> bool:
    value = getattr(settings, "reports_audited_manifests", None)
    if isinstance(value, bool):
        return value
    profile_id = getattr(settings, "resolved_runtime_profile_id", None)
    if not isinstance(profile_id, str) or not profile_id:
        profile_id = getattr(settings, "runtime_profile", None)
    profile = runtime_profile_by_id(profile_id)
    if profile is not None:
        return profile.reports_audited_manifests
    return _settings_runtime_engine(settings) == "vllm"


def _settings_current_model(settings: object) -> str | None:
    value = getattr(settings, "current_model", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(settings, "vllm_model", None)
    if isinstance(value, str) and value:
        return value
    return None


def resolved_model_manifest_digest(settings: NodeAgentSettings, model: str | None, operation: str | None) -> str | None:
    runtime_engine = _settings_runtime_engine(settings)
    if not _settings_reports_audited_manifests(settings):
        return None
    if isinstance(model, str) and model and isinstance(operation, str) and operation:
        artifact = find_model_artifact(model, operation, runtime_engine=runtime_engine)
        if artifact is not None:
            return artifact.model_manifest_digest
    override = settings.model_manifest_digest
    if isinstance(override, str) and override:
        return override
    return None


def resolved_tokenizer_digest(settings: NodeAgentSettings, model: str | None, operation: str | None) -> str | None:
    runtime_engine = _settings_runtime_engine(settings)
    if not _settings_reports_audited_manifests(settings):
        return None
    if isinstance(model, str) and model and isinstance(operation, str) and operation:
        artifact = find_model_artifact(model, operation, runtime_engine=runtime_engine)
        if artifact is not None:
            return artifact.tokenizer_digest
    override = settings.tokenizer_digest
    if isinstance(override, str) and override:
        return override
    return None


def resolved_chat_template_digest(settings: NodeAgentSettings, model: str | None, operation: str | None) -> str | None:
    runtime_engine = _settings_runtime_engine(settings)
    if not _settings_reports_audited_manifests(settings):
        return None
    if not isinstance(operation, str) or operation != "responses":
        return None
    if isinstance(model, str) and model:
        artifact = find_model_artifact(model, operation, runtime_engine=runtime_engine)
        if artifact is not None:
            return _chat_template_digest(artifact.tokenizer_manifest, operation)
    return None


def resolved_default_runtime_metadata(settings: NodeAgentSettings) -> tuple[str | None, str | None]:
    return (
        resolved_model_manifest_digest(settings, _settings_current_model(settings), "responses"),
        resolved_tokenizer_digest(settings, _settings_current_model(settings), "responses"),
    )
